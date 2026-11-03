"""pytest fixtures：FakeLLM —— 用脚本化响应替换真实 LLM 调用。

这样整个 agent loop 可以确定性测试，不需要真 API key、不花钱、不慢。
对应 CLAUDE.md 第五节的测试策略。
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def db(tmp_path):
    """一个独立的临时库。

    Agent 无状态化后 db 是**必需**依赖（会话历史的唯一真相在库里），所以连纯粹
    测循环逻辑的用例也得有它。test_persistence.py 有自己的同名 fixture，就近覆盖。
    """
    from mini_agent.persistence.session_db import MiniSessionDB

    return MiniSessionDB(str(tmp_path / "state.db"))


class FakeUsage:
    """长得像 openai SDK 的 response.usage。

    和真 SDK 一样，usage 挂在 **response 顶层**，不在 message 上 —— 所以它被塞进
    FakeMessage 只是为了方便脚本写法，FakeLLM 再把它挪到 response 壳上、交给
    真正的 llm._extract_usage 提取，与生产走同一条路径。
    """

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeMessage:
    """模拟 openai SDK 返回的 message 对象（鸭子类型：有 .content / .tool_calls）。

    usage=(prompt, completion) 表示"产生这条回复的那次请求"的用量；不给就是
    None —— 与上游不返回 usage 的情形一致（measure 那时会退回纯估算）。
    """

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list | None = None,
        usage: tuple[int, int] | None = None,
    ):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.usage = None if usage is None else FakeUsage(*usage)


class FakeLLM:
    """按预设脚本依次返回响应；记录每次调用的入参便于断言。"""

    def __init__(self, scripted: list[FakeMessage]):
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []   # 每次 call_llm 的 messages 快照

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        from mini_agent import llm
        from mini_agent.llm import LLMReply

        if not self._scripted:
            # 脚本耗尽：默认返回一个收尾文本，避免无限循环
            msg = FakeMessage(content="(no more scripted responses)")
        else:
            msg = self._scripted.pop(0)
        # ★ usage 不在这里自己拼，而是造一个"长得像 SDK 响应"的壳交给真的
        #   _extract_usage。 以前 FakeMessage.usage 被原样塞进 LLMReply，于是测试
        #   替身的形状（FakeUsage）和生产读的形状（agent 按元组解包）各走各的，
        #   agent 一改就读不到。走真实提取路径后，替身与生产的契约只剩一处：
        #   FakeUsage 必须有 prompt_tokens / completion_tokens —— 和真 SDK 一样。
        fake_response = SimpleNamespace(usage=msg.usage)
        return LLMReply(message=msg, usage=llm._extract_usage(fake_response))


@pytest.fixture
def fake_llm(monkeypatch):
    """用法：llm = fake_llm([FakeMessage(content="hi"), ...])。

    会把 mini_agent.agent.call_llm 替换成 FakeLLM 实例。

    patch 目标必须和模块的**真实全名**一致。全项目只用包绝对 import
    （from mini_agent.llm import call_llm），所以模块只有 mini_agent.agent
    这一个身份。若项目里同时存在裸名 `agent` 和 `mini_agent.agent`，同一份代码
    会被加载成两个模块对象，patch 打在副本上，对真正跑着的那个毫无作用。
    """
    def _make(scripted: list[FakeMessage]) -> FakeLLM:
        llm = FakeLLM(scripted)
        monkeypatch.setattr("mini_agent.agent.call_llm", llm)
        return llm

    return _make
