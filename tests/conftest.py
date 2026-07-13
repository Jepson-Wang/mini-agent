"""pytest fixtures：FakeLLM —— 用脚本化响应替换真实 LLM 调用。

这样整个 agent loop 可以确定性测试，不需要真 API key、不花钱、不慢。
对应 CLAUDE.md 第五节的测试策略。
"""
from __future__ import annotations

from typing import Any

import pytest


class FakeMessage:
    """模拟 openai SDK 返回的 message 对象（鸭子类型：有 .content / .tool_calls）。"""

    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class FakeLLM:
    """按预设脚本依次返回响应；记录每次调用的入参便于断言。"""

    def __init__(self, scripted: list[FakeMessage]):
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []   # 每次 call_llm 的 messages 快照

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if not self._scripted:
            # 脚本耗尽：默认返回一个收尾文本，避免无限循环
            return FakeMessage(content="(no more scripted responses)")
        return self._scripted.pop(0)


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
