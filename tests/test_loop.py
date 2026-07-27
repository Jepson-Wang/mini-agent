"""[M0] agent 主循环验收测试。

断言：消息累积正确、纯文本即终止、多轮对话连贯、max_turns 截断死循环。
全部用 FakeLLM，不需要真 API。
"""
from __future__ import annotations

import pytest

from mini_agent.agent import Agent
from tests.conftest import FakeMessage


def test_retry_after_failure_does_not_duplicate_the_user_message(db, monkeypatch):
    """调用中途炸了，调用方用同一条消息重试 → 库里的 user 消息只能有一条。

    user 消息是**先落库、再调 LLM** 的，所以 LLM 一炸，库里就已经留下了它。
    重试时若无脑再 _record 一条，transcript 就成了 [user, user]。
    高并发 + 自动重试的生产环境下这必然发生。
    """
    calls = {"n": 0}

    def flaky(messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("LLM 超时")
        return FakeMessage(content="终于答上了")

    monkeypatch.setattr("mini_agent.agent.call_llm", flaky)
    agent = Agent(db=db)

    with pytest.raises(RuntimeError):
        agent.run_conversation("sess_1", "帮我查天气")

    out = agent.run_conversation("sess_1", "帮我查天气")      # 调用方重试

    assert out["final_response"] == "终于答上了"
    history = db.recover("sess_1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert [m.content for m in history] == ["帮我查天气", "终于答上了"]


def test_one_agent_serves_multiple_sessions_without_leaking(fake_llm, db):
    """一个 Agent 实例交替服务两个 session，历史绝不能串。

    这是无状态化的**核心验收**：session 状态只存在于库里，不挂在实例上。
    有状态版本里 self.messages 是实例字段，第二个 session 会读到第一个的残留。
    """
    llm = fake_llm([
        FakeMessage(content="a-1"),
        FakeMessage(content="b-1"),
        FakeMessage(content="a-2"),
    ])
    agent = Agent(db=db)

    agent.run_conversation("sess_a", "问题 A1")
    agent.run_conversation("sess_b", "问题 B1")
    agent.run_conversation("sess_a", "问题 A2")

    # sess_a 第二轮发出去的上下文里，不许出现 B 的任何痕迹
    third = llm.calls[2]["messages"]
    assert [m["role"] for m in third] == ["system", "user", "assistant", "user"]
    contents = [m.get("content") for m in third]
    assert "问题 B1" not in contents and "b-1" not in contents

    # 两个 session 在库里各自完整、互不干涉
    assert [m.content for m in db.recover("sess_a")] == [
        "问题 A1", "a-1", "问题 A2", "a-2",
    ]
    assert [m.content for m in db.recover("sess_b")] == ["问题 B1", "b-1"]


def test_plain_text_terminates(fake_llm, db):
    """模型返回纯文本 → 一轮即终止，final_response 正确。"""
    fake_llm([FakeMessage(content="你好，我是 mini-agent")])
    agent = Agent(db=db)

    out = agent.run_conversation("sess_1", "hi")

    assert out["completed"] is True
    assert out["final_response"] == "你好，我是 mini-agent"
    assert out["used_turns"] == 1


def test_messages_accumulate(fake_llm, db):
    """消息应按 user → assistant 顺序累积。

    无状态化后断言对象从 agent.messages 换成库里的事实：实例上已经不存在
    per-session 状态了，而磁盘本就是唯一真相 —— 这比断言内存副本更强。
    """
    fake_llm([FakeMessage(content="answer")])
    agent = Agent(db=db)

    agent.run_conversation("sess_1", "question")

    history = db.recover("sess_1")
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[0].content == "question"
    assert history[1].content == "answer"


def test_multi_turn_keeps_history(fake_llm, db):
    """跨多次 run_conversation，同一 session 的历史持续累积并被完整带上。

    有状态版本靠 self.messages 攒历史；无状态版本靠每轮 recover 从库里取回来。
    对外可观察的行为必须一模一样。
    """
    llm = fake_llm([FakeMessage(content="a1"), FakeMessage(content="a2")])
    agent = Agent(db=db)

    agent.run_conversation("sess_1", "q1")
    agent.run_conversation("sess_1", "q2")

    assert [m.role for m in db.recover("sess_1")] == [
        "user", "assistant", "user", "assistant",
    ]
    # 第二次调用发出的 messages 应包含 system + 前 2 条历史 + 这次的 user
    second_call_msgs = llm.calls[1]["messages"]
    assert second_call_msgs[0]["role"] == "system"
    roles = [m["role"] for m in second_call_msgs]
    assert roles == ["system", "user", "assistant", "user"]


def test_budget_exhaustion_stops(db, monkeypatch):
    """若模型永远返回 tool_calls（M0 不该发生，但要防失控），max_turns 截断。"""
    # 造一个永远带 tool_calls 的假响应，逼循环走到预算耗尽
    forever_tool = FakeMessage(
        content=None,
        tool_calls=[type("TC", (), {
            "id": "x", "function": type("F", (), {
                "name": "noop", "arguments": "{}"})()})()],
    )

    class Loop:
        calls = []

        def __call__(self, messages, tools=None):
            self.calls.append(messages)
            return forever_tool

    monkeypatch.setattr("mini_agent.agent.call_llm", Loop())
    agent = Agent(db=db, max_turns=3)

    out = agent.run_conversation("sess_1", "go")

    assert out["completed"] is False
    assert out["used_turns"] == 3
    assert "budget" in out["final_response"]
