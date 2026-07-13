"""[M0] agent 主循环验收测试。

断言：消息累积正确、纯文本即终止、多轮对话连贯、max_turns 截断死循环。
全部用 FakeLLM，不需要真 API。
"""
from __future__ import annotations

from mini_agent.agent import Agent
from tests.conftest import FakeMessage


def test_plain_text_terminates(fake_llm):
    """模型返回纯文本 → 一轮即终止，final_response 正确。"""
    fake_llm([FakeMessage(content="你好，我是 mini-agent")])
    agent = Agent()

    out = agent.run_conversation("hi")

    assert out["completed"] is True
    assert out["final_response"] == "你好，我是 mini-agent"
    assert out["used_turns"] == 1


def test_messages_accumulate(fake_llm):
    """messages 应按 user → assistant 顺序累积。"""
    fake_llm([FakeMessage(content="answer")])
    agent = Agent()

    agent.run_conversation("question")

    assert [m.role for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[0].content == "question"
    assert agent.messages[1].content == "answer"


def test_multi_turn_keeps_history(fake_llm):
    """跨多次 run_conversation，历史持续累积；每次都带上完整历史发给 LLM。"""
    llm = fake_llm([FakeMessage(content="a1"), FakeMessage(content="a2")])
    agent = Agent()

    agent.run_conversation("q1")
    agent.run_conversation("q2")

    assert [m.role for m in agent.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    # 第二次调用发出的 messages 应包含 system + 前 3 条历史 + 这次没有额外项
    second_call_msgs = llm.calls[1]["messages"]
    assert second_call_msgs[0]["role"] == "system"
    roles = [m["role"] for m in second_call_msgs]
    assert roles == ["system", "user", "assistant", "user"]


def test_budget_exhaustion_stops(fake_llm, monkeypatch):
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
    agent = Agent(max_turns=3)

    out = agent.run_conversation("go")

    assert out["completed"] is False
    assert out["used_turns"] == 3
    assert "budget" in out["final_response"]
