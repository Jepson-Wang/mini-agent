"""[M0] agent 主循环验收测试。

断言：消息累积正确、纯文本即终止、多轮对话连贯、max_turns 截断死循环。
全部用 FakeLLM，不需要真 API。
"""
from __future__ import annotations

import json
import logging

import pytest

from mini_agent.agent import Agent, _Turn
from mini_agent.schema import Message, ToolCall
from mini_agent.tools.registry import registry
from tests.conftest import FakeMessage


class FakeToolCall:
    """长得像 openai SDK 的 tool_call（ToolCall.from_openai 要 .id/.function）。"""

    def __init__(self, call_id: str, name: str, args: dict):
        self.id = call_id
        self.type = "function"
        self.function = type(
            "F", (), {"name": name, "arguments": json.dumps(args)}
        )()


@pytest.fixture
def loop_echo():
    """一个确定性的工具，用完摘掉。注册表是模块级单例，不清理会漏到别的测试里。"""
    registry.register(
        name="loop_echo",
        toolset="looptest",
        description="原样回显",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda text="": f"echoed:{text}",
    )
    yield "loop_echo"
    registry._tools.pop("loop_echo", None)


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


def test_duplicate_execution_uses_canonical_result_and_warns(
    db, loop_echo, caplog, monkeypatch
):
    """同一个 (session, assistant_seq, idx) 撞键时，tool 消息必须用幂等表里的权威结果。

    新键完全由服务端数据构成，单写者走 run_conversation 根本撞不上，所以这里
    手工先记一行、再直接调 _execute_tool_calls，模拟"同一条 assistant 的 tool_calls 被重放"。
    若 tool 消息用本次的新结果，幂等表和 transcript 就分叉了：崩溃恢复补回来的
    是"权威结果"，正常路径落的却是另一个答案。
    """
    # mini_agent 根 logger 关了 propagate，caplog 的 handler 挂在全局 root 上收不到
    monkeypatch.setattr(logging.getLogger("mini_agent"), "propagate", True)
    sid = db.create_session()
    agent = Agent(db=db)

    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="call_0", name=loop_echo, arguments={"text": "新结果"})],
    )
    assistant_seq = db.append_message(sid, assistant)
    db.record_executed_key(sid, assistant_seq, 0, "call_0", {"content": "权威结果"})

    turn = _Turn(session_id=sid, messages=[assistant])
    with caplog.at_level("WARNING", logger="mini_agent.agent"):
        agent._execute_tool_calls(turn, assistant, assistant_seq)

    tool_msgs = [m for m in db.recover(sid) if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_0"
    assert tool_msgs[0].content == "权威结果"          # ← 不是 "echoed:新结果"
    assert turn.messages[-1].content == "权威结果"     # 内存副本和库一致
    assert "被执行了第二次" in caplog.text


def test_reused_tool_call_id_across_turns_pairs_each_with_its_own_result(
    fake_llm, db, loop_echo
):
    """模型在两轮里复用同一个 tool_call id，每个 assistant 只能配到自己那轮的结果。

    旧实现两处都会错：幂等表按 id 全局去重，第二轮的写入被 DO NOTHING 吞掉；
    recover 用全会话的 {tool_call_id: msg} 建索引，后写的覆盖先写的，第一轮的
    assistant 会认领第二轮的结果。无状态化后每次 run_conversation 都走 recover，
    所以这**不需要崩溃**，正常跑两轮就会让模型看到错位的历史。
    """
    llm = fake_llm([
        FakeMessage(tool_calls=[FakeToolCall("call_0", loop_echo, {"text": "first"})]),
        FakeMessage(content="done-1"),
        FakeMessage(tool_calls=[FakeToolCall("call_0", loop_echo, {"text": "second"})]),
        FakeMessage(content="done-2"),
    ])
    agent = Agent(db=db, toolset={"looptest"})

    agent.run_conversation("sess_1", "q1")
    agent.run_conversation("sess_1", "q2")

    history = db.recover("sess_1")
    assert [m.role for m in history] == [
        "user", "assistant", "tool", "assistant",
        "user", "assistant", "tool", "assistant",
    ]
    assert history[2].tool_call_id == "call_0"
    assert history[2].content == "echoed:first"
    assert history[6].tool_call_id == "call_0"
    assert history[6].content == "echoed:second"
    assert [m.content for m in history if m.role != "tool"] == [
        "q1", None, "done-1", "q2", None, "done-2",
    ]

    # 第二轮最后一次调用发给模型的上下文里，两条 tool 结果也必须各归各位
    last_sent = llm.calls[3]["messages"]
    tool_contents = [m["content"] for m in last_sent if m["role"] == "tool"]
    assert tool_contents == ["echoed:first", "echoed:second"]
