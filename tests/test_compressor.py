"""M3 的触发：Agent 在每次调 LLM 之前，按 CompressionPolicy 决定要不要压缩。

这次只测"什么时候触发"，不测压缩本身（compress 的真实实现是下一块）。所以
SpyCompressor 只记调用次数，策略给一个很小的 context_window，这样不用造巨量文本。

★ 阈值是**算出来的**，不是猜的：第二次请求前，agent 数的是
  usage.prompt + usage.completion + estimate(锚点之后的每条消息)
  （见 context.tokens.measure 的 usage 分支）。测试先算出其中与 usage 无关的
  尾部估算，再让 usage 凑成目标值，使第二次请求恰好落在 / 差 1 离开阈值。
  这样断言钉住的是 >= 的边界本身，而不是某个魔法数字。
"""
from __future__ import annotations

import json

import pytest

from mini_agent.agent import Agent
from mini_agent.context.compressor import CompressionPolicy
from mini_agent.context.tokens import (
    SOURCE_ESTIMATE,
    SOURCE_USAGE,
    estimate_message,
    request_fingerprint,
)
from mini_agent.schema import Message
from mini_agent.tools.registry import registry
from tests.conftest import FakeMessage

WINDOW = 1000
RATIO = 0.5
# 500，正好是整数，边界好钉。★ 必须写成 int(...)：WINDOW * RATIO 是 **float**
# （500.0），拿它去凑 usage.prompt 就是给 FakeUsage 塞浮点数；而真 SDK 的计数字段
# 是 int，llm._extract_usage 会（正确地）把浮点丢掉 → 锚点消失 → 边界钉不住，
# 症状是"阈值上那条"莫名其妙不触发。
THRESHOLD = int(WINDOW * RATIO)


class SpyCompressor:
    """记录被压缩了几次；可选地在压缩时抛异常。"""

    def __init__(self, raises: bool = False):
        self.calls = 0
        self._raises = raises

    def compress(self, turn) -> None:
        self.calls += 1
        if self._raises:
            raise RuntimeError("压缩器炸了")


def _policy() -> CompressionPolicy:
    return CompressionPolicy(context_window=WINDOW, trigger_ratio=RATIO)


def _tool_call_msg(call_id: str, name: str, arguments: dict, usage=None) -> FakeMessage:
    tc = type("TC", (), {
        "id": call_id,
        "type": "function",
        "function": type("F", (), {"name": name, "arguments": json.dumps(arguments)})(),
    })()
    return FakeMessage(content=None, tool_calls=[tc], usage=usage)


def _long_text(chars: int) -> str:
    """纯 ASCII，约 0.3 token/字符：6000 字符 ≈ 1800 token，远超 500。"""
    return "x" * chars


def _fake_turn(sid: str, messages: list[Message]):
    return type("T", (), {"session_id": sid, "messages": messages})()


@pytest.fixture
def noop_tool():
    registry.register(
        name="noop_ctx", toolset="ctxtest",
        schema={"description": "echo"}, handler=lambda text="": f"echoed:{text}",
    )
    yield "noop_ctx"
    registry._tools.pop("noop_ctx", None)


# ---------------------------------------------------------------------------
# 8. 到达阈值触发（>= 的边界，正反两面）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target,expected", [
    (THRESHOLD, 1),          # 正好在阈值上 → 触发（钉住 >=；改成 > 这条会红）
    (THRESHOLD - 1, 0),      # 差 1 token → 不触发（钉住"不到阈值绝不压"）
])
def test_threshold_boundary_triggers_on_greater_equal(
    fake_llm, db, noop_tool, target, expected
):
    """让第二次请求前的计数恰好 = target，看压缩器被调用几次。

    第二次请求时 turn.messages = [user, assistant(tool_calls), tool]，锚点是那条
    assistant（它带 tools 的 fingerprint，与本次请求一致），所以
      计数 = usage.prompt + usage.completion + estimate_message(tool)
    让 usage 两项凑成 target - estimate_message(tool)。
    """
    sid = db.create_session()
    tool_text = _long_text(100)
    tool_result = f"echoed:{tool_text}"
    completion = 0

    # 尾部估算用**与生产同一把尺子**的 estimate_message，而不是 _measure：
    # _measure 对"只有 tool 消息、没有锚点"的假 turn 会走整条估算（还含 system、
    # tool 定义和结构开销），和真实第二次请求的计数不是一回事。
    tail = estimate_message(Message(role="tool", tool_call_id="c1", content=tool_result))
    prompt = target - completion - tail      # target 已是 int，这里不能出现浮点
    assert prompt > 0, "测试参数不合理：尾部估算已超过目标，锚点用量算成了非正数"

    # 第一次请求前没有任何 assistant → 只有整条估算。它必须低于阈值，否则 spy 的
    # 计数会被第一轮污染，边界就钉不住了。
    agent = Agent(db=db, toolset={"ctxtest"}, compressor=SpyCompressor(), policy=_policy())
    first = agent._measure(
        _fake_turn(sid, [Message(role="user", content="q")]), agent._tool_defs()
    )
    assert first.source == SOURCE_ESTIMATE
    assert first.tokens < THRESHOLD

    fake_llm([
        _tool_call_msg("c1", noop_tool, {"text": tool_text}, usage=(prompt, completion)),
        FakeMessage(content="done", usage=(prompt, completion)),
    ])

    out = agent.run_conversation(sid, "q")

    assert out["completed"] is True
    assert agent.compressor.calls == expected


# ---------------------------------------------------------------------------
# 9. usage 全部缺失 → 靠估算触发
# ---------------------------------------------------------------------------

def test_estimate_source_triggers_when_usage_missing(fake_llm, db, noop_tool, monkeypatch):
    """上游一次都不给 usage：计数只能走估算，大工具结果照样要触发。"""
    fake_llm([
        _tool_call_msg("c1", noop_tool, {"text": _long_text(6000)}),   # ~1800 token
        FakeMessage(content="done"),
    ])
    spy = SpyCompressor()
    agent = Agent(db=db, toolset={"ctxtest"}, compressor=spy, policy=_policy())

    sources: list[str] = []
    real_measure = agent._measure

    def recording_measure(turn, tools):
        count = real_measure(turn, tools)
        sources.append(count.source)
        return count

    monkeypatch.setattr(agent, "_measure", recording_measure)

    agent.run_conversation(db.create_session(), "q")

    assert spy.calls == 1
    # _measure 被调 3 次：第一次请求前、第二次请求前（触发那一次）、压缩后的复核。
    # 钉住的是"触发那一刻"决定压缩的计数来源 —— 它必须来自估算。全程没有 usage，
    # 三次都是 estimate；这里只断言触发那一次，不把复核也算进来。
    assert sources[0] == SOURCE_ESTIMATE
    assert len(sources) >= 2
    assert sources[1] == SOURCE_ESTIMATE
    assert all(s == SOURCE_ESTIMATE for s in sources)


# ---------------------------------------------------------------------------
# 10. usage 有效但很小，增量很大 → 仍然触发（证明"只看 usage 会漏"的修正是对的）
# ---------------------------------------------------------------------------

def test_small_anchor_plus_large_increment_still_triggers(fake_llm, db, noop_tool):
    """锚点 usage 远低于阈值，但之后塞进一个巨大的工具结果 → 必须触发。

    如果 measure 找到锚点后只返回 prompt+completion（不加增量），这条会红。
    """
    fake_llm([
        _tool_call_msg("c1", noop_tool, {"text": _long_text(6000)}, usage=(50, 5)),
        FakeMessage(content="done", usage=(50, 5)),
    ])
    spy = SpyCompressor()
    agent = Agent(db=db, toolset={"ctxtest"}, compressor=spy, policy=_policy())

    agent.run_conversation(db.create_session(), "q")

    assert spy.calls == 1           # 锚点 55 token，但 + 1800 的增量后远超 500


# ---------------------------------------------------------------------------
# 11. 无状态下锚点跨调用存活
# ---------------------------------------------------------------------------

def test_anchor_survives_across_run_conversation(fake_llm, db):
    """第一次调用结束后，第二次调用的第一轮要靠库里 recover 回的 usage 当锚点。

    不落库的话，这里会是 estimate —— 而"每次调用的第一轮"恰恰是上下文最长的一轮。
    """
    fake_llm([FakeMessage(content="a1", usage=(777, 33))])
    agent = Agent(db=db)

    agent.run_conversation("sess_1", "q1")

    # 新实例（模拟 gRPC 换了 worker）再跑同一个 session
    agent2 = Agent(db=db)
    turn = _fake_turn("sess_1", db.recover("sess_1") + [Message(role="user", content="q2")])
    count = agent2._measure(turn, None)

    assert count.source == SOURCE_USAGE
    # 锚点之后恰好多了一条 user="q2"
    assert count.tokens == 777 + 33 + estimate_message(turn.messages[-1])


# ---------------------------------------------------------------------------
# 12. 压缩器抛异常：任务正常完成 + 告警
# ---------------------------------------------------------------------------

def test_compressor_exception_does_not_kill_the_task(fake_llm, db, noop_tool, caplog):
    fake_llm([
        _tool_call_msg("c1", noop_tool, {"text": _long_text(6000)}),
        FakeMessage(content="最终答案"),
    ])
    spy = SpyCompressor(raises=True)
    agent = Agent(db=db, toolset={"ctxtest"}, compressor=spy, policy=_policy())

    with caplog.at_level("WARNING", logger="mini_agent.agent"):
        out = agent.run_conversation(db.create_session(), "q")

    assert out["completed"] is True
    assert out["final_response"] == "最终答案"
    assert spy.calls >= 1
    assert "压缩器抛异常" in caplog.text


# ---------------------------------------------------------------------------
# 13. 不注入 compressor / policy → 完全不做检查
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_compressor", [False, True])
def test_no_check_without_both_dependencies(fake_llm, db, noop_tool, with_compressor):
    """两个依赖缺一个就完全不介入，即便上下文早已超阈值也一次不压。"""
    fake_llm([
        _tool_call_msg("c1", noop_tool, {"text": _long_text(6000)}),
        FakeMessage(content="done"),
    ])
    spy = SpyCompressor()
    # 两条参数都不缺 spy：with_compressor=True 时**只给** compressor（缺 policy），
    # 否则只给 policy（缺 compressor）。spy 始终在场，calls == 0 才是真断言，
    # 而不是"压根没有 spy 所以自然是 0"。
    kwargs = {"compressor": spy} if with_compressor else {"policy": _policy()}
    agent = Agent(db=db, toolset={"ctxtest"}, **kwargs)

    agent.run_conversation(db.create_session(), "q")
    assert spy.calls == 0


def test_measure_is_not_called_without_policy(fake_llm, db, monkeypatch):
    """没有 policy 时连计数都不做（零开销，行为与加这个功能之前逐字节一致）。"""
    fake_llm([FakeMessage(content="hi")])
    agent = Agent(db=db, compressor=SpyCompressor())   # 只有 compressor，没有 policy
    calls = []
    monkeypatch.setattr(agent, "_measure", lambda turn, tools: calls.append(1))

    agent.run_conversation(db.create_session(), "q")
    assert calls == []


# ---------------------------------------------------------------------------
# 补充：fingerprint 让 tools=None 的收尾调用不与带工具的锚点混淆
# ---------------------------------------------------------------------------

def test_finishing_call_without_tools_has_a_different_fingerprint(
    fake_llm, db, noop_tool
):
    """预算耗尽的收尾调用 tools=None，它的指纹不能被当成带工具的锚点。

    预算 max_turns=2：两轮都调工具，然后收尾调用。收尾调用若仍返回 tool_calls，
    agent **不 record**（避免悬空 tool_calls），库里就没有它的 usage 可看 —— 所以
    脚本第 3 条给纯文本，让收尾那次落库。
    """
    sid = db.create_session()
    first_tool = _tool_call_msg("c0", noop_tool, {"text": "x"}, usage=(10, 1))
    second_tool = _tool_call_msg("c1", noop_tool, {"text": "x"}, usage=(20, 2))
    fake_llm([first_tool, second_tool, FakeMessage(content="收尾", usage=(30, 3))])
    agent = Agent(db=db, toolset={"ctxtest"}, max_turns=2)

    out = agent.run_conversation(sid, "go")
    assert out["completed"] is True
    assert out["final_response"] == "收尾"
    assert out["used_turns"] == 2

    history = db.recover(sid)
    assistants = [m for m in history if m.role == "assistant" and m.usage is not None]
    with_tools_fp = request_fingerprint(agent.system_prompt, agent._tool_defs())
    without_tools_fp = request_fingerprint(agent.system_prompt, None)

    assert with_tools_fp != without_tools_fp
    # 最后一条 assistant 是收尾调用：指纹是 tools=None 的那个
    assert assistants[-1].content == "收尾"
    assert assistants[-1].usage.fingerprint == without_tools_fp
    # 前面那些（带 tool_calls 的）是带工具的指纹
    assert len(assistants) == 3
    assert all(a.usage.fingerprint == with_tools_fp for a in assistants[:-1])
