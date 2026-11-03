"""上下文超长 400 的被动兜底：识别（llm.is_context_overflow）+ 强制压缩 + 重试一次。

对应 agent._call_model / _maybe_compress(force=True)。

★ 所有 LLM 调用都是替身：主对话用本文件的 ScriptedLLM（按脚本"抛异常 / 返回回复"），
  总结用 test_compaction_agent.SummaryLLM；没有任何一处走网络。
★ "计数在阈值以下"靠把 policy 的窗口设得极大（1e9）造出来，而 LLMCompressor 的 window
  仍是 2000（决定能不能选出边界）。两者本来就是独立参数，这样能证明 force 越过了阈值，
  而不是碰巧也超阈值。
"""
from __future__ import annotations

import json

import httpx
import openai
import pytest

from mini_agent.agent import Agent, TaskCancelled, _Turn
from mini_agent.context.compressor import CompressionPolicy, LLMCompressor
from mini_agent.llm import LLMReply, is_context_overflow
from mini_agent.schema import Message
from tests.conftest import FakeMessage
from tests.test_compaction_agent import (
    BIG_CHARS,
    OLD_LONG,
    SummaryLLM,
    WINDOW,
    _n_compactions,
    _seed_old_tool_round,
)

HUGE_WINDOW = 10**9
RATIO = 0.9
URL = "https://api.deepseek.com/chat/completions"
OVERFLOW_MSG = (
    "This model's maximum context length is 131072 tokens. "
    "However, you requested 131329 tokens (131000 in the messages, 329 in the completion)."
)


# ---------------------------------------------------------------------------
# 构造异常 / 替身
# ---------------------------------------------------------------------------

def _api_exc(cls, message, body=None, status=400):
    resp = httpx.Response(status, request=httpx.Request("POST", URL))
    return cls(message, response=resp, body=body)


def _overflow() -> openai.BadRequestError:
    return _api_exc(
        openai.BadRequestError, OVERFLOW_MSG,
        body={"error": {"message": OVERFLOW_MSG, "type": "invalid_request_error"}},
    )


def _plain_400() -> openai.BadRequestError:
    return _api_exc(openai.BadRequestError, "Invalid request body format")


class ScriptedLLM:
    """主对话替身：脚本里是异常就抛、是 FakeMessage 就返回；脚本耗尽后重复 default。"""

    def __init__(self, script, default=None):
        self.script = list(script)
        self.default = default if default is not None else FakeMessage(content="(exhausted)")
        self.calls: list[dict] = []

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, BaseException):
            raise item
        return LLMReply(message=item, usage=None)


def _chars(req: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in req)


def _agent(db, summary_llm, **kw) -> Agent:
    """阈值形同虚设（窗口 1e9）：只有 force 才可能触发压缩。"""
    return Agent(
        db=db,
        compressor=LLMCompressor(db, WINDOW, llm=summary_llm),
        policy=CompressionPolicy(HUGE_WINDOW, RATIO),
        **kw,
    )


def _seed_big(db) -> str:
    """两条各 ≈ 6k token 的历史：本轮 user 进来后，中段 ≥ 2k 且超尾部预算 → 选得出边界。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="u" + "x" * BIG_CHARS))
    db.append_message(sid, Message(role="assistant", content="a" + "x" * BIG_CHARS))
    return sid


def _run(monkeypatch, agent, sid, llm, *, cancel=None):
    monkeypatch.setattr("mini_agent.agent.call_llm", llm)
    events: list[tuple[str, str]] = []
    out = agent.run_conversation(
        sid, "这次的问题", cancel_check=cancel, on_event=lambda t, p: events.append((t, p)),
    )
    return out, events


# ===========================================================================
# 1. is_context_overflow：窄识别
# ===========================================================================

class _Weird(openai.BadRequestError):
    """message / body 都是怪东西。"""


@pytest.mark.parametrize(
    "exc, expected",
    [
        # --- 该认的 ---
        (_api_exc(openai.BadRequestError, OVERFLOW_MSG), True),
        (_api_exc(openai.BadRequestError, "MAXIMUM CONTEXT LENGTH exceeded"), True),
        (_api_exc(openai.BadRequestError, "too long",
                  body={"code": "context_length_exceeded", "message": "too long"}), True),
        (_api_exc(openai.BadRequestError, "too long",
                  body={"error": {"code": "context_length_exceeded", "message": "too long"}}), True),
        (_api_exc(openai.UnprocessableEntityError, OVERFLOW_MSG, status=422), True),
        # --- 不该认的 ---
        (_api_exc(openai.BadRequestError, "Invalid request body format"), False),
        (_api_exc(openai.BadRequestError, "bad",
                  body={"error": {"code": "invalid_request_error", "message": "bad"}}), False),
        (_api_exc(openai.AuthenticationError, OVERFLOW_MSG, status=401), False),
        (_api_exc(openai.RateLimitError, OVERFLOW_MSG, status=429), False),
        (_api_exc(openai.InternalServerError, OVERFLOW_MSG, status=500), False),
        (RuntimeError("maximum context length"), False),
        (ValueError(OVERFLOW_MSG), False),
        # --- 怪对象：不抛、返回 False ---
        (_api_exc(openai.BadRequestError, "x", body=None), False),
        (_api_exc(openai.BadRequestError, 12345, body=None), False),
        (_api_exc(openai.BadRequestError, None, body="garbage"), False),
        (_api_exc(openai.BadRequestError, ["maximum context length"], body=["x"]), False),
        (_api_exc(openai.BadRequestError, "x", body={"error": "not-a-dict", "code": 5}), False),
    ],
)
def test_is_context_overflow(exc, expected):
    assert is_context_overflow(exc) is expected


def test_is_context_overflow_never_raises_even_if_attribute_access_blows_up():
    class Boom(openai.BadRequestError):
        @property
        def body(self):
            raise RuntimeError("body 炸了")

        @body.setter
        def body(self, v):
            pass

    assert is_context_overflow(Boom("x", response=httpx.Response(
        400, request=httpx.Request("POST", URL)), body=None)) is False
    assert is_context_overflow(None) is False        # type: ignore[arg-type]


# ===========================================================================
# 2. 超长 → 强制压缩 → 重试一次 → 完成（计数在阈值以下）
# ===========================================================================

def test_overflow_forces_compaction_and_retries_once(monkeypatch, db):
    summary = SummaryLLM(text="SUMMARY-X")
    agent = _agent(db, summary)
    sid = _seed_big(db)
    llm = ScriptedLLM([_overflow(), FakeMessage(content="最终答案")])

    out, events = _run(monkeypatch, agent, sid, llm)

    assert out == {"final_response": "最终答案", "completed": True, "used_turns": 1}
    assert len(llm.calls) == 2
    # 中段两条各 ≈6k token，超过 window*0.5 的分块预算 → 2 块 → 2 次总结调用，但只落 1 个检查点
    assert len(summary.calls) == 2
    assert _n_compactions(db) == 1
    assert _chars(llm.calls[1]["messages"]) < _chars(llm.calls[0]["messages"])
    assert "SUMMARY-X" in llm.calls[1]["messages"][1]["content"]

    # 事件成对，且只有强制那一次带 reason；计数远低于阈值，证明不是阈值触发的
    types = [t for t, _ in events]
    assert types.count("compacting") == 1 and types.count("compacted") == 1
    assert types.index("compacting") < types.index("compacted")
    compacting = json.loads(next(p for t, p in events if t == "compacting"))
    assert compacting["reason"] == "overflow"
    assert compacting["tokens"] < HUGE_WINDOW * RATIO
    compacted = json.loads(next(p for t, p in events if t == "compacted"))
    assert compacted["ok"] is True


def test_non_forced_compacting_event_has_no_reason(monkeypatch, db):
    """正常（阈值触发）路径的事件载荷不变：没有 reason 字段。"""
    summary = SummaryLLM()
    agent = Agent(
        db=db,
        compressor=LLMCompressor(db, WINDOW, llm=summary),
        policy=CompressionPolicy(WINDOW, 0.5),
    )
    sid = _seed_big(db)
    _out, events = _run(monkeypatch, agent, sid, ScriptedLLM([FakeMessage(content="ok")]))
    compacting = json.loads(next(p for t, p in events if t == "compacting"))
    assert set(compacting) == {"tokens", "source"}


# ===========================================================================
# 3. 最多重试一次
# ===========================================================================

def test_retries_at_most_once(monkeypatch, db):
    summary = SummaryLLM()
    agent = _agent(db, summary)
    sid = _seed_big(db)
    llm = ScriptedLLM([], default=_overflow())          # 每次都超长

    with pytest.raises(openai.BadRequestError):
        _run(monkeypatch, agent, sid, llm)

    assert len(llm.calls) == 2, "第 2 次仍超长必须直接抛，不能第 3 次"
    assert len(summary.calls) == 2          # 一次压缩 = 2 块（见测试 2）；没有第二次压缩
    assert _n_compactions(db) == 1


# ===========================================================================
# 4. 没有 compressor / policy → 原样上抛
# ===========================================================================

@pytest.mark.parametrize("variant", ["neither", "no_policy", "no_compressor"])
def test_without_compressor_or_policy_overflow_propagates(monkeypatch, db, variant):
    summary = SummaryLLM()
    kw = {}
    if variant == "no_policy":
        kw["compressor"] = LLMCompressor(db, WINDOW, llm=summary)
    elif variant == "no_compressor":
        kw["policy"] = CompressionPolicy(HUGE_WINDOW, RATIO)
    agent = Agent(db=db, **kw)
    sid = _seed_big(db)
    llm = ScriptedLLM([], default=_overflow())

    with pytest.raises(openai.BadRequestError):
        _run(monkeypatch, agent, sid, llm)

    assert len(llm.calls) == 1
    assert summary.calls == []


# ===========================================================================
# 5. 没有可压的东西 → 视图没变 → 不重试
# ===========================================================================

def test_unchanged_view_is_not_retried(monkeypatch, db):
    summary = SummaryLLM()
    agent = _agent(db, summary)
    sid = db.create_session()                               # 只有本轮一条短 user
    llm = ScriptedLLM([], default=_overflow())

    with pytest.raises(openai.BadRequestError):
        _out, _events = _run(monkeypatch, agent, sid, llm)

    assert len(llm.calls) == 1
    assert summary.calls == []
    assert db.latest_compaction(sid) is None


def test_unchanged_view_still_emits_paired_noop_events(monkeypatch, db):
    agent = _agent(db, SummaryLLM())
    sid = db.create_session()
    events: list[tuple[str, str]] = []
    monkeypatch.setattr("mini_agent.agent.call_llm", ScriptedLLM([], default=_overflow()))
    with pytest.raises(openai.BadRequestError):
        agent.run_conversation(sid, "q", on_event=lambda t, p: events.append((t, p)))
    assert [t for t, _ in events] == ["compacting", "compacted"]
    assert json.loads(events[1][1]) == {"ok": False, "reason": "noop"}


# ===========================================================================
# 6. 强制压缩时总结失败 → D1 裁剪生效 → 重试一次
# ===========================================================================

def test_summary_failure_during_forced_compaction_prunes_and_retries(monkeypatch, db):
    summary = SummaryLLM(raises=True)
    agent = _agent(db, summary)
    sid = db.create_session()
    _seed_old_tool_round(db, sid)                           # 含一条 ≈6k token 的旧长 tool 输出
    llm = ScriptedLLM([_overflow(), FakeMessage(content="最终答案")])

    out, events = _run(monkeypatch, agent, sid, llm)

    assert out["final_response"] == "最终答案"
    assert len(llm.calls) == 2
    assert len(summary.calls) == 1
    first = [m.get("content") or "" for m in llm.calls[0]["messages"]]
    second = [m.get("content") or "" for m in llm.calls[1]["messages"]]
    assert any(OLD_LONG in c for c in first)
    assert not any(OLD_LONG in c for c in second)
    assert any(c.startswith("[Old tool output") for c in second)
    assert db.latest_compaction(sid) is None
    compacted = json.loads(next(p for t, p in events if t == "compacted"))
    assert compacted["ok"] is False and compacted["prune_before"] is not None


def test_forced_degrade_may_move_prune_point_forward_but_never_back():
    """强制路径的例外：已裁剪过时，裁剪点取 max(旧, 新)；非强制路径仍是"固定"。"""
    from mini_agent.context.compressor import _prune_placeholder  # noqa: F401  (确认符号存在)
    from mini_agent.context.tokens import TokenCount

    def build(prune_before):
        msgs = [Message(role="tool", tool_call_id="t", content="L" * 300) for _ in range(3)]
        t = _Turn(session_id="s", messages=msgs, seqs=[1, 2, 3])
        t.prune_before = prune_before
        return t

    agent = Agent(db=None)                                   # type: ignore[arg-type]
    count = TokenCount(tokens=1, source="estimate")

    t = build(2)                                             # 旧裁剪点 2，新候选 3 → 前移
    agent._degrade_after_compaction_failure(t, 3, count, force=True)
    assert t.prune_before == 3
    assert [m.content.startswith("[Old tool output") for m in t.messages] == [True, True, False]

    t = build(3)                                             # 新候选 2 < 旧 3 → 不后退
    agent._degrade_after_compaction_failure(t, 2, count, force=True)
    assert t.prune_before == 3

    t = build(3)                                             # 非强制：沿用旧语义（替换为新边界）
    agent._degrade_after_compaction_failure(t, 2, count)
    assert t.prune_before == 2


# ===========================================================================
# 7. 非超长的 400 → 原样上抛，不触发压缩
# ===========================================================================

def test_plain_400_does_not_trigger_compaction(monkeypatch, db):
    summary = SummaryLLM()
    agent = _agent(db, summary)
    sid = _seed_big(db)                                      # 有得压：若误判就会压
    llm = ScriptedLLM([_plain_400()])

    events: list[tuple[str, str]] = []
    monkeypatch.setattr("mini_agent.agent.call_llm", llm)
    with pytest.raises(openai.BadRequestError):
        agent.run_conversation(sid, "q", on_event=lambda t, p: events.append((t, p)))

    assert len(llm.calls) == 1
    assert summary.calls == []
    assert not [t for t, _ in events if t == "compacting"]
    assert db.latest_compaction(sid) is None


# ===========================================================================
# 8. 强制压缩中取消 → TaskCancelled 上抛，没有检查点
# ===========================================================================

def test_cancel_during_forced_compaction_propagates(monkeypatch, db):
    summary = SummaryLLM()
    agent = _agent(db, summary)
    sid = _seed_big(db)
    llm = ScriptedLLM([_overflow(), FakeMessage(content="不该走到这里")])
    seq = iter([False])                 # 循环顶放行；之后（进强制压缩时）视为已取消

    with pytest.raises(TaskCancelled):
        _run(monkeypatch, agent, sid, llm, cancel=lambda: next(seq, True))

    assert len(llm.calls) == 1
    assert summary.calls == []
    assert _n_compactions(db) == 0
