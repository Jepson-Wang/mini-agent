"""单次工具输出的 token 上限：纯函数 + Agent 接线 + 配置 + 两个入口。

要堵的缺口：压缩永远不压"最新一轮"，单次工具输出本身超预算时压缩帮不上忙、
重试照样 400。所以截断是独立于 compressor 的运行时安全网。
"""
from __future__ import annotations

import dataclasses
import json
import sys

import pytest

from mini_agent.agent import Agent
from mini_agent.context.tokens import estimate_text, truncate_tool_output
from mini_agent.tools import registry as registry_mod
from mini_agent.tools.registry import registry
from tests.conftest import FakeMessage

ASCII = "".join(chr(97 + (i * 7) % 26) for i in range(20000))
CHINESE = "".join(chr(0x4E00 + (i * 13) % 2000) for i in range(8000))
MIXED = "".join(
    (chr(0x4E00 + (i * 13) % 2000) if i % 3 == 0 else chr(97 + i % 26)) for i in range(12000)
)


# ---------------------------------------------------------------------------
# 1. 纯函数
# ---------------------------------------------------------------------------

def test_within_budget_returns_same_object():
    text = "hello " * 10
    assert truncate_tool_output(text, estimate_text(text)) is text
    assert truncate_tool_output("", 0) == ""


@pytest.mark.parametrize("text", [ASCII, CHINESE, MIXED], ids=["ascii", "chinese", "mixed"])
@pytest.mark.parametrize("max_tokens", [200, 1500])
def test_over_budget_fits_and_keeps_head_and_tail(text, max_tokens):
    out = truncate_tool_output(text, max_tokens)

    assert estimate_text(out) <= max_tokens
    assert out.startswith(text[:20])
    assert out.endswith(text[-20:])
    assert "运行时截断" in out
    assert str(len(text)) in out
    # 二分要"尽量多留"：不能只留一点点就收手
    assert estimate_text(out) > max_tokens * 0.9


def test_head_gets_more_than_tail():
    out = truncate_tool_output(ASCII, 500)
    head = out.split("\n…[运行时截断")[0]
    tail = out.rsplit("]…\n", 1)[1]
    assert len(head) > len(tail) * 2


@pytest.mark.parametrize("text", [ASCII, CHINESE, MIXED], ids=["ascii", "chinese", "mixed"])
def test_deterministic(text):
    assert truncate_tool_output(text, 700) == truncate_tool_output(text, 700)


@pytest.mark.parametrize("max_tokens", [0, 1, 5, 20])
def test_tiny_budget_never_raises(max_tokens):
    out = truncate_tool_output(ASCII, max_tokens)
    assert isinstance(out, str) and out
    assert "截断" in out


# ---------------------------------------------------------------------------
# 2. 中文：防"按字符数 × 0.3 近似"的实现
# ---------------------------------------------------------------------------

def test_chinese_text_fits_budget_even_though_char_approx_says_it_does():
    limit = 1000
    text = CHINESE[: int(limit / 0.3) - 200]       # 按 0.3 算 < limit，按 0.6 算约 2 倍
    assert len(text) * 0.3 < limit < len(text) * 0.6
    out = truncate_tool_output(text, limit)
    assert out is not text
    assert estimate_text(out) <= limit


# ---------------------------------------------------------------------------
# 3-5. Agent 端到端
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry():
    registry_mod.discover_builtin_tools()
    snapshot = dict(registry._tools)
    try:
        yield registry
    finally:
        registry._tools.clear()
        registry._tools.update(snapshot)


def _tool_call_msg(call_id, name, arguments):
    tc = type("TC", (), {
        "id": call_id, "type": "function",
        "function": type("F", (), {"name": name, "arguments": json.dumps(arguments)})(),
    })()
    return FakeMessage(content=None, tool_calls=[tc])


LIMIT = 300


def _run(isolated_registry, fake_llm, db, cap, sid="sess_cap"):
    isolated_registry.register(name="big", toolset="cap", schema={"description": "x"},
                               handler=lambda: MIXED)
    fake_llm([_tool_call_msg("c1", "big", {}), FakeMessage(content="done")])
    events = []
    agent = Agent(db=db, toolset={"cap"}, max_tool_output_tokens=cap)
    agent.run_conversation(sid, "go", on_event=lambda k, p: events.append((k, p)))
    tool_msg = [m for m in db.recover(sid) if m.role == "tool"][0]
    return events, tool_msg


def test_agent_truncates_persisted_tool_message_and_event(isolated_registry, fake_llm, db):
    events, tool_msg = _run(isolated_registry, fake_llm, db, LIMIT)

    assert estimate_text(tool_msg.content) <= LIMIT
    assert "运行时截断" in tool_msg.content
    results = [json.loads(p) for k, p in events if k == "tool_result"]
    assert len(results) == 1
    assert results[0]["content"] == tool_msg.content       # 事件发的也是截断后的


def test_executed_key_matches_transcript_and_recover_backfills_truncated(
    isolated_registry, fake_llm, db
):
    sid = "sess_cap"
    _, tool_msg = _run(isolated_registry, fake_llm, db, LIMIT, sid)
    assistant_seq = db.conn.execute(
        "SELECT seq FROM messages WHERE session_id=? AND role='assistant' ORDER BY seq LIMIT 1",
        (sid,),
    ).fetchone()["seq"]

    cached = db.get_executed_result(sid, assistant_seq, 0, "c1")
    assert cached["content"] == tool_msg.content
    assert estimate_text(cached["content"]) <= LIMIT

    # 模拟崩溃：幂等表记了、tool 消息没落库
    with db._conn_lock:
        db.conn.execute("DELETE FROM messages WHERE session_id=? AND role='tool'", (sid,))
        db.conn.commit()
    assert not db.conn.execute(
        "SELECT 1 FROM messages WHERE session_id=? AND role='tool'", (sid,)
    ).fetchall()
    revived = [m for m in db.recover(sid) if m.role == "tool"]
    assert len(revived) == 1
    assert revived[0].content == tool_msg.content
    assert estimate_text(revived[0].content) <= LIMIT


def test_none_means_no_truncation(isolated_registry, fake_llm, db):
    _, tool_msg = _run(isolated_registry, fake_llm, db, None)
    assert tool_msg.content == MIXED


# ---------------------------------------------------------------------------
# 6. 配置
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    for k in ("MINI_AGENT_CONTEXT_WINDOW", "MINI_AGENT_TOOL_OUTPUT_MAX_TOKENS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def _load():
    from mini_agent.config import _load_settings
    return _load_settings()


def test_default_is_context_window_over_ten(clean_env):
    s = _load()
    assert s.context_window == 128_000
    assert s.tool_output_max_tokens == s.context_window // 10 == 12_800


def test_default_follows_context_window(clean_env):
    clean_env.setenv("MINI_AGENT_CONTEXT_WINDOW", "50000")
    assert _load().tool_output_max_tokens == 5000


def test_directly_constructed_settings_also_derive_the_default():
    """不经 _load_settings、直接构造 Settings 时，默认值同样随 context_window 推出。

    回归：字段曾写死 12_800 —— Settings(context_window=50_000) 拿到的上限比窗口的
    10% 大一倍多，"最新一轮的单个工具输出放得进尾部"的保证悄悄失效。
    """
    from mini_agent.config import Settings

    assert Settings(deepseek_api_key="x", context_window=50_000).tool_output_max_tokens == 5000
    assert Settings(deepseek_api_key="x", context_window=5).tool_output_max_tokens == 1
    assert Settings(deepseek_api_key="x", tool_output_max_tokens=777).tool_output_max_tokens == 777


def test_explicit_env_overrides(clean_env):
    clean_env.setenv("MINI_AGENT_CONTEXT_WINDOW", "50000")
    clean_env.setenv("MINI_AGENT_TOOL_OUTPUT_MAX_TOKENS", "3000")
    assert _load().tool_output_max_tokens == 3000


@pytest.mark.parametrize("raw", ["0", "-1", "abc"])
def test_invalid_env_falls_back_to_default(clean_env, raw):
    clean_env.setenv("MINI_AGENT_CONTEXT_WINDOW", "50000")
    clean_env.setenv("MINI_AGENT_TOOL_OUTPUT_MAX_TOKENS", raw)
    assert _load().tool_output_max_tokens == 5000


# ---------------------------------------------------------------------------
# 7. 两个入口都把配置传进了 Agent
# ---------------------------------------------------------------------------

def test_cli_passes_cap_to_agent(tmp_path, monkeypatch):
    import mini_agent.__main__ as M

    seen = {}

    class Spy(Agent):
        def __init__(self, **kw):
            seen.update(kw)
            super().__init__(**kw)

    monkeypatch.setattr(M, "settings", dataclasses.replace(
        M.settings, home=tmp_path, deepseek_api_key="dummy", tool_output_max_tokens=4321))
    monkeypatch.setattr(M, "Agent", Spy)
    monkeypatch.setattr("builtins.input", lambda *a: "exit")
    monkeypatch.setattr(sys, "argv", ["mini_agent"])
    M.main()
    assert seen["max_tool_output_tokens"] == 4321


def test_grpc_serve_passes_cap_to_agent(tmp_path, monkeypatch):
    import mini_agent.grpcserver.server as S

    seen = {}

    class Stop(Exception):
        pass

    def spy(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(S, "settings", dataclasses.replace(
        S.settings, home=tmp_path, deepseek_api_key="dummy", tool_output_max_tokens=4321))
    monkeypatch.setattr(S, "Agent", spy)
    with pytest.raises(Stop):
        S.serve(port=0, host="127.0.0.1", token="")
    assert seen["max_tool_output_tokens"] == 4321
