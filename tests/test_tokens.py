"""M3 的计数：估算规则、usage 锚点、退回估算。

这里的每个断言都对着 context/tokens.py 的 docstring 写 —— 尤其是"找的是有有效
usage 的 assistant，而不是最后一条 assistant"和"fingerprint 不一就整批作废"，
它们是 measure 最容易被"简化"错的两处。
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from mini_agent.context.tokens import (
    SOURCE_ESTIMATE,
    SOURCE_USAGE,
    estimate_message,
    estimate_request,
    estimate_text,
    measure,
    request_fingerprint,
)
from mini_agent.schema import Message, TokenUsage, ToolCall

SYSTEM = "you are a test agent"
TOOLS = [{"type": "function", "function": {"name": "echo", "parameters": {}}}]


def _usage(prompt: int, completion: int, system=SYSTEM, tools=TOOLS) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        fingerprint=request_fingerprint(system, tools),
    )


# ---------------------------------------------------------------------------
# 1. estimate_text：0.3 / 0.6 规则
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("", 0),                                  # 空串
    ("a" * 10, 3),                            # 10 × 0.3 = 3
    ("a" * 11, 4),                            # 3.3 → 向上取整
    ("中" * 10, 6),                           # 10 × 0.6 = 6
    ("中" * 11, 7),                           # 6.6 → 7
    ("ab中", 2),                              # 0.6 + 0.6 = 1.2 → 2（混排一次性取整）
    ("。", 1),                                # 中日韩标点按 CJK 算
])
def test_estimate_text(text, expected):
    assert estimate_text(text) == expected


def test_estimate_text_ceil_once_not_per_character():
    """混排必须**一次性** ceil：三类的零头各自 ceil 会把 1.2 算成 3。"""
    assert estimate_text("ab中") == 2          # 0.3+0.3+0.6 = 1.2
    assert estimate_text("a中") == 1           # 0.3+0.6 = 0.9 → 1


# ---------------------------------------------------------------------------
# 2. estimate_message：tool_calls 参数 + 结构开销
# ---------------------------------------------------------------------------

def test_estimate_message_counts_tool_call_arguments_and_overhead():
    plain = Message(role="user", content="hi")
    with_call = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hello"})],
    )

    # 每条消息 4 的结构开销是硬编码在 docstring 里的数：两边都至少 4
    assert estimate_message(plain) >= 4
    # 参数 JSON 必须被算进去：同一量级的文本放进 arguments 里，消息估算会明显变大
    assert estimate_message(with_call) > estimate_message(plain)
    # 参数文本本身占的份额：{"text": "hello"} 比纯 content "hi" 长得多
    assert estimate_message(Message(role="tool", tool_call_id="c", content="hello")) > 4


def test_estimate_request_includes_system_and_tool_defs():
    msgs = [Message(role="user", content="hello")]
    no_tools = estimate_request(msgs, SYSTEM, None)
    with_tools = estimate_request(msgs, SYSTEM, TOOLS)
    assert with_tools > no_tools


# ---------------------------------------------------------------------------
# 3. 有效锚点
# ---------------------------------------------------------------------------

def test_valid_anchor_uses_usage_plus_estimate_of_tail():
    anchor = Message(role="assistant", content="ok", usage=_usage(1000, 20))
    tail1 = Message(role="user", content="a" * 100)
    tail2 = Message(role="assistant", content="b" * 50)
    msgs = [Message(role="user", content="q"), anchor, tail1, tail2]

    count = measure(msgs, SYSTEM, TOOLS)

    expected_tail = estimate_message(tail1) + estimate_message(tail2)
    assert count.source == SOURCE_USAGE
    assert count.tokens == 1000 + 20 + expected_tail


# ---------------------------------------------------------------------------
# 4. 多条 assistant：取最后一条**有效**的；最后一条没有 usage 时往前找
# ---------------------------------------------------------------------------

def test_last_effective_anchor_is_used_not_first():
    first = Message(role="assistant", content="a1", usage=_usage(100, 1))
    second = Message(role="assistant", content="a2", usage=_usage(500, 2))
    msgs = [Message(role="user", content="q"), first, Message(role="user", content="q2"), second]

    count = measure(msgs, SYSTEM, TOOLS)
    assert count.source == SOURCE_USAGE
    assert count.tokens == 500 + 2           # 锚点之后没有消息


def test_last_assistant_without_usage_falls_back_to_earlier_anchor():
    """取消封口消息（没有 usage）必须被跳过，它的开销落进增量估算。"""
    anchor = Message(role="assistant", content="a1", usage=_usage(100, 1))
    seal = Message(role="assistant", content="(上一轮任务被取消，未完成)")
    msgs = [Message(role="user", content="q"), anchor, Message(role="user", content="q2"), seal]

    count = measure(msgs, SYSTEM, TOOLS)
    assert count.source == SOURCE_USAGE
    assert count.tokens == 100 + 1 + estimate_message(msgs[2]) + estimate_message(seal)


# ---------------------------------------------------------------------------
# 5. usage 不好用 → 退回整条请求估算
# ---------------------------------------------------------------------------

def _bad_usage(case: str) -> Message:
    good = _usage(1000, 10)
    if case == "none":
        return Message(role="assistant", content="a")
    if case == "zero_prompt":
        return Message(role="assistant", content="a",
                       usage=TokenUsage(prompt_tokens=0, completion_tokens=10,
                                        fingerprint=good.fingerprint))
    # 下面两种过不了 TokenUsage 的校验（StrictInt / >0 的规则在 measure 里），
    # 用 model_construct 绕过校验，模拟"从别处来的、没经过校验的 usage"。
    if case == "negative_prompt":
        usage = TokenUsage.model_construct(prompt_tokens=-5, completion_tokens=10,
                                           fingerprint=good.fingerprint)
    elif case == "str_prompt":
        usage = TokenUsage.model_construct(prompt_tokens="12", completion_tokens=10,
                                           fingerprint=good.fingerprint)
    elif case == "fingerprint_mismatch":
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=10,
                           fingerprint="0" * 16)
    else:  # pragma: no cover
        raise AssertionError(case)
    return Message(role="assistant", content="a", usage=usage)


@pytest.mark.parametrize("case", [
    "none", "zero_prompt", "negative_prompt", "str_prompt", "fingerprint_mismatch",
])
def test_unusable_usage_falls_back_to_full_estimate(case):
    anchor = Message(role="user", content="q")
    msgs = [anchor, _bad_usage(case)]

    count = measure(msgs, SYSTEM, TOOLS)
    assert count.source == SOURCE_ESTIMATE
    assert count.tokens == estimate_request(msgs, SYSTEM, TOOLS)


# ---------------------------------------------------------------------------
# 6. 没有 assistant：新会话第一轮走纯估算
# ---------------------------------------------------------------------------

def test_no_assistant_means_pure_estimate():
    msgs = [Message(role="user", content="第一句话")]
    count = measure(msgs, SYSTEM, TOOLS)
    assert count.source == SOURCE_ESTIMATE
    assert count.tokens == estimate_request(msgs, SYSTEM, TOOLS)


# ---------------------------------------------------------------------------
# 7. 旧格式兼容 + 结构约束
# ---------------------------------------------------------------------------

def test_old_message_json_without_usage_loads_as_none():
    """旧行没有 usage 这个 key：解析成功、usage 为 None，不需要 schema 迁移。"""
    raw = '{"role": "assistant", "content": "old"}'
    msg = Message.model_validate_json(raw)
    assert msg.usage is None


def test_usage_is_not_sent_to_the_model():
    """to_openai 是逐字段手工构造的，usage 绝不能混进请求体。"""
    msg = Message(role="assistant", content="hi", usage=_usage(1, 1))
    assert "usage" not in msg.to_openai()


@pytest.mark.parametrize("bad", [
    {"prompt_tokens": True, "completion_tokens": 1, "fingerprint": "f"},
    {"prompt_tokens": "12", "completion_tokens": 1, "fingerprint": "f"},
    {"prompt_tokens": 1, "completion_tokens": 1.5, "fingerprint": "f"},
])
def test_token_usage_rejects_wrong_types(bad):
    """StrictInt：True / "12" / 1.5 都不收 —— 库里读到假计数会变成假锚点。"""
    with pytest.raises(ValidationError):
        TokenUsage(**bad)


def test_usage_round_trips_through_json():
    """落库/恢复走的就是 model_dump_json → model_validate_json 这一对。"""
    msg = Message(role="assistant", content="a", usage=_usage(100, 7))
    revived = Message.model_validate_json(msg.model_dump_json())
    assert revived.usage == msg.usage


# ---------------------------------------------------------------------------
# 8. usage 坏掉只降级 usage，不连累整条消息（recover 金规则）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_usage", [
    '{"prompt_tokens": "12", "completion_tokens": 1, "fingerprint": "x"}',
    '{"prompt_tokens": true, "completion_tokens": 1, "fingerprint": "x"}',
    '{"prompt_tokens": 1, "completion_tokens": 1}',           # 缺 fingerprint
    '{"prompt_tokens": 1, "completion_tokens": 1, "fingerprint": null}',
    '"not-an-object"',
])
def test_bad_usage_degrades_to_none_but_keeps_the_message(bad_usage, caplog):
    """一条遥测字段坏了，真实的 assistant 必须还在。

    TokenUsage 自己仍是 StrictInt、仍然拒绝坏值（见 test_token_usage_rejects_wrong_types）；
    但 Message 用 wrap 校验把它兜成 None。否则 recover 会按"坏行跳过"把整条
    assistant（连同它的 tool_calls）扔掉，它窗口里的 tool 结果变成孤儿也被丢弃 ——
    一条脏计数换来历史凭空少一段。usage=None 的结局正是"没有锚点、退回估算"。
    """
    raw = (
        '{"role": "assistant", "content": "真实回答", '
        '"tool_calls": [{"id": "c1", "name": "echo", "arguments": {}}], '
        f'"usage": {bad_usage}}}'
    )
    with caplog.at_level("WARNING", logger="mini_agent.schema"):
        msg = Message.model_validate_json(raw)

    assert msg.content == "真实回答"
    assert msg.tool_calls is not None and msg.tool_calls[0].id == "c1"
    assert msg.usage is None
    assert "usage 校验失败" in caplog.text


def test_bad_usage_in_db_keeps_the_row_through_recover(db, tmp_path, caplog):
    """落库层同样不能丢：坏 usage 的 assistant 从库里 recover 回来仍在，只是没有锚点。"""
    import sqlite3

    db_path = str(tmp_path / "state.db")   # conftest 的 db fixture 就建在这里
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q"))
    # 绕过 MiniSessionDB 直接塞一行"字段类型坏了"的 JSON（模拟旧版本/脏数据）
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute(
        "INSERT INTO messages (session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?);",
        (sid, 2, "assistant",
         '{"role": "assistant", "content": "真实回答", '
         '"usage": {"prompt_tokens": "12", "completion_tokens": 1, "fingerprint": "x"}}',
         int(time.time())),
    )
    raw.close()

    with caplog.at_level("WARNING", logger="mini_agent.persistence.session_db"):
        history = db.recover(sid)

    # 消息还在（没有被当成坏行跳过），usage 降级为 None
    assert [m.content for m in history] == ["q", "真实回答"]
    assert history[-1].usage is None
    assert "跳过损坏的消息行" not in caplog.text
    # 它作为"无锚点"参与计数：退回估算，而不是拿假数字当锚点
    assert measure(history, SYSTEM, TOOLS).source == SOURCE_ESTIMATE


# ---------------------------------------------------------------------------
# 9. llm._extract_usage：上游 usage 的形状校验（bool 不能冒充 1）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("usage", [
    SimpleNamespace(prompt_tokens=True, completion_tokens=1),
    SimpleNamespace(prompt_tokens=1, completion_tokens=False),
    SimpleNamespace(prompt_tokens="12", completion_tokens=1),
    SimpleNamespace(prompt_tokens=None, completion_tokens=1),
    SimpleNamespace(prompt_tokens=1, completion_tokens=1.5),
    SimpleNamespace(completion_tokens=1),                    # 缺属性
    None,
])
def test_extract_usage_rejects_bad_values(usage):
    """任何不合法都返回 None，**绝不抛**：一次成功的调用不能因计数字段而失败。"""
    from mini_agent.llm import _extract_usage

    assert _extract_usage(SimpleNamespace(usage=usage)) is None


def test_extract_usage_accepts_real_ints():
    from mini_agent.llm import RawUsage, _extract_usage

    got = _extract_usage(
        SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3))
    )
    assert got == RawUsage(prompt_tokens=12, completion_tokens=3)


def test_extract_usage_tolerates_response_without_usage():
    from mini_agent.llm import _extract_usage

    assert _extract_usage(SimpleNamespace()) is None
