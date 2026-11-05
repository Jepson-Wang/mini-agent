"""压缩的三个边界：覆盖率报告里没被任何测试走到的分支。

1. 检查点写入被拒（append_compaction 返回 False）→ CompactionError 必须带候选边界，
   Agent 才能做 D1 降级；不能把"没写进去"当成成功。
2. 上一个检查点没有逐字原话章节（旧版本写的 / 边界前没有 user）→ 滚动更新时
   原样交给模型，不能被 _strip_verbatim 切掉内容。
3. 视图里没有任何一行能当边界（全是 tool / 补桩）→ select_boundary 返回 None。
"""
from __future__ import annotations

import pytest

from mini_agent.context.compressor import (
    VERBATIM_END,
    VERBATIM_HEADER,
    CompactionError,
    LLMCompressor,
    _strip_verbatim,
    select_boundary,
)
from mini_agent.llm import LLMReply
from mini_agent.schema import Message
from tests.conftest import FakeMessage

WINDOW = 2000
BIG = "x" * 21000          # 约 6300 估算 token：单条就超过尾部预算 4000


class SummaryLLM:
    def __init__(self, text: str = "新摘要"):
        self.text = text
        self.calls: list[list[dict]] = []

    def __call__(self, messages, tools=None, max_tokens=None):
        self.calls.append(messages)
        return LLMReply(message=FakeMessage(content=self.text), usage=None)


def _turn(sid, ctx):
    return type("T", (), {
        "session_id": sid, "messages": ctx.messages, "seqs": ctx.seqs,
        "boundary": ctx.boundary, "prune_before": None, "cancel_check": None,
    })()


def _seed(db) -> str:
    """[user, assistant(大), user, assistant(大)]：最后那条 assistant 是最新一轮、留尾部，
    边界落在它身上，中段 = 前三条（远超 2k）。"""
    sid = db.create_session()
    for role, content in (("user", "第一问"), ("assistant", BIG),
                          ("user", "第二问"), ("assistant", BIG)):
        db.append_message(sid, Message(role=role, content=content))
    return sid


# --------------------------------------------------------------------------
# 1. 写入被拒 → CompactionError(boundary=候选)
# --------------------------------------------------------------------------

def test_rejected_checkpoint_write_raises_compaction_error_with_candidate(db, monkeypatch):
    sid = _seed(db)
    ctx = db.load_context(sid)
    expected = select_boundary(ctx.messages, ctx.seqs, None, WINDOW)
    assert expected is not None                       # 前提：确实有东西可压

    # 模拟"另一个进程刚写了一个更大的边界"之类的拒绝
    monkeypatch.setattr(db, "append_compaction", lambda *a, **k: False)
    llm = SummaryLLM()
    with pytest.raises(CompactionError) as ei:
        LLMCompressor(db, WINDOW, llm=llm).compress(_turn(sid, ctx))

    assert ei.value.boundary == expected               # 降级需要它
    assert llm.calls                                   # 总结确实调了，失败发生在落库
    assert db.latest_compaction(sid) is None


# --------------------------------------------------------------------------
# 2. 上一个检查点没有逐字原话章节
# --------------------------------------------------------------------------

@pytest.mark.parametrize("summary", [
    "旧格式摘要，没有原话章节",
    VERBATIM_HEADER + "只有开头标记、没有结束标记",       # 半截标记：同样原样返回
])
def test_strip_verbatim_returns_summary_unchanged_without_a_complete_marker(summary):
    assert _strip_verbatim(summary) == summary


def test_strip_verbatim_removes_only_the_verbatim_section():
    s = VERBATIM_HEADER + "用户原话" + VERBATIM_END + "模型摘要正文"
    assert _strip_verbatim(s) == "模型摘要正文"


def test_rolling_update_over_a_legacy_checkpoint_keeps_its_whole_text(db):
    """旧检查点没有原话章节：交给模型的 "Previous summary" 必须是它的全文。"""
    sid = _seed(db)
    # 先手工写一个"旧格式"检查点（边界 = 第二条 user，seq 2）
    assert db.append_compaction(sid, 2, "旧格式摘要：用户问过第一问")
    # 再追加一轮，让新边界 > 2
    for role, content in (("user", "第三问"), ("assistant", BIG)):
        db.append_message(sid, Message(role=role, content=content))
    ctx = db.load_context(sid)
    assert ctx.boundary == 2

    llm = SummaryLLM()
    LLMCompressor(db, WINDOW, llm=llm).compress(_turn(sid, ctx))

    prompt = llm.calls[0][1]["content"]
    assert "旧格式摘要：用户问过第一问" in prompt
    new_boundary, new_summary = db.latest_compaction(sid)
    assert new_boundary > 2
    assert new_summary.startswith(VERBATIM_HEADER)    # 新检查点重新带上原话章节


# --------------------------------------------------------------------------
# 3. 没有任何一行能当边界
# --------------------------------------------------------------------------

def test_select_boundary_none_when_no_row_can_be_a_boundary():
    tools_only = [Message(role="tool", tool_call_id=f"c{i}", content=BIG) for i in range(3)]
    # 全是补桩（seq=None）
    assert select_boundary(tools_only, [None, None, None], None, WINDOW) is None
    # 全是真实 tool 行（有 seq，但 role 不能当边界）
    assert select_boundary(tools_only, [1, 2, 3], None, WINDOW) is None


def test_select_boundary_none_on_malformed_input():
    msgs = [Message(role="user", content=BIG), Message(role="assistant", content=BIG)]
    assert select_boundary(msgs, [1], None, WINDOW) is None            # 长度不一致
    assert select_boundary([], [], None, WINDOW) is None               # 空
    assert select_boundary(msgs, [1, 2], None, 0) is None              # 非法窗口
