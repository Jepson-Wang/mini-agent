"""M3 压缩总结 A 轮：检查点落库 / 加载 / summary 渲染。

对应 plan 第 3 节的测试 6–10 与 D4 渲染。B 轮（压缩器、Agent 接入）不在这里。

★ "造坏数据"一律走 MiniSessionDB 自己的连接(db.conn)。 它开了 foreign_keys=ON,
  和写侧走同一条连接级配置;用裸 sqlite3.connect 时外键默认关闭,很多"本该被拒绝"
  的用例会恒绿。CHECK 约束挡得住的东西(空白 summary)用 PRAGMA
  ignore_check_constraints 临时关掉,造完立刻关回来 —— 否则同一连接上后续的
  合法写入也会绕过约束,测试就失去意义了。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pytest

from mini_agent.context.compressor import SUMMARY_PREFIX, apply_summary
from mini_agent.persistence.session_db import (
    _INTERRUPTED,
    LoadedContext,
    MiniSessionDB,
)
from mini_agent.schema import Message, ToolCall

LOG = "mini_agent.persistence.session_db"

# 一个"空白" summary。必须用转义写:写成 '  \n\t' 字面量的话,库里会是一段
# 反斜杠 + n 的**可见**文本,CHECK 和 strip() 都认为它非空 —— 用例会恒绿
# (或恒红),却完全没测到"空白 summary"这条路径。
_BLANK_SUMMARY = "  \n\t"


@pytest.fixture
def db(tmp_path):
    return MiniSessionDB(str(tmp_path / "state.db"))


def _seed(db: MiniSessionDB, n: int = 6) -> tuple[str, list[int]]:
    """落一段最普通的历史:user / assistant / user / assistant ...，返回 (sid, seqs)。

    偶数 seq 是 user、奇数是 assistant,后面很多用例直接按这个约定挑边界 ——
    比每条都手写 Message 短,也更容易看出"边界落在哪种 role 上"。
    """
    sid = db.create_session()
    seqs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        seqs.append(db.append_message(sid, Message(role=role, content=f"msg-{i}")))
    return sid, seqs


def _count_compactions(db: MiniSessionDB) -> int:
    return db.conn.execute("SELECT COUNT(*) FROM compactions;").fetchone()[0]


def _raw_row(db: MiniSessionDB, sid: str, seq: int) -> tuple[str, str]:
    row = db.conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? AND seq = ?;",
        (sid, seq),
    ).fetchone()
    return row["role"], row["content"]


def _json_of(msgs) -> list[str]:
    return [m.model_dump_json() for m in msgs]


@contextmanager
def _checks_disabled(db: MiniSessionDB):
    """临时关掉 CHECK 约束,用来造写侧永远造不出来的坏数据。"""
    db.conn.execute("PRAGMA ignore_check_constraints = ON;")
    try:
        yield
    finally:
        db.conn.execute("PRAGMA ignore_check_constraints = OFF;")


# --------------------------------------------------------------------------
# 6. append_compaction 的拒绝
# --------------------------------------------------------------------------

def test_append_compaction_accepts_a_valid_checkpoint(db):
    sid, seqs = _seed(db)

    assert db.append_compaction(sid, seqs[2], "前半段的摘要") is True

    assert _count_compactions(db) == 1
    assert db.latest_compaction(sid) == (seqs[2], "前半段的摘要")


def test_append_compaction_rejects_boundary_not_greater_than_existing(db, caplog):
    """等于也拒:主键能挡完全相等,但 DAO 不能依赖 IntegrityError 来表达"拒绝"。"""
    sid, seqs = _seed(db)
    assert db.append_compaction(sid, seqs[2], "第一份") is True

    with caplog.at_level("WARNING", logger=LOG):
        # 等于已有最大值
        assert db.append_compaction(sid, seqs[2], "第二份") is False

    assert _count_compactions(db) == 1
    assert db.latest_compaction(sid) == (seqs[2], "第一份")
    assert "没有大于已有检查点" in caplog.text


def test_append_compaction_rejects_a_boundary_that_moves_backwards(db, caplog):
    """倒退必须单独成一条测试:这才是单调条件**独有**的保护。

    "等于"那种情况主键本来就挡得住(去掉单调条件也只是从 False 变成抛异常);
    倒退的边界和已有检查点主键不同,主键放行 —— 只有条件 INSERT 能拒绝它。
    和"等于"写在同一条测试里时,变异下"等于"先抛异常,倒退这一步根本跑不到。
    """
    sid, seqs = _seed(db)
    assert db.append_compaction(sid, seqs[2], "第一份") is True

    with caplog.at_level("WARNING", logger=LOG):
        assert db.append_compaction(sid, seqs[0], "倒退的") is False

    assert _count_compactions(db) == 1
    assert db.latest_compaction(sid) == (seqs[2], "第一份")
    assert "没有大于已有检查点" in caplog.text


def test_append_compaction_allows_boundary_greater_than_existing(db):
    sid, seqs = _seed(db)
    assert db.append_compaction(sid, seqs[2], "第一份") is True
    assert db.append_compaction(sid, seqs[4], "第二份") is True

    assert _count_compactions(db) == 2
    assert db.latest_compaction(sid) == (seqs[4], "第二份")


def test_append_compaction_rejects_boundary_on_a_tool_row(db, caplog):
    """边界切在 tool 行上 = 把一轮工具调用劈成两半,summary 后面会跟一条孤儿 tool。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="发邮件"))
    db.append_message(sid, Message(role="assistant", content="好的"))
    tc_seq = db.append_message(
        sid, Message(role="assistant", tool_calls=[ToolCall(id="call_1", name="send", arguments={})])
    )
    tool_seq = db.append_message(sid, Message(role="tool", tool_call_id="call_1", content="ok"))
    assert _raw_row(db, sid, tool_seq)[0] == "tool"

    with caplog.at_level("WARNING", logger=LOG):
        assert db.append_compaction(sid, tool_seq, "摘要") is False

    assert _count_compactions(db) == 0
    assert "role=tool" in caplog.text
    # 用 tool 行前面的 assistant 作边界是合法的 —— 证明拒绝的确实是 role,不是位置
    assert db.append_compaction(sid, tc_seq, "摘要") is True


def test_append_compaction_rejects_missing_boundary_row(db, caplog):
    sid, seqs = _seed(db)

    with caplog.at_level("WARNING", logger=LOG):
        assert db.append_compaction(sid, 999, "摘要") is False

    assert _count_compactions(db) == 0
    assert "边界行不存在" in caplog.text


@pytest.mark.parametrize("blank", ["", "  \n\t", "   ", "\r\n"])
def test_append_compaction_rejects_blank_summary(db, caplog, blank):
    sid, seqs = _seed(db)

    with caplog.at_level("WARNING", logger=LOG):
        assert db.append_compaction(sid, seqs[2], blank) is False

    assert _count_compactions(db) == 0
    assert "summary 为空" in caplog.text


def test_append_compaction_rejects_non_str_summary(db):
    """Python 层先挡:None / bytes 走到 SQL 层会变成 CHECK 失败或类型怪癖,都不是"拒绝"。"""
    sid, seqs = _seed(db)

    assert db.append_compaction(sid, seqs[2], None) is False
    assert db.append_compaction(sid, seqs[2], b"bytes") is False
    assert _count_compactions(db) == 0


# --------------------------------------------------------------------------
# 7. latest_compaction
# --------------------------------------------------------------------------

def test_latest_compaction_returns_none_without_checkpoints(db):
    sid, _ = _seed(db)
    assert db.latest_compaction(sid) is None


def test_latest_compaction_returns_the_largest_boundary(db):
    """用裸连接写入(倒序、乱序插入),证明"取最新"靠 ORDER BY,不是插入顺序。"""
    sid, seqs = _seed(db)
    for seq, summary in ((seqs[4], "第四"), (seqs[2], "第二"), (seqs[0], "第零")):
        db.conn.execute(
            "INSERT INTO compactions (session_id, first_kept_seq, summary, created_at) "
            "VALUES (?, ?, ?, 0);",
            (sid, seq, summary),
        )

    assert db.latest_compaction(sid) == (seqs[4], "第四")


def test_latest_compaction_is_per_session(db):
    a, a_seqs = _seed(db)
    b, b_seqs = _seed(db)
    db.append_compaction(a, a_seqs[2], "A 的")
    db.append_compaction(b, b_seqs[4], "B 的")

    assert db.latest_compaction(a) == (a_seqs[2], "A 的")
    assert db.latest_compaction(b) == (b_seqs[4], "B 的")


# --------------------------------------------------------------------------
# 8. load_context + apply_summary（D4 渲染）
# --------------------------------------------------------------------------

def test_load_context_without_checkpoint_is_full_history_with_none_boundary(db):
    sid, seqs = _seed(db)

    ctx = db.load_context(sid)

    assert isinstance(ctx, LoadedContext)
    assert ctx.boundary is None and ctx.summary is None
    assert ctx.seqs == seqs
    assert _json_of(ctx.messages) == _json_of(db.recover(sid))


def test_load_context_with_checkpoint_loads_only_after_boundary(db):
    """8a:只含 seq >= boundary 的行，seqs 与之对齐。"""
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "前两条的摘要")

    ctx = db.load_context(sid)

    assert ctx.boundary == seqs[2]
    assert ctx.summary == "前两条的摘要"
    assert [m.content for m in ctx.messages] == ["msg-2", "msg-3", "msg-4", "msg-5"]
    assert ctx.seqs == seqs[2:]
    assert len(ctx.seqs) == len(ctx.messages)


def _assert_render_invariants(msgs):
    """system 之后第一条是 user、任意相邻两条不同时为 user。"""
    assert msgs[0].role == "user"
    for prev, cur in zip(msgs, msgs[1:]):
        assert not (prev.role == "user" and cur.role == "user"), (
            f"出现了连续两条 user: {prev.content!r} -> {cur.content!r}"
        )


def _render_twice(ctx: LoadedContext):
    first, first_seqs = apply_summary(ctx.messages, ctx.seqs, ctx.summary)
    second, second_seqs = apply_summary(ctx.messages, ctx.seqs, ctx.summary)
    assert _json_of(first) == _json_of(second)       # 渲染确定性:逐字节相同
    assert first_seqs == second_seqs
    return first, first_seqs


def test_render_summary_inserts_user_when_tail_starts_with_assistant(db):
    """8b(其一):第一条保留是 assistant → 前面多一条 user summary。"""
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[1], "边界之前的摘要")
    ctx = db.load_context(sid)
    assert ctx.messages[0].role == "assistant"

    msgs, out_seqs = _render_twice(ctx)

    assert msgs[0].role == "user"
    assert msgs[0].content == SUMMARY_PREFIX + "边界之前的摘要"
    assert msgs[1].content == "msg-1"
    assert out_seqs == [None, *seqs[1:]]
    _assert_render_invariants(msgs)


def test_render_summary_merges_into_user_when_tail_starts_with_user(db):
    """8b(其二):第一条保留是 user → 合并成一条，不出现连续两条 user。"""
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "边界之前的摘要")
    ctx = db.load_context(sid)
    assert ctx.messages[0].role == "user"

    msgs, out_seqs = _render_twice(ctx)

    # 合并后条数不变、第一条内容 = 前缀 + 摘要 + 分隔 + 原文
    assert len(msgs) == len(ctx.messages)
    assert msgs[0].content == (
        SUMMARY_PREFIX + "边界之前的摘要" + "\n\n---\n\n" + "msg-2"
    )
    assert msgs[1].content == "msg-3"
    assert out_seqs == [None, *seqs[3:]]
    _assert_render_invariants(msgs)


def test_apply_summary_does_not_mutate_inputs(db):
    """8c:入参与模型对象都不改（合并分支最容易踩：就地改 content 最省事）。"""
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "摘要")
    ctx = db.load_context(sid)
    assert ctx.messages[0].role == "user"      # 走合并分支，否则证明不了什么
    before_msgs, before_seqs = _json_of(ctx.messages), list(ctx.seqs)
    head_identity, head_content = ctx.messages[0], ctx.messages[0].content

    out, out_seqs = apply_summary(ctx.messages, ctx.seqs, ctx.summary)

    assert _json_of(ctx.messages) == before_msgs
    assert ctx.seqs == before_seqs
    assert ctx.messages[0] is head_identity
    assert ctx.messages[0].content == head_content
    # 返回的是新对象，且返回的列表和入参列表不是同一个
    assert out[0] is not head_identity
    assert out is not ctx.messages
    assert out_seqs is not ctx.seqs


def test_apply_summary_with_none_returns_a_copy(db):
    sid, _ = _seed(db)
    ctx = db.load_context(sid)
    before = _json_of(ctx.messages)

    out, out_seqs = apply_summary(ctx.messages, ctx.seqs, None)

    assert _json_of(out) == before
    assert out_seqs == ctx.seqs
    assert out is not ctx.messages
    assert out_seqs is not ctx.seqs


# --------------------------------------------------------------------------
# 9. load_context 的读侧降级：忽略检查点、读全量、告警、不抛
# --------------------------------------------------------------------------

def _corrupt_boundary_json(db, sid, seq):
    db.conn.execute(
        "UPDATE messages SET content = '{not json' WHERE session_id = ? AND seq = ?;",
        (sid, seq),
    )


def _corrupt_boundary_role(db, sid, seq):
    """把边界行的 role 列改成 tool。绕过写侧校验,模拟被别的东西改坏/旧代码写坏。"""
    db.conn.execute(
        "UPDATE messages SET role = 'tool' WHERE session_id = ? AND seq = ?;",
        (sid, seq),
    )


def _boundary_is_system(db, sid, seq):
    """边界行的 role 列和 JSON **一致地**是 system。

    只有读侧的 role 检查挡得住这一种：它能正常解析（列与 JSON 不冲突），也不是
    tool（不会被 _sanitize 当孤儿丢掉），于是 seqs[0] 恰好等于边界 —— 删掉 role
    检查时，这条 system 消息会被当成第一条保留消息。boundary_role_tool 证明不了这点：
    它同时会被"边界行无法解析"的兜底拦下。
    """
    db.conn.execute(
        "UPDATE messages SET role = 'system', content = ? WHERE session_id = ? AND seq = ?;",
        (Message(role="system", content="伪造的 system").model_dump_json(), sid, seq),
    )


def _blank_summary(db, sid, seq):
    """CHECK 挡得住空白 summary,用 ignore_check_constraints 造出读侧必须扛的坏数据。"""
    with _checks_disabled(db):
        db.conn.execute(
            "UPDATE compactions SET summary = ? "
            "WHERE session_id = ? AND first_kept_seq = ?;",
            (_BLANK_SUMMARY, sid, seq),
        )


_CORRUPTIONS = {
    "boundary_json_broken": _corrupt_boundary_json,
    "boundary_role_tool": _corrupt_boundary_role,
    "boundary_role_system": _boundary_is_system,
    "blank_summary": _blank_summary,
}


@pytest.mark.parametrize("corruption", sorted(_CORRUPTIONS))
def test_load_context_degrades_to_full_history_on_a_bad_checkpoint(db, caplog, corruption):
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "摘要")
    _CORRUPTIONS[corruption](db, sid, seqs[2])

    with caplog.at_level("WARNING", logger=LOG):
        ctx = db.load_context(sid)              # 不得抛

    assert ctx.boundary is None
    assert ctx.summary is None
    assert "load_context: 检查点" in caplog.text
    # 全量历史:条数和 recover 一致(recover 自己的降级 —— 例如坏行跳过 —— 也算在内)
    assert [m.content for m in ctx.messages] == [m.content for m in db.recover(sid)]


def test_bad_data_fixtures_really_land(db):
    """证明"造坏数据"真的生效，否则上面三条降级测试可能只是在测一个完好的检查点。

    尤其是空白 summary：CHECK 默认会拒绝，若 PRAGMA 没生效就会抛 IntegrityError ——
    那说明这个用例造出来的数据是假的。
    """
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "摘要")

    # 先证明 CHECK 确实会拒绝，再用 PRAGMA 绕过 —— 否则"改成功了"可能只是因为约束没用
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.conn.execute("UPDATE compactions SET summary = ?;", (_BLANK_SUMMARY,))
    _blank_summary(db, sid, seqs[2])
    assert _raw_row(db, sid, seqs[2])[0] == "user"
    assert db.conn.execute("SELECT summary FROM compactions;").fetchone()[0].strip() == ""

    # PRAGMA 必须是"关掉又关回来"：关回来之后合法的插入照常被 CHECK 拦住
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        db.conn.execute("UPDATE compactions SET summary = ' ';")


def test_bad_data_fixtures_really_land_other_corruptions(db):
    sid, seqs = _seed(db)
    db.append_compaction(sid, seqs[2], "摘要")

    _corrupt_boundary_json(db, sid, seqs[2])
    assert _raw_row(db, sid, seqs[2])[1] == "{not json"
    _corrupt_boundary_role(db, sid, seqs[2])
    assert _raw_row(db, sid, seqs[2])[0] == "tool"


# --------------------------------------------------------------------------
# 10. 检查点之后仍有悬空 tool_call：补回真结果,seq 为 None
# --------------------------------------------------------------------------

def test_load_context_restores_real_tool_result_from_executed_keys(db):
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="老问题"))
    db.append_message(sid, Message(role="assistant", content="老回答"))
    boundary = db.append_message(sid, Message(role="user", content="邮件清单"))
    db.append_compaction(sid, boundary, "前面的摘要")

    tc = ToolCall(id="call_1", name="list_mail", arguments={})
    a_seq = db.append_message(sid, Message(role="assistant", tool_calls=[tc]))
    # 崩在"记幂等表 → 写 tool 消息"之间:真结果在 executed_keys,没有 tool 行
    db.record_executed_key(sid, a_seq, 0, "call_1", {"content": "三封未读邮件"})

    ctx = db.load_context(sid)

    assert ctx.boundary == boundary
    assert [m.role for m in ctx.messages] == ["user", "assistant", "tool"]
    tool_msg = ctx.messages[2]
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == "三封未读邮件"
    assert tool_msg.content != _INTERRUPTED
    assert ctx.seqs == [boundary, a_seq, None]
    assert len(ctx.seqs) == len(ctx.messages)
