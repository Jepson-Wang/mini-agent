"""[M2] 持久化与崩溃恢复验收测试。

核心问题只有一个：**进程被 kill 掉之后，我还能不能把对话原样接上？**
所以这里几乎每个测试都遵循同一个套路：
    写库 → 丢掉这个连接 → 用一个全新连接读回来 → 断言一致。
"全新连接"就是"新进程"的替身，它保证我们断言的是**磁盘上的事实**，
而不是某个还活在内存里的缓存。

不需要真 API key，跑得飞快。
"""
from __future__ import annotations

import json
import sqlite3
import time

import pytest

from mini_agent.persistence.session_db import (
    _INTERRUPTED,
    MiniSessionDB,
    PersistenceError,
)
from mini_agent.schema import Message, ToolCall


@pytest.fixture
def db_path(tmp_path):
    """每个测试一个独立的临时库，绝不碰 ~/.mini_agent/state.db。"""
    return str(tmp_path / "state.db")


@pytest.fixture
def db(db_path):
    return MiniSessionDB(db_path)


def _tool_call(call_id: str = "call_1", path: str = "a.txt") -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments={"path": path})


# --------------------------------------------------------------------------
# 1. 底层配置：WAL 是崩溃恢复的地基，外键是"孤儿消息"的守门人
# --------------------------------------------------------------------------

def test_wal_and_foreign_keys_are_on(db):
    """WAL 模式和外键必须真的生效——这两个都不是 SQLite 的默认值。

    外键尤其容易漏：它是**连接级**开关，每建一个连接都得重新打开
    （WAL 相反，它是写进库文件的持久属性）。
    """
    assert db.conn.execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"
    assert db.conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1


# --------------------------------------------------------------------------
# 2. session 生命周期
# --------------------------------------------------------------------------

def test_corrupt_db_file_raises_persistence_error_not_raw_sqlite(tmp_path):
    """损坏的库文件必须抛 PersistenceError，不许漏裸 sqlite3 异常。

    _db_guard 的全部意义是「持久化层是模块边界，边界上只暴露自己的异常类型」。
    漏一个出去，调用方那句 except PersistenceError 就白写了 —— __main__ 里
    「数据还在、可以开新会话」的友好提示曾经就是这么失效的。

    注意 sqlite3.connect() 是惰性的：它根本不读文件头，所以「这不是个数据库」
    要到 _configure 的第一条 PRAGMA 才暴露。构造函数整体都得在 guard 里。
    """
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"SQLite format 3\x00" + b"\xde\xad\xbe\xef" * 500)

    with pytest.raises(PersistenceError) as exc:
        MiniSessionDB(str(bad))
    assert isinstance(exc.value.__cause__, sqlite3.Error)   # 异常链没断


def test_opening_existing_db_does_not_need_the_write_lock(db, db_path):
    """表已就绪时，构造一个 handle 不该抢写锁 —— 纯读路径不能被写者挡在门口。

    原本 _create_tables 无条件走 BEGIN IMMEDIATE，于是别人正在写时，一个只想
    读的 --resume 会被卡满 busy_timeout（实测 5.5s）甚至失败。WAL 下读者本该
    永不被写者阻塞，那样等于在门口就把 WAL 的好处丢了。
    """
    db.create_session()

    hog = sqlite3.connect(db_path, isolation_level=None)
    hog.execute("BEGIN IMMEDIATE;")          # 死死占住写锁
    try:
        t0 = time.perf_counter()
        MiniSessionDB(db_path)               # 不该在这里卡住
        elapsed = time.perf_counter() - t0
    finally:
        hog.execute("ROLLBACK;")
        hog.close()

    # busy_timeout 是 5s；真去抢写锁的话这里必然是秒级
    assert elapsed < 1.0, f"打开已存在的库花了 {elapsed:.2f}s，说明还在抢写锁"


def test_create_session_and_exists(db):
    sid = db.create_session()

    assert sid.startswith("sess_")
    assert db.session_exists(sid) is True
    # 不存在的 session 要返回 False，而不是抛 TypeError
    # （fetchone() 未命中时返回 None，不是空 tuple —— 这里踩过坑）
    assert db.session_exists("sess_does_not_exist") is False


def test_end_session_marks_done(db):
    sid = db.create_session()
    row = db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()
    assert row["status"] == "running"

    db.end_session(sid)

    row = db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()
    assert row["status"] == "done"


def test_transition_status_cas_returns_whether_it_won(db):
    """CAS 原语：只有 from_status 和当前状态匹配时才转移，返回值如实报告是否命中。"""
    sid = db.create_session()                                   # running
    assert db._transition_status(sid, "running", "done") is True     # 合法转移
    assert db._transition_status(sid, "running", "done") is False    # 当前已 done，from 不匹配
    assert db._transition_status(sid, "done", "failed") is True      # done -> failed 合法
    assert db._transition_status("sess_ghost", "running", "done") is False  # 会话不存在


def test_end_session_is_idempotent_cas(db):
    """end_session 跑在 finally 里、可能被重复触发：第二次是安全 no-op，不抛、不覆盖终态。

    无条件 UPDATE 也能重复调，但一旦将来有 failed/killed 终态，无条件写会把终态
    覆盖回 done。CAS 的 WHERE status='running' 把「已是终态」挡在门外。
    """
    sid = db.create_session()
    db.end_session(sid)          # running -> done
    db.end_session(sid)          # 已是 done：CAS 命中 0 行，no-op，不抛

    row = db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()
    assert row["status"] == "done"


def _status(db, sid):
    return db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()["status"]


def test_reopen_session_pulls_done_back_to_running(db):
    """--resume 缺的那条反向边：done -> running。"""
    sid = db.create_session()
    db.end_session(sid)
    assert _status(db, sid) == "done"

    assert db.reopen_session(sid) is True
    assert _status(db, sid) == "running"


def test_reopen_session_returns_false_when_already_running(db):
    """会话已经是 running（上次被 kill 没走完 finally）→ CAS 不命中，如实返回 False。

    调用方已用 session_exists 排除了「不存在」，所以 False 只剩这一种含义。
    它不是错误：正是崩溃恢复要处理的常态。状态也不能被改坏。
    """
    sid = db.create_session()                 # 从没 end 过，还是 running

    assert db.reopen_session(sid) is False
    assert _status(db, sid) == "running"      # 没被动


def test_resume_cycle_keeps_status_honest_and_quiet(db, db_path, caplog):
    """整条 --resume 生命周期的回归测试：状态标签不许撒谎，正常路径不许打告警。

    没有 reopen_session 时的老行为（这测试就是来钉死它的）：
      - 续聊全程 status 停在 done —— 会话活着但标签写着结束；
      - 退出时 end_session 的 CAS 必然落空，在**完全正常**的路径上打一条
        「不在 running 态」告警。那条告警本是给「重复调用」用的，正常路径必响
        就成了狼来了，真出事时会被无视。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q1"))
    db.end_session(sid)                                     # 第一次运行退出

    with caplog.at_level("WARNING", logger="mini_agent.persistence.session_db"):
        db.reopen_session(sid)                              # --resume
        db.append_message(sid, Message(role="user", content="q2"))
        assert _status(db, sid) == "running"                # ← 续聊期间不撒谎
        db.end_session(sid)                                 # 退出

    assert _status(db, sid) == "done"
    assert caplog.text == ""                                # ← 正常路径必须安静
    # 历史没丢：两次运行的消息都在
    assert [m.content for m in MiniSessionDB(db_path).recover(sid)] == ["q1", "q2"]


def test_root_session_defaults_genealogy_columns(db):
    """不带参数建的 root 会话：parent 为 NULL、depth=0、spawn_budget_used=0。"""
    sid = db.create_session()
    row = db.conn.execute(
        "SELECT parent_session_id, depth, spawn_budget_used "
        "FROM sessions WHERE session_id = ?;",
        (sid,),
    ).fetchone()
    assert row["parent_session_id"] is None
    assert row["depth"] == 0
    assert row["spawn_budget_used"] == 0


def test_child_session_records_parent_and_depth(db):
    """子代理会话：parent 指向父 sid、depth 逐层 +1。谱系链能查回来。"""
    root = db.create_session()
    child = db.create_session(parent_session_id=root, depth=1)
    grandchild = db.create_session(parent_session_id=child, depth=2)

    rows = {
        r["session_id"]: r
        for r in db.conn.execute(
            "SELECT session_id, parent_session_id, depth FROM sessions;"
        ).fetchall()
    }
    assert rows[child]["parent_session_id"] == root
    assert rows[child]["depth"] == 1
    assert rows[grandchild]["parent_session_id"] == child
    assert rows[grandchild]["depth"] == 2


def test_child_session_with_nonexistent_parent_raises(db):
    """外键守门：parent_session_id 指向不存在的会话 → PersistenceError，不留孤儿。"""
    with pytest.raises(PersistenceError) as exc:
        db.create_session(parent_session_id="sess_ghost", depth=1)
    assert isinstance(exc.value.__cause__, sqlite3.IntegrityError)


# --------------------------------------------------------------------------
# 3. append_message：seq 单调、updated_at 推进、约束校验
# --------------------------------------------------------------------------

def test_seq_is_monotonic_per_session(db):
    """seq 是 session 内自增，不是全局 rowid —— 两个 session 各自从 0 开始。"""
    a, b = db.create_session(), db.create_session()

    assert db.append_message(a, Message(role="user", content="a0")) == 0
    assert db.append_message(a, Message(role="assistant", content="a1")) == 1
    assert db.append_message(b, Message(role="user", content="b0")) == 0
    assert db.append_message(a, Message(role="user", content="a2")) == 2


def test_append_message_advances_session_updated_at(db):
    sid = db.create_session()
    before = db.conn.execute(
        "SELECT updated_at FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()["updated_at"]

    time.sleep(1.05)  # updated_at 是 epoch 秒，不睡满 1 秒看不出变化
    db.append_message(sid, Message(role="user", content="hi"))

    after = db.conn.execute(
        "SELECT updated_at FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()["updated_at"]
    assert after > before


def test_append_to_unknown_session_raises_persistence_error(db):
    """外键挡住孤儿消息，且异常必须是 PersistenceError —— 调用方不该 import sqlite3。"""
    with pytest.raises(PersistenceError) as exc:
        db.append_message("sess_ghost", Message(role="user", content="hi"))

    # 异常链没断：底层的 IntegrityError 还挂在 __cause__ 上，traceback 里能看到
    assert isinstance(exc.value.__cause__, sqlite3.IntegrityError)
    # 事务回滚干净：一条消息都没落下
    assert db.conn.execute("SELECT COUNT(*) FROM messages;").fetchone()[0] == 0


def test_append_message_rejects_bad_role(db):
    sid = db.create_session()
    bad = Message(role="user", content="x")
    object.__setattr__(bad, "role", "wizard")  # 绕过 pydantic，模拟脏数据

    with pytest.raises(PersistenceError):
        db.append_message(sid, bad)


# --------------------------------------------------------------------------
# 4. 崩溃恢复主线：写 → 换连接 → 读回来一致
# --------------------------------------------------------------------------

def test_crash_recovery_roundtrip(db, db_path):
    """M2 的核心验收：完整的一轮工具调用，换个连接读回来必须一模一样。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="帮我读 a.txt"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call()]))
    db.append_message(
        sid, Message(role="tool", tool_call_id="call_1", content='{"ok": true}')
    )
    db.append_message(sid, Message(role="assistant", content="内容是 ok"))

    # ★ 新连接 = 新进程。从这行往下，我们只信磁盘。
    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool", "assistant"]
    assert revived[0].content == "帮我读 a.txt"          # 中文没被存坏
    assert revived[3].content == "内容是 ok"

    # tool_calls 的往返是最容易写错的地方：arguments 必须还是 dict，
    # 不能变成 '{"path": "a.txt"}' 这种字符串（那是 wire format，不是内部格式）
    restored = revived[1].tool_calls[0]
    assert restored.id == "call_1"
    assert restored.name == "read_file"
    assert restored.arguments == {"path": "a.txt"}
    assert isinstance(restored.arguments, dict)

    # tool 结果靠 tool_call_id 认领它的 assistant，这个 id 必须活下来
    assert revived[2].tool_call_id == "call_1"


def test_recover_empty_session_returns_empty_list(db):
    assert db.recover(db.create_session()) == []


def test_recover_stubs_dangling_tool_call(db, db_path):
    """崩在「assistant 已落库、tool 结果还没落库」的瞬间。

    这是崩溃恢复里唯一真正危险的状态：直接把这段历史发给 DeepSeek 会 400
    （tool_use 后面必须紧跟数量匹配的 tool_result）。
    但**不能丢**这一轮 —— 丢掉 = 抹掉「工具可能已经跑过」这个事实，模型看不见
    就会重做一遍（邮件重发、文件重写）。正确解法是补一条 interrupted 桩：
    配对补齐（不 400），同时如实告诉模型「这次调用结果未知」，让它自己决定重试。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="帮我读 a.txt"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call()]))
    # ← 进程在这里被 kill：tool 结果永远不会落库

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool"]
    # 桩必须认领它的 assistant，否则配对不上，补了等于没补
    assert revived[2].tool_call_id == "call_1"
    assert revived[2].content == _INTERRUPTED


def test_recover_stubs_only_the_incomplete_round(db, db_path):
    """一轮残缺不该连累另一轮完整的：完整的原样带出，只有残缺的那轮补桩。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q1"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))
    db.append_message(sid, Message(role="tool", tool_call_id="call_1", content="r1"))
    db.append_message(sid, Message(role="user", content="q2"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_2")]))
    # call_2 的结果没落库 → 只有第二轮该补桩

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == [
        "user", "assistant", "tool", "user", "assistant", "tool",
    ]
    # 第一轮完全没被动过：真结果就是真结果，绝不能被桩覆盖
    assert revived[1].tool_calls[0].id == "call_1"
    assert revived[2].content == "r1"
    # 第二轮补了桩
    assert revived[5].tool_call_id == "call_2"
    assert revived[5].content == _INTERRUPTED


def test_recover_stubs_only_the_missing_call_in_a_multi_tool_round(db, db_path):
    """一个 assistant 发起两个 tool_call，只回来一个结果 → 只补缺的那个。

    配对规则要求「数量一致」，所以两个 tool_call 必须都有对应的 tool 消息；
    但已经拿到的那个真结果要原样保留 —— 它代表已经发生的副作用。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="读两个文件"))
    db.append_message(
        sid, Message(role="assistant", tool_calls=[_tool_call("c1"), _tool_call("c2")])
    )
    db.append_message(sid, Message(role="tool", tool_call_id="c1", content="r1"))
    # c2 的结果缺失

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool", "tool"]
    assert (revived[2].tool_call_id, revived[2].content) == ("c1", "r1")
    assert (revived[3].tool_call_id, revived[3].content) == ("c2", _INTERRUPTED)


def test_recover_backfills_missing_result_from_idempotency_table(db, db_path):
    """幂等表存在的全部意义，就是这个测试。

    三个写入点是有先后的：
        handler() 跑完 ──► record_executed_key ──► append_message(tool 结果)
                              ②                          ③
    崩在 ②③ 之间时，工具**真的跑过了**、结果**真的存下来了**，只是没进 messages。
    这时补 interrupted 桩就是在撒谎，会害模型白白重跑一次有副作用的操作。
    所以补桩前必须先查幂等表：查得到 → 无损恢复出真结果，一次都不用重跑。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="发封邮件"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))
    # 工具跑完了、幂等表也写了，但进程在 append_message 之前就死了
    db.record_executed_key("call_1", sid, {"content": "邮件已发送"})

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool"]
    assert revived[2].tool_call_id == "call_1"
    assert revived[2].content == "邮件已发送"      # ← 真结果，不是桩
    assert revived[2].content != _INTERRUPTED


def test_recover_stubs_when_idempotency_table_has_no_matching_key(db, db_path):
    """幂等表里有别的 key，但没有这个 tool_call 的 → 照样补桩。

    和上一个测试配成一对：证明补桩走的是 **按 tool_call_id 精确查表**，
    而不是「表里有行就当命中」。少了这个测试，一个把 key 写错的实现也能跑绿。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="发封邮件"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))
    db.record_executed_key("call_999", sid, {"content": "别人的结果"})  # 不相干的 key

    revived = MiniSessionDB(db_path).recover(sid)

    assert revived[2].content == _INTERRUPTED


def test_recover_skips_one_corrupt_row(db, db_path):
    """一行脏数据不该把整个 --resume 打死：跳过坏行，其余照常恢复。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="good-0"))
    db.append_message(sid, Message(role="assistant", content="good-1"))

    # 绕过 MiniSessionDB，直接往库里塞一行坏 JSON（模拟写到一半断电）
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute(
        "INSERT INTO messages (session_id, seq, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?);",
        (sid, 2, "user", '{"role": "user", "content": ', int(time.time())),
    )
    raw.close()
    db.append_message(sid, Message(role="user", content="good-3"))

    revived = MiniSessionDB(db_path).recover(sid)

    # 坏的那条被跳过，好的三条都在，顺序不乱
    assert [m.content for m in revived] == ["good-0", "good-1", "good-3"]


def test_recover_stubs_tool_call_with_empty_id_instead_of_raising(db, db_path):
    """tool_call 的 id 是空串 → 补桩，不许抛。

    空 id 不是假想情况：ToolCall.id 没有 min_length，而 ToolCall.from_openai 在
    SDK 漏给 id 时默认就填 ""。这样一条消息会被落库。
    危险在于 get_executed_result 有个「key 不能为空」的守卫——它是用来抓**调用方
    编程错误**的，完全正当；但 recover 传进去的是**从库里读出来的历史数据**。
    同一个守卫在这里会把一次数据异常升级成 PersistenceError，让这个会话
    **永远 --resume 不了**（坏数据在库里，每次读都一样，重试无用）。

    recover 面对历史数据的铁律：任何「这不该发生」都降级，不抛。
    """
    sid = db.create_session()
    db.append_message(
        sid,
        Message(role="assistant", tool_calls=[ToolCall(id="", name="f", arguments={})]),
    )

    revived = MiniSessionDB(db_path).recover(sid)   # 不抛，就是重点

    assert [m.role for m in revived] == ["assistant", "tool"]
    assert revived[1].content == _INTERRUPTED


@pytest.mark.parametrize("bad_content", [{"nested": 1}, 42, ["x"], True, None])
def test_recover_stubs_when_cached_content_is_not_a_string(db, db_path, bad_content):
    """幂等表里的 content 不是字符串 → 按未命中补桩，不许抛。

    Message.content 是 str|None，塞个 dict/int 进去会抛 pydantic ValidationError。
    这条比其它坏数据更阴：ValidationError **不是** PersistenceError，_db_guard
    只认 sqlite3/json 异常拦不住它，__main__ 的 except PersistenceError 也接不住
    —— 用户拿到裸 traceback，且这个会话永久 --resume 不了。

    而 {"content": <str>} 只是 agent.py 和 session_db.py 之间的口头约定：
    record_executed_key 的签名收任意 dict，没有任何东西强制形状。读侧必须自己扛。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))
    db.record_executed_key("call_1", sid, {"content": bad_content})

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["assistant", "tool"]
    assert revived[1].content == _INTERRUPTED


def test_recover_stubs_when_idempotency_row_is_corrupt(db, db_path):
    """幂等表的坏行 = 未命中，降级补桩，不许把整个 --resume 炸掉。

    和上一个测试是同一条原则的两半：messages 表的坏行跳过，executed_keys 表的
    坏行按未命中处理。两张表的容错态度必须一致——否则一行坏 JSON 会被 _db_guard
    翻译成 PersistenceError 抛出，一路冒到 main()，这个会话从此再也 --resume
    不了：明明只是缓存里一行脏数据，代价却是整段对话再也回不来。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="发封邮件"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))

    # 绕过 MiniSessionDB，直接塞一行坏 JSON（模拟 record_executed_key 写到一半断电）
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute(
        "INSERT INTO executed_keys (idempotency_key, session_id, result, created_at) "
        "VALUES (?, ?, ?, ?);",
        ("call_1", sid, '{"content": "邮件已发', int(time.time())),
    )
    raw.close()

    revived = MiniSessionDB(db_path).recover(sid)   # 不抛，就是这个测试的重点

    assert [m.role for m in revived] == ["user", "assistant", "tool"]
    assert revived[2].content == _INTERRUPTED


# --------------------------------------------------------------------------
# 5. 幂等表：first-write-wins
# --------------------------------------------------------------------------

def test_idempotency_first_write_wins(db):
    """幂等的全部意义：第一次写进去的结果，就是唯一权威值。

    重放时哪怕传了个不同的 result，也必须原样拿回第一次那个 ——
    否则「重放」就变成了「覆盖」，幂等表就白建了。
    """
    sid = db.create_session()

    is_first, canonical = db.record_executed_key("k1", sid, {"ok": True, "text": "第一次"})
    assert is_first is True
    assert canonical == {"ok": True, "text": "第一次"}

    is_first, canonical = db.record_executed_key("k1", sid, {"ok": False, "text": "重放"})
    assert is_first is False
    assert canonical == {"ok": True, "text": "第一次"}   # ← 没被覆盖

    # 表里也只有一行
    assert db.conn.execute("SELECT COUNT(*) FROM executed_keys;").fetchone()[0] == 1


def test_idempotency_survives_restart(db, db_path):
    """幂等表也得抗崩溃：重启后同一个 key 依然认得出这是重放。"""
    sid = db.create_session()
    db.record_executed_key("k1", sid, {"ok": True})

    revived = MiniSessionDB(db_path)
    assert revived.get_executed_result("k1") == {"ok": True}
    assert revived.record_executed_key("k1", sid, {"ok": False}) == (False, {"ok": True})


def test_get_executed_result_miss_returns_none(db):
    assert db.get_executed_result("never_written") is None


def test_get_executed_result_rejects_empty_key(db):
    with pytest.raises(PersistenceError):
        db.get_executed_result("")


def test_record_executed_key_rejects_non_dict(db):
    with pytest.raises(PersistenceError):
        db.record_executed_key("k", db.create_session(), "不是字典")


def test_chinese_result_stored_unescaped(db):
    r"""ensure_ascii=False：库里存的该是「中文」，不是 中文。

    不只是好看 —— 转义后体积翻三倍，而且拿 sqlite3 命令行查库时根本没法读。
    """
    sid = db.create_session()
    db.record_executed_key("k", sid, {"text": "中文"})

    stored = db.conn.execute(
        "SELECT result FROM executed_keys WHERE idempotency_key = 'k';"
    ).fetchone()["result"]
    assert "中文" in stored
    assert "\\u4e2d" not in stored
    assert json.loads(stored) == {"text": "中文"}
