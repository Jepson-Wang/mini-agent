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

from mini_agent.persistence.session_db import MiniSessionDB, PersistenceError
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


def test_recover_drops_dangling_tool_call(db, db_path):
    """崩在「assistant 已落库、tool 结果还没落库」的瞬间。

    这是崩溃恢复里唯一真正危险的状态：直接把这段历史发给 DeepSeek 会 400
    （tool_use 后面必须紧跟数量匹配的 tool_result）。所以整轮必须丢掉。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="帮我读 a.txt"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call()]))
    # ← 进程在这里被 kill：tool 结果永远不会落库

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user"]
    assert not any(m.tool_calls for m in revived)


def test_recover_drops_only_the_incomplete_round(db, db_path):
    """一轮残缺不该连累另一轮完整的：完整的留下，残缺的丢掉。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q1"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_1")]))
    db.append_message(sid, Message(role="tool", tool_call_id="call_1", content="r1"))
    db.append_message(sid, Message(role="user", content="q2"))
    db.append_message(sid, Message(role="assistant", tool_calls=[_tool_call("call_2")]))
    # call_2 的结果没落库 → 只有第二轮该被丢

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool", "user"]
    assert revived[1].tool_calls[0].id == "call_1"


def test_recover_partial_multi_tool_round_is_dropped_whole(db, db_path):
    """一个 assistant 发起两个 tool_call，只回来一个结果 → 整轮丢弃。

    不能只丢那个缺结果的 tool_call、留下另一个：配对规则要求「数量一致」，
    留半轮照样 400。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="读两个文件"))
    db.append_message(
        sid, Message(role="assistant", tool_calls=[_tool_call("c1"), _tool_call("c2")])
    )
    db.append_message(sid, Message(role="tool", tool_call_id="c1", content="r1"))
    # c2 的结果缺失

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.role for m in revived] == ["user"]


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
