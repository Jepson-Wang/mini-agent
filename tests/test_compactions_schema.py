"""M3 压缩检查点：compactions 表 + schema v1→v2 迁移。

本次只做地基（建表 + 迁移），读写 DAO / recover 改造 / compressor 都不在这里。
这里要证明的是三件事：
  1. DDL 写对了（列、主键、外键）；
  2. 各种起点的库打开后都收敛到 v2，原有数据一条不动；
  3. 约束**真的生效**，而不只是声明在 sqlite_master 里。

★ 约束类测试（4/5/6）一律通过 MiniSessionDB 的**连接**写入。它在 _configure 里开了
  PRAGMA foreign_keys=ON；用裸 sqlite3.connect 时外键默认关闭，测试 4 会恒绿。
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager

import pytest

from mini_agent.persistence import session_db as session_db_module
from mini_agent.persistence.schema import (
    EXECUTED_KEYS_CREATE_SQL,
    MESSAGES_CREATE_SQL,
    SCHEMA_VERSION,
    SESSION_CREATE_SQL,
)
from mini_agent.persistence.session_db import MiniSessionDB, PersistenceError
from mini_agent.schema import Message


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "state.db")


# --------------------------------------------------------------------------
# 小工具（从 test_idempotency_scope.py 复制一份，不去 import 别的测试模块）
# --------------------------------------------------------------------------

def _user_version(path: str) -> int:
    raw = sqlite3.connect(path)
    try:
        return raw.execute("PRAGMA user_version;").fetchone()[0]
    finally:
        raw.close()


def _tables(path: str) -> set[str]:
    raw = sqlite3.connect(path)
    try:
        return {
            r[0]
            for r in raw.execute("SELECT name FROM sqlite_master WHERE type='table';")
        }
    finally:
        raw.close()


def _columns(path: str, table: str) -> set[str]:
    raw = sqlite3.connect(path)
    try:
        return {r[1] for r in raw.execute(f'PRAGMA table_info("{table}");')}
    finally:
        raw.close()


# --------------------------------------------------------------------------
# 造库 helper
# --------------------------------------------------------------------------

_V0_EXECUTED_KEYS_SQL = """
CREATE TABLE executed_keys (
    idempotency_key TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    result          TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
"""


def _make_v1_db(path: str) -> None:
    """造一份 v1 库：三张表都是当前结构、version=1。

    三张表 v1→v2 没变，直接用 schema.py 的常量，不手抄 DDL —— 这样真实结构一改，
    这里跟着改，不会出现测试库和真实库悄悄分叉。
    这里只管建表，数据由测试自己用裸连接写。
    """
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA journal_mode=WAL;")
    raw.execute(SESSION_CREATE_SQL)
    raw.execute(MESSAGES_CREATE_SQL)
    raw.execute(EXECUTED_KEYS_CREATE_SQL)
    raw.execute("PRAGMA user_version = 1;")
    assert raw.execute("PRAGMA user_version;").fetchone()[0] == 1
    raw.close()


def _make_legacy_db(path: str) -> None:
    """造一份 v0 库：executed_keys 是旧结构，带一行旧数据。"""
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA journal_mode=WAL;")
    raw.execute(SESSION_CREATE_SQL)
    raw.execute(MESSAGES_CREATE_SQL)
    raw.execute(_V0_EXECUTED_KEYS_SQL)
    raw.execute(
        "INSERT INTO executed_keys VALUES ('call_1', 'sess_old', '{\"content\": \"旧\"}', 0);"
    )
    assert raw.execute("PRAGMA user_version;").fetchone()[0] == 0
    raw.close()


def _seed_v1(path: str) -> tuple[str, int]:
    """造一份**带数据**的 v1 库，返回 (sid, assistant 的 seq)。

    ★ 顺序很重要：先建 v1 三张表，再用裸连接写数据。**不能**先用 MiniSessionDB
      打开 —— 它一打开就把库迁到 v2 了，反过来把 user_version 改回 1 也撤不掉
      已经建好的 compactions 表（迁移事务早已提交）。测试 7 要断言"迁移失败后
      没有 compactions"，就必须从一个**本来就没有**它的库出发。
    消息 JSON 用 Message.model_dump_json() 生成，和真实写入格式一致。
    """
    _make_v1_db(path)
    sid = "sess_v1"
    raw = sqlite3.connect(path, isolation_level=None)
    raw.execute("PRAGMA foreign_keys=ON;")   # 裸连接默认不开，别造出坏数据
    raw.execute(
        "INSERT INTO sessions (session_id, status, created_at, updated_at) "
        "VALUES (?, 'done', 0, 0);",
        (sid,),
    )
    for seq, role, content in ((0, "user", "你好"), (1, "assistant", "在的")):
        raw.execute(
            "INSERT INTO messages (session_id, seq, role, content, created_at) "
            "VALUES (?, ?, ?, ?, 0);",
            (sid, seq, role, Message(role=role, content=content).model_dump_json()),
        )
    # 幂等记录指向 assistant(seq=1)
    raw.execute(
        "INSERT INTO executed_keys "
        "(session_id, assistant_seq, tool_index, tool_call_id, result, created_at) "
        "VALUES (?, 1, 0, 'call_1', '{\"content\": \"工具结果\"}', 0);",
        (sid,),
    )
    raw.execute("PRAGMA user_version = 1;")
    assert raw.execute("PRAGMA user_version;").fetchone()[0] == 1
    raw.close()
    return sid, 1


def _db_with_one_message(db_path) -> tuple[MiniSessionDB, str, int]:
    db = MiniSessionDB(db_path)
    sid = db.create_session()
    seq = db.append_message(sid, Message(role="user", content="hi"))
    return db, sid, seq


def _write_compaction(conn: sqlite3.Connection, sid, seq, summary) -> None:
    """写一行检查点，失败时把 sqlite3.IntegrityError 原样抛给测试断言。

    必须传 MiniSessionDB.conn：它在 _configure 里开了 foreign_keys=ON，裸连接默认关着。
    外键那条测试在裸连接上会恒绿。也正因为走的是裸 conn，不经过 _db_guard，
    所以断言的是 sqlite3.IntegrityError，而不是被包一层的 PersistenceError。
    """
    conn.execute(
        "INSERT INTO compactions (session_id, first_kept_seq, summary, created_at) "
        "VALUES (?, ?, ?, 0);",
        (sid, seq, summary),
    )


# --------------------------------------------------------------------------
# 1. 新库 DDL：列 / 主键 / 外键
# --------------------------------------------------------------------------

def test_new_db_has_compactions_with_designed_shape(db_path):
    """只证明 DDL 写对了；"约束真的生效"由下面 4/5/6 证明。"""
    MiniSessionDB(db_path)
    assert _user_version(db_path) == SCHEMA_VERSION

    raw = sqlite3.connect(db_path)
    try:
        info = {r[1]: r for r in raw.execute("PRAGMA table_info(compactions);")}
        assert set(info) == {"session_id", "first_kept_seq", "summary", "created_at"}
        # table_info 第 5 列是 pk 序号：> 0 的列按序号组成主键
        pk = sorted((r[5], name) for name, r in info.items() if r[5] > 0)
        assert [name for _, name in pk] == ["session_id", "first_kept_seq"]
        for name in info:   # table_info 第 3 列是 notnull
            assert info[name][3] == 1, f"{name} 应为 NOT NULL"

        # ★ 复合外键在 foreign_key_list 里**按列拆成多行**、共享同一个 id（第 0 列），
        #   不是一行。所以按 id 分组比对，别断言 len(fks) == 1。
        fks = list(raw.execute("PRAGMA foreign_key_list(compactions);"))
        assert len({fk[0] for fk in fks}) == 1
        assert {(fk[2], fk[3], fk[4]) for fk in fks} == {
            ("messages", "session_id", "session_id"),
            ("messages", "first_kept_seq", "seq"),
        }
    finally:
        raw.close()


# --------------------------------------------------------------------------
# 2/3. 迁移：v1 → v2、v0 → v2
# --------------------------------------------------------------------------

def test_v1_db_migrates_to_v2_keeping_data(db_path):
    sid, seq = _seed_v1(db_path)

    db = MiniSessionDB(db_path)

    assert _user_version(db_path) == SCHEMA_VERSION
    assert "compactions" in _tables(db_path)
    # 原有数据原样可读：消息按 seq 顺序、幂等记录仍命中
    assert [(m.role, m.content) for m in db.recover(sid)] == [
        ("user", "你好"), ("assistant", "在的"),
    ]
    assert db.get_executed_result(sid, seq, 0, "call_1") == {"content": "工具结果"}


def test_v0_db_converges_to_v2_in_one_open(db_path):
    """v0 一次打开就收敛到 v2：改名封存 + compactions 都在，不分两次迁移。"""
    _make_legacy_db(db_path)

    MiniSessionDB(db_path)

    assert _user_version(db_path) == SCHEMA_VERSION
    assert "compactions" in _tables(db_path)
    assert "idempotency_key" in _columns(db_path, "executed_keys_legacy_v0")
    assert "idempotency_key" not in _columns(db_path, "executed_keys")


# --------------------------------------------------------------------------
# 4/5/6. 约束生效（全部走 MiniSessionDB 的连接，foreign_keys=ON）
# --------------------------------------------------------------------------

def test_first_kept_seq_must_reference_a_real_message(db_path):
    """外键正反两面：失败确实来自外键，而不是别的原因。"""
    db, sid, seq = _db_with_one_message(db_path)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        _write_compaction(db.conn, sid, seq + 100, "指向不存在的消息")
    _write_compaction(db.conn, sid, seq, "指向存在的消息")
    assert db.conn.execute("SELECT COUNT(*) FROM compactions;").fetchone()[0] == 1


def test_primary_key_rejects_same_boundary_twice(db_path):
    db, sid, seq = _db_with_one_message(db_path)

    _write_compaction(db.conn, sid, seq, "第一次")
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        _write_compaction(db.conn, sid, seq, "第二次")


@pytest.mark.parametrize("blank", ["", "   ", "\n\t\r\n"])
def test_blank_summary_is_rejected(db_path, blank):
    """空 / 纯空白 summary 一定是写侧 bug。

    第三个参数专门防"CHECK 写成单参 trim"的回归：SQLite 单参 trim() 只去空格，
    trim('\\n\\t\\r\\n') 仍然是 '\\n\\t\\r\\n'，写成单参时这条会变红。
    """
    db, sid, seq = _db_with_one_message(db_path)

    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        _write_compaction(db.conn, sid, seq, blank)


def test_summary_with_surrounding_space_is_accepted(db_path):
    db, sid, seq = _db_with_one_message(db_path)

    _write_compaction(db.conn, sid, seq, " x ")
    assert db.conn.execute("SELECT summary FROM compactions;").fetchone()[0] == " x "


# --------------------------------------------------------------------------
# 7/8. 迁移的原子性与并发
# --------------------------------------------------------------------------

class _FailOnStamp:
    """包一层真连接：转发一切，唯独在写版本号那一步抛异常，并记下执行过的 SQL。

    execute(...).fetchone() / 可迭代的结果都要照常工作 —— _create_tables 里的
    double-check(_schema_is_current) 和 _retire_legacy_executed_keys 都会经过它。
    用 __getattr__ 把其余属性透传给真连接，就不用逐个手写。
    """

    def __init__(self, conn, seen: list[str]):
        self._conn = conn
        self._seen = seen

    def execute(self, sql, *args):
        self._seen.append(sql)
        if sql.lstrip().startswith("PRAGMA user_version ="):
            raise sqlite3.OperationalError("injected")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_v1_migration_failure_rolls_back_atomically(db_path, monkeypatch):
    """v1→v2 在**写版本号**这一步失败 → 整段撤销：版本仍是 1、无 compactions。

    失败注入在最后一步而不是建表那一步，证明的是"建表与版本号同一事务"：
    前面的 CREATE IF NOT EXISTS 对 v1 库都是空操作，真正要证明的原子性是
    **compactions 已经建好之后写版本号失败 → 这张表随事务回滚消失**。

    关键是断言 seen 里确实执行过 COMPACTIONS_CREATE_SQL。若它没执行（比如
    外键/约束检查在建表时就失败），这条测试即使绿也证明不了任何回滚。
    """
    sid, _ = _seed_v1(db_path)
    seen: list[str] = []
    real_write_txn = MiniSessionDB._write_txn

    @contextmanager
    def failing_write_txn(self):
        with real_write_txn(self) as c:
            yield _FailOnStamp(c, seen)

    monkeypatch.setattr(MiniSessionDB, "_write_txn", failing_write_txn)

    with pytest.raises(PersistenceError):
        MiniSessionDB(db_path)

    assert any("CREATE TABLE IF NOT EXISTS compactions" in s for s in seen), \
        "失败点必须在建表之后，否则证明不了回滚"
    assert _user_version(db_path) == 1
    assert "compactions" not in _tables(db_path)
    # 原有三张表和它们的数据原样还在
    assert {"sessions", "messages", "executed_keys"} <= _tables(db_path)
    raw = sqlite3.connect(db_path)
    assert raw.execute("SELECT COUNT(*) FROM messages;").fetchone()[0] == 2
    raw.close()

    # 解除 patch 后能正常迁移，说明库没被这次失败弄坏
    monkeypatch.undo()
    db = MiniSessionDB(db_path)
    assert _user_version(db_path) == SCHEMA_VERSION
    assert db.session_exists(sid)


def test_concurrent_v1_migration_happens_once(db_path, monkeypatch):
    """v1 库并发打开只迁一次：BEGIN IMMEDIATE 里的二次检查挡住后到的连接。

    ★ 直接数慢路径"真正动手"的次数，而不是只看结果。只断言"收敛到 v2、表都对"
      抓不到任何东西：迁移写成 IF NOT EXISTS + 幂等判断，二次检查被删掉后慢路径
      重复执行结果照样正确。计数点选在 _retire_legacy_executed_keys —— 它在慢路径
      double-check **之后**调用，所以 4 个连接都白进慢路径时 calls == 4。
    """
    _make_v1_db(db_path)

    calls: list[int] = []
    real_retire = MiniSessionDB._retire_legacy_executed_keys

    def counting_retire(*args, **kwargs):
        calls.append(1)
        return real_retire(*args, **kwargs)

    # 原方法是 staticmethod，替身也得包成 staticmethod，否则会被当实例方法多传一个 self
    monkeypatch.setattr(
        MiniSessionDB, "_retire_legacy_executed_keys", staticmethod(counting_retire)
    )

    n = 4
    barrier = threading.Barrier(n)
    errors: list[str] = []

    # ★ 必须把竞态逼出来。不加这道闸，第一个连接迁得太快，其余线程到快路径时
    #   已经读到新版本直接返回，根本走不到慢路径 —— 删掉二次检查测试照样绿。
    #   闸设在 _write_txn 入口：谁都得等 n 个线程**全部**过完快路径（都读到旧版本）
    #   才能去抢 BEGIN IMMEDIATE，于是后进去的那几个只能靠二次检查挡回。
    real_write_txn = MiniSessionDB._write_txn
    gate = threading.Barrier(n, timeout=10)

    @contextmanager
    def gated_write_txn(self):
        try:
            gate.wait()
        except threading.BrokenBarrierError:
            pass
        with real_write_txn(self) as c:
            yield c

    monkeypatch.setattr(MiniSessionDB, "_write_txn", gated_write_txn)

    def opener():
        try:
            barrier.wait()
            MiniSessionDB(db_path)
        except Exception as e:                      # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=opener) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    # 只有一个连接真正进了迁移，其余被二次检查挡回；删掉二次检查这里会变成 == n
    assert len(calls) == 1
    assert _user_version(db_path) == SCHEMA_VERSION
    assert "compactions" in _tables(db_path)
    assert _columns(db_path, "compactions") == {
        "session_id", "first_kept_seq", "summary", "created_at"
    }
