"""幂等键作用域修复：(session_id, assistant_seq, tool_index)。

旧 executed_keys 的主键只有模型给的 tool_call.id，recover 的配对也用一个
全会话共享的 {tool_call_id: msg} 索引。于是"模型复用 id"会造成三种串结果：
  1. 跨会话：B 撞上 A 的 id，崩溃恢复时拿回 A 的工具输出（隐私泄露）
  2. 会话内跨轮：第 3 轮复用第 1 轮的 id，恢复时拿回第 1 轮的结果
  3. ★ 不崩溃也会发生：id 重复时后写的覆盖先写的，每次 recover 都错配
     —— 无状态化以后每轮开头都走 recover，所以这是**正常路径**上的 bug

修法的核心：键里不含任何模型输入；tool 结果只从紧跟 assistant 的那段窗口里认领。
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager

import pytest

from mini_agent.persistence import session_db as session_db_module
from mini_agent.persistence.schema import SCHEMA_VERSION
from mini_agent.persistence.session_db import (
    _INTERRUPTED,
    MiniSessionDB,
    PersistenceError,
)
from mini_agent.schema import Message, ToolCall


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "state.db")


@pytest.fixture
def db(db_path):
    return MiniSessionDB(db_path)


def _tc(call_id: str) -> ToolCall:
    return ToolCall(id=call_id, name="send_email", arguments={})


def _assistant(db, sid, *ids) -> int:
    """落一条 assistant(tool_calls)，返回它的 seq —— 幂等表的外键要指向它。"""
    return db.append_message(
        sid, Message(role="assistant", tool_calls=[_tc(i) for i in ids])
    )


def _tool(db, sid, call_id, content) -> int:
    return db.append_message(
        sid, Message(role="tool", tool_call_id=call_id, content=content)
    )


def _pairs(msgs):
    return [(m.role, m.tool_call_id, m.content) for m in msgs]


# --------------------------------------------------------------------------
# 1-3. 三个串结果的 bug
# --------------------------------------------------------------------------

def test_cross_session_same_id_gets_own_result_not_others(db, db_path):
    """会话 A、B 用了同一个 call_1，B 崩在"记幂等表 → 写 tool 消息"之间。

    旧代码：B 的写入被 ON CONFLICT DO NOTHING 静默吞掉，recover(B) 补回的是
    **A 的工具输出** —— 一个会话读到另一个会话的数据。
    """
    a = db.create_session()
    seq_a = _assistant(db, a, "call_1")
    db.record_executed_key(a, seq_a, 0, "call_1", {"content": "A 的秘密"})
    _tool(db, a, "call_1", "A 的秘密")

    b = db.create_session()
    seq_b = _assistant(db, b, "call_1")
    is_first, _ = db.record_executed_key(b, seq_b, 0, "call_1", {"content": "B 的结果"})
    assert is_first is True          # 不同会话的同 id 不再撞键
    # ← B 在这里被 kill

    revived = MiniSessionDB(db_path).recover(b)

    assert revived[-1].content == "B 的结果"
    assert "A 的秘密" not in [m.content for m in revived]


def test_same_session_reused_id_across_rounds_gets_its_own_round(db, db_path):
    """第 1 轮和第 3 轮都叫 call_0，第 3 轮崩在窗口里 → 拿回第 3 轮的结果。

    只给键加 session_id 前缀修不了这一条，这正是选 (session, assistant_seq, idx)
    的原因。
    """
    sid = db.create_session()
    s1 = _assistant(db, sid, "call_0")
    db.record_executed_key(sid, s1, 0, "call_0", {"content": "r1"})
    _tool(db, sid, "call_0", "r1")
    db.append_message(sid, Message(role="user", content="q3"))
    s3 = _assistant(db, sid, "call_0")
    db.record_executed_key(sid, s3, 0, "call_0", {"content": "r3"})
    # ← 崩在这里

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived) == [
        ("assistant", None, None),
        ("tool", "call_0", "r1"),
        ("user", None, "q3"),
        ("assistant", None, None),
        ("tool", "call_0", "r3"),
    ]


def test_reused_id_without_any_crash_pairs_each_round_with_its_own_result(db, db_path):
    """★ 不崩溃的错配：两轮同 id、两轮结果都好好地在库里。

    旧代码的 results_by_id 是全会话共享的 dict，后写的覆盖先写的 —— 两个
    assistant 都认领第 3 轮的结果，第 1 轮的结果凭空消失。每次 recover 都这样，
    根本不需要崩溃。
    """
    sid = db.create_session()
    _assistant(db, sid, "call_0")
    _tool(db, sid, "call_0", "r1")
    db.append_message(sid, Message(role="user", content="q3"))
    _assistant(db, sid, "call_0")
    _tool(db, sid, "call_0", "r3")

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.content for m in revived if m.role == "tool"] == ["r1", "r3"]
    assert [m.role for m in revived] == ["assistant", "tool", "user", "assistant", "tool"]


# --------------------------------------------------------------------------
# 4-5. 窗口内配对：每个 id 一个先进先出队列
# --------------------------------------------------------------------------

def test_duplicate_ids_in_one_assistant_each_claim_their_own_result(db, db_path):
    """同一条 assistant 里两个 tool_call 同 id、两条结果都在 → 按先进先出各配各的。

    窗口内若按 dict 匹配，两个 tc 都会拿到后一条结果 —— 第 3 个 bug 原样回来。
    """
    sid = db.create_session()
    _assistant(db, sid, "dup", "dup")
    _tool(db, sid, "dup", "r0")
    _tool(db, sid, "dup", "r1")

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.content for m in revived[1:]] == ["r0", "r1"]


def test_duplicate_ids_crash_before_second_backfills_from_table(db, db_path):
    """同 id 两个 tc，崩在第 2 个的 tool 消息落库之前：第 1 个保留真结果，
    第 2 个按 (seq, idx=1) 从幂等表取回 —— 不会去认领第 1 个的结果。"""
    sid = db.create_session()
    seq = _assistant(db, sid, "dup", "dup")
    db.record_executed_key(sid, seq, 0, "dup", {"content": "r0"})
    _tool(db, sid, "dup", "r0")
    db.record_executed_key(sid, seq, 1, "dup", {"content": "r1"})
    # ← 崩在这里

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.content for m in revived[1:]] == ["r0", "r1"]


def test_duplicate_ids_crash_before_second_without_record_stubs_it(db, db_path):
    sid = db.create_session()
    _assistant(db, sid, "dup", "dup")
    _tool(db, sid, "dup", "r0")

    revived = MiniSessionDB(db_path).recover(sid)

    assert [m.content for m in revived[1:]] == ["r0", _INTERRUPTED]


def test_partial_window_ignores_records_of_other_indexes(db, db_path):
    """[assistant(c1,c2), tool(c1)]，幂等表里只有 idx=0 的记录 → c2 补桩。

    证明补桩查的是 (seq, idx) 精确位置，不是"这个 assistant 有记录就当命中"。
    """
    sid = db.create_session()
    seq = _assistant(db, sid, "c1", "c2")
    db.record_executed_key(sid, seq, 0, "c1", {"content": "r1"})
    _tool(db, sid, "c1", "r1")

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived[1:]) == [("tool", "c1", "r1"), ("tool", "c2", _INTERRUPTED)]


# --------------------------------------------------------------------------
# 6-7. 读侧校验 / 空 id
# --------------------------------------------------------------------------

def test_stored_tool_call_id_mismatch_is_a_miss(db, db_path, caplog):
    """位置对上了、但存的 tool_call_id 和当前 tc.id 不一致 → 按未命中补桩并告警。

    tool_call_id 不在主键里，它的作用就是这道纵深防御。
    """
    sid = db.create_session()
    seq = _assistant(db, sid, "call_1")
    db.record_executed_key(sid, seq, 0, "call_other", {"content": "不该被取回"})

    with caplog.at_level("WARNING", logger="mini_agent.persistence.session_db"):
        revived = MiniSessionDB(db_path).recover(sid)

    assert revived[-1].content == _INTERRUPTED
    assert "call_other" in caplog.text


def test_empty_tool_call_id_backfills_real_result_from_table(db, db_path):
    """空 id 也照常查表 —— 键里不含 id，空不空和"能不能查"无关。

    旧代码对空 id 直接补桩，白白丢掉一个已经存下来的真结果，害模型重跑。
    """
    sid = db.create_session()
    seq = _assistant(db, sid, "")
    db.record_executed_key(sid, seq, 0, "", {"content": "邮件已发送"})

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived[1:]) == [("tool", "", "邮件已发送")]


def test_get_executed_result_accepts_empty_expected_id(db):
    """空 id 是合法输入：旧的空 key 守卫在新签名下会让会话永久 --resume 不了。"""
    sid = db.create_session()
    seq = _assistant(db, sid, "")
    assert db.get_executed_result(sid, seq, 0, "") is None      # 未命中，不抛
    db.record_executed_key(sid, seq, 0, "", {"content": "x"})
    assert db.get_executed_result(sid, seq, 0, "") == {"content": "x"}


# --------------------------------------------------------------------------
# 8-9. 窗口边界
# --------------------------------------------------------------------------

def test_corrupt_assistant_row_still_cuts_the_window(db, db_path):
    """assistant 那行 JSON 坏了被跳过，它后面的 tool 消息不能并进上一条 assistant 的窗口。

    截断看的是 role **列**，不是解析成功的消息。否则坏行一跳过，上一条
    assistant(c1,c2) 的窗口就会一路延伸，把坏 assistant 的 c2 结果认领走。
    """
    sid = db.create_session()
    _assistant(db, sid, "c1", "c2")                       # seq 0
    _tool(db, sid, "c1", "r1")                            # seq 1
    raw = sqlite3.connect(db_path, isolation_level=None)  # seq 2：坏掉的 assistant
    raw.execute(
        "INSERT INTO messages (session_id, seq, role, content, created_at) "
        "VALUES (?, 2, 'assistant', '{\"role\": \"assistant\", ', ?);",
        (sid, int(time.time())),
    )
    raw.close()
    _tool(db, sid, "c2", "属于坏 assistant 的结果")         # seq 3：孤儿

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived) == [
        ("assistant", None, None),
        ("tool", "c1", "r1"),
        ("tool", "c2", _INTERRUPTED),
    ]


def test_cancel_seal_after_partial_window_gets_stub_before_seal(db, db_path):
    """取消封口：a(tc0,tc1), t0, seal → tc1 的桩补在 seal 之前，配对仍然合法。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q"))
    _assistant(db, sid, "tc0", "tc1")
    _tool(db, sid, "tc0", "r0")
    db.append_message(sid, Message(role="assistant", content="(被取消)"))

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived) == [
        ("user", None, "q"),
        ("assistant", None, None),
        ("tool", "tc0", "r0"),
        ("tool", "tc1", _INTERRUPTED),
        ("assistant", None, "(被取消)"),
    ]


# --------------------------------------------------------------------------
# 11（持久化那一半）. first-write-wins 返回权威结果
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "col_role, body",
    [
        # JSON 是 tool、列写着 user：若照"列"截断、照"JSON"认领，这条 tool 会被
        # 当普通消息原样留下 —— 一条不跟在 assistant 后面的 tool，之后每次请求都 400
        ("user", '{"role": "tool", "tool_call_id": "c9", "content": "错位"}'),
        # JSON 是 user、列写着 tool：会被当孤儿 tool 丢掉，日志还写错原因
        ("tool", '{"role": "user", "content": "错位"}'),
    ],
)
def test_role_column_disagreeing_with_content_is_a_corrupt_row(db, db_path, col_role, body):
    """role 列与 JSON 里的 role 不一致 → 整行按坏行跳过，两个来源只认一个口径。"""
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="hi"))   # seq 0
    raw = sqlite3.connect(db_path, isolation_level=None)          # seq 1：口径打架
    raw.execute(
        "INSERT INTO messages (session_id, seq, role, content, created_at) "
        "VALUES (?, 1, ?, ?, ?);",
        (sid, col_role, body, int(time.time())),
    )
    raw.close()
    db.append_message(sid, Message(role="assistant", content="ok"))  # seq 2

    revived = MiniSessionDB(db_path).recover(sid)

    assert _pairs(revived) == [("user", None, "hi"), ("assistant", None, "ok")]
    assert all(m.content != "错位" for m in revived)


def test_corrupt_canonical_row_on_conflict_degrades_instead_of_raising(db, db_path):
    """冲突分支读回的权威行是坏 JSON → 返回 (False, {})，不抛。

    走到这里时工具已经执行完了；为一行坏掉的旧记录让整轮失败是最坏的取舍。
    """
    sid = db.create_session()
    seq = _assistant(db, sid, "call_1")
    db.record_executed_key(sid, seq, 0, "call_1", {"content": "第一次"})
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute("UPDATE executed_keys SET result = '{坏' WHERE session_id = ?;", (sid,))
    raw.close()

    assert db.record_executed_key(sid, seq, 0, "call_1", {"content": "重放"}) == (False, {})


def test_record_executed_key_returns_canonical_on_conflict(db):
    sid = db.create_session()
    seq = _assistant(db, sid, "call_1")

    assert db.record_executed_key(sid, seq, 0, "call_1", {"content": "第一次"}) == (
        True, {"content": "第一次"},
    )
    assert db.record_executed_key(sid, seq, 0, "call_1", {"content": "重放"}) == (
        False, {"content": "第一次"},
    )
    # 同一条 assistant 的另一个下标不是冲突
    assert db.record_executed_key(sid, seq, 1, "call_1", {"content": "idx1"})[0] is True


def test_record_executed_key_requires_existing_assistant_message(db):
    """外键：幂等记录必须指向一条真实存在的消息，不能凭空写。"""
    sid = db.create_session()
    with pytest.raises(PersistenceError) as exc:
        db.record_executed_key(sid, 42, 0, "call_1", {"content": "x"})
    assert isinstance(exc.value.__cause__, sqlite3.IntegrityError)


# --------------------------------------------------------------------------
# 10. 迁移：PRAGMA user_version
# --------------------------------------------------------------------------

_V0_EXECUTED_KEYS_SQL = """
CREATE TABLE executed_keys (
    idempotency_key TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    result          TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
"""


def _make_legacy_db(path: str) -> None:
    """造一份升级前的库：user_version=0，executed_keys 是旧结构，带一行旧数据。"""
    from mini_agent.persistence.schema import MESSAGES_CREATE_SQL, SESSION_CREATE_SQL

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


def test_new_db_is_stamped_with_schema_version(db_path):
    MiniSessionDB(db_path)
    assert _user_version(db_path) == SCHEMA_VERSION
    assert "tool_index" in _columns(db_path, "executed_keys")


def test_legacy_db_is_migrated_and_old_table_kept(db_path):
    """旧库打开后：版本升到当前版本，旧表改名保留（不做不可逆的事），新表可读写。"""
    _make_legacy_db(db_path)

    db = MiniSessionDB(db_path)

    assert _user_version(db_path) == SCHEMA_VERSION
    assert "executed_keys_legacy_v0" in _tables(db_path)
    assert "idempotency_key" in _columns(db_path, "executed_keys_legacy_v0")
    assert "idempotency_key" not in _columns(db_path, "executed_keys")
    raw = sqlite3.connect(db_path)
    assert raw.execute("SELECT result FROM executed_keys_legacy_v0;").fetchone()[0] == (
        '{"content": "旧"}'
    )
    raw.close()

    sid = db.create_session()
    seq = _assistant(db, sid, "call_1")
    assert db.record_executed_key(sid, seq, 0, "call_1", {"content": "新"})[0] is True
    assert db.get_executed_result(sid, seq, 0, "call_1") == {"content": "新"}


def test_migrated_db_reopens_on_fast_path_without_write_lock(db_path):
    _make_legacy_db(db_path)
    MiniSessionDB(db_path)                   # 迁移

    hog = sqlite3.connect(db_path, isolation_level=None)
    hog.execute("BEGIN IMMEDIATE;")
    try:
        t0 = time.perf_counter()
        MiniSessionDB(db_path)               # 快路径：不该碰写锁
        elapsed = time.perf_counter() - t0
    finally:
        hog.execute("ROLLBACK;")
        hog.close()

    assert elapsed < 1.0, f"重开已迁移的库花了 {elapsed:.2f}s，说明还在抢写锁"


def test_failed_migration_rolls_back_atomically(db_path, monkeypatch):
    """迁移中途失败（改名之后、建新表时）→ 整段撤销：版本仍是 0，旧表原名原样。

    user_version 和 ALTER TABLE 都在同一个 BEGIN IMMEDIATE 里，随事务回滚。
    """
    _make_legacy_db(db_path)
    monkeypatch.setattr(
        session_db_module, "EXECUTED_KEYS_CREATE_SQL", "CREATE TABLE executed_keys ("
    )

    with pytest.raises(PersistenceError):
        MiniSessionDB(db_path)

    assert _user_version(db_path) == 0
    assert "executed_keys_legacy_v0" not in _tables(db_path)
    assert "idempotency_key" in _columns(db_path, "executed_keys")


def test_migration_tolerates_legacy_name_already_taken(db_path):
    _make_legacy_db(db_path)
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute("CREATE TABLE executed_keys_legacy_v0 (x INTEGER);")   # 名字被占了
    raw.close()

    MiniSessionDB(db_path)

    tables = _tables(db_path)
    assert _user_version(db_path) == SCHEMA_VERSION
    assert _columns(db_path, "executed_keys_legacy_v0") == {"x"}       # 占位的没被动
    renamed = [t for t in tables if t.startswith("executed_keys_legacy_v0_")]
    assert len(renamed) == 1
    assert "idempotency_key" in _columns(db_path, renamed[0])


def test_concurrent_migration_happens_once(db_path, monkeypatch):
    """多个连接同时打开旧库：BEGIN IMMEDIATE 里的二次检查保证只迁一次。

    ★ 直接数"真正动手迁移"的次数，而不是只看结果。只看结果证明不了二次检查：
      _retire_legacy_executed_keys 自己会判断有没有 idempotency_key 列、建表又是
      IF NOT EXISTS，于是把二次检查删掉，慢路径重复跑也一样"结果正确" —— 那样
      测的只是"迁移重复执行无害"。数次数才能在二次检查被删时变红。
    """
    _make_legacy_db(db_path)
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
    #   已经读到 v1 直接返回，根本走不到慢路径 —— 删掉二次检查测试照样绿（实测）。
    #   闸设在 _write_txn 入口：谁都得等 n 个线程**全部**过完快路径（都读到 v0）
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
    assert len(calls) == 1          # 只有一个连接真正进了迁移，其余被二次检查挡回
    assert _user_version(db_path) == SCHEMA_VERSION
    legacy = [t for t in _tables(db_path) if t.startswith("executed_keys_legacy")]
    assert legacy == ["executed_keys_legacy_v0"]
    raw = sqlite3.connect(db_path)
    assert raw.execute("SELECT COUNT(*) FROM executed_keys_legacy_v0;").fetchone()[0] == 1
    raw.close()


def test_db_from_newer_version_is_refused(db_path):
    """user_version 比代码认识的还新（用新库跑旧代码）→ 说清原因地拒绝，不去猜。"""
    MiniSessionDB(db_path)
    raw = sqlite3.connect(db_path, isolation_level=None)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1};")
    raw.close()

    with pytest.raises(PersistenceError, match="更新"):
        MiniSessionDB(db_path)
