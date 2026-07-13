import json
import sqlite3
import time
import uuid
from contextlib import contextmanager

from persistence.schema import (
    SESSION_CREATE_SQL,
    MESSAGES_CREATE_SQL,
    EXECUTED_KEYS_CREATE_SQL,
)
from schema import Message


class PersistenceError(Exception):
    """持久化层的统一异常。把底层 sqlite/json 异常包一层，附带上下文。"""


_VALID_ROLES = {"system", "user", "assistant", "tool"}

class MiniSessionDB:

    def __init__(self, path: str):
        # isolation_level=None → 关掉 Python 的隐式事务管理(见下方"坑")
        self.conn = sqlite3.connect(path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self._configure()
        self._create_tables()

    def _configure(self):
        c = self.conn
        c.execute("PRAGMA journal_mode=WAL;")  # 你学过:redo 日志,崩溃可重放
        c.execute("PRAGMA foreign_keys=ON;")  # SQLite 默认关!你 messages 有 FK
        c.execute("PRAGMA busy_timeout=5000;")  # 拿不到锁时等 5s 再报 BUSY
        c.execute("PRAGMA synchronous=NORMAL;")  # ← 这个留给你决定

    def create_session(self) -> str:
        """建一个新 session，返回 sid。append_message 的前置。"""
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        now = int(time.time())
        with self._write_txn() as c:
            c.execute(
                "INSERT INTO sessions (session_id,status, created_at, updated_at) "                    
                "VALUES (?, ?, ?, ?);",(sid, "running", now, now,)
            )
        return sid

    @contextmanager
    def _write_txn(self):
        """一次原子写。BEGIN IMMEDIATE 立即拿写锁,失败即回滚。"""
        c = self.conn
        c.execute("BEGIN IMMEDIATE;")  # 立即申请写锁,不拖到第一条写语句
        try:
            yield c
            c.execute("COMMIT;")
        except Exception:
            c.execute("ROLLBACK;")
            raise

    def _create_tables(self):
        with self._write_txn():
            c = self.conn
            c.execute(SESSION_CREATE_SQL)
            c.execute(MESSAGES_CREATE_SQL)
            c.execute(EXECUTED_KEYS_CREATE_SQL)

    def session_exists(self, session_id) -> bool:
        """
        用户 - -resume sess_xxx 时先校验，不存在就早报错
        """
        c = self.conn
        row = c.execute(
            "SELECT 1 FROM sessions WHERE session_id=? LIMIT 1;",(session_id,)
        ).fetchone()
        return row is not None


