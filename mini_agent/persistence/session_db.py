import json
import sqlite3
import time
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

