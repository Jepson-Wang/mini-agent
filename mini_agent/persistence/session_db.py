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

    def end_session(self, session_id) -> None:
        """
        __main__.py 正常退出时会调它
        """
        with self._write_txn() as c:
            c.execute(
                "UPDATE sessions SET status='done',updated_at=? WHERE session_id = ?;",
                (int(time.time()),session_id,)
            )

    def _sanitize_dangling_tool_calls(self, messages: list[Message]) -> list[Message]:
        """
        判断一对assistant - tool call 是否是正常结束的
        正常结束就原样返回，否则就删除这这一对messages
        """
        # 第一遍:把所有 tool result 按 tool_call_id 建索引(不管它在哪个位置)
        results_by_id = {}
        for msg in messages:
            if msg.role == "tool":
                results_by_id[msg.tool_call_id] = msg

        # 第二遍:逐轮认领
        kept = []
        for msg in messages:
            if msg.role == "tool":
                continue  # tool 消息不单独保留,由所属轮次带出

            if msg.role != "assistant" or not msg.tool_calls:
                kept.append(msg)  # system/user/纯文本 assistant,天然完整
                continue

            expected_ids = [tc.id for tc in msg.tool_calls]
            claimed = [results_by_id.get(tid) for tid in expected_ids]

            if all(r is not None for r in claimed):
                kept.append(msg)
                kept.extend(claimed)  # ★ 按 expected_ids 顺序输出,顺便修复乱序
            # else: 这一轮残缺 → 整轮丢弃(assistant 和它那些孤儿 result 都不进 kept)

        return kept

    def recover(self,session_id) -> list[Message]:
        """
        从磁盘上的事实,重建出一个"可以安全地继续跑主循环"的内存状态。
        就这一件。它不执行工具、不调 LLM、不写任何东西(除了可能改 session status)。
        它是纯读 + 重建。
        """
        c = self.conn
        rows = c.execute(
            "SELECT content FROM MESSAGES WHERE session_id = ? ORDER BY seq;",(session_id,)
        ).fetchall()
        messages = [Message.model_validate_json(row[0]) for row in rows]
        messages = self._sanitize_dangling_tool_calls(messages)
        return messages

    # 幂等表的相关操作，两个方法分别用于获取tool结果和记录tool结果

    def get_executed_result(self, key: str) -> dict | None:
        """查幂等缓存。命中返回 result dict,未命中返回 None。纯读,不开写事务。"""
        if not key:
            raise PersistenceError("get_executed_result: key 不能为空")
        row = self.conn.execute(
            "SELECT result FROM executed_keys WHERE idempotency_key = ?;", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def record_executed_key(self, key: str, session_id: str, result: dict) -> tuple[bool, dict]:
        """
        返回 (is_first, canonical_result)。
        - 用 ON CONFLICT(idempotency_key) DO NOTHING,看 rowcount 判断 is_first。
        - is_first=True  → canonical 就是你传进来的 result
        - is_first=False → 你必须把表里那条已存在的 result 读出来返回
          ★ 关键:这个"读回来"的 SELECT,必须和上面的 INSERT 在同一个
            BEGIN IMMEDIATE 事务里 —— 想清楚为什么(如果分开,读回来的
            权威值可能又被第三个并发写者改掉吗?或者说,你能保证读到的
            就是你 INSERT 时撞上的那一条吗?)
        """
        if not isinstance(result,dict):
            raise PersistenceError("record_executed_key: result应该为字典格式")
        result_json = json.dumps(result)

        with self._write_txn() as c:
            cur = c.execute(
                "INSERT INTO executed_keys "
                "(idempotency_key, session_id, result, created_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(idempotency_key) DO NOTHING;",
                (key, session_id, result_json, int(time.time()),),
            )
            if cur.rowcount == 1:
                return True,result
            elif cur.rowcount == 0:
                # 将INSERT 和 SELECT 放入同一个事务中，原因：
                # 1. 少一次锁获取、少一次事务开销。 事务已经开着了,SELECT 顺手就做了
                # 2. (更重要)不要让代码的正确性,依赖一个"读代码的人看不见的假设"。
                row = c.execute(
                    "SELECT result FROM executed_keys WHERE idempotency_key = ?;", (key,)
                ).fetchone()
                if row is None:
                    raise PersistenceError(f"executed_keys 不变量被破坏: key={key} 冲突但不存在")
                return False, json.loads(row[0])

