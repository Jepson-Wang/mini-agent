import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

from pydantic import ValidationError

from mini_agent.log import get_logger
from mini_agent.persistence.schema import (
    SESSION_CREATE_SQL,
    MESSAGES_CREATE_SQL,
    EXECUTED_KEYS_CREATE_SQL,
)
from mini_agent.schema import Message

logger = get_logger(__name__)


class PersistenceError(Exception):
    """持久化层的统一异常。把底层 sqlite/json 异常包一层，附带上下文。"""


_VALID_ROLES = {"system", "user", "assistant", "tool"}

# 崩在"tool 结果没落库"的瞬间、且幂等表里也没有真结果时,用它补桩。
_INTERRUPTED = json.dumps(
    {"error": "interrupted: 上次运行崩溃，这个工具调用的结果未知"},
    ensure_ascii=False,
)


class MiniSessionDB:

    def __init__(self, path: str):
        # isolation_level=None → 关掉 Python 的隐式事务管理(见下方"坑")
        # check_same_thread=False → M1 的并行只读工具、M4 的子代理都会跨线程用同一个 db。
        #   但光开这个不够:两个线程同时 BEGIN IMMEDIATE 会撞
        #   "cannot start a transaction within a transaction"(那是连接级状态,
        #   不是数据库级锁)。所以写事务再用一把 RLock 串行化,见 _write_txn。
        self.conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._txn_lock = threading.RLock()
        self._configure()
        self._create_tables()

    def _configure(self):
        c = self.conn
        c.execute("PRAGMA journal_mode=WAL;")  # 你学过:redo 日志,崩溃可重放
        c.execute("PRAGMA foreign_keys=ON;")  # SQLite 默认关!你 messages 有 FK
        c.execute("PRAGMA busy_timeout=5000;")  # 拿不到锁时等 5s 再报 BUSY
        c.execute("PRAGMA synchronous=NORMAL;")  # ← 这个留给你决定

    # ---------- 异常收敛：底层 sqlite/json 异常 → PersistenceError ----------

    @contextmanager
    def _db_guard(self, what: str, hint: str = ""):
        """把底层异常统一翻译成 PersistenceError,附带"是哪个操作、什么上下文"。

        只翻译,不吞:异常照样往上抛,只是换了个类型 + 加了上下文,
        并用 `raise ... from e` 保住异常链(traceback 里还能看到原始 sqlite 错误)。

        为什么值得做:调用方(agent / __main__)不该 import sqlite3 才能写 except。
        持久化层是一个模块边界,边界上就该只暴露自己的异常类型。
        注意 PersistenceError 本身直接放行,避免被二次包裹成套娃。
        """
        tail = f" {hint}" if hint else ""
        try:
            yield
        except PersistenceError:
            raise  # 已经是本层异常,原样上抛
        except sqlite3.IntegrityError as e:
            # 外键缺失 / 主键冲突 —— 通常是调用方的逻辑错误
            raise PersistenceError(
                f"{what}: 完整性约束失败(外键缺失 / 主键冲突)。{tail} 原始错误: {e}"
            ) from e
        except sqlite3.OperationalError as e:
            # 锁超时 / 磁盘问题 / SQL 写错 —— 环境或 SQL 问题(这里也可以加重试)
            raise PersistenceError(
                f"{what}: 数据库操作失败(锁超时 / 磁盘 / SQL 错误)。{tail} 原始错误: {e}"
            ) from e
        except sqlite3.Error as e:
            # 兜底:sqlite3 的其余异常(DatabaseError / ProgrammingError ...)
            raise PersistenceError(
                f"{what}: sqlite 错误 {type(e).__name__}。{tail} 原始错误: {e}"
            ) from e
        except json.JSONDecodeError as e:
            # 库里存的 JSON 坏了 —— 数据损坏,不是程序 bug
            raise PersistenceError(
                f"{what}: 库中 JSON 无法反序列化(数据损坏)。{tail} 原始错误: {e}"
            ) from e

    @contextmanager
    def _write_txn(self):
        """一次原子写。BEGIN IMMEDIATE 立即拿写锁,失败即回滚。

        RLock:同一连接上不能并发开事务(见 __init__ 注释)。用 R 版而不是普通
        Lock,是为了让"不小心嵌套 _write_txn"暴露成 sqlite 的显式报错,
        而不是变成一个静默死锁。
        """
        with self._txn_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE;")  # 立即申请写锁,不拖到第一条写语句
            try:
                yield c
                c.execute("COMMIT;")
            except Exception:
                c.execute("ROLLBACK;")
                raise

    def _create_tables(self):
        with self._db_guard("_create_tables"):
            with self._write_txn() as c:
                c.execute(SESSION_CREATE_SQL)
                c.execute(MESSAGES_CREATE_SQL)
                c.execute(EXECUTED_KEYS_CREATE_SQL)

    # ---------- session 生命周期 ----------

    def create_session(self) -> str:
        """建一个新 session，返回 sid。append_message 的前置。"""
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        now = int(time.time())
        with self._db_guard("create_session"):
            with self._write_txn() as c:
                c.execute(
                    "INSERT INTO sessions (session_id, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?);",
                    (sid, "running", now, now),
                )
        return sid

    def session_exists(self, session_id) -> bool:
        """
        用户 --resume sess_xxx 时先校验，不存在就早报错
        """
        with self._db_guard(f"session_exists(session={session_id})"):
            row = self.conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1;", (session_id,)
            ).fetchone()
        return row is not None

    def end_session(self, session_id) -> None:
        """
        __main__.py 正常退出时会调它
        """
        with self._db_guard(f"end_session(session={session_id})"):
            with self._write_txn() as c:
                c.execute(
                    "UPDATE sessions SET status = 'done', updated_at = ? "
                    "WHERE session_id = ?;",
                    (int(time.time()), session_id),
                )

    # ---------- 崩溃恢复 ----------

    def _sanitize_dangling_tool_calls(self, messages: list[Message]) -> list[Message]:
        """把每一轮 assistant(tool_calls) 补足成配对完整、可安全发给 DeepSeek 的形状。

        不变量:在【串行】主循环里,_execute_tool_calls 会把一轮的 N 个 tool 结果
        全部落库后才回到循环顶去取下一个 assistant。所以一个残缺的轮次后面不可能
        再挂消息 —— 悬空【至多一个,且必然在尾部】。

        那为什么还遍历所有轮、而不是只查最后一个 assistant?
        防御:M4 子代理 / M6 并行只读工具将来会引入【乱序回填】,一旦上面的不变量
        被打破,只查尾部就会静默产出一个 400 的 transcript。遍历所有轮的成本可忽略
        (complete 轮走 fast path,只是 dict 查一下),换来的是不依赖这个不变量。

        补桩而非丢弃:残缺轮里已经落库的真实结果代表【已经发生的副作用】(邮件已发、
        文件已写),丢掉它 = 恢复后模型不知情、必重做。所以整轮保留,缺失的位置
        先查幂等表拿真结果,查不到才用 _INTERRUPTED 桩占位。
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

            kept.append(msg)

            for tc in msg.tool_calls:
                r = results_by_id.get(tc.id)
                if r is not None:
                    kept.append(r)  # 真结果,原样带出(顺便按 tool_calls 顺序修复乱序)
                    continue

                # 结果没落库 → 先看幂等表里有没有 handler 真跑出来的结果。
                # 命中说明崩在"已 record_executed_key、还没 append_message"之间,
                # 这一步是无损恢复;未命中才是真丢了。
                cached = self.get_executed_result(tc.id)
                content = cached.get("content") if isinstance(cached, dict) else None
                if content is not None:
                    logger.warning("recover: tc=%s 用幂等表里的真结果补回", tc.id)
                else:
                    content = _INTERRUPTED
                    logger.warning("recover: tc=%s 结果丢失,补 interrupted 桩", tc.id)

                # 补出来的这条必须和真结果长得一模一样,否则照样 400
                kept.append(Message(role="tool", tool_call_id=tc.id, content=content))

        return kept

    def recover(self, session_id) -> list[Message]:
        """
        从磁盘上的事实,重建出一个"可以安全地继续跑主循环"的内存状态。
        就这一件。它不执行工具、不调 LLM、不写任何东西(除了可能改 session status)。
        它是纯读 + 重建。

        容错:单行 content 损坏(JSON 坏了 / schema 对不上)时,跳过这一行并告警,
        而不是让整个 --resume 崩掉 —— 崩溃恢复本身就不该被一行脏数据打死。
        """
        with self._db_guard(f"recover(session={session_id})"):
            rows = self.conn.execute(
                "SELECT seq, content FROM messages WHERE session_id = ? ORDER BY seq;",
                (session_id,),
            ).fetchall()

        messages: list[Message] = []
        for row in rows:
            try:
                messages.append(Message.model_validate_json(row["content"]))
            except (ValidationError, ValueError) as e:
                # ValueError 覆盖 JSONDecodeError;两者都属于"这一行的数据坏了"
                logger.warning(
                    "recover: 跳过损坏的消息行 session=%s seq=%s: %s",
                    session_id,
                    row["seq"],
                    e,
                )

        return self._sanitize_dangling_tool_calls(messages)

    # ---------- 幂等表:两个方法分别用于获取 tool 结果和记录 tool 结果 ----------

    def get_executed_result(self, key: str) -> dict | None:
        """查幂等缓存。命中返回 result dict,未命中返回 None。纯读,不开写事务。"""
        if not key:
            raise PersistenceError("get_executed_result: key 不能为空")

        with self._db_guard(f"get_executed_result(key={key})"):
            row = self.conn.execute(
                "SELECT result FROM executed_keys WHERE idempotency_key = ?;", (key,)
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["result"])

    def record_executed_key(
        self, key: str, session_id: str, result: dict
    ) -> tuple[bool, dict]:
        """
        返回 (is_first, canonical_result)。
        - 用 ON CONFLICT(idempotency_key) DO NOTHING,看 rowcount 判断 is_first。
        - is_first=True  → canonical 就是你传进来的 result
        - is_first=False → 把表里那条已存在的 result 读出来返回(它才是权威值)
          ★ 关键:这个"读回来"的 SELECT,必须和上面的 INSERT 在同一个
            BEGIN IMMEDIATE 事务里 —— 否则读到的可能不是你刚撞上的那一条。
        """
        if not isinstance(result, dict):
            raise PersistenceError("record_executed_key: result应该为字典格式")

        try:
            # ensure_ascii=False:中文原样存,库里可读,也省一半空间
            result_json = json.dumps(result, ensure_ascii=False)
        except TypeError as e:
            # 只窄窄地包这一处的 TypeError,不放进 _db_guard —— 别让宽泛的
            # except TypeError 顺手吞掉真正的程序 bug
            raise PersistenceError(
                f"record_executed_key: result 不可 JSON 序列化: {e}"
            ) from e

        with self._db_guard(f"record_executed_key(key={key})"):
            with self._write_txn() as c:
                cur = c.execute(
                    "INSERT INTO executed_keys "
                    "(idempotency_key, session_id, result, created_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(idempotency_key) DO NOTHING;",
                    (key, session_id, result_json, int(time.time())),
                )
                if cur.rowcount == 1:
                    return True, result

                # rowcount == 0 → 这个 key 已经存在,读回权威结果
                # 把 INSERT 和 SELECT 放同一个事务里,原因:
                # 1. 少一次锁获取、少一次事务开销。事务已经开着了,SELECT 顺手就做了
                # 2. (更重要)不要让代码的正确性,依赖一个"读代码的人看不见的假设"。
                row = c.execute(
                    "SELECT result FROM executed_keys WHERE idempotency_key = ?;",
                    (key,),
                ).fetchone()
                if row is None:
                    raise PersistenceError(
                        f"executed_keys 不变量被破坏: key={key} 冲突但不存在"
                    )
                return False, json.loads(row["result"])

    # ---------- 消息落库 ----------

    def append_message(self, session_id: str, msg: Message) -> int:
        """
        写一条 message,返回它的 seq。
        约束:
        - seq 是 session 内自增(不是全局 rowid)。算 seq 和插入必须在同一个
          BEGIN IMMEDIATE 里,否则并发/重跑会算出同一个 seq 撞主键。
        - content 存整条 Message 的 JSON(model_dump_json,能无损 round-trip)。
        - 顺手把 sessions.updated_at 推进,这样"最近活跃的 session"可查。
        """
        if not session_id:
            raise PersistenceError("append_message: session_id 不能为空")
        if msg.role not in _VALID_ROLES:
            raise PersistenceError(f"append_message: 非法 role={msg.role!r}")

        now = int(time.time())
        with self._db_guard(
            f"append_message(session={session_id}, role={msg.role})",
            hint="(IntegrityError 通常是没先 create_session)",
        ):
            with self._write_txn() as c:
                row = c.execute(
                    "SELECT MAX(seq) FROM messages WHERE session_id = ?;",
                    (session_id,),
                ).fetchone()
                # 聚合函数必然返回一行,所以这里判的是 row[0] is None(空表),
                # 而不是 row is None
                seq = 0 if row[0] is None else row[0] + 1
                c.execute(
                    "INSERT INTO messages (session_id, seq, role, content, created_at) "
                    "VALUES (?, ?, ?, ?, ?);",
                    (session_id, seq, msg.role, msg.model_dump_json(), now),
                )
                # 和 INSERT 同一个事务:要么消息和 updated_at 一起生效,要么都不生效
                c.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?;",
                    (now, session_id),
                )
            return seq
