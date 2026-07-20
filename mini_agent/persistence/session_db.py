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

    def create_session(
        self, parent_session_id: str | None = None, depth: int = 0
    ) -> str:
        """建一个新 session，返回 sid。append_message 的前置。

        M4 谱系链：CLI 主会话（root）用默认值 parent_session_id=None、depth=0；
        子代理由父会话创建，传入父 sid 和 depth+1。spawn_budget_used 一律从 0 起，
        之后只有 root 行会被累加（全树 spawn 预算记在 root 上，见 M4）。
        """
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        now = int(time.time())
        with self._db_guard("create_session"):
            with self._write_txn() as c:
                c.execute(
                    "INSERT INTO sessions "
                    "(session_id, status, created_at, updated_at, "
                    " parent_session_id, depth, spawn_budget_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0);",
                    (sid, "running", now, now, parent_session_id, depth),
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

    def _transition_status(
        self, session_id: str, from_status: str, to_status: str
    ) -> bool:
        """CAS 状态转移：仅当当前 status == from_status 时，才把它置为 to_status。

        返回本次是否真的转移成功（UPDATE 命中 1 行）。rowcount == 0 有两种可能：
        会话不存在，或它当前状态不是 from_status（已被别的路径改过 / 重复调用）。
        本方法**不抛**这种情况，交给调用方判断——「转移没发生」常常是合法的幂等
        重入，不该当错误。

        为什么用 CAS 而不是无条件 UPDATE：状态转移必须「从某个已知态出发」才合法。
        无条件 SET status='done' 会把一个已经 failed/killed 的会话悄悄改回 done、
        丢掉真实结局；带上 WHERE status = from_status，非法转移自然命中 0 行、被
        rowcount 拦下。M4 里父代理 kill 子代理(running→killed)与子代理自己正常结束
        (running→done)会竞争同一行，CAS 保证只有一个赢、且赢家可判定。
        """
        now = int(time.time())
        with self._db_guard(
            f"_transition_status(session={session_id}, {from_status}->{to_status})"
        ):
            with self._write_txn() as c:
                cur = c.execute(
                    "UPDATE sessions SET status = ?, updated_at = ? "
                    "WHERE session_id = ? AND status = ?;",
                    (to_status, now, session_id, from_status),
                )
                # rowcount 在 execute 后即确定（DML 立刻可读），趁事务内读掉
                changed = cur.rowcount == 1
        return changed

    def end_session(self, session_id) -> None:
        """__main__.py 退出时（正常 / Ctrl-C / 异常）在 finally 里调它。

        CAS：只把 running 的会话置为 done。重复调用、或会话已是终态时是安全的
        no-op（只告警不抛）——它跑在 finally 里，一旦抛异常会盖掉 try 里真正的错误。
        """
        if not self._transition_status(session_id, "running", "done"):
            logger.warning(
                "end_session: 会话 %s 不在 running 态（已结束或不存在），跳过",
                session_id,
            )

    # ---------- 崩溃恢复 ----------

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
        """查幂等缓存。命中返回 result dict,未命中返回 None。纯读,不开写事务。

        损坏的行按「未命中」处理,不抛。理由:本方法唯一的调用方是 recover 的补桩
        分支,它对 None 的语义已经是「查不到真结果,补 interrupted 桩」——一行坏
        JSON 的正确降级就是走同一条路。recover 对 messages 表的坏行也是跳过+告警,
        两张表的容错态度必须一致,否则一行脏数据就能打死整个 --resume。
        ★ json.loads 必须留在 _db_guard **外面**:guard 里有一个
          except json.JSONDecodeError 分支,放进去会被它抢先翻译成
          PersistenceError,下面这个 except ValueError 就永远接不到了。
        """
        if not key:
            raise PersistenceError("get_executed_result: key 不能为空")

        with self._db_guard(f"get_executed_result(key={key})"):
            row = self.conn.execute(
                "SELECT result FROM executed_keys WHERE idempotency_key = ?;", (key,)
            ).fetchone()
            if row is None:
                return None
            raw = row["result"]

        try:
            return json.loads(raw)
        except ValueError as e:
            # ValueError 覆盖 JSONDecodeError
            logger.warning(
                "get_executed_result: 幂等表行损坏 key=%s,按未命中处理: %s", key, e
            )
            return None

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
