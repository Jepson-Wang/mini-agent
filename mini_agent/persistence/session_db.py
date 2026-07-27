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

# _create_tables 用它判断"表是否已就绪"，从而跳过写锁。改动建表语句时要同步改这里。
_TABLES = ("sessions", "messages", "executed_keys")

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
        # 整个构造过程都要收进 _db_guard:它是持久化层的边界,边界上只能抛
        # PersistenceError。漏一个裸 sqlite3 异常出去,调用方那句
        # `except PersistenceError` 就白写了 —— __main__ 里那条"数据还在、可以开
        # 新会话"的友好提示曾经就是这么失效的(库文件损坏 → 裸 DatabaseError)。
        with self._db_guard(f"打开数据库({path})", hint="(路径不存在 / 无权限 / 文件损坏)"):
            self.conn = sqlite3.connect(
                path, isolation_level=None, check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
            self._txn_lock = threading.RLock()
            self._configure()
        self._create_tables()

    def _configure(self):
        """连接级设置。注意 sqlite3.connect() 是惰性的——它不读文件头,所以
        "文件不是数据库"这类损坏要到这里第一条 PRAGMA 才会暴露。
        本方法由 __init__ 在 _db_guard 内调用,自己不再重复包一层。
        """
        c = self.conn
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA foreign_keys=ON;")  # SQLite 默认关
        c.execute("PRAGMA busy_timeout=5000;")  # 拿不到锁时等 5s 再报 BUSY
        c.execute("PRAGMA synchronous=NORMAL;")

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
        """建表。★ 先读后写:表已就绪时直接返回,不碰写锁。

        原本无条件走 _write_txn,意味着**哪怕只想读**,构造一个 handle 也要抢
        BEGIN IMMEDIATE 的写锁 —— 别人正在写时,一个纯读的 --resume 会被卡满
        busy_timeout(实测 5.5s)甚至失败。而 WAL 下读者本该永不被写者阻塞,这等于
        把 WAL 的好处在门口就丢掉了。M4 多进程共享 db 时这会变成常态。

        竞态安全:两个进程可能都看到表不全、都去建,但 CREATE TABLE IF NOT EXISTS
        幂等,且在 BEGIN IMMEDIATE 里串行,最多白跑一次。
        """
        with self._db_guard("_create_tables"):
            placeholders = ",".join("?" * len(_TABLES))
            n = self.conn.execute(
                f"SELECT COUNT(*) FROM sqlite_master "
                f"WHERE type='table' AND name IN ({placeholders});",
                _TABLES,
            ).fetchone()[0]
            if n == len(_TABLES):
                return

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

    def session_exists(self, session_id: str) -> bool:
        """
        用户 --resume sess_xxx 时先校验，不存在就早报错
        """
        with self._db_guard(f"session_exists(session={session_id})"):
            row = self.conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1;", (session_id,)
            ).fetchone()
        return row is not None

    def ensure_session(
        self, session_id: str, parent_session_id: str | None = None, depth: int = 0
    ) -> bool:
        """幂等地保证 sessions 表里有这一行；返回本次是否**新建**。

        服务化之后 session 身份由调用方（Go）铸造，我们这一行只是它的从属子记录：
        存在的意义是满足 messages / executed_keys 的外键约束，外加存 depth/parent。
        所以这里**不 mint id**，给什么 id 就落什么 id —— 这正是它和 create_session
        的分工（后者自己 mint，供测试和 CLI 用）。

        ★ 先探一次再决定要不要写。 每一轮对话开头都会调它，而绝大多数请求落在
          已存在的 session 上；探到就直接返回，全程不碰写锁。否则 WAL「读者永不
          被写者阻塞」的好处会在每一轮开头都被丢掉一次。同 _create_tables 的套路。

        竞态安全：两个进程可能都探到不存在、都去 INSERT，ON CONFLICT DO NOTHING
        保证幂等，最多白跑一次；rowcount 如实反映"这行到底是不是我建的"。
        """
        if not session_id:
            raise PersistenceError("ensure_session: session_id 不能为空")

        if self.session_exists(session_id):   # 自带 _db_guard，不必再套一层
            return False

        now = int(time.time())
        with self._db_guard(f"ensure_session(session={session_id})"):
            with self._write_txn() as c:
                cur = c.execute(
                    "INSERT INTO sessions "
                    "(session_id, status, created_at, updated_at, "
                    " parent_session_id, depth, spawn_budget_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0) "
                    "ON CONFLICT(session_id) DO NOTHING;",
                    (session_id, "running", now, now, parent_session_id, depth),
                )
                # rowcount 在 execute 后即确定，趁事务内读掉（同 _transition_status）
                created = cur.rowcount == 1
        return created

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

    def end_session(self, session_id: str) -> None:
        """__main__.py 退出时（正常 / Ctrl-C / 异常）在 finally 里调它。

        CAS：只把 running 的会话置为 done。重复调用、或会话已是终态时是安全的
        no-op（只告警不抛）——它跑在 finally 里，一旦抛异常会盖掉 try 里真正的错误。
        """
        if not self._transition_status(session_id, "running", "done"):
            logger.warning(
                "end_session: 会话 %s 不在 running 态（已结束或不存在），跳过",
                session_id,
            )

    def reopen_session(self, session_id: str) -> bool:
        """--resume 时把会话从 done 拉回 running。补上状态机缺的那条反向边。

        没有它，--resume 一个已正常退出的会话会留下两个毛病：续聊全程 status 写着
        done（标签撒谎，session list / M4 存活检查全都不可信），且退出时 finally 里的
        end_session 必然 CAS 失败、打一条假告警——那条告警本是为「重复调用」设计的，
        在正常路径上必响就成了狼来了，真出事时没人会看。

        ★ 返回 False 在这里的含义是唯一的。 调用方（__main__）已经先用
        session_exists 排除了「会话不存在」，所以 CAS 没命中只剩一种解释：
        **它当前就是 running** —— 要么上次被 kill 没走完 finally（合法，正是崩溃
        恢复要处理的情况），要么另一个进程正开着它（并发 --resume，危险）。
        本层区分不了这两者，所以只如实返回、不抛，把判断留给调用方。
        M4 父子代理共享同一个 db 时，这里要升级成带 owner/heartbeat 的租约才能真正
        分辨；那之前它至少是个探测点，别把这个信息浪费掉。
        """
        return self._transition_status(session_id, "done", "running")

    # ---------- 崩溃恢复 ----------

    def _sanitize_dangling_tool_calls(self, messages: list[Message]) -> list[Message]:
        """把每一轮 assistant(tool_calls) 补足成配对完整、可安全发给 DeepSeek 的形状。

        ★ 补桩,不是丢弃。 残缺的那一轮**绝不能删**:删掉 = 抹掉「这个工具可能
        已经跑过」这个事实,模型看不见就会重做一遍(邮件重发、文件重写)。正确解法
        是给每个没有结果的 tool_call 补一条 role="tool" 消息 —— 配对补齐(不 400),
        同时如实告诉模型「这次调用结果未知」,让它自己决定要不要重试。
        补桩前先查幂等表:命中说明崩在「已 record_executed_key、还没 append_message」
        之间,能拿回真结果,一次都不用重跑。

        ★ 补桩 != 重跑。 本方法执行零个 handler、零次 LLM 调用,纯读 + 纯计算。
        恢复过程一旦有副作用,你就得为恢复写恢复,那是个无底洞。

        不变量:在【串行】主循环里,_execute_tool_calls 会把一轮的 N 个 tool 结果
        全部落库后才回到循环顶去取下一个 assistant。所以残缺的轮次后面不可能再挂
        消息 —— 悬空【至多一个,且必然在尾部】。这里仍然遍历全部消息而不是只看
        尾巴,是防御性的:M4 子代理 / M6 并行工具会打破这个串行前提,到那时只处理
        尾部就会漏。顺带,按 tool_calls 的顺序重排结果,乱序也能被修正。
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
                #
                # ★ 空 id 必须自己挡掉,不能交给 get_executed_result。
                #   那里的「key 不能为空」守卫是用来抓**调用方的编程错误**的,完全
                #   正当;但 recover 传进去的不是程序算出来的值,而是**从库里读出来
                #   的历史数据** —— ToolCall.id 允许空串,且 from_openai 在 SDK 漏
                #   给 id 时默认就是 ""。同一个守卫在这里会把一次数据异常升级成
                #   PersistenceError,让这个会话**永远 --resume 不了**。
                #   recover 面对历史数据的铁律:任何"这不该发生"都降级,不抛。
                if not tc.id:
                    # 没 id 就无从查表,也无从配对,直接补桩收场。
                    content = _INTERRUPTED
                    logger.warning("recover: tool_call 缺 id,无法配对,补 interrupted 桩")
                else:
                    cached = self.get_executed_result(tc.id)
                    content = cached.get("content") if isinstance(cached, dict) else None
                    # ★ 必须是字符串才认。 Message.content 是 str|None,塞个 dict/int
                    #   进去会抛 pydantic ValidationError —— 那不是 PersistenceError,
                    #   _db_guard 拦不住(它只认 sqlite3/json 异常),__main__ 的
                    #   except PersistenceError 也接不住,用户得到裸 traceback 且这个
                    #   会话永久 --resume 不了。
                    #   而 {"content": <str>} 只是 agent.py 和这里的口头约定:
                    #   record_executed_key 的签名收任意 dict,没有任何东西强制形状。
                    #   读侧必须自己扛住 —— 库里的东西是谁写的、什么时候写的,recover
                    #   一概不知道,只能假设它可能是任何形状。
                    if isinstance(content, str):
                        logger.warning("recover: tc=%s 用幂等表里的真结果补回", tc.id)
                    else:
                        if content is not None:
                            logger.warning(
                                "recover: tc=%s 幂等表里的 content 是 %s 不是 str,"
                                "按未命中处理", tc.id, type(content).__name__,
                            )
                        content = _INTERRUPTED
                        logger.warning("recover: tc=%s 结果丢失,补 interrupted 桩", tc.id)

                # 补出来的这条必须和真结果长得一模一样,否则照样 400
                kept.append(Message(role="tool", tool_call_id=tc.id, content=content))

        return kept

    def recover(self, session_id: str) -> list[Message]:
        """
        从磁盘上的事实,重建出一个"可以安全地继续跑主循环"的内存状态。
        就这一件。它不执行工具、不调 LLM、不写任何东西(除了可能改 session status)。
        它是纯读 + 重建。

        ★ 容错契约:recover 的输入是【历史数据】,不是程序状态。
          对程序状态该 fail-fast 的地方,对历史数据必须降级。库里的东西是谁写的、
          什么时候写的、被什么版本的代码写的,recover 一概不知道,只能假设它可能
          是任何形状。已经踩过四次同一个坑,每次症状都一样:一处"这不该发生"的
          检查被历史数据触发 → 抛异常 → 那个会话**永久** --resume 不了(坏数据
          在库里,重试一万次结果相同):
            1. messages 行 JSON 坏 → 跳过该行 + 告警
            2. executed_keys 行 JSON 坏 → 按未命中处理
            3. tool_call 的 id 是空串 → 直接补桩,不去踩空 key 守卫
            4. 幂等表里的 content 不是 str → 按未命中处理
                 (这条最阴:抛的是 pydantic ValidationError,不是 PersistenceError,
                  _db_guard 拦不住、__main__ 也接不住)

        降级的边界:影响【一条历史】的错误就地降级(跳过/补桩)+ 告警,整体照常返回;
        影响【访问历史的能力】的错误(整张表读不了、库损坏)才抛,交上层决定。

        对应地,这里没有任何重试:失败的全是确定性错误,同样的输入必然同样的结果。
        重试只在"两次尝试的输入可能不同"时才有意义。锁争用那类瞬时错误由
        PRAGMA busy_timeout 在 SQLite 层兜住,不必在应用层再套一层。
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
