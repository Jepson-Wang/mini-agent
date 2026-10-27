import json
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager

from pydantic import ValidationError

from mini_agent.log import get_logger
from mini_agent.persistence.schema import (
    SCHEMA_VERSION,
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

# 旧版 executed_keys 迁移时改名到这里。只改名不 DROP：自动迁移不该做不可逆的事。
_LEGACY_EXECUTED_KEYS = "executed_keys_legacy_v0"

# 崩在"tool 结果没落库"的瞬间、且幂等表里也没有真结果时,用它补桩。
_INTERRUPTED = json.dumps(
    {"error": "interrupted: 上次运行崩溃，这个工具调用的结果未知"},
    ensure_ascii=False,
)

class MiniSessionDB:

    def __init__(self, path: str):
        # isolation_level=None → 关掉 Python 的隐式事务管理(见下方"坑")
        # check_same_thread=False → M1 的并行只读工具、M4 的子代理、gRPC surface
        #   都会跨线程用同一个 db。
        #
        #   ★ 但这个参数只是**关掉检查**,它不会让 Connection 变成线程安全的。
        #   底层 sqlite 是 serialized 模式没错,可 Python 的 sqlite3.Connection
        #   自己带一个语句缓存,而 conn.execute() 是"建游标 + 用缓存"的快捷方式 ——
        #   多个线程同时调它就是在并发改同一份缓存。实测(四个线程各 300 轮读写)
        #   会随机踩到:
        #     - sqlite3.InterfaceError: bad parameter or other API misuse
        #     - IndexError: tuple index out of range(抛在 sqlite3 模块内部)
        #     - **读回来的行内容是错的**:recover() 拿到 content=None 或空串
        #   最后一条最危险:recover() 的既定策略是"坏行跳过 + 记日志,绝不抛",
        #   所以这个竞态的表现是**一条消息从 transcript 里静悄悄消失**,没有任何
        #   人报错。这正是 _SessionLocks 拼命要防的那种损坏,只不过来自更下面一层。
        #
        #   另外两个线程同时 BEGIN IMMEDIATE 还会撞
        #   "cannot start a transaction within a transaction"(那是连接级状态,
        #   不是数据库级锁)。
        #
        #   所以:**所有**碰 self.conn 的地方都必须先拿 _conn_lock,读也一样。
        #   见 _read / _write_txn。
        # 整个构造过程都要收进 _db_guard:它是持久化层的边界,边界上只能抛
        # PersistenceError。漏一个裸 sqlite3 异常出去,调用方那句
        # `except PersistenceError` 就白写了 —— __main__ 里那条"数据还在、可以开
        # 新会话"的友好提示曾经就是这么失效的(库文件损坏 → 裸 DatabaseError)。
        with self._db_guard(f"打开数据库({path})", hint="(路径不存在 / 无权限 / 文件损坏)"):
            self.conn = sqlite3.connect(
                path, isolation_level=None, check_same_thread=False
            )
            self.conn.row_factory = sqlite3.Row
            self._conn_lock = threading.RLock()
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
    def _read(self):
        """一次读。**拿的是和写同一把锁**。

        读也要串行,理由见 __init__ 里那段:并发调 conn.execute() 会踩到
        Connection 的语句缓存竞态,轻则 InterfaceError,重则读回错误的行内容
        而没有任何人报错。这里没有"读不需要加锁"的余地。

        代价是一个进程内的读被串行化了。可以接受:sqlite3 在 step 期间会放开
        GIL,而且这个库的量级本来就不靠单进程并发读撑吞吐。真需要并发读的话,
        正确做法是每个线程一条连接(WAL 下读者不会被写者阻塞),不是把锁去掉。
        """
        with self._conn_lock:
            yield self.conn

    @contextmanager
    def _write_txn(self):
        """一次原子写。BEGIN IMMEDIATE 立即拿写锁,失败即回滚。

        RLock:同一连接上不能并发开事务(见 __init__ 注释)。用 R 版而不是普通
        Lock,是为了让"不小心嵌套 _write_txn"暴露成 sqlite 的显式报错,
        而不是变成一个静默死锁。R 版同时也让 _read 能安全地嵌在写事务里。
        """
        with self._conn_lock:
            c = self.conn
            c.execute("BEGIN IMMEDIATE;")  # 立即申请写锁,不拖到第一条写语句
            try:
                yield c
                c.execute("COMMIT;")
            except Exception:
                c.execute("ROLLBACK;")
                raise

    def ping(self) -> None:
        """自检：库还能读吗。读得到就正常返回，读不到抛 PersistenceError。

        给 gRPC 的 Health 用。查的是 sessions 表而不是 `SELECT 1`：后者只证明
        sqlite 这个库还在，连文件被删掉都发现不了（连接句柄还握着已删除的
        inode，照样返回 1）。读一张真表才会碰到页缓存之外的东西。

        LIMIT 0 是刻意的：只要求 sqlite 解析 + 打开表，不真取行，
        所以库多大都是常数开销 —— 这条路径会被探针按秒轮询。
        """
        with self._db_guard("数据库自检", hint="(文件被删 / 权限变了 / 磁盘故障)"):
            with self._read() as c:
                c.execute("SELECT session_id FROM sessions LIMIT 0;").fetchall()

    def _create_tables(self):
        """建表。★ 先读后写:表已就绪时直接返回,不碰写锁。

        原本无条件走 _write_txn,意味着**哪怕只想读**,构造一个 handle 也要抢
        BEGIN IMMEDIATE 的写锁 —— 别人正在写时,一个纯读的 --resume 会被卡满
        busy_timeout(实测 5.5s)甚至失败。而 WAL 下读者本该永不被写者阻塞,这等于
        把 WAL 的好处在门口就丢掉了。M4 多进程共享 db 时这会变成常态。

        ★ 兼做 schema 迁移,版本号存在 PRAGMA user_version 里(见 schema.SCHEMA_VERSION)。
          SQLite 改不了已有表的主键,executed_keys 换主键只能重建表,所以必须有
          版本号来区分"新库 / 旧库 / 已迁移的库"。

        快路径(只拿读锁):版本对、三张表齐 → 直接返回。版本比代码认识的还新
        (用新库跑旧代码)→ 抛 PersistenceError,不去猜那份结构该怎么用。

        慢路径(BEGIN IMMEDIATE):**再查一次**版本(double-checked)。两个进程
        可能同时在快路径上看到旧版本,但写锁让它们串行,后到的那个二次检查时
        读到的已经是新版本,直接退出 —— 只迁一次。快路径不拿写锁也是安全的:
        WAL 下别人的迁移没提交前,读到的始终是迁移前的完整结构;版本只增不减,
        所以"读到旧版本"最多导致白进一次慢路径,不会导致用错结构。

        原子性:ALTER TABLE、CREATE TABLE、PRAGMA user_version 都在同一个事务里,
        user_version 写在库文件头里、随事务回滚。迁移中途失败 = 整段撤销,
        库停在旧版本、旧表原名原样,下次打开再试。
        """
        with self._db_guard("_create_tables"):
            with self._read() as c:
                if self._schema_is_current(c):
                    return

            with self._write_txn() as c:
                if self._schema_is_current(c):   # double-checked:别人刚迁完
                    return
                self._retire_legacy_executed_keys(c)
                c.execute(SESSION_CREATE_SQL)
                c.execute(MESSAGES_CREATE_SQL)
                c.execute(EXECUTED_KEYS_CREATE_SQL)
                # ★ PRAGMA 不接受 ? 占位符,只能把整数常量拼进 SQL。
                #   SCHEMA_VERSION 是代码里的 int 常量,不是外部输入,没有注入面。
                c.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)};")

    @staticmethod
    def _schema_is_current(c: sqlite3.Connection) -> bool:
        """版本 == SCHEMA_VERSION 且三张表齐全。版本更新则直接拒绝。

        版本对了还要数表,是因为"版本 1 但缺表"也可能出现(有人手工删了表),
        那种情况让慢路径用 IF NOT EXISTS 补齐即可。
        """
        version = c.execute("PRAGMA user_version;").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise PersistenceError(
                f"数据库 schema 版本是 {version},比当前代码认识的 {SCHEMA_VERSION} 更新 "
                f"—— 它是被更新版本的 mini_agent 写过的。请升级代码后再打开,"
                f"不要用旧代码读写它(旧代码不认识新结构,会写坏数据)。"
            )
        if version < SCHEMA_VERSION:
            return False
        placeholders = ",".join("?" * len(_TABLES))
        n = c.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type='table' AND name IN ({placeholders});",
            _TABLES,
        ).fetchone()[0]
        return n == len(_TABLES)

    @staticmethod
    def _retire_legacy_executed_keys(c: sqlite3.Connection) -> None:
        """v0 → v1:旧的 executed_keys(以模型给的 tool_call.id 为主键)改名封存。

        必须在 _write_txn 里调用。判断旧库看的是**有没有 idempotency_key 列**,
        而不是"表存在":新表也叫 executed_keys,只看名字会把刚建好的新表也改名。

        旧数据不迁、不删、以后也不再读:
          - 旧行无法可靠地映射到 (assistant_seq, tool_index) —— 用 id 反查本身
            就是这次要修的 bug;
          - 它唯一的用处是"升级前恰好崩在记表与写 tool 消息之间"的会话,丢了也只
            是降级成 interrupted 桩,符合 recover 的金规则。
        改名而不是 DROP:自动迁移不该做不可逆的事,真要排查时数据还在。
        RENAME 会把旧表的自动索引一起改名,不会和新表的主键索引撞名;没有别的
        表引用 executed_keys,外键也不受影响。
        """
        cols = {r["name"] for r in c.execute("PRAGMA table_info(executed_keys);")}
        if "idempotency_key" not in cols:
            return   # 没有这张表(新库),或者已经是新结构

        target = _LEGACY_EXECUTED_KEYS
        taken = c.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?;", (target,)
        ).fetchone()
        if taken is not None:
            # 名字被占了(比如上次手工迁过一半)。别去动占位的那张表,换个名字。
            target = f"{_LEGACY_EXECUTED_KEYS}_{int(time.time())}"
        logger.info("schema 迁移 v0→v%d: 旧 executed_keys 改名为 %s", SCHEMA_VERSION, target)
        # 表名同样不能用 ? 传,这里拼的是常量 + 整数时间戳,不是外部输入
        c.execute(f'ALTER TABLE executed_keys RENAME TO "{target}";')

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
            with self._read() as c:
                row = c.execute(
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

    def _sanitize_dangling_tool_calls(
        self, session_id: str, rows: list[tuple[int, str, Message | None]]
    ) -> list[Message]:
        """把每一轮 assistant(tool_calls) 补足成配对完整、可安全发给 DeepSeek 的形状。

        输入是 recover 读出来的 (seq, role 列, 解析后的 Message 或 None) 三元组,
        None 表示那一行 JSON 坏了、已被跳过。

        ★ 补桩,不是丢弃。 残缺的那一轮**绝不能删**:删掉 = 抹掉「这个工具可能
        已经跑过」这个事实,模型看不见就会重做一遍(邮件重发、文件重写)。正确解法
        是给每个没有结果的 tool_call 补一条 role="tool" 消息 —— 配对补齐(不 400),
        同时如实告诉模型「这次调用结果未知」,让它自己决定要不要重试。
        补桩前先按 (session_id, assistant_seq, tool_index) 查幂等表:命中说明崩在
        「已 record_executed_key、还没 append_message」之间,能拿回真结果,一次都
        不用重跑。

        ★ 补桩 != 重跑。 本方法执行零个 handler、零次 LLM 调用,纯读 + 纯计算。
        恢复过程一旦有副作用,你就得为恢复写恢复,那是个无底洞。

        ★ 按窗口配对,不按全会话的 id 配对。 一条 assistant(tool_calls) 的结果,
        只从**紧跟它的那段 tool 行**里认领;窗口遇到下一条 role 不是 tool 的行就截断。
          - 旧实现给整个会话建 {tool_call_id: msg} 索引:模型只要在两轮里复用同一个
            id,后写的就覆盖先写的,第 1 轮的 assistant 认领第 3 轮的结果。无状态化
            之后每轮开头都要 recover,所以这是**正常路径**上的错配,不需要崩溃。
            结果能不能取对,不能取决于模型给的 id 是否唯一 —— 那是外部输入。
          - 截断看的是 **role 列**,不是解析成功的消息。坏掉被跳过的 assistant 行
            照样截断窗口,它后面的 tool 行成为孤儿被丢弃,而不是并进上一条
            assistant 的窗口、被错误地认领。
          - 窗口内每个 id 一个**先进先出队列**,取走的不再复用。同一条 assistant 里
            两个 tool_call 同 id 时,按 dict 匹配会让两个 tc 拿到同一条结果;串行下
            tool 行的落库顺序就是 tool_calls 的下标顺序,所以先进先出等价于按位置配,
            同时还能容忍不同 id 之间的乱序(顺带按 tool_calls 顺序重排)。
          - 窗口里多出来的、认领不到的 tool 行,以及不在任何窗口里的孤儿,丢弃 + 告警。

        不变量:【串行】主循环里,_execute_tool_calls 按 tool_calls 的下标顺序逐个
        执行、逐个落库,所以窗口里 tool 行的顺序 == 下标顺序。残缺的窗口要么在尾部
        (崩溃),要么后面紧跟一条封口 assistant(取消,见 agent._seal_cancelled_turn)
        —— 两种情况桩都补在窗口末尾、下一条非 tool 行之前,配对仍然合法。
        这里遍历全部行而不是只看尾巴,是防御性的。
          ★ M6 并行工具会打破"执行顺序 == 下标顺序":到那时必须在 dispatch **之前**
            就把 idx 绑定到每个 tool_call 上(record_executed_key 用的是它),并且
            tool 消息仍然**按 idx 顺序**落库 —— 否则同 id 的先进先出就不再等价于
            按位置配。
        """
        kept: list[Message] = []
        i, n = 0, len(rows)
        while i < n:
            seq, role, msg = rows[i]
            i += 1

            if role == "tool":
                # 能走到这里,说明它不在任何 assistant(tool_calls) 的窗口里
                if msg is not None:
                    logger.warning(
                        "recover: 丢弃孤儿 tool 消息 session=%s seq=%s tool_call_id=%r",
                        session_id, seq, msg.tool_call_id,
                    )
                continue

            if msg is None:
                continue  # 坏行,recover 已告警;它的 role 列照样截断了上一个窗口

            if msg.role != "assistant" or not msg.tool_calls:
                kept.append(msg)  # system/user/纯文本 assistant,天然完整
                continue

            # 收集窗口:紧跟在后面、role 列为 tool 的那一段
            queues: dict[str, deque[Message]] = defaultdict(deque)
            while i < n and rows[i][1] == "tool":
                if rows[i][2] is not None:
                    queues[rows[i][2].tool_call_id].append(rows[i][2])
                i += 1

            kept.append(msg)
            for idx, tc in enumerate(msg.tool_calls):
                q = queues.get(tc.id)
                if q:
                    kept.append(q.popleft())  # 真结果,原样带出
                    continue
                content = self._backfill_from_executed_keys(session_id, seq, idx, tc.id)
                # 补出来的这条必须和真结果长得一模一样,否则照样 400
                kept.append(Message(role="tool", tool_call_id=tc.id, content=content))

            leftover = sum(len(q) for q in queues.values())
            if leftover:
                logger.warning(
                    "recover: assistant seq=%s 的窗口里有 %d 条 tool 消息认领不到,丢弃 "
                    "session=%s", seq, leftover, session_id,
                )

        return kept

    def _backfill_from_executed_keys(
        self, session_id: str, assistant_seq: int, idx: int, tc_id: str
    ) -> str:
        """窗口里没有这个 tool_call 的结果 → 查幂等表;查不到就返回 interrupted 桩。

        命中说明崩在"已 record_executed_key、还没 append_message"之间,这一步是
        无损恢复;未命中才是真丢了。

        ★ 空 tc.id 照常查。 键是 (session, assistant_seq, idx),里面不含模型给的 id,
          空不空和"能不能查"无关;id 只参与读侧的一致性比对。旧实现对空 id 直接补桩,
          是因为旧键就是 id 本身 —— 换了键之后再这么做,只会白白丢掉一个已经存下来的
          真结果,害模型重跑一次有副作用的操作。
        """
        cached = self.get_executed_result(session_id, assistant_seq, idx, tc_id)
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
            logger.warning(
                "recover: seq=%s idx=%s tc=%r 用幂等表里的真结果补回",
                assistant_seq, idx, tc_id,
            )
            return content
        if content is not None:
            logger.warning(
                "recover: seq=%s idx=%s tc=%r 幂等表里的 content 是 %s 不是 str,"
                "按未命中处理", assistant_seq, idx, tc_id, type(content).__name__,
            )
        logger.warning(
            "recover: seq=%s idx=%s tc=%r 结果丢失,补 interrupted 桩",
            assistant_seq, idx, tc_id,
        )
        return _INTERRUPTED

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
            1. messages 行 JSON 坏 → 跳过该行 + 告警(它的 role 列照样截断配对窗口)
            2. executed_keys 行 JSON 坏 / 存的 tool_call_id 对不上 → 按未命中处理
            3. tool_call 的 id 是空串 → 照常按 (seq, idx) 查表;get_executed_result
               不再有空 key 守卫,空 id 是合法输入
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
            with self._read() as c:
                # role 列单独取出来:坏行解析不了,但它的 role 仍然要参与窗口截断
                db_rows = c.execute(
                    "SELECT seq, role, content FROM messages "
                    "WHERE session_id = ? ORDER BY seq;",
                    (session_id,),
                ).fetchall()

        rows: list[tuple[int, str, Message | None]] = []
        for row in db_rows:
            try:
                msg = Message.model_validate_json(row["content"])
            except (ValidationError, ValueError) as e:
                # ValueError 覆盖 JSONDecodeError;两者都属于"这一行的数据坏了"
                logger.warning(
                    "recover: 跳过损坏的消息行 session=%s seq=%s: %s",
                    session_id,
                    row["seq"],
                    e,
                )
                msg = None
            # ★ role 列和 JSON 里的 role 必须一致,不一致按坏行处理(msg=None)。
            #   窗口截断看的是 role 列,认领看的是 msg.role —— 两个来源一旦打架,
            #   一条 JSON 是 tool、列写着 user 的行就会被当普通消息原样留下:一条
            #   不跟在 assistant 后面的 tool 消息,之后**每次**请求都 400,会话报废
            #   且没有任何异常。统一成"列说了算、JSON 对不上就是坏行",两边口径一致。
            if msg is not None and msg.role != row["role"]:
                logger.warning(
                    "recover: 消息行 role 列(%s)与内容(%s)不一致,按坏行跳过 session=%s seq=%s",
                    row["role"], msg.role, session_id, row["seq"],
                )
                msg = None
            rows.append((row["seq"], row["role"], msg))

        return self._sanitize_dangling_tool_calls(session_id, rows)

    # ---------- 幂等表:两个方法分别用于获取 tool 结果和记录 tool 结果 ----------
    #
    # ★ 键是 (session_id, assistant_seq, tool_index),**里面不含任何模型输入**。
    #   旧键就是模型给的 tool_call.id:不同会话撞 id → 一个会话读到另一个会话的工具
    #   输出;同一会话跨轮复用 id → 第 3 轮的写入被吞、恢复时拿回第 1 轮的结果。
    #   id 的格式和唯一性我们控制不了,也不该依赖。现在键全由我们自己的数据构成:
    #   assistant_seq 是 append_message 分配的,tool_index 是 tool_calls 的下标
    #   (随 JSON 原样落库、按这个顺序串行执行,两边天然对齐)。
    #   外键 (session_id, assistant_seq) → messages 让数据库保证每条记录都指向一条
    #   真实存在的消息。tool_call_id 只作普通列,读侧拿来做一致性比对 + 排查用。

    def get_executed_result(
        self,
        session_id: str,
        assistant_seq: int,
        tool_index: int,
        expected_tool_call_id: str,
    ) -> dict | None:
        """查幂等缓存。命中返回 result dict,未命中返回 None。纯读,不开写事务。

        以下都按「未命中」处理,告警但不抛:
          - 行不存在;
          - 行的 JSON 坏了;
          - ★ 存的 tool_call_id 和 expected_tool_call_id 不一致。位置对上了、id 对不上,
            说明这条记录描述的不是我们以为的那次调用(数据被改过、或写侧有 bug)。
            宁可补一个 interrupted 桩让模型自己判断,也不把一个可能张冠李戴的结果
            喂回去。这是 tool_call_id 不进主键、却仍然要存的全部理由:纵深防御。

        ★ 不再有"空 key"守卫。 键里不含 id,空 id 只是一个参与比对的普通值。
          本方法唯一的调用方是 recover 的补桩分支,传进来的是**从库里读出的历史数据**
          —— 守卫在这里会把一次数据异常升级成 PersistenceError,让会话永久 --resume
          不了。

        损坏的行按「未命中」处理而不是抛,理由同上:recover 对 None 的语义已经是
        「查不到真结果,补 interrupted 桩」—— 一行坏 JSON 的正确降级就是走同一条路。
        recover 对 messages 表的坏行也是跳过+告警,两张表的容错态度必须一致。
        ★ json.loads 必须留在 _db_guard **外面**:guard 里有一个
          except json.JSONDecodeError 分支,放进去会被它抢先翻译成
          PersistenceError,下面这个 except ValueError 就永远接不到了。
        """
        where = f"session={session_id}, seq={assistant_seq}, idx={tool_index}"
        with self._db_guard(f"get_executed_result({where})"):
            with self._read() as c:
                row = c.execute(
                    "SELECT tool_call_id, result FROM executed_keys "
                    "WHERE session_id = ? AND assistant_seq = ? AND tool_index = ?;",
                    (session_id, assistant_seq, tool_index),
                ).fetchone()
            if row is None:
                return None
            stored_id, raw = row["tool_call_id"], row["result"]

        if stored_id != expected_tool_call_id:
            logger.warning(
                "get_executed_result: %s 存的 tool_call_id=%r 与期望的 %r 不一致,"
                "按未命中处理", where, stored_id, expected_tool_call_id,
            )
            return None

        try:
            return json.loads(raw)
        except ValueError as e:
            # ValueError 覆盖 JSONDecodeError
            logger.warning(
                "get_executed_result: 幂等表行损坏 %s,按未命中处理: %s", where, e
            )
            return None

    def record_executed_key(
        self,
        session_id: str,
        assistant_seq: int,
        tool_index: int,
        tool_call_id: str,
        result: dict,
    ) -> tuple[bool, dict]:
        """
        返回 (is_first, canonical_result)。
        - 用 ON CONFLICT(session_id, assistant_seq, tool_index) DO NOTHING,
          看 rowcount 判断 is_first。
        - is_first=True  → canonical 就是你传进来的 result
        - is_first=False → 把表里那条已存在的 result 读出来返回(它才是权威值)
          ★ 关键:这个"读回来"的 SELECT,必须和上面的 INSERT 在同一个
            BEGIN IMMEDIATE 事务里 —— 否则读到的可能不是你刚撞上的那一条。
        - assistant_seq 必须是一条已落库的消息(外键),否则 PersistenceError。
          写入顺序天然满足:assistant 先落库,工具后执行。
        - tool_call_id 原样存,空串也存:它不参与主键,只供读侧比对。
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

        where = f"session={session_id}, seq={assistant_seq}, idx={tool_index}"
        with self._db_guard(
            f"record_executed_key({where})",
            hint="(IntegrityError 通常是 assistant_seq 指向的消息不存在)",
        ):
            with self._write_txn() as c:
                cur = c.execute(
                    "INSERT INTO executed_keys "
                    "(session_id, assistant_seq, tool_index, tool_call_id, result, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id, assistant_seq, tool_index) DO NOTHING;",
                    (session_id, assistant_seq, tool_index, tool_call_id,
                     result_json, int(time.time())),
                )
                if cur.rowcount == 1:
                    return True, result

                # rowcount == 0 → 这个位置已经记过了,读回权威结果
                # 把 INSERT 和 SELECT 放同一个事务里,原因:
                # 1. 少一次锁获取、少一次事务开销。事务已经开着了,SELECT 顺手就做了
                # 2. (更重要)不要让代码的正确性,依赖一个"读代码的人看不见的假设"。
                row = c.execute(
                    "SELECT result FROM executed_keys "
                    "WHERE session_id = ? AND assistant_seq = ? AND tool_index = ?;",
                    (session_id, assistant_seq, tool_index),
                ).fetchone()
                if row is None:
                    raise PersistenceError(
                        f"executed_keys 不变量被破坏: {where} 冲突但不存在"
                    )
                raw = row["result"]

        # ★ json.loads 留在 _db_guard 外面,理由同 get_executed_result:放进去会被
        #   guard 的 JSONDecodeError 分支翻译成 PersistenceError —— 而走到这里时工具
        #   **已经执行完了**,为一行坏掉的旧记录让整轮失败是最坏的取舍。降级成空
        #   dict:调用方拿不到 str 的 content,就退回用本次的新结果并告警。
        try:
            return False, json.loads(raw)
        except ValueError as e:
            logger.warning(
                "record_executed_key: 权威行损坏 %s,按无权威值处理: %s", where, e
            )
            return False, {}

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
