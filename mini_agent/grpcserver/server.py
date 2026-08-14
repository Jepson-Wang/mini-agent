"""AgentService 的 gRPC 实现：把无状态的 Agent 包成 Go 后端能调的服务。

这是继 CLI 之后的第二个 surface，和 __main__.py 平级、一样薄：session_id 由 Go
铸造并传进来，历史加载 / 崩溃补桩全在 run_conversation 内部完成，本模块只负责
**协议层**的三件事 ——

  1. 把 Agent 的进度回调翻成 TaskEvent 流（RunTask 是服务端流式）；
  2. 维护一张"在跑的任务"表，让 CancelTask 找得到该停哪一个；
  3. 把异常收敛成 EVENT_TYPE_ERROR，而不是让流裸断。

★ 线程模型（这是本文件最容易看错的地方）

    gRPC handler 线程            worker 线程
    ── RunTask 生成器 ──         ── run_conversation ──
         yield started
         q.get()  ◄──────────────── on_event() → q.put()
         yield ...                    ...
         q.get()  ◄──────────────── finished / error → q.put()
         return                     q.put(SENTINEL)

  run_conversation 是**阻塞**的，中途只通过回调对外说话；而 gRPC 的服务端流式
  要求 handler 不断 yield。两者对不上，所以中间垫一个队列，用两个线程把"推"
  变成"拉"。取消则是反方向的一位信号（threading.Event），由 handler 侧置位、
  worker 侧在 run_conversation 的取消点轮询。

★ 事件类型的字符串（"started"/"thinking"/...）和 Go 那边 agentclient.EventType
  的常量逐字对齐。它俩是同一份契约的两侧，改一边必须改另一边。
"""
from __future__ import annotations

import itertools
import json
import queue
import signal
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from importlib import metadata
from typing import Iterator, Optional

import grpc

from mini_agent.agent import Agent, TaskCancelled
from mini_agent.config import DEFAULT_TOOLSET, settings
from mini_agent.gen.agent.v1 import agent_pb2, agent_pb2_grpc
from mini_agent.log import get_logger
from mini_agent.persistence.session_db import MiniSessionDB, PersistenceError

logger = get_logger(__name__)


def _version() -> str:
    """从包元数据读版本，避免和 pyproject.toml 各写一份。

    没装（直接从源码树跑）时给一个显眼的兜底，而不是假装是某个版本 ——
    Health 的 version 是给运维看"对面到底是哪个构建"的，猜一个比说不知道更坏。
    """
    try:
        return metadata.version("mini-agent")
    except metadata.PackageNotFoundError:
        return "0.0.0+source"


VERSION = _version()

# 事件队列容量。满了说明发送侧（gRPC）跟不上或者已经没了，见 _offer。
_QUEUE_MAXSIZE = 256
# 往满队列里塞一条事件最多等多久，超时即丢弃并取消任务。
_PUT_TIMEOUT = 5.0
# 消费侧轮询间隔：决定"worker 悄悄死了"最晚多久被发现。
_POLL_INTERVAL = 0.5
# 流结束后等 worker 线程退出的时间。
_JOIN_TIMEOUT = 10.0
# 同一个 session 上排队等前一个任务的上限，见 _SessionLocks。
_SESSION_WAIT_TIMEOUT = 300.0
# 优雅关闭时留给在途请求的时间。
_GRACE_PERIOD = 15.0

# payload 上限。gRPC 这条路没有 HTTP binding tag 挡着，不截的话一次 read_file
# 就能把几 MB 推给对面（而且会跟着任务详情回显给用户）。
_MAX_PAYLOAD = 8192
# error 的 payload 单独收紧：Go 那侧要把它写进 tasks.error_msg，
# 那一列对齐的是 dto.TaskCallbackReq 的 max=2000。这里先截，对面还会再截一次。
_MAX_ERROR_PAYLOAD = 2000

# 本地事件名 → proto 枚举。键必须和 Go 的 agentclient.EventType 逐字一致。
_EVENT_TYPES = {
    "started": agent_pb2.EVENT_TYPE_STARTED,
    "thinking": agent_pb2.EVENT_TYPE_THINKING,
    "tool_call": agent_pb2.EVENT_TYPE_TOOL_CALL,
    "tool_result": agent_pb2.EVENT_TYPE_TOOL_RESULT,
    "finished": agent_pb2.EVENT_TYPE_FINISHED,
    "error": agent_pb2.EVENT_TYPE_ERROR,
}

# 队列里的"没有更多事件了"标记。用独一无二的哨兵对象，不用 None ——
# None 是个合法的普通值，将来谁往队列里放了个 None 就会提前截断整条流。
_SENTINEL = object()


def _truncate(text: str, limit: int) -> str:
    """超长就截断并留个明显的尾巴，别让对面以为这就是全部内容。"""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…[truncated {len(text) - limit} chars]"


def _trace_id(context: grpc.ServicerContext, from_field: str = "") -> str:
    """取本次调用的 trace_id：优先消息体里的字段，其次 metadata。

    Go 那侧目前把 trace_id 放在**消息体**里（见 agentclient.StartTaskReq），
    所以只读 metadata 的话永远是 '-'，链路在我们这里就断了。metadata 作为
    兜底保留，方便 grpcurl 之类的工具手动调试。
    """
    if from_field:
        return from_field
    md = dict(context.invocation_metadata() or ())
    return md.get("x-trace-id", "-")


class _TaskHandle:
    """一个在跑的任务在服务端的全部把手。

    目前只有一位取消信号。它是 threading.Event 而不是 bool，因为置位方
    （handler 线程 / CancelTask 的 RPC 线程）和读取方（worker 线程）不是同一个，
    需要一个有内存屏障保证的东西。
    """

    __slots__ = ("task_uid", "cancel")

    def __init__(self, task_uid: str):
        self.task_uid = task_uid
        self.cancel = threading.Event()


class _TaskRegistry:
    """task_uid → _TaskHandle。CancelTask 靠它找到该停哪个任务。

    进程内的表，重启即空 —— 这没问题：重启后本来也没有任务在跑，取消一个
    不存在的任务如实返回 cancelled=false 就是正确答案。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, _TaskHandle] = {}

    def register(self, task_uid: str) -> Optional[_TaskHandle]:
        """登记一个任务。task_uid 已在跑则返回 None（调用方要拒绝这次请求）。

        拒绝重复而不是接受：同一个 task_uid 跑两遍会把两份事件流写到同一条
        任务上，Go 那边按 task_uid 更新状态，最后谁赢取决于时序。宁可让重复
        的那次响亮失败。
        """
        with self._lock:
            if task_uid in self._tasks:
                return None
            handle = _TaskHandle(task_uid)
            self._tasks[task_uid] = handle
            return handle

    def unregister(self, task_uid: str) -> None:
        with self._lock:
            self._tasks.pop(task_uid, None)

    def cancel(self, task_uid: str) -> bool:
        """置位取消信号。返回 False 表示这个任务不在跑（早结束了 / 从没来过）。"""
        with self._lock:
            handle = self._tasks.get(task_uid)
        if handle is None:
            return False
        handle.cancel.set()
        return True

    def cancel_all(self) -> int:
        """关服务时把在途任务全叫停。返回叫停了几个。"""
        with self._lock:
            handles = list(self._tasks.values())
        for h in handles:
            h.cancel.set()
        return len(handles)


class _SessionBusy(Exception):
    """等不到 session 的执行权（前一个任务跑太久）。"""


class _SessionLocks:
    """按 session_id 串行化 run_conversation。**这不是优化，是正确性。**

    一个 session 的历史是一条**线性 transcript**，而 run_conversation 的形状是
    「读全量历史 → 往末尾追加若干条」。两个任务同时对同一个 session 跑，两条
    对话就会在库里交织成一坨：

        user(A) user(B) assistant(B) tool(B) assistant(A) tool(A) ...

    而且这个损坏是**永久**的 —— 之后每一轮都会把这坨东西发给模型。两个任务还
    都会报成功，没有任何告警。CLI 时代碰不到（单线程 REPL），是 gRPC 这个
    surface 让它变得可达，所以必须在这一层挡住。

    为什么是排队而不是直接拒：一个会话里的消息本来就是有先后的，用户连着发两句
    话是完全正常的操作，让第二句失败很难解释。排队则两句都能被回答、且顺序正确。
    但等待有上限（_SESSION_WAIT_TIMEOUT）：等不到就如实失败，不能让一个卡死的
    任务把后面所有请求连同 worker 线程一起拖住。

    锁按需创建、用完即弃（引用计数），否则长期运行的进程里 session 一多，
    这张表就只涨不缩。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._refs: dict[str, int] = {}

    def _take(self, session_id: str) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            self._refs[session_id] = self._refs.get(session_id, 0) + 1
            return lock

    def _drop(self, session_id: str) -> None:
        with self._guard:
            n = self._refs.get(session_id, 0) - 1
            if n <= 0:
                # 引用计数覆盖「持有者 + 所有等待者」，归零说明没人再碰它了，
                # 可以安全丢弃 —— 不会有人还攥着这把锁。
                self._refs.pop(session_id, None)
                self._locks.pop(session_id, None)
            else:
                self._refs[session_id] = n

    @contextmanager
    def hold(self, session_id: str, timeout: float, cancel: threading.Event):
        """拿到该 session 的执行权。拿不到抛 _SessionBusy，等待中被取消抛 TaskCancelled。

        分段 acquire 而不是一次性 acquire(timeout=全程)，是为了在排队期间也能
        响应取消 —— 用户在排队时点了取消，不该等到轮上了才反应过来。
        """
        lock = self._take(session_id)
        deadline = time.monotonic() + timeout
        acquired = False
        try:
            while True:
                if cancel.is_set():
                    raise TaskCancelled(f"session={session_id} 排队期间被取消")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _SessionBusy(session_id)
                if lock.acquire(timeout=min(0.5, remaining)):
                    acquired = True
                    break
            yield
        finally:
            if acquired:
                lock.release()
            self._drop(session_id)


class AgentServicer(agent_pb2_grpc.AgentServiceServicer):
    """AgentService 的实现。一个实例服务所有请求，内部无 per-session 状态。"""

    def __init__(self, agent: Agent):
        self._agent = agent
        self._tasks = _TaskRegistry()
        self._sessions = _SessionLocks()

    # ---------- Health ----------

    def Health(self, request, context):
        logger.debug("Health trace=%s", _trace_id(context))
        return agent_pb2.HealthResponse(ok=True, version=VERSION)

    # ---------- RunTask ----------

    def RunTask(self, request, context) -> Iterator[agent_pb2.TaskEvent]:
        task_uid = request.task_uid
        trace_id = _trace_id(context, request.trace_id)

        # 入参校验放最前面。task_uid 是整条链路的关联键，session_id 决定历史
        # 落在哪 —— 任一为空都会让后面所有事件失去归属，早拒比晚错好。
        if not task_uid:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "task_uid 不能为空")
        if not request.session_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id 不能为空")

        handle = self._tasks.register(task_uid)
        if handle is None:
            context.abort(
                grpc.StatusCode.ALREADY_EXISTS,
                f"任务 {task_uid} 已在运行中，拒绝重复发起",
            )

        # 客户端断开 / 超时 / 主动 cancel 时，gRPC 会回调这里 —— 顺手把取消位
        # 置上，让 worker 别再对着一条没人收的流干活。
        context.add_callback(handle.cancel.set)

        logger.info(
            "RunTask 开始 task=%s session=%s trace=%s input=%r",
            task_uid, request.session_id, trace_id, _truncate(request.input, 80),
        )
        try:
            yield from self._stream(request, handle, trace_id)
        finally:
            # 无论正常收尾、客户端跑路（GeneratorExit）还是异常，都要摘牌，
            # 否则这个 task_uid 会永远占着注册表、再也发不起来。
            handle.cancel.set()
            self._tasks.unregister(task_uid)
            logger.info("RunTask 结束 task=%s trace=%s", task_uid, trace_id)

    def _stream(
        self, request, handle: _TaskHandle, trace_id: str
    ) -> Iterator[agent_pb2.TaskEvent]:
        """把 worker 推进队列的事件拉出来变成流。见文件头的线程模型图。"""
        task_uid = handle.task_uid
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        # seq 从 1 开始单调递增（proto 里写死的约定，Go 拿它排序去重）。
        # itertoaols.count 本身不是线程安全的，所以**只在这个线程**里取号 ——
        # worker 只管往队列里放内容，编号一律由消费侧统一发。
        seq = itertools.count(1)

        def on_event(event_type: str, payload: str) -> None:
            _offer(q, handle, (event_type, payload), trace_id)

        def cancel_check() -> bool:
            return handle.cancel.is_set()

        def worker() -> None:
            try:
                # 同一个 session 上串行，见 _SessionLocks。锁在 worker 线程里拿，
                # 不在 handler 线程里 —— 排队期间 handler 还得能把 started 发出去
                # 并保持流活着，不然对面会以为我们死了。
                with self._sessions.hold(
                    request.session_id, _SESSION_WAIT_TIMEOUT, handle.cancel
                ):
                    out = self._agent.run_conversation(
                        request.session_id,
                        request.input,
                        on_event=on_event,
                        cancel_check=cancel_check,
                    )
                # ★ completed=False 必须发 error，不能发 finished。
                #   Go 那侧只看事件类型定状态（finished→succeeded），根本不解析
                #   payload。发 finished 的话，一个撞了 IterationBudget 上限、
                #   压根没回答用户的任务会显示成"成功"，正文是
                #   "(stopped: iteration budget exhausted)" —— 那是最坏的一种
                #   错误：它不报警，只是悄悄给了个不对的答案。
                if out.get("completed"):
                    _offer(q, handle, ("finished", json.dumps(out, ensure_ascii=False)), trace_id)
                else:
                    logger.warning(
                        "任务未完成（预算耗尽）task=%s trace=%s used_turns=%s",
                        task_uid, trace_id, out.get("used_turns"),
                    )
                    _offer(q, handle, ("error", out.get("final_response") or "任务未完成"), trace_id)
            except TaskCancelled:
                # 取消是**正常结局**，不是失败：不发 finished 也不发 error，
                # 直接让流干净地 EOF。Go 那边收到 EOF 不改状态 —— 正确，因为
                # 库里在通知我们之前就已经是 canceled 了（先落库、再通知）。
                logger.info("任务被取消 task=%s trace=%s", task_uid, trace_id)
            except _SessionBusy:
                # 前一个任务占着这个 session 超过了等待上限。如实失败 ——
                # 硬跑下去会把两段对话交织进同一条 transcript，那是永久损坏。
                logger.error(
                    "等 session 执行权超时 task=%s session=%s trace=%s",
                    task_uid, request.session_id, trace_id,
                )
                _offer(q, handle, (
                    "error",
                    f"会话 {request.session_id} 上有任务已运行超过 "
                    f"{_SESSION_WAIT_TIMEOUT:.0f}s，本任务未执行",
                ), trace_id)
            except PersistenceError as e:
                # 单独接一手只为日志分级：这类错误是环境问题（库锁死、磁盘满），
                # 不是模型或工具的问题，排查方向完全不同。
                logger.error("任务持久化失败 task=%s trace=%s: %s", task_uid, trace_id, e)
                _offer(q, handle, ("error", f"持久化失败: {e}"), trace_id)
            except Exception as e:  # noqa: BLE001 —— 兜底必须够宽，见下
                # 这个 except 是流的**唯一**保险。漏掉它的话，worker 线程静静
                # 死掉，队列里再没有东西，消费侧只能等到轮询发现线程没了才收摊，
                # 而 Go 那边看到的是一条毫无解释的 EOF —— 任务会被当成成功。
                #
                # 只把类型和消息放进 payload，不带 traceback：那东西会跟着任务
                # 详情回显给最终用户，里面有文件路径、甚至可能有请求 URL。
                logger.exception("任务异常 task=%s trace=%s", task_uid, trace_id)
                _offer(q, handle, ("error", f"{type(e).__name__}: {e}"), trace_id)
            finally:
                # 哨兵也可能投不进去（消费侧已经走了），所以消费侧不能只靠它，
                # 还得会自己发现线程死了 —— 见下面的 is_alive 分支。
                _offer(q, handle, _SENTINEL, trace_id)

        thread = threading.Thread(
            target=worker, name=f"agent-task-{task_uid}", daemon=True
        )
        thread.start()

        try:
            yield _event(task_uid, "started", "", next(seq))

            while True:
                try:
                    item = q.get(timeout=_POLL_INTERVAL)
                except queue.Empty:
                    if thread.is_alive():
                        continue
                    # worker 没了却没留下哨兵（投递超时被丢）。它已经死了，
                    # 不会再有新事件，把队列里压着的排空就可以收摊了。
                    yield from _drain(q, task_uid, seq)
                    break

                if item is _SENTINEL:
                    break
                yield _event(task_uid, item[0], item[1], next(seq))
        finally:
            # 流不管怎么结束，worker 都得跟着停：客户端跑了还继续烧 LLM 额度
            # 是最贵的一种泄漏。置位 + 等它到下一个取消点。
            handle.cancel.set()
            thread.join(timeout=_JOIN_TIMEOUT)
            if thread.is_alive():
                # 等不到通常意味着卡在一次没有取消点的长调用里（比如一次很慢的
                # LLM 请求）。线程是 daemon，不会拖住进程退出，但值得留一条记录。
                logger.warning(
                    "worker 线程在 %.0fs 内没停下 task=%s trace=%s",
                    _JOIN_TIMEOUT, task_uid, trace_id,
                )

    # ---------- CancelTask ----------

    def CancelTask(self, request, context):
        trace_id = _trace_id(context, request.trace_id)
        task_uid = request.task_uid

        if not task_uid:
            return agent_pb2.CancelTaskResponse(
                cancelled=False, message="task_uid 不能为空"
            )

        logger.info(
            "CancelTask task=%s trace=%s reason=%r",
            task_uid, trace_id, _truncate(request.reason, 200),
        )

        if not self._tasks.cancel(task_uid):
            # 不在表里 = 早就跑完了、从没发起过、或者发起它的进程已经重启。
            # 这三种我们区分不了，也不必区分：对面是 best-effort 调用，库里
            # 已经是 canceled 了。message 必须说人话 —— 那是对面唯一能拿到的
            # 失败解释（见 proto 里的注释）。
            return agent_pb2.CancelTaskResponse(
                cancelled=False,
                message=f"任务 {task_uid} 不在运行中（已结束、从未发起，或本进程重启过）",
            )

        # 置位即返回，不等它真的停。取消点最坏要等一次 LLM 调用返回，
        # 而对面只给了 10s（agentclient.cancelTimeout）—— 等下去只会双双超时。
        return agent_pb2.CancelTaskResponse(cancelled=True, message="")

    # ---------- 关闭 ----------

    def shutdown(self) -> None:
        """叫停所有在途任务。给 serve() 在优雅关闭时调。"""
        n = self._tasks.cancel_all()
        if n:
            logger.info("正在叫停 %d 个在途任务", n)


# ---------- 模块级小工具 ----------


def _event(task_uid: str, event_type: str, payload: str, seq: int) -> agent_pb2.TaskEvent:
    """组一条 TaskEvent。时间戳是 Unix 毫秒（proto 里写死的约定）。"""
    proto_type = _EVENT_TYPES.get(event_type)
    if proto_type is None:
        # 我们自己发出了一个没登记的类型 —— 是本进程的 bug，不是对面的。
        # 发 UNSPECIFIED 而不是丢弃：payload 还在，对面记一条 unknown 就能
        # 发现"你们加了新事件却没跟 Go 那边对齐"。
        logger.warning("未登记的事件类型 %r，按 UNSPECIFIED 发出", event_type)
        proto_type = agent_pb2.EVENT_TYPE_UNSPECIFIED

    limit = _MAX_ERROR_PAYLOAD if event_type == "error" else _MAX_PAYLOAD
    return agent_pb2.TaskEvent(
        task_uid=task_uid,
        type=proto_type,
        seq=seq,
        payload=_truncate(payload, limit),
        timestamp=int(time.time() * 1000),
    )


def _offer(q: queue.Queue, handle: _TaskHandle, item, trace_id: str) -> bool:
    """尽力把一条事件放进队列；放不进去就丢掉并叫停任务。

    ★ 绝不无限阻塞。 队列满意味着消费侧跟不上或者已经没了（客户端断开后
    没人再 get）。这时候把 worker 阻塞在 put 上是最坏的选择：任务停不下来、
    线程回收不了、LLM 额度继续烧。丢事件 + 置取消位，让任务在下一个取消点
    自己收场 —— 事件是可丢的，一个停不下来的任务不是。
    """
    try:
        q.put(item, timeout=_PUT_TIMEOUT)
        return True
    except queue.Full:
        logger.warning(
            "事件队列 %.0fs 内仍满，丢弃事件并取消任务 task=%s trace=%s",
            _PUT_TIMEOUT, handle.task_uid, trace_id,
        )
        handle.cancel.set()
        return False


def _drain(q: queue.Queue, task_uid: str, seq) -> Iterator[agent_pb2.TaskEvent]:
    """把队列里**现存**的事件全吐出来，不等新的。

    只在确认 worker 已经退出之后调用 —— 那时不会再有新事件进来，排空即完整。
    """
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return
        if item is _SENTINEL:
            return
        yield _event(task_uid, item[0], item[1], next(seq))


# ---------- 启动 ----------


def serve(port: Optional[int] = None, max_workers: Optional[int] = None) -> None:
    """起 gRPC server 并阻塞，直到收到 SIGINT / SIGTERM。"""
    port = settings.grpc_port if port is None else port
    max_workers = settings.grpc_max_workers if max_workers is None else max_workers

    if not settings.deepseek_api_key:
        # 和 CLI 同样的前置检查。没有 key 也能把端口听起来，但每个任务都会在
        # 第一次 LLM 调用时炸 —— 那时错误已经变成一条 error 事件传到 Go 那边，
        # 排查要绕一大圈。启动时就说清楚。
        raise SystemExit("缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 key。")

    # 开库失败（路径无权限 / 磁盘满 / 文件损坏）给一句人话，别甩 traceback。
    # 同 __main__.py 的处理，理由也一样：这是启动期的环境问题，不是程序 bug。
    try:
        db = MiniSessionDB(settings.db_path())
    except PersistenceError as e:
        raise SystemExit(
            f"无法打开数据库：{e}\n位置：{settings.db_path()}\n"
            f"若确认这个文件已损坏，删掉它即可重新开始（会丢失全部历史会话）。"
        )
    agent = Agent(db=db, toolset=DEFAULT_TOOLSET, max_turns=settings.max_turns)
    servicer = AgentServicer(agent)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(servicer, server)

    # add_insecure_port 返回 0 表示端口没绑上（被占用 / 无权限）。不检查的话
    # 服务会"正常启动"然后谁也连不上。
    if server.add_insecure_port(f"[::]:{port}") == 0:
        raise SystemExit(f"端口 {port} 绑定失败（被占用或无权限）")

    server.start()
    logger.info(
        "gRPC server 已启动 port=%d workers=%d model=%s version=%s",
        port, max_workers, settings.model, VERSION,
    )

    stop = threading.Event()

    def _on_signal(signum, _frame) -> None:
        # 信号处理器里只置位，真正的关闭动作留给主线程 —— 处理器跑在中断
        # 上下文里，在里面做重活（等线程、关连接）容易踩到重入。
        logger.info("收到 %s，开始优雅关闭", signal.Signals(signum).name)
        stop.set()

    # 只有主线程能装信号处理器；serve() 本来就该在主线程里跑。
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    stop.wait()

    # 顺序要紧：先叫停在途任务，再 stop(grace)。反过来的话，那些任务在整个
    # grace 期间还在跑，等于白等一场再被硬砍。
    servicer.shutdown()
    server.stop(_GRACE_PERIOD).wait()
    logger.info("gRPC server 已关闭")


if __name__ == "__main__":
    serve()
