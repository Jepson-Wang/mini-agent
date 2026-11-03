"""核心 Agent 类与主对话循环。

演进：
  [M0] 最小对话循环：无工具、单 provider（DeepSeek）。
  [M1] 工具系统：每轮把 get_definitions() 传给模型，解析 tool_calls、
       dispatch 执行、把结果作为 role="tool" 回填，再回到循环顶。
  [M2] 持久化：每 append 一条内存 Message 就同步落库。
  [M2.5] **无状态化**：session_id 由调用方（Go 服务端经 gRPC）每次显式传入，
       会话历史每轮从库里 recover。一个 Agent 实例可并发服务任意多个 session，
       进程里不留任何 per-session 状态 —— 这样任何 worker 都能接任何 session，
       横向扩容不需要 sticky 路由。
后续：
  M3 上下文压缩（preflight 检查）
  M4 subagent 委派（delegate_task 拦截）
  M5 self-improving（memory/skills 注入）
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from mini_agent.context.compressor import (
    CompressionPolicy,
    CompactionError,
    Compressor,
    apply_summary,
    prune_old_tool_outputs,
)
from mini_agent.context.tokens import (
    TokenCount,
    estimate_text,
    measure,
    request_fingerprint,
    truncate_tool_output,
)
from mini_agent.llm import call_llm, is_context_overflow
from mini_agent.log import get_logger
from mini_agent.schema import Message, TokenUsage
from mini_agent.tools.registry import registry, discover_builtin_tools

if TYPE_CHECKING:
    from mini_agent.persistence.session_db import MiniSessionDB

logger = get_logger(__name__)


class TaskCancelled(Exception):
    """外部要求中止本次 run_conversation（gRPC 的 CancelTask / 客户端断开）。

    刻意用异常而不是返回 `completed=False`：后者已经被"预算耗尽"占用了，
    两种结局混在一个返回值里，调用方就分不出"跑不完"和"被叫停"——而这两件事
    对上游意味着完全不同的东西（前者是失败，后者是用户自己要的）。
    """


# 被取消的回合在 transcript 上的封口标记，见 _seal_cancelled_turn。
CANCELLED_MARKER = "(上一轮任务被取消，未完成)"

DEFAULT_SYSTEM_PROMPT = (
    "You are mini-agent, a helpful AI assistant with access to tools.\n"
    "When a task needs external information or an action, CALL the appropriate tool "
    "immediately. Do NOT announce what you are about to do or ask the user to wait — "
    "just call the tool, then answer from its result.\n"
    "When you used a web tool, cite the source URLs you relied on.\n"
    "Answer concisely and accurately."
)


class IterationBudget:
    """限制单次 run_conversation 的循环轮数，防止失控（Hermes 同名概念）。"""

    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        self._used = 0

    def consume(self) -> bool:
        """消耗一轮预算；返回 False 表示预算已耗尽（此时不计入 used）。"""
        with self._lock:
            if self._used >= self.max_turns:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0,self.max_turns-self._used)


def _turn_in_flight(history: list[Message]) -> bool:
    """上一回合是否没跑完 —— 判据就是主循环自己终止条件的反面。

    run_conversation 只在模型返回**纯文本 assistant**（不带 tool_calls）时才算
    收尾，所以「transcript 以纯文本 assistant 结尾」⟺「上一回合已完成」。
    其余任何结尾（user / tool / 带 tool_calls 的 assistant）都意味着上一次调用
    半路死了，而**用户那句话早已落库**（它是先写库再调 LLM 的）—— 这次是调用方
    在重试，绝不能再记一遍，否则 transcript 里会多出一条重复的 user。

    ★ 别用「末条是不是 user」这个朴素判据。 它只抓得住"崩在首次 LLM 调用"这一种
      崩溃点；崩在工具执行之后时历史以 tool 结尾，会被漏判，插出
      [user, assistant(tool_calls), tool, user] 这种更难看的东西。

    空历史 = 全新 session，不算 in-flight。
    """
    if not history:
        return False
    last = history[-1]
    return not (last.role == "assistant" and not last.tool_calls)


@dataclass
class _Turn:
    """一次 run_conversation 调用的全部易变状态。

    无状态化的关键：session_id 和 messages 从**实例字段**降级成**调用局部变量**。
    它俩必须一起拆走 —— 只把 session_id 改成传参、messages 还挂在 self 上的话，
    第二个 session 会读到第一个残留的历史，比原来更糟。

    on_event / cancel_check 同理，**必须挂在这里而不是 Agent 实例上**：一个实例
    并发服务多个 session，挂实例上就等于把 A 的事件推给了 B 的流、用 B 的取消
    信号停掉 A 的任务。
    """

    session_id: str
    messages: list[Message]

    # 进度回调：(event_type, payload) -> None。gRPC 那侧把它塞进事件队列变成
    # TaskEvent 流。None = 没人听（CLI、测试），所有 _emit 直接空转。
    on_event: Optional[Callable[[str, str], None]] = None

    # 取消探针：返回 True 表示"别跑了"。在每个可中断的点被调用，见 _check_cancelled。
    cancel_check: Optional[Callable[[], bool]] = None

    # M3：和 messages 一一对应，是每条消息在库里的 seq。合成消息（补桩、summary、
    # 合并后的 user）是 None —— 它们不是库里的行。压缩选边界、D1 裁剪都靠它
    # 判断"这条是不是库里的、有多旧"。
    # ★ 默认 None 而不是 []：外部（测试、将来别的入口）可能只给 session_id +
    #   messages 来构造 _Turn。列表默认值不能在 dataclass 里直接写 []，而在
    #   __post_init__ 里补成 [None] * len(messages) 才能保证和 messages 等长 ——
    #   否则 _record 往后 append 时 seqs 和 messages 错一位，后面全错。
    seqs: Optional[list[int | None]] = None

    # 生效的检查点边界（load_context 给的 first_kept_seq）。进 fingerprint。
    boundary: Optional[int] = None
    # D1 降级的裁剪界线：本次 run 内固定，seq < 它的旧 tool 输出已换成占位符。
    # 也进 fingerprint —— 替换之后旧锚点的 prompt_tokens 描述的已经不是眼前的前缀。
    prune_before: Optional[int] = None
    # 本次 run 内总结已经失败过一次：不再重试（同一份历史、同样的输入，大概率
    # 同样的结果，只是再白花一次调用和一轮等待）。
    compaction_failed: bool = False

    def __post_init__(self) -> None:
        if self.seqs is None:
            self.seqs = [None] * len(self.messages)


def _view_state(turn) -> Optional[tuple[int | None, int | None]]:
    """turn 的视图状态 (boundary, prune_before)，(None, None) 归一化成 None。

    ★ _measure 与 _call_model **必须**都走这一个函数。 两处各算各的（比如一处漏了
      prune_before）就会得到两把尺子：写进去的锚点永远对不上比对时的指纹，usage
      锚点等于不存在，每轮都退回全量估算。
    ★ 用 getattr 而不是 turn.view_state：外部构造的鸭子类型 turn 可能没有这些
      字段，属性不存在就等于"没有检查点、没有裁剪"。
    """
    boundary = getattr(turn, "boundary", None)
    prune_before = getattr(turn, "prune_before", None)
    if boundary is None and prune_before is None:
        return None
    return (boundary, prune_before)


class Agent:
    """无状态的对话 handler：一个实例服务所有 session。

    实例上只放**跨 session 共享**的配置与依赖；某个会话的历史一律从库里取。
    """

    def __init__(
        self,
        *,
        db: "MiniSessionDB",
        toolset: Optional[set[str]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 25,
        compressor: Optional[Compressor] = None,
        policy: Optional[CompressionPolicy] = None,
        max_tool_output_tokens: Optional[int] = None,
    ):
        # db 是**必需**依赖，不再允许 None。会话历史的唯一真相在库里，没有 db
        # 就没有历史可言；而 sessions 表里的 parent_session_id/depth 说明连将来的
        # 子代理也要落库，纯内存模式没有任何真实用途。去掉它顺带消掉了 _record /
        # _execute_tool_calls 里两处 `if self.db is not None` 分支。
        self.db = db
        # M1：toolset 决定这个 agent 能看到哪些工具（None = 不带工具，退回 M0）。
        #     M4 子代理靠"父 toolset 的子集"实现最小权限。
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.max_turns = max_turns

        # M3：上下文压缩。两个**可选**依赖，缺一个就完全不做检查（_maybe_compress
        # 直接返回）—— 于是不传时现有行为逐字节不变，可以零风险先合并这一块，
        # 等 compressor 的真实实现落地、两个入口各自决定接不接。
        self.compressor = compressor
        self.policy = policy

        # 单次工具输出的 token 上限（None = 不截断）。独立于 compressor/policy：
        # 压缩永远不压"最新一轮"（模型还没读过刚返回的结果），单次输出本身超预算时
        # 压缩帮不上忙、重试照样 400 —— 所以这是压缩关掉也要在的安全网。
        # ★ 已知限制：只限**单次调用**。一轮里多个 tool_call 的总量不在此处管，
        #   剩余风险由 400 被动兜底 + 旧工具输出裁剪承担。
        self.max_tool_output_tokens = max_tool_output_tokens

        if toolset:
            # 显式、幂等地做一次工具发现：一个"带工具的 agent"就该保证工具已注册。
            # import 已缓存，多个 agent 重复调也只有第一次真正扫盘。
            discover_builtin_tools()

    @property
    def system_dict(self) -> dict:
        return {"role": "system", "content": self.system_prompt}

    # ---------- 内部：进度上报 + 取消 ----------

    @staticmethod
    def _emit(turn: _Turn, event_type: str, payload: str) -> None:
        """尽力上报一条进度事件。**永不抛异常。**

        进度是旁路，不是主线：观察者（gRPC 事件队列）出问题不该让一个正在跑的
        任务失败 —— 那等于让"看任务的人"有权杀掉任务。同 registry.dispatch 的取向。
        真要停，走 cancel_check 那条明路。
        """
        if turn.on_event is None:
            return
        try:
            turn.on_event(event_type, payload)
        except Exception:
            logger.exception(
                "on_event 回调抛异常，已忽略 session=%s type=%s",
                turn.session_id, event_type,
            )

    @staticmethod
    def _check_cancelled(turn: _Turn) -> None:
        """到取消点了就抛 TaskCancelled。cancel_check 自己抛异常视为"不取消"。

        探针是外部传进来的代码，它坏掉不该被解读成"用户要取消" —— 那是在最坏的
        方向上猜。记一条日志继续跑，任务顶多多活到下一个探针点或自然结束。
        """
        if turn.cancel_check is None:
            return
        try:
            cancelled = turn.cancel_check()
        except Exception:
            logger.exception("cancel_check 抛异常，按未取消处理 session=%s", turn.session_id)
            return
        if cancelled:
            raise TaskCancelled(f"session={turn.session_id} 收到取消信号")

    def _seal_cancelled_turn(self, turn: _Turn) -> None:
        """给被取消的回合在 transcript 上封口，让它不再看起来像"跑了一半"。

        ★ 少了这一步会静默丢用户输入。 _turn_in_flight 的判据是"末条不是纯文本
        assistant"，它原本只可能由**崩溃**造成，所以推断"调用方在重试、用户那句话
        早已落库"是对的。取消打破了这个前提：取消之后同一个 session 的下一个任务
        带的是**全新的用户输入**，却会被当成重试而不予记录 —— 用户换了个问题，
        模型接着答上一个被取消的问题，而新问题永远丢了。

        封口的方式是补一条纯文本 assistant：
          - 纯文本（不带 tool_calls）正是主循环的正常终止形态，_turn_in_flight
            据此判定"这回合结束了"，下一轮的用户输入就能正常落库；
          - 内容如实写明被取消，下一轮模型看得见"上一件事被叫停了"，这本就是它
            该有的上下文；
          - 残缺的 tool 配对不必在这里补 —— 下一轮 recover 会按 assistant 窗口
            统一补桩（封口消息不是 tool，正好截断窗口），补出来的位置紧跟在
            assistant(tool_calls) 后面，仍然合法，这条封口消息排在它们之后。

        写库失败只记日志、不抛：这条路径本身是在处理取消，再抛一个新异常只会把
        "任务被取消了"这个真正的结局盖掉。
        """
        try:
            self._record(turn, Message(role="assistant", content=CANCELLED_MARKER))
        except Exception:
            logger.exception("给被取消的回合封口时写库失败 session=%s", turn.session_id)

    # ---------- 内部：落库 + 工具 ----------

    def _record(self, turn: _Turn, msg: Message) -> int:
        """把一条 Message 同时写进本轮内存和库，返回它在库里的 seq。

        M2 的核心：每产生一条消息就立刻落库，崩溃时丢的最多是"最后一条还没写完
        的"。append_message 失败会抛 PersistenceError —— 故意不吞：内存和库一旦
        静默分叉，崩溃恢复就失去意义（fail-fast）。

        返回 seq 是给幂等键用的：assistant(tool_calls) 的 seq 是
        (session_id, assistant_seq, tool_index) 里我们自己掌握的那一维，
        见 _execute_tool_calls。其余调用方忽略返回值即可。
        """
        turn.messages.append(msg)
        seq = self.db.append_message(turn.session_id, msg)
        # seqs 必须和 messages 同步追加，否则"用 seq 判新旧"的所有上游（压缩边界、
        # D1 裁剪、指纹）都会拿错行。append_message 抛异常时上面已 append 的
        # messages 会多一条 —— 与 seq 错位的风险同源，但这是 fail-fast 路径：
        # 异常会一路抛到调用方，这一轮不会再往下用 turn。
        turn.seqs.append(seq)
        return seq

    def _tool_defs(self) -> Optional[list[dict]]:
        """本轮要暴露给模型的工具列表。无 toolset → None（等价 M0，不带工具）。"""
        if not self.toolset:
            return None
        return registry.get_definitions(toolset=self.toolset) or None

    def _call_model(self, turn: _Turn, tools: Optional[list[dict]]) -> Message:
        """拼上 system、发一次调用、解析回内部 Message，并把 usage 挂上去。

        ★ usage 必须落在 assistant 消息上、随 _record 一起落库。 Agent 是无状态的，
          每次 run_conversation 都从库里 recover，内存里攒的东西活不过一次调用 ——
          不落库的话，每次调用的**第一轮**永远没有锚点，只能整条估算（而第一轮恰好
          是上下文最长、最需要数准的那一轮）。

        ★ fingerprint 和 usage 在这里**一起**写：它描述的是"这次请求的前缀"
          （system + 本轮 tools + 视图状态），只有发请求的这一处同时知道这三样。
          收尾调用 tools=None，指纹天然不同，下一轮 measure 会跳过这个锚点。
          视图状态取自 turn.view_state，和 _measure 是同一个属性 —— 见它的说明。

        ★ 上下文超长的 400 是这里的被动兜底：计数是估算 + 锚点，偏低一点就会在阈值
          之下照样撞墙，而撞墙时唯一的出路是压缩。所以收到 is_context_overflow 的
          异常后**强制压缩一次**（越过阈值和 compaction_failed 的闸），**只有视图真的
          变了**才重建请求、重试**一次**：
          - 视图没变（没东西可压、也没东西可裁）→ 原样请求必然再 400，直接抛原异常；
          - 重试再超长 → 直接抛，绝不第三次（压缩已经尽力，再来只是死循环）；
          - 没有 compressor/policy → 没有可用的补救手段，原样抛。
          预算耗尽的收尾调用（tools=None）同样走这里，自动覆盖。
          重试后的 fingerprint 照常取当时的 _view_state(turn)，锚点描述的是真正发出去的前缀。
        """
        api_msgs = [self.system_dict] + [m.to_openai() for m in turn.messages]
        try:
            reply = call_llm(api_msgs, tools=tools)
        except Exception as e:
            if (
                not is_context_overflow(e)
                or self.compressor is None
                or self.policy is None
            ):
                raise
            logger.warning(
                "上游报告上下文超长，强制压缩后重试一次 session=%s", turn.session_id,
            )
            before = self._view_snapshot(turn)
            self._maybe_compress(turn, tools, force=True)   # TaskCancelled 原样上抛
            if self._view_snapshot(turn) == before:
                logger.warning(
                    "强制压缩没有改变视图，不重试 session=%s", turn.session_id,
                )
                raise
            api_msgs = [self.system_dict] + [m.to_openai() for m in turn.messages]
            reply = call_llm(api_msgs, tools=tools)   # 只重试这一次：再抛就原样上抛
        msg = Message.from_openai(reply.message)
        if reply.usage is not None:
            # 注意 fingerprint 用的是**本次实际发出去的** tools，不是 self._tool_defs()：
            # 收尾调用传的是 None，重新算一遍会得到带工具的指纹，锚点就是错的。
            try:
                msg.usage = TokenUsage(
                    prompt_tokens=reply.usage.prompt_tokens,
                    completion_tokens=reply.usage.completion_tokens,
                    fingerprint=request_fingerprint(
                        self.system_prompt, tools, _view_state(turn)
                    ),
                )
            except Exception:
                # 最后一道闸：usage 是纯遥测，构造失败只意味着"这条没有锚点"，
                # 绝不能让一次已经成功的 LLM 调用因为它而失败。
                logger.warning(
                    "usage 挂到消息上失败，按无锚点处理 session=%s", turn.session_id,
                    exc_info=True,
                )
        return msg

    @staticmethod
    def _view_snapshot(turn: _Turn) -> tuple:
        """视图指纹：(boundary, prune_before, 每条消息的 JSON)。

        判"强制压缩有没有改变要发的请求"用。只比 boundary/prune_before 不够 ——
        裁剪点前移到同一值、或内容被换而两个标量没变的情形会漏判；逐条序列化比内容
        才是"请求会不会变"的真相。
        """
        return (
            getattr(turn, "boundary", None),
            getattr(turn, "prune_before", None),
            [m.model_dump_json() for m in turn.messages],
        )

    def _maybe_compress(
        self, turn: _Turn, tools: Optional[list[dict]], *, force: bool = False
    ) -> None:
        """发请求前检查一次上下文用量；超阈值就压缩，最多一次。

        刻意**不返回**任何状态：压缩成功与否都不改变调用方接下来要做的事（发请求），
        所以没有必要让它参与控制流 —— 与 _emit 同一个取向。

        ★ 压缩器抛异常不能杀死任务。 压缩是优化手段，不是正确性的前提；一次失败的
          压缩换来的是"这一轮请求偏长"，而把异常抛出去换来的是整个任务失败。
          记日志后照常发请求。

        ★ 压完还超阈值只告警、不重试。 再压一次是纯浪费（同一份历史、同样的输入、
          大概率同样的结果），真要死循环也是在这里。真正超限的处理交给后续的
          "400 被动兜底"。

        三条路径：
          - 成功：按新检查点重新 load_context + apply_summary 替换 turn 的视图。
            压缩器**只落检查点、不改内存**，不重载的话 compress 等于没发生，请求
            原样发出（plan 2.2 的 TODO）。boundary 随之更新，指纹换一把，旧锚点
            整批作废，下一次计数退回估算。
          - CompactionError：本次 run 不再尝试（compaction_failed），并做 D1 降级
            —— 把候选边界之前的旧工具输出换成占位符，替换点固定为 e.boundary，
            本次 run 内不再移动。只改内存视图，库里一字不动。
          - TaskCancelled：**原样上抛**。它虽然是 Exception 子类，但代表的是"用户
            不要了"，不是"压缩失败了"；吞掉的话封口逻辑（_seal_cancelled_turn）和
            调用方都看不到取消，gRPC 的 CancelTask 会变成一个跑完的任务。
          - 其它异常：同样视为失败，但没有可信的候选边界，不做裁剪。

        force=True 是 400 被动兜底用的（见 _call_model）：上游已经说"超长"了，
          - **跳过阈值判断**：计数偏低正是 400 的来源，再拿它做门槛就自相矛盾；
          - **越过 compaction_failed 的闸一次**：不压这次请求必然再失败；
          - 总结失败时的降级裁剪点允许前移（见 _degrade_after_compaction_failure）；
          - 成功后不做"压完是否仍超阈值"的复核 —— 紧接着的那次重试就是真复核；
          - compacting 事件带 "reason": "overflow"，compacted 事件语义不变，仍成对。
        force=False 的路径与加这个参数之前逐字节一致。
        """
        if self.compressor is None or self.policy is None:
            return
        if turn.compaction_failed and not force:
            # 本次 run 已经失败过：同一份历史、同样的输入，再试只是再花一次调用。
            # 降级（如果有）已经作用在 turn.messages 上，照常发请求即可。
            return
        try:
            count = self._measure(turn, tools)
        except Exception:
            # 计数本身是纯计算，正常不会抛；真抛了也只是"这轮不压"，别把请求也拖下去
            logger.exception("上下文计数失败，跳过本轮压缩检查 session=%s", turn.session_id)
            if not force:
                return
            # 强制路径：计数只用于事件/日志展示，不是门槛，算不出来也照压
            count = TokenCount(tokens=0, source="unknown")
        if not force and not self.policy.should_compress(count):
            return
        compacting = {"tokens": count.tokens, "source": count.source}
        if force:
            compacting["reason"] = "overflow"
        self._emit(turn, "compacting", json.dumps(compacting, ensure_ascii=False))
        latest_before = self.db.latest_compaction(turn.session_id)
        try:
            self.compressor.compress(turn)
        except TaskCancelled:
            # ★ 必须排在 except Exception 前面：TaskCancelled 是 Exception 子类。
            raise
        except CompactionError as e:
            self._degrade_after_compaction_failure(turn, e.boundary, count, force=force)
            return
        except Exception:
            logger.exception(
                "压缩器抛异常，本次 run 不再尝试压缩 session=%s tokens=%s source=%s",
                turn.session_id, count.tokens, count.source,
            )
            turn.compaction_failed = True
            self._emit(turn, "compacted", json.dumps(
                {"ok": False, "reason": "error"}, ensure_ascii=False,
            ))
            return
        # ★ 判定"有没有落新检查点"看的是**压缩前后**最新检查点变没变，不是拿它和
        #   turn.boundary 比：库里的检查点可能被 load_context 降级忽略了（turn.boundary
        #   为 None、latest 却不是），那样一次 noop 会被误判成成功、给 Go 发
        #   compacted{ok: true, boundary: null}。
        latest = self.db.latest_compaction(turn.session_id)
        if latest is None or latest == latest_before:
            # ★ 压缩器**没有**落新检查点（select_boundary 返回 None：中段 < 2k /
            #   装得下尾部）≠"压缩生效"。旧代码照常重载、记 "压缩生效 boundary=None"、
            #   发 ok:true，订阅方会以为上下文已被折叠。这里如实报 ok:false + noop。
            #   ★ 不设 compaction_failed：下一轮历史又长了一点，可能就够压了。
            logger.info(
                "本轮未产生检查点，跳过压缩（noop）session=%s tokens=%s source=%s",
                turn.session_id, count.tokens, count.source,
            )
            self._emit(turn, "compacted", json.dumps(
                {"ok": False, "reason": "noop"}, ensure_ascii=False,
            ))
            return
        self._reload_after_compaction(turn)
        self._emit(turn, "compacted", json.dumps(
            {"ok": True, "boundary": turn.boundary}, ensure_ascii=False,
        ))
        if force:
            return   # 强制路径的复核就是随后的那次重试，不再单独量一遍
        # 压缩后复核：压完仍超阈值说明这次压得不够（尾部边界 + 保留量本身就超标）。
        # 不重试，只告警 —— 再压一次是同一份历史、同样的输入，大概率同样的结果。
        try:
            after = self._measure(turn, tools)
        except Exception:
            logger.exception(
                "压缩后重新计数失败，跳过校验 session=%s", turn.session_id,
            )
            return
        if self.policy.should_compress(after):
            logger.warning(
                "压缩后仍超过阈值，照常发请求（真超限交给 400 被动兜底）session=%s",
                turn.session_id,
            )

    def _reload_after_compaction(self, turn: _Turn) -> None:
        """压缩成功：按最新检查点从库里重建 turn 的视图（messages / seqs / boundary）。

        ★ 不能直接改 turn.messages 里的内容。 压缩的正确语义是"库里多了一条检查点"，
          视图是它的投影。从库里读回来，视图和库就不可能分叉；自己拼一份"少了几条"
          的列表，一旦和 load_context 的渲染规则（summary 合并、补桩、配对）差一点，
          表现就是请求 400。
        ★ prune_before 清空：检查点生效后，旧工具输出本来就不在视图里了；残留的
          裁剪点只会让指纹多带一个无意义的维度。
        """
        ctx = self.db.load_context(turn.session_id)
        messages, seqs = apply_summary(ctx.messages, ctx.seqs, ctx.summary)
        turn.messages = messages
        turn.seqs = seqs
        turn.boundary = ctx.boundary
        turn.prune_before = None
        logger.info(
            "压缩生效 session=%s boundary=%s messages=%s",
            turn.session_id, turn.boundary, len(turn.messages),
        )

    def _degrade_after_compaction_failure(
        self, turn: _Turn, boundary: Optional[int], count: TokenCount,
        *, force: bool = False,
    ) -> None:
        """总结失败的降级（D1）：把候选边界之前的旧工具输出换成占位符。

        ★ 只改内存视图。 库里的原始消息一条不动 —— append-only、审计、
          exactly-once 都建立在原始数据之上；而且 Agent 是无状态的，下一次
          run_conversation 会重新 load_context，自然拿到完整原文并重新尝试总结。
        ★ 替换点固定为这次选出的边界，不随对话增长往后移。 前缀只在替换发生的那一刻
          变一次，之后每轮的请求前缀都相同，缓存才有机会命中。
        ★ prune_before 进视图状态 → 进指纹。 替换之后旧锚点的 prompt_tokens 描述的
          是替换前的长前缀，必须作废，计数退回估算。
        ★ e.boundary 为 None 时没有可信的裁剪界线，只禁用本次 run 的重试。
        ★ 例外：force（400 兜底）路径允许裁剪点**前移**到 max(旧, 新)。平时"替换点固定"
          是为了前缀稳定、缓存命中；但这里请求马上就要失败，只裁到旧界线不够短的话
          重试必然再 400，正确性优先于缓存。只前移、不后退（后退等于把已替换的又
          恢复原文，请求更长）；占位符本身短于裁剪阈值，重复裁剪是幂等的。
        """
        turn.compaction_failed = True
        if boundary is not None:
            if force and turn.prune_before is not None:
                boundary = max(turn.prune_before, boundary)
            turn.messages = prune_old_tool_outputs(turn.messages, turn.seqs, boundary)
            turn.prune_before = boundary
        logger.warning(
            "总结失败，本次 run 不再尝试压缩 session=%s tokens=%s source=%s prune_before=%s",
            turn.session_id, count.tokens, count.source, turn.prune_before,
        )
        self._emit(turn, "compacted", json.dumps(
            {"ok": False, "prune_before": turn.prune_before}, ensure_ascii=False,
        ))

    def _measure(self, turn: _Turn, tools: Optional[list[dict]]) -> TokenCount:
        """当前 turn.messages 发出去的上下文占用估算。见 context.tokens.measure。

        视图状态和 _call_model 写进 usage 的指纹**来自同一个 turn.view_state**：
        检查点边界 + D1 裁剪点，任一变化都让旧锚点作废。
        ★ 用 _view_state(turn) 而不是 turn.view_state 属性：测试/其它入口会拿
          鸭子类型的假 turn（只有 session_id + messages）直接调 _measure。
          属性不存在 = 没有检查点也没有裁剪，视图状态就是 None。
        """
        return measure(turn.messages, self.system_prompt, tools, view_state=_view_state(turn))

    def _execute_tool_calls(
        self, turn: _Turn, assistant: Message, assistant_seq: int
    ) -> None:
        """串行执行 assistant 的每个 tool_call，把结果作为 role="tool" 回填。

        assistant_seq 是这条 assistant 刚落库时 append_message 给的 seq。

        ★ 配对铁律：带 tool_calls 的 assistant 后面，必须紧跟"数量一致、
        tool_call_id 一一对应"的 tool 消息，否则下一次请求会 400。串行执行、
        每个 tc 都回填一条，天然满足；关键是**一个都不能漏**。

        ★ 写入顺序（②在③前）：先记幂等表、再落 tool 结果。若崩在这两步之间，
        recover 能从幂等表查回真结果补桩，不必重跑有副作用的工具。
        幂等键是 (session_id, assistant_seq, tool_index)，**完全由服务端数据构成，
        不含任何模型输入**。以前用模型给的 tc.id 当全局键，模型一复用 id（跨会话
        撞、同会话跨轮撞），后写的就被 DO NOTHING 吞掉，恢复时拿回的是别人的结果。
        tc.id 只作为普通列存着，供 recover 读侧校验"存的和当前的是不是同一个"。
        tool_index 能当键，是因为 assistant.tool_calls 的顺序随 JSON 原样落库，
        这里又按同一顺序串行执行，两边天然对齐。
        （M6 若改成并行执行：idx 必须在提交任务**之前**就绑定到每个 tc 上，
        tool 消息也要按 idx 顺序落库，不能按完成顺序 —— 否则对齐就断了。）
        result 包成 {"content": result} 是因为 dispatch 返回的是裸字符串（不一定是
        JSON），而幂等表按 dict 存、recover 按 cached["content"] 取——两头必须对齐。

        ★ is_first=False 不能静默吞掉。 新键下正常路径根本撞不上：哪怕同一会话被
        两个回合并发跑，它们各自 append_message、拿到的 assistant_seq 也必然不同。
        能撞上只有一种情况：**同一条已落库的 assistant 的 tool_calls 被重放了**
        （重试 / 重放类 bug，或日后有人写了"重跑悬空调用"的逻辑）—— 也就是有副作用的
        工具真的执行了第二次。这是必须被看见的信号。
        此时 tool 消息改用表里的权威结果（先写者赢），保证幂等表和 transcript
        永远一致，否则崩溃恢复补回来的和正常路径落的会是两个不同的答案。
        权威值的 content 不是 str（脏数据）时退回本次结果并告警，绝不抛：工具已经
        跑完了，为一行脏数据把整轮打死只会更糟。

        ★ 取消点在 dispatch **之前**，一个都不放在之后。 在这里中断会留下一轮
        配对残缺的 transcript（assistant 有 tool_calls，tool 结果缺几条），那正是
        recover / _sanitize_dangling_tool_calls 每轮开头都在处理的情况，安全。
        但取消点若放在 dispatch 之后、record 之前，就等于"工具跑了、结果扔了"——
        对有副作用的工具，下次 recover 只能从幂等表捞，捞不到就得重跑。放前面则
        这个工具压根没启动过，没有任何东西需要善后。
        """
        for idx, tc in enumerate(assistant.tool_calls):
            self._check_cancelled(turn)
            self._emit(turn, "tool_call", json.dumps(
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                ensure_ascii=False,
            ))
            result = registry.dispatch(tc.name, tc.arguments)
            # ★ 截断必须在 record_executed_key **之前**：幂等表里的 result 和
            #   transcript 里的 tool 消息得是同一份。崩在两次写之间时 recover 从幂等
            #   表补回的就是模型本来会看到的那份；若先记全文再截断，补回来的是未截断
            #   全文，同一次调用就有两种答案（还可能一补回来就把上下文撑爆）。
            if self.max_tool_output_tokens is not None:
                truncated = truncate_tool_output(result, self.max_tool_output_tokens)
                if truncated is not result:
                    # 只记长度和工具名，不记内容（可能含文件/网页里的敏感信息）
                    logger.info(
                        "工具输出超限已截断 tool=%s 原长=%d字符/约%dtoken 截后=%d字符 上限=%dtoken",
                        tc.name, len(result), estimate_text(result),
                        len(truncated), self.max_tool_output_tokens,
                    )
                    result = truncated
            is_first, canonical = self.db.record_executed_key(
                turn.session_id, assistant_seq, idx, tc.id, {"content": result}
            )
            if not is_first:
                cached = canonical.get("content") if isinstance(canonical, dict) else None
                if isinstance(cached, str):
                    logger.warning(
                        "同一个 (session, assistant_seq, idx) 被执行了第二次（同一条"
                        "assistant 的 tool_calls 被重放）；改用幂等表里的权威结果 "
                        "session=%s assistant_seq=%s idx=%s tool_call_id=%s",
                        turn.session_id, assistant_seq, idx, tc.id,
                    )
                    result = cached
                else:
                    logger.warning(
                        "同一个 (session, assistant_seq, idx) 被执行了第二次（同一条"
                        "assistant 的 tool_calls 被重放）；权威结果的 content 不是 str，保留本次结果 "
                        "session=%s assistant_seq=%s idx=%s tool_call_id=%s",
                        turn.session_id, assistant_seq, idx, tc.id,
                    )
            self._record(turn, Message(role="tool", tool_call_id=tc.id, content=result))
            self._emit(turn, "tool_result", json.dumps(
                {"id": tc.id, "name": tc.name, "content": result},
                ensure_ascii=False,
            ))

    # ---------- 主循环 ----------

    def run_conversation(
        self,
        session_id: str,
        user_message: str,
        *,
        on_event: Optional[Callable[[str, str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """跑一轮用户输入到最终文本回复。返回 {final_response, completed, used_turns}。

        session_id 由调用方给（Go 铸造、经 gRPC 传入；CLI 自己 mint），本方法不
        持有它 —— 这正是"一个实例服务所有 session"的前提。

        on_event / cancel_check 也是**每次调用**给的，理由同上：实例是共享的，
        这两个钩子却是一个任务私有的。两者都可以不给（CLI、测试就不给），
        此时行为与加钩子之前逐字节一致。

        被取消时抛 TaskCancelled（不是返回值），见该异常的说明。

        起手三步：
          1. ensure_session —— 幂等补上 sessions 行。Go 那边"有"这个 session 不等于
             我们库里有，而 messages 有外键指向 sessions，不补首次 append 必炸。
          2. load_context —— 历史的唯一真相在库里；有压缩检查点时只读边界之后的
             那一段，summary 由 apply_summary 渲染进视图（M3）。它顺带把上次崩溃留下的残缺轮补桩、
             并从幂等表回填已执行工具的真结果，所以"上次跑到一半挂了"在每轮开头
             就被免费处理掉了。新 session 返回 []，于是新建/续跑两条路合并成一条。
          3. 记下用户这句话 —— 除非上一回合没跑完（见 _turn_in_flight）。那说明
             这次是调用方在重试，用户那句话早已落库，再记一遍就重复了；此时直接
             从补桩后的历史续跑。**取消不会走到这条分支**：被取消的回合在退出前
             就被 _seal_cancelled_turn 封了口，所以下一个任务的新输入照常记录。

        终止：模型返回纯文本（不带 tool_calls）即完成。
        失控保护：IterationBudget 到顶后，再给最后一次"不带 tools"的收尾调用，
        逼模型用文本作答，而不是无限调工具。
        """
        self.db.ensure_session(session_id)
        # ★ 用 load_context 而不是 recover：有检查点时只读边界之后那一段，并把
        #   summary 渲染进视图（apply_summary）。recover 仍然读全量，它是崩溃恢复
        #   的真相来源，两者共用 _load_rows，降级规则一致。
        #   _turn_in_flight 看的是**渲染前**的 ctx.messages：判据关心的只是末尾
        #   那条是什么，summary 在开头，不影响结论。
        ctx = self.db.load_context(session_id)
        messages, seqs = apply_summary(ctx.messages, ctx.seqs, ctx.summary)
        turn = _Turn(
            session_id=session_id,
            messages=messages,
            seqs=seqs,
            boundary=ctx.boundary,
            on_event=on_event,
            cancel_check=cancel_check,
        )

        if not _turn_in_flight(ctx.messages):
            self._record(turn, Message(role="user", content=user_message))

        try:
            return self._run_loop(turn)
        except TaskCancelled:
            # 封口后原样上抛：调用方要的是"被取消"这个结局，不是一个返回值。
            self._seal_cancelled_turn(turn)
            raise

    def _run_loop(self, turn: _Turn) -> dict:
        """主循环本体。从 run_conversation 拆出来，只为让取消能包一层 try。"""
        budget = IterationBudget(self.max_turns)
        while budget.consume():
            # 循环顶是最主要的取消点：此时 transcript 处于完整可发送状态，
            # 中断不留任何需要 recover 善后的残缺轮。
            self._check_cancelled(turn)
            # 压缩检查放在**取消之后、发请求之前**：取消点优先（用户已经不要这次请求了，
            # 不必再为它花一次总结调用）；而检查必须在发请求之前，否则这次的超长
            # 请求已经发出去了，压缩只能对下一次生效、等于晚了一轮。
            tools = self._tool_defs()
            self._maybe_compress(turn, tools)
            assistant = self._call_model(turn, tools)
            assistant_seq = self._record(turn, assistant)

            # 带 tool_calls 时 content 是模型的"我打算干什么"，纯文本时它就是
            # 最终答案 —— 后者由 finished 事件带走，这里只报前者，避免重复推送。
            if assistant.content and assistant.tool_calls:
                self._emit(turn, "thinking", assistant.content)

            if not assistant.tool_calls:
                return {
                    "final_response": assistant.content or "",
                    "completed": True,
                    "used_turns": budget.used,
                }

            self._execute_tool_calls(turn, assistant, assistant_seq)   # 执行 + 回填，然后回循环顶

        # 预算耗尽：给最后一次机会，强制不带 tools 逼模型收尾成文本。
        # 此时 turn.messages 以一组 tool 结果结尾（合法可发送状态）。
        # 收尾调用同样先检查：预算耗尽时上下文往往最长，这里恰恰最不该漏。
        self._maybe_compress(turn, tools=None)
        assistant = self._call_model(turn, tools=None)
        if assistant.tool_calls:
            # 收尾调用模型还想调工具（真实场景 tools=None 不会发生）：
            # 不执行、也不 record —— 避免在 messages 末尾留下悬空的 tool_calls，
            # 否则下一轮把它发出去就是 400。
            return {
                "final_response": "(stopped: iteration budget exhausted)",
                "completed": False,
                "used_turns": budget.used,
            }
        self._record(turn, assistant)
        return {
            "final_response": assistant.content or "",
            "completed": True,
            "used_turns": budget.used,
        }
