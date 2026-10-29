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

from mini_agent.llm import call_llm
from mini_agent.log import get_logger
from mini_agent.schema import Message
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
        return self.db.append_message(turn.session_id, msg)

    def _tool_defs(self) -> Optional[list[dict]]:
        """本轮要暴露给模型的工具列表。无 toolset → None（等价 M0，不带工具）。"""
        if not self.toolset:
            return None
        return registry.get_definitions(toolset=self.toolset) or None

    def _call_model(self, turn: _Turn, tools: Optional[list[dict]]) -> Message:
        """拼上 system、发一次调用、解析回内部 Message。"""
        api_msgs = [self.system_dict] + [m.to_openai() for m in turn.messages]
        raw = call_llm(api_msgs, tools=tools)
        return Message.from_openai(raw)

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
          2. recover —— 历史的唯一真相在库里。它顺带把上次崩溃留下的残缺轮补桩、
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
        history = self.db.recover(session_id)
        turn = _Turn(
            session_id=session_id,
            messages=history,
            on_event=on_event,
            cancel_check=cancel_check,
        )

        if not _turn_in_flight(history):
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
            assistant = self._call_model(turn, self._tool_defs())
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
