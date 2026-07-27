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

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from mini_agent.llm import call_llm
from mini_agent.schema import Message
from mini_agent.tools.registry import registry, discover_builtin_tools

if TYPE_CHECKING:
    from mini_agent.persistence.session_db import MiniSessionDB

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
    """

    session_id: str
    messages: list[Message]


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

    # ---------- 内部：落库 + 工具 ----------

    def _record(self, turn: _Turn, msg: Message) -> None:
        """把一条 Message 同时写进本轮内存和库。

        M2 的核心：每产生一条消息就立刻落库，崩溃时丢的最多是"最后一条还没写完
        的"。append_message 失败会抛 PersistenceError —— 故意不吞：内存和库一旦
        静默分叉，崩溃恢复就失去意义（fail-fast）。
        """
        turn.messages.append(msg)
        self.db.append_message(turn.session_id, msg)

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

    def _execute_tool_calls(self, turn: _Turn, assistant: Message) -> None:
        """串行执行 assistant 的每个 tool_call，把结果作为 role="tool" 回填。

        ★ 配对铁律：带 tool_calls 的 assistant 后面，必须紧跟"数量一致、
        tool_call_id 一一对应"的 tool 消息，否则下一次请求会 400。串行执行、
        每个 tc 都回填一条，天然满足；关键是**一个都不能漏**。

        ★ 写入顺序（②在③前）：先记幂等表、再落 tool 结果。若崩在这两步之间，
        recover 能按 tool_call_id 从幂等表查回真结果补桩，不必重跑有副作用的工具。
        result 包成 {"content": result} 是因为 dispatch 返回的是裸字符串（不一定是
        JSON），而幂等表按 dict 存、recover 按 cached["content"] 取——两头必须对齐。
        """
        for tc in assistant.tool_calls:
            result = registry.dispatch(tc.name, tc.arguments)
            self.db.record_executed_key(tc.id, turn.session_id, {"content": result})
            self._record(turn, Message(role="tool", tool_call_id=tc.id, content=result))

    # ---------- 主循环 ----------

    def run_conversation(self, session_id: str, user_message: str) -> dict:
        """跑一轮用户输入到最终文本回复。返回 {final_response, completed, used_turns}。

        session_id 由调用方给（Go 铸造、经 gRPC 传入；CLI 自己 mint），本方法不
        持有它 —— 这正是"一个实例服务所有 session"的前提。

        起手三步：
          1. ensure_session —— 幂等补上 sessions 行。Go 那边"有"这个 session 不等于
             我们库里有，而 messages 有外键指向 sessions，不补首次 append 必炸。
          2. recover —— 历史的唯一真相在库里。它顺带把上次崩溃留下的残缺轮补桩、
             并从幂等表回填已执行工具的真结果，所以"上次跑到一半挂了"在每轮开头
             就被免费处理掉了。新 session 返回 []，于是新建/续跑两条路合并成一条。
          3. 记下用户这句话 —— 除非上一回合没跑完（见 _turn_in_flight）。那说明
             这次是调用方在重试，用户那句话早已落库，再记一遍就重复了；此时直接
             从补桩后的历史续跑。

        终止：模型返回纯文本（不带 tool_calls）即完成。
        失控保护：IterationBudget 到顶后，再给最后一次"不带 tools"的收尾调用，
        逼模型用文本作答，而不是无限调工具。
        """
        self.db.ensure_session(session_id)
        history = self.db.recover(session_id)
        turn = _Turn(session_id=session_id, messages=history)

        if not _turn_in_flight(history):
            self._record(turn, Message(role="user", content=user_message))

        budget = IterationBudget(self.max_turns)
        while budget.consume():
            assistant = self._call_model(turn, self._tool_defs())
            self._record(turn, assistant)

            if not assistant.tool_calls:
                return {
                    "final_response": assistant.content or "",
                    "completed": True,
                    "used_turns": budget.used,
                }

            self._execute_tool_calls(turn, assistant)   # 执行 + 回填，然后回循环顶

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
