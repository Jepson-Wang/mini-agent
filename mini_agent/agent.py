"""核心 Agent 类与主对话循环。

演进：
  [M0] 最小对话循环：无工具、单 provider（DeepSeek）。
  [M1] 工具系统：每轮把 get_definitions() 传给模型，解析 tool_calls、
       dispatch 执行、把结果作为 role="tool" 回填，再回到循环顶。
  [M2] 持久化：每 append 一条内存 Message 就同步落库；--resume 时用
       db.recover() 从库里重建对话继续。
后续：
  M3 上下文压缩（preflight 检查）
  M4 subagent 委派（delegate_task 拦截）
  M5 self-improving（memory/skills 注入）
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from mini_agent.llm import call_llm
from mini_agent.schema import Message
from mini_agent.tools.registry import registry, discover_builtin_tools

if TYPE_CHECKING:
    from mini_agent.persistence.session_db import MiniSessionDB

DEFAULT_SYSTEM_PROMPT = (
    "You are mini-agent, a helpful AI assistant. "
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


class Agent:
    def __init__(
        self,
        *,
        session_id: Optional[str] = None,
        db: "Optional[MiniSessionDB]" = None,
        toolset: Optional[set[str]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 25,
        depth: int = 0,
    ):
        # M2：session_id + db 同时给才开启持久化。两者缺一即纯内存模式
        #     （M0 测试、一次性调用都走这条），行为与最初完全一致。
        self.session_id = session_id
        self.db = db
        # M1：toolset 决定这个 agent 能看到哪些工具（None = 不带工具，退回 M0）。
        #     M4 子代理靠"父 toolset 的子集"实现最小权限。
        self.toolset = toolset
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.depth = depth                    # M4 递归深度，M1/M2 恒为 0
        self.messages: list[Message] = []     # 不含 system；system 在发送时拼接

        if toolset:
            # 显式、幂等地做一次工具发现：一个"带工具的 agent"就该保证工具已注册。
            # import 已缓存，多个 agent 重复调也只有第一次真正扫盘。
            discover_builtin_tools()

    @property
    def system_dict(self) -> dict:
        return {"role": "system", "content": self.system_prompt}

    # ---------- 内部：落库 + 工具 ----------

    def _record(self, msg: Message) -> None:
        """把一条 Message 同时写进内存和库。

        M2 的核心：每产生一条消息就立刻落库，崩溃时丢的最多是"最后一条还没写完
        的"。持久化未开启（db/session_id 缺）时退化成纯内存 append，M0 行为不变。
        append_message 失败会抛 PersistenceError —— 故意不吞：内存和库一旦静默
        分叉，崩溃恢复就失去意义（fail-fast）。
        """
        self.messages.append(msg)
        if self.db is not None and self.session_id is not None:
            self.db.append_message(self.session_id, msg)

    def _tool_defs(self) -> Optional[list[dict]]:
        """本轮要暴露给模型的工具列表。无 toolset → None（等价 M0，不带工具）。"""
        if not self.toolset:
            return None
        return registry.get_definitions(toolset=self.toolset) or None

    def _call_model(self, tools: Optional[list[dict]]) -> Message:
        """拼上 system、发一次调用、解析回内部 Message。"""
        api_msgs = [self.system_dict] + [m.to_openai() for m in self.messages]
        raw = call_llm(api_msgs, tools=tools)
        return Message.from_openai(raw)

    def _execute_tool_calls(self, assistant: Message) -> None:
        """串行执行 assistant 的每个 tool_call，把结果作为 role="tool" 回填。

        ★ 配对铁律：带 tool_calls 的 assistant 后面，必须紧跟"数量一致、
        tool_call_id 一一对应"的 tool 消息，否则下一次请求会 400。串行执行、
        每个 tc 都回填一条，天然满足；关键是**一个都不能漏**。
        dispatch 保证永远返回合法 JSON 字符串，所以这里不会抛。
        """
        for tc in assistant.tool_calls:
            result = registry.dispatch(tc.name, tc.arguments)
            self._record(Message(role="tool", tool_call_id=tc.id, content=result))

    # ---------- 主循环 ----------

    def run_conversation(self, user_message: str) -> dict:
        """跑一轮用户输入到最终文本回复。返回 {final_response, completed, used_turns}。

        终止：模型返回纯文本（不带 tool_calls）即完成。
        失控保护：IterationBudget 到顶后，再给最后一次"不带 tools"的收尾调用，
        逼模型用文本作答，而不是无限调工具。
        """
        self._record(Message(role="user", content=user_message))

        budget = IterationBudget(self.max_turns)
        while budget.consume():
            assistant = self._call_model(self._tool_defs())
            self._record(assistant)

            if not assistant.tool_calls:
                return {
                    "final_response": assistant.content or "",
                    "completed": True,
                    "used_turns": budget.used,
                }

            self._execute_tool_calls(assistant)   # 执行 + 回填，然后回循环顶

        # 预算耗尽：给最后一次机会，强制不带 tools 逼模型收尾成文本。
        # 此时 self.messages 以一组 tool 结果结尾（合法可发送状态）。
        assistant = self._call_model(tools=None)
        if assistant.tool_calls:
            # 收尾调用模型还想调工具（真实场景 tools=None 不会发生）：
            # 不执行、也不 record —— 避免在 messages 末尾留下悬空的 tool_calls，
            # 否则下一轮把它发出去就是 400。
            return {
                "final_response": "(stopped: iteration budget exhausted)",
                "completed": False,
                "used_turns": budget.used,
            }
        self._record(assistant)
        return {
            "final_response": assistant.content or "",
            "completed": True,
            "used_turns": budget.used,
        }
