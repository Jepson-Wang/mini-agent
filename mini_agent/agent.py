"""核心 Agent 类与主对话循环。

[M0] 最小可运行版本：无工具、单 provider（DeepSeek）。
后续里程碑在此基础上增量叠加：
  M1 工具系统（tool_calls 解析/执行/回填）
  M2 持久化（每轮即时落库）
  M3 上下文压缩（preflight 检查）
  M4 subagent 委派（delegate_task 拦截）
  M5 self-improving（memory/skills 注入）
"""
from __future__ import annotations

import threading

from mini_agent.llm import call_llm
from mini_agent.schema import Message

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
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 25,
    ):
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.messages: list[Message] = []   # 不含 system；system 在发送时拼接
        self._budget_grace_call = True      # 在 budget 耗尽之后给llm最后一次机会，输出答案

    @property
    def system_dict(self) -> dict:
        return {"role": "system", "content": self.system_prompt}

    def run_conversation(self, user_message: str) -> dict:
        """跑一轮用户输入到最终文本回复。返回 dict。

        [M0] 无工具：模型一旦返回纯文本即终止。
        """
        self.messages.append(Message(role="user", content=user_message))

        budget = IterationBudget(self.max_turns)
        while budget.consume() or self._budget_grace_call:
            api_msgs = [self.system_dict] + [m.to_openai() for m in self.messages]
            raw = call_llm(api_msgs)                  # M0：不传 tools
            assistant = Message.from_openai(raw)
            self.messages.append(assistant)

            if not assistant.tool_calls:
                # 纯文本回复 = 终止条件
                return {
                    "final_response": assistant.content or "",
                    "completed": True,
                    "used_turns": budget.used,
                }

            if self._budget_grace_call:
                # 在budget耗尽的最后一次call之后，将这个标志位设为False，防止一直调用
                self._budget_grace_call = False

            # M1 起：此处执行 tool_calls、回填 tool 结果，然后 continue
            # M0 没有工具注入，理论上不会走到这里


        # 预算耗尽
        return {
            "final_response": "(stopped: iteration budget exhausted)",
            "completed": False,
            "used_turns": budget.used,
        }
