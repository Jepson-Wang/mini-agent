"""内部规范消息格式（narrow waist）。

[M0] 内部统一用 OpenAI 风格的消息。因为本项目只接 DeepSeek（OpenAI 兼容），
所以「内部格式 ↔ provider 格式」的转换几乎是恒等的——这正是选对内部规范格式
的回报：provider 适配几乎零成本。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

#   各个 Role 的功能
#   system     : 你是 mini-agent...           (开发者设定)
#   user       : 帮我读一下 a.txt             (用户提问)
#   assistant  : (content=None,               (模型决定调工具)
#                 tool_calls=[{id:"call_1", name:"read_file", args:{path:"a.txt"}}])
#   tool       : (tool_call_id="call_1",      (你执行后回填结果)
#                 content='{"ok":true,"text":"..."}')
Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """模型发起的一次工具调用（arguments 已解析成 dict）。"""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }

    @classmethod
    def from_openai(cls, raw: Any) -> "ToolCall":
        """从 openai SDK 的 tool_call 对象（或等价 dict）构造。"""
        import json

        if isinstance(raw, dict):
            fn = raw.get("function", {})
            call_id = raw.get("id", "")
            name = fn.get("name", "")
            args_str = fn.get("arguments") or "{}"
        else:  # openai SDK 对象
            call_id = raw.id
            name = raw.function.name
            args_str = raw.function.arguments or "{}"
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {}
        return cls(id=call_id, name=name, arguments=arguments)


class Message(BaseModel):
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None   # 仅 assistant
    tool_call_id: str | None = None            # 仅 tool

    def to_openai(self) -> dict[str, Any]:
        """转成 openai/DeepSeek chat.completions 接受的 dict。"""
        d: dict[str, Any] = {"role": self.role}
        # assistant 带 tool_calls 时 content 可为 None；其余角色给空串兜底
        if self.content is not None:
            d["content"] = self.content
        elif not self.tool_calls:
            d["content"] = ""
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d

    @classmethod
    def from_openai(cls, raw: Any) -> "Message":
        """从 openai SDK 返回的 message 对象（或等价 dict）构造内部 Message。"""
        if isinstance(raw, dict):
            role = raw.get("role", "assistant")
            content = raw.get("content")
            raw_tcs = raw.get("tool_calls")
        else:  # openai SDK 对象
            role = raw.role
            content = raw.content
            raw_tcs = getattr(raw, "tool_calls", None)
        tool_calls = [ToolCall.from_openai(tc) for tc in raw_tcs] if raw_tcs else None
        return cls(role=role, content=content, tool_calls=tool_calls)
