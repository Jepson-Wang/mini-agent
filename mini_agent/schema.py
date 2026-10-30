"""内部规范消息格式（narrow waist）。

[M0] 内部统一用 OpenAI 风格的消息。因为本项目只接 DeepSeek（OpenAI 兼容），
所以「内部格式 ↔ provider 格式」的转换几乎是恒等的——这正是选对内部规范格式
的回报：provider 适配几乎零成本。

[M3] assistant 消息可以随身携带 TokenUsage：那是上游对**产生它的那次请求**的
真实计数，是上下文压缩决定"要不要压"时唯一可信的锚点，见 context/tokens.py。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, StrictInt, ValidationError, field_validator

from mini_agent.log import get_logger

logger = get_logger(__name__)

#   各个 Role 的功能
#   system     : 你是 mini-agent...           (开发者设定)
#   user       : 帮我读一下 a.txt             (用户提问)
#   assistant  : (content=None,               (模型决定调工具)
#                 tool_calls=[{id:"call_1", name:"read_file", args:{path:"a.txt"}}])
#   tool       : (tool_call_id="call_1",      (你执行后回填结果)
#                 content='{"ok":true,"text":"..."}')
Role = Literal["system", "user", "assistant", "tool"]


class TokenUsage(BaseModel):
    """上游返回的一次真实用量，挂在产生它的 assistant 消息上。

    ★ 用 StrictInt 而不是 int。 pydantic 的宽松模式会把 True 当 1、把 "12" 当 12
      收下来；而这里的输入有一部分来自**库里的历史 JSON**，recover 对坏行是
      "跳过 + 告警"的降级策略 —— 一个字符串计数被悄悄当成数字，等于拿一个假锚点
      去决定压不压缩，且没有任何人报错。宁可这条 usage 当不存在，也不猜。
      ★ "不存在"的范围只限 usage 自己，不外溢到整条消息：usage 坏掉的结局由
      Message 的 wrap 校验兜成 None，见 Message.usage。

    fingerprint 记录的是"这个数字描述的是哪种请求前缀"（system + tools + 压缩边界），
    见 context/tokens.py :: request_fingerprint。前缀一变它就作废，measure 会把这条 usage
    当不存在、退回估算。
    """

    prompt_tokens: StrictInt
    completion_tokens: StrictInt
    fingerprint: str


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
    # 仅 assistant：产生这条消息的那次请求的用量，见 TokenUsage。
    #
    # ★ 不用 schema 迁移。 messages 表存的是整条 Message 的 JSON，旧行没有这个 key，
    #   model_validate_json 得到的就是 None —— 正好是"没有锚点"的正确语义。
    # ★ 也不会发给模型。 to_openai() 是逐字段**手工**构造的，不读这个字段；
    #   若哪天有人把它改成 model_dump()，这条 usage 就会混进请求体，DeepSeek 会 400。
    usage: TokenUsage | None = None

    @field_validator("usage", mode="wrap")
    @classmethod
    def _tolerate_bad_usage(cls, value: Any, handler) -> TokenUsage | None:
        """usage 校验失败 → 当没有锚点（None），**不要连累整条消息**。

        ★ 为什么必须 wrap 而不是让 TokenUsage 直接拒绝。 usage 是纯遥测：它只影响
          "压不压缩"，不影响 transcript 本身。可它一旦把异常抛给整条 Message，
          recover 的金规则（坏行跳过 + 告警）就会把一条**完全能用的 assistant**
          连同它的 tool_calls 一起丢掉，它窗口里的 tool 结果随之变成孤儿也被丢弃。
          一条遥测字段的脏值，换来的是历史凭空少一段，而且只打一条"跳过"日志。
          正确降级是"坏掉的那部分没了"：usage=None 等于没有锚点，measure 退回估算，
          偏保守、只可能多压，不会漏压。
        """
        try:
            return handler(value)
        except ValidationError as exc:
            # 只接 ValidationError：handler 里若冒出别的异常，那是代码 bug，不该被吞。
            logger.warning(
                "消息的 usage 校验失败，按无锚点处理（消息本身保留）：%s",
                exc.errors(include_url=False),
            )
            return None

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
        if isinstance(raw, dict):
            role = raw.get("role", "assistant")
            content = raw.get("content")
            raw_tcs = raw.get("tool_calls")
            tool_call_id = raw.get("tool_call_id")  # ← 补
        else:
            role = raw.role
            content = raw.content
            raw_tcs = getattr(raw, "tool_calls", None)
            tool_call_id = getattr(raw, "tool_call_id", None)  # ← 补
        tool_calls = [ToolCall.from_openai(tc) for tc in raw_tcs] if raw_tcs else None
        # 刻意不读上游的 usage：这里的 raw 只有 message，usage 在 response 顶层，
        # 由 llm.py 取出来交给 agent.py 连同 fingerprint 一起挂上。
        return cls(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
