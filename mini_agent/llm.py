"""唯一的 LLM 适配层（DeepSeek-only）。

[M0] DeepSeek 与 OpenAI 协议兼容：同样的 /chat/completions 形状、同样的
OpenAI 风格 tools / tool_calls。所以这里只是「openai SDK + base_url 改写」，
内部消息本来就是 OpenAI 风格，无需任何格式转换。

对照 CLAUDE.md 的 providers.py：原设计有 to_openai()/to_anthropic() 双分支 +
provider 参数贯穿 loop；DeepSeek-only 把它坍缩成一个函数。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import openai
from openai import OpenAI

from mini_agent.config import settings
from mini_agent.log import get_logger

_client: OpenAI | None = None

logger = get_logger(__name__)


@dataclass(frozen=True)
class RawUsage:
    """上游给的原始计数，已通过"是两个真 int"的校验。

    ★ 为什么不用 tuple。 两头（生产者 llm、消费者 agent）用 tuple 时，第几个是
      prompt、第几个是 completion 全靠"记得住"，测试替身也各造各的形状。
      具名 dataclass 让两边只共享这一个契约，且这里**只做校验不做换算**
      （>0 / fingerprint 是 context.tokens 的事）。
    """

    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class LLMReply:
    """一次调用的完整产出：assistant 消息 + 上游给的真实用量。

    ★ 为什么把 usage 单独拆出来，而不是塞进 message 对象。 message 是"要回填进
      transcript 的内容"，usage 是"这次请求花了多少"的元数据，两者生命周期不同：
      usage 只会被挂到**产生它的那条 assistant** 上（见 agent._call_model），
      而且 Message.from_openai 根本看不到 response 顶层的 usage —— 它只拿到 message。
    """

    message: Any
    # 上游没给或不合法时是 None。连同 fingerprint 落库是 agent 的事
    # （schema.TokenUsage），这里已经过了类型校验，agent 只负责挂上去。
    usage: RawUsage | None


def _get_client() -> OpenAI:
    """惰性构造客户端：避免 import 时就因缺 key 而报错（方便测试 mock）。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.base_url,          # DeepSeek = OpenAI 兼容端点
            timeout=settings.request_timeout,    # SDK 默认 600s 太长：工具循环里一个
                                                 # 卡住的连接不该让整个 REPL 干等
            # max_retries 用 SDK 默认的 2：对 429 / 5xx / 网络错误自动指数退避重试
        )
    return _client


def _extract_usage(resp: Any) -> RawUsage | None:
    """从 response.usage 取 RawUsage(prompt, completion)。取不到或不合法都返回 None。

    ★ 不抛异常、不做类型转换。 这里的输入是**网络的另一端**给的数据，而它的用途是
      "拿来做压缩的锚点"——一个残值的正确结局是"当没有锚点、退回估算"，不是让
      一次正常的 LLM 调用因为计数字段长得怪而失败。上游若给了怪东西，记一条日志
      就走。合法性（> 0 / >= 0）由 context.tokens 统一判定，这里只负责"有没有"。
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None)
    # ★ type(x) is int，不是 isinstance：bool 是 int 的子类，isinstance(True, int)
    #   为真，于是 True 会冒充 "1 个 token" 一路走到 TokenUsage —— 那里的 StrictInt
    #   会抛 ValidationError，把一次**成功**的调用打死。注释承诺的"顺手排除"必须
    #   真的排除。OpenAI SDK 的计数字段就是普通 int，没有要接受的 int 子类。
    if not (type(prompt) is int and type(completion) is int):
        logger.warning("上游 usage 字段不是整数，按无 usage 处理：%r", usage)
        return None
    return RawUsage(prompt_tokens=prompt, completion_tokens=completion)


# 上下文超长的判据。DeepSeek 官方文档没写这个错误的格式，下面两条来自第三方报告：
# HTTP 400 / invalid_request_error，message 形如 "This model's maximum context
# length is N tokens. However, you requested M tokens ..."；而 `code` 字段**不稳定**，
# 有时是 "context_length_exceeded"，有时只是 "invalid_request_error"。所以 code 和
# message 任一命中即可；另有托管商对"prompt + max_tokens 超窗口"返回 422，一并认。
_OVERFLOW_CODE = "context_length_exceeded"
_OVERFLOW_PHRASE = "maximum context length"


def is_context_overflow(exc: BaseException) -> bool:
    """这个异常是不是"请求超过模型上下文窗口"。**永不抛异常。**

    ★ 识别必须窄。 误判的代价是白花一次总结调用、压完原样再 400；一个普通的格式
      错误 400（比如 "Invalid request body format"）绝不能触发压缩。所以先卡类型
      （只认 400 / 422 两个子类，401/429/5xx 和非 openai 异常一律不认），再卡
      code 或 message 的特征，缺一不可。
    ★ SDK 细节只留在这里：llm.py 是唯一碰网络的地方，agent 只拿一个布尔值，
      不必认识 openai 的异常层级。
    ★ 全程防御性取值：这是在**错误处理路径**上跑的代码，它自己再抛异常会把原始
      错误盖掉。取不到的字段一律当"没有"。
    """
    try:
        if not isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
            return False
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None)
        body = getattr(exc, "body", None)
        # body 可能是 {"error": {...}}（原始响应体），也可能已经是内层 {...}
        if isinstance(body, dict):
            inner = body.get("error")
            if isinstance(inner, dict):
                body = inner
            if code is None:
                code = body.get("code")
            if not isinstance(message, str):
                message = body.get("message")
        if isinstance(code, str) and code == _OVERFLOW_CODE:
            return True
        return isinstance(message, str) and _OVERFLOW_PHRASE in message.lower()
    except Exception:
        return False


def call_llm(
    messages: list[dict[str, Any]],
    tools: list[dict] | None = None,
    *,
    max_tokens: int | None = None,
) -> LLMReply:
    """发一次 DeepSeek 调用，返回 LLMReply(message, usage)。

    messages 已是 OpenAI 风格 dict；DeepSeek 直接吃，无需转换。
    reply.message 含 .content 与 .tool_calls（供 Message.from_openai 消费）。

    ★ max_tokens=None 时**不传**这个参数，而不是传 None 或 0。 主对话走 SDK
      默认（模型能吐多少吐多少）；只有压缩总结才设上限。把 None 原样塞进请求体，
      SDK 会把它序列化成 null，上游对 null 和在不在是两回事 —— 主对话的行为不该
      因为加了"可选参数"而变。

    超时由客户端层统一兜底（见 _get_client 的 timeout=settings.request_timeout），
    覆盖每一次调用，这里不必逐次穿参。将来若某一炮需要不同超时，再加
    关键字可选参数 `*, timeout=None` 覆盖即可。
    """
    extra: dict[str, Any] = {}
    if max_tokens is not None:
        extra["max_tokens"] = max_tokens
    resp = _get_client().chat.completions.create(
        model=settings.model,
        messages=messages,
        tools=tools or None,   # 为空时传 None，不要传 []
        **extra,
    )
    return LLMReply(message=resp.choices[0].message, usage=_extract_usage(resp))
