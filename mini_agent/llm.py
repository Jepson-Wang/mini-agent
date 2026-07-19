"""唯一的 LLM 适配层（DeepSeek-only）。

[M0] DeepSeek 与 OpenAI 协议兼容：同样的 /chat/completions 形状、同样的
OpenAI 风格 tools / tool_calls。所以这里只是「openai SDK + base_url 改写」，
内部消息本来就是 OpenAI 风格，无需任何格式转换。

对照 CLAUDE.md 的 providers.py：原设计有 to_openai()/to_anthropic() 双分支 +
provider 参数贯穿 loop；DeepSeek-only 把它坍缩成一个函数。
"""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from mini_agent.config import settings

_client: OpenAI | None = None


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


def call_llm(messages: list[dict[str, Any]], tools: list[dict] | None = None) -> Any:
    """发一次 DeepSeek 调用，返回 OpenAI 风格的 message 对象。

    messages 已是 OpenAI 风格 dict；DeepSeek 直接吃，无需转换。
    返回对象含 .content 与 .tool_calls（供 Message.from_openai 消费）。

    超时由客户端层统一兜底（见 _get_client 的 timeout=settings.request_timeout），
    覆盖每一次调用，这里不必逐次穿参。将来若某一炮需要不同超时，再加
    关键字可选参数 `*, timeout=None` 覆盖即可。
    """
    resp = _get_client().chat.completions.create(
        model=settings.model,
        messages=messages,
        tools=tools or None,   # 为空时传 None，不要传 []
    )
    return resp.choices[0].message
