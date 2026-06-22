"""集中配置：读取 .env，暴露一个全局 settings 对象。

[M0] DeepSeek-only。DeepSeek 的 API 与 OpenAI 兼容，所以我们用 openai SDK
+ base_url 即可，不需要 anthropic / 多 provider 配置。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# import 时自动加载项目根目录的 .env（若不存在则静默跳过）
load_dotenv()

# DeepSeek 的 OpenAI 兼容端点
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    base_url: str = DEEPSEEK_BASE_URL
    model: str = "deepseek-chat"        # V3 通用模型，完整支持 function calling
    max_turns: int = 25                 # IterationBudget 默认上限（主 agent）


def _load_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("MINI_AGENT_BASE_URL", DEEPSEEK_BASE_URL),
        model=os.getenv("MINI_AGENT_MODEL", "deepseek-chat"),
        max_turns=int(os.getenv("MINI_AGENT_MAX_TURNS", "25")),
    )


settings = _load_settings()
