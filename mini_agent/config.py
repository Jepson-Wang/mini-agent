"""集中配置：读取 .env，暴露一个全局 settings 对象。

[M0] DeepSeek-only。DeepSeek 的 API 与 OpenAI 兼容，所以我们用 openai SDK
+ base_url 即可，不需要 anthropic / 多 provider 配置。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# import 时自动加载项目根目录的 .env（若不存在则静默跳过）
load_dotenv()

# DeepSeek 的 OpenAI 兼容端点
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 本文件在 mini_agent/ 下：parent 是包目录，parent.parent 才是项目根。
# 用 __file__ 推导而不是 Path.cwd()，是因为包已 editable 安装，在任意目录
# 都能 `python -m mini_agent`——用 cwd 的话库和日志会跟着启动目录到处跑。
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# agent 的全部运行时状态（state.db / logs / 将来的 MEMORY.md、skills）都挂在这里。
# 放项目内而不是 ~/.mini_agent：每个 checkout 一份干净的状态，删了重来即可。
# （对照 Hermes：它放 ~/.hermes，因为它是跨项目共享的全局 CLI 工具。）
DEFAULT_HOME = PROJECT_ROOT / ".mini_agent"


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    base_url: str = DEEPSEEK_BASE_URL
    model: str = "deepseek-chat"        # V3 通用模型，完整支持 function calling
    max_turns: int = 25                 # IterationBudget 默认上限（主 agent）
    request_timeout: float = 120.0      # 单次 LLM 调用超时（秒）；SDK 默认 600s 太长
    home: Path = DEFAULT_HOME           # 运行时状态根目录，见上


def _num_env(name: str, default, cast):
    """读数值型环境变量：解析失败时退回默认值并告警，绝不让一个坏值炸掉 import。

    不 import log —— log 模块依赖 config，反过来 import 会成环。config 处在最
    底层，只能用 stderr 直接喊。
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        print(
            f"[config] 环境变量 {name}={raw!r} 无法解析为 {cast.__name__}，"
            f"退回默认值 {default}",
            file=sys.stderr,
        )
        return default


def _load_settings() -> Settings:
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("MINI_AGENT_BASE_URL", DEEPSEEK_BASE_URL),
        model=os.getenv("MINI_AGENT_MODEL", "deepseek-chat"),
        max_turns=_num_env("MINI_AGENT_MAX_TURNS", 25, int),
        request_timeout=_num_env("MINI_AGENT_REQUEST_TIMEOUT", 120.0, float),
        home=Path(os.getenv("MINI_AGENT_HOME", DEFAULT_HOME)).expanduser(),
    )


settings = _load_settings()
