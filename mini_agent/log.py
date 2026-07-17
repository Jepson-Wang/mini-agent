"""集中日志模块。

给全项目一个统一的 logger 工厂：其他模块只需 `from log import get_logger`，
`logger = get_logger(__name__)`，然后 `logger.info(...)` / `logger.debug(...)`。
本模块只负责“配置日志系统”，不替其他文件写任何日志调用。

设计要点（对照生产做法）：
- 只在**根 logger `mini_agent`** 上挂 handler，子 logger（如 `mini_agent.agent`）
  靠 logging 的层级传播（propagate）自动继承，避免每个模块各自加 handler
  导致日志重复打印。
- **幂等初始化**：`_setup()` 用一个模块级标志位保证只配置一次，import 多次也安全。
- **双 handler**：控制台（给人看，简洁）+ 滚动文件（留证据，带时间/位置，
  `RotatingFileHandler` 防止无限增长）。
- 日志自身的行为全部走**环境变量**，不往 Settings 里塞日志字段：
  - `MINI_AGENT_LOG_LEVEL`   日志级别，默认 INFO（DEBUG/INFO/WARNING/ERROR）
  - `MINI_AGENT_LOG_DIR`     日志目录，默认 `settings.home / "logs"`
  - `MINI_AGENT_LOG_CONSOLE` 是否打到控制台，默认 1（设 0 关闭，跑测试时有用）
  唯一从 config 拿的东西是 `settings.home`——「状态放哪」是全局决策，
  必须单点定义，不能让 log.py 和 __main__.py 各自硬编码一遍路径。
  （依赖方向：log → config。config 不 import 项目内任何模块，所以不成环；
   这条规矩要守住，否则哪天 config 想打条日志就死锁在 import 上。）
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from mini_agent.config import settings

# 整个项目共用的根 logger 名字；所有子 logger 都是它的后代
ROOT_LOGGER_NAME = "mini_agent"

# 保证 handler 只挂一次
_configured = False

# 文件格式带足够定位信息（时间 / 级别 / logger 名 / 行号 / 消息）
_FILE_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s:%(lineno)d] %(message)s"
# 控制台给人看，去掉冗余，保留级别和消息
_CONSOLE_FORMAT = "%(levelname)-7s %(name)s | %(message)s"

_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_level() -> int:
    """把 env 里的级别字符串解析成 logging 常量，非法值回退 INFO。"""
    raw = os.getenv("MINI_AGENT_LOG_LEVEL", "INFO").upper()
    # getLevelName 对合法名字返回 int，对未知名字返回字符串，所以要判类型
    level = logging.getLevelName(raw)
    return level if isinstance(level, int) else logging.INFO


def _resolve_log_dir() -> Path:
    """日志目录，默认与 state.db 同在 settings.home 下。目录不存在则创建。"""
    default = settings.home / "logs"
    log_dir = Path(os.getenv("MINI_AGENT_LOG_DIR", str(default))).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _setup() -> None:
    """给根 logger 配置好 handler。幂等：重复调用只生效一次。"""
    global _configured
    if _configured:
        return

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(_resolve_level())
    # 不要向 Python 全局 root logger 冒泡，避免和 basicConfig 之类重复输出
    root.propagate = False

    # —— 滚动文件 handler：单文件上限 2MB，保留 5 个备份 ——
    file_handler = RotatingFileHandler(
        _resolve_log_dir() / "mini_agent.log",
        maxBytes=2 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, _DATE_FORMAT))
    root.addHandler(file_handler)

    # —— 控制台 handler（可用 env 关掉，测试时清爽）——
    if os.getenv("MINI_AGENT_LOG_CONSOLE", "1") != "0":
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
        root.addHandler(console)

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """取一个已配置好的 logger。

    用法：`logger = get_logger(__name__)`。传入模块的 `__name__`（如
    `mini_agent.agent`）会得到根 logger 的子 logger，自动继承 handler 与级别。
    传 None 则返回根 logger 本身。
    """
    _setup()
    if name is None or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    # 确保拿到的是 mini_agent.* 命名空间下的 logger，才能挂到我们的根上
    if not name.startswith(ROOT_LOGGER_NAME + "."):
        name = f"{ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
