"""集中配置：读取 .env，暴露一个全局 settings 对象。

[M0] DeepSeek-only。DeepSeek 的 API 与 OpenAI 兼容，所以我们用 openai SDK
+ base_url 即可，不需要 anthropic / 多 provider 配置。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
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

# 文件工具（read_file / write_file / list_dir）的沙箱根目录。
#
# ★ 默认值绝不能是 os.getcwd()。file_tools 的沙箱边界（resolve + is_relative_to）
#   本身是对的，但边界圈住的是**根目录**——根设成 cwd，而项目通常就是从仓库根
#   启动的，于是 .env 恰好落在沙箱内，模型一句 read_file(".env") 就能把 API key
#   读出来，还会跟着 final_response 回显给用户。这正是 file_tools 模块开头声称
#   要防的那件事。
#
# 所以给它一个**专用的空目录**：默认 <home>/workspace。它和 state.db 是兄弟而不是
# 祖先，模型也读不到库文件。想让模型在别处干活，显式设 MINI_AGENT_WORKSPACE。
DEFAULT_WORKSPACE_NAME = "workspace"

# 默认工具集。web_fetch 另受 ALLOW_WEB 门控（没开就不会暴露给模型）。
#
# ★ 放在这里而不是某个 surface 里：CLI 和 gRPC server 是两个平级入口，谁持有它
#   另一个就得 import 对方。尤其不能留在 __main__.py —— `python -m mini_agent`
#   会让那个文件同时以 `__main__` 和 `mini_agent.__main__` 两个身份存在，import
#   它等于把整个 CLI 再执行一遍。两边各抄一份则更糟：改了一处，另一个 surface
#   的模型权限静默不同步。
DEFAULT_TOOLSET = {"file", "web", "math"}


@dataclass(frozen=True)
class Settings:
    # ★ repr=False 的两个字段都是密钥。
    #
    # dataclass 默认生成的 __repr__ 会把每个字段原样打出来，于是一句
    # logger.debug("settings=%s", settings) 就能把真 API key 写进日志系统 ——
    # 而日志通常是收集、长期保存、很多人能看的。异常处理器打印帧局部变量时同理。
    #
    # 对照 Go 那侧的做法：密钥字段一律 yaml:"-"，启动日志只打
    # "带没带凭证"这个布尔值，绝不打密钥、也不打它的长度或指纹。
    # 这里是同一条规矩在 Python 上的落地。
    #
    # 注意它只挡 repr，不挡 asdict()/序列化 —— 真要往外发之前还是得自己想清楚。
    deepseek_api_key: str = field(repr=False)
    base_url: str = DEEPSEEK_BASE_URL
    model: str = "deepseek-v4-flash"    # deepseek-chat 已下线，flash 为新命名下最接近的通用档位
    max_turns: int = 25                 # IterationBudget 默认上限（主 agent）
    request_timeout: float = 120.0      # 单次 LLM 调用超时（秒）；SDK 默认 600s 太长
    home: Path = DEFAULT_HOME           # 运行时状态根目录，见上

    # 文件工具的沙箱根，见 DEFAULT_WORKSPACE_NAME。真正取值走 workspace_root()。
    workspace: Path = DEFAULT_HOME / DEFAULT_WORKSPACE_NAME

    # ---- gRPC surface（mini_agent/grpcserver）----
    grpc_port: int = 50051              # 要和 Go 那边的 AGENT_ADDR 对上
    # 并发上限。一个任务占**两个**线程：gRPC 的 handler 线程（跑生成器、发事件）
    # 和它拉起的 worker 线程（跑 run_conversation）。所以这个数是"同时在跑的任务
    # 数"的上限，不是它的两倍。
    grpc_max_workers: int = 16

    # 监听地址。默认只听回环——这个 surface 能跑任意任务、读写工作区、烧 LLM
    # 额度，权限比 Go 那边的 /internal 还大，不该因为"默认值方便"就开到全网卡。
    # 要跨机器/跨容器访问，显式设 MINI_AGENT_GRPC_HOST 并同时配好 token。
    grpc_host: str = "127.0.0.1"

    # 服务间共享密钥。客户端放在 metadata 的 x-agent-token 里，
    # 对应 Go 那侧的 config.Agent.Token（AGENT_GRPC_TOKEN）。
    #
    # 空值只在绑回环时才允许（见 grpcserver.serve）：本机开发不必配，
    # 一旦往外绑就必须有。这个取向和 Go 的 AGENT_CALLBACK_SECRET 一致——
    # 密钥缺失时宁可拒绝服务，也不要"能跑但没防护"。
    grpc_token: str = field(default="", repr=False)

    def workspace_path(self) -> Path:
        """沙箱根目录（会建出来）。别直接用它——用模块级的 workspace_root()。"""
        self.workspace.mkdir(parents=True, exist_ok=True)
        return self.workspace

    def db_path(self) -> str:
        """<home>/state.db；目录不存在先建。CLI 和 gRPC server 共用同一个库。"""
        self.home.mkdir(parents=True, exist_ok=True)
        return str(self.home / "state.db")


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


def _workspace_from_env(home: Path) -> Path:
    """沙箱根：显式给了就用，没给就挂在 home 下面。绝不退回 cwd。"""
    raw = os.getenv("MINI_AGENT_WORKSPACE")
    if raw:
        return Path(raw).expanduser()
    return home / DEFAULT_WORKSPACE_NAME


def _load_settings() -> Settings:
    home = Path(os.getenv("MINI_AGENT_HOME", DEFAULT_HOME)).expanduser()
    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        base_url=os.getenv("MINI_AGENT_BASE_URL", DEEPSEEK_BASE_URL),
        model=os.getenv("MINI_AGENT_MODEL", "deepseek-v4-flash"),
        max_turns=_num_env("MINI_AGENT_MAX_TURNS", 25, int),
        request_timeout=_num_env("MINI_AGENT_REQUEST_TIMEOUT", 120.0, float),
        home=home,
        workspace=_workspace_from_env(home),
        grpc_port=_num_env("MINI_AGENT_GRPC_PORT", 50051, int),
        grpc_max_workers=_num_env("MINI_AGENT_GRPC_MAX_WORKERS", 16, int),
        grpc_host=os.getenv("MINI_AGENT_GRPC_HOST", "127.0.0.1"),
        grpc_token=os.getenv("MINI_AGENT_GRPC_TOKEN", ""),
    )


def workspace_root() -> Path:
    """文件工具的沙箱根：绝对路径，且保证目录存在。

    每次调用都重新读一次 MINI_AGENT_WORKSPACE，而不是只认 import 时算好的
    settings.workspace —— 测试要靠 monkeypatch.setenv 把沙箱钉在 tmp_path 上，
    settings 是 import 期一次性构造的，读它的话 env 改了也没用。

    没设 env 时退回 settings.workspace（<home>/workspace），**不是 cwd**。
    理由见 DEFAULT_WORKSPACE_NAME。
    """
    raw = os.getenv("MINI_AGENT_WORKSPACE")
    root = (Path(raw).expanduser() if raw else settings.workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


settings = _load_settings()
