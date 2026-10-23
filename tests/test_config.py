"""配置层的约束。目前只有一条，但它值得单独一个文件：密钥不能进 repr。"""
from __future__ import annotations

from pathlib import Path

from mini_agent.config import DEFAULT_WORKSPACE_NAME, Settings, workspace_root


# ---------------------------------------------------------------------------
# 密钥不得出现在 repr 里
# ---------------------------------------------------------------------------

def test_repr_hides_secrets():
    """dataclass 默认的 __repr__ 会把每个字段原样打出来。

    也就是说一句 logger.debug("settings=%s", settings) 就能把真 API key 写进
    日志系统 —— 而日志通常是收集走、长期保存、很多人能看的。异常处理器打印
    帧局部变量时同理。

    对照 Go 那侧：密钥字段一律 yaml:"-"，启动日志只打"带没带凭证"这个布尔值。
    这条用例是同一条规矩在 Python 上的护栏。
    """
    s = Settings(deepseek_api_key="sk-REAL-KEY-DO-NOT-LOG",
                 grpc_token="TOKEN-DO-NOT-LOG")
    r = repr(s)

    assert "sk-REAL-KEY-DO-NOT-LOG" not in r
    assert "TOKEN-DO-NOT-LOG" not in r
    # 但非密钥字段还要看得见，否则 repr 就没用了。
    assert "deepseek-v4-flash" in r or "model=" in r


def test_str_hides_secrets_too():
    """dataclass 的 __str__ 默认就是 __repr__，顺手钉住 —— 日志里更常见的是 %s。"""
    s = Settings(deepseek_api_key="sk-REAL-KEY", grpc_token="TOKEN-X")
    assert "sk-REAL-KEY" not in str(s)
    assert "TOKEN-X" not in str(s)


# ---------------------------------------------------------------------------
# 沙箱根目录的默认值（和 test_file_tools 里那两条互补：这里只看 config 本身）
# ---------------------------------------------------------------------------

def test_default_workspace_hangs_off_home(monkeypatch):
    monkeypatch.delenv("MINI_AGENT_WORKSPACE", raising=False)
    s = Settings(deepseek_api_key="x", home=Path("/tmp/whatever"))
    # 构造 Settings 时不读 env，默认值就是挂在 DEFAULT_HOME 下面的那个。
    assert s.workspace.name == DEFAULT_WORKSPACE_NAME


def test_env_overrides_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("MINI_AGENT_WORKSPACE", str(tmp_path / "ws"))
    assert workspace_root() == (tmp_path / "ws").resolve()
