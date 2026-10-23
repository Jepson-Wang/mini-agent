"""[M1] 文件工具测试 —— 重点是沙箱：模型不能读写工作区之外。

直接 import handler 调用（绕过 dispatch），因为要测的是工具本身的逻辑。
每个测试把 MINI_AGENT_WORKSPACE 指到临时目录，绝不碰真实文件系统。
"""
from __future__ import annotations

import json

import pytest

from mini_agent.tools.builtin.file_tools import read_file, write_file, list_dir


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """把工作区根目录钉在一个临时目录。"""
    monkeypatch.setenv("MINI_AGENT_WORKSPACE", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# 正常读写
# ---------------------------------------------------------------------------

def test_write_then_read_roundtrip(ws):
    w = json.loads(write_file("notes/hello.txt", "你好，世界"))
    assert w["ok"] is True
    assert w["path"] == "notes/hello.txt"      # 相对工作区，正斜杠
    assert (ws / "notes" / "hello.txt").exists()   # 父目录被自动创建

    r = json.loads(read_file("notes/hello.txt"))
    assert r["text"] == "你好，世界"
    assert r["truncated"] is False


def test_read_missing_file(ws):
    assert "不存在" in json.loads(read_file("nope.txt"))["error"]


def test_read_directory_is_error(ws):
    (ws / "sub").mkdir()
    assert "目录" in json.loads(read_file("sub"))["error"]


def test_read_non_utf8_is_error(ws):
    (ws / "bin.dat").write_bytes(b"\xff\xfe\x00\x01\x80")
    assert "UTF-8" in json.loads(read_file("bin.dat"))["error"]


def test_list_dir(ws):
    (ws / "a.txt").write_text("x", encoding="utf-8")
    (ws / "sub").mkdir()
    out = json.loads(list_dir("."))
    names = {e["name"]: e["type"] for e in out["entries"]}
    assert names == {"a.txt": "file", "sub": "dir"}


def test_write_rejects_non_string_content(ws):
    assert "字符串" in json.loads(write_file("x.txt", 123))["error"]


# ---------------------------------------------------------------------------
# 沙箱：越界一律拒（安全关键）
# ---------------------------------------------------------------------------

def test_read_rejects_parent_traversal(ws):
    """../ 逃出工作区 → 拒。模型不能用 ../../ 读到工作区外的文件。"""
    secret = ws.parent / "secret.txt"
    secret.write_text("TOP-SECRET", encoding="utf-8")

    out = json.loads(read_file("../secret.txt"))
    assert "工作区外" in out["error"]
    assert "TOP-SECRET" not in json.dumps(out, ensure_ascii=False)


def test_read_rejects_absolute_path_escape(ws):
    """绝对路径逃出工作区 → 拒。read_file('/etc/passwd') 这类要被挡住。"""
    secret = ws.parent / "abs_secret.txt"
    secret.write_text("SECRET", encoding="utf-8")

    out = json.loads(read_file(str(secret)))
    assert "工作区外" in out["error"]


def test_write_rejects_parent_traversal(ws):
    """写操作同样受沙箱约束：不能往工作区外覆盖文件。"""
    out = json.loads(write_file("../escaped.txt", "x"))
    assert "工作区外" in out["error"]
    assert not (ws.parent / "escaped.txt").exists()   # 确实没写出去


def test_empty_path_rejected(ws):
    assert "不能为空" in json.loads(read_file(""))["error"]


# ---------------------------------------------------------------------------
# 沙箱根目录本身（这一层出过事：边界对，根选错了）
# ---------------------------------------------------------------------------

def test_default_root_is_not_cwd(tmp_path, monkeypatch):
    """没设 MINI_AGENT_WORKSPACE 时，沙箱根**不能**是当前目录。

    这条是回归测试，护的是一个真实漏洞：根目录曾经默认 os.getcwd()，而进程
    通常就是从仓库根启动的，于是 .env 落在沙箱内 —— 模型一句 read_file('.env')
    就能把 API key 读出来并跟着回答回显给用户，正好是 file_tools 模块开头
    声称要防的那件事。边界（resolve + is_relative_to）一直是对的，错的是根。
    """
    from mini_agent.config import workspace_root

    monkeypatch.delenv("MINI_AGENT_WORKSPACE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=sk-should-not-be-readable",
                                   encoding="utf-8")

    assert workspace_root() != tmp_path.resolve()

    out = json.loads(read_file(".env"))
    assert "error" in out                       # 不存在 或 越界，都行
    assert "sk-should-not-be-readable" not in json.dumps(out, ensure_ascii=False)


def test_default_root_lives_under_home(monkeypatch):
    """默认沙箱挂在 <home>/workspace 下，和 state.db 是兄弟而不是祖先。

    是兄弟很关键：settings.home 本身当沙箱根的话，模型能直接 read_file('state.db')
    （虽然会因为不是 UTF-8 被挡回来）和 write_file('state.db', ...) 把库写坏。
    """
    from mini_agent.config import settings, workspace_root

    monkeypatch.delenv("MINI_AGENT_WORKSPACE", raising=False)
    root = workspace_root()
    assert root.parent == settings.home.resolve()
    assert root != settings.home.resolve()


def test_unusable_workspace_reports_json_not_raw_exception(tmp_path, monkeypatch):
    """工作区根目录建不出来时，三个工具都要给 {"error": ...}，且不带绝对路径。

    护的是一条真实的泄露路径：workspace_root() 内部会 mkdir，失败抛的是裸
    OSError。不接住的话它会穿过 read_file —— registry.dispatch 兜得住不至于
    打死 agent，但它会把异常文本包成 "Tool failed: ..." 回给模型，而那段文本
    里带着**绝对路径**，跟着 tool_result 事件流到 Go、写进任务详情、回显给用户。

    `_rel()` 存在的全部理由就是不让绝对路径漏给模型，这条路等于从后门送出去。
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("我是个文件，不是目录", encoding="utf-8")
    monkeypatch.setenv("MINI_AGENT_WORKSPACE", str(blocker / "sub"))

    for label, call in [
        ("read_file", lambda: read_file("a.txt")),
        ("write_file", lambda: write_file("a.txt", "x")),
        ("list_dir", lambda: list_dir(".")),
    ]:
        out = call()                       # 不抛就是第一关过了
        parsed = json.loads(out)           # 一定是合法 JSON
        assert "error" in parsed, label
        assert str(tmp_path) not in out, f"{label} 把绝对路径漏出去了"
        assert "工作区根目录不可用" in parsed["error"], label
