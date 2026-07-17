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
