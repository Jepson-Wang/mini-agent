"""[M1] 文件工具：read_file / write_file / list_dir。

和 web_fetch 一样，这是安全边界——**路径由模型说了算**。放任不管的话，模型
一句 read_file('.env') 就能把你的 API key 读出来，write_file('/etc/hosts', ...)
就能改系统文件。所以这个文件的核心不是「读写」，而是「把读写关进一个工作区沙箱」：

  所有路径都先 resolve()（会展开 .. 和符号链接），再校验它没有逃出工作区根目录。
  越界的一律拒绝。这就是 web_fetch 那套「解析后校验目标」在文件系统上的翻版。

工作区根目录：环境变量 MINI_AGENT_WORKSPACE，默认是启动时的当前目录。
读写只用标准库（pathlib），不引入任何依赖。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mini_agent.log import get_logger
from mini_agent.tools.registry import registry

logger = get_logger(__name__)

# 单次 read_file 的字节上限：防止一个大文件把上下文/内存撑爆
FILE_READ_MAX_BYTES = 256 * 1024
# list_dir 一次最多列多少条
LIST_MAX_ENTRIES = 200


class FileToolError(Exception):
    """策略拒绝（越界、路径为空……）。会被转成 {"error": ...} 回给模型。"""


def _err(message: str) -> str:
    """文件工具的错误出口：永远是 {"error": ...} 的合法 JSON 字符串。"""
    return json.dumps({"error": message}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 沙箱：所有路径都必须落在工作区根目录内
# ---------------------------------------------------------------------------

def _workspace_root() -> Path:
    """工作区根目录。每次调用都现算，好让测试能用 env 临时改。"""
    raw = os.getenv("MINI_AGENT_WORKSPACE") or os.getcwd()
    root = Path(raw).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _resolve_in_workspace(path: str) -> Path:
    """把模型给的 path 解析成绝对路径，并确认它没逃出工作区。越界就抛 FileToolError。

    关键是 resolve()：它会把 '../../etc/passwd'、绝对路径、符号链接**全部展开**成
    真实路径，我们再用 is_relative_to 判断真实路径在不在根目录里。所以：
      - '../x'（相对越界）→ 展开后在根外 → 拒
      - '/etc/passwd'（绝对路径）→ root / 绝对路径 = 绝对路径 → 在根外 → 拒
      - 指向根外的符号链接 → 展开后在根外 → 拒
    """
    if not path or not str(path).strip():
        raise FileToolError("path 不能为空")
    root = _workspace_root()
    candidate = (root / path).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise FileToolError(
            f"拒绝访问工作区外的路径: {path!r}。文件工具只能在工作区内读写。"
        )
    return candidate


def _rel(p: Path) -> str:
    """把绝对路径显示成相对工作区的形式，不向模型泄露绝对路径。"""
    try:
        return str(p.relative_to(_workspace_root())).replace("\\", "/")
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

def read_file(path: str) -> str:
    """读取工作区内一个 UTF-8 文本文件，返回内容（JSON 字符串）。"""
    try:
        target = _resolve_in_workspace(path)
    except FileToolError as e:
        return _err(str(e))

    if not target.exists():
        return _err(f"文件不存在: {path}")
    if target.is_dir():
        return _err(f"这是目录不是文件: {path}（列目录请用 list_dir）")

    try:
        with target.open("rb") as f:
            raw = f.read(FILE_READ_MAX_BYTES + 1)   # 多读 1 字节判断是否截断
    except OSError as e:
        return _err(f"读取失败: {type(e).__name__}: {e}")

    truncated = len(raw) > FILE_READ_MAX_BYTES
    raw = raw[:FILE_READ_MAX_BYTES]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _err(f"不是 UTF-8 文本文件（可能是二进制）: {path}")

    return json.dumps(
        {"path": _rel(target), "chars": len(text), "truncated": truncated, "text": text},
        ensure_ascii=False,
    )


def write_file(path: str, content: str) -> str:
    """把 content 写入工作区内的文件（覆盖已有内容），返回写入字节数。

    父目录不存在会自动创建（仍限定在工作区内）。
    """
    if not isinstance(content, str):
        return _err(f"content 必须是字符串，收到 {type(content).__name__}")
    try:
        target = _resolve_in_workspace(path)
    except FileToolError as e:
        return _err(str(e))

    if target.is_dir():
        return _err(f"目标是一个目录，不能当文件写: {path}")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
    except OSError as e:
        return _err(f"写入失败: {type(e).__name__}: {e}")

    logger.info("write_file 写入 %s（%d 字节）", _rel(target), len(data))
    return json.dumps({"path": _rel(target), "bytes": len(data), "ok": True}, ensure_ascii=False)


def list_dir(path: str = ".") -> str:
    """列出工作区内某个目录的条目（文件/子目录），返回 JSON。"""
    try:
        target = _resolve_in_workspace(path)
    except FileToolError as e:
        return _err(str(e))

    if not target.exists():
        return _err(f"目录不存在: {path}")
    if not target.is_dir():
        return _err(f"不是目录: {path}（读文件请用 read_file）")

    entries: list[dict] = []
    truncated = False
    try:
        # 目录在前、同类按名排序，方便模型阅读
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if len(entries) >= LIST_MAX_ENTRIES:
                truncated = True
                break
            is_dir = child.is_dir()
            size = None
            if not is_dir:
                try:
                    size = child.stat().st_size
                except OSError:
                    size = None
            entries.append({"name": child.name, "type": "dir" if is_dir else "file", "size": size})
    except OSError as e:
        return _err(f"列目录失败: {type(e).__name__}: {e}")

    return json.dumps(
        {"path": _rel(target), "count": len(entries), "truncated": truncated, "entries": entries},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 自注册（必须在模块顶层——discover_builtin_tools 靠 AST 扫的就是这些调用）
# ---------------------------------------------------------------------------

READ_FILE_SCHEMA = {
    "name": "read_file",
    "description": "读取工作区内一个 UTF-8 文本文件的内容。路径相对工作区根目录。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的文件路径，如 notes/a.txt"},
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA = {
    "name": "write_file",
    "description": "把文本写入工作区内的文件（覆盖已有内容，父目录会自动创建）。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的文件路径"},
            "content": {"type": "string", "description": "要写入的完整文本内容"},
        },
        "required": ["path", "content"],
    },
}

LIST_DIR_SCHEMA = {
    "name": "list_dir",
    "description": "列出工作区内某个目录下的文件和子目录。默认列工作区根目录。",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对工作区的目录路径，默认 '.'"},
        },
        "required": [],
    },
}

# 不设 check_fn：安全性由工作区沙箱保证（越界一律拒），不靠开关。
# 若想更保守，可给 write_file 单独加一个 MINI_AGENT_ALLOW_WRITE 门控。
registry.register(name="read_file", toolset="file", schema=READ_FILE_SCHEMA,
                  handler=read_file, description=READ_FILE_SCHEMA["description"])
registry.register(name="write_file", toolset="file", schema=WRITE_FILE_SCHEMA,
                  handler=write_file, description=WRITE_FILE_SCHEMA["description"])
registry.register(name="list_dir", toolset="file", schema=LIST_DIR_SCHEMA,
                  handler=list_dir, description=LIST_DIR_SCHEMA["description"])
