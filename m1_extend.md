# M1 工具系统补完清单（m1_exetend）

> 目标：把 tool 系统从「只有骨架」补到「模型能真正调用 read_file/write_file 并继续对话」的可 demo 状态。
> 现状结论：**M1 基本只有骨架，跑不起来**。大部分文件为空，唯一有内容的 `registry.py` 存在运行时 bug。

---

## 0. 现状快照

| 文件 | 状态 | 说明 |
|---|---|---|
| `tools/registry.py` | ⚠️ 写了但有 bug | `discover_builtin_tools` / `ToolEntry` / `ToolRegistry`，跑不通 |
| `tools/model_tools.py` | 🔶 部分 | 只有 `_run_async`（async 桥接），无实际工具 |
| `tools/dispatch.py` | ❌ 空 | — |
| `tools/executor.py` | ❌ 空 | — |
| `tools/state_tools.py` | ❌ 空 | — |
| `tools/__init__.py` | ❌ 空 | — |
| `tools/builtin/file_tools.py` | ❌ 空 | 无 read_file / write_file |
| `tools/builtin/web_tool.py` | ❌ 空 | — |
| `tools/builtin/shell_tool.py` | ❌ 空 | — |
| `tests/test_tools.py` | ❌ 空 | — |

另外 `agent.py` 主循环停在 `# M1 起：此处执行 tool_calls` 占位注释，**Agent 未持有 / 未执行工具**，`call_llm` 也没传 `tools=`。

---

## 1. 🔴 必须完成的项（M1 跑通的最小集）

按建议实现顺序排列。

### 1.1 修 `registry.py` 的 bug + 补 `registry` 单例

**要修的 bug：**

- `registry.py:74-75`
  ```python
  self._tools = Dict[str,ToolEntry] = {}          # ❌ 运行时 TypeError（链式赋值给下标表达式）
  self._toolset_checks = Dict[str,Callable] = {}  # ❌ 同上
  ```
  应改为类型注解：
  ```python
  self._tools: Dict[str, ToolEntry] = {}
  self._toolset_checks: Dict[str, Callable] = {}
  ```

- `discover_builtin_tools`（line 36-53）
  - `_module_registers_tools(path)` 传的是**目录**，应传每个文件 `p`。
  - import 名 `f"builtin.{p.stem}"` 作为包无法解析，应是完整包路径 `f"mini_agent.tools.builtin.{p.stem}"`（或用相对 import）。
  - `path.glob('*.py')` 扫的是 `tools/` 目录，但 builtin 工具在 `tools/builtin/`，路径要对上。

- `dispatch`（line 126-136）
  - `from model_tools import _run_async` 相对路径错误，应为 `from .model_tools import _run_async`。
  - 调用约定 `entry.handler(args, **kwargs)` 把 dict 当第一个位置参数传；与设计 `handler(**args)` 不一致。**统一成 `handler(**args)`**。
  - 异常分支 `return json.dumps({"error":"调用工具函数错误"})` 丢掉了错误信息，应保留 `type(ex).__name__: ex`。

**要补的单例：** 模块末尾导出全局单例（AST 扫描器找的就是它）：
```python
registry = ToolRegistry()
```

### 1.2 `@tool` 装饰器 + 自动 JSON Schema 生成（新增 `tools/schemas.py`）

CLAUDE.md 设计的 `@tool` 装饰器**当前完全不存在**。用 `inspect` 签名 + Pydantic 自动生成 schema，免去每个 builtin 手写 schema dict。

```python
# tools/schemas.py
import inspect
from pydantic import create_model
from .registry import registry, ToolEntry   # 注意相对 import

def tool(toolset: str = "core", read_only: bool = False, check_fn=None):
    def deco(fn):
        sig = inspect.signature(fn)
        fields = {}
        for pname, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect._empty else str
            default = ... if p.default is inspect._empty else p.default
            fields[pname] = (ann, default)
        ArgsModel = create_model(f"{fn.__name__}_Args", **fields)
        registry.register(
            name=fn.__name__,
            toolset=toolset,
            schema={
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": ArgsModel.model_json_schema(),
            },
            handler=fn,
            check_fn=check_fn,
        )
        return fn
    return deco
```

> 备注：现有 `ToolEntry` 需要补一个 `read_only` 字段（用于将来 M6 并行判定）；M1 可先忽略并行，但字段先留好。

### 1.3 `ToolRegistry.get_definitions(toolset)` —— **当前完全缺失**

生成给 LLM 的 tool schema 列表。没有它，loop 无法给 `call_llm(tools=...)` 传工具，模型就调不了工具。

```python
# registry.py 内 ToolRegistry 追加
def get_definitions(self, toolset: set[str]) -> list[dict]:
    """按 toolset 过滤 + 跑 check_fn，返回 OpenAI/DeepSeek 格式 tool schema 列表。"""
    defs = []
    entries, _ = self._snapshot_state()
    for e in entries:
        if e.toolset not in toolset:
            continue
        if e.check_fn and not self._safe_check(e.check_fn):
            continue
        defs.append({
            "type": "function",
            "function": {
                "name": e.schema["name"],
                "description": e.schema.get("description", ""),
                "parameters": e.schema["parameters"],
            },
        })
    return defs

@staticmethod
def _safe_check(check_fn) -> bool:
    try:
        return bool(check_fn())
    except Exception:
        return False
```

> DeepSeek = OpenAI 兼容，外层固定 `{"type":"function","function":{...}}`。本项目单 provider，无需 anthropic 分支。

### 1.4 `dispatch.py` —— 两层 JSON 异常包裹（健壮性契约）

保证「模型永远收到合法 JSON」。核心逻辑虽然 `registry.dispatch` 里有雏形，但建议独立成 `handle_tool_call`，承担：参数校验 → 调用 → 结果统一成 JSON 字符串 → 异常包裹。

```python
# tools/dispatch.py
import json
from .registry import registry
from ..schema import ToolCall

def handle_tool_call(tc: ToolCall) -> str:
    """执行单个 tool_call，永远返回合法 JSON 字符串。"""
    entry = registry.get_entry(tc.name)
    if entry is None:
        return json.dumps({"error": f"unknown tool: {tc.name}"}, ensure_ascii=False)
    try:
        result = registry.dispatch(tc.name, tc.arguments)  # 第一层包裹在 registry.dispatch
        # 结果统一成字符串
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception as ex:  # 第二层兜底
        return json.dumps(
            {"error": f"dispatch failed: {type(ex).__name__}: {ex}"},
            ensure_ascii=False,
        )
```

> 约定：builtin handler 内部自己也 `return json.dumps(...)`；异常既可以 `raise`（被这里包）也可以自己返回 `{"error":...}`。两层保护缺一不可。

### 1.5 真正的 builtin 工具 —— 至少 `file_tools.py`

M1 demo 的主角。`read_file` / `write_file` 目前是空文件。

```python
# tools/builtin/file_tools.py
import json
from pathlib import Path
from ..schemas import tool

@tool(toolset="file", read_only=True)
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its content."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

@tool(toolset="file", read_only=False)
def write_file(path: str, content: str) -> str:
    """Write text to a file (overwrites). Returns bytes written."""
    Path(path).write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "bytes": len(content.encode())}, ensure_ascii=False)
```

> `web_tool.py` / `shell_tool.py` M1 不是必须，可留空到后续。

### 1.6 接进 agent loop（改 `agent.py`）

现在 `run_conversation` 只有占位注释，需要真正执行工具并回填。

- `Agent.__init__` 增加 `toolset: set[str]`（如 `{"file"}`）。
- `run_conversation` 里：
  1. `tools = registry.get_definitions(self.toolset)`；
  2. `call_llm(api_msgs, tools=tools)`；
  3. 若 `assistant.tool_calls`：逐个 `handle_tool_call(tc)`，用 `Message(role="tool", tool_call_id=tc.id, content=result)` 回填，然后 `continue`；
  4. 纯文本回复才终止。

骨架：
```python
if assistant.tool_calls:
    for tc in assistant.tool_calls:
        result = handle_tool_call(tc)
        self.messages.append(
            Message(role="tool", tool_call_id=tc.id, content=result)
        )
    continue   # 回填后继续下一轮，让模型消费工具结果
```

---

## 2. 🟡 可延后（M1 不需要）

| 项 | 归属里程碑 | 说明 |
|---|---|---|
| `executor.py`（只读工具并行） | M6 | M1 串行足够 |
| `state_tools.py`（memory / delegate_task 拦截） | M4 / M5 | 需要访问 agent 内部状态 |
| `web_tool.py` / `shell_tool.py` | 后续 | 代表性工具，非 M1 必须 |
| `tests/test_tools.py` | 建议尽早 | 保质量，非跑通必须 |

---

## 3. ⚠️ 横跨全项目的坑：import 方案必须先定

当前全项目在用**非包相对 import**：`from llm import ...`、`from config import ...`、`from registry import discover_builtin_tools`、`from model_tools import _run_async` 等。一旦把 `mini_agent/` 当包 import，这些会全崩，`discover_builtin_tools` 的动态 import 也会连锁失败。

**动手写 tool 系统前先统一为其一：**
- 方案 A（推荐）：包内一律相对 import（`from .llm import ...`、`from ..schema import ...`），入口用 `python -m mini_agent`。
- 方案 B：全部 `from mini_agent.xxx import ...`。

本文档 1.1–1.6 的示例默认走**方案 A（相对 import）**。

---

## 4. 实现顺序 & 验收

**顺序（最小依赖链）：**
```
1.1 修 registry + 单例
      ↓
1.2 @tool 装饰器（schemas.py）
      ↓
1.3 get_definitions
      ↓
1.4 dispatch.py（两层包裹）
      ↓
1.5 file_tools.py（read_file/write_file）
      ↓
1.6 接进 agent loop
```

**验收标准（M1 完成的判定）：**
1. `discover_builtin_tools()` 能成功 import 并注册 `read_file` / `write_file`；
2. `registry.get_definitions({"file"})` 返回含两个工具的 OpenAI 格式 schema 列表；
3. 注册一个会抛异常的工具，`handle_tool_call` 返回的是 `{"error": ...}` JSON 而非崩溃；
4. 端到端：`python -m mini_agent`，让模型「读取某文件」→ 模型发 `read_file` tool_call → 执行回填 → 模型基于结果给出文本回复；
5. `check_fn` 返回 False 的工具不出现在 `get_definitions` 结果里。

**建议补的测试（`tests/test_tools.py`，用 FakeLLM）：**
- 抛异常工具 → 断言返回 `{"error":...}`；
- toolset 过滤 / `check_fn` 过滤生效；
- `@tool` 自动生成的 schema 结构正确（含 name/description/parameters）；
- loop 脚本「先 tool_call 再纯文本」→ 断言角色交替合法、工具被执行、最终返回 final_response。
