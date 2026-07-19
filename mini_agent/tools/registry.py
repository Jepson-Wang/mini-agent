import ast
import importlib
import inspect
import json
import threading
from pathlib import Path
from typing import Any, Optional, List, Dict, Callable

from mini_agent.log import get_logger

logger = get_logger(__name__)


def _error(message: str) -> str:
    """工具错误的唯一出口：永远是 {"error": "..."} 形状的合法 JSON 字符串。

    Hermes 的原则：Handlers MUST return a JSON string, errors as {"error": ...},
    never raised。模型永远收得到能解析的东西。
    """
    return json.dumps({"error": message}, ensure_ascii=False)


def _check_available(entry: "ToolEntry") -> bool:
    """跑工具的 check_fn，判断它此刻是否可用。没有 check_fn 就当作永远可用。

    check_fn 自己抛异常 → 一律视为不可用。一个"探测 API key 在不在"的函数
    自己炸了，不该把整个 agent 拖下水；最坏结果只是少一个工具。
    """
    if entry.check_fn is None:
        return True
    try:
        return bool(entry.check_fn())
    except Exception as e:
        logger.warning("工具 %s 的 check_fn 抛异常，视为不可用: %s: %s",
                       entry.name, type(e).__name__, e)
        return False


def _to_openai_tool_schema(entry: "ToolEntry") -> dict:
    """ToolEntry → DeepSeek / OpenAI 的 tools 元素形状。

    外层必须是 {"type": "function", "function": {...}}——这是 OpenAI Chat
    Completions 的规定形状，DeepSeek 原生兼容。（对比 Anthropic 是扁平的
    {"name", "description", "input_schema"}，形状完全不同；provider adapter
    要负责的就是这层转换。我们只接 DeepSeek，所以只有这一种。）

    parameters 必须是合法的 JSON Schema object。工具没声明参数时也要给一个
    空的 object 骨架，不能省略——省略了部分实现会 400。
    """
    schema = entry.schema or {}
    parameters = schema.get("parameters") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": entry.name,
            "description": entry.description or schema.get("description", ""),
            "parameters": parameters,
        },
    }


def _is_registry_register_call(node: ast.AST) -> bool:
    """如果node是一个registry.register()的形式就返回True"""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (
            isinstance(func, ast.Attribute)
            and func.attr == "register"
            and isinstance(func.value, ast.Name)
            and func.value.id == "registry"
    )
def _module_registers_tools(module_path: Path) -> bool:
    """
    判断一个 .py 文件是否在模块顶层调用了 registry.register()
    如果 registry.register() 出现在函数内部、if 分支内，它不会被检测到
    """
    try:
        #先读取path对应的文件代码，然后通过parse来解析并返回一个ast.Module节点
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, SyntaxError):
        return False
    # body这个属性是一个列表，包含模块级别的所有语句节点
    #   但是只包含模块顶层的语句,比如registry.register(...)
    #   但如果registry.register(...)  在一个函数或if中被使用，那么将不会被检测
    return any(_is_registry_register_call(stmt) for stmt in tree.body)

def discover_builtin_tools(path_lib: Optional[Path] = None) -> List[str]:
    """扫描 builtin/ 目录，import 所有在顶层调用了 registry.register() 的模块。

    工具的「自注册」靠的是 import 副作用：模块被 import → 顶层的
    registry.register(...) 执行 → 工具进 _tools。所以这里只需要负责
    「找到该 import 谁」并 import，不需要关心工具本身。

    返回成功 import 的模块全名列表。
    """
    # 扫的是 builtin/ 目录，不是 tools/ 自己 —— 否则会去 import
    # mini_agent.tools.builtin.registry 这种根本不存在的模块。
    path = path_lib if path_lib else Path(__file__).resolve().parent / "builtin"

    # 动态 import 必须用**包全名**（mini_agent.tools.builtin.xxx），不能用裸名。
    # 裸名要么 import 不到，要么把同一份代码加载成第二个模块对象——那样 @tool
    # 会注册进两个不同的 _tools 字典，是最难 debug 的一类 bug。
    # 用 __package__（= "mini_agent.tools"）拼，将来改包名也不会坏。
    module_names = [
        f"{__package__}.builtin.{p.stem}"
        for p in sorted(path.glob("*.py"))
        if p.name != "__init__.py"
        and _module_registers_tools(p)   # 传文件，不是目录
    ]

    tools = []
    for mod in module_names:
        try:
            importlib.import_module(mod)
            tools.append(mod)
        except Exception as e:
            # 绝不能静默吞：一个工具模块 import 失败 = 这个工具直接从模型眼前消失，
            # 而模型不会告诉你「我少了个工具」，它只会表现得莫名其妙。
            logger.warning("工具模块导入失败，该工具将不可用: %s (%s: %s)",
                           mod, type(e).__name__, e)
    return tools

class ToolEntry:

    __slots__ = ("name","toolset","schema",
                 "handler","check_fn","is_async","description")

    def __init__(self,name,toolset,schema,handler,description
                 ,check_fn,is_async):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.description = description
        self.check_fn = check_fn
        self.is_async = is_async

class ToolRegistry:

    def __init__(self):
        # 注意是冒号（类型标注），不是等号。
        # 写成 `self._tools = Dict[str, ToolEntry] = {}` 是**链式赋值**：
        # Python 会去执行 Dict.__setitem__((str, ToolEntry), {})，运行时直接 TypeError。
        self._tools: Dict[str, ToolEntry] = {}
        self._lock = threading.RLock()

    def get_entry(self,name: str) -> Optional[ToolEntry]:
        with self._lock:
            return self._tools.get(name)

    def get_definitions(
        self,
        toolset: Optional[set[str]] = None,
        disabled: Optional[set[str]] = None,
    ) -> List[dict]:
        """把注册表里可用的工具，转成 DeepSeek / OpenAI 要的 tools 参数。

        这是「注册表」和「模型」之间唯一的接口：agent 每轮把它的返回值传给
        call_llm(messages, tools=...)。工具没出现在这里，对模型来说就等于不存在。

        三步过滤：
          1. toolset 过滤 —— 只暴露 agent 被授权的工具集。M4 的子代理靠这一步做
             最小权限（子 toolset ⊆ 父 toolset）。传 None 表示不过滤（全部）。
          2. disabled 黑名单 —— 按工具名精确剔除。M4 从子代理身上剥离
             delegate_task（禁递归）走的就是这条。
          3. check_fn —— 运行时可用性检查（API key 在不在、二进制装没装）。
             check_fn 抛异常一律视为不可用：一个探测函数自己炸了，不该把
             整个 agent 拖下水。

        ★ 输出按工具名排序，顺序**必须稳定**。
          DeepSeek 有自动的上下文缓存，命中的前提是请求前缀逐字节不变；tools 是
          前缀的一部分，今天 [read_file, web_fetch]、明天 [web_fetch, read_file]，
          缓存就整个失效了。而 dict 的迭代顺序会随注册顺序（= import 顺序）变化，
          所以这里显式排序，不依赖插入顺序。
        """
        with self._lock:
            entries = list(self._tools.values())

        defs: List[dict] = []
        for entry in sorted(entries, key=lambda e: e.name):
            if toolset is not None and entry.toolset not in toolset:
                continue
            if disabled and entry.name in disabled:
                continue
            if not _check_available(entry):
                continue
            defs.append(_to_openai_tool_schema(entry))
        return defs

    def register(
            self,
            name:str,
            toolset:str,
            schema:dict,
            handler:Callable = None,
            check_fn:Callable = None,
            is_async:bool = False,
            description:str = "",
                 ):
        with self._lock:
            existing = self._tools.get(name)
            if existing and existing.toolset != toolset:
                # 同名工具在不同 toolset 里重复注册 —— 多半是命名撞车或误注册。
                # 保持"后注册者覆盖"，但绝不静默：不报出来的话，一个工具被另一个
                # 顶掉、模型调到的是意外的那个，是能查一整天的 bug。
                # （同 toolset 的重复注册是幂等 re-import，属正常，不告警。）
                logger.warning(
                    "工具名冲突：%r 已在 toolset=%r 注册，现被 toolset=%r 覆盖",
                    name, existing.toolset, toolset,
                )
            self._tools[name] =ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                is_async=is_async,
                description=description or schema.get("description",""),
                handler = handler,
                check_fn = check_fn
            )

    def dispatch(self, name: str, args: Optional[dict] = None, **kwargs) -> str:
        """执行一次工具调用。**永远返回字符串、绝不抛异常** —— 健壮性契约的硬核。

        「是不是 JSON」要说清楚，别让将来读代码的人误用：
          - 错误一律是 {"error": ...} 的合法 JSON（_error 出口）；
          - handler 返回非字符串时，本层 json.dumps 兜底成 JSON；
          - 但 handler 自己返回的**字符串是原样放行、不校验**的。出厂 builtin 都按
            约定返回 JSON，但这只是约定、不是强制。所以调用方**不能**假设 dispatch
            的返回一定能 json.loads（别写 json.loads(dispatch(...)) —— read_file
            这类工具完全可以直接返回纯文本给模型读）。

        工具炸了就回 {"error": ...}，让模型自己决定重试还是换路子；
        要是让异常冒到 loop 里，一次 read_file 读不到文件就能把整个 agent 打死。

        调用约定：**args 按具名参数展开**（handler(**args)）。
        因为 entry.schema 里存的是 JSON Schema，描述的本来就是一组具名参数，
        模型发来的 arguments 也是 {"path": "a.txt"} 这种形状。所以工具就写成
        普通的 Python 函数即可：

            def read_file(path: str) -> str: ...

        而不是让每个工具自己去 args["path"] 里掏 —— 那样每个工具都要重写一遍
        取参 + 校验，还没法用 inspect 自动生成 schema。

        kwargs 是 loop 注入的额外上下文（如 session_id），和 args 合并后一起传。
        """
        entry = self.get_entry(name)
        if entry is None:
            return _error(f"Unknown tool: {name}")
        if entry.handler is None:
            return _error(f"Tool has no handler: {name}")

        if args is None:
            args = {}
        if not isinstance(args, dict):
            # 模型偶尔会把 arguments 发成字符串或数组
            return _error(
                f"Invalid arguments for {name}: expected an object, "
                f"got {type(args).__name__}"
            )

        # kwargs 是 loop 注入的可信上下文（如 session_id），args 是模型发来的不可信
        # 参数。两者撞名时**绝不静默覆盖** —— 否则模型只要发一个同名 session_id 就能
        # 顶掉运行时注入的真值，是一类安全隐患。撞了就明确报错，让它当场暴露。
        collisions = set(args) & set(kwargs)
        if collisions:
            return _error(
                f"Argument name conflict for {name}: {sorted(collisions)} "
                "由运行时注入，模型不能提供同名参数"
            )
        call_args: dict[str, Any] = {**args, **kwargs}

        # 先做参数绑定校验，再执行。这样「模型把参数名写错了」和「工具内部炸了」
        # 是两类不同的错误，给模型两种不同的提示：前者它改个参数名就能自救，
        # 后者它得换个思路。混成一句 TypeError 的话，模型只会原地打转。
        #
        # signature() 对少数可调用对象（C 实现的内置函数、某些 partial）会抛
        # ValueError；此时无法内省，就**跳过预校验**、把判断交给真正的调用，
        # 而不是把它当成参数错误 —— 否则 ValueError 会逃出 dispatch，破坏
        # 「绝不抛异常」的契约。
        try:
            sig = inspect.signature(entry.handler)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            try:
                sig.bind(**call_args)
            except TypeError as e:
                return _error(f"Invalid arguments for {name}: {e}")

        try:
            if entry.is_async:
                # 延迟 import：model_tools 在顶层 import 了本模块，
                # 提到文件顶部会变成循环 import。
                from mini_agent.tools.model_tools import _run_async
                result = _run_async(entry.handler(**call_args))
            else:
                result = entry.handler(**call_args)
        except Exception as e:
            # 把类型和消息都带上。只回一句"调用工具函数错误"的话，模型看不出
            # 是文件不存在、权限不够还是磁盘满了，只能瞎猜。
            logger.warning("工具执行失败: %s(%s) -> %s: %s",
                           name, call_args, type(e).__name__, e)
            return _error(f"Tool failed: {name}: {type(e).__name__}: {e}")

        # handler 返回非字符串时兜底序列化，保证「永远是合法 JSON 字符串」成立
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except TypeError as e:
            return _error(f"Tool {name} returned a non-JSON-serializable value: {e}")


# 全局唯一的注册表实例。工具模块靠 import 副作用自注册：
#
#     from mini_agent.tools.registry import registry
#
#     def read_file(path: str) -> str: ...
#
#     registry.register(                       # ← 必须在模块顶层
#         name="read_file", toolset="file",
#         schema={"description": "...", "parameters": {...}},
#         handler=read_file,
#     )
#
# discover_builtin_tools() 用 AST 找的就是这个顶层 `registry.register(...)` 调用
# —— 名字必须叫 registry，写成 `from ... import registry as reg` 会扫不到。
#
# 这个单例是模块级的，所以「只有一个 _tools 字典」这件事，前提是 registry 模块
# 本身只有一个模块身份。这正是全项目统一包绝对 import 的意义所在。
registry = ToolRegistry()