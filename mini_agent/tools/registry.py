import ast
import importlib
import json
import threading
from pathlib import Path
from typing import Optional, List, Dict, Callable


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

def discover_builtin_tools(path_lib : Optional[Path] = None) -> List[str]:
    """实现工具自注册"""
    path = path_lib if path_lib else Path(__file__).resolve().parent
    # 动态 import 必须用**包全名**（mini_agent.tools.builtin.xxx），不能用裸名。
    # 裸名要么 import 不到，要么把同一份代码加载成第二个模块对象——那样 @tool
    # 会注册进两个不同的 _TOOLS 字典，是最难 debug 的一类 bug。
    # 用 __package__（= "mini_agent.tools"）拼，将来改包名也不会坏。
    module_name = [
        f"{__package__}.builtin.{p.stem}"
        for p in sorted(path.glob('*.py'))
            if p.name not in {'__init__'}
            and _module_registers_tools(path)
    ]

    tools = []
    for mod in module_name:
        try:
            importlib.import_module(mod)
            tools.append(mod)
        except Exception:
            pass # TODO 导入日志系统
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

        self._tools = Dict[str,ToolEntry] = {}
        self._toolset_checks = Dict[str,Callable] = {}
        self._lock = threading.RLock()

    def get_entry(self,name: str) -> Optional[ToolEntry]:
        with self._lock:
            return self._tools.get(name)

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
                pass    #TODO 后续需要补全
            self._tools[name] =ToolEntry(
                name=name,
                toolset=toolset,
                schema=schema,
                is_async=is_async,
                description=description or schema.get("description",""),
                handler = handler,
                check_fn = check_fn
            )
            self._toolset_checks[toolset] = check_fn

    def deregister(self,name) -> None:
        """
        1. 加可重入锁
        2.删除一个ToolEntry
        3.判断是否还有其他Toolentry,没有则返回，有就删除
        :param self:
        :return:
        """
        with self._lock:
            entry = self._tools.pop(name,None)
            if entry is None:
                return
            toolset_still_exists = any(
                e.toolset ==entry.toolset for e in self._tools.values()
            )
            if not toolset_still_exists:
                self._toolset_checks.pop(entry.toolset,None)


    def dispatch(self,name,args,**kwargs) -> str:
        entry = self.get_entry(name)
        if not entry:
            return json.dumps({"error":f"Unknow tool:{name}"})
        try:
            if entry.is_async:
                from model_tools import _run_async
                return _run_async(entry.handler(args,**kwargs))
            return entry.handler(args,**kwargs)
        except Exception as e:
            return json.dumps({"error":"调用工具函数错误"})

    def _snapshot_state(self) -> tuple[list[ToolEntry],Dict[str,Callable]]:
        with self._lock:
            return list(self._tools.values()),dict(self._toolset_checks)