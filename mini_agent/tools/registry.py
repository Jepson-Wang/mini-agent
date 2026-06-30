import ast
import importlib
from pathlib import Path
from typing import Optional, List


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


class ToolEntry:

    __slots__ = ("name","toolset","schema",
                 "handler","check_fn","is_async")

    def __init__(self,name,toolset,schema,handler
                 ,check_fn,is_async):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.handler = handler
        self.check_fn = check_fn
        self.is_async = is_async
