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
