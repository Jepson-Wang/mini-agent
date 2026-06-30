import threading
from typing import Dict, Callable
from xml.sax import handler


class ToolEntry:

    __slots__ = ("name","toolset","schema","description",
                 "handler","check_fn","is_async")

    def __init__(self,name,toolset,schema,description,handler
                 ,check_fn,is_async):
        self.name = name
        self.toolset = toolset
        self.schema = schema
        self.description = description
        self.handler = handler
        self.check_fn = check_fn
        self.is_async = is_async

class ToolRegistry:

    def __init__(self):

        self._tools = Dict[str,ToolEntry] = {}
        self._toolset_checks = Dict[str,Callable] = {}
        self._lock = threading.RLock()

    def registry(
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
            if check_fn and toolset not in self._toolset_checks:
                self._toolset_checks[toolset] = check_fn