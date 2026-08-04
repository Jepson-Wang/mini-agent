'''这是一个加法运算工具'''

from __future__ import annotations
import json
from mini_agent.tools.registry import registry
from mini_agent.log import get_logger

logger = get_logger(__name__)


def add(a:float,b:float) ->str:
    return json.dumps({"result": a+b})


ADD_SCHEMA = {
    "name": "add",
    "description": "这是一个加法运算函数",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "第一个加数"},
            "b": {"type": "number", "description": "第二个加数"},
        },
        "required": ["a", "b"],
    },
}

registry.register(
         name="add",
         toolset="math",
         schema=ADD_SCHEMA,
         handler=add,
         description=ADD_SCHEMA["description"],
     )