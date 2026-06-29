import asyncio
import threading

_tool_loop = None
_tool_thread_local = threading.local()
_tool_lock = threading.Lock()

def _get_tool_loop():
    global _tool_loop
    with _tool_lock:
        if _tool_loop is None or _tool_loop.is_closed:
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop

def _get_worker_loop():
    loop = getattr(_tool_thread_local,'loop',None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop

