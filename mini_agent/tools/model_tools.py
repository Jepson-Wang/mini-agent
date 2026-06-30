import asyncio
import threading
from typing import Optional
from registry import discover_builtin_tools

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

discover_builtin_tools()

def _run_async(fn):
    """
    执行异步任务
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # 分支一: 如果loop存在并且在跑，此时需要创建一个一次性线程来跑
    if loop and loop.is_running():
        import concurrent.futures

        worker_loop : Optional[asyncio.AbstractEventLoop] = None
        event = threading.Event()

        def _worker():
            """
            获取worker_loop并执行相关代码并返回
            """
            nonlocal worker_loop
            worker_loop = asyncio.get_event_loop()
            event.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(fn)
            finally:
                try:
                    pending = asyncio.all_tasks()
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = worker.submit(_worker)
        try:
            future.result()
        except concurrent.futures.TimeoutError:
            try:
                for t in asyncio.all_tasks(worker_loop):
                    worker_loop.call_soon_threadsafe(t.cancel)
            except RuntimeError:
                pass

        finally:
            worker.shutdown(wait = False)

    # 分支二: 如果此时在一个线程中跑，但是这个线程不是主线程，那么就获取worker_loop并执行
    if threading.current_thread() is not threading.main_thread():
        loop = _get_worker_loop()
        return loop.run_until_complete(fn)

    # 分支三: 如果此时在主线程跑，那么就直接拿一个tool_loop并执行fn
    loop = _get_tool_loop()
    return loop.run_until_complete(fn)