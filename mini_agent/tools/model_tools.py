"""同步 → 异步的桥接：让同步的 dispatch 能调用 async 工具 handler。

本项目的主循环是同步的（CLI 单线程），但个别工具可能想写成 async。
registry.dispatch 在 entry.is_async 时会把 handler(...) 产生的协程交给这里跑完。

设计取舍：不做「按线程缓存事件循环」那套——工具调用不是热路径，每次
asyncio.run 新建一个 loop 的开销可以忽略，换来的是「绝不出错」的简单。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Coroutine


def _run_async(coro: Coroutine) -> Any:
    """把一个协程跑到完成并返回它的结果。

    两种情形：
    - 当前线程没有正在运行的事件循环（CLI 主线程的常态）→ asyncio.run 直接跑。
    - 当前线程**已经**有 loop 在跑（我们是同步架构，正常不会发生，但防一手）→
      不能在运行中的 loop 里再 asyncio.run（会 RuntimeError），只能把协程丢到
      一个临时工作线程里，让它在自己干净的 loop 上跑完。
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        running = False
    else:
        running = True

    if not running:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
