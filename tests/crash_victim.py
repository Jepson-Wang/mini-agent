"""被真·kill 的那个进程。

★ 这不是测试模块（文件名不以 test_ 开头，pytest 不会收集它），是 test_crash_real_kill.py
  拿 subprocess 拉起来、然后从外面杀掉的靶子。

★ 生产代码里没有任何钩子。 没有 sentinel 环境变量、没有 if TESTING、没有注入点。
  kill 从**进程外**来，靶子只用 mini_agent 的公开 API 做真实写入。所以线上不存在
  "误触发注入点"的问题 —— 根本没有注入点可触发。

用法: python crash_victim.py <mode> <db_path> <sentinel_path>
  mode 决定它写到哪一步就停下来等死，见下面各分支。
"""
from __future__ import annotations

import os
import sys
import time

from mini_agent.persistence.session_db import MiniSessionDB
from mini_agent.schema import Message, ToolCall

# churn 模式的硬上限：父进程万一没杀成，别让它无限写下去把磁盘塞满
_MAX_ROUNDS = 100_000


def _signal_ready(sentinel: str, payload: str) -> None:
    """告诉父进程「可以动手了」。

    先写临时文件 + fsync，再 os.replace 原子改名。父进程只要看见目标文件存在，
    内容就一定是完整的 —— 否则可能读到写了一半的空文件，然后拿着空 sid 去查库。
    进程随时会消失，缓冲区里的字节不算数，所以 fsync 不能省。
    """
    tmp = sentinel + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, sentinel)


def _tc(i: int) -> ToolCall:
    return ToolCall(id=f"call_{i}", name="send_email", arguments={"n": i})


def main() -> None:
    mode, db_path, sentinel = sys.argv[1], sys.argv[2], sys.argv[3]
    db = MiniSessionDB(db_path)
    sid = db.create_session()

    if mode == "after-assistant":
        # 崩在【W2 之后、E2/W3 之前】：assistant 的 tool_calls 已落库，工具还没跑完，
        # 幂等表里什么都没有。recover 应当补 _INTERRUPTED 桩（如实说"结果未知"）。
        db.append_message(sid, Message(role="user", content="发封邮件"))
        db.append_message(sid, Message(role="assistant", tool_calls=[_tc(1)]))
        _signal_ready(sentinel, sid)

    elif mode == "after-record":
        # 崩在【W3 之后、W4 之前】：工具真跑过了、幂等表也写了，只差 tool 结果没进
        # messages。recover 应当从幂等表把**真结果**捞回来，而不是补桩——补桩会害
        # 模型以为没发成、再发一遍邮件。这个窗口就是幂等表存在的全部理由。
        db.append_message(sid, Message(role="user", content="发封邮件"))
        db.append_message(sid, Message(role="assistant", tool_calls=[_tc(1)]))
        db.record_executed_key("call_1", sid, {"content": "邮件已发送"})
        _signal_ready(sentinel, sid)

    elif mode == "churn":
        # 不停地按真实协议写：user → assistant(tool_calls) → 幂等表 → tool 结果。
        # 父进程在随机时刻开枪，于是 kill 会落在协议的任意一条缝里。
        # 这才是真 kill 相对"构造状态"的增量价值：构造状态只能验证我**想到的**那几个
        # 状态处理得对，随机 kill 验证的是我对可达状态的**枚举是否完整**。
        _signal_ready(sentinel, sid)
        for i in range(_MAX_ROUNDS):
            db.append_message(sid, Message(role="user", content=f"q{i}"))
            db.append_message(sid, Message(role="assistant", tool_calls=[_tc(i)]))
            db.record_executed_key(f"call_{i}", sid, {"content": f"r{i}"})
            db.append_message(
                sid, Message(role="tool", tool_call_id=f"call_{i}", content=f"r{i}")
            )
    else:
        raise SystemExit(f"unknown mode: {mode}")

    # 等着挨枪子。绝不主动退出，也绝不 end_session —— 会话必须停在 running。
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
