"""[M2] 真·进程级 kill 的崩溃恢复测试。

其余测试（test_persistence.py）都是**构造状态**：直接往库里写出"崩溃后会留下的
样子"，再 recover。那验证的是「recover 对每个状态处理得对不对」。
这个文件不一样 —— 它真的拉起一个进程、让它按真实协议写库、然后从外面一枪打死。
增量价值有两条，构造状态永远拿不到：

  ① 持久性。 构造状态跑在同一个进程里，你测的是"我写进去的东西我读得出来"。
     真 kill 测的是"进程消失之后，磁盘上还剩什么" —— 那是 WAL +
     synchronous=NORMAL 的承诺，不是我代码的承诺。
  ② 枚举完备性。 构造状态只能覆盖我**想到的**状态；我没想到的状态永远不会有测试，
     因为我不会给一个我没想到的情况写断言。随机时刻开枪是对崩溃瞬间做蒙特卡洛
     采样，它能撞到我漏掉的缝。

★ 生产代码零钩子。 kill 从进程外来，mini_agent/ 里没有任何 sentinel / 环境变量 /
  if TESTING 分支。所以"线上会不会被误触发"这个问题不存在——没有可触发的东西。

这些测试要拉子进程，比其它测试慢一两个数量级，打了 slow 标记：
    pytest -m "not slow"     # 平时
    pytest -m slow           # 单独跑崩溃恢复
"""
from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mini_agent.persistence.session_db import _INTERRUPTED, MiniSessionDB
from mini_agent.schema import Message

pytestmark = pytest.mark.slow

_VICTIM = str(Path(__file__).parent / "crash_victim.py")
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# --------------------------------------------------------------------------
# 工具：拉起靶子 / 等它就位 / 一枪打死
# --------------------------------------------------------------------------

def _spawn(mode: str, db_path: Path, sentinel: Path, log: Path) -> subprocess.Popen:
    """把靶子拉起来。stderr 落到文件——靶子被 kill 之后管道读不出东西，
    真出错时（比如 import 失败）只有文件里还留着现场。"""
    return subprocess.Popen(
        [sys.executable, _VICTIM, mode, str(db_path), str(sentinel)],
        cwd=_PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=open(log, "w", encoding="utf-8"),
    )


def _wait_ready(proc: subprocess.Popen, sentinel: Path, log: Path,
                timeout: float = 30.0) -> str:
    """等靶子写出 sentinel。返回它建的 session_id。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sentinel.exists():
            return sentinel.read_text(encoding="utf-8").strip()
        if proc.poll() is not None:            # 靶子自己先死了 = 它出错了
            err = log.read_text(encoding="utf-8", errors="replace")
            raise AssertionError(f"靶子提前退出（code={proc.returncode}）：\n{err}")
        time.sleep(0.01)
    proc.kill()
    raise AssertionError(f"靶子 {timeout}s 内没就位")


def _hard_kill(proc: subprocess.Popen) -> None:
    """真·不可捕获地干掉进程。

    POSIX：SIGKILL（信号 9）。它和 SIGSTOP 是仅有的两个进程无法捕获、无法屏蔽、
      无法忽略的信号。不跑 finally、不跑 atexit、不 flush 任何缓冲区。
    Windows：**没有 SIGKILL**（signal.SIGKILL 都不存在）。Popen.kill() 走的是
      TerminateProcess()，语义上是等价物：内核直接干掉进程，用户态一行清理代码都
      不会跑。对我们关心的东西（未 flush 的写会不会丢、WAL 能不能 replay 回来）
      两者一致。

    所以这里统一用 Popen.kill()：POSIX 上它就是 SIGKILL，Windows 上是最接近的等价物。
    """
    proc.kill()
    proc.wait(timeout=15)
    # 确认它真的是被杀死的，不是自己正常退出的 —— 否则这个测试什么都没测到
    if sys.platform != "win32":
        assert proc.returncode == -signal.SIGKILL, (
            f"期望被 SIGKILL 干掉，实际 returncode={proc.returncode}"
        )
    else:
        assert proc.returncode != 0, "Windows 上被 TerminateProcess 的进程不该返回 0"


@pytest.fixture
def crash_env(tmp_path):
    """返回 (db_path, sentinel, log)。每个测试一套独立的临时文件。"""
    return tmp_path / "state.db", tmp_path / "ready.txt", tmp_path / "victim.err"


def _assert_pairing_complete(messages: list[Message]) -> None:
    """带 tool_calls 的 assistant 后面必须紧跟「数量一致、id 一一对应」的 tool 消息。

    这就是把历史发给 DeepSeek 会不会 400 的判据，也是崩溃恢复要保住的唯一硬约束。
    顺带查孤立的 tool 消息（没有 assistant 认领它）—— 那同样会 400。
    """
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.role == "assistant" and m.tool_calls:
            n = len(m.tool_calls)
            following = messages[i + 1: i + 1 + n]
            assert [x.role for x in following] == ["tool"] * n, (
                f"第 {i} 条 assistant 有 {n} 个 tool_call，"
                f"后面跟的却是 {[x.role for x in following]}"
            )
            assert [x.tool_call_id for x in following] == [tc.id for tc in m.tool_calls], (
                f"第 {i} 条 assistant 的 tool_call id 和后面的 tool 结果对不上"
            )
            i += 1 + n
        else:
            assert m.role != "tool", f"第 {i} 条是孤立的 tool 消息，没有 assistant 认领"
            i += 1


# --------------------------------------------------------------------------
# 1. 确定性开枪：把 ②③ 窗口的两侧各打一枪
# --------------------------------------------------------------------------

def test_real_kill_before_tool_result_lands_stubs_it(crash_env):
    """崩在【assistant 已落库、工具结果未知】：补 interrupted 桩。

    幂等表里没有这个 key，说明工具没跑完（或者跑完了但我们不知道）。这时候
    **不能撒谎说它成功了**，也不能丢掉这一轮 —— 丢掉等于抹掉"工具可能已经跑过"
    这个事实，模型看不见就会重发一遍邮件。补桩是唯一诚实的选择。
    """
    db_path, sentinel, log = crash_env
    proc = _spawn("after-assistant", db_path, sentinel, log)
    sid = _wait_ready(proc, sentinel, log)
    _hard_kill(proc)

    # 从这行往下，我们只信磁盘上剩下的东西
    revived = MiniSessionDB(str(db_path)).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool"]
    assert revived[2].tool_call_id == "call_1"
    assert revived[2].content == _INTERRUPTED
    _assert_pairing_complete(revived)


def test_real_kill_after_idempotency_write_recovers_the_real_result(crash_env):
    """崩在【幂等表已写、tool 结果未落库】：捞回真结果，一次都不用重跑。

    ★ 这个测试是幂等表存在的全部理由，而且真 kill 版本比构造状态版本多验一件事：
      record_executed_key 的那次 COMMIT 是**真的落到磁盘上了**，进程被内核干掉、
      没有任何 flush 机会的情况下仍然读得回来。那是 WAL + synchronous=NORMAL 的
      承诺，构造状态（同进程读写）永远验不到。
    """
    db_path, sentinel, log = crash_env
    proc = _spawn("after-record", db_path, sentinel, log)
    sid = _wait_ready(proc, sentinel, log)
    _hard_kill(proc)

    revived = MiniSessionDB(str(db_path)).recover(sid)

    assert [m.role for m in revived] == ["user", "assistant", "tool"]
    assert revived[2].content == "邮件已发送"        # ← 真结果
    assert revived[2].content != _INTERRUPTED       # ← 不是桩
    _assert_pairing_complete(revived)


def test_session_killed_mid_run_stays_running(crash_env):
    """被 kill 的会话停在 running —— end_session 在 finally 里，而 kill 不跑 finally。

    这个断言有两层意思：
      1. 证明这一枪真的是"不可捕获"的。如果 status 变成了 done，说明有清理代码跑过，
         那这整个文件测的就不是 kill 了。
      2. 钉住「status 是咨询性的、不参与重建」这个设计：recover 压根不看 status，
         所以一个永远停在 running 的会话照样恢复得了。
    """
    db_path, sentinel, log = crash_env
    proc = _spawn("after-record", db_path, sentinel, log)
    sid = _wait_ready(proc, sentinel, log)
    _hard_kill(proc)

    db = MiniSessionDB(str(db_path))
    status = db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()["status"]
    assert status == "running"

    # --resume 的真实路径：reopen 会 CAS 失败（本来就是 running），如实返回 False。
    # 这正是「上次被 kill 没走完 finally」的信号，不是错误。
    assert db.reopen_session(sid) is False
    assert db.recover(sid)                     # 照样恢复得了


# --------------------------------------------------------------------------
# 2. 随机开枪：对崩溃瞬间做蒙特卡洛
# --------------------------------------------------------------------------

_ROUNDS = 8       # 开几枪。实测约一半会落在残缺轮里，8 枪全落空的概率 <0.5%


def test_kill_at_a_random_instant_always_leaves_a_recoverable_db(tmp_path):
    """靶子按真实协议疯狂写库，在随机时刻开枪，恢复必须永远成立。

    ★ 这条是唯一能证伪「我对可达状态的枚举是完整的」的测试。 其它测试都只覆盖我
      **想到的**状态；这里子弹落在哪条缝上我不知道，跑够多次就会撞到我漏掉的。

    ★ 顺带做覆盖率自检（最后那条断言）。 随机开枪有一半概率落在两轮之间的干净
      边界上——那种情况下历史本来就是配对完整的，配对断言恒真，这一枪什么都没测到。
      不检查的话，哪天 kill 时机整体偏移、每一枪都落空，测试照样全绿，你却以为
      崩溃恢复被覆盖着。**一个可能恒真的断言，必须有东西证明它不恒真。**

    每一枪断言三条，都是"能不能接着跑"的必要条件：
      1. recover 不抛（不管磁盘上剩的是什么形状）；
      2. 配对完整（否则下一次请求就是 400，恢复了也白恢复）；
      3. 恢复之后还能继续写（被 kill 的进程没有把写锁留成孤儿锁）。
    """
    hit_incomplete = 0

    for i in range(_ROUNDS):
        db_path = tmp_path / f"state_{i}.db"
        sentinel = tmp_path / f"ready_{i}.txt"
        log = tmp_path / f"victim_{i}.err"
        # seed 只固定"等多久再开枪"，固定不了进程内部的调度——所以这不是可完全复现的
        # 测试，而是一次采样。它的价值在于跑得多、撞得广，不在于可复现。
        delay = random.Random(i).uniform(0.02, 0.40)

        proc = _spawn("churn", db_path, sentinel, log)
        sid = _wait_ready(proc, sentinel, log)
        time.sleep(delay)
        _hard_kill(proc)

        db = MiniSessionDB(str(db_path))
        revived = db.recover(sid)                            # ① 不抛
        _assert_pairing_complete(revived)                    # ② 配对完整

        # ③ 还能接着写：被 kill 的进程不该把 SQLite 的写锁永久锁死在库上
        db.append_message(sid, Message(role="user", content="恢复之后继续聊"))
        assert db.recover(sid)[-1].content == "恢复之后继续聊", f"第 {i} 枪：续写没生效"

        if _landed_mid_round(db, sid, revived):
            hit_incomplete += 1

    assert hit_incomplete > 0, (
        f"{_ROUNDS} 枪全打在两轮之间的干净边界上，一次残缺轮都没造出来 —— "
        f"配对断言在这种情况下恒真，这个测试其实什么都没验证。"
        f"该调 crash_victim 的写节奏或开枪时机分布。"
    )


def _landed_mid_round(db: MiniSessionDB, sid: str, revived: list[Message]) -> bool:
    """这一枪是不是打在了一轮的中间（= recover 真的补了东西）？

    判据：库里**原始**的 messages 条数 < recover 返回的条数。recover 只增不减
    （补桩/回填），所以两者不等就说明它确实修补过，也就说明子弹落在了残缺轮里。
    注意要减掉我们自己刚 append 的那一条。
    """
    raw = db.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?;", (sid,)
    ).fetchone()[0] - 1        # 减掉上面那条"恢复之后继续聊"
    return raw < len(revived)
