"""CLI 薄入口：python -m mini_agent 进入 REPL。

用法：
  python -m mini_agent                    新建会话
  python -m mini_agent --resume sess_xxx  从库里恢复既有会话继续聊

[M0] 只做一个 surface（CLI）。Agent 类本身 surface 无关，方便后续复用
（如 subagent / gRPC server）。
[M2] 每条消息即时落库。会话 id 会打印出来，方便下次 --resume。
[M2.5] Agent 无状态化后，CLI 的职责收缩成两件事：**铸造 / 认领一个 session_id**，
     然后每轮把它连同用户输入一起传给共享的 agent。历史加载、崩溃补桩都在
     run_conversation 内部完成，所以这里不再有"新建 vs 续跑"两条装配路径。
     将来的 gRPC server 是第二个同样薄的 surface：session_id 从 Go 那边来，
     其余一模一样。
"""
from __future__ import annotations

import sys

from mini_agent.agent import Agent
from mini_agent.config import settings
from mini_agent.persistence.session_db import MiniSessionDB, PersistenceError

# 默认工具集。web_fetch 另受 ALLOW_WEB 门控（没开就不会暴露给模型）；
# file 工具目前还没实现，写好后会自动出现在这个集合里。
DEFAULT_TOOLSET = {"file", "web","math"}


def _db_path() -> str:
    """<settings.home>/state.db；目录不存在先建。默认落在项目根的 .mini_agent/。"""
    settings.home.mkdir(parents=True, exist_ok=True)
    return str(settings.home / "state.db")


def _parse_resume(argv: list[str]) -> str | None:
    """从 argv 里取 --resume 的值；没有则返回 None。"""
    if "--resume" not in argv:
        return None
    i = argv.index("--resume")
    if i + 1 >= len(argv):
        print("--resume 后面要跟一个 session_id", file=sys.stderr)
        sys.exit(2)
    return argv[i + 1]


def main() -> None:
    if not settings.deepseek_api_key:
        print("缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 key。",
              file=sys.stderr)
        sys.exit(1)

    # 打开库本身也会失败：路径无权限、磁盘满、文件损坏（sqlite3.connect 是惰性的，
    # 损坏要到第一条 PRAGMA 才暴露）。持久化层已把这些收敛成 PersistenceError，
    # 这里接住给一句人话，而不是甩一坨 traceback。
    try:
        db = MiniSessionDB(_db_path())
    except PersistenceError as e:
        print(f"无法打开数据库：{e}\n"
              f"位置：{_db_path()}\n"
              f"若确认这个文件已损坏，删掉它即可重新开始（会丢失全部历史会话）。",
              file=sys.stderr)
        sys.exit(1)

    resume_sid = _parse_resume(sys.argv[1:])

    if resume_sid is not None:
        if not db.session_exists(resume_sid):
            print(f"会话不存在: {resume_sid}", file=sys.stderr)
            sys.exit(1)
        session_id = resume_sid

        # ★ 这次 recover 是**探路**，不是装配。 历史真正的加载在 run_conversation
        #   里（每轮都做）；这里跑一遍只为两件 CLI 专属的事：把"这段历史还能不能
        #   读"提前问出来、好给一句人话而不是等第一条消息甩 traceback，以及打印
        #   条数。多一次读，只发生在 CLI 启动时，不在服务热路径上。
        # ★ 顺序要紧：先 recover，再 reopen。
        #   recover 是纯读的，失败了库还和没跑过一样，可以原地重试；而
        #   reopen_session 是一个**承诺**——「我现在接管这个会话了」。不该在确认
        #   自己接管得了之前就把 status 改掉：反过来写的话，recover 一抛就把会话
        #   永久卡在 running，下次 --resume 还会误报「上次未正常退出」。
        try:
            history = db.recover(resume_sid)  # 残缺轮已补桩，可安全续跑
        except PersistenceError as e:
            # 恢复不了就响亮地死，绝不静默退化成空历史 —— 那样用户以为在续聊，
            # 实际在开新的，而新消息还会以 seq 接着写进同一个 session，把库里那段
            # transcript 变成两段无关对话的拼接，事后无法分辨。
            print(f"无法恢复会话 {resume_sid}：{e}\n"
                  f"这段历史仍在库里（{_db_path()}），没有被改动。\n"
                  f"可以直接 `python -m mini_agent` 开一个新会话继续工作。",
                  file=sys.stderr)
            sys.exit(1)

        # CAS 没命中说明它本来就是 running：要么上次被 kill 没走完 finally
        # （正常，继续），要么另一个进程正开着它（危险）。M2 是单进程 CLI，
        # 这里只提示放行；M4 共享 db 时要换成带 owner/heartbeat 的租约。
        if not db.reopen_session(resume_sid):
            print(f"提示：会话 {resume_sid} 上次未正常退出（或正被另一个进程使用）",
                  file=sys.stderr)
        print(f"恢复会话 {session_id}（{len(history)} 条历史）")
    else:
        session_id = db.create_session()
        print(f"会话 {session_id}")

    # 无状态 agent：不绑 session，构造一次、整个 REPL 复用。
    agent = Agent(db=db, toolset=DEFAULT_TOOLSET, max_turns=settings.max_turns)

    print(f"mini-agent ({settings.model}) — 输入 'exit' 退出；"
          f"下次可用 --resume {session_id} 继续")

    try:
        while True:
            try:
                user = input("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user.strip() in {"exit", "quit"}:
                break
            if not user.strip():
                continue
            out = agent.run_conversation(session_id, user)
            print(out["final_response"])
    finally:
        # 正常退出 / Ctrl-C / 异常都走这里，把会话标记成结束
        db.end_session(session_id)


if __name__ == "__main__":
    main()
