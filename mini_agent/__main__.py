"""CLI 薄入口：python -m mini_agent 进入 REPL。

用法：
  python -m mini_agent                    新建会话
  python -m mini_agent --resume sess_xxx  从库里恢复既有会话继续聊

[M0] 只做一个 surface（CLI）。Agent 类本身 surface 无关，方便后续复用
（如 subagent / HTTP server）。
[M2] 每条消息即时落库；--resume 靠 db.recover() 重建对话。会话 id 会打印出来，
     方便下次 --resume。
"""
from __future__ import annotations

import sys

from mini_agent.agent import Agent
from mini_agent.config import settings
from mini_agent.persistence.session_db import MiniSessionDB

# 默认工具集。web_fetch 另受 ALLOW_WEB 门控（没开就不会暴露给模型）；
# file 工具目前还没实现，写好后会自动出现在这个集合里。
DEFAULT_TOOLSET = {"file", "web"}


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

    db = MiniSessionDB(_db_path())
    resume_sid = _parse_resume(sys.argv[1:])

    if resume_sid is not None:
        if not db.session_exists(resume_sid):
            print(f"会话不存在: {resume_sid}", file=sys.stderr)
            sys.exit(1)
        agent = Agent(session_id=resume_sid, db=db, toolset=DEFAULT_TOOLSET,
                      max_turns=settings.max_turns)
        agent.messages = db.recover(resume_sid)   # 从库里重建历史（已清理悬空 tool_calls）
        session_id = resume_sid
        print(f"恢复会话 {session_id}（{len(agent.messages)} 条历史）")
    else:
        session_id = db.create_session()
        agent = Agent(session_id=session_id, db=db, toolset=DEFAULT_TOOLSET,
                      max_turns=settings.max_turns)
        print(f"会话 {session_id}")

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
            out = agent.run_conversation(user)
            print(out["final_response"])
    finally:
        # 正常退出 / Ctrl-C / 异常都走这里，把会话标记成结束
        db.end_session(session_id)


if __name__ == "__main__":
    main()
