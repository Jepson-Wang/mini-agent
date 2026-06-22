"""CLI 薄入口：python -m mini_agent 进入 REPL。

[M0] 只做一个 surface（CLI）。Agent 类本身 surface 无关，方便后续复用
（如 subagent / HTTP server）。
"""
from __future__ import annotations

import sys

from agent import Agent
from config import settings


def main() -> None:
    if not settings.deepseek_api_key:
        print("缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 key。",
              file=sys.stderr)
        sys.exit(1)

    agent = Agent(max_turns=settings.max_turns)
    print(f"mini-agent ({settings.model}) — 输入 'exit' 退出")
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


if __name__ == "__main__":
    main()
