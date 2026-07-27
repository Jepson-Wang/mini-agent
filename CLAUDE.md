# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning-oriented, from-scratch minimal AI agent runtime (a "mini Hermes"), built to be discussed in backend interviews. A synchronous CLI REPL drives an `Agent` that runs a ReAct-style tool-calling loop against DeepSeek, with every message persisted to SQLite so a killed process can `--resume` and continue without re-running side effects.

`开发路线.md` is the original 6-milestone (M0→M5) **design blueprint**, not a spec of current state. Code comments and docstrings that say "对照 CLAUDE.md" / "见 CLAUDE.md 第五节" refer to *that blueprint*, not this file. Read it for intent, but **two hard constraints below override it** — do not implement what the blueprint says where they conflict.

## Two hard constraints (override the blueprint)

1. **DeepSeek-only.** The LLM adapter supports DeepSeek exclusively via the OpenAI-compatible SDK (`base_url=https://api.deepseek.com`). The internal message format (`mini_agent/schema.py`) is already OpenAI-shaped, so `call_llm` collapses to a single `base_url` rewrite. **Do not** build OpenAI/Anthropic dual adapters, `to_openai()`/`to_anthropic()` branches, or a `provider` parameter threaded through the loop — the blueprint's multi-provider design is dead.

2. **Absolute imports only:** always `from mini_agent.xxx import ...`. Never relative (`from .schema import`) and never bare (`from schema import`). Bare names require `mini_agent/` itself on `sys.path`, which breaks `python -m mini_agent` and loads the same code under two module identities — that silently double-registers tools, defeats `monkeypatch`, and breaks `isinstance`. `pyproject.toml` sets pytest `pythonpath = ["."]` (project root only) for exactly this reason; **do not** add `mini_agent` to the path.

## Commands

```bash
# Setup (editable install + dev deps)
pip install -e ".[dev]"
cp .env.example .env          # then put a real DEEPSEEK_API_KEY in .env

# Run the REPL
python -m mini_agent                     # new session (prints its session id)
python -m mini_agent --resume sess_xxxx  # recover & continue a prior session

# Tests
pytest                        # everything
pytest -m "not slow"          # day-to-day: skips the subprocess crash-kill tests
pytest -m slow                # only the real-SIGKILL crash-recovery tests (slow)
pytest tests/test_persistence.py                 # one file
pytest tests/test_persistence.py::test_name -x   # one test, stop on first failure
```

Tests need no API key or network: `tests/conftest.py` provides a `fake_llm` fixture that monkeypatches `mini_agent.agent.call_llm` with scripted `FakeMessage` responses, making the whole loop deterministic.

## Architecture

The loop lives in **`agent.py :: Agent.run_conversation`**: record the user message → call the model with the current tool defs → if the assistant returned `tool_calls`, execute them and append `role="tool"` results, then loop; otherwise the plain-text reply ends the turn. `IterationBudget` caps turns; when exhausted, one final `tools=None` call forces a text answer instead of looping forever. An `Agent` with no `session_id`/`db` is pure in-memory (no persistence); with no `toolset` it degrades to a plain chat loop (no tools exposed).

Four subsystems the loop leans on:

- **LLM adapter (`llm.py`)** — the only place that talks to the network. `call_llm(messages, tools)` → OpenAI-style message object. Client is lazily built so importing without a key won't crash (keeps tests mockable).

- **Tool registry (`tools/registry.py`)** — one module-level `registry` singleton. Tools self-register via **import side effect**: a `builtin/*.py` module calls `registry.register(...)` at top level. `discover_builtin_tools()` AST-scans `builtin/` for exactly that top-level `registry.register(...)` call and `import_module`s the matches (full package names only — bare names would create a second `_tools` dict). `get_definitions()` filters by `toolset` (⊆ authorization, for future subagents), a `disabled` blacklist, and each tool's runtime `check_fn`, then **returns tools sorted by name** — a stable prefix is required for DeepSeek's context cache to hit. `dispatch()` is the robustness contract: it **never raises and always returns a string**; errors come back as `{"error": ...}` JSON so a failed tool can't kill the loop. Runtime-injected kwargs (e.g. `session_id`) that collide with model-supplied args are rejected, not silently overwritten.

- **Persistence + crash recovery (`persistence/session_db.py`, `schema.py`)** — single SQLite file in WAL mode, three tables: `sessions`, `messages` (`(session_id, seq)` PK, whole `Message` stored as JSON), `executed_keys` (idempotency). Every produced message is written immediately (`_record` in `agent.py`), so a crash loses at most the last in-flight message. Session status is a **CAS state machine** (`running`→`done` on exit, `done`→`running` on `--resume`) so illegal/concurrent transitions are caught rather than silently overwriting a real outcome. All sqlite/json exceptions are funneled to `PersistenceError` at the module boundary via `_db_guard` — callers never import `sqlite3`.

  **The exactly-once mechanism:** in `_execute_tool_calls`, `record_executed_key` is written *before* the `tool` result message. If the process dies between those two writes, `recover()` → `_sanitize_dangling_tool_calls` finds the assistant's `tool_call` with no matching result, looks it up in `executed_keys` by `tool_call_id`, and **stubs the real result back** — no re-execution of a non-idempotent tool. If nothing is cached, it stubs an `interrupted` marker so the transcript stays validly paired (a dangling `tool_calls` would 400 on the next request) and lets the model decide whether to retry. **`recover()`'s golden rule:** its input is *historical data*, not program state — anything that "should never happen" (corrupt JSON row, empty `tool_call.id`, non-str cached content) is **degraded (skip/stub) + logged, never raised**. A single bad row must not permanently brick a session's `--resume`. Only errors that break *access to history* (whole table unreadable, DB corrupt) propagate.

- **CLI entry (`__main__.py`)** — thin REPL. Opens the DB (`<MINI_AGENT_HOME>/state.db`, default `./.mini_agent/`), and on `--resume` does **recover *then* reopen** (recover is pure-read and retryable; reopen is the "I'm taking over" commitment — never flip status before you know you can recover).

**Runtime state** lives under `settings.home` (default `./.mini_agent/`, gitignored) — a fresh checkout starts clean; delete the dir to reset. Config is centralized in `config.py` (`settings` object from `.env`); default model is `deepseek-v4-flash`.

## Implemented vs. stubbed

Much of the blueprint is **empty stub files** (0 bytes) — do not assume a module works because it exists. As of now:

- **Working:** `agent.py`, `llm.py`, `config.py`, `schema.py`, `log.py`, `__main__.py`, `tools/registry.py`, `tools/model_tools.py` (sync→async bridge), `tools/builtin/{file_tools,web_tool}.py`, all of `persistence/`.
- **Empty stubs:** `loop.py`, `budget.py`, `delegate.py` (subagents), `prompt.py`, `cli.py`, `context/*` (compression), `memory/*` (self-improvement), `tools/{dispatch,executor,state_tools}.py`, `tools/builtin/shell_tool.py`. Their `tests/test_*` files exist but exercise unwritten code — expect skips/failures there until implemented.

When implementing a stubbed subsystem, follow the milestone intent in `开发路线.md` **as filtered through the two hard constraints above**.
