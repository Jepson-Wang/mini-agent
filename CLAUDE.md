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

  **Every access to `self.conn` — reads included — must hold `_conn_lock`** (`_read` / `_write_txn`). `check_same_thread=False` only disables the *check*; it does not make a `Connection` thread-safe. Python's `Connection` carries a shared statement cache and `conn.execute()` is the "make a cursor + use that cache" shortcut, so concurrent calls race on it. Unguarded reads produced `InterfaceError: bad parameter or other API misuse`, an `IndexError` from inside the `sqlite3` module, and — worst — **rows read back with `content=None` or empty**. That last one is silent data loss: `recover()`'s golden rule is to skip-and-log a corrupt row rather than raise, so a message just disappears from the transcript and nobody errors. The CLI never hit this (single-threaded); the gRPC surface made it reachable. Regression test: `test_concurrent_read_write_does_not_corrupt` (it shrinks `sys.setswitchinterval` so the race reproduces every run rather than one in three). If you ever need real read concurrency, give each thread its own connection — WAL readers don't block on writers — don't drop the lock.

  **The exactly-once mechanism:** in `_execute_tool_calls`, `record_executed_key` is written *before* the `tool` result message. If the process dies between those two writes, `recover()` → `_sanitize_dangling_tool_calls` finds the assistant's `tool_call` with no matching result, looks it up in `executed_keys` by `(session_id, assistant_seq, tool_index)`, and **stubs the real result back** — no re-execution of a non-idempotent tool. If nothing is cached, it stubs an `interrupted` marker so the transcript stays validly paired (a dangling `tool_calls` would 400 on the next request) and lets the model decide whether to retry. **`recover()`'s golden rule:** its input is *historical data*, not program state — anything that "should never happen" (corrupt JSON row, empty `tool_call.id`, non-str cached content) is **degraded (skip/stub) + logged, never raised**. A single bad row must not permanently brick a session's `--resume`. Only errors that break *access to history* (whole table unreadable, DB corrupt) propagate.

  **The idempotency key contains no model input.** It used to be the model-supplied `tool_call.id`, global across sessions: a colliding id let session B recover session A's tool output (a privacy leak), and an id reused across rounds made round 3 recover round 1's result. Now the PK is `(session_id, assistant_seq, tool_index)` — `assistant_seq` is the seq `append_message` assigned to the assistant that issued the calls (FK → `messages(session_id, seq)`), `tool_index` its position in `tool_calls` (aligned because calls run serially in that order; M6 parallel tools must bind idx *before* dispatch and persist tool messages in idx order). `tool_call_id` is only a validation column: a stored id that differs from `tc.id` is treated as a miss. Likewise, **pairing tool messages to calls is per window, never per session**: an assistant owns only the run of `tool` rows directly after it (cut by the DB `role` column, so a corrupt assistant row still ends the window), with one FIFO queue per id. A session-wide `{tool_call_id: msg}` index used to mis-pair reused ids on *every* `run_conversation` — the stateless agent recovers history each call, so this wasn't a crash-only bug. A row whose `role` column disagrees with its JSON is treated as corrupt. If `record_executed_key` returns `is_first=False` (only possible when an already-persisted assistant's calls are replayed), the agent uses the canonical stored result and warns.

  **Schema is versioned with `PRAGMA user_version` (currently 1).** `_create_tables` has a read-only fast path (version current → return, no write lock; version newer than the code → `PersistenceError`) and a migration slow path inside `BEGIN IMMEDIATE` that re-checks the version first (double-checked — concurrent openers migrate once; regression test `test_concurrent_migration_happens_once` forces the race with a barrier). The v0 `executed_keys` is renamed to `executed_keys_legacy_v0`, never dropped and never read. **Deployment constraint:** old code against a v1 DB takes its fast path (all tables exist) and then fails on INSERT with `no column named idempotency_key` — *after* the tool already ran. Upgrade the CLI and gRPC surfaces together.

- **CLI entry (`__main__.py`)** — thin REPL. Opens the DB (`<MINI_AGENT_HOME>/state.db`, default `./.mini_agent/`), and on `--resume` does **recover *then* reopen** (recover is pure-read and retryable; reopen is the "I'm taking over" commitment — never flip status before you know you can recover).

**Runtime state** lives under `settings.home` (default `./.mini_agent/`, gitignored) — a fresh checkout starts clean; delete the dir to reset. Config is centralized in `config.py` (`settings` object from `.env`); default model is `deepseek-v4-flash`.

## Implemented vs. stubbed

Much of the blueprint is **empty stub files** (0 bytes) — do not assume a module works because it exists. As of now:

- **Working:** `agent.py`, `llm.py`, `config.py`, `schema.py`, `log.py`, `__main__.py`, `grpcserver/server.py`, `gen/` (generated protobuf), `tools/registry.py`, `tools/model_tools.py` (sync→async bridge), `tools/builtin/{file_tools,web_tool}.py`, all of `persistence/`.
- **Empty stubs:** `loop.py`, `budget.py`, `delegate.py` (subagents), `prompt.py`, `cli.py`, `context/*` (compression), `memory/*` (self-improvement), `tools/{dispatch,executor,state_tools}.py`, `tools/builtin/shell_tool.py`.
- **Their tests are empty too**, which matters more than it sounds: `tests/test_{compressor,delegate,memory}.py` are **0 bytes** — they collect nothing, so the suite is green without covering any of it. Don't read "113 passed" as "the stubs are fine". (`tests/test_loop.py` is *not* a stub test despite the name — it exercises `agent.py`'s loop.)

## Two surfaces, not one

`Agent` is driven by **two** equal-footing entry points, and neither may import the other:

- **CLI (`__main__.py`)** — the REPL described above.
- **gRPC (`grpcserver/server.py`)** — server-streaming `RunTask`, consumed by the Go backend (which is a gRPC *client*). `session_id` is minted by Go and passed in; this module is protocol-only.

Anything both need (`DEFAULT_TOOLSET`) lives in `config.py`. Putting it in `__main__.py` is specifically wrong: `python -m mini_agent` makes that file exist as **both** `__main__` and `mini_agent.__main__`, so importing it re-runs the whole CLI.

**Secrets never enter `repr()`.** `Settings.deepseek_api_key` and `Settings.grpc_token` are declared `field(repr=False)`, because a dataclass's generated `__repr__` prints every field — so one `logger.debug("settings=%s", settings)` would write the live API key into a log system that is collected, retained, and widely readable. Same reasoning as Go's `yaml:"-"` on its secret fields plus a startup log that prints only *whether* a credential is present. Guarded by `test_repr_hides_secrets`. Note it only covers `repr`/`str`, not `asdict()`.

**The gRPC surface is the higher-privilege one** — it runs arbitrary tasks with a caller-chosen `session_id`, reads/writes the workspace, and spends LLM budget with no ceiling. So it defaults to binding `127.0.0.1` and, the moment `MINI_AGENT_GRPC_HOST` is anything else, **refuses to start without `MINI_AGENT_GRPC_TOKEN`** (≥32 **bytes**, visible ASCII only, matched by Go's `AGENT_GRPC_TOKEN`). Fail-closed, same posture as Go's `middleware.InternalAuth`.

  Two details that cost real debugging time: the token must be visible ASCII (`0x21..0x7E`) because gRPC metadata rejects anything else — a Chinese-character token passes every length check, the server starts clean, and then *every* call dies in the transport with `Illegal header value` → `INTERNAL`, with nothing in the logs pointing at the credential. And the length is counted in **bytes**, matching Go's `len()`; counting characters would let a 20-char/40-byte token be accepted by one side and rejected by the other. The interceptor compares `bytes`, not `str`, because `hmac.compare_digest` raises `TypeError` on non-ASCII `str` — a security component's worst failure mode should be "deny", not "raise".

**The file-tool sandbox root is a security boundary, and the root — not the check — is the part that has been wrong.** `_resolve_in_workspace` (resolve + `is_relative_to`) is sound. But the root used to default to `os.getcwd()`, and the process is normally started from the repo root, which put `.env` *inside* the sandbox — `read_file('.env')` returned the live API key, and it would ride back out through `final_response`. The root now comes from `config.workspace_root()`, defaulting to `<home>/workspace`. Regression tests: `test_default_root_is_not_cwd`, `test_default_root_lives_under_home`.

  The tools' other contract is that **every** failure comes back as `{"error": ...}` JSON. `workspace_root()` calls `mkdir`, which raises `OSError`; letting that escape breaks the contract *and* leaks an absolute path — `registry.dispatch` wraps it as `Tool failed: ...`, which rides the `tool_result` event to Go, into the task detail, and out to the user. Defeating `_rel()`'s entire purpose through the back door. `_root()` converts it, logging the real cause (with the path) locally and returning a path-free message. Regression test: `test_unusable_workspace_reports_json_not_raw_exception`.

**`Health` reports a real self-check**, not a constant. It pings the DB (`MiniSessionDB.ping()`) so that "the process is up but can't work" is expressible — Go's client deliberately distinguishes "couldn't ask" from "asked, and it says it's unwell", and hardcoding `ok=True` made the second state unreachable.

When implementing a stubbed subsystem, follow the milestone intent in `开发路线.md` **as filtered through the two hard constraints above**.
