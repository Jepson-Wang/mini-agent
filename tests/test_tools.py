"""[M1] 工具系统 + 主循环接线测试。

分三块：
  1. dispatch 的健壮性契约（含这次补的两个洞：参数撞名、signature 无法内省）
  2. get_definitions 的过滤与顺序
  3. agent 把工具接进主循环（tool_call → 执行 → 回填 → 继续）
     以及每条消息都落库（M2）

全部用 FakeLLM，不需要真 API、不联网。
"""
from __future__ import annotations

import json

import pytest

from mini_agent.agent import Agent
from mini_agent.schema import Message
from mini_agent.tools import registry as registry_mod
from mini_agent.tools.registry import registry
from tests.conftest import FakeMessage


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_registry():
    """隔离全局注册表：测试里注册的工具在结束后清掉，不污染别的测试。

    先跑一次 discover 把内置工具装进来再快照，这样内置工具（web_fetch）留在
    基线里不会被误删——注意 import 是缓存的，删了就再也 register 不回来了。
    """
    registry_mod.discover_builtin_tools()
    tools_snapshot = dict(registry._tools)
    try:
        yield registry
    finally:
        registry._tools.clear()
        registry._tools.update(tools_snapshot)


def _tool_call_msg(call_id: str, name: str, arguments: dict) -> FakeMessage:
    """造一个带 tool_calls 的假响应，形状与 openai SDK 的 message 一致。"""
    tc = type("TC", (), {
        "id": call_id,
        "type": "function",
        "function": type("F", (), {
            "name": name,
            "arguments": json.dumps(arguments),
        })(),
    })()
    return FakeMessage(content=None, tool_calls=[tc])


# ---------------------------------------------------------------------------
# 1. dispatch 健壮性契约
# ---------------------------------------------------------------------------

def test_dispatch_unknown_tool(isolated_registry):
    out = json.loads(isolated_registry.dispatch("does_not_exist", {}))
    assert "Unknown tool" in out["error"]


def test_dispatch_wraps_handler_exception(isolated_registry):
    def boom(path: str) -> str:
        raise FileNotFoundError(path)

    isolated_registry.register(name="boom", toolset="t",
                               schema={"description": "x"}, handler=boom)
    out = json.loads(isolated_registry.dispatch("boom", {"path": "a.txt"}))
    assert out["error"].startswith("Tool failed: boom: FileNotFoundError")


def test_dispatch_bad_arg_name_is_distinct_from_crash(isolated_registry):
    """参数名写错 → "Invalid arguments"，和"工具内部炸了"是两类错误。"""
    isolated_registry.register(name="reader", toolset="t",
                               schema={"description": "x"}, handler=lambda path: path)
    out = json.loads(isolated_registry.dispatch("reader", {"wrong_name": "a.txt"}))
    assert "Invalid arguments" in out["error"]


def test_dispatch_non_dict_args(isolated_registry):
    isolated_registry.register(name="reader", toolset="t",
                               schema={"description": "x"}, handler=lambda path: path)
    out = json.loads(isolated_registry.dispatch("reader", "just a string"))
    assert "expected an object" in out["error"]


def test_dispatch_serializes_non_string_return(isolated_registry):
    isolated_registry.register(name="d", toolset="t", schema={"description": "x"},
                               handler=lambda: {"ok": True, "msg": "中文"})
    out = isolated_registry.dispatch("d", {})
    assert isinstance(out, str)
    assert json.loads(out) == {"ok": True, "msg": "中文"}


# --- 洞 1：注入上下文与模型参数撞名，绝不静默覆盖 ---

def test_dispatch_rejects_arg_name_collision(isolated_registry):
    """模型发来的 args 里若混进了运行时注入名（session_id），要报错而非覆盖。"""
    isolated_registry.register(name="ctx", toolset="t", schema={"description": "x"},
                               handler=lambda session_id, x=None: session_id)
    # 模型试图自己塞一个 session_id，同时 loop 也注入一个 → 冲突
    out = json.loads(
        isolated_registry.dispatch("ctx", {"session_id": "evil"}, session_id="real")
    )
    assert "conflict" in out["error"].lower()
    assert "session_id" in out["error"]


def test_dispatch_injected_kwargs_reach_handler(isolated_registry):
    """不撞名时，注入的 kwargs 应该正常传进 handler（模型看不见这个参数）。"""
    isolated_registry.register(name="ctx", toolset="t", schema={"description": "x"},
                               handler=lambda text, session_id: f"{text}@{session_id}")
    out = isolated_registry.dispatch("ctx", {"text": "hi"}, session_id="sess_1")
    assert out == "hi@sess_1"


# --- 洞 2：signature 无法内省时，dispatch 不能崩 ---

def test_dispatch_runs_async_handler(isolated_registry):
    """is_async=True 的工具：dispatch 靠 _run_async 把协程跑完，返回其结果。"""
    async def fetch(url: str) -> dict:
        return {"url": url, "ok": True}

    isolated_registry.register(name="afetch", toolset="t", schema={"description": "x"},
                               handler=fetch, is_async=True)
    out = json.loads(isolated_registry.dispatch("afetch", {"url": "http://x"}))
    assert out == {"url": "http://x", "ok": True}


def test_dispatch_survives_uninspectable_handler(isolated_registry, monkeypatch):
    """inspect.signature 抛 ValueError（某些 C 内置/partial）时，跳过预校验、
    照常执行，绝不让 ValueError 逃出 dispatch 破坏「永远返回 JSON」的契约。"""
    def raise_valueerror(_):
        raise ValueError("no signature for builtin")

    monkeypatch.setattr(registry_mod.inspect, "signature", raise_valueerror)
    isolated_registry.register(name="ok", toolset="t", schema={"description": "x"},
                               handler=lambda **kw: "executed")
    out = isolated_registry.dispatch("ok", {"anything": 1})
    assert out == "executed"   # 没有变成 Invalid arguments，也没有抛异常


# ---------------------------------------------------------------------------
# 2. get_definitions 过滤 + 顺序
# ---------------------------------------------------------------------------

def test_get_definitions_filters_by_toolset_and_is_sorted(isolated_registry):
    # 用测试专属的 toolset 名，避免和真实内置工具（file/web 集）撞在一起
    isolated_registry.register(name="z_tool", toolset="ftest",
                               schema={"description": "z", "parameters": {}}, handler=lambda: "")
    isolated_registry.register(name="a_tool", toolset="ftest",
                               schema={"description": "a", "parameters": {}}, handler=lambda: "")
    isolated_registry.register(name="other", toolset="wtest",
                               schema={"description": "o", "parameters": {}}, handler=lambda: "")

    defs = isolated_registry.get_definitions(toolset={"ftest"})
    names = [d["function"]["name"] for d in defs]

    assert names == ["a_tool", "z_tool"]         # 只剩 ftest 集，且按名排序（缓存友好）
    assert defs[0]["type"] == "function"          # DeepSeek/OpenAI 形状


def test_get_definitions_check_fn_hides_unavailable(isolated_registry):
    isolated_registry.register(name="gated", toolset="file", schema={"description": "x"},
                               handler=lambda: "", check_fn=lambda: False)
    names = [d["function"]["name"] for d in isolated_registry.get_definitions(toolset={"file"})]
    assert "gated" not in names


# ---------------------------------------------------------------------------
# 3. agent 主循环接工具（M1）+ 每条消息落库（M2）
# ---------------------------------------------------------------------------

def test_agent_runs_tool_then_answers(isolated_registry, fake_llm, db):
    """完整一轮：模型发 tool_call → agent 执行 → 回填 → 模型据结果给文本答复。"""
    isolated_registry.register(
        name="echo", toolset="t", schema={"description": "echo"},
        handler=lambda text: {"echoed": text},
    )
    fake_llm([
        _tool_call_msg("call_1", "echo", {"text": "hello"}),
        FakeMessage(content="结果是 hello"),
    ])

    agent = Agent(db=db, toolset={"t"})
    out = agent.run_conversation("sess_1", "说 hello")

    assert out["completed"] is True
    assert out["final_response"] == "结果是 hello"
    assert out["used_turns"] == 2

    # 配对铁律：assistant(tool_calls) 后面紧跟 id 对应的 tool 消息
    history = db.recover("sess_1")
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert history[1].tool_calls[0].id == "call_1"
    assert history[2].tool_call_id == "call_1"
    assert json.loads(history[2].content) == {"echoed": "hello"}


def test_agent_tool_error_does_not_crash_loop(isolated_registry, fake_llm, db):
    """工具抛异常时，模型收到的是 {"error":...} 回填，循环继续、能正常收尾。"""
    def boom(**kw):
        raise RuntimeError("炸了")

    isolated_registry.register(name="boom", toolset="t",
                               schema={"description": "x"}, handler=boom)
    fake_llm([
        _tool_call_msg("c1", "boom", {}),
        FakeMessage(content="工具失败了，我换个方式"),
    ])

    agent = Agent(db=db, toolset={"t"})
    out = agent.run_conversation("sess_1", "go")

    assert out["completed"] is True
    tool_msg = db.recover("sess_1")[2]
    assert tool_msg.role == "tool"
    assert "Tool failed" in json.loads(tool_msg.content)["error"]


def test_agent_persists_full_transcript(isolated_registry, fake_llm, tmp_path):
    """M2 接线：一轮带工具的对话，每条消息都要落库，换连接能原样 recover。"""
    from mini_agent.persistence.session_db import MiniSessionDB

    isolated_registry.register(name="echo", toolset="t",
                               schema={"description": "x"}, handler=lambda text: {"echoed": text})
    fake_llm([
        _tool_call_msg("call_1", "echo", {"text": "hi"}),
        FakeMessage(content="done"),
    ])

    db_file = str(tmp_path / "state.db")
    db = MiniSessionDB(db_file)
    sid = db.create_session()
    agent = Agent(db=db, toolset={"t"})
    agent.run_conversation(sid, "go")

    # 新连接 = 新进程：只信磁盘
    revived = MiniSessionDB(db_file).recover(sid)
    assert [m.role for m in revived] == ["user", "assistant", "tool", "assistant"]
    assert revived[0].content == "go"
    assert revived[1].tool_calls[0].name == "echo"
    assert json.loads(revived[2].content) == {"echoed": "hi"}
    assert revived[3].content == "done"


def test_running_an_existing_session_does_not_rewrite_its_history(fake_llm, db):
    """对已有历史的 session 再跑一轮：历史只能往后增，绝不能被重写一遍。

    前身是 test_initial_messages_seed_memory_without_rewriting_history。那时
    --resume 靠构造器的 initial_messages 播种内存，边界是"播种绝不能走 _record"。
    无状态化后 initial_messages 没了、历史改由 run_conversation 内部 recover 取回，
    但守的不变量一模一样：recover 出来的消息本就来自库，若被重新落库一遍，
    seq 会从 MAX+1 续着涨、行数翻倍，历史彻底乱套。
    """
    db.ensure_session("sess_1")
    db.append_message("sess_1", Message(role="user", content="q1"))
    db.append_message("sess_1", Message(role="assistant", content="a1"))

    fake_llm([FakeMessage(content="a2")])
    agent = Agent(db=db)
    agent.run_conversation("sess_1", "q2")

    # 恰好多出 q2 + a2 两条，原有两条原封不动
    assert [m.content for m in db.recover("sess_1")] == ["q1", "a1", "q2", "a2"]
    n = db.conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?;", ("sess_1",)
    ).fetchone()[0]
    assert n == 4


def test_retry_after_crash_mid_tool_turn_does_not_duplicate_the_user_message(
    isolated_registry, db, monkeypatch
):
    """崩在**工具已执行之后**：历史以 tool 结尾，重试同样不该再插一条 user。

    这一例专门盯着"末条是不是 user"那种朴素判据 —— 它在这里会漏判，插出
    [user, assistant(tc), tool, user] 这种更难看的东西。正确的判据是
    「末条是不是纯文本 assistant」，也就是主循环自己的终止条件。
    """
    isolated_registry.register(name="echo", toolset="t",
                               schema={"description": "x"},
                               handler=lambda text: {"echoed": text})
    calls = {"n": 0}

    def flaky(messages, tools=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _tool_call_msg("call_1", "echo", {"text": "hi"})
        if calls["n"] == 2:
            raise RuntimeError("LLM 超时")          # 工具已跑完、结果已落库
        return FakeMessage(content="done")

    monkeypatch.setattr("mini_agent.agent.call_llm", flaky)
    agent = Agent(db=db, toolset={"t"})

    with pytest.raises(RuntimeError):
        agent.run_conversation("sess_1", "说 hi")

    agent.run_conversation("sess_1", "说 hi")        # 调用方重试

    history = db.recover("sess_1")
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert [m.content for m in history].count("说 hi") == 1
