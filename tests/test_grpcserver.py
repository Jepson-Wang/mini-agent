"""gRPC surface 的测试。

大部分用例是**绊线**，不是行为测试：它们钉住的是三个"不报警、只是悄悄给错
答案"的坑，每一个都真的踩过（见各自的 docstring）。删掉它们，坑会原样回来。

这里一律走真 gRPC round-trip（起 server、连 channel），不直接调 servicer 方法。
RunTask 是生成器 + 后台线程 + 队列的组合，最容易出问题的恰恰是这三者的接缝，
直接调方法会把接缝全绕过去。
"""
from __future__ import annotations

import json
import threading
import time
from concurrent import futures

import grpc
import pytest

import mini_agent.agent as agent_mod
from mini_agent.agent import Agent, CANCELLED_MARKER
from mini_agent.gen.agent.v1 import agent_pb2, agent_pb2_grpc
from mini_agent.grpcserver.server import AgentServicer
from mini_agent.tools.registry import registry


# ---------- 脚手架 ----------


class FakeToolCall:
    """长得像 openai SDK 的 tool_call（ToolCall.from_openai 要 .id/.function）。"""

    def __init__(self, call_id: str, name: str, args: dict):
        self.id = call_id
        self.type = "function"
        self.function = type(
            "F", (), {"name": name, "arguments": json.dumps(args)}
        )()


@pytest.fixture(autouse=True)
def echo_tool():
    """一个确定性的工具。注册表是模块级单例，重复注册是幂等 re-import，安全。"""
    registry.register(
        name="echo",
        toolset="grpctest",
        description="原样回显",
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        handler=lambda text="": f"echoed:{text}",
    )


@pytest.fixture
def serve(db, monkeypatch):
    """起一个真 gRPC server，返回 (stub, agent)。用完自动关。

    llm 参数是 call_llm 的替身；monkeypatch 打在 mini_agent.agent 上，
    理由见 conftest.fake_llm 的说明（模块必须只有一个身份）。
    """
    servers = []

    def _make(llm, max_turns: int = 25):
        monkeypatch.setattr("mini_agent.agent.call_llm", llm)
        agent = Agent(db=db, toolset={"grpctest"}, max_turns=max_turns)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
        agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentServicer(agent), server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        servers.append((server, channel))
        return agent_pb2_grpc.AgentServiceStub(channel), agent

    yield _make

    for server, channel in servers:
        channel.close()
        server.stop(0)


class Msg:
    """假的 assistant 回复。"""

    def __init__(self, content=None, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


def run_task(stub, task_uid: str, session_id: str, text: str) -> list:
    return list(stub.RunTask(agent_pb2.RunTaskRequest(
        task_uid=task_uid, session_id=session_id, input=text)))


def type_names(events) -> list[str]:
    return [agent_pb2.EventType.Name(e.type) for e in events]


# ---------- 基本形状 ----------


def test_event_stream_shape(serve):
    """一个带工具调用的任务，事件类型、顺序、seq、task_uid 都要对。"""
    stub, _ = serve(iter_llm([
        Msg(content="先想想", tool_calls=[FakeToolCall("c1", "echo", {"text": "hi"})]),
        Msg(content="最终答案"),
    ]))

    events = run_task(stub, "t1", "s1", "问题")

    assert type_names(events) == [
        "EVENT_TYPE_STARTED",
        "EVENT_TYPE_THINKING",
        "EVENT_TYPE_TOOL_CALL",
        "EVENT_TYPE_TOOL_RESULT",
        "EVENT_TYPE_FINISHED",
    ]
    # seq 从 1 开始连续 —— Go 那侧靠它排序去重
    assert [e.seq for e in events] == [1, 2, 3, 4, 5]
    assert {e.task_uid for e in events} == {"t1"}
    assert all(e.timestamp > 0 for e in events)
    assert json.loads(events[-1].payload)["final_response"] == "最终答案"


def test_health_reports_version(serve):
    stub, _ = serve(iter_llm([]))
    resp = stub.Health(agent_pb2.HealthRequest())
    assert resp.ok is True
    assert resp.version


@pytest.mark.parametrize("req,missing", [
    (agent_pb2.RunTaskRequest(session_id="s", input="q"), "task_uid"),
    (agent_pb2.RunTaskRequest(task_uid="t", input="q"), "session_id"),
])
def test_runtask_rejects_missing_ids(serve, req, missing):
    """task_uid / session_id 空 → INVALID_ARGUMENT，不能默默跑。

    两者任一为空，后面所有事件都失去归属：task_uid 是链路关联键，
    session_id 决定历史落在哪。
    """
    stub, _ = serve(iter_llm([Msg(content="x")]))
    with pytest.raises(grpc.RpcError) as exc:
        list(stub.RunTask(req))
    assert exc.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert missing in exc.value.details()


def test_duplicate_task_uid_is_rejected(serve):
    """同一个 task_uid 不许同时跑两遍。

    放行的话两份事件流会写到同一条任务上，Go 那边按 task_uid 更新状态，
    最后谁赢取决于时序 —— 不可复现，也没法解释。
    """
    stub, _ = serve(endless_tools(delay=0.4))

    first = threading.Thread(
        target=lambda: run_task(stub, "dup", "s", "q1"), daemon=True)
    first.start()
    time.sleep(0.5)

    with pytest.raises(grpc.RpcError) as exc:
        run_task(stub, "dup", "s", "q2")
    assert exc.value.code() == grpc.StatusCode.ALREADY_EXISTS

    stub.CancelTask(agent_pb2.CancelTaskRequest(task_uid="dup"))
    first.join(timeout=15)


# ---------- 绊线一：预算耗尽不能报成功 ----------


def test_budget_exhausted_reports_error_not_finished(serve):
    """撞了 IterationBudget 上限的任务必须发 error，不能发 finished。

    Go 那侧只看事件**类型**定状态（finished→succeeded），根本不解析 payload。
    发 finished 的话，一个压根没回答用户的任务会在界面上显示成"成功"，正文是
    "(stopped: iteration budget exhausted)"。这是最坏的一种错误：不报警，
    只是悄悄给了个不对的结果。
    """
    step = iter(range(1000))
    stub, _ = serve(
        lambda messages, tools=None: Msg(
            content="还要调",
            tool_calls=[FakeToolCall(f"c{next(step)}", "echo", {"text": "x"})],
        ),
        max_turns=3,
    )

    events = run_task(stub, "t_budget", "s_budget", "永远做不完的事")

    assert type_names(events)[-1] == "EVENT_TYPE_ERROR"
    assert "budget" in events[-1].payload.lower()


# ---------- 绊线二：取消之后不能吞掉下一句话 ----------


def test_cancel_does_not_swallow_next_task_input(serve, db):
    """取消一个任务后，同一个 session 的下一个任务的用户输入必须照常落库。

    run_conversation 有个 _turn_in_flight 判据：transcript 没以纯文本 assistant
    结尾，就认为"调用方在重试、用户那句话早已落库"，于是**不记**这次的输入。
    这个推断在只有崩溃的年代是对的，取消打破了它 —— 取消之后下一个任务带的是
    全新的输入，却会被当成重试丢掉，用户换了问题、模型接着答上一个被取消的
    问题，而新问题永远丢了。

    修法是 _seal_cancelled_turn：被取消的回合退出前补一条纯文本 assistant 封口。
    """
    stub, _ = serve(endless_tools(delay=0.4))

    done = threading.Event()

    def bg():
        try:
            run_task(stub, "t_cancel", "s_shared", "任务A：做X")
        finally:
            done.set()

    threading.Thread(target=bg, daemon=True).start()
    time.sleep(0.9)

    assert stub.CancelTask(agent_pb2.CancelTaskRequest(task_uid="t_cancel")).cancelled
    assert done.wait(timeout=15), "取消后流没有及时结束"

    # 封口标记在，且 transcript 不再是"跑了一半"的样子
    history = db.recover("s_shared")
    assert history[-1].role == "assistant"
    assert history[-1].content == CANCELLED_MARKER
    assert not history[-1].tool_calls

    # 关键断言：下一个任务的输入必须进库
    stub2, _ = serve(iter_llm([Msg(content="任务B的答案")]))
    run_task(stub2, "t_next", "s_shared", "任务B：做完全不同的Y")

    users = [m.content for m in db.recover("s_shared") if m.role == "user"]
    assert users == ["任务A：做X", "任务B：做完全不同的Y"], users


def test_cancel_emits_no_terminal_event(serve):
    """取消是正常结局，不发 finished 也不发 error，让流干净 EOF。

    库里在通知我们之前就已经是 canceled 了（Go 是先落库、再通知），
    再发一个终态事件会把它盖掉。
    """
    stub, _ = serve(endless_tools(delay=0.4))

    events: list = []
    done = threading.Event()

    def bg():
        try:
            for ev in stub.RunTask(agent_pb2.RunTaskRequest(
                    task_uid="t_c2", session_id="s_c2", input="q")):
                events.append(ev)
        finally:
            done.set()

    threading.Thread(target=bg, daemon=True).start()
    time.sleep(0.9)
    stub.CancelTask(agent_pb2.CancelTaskRequest(task_uid="t_c2"))
    assert done.wait(timeout=15)

    names = type_names(events)
    assert "EVENT_TYPE_FINISHED" not in names
    assert "EVENT_TYPE_ERROR" not in names


def test_cancel_unknown_task_explains_why(serve):
    """取消一个不在跑的任务：cancelled=false，且 message 必须说人话。

    那是对面唯一能拿到的失败解释（proto 里写明了"别留空"），
    agentclient.CancelTask 会把它拼进 error 里。
    """
    stub, _ = serve(iter_llm([]))
    resp = stub.CancelTask(agent_pb2.CancelTaskRequest(task_uid="从来没有过"))
    assert resp.cancelled is False
    assert resp.message.strip()


# ---------- 绊线三：同一 session 并发不能交织 ----------


def test_same_session_tasks_are_serialized(serve, db):
    """同一个 session 上并发两个任务，transcript 必须是两段完整的回合，不能交织。

    run_conversation 的形状是「读全量历史 → 往末尾追加」，两个任务同时跑会把
    两段对话搅成 user,user,assistant,tool,assistant,tool,... 而且这个损坏是
    **永久**的：之后每一轮都会把这坨东西发给模型。两个任务还都报成功。
    CLI 时代碰不到（单线程 REPL），是 gRPC 这个 surface 让它变得可达。
    """
    def llm(messages, tools=None):
        time.sleep(0.15)
        last_user = [m["content"] for m in messages if m.get("role") == "user"][-1]
        # 已经为这句话调过工具了就收尾，否则先调一次工具
        called = any(
            m.get("role") == "assistant" and m.get("tool_calls")
            for m in messages[messages.index(
                next(m for m in messages if m.get("content") == last_user)):]
        )
        if called:
            return Msg(content=f"答案-{last_user}")
        return Msg(content=f"想-{last_user}",
                   tool_calls=[FakeToolCall(f"{last_user}-c", "echo", {"text": "x"})])

    stub, _ = serve(llm)

    threads = [
        threading.Thread(target=run_task, args=(stub, f"tc{i}", "s_race", f"Q{i}"))
        for i in (1, 2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    roles = [m.role for m in db.recover("s_race")]
    # 两段完整的回合首尾相接；交织的话会出现连续两个 user
    assert roles == ["user", "assistant", "tool", "assistant"] * 2, roles


def test_different_sessions_run_concurrently(serve, db):
    """不同 session 之间不该互相串扰，也不该被 session 锁挡住。"""
    def llm(messages, tools=None):
        time.sleep(0.2)
        return Msg(content="答-" + [m["content"] for m in messages
                                    if m.get("role") == "user"][-1])

    stub, _ = serve(llm)

    results: dict[int, list] = {}
    threads = [
        threading.Thread(
            target=lambda i=i: results.__setitem__(
                i, run_task(stub, f"t{i}", f"s{i}", f"Q{i}"))
        )
        for i in range(4)
    ]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    elapsed = time.monotonic() - start

    for i in range(4):
        assert json.loads(results[i][-1].payload)["final_response"] == f"答-Q{i}"
        assert {e.task_uid for e in results[i]} == {f"t{i}"}
    # 真并发：4 个各 0.2s 的任务不该退化成串行的 0.8s
    assert elapsed < 0.7, f"不同 session 被串行化了，用了 {elapsed:.2f}s"


# ---------- 异常映射 ----------


def test_agent_exception_becomes_error_event(serve):
    """agent 抛异常必须变成 error 事件，而不是让流裸断。

    裸断的话 Go 那边收到的是一条没有解释的 EOF —— 任务会被当成成功。
    payload 里只放类型和消息，不带 traceback（那东西会跟着任务详情回显给用户）。
    """
    def boom(messages, tools=None):
        raise RuntimeError("模型炸了")

    stub, _ = serve(boom)
    events = run_task(stub, "t_err", "s_err", "q")

    assert type_names(events)[-1] == "EVENT_TYPE_ERROR"
    assert "RuntimeError" in events[-1].payload
    assert "模型炸了" in events[-1].payload
    assert "Traceback" not in events[-1].payload


# ---------- helper ----------


def iter_llm(scripted: list, delay: float = 0.0):
    """按脚本依次返回；用完给一个收尾文本，避免无限循环。"""
    box = list(scripted)

    def _call(messages, tools=None):
        if delay:
            time.sleep(delay)
        return box.pop(0) if box else Msg(content="(脚本用尽)")

    return _call


def endless_tools(delay: float = 0.4):
    """一个**永远不收尾**的模型：每轮都要求再调一次工具。

    专给取消类用例用。★ 别拿"纯文本 Msg + sleep"当慢任务 —— 纯文本正是主循环的
    终止条件，那样的任务一轮就结束了，测试里的 CancelTask 会打在一个早已跑完的
    任务上，于是断言"没有终态事件"必然失败。要让循环转下去，每一轮都得带
    tool_calls。
    """
    step = iter(range(10_000))

    def _call(messages, tools=None):
        time.sleep(delay)
        return Msg(
            content="继续干活",
            tool_calls=[FakeToolCall(f"c{next(step)}", "echo", {"text": "x"})],
        )

    return _call
