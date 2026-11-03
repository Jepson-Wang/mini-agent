"""M3 压缩总结 B 轮：压缩器、Agent 接入、失败降级、两个入口的默认开启。

对应 plan 第 3 节的测试 1–5（纯函数）与 11–19（端到端 / 装配）。6–10 在
test_compaction.py（A 轮）。

★ 所有总结调用都注入替身。 LLMCompressor(db, llm=替身)，替身只返回固定文本、
  记下每次调用的入参；主对话用 tests.conftest 的 FakeLLM（它自己会调用真的
  llm._extract_usage）。两者都不碰网络。
★ 阈值和尾部预算都是按常量推的，不猜魔法数字：
  window=2000 → 触发线 1000（ratio 0.5）；tail_budget = clamp(300, 4000, 24000) = 4000。
  尾部 = [assistant(tool_calls, 很小), tool(远大于 4000)]，累加第一步就超预算，
  切点落在 tool 上、往新对齐到下一条 user —— 边界 B 就是那条 user 的 seq（D7）。
★ 工具输出的大小按 estimate_message 反推（纯 ASCII 0.3 token/字符，每条 +4）：
  改了估算系数，这些用例的意图（"远超 tail_budget"）仍然成立。

S1–S6 是 B 轮开工前查明的接口缝隙，用例按它们编号对照。
"""
from __future__ import annotations

import hashlib
import json

import pytest

from mini_agent.agent import Agent, TaskCancelled
from mini_agent.context.compressor import (
    SUMMARY_PREFIX,
    VERBATIM_END,
    VERBATIM_HEADER,
    CompressionPolicy,
    CompactionError,
    LLMCompressor,
    _chunk,
    build_compression,
    prune_old_tool_outputs,
    render_transcript,
    select_boundary,
)
from mini_agent.context.tokens import (
    SOURCE_ESTIMATE,
    estimate_message,
    request_fingerprint,
)
from mini_agent.llm import LLMReply
from mini_agent.schema import Message, ToolCall
from mini_agent.tools.registry import registry
from tests.conftest import FakeLLM, FakeMessage

WINDOW = 2000
RATIO = 0.5
TAIL_BUDGET = 4000                       # clamp(2000 * 0.15 = 300, 4000, 24000)
MIDDLE_MIN = 2000                        # 中段估算下限


# ---------------------------------------------------------------------------
# 尺寸工具：一律用 estimate_message 反推，不在测试里写死魔法字符数
# ---------------------------------------------------------------------------

def _tokens_of(chars: int) -> int:
    """一条纯 ASCII tool 消息的估算 token（含每条消息的 +4 结构开销）。"""
    return estimate_message(Message(role="tool", tool_call_id="c", content="x" * chars))


def _chars_for(tokens: int) -> int:
    """反推：让一条纯 ASCII tool 消息的估算恰好 >= tokens。"""
    chars = int(tokens / 0.3) + 1
    while _tokens_of(chars) < tokens:
        chars += 1
    return chars


BIG_CHARS = _chars_for(TAIL_BUDGET + 2000)      # 单条就超过整个尾部预算
SMALL_CHARS = _chars_for(MIDDLE_MIN + 500)      # 中段里一条就够 2k
# 端到端种子用：两条加起来 ≈ 2.4k —— 够中段下限（2k），又装得进尾部预算（4k），
# 所以第一次请求前 select_boundary 返回 None（不压），工具返回后才压**一次**。
SEED_CHARS = _chars_for(1200)


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------

class SummaryLLM:
    """总结调用替身：记下每次入参，返回固定文本（或按开关抛异常）。"""

    def __init__(self, text: str = "SUMMARY-1", raises: bool = False):
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def __call__(self, messages, tools=None, max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools, "max_tokens": max_tokens})
        if self.raises:
            raise RuntimeError("summary llm 炸了")
        # 形状对齐生产契约 LLMReply(message, usage)。usage=None：总结调用的用量
        # 本来就不该当锚点，替身也不给。
        return LLMReply(message=FakeMessage(content=self.text), usage=None)


class _TC:
    """长得像 openai SDK 的 tool_call。"""

    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.type = "function"
        self.function = type("F", (), {"name": name, "arguments": json.dumps(arguments)})()


def _tool_call_msg(call_id: str, name: str, arguments: dict, usage=None) -> FakeMessage:
    return FakeMessage(
        content=None, tool_calls=[_TC(call_id, name, arguments)], usage=usage
    )


@pytest.fixture
def echo_tool():
    """toolset="cmptest" 下的确定性工具：echoed:<text>。用完从注册表移除。"""
    registry.register(
        name="echo_cmp", toolset="cmptest",
        schema={"description": "echo"}, handler=lambda text="": f"echoed:{text}",
    )
    yield "echo_cmp"
    with registry._lock:
        registry._tools.pop("echo_cmp", None)


def _agent(db, summary_llm, **kw) -> Agent:
    return Agent(
        db=db, toolset={"cmptest"},
        compressor=LLMCompressor(db, WINDOW, llm=summary_llm),
        policy=CompressionPolicy(WINDOW, RATIO),
        **kw,
    )


def _n_compactions(db) -> int:
    with db._read() as c:
        return c.execute("SELECT COUNT(*) FROM compactions;").fetchone()[0]


def _tool_rows(db, sid: str) -> list[tuple[int, str]]:
    """库里 role=tool 的原始行：(seq, 库里 JSON 里的 content)。"""
    with db._read() as c:
        rows = c.execute(
            "SELECT seq, content FROM messages WHERE session_id=? AND role='tool' "
            "ORDER BY seq;",
            (sid,),
        ).fetchall()
    return [(r["seq"], json.loads(r["content"])["content"]) for r in rows]


def _turn(sid, messages, seqs, boundary=None, cancel_check=None):
    """鸭子类型的 turn。和 _Turn 一样带 view_state —— _measure 靠它算指纹。"""
    prune_before = None
    return type("T", (), {
        "session_id": sid,
        "messages": messages,
        "seqs": seqs,
        "boundary": boundary,
        "prune_before": prune_before,
        "cancel_check": cancel_check,
    })()


# 分块用例里每条历史消息的正文大小：BIG_CHARS ≈ 6005 token，单条就超过
# window*0.5=1000 → 一条消息独占一块，块数 = 历史消息条数，断言才好写。
CHUNK_CHARS = BIG_CHARS


def _seed_for_compressor(db, *, rounds: int = 1):
    """造一段"中段足够大、尾部一上来就超预算"的历史，返回 (sid, ctx)。

    每轮 = 一条 user + 一条 assistant（各 ≈ 6005 token）。最后那条 assistant 一条
    就超尾部预算 → 切点落在它身上 → 边界 = 它，中段 = 它之前的全部历史。

    rounds=1：中段 ≈ 2.5k，按 window*0.5=1000 分块 → 2 块；
    rounds=2：中段 4 块（块数不写死，用例里按 _chunk 实测算）。
    """
    sid = db.create_session()
    for i in range(rounds):
        db.append_message(sid, Message(role="user", content=f"u{i}" + "x" * CHUNK_CHARS))
        db.append_message(sid, Message(role="assistant", content=f"a{i}" + "x" * CHUNK_CHARS))
    # 最后一条 user 留在尾部（新一轮的问题），由统计 assistant 撑爆尾部预算。
    # 那一轮之后若还有真实的 user，它就是"最后一条 user"，D6 允许它进中段——
    # 这里刻意让最后一条 user 留在尾部，是为了让下面的断言不依赖 VERBATIM 段。
    db.append_message(sid, Message(role="user", content="这次的问题"))
    db.append_message(sid, Message(role="assistant", content="x" * CHUNK_CHARS))
    return sid, db.load_context(sid)


def _db_middle(db, sid: str, boundary: int):
    """compress 实际总结的那一段：库里 seq ∈ [0, boundary) 的原始行（不是视图）。"""
    rows, seqs = db._load_rows(sid, from_seq=0)
    return [m for m, s in zip(rows, seqs) if s is not None and s < boundary]


# ===========================================================================
# 1. select_boundary：尾部预算（D6：最后一条 user 不再强制保留）
# ===========================================================================

def test_select_boundary_can_cut_past_the_last_user_message():
    """D6：**没有**"最后一条 user 留在尾部"的硬规则 —— 边界可以落在它之后。

    这个用例是 D6 的核心：会话里已经有一轮完整的 user → assistant（它们在库里
    是真实的 user/assistant 行），**最后一条 user 后面还跟着一条 huge assistant**。
    右到左累加时一条 assistant 就超尾部预算 → 切点落在它身上 → 边界 = 它的 seq，
    于是最后一条 user 进了中段。旧规则会把切点拽回那条 user 之前，边界退回去、
    与上一次相同，下一次就永远 None。
    """
    msgs = [
        Message(role="user", content="x" * SMALL_CHARS),                     # 中段
        Message(role="assistant", content="x" * SMALL_CHARS),                # 中段
        Message(role="user", content="这次的问题"),                          # 最后一条 user
        Message(role="assistant", content="x" * BIG_CHARS),                 # 朴素切点
    ]
    assert estimate_message(msgs[3]) > TAIL_BUDGET       # 构造前提成立
    assert select_boundary(msgs, [1, 2, 3, 4], None, WINDOW) == 4

    # 最后一条 user 自己撑爆尾部预算：朴素切点落在它之后，往新走到末尾 → 回退
    # 落在它身上（它进中段、原话由 compress 逐字兜底）。中段 = 前两条 ≈ 5k，够 2k。
    # 旧规则在这里是"拽到它前面"，边界 = 2；D6 之后 = 3。
    msgs2 = [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", content="x" * SMALL_CHARS),
        Message(role="user", content="x" * BIG_CHARS),
    ]
    assert estimate_message(msgs2[2]) > TAIL_BUDGET
    assert select_boundary(msgs2, [1, 2, 3], None, WINDOW) == 3

    # 也不能一路拖到最后一条 user 前面：尾部 = 那条 assistant(≈4k) + 短 user，
    # 累加 assistant 就超预算 → 切点落在 user 上，边界 = 3。
    msgs3 = [
        Message(role="user", content="x" * SMALL_CHARS),                   # 中段
        Message(role="assistant", content="x" * _chars_for(TAIL_BUDGET)),
        Message(role="user", content="这次的问题"),
    ]
    assert select_boundary(msgs3, [1, 2, 3], None, WINDOW) == 3


def test_select_boundary_none_when_everything_fits_in_tail_budget():
    msgs = [Message(role="user", content="短"), Message(role="assistant", content="也短")]
    assert select_boundary(msgs, [1, 2], None, WINDOW) is None
    # 尾部预算极小时切点走到末尾 → 回退到 assistant（2）；中段只有一条短 user
    # < 2k → None（D7 第 3 条回退之后，仍受中段下限约束）
    assert select_boundary(msgs, [1, 2], None, 1) is None


# ===========================================================================
# 2. 对齐（D7）：落在 tool / 补桩上 → 往**新**的方向走
# ===========================================================================

def test_select_boundary_aligns_a_tool_cut_forward_to_the_next_message():
    """D7：尾部累加在 tool 上超预算 → 切点落在 tool（不能当边界）→ **往新**对齐。

    超预算的 tool 连同触发它的 assistant 一起进中段（由 summary 代表），尾部
    反而更小。补桩（seq=None）同理，跳过它，落到下一条合法行。

    ★ 为什么种一段"旧的 user/assistant"：D7 往新对齐会把 tool 之后那条 user
      留作尾部，中段只剩 tool 之前的部分。会话从空开始（[user, assistant, tool]）
      时中段只有那条 user < 2k → 正确的结果是 None —— 那样这个用例测不到"对齐
      方向"。种一条真实的历史 assistant 后，中段 ≈ 2.5k，边界才会真的落到对齐目标。
    """
    history = [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", content="x" * SMALL_CHARS),
    ]
    msgs = history + [
        Message(role="user", content="x" * SMALL_CHARS),                     # 中段
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(role="tool", tool_call_id="c1", content="x" * BIG_CHARS),   # 朴素切点
        Message(role="user", content="这次的问题"),                          # 对齐目标
    ]
    assert select_boundary(msgs, [1, 2, 3, 4, 5, 6], None, WINDOW) == 6
    # 尾部预算极小时，朴素切点会落在最后一条 user 自己身上：它是合法边界 → 6
    # （D6：不再有"最后一条 user 必须留下"的规则）。
    assert select_boundary(msgs, [1, 2, 3, 4, 5, 6], None, 1) == 6

    # tool 后面跟着补桩（seq=None）和另一条 tool：一路跳到 assistant。
    msgs2 = history + [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(role="tool", tool_call_id="c1", content="x" * BIG_CHARS),   # 朴素切点
        Message(role="tool", tool_call_id="c2", content="x" * BIG_CHARS),   # 补桩
        Message(role="assistant", content="a"),
        Message(role="user", content="这次的问题"),
    ]
    assert select_boundary(msgs2, [1, 2, 3, 4, 5, None, 7, 8], None, WINDOW) == 7

    # 切点**恰好停在 tool 行上**的形态：超预算的是一条带大 content 的
    # assistant(tool_calls)，cut = 它之后 = 那条短 tool。上面两组里超预算的都是 tool
    # 本身（cut 直接越过它），测不到"不能落在 tool 上"——这一组才是那条规则的回归点。
    msgs3 = history + [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", content="x" * BIG_CHARS,
                tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(role="tool", tool_call_id="c1", content="短"),             # 朴素切点
        Message(role="user", content="这次的问题"),                          # 对齐目标
    ]
    assert select_boundary(msgs3, [1, 2, 3, 4, 5, 6], None, WINDOW) == 6


def test_select_boundary_only_returns_seqs_of_real_user_or_assistant_rows():
    """扫一串窗口：返回值只能是库里 user/assistant 行的 seq，绝不在 tool 行上。"""
    msgs = [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", content="x" * SMALL_CHARS),
        Message(role="assistant", tool_calls=[ToolCall(id="c1", name="t", arguments={})]),
        Message(role="tool", tool_call_id="c1", content="x" * BIG_CHARS),
        Message(role="user", content="x" * 300),
    ]
    seqs = [1, 2, 3, 4, 5]
    legal = {1, 2, 3, 5}
    for w in (1, 100, 1000, 2000, 5000, 10_000, 50_000):
        b = select_boundary(msgs, seqs, None, w)
        assert b is None or b in legal, (w, b)


# ===========================================================================
# 3. 返回 None：中段为空 / 中段 < 2k / 新边界 ≤ 当前边界
# ===========================================================================

def test_select_boundary_none_when_middle_is_below_minimum_or_empty():
    # 中段只有一条短 user（< 2k）。构造：最后一条重 user 自己就超尾部预算，
    # 切点落在它身上（下标 1），中段 = msgs[:1] = 一条短 user。
    msgs = [
        Message(role="user", content="短"),
        Message(role="user", content="x" * BIG_CHARS),
    ]
    assert select_boundary(msgs, [1, 2], None, WINDOW) is None
    # 中段为空
    assert select_boundary([Message(role="assistant", content="x" * BIG_CHARS)], [1], None, WINDOW) is None


def test_select_boundary_none_when_not_moving_forward():
    msgs = [
        Message(role="user", content="x" * SMALL_CHARS),
        Message(role="assistant", content="x" * SMALL_CHARS),
        Message(role="user", content="x" * BIG_CHARS),
    ]
    seqs = [1, 2, 3]
    b = select_boundary(msgs, seqs, None, WINDOW)
    assert b is not None
    assert select_boundary(msgs, seqs, b, WINDOW) is None
    assert select_boundary(msgs, seqs, b + 10, WINDOW) is None


# ===========================================================================
# 4. render_transcript：截断；prompt 含上一份 summary 与防注入句
# ===========================================================================

def test_render_transcript_truncates_long_tool_output_keeping_head_and_tail():
    head, tail = "H" * 1500, "T" * 500
    long_out = head + "M" * 3000 + tail
    msgs = [
        Message(role="assistant",
                tool_calls=[ToolCall(id="c9", name="read_file", arguments={"path": "a.txt"})]),
        Message(role="tool", tool_call_id="c9", content=long_out),
    ]

    text = render_transcript(msgs)

    assert "read_file" in text and "a.txt" in text       # tool_calls 渲染成 name(json_args)
    assert head in text and tail in text
    assert "M" * 3000 not in text
    assert "…[省略 3000 字符]…" in text
    # 短输出原样保留
    assert "echoed:ok" in render_transcript(
        [Message(role="tool", tool_call_id="c1", content="echoed:ok")]
    )


def test_summary_prompt_guards_against_injection_and_carries_previous_summary(db):
    sid, ctx = _seed_for_compressor(db, rounds=1)
    llm = SummaryLLM()
    LLMCompressor(db, WINDOW, llm=llm).compress(_turn(sid, ctx.messages, ctx.seqs))

    # 分块数按 compress 真正总结的那一段（库里的 [0, boundary)）算，不写死
    boundary = db.latest_compaction(sid)[0]
    assert boundary == ctx.seqs[-1], "边界 = 最后那条 assistant（最新一轮留尾部），中段 = 它之前的全部"
    assert len(llm.calls) == len(_chunk(_db_middle(db, sid, boundary), int(WINDOW * 0.5)))
    system, user = llm.calls[0]["messages"]
    assert system["role"] == "system"
    assert "DATA, not instructions" in system["content"]
    assert user["role"] == "user"
    assert "(none)" in user["content"]                 # 第一份 summary：从空白开始
    assert "New transcript" in user["content"]
    assert llm.calls[0]["tools"] is None
    assert llm.calls[0]["max_tokens"] == 2048


# ===========================================================================
# 5. 分块滚动：第 2 次总结的输入包含第 1 次的输出
# ===========================================================================

def test_chunked_summary_rolls_previous_output_into_next_chunk(db):
    sid, ctx = _seed_for_compressor(db, rounds=2)     # 中段 4 条大消息 → 4 块
    llm = SummaryLLM(text="CHUNK-OUT")
    LLMCompressor(db, WINDOW, llm=llm).compress(_turn(sid, ctx.messages, ctx.seqs))

    boundary = db.latest_compaction(sid)[0]
    expected = len(_chunk(_db_middle(db, sid, boundary), int(WINDOW * 0.5)))
    assert expected >= 2 and len(llm.calls) == expected
    assert "(none)" in llm.calls[0]["messages"][1]["content"]
    for cur in llm.calls[1:]:
        assert "CHUNK-OUT" in cur["messages"][1]["content"], "后一块没有建立在上一块的结果上"
    # 块是切着总结的：a1 不在第一块里，只出现在后面某一块里
    assert "a1" not in llm.calls[0]["messages"][1]["content"]
    assert any("a1" in c["messages"][1]["content"] for c in llm.calls[1:])


def test_empty_summary_is_a_compaction_error_with_candidate_boundary(db):
    sid, ctx = _seed_for_compressor(db, rounds=1)
    with pytest.raises(CompactionError) as ei:
        LLMCompressor(db, WINDOW, llm=SummaryLLM(text="   \n ")).compress(
            _turn(sid, ctx.messages, ctx.seqs)
        )
    assert ei.value.boundary is not None
    assert _n_compactions(db) == 0


# ===========================================================================
# 端到端脚手架：第一轮大工具输出触发压缩，第二轮纯文本收尾
# ===========================================================================

FINAL_ANSWER = "最终答案"
LONG_TEXT = "L" * BIG_CHARS


OLD_LONG = "O" * BIG_CHARS          # 旧工具轮里的长输出（D1 应当被替换）
OLD_SHORT = "s" * 150                # 旧工具轮里的短输出（<=200 字符，D1 应当保留）


def _seed_old_tool_round(db, sid: str) -> None:
    """种一轮**已经完成**的旧工具调用：user -> assistant(两个调用) -> 长/短两条 tool -> 文本收尾。

    D1 只替换候选边界 B **之前**的旧工具输出；最新一轮永远在尾部、不会被替换。
    所以要测 D1 真正起作用的形态，历史里必须先有一轮旧的长工具输出。
    """
    db.append_message(sid, Message(role="user", content="旧任务"))
    db.append_message(sid, Message(role="assistant", tool_calls=[
        ToolCall(id="old0", name="echo_cmp", arguments={"text": "long"}),
        ToolCall(id="old1", name="echo_cmp", arguments={"text": "short"}),
    ]))
    db.append_message(sid, Message(role="tool", tool_call_id="old0", content=f"echoed:{OLD_LONG}"))
    db.append_message(sid, Message(role="tool", tool_call_id="old1", content=f"echoed:{OLD_SHORT}"))
    db.append_message(sid, Message(role="assistant", content="旧任务完成"))


def _scenario(monkeypatch, db, summary_llm, *, usage=(10, 1), cancel=None,
              tool_texts=(LONG_TEXT,), max_turns=25, history=True, on_event=None):
    """端到端场景：一轮 user → 工具调用（大输出）→ 纯文本收尾。

    ★ history=True 时先往库里种一轮早先的对话（user + assistant，各 ≈ 2.5k token）。
      为什么需要：select_boundary 永远把**最新一轮**（assistant(tool_calls) + 它的
      大 tool 结果，模型还没读过）留在尾部，中段 = 本轮 assistant 之前的全部历史
      （种子 + 本轮 user）。会话从空开始时中段只有本轮那条短 user < 2k → 不压。
      种子落在 [2k, 4k)：第一次请求前整段装得进尾部预算（不压），工具返回后中段
      ≥ 2k（压且只压一次）—— 这是测试**前提**的调整，不是实现或断言的放宽。
    """
    fake = FakeLLM(
        [_tool_call_msg(f"c{i}", "echo_cmp", {"text": t}, usage=usage)
         for i, t in enumerate(tool_texts)]
        + [FakeMessage(content=FINAL_ANSWER, usage=usage)]
    )
    monkeypatch.setattr("mini_agent.agent.call_llm", fake)
    agent = _agent(db, summary_llm, max_turns=max_turns)
    sid = db.create_session()
    if history == "old_tool":
        _seed_old_tool_round(db, sid)
    elif history:
        db.append_message(sid, Message(role="user", content="早先的问题" + "x" * SEED_CHARS))
        db.append_message(sid, Message(role="assistant", content="早先的回答" + "x" * SEED_CHARS))
    out = agent.run_conversation(sid, "读一个大文件", cancel_check=cancel, on_event=on_event)
    return agent, sid, fake, out


# ===========================================================================
# 11. 越过阈值：检查点落库；压缩后的请求 = [system, summary…, 尾部] 且更短
# ===========================================================================

def test_compaction_writes_checkpoint_and_shrinks_next_request(monkeypatch, db, echo_tool):
    llm = SummaryLLM(text="SUMMARY-1")
    agent, sid, fake, out = _scenario(monkeypatch, db, llm)

    assert out["completed"] is True
    assert out["final_response"] == FINAL_ANSWER
    assert _n_compactions(db) == 1
    boundary, summary = db.latest_compaction(sid)
    # 中段 = 种子那两条 + "读一个大文件" ≈ 5k > window*0.5=1000 → 分块，但块数
    # 由 _chunk 决定，这里只验证内容。
    # D6：被压缩段内最后一条 user 的原话逐字放最前面（这句会被折进中段）。
    checkpoint_body = (
        VERBATIM_HEADER + "读一个大文件" + VERBATIM_END + "SUMMARY-1"
    )
    assert summary == checkpoint_body

    assert len(fake.calls) == 2
    before, after = fake.calls[0]["messages"], fake.calls[1]["messages"]
    # 第 1 次请求：视图里还没有检查点，第一句 user 就是种子的"早先的问题"
    assert before[1]["content"].startswith("早先的问题")
    assert after[0]["role"] == "system"
    assert after[1]["role"] == "user"
    # D4：尾部第一条是 assistant(tool_calls) → summary 单独成一条 user 插在它前面
    # （种子和本轮 user 都已被压进中段，没有可合并的 user）。
    assert after[1]["content"] == SUMMARY_PREFIX + checkpoint_body
    assert "早先的问题" not in after[1]["content"]
    assert "读一个大文件" in after[1]["content"]              # D6：原话逐字保留
    # 尾部 = 边界 B 之后的行：assistant(tool_calls) + tool
    assert [m["role"] for m in after[2:]] == ["assistant", "tool"]
    assert after[2]["tool_calls"][0]["id"] == "c0"
    assert boundary == db.load_context(sid).boundary
    # 请求确实变短：和"不压缩时这次请求本该多大"比（第一次请求里还没有那个大工具
    # 输出，拿它比没有意义）。不压缩的版本 = system + 库里全量历史中截至那条 tool 的部分。
    full = [m.to_openai() for m in db.recover(sid)]
    uncompressed = [after[0]] + full[: len(full) - 1]      # 去掉最后那条最终答案

    def size(req):
        return sum(len(m.get("content") or "") for m in req)

    assert size(after) < size(uncompressed)


# ===========================================================================
# 12. 压缩后旧锚点失效；新 assistant 的指纹带上新边界
# ===========================================================================

def test_fingerprint_carries_boundary_so_old_anchors_are_dropped(monkeypatch, db, echo_tool):
    llm = SummaryLLM(text="SUMMARY-1")
    seen: list[tuple] = []
    real_measure = Agent._measure

    def recording(self, turn, tools):
        count = real_measure(self, turn, tools)
        seen.append((turn.boundary, count.source))
        return count

    monkeypatch.setattr(Agent, "_measure", recording)
    agent, sid, fake, out = _scenario(monkeypatch, db, llm)

    boundary, _ = db.latest_compaction(sid)
    tools = agent._tool_defs()
    # 压缩刚完成后的那次计数（视图已带上新边界）：旧锚点的指纹是 (None, None)，
    # 与 (B, None) 对不上 → 没有可用锚点 → 整条估算
    after_compaction = [src for b, src in seen if b == boundary]
    assert after_compaction and after_compaction[0] == SOURCE_ESTIMATE

    # 压缩后发出的那次请求落库的 assistant：指纹 = (B, None)
    final = [m for m in db.recover(sid)
             if m.role == "assistant" and m.content == FINAL_ANSWER]
    assert len(final) == 1 and final[0].usage is not None
    assert final[0].usage.fingerprint == request_fingerprint(
        agent.system_prompt, tools, (boundary, None)
    )
    # 旧尺子（没有边界）不再匹配
    assert final[0].usage.fingerprint != request_fingerprint(agent.system_prompt, tools)


def test_view_state_none_normalizes_to_the_legacy_fingerprint():
    """S5：(None, None) 与不传 view_state 必须是同一个指纹，否则升级即让旧锚点全数失效。"""
    assert request_fingerprint("s", [{"a": 1}], (None, None)) == request_fingerprint(
        "s", [{"a": 1}]
    )
    assert request_fingerprint("s", [{"a": 1}], (5, None)) != request_fingerprint("s", [{"a": 1}])
    assert request_fingerprint("s", [{"a": 1}], (None, 7)) != request_fingerprint("s", [{"a": 1}])


def _hand_fingerprint(payload_dict: dict) -> str:
    """手算：与 9f69ea4 的 payload 形状对齐，不复用被测函数。"""
    payload = json.dumps(payload_dict, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def test_fingerprint_is_legacy_compatible_and_hand_computable():
    """F1：None / (None, None) 与旧版本逐字节相同；有视图状态时只多一个 prune_before 键。"""
    s, tools = "sys", [{"type": "function", "function": {"name": "t"}}]
    assert request_fingerprint(s, tools) == _hand_fingerprint(
        {"system": s, "tools": tools, "boundary": None}
    )
    assert request_fingerprint(s, tools, (None, None)) == request_fingerprint(s, tools)
    assert request_fingerprint(s, tools, (5, None)) == _hand_fingerprint(
        {"system": s, "tools": tools, "boundary": 5}
    )
    assert request_fingerprint(s, tools, (None, 7)) == _hand_fingerprint(
        {"system": s, "tools": tools, "boundary": None, "prune_before": 7}
    )
    # tools=None 与 [] 同形（旧版就是 `tools or []`）
    assert request_fingerprint(s, None) == _hand_fingerprint(
        {"system": s, "tools": [], "boundary": None}
    )


# ===========================================================================
# 13. 无状态：新 Agent 直接用检查点，不再总结
# ===========================================================================

def test_new_agent_reuses_checkpoint_without_summarizing(monkeypatch, db, echo_tool):
    llm = SummaryLLM(text="SUMMARY-1")
    agent, sid, fake, out = _scenario(monkeypatch, db, llm)
    n_first = len(llm.calls)
    assert n_first >= 1                              # 中段按 window/2 分块，次数不写死
    checkpoint_summary = db.latest_compaction(sid)[1]

    fake2 = FakeLLM([FakeMessage(content="第二个答案", usage=(10, 1))])
    monkeypatch.setattr("mini_agent.agent.call_llm", fake2)
    agent2 = _agent(db, llm)
    out2 = agent2.run_conversation(sid, "第二个问题")

    assert out2["completed"] is True
    assert len(llm.calls) == n_first, "新实例不该再总结一次"
    req = fake2.calls[0]["messages"]
    assert req[0]["role"] == "system"
    assert req[1]["role"] == "user"
    assert req[1]["content"].startswith(SUMMARY_PREFIX + checkpoint_summary)
    assert [m["role"] for m in req[2:]] == ["assistant", "tool", "assistant", "user"]
    assert req[-1]["content"] == "第二个问题"


# ===========================================================================
# 14. 总结失败：任务完成、无检查点、本次 run 内只尝试一次
# ===========================================================================

def test_summary_failure_completes_task_and_only_tries_once_per_run(monkeypatch, db, echo_tool):
    llm = SummaryLLM(raises=True)
    # 两轮大工具输出：第一轮失败、被裁剪之后，第二轮若还会触发阈值也**不应**再调总结
    agent, sid, fake, out = _scenario(
        monkeypatch, db, llm, tool_texts=(LONG_TEXT, "M" * BIG_CHARS)
    )

    assert out["completed"] is True
    assert out["final_response"] == FINAL_ANSWER
    assert _n_compactions(db) == 0
    assert len(llm.calls) == 1, "同一 run 内总结失败后不应重试"


# ===========================================================================
# 15. 总结前取消：TaskCancelled 原样抛出、无检查点（S1 回归）
# ===========================================================================

def test_cancel_before_summary_raises_task_cancelled(monkeypatch, db, echo_tool):
    llm = SummaryLLM(text="SUMMARY-1")
    seq = iter([False, False, False])           # 循环顶、工具前、下一循环顶 都放行

    def cancel_check() -> bool:
        return next(seq, True)                  # 之后（进压缩时）视为用户已取消

    with pytest.raises(TaskCancelled):
        _scenario(monkeypatch, db, llm, cancel=cancel_check)

    assert llm.calls == []
    assert _n_compactions(db) == 0


# ===========================================================================
# 16. D1：只裁旧的长 tool 输出；tool_call_id 不变；库里一字未改
# ===========================================================================

def test_failed_summary_prunes_only_old_long_tool_outputs(monkeypatch, db, echo_tool):
    """总结失败 -> 候选边界 B 之前、超过 200 字符的旧 tool 输出换成占位符。

    第一次请求前就会触发（旧长输出让整段超阈值）：B 落在旧任务收尾那条 assistant
    上，旧的长/短两条 tool 在中段，本轮的大工具输出在 B 之后（最新一轮）。
    """
    llm = SummaryLLM(raises=True)
    new_out = "N" * BIG_CHARS
    agent, sid, fake, out = _scenario(
        monkeypatch, db, llm, tool_texts=(new_out,), history="old_tool"
    )
    assert out["completed"] is True
    assert len(llm.calls) == 1

    last_req = fake.calls[-1]["messages"]
    tools_msg = [m for m in last_req if m["role"] == "tool"]
    assert [t["tool_call_id"] for t in tools_msg] == ["old0", "old1", "c0"]   # 配对不断
    long_len = len(f"echoed:{OLD_LONG}")
    assert tools_msg[0]["content"] == (
        f"[Old tool output ({long_len} chars) was cleared because "
        f"it is too old. Re-run the tool if you still need it.]"
    )
    assert tools_msg[1]["content"] == f"echoed:{OLD_SHORT}"       # 短：原样
    assert tools_msg[2]["content"] == f"echoed:{new_out}"         # seq >= B：原样

    # 库里一字未改
    assert [content for _seq, content in _tool_rows(db, sid)] == [
        f"echoed:{OLD_LONG}", f"echoed:{OLD_SHORT}", f"echoed:{new_out}",
    ]

    # prune_before 进了指纹：裁剪之后那次请求产出的 assistant，指纹是 (None, B)。
    # 只直接测 request_fingerprint 证明不了 Agent 真的把它传进去了。
    with db._read() as c:
        b = c.execute(
            "SELECT seq FROM messages WHERE session_id=? AND role='assistant' "
            "AND content LIKE '%旧任务完成%';", (sid,),
        ).fetchone()["seq"]
    tools = agent._tool_defs()
    produced = [m for m in db.recover(sid)
                if m.role == "assistant" and m.usage is not None]
    assert produced and all(
        m.usage.fingerprint == request_fingerprint(agent.system_prompt, tools, (None, b))
        for m in produced
    )


def test_failed_summary_never_prunes_the_newest_round(monkeypatch, db, echo_tool):
    """失败时尾部（seq >= B）里已经有一个大工具输出：它必须原样留着。

    单任务场景里第一次压缩发生在工具返回之后：B = 发起这次调用的 assistant，
    那个大 tool 结果在 B 之后、模型还没读过。上一条用例的失败发生在第一次请求
    之前，那时 B 之后还没有大输出，测不到"尾部不动"。
    """
    llm = SummaryLLM(raises=True)
    agent, sid, fake, out = _scenario(monkeypatch, db, llm)      # 聊天种子 + 一个大工具输出
    assert len(llm.calls) == 1 and out["completed"] is True

    after_failure = fake.calls[-1]["messages"]
    tool_contents = [m["content"] for m in after_failure if m["role"] == "tool"]
    assert tool_contents == [f"echoed:{LONG_TEXT}"]               # 原样，没被换成占位符


def test_prune_old_tool_outputs_is_pure_and_keeps_pairing():
    msgs = [
        Message(role="tool", tool_call_id="a", content="x" * 500),
        Message(role="tool", tool_call_id="b", content="y" * 50),
        Message(role="assistant", content="z" * 500),
    ]
    before = [m.model_dump_json() for m in msgs]
    out = prune_old_tool_outputs(msgs, [1, 2, 3], before_seq=2)

    assert [m.model_dump_json() for m in msgs] == before          # 入参不变
    assert out[0] is not msgs[0] and out[0].tool_call_id == "a"
    assert "(500 chars)" in out[0].content
    assert out[1] is msgs[1] and out[2] is msgs[2]


# ===========================================================================
# 17. D1 的缓存与指纹：裁剪后紧接着是 estimate；之后的前缀逐字节相同
# ===========================================================================

def test_pruned_view_invalidates_anchor_and_keeps_prefix_stable(monkeypatch, db, echo_tool):
    llm = SummaryLLM(raises=True)
    agent, sid, fake, out = _scenario(
        monkeypatch, db, llm,
        tool_texts=("M" * BIG_CHARS, "N" * BIG_CHARS), history="old_tool",
    )

    pruned_requests = [
        req["messages"] for req in fake.calls
        if any("was cleared" in (m.get("content") or "") for m in req["messages"])
    ]
    assert len(pruned_requests) >= 2, "需要至少两次裁剪后的请求才能比较前缀"

    def pruned_part(req):
        return [json.dumps(m, ensure_ascii=False, sort_keys=True) for m in req]

    # 替换点固定为 B：之后每次请求里被替换过的那一段逐字节相同
    assert pruned_part(pruned_requests[0])[:4] == pruned_part(pruned_requests[1])[:4]

    # 裁剪之后计数：旧锚点（描述裁剪前的长前缀）不可用 → estimate。
    # 视图状态 (None, B) 就是"裁剪了、但还没有检查点"，与 Agent 的行为一致。
    last = pruned_requests[-1]
    # 请求体是 OpenAI 形状（tool_calls 嵌在 function 里），要用 from_openai 还原
    msgs = [Message.from_openai(m) for m in last[1:]]
    turn = _turn(sid, msgs, [None] * len(msgs), None)
    turn.prune_before = 1                      # 不算 None 的视图状态
    count = agent._measure(turn, agent._tool_defs())
    assert count.source == SOURCE_ESTIMATE
    assert request_fingerprint(agent.system_prompt, agent._tool_defs(), (None, 1)) != \
        request_fingerprint(agent.system_prompt, agent._tool_defs())


def test_prune_before_is_part_of_the_fingerprint(monkeypatch, db, echo_tool):
    """S5/17：裁剪发生后锚点必须整批作废 —— prune_before 不进指纹时 usage 会被误用。"""
    llm = SummaryLLM(raises=True)
    agent, sid, fake, out = _scenario(monkeypatch, db, llm)
    with db._read() as c:
        tool_seq = c.execute(
            "SELECT seq FROM messages WHERE session_id=? AND role='tool';", (sid,)
        ).fetchone()["seq"]
    tools = agent._tool_defs()

    assert request_fingerprint(agent.system_prompt, tools, (None, tool_seq)) != \
        request_fingerprint(agent.system_prompt, tools, (None, None))


# ===========================================================================
# 18. D1 不落库：下一次 run 看到原文，并重新尝试总结
# ===========================================================================

def test_pruning_is_not_persisted_and_next_run_retries(monkeypatch, db, echo_tool):
    llm = SummaryLLM(raises=True)
    agent, sid, fake, out = _scenario(monkeypatch, db, llm, history="old_tool")
    assert len(llm.calls) == 1
    # 第一次 run 的请求里确实被替换过（否则下面的"没落库"就什么都没证明）
    assert any("was cleared" in (m.get("content") or "")
               for m in fake.calls[-1]["messages"])

    # 不落库：下一次 run 装载的视图里是完整原文
    ctx = db.load_context(sid)
    tool_contents = [m.content for m in ctx.messages if m.role == "tool"]
    assert f"echoed:{OLD_LONG}" in tool_contents
    assert all("was cleared" not in c for c in tool_contents)

    # 下一次 run 会重新尝试总结（失败标记只活在一次 run 里）
    fake2 = FakeLLM([FakeMessage(content="第二次答案", usage=(10, 1))])
    monkeypatch.setattr("mini_agent.agent.call_llm", fake2)
    _agent(db, llm).run_conversation(sid, "第二次")
    assert len(llm.calls) == 2


def test_successful_compaction_emits_compacting_then_compacted(monkeypatch, db, echo_tool):
    """事件成对出现：每个 compacting 后面紧跟一个 compacted；成功那次带 boundary。

    第一次请求前也超阈值，但那时没有可压的东西（noop）—— 它同样发一对事件
    （compacted 里 ok=false, reason=noop），Go 那边不会看到悬空的 compacting。
    """
    events: list[tuple[str, str]] = []
    llm = SummaryLLM(text="SUMMARY-1")
    _a, sid, fake, out = _scenario(
        monkeypatch, db, llm, on_event=lambda t, p: events.append((t, p))
    )

    pairs = [(t, p) for t, p in events if t in ("compacting", "compacted")]
    kinds = [t for t, _ in pairs]
    assert kinds and kinds == ["compacting", "compacted"] * (len(kinds) // 2)
    boundary, _ = db.latest_compaction(sid)
    assert json.loads(pairs[-1][1]) == {"ok": True, "boundary": boundary}
    # 成功之前的那些都是 noop，不是失败
    for t, p in pairs[:-1]:
        if t == "compacted":
            assert json.loads(p) == {"ok": False, "reason": "noop"}


# ===========================================================================
# 19. build_compression（S3）
# ===========================================================================

def test_build_compression_defaults_to_enabled(db):
    from mini_agent.config import settings
    assert settings.compression_enabled is True
    kwargs = build_compression(db)
    assert isinstance(kwargs["compressor"], LLMCompressor)
    assert isinstance(kwargs["policy"], CompressionPolicy)
    assert kwargs["policy"].context_window == settings.context_window
    Agent(db=db, **kwargs)                          # kwargs 可直接展开给 Agent


def test_build_compression_returns_empty_when_disabled(db, monkeypatch):
    """settings 是 frozen dataclass：monkeypatch 直接 setattr 会 FrozenInstanceError。

    换成"replace 出新对象、再把模块级 settings 整体换掉、由 monkeypatch 负责还原"。
    build_compression 在函数体内 `from mini_agent.config import settings`，每次调用
    都读模块属性，所以换掉模块属性就能生效。
    """
    import dataclasses
    import mini_agent.config as config_mod
    from mini_agent.config import settings

    replaced = dataclasses.replace(settings, compression_enabled=False)
    monkeypatch.setattr(config_mod, "settings", replaced)
    assert build_compression(db) == {}


@pytest.mark.parametrize("raw", ["0", "false", "OFF", "No", " off "])
def test_env_parses_false_values(monkeypatch, raw):
    from mini_agent.config import _bool_env
    monkeypatch.setenv("MINI_AGENT_COMPRESSION", raw)
    assert _bool_env("MINI_AGENT_COMPRESSION", True) is False


@pytest.mark.parametrize("raw", ["1", "true", "on", "yes", ""])
def test_env_parses_true_values(monkeypatch, raw):
    from mini_agent.config import _bool_env
    monkeypatch.setenv("MINI_AGENT_COMPRESSION", raw)
    assert _bool_env("MINI_AGENT_COMPRESSION", True) is True


def test_env_unset_uses_default(monkeypatch):
    from mini_agent.config import _bool_env
    monkeypatch.delenv("MINI_AGENT_COMPRESSION", raising=False)
    assert _bool_env("MINI_AGENT_COMPRESSION", True) is True


def test_noop_is_reported_as_noop_even_when_an_ignored_checkpoint_exists(db):
    """压缩器没写新检查点 → 必须报 noop，哪怕库里已经有一个（被降级忽略的）检查点。

    旧的判定是"latest is None 或 latest == turn.boundary"：检查点坏了被 load_context
    忽略时 turn.boundary 是 None、latest 却不是 None，于是一次 noop 被当成成功 ——
    重载一遍、给 Go 发 compacted{ok: true, boundary: null}。正确的判定是"压缩前后
    最新检查点有没有变"。
    """
    sid = db.create_session()
    db.append_message(sid, Message(role="user", content="q0"))
    s1 = db.append_message(sid, Message(role="assistant", content="a0"))
    assert db.append_compaction(sid, s1, "旧摘要")
    # 把边界行改成 role 一致的 system：load_context 会忽略这个检查点
    db.conn.execute(
        "UPDATE messages SET role='system', content=? WHERE session_id=? AND seq=?;",
        (Message(role="system", content="x").model_dump_json(), sid, s1),
    )
    ctx = db.load_context(sid)
    assert ctx.boundary is None                         # 前提：检查点确实被忽略了

    class NoopCompressor:
        def compress(self, turn):
            return None                                 # 什么都没写

    events: list[tuple[str, str]] = []
    agent = Agent(db=db, compressor=NoopCompressor(),
                  policy=CompressionPolicy(context_window=1, trigger_ratio=0.5))
    from mini_agent.agent import _Turn
    turn = _Turn(session_id=sid, messages=ctx.messages, seqs=ctx.seqs,
                 on_event=lambda t, p: events.append((t, p)))

    agent._maybe_compress(turn, None)

    compacted = [json.loads(p) for t, p in events if t == "compacted"]
    assert compacted == [{"ok": False, "reason": "noop"}]
    assert turn.boundary is None
