#!/usr/bin/env python
"""真机端到端验证：压缩链路 / 工具输出截断 / 无状态续跑（真实 DeepSeek key）。

这是一个**手动**验证脚本，不进 pytest：日常测试与 CI 不该意外联网、花钱。

用法（在仓库根目录）：
  python scripts/e2e_compaction.py --out-dir <临时目录> --dry-run   # 不联网：生成文件、校验估算与工具集
  python scripts/e2e_compaction.py --out-dir <临时目录>             # 真实运行 S1 / S2 / S3，写 report.md

场景（每个场景一个**独立子进程**）：
  S1  压缩链路：window=8000、cap=2500，链式读 5 个约 2.2k token 的文件，期望压缩 3~4 次
  S2  工具输出截断：cap=300，读一个约 3k token 的文件，落库内容 <= 300 token 且与幂等表一致
  S3  无状态续跑：S1 的同一 session，新进程 + 新连接再问一句，不应再触发总结

设计要点：
  * 子进程在 import mini_agent **之前**已拿到全部配置（env 由父进程显式构造），
    import 后、开库前断言 settings.home 在 --out-dir 之下，否则立即退出。
  * DEEPSEEK_API_KEY 由 config 照常从 .env 读；脚本不读 .env、不打印、不写报告。
    报告里的有效配置只含 model / window / ratio / cap / home。
  * 调用次数与 token 有硬上限（总计 40 次 / 15 万 token；分场景 S1<=14、S2<=4、S3<=3）。
  * 产物（库、日志、报告）只写 --out-dir。
  * 质量类观察（答对 / 答错）只记录，不算断言失败。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()

HARD_MAX_CALLS = 40
HARD_MAX_TOKENS = 150_000
SCENARIO_CALL_CAP = {"S1": 14, "S2": 4, "S3": 3}
# 场景 -> (context_window, 单次工具输出上限)。S3 必须与 S1 同配置（同一份库、同一把指纹）。
SCENARIO_CFG = {
    "S1": {"window": 8000, "cap": 2500},   # 8000
    "S2": {"window": 128_000, "cap": 300},
    "S3": {"window": 8000, "cap": 2500},
}
# 场景 -> 子目录（S3 复用 S1 的 home / workspace）
SCENARIO_DIR = {"S1": "s1", "S2": "s2", "S3": "s1"}

EXIT_HOME_OUTSIDE = 3
EXIT_NO_KEY = 4
EXIT_PREP_FAILED = 5

# 文件的验证暗号 / 中部校准编号（合成数据，无敏感信息）
FRONT_CODES = {1: "青岚-7314", 2: "云杉-2058", 3: "白榆-5601", 4: "灰雁-8830", 5: "墨竹-4417"}
MID_CODES = {1: "赤松-9482", 2: "朱雀-3391", 3: "苍梧-7720", 4: "玄鹤-6154", 5: "碧潭-2286"}

_SENTENCES = [
    "天枢站的仓库按月轮换盘点，每次盘点都会留下一份签字记录。",
    "三号货架靠近西侧通道，存放的主要是包装材料与备用的封条。",
    "夜班值守人员需要在凌晨两点前完成温湿度读数，并写入登记簿。",
    "运输车辆进站前必须先在门岗登记，核对车牌与随车单据是否一致。",
    "冷藏区的门禁每季度检修一次，检修期间由临时通道代替使用。",
    "过期物料统一集中到东侧隔离区，等待每月第二个周四的统一处理。",
    "站内的照明系统分为四个回路，任何一个回路故障都不会影响其余部分。",
    "新入职的员工要先完成为期三天的安全培训，再由组长带着熟悉流程。",
    "备用发电机每半年空载运行一小时，运行记录同样需要主管签字。",
    "客户退回的货物先登记原因，再决定是重新上架还是转入报废流程。",
    "月度报表由财务与仓储两边各自核对一遍，差异超过百分之二要复查。",
    "消防器材的有效期贴在瓶身侧面，过期前一个月由安全员统一更换。",
    "地面划线每年重新喷涂一次，叉车通行区域与行人通道用不同颜色区分。",
    "出库单一式三联，白联留存、黄联随货、红联交回客户确认签收。",
    "遇到极端天气时，露天堆放的货物要在两小时内转移到棚内存放。",
    "设备维修申请先提交给组长审核，再由行政部门统一安排外部服务商。",
]


# ---------------------------------------------------------------------------
# 文件生成（纯函数，只依赖 random.Random(seed)；中文为主，估算与真实 token 接近）
# ---------------------------------------------------------------------------

def _filler(rng: random.Random, n_chars: int) -> str:
    parts: list[str] = []
    total = 0
    while total < n_chars:
        s = f"第{rng.randint(10, 999)}号记录：{rng.choice(_SENTENCES)}"
        if rng.random() < 0.3:
            s += "\n"
        parts.append(s)
        total += len(s)
    return "".join(parts)[:n_chars]


def make_chain_file(k: int, filler_chars: int, last: bool) -> str:
    """链式文件 f{k}：前部关键事实（<1000 字符内）、中部对照事实、末尾指向下一个文件。"""
    rng = random.Random(1000 + k)
    head = f"# 文件 {k}（f{k}.txt）\n本文件是天枢站档案的第 {k} 卷。\n"
    a = _filler(rng, 250)
    front = f"\n关键事实：第 {k} 卷的验证暗号是「{FRONT_CODES[k]}」。\n"
    rest = filler_chars - 250
    b1 = _filler(rng, rest // 2)
    mid = f"\n对照事实：第 {k} 卷中部记录的校准编号是「{MID_CODES[k]}」。\n"
    b2 = _filler(rng, rest - rest // 2)
    tail = "\n本文件到此结束，这是最后一个文件。\n" if last else f"\n下一个文件是 f{k + 1}.txt。\n"
    return head + a + front + b1 + mid + b2 + tail


def make_big_file(filler_chars: int) -> str:
    """S2 用：头 / 中 / 尾各一条事实。"""
    rng = random.Random(2000)
    first = "暗号甲：青岚-7314。\n"
    half = filler_chars // 2
    mid = "\n暗号乙：赤松-9482。\n"
    last = "\n暗号丙：白榆-5601。"
    return first + _filler(rng, half) + mid + _filler(rng, filler_chars - half) + last


# ---------------------------------------------------------------------------
# 调用记录 + 硬上限
# ---------------------------------------------------------------------------

class Abort(BaseException):
    """超过调用 / token 上限。继承 BaseException：agent 里的 `except Exception`
    （压缩失败降级、on_event 吞异常）不能把"该停了"吃掉。"""


class Rec:
    def __init__(self, scenario: str, max_calls: int, max_tokens: int):
        self.scenario = scenario
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.calls: list[dict] = []
        self.events: list[dict] = []
        self.total_tokens = 0
        self.last_raw: dict | None = None
        self.create_count = 0
        self.keep_api1 = False

    def on_event(self, etype: str, payload: str) -> None:
        text = payload if etype != "tool_result" else payload[:120]
        self.events.append({"n": len(self.calls), "type": etype, "payload": text[:400]})


def install_recorder(rec: Rec) -> None:
    from mini_agent import agent as agent_mod
    from mini_agent import llm

    orig_call = llm.call_llm
    client = llm._get_client()
    orig_create = client.chat.completions.create

    def create(*a, **kw):
        resp = orig_create(*a, **kw)
        u = getattr(resp, "usage", None)
        rec.create_count += 1
        rec.last_raw = {
            "model": getattr(resp, "model", None),
            "finish_reason": resp.choices[0].finish_reason if resp.choices else None,
            "cache_hit": getattr(u, "prompt_cache_hit_tokens", None),
        }
        return resp

    client.chat.completions.create = create

    def wrapped(messages, tools=None, *, max_tokens=None):
        if len(rec.calls) >= rec.max_calls:
            raise Abort(f"{rec.scenario}: 调用次数达到上限 {rec.max_calls}")
        if rec.total_tokens >= rec.max_tokens:
            raise Abort(f"{rec.scenario}: token 达到上限 {rec.max_tokens}")
        kind = "summary" if max_tokens is not None else "main"
        rec.last_raw = None
        entry: dict = {
            "i": len(rec.calls) + 1,
            "kind": kind,
            "n_msgs": len(messages),
            "n_tool_msgs": sum(1 for m in messages if m.get("role") == "tool"),
            "has_tools": bool(tools),
        }
        if rec.keep_api1 and kind == "main" and len(messages) > 1:
            m1 = messages[1]
            entry["api1_role"] = m1.get("role")
            entry["api1_head"] = (m1.get("content") or "")[:160]
        rec.calls.append(entry)
        t0 = time.time()
        try:
            reply = orig_call(messages, tools=tools, max_tokens=max_tokens)
        except Exception as e:
            entry["error"] = type(e).__name__
            entry["secs"] = round(time.time() - t0, 2)
            raise
        entry["secs"] = round(time.time() - t0, 2)
        if reply.usage is not None:
            entry["prompt_tokens"] = reply.usage.prompt_tokens
            entry["completion_tokens"] = reply.usage.completion_tokens
            rec.total_tokens += reply.usage.prompt_tokens + reply.usage.completion_tokens
        raw = rec.last_raw or {}
        entry["model"] = raw.get("model")
        entry["finish"] = raw.get("finish_reason")
        entry["cache_hit"] = raw.get("cache_hit")
        return reply

    llm.call_llm = wrapped
    agent_mod.call_llm = wrapped


# ---------------------------------------------------------------------------
# 子进程：单个场景
# ---------------------------------------------------------------------------

def _check(results: list, name: str, ok: bool, detail: str = "") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def _tool_names(agent) -> set[str]:
    return {d["function"]["name"] for d in (agent._tool_defs() or [])}


def _measure_tool_msg(read_file, name: str) -> tuple[int, str]:
    from mini_agent.context.tokens import estimate_message
    from mini_agent.schema import Message

    result = read_file(name)
    return estimate_message(Message(role="tool", tool_call_id="x", content=result)), result


def _calibrate(ws: Path, name: str, builder, lo: int, hi: int, target: int, read_file) -> int:
    """调文件长度，直到 read_file 返回的 tool 消息估算 T 落入 [lo, hi]。失败抛 RuntimeError。"""
    chars = 3400
    for _ in range(14):
        (ws / name).write_bytes(builder(chars).encode("utf-8"))
        t, result = _measure_tool_msg(read_file, name)
        if not result.startswith('{"path"'):
            raise RuntimeError(f"read_file({name}) 没有返回正常结果: {result[:80]}")
        if lo <= t <= hi:
            return t
        chars += int((target - t) / 0.55)
    raise RuntimeError(f"{name} 无法校准到 [{lo}, {hi}]，最后 T={t}")


def child_main(args) -> int:
    out_dir = Path(args.out_dir).resolve()
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    from mini_agent.config import DEFAULT_TOOLSET, settings  # noqa: F401

    # ★ 开库、写文件之前先确认没指到别处（比如 env 没生效退回了 ./.mini_agent）
    home = settings.home.resolve()
    if not home.is_relative_to(out_dir):
        print(f"[{args.scenario}] settings.home 不在 --out-dir 之下，拒绝运行", file=sys.stderr)
        return EXIT_HOME_OUTSIDE
    if not args.dry_run and not settings.deepseek_api_key:
        print("缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 key。", file=sys.stderr)
        return EXIT_NO_KEY

    import logging

    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    from mini_agent.agent import Agent
    from mini_agent.context.compressor import SUMMARY_PREFIX, VERBATIM_HEADER, build_compression
    from mini_agent.context.tokens import estimate_message, estimate_request, estimate_text
    from mini_agent.persistence.session_db import MiniSessionDB
    from mini_agent.tools.builtin.file_tools import read_file

    scenario = args.scenario
    results: list[dict] = []
    findings: list[str] = []
    out: dict = {
        "scenario": scenario,
        "config": {
            "model": settings.model,
            "window": settings.context_window,
            "ratio": settings.compress_trigger_ratio,
            "cap": settings.tool_output_max_tokens,
            "compression": settings.compression_enabled,
            "max_turns": settings.max_turns,
            "home": str(home),
        },
        "assertions": results,
        "findings": findings,
    }
    result_path = out_dir / "run" / f"{scenario}_result.json"

    def finish(code: int = 0) -> int:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        return code

    ws = Path(os.environ["MINI_AGENT_WORKSPACE"]).resolve()
    if not ws.is_relative_to(out_dir):
        print(f"[{scenario}] 工作区不在 --out-dir 之下，拒绝运行", file=sys.stderr)
        return EXIT_HOME_OUTSIDE

    # ---- 工具集：显式 {"file"}，且恰为三个文件工具 ----
    probe = Agent(db=None, toolset={"file"})  # type: ignore[arg-type]  # 只为取工具定义
    names = _tool_names(probe)
    _check(results, "toolset 恰为 {list_dir, read_file, write_file}",
           names == {"list_dir", "read_file", "write_file"}, f"实际: {sorted(names)}")

    # ---- 准备文件（联网之前校验估算）----
    try:
        if scenario in ("S1", "S3"):
            ws.mkdir(parents=True, exist_ok=True)
            if scenario == "S1":
                ts = []
                for k in range(1, 6):
                    ts.append(_calibrate(
                        ws, f"f{k}.txt",
                        lambda n, k=k: make_chain_file(k, n, last=(k == 5)),
                        2100, 2400, 2250, read_file))
                out["file_tokens"] = ts
                _check(results, "S1 每个文件估算 T 落在 [2100, 2400]",
                       all(2100 <= t <= 2400 for t in ts), f"T={ts}")
        elif scenario == "S2":
            ws.mkdir(parents=True, exist_ok=True)
            t = _calibrate(ws, "big.txt", make_big_file, 2700, 3500, 3000, read_file)
            out["file_tokens"] = [t]
            _check(results, "S2 文件估算 T 落在 [2700, 3500]", True, f"T={t}")
    except Exception as e:  # 校准失败 = 联网前中止
        print(f"[{scenario}] 准备文件失败: {e}", file=sys.stderr)
        _check(results, "文件准备 / 估算校验", False, str(e))
        finish()
        return EXIT_PREP_FAILED

    if args.dry_run:
        out["dry_run"] = True
        return finish()

    # ---- 真实运行 ----
    rec = Rec(scenario, args.max_calls, args.max_tokens)
    install_recorder(rec)

    db = MiniSessionDB(settings.db_path())
    comp = build_compression(db)
    agent = Agent(
        db=db,
        toolset={"file"},
        max_turns=settings.max_turns,
        max_tool_output_tokens=settings.tool_output_max_tokens,
        **comp,
    )
    _check(results, "压缩已装配（compressor 与 policy 非空）",
           agent.compressor is not None and agent.policy is not None)
    _check(results, "Agent 工具定义恰为三个文件工具",
           _tool_names(agent) == {"list_dir", "read_file", "write_file"})

    if scenario == "S1":
        sid = db.create_session()
        user_msg = (
            "请从 f1.txt 开始读。每个文件末尾都会告诉你下一个文件是哪一个，请一直读到最后一个文件为止"
            "（每次只读一个文件，不要猜文件名）。全部读完后回答两个问题，只依据文件内容作答：\n"
            "1) f1.txt 里记录的验证暗号是什么？\n"
            "2) f1.txt 中部记录的校准编号是什么？"
        )
    elif scenario == "S2":
        sid = db.create_session()
        user_msg = (
            "请读取 big.txt，并告诉我：暗号甲、暗号乙、暗号丙分别是什么。"
            "如果文件内容不完整或你没能看到某一部分，请明确说明。"
        )
    else:  # S3
        sid = args.session
        if not sid or not db.session_exists(sid):
            _check(results, "S3 能找到 S1 的 session", False, f"session={sid}")
            return finish()
        user_msg = "不要调用任何工具，直接根据你已有的上下文回答：f1.txt 里记录的验证暗号是什么？"
        rec.keep_api1 = True
    out["session_id"] = sid

    def comp_rows():
        with db._read() as c:
            return [dict(r) for r in c.execute(
                "SELECT first_kept_seq, summary, created_at FROM compactions "
                "WHERE session_id = ? ORDER BY first_kept_seq;", (sid,)).fetchall()]

    before_rows = comp_rows() if scenario == "S3" else []
    error = None
    resp: dict = {}
    try:
        resp = agent.run_conversation(sid, user_msg, on_event=rec.on_event)
    except Abort as e:
        error = f"Abort: {e}"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-1500:]
    finally:
        try:
            db.end_session(sid)
        except Exception:
            pass

    out["run_error"] = error
    out["response"] = resp
    out["final_answer"] = resp.get("final_response", "")
    out["calls"] = rec.calls
    out["events"] = rec.events
    out["total_calls"] = len(rec.calls)
    out["total_tokens"] = rec.total_tokens
    models = sorted({c.get("model") for c in rec.calls if c.get("model")})
    out["response_models"] = models
    if error:
        findings.append(f"运行异常：{error}")
    if rec.create_count != len(rec.calls) - sum(1 for c in rec.calls if c.get("error")):
        findings.append(
            f"create 级记录数({rec.create_count})与成功调用数不一致，model 字段可能不完整")
    if models and models != [settings.model]:
        findings.append(f"响应 model 字段 {models} 与配置的 {settings.model!r} 不同（记录，不算失败）")

    rows, seqs = db._load_rows(sid, from_seq=0)
    out["transcript_len"] = len(rows)
    ckpts = comp_rows()
    out["compactions"] = [
        {"first_kept_seq": r["first_kept_seq"], "summary_len": len(r["summary"]),
         "summary_head": r["summary"][:500]} for r in ckpts]
    with db._read() as c:
        exe = [dict(r) for r in c.execute(
            "SELECT assistant_seq, tool_index, tool_call_id, result FROM executed_keys "
            "WHERE session_id = ? ORDER BY assistant_seq, tool_index;", (sid,)).fetchall()]

    main_calls = [c for c in rec.calls if c["kind"] == "main"]
    summary_calls = [c for c in rec.calls if c["kind"] == "summary"]
    out["main_prompt_tokens"] = [c.get("prompt_tokens") for c in main_calls]
    ok_events = [e for e in rec.events if e["type"] == "compacted" and '"ok": true' in e["payload"]]
    out["compaction_ok_count"] = len(ok_events)

    # ======== 各场景的断言 ========
    # ★ 断言分两类。产品不变量（下面的 _check）不成立就是 bug；模型行为相关的预测
    #   （每轮只调一个工具、压缩恰好几次、prompt 有固定上界）只记进 findings ——
    #   真机上模型会主动重读文件、偶尔一轮调两个工具，这些不是产品缺陷，把它们当
    #   断言只会制造假 FAIL（首次真机运行时就出现过 3 条）。
    if scenario == "S1":
        asst = [m for m in rows if m.role == "assistant"]
        multi = [i for i, m in enumerate(asst) if m.tool_calls and len(m.tool_calls) != 1]
        reads = [tc.arguments.get("path") for m in asst for tc in (m.tool_calls or [])]
        out["read_order"] = reads
        if multi:
            findings.append(f"观察：assistant #{multi} 一轮调用了多个工具（该轮是最新一轮时不可压缩，"
                            f"属已知限制：单轮多工具的总量不设上限）")
        rereads = len(reads) - len(set(reads))
        if rereads:
            findings.append(f"观察：模型重读了 {rereads} 次文件（读取顺序 {reads}）—— 摘要有损时"
                            f"模型靠重新调用工具找回细节")
        _check(results, "运行完成（completed=True，无异常）",
               error is None and bool(resp.get("completed")), str(error or resp.get("used_turns")))
        _check(results, "压缩在真机上至少成功执行 1 次", len(ok_events) >= 1, f"实际 {len(ok_events)}")
        if not 3 <= len(ok_events) <= 4:
            findings.append(f"观察：压缩成功 {len(ok_events)} 次，推导预期 [3, 4]（模型多读/少读几轮时会偏离）")
        _check(results, "compactions 表行数 == 压缩成功次数", len(ckpts) == len(ok_events),
               f"表 {len(ckpts)} 行 / 事件 {len(ok_events)} 次")
        _check(results, "每份 summary 以逐字原话章节开头",
               bool(ckpts) and all(r["summary"].startswith(VERBATIM_HEADER) for r in ckpts))
        n_trunc = sum(1 for m in rows if m.role == "tool" and "运行时截断" in (m.content or ""))
        _check(results, "cap=2500 下没有工具输出被截断（文件 T<=2400）", n_trunc == 0, f"被截断 {n_trunc} 条")

        # "变短"：每次成功压缩后的下一次主请求，prompt_tokens 小于 compacting 事件里的计数
        shorter: list[dict] = []
        evs = rec.events
        for idx, e in enumerate(evs):
            if e["type"] != "compacted" or '"ok": true' not in e["payload"]:
                continue
            before = next((json.loads(x["payload"]) for x in reversed(evs[:idx]) if x["type"] == "compacting"), None)
            nxt = next((c for c in main_calls if c["i"] > e["n"]), None)
            shorter.append({
                "count_before": before["tokens"] if before else None,
                "source": before["source"] if before else None,
                "next_main_prompt": nxt.get("prompt_tokens") if nxt else None,
            })
        out["shorter_check"] = shorter
        _check(results, "压缩后下一次主请求 prompt_tokens < compacting 事件里的计数",
               bool(shorter) and all(s["count_before"] and s["next_main_prompt"]
                                     and s["next_main_prompt"] < s["count_before"] for s in shorter),
               json.dumps(shorter, ensure_ascii=False))
        late = [c.get("prompt_tokens") for n, c in enumerate(main_calls, 1) if n >= 3]
        if not (late and all(p is not None and p <= 4500 for p in late)):
            findings.append(f"观察：call3 起 prompt_tokens 有超过推导上界 4500 的：{late}"
                            f"（上界按摘要约 1k token 推导；摘要偏长、或某一轮有多个工具输出"
                            f"而该轮不可压缩时都会超出 —— 硬性判据见上面的 70% 反事实断言）")

        # 反事实：不压缩时各次主请求的估算
        tools = agent._tool_defs()
        asst_pos = [i for i, m in enumerate(rows) if m.role == "assistant"]
        out["counterfactual_estimate"] = [
            estimate_request(rows[:p], agent.system_prompt, tools) for p in asst_pos]
        # 不变量：压缩确实把上下文压住了 —— 最后一次主请求明显小于不压缩时的估算
        cf_last = out["counterfactual_estimate"][len(main_calls) - 1] if main_calls else None
        last_prompt = main_calls[-1].get("prompt_tokens") if main_calls else None
        _check(results, "最后一次主请求 prompt_tokens < 不压缩时估算的 70%",
               bool(cf_last and last_prompt) and last_prompt < 0.7 * cf_last,
               f"实际 {last_prompt} / 反事实 {cf_last}")

        ans = out["final_answer"]
        out["quality"] = {
            "front_code_7314_in_answer": "7314" in ans,
            "mid_code_9482_in_answer": "9482" in ans,
        }

    elif scenario == "S2":
        # 按工具名找 read_file 的那条结果：模型可能先 list_dir 看一眼（首次真机运行就这样），
        # 拿"第一条 tool 消息"去查截断标记会查错对象。
        read_ids = {tc.id for m in rows if m.role == "assistant"
                    for tc in (m.tool_calls or []) if tc.name == "read_file"}
        read_rows = [m for m in rows if m.role == "tool" and m.tool_call_id in read_ids]
        _check(results, "S2 至少有 1 次 read_file 结果", bool(read_rows), f"实际 {len(read_rows)}")
        if read_rows:
            content = read_rows[0].content or ""
            t = estimate_text(content)
            out["tool_content_tokens"] = t
            _check(results, "estimate_text(read_file 结果) <= 300", t <= 300, f"T={t}")
            _check(results, "read_file 结果含截断标记", "运行时截断" in content)
        # 幂等表与 transcript：每一次工具调用都要逐字节相等（按 (assistant_seq, idx) 顺序配对）
        tool_contents = [m.content or "" for m in rows if m.role == "tool"]
        cached_contents = []
        for r in exe:
            try:
                cached_contents.append(json.loads(r["result"]).get("content"))
            except Exception:
                cached_contents.append(None)
        _check(results, "executed_keys 与 transcript 的每条 tool 结果逐字节相等",
               len(cached_contents) == len(tool_contents) and all(
                   isinstance(a, str) and a.encode("utf-8") == b.encode("utf-8")
                   for a, b in zip(cached_contents, tool_contents)),
               f"幂等表 {len(cached_contents)} 行 / transcript {len(tool_contents)} 条")
        _check(results, "运行完成（completed=True，无异常）",
               error is None and bool(resp.get("completed")), str(error))
        ans = out["final_answer"]
        out["quality"] = {
            "head_7314": "7314" in ans, "mid_9482": "9482" in ans, "tail_5601": "5601" in ans,
            "mentions_truncation": any(w in ans for w in ("截断", "不完整", "省略", "未能", "无法看到")),
        }

    else:  # S3
        out["compactions_before"] = len(before_rows)
        _check(results, "运行完成（无异常）", error is None, str(error))
        _check(results, "compactions 行数不变", len(ckpts) == len(before_rows),
               f"前 {len(before_rows)} 后 {len(ckpts)}")
        _check(results, "总结调用计数为 0（不再触发总结）", len(summary_calls) == 0,
               f"总结调用 {len(summary_calls)} 次")
        first = main_calls[0] if main_calls else {}
        _check(results, "首个主请求 api_msgs[1] 是 user 且以 SUMMARY_PREFIX 开头",
               first.get("api1_role") == "user"
               and (first.get("api1_head") or "").startswith(SUMMARY_PREFIX),
               f"role={first.get('api1_role')} head={(first.get('api1_head') or '')[:60]!r}")
        out["quality"] = {"front_code_7314_in_answer": "7314" in out["final_answer"],
                          "tool_calls_made": sum(1 for m in rows if m.role == "assistant" and m.tool_calls)}
    return finish()


# ---------------------------------------------------------------------------
# 父进程：编排 + 报告
# ---------------------------------------------------------------------------

def scenario_env(name: str, out_dir: Path) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items()
            if not k.startswith("MINI_AGENT_") and k not in ("ALLOW_WEB", "OPENAI_LOG")}
    d = out_dir / "run" / SCENARIO_DIR[name]
    cfg = SCENARIO_CFG[name]
    base.update({
        "MINI_AGENT_HOME": str(d / "home"),
        "MINI_AGENT_WORKSPACE": str(d / "ws"),
        "MINI_AGENT_CONTEXT_WINDOW": str(cfg["window"]),
        "MINI_AGENT_COMPRESS_TRIGGER_RATIO": "0.5",
        "MINI_AGENT_COMPRESSION": "1",
        "MINI_AGENT_TOOL_OUTPUT_MAX_TOKENS": str(cfg["cap"]),
        "MINI_AGENT_MAX_TURNS": "12",
        "MINI_AGENT_LOG_CONSOLE": "0",
        "ALLOW_WEB": "",
        "OPENAI_LOG": "",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    return base


def run_child(name: str, out_dir: Path, *, dry: bool, max_calls: int, max_tokens: int,
              session: str = "") -> tuple[int, dict | None]:
    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SELF), "--scenario", name, "--out-dir", str(out_dir),
           "--max-calls", str(max_calls), "--max-tokens", str(max_tokens)]
    if session:
        cmd += ["--session", session]
    if dry:
        cmd.append("--dry-run")
    log = run_dir / f"{name}{'_dry' if dry else ''}.log"
    with open(log, "wb") as f:
        p = subprocess.run(cmd, env=scenario_env(name, out_dir), cwd=str(REPO),
                           stdout=f, stderr=subprocess.STDOUT, timeout=900)
    rp = run_dir / f"{name}_result.json"
    data = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    if p.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-600:]
        print(f"[{name}] 子进程退出码 {p.returncode}\n{tail}", file=sys.stderr)
    return p.returncode, data


def render_report(out_dir: Path, results: dict[str, dict | None], totals: dict) -> str:
    L: list[str] = ["# e2e 报告：压缩链路 / 工具输出截断 / 无状态续跑", ""]
    L.append(f"- 总调用 {totals['calls']} 次，总 token {totals['tokens']}（上限 {HARD_MAX_CALLS} 次 / {HARD_MAX_TOKENS}）")
    L.append("- key present: True（不记录其值）")
    L.append("")
    failed_all: list[str] = []
    for name in ("S1", "S2", "S3"):
        d = results.get(name)
        L.append(f"## {name}")
        if not d:
            L += ["（未运行或无结果）", ""]
            continue
        c = d["config"]
        L.append(f"- 有效配置：model={c['model']} window={c['window']} ratio={c['ratio']} "
                 f"cap={c['cap']} home={c['home']}")
        L.append(f"- session={d.get('session_id')} 响应 model 字段={d.get('response_models')}")
        L.append(f"- 调用 {d.get('total_calls')} 次，token {d.get('total_tokens')}，"
                 f"压缩成功 {d.get('compaction_ok_count')} 次，used_turns={d.get('response', {}).get('used_turns')}")
        if d.get("file_tokens"):
            L.append(f"- 文件估算 T：{d['file_tokens']}")
        if d.get("run_error"):
            L.append(f"- 运行异常：{d['run_error']}")
        L += ["", "### 事件序列", ""]
        for e in d.get("events", []):
            if e["type"] in ("tool_call", "compacting", "compacted", "thinking", "tool_result"):
                L.append(f"- [after {e['n']} calls] {e['type']} {e['payload'][:200]}")
        L += ["", "### 调用表", "",
              "| # | kind | msgs | tool_msgs | tools | prompt | completion | cache_hit | secs | model | finish |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k in d.get("calls", []):
            L.append(f"| {k['i']} | {k['kind']} | {k['n_msgs']} | {k['n_tool_msgs']} | {k['has_tools']} | "
                     f"{k.get('prompt_tokens')} | {k.get('completion_tokens')} | {k.get('cache_hit')} | "
                     f"{k.get('secs')} | {k.get('model')} | {k.get('finish') or k.get('error')} |")
        if d.get("main_prompt_tokens") is not None:
            L += ["", f"主请求 prompt_tokens 序列：{d['main_prompt_tokens']}"]
        if d.get("counterfactual_estimate"):
            L.append(f"反事实（不压缩的估算）序列：{d['counterfactual_estimate']}")
        if d.get("shorter_check"):
            L.append(f"压缩前计数 vs 下一次主请求 prompt：{json.dumps(d['shorter_check'], ensure_ascii=False)}")
        if d.get("compactions"):
            L += ["", "### 检查点（summary 前 500 字）", ""]
            for r in d["compactions"]:
                L += [f"- first_kept_seq={r['first_kept_seq']} 全长 {r['summary_len']} 字符", "  ```",
                      *("  " + ln for ln in r["summary_head"].splitlines()), "  ```"]
        L += ["", "### 最终答案", "", "> " + (d.get("final_answer") or "").replace("\n", "\n> "), ""]
        if d.get("quality"):
            L.append(f"质量观察（只记录）：{json.dumps(d['quality'], ensure_ascii=False)}")
            L.append("")
        L += ["### 断言", ""]
        for a in d.get("assertions", []):
            mark = "PASS" if a["ok"] else "FAIL"
            L.append(f"- {mark} {a['name']}" + (f" —— {a['detail']}" if a.get("detail") else ""))
            if not a["ok"]:
                failed_all.append(f"{name}: {a['name']}")
        L.append("")
        if d.get("findings"):
            L += ["### 发现的问题", ""] + [f"- {x}" for x in d["findings"]] + [""]
        if d.get("traceback"):
            L += ["### traceback（尾部）", "", "```", d["traceback"], "```", ""]
    L += ["## 汇总", "", f"- 断言失败 {len(failed_all)} 条"] + [f"  - {x}" for x in failed_all]
    return "\n".join(L) + "\n"


def parent_main(args) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out_dir = Path(args.out_dir).resolve()
    run_dir = out_dir / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)       # 只删 --out-dir 下自己的 run/ 子目录
    run_dir.mkdir(parents=True)

    # 阶段 1：不联网。生成文件 + 估算校验 + 工具集断言（每个场景一个子进程）
    print("== 阶段 1：dry-run（不联网）==")
    dry_results = {}
    for name in ("S1", "S2"):
        code, data = run_child(name, out_dir, dry=True, max_calls=0, max_tokens=0)
        dry_results[name] = data
        bad = [a for a in (data or {}).get("assertions", []) if not a["ok"]]
        print(f"  {name}: exit={code} 断言 {len(data['assertions']) if data else '?'} 条, 失败 {len(bad)}"
              + (f" 文件 T={data.get('file_tokens')}" if data and data.get("file_tokens") else ""))
        if code != 0 or bad or data is None:
            print(f"{name} 联网前校验未通过，中止（未调用任何 LLM）", file=sys.stderr)
            return 1
    if args.dry_run:
        print("dry-run 通过。")
        return 0

    # 阶段 2：真实运行
    print("== 阶段 2：真实运行 ==")
    totals = {"calls": 0, "tokens": 0}
    results: dict[str, dict | None] = {}
    s1_sid = ""
    for name in ("S1", "S2", "S3"):
        if name == "S3" and not s1_sid:
            print("S3 跳过：S1 没有产生 session")
            results[name] = None
            continue
        left_calls = HARD_MAX_CALLS - totals["calls"]
        left_tokens = HARD_MAX_TOKENS - totals["tokens"]
        if left_calls <= 0 or left_tokens <= 0:
            print(f"{name} 跳过：总上限已用完", file=sys.stderr)
            continue
        cap_calls = min(SCENARIO_CALL_CAP[name], left_calls)
        code, data = run_child(name, out_dir, dry=False, max_calls=cap_calls,
                               max_tokens=left_tokens, session=s1_sid if name == "S3" else "")
        if code == EXIT_NO_KEY:
            print("缺少 DEEPSEEK_API_KEY：请复制 .env.example 为 .env 并填入 key（脚本不会读取或打印它）。",
                  file=sys.stderr)
            return EXIT_NO_KEY
        results[name] = data
        if data:
            totals["calls"] += data.get("total_calls", 0)
            totals["tokens"] += data.get("total_tokens", 0)
            if name == "S1":
                s1_sid = data.get("session_id", "")
        fails = [a for a in (data or {}).get("assertions", []) if not a["ok"]]
        print(f"  {name}: exit={code} 调用 {data.get('total_calls') if data else '?'} 次, "
              f"断言失败 {len(fails)} 条")

    report = render_report(out_dir, results, totals)
    rp = out_dir / "report.md"
    rp.write_text(report, encoding="utf-8")
    print(f"总调用 {totals['calls']} 次，总 token {totals['tokens']}")
    print(f"报告：{rp}")
    any_fail = any(not a["ok"] for d in results.values() if d for a in d.get("assertions", []))
    return 1 if any_fail else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="所有产物（库 / 日志 / 报告）写到这里")
    ap.add_argument("--dry-run", action="store_true", help="不联网：只生成文件并做估算 / 工具集校验")
    ap.add_argument("--scenario", choices=["S1", "S2", "S3"], help="（内部）子进程只跑这一个场景")
    ap.add_argument("--max-calls", type=int, default=0, help="（内部）子进程调用次数上限")
    ap.add_argument("--max-tokens", type=int, default=0, help="（内部）子进程 token 上限")
    ap.add_argument("--session", default="", help="（内部）S3 要续跑的 session")
    args = ap.parse_args()
    if args.scenario:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
