"""上下文占用的可信计数：usage 锚点 + 增量估算。

[M3] 回答"这次请求大概要花多少 context"，供 CompressionPolicy 判断要不要压缩。

为什么不能直接用 usage：
  1. 它是**滞后**的。 usage.prompt_tokens 描述的是**上一次**请求，从上一次到
     这一次之间新增的消息（典型：一次 web_fetch 返回几千字）它从没见过。
     只看 usage，"上次 60% → 这次塞进一个大工具结果 → 直接超限 400"必然漏触发。
  2. Agent 是**无状态**的。 每次 run_conversation 都从库里 recover 历史，内存里
     攒的 usage 活不过一次调用，所以 usage 必须随产生它的那条 assistant 落库
     （见 agent._call_model / schema.TokenUsage）。

所以算法是：**usage 当锚点，锚点之后的消息用估算补上**；没有可用锚点时整条请求
都估算。两条路都**偏向高估**，最坏结果是提前压缩，而不是漏触发。

为什么"锚点"需要 fingerprint：
  prompt_tokens 只在"这次请求的前缀和它当时一样"时才可信。前缀由 system prompt、
  tool 定义、当前生效的视图状态（压缩边界 / 降级裁剪点）决定 —— 任何一项变了，锚点的数字就不再描述
  眼前这条请求。把前缀的指纹记在 usage 上，一变就整批作废、退回估算，
  不需要在压缩发生时去逐条清理旧锚点。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from mini_agent.schema import Message, TokenUsage

# 估算系数（DeepSeek 官方经验值，按**字符**计）。中日韩文字切得密，约 0.6
# token/字符；其余文字（英文为主）实际约 0.25，取 0.3 本身就偏保守。不再额外乘
# 安全系数：已知偏差（推理模型的 completion_tokens 含不回流的内容、recover 跳过
# 的坏行会让前缀变短）方向都一致 —— 只会估高、提前压缩，不会估低、漏触发。
_CJK_TOKENS_PER_CHAR = 0.6
_OTHER_TOKENS_PER_CHAR = 0.3

# 每条消息的聊天模板结构开销（<|im_start|>role ... <|im_end|> 之类）
_PER_MESSAGE_OVERHEAD = 4

# 计数来源，只给日志/测试看，不参与决策
SOURCE_USAGE = "usage"
SOURCE_ESTIMATE = "estimate"


@dataclass(frozen=True)
class TokenCount:
    """一次计数的结果。source 说明这个数是"锚点 + 增量"还是"整条估算"。"""

    tokens: int
    source: str


def _is_cjk(ch: str) -> bool:
    """粗略判定中日韩文字。认错的代价只是十来个百分点，不值得引一个分词库。"""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF        # CJK 统一表意文字
        or 0x3400 <= cp <= 0x4DBF     # 扩展 A
        or 0x3000 <= cp <= 0x303F     # 中日韩标点（。、「」等）
        or 0xFF00 <= cp <= 0xFFEF     # 全角字符
        or 0x3040 <= cp <= 0x30FF     # 日文假名
        or 0xAC00 <= cp <= 0xD7AF     # 韩文音节
    )


def estimate_text(text: str) -> int:
    """一段文本的 token 估算：CJK 字符 0.6、其余 0.3，最后一次性向上取整。

    混排也是先分类累加再 ceil，而不是每类各 ceil 再加 —— 后者会把两个不到 1 的
    零头各自凑成 1，凭空多算将近一个 token。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return math.ceil(cjk * _CJK_TOKENS_PER_CHAR + (len(text) - cjk) * _OTHER_TOKENS_PER_CHAR)


# 截断时头:尾的字符配比。开头通常带"这是什么"（文件头、导入、标题），结尾带
# "结果如何"（报错栈、汇总行），两头都比中间值钱；头多给一点是因为开头决定模型
# 能不能看懂后面。
_TRUNCATE_HEAD_RATIO = 0.75

# 预算连完整标记都放不下时的退路。已经是能表达"被截断了"的最短形态。
_MIN_MARKER = "…[截断]…"


def truncate_tool_output(text: str, max_tokens: int) -> str:
    """把单次工具输出压到 estimate_text(结果) <= max_tokens；不超限则原样返回**同一对象**。

    超限时保留开头约 75%、结尾约 25% 的字符，中间换成一行说明原长度的标记。

    ★ 用二分而不是"max_tokens / 0.3 个字符"这类近似：中文是 0.6 token/字符，
      按 0.3 反推字符数会让纯中文结果超预算整整一倍 —— 而这个函数的全部意义就是
      "截完一定放得下"。estimate_text 对 (head, tail) 同时增长是单调的（多一个字符
      只会多不会少），所以能二分"保留字符总数 k"，标记本身的开销一并计入。

    ★ 确定性：结果只由 (text, max_tokens) 决定，不读时间/随机数/环境。它会被写进
      幂等表和 transcript，并成为之后每次请求的前缀 —— 同一输入两次得到不同字节，
      会让前缀缓存失效，也会让崩溃恢复补回的内容和当初模型看到的对不上。

    ★ 兜底语义（尽力而为，不抛异常）：max_tokens 小到完整标记都放不下时退到最短
      标记 _MIN_MARKER；若连它都超（max_tokens 个位数），也照样返回它 —— 此时
      "放得下"已不可能，给模型一个明确的"被截断"信号比返回空串好。
    """
    if estimate_text(text) <= max_tokens:
        return text

    n_chars = len(text)
    marker = (
        f"\n…[运行时截断：原输出 {n_chars} 字符，约 {estimate_text(text)} token，"
        f"超过单次上限 {max_tokens} token；中间部分已省略，以下为结尾部分]…\n"
    )

    def build(k: int) -> str:
        head = int(k * _TRUNCATE_HEAD_RATIO)
        tail = k - head
        # tail == 0 时不能写 text[-0:]（那是整串）
        return text[:head] + marker + (text[n_chars - tail:] if tail else "")

    if estimate_text(marker) > max_tokens:
        return _MIN_MARKER
    lo, hi = 0, n_chars          # 不变式：build(lo) 放得下
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_text(build(mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return build(lo)


def estimate_message(msg: Message) -> int:
    """一条消息的估算：正文 + tool_calls（name + 参数 JSON）+ 结构开销。"""
    tokens = _PER_MESSAGE_OVERHEAD
    if msg.content:
        tokens += estimate_text(msg.content)
    for tc in msg.tool_calls or ():
        tokens += estimate_text(tc.name)
        # 参数按序列化后的文本估，而不是只估里面的字符串值：键名、引号、括号
        # 同样是发给模型的内容，漏掉它们会系统性低估。
        tokens += estimate_text(json.dumps(tc.arguments, ensure_ascii=False))
    return tokens


def estimate_request(
    messages: Iterable[Message],
    system_prompt: str,
    tool_defs: Optional[list[dict]],
) -> int:
    """整条请求的估算：system + 全部消息 + tool 定义。

    tool 定义按 json.dumps 后的文本估：它是请求前缀里实打实的一部分，一个带
    完整 JSON Schema 的工具就好几百 token，几十个就是上万。
    """
    total = _PER_MESSAGE_OVERHEAD + estimate_text(system_prompt)
    for msg in messages:
        total += estimate_message(msg)
    for definition in tool_defs or ():
        total += estimate_text(json.dumps(definition, ensure_ascii=False, sort_keys=True))
    return total


# view_state 的类型：(压缩边界, 降级裁剪点)，每一项可为 None。
_VIEW_STATE = tuple[int | None, int | None]


def request_fingerprint(
    system_prompt: str,
    tool_defs: Optional[list[dict]],
    view_state: Optional[_VIEW_STATE] = None,
) -> str:
    """请求前缀的指纹：system + tools + 视图状态，取 sha256 前 16 位。

    dict 用 sort_keys 序列化：键序不影响请求语义，不该让指纹抖动。但工具**列表**
    的顺序不排序 —— 它对模型有实际影响，宁可指纹多变几次，也不掩盖真实差异。

    view_state = (boundary, prune_before)，描述"这条请求看到的**视图**和库里的
    原始历史差在哪"：
      - boundary：生效的最新检查点的 first_kept_seq。压缩一发生它就变，旧锚点
        随之全部失效 —— 前缀已经面目全非，旧的 prompt_tokens 没有参考价值；
      - prune_before：D1 降级时旧工具输出被替换成占位符的界线。替换之后同样
        作废旧锚点，否则 usage 描述的是替换前的长前缀，会把占用数**高估**。
    ★ (None, None) 与 None 产出**同一个**指纹。历史上有锚点时 view_state 恒为
      None，若把它序列化成 [null, null]，升级这次代码会让库里所有锚点集体失效
      一次（凭空多一次全量估算、多一次压缩）。归一化在这里做，调用方不必记得。
    ★ 不在这里 import 别的东西：compressor 会用同一个函数算指纹，两边必须是
      同一份实现，否则"写进去的锚点和比对时的锚点"用的不是一把尺子。
    """
    # 先归一化，再拼 payload：None 和 (None, None) 都落到**旧版形状**
    # {"system", "tools", "boundary": None}。只拼"非默认"的键，让没有触发压缩/
    # 降级的请求与 9f69ea4 之前写进库的指纹逐字节相同 —— 升级代码不清空存量锚点。
    state = view_state if view_state is not None else (None, None)
    boundary, prune_before = state
    payload_dict: dict = {"system": system_prompt, "tools": tool_defs or [], "boundary": boundary}
    if prune_before is not None:
        payload_dict["prune_before"] = prune_before
    payload = json.dumps(payload_dict, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _usage_is_valid(usage: Optional[TokenUsage]) -> bool:
    """这条 usage 在结构上能不能当锚点（不含 fingerprint 比对）。

    prompt_tokens > 0：0 意味着上游没填或填错了，拿它当锚点等于把整条请求当成
    空请求，必然漏触发。completion_tokens >= 0：可以为 0，但负数没有意义。
    bool 是 int 的子类，先排掉，免得 True 被当成"1 个 token"的锚点。
    """
    if usage is None:
        return False
    prompt, completion = usage.prompt_tokens, usage.completion_tokens
    if isinstance(prompt, bool) or isinstance(completion, bool):
        return False
    return isinstance(prompt, int) and prompt > 0 and isinstance(completion, int) and completion >= 0


def measure(
    messages: list[Message],
    system_prompt: str,
    tool_defs: Optional[list[dict]],
    view_state: Optional[_VIEW_STATE] = None,
) -> TokenCount:
    """估算"把 messages 全发出去"这次请求要花多少 token。

    从后往前找**最后一条有有效锚点**的 assistant：
      - usage 不为 None，且 _usage_is_valid；
      - fingerprint 与**现在**的前缀一致。
      找到 → prompt + completion + 估算(锚点之后的消息)，source="usage"。
    找不到 → 整条请求估算，source="estimate"（新会话第一轮就走这条路）。

    ★ 找的是"有有效 usage 的 assistant"，不是"最后一条 assistant"。
      取消封口消息、旧代码写的消息、预算耗尽时 tools=None 的收尾调用（fingerprint
      不同）都没有可用锚点，一律跳过、开销落进增量估算里。拿"最后一条 assistant"
      当锚点，会把这些开销整个漏掉 —— 而漏掉恰恰发生在"刚塞进一个大工具结果"
      这种最需要数准的时刻。

    ★ completion_tokens 也要算。 锚点那次请求产生的补全也在 transcript 里，是
      下一次请求前缀的一部分，而 prompt_tokens 描述前缀时不含它。
    """
    fingerprint = request_fingerprint(system_prompt, tool_defs, view_state)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.role != "assistant" or not _usage_is_valid(msg.usage):
            continue
        if msg.usage.fingerprint != fingerprint:
            continue
        tail = sum(estimate_message(m) for m in messages[i + 1:])
        return TokenCount(
            tokens=msg.usage.prompt_tokens + msg.usage.completion_tokens + tail,
            source=SOURCE_USAGE,
        )
    return TokenCount(
        tokens=estimate_request(messages, system_prompt, tool_defs),
        source=SOURCE_ESTIMATE,
    )
