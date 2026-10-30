"""压缩的**触发**：阈值策略 + 压缩器接口。

[M3] 回答"什么时候该压缩"。真正的总结逻辑（调 LLM 把中段折叠成摘要、写
compactions 检查点）是下一块，本模块只定接口和策略。

为什么"判断要不要压"和"怎么压"分成两个东西：
  - 策略是纯函数（TokenCount → bool），能脱库、脱网络单测，不同入口也能用不同
    阈值；
  - 压缩器有副作用（写检查点、花一次 LLM 调用），实现它注定要碰库和网络。
  拆开后 Agent 只依赖 "policy + compressor" 两个可选依赖，缺一个就完全不介入。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mini_agent.context.tokens import TokenCount


class Compressor(Protocol):
    """压缩器接口。这次只定义，不实现。

    ★ 契约是 `compress(turn) -> None`（把检查点落库），不是"返回一份新的消息
      列表"。压缩后历史要按检查点从**库里**读（recover 改造是下一块）；由这里
      往内存塞一份改好的副本，一旦和库分叉，崩溃恢复就失去意义。

    ★ 可以抛异常。主循环会接住并记日志（压缩是优化，不是正确性前提），但压缩器
      不该依赖这一点 —— 半途失败时别留下一个引用不存在的消息或指向 tool 消息的
      检查点。外键只挡得住"消息行不存在"。

    ★ 参数不写成 _Turn。agent.py 只要从 turn 里拿 session_id / messages，写死具体
      类型会让 compressor 反向 import agent 而成环；Protocol 本就是结构化类型。
    """

    def compress(self, turn) -> None: ...


@dataclass(frozen=True)
class CompressionPolicy:
    """按"估算 token 占 context_window 的比例"决定要不要压缩。

    ★ 用 >= 而不是 >。阈值本身就算触发：ratio=0.5、window=1000 时阈值恰好 500，
      写成 > 的话"正好在阈值上"的请求要再塞一轮工具结果才压，而超限的代价是 400。
    """

    context_window: int
    trigger_ratio: float

    def should_compress(self, count: TokenCount) -> bool:
        return count.tokens >= self.context_window * self.trigger_ratio
