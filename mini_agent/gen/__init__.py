"""protoc 生成代码的容器包。

★ 这里的 sys.path 补丁是必须的，别删。

`grpc_tools.protoc` 生成的 `agent_pb2_grpc.py` 里写的是**顶层绝对导入**：

    from agent.v1 import agent_pb2 as agent_dot_v1_dot_agent__pb2

这个路径来自 .proto 里的 `package agent.v1`，和 `--python_out` 指定的输出目录
无关 —— protoc 不知道我们把生成结果放进了 `mini_agent/gen/` 下面。也就是说，
只有当 **本目录**（gen/）在 sys.path 上时，`agent.v1` 才解析得到。

所以在包被 import 的第一时间把 gen/ 塞进 sys.path。这样上层照常写符合项目规范的
绝对导入（`from mini_agent.gen.agent.v1 import agent_pb2`），import 机制在加载
`agent_pb2_grpc` 时会顺着这条补丁把它内部那句顶层导入解析掉。

为什么不改生成代码：那是 protoc 的产物，重新生成一次就被覆盖回去了。补丁放在
这个手写的 `__init__.py` 里，才是唯一不会被覆盖的位置。

注意这**不违反** CLAUDE.md 的「绝对导入」硬约束：那条约束禁止的是把
`mini_agent/` 本身加进 sys.path（会让 schema/config/log 这些通用名变成顶层模块，
同一份代码有两个模块身份）。这里加的是 `mini_agent/gen/`，它下面只有 protoc
生成的 `agent/` 一个包，撞名风险自成一体。
"""
from __future__ import annotations

import sys
from pathlib import Path

_GEN_DIR = str(Path(__file__).resolve().parent)
if _GEN_DIR not in sys.path:
    sys.path.insert(0, _GEN_DIR)
