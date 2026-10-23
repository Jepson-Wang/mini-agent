# mini-agent

学习型最小 AI Agent 运行时（DeepSeek-only）。两个 surface：CLI REPL 和 gRPC server。

## 安装

```bash
pip install -e ".[dev]"
cp .env.example .env          # 然后把真实的 DEEPSEEK_API_KEY 填进去
```

## 跑起来

```bash
# CLI
python -m mini_agent                     # 新建会话
python -m mini_agent --resume sess_xxxx  # 续跑之前的会话

# gRPC server（给 Go 后端调，默认 :50051）
python -m mini_agent.grpcserver.server
```

gRPC 端口和并发数走 `MINI_AGENT_GRPC_PORT` / `MINI_AGENT_GRPC_MAX_WORKERS`，
默认值见 `mini_agent/config.py`。端口要和 Go 那边的 `AGENT_ADDR` 对上。

## 测试

```bash
pytest                  # 全部
pytest -m "not slow"    # 平时开发，跳过拉子进程的崩溃恢复测试
pytest -m slow          # 只跑真 SIGKILL 的崩溃恢复
```

测试不需要 API key 和网络：`tests/conftest.py` 的 `fake_llm` fixture 会把
`mini_agent.agent.call_llm` 换成脚本化的假响应。

干净 clone 上要先跑一次下面「protoc」一节的生成命令，否则 `test_grpcserver.py`
在收集阶段就 ImportError。

### CI

`.github/workflows/ci.yml` 跑的就是上面这几条命令，没有 CI 专属的魔法：
`pip install -e ".[dev]"` → protoc 生成 → `pytest -m "not slow"` → `pytest -m slow`，
在 ubuntu/3.11、ubuntu/3.15、windows/3.14 三个组合上各跑一遍。本地能复现 CI 的一切。

CI **不检查** `mini_agent/proto/` 和 backend 那份权威副本是否一致（它只 clone 本仓库）。
两份是否逐字节相同，目前只能靠改协议时照下面的顺序手工保证。

## protoc：改了 .proto 之后

`.proto` 的**权威副本在 backend 仓库**（`backend/proto/agent/v1/agent.proto`），
这边 `mini_agent/proto/` 下的是它的复制品。改协议的正确顺序是：改 backend 那份 →
复制过来 → 两边各自重新生成。

```bash
# 1. 从 backend 同步（两份必须逐字节一致）
cp ../backend/proto/agent/v1/agent.proto mini_agent/proto/agent/v1/agent.proto

# 2. 重新生成 mini_agent/gen/ 下的 stub
python -m grpc_tools.protoc \
  --proto_path=mini_agent/proto \
  --python_out=mini_agent/gen \
  --grpc_python_out=mini_agent/gen \
  --pyi_out=mini_agent/gen \
  mini_agent/proto/agent/v1/agent.proto
```

生成出来的 `*_pb2*.py` 已被 `.gitignore` 忽略，不入库；`mini_agent/gen/` 下那几个
手写的 `__init__.py` 要入库 —— 顶层那个带着一段**必需**的 sys.path 补丁，
原因见文件里的说明（protoc 生成的代码内部用的是顶层绝对导入 `from agent.v1 import ...`）。

## 事件契约

`RunTask` 是服务端流式。事件类型的字符串和 Go 那边 `agentclient.EventType`
的常量逐字对齐，改一边必须改另一边：

| 事件 | 什么时候发 | Go 那边的动作 |
|---|---|---|
| `started` | 流一建立就发 | → `running` |
| `thinking` | 模型带着 tool_calls 返回时的那段文字 | 只记日志 |
| `tool_call` | 每次 dispatch **之前** | 只记日志 |
| `tool_result` | 每次工具执行完 | 只记日志 |
| `finished` | `run_conversation` 正常返回 | → `succeeded` |
| `error` | 任务抛异常 | → `failed`，payload 当失败原因 |

**被取消时不发任何终态事件**，直接让流干净 EOF —— 库里在通知我们之前就已经是
`canceled` 了（Go 那边是先落库、再通知），再发一个终态事件反而会盖掉它。
