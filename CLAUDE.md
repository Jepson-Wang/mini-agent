# 学习型 MVP 实施方案：从零手写一个仿 Hermes Agent 架构的最小可运行 AI Agent 框架

> 面向准备字节跳动后端实习面试的 Python 开发者 | 学习导向、循序渐进、可落地

## TL;DR

- **这是一份可照着写的 6 里程碑(M0→M5)实施蓝图**:用纯 Python + 官方 openai/anthropic SDK + 标准库 sqlite3/concurrent.futures + Pydantic,从一个 50 行的对话循环逐步长成一个具备「agent loop / 工具系统 / subagent / self-improving / 持久化 + 简化上下文压缩」五大子系统的最小 agent。每个里程碑都是一个能独立 demo 的可运行增量,总工作量约 5–8 个周末(作者估算)。
- **核心方法论是「Hermes 真实做法 → MVP 简化」的逐子系统对照**:保留 Hermes 最有教学价值的架构决策(单一 AIAgent 类跑遍所有 surface、规范化内部 transcript + 薄 provider adapter、工具结果统一 JSON 契约、状态工具在 loop 内拦截、prompt 缓存前缀神圣性、子代理上下文隔离与 toolset 父子交集),砍掉生产级才需要的复杂度(多 surface gateway、MCP、插件系统、6 种 terminal backend、RL 轨迹导出、外部记忆 provider、并行流式回调)。
- **面试价值在于能讲清「生产级 agent 工程权衡」**:context 管理触发阈值与 tool_call/tool_result 配对保护、工具并发安全(只读白名单可并行)、子代理隔离与最小权限(父子 toolset 交集 + 递归深度限制)、崩溃恢复(每条消息即时落库 + WAL)、prompt cache 命中(稳定系统提示前缀)——这些都是真实大厂 agent 平台的考点,而你能同时说出「Hermes 怎么做的」和「我的 MVP 做了哪些简化及为什么」。

---

## 一、整体架构与设计哲学

### 1.1 设计哲学:从 Hermes 学什么

Hermes Agent 的官方架构文档把它最核心的设计原则总结为「One AIAgent class serves CLI, gateway, ACP, batch, and API server. Platform differences live in the entry point, not the agent.」——单一 `AIAgent` 类承载所有交互表面,平台差异只存在于入口点。这是我们 MVP 要继承的第一性原则。

这也呼应 Anthropic《Building Effective Agents》(Schluntz & Zhang, 2024 年 12 月)对 agent 的定义:「agents are typically just LLMs using tools based on environmental feedback in a loop」,并强调要有终止条件「a stopping condition (such as a maximum number of iterations)」——这正是 MVP 里 `IterationBudget` 的设计依据。

我把 Hermes 值得学的设计哲学拆成 6 条,并标注 MVP 的取舍:

| Hermes 设计哲学 | 含义 | MVP 取舍 | 理由 |
|---|---|---|---|
| **单一 AIAgent 跑遍所有 surface** | 一个类承载 CLI/gateway/cron/batch,surface 只是薄入口 | **保留**(只做 CLI 一个 surface,但 `Agent` 类设计成 surface 无关) | 这是「子代理复用主类」的前提,教学价值高 |
| **Narrow waist / 规范化内部 transcript** | 内部统一用 OpenAI 风格 `role/content/tool_calls` dict,三种 API mode 都收敛到它 | **保留**(内部规范格式 + 薄 adapter) | 这是「provider adapter 薄层」的精华,面试高频 |
| **薄 provider adapter** | `ProviderTransport` ABC,各 transport 只管消息转换/工具转换/响应归一化 | **简化**(两个函数 `to_openai()` / `to_anthropic()`,不做 ABC) | MVP 只需 2 个 provider,函数足够 |
| **Prompt cache 前缀神圣性** | 系统提示在会话中字节不变,「No cache-breaking mutations except explicit user actions」 | **保留为原则 + 一个 CI 测试** | 不实际接 Anthropic 缓存也要理解,面试谈资 |
| **状态工具在 loop 内拦截** | `todo/memory/session_search/delegate_task` 需要 agent 级状态,在 loop 里拦截,不走 registry | **保留**(`memory` / `delegate_task` / `session_search` 拦截) | 这是「为什么有些工具特殊」的关键认知 |
| **工具结果统一 JSON 契约** | Handler 必须返回 JSON 字符串,异常包成 `{"error": "..."}`,模型永远收到合法 JSON | **完全保留** | 这是 agent 健壮性的核心,实现成本极低 |

### 1.2 砍掉的东西(及为什么)

为了让 MVP 聚焦学习价值,以下 Hermes 子系统**明确不做**:多平台 gateway(Telegram/Discord/Slack 等)、MCP server 接入、插件系统(三来源发现 + hook)、6 种 terminal backend(docker/ssh/modal 等)、危险命令审批(DANGEROUS_PATTERNS)、外部记忆 provider(Honcho/Mem0 等 8 个)、RL 轨迹导出(Atropos/ShareGPT)、prompt caching 的实际接入、并行流式 callback、cron 调度、TUI(React/Ink)。这些要么是「生产级运维需求」,要么是「特定生态集成」,对理解 agent 核心原理边际收益低。这也符合 Anthropic 的核心建议:「When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all.」

### 1.3 整体架构图(ASCII)

```
                          ┌─────────────────────────────┐
       用户 (CLI)  ───────▶│   main.py  (薄入口 / REPL)    │
                          └──────────────┬──────────────┘
                                         │ Agent(session_id, toolset)
                                         ▼
        ┌────────────────────────────────────────────────────────────┐
        │                      agent.py :: Agent                       │
        │   run_conversation()  ── 主循环 (OODA: 调LLM→解析→执行→回填)   │
        │     · IterationBudget (max_turns 防失控)                      │
        │     · 状态工具拦截 (memory / delegate_task / session_search)  │
        │     · 错误重试 / 空回复处理 / 终止条件                          │
        └───┬───────────┬───────────┬───────────┬───────────┬──────────┘
            │           │           │           │           │
            ▼           ▼           ▼           ▼           ▼
   ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │providers │ │  tools/   │ │ context  │ │ memory & │ │ persistence  │
   │ .py      │ │ registry  │ │ .py      │ │ skills   │ │ (sqlite)     │
   │薄adapter │ │+dispatch  │ │压缩(简版)│ │自我改进   │ │session/msgs  │
   └────┬─────┘ └─────┬─────┘ └──────────┘ └────┬─────┘ └──────┬───────┘
        │             │                          │              │
        ▼             ▼                          ▼              ▼
   OpenAI/        @tool 装饰器             MEMORY.md /      state.db (WAL)
   Anthropic      自动生成 schema          USER.md /        sessions
   官方 SDK       (read_file/write_file/   skills/*.md       messages
                  web_fetch/...)          (frontmatter)     parent_session_id

                         delegate_task 拦截 → spawn 子 Agent(复用 Agent 类)
                         子 agent: 独立 messages / budget / session, toolset⊆父
```

### 1.4 目录结构(每个 .py 文件职责)

```
mini-hermes/
├── README.md                      # 项目说明、架构图、运行方式
├── pyproject.toml                 # 依赖与打包(openai/anthropic/pydantic/tiktoken/pytest)
├── .env.example                   # ANTHROPIC_API_KEY / OPENAI_API_KEY 模板
├── .gitignore                     # 忽略 .env、__pycache__、*.db、.mini_hermes/
│
├── mini_hermes/                   # 主包
│   ├── __init__.py
│   ├── config.py                  # [M0] 读 env、模型名、阈值等集中配置(dataclass/pydantic-settings)
│   │
│   ├── agent.py                   # [M0→M4] 核心 Agent 类(对应 Hermes run_agent.py 的 AIAgent)
│   │                              #   持有 messages / tools / compressor / session_db / memory / budget
│   ├── loop.py                    # [M0] run_conversation 主循环(对应 conversation_loop.py)
│   │                              #   组装→调LLM→解析tool_calls→执行→回填→循环
│   ├── budget.py                  # [M0] IterationBudget(max_turns 防失控;子代理独立预算)
│   │
│   ├── llm/                       # provider 适配薄层(对应 Hermes 的 adapter 层)
│   │   ├── __init__.py
│   │   ├── base.py                # [M0] 统一内部消息格式 + LLMClient 抽象接口
│   │   ├── anthropic_client.py    # [M0] anthropic SDK 封装(messages 格式 + tool schema 转换)
│   │   └── openai_client.py       # [M0] openai SDK 封装(chat completions 格式)
│   │
│   ├── prompt.py                  # [M0→M5] system prompt 分层组装(stable/context/volatile)
│   │                              #   对应 prompt_builder.py;memory/skills 注入点在这里
│   │
│   ├── tools/                     # [M1] 工具系统(对应 tools/ + model_tools.py + toolsets.py)
│   │   ├── __init__.py
│   │   ├── registry.py            # [M1] @tool 装饰器 + ToolRegistry 单例 + 自动 JSON Schema 生成
│   │   ├── dispatch.py            # [M1] handle_tool_call:参数校验→执行→结果统一成 JSON 字符串→异常包裹
│   │   ├── executor.py            # [M1+] 串行执行;(可选)并行判定+ThreadPoolExecutor
│   │   ├── builtin/               # 内置工具实现
│   │   │   ├── __init__.py
│   │   │   ├── file_tools.py      # [M1] read_file / write_file(代表性读/写工具)
│   │   │   ├── web_tools.py       # [M1] web_search 或 http_get(代表性外部工具)
│   │   │   └── shell_tool.py      # [M1] (可选)run_command,带审批
│   │   └── state_tools.py         # [M4/M5] 被 loop 拦截的状态工具:memory / delegate_task
│   │                              #   (需访问 agent 内部状态,不走 registry)
│   │
│   ├── delegate.py                # [M4] subagent 委派:_build_child_agent、工具集父子交集、
│   │                              #   独立 budget、递归深度限制、结果回填(对应 delegate_tool.py)
│   │
│   ├── memory/                    # [M5] self-improving(memory + skills)
│   │   ├── __init__.py
│   │   ├── store.py               # [M5] MemoryStore:MEMORY.md/USER.md 读写、字符上限、注入快照
│   │   ├── skills.py              # [M5] skill 加载器:扫描 SKILL.md + frontmatter、progressive disclosure
│   │   └── review.py              # [M5] (可选)background review:回复后总结、更新 memory
│   │
│   ├── context/                   # [M3] 上下文管理(简化版压缩)
│   │   ├── __init__.py
│   │   ├── tokens.py              # [M3] token 计数(tiktoken 或字符粗估)
│   │   └── compressor.py          # [M3] SimpleCompressor:阈值触发、老消息总结、tool_call/result 配对保护
│   │
│   ├── persistence/               # [M2] 持久化与崩溃恢复(对应 hermes_state.py)
│   │   ├── __init__.py
│   │   ├── session_db.py          # [M2] SessionDB:SQLite WAL、每条消息即时写库、parent_session_id
│   │   └── schema.sql             # [M2] sessions / messages 表 DDL(可选 FTS5)
│   │
│   └── cli.py                     # [M0→] 命令行入口:mini-hermes "你的问题"
│
├── skills/                        # [M5] 示例 skill 目录(运行时也可指向 ~/.mini_hermes/skills/)
│   └── example-skill/
│       └── SKILL.md               # name/description frontmatter + 正文步骤
│
├── tests/                         # 测试(对应每个子系统)
│   ├── __init__.py
│   ├── conftest.py                # FakeLLM fixture(mock LLM 调用,避免真调 API)
│   ├── test_loop.py               # [M0] 主循环、终止条件、budget
│   ├── test_tools.py              # [M1] 注册/发现/dispatch/异常包裹/JSON 契约
│   ├── test_persistence.py        # [M2] 写库、崩溃后从 DB 重建 messages
│   ├── test_compressor.py         # [M3] 阈值触发、配对保护
│   ├── test_delegate.py           # [M4] 子代理隔离、工具交集、结果回填
│   └── test_memory.py             # [M5] memory 读写上限、skill 加载
│
└── scripts/
    └── demo.py                    # 一个"装好就能跑"的最小端到端 demo
```

### 1.5 技术选型表

| 关注点 | 选型 | 替代/Hermes 对照 | 理由 |
|---|---|---|---|
| LLM 调用 | `openai` 官方 SDK(主)+ `anthropic` 官方 SDK(可选第二 provider) | Hermes 用 `ProviderTransport` ABC 抽象 4 种 transport | 直接用官方 SDK 吃透底层,不引入 LangChain/LiteLLM |
| 内部消息格式 | OpenAI 风格 `{role, content, tool_calls}` dict + Pydantic 校验 | Hermes「All three converge on the same internal message format (OpenAI-style)」 | 与 Hermes narrow-waist 一致 |
| Schema 校验 / 工具 schema 生成 | Pydantic v2 + `inspect` 签名 | Hermes 手写 schema dict | Pydantic `model_json_schema()` 自动生成更省事 |
| 持久化 | 标准库 `sqlite3` + WAL 模式 | Hermes 用 `hermes_state.py` 包装 SQLite,schema v11 | 完全一致的设计,标准库零依赖 |
| 全文检索(可选) | SQLite FTS5 | Hermes 用 `messages_fts` + trigram tokenizer | FTS5 是 sqlite 内置 |
| 并发(并行工具/子代理) | `concurrent.futures.ThreadPoolExecutor` | Hermes 多 tool_call 时用 ThreadPoolExecutor | 标准库,够用 |
| token 计数 | `tiktoken`(OpenAI)+ 字符估算回退 | Hermes 优先用 API 实报 token,回退字符估算 | tiktoken 是 OpenAI 官方 |
| 测试 | `pytest` + FakeLLM mock | — | mock 掉 LLM 避免每次真调 API |
| 配置 | `pyproject.toml` + `.env`(`python-dotenv`) | Hermes 用 `config.yaml` | MVP 用 env 足够 |

---

## 二、分阶段实施路线图

设计原则:**每个里程碑结束时,`python main.py` 都能跑起来并 demo 出新能力**。后一阶段在前一阶段基础上增量叠加,不推倒重来。

| 里程碑 | 目标 | 新增模块/函数 | 验收标准 | 工作量(估算) |
|---|---|---|---|---|
| **M0** | 最小对话循环 | `agent.py`(无工具版 run_conversation)、`providers.py`、`schema.py`、`main.py` | CLI 能多轮对话,messages 列表正确累积,`max_turns` 生效 | 半天 |
| **M1** | 工具系统 | `tools/registry.py`、`tools/builtins.py`、`tools/schemas.py`;loop 接入 tool_calls 解析/执行/回填 | 模型能调用 `read_file`/`write_file`,异常被包成 JSON,工具结果回填后模型继续 | 1–2 天 |
| **M2** | 持久化与崩溃恢复 | `persistence.py`(SessionDB);agent 每轮即时写库 | kill 掉进程后重启,`--resume <session_id>` 能从 DB 重建 messages 继续对话 | 1 天 |
| **M3** | 简化上下文管理 | `context.py`;loop 内 preflight 压缩检查 | 构造超长对话,触发阈值后老消息被总结成一条 summary,tool 配对不被拆散,API 不报 400 | 1–2 天 |
| **M4** | subagent 委派 | `delegate.py`;`delegate_task` 在 loop 内拦截 | 主 agent 能 `delegate_task(goal=...)` 派生子 agent,子 agent 独立跑完返回 summary 回填父;toolset⊆父;递归深度=1 拦截 | 1–2 天 |
| **M5** | self-improving | `memory.py`、`skills.py`;`build_system_prompt` 注入;`memory` 工具拦截;周期性 nudge | 记忆能跨 session 持久(写 MEMORY.md,下次启动注入);skill 索引注入 + `skill_view` 按需加载全文 | 2–3 天 |

> **进阶(M6+,可选)**:并行只读工具、两层压缩(preflight 50% + gateway 85%)、Anthropic prompt cache 实接、session_search FTS 工具、第二 surface(简易 HTTP server)、MCP 接入。

---

## 三、各核心子系统详细设计

### 3.1 Agent Loop(主对话循环)

#### Hermes 真实做法(回顾)
Hermes 的 `AIAgent.run_conversation()` 每轮遵循固定序列:生成 task_id → 追加 user 消息 → 构建/复用缓存的系统提示 → 检查是否需 preflight 压缩(>50% context)→ 构建 API messages(三种 mode 各自转换)→ 注入 ephemeral 层 → 应用 prompt 缓存标记 → 发起可中断 API 调用 → 解析响应:有 tool_calls 就执行并回填、循环;纯文本就持久化、flush 记忆、返回。它强制严格的消息角色交替(绝不连续两条 assistant 或两条 user,只有 tool 角色可连续),并用 `IterationBudget` 控制失控(默认 90 轮,子代理独立 budget 默认 50)。三种 API mode(`chat_completions` / `codex_responses` / `anthropic_messages`)全部收敛到统一的内部 OpenAI 风格格式。

#### MVP 如何简化
- 只做两种 provider mode:`chat_completions`(OpenAI 兼容)和 `anthropic_messages`。
- 内部统一用 OpenAI 风格 dict;`providers.py` 只在「发出去前」和「收回来后」做转换。
- 不做可中断调用(无后台线程监控 interrupt),只做简单的同步调用 + 重试。
- `max_turns` 用一个简单计数器(默认 25,子代理 10),到顶返回「已尽力」总结。
- 不做流式;不做 callback 体系(M6 可加)。

#### 关键数据结构与函数签名

```python
# schema.py
from pydantic import BaseModel
from typing import Literal, Any

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]   # 内部已解析好的 dict(从 JSON string 解析)

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None   # 仅 assistant
    tool_call_id: str | None = None            # 仅 tool

# agent.py
class IterationBudget:
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self.used = 0
    def tick(self) -> bool:        # 返回 False 表示预算耗尽
        self.used += 1
        return self.used <= self.max_turns

class Agent:
    def __init__(self, *, session_id: str, db: "SessionDB",
                 toolset: set[str], provider: str = "openai",
                 model: str = "gpt-4o", max_turns: int = 25,
                 depth: int = 0):
        ...
    def run_conversation(self, user_message: str) -> dict:
        """返回 {final_response, messages, completed, used_turns}"""
```

#### 主循环核心骨架(可运行级伪代码)

```python
def run_conversation(self, user_message: str) -> dict:
    self.messages.append(Message(role="user", content=user_message))
    self.db.append_message(self.session_id, "user", user_message)  # M2: 即时落库

    budget = IterationBudget(self.max_turns)
    while budget.tick():
        # M3: preflight 压缩检查
        if self.context.should_compress(self.messages):
            self.messages = self.context.compress(self.messages)

        system = build_system_prompt(self)           # M5: 注入 memory/skills 索引
        api_msgs = to_provider_format(system, self.messages, self.provider)
        tools = self.registry.get_definitions(self.toolset, self.provider)

        try:
            resp = call_llm(api_msgs, tools, self.provider, self.model)
        except RetryableError:
            if not self._retry(): raise
            continue

        assistant_msg = parse_response(resp, self.provider)  # → Message
        self.messages.append(assistant_msg)
        self.db.append_message(self.session_id, "assistant",
                               assistant_msg.content, assistant_msg.tool_calls)

        if not assistant_msg.tool_calls:
            # 终止条件:纯文本回复
            if not (assistant_msg.content or "").strip():
                # 空回复处理:补一个 nudge 再试一轮
                self.messages.append(Message(role="user",
                    content="(empty response — please continue or answer)"))
                continue
            self._maybe_background_review()          # M5: 周期性 nudge
            return {"final_response": assistant_msg.content, ...}

        # 执行工具(M1)+ 状态工具拦截(M4/M5)
        for tc in assistant_msg.tool_calls:
            result = self._dispatch_tool(tc)         # 见 3.2 / 3.3 / 3.5
            tool_msg = Message(role="tool", tool_call_id=tc.id, content=result)
            self.messages.append(tool_msg)
            self.db.append_message(self.session_id, "tool", result,
                                   tool_call_id=tc.id)
    # 预算耗尽
    return {"final_response": "(stopped: iteration budget exhausted)",
            "completed": False, ...}
```

#### provider adapter 薄层

```python
# providers.py
def to_provider_format(system, messages, provider):
    if provider == "openai":
        out = [{"role": "system", "content": system}]
        for m in messages:
            d = {"role": m.role}
            if m.content is not None: d["content"] = m.content
            if m.tool_calls:
                d["tool_calls"] = [{"id": tc.id, "type": "function",
                    "function": {"name": tc.name,
                                 "arguments": json.dumps(tc.arguments)}}
                    for tc in m.tool_calls]
            if m.tool_call_id: d["tool_call_id"] = m.tool_call_id
            out.append(d)
        return out
    elif provider == "anthropic":
        # system 走顶层 system 参数;tool_use/tool_result 用 content block
        # assistant 的 tool_calls → content=[{"type":"tool_use",...}]
        # tool 角色 → user 消息里 content=[{"type":"tool_result",
        #            "tool_use_id":..., "content":...}]
        ...
```

> **关键坑**:OpenAI Chat Completions 的工具 schema 外层是 `{"type":"function","function":{...}}`,而 Anthropic 是 `{"name","description","input_schema"}`,两者 schema 形状不同,adapter 要负责转换。Anthropic 还要求**带 tool_use/tool_result 的请求必须定义 tools**,否则 400;且 tool_use 块后面必须紧跟匹配数量的 tool_result 块——这正是 M3 压缩时要保护配对的原因。

#### 验收/测试方法
用 FakeLLM(见第五节)按脚本返回「先 tool_call 再纯文本」,断言:messages 序列符合角色交替、工具被执行、最终返回 final_response、`max_turns` 能截断死循环。

---

### 3.2 工具系统

#### Hermes 真实做法(回顾)
Hermes 工具是「自注册函数」:每个 `tools/*.py` 在 import 时调 `registry.register(name, toolset, schema, handler, check_fn, is_async, ...)`,创建 `ToolEntry` 存进单例 `ToolRegistry._tools`。`discover_builtin_tools()` 用 AST 解析扫描 `tools/` 目录,找到含顶层 `registry.register()` 调用的模块并 import,实现自动发现。`get_tool_definitions(enabled, disabled)` 解析 toolset → 跑每个工具的 `check_fn`(API key 在不在、binary 装没装,异常即视为不可用)→ 返回 OpenAI 格式 schema。Dispatch 流程:模型 tool_call → `handle_function_call()` → 先看是不是 agent-loop 工具(拦截)→ 插件 pre-hook → `registry.dispatch()` 查 ToolEntry → async handler 用 `_run_async()` 桥接 → 返回结果字符串。**两层错误包裹**保证模型永远收到合法 JSON:`registry.dispatch()` 捕获异常返回 `{"error": "Tool execution failed: ..."}`,`handle_function_call()` 再包一层。关键规则(官方 adding-tools 文档原文):**Handlers MUST return a JSON string(`json.dumps()`),错误返回 `{"error": "..."}` 而非抛异常**。并发上,单个 tool_call 直接在主线程执行,多个 tool_call 用 `ThreadPoolExecutor` 并发(interactive 工具如 clarify 强制串行),结果按原始顺序重排回填。

#### MVP 如何简化
- 用 **`@tool` 装饰器** + `inspect` 签名 + Pydantic 自动生成 JSON Schema,免去手写 schema dict。
- 注册表用一个模块级单例 dict,装饰器在 import 时注册;发现就靠「在 `builtins.py` 里 import 一遍」,不做 AST 扫描(M6 可加)。
- `check_fn` 简化为可选 lambda(如 web 工具检查 env)。
- **完全保留 JSON 契约 + 两层异常包裹**(成本极低、收益极高)。
- 并发:MVP 默认全串行;可选实现「只读白名单可并行」的最简版(见末尾)。

#### 关键数据结构与函数签名

```python
# tools/registry.py
@dataclass
class ToolEntry:
    name: str
    toolset: str
    schema: dict           # {"name","description","parameters": {JSON Schema}}
    handler: Callable[..., str]
    check_fn: Callable[[], bool] | None = None
    read_only: bool = False   # 用于并行判定

class ToolRegistry:
    _tools: dict[str, ToolEntry] = {}

    @classmethod
    def register(cls, entry: ToolEntry): cls._tools[entry.name] = entry

    @classmethod
    def get_definitions(cls, toolset: set[str], provider: str) -> list[dict]:
        defs = []
        for e in cls._tools.values():
            if e.toolset not in toolset: continue
            if e.check_fn and not _safe_check(e.check_fn): continue
            defs.append(_to_provider_tool_schema(e.schema, provider))
        return defs

    @classmethod
    def dispatch(cls, name: str, args: dict) -> str:
        e = cls._tools.get(name)
        if not e:
            return json.dumps({"error": f"unknown tool: {name}"})
        try:
            result = e.handler(**args)
            return result if isinstance(result, str) else json.dumps(result)
        except Exception as ex:          # 第一层包裹
            return json.dumps({"error": f"Tool failed: {type(ex).__name__}: {ex}"})
```

#### `@tool` 装饰器 + 自动 schema 生成

```python
# tools/schemas.py
from pydantic import create_model
import inspect

def tool(toolset: str = "core", read_only: bool = False, check_fn=None):
    def deco(fn):
        sig = inspect.signature(fn)
        fields = {}
        for pname, p in sig.parameters.items():
            ann = p.annotation if p.annotation is not inspect._empty else str
            default = ... if p.default is inspect._empty else p.default
            fields[pname] = (ann, default)
        ArgsModel = create_model(f"{fn.__name__}_Args", **fields)
        json_schema = ArgsModel.model_json_schema()
        ToolRegistry.register(ToolEntry(
            name=fn.__name__, toolset=toolset,
            schema={"name": fn.__name__,
                    "description": (fn.__doc__ or "").strip(),
                    "parameters": json_schema},
            handler=fn, check_fn=check_fn, read_only=read_only))
        return fn
    return deco
```

#### 代表性工具完整示例

```python
# tools/builtins.py
@tool(toolset="file", read_only=True)
def read_file(path: str) -> str:
    """Read a UTF-8 text file and return its content."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": str(e)})  # 也可直接 raise,registry 会包

@tool(toolset="file", read_only=False)
def write_file(path: str, content: str) -> str:
    """Write text to a file (overwrites). Returns bytes written."""
    Path(path).write_text(content, encoding="utf-8")
    return json.dumps({"ok": True, "bytes": len(content.encode())})

@tool(toolset="web", read_only=True,
      check_fn=lambda: bool(os.getenv("ALLOW_WEB")))
def web_fetch(url: str) -> str:
    """Fetch a URL and return text (truncated to 8000 chars)."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.read().decode("utf-8", "ignore")[:8000]
```

状态工具示例(见 3.5,`memory` 不走 registry,在 loop 内拦截因为它要访问 agent 的 MemoryStore)。

> **Hermes 工具文件骨架对照**(官方 adding-tools 文档):一个内置工具文件包含 `check_fn`、handler(返回 `json.dumps(...)`)、schema dict、`registry.register(...)`,且「Handlers MUST return a JSON string, never raw dicts」「Errors MUST be returned as `{"error": "message"}`, never raised」。MVP 的 `@tool` 装饰器把这套样板自动化了。

#### 同步/异步桥接
MVP 默认全同步。若某工具必须 async,用 `asyncio.run(coro)` 在 dispatch 里桥接即可(单线程 CLI 场景安全);这是 Hermes `_run_async()` CLI 路径的极简版。

#### (可选)并行 vs 串行最简判定
```python
ro = [tc for tc in tool_calls if registry._tools[tc.name].read_only]
rw = [tc for tc in tool_calls if not registry._tools[tc.name].read_only]
with ThreadPoolExecutor(max_workers=4) as ex:
    ro_results = list(ex.map(lambda tc: (tc.id, dispatch(tc)), ro))  # 只读并行
rw_results = [(tc.id, dispatch(tc)) for tc in rw]                     # 写操作串行
# 最后按 tool_calls 原始顺序重排回填,避免乱序破坏配对
```

#### 验收/测试方法
注册一个会抛异常的工具,断言 dispatch 返回的是 `{"error":...}` JSON 而非崩溃;断言 `get_definitions` 能按 toolset 过滤、`check_fn` 返回 False 的工具被排除。

---

### 3.3 Subagent / 委派

#### Hermes 真实做法(回顾)
Hermes 的 `delegate_task` 在 loop 内被拦截(属于 4 个 agent-loop 工具之一)。子 agent 是一个完整的 `AIAgent` 实例,与父**完全隔离**:全新对话(零父历史)、独立 terminal session、独立 toolset、独立 `task_id`,并以 `skip_context_files=True` / `skip_memory=True` 构造。

子代理的系统提示由 `_build_child_system_prompt()` 构造,开头是「You are a focused subagent working on a specific delegated task.」,然后是 `YOUR TASK:\n{goal}`、可选 `CONTEXT:` 块,并附上明确的总结指令——要求子 agent 完成后给出包含**四个固定段落**的总结。源码原文(`tools/delegate_tool.py`):

```
"Complete this task using the tools available to you. "
"When finished, provide a clear, concise summary of:\n"
"- What you did\n"
"- What you found or accomplished\n"
"- Any files you created or modified\n"
"- Any issues encountered\n\n"
... "Be thorough but concise -- your response is returned to the parent agent as a summary."
```

**结果契约**(源码 `_run_single_child()`):子 agent 的 `run_conversation()` 返回 dict 的 `final_response` 字段被取出作为 `summary`(`summary = result.get("final_response") or ""`),打包进 per-task 结果 dict(`{task_index, status, summary, exit_reason, tokens, ...}`),多个子任务的 dict 收集进 `results` 数组,最终用 `json.dumps({"results": [...], "total_duration_seconds": ...}, ensure_ascii=False)` 作为 tool_result 回填父。**父上下文只看到 delegation 调用和最终 summary,绝不看到子 agent 的中间推理或 tool 调用**(模块 docstring 原文:「The parent's context only sees the delegation call and the summary result, never the child's intermediate tool calls or reasoning.」)。

**toolset 父子交集**由 `_build_child_agent()` 强制,源码原文:`child_toolsets = [t for t in toolsets if t in expanded_parent]`,注释写「subagent must not gain tools the parent lacks」;复合 toolset(如 `hermes-cli`)会先 `_expand_parent_toolsets()` 展开成 `web`/`terminal` 等单名再求交。`DELEGATE_BLOCKED_TOOLS = frozenset(["delegate_task", "clarify", "memory", "send_message", "execute_code"])` 永远从子 agent 剥离(`delegate_task` 禁递归、`clarify` 禁用户交互、`memory` 禁写共享记忆等)。**独立 IterationBudget** 默认 50(源码常量 `DEFAULT_MAX_ITERATIONS = 50`,config `delegation.max_iterations`,且模型自报的 max_iterations 被忽略以保证可预测)。**递归深度** `MAX_DEPTH = 1`(源码注释:「flat by default: parent (0) -> child (1); grandchild rejected unless max_spawn_depth raised」),可通过 `delegation.max_spawn_depth`(1–3)放开,默认子代理是 `leaf` 角色不能再委派(若默认深度下传 `role="orchestrator"` 会静默降级为 leaf)。源码中 `delegate_task` 是**同步阻塞**的(父在 tool 调用里阻塞直到所有子完成,batch 模式用 `ThreadPoolExecutor`,`delegation.max_concurrent_children` 默认 3);Nous 后来于 2026 年 6 月 15 日通过 `async_delegation` toolset(GitHub Issue #5586)加了非阻塞路径。

#### MVP 如何简化
- **完全保留**:子 agent 复用 `Agent` 类、独立 messages/budget/session、toolset⊆父交集、`delegate_task` 自身从子剥离(禁递归)、final_response 作为 summary 回填、父只看 summary。
- **保留**:递归深度限制(MVP 直接 `depth >= 1` 就拒绝,等价 MAX_DEPTH=1)。
- **简化**:只做**同步阻塞的单子代理**(不做 batch 并行、不做 orchestrator 角色、不做 async)。M6 可加 ThreadPoolExecutor batch。
- **简化**:子 agent 系统提示直接硬编码上面那段四段式总结指令。

#### `delegate_task` 拦截 + 子 agent spawn 骨架

```python
# delegate.py
CHILD_SUMMARY_INSTRUCTION = (
    "You are a focused subagent working on a specific delegated task.\n"
    "YOUR TASK:\n{goal}\n\n{context_block}"
    "Complete this task using the tools available to you. "
    "When finished, provide a clear, concise summary of:\n"
    "- What you did\n- What you found or accomplished\n"
    "- Any files you created or modified\n- Any issues encountered\n\n"
    "Be thorough but concise -- your response is returned to the parent agent."
)
BLOCKED_FOR_CHILD = {"delegation", "memory", "clarify"}  # MVP 版剥离集

def handle_delegate_task(parent: "Agent", goal: str,
                         context: str = "", toolsets: list[str] | None = None) -> str:
    # 1. 递归深度限制(等价 MAX_DEPTH=1)
    if parent.depth >= 1:
        return json.dumps({"error": "max delegation depth reached"})

    # 2. toolset 父子交集 + 剥离 delegate_task(禁递归)
    requested = set(toolsets) if toolsets else set(parent.toolset)
    child_toolset = (requested & set(parent.toolset)) - BLOCKED_FOR_CHILD

    # 3. 构造隔离的子 agent(复用 Agent 类!),独立 session 走谱系链
    child_session = parent.db.create_session(
        source="subagent", parent_session_id=parent.session_id)
    child = Agent(session_id=child_session, db=parent.db,
                  toolset=child_toolset, provider=parent.provider,
                  model=parent.model, max_turns=10,   # 独立 budget
                  depth=parent.depth + 1)

    # 4. 同步阻塞跑完
    ctx_block = f"CONTEXT:\n{context}\n\n" if context else ""
    prompt = CHILD_SUMMARY_INSTRUCTION.format(goal=goal, context_block=ctx_block)
    result = child.run_conversation(prompt)

    # 5. final_response 作为 summary 回填(父只看到这个)
    return json.dumps({
        "status": "completed" if result.get("completed") else "incomplete",
        "summary": result["final_response"],
        "child_session_id": child_session,
    }, ensure_ascii=False)
```

在 `Agent._dispatch_tool` 里,先判断 `tc.name == "delegate_task"` → 调 `handle_delegate_task(self, **tc.arguments)`(拦截,不走 registry)。

#### 验收/测试方法
用 FakeLLM 让父 agent 发一个 `delegate_task`,子 agent 用另一个脚本跑完返回总结;断言:父 messages 里只有 delegation 的 tool_call + summary 字符串,**没有**子 agent 的中间 tool 调用;断言子 agent toolset 是父的子集;断言 depth=1 的子 agent 再调 delegate_task 被拒。

---

### 3.4 Self-improving(skills + memory)

#### Hermes 真实做法(回顾)
**Memory 子系统**:Hermes 维护两个 curated Markdown 文件 `MEMORY.md`(环境/约定/教训)和 `USER.md`(用户画像),存在 `$HERMES_HOME/memories/`。两者在 **session 启动时作为「冻结快照」注入系统提示**;mid-session 写入立即落盘但**直到下次 session 或 prompt 重建才进系统提示**(为了 prompt cache 稳定性)。`memory` 工具只支持 `add`/`replace`/`remove`(**没有 read**,因为已注入提示)。有**字符上限**:`memory_char_limit: 2200`(约 800 token)、`user_char_limit: 1375`(约 500 token);超限时工具**返回错误**而非静默丢弃,agent 必须自己腾地方(合并/删除)再重试。

**Skills 子系统**:skills 是 `~/.hermes/skills/` 下的 Markdown 文件,遵循 **agentskills.io 开放标准**和**渐进式披露(progressive disclosure)**:Level 0 系统提示只放每个 skill 的 name+description(整个目录约 3000 token);Level 1 用到时才 `skill_view(name)` 加载全文(建议 <5000 token);Level 2 按需加载 `references/`、`scripts/`、`assets/`。`SKILL.md` 格式是 YAML frontmatter(必填 `name`、`description`,name 须等于父目录名、小写字母数字连字符、≤64 字符;description ≤1024 字符且须说明「做什么 + 何时用」)+ Markdown 正文。

**自我改进闭环**:四个动作——任务完成后(5+ tool calls)写 skill;用到时精修 skill;**周期性 nudge**(`nudge_interval` 默认 10)agent 收到内部系统提示自评是否有值得持久化的东西;**回复交付后的后台 review**(`_spawn_background_review` 创建全新 AIAgent 跑 review,可写 memory/skill)。判断「该进哪层」的边界是「是否每次对话都需要」:每次都要→MEMORY/USER;特定话题才要→留在 session 归档靠 session_search 召回。

#### MVP 如何简化
- **Memory**:实现 `MEMORY.md` + `USER.md`,session 启动注入(冻结快照语义),`memory` 工具拦截支持 `add`/`replace`/`remove`,带字符上限 + 超限返回错误。
- **Skills**:实现 `~/.mini_agent/skills/*/SKILL.md` 扫描,解析 frontmatter 生成 Level-0 索引注入系统提示;`skill_view(name)` 工具加载全文(Level 1)。Level 2 引用文件先不做。
- **自我改进**:把「后台 review」简化为**同步触发**——纯文本回复返回前,若本轮 tool_calls 数 ≥ 阈值(如 5),追加一个内部 nudge 让模型决定是否 `memory.add` 或写 skill;或更简单地做成**手动 `/review` 命令**触发。不做独立后台线程/全新 AIAgent。

#### memory 工具与 skill 加载骨架

```python
# memory.py
class MemoryStore:
    MEMORY_LIMIT = 2200
    USER_LIMIT = 1375
    def __init__(self, home: Path):
        self.mem_path = home / "MEMORY.md"
        self.user_path = home / "USER.md"
    def snapshot(self) -> str:        # session 启动时调用,注入系统提示
        parts = []
        if self.mem_path.exists():
            parts.append("## Persistent Memory\n" + self.mem_path.read_text())
        if self.user_path.exists():
            parts.append("## User Profile\n" + self.user_path.read_text())
        return "\n\n".join(parts)
    def apply(self, action: str, target: str, content: str = "",
              old: str = "") -> str:
        path = self.mem_path if target == "memory" else self.user_path
        limit = self.MEMORY_LIMIT if target == "memory" else self.USER_LIMIT
        cur = path.read_text() if path.exists() else ""
        if action == "add":      new = (cur + "\n- " + content).strip()
        elif action == "replace": new = cur.replace(old, content)
        elif action == "remove":  new = cur.replace(old, "").strip()
        else: return json.dumps({"error": f"bad action {action}"})
        if len(new) > limit:     # 超限返回错误,逼模型自己腾地方
            return json.dumps({"error": f"{target} over {limit} chars; "
                               "consolidate or remove an entry first"})
        path.write_text(new)
        return json.dumps({"ok": True, "chars": len(new)})

# skills.py
def build_skill_index(skills_dir: Path) -> str:
    lines = ["## Skills (load with skill_view(name) when relevant)"]
    for skill_md in skills_dir.glob("*/SKILL.md"):
        fm = parse_frontmatter(skill_md.read_text())   # 取 name/description
        lines.append(f"- {fm['name']}: {fm['description']}")
    return "\n".join(lines)

@tool(toolset="skills", read_only=True)
def skill_view(name: str) -> str:
    """Load the full body of a skill by name (progressive disclosure L1)."""
    p = SKILLS_DIR / name / "SKILL.md"
    return p.read_text() if p.exists() else json.dumps({"error": "not found"})
```

`memory` 在 loop 内拦截(像 `delegate_task` 一样),因为它要访问 agent 持有的 `MemoryStore`。

#### SKILL.md 文件格式示例(agentskills.io 标准)
```markdown
---
name: code-review
description: Reviews code for bugs, security issues, and style violations. Use when the user asks to review code, check a PR, or find issues.
---

# Code Review
1. 先用 read_file 读目标文件
2. 检查:空指针、未处理异常、SQL 注入、硬编码密钥
3. 输出:按严重程度分级的 issue 列表 + 修复建议
```

#### 验收/测试方法
写一条 memory → 重启 agent → 断言新 session 的系统提示里包含该条;写超长 memory → 断言返回 error;放一个 SKILL.md → 断言系统提示出现其 name+description,`skill_view` 能取全文。

---

### 3.5 持久化与崩溃恢复

#### Hermes 真实做法(回顾)
Hermes 用单个 SQLite 库 `~/.hermes/state.db`(WAL 模式),`hermes_state.py` 封装,schema 已到 v11。核心表:`sessions`(元数据 + token/billing 统计 + `parent_session_id` 谱系链)、`messages`(完整消息历史,`tool_calls` 存 JSON 字符串)、`messages_fts`(FTS5 虚表 + trigram tokenizer 支持 CJK/子串检索)、`state_meta`、`schema_version`。设计决策:**WAL 模式**支持「多读一写」并发;**每轮即时落库**(append_message);谱系链由 `parent_session_id` 串起(压缩触发的 session split 会生成新 session)。写竞争处理:1 秒短超时 + 应用层重试(20–150ms 抖动,最多 15 次)+ `BEGIN IMMEDIATE` + 每 50 次写做一次 PASSIVE checkpoint。崩溃恢复靠 `get_messages_as_conversation()` 从 DB 重建 OpenAI 格式对话。

#### MVP 如何简化
- 保留 `sessions` + `messages` 两张核心表 + `parent_session_id` 谱系链。
- 保留 **WAL 模式** + **每条消息即时写库**(崩溃恢复的关键)。
- FTS5 列为可选(M6,配 `session_search` 工具)。
- 写竞争:MVP 单进程,不做复杂重试,但开 WAL + `synchronous=NORMAL`。

#### SQLite schema DDL

```sql
PRAGMA journal_mode=WAL;        -- 并发读 + 崩溃后能 replay
PRAGMA synchronous=NORMAL;      -- WAL 下兼顾性能与安全

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,                 -- 'cli' / 'subagent'
    parent_session_id TEXT,               -- 谱系链
    model TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,                      -- JSON 字符串
    tool_call_id TEXT,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
```

#### 持久化层骨架

```python
# persistence.py
class SessionDB:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA_DDL)
    def create_session(self, source, parent_session_id=None, model=None) -> str:
        sid = f"sess_{uuid4().hex[:12]}"
        self.conn.execute("INSERT INTO sessions(id,source,parent_session_id,"
            "model,started_at) VALUES (?,?,?,?,?)",
            (sid, source, parent_session_id, model, time.time()))
        return sid
    def append_message(self, session_id, role, content,
                       tool_calls=None, tool_call_id=None):
        self.conn.execute("INSERT INTO messages(session_id,role,content,"
            "tool_calls,tool_call_id,timestamp) VALUES (?,?,?,?,?,?)",
            (session_id, role, content,
             json.dumps([tc.model_dump() for tc in tool_calls]) if tool_calls else None,
             tool_call_id, time.time()))
    def load_conversation(self, session_id) -> list[Message]:
        rows = self.conn.execute("SELECT role,content,tool_calls,tool_call_id "
            "FROM messages WHERE session_id=? ORDER BY timestamp,id", (session_id,))
        return [_row_to_message(r) for r in rows]   # 崩溃恢复:重建 messages
```

> **WAL 崩溃恢复的原理(面试点)**:WAL 模式下,COMMIT 先把变更写进 `-wal` 文件并 flush 到磁盘,只有 commit 记录落盘后变更才算 durable;进程崩溃/掉电后,下次打开数据库时 SQLite 会自动「replay」WAL 里已提交但未 checkpoint 的事务到主库。这就是「每条消息即时 append + 崩溃后能恢复」成立的底层保证。注意 `synchronous=NORMAL` 在 WAL 下牺牲的是「掉电时最后几个事务的持久性」,但仍保证数据库不损坏,对 CLI agent 足够。

#### 验收/测试方法
跑一轮对话写若干消息 → `kill -9` 进程 → 重启用同 session_id 调 `load_conversation` → 断言 messages 完整、顺序正确、tool_calls 能反序列化。

---

## 四、上下文管理(简化版压缩)

#### Hermes 真实做法(回顾)
Hermes 有**双层压缩**:agent 内 `ContextCompressor`(默认 50% 阈值,API 实报 token)+ gateway session hygiene(85% 阈值,字符估算,安全网)。4 阶段算法:① 剪枝老 tool 结果(>200 char 的换成 `[Old tool output cleared to save context space]` 占位符,无 LLM 调用);② 确定边界(保护 `protect_first_n=3` 头部 + `protect_last_n=20` 尾部,边界对齐避免拆散 tool_call/tool_result 组,`_align_boundary_backward` 往回走到父 assistant 消息);③ 用辅助 LLM 把中间段生成结构化总结(Goal/Constraints & Preferences/Progress/Key Decisions/Relevant Files/Next Steps/Critical Context 等模板);④ 重组消息并 `_sanitize_tool_pairs()` 清理孤立配对(没结果的 tool_call 注入 stub 结果)。再压缩时把上次 summary 传给 LLM 让它「更新」而非重写。

#### MVP 如何简化(这是 ContextCompressor 的极简版)
- 单层、单阈值(如 token 数 > 模型上下文 50%)。
- 用 tiktoken 估 token(Anthropic 用字符/4 粗估)。
- 策略:保护头部前 2 条 + 尾部最近 N 条(按数量,不按 token budget),把中间段塞给 LLM 总结成**一条** summary 消息。
- **必须保护 tool_call/tool_result 配对**:边界往前/后对齐,确保不会把一个 assistant(带 tool_calls)和它的 tool 结果拆到压缩边界两侧——否则 OpenAI/Anthropic 都会 400。

```python
# context.py
class SimpleCompressor:
    def __init__(self, model, threshold=0.5, protect_last_n=8):
        self.ctx_limit = CONTEXT_LIMITS[model]   # e.g. 128000
        self.threshold = threshold
        self.protect_last_n = protect_last_n
    def count(self, messages) -> int:
        enc = tiktoken.encoding_for_model(self.model)
        return sum(len(enc.encode(m.content or "")) for m in messages) + \
               sum(200 for m in messages if m.tool_calls)   # 工具调用粗加
    def should_compress(self, messages) -> bool:
        return self.count(messages) > self.ctx_limit * self.threshold
    def compress(self, messages) -> list[Message]:
        head = messages[:2]                       # 保护 system 后首轮
        tail = messages[-self.protect_last_n:]
        middle = messages[2:-self.protect_last_n]
        # 关键:把 tail 起点往前对齐到一个 assistant 边界,避免拆散配对
        tail = _align_to_assistant_boundary(messages, tail)
        summary = call_llm_summarize(middle)      # 用 STRUCTURED 模板
        summary_msg = Message(role="user",
            content=f"[CONTEXT COMPACTION] Summary of earlier turns:\n{summary}")
        return head + [summary_msg] + tail
```

> **面试点**:能说出「为什么压缩要保护 tool_call/tool_result 配对」——因为 Anthropic 要求「tool_use 块后必须紧跟匹配数量的 tool_result 块」(否则报 `Did not find tool_result block(s) at the beginning of this message` 的 400),OpenAI 也要求 tool 消息必须对应前面的 tool_calls,拆散就 400。这是真实踩坑点。

---

## 五、测试与验证策略

### 5.1 Mock LLM(避免每次真调 API)
核心是一个 `FakeLLM`,按预设脚本依次返回响应,让你能确定性地测整个 loop:

```python
# tests/conftest.py
class FakeLLM:
    def __init__(self, scripted_responses: list):
        self.responses = scripted_responses
        self.calls = []
    def __call__(self, messages, tools, provider, model):
        self.calls.append(messages)
        return self.responses.pop(0)   # 预设的 tool_call 或文本响应

@pytest.fixture
def fake_llm(monkeypatch):
    def _make(responses):
        llm = FakeLLM(responses)
        monkeypatch.setattr("mini_agent.agent.call_llm", llm)
        return llm
    return _make
```

### 5.2 各子系统测试要点
- **agent loop**:脚本「tool_call→文本」,断言角色交替合法、终止正确、`max_turns` 截断。
- **tools**:抛异常工具返回 `{"error":...}`;`check_fn`/toolset 过滤生效;自动 schema 正确。
- **persistence**:写后用新连接 `load_conversation` 重建一致(模拟崩溃)。
- **context**:构造超阈值对话,断言压缩后 token 下降且 tool 配对完整。
- **delegate**:断言父只见 summary、子 toolset⊆父、depth 限制。
- **memory/skills**:断言注入、字符上限、skill 索引与全文加载。

### 5.3 端到端 smoke test
一个 `tests/test_smoke.py` 用真 API key(打 `@pytest.mark.skipif(no key)`)跑一轮「请创建 hello.txt 并写入 hi」,断言文件真被创建——验证整条链路。

### 5.4 「装好就能跑」最小 demo

```python
# main.py
def main():
    db = SessionDB(Path.home() / ".mini_agent" / "state.db")
    sid = sys.argv[2] if "--resume" in sys.argv else db.create_session("cli")
    agent = Agent(session_id=sid, db=db,
                  toolset={"file", "web", "skills"}, provider="openai")
    if "--resume" in sys.argv:
        agent.messages = db.load_conversation(sid)   # 崩溃恢复
    print(f"session {sid} — type 'exit' to quit")
    while True:
        user = input("> ")
        if user.strip() == "exit": break
        out = agent.run_conversation(user)
        print(out["final_response"])
```

---

## 六、学习价值与面试包装

### 6.1 能讲出的面试亮点(对照大厂 agent 工程考点)

| 考点 | 你能讲什么 |
|---|---|
| **Context 管理** | token 计数 + 阈值触发 + 老消息总结;尤其能讲「保护 tool_call/tool_result 配对避免 400」这个真实踩坑;并知道 Hermes 是双层(50%/85%)+ 4 阶段算法的生产版 |
| **工具并发安全** | 只读白名单可并行、写操作串行、结果按原序重排;Hermes 用 ThreadPoolExecutor + interactive 工具强制串行 |
| **子代理隔离 / 最小权限** | 子 agent 全新上下文、独立 budget、toolset⊆父交集、禁递归(剥离 delegate_task)、深度限制(MAX_DEPTH=1);父只见 summary 不见中间过程,省 context |
| **崩溃恢复** | 每条消息即时落库 + WAL 模式 + 从 DB replay 重建 messages;能讲 WAL 为何能崩溃恢复(commit 先写 WAL,重开时 replay) |
| **Prompt cache** | 系统提示前缀字节稳定才能命中缓存;Hermes「memory mid-session 写盘但不进当前系统提示」就是为了缓存稳定;据 Anthropic Prompt caching 官方文档:cache read 为标准输入价的 0.1×(省 90%)、cache write 为 1.25×(贵 25%)、默认 TTL 5 分钟(可选 1 小时为 2× write),最小可缓存前缀 Sonnet/Haiku 1024 token、部分模型需 2048–4096 token,break-even 约在第 2–3 次读取。对照 OpenAI:自动前缀缓存对 ≥1024 token 的 prompt 生效、cache write 免费、cache read 折扣(gpt-4o 系列约 50%),官方称长 prompt 可降延迟最多 80%、降成本约 50% |
| **健壮性契约** | 工具结果统一 JSON、两层异常包裹,模型永远收到合法返回(Hermes 原则:Handler MUST return JSON string,errors as `{"error":...}`) |
| **provider 抽象** | narrow waist:内部规范 transcript + 薄 adapter,换 provider 不改 loop |

### 6.2 简历 / 面试措辞建议
- 一句话定位:「从零手写了一个仿 Hermes Agent 架构的最小 agent 框架(纯 Python,不依赖 LangChain),实现了 agent loop、工具系统、子代理委派、自我改进记忆/技能闭环、SQLite 崩溃恢复五大子系统。」
- 强调权衡意识:「我研读了 Hermes 生产级源码,理解它的双层压缩 / 4-transport provider 抽象 / 8 个外部记忆 provider 等设计;我的 MVP 做了有意识的简化(单层压缩、2 个 provider、文件式记忆),并能说清每处简化的代价与适用边界。」——这种「知道生产怎么做 + 知道自己砍了什么」的对照,是最强的信号。

### 6.3 进阶方向(做完 MVP 继续加)
并行 batch 子代理(ThreadPoolExecutor + max_concurrent,默认 3)→ 两层压缩(preflight 50% + hygiene 85%)→ Anthropic prompt cache 实接(cache_control 打在系统提示末尾,5m/1h TTL)→ orchestrator 角色 + 深度 2 → session_search FTS5 工具 → 第二 surface(FastAPI HTTP)→ MCP 接入 → 后台 review 独立线程 → curator/skill 自动精修。

---

## 七、依赖与脚手架

### 7.1 pyproject.toml

```toml
[project]
name = "mini-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.40",
    "anthropic>=0.34",      # 可选第二 provider
    "pydantic>=2.7",
    "tiktoken>=0.7",
    "python-dotenv>=1.0",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]
```
（标准库即可覆盖:`sqlite3`、`concurrent.futures`、`inspect`、`pathlib`、`json`、`uuid`。）

### 7.2 项目初始化步骤清单
1. `mkdir mini_agent && cd mini_agent && python -m venv .venv && source .venv/bin/activate`
2. 创建上面的目录结构(先建空文件占位)。
3. `pip install -e ".[dev]"`;`.env` 里放 `OPENAI_API_KEY=...`(可选 `ANTHROPIC_API_KEY`、`ALLOW_WEB=1`)。
4. 按 **M0→M5** 顺序写,每个里程碑结束先写测试再 `pytest`,确认能 demo 再进下一阶段。
5. 每阶段提交一个 git tag(`m0-loop`、`m1-tools`…),方便面试时按演进讲解。
6. 写一个 `README.md` 记录「Hermes 真实做法 → 我的简化」对照表——这就是你的面试讲稿。

> **总原则(呼应 Anthropic《Building Effective Agents》)**:从最简方案起步,只在确有需要时增加复杂度(原文:「we recommend finding the simplest solution possible, and only increasing complexity when needed」);agent 的本质就是「LLMs using tools based on environmental feedback in a loop」,框架只是这个循环的封装。先把 50 行的循环吃透,再逐层加 memory、子代理、压缩,你对「chatbot 到 agent 的那条线」会有第一手的、能在面试里讲清楚的理解。

---

## 附:来源与确定性说明

- **Hermes 内部机制**(agent loop 序列、API mode、状态工具拦截、ContextCompressor 4 阶段与阈值、SQLite schema、prompt assembly 分层、memory/skills 闭环、delegate_task 源码细节如 `DEFAULT_MAX_ITERATIONS=50`/`MAX_DEPTH=1`/四段式总结模板/`final_response`→`summary`/`json.dumps({"results":...})` 结果契约/toolset 交集)均来自 Hermes 官方开发者文档(hermes-agent.nousresearch.com/docs)与 `tools/delegate_tool.py` 源码。其中部分版本相关数值(schema v11、nudge_interval=10、async_delegation Issue #5586 等)可能随版本演进,落地时建议对照你研读的具体 commit 复核。
- **SDK 行为**(OpenAI function calling/structured outputs 的 `{"type":"function",...}` 形状与 `parallel_tool_calls`;Anthropic Messages API 的 tool_use/tool_result 配对与 400 规则;prompt caching 的 0.1×/1.25× 价格、TTL、最小 token)来自 OpenAI、Anthropic 官方文档。
- **里程碑工作量估算、目录结构、`@tool` 装饰器与 `SimpleCompressor` 等代码骨架**为本方案的工程设计建议,非 Hermes 原样移植——它们是「照着 Hermes 思路自己写」的 MVP 实现,意在可落地与教学,具体接口可按你的习惯调整。