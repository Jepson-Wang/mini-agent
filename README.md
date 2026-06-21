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