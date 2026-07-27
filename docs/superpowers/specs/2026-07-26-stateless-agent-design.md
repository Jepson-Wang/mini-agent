# Agent 无状态化改造（面向 Go/gRPC 服务端）

> 2026-07-26 · 设计文档

## 1. 背景与目标

现在 `Agent` 是**有状态**的：`self.session_id` + `self.messages` 把一个实例绑死在一个会话上，CLI 里一个进程服务一个 session。

服务化之后，后端是 **Go**，通过 **gRPC** 调用 Python 的 agent 程序。高并发下「一个 session 一个 Agent 实例」不成立：

- 每个活跃 session 的完整 transcript 常驻内存，无界增长；
- 进程重启即全丢，已写好的 SQLite 持久化 + `recover` 只在冷启动兜底，热路径没上保险；
- 实例的 session 状态绑在某个进程/pod 上，横向扩容要么上 sticky 路由，要么在「错的」worker 上还是得从库加载。

**目标：把 `Agent` 改造成无状态 handler —— 一个实例服务所有 session，`session_id` 由服务端每次显式传入，会话状态的唯一真相是数据库。**

## 2. 核心决策

**缓存数据，不缓存 handler。**

- **真相唯一在 SQLite。** session 状态 = 库里的行。
- **`Agent` = 无状态 handler。** 只持有跨 session 共享的配置与依赖（`db` / `toolset` / `system_prompt` / `max_turns`）。
- **`session_id` + 该会话历史 = 每次调用的局部变量。**

### 被否决的方案

**`dict[session_id → Agent]`（服务端维护实例表）** —— 单进程、中等并发下够用且好写，但与「面对高并发」的初衷相悖：

1. 内存随活跃 session 无界增长，终究要加 LRU+TTL；一旦淘汰，miss 时还是得 `recover()`，等于在 recover 前面又加了一个缓存失效问题。
2. 进程重启 dict 全没；持久化层的价值被降级为冷启动兜底。
3. 多 worker/多 pod 下同 session 的实例只活在某一个进程里，扩不动。

**服务端传 `messages` 进来** —— 最灵活，但把状态管理责任推给每个调用方；Go 侧要重复实现加载/缓存逻辑，且 Python 的 `recover` 补桩能力被绕过。

**未来的优化路径**（现在不做）：若 profiling 证明每回合 `recover` 太贵（短期内不会——一次本地 WAL 索引读 vs 数秒级 LLM 调用），在库前面挂**有界 LRU+TTL 的 messages 缓存**。缓存的是数据不是 handler，有界，且库始终权威，所以冷 worker / 缓存失效只是重新加载，**不需要 sticky 路由**。

## 3. 详细设计

### 3.1 `Agent` 字段拆分

| 类别 | 字段 | 去向 |
|---|---|---|
| 跨 session 共享 | `db`、`toolset`、`system_prompt`、`max_turns` | 留在实例上，构造一次、所有 session 复用 |
| 单次调用易变 | `session_id`、`messages` | 拆下来，`run_conversation` 局部持有 |
| 废弃 | `initial_messages`、`depth` | 删除 |

- `initial_messages`（原 `--resume` 播种用）废弃：新设计里历史统一由 `run_conversation` 内部 `recover()` 加载。
- `depth` 废弃：它属于 session 的属性，存在 `sessions` 表里即可，不必挂在 handler 上。
- **`db` 从可选改为必需**（关键字参数，无默认值）。纯内存多轮模式取消——`sessions` 表的 `parent_session_id`/`depth` 说明连未来的子代理也要落库，没有任何东西真的需要它。收益：`_record` 和 `_execute_tool_calls` 里的 `if self.db is not None and self.session_id is not None` 两处分支消失。

### 3.2 单次调用的状态载体 `_Turn`

`self.messages`/`self.session_id` 拆下来后，`_record` / `_call_model` / `_execute_tool_calls` 不能再读 `self`。不给每个 helper 塞一串参数，而是引入一个小状态对象串下去：

```python
@dataclass
class _Turn:
    """一次 run_conversation 的全部易变状态。Agent 实例不再持有它。"""
    session_id: str
    messages: list[Message]
```

helper 签名统一改为 `_record(self, turn, msg)` / `_call_model(self, turn, tools)` / `_execute_tool_calls(self, turn, assistant)`，内部把 `self.messages` 换成 `turn.messages`、`self.session_id` 换成 `turn.session_id`。

### 3.3 `run_conversation` 新签名与流程

```python
def run_conversation(self, session_id: str, user_message: str) -> dict:
    self.db.ensure_session(session_id)        # ① 幂等补行，满足 FK
    history = self.db.recover(session_id)     # ② 唯一真相在库；新 session 返回 []
    turn = _Turn(session_id=session_id, messages=history)

    if not _turn_in_flight(history):          # ③ 重试保护，见 3.5
        self._record(turn, Message(role="user", content=user_message))

    # ④ 以下为原 ReAct 循环，仅把 self.messages 换成 turn.messages
    ...
```

返回值不变：`{final_response, completed, used_turns}`。

**三个白捡的好处：**

1. `recover()` 一句话把「新建 / resume」两条路合并 —— 新 session 返回 `[]`，老 session 返回历史。`__main__` 里 create-vs-resume 的分支消失。
2. `recover()` 本就带**崩溃补桩 + 幂等表回填**。「上一次调用跑到一半进程挂了」这个 case，在每一回合开头**免费**处理掉：加载即修复，无需另写恢复逻辑。
3. `Agent` 成为可被任意并发请求共享的无状态 handler，正好贴 gRPC servicer 的形状。

### 3.4 `ensure_session`（持久化层新增）

**session 身份归 Go**：Go 生成 id、存自己的业务库、管产品级生命周期、驱动前端创建流程。**创建过程零 gRPC 往返**。

但 Python 的 SQLite 有外键约束（`messages.session_id → sessions.session_id`，且 `PRAGMA foreign_keys=ON`）。Go 那边「有」这个 session ≠ Python 的库里「有」——直接读没问题（`recover` 返回 `[]`），但首轮 `append_message` 会因 `sessions` 表缺行抛 `IntegrityError`（收敛为 `PersistenceError`）。**第一回合必炸。**

因此新增幂等的 `ensure_session`：

```python
def ensure_session(self, session_id: str, parent_session_id=None, depth=0) -> bool:
    """幂等地保证 sessions 表里有这一行；返回本次是否新建。

    Python 的 sessions 行降级为 Go session 的从属子记录，只为满足 FK 约束
    并存放 depth/parent 等 agent 侧属性。Go 给什么 id 就落什么 id。

    先 SELECT 探一次：老 session 的热路径（绝大多数请求）零写锁开销 ——
    照抄 _create_tables 的套路。竞态安全：两个进程可能都 SELECT-miss、
    都去 INSERT，ON CONFLICT DO NOTHING 保证幂等，最多白跑一次。
    """
```

比「Python 查不到就报错、让前端回头找 Go 创建」更好：**少一次协调往返**，Go 不用先探「Python 建过没」，直接发对话，Python 自愈式补行。

代价：buggy 的 Go 发来垃圾 sid 会静默建垃圾行。但 gRPC 是内网服务间调用，边界可信，加个 sid 格式校验足够，不值得为它引入额外 RPC。

**`create_session` 原样保留，不重构。** 两者并存、职责不同：

| 函数 | 职责 | 调用方 |
|---|---|---|
| `create_session()` | 自己 mint id + insert | 测试、CLI |
| `ensure_session(sid)` | 外部 id 的幂等 upsert | `run_conversation` |

有轻微重复，但换来 590 行 `test_persistence.py` 零改动。

### 3.5 重试保护：`_turn_in_flight`

**问题推演：**

```
RPC(sid, "帮我查天气") → 落库 user 消息 → LLM 调用超时 → 返回 UNAVAILABLE
Go 重试 RPC(sid, "帮我查天气") → recover 得到 [user "帮我查天气"]
                              → 又 _record 一条 user
                              → transcript = [user, user]   ← 重复
```

崩在更后面（工具已跑完）时更难看：`[.., user, assistant(tc), tool, user]`。高并发 + 自动重试的生产环境，这个必然发生。

**判据用主循环自己的终止条件：transcript 以「纯文本 assistant」结尾 ⟺ 上一回合已完成。**

```python
def _turn_in_flight(history: list[Message]) -> bool:
    """上一回合是否没跑完 —— 主循环终止条件的反面。"""
    if not history:
        return False                      # 新 session，不算 in-flight
    last = history[-1]
    return not (last.role == "assistant" and not last.tool_calls)
```

in-flight 时**不追加 user 消息**，直接从补桩后的历史续跑。与 `recover` 的补桩天然咬合：补桩负责把残缺轮补成合法可发送形状，这一步负责不重复计入用户输入。

**代价（明确接受）：** 若 Go 在未完成的回合上重试时换了条不同的消息，新消息会被丢弃、转而续跑旧回合。可接受——回合没结束本就不该发新消息，且 Go 侧已保证同 session 串行。

**升级路径（现在不做）：** Go 传 `request_id` 做 RPC 级幂等（复用 `executed_keys` 的思路）。需要 proto 字段 + 新表 + Go 侧纪律，YAGNI。

### 3.6 并发控制

**唯一机制：Go 侧保证同一 session 串行。** 一段对话本就是顺序的，而 Go 握着 session 身份和路由，天生该由它保证。

**Python 侧不加任何兜底锁。** 理由：

- **进程内 `dict[sid → Lock]`** 只在同一进程内有效。Python 一旦多实例（无状态化的初衷正是能多实例），同 session 的两个并发请求落到不同 pod 时互相看不见 —— 是**虚假的安全感**。
- **CAS 状态机当锁**（`reopen_session`/`end_session`）虽然锁在共享库里能跨实例，但有两个硬伤：其一，**崩溃即砖** —— 进程中途挂掉 status 永远卡在 `running`，此后每次 CAS 都失败，session 被永久锁死，与既有的崩溃恢复属性直接打架；其二，生产环境数据库不一定只有一个，锁在 shard A 的库里管不住打到 shard B 的请求。跨实例真锁应当是外部的（Redis/etcd），那是 Go 的地盘。

**`status` 字段的归属：** 服务化路径下不再由 Python 驱动 —— `ensure_session` 建行时置 `running`，之后 Python 不再改它，产品级生命周期归 Go。CLI surface 保留 `finally: end_session(...)` 的既有行为。

## 4. 错误处理

| 情形 | 处理 |
|---|---|
| `PersistenceError` | 向上抛；未来 servicer 层映射为 gRPC `INTERNAL` |
| LLM 调用失败（超时/429/网络） | 向上抛；未来映射为 `UNAVAILABLE` / `DEADLINE_EXCEEDED` |
| 工具异常 | **不存在** —— `registry.dispatch` 契约保证永不抛，错误回填为 `{"error": ...}` |
| 迭代预算耗尽 | **不是错误** —— 正常返回 `completed=false` |

**铁律（服务化时生效）：** servicer 层必须兜住一切异常 —— 一个请求炸掉不能带走整个 server。现在 REPL 里 LLM 异常是直接往上冒的，服务化后不可接受。

**自愈属性：** 因每条消息即时落库、且 `recover` 会补桩，一次失败的调用留下的是**可恢复的残缺 transcript**。下次调用开头的 `recover` + `_turn_in_flight` 自动把它补齐并避免重复。Go 直接重试即可，零额外机制。

## 5. 入口 / surface

本次范围内 **`__main__.py` 仍是 CLI REPL**，只做适配：

- 启动时 mint 一个 session_id（或 `--resume <sid>` 沿用既有 id），然后循环调 `agent.run_conversation(sid, user_input)`；
- create-vs-resume 的分支收敛：`run_conversation` 内部统一 `ensure_session` + `recover`，CLI 不再需要按分支决定是否传 `initial_messages`；
- `--resume` 时保留 `session_exists` 校验以给出友好的「会话不存在」提示（`ensure_session` 会静默建行，对 CLI 来说不是好 UX）；
- `finally: end_session(sid)` 保留。

**gRPC server 不在本次范围内。** 它落地时作为第二个瘦 surface（`mini_agent/server.py`）套在同一个无状态 `Agent` 上 —— 这正是蓝图「一个 Agent 类、多个 surface」原则的兑现。

## 6. 测试策略

### 爆炸半径

| 文件 | 行数 | 命运 |
|---|---|---|
| `test_persistence.py` | 590 | **零改动**（`create_session` 保留） |
| `test_crash_real_kill.py` + `crash_victim.py` | 353 | **零改动**（db 层，未构造 `Agent`） |
| `test_file_tools.py` | 94 | 零改动 |
| `test_loop.py` | 78 | 重写 4 处 `Agent()` |
| `test_tools.py` | 4 处 | 重写；`test_initial_messages_...` 作废并替换 |
| `test_cli.py` | 134 | 跟随 REPL 适配（它测的正是被收敛掉的 create-vs-resume 顺序） |

约 1037 行不受影响。

### 迁移手法

- `conftest.py` 增加一个 `tmp_path` SQLite 的 db fixture，供 `test_loop` / `test_tools` 复用。
- 断言 `agent.messages` 的地方改为从库读（`db.recover(sid)`）。此 pattern 代码里已有先例（`test_tools.py` 的「新连接 = 新进程：只信磁盘」），且**比原来更强** —— 验的是磁盘上的事实而非内存副本。
- `test_initial_messages_seed_memory_without_rewriting_history` 作废，但它守的不变量仍然成立，改写为：**对已有历史的 session 调 `run_conversation`，不得重复写入历史行**。

### 新增测试

1. **`test_one_agent_serves_multiple_sessions`** —— 同一个 `Agent` 实例交替跑两个 `session_id`，断言历史互不串扰。**这是本次重构的核心验收。**
2. **`test_retry_does_not_duplicate_user_message`** —— 模拟首次 LLM 抛异常，用同一条消息重试，断言库里 user 消息只有一条。
3. **`test_ensure_session_is_idempotent`** —— 重复调用不新增行、返回值正确。
4. **`test_run_conversation_on_unknown_session_creates_row`** —— 外部 id 首轮不触发 FK 错误。

## 7. 明确不做（YAGNI）

- `.proto` 契约与 gRPC server 实现（后续独立进行）
- Python 侧任何形式的并发锁 / CAS 租约
- messages 的 LRU+TTL 缓存
- RPC 级 `request_id` 幂等
- 多 provider 适配（项目硬约束：DeepSeek-only）

## 8. 待定

- 生产环境「数据库不止一个」若成真，`Agent` 实际只依赖 `db` 的 4 个方法（`ensure_session` / `recover` / `append_message` / `record_executed_key`）。是否把这个接缝显式化为 `typing.Protocol`，本次未决，留待 gRPC server 落地时一并考虑。
