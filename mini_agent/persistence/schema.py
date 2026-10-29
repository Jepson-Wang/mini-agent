"""[M2] SQLite 建表语句（DDL）。

对照 Hermes hermes_state.py：单库 + WAL 模式 + sessions/messages 两表。
messages 用 (session_id, seq) 复合主键，靠 seq 单调递增保证消息顺序，
content 直接存整条 Message 序列化后的 JSON。
"""
from __future__ import annotations

# 写进 PRAGMA user_version 的 schema 版本号，_create_tables 据此决定要不要迁移。
# 改动任何一张表的结构都要 +1，并在 _create_tables 的慢路径里补对应的迁移步骤。
#   0：旧库（executed_keys 以模型给的 tool_call.id 为主键）；
#   1：executed_keys 改为 (session_id, assistant_seq, tool_index) 复合主键；
#   2：新增 compactions 表（M3 压缩检查点）。
# 约定：每一步都用 IF NOT EXISTS + 改名探测写成幂等的"收敛式"迁移，单一慢路径就够了；
# 等哪天出现非加法变更（改列 / 改约束 / 需要搬数据），再重构成 {目标版本: 迁移函数}。
SCHEMA_VERSION = 2

SESSION_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id         TEXT    PRIMARY KEY,
    status             TEXT    NOT NULL,          -- running / done / failed
    created_at         INTEGER NOT NULL,          -- epoch 秒
    updated_at         INTEGER NOT NULL,
    
    parent_session_id  TEXT,                       -- 子代理指向其父 session；root 为 NULL
    depth              INTEGER NOT NULL DEFAULT 0,  -- 递归深度，root=0，每下一层 +1
    spawn_budget_used  INTEGER NOT NULL DEFAULT 0,  -- 全树已 spawn 的子代理数，只在 root 行有意义
    FOREIGN KEY (parent_session_id) REFERENCES sessions(session_id)
);
"""

MESSAGES_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    session_id  TEXT    NOT NULL,
    seq         INTEGER NOT NULL,
    role        TEXT    NOT NULL,     -- 'system' / 'user' / 'assistant' / 'tool'
    content     TEXT    NOT NULL,     -- 整条 message 序列化成的 JSON
    created_at  INTEGER NOT NULL,
    PRIMARY KEY (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

# 幂等键完全由我们自己的数据构成，主键里没有任何模型输入：
#   旧版主键是模型给的 tool_call.id —— 跨会话撞 id 会串结果（隐私泄露），同会话
#   跨轮复用 id 会让后一轮的写入被 ON CONFLICT 吞掉。
# 外键让数据库保证每条记录都指向一条真实存在的 assistant 消息。
EXECUTED_KEYS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS executed_keys (
    session_id     TEXT    NOT NULL,
    assistant_seq  INTEGER NOT NULL,   -- 发起这批 tool_calls 的 assistant 消息在 messages 里的 seq
    tool_index     INTEGER NOT NULL,   -- 该 tool_call 在 assistant.tool_calls 里的下标
    tool_call_id   TEXT    NOT NULL,   -- 模型给的 id：不参与主键，只用于读侧校验 + 排查
    result         TEXT    NOT NULL,   -- 这次调用产生的结果(JSON)
    created_at     INTEGER NOT NULL,
    PRIMARY KEY (session_id, assistant_seq, tool_index),
    FOREIGN KEY (session_id, assistant_seq) REFERENCES messages(session_id, seq)
);
"""

# [M3] 压缩检查点。原始 messages 一条不动，压缩结果作为一条新事实追加进来：
#   加载时读「最新检查点 summary + seq >= first_kept_seq 的消息」（快照 + 增量日志）。
#
# 边界列是 first_kept_seq（第一条**保留**的 seq），不是"最后一条被压的 seq"：
#   外键让数据库保证边界落在一条真实存在的消息上，所以 tail 至少有这一行；
#   但它**不**保证这一行是 user/assistant —— 那条边界不变量由写侧（compressor/DAO）
#   保证，读侧校验并降级；要校验的恰好就是外键指向的这一行，一次查询。
#   messages 不存 system prompt（它在 _call_model 里现拼），边界不会和 system 纠缠。
#
# 主键天然幂等：同一边界重复写 → 冲突，写侧可以 ON CONFLICT DO NOTHING；
#   "取最新检查点" = ORDER BY first_kept_seq DESC LIMIT 1，直接走主键 B-tree。
# 只追加、不覆盖：审计/排障时能看到每次压缩把什么压成了什么；回退靠的是原始历史
#   还在、可以重新压缩，不是读路径的能力。
#
# ★ 与 executed_keys 一样，外键是**连接级开关**：MiniSessionDB._configure 里
#   PRAGMA foreign_keys=ON 才对"走我们连接"的写入生效，外部 sqlite3 CLI 默认不检查。
# ★ 不加 ON DELETE 动作（默认 NO ACTION）：messages 只追加，现在没有删除路径；
#   将来要清理时，必须**先删 compactions、再删 messages**，否则外键会拒绝删除。
# ★ summary 只存纯文本：渲染成 role="user" + [CONTEXT COMPACTION] 前缀是代码的事，
#   以后改渲染格式不需要迁移数据。
#   CHECK 挡住空 / 纯空白 summary（写侧 bug 的 fail-fast）。trim 必须带第二个参数：
#   单参 trim() 只去空格，trim('\n') 仍然是 '\n'。它是最后一道防线，读侧仍按
#   recover 金规则降级，不依赖它。
# ★ 单调性（新检查点 first_kept_seq 必须大于已有最大值）不在表上约束 —— CHECK 不能
#   引用其他行，留给 DAO 用条件 INSERT 保证。
COMPACTIONS_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS compactions (
    session_id      TEXT    NOT NULL,
    first_kept_seq  INTEGER NOT NULL,   -- 压缩边界：seq < 它的消息由 summary 代表，seq >= 它的原样保留
    summary         TEXT    NOT NULL CHECK (length(trim(summary, char(32, 9, 10, 12, 13))) > 0),
    created_at      INTEGER NOT NULL,   -- epoch 秒，同其他表
    PRIMARY KEY (session_id, first_kept_seq),
    FOREIGN KEY (session_id, first_kept_seq) REFERENCES messages(session_id, seq)
);
"""
