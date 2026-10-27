"""[M2] SQLite 建表语句（DDL）。

对照 Hermes hermes_state.py：单库 + WAL 模式 + sessions/messages 两表。
messages 用 (session_id, seq) 复合主键，靠 seq 单调递增保证消息顺序，
content 直接存整条 Message 序列化后的 JSON。
"""
from __future__ import annotations

# 写进 PRAGMA user_version 的 schema 版本号，_create_tables 据此决定要不要迁移。
# 改动任何一张表的结构都要 +1，并在 _create_tables 的慢路径里补对应的迁移步骤。
#   0：旧库（executed_keys 以模型给的 tool_call.id 为主键）
#   1：executed_keys 改为 (session_id, assistant_seq, tool_index) 复合主键
SCHEMA_VERSION = 1

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
