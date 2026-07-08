"""[M2] SQLite 建表语句（DDL）。

对照 Hermes hermes_state.py：单库 + WAL 模式 + sessions/messages 两表。
messages 用 (session_id, seq) 复合主键，靠 seq 单调递增保证消息顺序，
content 直接存整条 Message 序列化后的 JSON。
"""
from __future__ import annotations

SESSION_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT    PRIMARY KEY,
    status      TEXT    NOT NULL,     -- running / done / failed
    created_at  INTEGER NOT NULL,     -- epoch 秒
    updated_at  INTEGER NOT NULL
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
