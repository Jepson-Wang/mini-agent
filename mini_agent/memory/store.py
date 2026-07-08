SESSION_CREATE_SQL = (
    "CREATE TABLE sessions ( "
    "session_id   TEXT PRIMARY KEY,"
    "status       TEXT NOT NULL,      -- running / done / failed"
    "created_at   INTEGER NOT NULL,   -- epoch 秒"
    "updated_at   INTEGER NOT NULL);"
)

MESSAGES_CREATE_SQL = (
    "CREATE TABLE messages ("
    "session_id  TEXT    NOT NULL,"
    "seq  NOT NULL,"
    "role        TEXT    NOT NULL,"
    "role        TEXT    NOT NULL,     -- 'assistant' / 'tool' / 'user' / 'system'"
    "content       TEXT    NOT NULL,     -- 整条 message 序列化成的 JSON"
    "created_at  INTEGER NOT NULL,"
    "PRIMARY KEY (session_id, seq),"
    "FOREIGN KEY (session_id) REFERENCES sessions(session_id));"
)