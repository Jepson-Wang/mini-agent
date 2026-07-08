SESSION_CREATE_SQL = (
    "CREATE TABLE sessions ( "
    "session_id   TEXT PRIMARY KEY,"
    "status       TEXT NOT NULL,      -- running / done / failed"
    "created_at   INTEGER NOT NULL,   -- epoch 秒"
    "updated_at   INTEGER NOT NULL);"
)

