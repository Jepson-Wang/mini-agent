"""[M2] CLI 入口（__main__.py）的接线测试。

这里测的不是持久化层的逻辑（那在 test_persistence.py），而是 **main() 把这些
零件按什么顺序装起来**。--resume 路径上 session_exists / recover /
reopen_session 三步的先后是有正确性含义的，装反了单看每个零件都对、合起来是错的。

不需要真 API key：只往 stdin 喂 "exit"，一次 LLM 都不会调。
"""
from __future__ import annotations

import dataclasses
import sys

import pytest

import mini_agent.__main__ as M
from mini_agent.persistence.session_db import MiniSessionDB, PersistenceError
from mini_agent.schema import Message


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """只做环境改写：把 CLI 的运行时状态搬到 tmp_path，塞个假 key 让 main() 肯往下走。

    settings 是 frozen dataclass，改不了字段，所以用 dataclasses.replace 造一个新的
    整体换掉 __main__ 模块里的那个名字。_db_path() 查的是模块全局 settings，
    所以它会跟着一起改。

    ★ 故意**不建库**。需要一个损坏库的测试必须自己往一个干净路径上写垃圾字节——
      要是这里先建了合法库，它的连接和 -wal 文件还开着，事后覆盖主库文件
      SQLite 会从 WAL 把内容replay 回来，垃圾字节根本不生效。
    """
    monkeypatch.setattr(
        M, "settings", dataclasses.replace(
            M.settings, home=tmp_path, deepseek_api_key="dummy-never-used"
        )
    )
    monkeypatch.setattr("builtins.input", lambda *a: "exit")
    return tmp_path


@pytest.fixture
def cli(cli_env):
    """cli_env + 一个建好的空库，返回该 db 句柄。"""
    return MiniSessionDB(str(cli_env / "state.db"))


def _status(db, sid):
    return db.conn.execute(
        "SELECT status FROM sessions WHERE session_id = ?;", (sid,)
    ).fetchone()["status"]


def test_resume_roundtrip_leaves_session_done(cli, monkeypatch, capsys):
    """正常 --resume 一个已结束的会话：跑完之后状态回到 done，历史读得回来。"""
    sid = cli.create_session()
    cli.append_message(sid, Message(role="user", content="q1"))
    cli.end_session(sid)

    monkeypatch.setattr(sys, "argv", ["mini_agent", "--resume", sid])
    M.main()

    assert _status(cli, sid) == "done"
    # readouterr() 会清空缓冲区，只能取一次
    captured = capsys.readouterr()
    assert "恢复会话" in captured.out
    # 正常路径不该有任何「上次未正常退出」的提示
    assert "未正常退出" not in captured.err


def test_failed_recover_does_not_touch_session_status(cli, monkeypatch, capsys):
    """recover 抛异常时，会话状态必须原封不动。

    ★ 这条测试钉的是 main() 里 recover 和 reopen_session 的**先后顺序**。
    reopen_session 是一个承诺——「我现在接管这个会话了」。如果它排在 recover 前面，
    recover 一抛就把会话永久卡在 running：
      - 状态撒谎（会话没在跑，标签写着 running）；
      - 下次 --resume 会误报「上次未正常退出」，而真相是 recover 坏了。
        我们刚花力气让那条提示只在真出事时响，装反顺序就又变成狼来了。
    recover 是纯读的，失败了库还和没跑过一样——把它排在前面，这个性质才用得上。
    """
    sid = cli.create_session()
    cli.end_session(sid)
    assert _status(cli, sid) == "done"

    def boom(self, session_id):
        raise PersistenceError("database disk image is malformed")

    monkeypatch.setattr(MiniSessionDB, "recover", boom)
    monkeypatch.setattr(sys, "argv", ["mini_agent", "--resume", sid])

    with pytest.raises(SystemExit) as exc:
        M.main()

    assert exc.value.code == 1
    assert _status(cli, sid) == "done"          # ← 没被 reopen 抢先改掉

    err = capsys.readouterr().err
    assert "无法恢复会话" in err
    # 恢复失败最吓人的是「我的东西是不是没了」——必须明确告诉用户数据还在，
    # 并给一条出路，而不是甩一坨 traceback。
    assert "没有被改动" in err
    assert "python -m mini_agent" in err


def test_corrupt_db_file_gives_a_readable_message_not_a_traceback(cli_env, monkeypatch,
                                                                  capsys):
    """库文件损坏时，用户该看到一句人话 + 一条出路，而不是一坨 traceback。

    这条钉的是 __main__ 和持久化层之间的异常契约：MiniSessionDB(...) 曾经会漏出
    裸 sqlite3.DatabaseError（_configure 不在 _db_guard 里），于是这里的
    except PersistenceError 接不住，精心写的提示一次都不会显示。
    """
    (cli_env / "state.db").write_bytes(b"SQLite format 3\x00" + b"\xde\xad" * 900)
    monkeypatch.setattr(sys, "argv", ["mini_agent"])

    with pytest.raises(SystemExit) as exc:
        M.main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "无法打开数据库" in err
    assert "删掉它即可重新开始" in err          # 必须给出路


def test_resume_nonexistent_session_exits_before_touching_anything(cli, monkeypatch):
    """--resume 一个不存在的 sid：早报错退出，不建会话、不写库。"""
    monkeypatch.setattr(sys, "argv", ["mini_agent", "--resume", "sess_ghost"])

    with pytest.raises(SystemExit) as exc:
        M.main()

    assert exc.value.code == 1
    assert cli.conn.execute("SELECT COUNT(*) FROM sessions;").fetchone()[0] == 0
