import json

from xueliu_ai.game_logging.game_logger import GameLogger


def test_game_logger_rotates_by_size_and_keeps_bounded_backups(tmp_path) -> None:
    path = tmp_path / "realtime_ui.jsonl"
    logger = GameLogger(path, max_bytes=150, backup_count=2)

    for index in range(12):
        logger.log("tick", {"index": index, "message": "牌" * 30})

    assert path.exists()
    assert path.with_name("realtime_ui.jsonl.1").exists()
    assert path.with_name("realtime_ui.jsonl.2").exists()
    assert not path.with_name("realtime_ui.jsonl.3").exists()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["index"] == 11


def test_game_logger_can_truncate_instead_of_keep_backups(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    logger = GameLogger(path, max_bytes=100, backup_count=0)
    logger.log("first", {"payload": "x" * 200})
    logger.log("second", {"payload": "ok"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["second"]
