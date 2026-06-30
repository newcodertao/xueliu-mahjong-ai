import json
from pathlib import Path

from xueliu_ai.dataset.sample_manifest import build_sample_manifest
from xueliu_ai.game_logging.review_report import generate_markdown_report, summarize_jsonl


def test_build_sample_manifest(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    image_dir = raw / "my_hand"
    image_dir.mkdir(parents=True)
    (image_dir / "one.png").write_bytes(b"not really an image")
    output = tmp_path / "manifest.csv"

    result = build_sample_manifest(raw, output)

    text = result.read_text(encoding="utf-8")
    assert "one.png" in text
    assert "has_label" in text


def test_generate_markdown_report(tmp_path: Path) -> None:
    log = tmp_path / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "timestamp": "2026-06-25T21:00:00",
                "event": "advice",
                "recommended": "1W",
                "explanation": "推荐打 1W",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.md"

    assert summarize_jsonl(log) == {"advice": 1}
    result = generate_markdown_report(log, output)

    text = result.read_text(encoding="utf-8")
    assert "血流成河复盘报告" in text
    assert "推荐 `1W`" in text
