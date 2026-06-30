from pathlib import Path

from xueliu_ai.vision.detection_exporter import copy_hard_case, export_detection_result
from xueliu_ai.vision.detection_types import Detection


def test_export_detection_result(tmp_path: Path) -> None:
    image = tmp_path / "hand.png"
    image.write_bytes(b"fake")
    output_dir = tmp_path / "detections"

    output = export_detection_result(
        image,
        [Detection("1W", 0.9, 1, 2, 3, 4)],
        valid=True,
        output_dir=output_dir,
    )

    assert output.exists()
    assert '"label": "1W"' in output.read_text(encoding="utf-8")


def test_copy_hard_case(tmp_path: Path) -> None:
    image = tmp_path / "bad.png"
    image.write_bytes(b"fake")

    target = copy_hard_case(image, kind="low_confidence", reason="too low")

    assert target.exists()
    assert target.with_suffix(".txt").read_text(encoding="utf-8") == "too low"
