from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.detection_validator import validate_hand_detections
from xueliu_ai.vision.yolo_detector import YoloDetector


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class BenchmarkCase:
    image: str
    expected: list[str]
    predicted: list[str]
    valid: bool
    exact: bool
    reason: str


@dataclass(frozen=True)
class BenchmarkSummary:
    total: int
    labeled: int
    exact: int
    valid: int
    exact_rate: float
    valid_rate: float
    cases: list[BenchmarkCase]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "labeled": self.labeled,
            "exact": self.exact,
            "valid": self.valid,
            "exact_rate": self.exact_rate,
            "valid_rate": self.valid_rate,
            "cases": [case.__dict__ for case in self.cases],
        }


def benchmark_hand_folder(
    model_path: str | Path,
    image_dir: str | Path = "data/raw/my_hand",
    label_dir: str | Path = "data/labeled/my_hand",
    conf: float = 0.6,
    iou: float = 0.5,
    output: str | Path = "data/reviews/benchmark.json",
) -> BenchmarkSummary:
    detector = YoloDetector(model_path)
    images = [
        path
        for path in sorted(resolve_path(image_dir).rglob("*"))
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    cases: list[BenchmarkCase] = []
    for image_path in images:
        expected = _read_expected_tiles(resolve_path(label_dir) / f"{image_path.stem}.txt")
        detections = detector.detect_image(image_path, conf=conf, iou=iou)
        result = validate_hand_detections(detections, conf, iou)
        exact = bool(expected) and expected == result.tiles
        cases.append(
            BenchmarkCase(
                image=str(image_path),
                expected=expected,
                predicted=result.tiles,
                valid=result.valid,
                exact=exact,
                reason=result.reason,
            )
        )

    labeled = sum(1 for case in cases if case.expected)
    exact_count = sum(1 for case in cases if case.exact)
    valid_count = sum(1 for case in cases if case.valid)
    summary = BenchmarkSummary(
        total=len(cases),
        labeled=labeled,
        exact=exact_count,
        valid=valid_count,
        exact_rate=exact_count / labeled if labeled else 0.0,
        valid_rate=valid_count / len(cases) if cases else 0.0,
        cases=cases,
    )
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, ensure_ascii=False, indent=2)
    return summary


def _read_expected_tiles(path: Path) -> list[str]:
    if not path.exists():
        return []
    tiles = [part.strip().upper() for part in path.read_text(encoding="utf-8").split(",")]
    return [tile for tile in tiles if tile in TILE_SET]
