from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.paths import resolve_path
from xueliu_ai.realtime_table import (
    classify_table_zones,
    diagnose_zones,
    draw_table_overlay,
    reconcile_zone_tile_limits,
)
from xueliu_ai.table.game_phase import PhaseContext, should_allow_recommend
from xueliu_ai.vision.detection_validator import non_max_suppression
from xueliu_ai.vision.yolo_detector import YoloDetector


@dataclass(frozen=True)
class GoldCase:
    case_id: str
    image: str
    tags: list[str] = field(default_factory=list)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayTestSummary:
    total: int
    checked: dict[str, int]
    passed: dict[str, int]
    failures: list[dict[str, Any]]
    output_dir: str

    def metric(self, name: str) -> float | None:
        checked = self.checked.get(name, 0)
        if checked == 0:
            return None
        return self.passed.get(name, 0) / checked


def load_gold_cases(path: str | Path) -> list[GoldCase]:
    gold_path = resolve_path(path)
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    rows = payload.get("cases", payload) if isinstance(payload, dict) else payload
    return [
        GoldCase(
            case_id=str(row.get("id") or row.get("case_id") or row.get("image")),
            image=str(row["image"]),
            tags=list(row.get("tags", [])),
            expected=dict(row.get("expected", {})),
        )
        for row in rows
    ]


def run_replay_test(
    gold_path: str | Path = "data/gold_replay/gold_cases.json",
    model_path: str | Path = "models/yolo/xueliu_final325_plus_longjing39_plus83_clean_v1_0709.pt",
    output_dir: str | Path | None = None,
    conf: float = 0.75,
    iou: float = 0.45,
    imgsz: int = 1280,
    save_images: bool = True,
) -> ReplayTestSummary:
    gold = resolve_path(gold_path)
    cases = load_gold_cases(gold)
    output = resolve_path(output_dir) if output_dir else resolve_path(
        Path("reports") / "replay_failures" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)

    detector = YoloDetector(model_path, image_size=imgsz) if cases else None
    failures: list[dict[str, Any]] = []
    checked: dict[str, int] = {}
    passed: dict[str, int] = {}

    for case in cases:
        image_path = _case_image_path(gold, case)
        frame = cv2.imread(str(image_path))
        if frame is None:
            failure = {"id": case.case_id, "image": str(image_path), "error": "cannot_read_image"}
            failures.append(failure)
            continue

        assert detector is not None
        detections = detector.detect_image(frame, conf=conf, iou=iou)
        detections = non_max_suppression([det for det in detections if det.label in TILE_SET], iou)
        zones = reconcile_zone_tile_limits(classify_table_zones(detections, frame.shape[1], frame.shape[0]))
        diagnostics = diagnose_zones(zones)
        decision = should_allow_recommend(
            PhaseContext(
                zones=zones,
                diagnostics=diagnostics,
                stable=bool(case.expected.get("stable", True)),
                missing_suit=case.expected.get("missing_suit", "W"),
                detections=len(detections),
            )
        )
        actual = {
            "phase": decision.phase.value,
            "allow_recommend": decision.allow,
            "my_hand_count": len(zones.hand),
            "my_meld_count": diagnostics.open_melds,
            "diagnostics_valid": diagnostics.valid,
        }
        case_failures = evaluate_expected(case.expected, actual, checked, passed)
        if case_failures:
            failure = {
                "id": case.case_id,
                "image": str(image_path),
                "tags": case.tags,
                "expected": case.expected,
                "actual": actual,
                "failures": case_failures,
                "zones": zones.to_dict(),
                "diagnostics": asdict(diagnostics),
                "decision": asdict(decision),
            }
            failures.append(failure)
            (output / f"{_safe_name(case.case_id)}.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if save_images:
                overlay = draw_table_overlay(frame, detections, zones, None, decision.reason_text())
                cv2.imwrite(str(output / f"{_safe_name(case.case_id)}.jpg"), overlay)

    summary = ReplayTestSummary(
        total=len(cases),
        checked=checked,
        passed=passed,
        failures=failures,
        output_dir=str(output),
    )
    (output / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_expected(
    expected: dict[str, Any],
    actual: dict[str, Any],
    checked: dict[str, int] | None = None,
    passed: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    checked = checked if checked is not None else {}
    passed = passed if passed is not None else {}
    mapping = {
        "phase": "phase",
        "allow_recommend": "allow_recommend",
        "my_hand_count": "my_hand_count",
        "my_meld_count": "my_meld_count",
        "diagnostics_valid": "diagnostics_valid",
    }
    failures: list[dict[str, Any]] = []
    for expected_key, actual_key in mapping.items():
        if expected_key not in expected:
            continue
        checked[expected_key] = checked.get(expected_key, 0) + 1
        if expected[expected_key] == actual.get(actual_key):
            passed[expected_key] = passed.get(expected_key, 0) + 1
        else:
            failures.append(
                {
                    "field": expected_key,
                    "expected": expected[expected_key],
                    "actual": actual.get(actual_key),
                }
            )
    return failures


def _case_image_path(gold_path: Path, case: GoldCase) -> Path:
    image = Path(case.image)
    return image if image.is_absolute() else gold_path.parent / image


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)[:80]

