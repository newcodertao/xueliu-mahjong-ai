from __future__ import annotations

import json
import shutil
from pathlib import Path

from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.detection_types import Detection


def export_detection_result(
    image_path: str | Path,
    detections: list[Detection],
    valid: bool,
    reason: str = "",
    output_dir: str | Path = "data/reviews/detections",
) -> Path:
    image = resolve_path(image_path)
    root = resolve_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{image.stem}.json"
    payload = {
        "image": str(image),
        "valid": valid,
        "reason": reason,
        "detections": [det.to_dict() for det in detections],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def copy_hard_case(
    image_path: str | Path,
    kind: str = "mis_detected",
    reason: str = "",
) -> Path:
    image = resolve_path(image_path)
    target_dir = resolve_path("data/hard_cases") / kind
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / image.name
    shutil.copy2(image, target)
    if reason:
        target.with_suffix(".txt").write_text(reason, encoding="utf-8")
    return target
