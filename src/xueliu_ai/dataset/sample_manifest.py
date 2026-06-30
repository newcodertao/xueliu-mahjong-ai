from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from xueliu_ai.paths import resolve_path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class SampleRow:
    path: str
    kind: str
    label_path: str
    has_label: bool


def build_sample_manifest(
    raw_dir: str | Path = "data/raw",
    output: str | Path = "data/sample_manifest.csv",
) -> Path:
    root = resolve_path(raw_dir)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[SampleRow] = []
    if root.exists():
        for image_path in sorted(root.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            kind = image_path.parent.name
            label_path = _guess_label_path(image_path)
            rows.append(
                SampleRow(
                    path=str(image_path),
                    kind=kind,
                    label_path=str(label_path),
                    has_label=label_path.exists(),
                )
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "label_path", "has_label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return output_path


def _guess_label_path(image_path: Path) -> Path:
    return resolve_path("data/labeled") / image_path.parent.name / f"{image_path.stem}.txt"
