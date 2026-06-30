from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from xueliu_ai.paths import resolve_path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class DedupeResult:
    scanned: int
    kept: int
    skipped: int
    output_dir: str


def dedupe_images(
    source_dir: str | Path = "data/raw/my_hand",
    output_dir: str | Path = "data/curated/my_hand_unique",
    max_hamming_distance: int = 8,
) -> DedupeResult:
    source = resolve_path(source_dir)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    seen: list[np.ndarray] = []
    scanned = 0
    kept = 0
    for image_path in sorted(source.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        scanned += 1
        fingerprint = _dhash(image_path)
        if fingerprint is None:
            continue
        if any(_hamming_distance(fingerprint, old) <= max_hamming_distance for old in seen):
            continue
        seen.append(fingerprint)
        shutil.copy2(image_path, output / image_path.name)
        kept += 1
    return DedupeResult(scanned=scanned, kept=kept, skipped=scanned - kept, output_dir=str(output))


def _dhash(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    small = cv2.resize(image, (17, 16), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).astype(np.uint8).reshape(-1)


def _hamming_distance(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero(left != right))
