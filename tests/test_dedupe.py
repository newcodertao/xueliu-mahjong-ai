from pathlib import Path

import cv2
import numpy as np

from xueliu_ai.dataset.dedupe import dedupe_images


def test_dedupe_images_copies_unique_samples(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    first = np.zeros((32, 32, 3), dtype=np.uint8)
    second = np.zeros((32, 32, 3), dtype=np.uint8)
    second[:, 16:] = 255
    cv2.imwrite(str(source / "a.png"), first)
    cv2.imwrite(str(source / "b.png"), first)
    cv2.imwrite(str(source / "c.png"), second)

    result = dedupe_images(source, output, max_hamming_distance=0)

    assert result.scanned == 3
    assert result.kept == 2
    assert len(list(output.glob("*.png"))) == 2
