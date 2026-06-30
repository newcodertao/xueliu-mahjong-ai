from __future__ import annotations

from pathlib import Path

import cv2

from xueliu_ai.capture.roi_config import load_screen_profile
from xueliu_ai.paths import resolve_path
from xueliu_ai.vision.detection_types import Detection
from xueliu_ai.vision.yolo_detector import draw_detections


def show_debug_image(
    image_path: str | Path,
    detections: list[Detection] | None = None,
    profile_path: str | Path = "configs/screen_profile.yaml",
) -> None:
    image = cv2.imread(str(resolve_path(image_path)))
    if image is None:
        raise FileNotFoundError(resolve_path(image_path))

    profile = load_screen_profile(profile_path)
    for name, roi in profile.rois.items():
        if roi.is_empty:
            continue
        cv2.rectangle(image, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), (255, 0, 0), 2)
        cv2.putText(image, name, (roi.x, max(15, roi.y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if detections:
        image = draw_detections(image, detections)

    cv2.imshow("xueliu debug viewer", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
