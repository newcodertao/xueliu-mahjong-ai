from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from xueliu_ai.capture.roi_config import Roi, load_screen_profile
from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.paths import resolve_path
from xueliu_ai.realtime_table import (
    classify_table_zones,
    diagnose_zones,
    draw_table_overlay,
    reconcile_zone_tile_limits,
    visible_counts_from_zones,
)
from xueliu_ai.strategy.discard_advisor import advise_discard
from xueliu_ai.vision.detection_validator import non_max_suppression
from xueliu_ai.vision.yolo_detector import YoloDetector


@dataclass(frozen=True)
class ReplaySummary:
    video: str
    frames: int
    output_dir: str
    jsonl: str
    warnings: int


def replay_video(
    video_path: str | Path,
    model_path: str | Path,
    output_dir: str | Path = "data/replays/latest",
    every_seconds: float = 1.0,
    max_frames: int | None = 120,
    conf: float = 0.75,
    iou: float = 0.45,
    imgsz: int = 1280,
    save_images: bool = True,
    roi_name: str = "table",
) -> ReplaySummary:
    video = resolve_path(video_path)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "overlays"
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    profile = load_screen_profile()
    detector = YoloDetector(model_path, image_size=imgsz)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(fps * every_seconds))
    jsonl_path = output / "replay.jsonl"
    warnings = 0
    written = 0
    frame_index = 0

    with jsonl_path.open("w", encoding="utf-8") as fh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step != 0:
                frame_index += 1
                continue
            roi = _configured_or_fullscreen(profile.rois.get(roi_name), frame)
            crop = roi.crop(frame)
            detections = detector.detect_image(crop, conf=conf, iou=iou)
            detections = non_max_suppression([det for det in detections if det.label in TILE_SET], iou)
            zones = reconcile_zone_tile_limits(classify_table_zones(detections, crop.shape[1], crop.shape[0]))
            diagnostics = diagnose_zones(zones)
            if not diagnostics.valid:
                warnings += 1

            advice_payload = None
            if len(zones.hand) in diagnostics.expected_hand_counts and len(zones.hand) == 14 - diagnostics.open_melds * 3:
                try:
                    advice = advise_discard(
                        zones.hand,
                        visible_counts=visible_counts_from_zones(zones, include_hand=False),
                        open_melds=diagnostics.open_melds,
                    )
                    advice_payload = asdict(advice)
                except ValueError:
                    advice_payload = None

            payload = {
                "frame_index": frame_index,
                "time_seconds": frame_index / fps,
                "detections": len(detections),
                "zones": zones.to_dict(),
                "diagnostics": asdict(diagnostics),
                "advice": advice_payload,
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if save_images:
                overlay = draw_table_overlay(
                    crop,
                    detections,
                    zones,
                    advice_payload["recommended"] if advice_payload else None,
                    diagnostics.message(),
                )
                cv2.imwrite(str(image_dir / f"frame_{written:06d}.jpg"), overlay)

            written += 1
            frame_index += 1
            if max_frames is not None and written >= max_frames:
                break

    cap.release()
    summary = ReplaySummary(
        video=str(video),
        frames=written,
        output_dir=str(output),
        jsonl=str(jsonl_path),
        warnings=warnings,
    )
    (output / "summary.json").write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _configured_or_fullscreen(roi: Roi | None, frame) -> Roi:
    if roi and not roi.is_empty:
        return roi
    height, width = frame.shape[:2]
    return Roi(0, 0, width, height)
