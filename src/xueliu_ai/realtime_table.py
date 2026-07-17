from __future__ import annotations

import json
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import cv2
import numpy as np

from xueliu_ai.capture.roi_config import Roi, load_screen_profile
from xueliu_ai.capture.screen_capture import ScreenCapture
from xueliu_ai.game_logging.game_logger import GameLogger
from xueliu_ai.mahjong.tiles import TILE_SET
from xueliu_ai.strategy.discard_advisor import advise_discard
from xueliu_ai.table.meld_grouper import group_melds
from xueliu_ai.table.structured_types import MeldGroup, ZoneTile
from xueliu_ai.table.zone_assigner import assign_by_roi_priority
from xueliu_ai.vision.detection_types import Detection
from xueliu_ai.vision.detection_validator import non_max_suppression
from xueliu_ai.vision.yolo_detector import YoloDetector


AUTO_HAND_MIN_Y = 0.72
AUTO_DISCARD_MIN_X = 0.25
AUTO_DISCARD_MAX_X = 0.75
AUTO_DISCARD_MIN_Y = 0.16
AUTO_DISCARD_MAX_Y = 0.72


@dataclass(frozen=True)
class TableZones:
    hand: list[str]
    bottom_melds: list[str]
    left_melds: list[str]
    right_melds: list[str]
    top_melds: list[str]
    center_discards: list[str]
    all_tiles: list[str]
    my_discards: list[str] = field(default_factory=list)
    left_discards: list[str] = field(default_factory=list)
    top_discards: list[str] = field(default_factory=list)
    right_discards: list[str] = field(default_factory=list)
    unknown_tiles: list[str] = field(default_factory=list)
    candidate_meld_tiles: list[str] = field(default_factory=list)
    hu_display_tiles: list[str] = field(default_factory=list)
    event_tiles: list[str] = field(default_factory=list)
    table_decor_tiles: list[str] = field(default_factory=list)
    zone_tiles: list[ZoneTile] = field(default_factory=list)
    meld_groups: list[MeldGroup] = field(default_factory=list)

    @property
    def confirmed_open_melds(self) -> int:
        return sum(1 for group in self.meld_groups if group.zone == "bottom_melds" and group.is_confirmed)

    @property
    def suspected_open_melds(self) -> int:
        return sum(1 for group in self.meld_groups if group.zone == "bottom_melds" and group.is_suspected)

    def to_dict(self) -> dict[str, object]:
        return {
            "hand": self.hand,
            "bottom_melds": self.bottom_melds,
            "left_melds": self.left_melds,
            "right_melds": self.right_melds,
            "top_melds": self.top_melds,
            "center_discards": self.center_discards,
            "my_discards": self.my_discards,
            "left_discards": self.left_discards,
            "top_discards": self.top_discards,
            "right_discards": self.right_discards,
            "unknown_tiles": self.unknown_tiles,
            "candidate_meld_tiles": self.candidate_meld_tiles,
            "hu_display_tiles": self.hu_display_tiles,
            "event_tiles": self.event_tiles,
            "table_decor_tiles": self.table_decor_tiles,
            "all_tiles": self.all_tiles,
            "zone_tiles": [tile.to_dict() for tile in self.zone_tiles],
            "meld_groups": [group.to_dict() for group in self.meld_groups],
            "confirmed_open_melds": self.confirmed_open_melds,
            "suspected_open_melds": self.suspected_open_melds,
        }


@dataclass(frozen=True)
class TableTick:
    frame_index: int
    detections: int
    stable_hand: list[str]
    recommendation: str | None
    message: str
    zones: TableZones

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "detections": self.detections,
            "stable_hand": self.stable_hand,
            "recommendation": self.recommendation,
            "message": self.message,
            "zones": self.zones.to_dict(),
        }


@dataclass(frozen=True)
class ZoneDiagnostics:
    valid: bool
    warnings: list[str]
    expected_hand_counts: list[int]
    open_melds: int
    logical_warnings: list[str] = field(default_factory=list)

    def message(self) -> str:
        if not self.warnings:
            return "区域校验通过"
        return "；".join(self.warnings)


@dataclass
class StableTileList:
    stable_frames: int = 2
    history: deque[tuple[str, ...]] = field(default_factory=deque)

    def update(self, tiles: list[str], open_melds: int = 0) -> tuple[bool, list[str], str]:
        expected_counts = {13 - open_melds * 3, 14 - open_melds * 3}
        if len(tiles) not in expected_counts:
            self.history.clear()
            expected_text = "/".join(str(value) for value in sorted(expected_counts))
            return False, tiles, f"已开门 {open_melds} 组，暗手牌应为 {expected_text} 张，当前识别到 {len(tiles)} 张"
        over = [tile for tile, count in Counter(tiles).items() if count > 4]
        if over:
            self.history.clear()
            return False, tiles, f"单张牌数量超过 4：{', '.join(over)}"

        current = tuple(tiles)
        self.history.append(current)
        while len(self.history) > self.stable_frames:
            self.history.popleft()
        if len(self.history) < self.stable_frames:
            return False, tiles, "等待连续稳定帧"
        if len(set(self.history)) != 1:
            return False, tiles, "连续帧识别结果还不稳定"
        return True, tiles, "stable"


def run_realtime_table_loop(
    model_path: str | Path,
    missing_suit: str | None = None,
    interval_seconds: float = 0.25,
    limit: int | None = None,
    profile_path: str | Path = "configs/screen_profile.yaml",
    roi_name: str = "table",
    conf: float = 0.75,
    iou: float = 0.45,
    imgsz: int = 1280,
    show: bool = True,
    save_preview_dir: str | Path | None = None,
    log_path: str | Path = "data/games/realtime_table.jsonl",
) -> list[TableTick]:
    profile = load_screen_profile(profile_path)
    capture = ScreenCapture(profile.monitor)
    detector = YoloDetector(model_path, image_size=imgsz)
    tracker = StableTileList(stable_frames=2)
    logger = GameLogger(log_path)
    preview_dir = Path(save_preview_dir) if save_preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    ticks: list[TableTick] = []
    index = 0
    while limit is None or index < limit:
        frame = capture.grab().image_bgr
        roi = _select_roi(profile.rois.get(roi_name), frame)
        crop = roi.crop(frame)
        detections = detector.detect_image(crop, conf=conf, iou=iou)
        detections = _filter_tiles(detections, iou)
        zones = classify_table_zones(detections, crop.shape[1], crop.shape[0])
        open_melds = max(
            _open_melds_from_concealed_count(len(zones.hand)),
            _open_melds_from_groups(zones.meld_groups, "bottom_melds"),
        )
        stable, stable_hand, message = tracker.update(zones.hand, open_melds)
        recommendation = None
        if stable and len(stable_hand) == 14 - open_melds * 3:
            try:
                advice = advise_discard(stable_hand, missing_suit, open_melds=open_melds)
                recommendation = advice.recommended
                message = advice.explanation
            except ValueError as exc:
                message = str(exc)

        tick = TableTick(
            frame_index=index,
            detections=len(detections),
            stable_hand=stable_hand if stable else [],
            recommendation=recommendation,
            message=message,
            zones=zones,
        )
        payload = tick.to_dict()
        logger.log("realtime_table_tick", payload)
        print(json.dumps(payload, ensure_ascii=False))
        ticks.append(tick)

        overlay = draw_table_overlay(crop, detections, zones, recommendation, message)
        if preview_dir:
            cv2.imwrite(str(preview_dir / f"frame_{index:06d}.jpg"), overlay)
        if show:
            try:
                cv2.imshow("xueliu realtime table", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            except cv2.error as exc:
                print(f"OpenCV preview is unavailable; continue without preview: {exc}")
                show = False

        index += 1
        if limit is None or index < limit:
            time.sleep(interval_seconds)

    if show:
        try:
            cv2.destroyWindow("xueliu realtime table")
        except cv2.error:
            pass
    return ticks


def classify_table_zones(detections: list[Detection], width: int, height: int) -> TableZones:
    return _classify_table_zones_auto(detections, width, height)


def _diagnose_zones_legacy(zones: TableZones) -> ZoneDiagnostics:
    open_melds = _open_melds_from_groups(zones.meld_groups, "bottom_melds")
    expected_counts = sorted({13 - open_melds * 3, 14 - open_melds * 3})
    warnings: list[str] = []

    if len(zones.hand) not in expected_counts:
        expected_text = "/".join(str(value) for value in expected_counts)
        warnings.append(f"手牌数量异常：期望 {expected_text} 张，当前 {len(zones.hand)} 张")

    all_known_tiles: list[str] = []
    all_known_tiles.extend(zones.hand)
    all_known_tiles.extend(zones.bottom_melds)
    all_known_tiles.extend(zones.left_melds)
    all_known_tiles.extend(zones.right_melds)
    all_known_tiles.extend(zones.top_melds)
    all_known_tiles.extend(zones.center_discards)
    over_limit = [tile for tile, count in Counter(all_known_tiles).items() if count > 4]
    if over_limit:
        warnings.append("牌数超过4张：" + " ".join(f"{tile}x{Counter(all_known_tiles)[tile]}" for tile in over_limit))

    for name, tiles in [
        ("我的碰杠", zones.bottom_melds),
        ("上家碰杠", zones.left_melds),
        ("对家碰杠", zones.top_melds),
        ("下家碰杠", zones.right_melds),
    ]:
        if tiles and not _meld_labels_are_plausible(tiles):
            warnings.append(f"{name}数量可疑：{len(tiles)} 张")

    return ZoneDiagnostics(
        valid=not warnings,
        warnings=warnings,
        expected_hand_counts=expected_counts,
        open_melds=open_melds,
    )


def _classify_table_zones_auto(detections: list[Detection], width: int, height: int) -> TableZones:
    hand: list[Detection] = []
    bottom_melds: list[Detection] = []
    left_melds: list[Detection] = []
    right_melds: list[Detection] = []
    top_melds: list[Detection] = []
    center_discards: list[Detection] = []

    hand_run, bottom_play_melds = _partition_bottom_play_area(
        detections,
        width,
        height,
    )
    hand_ids = {id(det) for det in hand_run}
    bottom_meld_ids = {id(det) for det in bottom_play_melds}

    for det in detections:
        nx = det.center_x / max(1, width)
        ny = ((det.y1 + det.y2) / 2) / max(1, height)
        if id(det) in hand_ids:
            hand.append(det)
        elif id(det) in bottom_meld_ids:
            bottom_melds.append(det)
        elif not hand_run and ny >= AUTO_HAND_MIN_Y and 0.18 <= nx <= 0.82:
            hand.append(det)
        elif ny >= AUTO_HAND_MIN_Y:
            bottom_melds.append(det)
        # The four discard lattices occupy the inner table.  Resolve this
        # semantic area before the broad player-side bands so the top row of
        # discards is not mistaken for the top player's exposed melds.
        elif (
            AUTO_DISCARD_MIN_X <= nx <= AUTO_DISCARD_MAX_X
            and AUTO_DISCARD_MIN_Y <= ny <= AUTO_DISCARD_MAX_Y
        ):
            center_discards.append(det)
        elif ny <= AUTO_DISCARD_MIN_Y:
            top_melds.append(det)
        elif nx <= AUTO_DISCARD_MIN_X:
            left_melds.append(det)
        elif nx >= AUTO_DISCARD_MAX_X:
            right_melds.append(det)
        else:
            center_discards.append(det)

    bottom_meld_zone, bottom_isolated, bottom_groups = _split_meld_zone_tiles(bottom_melds, "bottom_melds", axis="horizontal")
    left_meld_zone, left_isolated, left_groups = _split_meld_zone_tiles(left_melds, "left_melds", axis="vertical")
    right_meld_zone, right_isolated, right_groups = _split_meld_zone_tiles(right_melds, "right_melds", axis="vertical")
    top_meld_zone, top_isolated, top_groups = _split_meld_zone_tiles(top_melds, "top_melds", axis="horizontal")
    meld_groups = [*bottom_groups, *left_groups, *right_groups, *top_groups]
    isolated_tiles = [*bottom_isolated, *left_isolated, *right_isolated, *top_isolated]
    hu_display_zone = [tile for tile in isolated_tiles if tile.zone == "hu_display_tiles"]
    event_zone = [tile for tile in isolated_tiles if tile.zone == "event_tiles"]
    unknown_zone = [tile for tile in isolated_tiles if tile.zone == "unknown_tiles"]
    candidate_meld_zone = [
        tile for tile in isolated_tiles if tile.zone == "candidate_meld_tiles"
    ]

    discard_groups = _split_discards_by_player(center_discards, width, height)
    zone_tiles = [
        *_zone_tiles_for_detections(hand, "hand"),
        *bottom_meld_zone,
        *left_meld_zone,
        *right_meld_zone,
        *top_meld_zone,
        *_zone_tiles_for_detections(center_discards, "center_discards"),
        *unknown_zone,
        *candidate_meld_zone,
        *hu_display_zone,
        *event_zone,
    ]
    return TableZones(
        hand=_labels_left_to_right(hand),
        bottom_melds=[tile.label for tile in bottom_meld_zone],
        left_melds=[tile.label for tile in left_meld_zone],
        right_melds=[tile.label for tile in right_meld_zone],
        top_melds=[tile.label for tile in top_meld_zone],
        center_discards=_labels_top_to_bottom(center_discards),
        all_tiles=_labels_top_to_bottom(detections),
        my_discards=_labels_left_to_right(discard_groups["my"]),
        left_discards=_labels_top_to_bottom(discard_groups["left"]),
        top_discards=_labels_left_to_right(discard_groups["top"]),
        right_discards=_labels_top_to_bottom(discard_groups["right"]),
        unknown_tiles=[tile.label for tile in unknown_zone],
        candidate_meld_tiles=[tile.label for tile in candidate_meld_zone],
        hu_display_tiles=[tile.label for tile in hu_display_zone],
        event_tiles=[tile.label for tile in event_zone],
        zone_tiles=zone_tiles,
        meld_groups=meld_groups,
    )


def _infer_bottom_hand_run(detections: list[Detection], width: int, height: int) -> list[Detection]:
    hand, _ = _partition_bottom_play_area(detections, width, height)
    return hand


def _partition_bottom_play_area(
    detections: list[Detection],
    width: int,
    height: int,
) -> tuple[list[Detection], list[Detection]]:
    """Split the bottom play area into concealed hand and exposed tiles."""
    bottom_candidates = [
        det
        for det in detections
        if _center_y(det) / max(1, height) >= AUTO_HAND_MIN_Y
        and 0.04 <= det.center_x / max(1, width) <= 0.96
    ]
    if not bottom_candidates:
        return [], []

    hand_sized = _hand_sized_bottom_candidates(bottom_candidates)
    runs = _horizontal_tile_runs(hand_sized)
    hand_runs = _horizontal_tile_runs(hand_sized, max_gap_multiplier=2.45)
    if not runs:
        return [], list(bottom_candidates)

    valid_hand_counts = {1, 2, 4, 5, 7, 8, 10, 11, 13, 14}
    valid_runs = [run for run in hand_runs if len(run) in valid_hand_counts]
    if valid_runs:
        median_height = sorted(
            max(1.0, det.y2 - det.y1) for det in bottom_candidates
        )[len(bottom_candidates) // 2]
        bottom_y = max(_run_center_y(run) for run in valid_runs)
        bottom_band_runs = [
            run
            for run in valid_runs
            if bottom_y - _run_center_y(run) <= median_height * 0.45
        ]
        selected = max(
            bottom_band_runs,
            key=lambda run: (
                len(run),
                _run_center_y(run),
                sum(det.area for det in run),
            ),
        )
        hand = _include_drawn_tile_if_split(selected, runs)
        hand_ids = {id(det) for det in hand}
        return hand, [det for det in bottom_candidates if id(det) not in hand_ids]

    selected = max(
        hand_runs,
        key=lambda run: (
            len(run),
            _run_center_y(run),
            sum(det.area for det in run),
        ),
    )
    hand = _include_drawn_tile_if_split(selected, runs)
    hand_ids = {id(det) for det in hand}
    return hand, [det for det in bottom_candidates if id(det) not in hand_ids]


def _hand_sized_bottom_candidates(
    bottom_candidates: list[Detection],
) -> list[Detection]:
    """Keep tiles matching the dominant large concealed-hand template."""
    if len(bottom_candidates) <= 2:
        return list(bottom_candidates)

    ranked = sorted(bottom_candidates, key=lambda det: det.area, reverse=True)
    reference_count = max(2, min(len(ranked), max(5, round(len(ranked) * 0.60))))
    reference = ranked[:reference_count]
    reference_width = median(max(1.0, det.x2 - det.x1) for det in reference)
    reference_height = median(max(1.0, det.y2 - det.y1) for det in reference)
    matching = [
        det
        for det in bottom_candidates
        if 0.78 <= max(1.0, det.x2 - det.x1) / reference_width <= 1.28
        and 0.78 <= max(1.0, det.y2 - det.y1) / reference_height <= 1.28
    ]
    return matching or list(bottom_candidates)


def _include_drawn_tile_if_split(hand_run: list[Detection], runs: list[list[Detection]]) -> list[Detection]:
    drawn_count_pairs = {(1, 2), (4, 5), (7, 8), (10, 11), (13, 14)}
    if not hand_run or (len(hand_run), len(hand_run) + 1) not in drawn_count_pairs:
        return hand_run

    median_width = sorted(max(1.0, det.x2 - det.x1) for det in hand_run)[len(hand_run) // 2]
    median_height = sorted(max(1.0, det.y2 - det.y1) for det in hand_run)[len(hand_run) // 2]
    hand_y = _run_center_y(hand_run)
    right_edge = max(det.x2 for det in hand_run)
    candidates: list[Detection] = []

    for run in runs:
        if run is hand_run or len(run) != 1:
            continue
        det = run[0]
        gap = det.x1 - right_edge
        if (
            det.center_x > right_edge
            and 0 <= gap <= median_width * 2.0
            and abs(_center_y(det) - hand_y) <= median_height * 0.45
            and 0.78 <= max(1.0, det.x2 - det.x1) / median_width <= 1.28
            and 0.78 <= max(1.0, det.y2 - det.y1) / median_height <= 1.28
        ):
            candidates.append(det)

    if not candidates:
        return hand_run
    drawn_tile = min(candidates, key=lambda det: det.x1 - right_edge)
    return sorted([*hand_run, drawn_tile], key=lambda item: item.center_x)


def _horizontal_tile_runs(detections: list[Detection], max_gap_multiplier: float = 1.95) -> list[list[Detection]]:
    if not detections:
        return []
    if len(detections) <= 2:
        return [sorted(detections, key=lambda item: item.center_x)]

    median_height = sorted(max(1.0, det.y2 - det.y1) for det in detections)[len(detections) // 2]
    rows: list[list[Detection]] = []
    for det in sorted(detections, key=_center_y):
        for row in rows:
            if abs(_center_y(det) - _run_center_y(row)) <= median_height * 0.45:
                row.append(det)
                break
        else:
            rows.append([det])

    runs: list[list[Detection]] = []
    for row in rows:
        ordered = sorted(row, key=lambda item: item.center_x)
        median_width = sorted(max(1.0, det.x2 - det.x1) for det in ordered)[len(ordered) // 2]
        max_gap = median_width * max_gap_multiplier
        current: list[Detection] = []
        for det in ordered:
            if not current or det.center_x - current[-1].center_x <= max_gap:
                current.append(det)
            else:
                runs.append(current)
                current = [det]
        if current:
            runs.append(current)
    return runs


def _center_y(det: Detection) -> float:
    return (det.y1 + det.y2) / 2


def _run_center_y(detections: list[Detection]) -> float:
    if not detections:
        return 0.0
    return sum(_center_y(det) for det in detections) / len(detections)


def _infer_hand_band(bottom_candidates: list[Detection]) -> tuple[float, float, float, float] | None:
    if len(bottom_candidates) < 5:
        return None
    ordered = sorted(bottom_candidates, key=lambda item: item.center_x)
    groups: list[list[Detection]] = []
    current: list[Detection] = []
    median_width = sorted(max(1.0, det.x2 - det.x1) for det in ordered)[len(ordered) // 2]
    max_gap = median_width * 1.8
    for det in ordered:
        if not current:
            current = [det]
            continue
        if det.center_x - current[-1].center_x <= max_gap:
            current.append(det)
        else:
            groups.append(current)
            current = [det]
    if current:
        groups.append(current)

    best = max(groups, key=len)
    if len(best) < 5:
        return None
    x1 = min(det.x1 for det in best) - median_width * 0.5
    x2 = max(det.x2 for det in best) + median_width * 0.5
    y1 = min(det.y1 for det in best) - median_width * 0.5
    y2 = max(det.y2 for det in best) + median_width * 0.8
    return x1, y1, x2, y2


def _in_hand_band(det: Detection, band: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = band
    cy = (det.y1 + det.y2) / 2
    return x1 <= det.center_x <= x2 and y1 <= cy <= y2


def classify_table_zones_by_rois(
    detections: list[Detection],
    table_roi: Roi,
    rois: dict[str, Roi],
    width: int,
    height: int,
) -> TableZones:
    buckets: dict[str, list[Detection]] = {
        "hand": [],
        "bottom_melds": [],
        "left_melds": [],
        "right_melds": [],
        "top_melds": [],
        "center_discards": [],
    }
    unknown: list[Detection] = []
    mapping = {
        "my_hand": "hand",
        "my_melds": "bottom_melds",
        "left_melds": "left_melds",
        "right_melds": "right_melds",
        "top_melds": "top_melds",
        "discards": "center_discards",
    }
    active_rois = {
        name: roi
        for name, roi in rois.items()
        if name in mapping and not roi.is_empty
    }
    if not active_rois:
        return classify_table_zones(detections, width, height)

    for det in detections:
        assignment = assign_by_roi_priority(det, table_roi, active_rois)
        if assignment is None:
            unknown.append(det)
        else:
            buckets[assignment.zone].append(det)

    fallback = classify_table_zones(unknown, width, height)
    bottom_meld_zone, bottom_isolated, bottom_groups = _split_meld_zone_tiles(buckets["bottom_melds"], "bottom_melds", axis="horizontal", source="manual_roi")
    left_meld_zone, left_isolated, left_groups = _split_meld_zone_tiles(buckets["left_melds"], "left_melds", axis="vertical", source="manual_roi")
    right_meld_zone, right_isolated, right_groups = _split_meld_zone_tiles(buckets["right_melds"], "right_melds", axis="vertical", source="manual_roi")
    top_meld_zone, top_isolated, top_groups = _split_meld_zone_tiles(buckets["top_melds"], "top_melds", axis="horizontal", source="manual_roi")
    isolated_tiles = [*bottom_isolated, *left_isolated, *right_isolated, *top_isolated]
    hu_display_zone = [tile for tile in isolated_tiles if tile.zone == "hu_display_tiles"]
    event_zone = [tile for tile in isolated_tiles if tile.zone == "event_tiles"]
    unknown_zone = [tile for tile in isolated_tiles if tile.zone == "unknown_tiles"]
    candidate_meld_zone = [
        tile for tile in isolated_tiles if tile.zone == "candidate_meld_tiles"
    ]
    use_manual_hand = "my_hand" in active_rois
    manual_bucket_names = {mapping[name] for name in active_rois}
    discard_groups = _split_discards_by_player(buckets["center_discards"], width, height)
    meld_groups = [*bottom_groups, *left_groups, *right_groups, *top_groups]
    for zone_name in ("bottom_melds", "left_melds", "right_melds", "top_melds"):
        if zone_name not in manual_bucket_names:
            meld_groups.extend(group for group in fallback.meld_groups if group.zone == zone_name)
    zone_tiles = [
        *_zone_tiles_for_detections(buckets["hand"], "hand", source="manual_roi"),
        *([] if use_manual_hand else _zone_tiles_from_zone(fallback, "hand")),
        *bottom_meld_zone,
        *([] if "bottom_melds" in manual_bucket_names else _zone_tiles_from_zone(fallback, "bottom_melds")),
        *left_meld_zone,
        *([] if "left_melds" in manual_bucket_names else _zone_tiles_from_zone(fallback, "left_melds")),
        *right_meld_zone,
        *([] if "right_melds" in manual_bucket_names else _zone_tiles_from_zone(fallback, "right_melds")),
        *top_meld_zone,
        *([] if "top_melds" in manual_bucket_names else _zone_tiles_from_zone(fallback, "top_melds")),
        *_zone_tiles_for_detections(buckets["center_discards"], "center_discards", source="manual_roi"),
        *_zone_tiles_from_zone(fallback, "center_discards"),
        *_zone_tiles_from_zone(fallback, "my_discards"),
        *_zone_tiles_from_zone(fallback, "left_discards"),
        *_zone_tiles_from_zone(fallback, "top_discards"),
        *_zone_tiles_from_zone(fallback, "right_discards"),
        *unknown_zone,
        *_zone_tiles_from_zone(fallback, "unknown_tiles"),
        *candidate_meld_zone,
        *_zone_tiles_from_zone(fallback, "candidate_meld_tiles"),
        *hu_display_zone,
        *_zone_tiles_from_zone(fallback, "hu_display_tiles"),
        *event_zone,
        *_zone_tiles_from_zone(fallback, "event_tiles"),
    ]
    return TableZones(
        hand=_labels_left_to_right(buckets["hand"]) if use_manual_hand else [*_labels_left_to_right(buckets["hand"]), *fallback.hand],
        bottom_melds=[tile.label for tile in bottom_meld_zone] if "bottom_melds" in manual_bucket_names else [*[tile.label for tile in bottom_meld_zone], *fallback.bottom_melds],
        left_melds=[tile.label for tile in left_meld_zone] if "left_melds" in manual_bucket_names else [*[tile.label for tile in left_meld_zone], *fallback.left_melds],
        right_melds=[tile.label for tile in right_meld_zone] if "right_melds" in manual_bucket_names else [*[tile.label for tile in right_meld_zone], *fallback.right_melds],
        top_melds=[tile.label for tile in top_meld_zone] if "top_melds" in manual_bucket_names else [*[tile.label for tile in top_meld_zone], *fallback.top_melds],
        center_discards=[*_labels_top_to_bottom(buckets["center_discards"]), *fallback.center_discards],
        all_tiles=_labels_top_to_bottom(detections),
        my_discards=[*_labels_left_to_right(discard_groups["my"]), *fallback.my_discards],
        left_discards=[*_labels_top_to_bottom(discard_groups["left"]), *fallback.left_discards],
        top_discards=[*_labels_left_to_right(discard_groups["top"]), *fallback.top_discards],
        right_discards=[*_labels_top_to_bottom(discard_groups["right"]), *fallback.right_discards],
        unknown_tiles=[
            *[tile.label for tile in unknown_zone],
            *fallback.unknown_tiles,
        ],
        candidate_meld_tiles=[
            *[tile.label for tile in candidate_meld_zone],
            *fallback.candidate_meld_tiles,
        ],
        hu_display_tiles=[*[tile.label for tile in hu_display_zone], *fallback.hu_display_tiles],
        event_tiles=[*[tile.label for tile in event_zone], *fallback.event_tiles],
        zone_tiles=zone_tiles,
        meld_groups=meld_groups,
    )


def visible_counts_from_zones(zones: TableZones, include_hand: bool = False) -> dict[str, int]:
    visible: list[str] = []
    if not zones.zone_tiles and not zones.meld_groups:
        if include_hand:
            visible.extend(zones.hand)
        visible.extend(zones.bottom_melds)
        visible.extend(zones.left_melds)
        visible.extend(zones.right_melds)
        visible.extend(zones.top_melds)
        visible.extend(zones.center_discards)
        return dict(Counter(visible))
    if include_hand:
        visible.extend(tile.label for tile in zones.zone_tiles if tile.zone == "hand" and not tile.inferred)
    visible.extend(
        tile.label
        for group in zones.meld_groups
        for tile in group.observed_only_tiles
    )
    visible.extend(
        tile.label
        for tile in zones.zone_tiles
        if tile.zone == "center_discards" and not tile.inferred
    )
    return dict(Counter(visible))


def logical_visible_counts_from_zones(zones: TableZones, include_hand: bool = False) -> dict[str, int]:
    visible: list[str] = []
    if include_hand:
        visible.extend(zones.hand)
    visible.extend(tile.label for group in zones.meld_groups for tile in group.logical_tiles)
    visible.extend(tile.label for tile in zones.zone_tiles if tile.zone == "center_discards")
    return dict(Counter(visible))


def diagnose_zones(zones: TableZones) -> ZoneDiagnostics:
    confirmed_open_melds = _open_melds_from_groups(zones.meld_groups, "bottom_melds")
    suspected_open_melds = sum(
        1
        for group in zones.meld_groups
        if group.zone == "bottom_melds" and group.is_suspected
    )
    expected_counts = sorted({13 - confirmed_open_melds * 3, 14 - confirmed_open_melds * 3})
    candidate_melds = confirmed_open_melds + suspected_open_melds
    candidate_expected_counts = sorted({13 - candidate_melds * 3, 14 - candidate_melds * 3})
    warnings: list[str] = []
    logical_warnings: list[str] = []

    if len(zones.hand) not in expected_counts:
        if suspected_open_melds and len(zones.hand) in candidate_expected_counts:
            logical_warnings.append(
                "hand count depends on unconfirmed meld: "
                f"candidate open melds {candidate_melds}, got {len(zones.hand)}"
            )
        else:
            expected_text = "/".join(str(value) for value in expected_counts)
            warnings.append(f"hand count invalid: expected {expected_text}, got {len(zones.hand)}")

    observed_counts = visible_counts_from_zones(zones, include_hand=True)
    logical_counts = logical_visible_counts_from_zones(zones, include_hand=True)
    observed_over = {tile: count for tile, count in observed_counts.items() if count > 4}
    logical_over = {tile: count for tile, count in logical_counts.items() if count > 4}
    if observed_over:
        warnings.append(
            "observed tile count over 4: "
            + " ".join(f"{tile}x{count}" for tile, count in sorted(observed_over.items()))
        )
    if logical_over:
        logical_warnings.append(
            "logical tile count over 4: "
            + " ".join(f"{tile}x{count}" for tile, count in sorted(logical_over.items()))
        )

    for name, tiles in (
        ("bottom_melds", zones.bottom_melds),
        ("left_melds", zones.left_melds),
        ("top_melds", zones.top_melds),
        ("right_melds", zones.right_melds),
    ):
        if tiles and not _meld_labels_are_plausible(tiles):
            warnings.append(f"{name} structure invalid: {len(tiles)} tiles")

    return ZoneDiagnostics(
        valid=not warnings,
        warnings=warnings,
        expected_hand_counts=expected_counts,
        open_melds=confirmed_open_melds,
        logical_warnings=logical_warnings,
    )


def reconcile_zone_tile_limits(zones: TableZones) -> TableZones:
    visible_counts = Counter(visible_counts_from_zones(zones, include_hand=False))

    remaining_by_tile = {tile: max(0, 4 - visible_counts[tile]) for tile in TILE_SET}
    kept_hand: list[str] = []
    kept_counts: Counter[str] = Counter()
    for tile in zones.hand:
        if kept_counts[tile] < remaining_by_tile.get(tile, 0):
            kept_hand.append(tile)
            kept_counts[tile] += 1

    if kept_hand == zones.hand:
        return zones
    removed = Counter(zones.hand) - Counter(kept_hand)
    removed_labels: list[str] = []
    kept_zone_tiles: list[ZoneTile] = []
    for tile in zones.zone_tiles:
        if tile.zone == "hand" and removed[tile.label] > 0:
            removed[tile.label] -= 1
            removed_labels.append(tile.label)
            kept_zone_tiles.append(
                ZoneTile(
                    label=tile.label,
                    confidence=tile.confidence,
                    x1=tile.x1,
                    y1=tile.y1,
                    x2=tile.x2,
                    y2=tile.y2,
                    zone="unknown_tiles",
                    group_id=tile.group_id,
                    source=tile.source,
                    reason="tile_count_limit",
                )
            )
            continue
        kept_zone_tiles.append(tile)
    return TableZones(
        hand=kept_hand,
        bottom_melds=zones.bottom_melds,
        left_melds=zones.left_melds,
        right_melds=zones.right_melds,
        top_melds=zones.top_melds,
        center_discards=zones.center_discards,
        all_tiles=zones.all_tiles,
        my_discards=zones.my_discards,
        left_discards=zones.left_discards,
        top_discards=zones.top_discards,
        right_discards=zones.right_discards,
        unknown_tiles=[*zones.unknown_tiles, *removed_labels],
        candidate_meld_tiles=zones.candidate_meld_tiles,
        hu_display_tiles=zones.hu_display_tiles,
        event_tiles=zones.event_tiles,
        table_decor_tiles=zones.table_decor_tiles,
        zone_tiles=kept_zone_tiles,
        meld_groups=zones.meld_groups,
    )


def _point_in_roi(x: float, y: float, roi: Roi) -> bool:
    return roi.x <= x <= roi.x + roi.width and roi.y <= y <= roi.y + roi.height


def _split_discards_by_player(
    discards: list[Detection],
    width: int,
    height: int,
) -> dict[str, list[Detection]]:
    groups: dict[str, list[Detection]] = {"my": [], "left": [], "top": [], "right": []}
    center_x = width / 2
    center_y = height / 2
    for det in discards:
        nx = det.center_x / max(1, width)
        ny = ((det.y1 + det.y2) / 2) / max(1, height)
        dx = det.center_x - center_x
        dy = ((det.y1 + det.y2) / 2) - center_y

        # Use perspective-aware discard lanes before the radial fallback.
        # The bottom and top lattices are wide horizontal bands, while the
        # side lattices occupy tall bands around the table edges.
        if ny >= 0.56 and 0.28 <= nx <= 0.72:
            owner = "my"
        elif ny <= 0.34 and 0.28 <= nx <= 0.72:
            owner = "top"
        elif nx < 0.45:
            owner = "left"
        elif nx > 0.55:
            owner = "right"
        else:
            normalized_dx = dx / max(1.0, width / 2)
            normalized_dy = dy / max(1.0, height / 2)
            if abs(normalized_dx) > abs(normalized_dy):
                owner = "right" if normalized_dx > 0 else "left"
            else:
                owner = "my" if normalized_dy > 0 else "top"
        groups[owner].append(det)
    return groups


def draw_table_overlay(
    image: np.ndarray,
    detections: list[Detection],
    zones: TableZones,
    recommendation: str | None,
    message: str,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    _draw_zone_guides(canvas, width, height)
    _draw_auto_zone_bands(canvas, width, height)

    if zones.zone_tiles:
        for tile in zones.zone_tiles:
            if tile.zone in {"my_discards", "left_discards", "top_discards", "right_discards"}:
                continue
            color = _final_zone_color(tile.zone)
            cv2.rectangle(canvas, (int(tile.x1), int(tile.y1)), (int(tile.x2), int(tile.y2)), color, 2)
            prefix = "I" if tile.inferred else _zone_short_name(tile.zone)
            track = f"#{tile.track_id}" if tile.track_id is not None else ""
            label = f"{prefix}{track}:{tile.label}"
            cv2.putText(
                canvas,
                label,
                (int(tile.x1), max(15, int(tile.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )
    else:
        for det in detections:
            color = _zone_color(det, width, height)
            cv2.rectangle(canvas, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
            cv2.putText(
                canvas,
                det.label,
                (int(det.x1), max(15, int(det.y1) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    lines = [
        f"hand({len(zones.hand)}): {' '.join(zones.hand)}",
        _zone_count_line(zones),
        _discard_zone_count_line(zones),
        f"discard: {recommendation or '-'}",
        _overlay_safe_text(message),
        "q/ESC exit",
    ]
    _draw_panel(canvas, lines)
    return canvas


def _select_roi(configured: Roi | None, frame: np.ndarray) -> Roi:
    if configured and not configured.is_empty:
        return configured
    height, width = frame.shape[:2]
    return Roi(x=0, y=0, width=width, height=height)


def _filter_tiles(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    return non_max_suppression([det for det in detections if det.label in TILE_SET], iou_threshold)


def _labels_left_to_right(detections: list[Detection]) -> list[str]:
    return [det.label for det in sorted(detections, key=lambda item: (item.center_x, item.y1))]


def _labels_top_to_bottom(detections: list[Detection]) -> list[str]:
    return [det.label for det in sorted(detections, key=lambda item: (item.y1, item.center_x))]


def _zone_tiles_for_detections(
    detections: list[Detection],
    zone: str,
    group_id: str | None = None,
    source: str = "auto",
    reason: str = "",
) -> list[ZoneTile]:
    ordered = sorted(detections, key=lambda item: (item.y1, item.center_x))
    if zone == "hand":
        ordered = sorted(detections, key=lambda item: (item.center_x, item.y1))
    return [ZoneTile.from_detection(det, zone, group_id, source=source, reason=reason) for det in ordered]


def _zone_tiles_from_zone(zones: TableZones, zone: str) -> list[ZoneTile]:
    return [tile for tile in zones.zone_tiles if tile.zone == zone]


def _split_meld_zone_tiles(
    detections: list[Detection],
    zone: str,
    axis: str,
    source: str = "auto",
) -> tuple[list[ZoneTile], list[ZoneTile], list[MeldGroup]]:
    result = group_melds(detections, zone=zone, axis=axis, source=source)
    return result.zone_tiles, result.isolated_tiles, result.groups


def _split_meld_run_zone_tiles(
    run: list[Detection],
    zone: str,
    axis: str,
    group_prefix: str,
    source: str,
) -> tuple[list[ZoneTile], list[ZoneTile]]:
    ordered = sorted(run, key=(lambda item: item.center_x) if axis == "horizontal" else _center_y)
    valid: list[ZoneTile] = []
    isolated: list[ZoneTile] = []
    segment: list[Detection] = []
    segment_index = 0
    for det in ordered:
        if not segment or segment[-1].label == det.label:
            segment.append(det)
            continue
        segment_index += 1
        _append_meld_segment_zone_tiles(segment, zone, f"{group_prefix}_{segment_index}", source, valid, isolated)
        segment = [det]
    if segment:
        segment_index += 1
        _append_meld_segment_zone_tiles(segment, zone, f"{group_prefix}_{segment_index}", source, valid, isolated)
    return valid, isolated


def _append_meld_segment_zone_tiles(
    segment: list[Detection],
    zone: str,
    group_id: str,
    source: str,
    valid: list[ZoneTile],
    isolated: list[ZoneTile],
) -> None:
    count = len(segment)
    if _is_valid_meld_segment_count(count):
        valid.extend(_zone_tiles_for_detections(segment, zone, group_id=group_id, source=source))
        return
    target_zone = "candidate_meld_tiles"
    reason = "isolated_near_meld" if count in (1, 2) else "invalid_meld_group"
    isolated.extend(_zone_tiles_for_detections(segment, target_zone, group_id=group_id, source=source, reason=reason))


def _is_valid_meld_segment_count(count: int) -> bool:
    return count in (3, 4) or (count >= 6 and count % 3 == 0)


def _split_valid_meld_detections(detections: list[Detection], axis: str) -> tuple[list[str], list[Detection]]:
    runs = _vertical_tile_runs(detections) if axis == "vertical" else _horizontal_tile_runs(detections)
    valid: list[str] = []
    unknown: list[Detection] = []
    for run in runs:
        run_valid, run_unknown = _split_valid_meld_run(run, axis)
        valid.extend(run_valid)
        unknown.extend(run_unknown)
    return valid, unknown


def _split_valid_meld_run(run: list[Detection], axis: str) -> tuple[list[str], list[Detection]]:
    ordered = sorted(run, key=(lambda item: item.center_x) if axis == "horizontal" else _center_y)
    valid: list[str] = []
    unknown: list[Detection] = []
    segment: list[Detection] = []
    for det in ordered:
        if not segment or segment[-1].label == det.label:
            segment.append(det)
            continue
        _append_meld_segment(segment, valid, unknown)
        segment = [det]
    if segment:
        _append_meld_segment(segment, valid, unknown)
    return valid, unknown


def _append_meld_segment(segment: list[Detection], valid: list[str], unknown: list[Detection]) -> None:
    count = len(segment)
    if _is_valid_meld_segment_count(count):
        valid.extend(det.label for det in segment)
    else:
        unknown.extend(segment)


def _vertical_tile_runs(detections: list[Detection], max_gap_multiplier: float = 1.95) -> list[list[Detection]]:
    if not detections:
        return []
    if len(detections) <= 2:
        return [sorted(detections, key=_center_y)]

    median_width = sorted(max(1.0, det.x2 - det.x1) for det in detections)[len(detections) // 2]
    columns: list[list[Detection]] = []
    for det in sorted(detections, key=lambda item: item.center_x):
        for column in columns:
            center_x = sum(item.center_x for item in column) / len(column)
            if abs(det.center_x - center_x) <= median_width * 0.55:
                column.append(det)
                break
        else:
            columns.append([det])

    runs: list[list[Detection]] = []
    for column in columns:
        ordered = sorted(column, key=_center_y)
        median_height = sorted(max(1.0, det.y2 - det.y1) for det in ordered)[len(ordered) // 2]
        max_gap = median_height * max_gap_multiplier
        current: list[Detection] = []
        for det in ordered:
            if not current or _center_y(det) - _center_y(current[-1]) <= max_gap:
                current.append(det)
            else:
                runs.append(current)
                current = [det]
        if current:
            runs.append(current)
    return runs


def _meld_labels_are_plausible(labels: list[str]) -> bool:
    if not labels:
        return True
    counts = Counter(labels)
    return all(count in (3, 4) or (count >= 6 and count % 3 == 0) for count in counts.values())


def _zone_count_line(zones: TableZones) -> str:
    open_melds = _open_melds_from_groups(zones.meld_groups, "bottom_melds")
    expected = sorted({13 - open_melds * 3, 14 - open_melds * 3})
    expected_text = "/".join(str(value) for value in expected)
    return (
        "zones: "
        f"hand={len(zones.hand)} "
        f"my_meld={len(zones.bottom_melds)} "
        f"left_meld={len(zones.left_melds)} "
        f"top_meld={len(zones.top_melds)} "
        f"right_meld={len(zones.right_melds)} "
        f"discards={len(zones.center_discards)} "
        f"candidate_meld={len(zones.candidate_meld_tiles)} "
        f"unknown={len(zones.unknown_tiles) + len(zones.hu_display_tiles) + len(zones.event_tiles)} "
        f"expected_hand={expected_text}"
    )


def _discard_zone_count_line(zones: TableZones) -> str:
    return (
        "discard areas: "
        f"me={len(zones.my_discards)} "
        f"left={len(zones.left_discards)} "
        f"top={len(zones.top_discards)} "
        f"right={len(zones.right_discards)}"
    )


def _zone_color(det: Detection, width: int, height: int) -> tuple[int, int, int]:
    nx = det.center_x / max(1, width)
    ny = ((det.y1 + det.y2) / 2) / max(1, height)
    if ny >= AUTO_HAND_MIN_Y and 0.04 <= nx <= 0.96:
        return (0, 220, 255)
    if (
        AUTO_DISCARD_MIN_X <= nx <= AUTO_DISCARD_MAX_X
        and AUTO_DISCARD_MIN_Y <= ny <= AUTO_DISCARD_MAX_Y
    ):
        return (255, 180, 0)
    return (0, 220, 80)


def _final_zone_color(zone: str) -> tuple[int, int, int]:
    colors = {
        "hand": (0, 220, 255),
        "bottom_melds": (0, 220, 80),
        "left_melds": (40, 220, 40),
        "right_melds": (40, 220, 40),
        "top_melds": (40, 220, 40),
        "center_discards": (255, 180, 0),
        "my_discards": (255, 210, 0),
        "left_discards": (255, 180, 0),
        "top_discards": (255, 180, 0),
        "right_discards": (255, 180, 0),
        "unknown_tiles": (180, 120, 255),
        "candidate_meld_tiles": (80, 210, 255),
        "hu_display_tiles": (255, 80, 220),
        "event_tiles": (0, 165, 255),
        "table_decor_tiles": (120, 120, 120),
    }
    return colors.get(zone, (255, 255, 255))


def _zone_short_name(zone: str) -> str:
    names = {
        "hand": "H",
        "bottom_melds": "M",
        "left_melds": "L",
        "right_melds": "R",
        "top_melds": "T",
        "center_discards": "D",
        "unknown_tiles": "U",
        "candidate_meld_tiles": "CM",
        "hu_display_tiles": "HU",
        "event_tiles": "EV",
        "table_decor_tiles": "TD",
    }
    return names.get(zone, zone[:2].upper())


def _draw_zone_guides(canvas: np.ndarray, width: int, height: int) -> None:
    guide_color = (80, 80, 80)
    cv2.line(
        canvas,
        (0, int(height * AUTO_HAND_MIN_Y)),
        (width, int(height * AUTO_HAND_MIN_Y)),
        guide_color,
        1,
    )
    cv2.line(
        canvas,
        (int(width * AUTO_DISCARD_MIN_X), 0),
        (int(width * AUTO_DISCARD_MIN_X), height),
        guide_color,
        1,
    )
    cv2.line(
        canvas,
        (int(width * AUTO_DISCARD_MAX_X), 0),
        (int(width * AUTO_DISCARD_MAX_X), height),
        guide_color,
        1,
    )


def _draw_auto_zone_bands(canvas: np.ndarray, width: int, height: int) -> None:
    bands = [
        (
            "BOTTOM PLAY AREA",
            (int(width * 0.04), int(height * AUTO_HAND_MIN_Y), int(width * 0.96), height),
            (0, 220, 255),
        ),
        (
            "DISCARD AREA",
            (
                int(width * AUTO_DISCARD_MIN_X),
                int(height * AUTO_DISCARD_MIN_Y),
                int(width * AUTO_DISCARD_MAX_X),
                int(height * AUTO_DISCARD_MAX_Y),
            ),
            (255, 180, 0),
        ),
        (
            "LEFT",
            (0, int(height * AUTO_DISCARD_MIN_Y), int(width * AUTO_DISCARD_MIN_X), int(height * AUTO_DISCARD_MAX_Y)),
            (80, 255, 80),
        ),
        (
            "RIGHT",
            (int(width * AUTO_DISCARD_MAX_X), int(height * AUTO_DISCARD_MIN_Y), width, int(height * AUTO_DISCARD_MAX_Y)),
            (80, 255, 80),
        ),
        (
            "TOP",
            (int(width * AUTO_DISCARD_MIN_X), 0, int(width * AUTO_DISCARD_MAX_X), int(height * AUTO_DISCARD_MIN_Y)),
            (80, 255, 80),
        ),
    ]
    for label, rect, color in bands:
        _draw_transparent_rect(canvas, rect, color, alpha=0.08)
        x1, y1, x2, y2 = rect
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            canvas,
            label,
            (x1 + 6, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_transparent_rect(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1, x2, y2 = rect
    x1 = max(0, min(canvas.shape[1] - 1, x1))
    x2 = max(0, min(canvas.shape[1] - 1, x2))
    y1 = max(0, min(canvas.shape[0] - 1, y1))
    y2 = max(0, min(canvas.shape[0] - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)


def _draw_panel(canvas: np.ndarray, lines: list[str]) -> None:
    x, y = 12, 24
    line_height = 24
    longest = max((len(line) for line in lines), default=20)
    width = min(canvas.shape[1] - 24, max(520, longest * 11))
    height = line_height * len(lines) + 16
    cv2.rectangle(canvas, (6, 6), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(canvas, 0.82, canvas, 0.18, 0, canvas)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line[:120],
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _overlay_safe_text(text: str) -> str:
    if not text:
        return "-"
    safe = "".join(char if 32 <= ord(char) < 127 else " " for char in str(text))
    safe = " ".join(safe.split())
    return safe or "status: see main UI"


def _open_melds_from_concealed_count(count: int) -> int:
    if count in (13, 14):
        return 0
    if count in (10, 11):
        return 1
    if count in (7, 8):
        return 2
    if count in (4, 5):
        return 3
    if count in (1, 2):
        return 4
    return 0


def _open_melds_from_groups(groups: list[MeldGroup], zone: str) -> int:
    return min(4, sum(group.open_meld_count for group in groups if group.zone == zone and group.is_confirmed))


def _suspected_open_melds_from_groups(groups: list[MeldGroup], zone: str) -> int:
    return min(4, sum(group.open_meld_count for group in groups if group.zone == zone and group.is_suspected))
