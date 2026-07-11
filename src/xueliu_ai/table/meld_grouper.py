from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import median

from xueliu_ai.table.structured_types import MeldGroup, MeldKind, ZoneTile
from xueliu_ai.vision.detection_types import Detection


@dataclass(frozen=True)
class MeldGroupingResult:
    groups: list[MeldGroup]
    isolated_tiles: list[ZoneTile]

    @property
    def zone_tiles(self) -> list[ZoneTile]:
        return [tile for group in self.groups for tile in group.all_tiles]


def group_melds(
    detections: list[Detection],
    zone: str,
    axis: str,
    source: str = "auto",
) -> MeldGroupingResult:
    """Group exposed tiles using geometry first and labels second."""
    if not detections:
        return MeldGroupingResult([], [])

    runs = _spatial_runs(detections, axis)
    groups: list[MeldGroup] = []
    isolated: list[ZoneTile] = []
    for run_index, run in enumerate(runs, start=1):
        for segment_index, segment in enumerate(_candidate_segments(run, axis), start=1):
            group_id = f"{zone}_{run_index}_{segment_index}"
            group = _make_group(segment, zone, axis, source, group_id)
            if group is not None:
                groups.append(group)
            else:
                reason = "isolated_near_meld" if len(segment) <= 2 else "invalid_meld_geometry"
                isolated.extend(
                    ZoneTile.from_detection(det, "unknown_tiles", group_id, source, reason)
                    for det in segment
                )
    return MeldGroupingResult(groups, isolated)


def _make_group(
    detections: list[Detection],
    zone: str,
    axis: str,
    source: str,
    group_id: str,
) -> MeldGroup | None:
    if not detections:
        return None
    observed_count = len(detections)
    if observed_count not in (2, 3, 4):
        return None
    ordered = sorted(detections, key=_axis_center(axis))
    if not _compact_enough(ordered, axis):
        return None

    votes: Counter[str] = Counter()
    for detection in ordered:
        votes[detection.label] += max(0.01, detection.confidence)
    label, winning_vote = votes.most_common(1)[0]
    same_count = sum(1 for detection in ordered if detection.label == label)
    if same_count < 2 or (observed_count == 4 and same_count < 3):
        return None
    total_vote = sum(votes.values())
    consistency = winning_vote / total_vote if total_vote else 0.0
    has_conflict = same_count != observed_count

    suspected_kong = observed_count == 3 and _looks_like_stacked_kong(ordered, axis)
    kind = (
        MeldKind.SUSPECTED_KONG
        if suspected_kong
        else MeldKind.SUSPECTED_PONG
        if has_conflict and observed_count == 3
        else MeldKind.SUSPECTED_KONG
        if has_conflict and observed_count == 4
        else {2: MeldKind.SUSPECTED_PONG, 3: MeldKind.PONG, 4: MeldKind.KONG}[
            observed_count
        ]
    )
    tiles = [
        ZoneTile.from_detection(det, zone, group_id, source, "spatial_meld_group")
        for det in ordered
    ]
    inferred: list[ZoneTile] = []
    if observed_count == 2 or suspected_kong:
        inferred.append(replace(_infer_missing_tile(tiles, axis, group_id), label=label))

    confidence = sum(tile.confidence for tile in tiles) / len(tiles)
    if inferred:
        confidence = min(confidence, inferred[0].confidence)
    return MeldGroup(
        group_id=group_id,
        zone=zone,
        kind=kind,
        label=label,
        observed_tiles=tiles,
        inferred_tiles=inferred,
        confidence=confidence,
        axis=axis,
        label_votes=dict(votes),
        conflicting_tiles=[tile for tile in tiles if tile.label != label],
        label_consistency=consistency,
        reason=(
            "two_tile_spatial_completion"
            if observed_count == 2
            else "stacked_kong_completion"
            if suspected_kong
            else "spatial_group_label_conflict"
            if has_conflict
            else "confirmed_spatial_group"
        ),
    )


def _infer_missing_tile(tiles: list[ZoneTile], axis: str, group_id: str) -> ZoneTile:
    ordered = sorted(tiles, key=(lambda tile: tile.center_x) if axis == "horizontal" else (lambda tile: tile.center_y))
    first, second = ordered[-2:]
    if axis == "horizontal":
        tile_size = (first.width + second.width) / 2
        gap = second.center_x - first.center_x
        cx = (first.center_x + second.center_x) / 2 if gap > tile_size * 1.45 else second.center_x + tile_size
        cy = (first.center_y + second.center_y) / 2
    else:
        tile_size = (first.height + second.height) / 2
        gap = second.center_y - first.center_y
        cx = (first.center_x + second.center_x) / 2
        cy = (first.center_y + second.center_y) / 2 if gap > tile_size * 1.45 else second.center_y + tile_size
    width = median([first.width, second.width])
    height = median([first.height, second.height])
    return ZoneTile(
        label=first.label,
        confidence=min(first.confidence, second.confidence) * 0.55,
        x1=cx - width / 2,
        y1=cy - height / 2,
        x2=cx + width / 2,
        y2=cy + height / 2,
        zone=first.zone,
        group_id=group_id,
        source="inferred",
        reason="spatial_pong_completion",
        inferred=True,
    )


def _spatial_runs(detections: list[Detection], axis: str) -> list[list[Detection]]:
    cross_size = median([
        max(1.0, det.y2 - det.y1) if axis == "horizontal" else max(1.0, det.x2 - det.x1)
        for det in detections
    ])
    primary_size = median([
        max(1.0, det.x2 - det.x1) if axis == "horizontal" else max(1.0, det.y2 - det.y1)
        for det in detections
    ])
    bands: list[list[Detection]] = []
    cross = _cross_center(axis)
    for det in sorted(detections, key=cross):
        for band in bands:
            band_center = sum(cross(item) for item in band) / len(band)
            if abs(cross(det) - band_center) <= cross_size * 0.6:
                band.append(det)
                break
        else:
            bands.append([det])

    runs: list[list[Detection]] = []
    primary = _axis_center(axis)
    for band in bands:
        current: list[Detection] = []
        for det in sorted(band, key=primary):
            if not current or primary(det) - primary(current[-1]) <= primary_size * 1.85:
                current.append(det)
            else:
                runs.append(current)
                current = [det]
        if current:
            runs.append(current)
    return runs


def _candidate_segments(run: list[Detection], axis: str) -> list[list[Detection]]:
    ordered = sorted(run, key=_axis_center(axis))
    if len(ordered) <= 4:
        return [ordered]
    segments: list[list[Detection]] = []
    current: list[Detection] = []
    for det in ordered:
        if not current or current[-1].label == det.label:
            current.append(det)
        else:
            segments.append(current)
            current = [det]
    if current:
        segments.append(current)
    expanded: list[list[Detection]] = []
    for segment in segments:
        if len(segment) <= 4:
            expanded.append(segment)
            continue
        if len(segment) % 3 == 0:
            expanded.extend(segment[index : index + 3] for index in range(0, len(segment), 3))
        elif len(segment) % 4 == 0:
            expanded.extend(segment[index : index + 4] for index in range(0, len(segment), 4))
        else:
            expanded.append(segment)
    return expanded


def _compact_enough(detections: list[Detection], axis: str) -> bool:
    if len(detections) < 2:
        return False
    primary = _axis_center(axis)
    sizes = [
        max(1.0, det.x2 - det.x1) if axis == "horizontal" else max(1.0, det.y2 - det.y1)
        for det in detections
    ]
    limit = median(sizes) * 1.9
    ordered = sorted(detections, key=primary)
    return all(primary(right) - primary(left) <= limit for left, right in zip(ordered, ordered[1:]))


def _looks_like_stacked_kong(detections: list[Detection], axis: str) -> bool:
    cross = _cross_center(axis)
    sizes = [
        max(1.0, det.y2 - det.y1) if axis == "horizontal" else max(1.0, det.x2 - det.x1)
        for det in detections
    ]
    spread = max(cross(det) for det in detections) - min(cross(det) for det in detections)
    return spread > median(sizes) * 0.35


def _axis_center(axis: str):
    return (lambda det: det.center_x) if axis == "horizontal" else (lambda det: (det.y1 + det.y2) / 2)


def _cross_center(axis: str):
    return (lambda det: (det.y1 + det.y2) / 2) if axis == "horizontal" else (lambda det: det.center_x)
