from __future__ import annotations

from dataclasses import dataclass

from xueliu_ai.table.structured_types import RegionState


@dataclass(frozen=True)
class StructuredStateResult:
    state: RegionState
    allow_recommend: bool
    reason: str


def validate_structured_state(
    zones,
    phase_stable: bool,
    consecutive_stable_frames: int,
    minimum_stable_frames: int = 3,
) -> StructuredStateResult:
    if zones.event_tiles:
        return StructuredStateResult(RegionState.TRANSIENT, False, "event_animation_active")
    if not phase_stable:
        return StructuredStateResult(RegionState.TRANSIENT, False, "phase_not_stable")
    if len(zones.unknown_tiles) >= 3:
        return StructuredStateResult(RegionState.INVALID, False, "too_many_unknown_tiles")
    if consecutive_stable_frames < minimum_stable_frames:
        return StructuredStateResult(RegionState.UNCERTAIN, False, "waiting_for_structural_stability")
    inferred = any(tile.inferred for tile in zones.zone_tiles)
    if inferred:
        return StructuredStateResult(RegionState.INFERRED_SAFE, True, "stable_with_safe_inference")
    return StructuredStateResult(RegionState.CONFIRMED, True, "confirmed")
