from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from xueliu_ai.table.structured_types import RegionState


@dataclass(frozen=True)
class StructuredStateResult:
    state: RegionState
    allow_recommend: bool
    reason: str

    @property
    def valid(self) -> bool:
        return self.state in {RegionState.CONFIRMED, RegionState.INFERRED_SAFE}

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "valid": self.valid,
            "allow_recommend": self.allow_recommend,
            "reason": self.reason,
        }


def validate_structured_state(
    state_or_zones,
    phase_stable: bool,
    consecutive_stable_frames: int,
    minimum_stable_frames: int = 3,
    diagnostics_valid: bool = True,
) -> StructuredStateResult:
    state = state_or_zones if hasattr(state_or_zones, "zones") else None
    zones = state.zones if state is not None else state_or_zones
    if state is not None:
        errors = state.consistency_errors()
        if errors:
            return StructuredStateResult(RegionState.INVALID, False, errors[0])
        if state.meld_history_transient:
            return StructuredStateResult(RegionState.TRANSIENT, False, "meld_track_temporarily_missing")
    if zones.event_tiles:
        return StructuredStateResult(RegionState.TRANSIENT, False, "event_animation_active")
    if not phase_stable:
        return StructuredStateResult(RegionState.TRANSIENT, False, "phase_not_stable")
    if not diagnostics_valid:
        return StructuredStateResult(RegionState.INVALID, False, "diagnostics_invalid")
    if zones.unknown_tiles:
        return StructuredStateResult(RegionState.UNCERTAIN, False, "unknown_tiles_present")
    if state is not None and any(group.is_suspected for group in state.meld_groups):
        return StructuredStateResult(RegionState.UNCERTAIN, False, "unconfirmed_meld_present")
    observed_counts = Counter(state.observed_visible_counts if state is not None else {})
    observed_counts.update(
        tile.label for tile in zones.zone_tiles if tile.zone == "hand" and not tile.inferred
    )
    if any(count > 4 for count in observed_counts.values()):
        return StructuredStateResult(RegionState.INVALID, False, "observed_tile_count_over_four")
    if state is not None and any(count > 4 for count in state.logical_visible_counts.values()):
        return StructuredStateResult(RegionState.UNCERTAIN, False, "logical_tile_count_over_four")
    inferred_hand = any(tile.inferred for tile in zones.zone_tiles if tile.zone == "hand")
    if inferred_hand:
        return StructuredStateResult(RegionState.UNCERTAIN, False, "inferred_hand_tile_present")
    if consecutive_stable_frames < minimum_stable_frames:
        return StructuredStateResult(RegionState.UNCERTAIN, False, "waiting_for_structural_stability")
    inferred = any(group.inferred_tiles for group in zones.meld_groups if group.is_confirmed)
    if inferred:
        return StructuredStateResult(RegionState.INFERRED_SAFE, True, "stable_with_safe_inference")
    return StructuredStateResult(RegionState.CONFIRMED, True, "confirmed")


class StructuredStateMachine:
    def __init__(self, minimum_stable_frames: int = 3) -> None:
        self.minimum_stable_frames = minimum_stable_frames
        self._last_signature = None
        self._stable_frames = 0

    def update(self, state_or_zones, phase_stable: bool, diagnostics_valid: bool) -> StructuredStateResult:
        zones = state_or_zones.zones if hasattr(state_or_zones, "zones") else state_or_zones
        safety = validate_structured_state(
            state_or_zones,
            phase_stable=phase_stable,
            consecutive_stable_frames=0,
            minimum_stable_frames=0,
            diagnostics_valid=diagnostics_valid,
        )
        if safety.state not in {RegionState.CONFIRMED, RegionState.INFERRED_SAFE}:
            self._last_signature = None
            self._stable_frames = 0
            return safety
        hand_signature = sorted(
            (
                tile.label,
                tile.inferred,
                tile.track_id if tile.track_id is not None else -1,
                _position_bucket(tile.center_x),
                _position_bucket(tile.center_y),
            )
            for tile in zones.zone_tiles
            if tile.zone == "hand"
        )
        meld_signature = sorted(
            (
                group.group_id,
                group.zone,
                group.kind.value,
                group.label,
                tuple(
                    sorted(
                        (
                            tile.track_id if tile.track_id is not None else -1,
                            tile.inferred,
                            _position_bucket(tile.center_x),
                            _position_bucket(tile.center_y),
                        )
                        for tile in group.logical_tiles
                    )
                ),
            )
            for group in zones.meld_groups
        )
        signature = (
            tuple(hand_signature),
            tuple(meld_signature),
            _zone_signature(zones, "unknown_tiles"),
            _zone_signature(zones, "event_tiles"),
            _zone_signature(zones, "hu_display_tiles"),
            tuple(sorted(getattr(state_or_zones, "observed_visible_counts", {}).items())),
            tuple(sorted(getattr(state_or_zones, "logical_visible_counts", {}).items())),
            diagnostics_valid,
        )
        if signature == self._last_signature:
            self._stable_frames += 1
        else:
            self._last_signature = signature
            self._stable_frames = 1
        return validate_structured_state(
            state_or_zones,
            phase_stable=phase_stable,
            consecutive_stable_frames=self._stable_frames,
            minimum_stable_frames=self.minimum_stable_frames,
            diagnostics_valid=diagnostics_valid,
        )


def _position_bucket(value: float, size: float = 5.0) -> int:
    """Keep structural stability sensitive to movement without reacting to detector jitter."""
    return round(value / size)


def _zone_signature(zones, zone: str) -> tuple:
    return tuple(
        sorted(
            (
                tile.label,
                tile.track_id if tile.track_id is not None else -1,
                _position_bucket(tile.center_x),
                _position_bucket(tile.center_y),
            )
            for tile in zones.zone_tiles
            if tile.zone == zone
        )
    )


def combine_recommendation_gates(
    validation: StructuredStateResult,
    legacy_state_machine_allow: bool,
    phase_allows_recommendation: bool,
) -> bool:
    return validation.allow_recommend and legacy_state_machine_allow and phase_allows_recommendation
