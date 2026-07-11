from dataclasses import replace

from xueliu_ai.realtime_table import TableZones, diagnose_zones
from xueliu_ai.table.hand_slot_tracker import HandSlotTracker
from xueliu_ai.table.state_fusion import TableStateFusion
from xueliu_ai.table.state_validator import (
    StructuredStateMachine,
    combine_recommendation_gates,
    validate_structured_state,
)
from xueliu_ai.table.structured_types import MeldKind, RegionState, ZoneTile
from xueliu_ai.table.tile_tracker import TileTracker


def _tile(label: str, x: float, zone: str = "hand", y: float = 100) -> ZoneTile:
    return ZoneTile(label, 0.9, x, y, x + 36, y + 52, zone)


def _zones(zone_tiles=None, hand=None) -> TableZones:
    return TableZones(
        hand=list(hand or []),
        bottom_melds=[],
        left_melds=[],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=[],
        zone_tiles=list(zone_tiles or []),
    )


def test_real_discard_expires_inferred_hand_tile() -> None:
    tracker = HandSlotTracker(max_missed=2)
    full = [_tile(f"{index % 9 + 1}W", index * 40) for index in range(14)]
    thirteen = full[:-1]
    assert len(tracker.update(full)) == 14
    assert len(tracker.update(thirteen)) == 14
    assert len(tracker.update(thirteen)) == 14
    final = tracker.update(thirteen)
    assert len(final) == 13
    assert not any(tile.inferred for tile in final)


def test_single_frame_hand_miss_can_be_temporarily_recovered() -> None:
    tracker = HandSlotTracker(max_missed=2)
    full = [_tile(label, index * 40) for index, label in enumerate(("1W", "2W", "3W", "4W"))]
    tracker.update(full)
    missed = tracker.update([full[0], full[1], full[3]])
    restored = tracker.update(full)
    assert len(missed) == 4 and missed[2].inferred
    assert len(restored) == 4 and not any(tile.inferred for tile in restored)


def test_inferred_meld_tile_not_counted_as_observed_visible_tile() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    pair = [_tile("7W", 0, "bottom_melds"), _tile("7W", 40, "bottom_melds")]
    fusion.update(_zones(pair))
    state = fusion.last_state
    assert state is not None
    assert state.observed_visible_counts["7W"] == 2
    assert state.logical_visible_counts["7W"] == 3
    assert state.confirmed_open_melds == 0
    assert state.suspected_open_melds == 1


def test_suspected_pong_promotes_after_required_stable_frames() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    pair = [_tile("7W", 0, "bottom_melds"), _tile("7W", 40, "bottom_melds")]
    kinds = []
    for _ in range(3):
        fusion.update(_zones(pair))
        kinds.append(fusion.last_state.meld_groups[0].kind)
    assert kinds == [MeldKind.SUSPECTED_PONG, MeldKind.SUSPECTED_PONG, MeldKind.PONG]
    assert fusion.last_state.confirmed_open_melds == 1


def test_unconfirmed_meld_blocks_recommendation() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    pair = [_tile("7W", 0, "bottom_melds"), _tile("7W", 40, "bottom_melds")]
    fusion.update(_zones(pair))
    validation = validate_structured_state(
        fusion.last_state, phase_stable=True, consecutive_stable_frames=5, diagnostics_valid=True
    )
    assert validation.state == RegionState.UNCERTAIN
    assert not validation.allow_recommend


def test_fused_zone_tiles_and_meld_groups_are_consistent() -> None:
    fusion = TableStateFusion()
    pong = [_tile("3T", x, "bottom_melds") for x in (0, 40, 80)]
    zones = fusion.update(_zones(pong))
    assert fusion.last_state.consistency_errors() == []
    assert [tile.label for tile in zones.zone_tiles if tile.zone == "bottom_melds"] == ["3T"] * 3


def test_two_confirmed_kongs_equal_two_open_melds() -> None:
    fusion = TableStateFusion()
    tiles = [_tile("2T", x, "bottom_melds", 100) for x in (0, 40, 80, 120)]
    tiles += [_tile("5W", x, "bottom_melds", 170) for x in (0, 40, 80, 120)]
    zones = fusion.update(_zones(tiles, hand=["1W"] * 7))
    assert fusion.last_state.confirmed_open_melds == 2
    assert diagnose_zones(zones).open_melds == 2


def test_state_inconsistency_returns_invalid() -> None:
    fusion = TableStateFusion()
    pong = [_tile("3T", x, "bottom_melds") for x in (0, 40, 80)]
    fusion.update(_zones(pong))
    broken = replace(fusion.last_state, confirmed_open_melds=2)
    validation = validate_structured_state(
        broken, phase_stable=True, consecutive_stable_frames=5, diagnostics_valid=True
    )
    assert validation.state == RegionState.INVALID


def test_class_jitter_reuses_one_track() -> None:
    tracker = TileTracker(max_missed=2)
    first = tracker.update([_tile("2T", 100)])
    second = tracker.update([_tile("4T", 102)])
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_one_frame_zone_boundary_crossing_does_not_leave_duplicate_track() -> None:
    tracker = TileTracker(max_missed=2)
    first = tracker.update([_tile("2T", 100, "center_discards")])
    second = tracker.update([_tile("2T", 102, "left_melds")])
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_legacy_gate_cannot_override_structured_validator() -> None:
    zones = _zones()
    validation = validate_structured_state(
        zones, phase_stable=True, consecutive_stable_frames=5, diagnostics_valid=False
    )
    assert not combine_recommendation_gates(validation, True, True)


def test_confirmed_state_allows_recommendation() -> None:
    machine = StructuredStateMachine(minimum_stable_frames=2)
    zones = _zones(hand=["1W"] * 14)
    machine.update(zones, phase_stable=True, diagnostics_valid=True)
    result = machine.update(zones, phase_stable=True, diagnostics_valid=True)
    assert result.state == RegionState.CONFIRMED
    assert combine_recommendation_gates(result, True, True)
