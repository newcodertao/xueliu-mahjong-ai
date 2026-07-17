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


def test_equal_count_alternating_misses_recover_physical_slots() -> None:
    tracker = HandSlotTracker(max_missed=2)
    full = [_tile(label, index * 40) for index, label in enumerate(("6W", "7W", "8W", "9W"))]
    first = tracker.update([full[0], full[1], full[3]])
    assert len(first) == 3

    second = tracker.update([full[0], full[1], full[2]])

    assert [tile.label for tile in second] == ["6W", "7W", "8W", "9W"]
    assert second[-1].inferred
    assert tracker.last_recovery["recovered_from_history"] == 1


def test_low_confidence_candidate_has_priority_over_slot_history() -> None:
    tracker = HandSlotTracker(max_missed=2)
    full = [_tile(label, index * 40) for index, label in enumerate(("7W", "8W", "9W"))]
    tracker.update(full)
    low_candidate = replace(full[1], confidence=0.38)

    recovered = tracker.update([full[0], full[2]], [low_candidate])

    assert [tile.label for tile in recovered] == ["7W", "8W", "9W"]
    assert recovered[1].reason == "low_confidence_slot_candidate"
    assert tracker.last_recovery["recovered_from_candidate"] == 1


def test_low_confidence_candidate_seeds_obvious_middle_gap_without_history() -> None:
    tracker = HandSlotTracker(max_missed=2)
    observed = [_tile("7W", 0.0), _tile("9W", 80.0)]
    candidate = replace(_tile("8W", 40.0), confidence=0.62)

    recovered = tracker.update(observed, [candidate])

    assert [tile.label for tile in recovered] == ["7W", "8W", "9W"]
    assert recovered[1].inferred is True
    assert recovered[1].reason == "low_confidence_geometric_gap"
    assert tracker.last_recovery["recovered_from_candidate"] == 1


def test_low_confidence_candidate_does_not_seed_without_two_sided_gap() -> None:
    tracker = HandSlotTracker(max_missed=2)
    observed = [_tile("7W", 0.0), _tile("9W", 40.0)]
    candidate = replace(_tile("8W", 120.0), confidence=0.62)

    recovered = tracker.update(observed, [candidate])

    assert [tile.label for tile in recovered] == ["7W", "9W"]
    assert tracker.last_recovery["recovered_from_candidate"] == 0


def test_repeated_low_confidence_evidence_keeps_recovered_slot_alive() -> None:
    tracker = HandSlotTracker(max_missed=2)
    observed = [_tile("7W", 0.0), _tile("9W", 80.0)]
    candidate = replace(_tile("8W", 40.0), confidence=0.62)

    for _ in range(6):
        recovered = tracker.update(observed, [candidate])
        assert [tile.label for tile in recovered] == ["7W", "8W", "9W"]

    recovered_without_candidate = tracker.update(observed, [])
    assert [tile.label for tile in recovered_without_candidate] == ["7W", "8W", "9W"]
    assert tracker.last_recovery["recovered_from_history"] == 1


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


def test_unknown_frame_resets_structured_stability() -> None:
    machine = StructuredStateMachine(minimum_stable_frames=3)
    normal = _zones(hand=["1W"] * 14)
    assert not machine.update(normal, True, True).allow_recommend
    assert not machine.update(normal, True, True).allow_recommend
    unknown_tile = _tile("2T", 200, "unknown_tiles")
    unknown = _zones([unknown_tile], hand=["1W"] * 14)
    assert machine.update(unknown, True, True).state == RegionState.UNCERTAIN
    recovered = machine.update(normal, True, True)
    assert recovered.reason == "waiting_for_structural_stability"
    assert not recovered.allow_recommend


def test_inferred_hand_frame_resets_structured_stability() -> None:
    machine = StructuredStateMachine(minimum_stable_frames=2)
    labels = ("1W", "1W", "2W", "2W", "3W", "3W", "4W", "4W", "5W", "5W", "6W", "6W", "7W", "8W")
    observed = [_tile(label, index * 40) for index, label in enumerate(labels)]
    normal = _zones(observed, hand=[tile.label for tile in observed])
    machine.update(normal, True, True)
    assert machine.update(normal, True, True).allow_recommend
    inferred_tiles = [*observed[:-1], replace(observed[-1], inferred=True)]
    inferred = _zones(inferred_tiles, hand=[tile.label for tile in inferred_tiles])
    assert machine.update(inferred, True, True).state == RegionState.UNCERTAIN
    assert not machine.update(normal, True, True).allow_recommend


def test_final_zones_rebuild_all_derived_counts() -> None:
    fusion = TableStateFusion()
    pong = [_tile("3T", x, "bottom_melds") for x in (0, 40, 80)]
    initial = fusion.update(_zones(pong))
    extra_discard = _tile("8W", 300, "center_discards")
    final_zones = replace(
        initial,
        center_discards=["8W"],
        zone_tiles=[*initial.zone_tiles, extra_discard],
    )
    rebuilt = fusion.build_structured_state(final_zones)
    assert rebuilt.zones is not final_zones
    assert rebuilt.observed_visible_counts["8W"] == 1
    assert rebuilt.logical_visible_counts["8W"] == 1
    assert rebuilt.consistency_errors() == []


def test_stale_visible_counts_are_detected() -> None:
    fusion = TableStateFusion()
    pong = [_tile("3T", x, "bottom_melds") for x in (0, 40, 80)]
    fusion.update(_zones(pong))
    stale = replace(fusion.last_state, observed_visible_counts={"9W": 4})
    assert "observed_visible_counts_mismatch" in stale.consistency_errors()

    stale = replace(fusion.last_state, logical_visible_counts={"9W": 4})
    assert "logical_visible_counts_mismatch" in stale.consistency_errors()


def test_logical_over_four_is_not_observed_hard_error() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    pair = [_tile("7W", 0, "bottom_melds"), _tile("7W", 40, "bottom_melds")]
    discards = [_tile("7W", 300, "center_discards"), _tile("7W", 350, "center_discards")]
    hand = [_tile(f"{index % 9 + 1}T", 500 + index * 40) for index in range(13)]
    zones = _zones([*pair, *discards, *hand], hand=[tile.label for tile in hand])
    fusion.update(zones)
    diagnostics = diagnose_zones(fusion.last_state.zones)
    assert diagnostics.valid
    assert diagnostics.logical_warnings


def test_confirmed_meld_history_survives_one_missing_frame() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    pong = [_tile("3T", x, "bottom_melds") for x in (0, 40, 80)]
    fusion.update(_zones(pong))
    assert fusion.last_state.meld_groups[0].kind == MeldKind.PONG
    fusion.update(_zones())
    assert fusion.last_state.meld_history_transient
    fusion.update(_zones(pong))
    assert fusion.last_state.meld_groups[0].kind == MeldKind.PONG


def test_final_rebuild_preserves_isolated_meld_tile_as_candidate() -> None:
    fusion = TableStateFusion()
    fusion.update(_zones([_tile("7W", 40, "left_melds")]))
    state = fusion.last_state
    assert state is not None
    assert state.zones.unknown_tiles == []
    assert state.zones.candidate_meld_tiles == ["7W"]
    assert [
        tile.label for tile in state.zones.zone_tiles if tile.zone == "candidate_meld_tiles"
    ] == [
        "7W"
    ]
    validation = validate_structured_state(state, True, 5, diagnostics_valid=True)
    assert validation.state == RegionState.PARTIAL
    assert validation.reason == "candidate_meld_unresolved"
    assert not validation.allow_recommend


def test_mismatched_tile_in_spatial_meld_is_not_dropped() -> None:
    fusion = TableStateFusion()
    tiles = [
        _tile("7W", 0, "bottom_melds"),
        _tile("7W", 40, "bottom_melds"),
        replace(_tile("8W", 80, "bottom_melds"), confidence=0.45),
    ]
    for _ in range(3):
        fusion.update(_zones(tiles))
    state = fusion.last_state
    assert state is not None
    assert len(state.meld_groups) == 1
    group = state.meld_groups[0]
    assert group.kind == MeldKind.SUSPECTED_PONG
    assert group.label == "7W"
    assert sorted(tile.label for tile in group.observed_tiles) == ["7W", "7W", "8W"]
    assert [tile.label for tile in group.conflicting_tiles] == ["8W"]
    assert len(group.observed_tiles) + len(state.zones.unknown_tiles) == 3


def test_high_confidence_conflict_is_not_forced_into_meld() -> None:
    fusion = TableStateFusion()
    tiles = [
        replace(_tile("2W", 0, "left_melds"), confidence=0.94),
        replace(_tile("7W", 0, "left_melds"), y1=60, y2=112, confidence=0.90),
        replace(_tile("7W", 0, "left_melds"), y1=120, y2=172, confidence=0.91),
    ]

    fusion.update(_zones(tiles))
    state = fusion.last_state

    assert state is not None
    assert state.meld_groups == []
    assert sorted(state.zones.candidate_meld_tiles) == ["2W", "7W", "7W"]


def test_low_confidence_meld_candidates_join_observed_group() -> None:
    fusion = TableStateFusion()
    high = [_tile("2T", 160, "bottom_melds")]
    low = [
        replace(_tile("2T", 80, "bottom_melds"), confidence=0.68),
        replace(_tile("2T", 120, "bottom_melds"), confidence=0.33),
    ]

    fusion.update(_zones(high), low_confidence_meld_tiles=low)
    state = fusion.last_state

    assert state is not None
    assert len(state.meld_groups) == 1
    assert state.meld_groups[0].kind == MeldKind.PONG
    assert state.meld_groups[0].observed_count == 3


def test_isolated_pair_retries_with_low_confidence_middle_tile() -> None:
    fusion = TableStateFusion()
    candidates = [
        replace(
            _tile("2T", 80, "candidate_meld_tiles"),
            group_id="bottom_melds_1_1",
            confidence=0.92,
        ),
        replace(
            _tile("2T", 160, "candidate_meld_tiles"),
            group_id="bottom_melds_1_1",
            confidence=0.87,
        ),
    ]
    low = [replace(_tile("2T", 120, "bottom_melds"), confidence=0.52)]
    zones = _zones(candidates)
    zones = replace(zones, candidate_meld_tiles=["2T", "2T"])

    fusion.update(zones, low_confidence_meld_tiles=low)
    state = fusion.last_state

    assert state is not None
    assert state.zones.candidate_meld_tiles == []
    assert state.zones.bottom_melds == ["2T", "2T", "2T"]
    assert state.meld_groups[0].kind == MeldKind.PONG
    assert state.meld_groups[0].observed_count == 3


def test_post_processed_meld_promotes_after_stable_frames() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    group_ids = []
    for _ in range(3):
        pair = [_tile("6T", 200, "bottom_melds"), _tile("6T", 240, "bottom_melds")]
        state = fusion.build_structured_state(_zones(pair))
        group_ids.append(state.meld_groups[0].group_id)
    assert len(set(group_ids)) == 1
    assert state.meld_groups[0].kind == MeldKind.PONG


def test_suspected_meld_explains_hand_count_without_hard_invalid() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    hand = [_tile(f"{index % 9 + 1}T", 400 + index * 40) for index in range(10)]
    pair = [_tile("7W", 0, "bottom_melds"), _tile("7W", 40, "bottom_melds")]
    fusion.update(_zones([*hand, *pair], hand=[tile.label for tile in hand]))
    state = fusion.last_state
    diagnostics = diagnose_zones(state.zones)
    assert diagnostics.valid
    assert any("unconfirmed meld" in warning for warning in diagnostics.logical_warnings)
    validation = validate_structured_state(state, True, 5, diagnostics_valid=diagnostics.valid)
    assert validation.state == RegionState.UNCERTAIN
    assert validation.reason == "unconfirmed_meld_present"


def test_meld_group_id_survives_center_bucket_boundary() -> None:
    fusion = TableStateFusion(meld_confirmation_frames=3)
    ids = []
    for offset in (0, 12, 4):
        pair = [
            _tile("3W", 19 + offset, "bottom_melds"),
            _tile("3W", 59 + offset, "bottom_melds"),
        ]
        fusion.update(_zones(pair))
        ids.append(fusion.last_state.meld_groups[0].group_id)
    assert len(set(ids)) == 1
    assert fusion.last_state.meld_groups[0].kind == MeldKind.PONG


def test_structured_stability_ignores_input_list_order() -> None:
    machine = StructuredStateMachine(minimum_stable_frames=2)
    labels = ("1W", "1W", "2W", "2W", "3W", "3W", "4W", "4W", "5W", "5W", "6W", "6W", "7W", "8W")
    tiles = [replace(_tile(label, index * 40), track_id=index + 1) for index, label in enumerate(labels)]
    first = _zones(tiles, hand=list(labels))
    second = _zones(list(reversed(tiles)), hand=list(labels))
    assert not machine.update(first, True, True).allow_recommend
    assert machine.update(second, True, True).allow_recommend
