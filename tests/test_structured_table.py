from xueliu_ai.realtime_table import TableZones, diagnose_zones
from xueliu_ai.capture.roi_config import Roi
from xueliu_ai.table.event_classifier import EventTileClassifier
from xueliu_ai.table.hand_slot_tracker import HandSlotTracker
from xueliu_ai.table.meld_grouper import group_melds
from xueliu_ai.table.state_fusion import TableStateFusion
from xueliu_ai.table.state_validator import StructuredStateMachine, validate_structured_state
from xueliu_ai.table.structured_types import MeldKind, RegionState, ZoneTile
from xueliu_ai.table.tile_tracker import TileTracker
from xueliu_ai.table.zone_assigner import assign_by_roi_priority
from xueliu_ai.vision.detection_types import Detection


def _det(label: str, x: float, y: float, w: float = 36, h: float = 52) -> Detection:
    return Detection(label, 0.9, x, y, x + w, y + h)


def _tile(label: str, x: float, zone: str = "hand") -> ZoneTile:
    return ZoneTile(label, 0.9, x, 100, x + 36, 152, zone)


def _zones(**changes) -> TableZones:
    values = dict(
        hand=[], bottom_melds=[], left_melds=[], right_melds=[], top_melds=[],
        center_discards=[], all_tiles=[],
    )
    values.update(changes)
    return TableZones(**values)


def test_spatial_meld_grouping_supports_pong_kong_and_suspected_pong() -> None:
    suspected = group_melds([_det("7W", 0, 0), _det("7W", 40, 0)], "bottom_melds", "horizontal")
    pong = group_melds([_det("7W", x, 0) for x in (0, 40, 80)], "bottom_melds", "horizontal")
    kong = group_melds([_det("7W", x, 0) for x in (0, 40, 80, 120)], "bottom_melds", "horizontal")

    assert suspected.groups[0].kind == MeldKind.SUSPECTED_PONG
    assert len(suspected.groups[0].all_tiles) == 3
    assert suspected.groups[0].inferred_tiles[0].inferred
    assert pong.groups[0].kind == MeldKind.PONG
    assert kong.groups[0].kind == MeldKind.KONG


def test_offset_three_tile_group_is_suspected_kong() -> None:
    detections = [_det("5T", 0, 0), _det("5T", 40, 0), _det("5T", 80, 22)]
    result = group_melds(detections, "bottom_melds", "horizontal")
    assert result.groups[0].kind == MeldKind.SUSPECTED_KONG
    assert len(result.groups[0].all_tiles) == 4


def test_single_tile_near_meld_is_candidate_not_hu() -> None:
    result = group_melds([_det("7W", 0, 0)], "left_melds", "vertical")
    assert result.groups == []
    assert result.isolated_tiles[0].zone == "candidate_meld_tiles"


def test_two_kongs_count_as_two_open_melds() -> None:
    first = group_melds([_det("2T", x, 0) for x in (0, 40, 80, 120)], "bottom_melds", "horizontal")
    second = group_melds([_det("5W", x, 80) for x in (0, 40, 80, 120)], "bottom_melds", "horizontal")
    zones = _zones(hand=["1W"] * 7, meld_groups=[*first.groups, *second.groups])
    assert diagnose_zones(zones).open_melds == 2


def test_tile_tracker_retains_short_miss_with_track_id() -> None:
    tracker = TileTracker(max_missed=2)
    first = tracker.update([_tile("8W", 100)])
    missed = tracker.update([])
    restored = tracker.update([_tile("8W", 102)])
    assert first[0].track_id == missed[0].track_id == restored[0].track_id
    assert missed[0].inferred
    assert missed[0].confidence < first[0].confidence


def test_hand_slot_tracker_recovers_middle_missing_tile() -> None:
    tracker = HandSlotTracker()
    full = [_tile(label, x) for label, x in zip(("1W", "2W", "3W", "4W"), (0, 40, 80, 120))]
    tracker.update(full)
    recovered = tracker.update([full[0], full[1], full[3]])
    assert [tile.label for tile in recovered] == ["1W", "2W", "3W", "4W"]
    assert recovered[2].inferred
    assert recovered[2].reason == "hand_slot_history"


def test_event_animation_blocks_recommendation() -> None:
    zones = _zones(event_tiles=["4T"])
    result = validate_structured_state(zones, phase_stable=True, consecutive_stable_frames=5)
    assert result.state == RegionState.TRANSIENT
    assert not result.allow_recommend


def test_roi_priority_assigns_overlap_to_hand() -> None:
    detection = _det("1W", 100, 100)
    table = Roi(100, 100, 1000, 800)
    rois = {
        "my_hand": Roi(180, 180, 200, 200),
        "my_melds": Roi(180, 180, 200, 200),
    }
    assignment = assign_by_roi_priority(detection, table, rois)
    assert assignment is not None
    assert assignment.zone == "hand"


def test_low_confidence_candidate_fills_actual_middle_slot() -> None:
    fusion = TableStateFusion(max_missed=2)
    full = [_tile(label, x) for label, x in zip(("1W", "2W", "3W", "4W"), (0, 40, 80, 120))]
    fusion.update(_zones(hand=[tile.label for tile in full], zone_tiles=full))
    current = [full[0], full[1], full[3]]
    candidate = ZoneTile("3W", 0.61, 80, 100, 116, 152, "hand")
    result = fusion.update(
        _zones(hand=[tile.label for tile in current], zone_tiles=current),
        low_confidence_hand_tiles=[candidate],
    )
    restored = sorted((tile for tile in result.zone_tiles if tile.zone == "hand"), key=lambda tile: tile.center_x)
    assert [tile.label for tile in restored] == ["1W", "2W", "3W", "4W"]
    assert restored[2].reason == "low_confidence_slot_candidate"


def test_isolated_tile_transitions_from_animation_to_stable_hu() -> None:
    classifier = EventTileClassifier(hu_stable_frames=3)
    tile = ZoneTile("7W", 0.9, 20, 200, 56, 252, "unknown_tiles")
    states = []
    zones = _zones(unknown_tiles=["7W"], zone_tiles=[tile])
    for _ in range(3):
        result = classifier.update(zones, width=1000, height=800)
        states.append((result.event_tiles, result.hu_display_tiles))
    assert states[:2] == [(["7W"], []), (["7W"], [])]
    assert states[2] == ([], ["7W"])


def test_candidate_meld_tile_never_transitions_to_hu() -> None:
    classifier = EventTileClassifier(hu_stable_frames=3)
    tile = ZoneTile("7W", 0.9, 20, 200, 56, 252, "candidate_meld_tiles")
    zones = _zones(candidate_meld_tiles=["7W"], zone_tiles=[tile])
    for _ in range(5):
        result = classifier.update(zones, width=1000, height=800)
        assert result.candidate_meld_tiles == ["7W"]
        assert result.hu_display_tiles == []
        assert result.event_tiles == []


def test_structured_state_machine_requires_consecutive_stability() -> None:
    machine = StructuredStateMachine(minimum_stable_frames=3)
    zones = _zones(hand=["1W"] * 14)
    states = [machine.update(zones, phase_stable=True, diagnostics_valid=True) for _ in range(3)]
    assert [state.state for state in states] == [RegionState.UNCERTAIN, RegionState.UNCERTAIN, RegionState.CONFIRMED]
    assert states[-1].allow_recommend
