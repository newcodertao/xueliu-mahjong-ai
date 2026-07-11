from xueliu_ai.realtime_table import TableZones
from xueliu_ai.ui.realtime_app import (
    RegionStateMachine,
    RealtimeStateStabilizer,
    _should_use_recovered_hand,
    _split_my_play_area,
)
from xueliu_ai.realtime_table import diagnose_zones
from xueliu_ai.vision.detection_types import Detection


def _zones(hand: list[str]) -> TableZones:
    return TableZones(
        hand=hand,
        bottom_melds=[],
        left_melds=[],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=hand,
        my_discards=[],
        left_discards=[],
        top_discards=[],
        right_discards=[],
    )


def test_stabilizer_keeps_tile_when_one_frame_misses() -> None:
    full = ["1W", "2W", "3W", "4W", "5W"]
    missed = ["1W", "2W", "4W", "5W"]
    stabilizer = RealtimeStateStabilizer(window_size=5)

    for hand in [full, full, missed, full, full]:
        zones, stable_hand = stabilizer.update(_zones(hand), hand)

    assert stable_hand == full
    assert zones.hand == full


def test_stabilizer_holds_previous_state_on_large_hand_jump() -> None:
    stable = ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"]
    broken = ["1W", "2W", "3W", "4W", "5W", "6W"]
    stabilizer = RealtimeStateStabilizer(window_size=5)

    zones, hand = stabilizer.update(_zones(stable), stable)
    zones, hand = stabilizer.update(_zones(broken), broken)

    assert hand == stable
    assert zones.hand == stable


def test_split_my_play_area_keeps_full_bottom_row_as_hand() -> None:
    detections = [_det(f"{index}W", index * 45, 120) for index in range(1, 10)]
    detections += [_det("2T", 520, 60), _det("2T", 565, 60), _det("2T", 610, 60)]

    hand, melds = _split_my_play_area(detections)

    assert hand == [f"{index}W" for index in range(1, 10)]
    assert melds == ["2T", "2T", "2T"]


def test_split_my_play_area_prefers_low_hand_when_meld_rows_are_longer() -> None:
    detections = [_det("3B", index * 45, 40) for index in range(12)]
    detections += [_det("8W", 200, 150), _det("9W", 245, 150)]

    hand, melds = _split_my_play_area(detections)

    assert hand == ["8W", "9W"]
    assert melds == ["3B"] * 12


def test_hand_recovery_accepts_one_missing_tile() -> None:
    current = ["1W", "2W", "3W", "4W", "5W", "6W", "7W"]
    recovered = ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W"]

    assert _should_use_recovered_hand(current, recovered)


def test_hand_recovery_rejects_over_count_or_large_change() -> None:
    assert not _should_use_recovered_hand(["1W", "2W"], ["1W", "1W", "1W", "1W", "1W"])
    assert not _should_use_recovered_hand(["1W", "2W", "3W", "4W"], ["5W", "6W", "7W", "8W"])


def test_region_state_blocks_invalid_zone_diagnostics() -> None:
    zones = TableZones(
        hand=["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"],
        bottom_melds=["7W", "7W", "7W"],
        left_melds=["7W", "7W", "7W"],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=[],
        my_discards=[],
        left_discards=[],
        top_discards=[],
        right_discards=[],
    )
    machine = RegionStateMachine()

    result = machine.update(zones, diagnose_zones(zones))

    assert not result.valid
    assert result.reason == "diagnostics_invalid"


def test_region_state_blocks_too_many_isolated_tiles() -> None:
    zones = TableZones(
        hand=["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"],
        bottom_melds=[],
        left_melds=[],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=[],
        my_discards=[],
        left_discards=[],
        top_discards=[],
        right_discards=[],
        hu_display_tiles=["2T", "2T", "3T"],
    )
    machine = RegionStateMachine()

    result = machine.update(zones, diagnose_zones(zones))

    assert not result.valid
    assert result.reason == "too_many_unknown_or_event_tiles"


def _det(label: str, x: float, y: float, width: float = 40, height: float = 60) -> Detection:
    return Detection(label, 0.9, x, y, x + width, y + height)
