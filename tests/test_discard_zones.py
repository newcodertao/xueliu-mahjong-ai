from xueliu_ai.capture.roi_config import Roi
from xueliu_ai.realtime_table import (
    TableZones,
    classify_table_zones,
    classify_table_zones_by_rois,
    diagnose_zones,
    reconcile_zone_tile_limits,
)
from xueliu_ai.table.structured_types import MeldGroup, MeldKind, ZoneTile
from xueliu_ai.vision.detection_types import Detection


def _det(label: str, x: int, y: int, w: int = 36, h: int = 52) -> Detection:
    return Detection(label=label, confidence=0.9, x1=x, y1=y, x2=x + w, y2=y + h)


def test_discards_are_split_by_table_side() -> None:
    detections = [
        _det("1W", 480, 520),
        _det("2W", 310, 380),
        _det("3W", 480, 230),
        _det("4W", 650, 380),
    ]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.my_discards == ["1W"]
    assert zones.left_discards == ["2W"]
    assert zones.top_discards == ["3W"]
    assert zones.right_discards == ["4W"]
    assert zones.center_discards == ["3W", "2W", "4W", "1W"]


def test_auto_zones_detect_full_bottom_hand_without_manual_roi() -> None:
    labels = ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T", "3T", "4T"]
    detections = [_det(label, 180 + index * 44, 700) for index, label in enumerate(labels)]
    detections += [_det("5B", 430, 360), _det("6B", 480, 360)]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.hand == labels
    assert zones.center_discards == ["5B", "6B"]


def test_auto_zones_includes_split_drawn_tile_in_hand() -> None:
    labels = ["1W", "4W", "5W", "5W", "6W", "6W", "7W", "7W", "9W", "1T", "2T", "4T", "6T"]
    detections = [_det(label, 95 + index * 43, 700) for index, label in enumerate(labels)]
    detections.append(_det("7T", 700, 700))

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.hand == [*labels, "7T"]
    assert zones.bottom_melds == []


def test_auto_zones_keeps_bottom_hand_together_when_one_tile_is_missed() -> None:
    left = ["4W", "5W", "5W", "6W", "6W", "7W"]
    right = ["1T", "2T", "2T", "4T", "6T", "7T"]
    detections = [_det(label, 95 + index * 43, 700) for index, label in enumerate(left)]
    detections += [_det(label, 95 + (index + len(left) + 1) * 43, 700) for index, label in enumerate(right)]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.hand == [*left, *right]
    assert zones.bottom_melds == []


def test_auto_zones_does_not_bridge_far_right_meld_into_hand() -> None:
    hand = ["4W", "5W", "5W", "6W", "6W", "7W", "8W", "4T", "6T", "7T"]
    meld = ["2T", "2T", "2T"]
    detections = [_det(label, 95 + index * 43, 700) for index, label in enumerate(hand)]
    detections += [_det(label, 690 + index * 43, 700) for index, label in enumerate(meld)]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.hand == hand
    assert zones.bottom_melds == meld


def test_auto_zones_prefers_low_short_hand_after_many_melds() -> None:
    melds = [_det("3B", 220 + index * 42, 670) for index in range(12)]
    hand = [_det("8W", 420, 735), _det("9W", 465, 735)]

    zones = classify_table_zones([*melds, *hand], width=1000, height=800)

    assert zones.hand == ["8W", "9W"]
    assert zones.bottom_melds == ["3B"] * 12


def test_auto_zones_completes_two_detected_tiles_as_suspected_pong() -> None:
    detections = [
        _det("7W", 80, 300),
        _det("7W", 75, 360),
        _det("1W", 180, 735),
        _det("2W", 225, 735),
    ]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.left_melds == ["7W", "7W", "7W"]
    assert zones.hu_display_tiles == []
    assert zones.meld_groups[0].kind == MeldKind.SUSPECTED_PONG
    assert sum(tile.inferred for tile in zones.meld_groups[0].all_tiles) == 1
    assert zones.hand == ["1W", "2W"]


def test_auto_zones_accepts_three_detected_tiles_in_meld_area() -> None:
    detections = [
        _det("7W", 80, 300),
        _det("7W", 75, 360),
        _det("7W", 78, 420),
        _det("1W", 180, 735),
        _det("2W", 225, 735),
    ]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.left_melds == ["7W", "7W", "7W"]
    assert zones.unknown_tiles == []
    assert zones.hand == ["1W", "2W"]


def test_auto_zones_does_not_complete_pairs_in_discard_area() -> None:
    detections = [
        _det("7W", 460, 380),
        _det("7W", 505, 380),
        _det("1W", 180, 735),
        _det("2W", 225, 735),
    ]

    zones = classify_table_zones(detections, width=1000, height=800)

    assert zones.center_discards == ["7W", "7W"]


def test_reconcile_zone_tile_limits_removes_hand_overflow_against_visible_melds() -> None:
    zones = TableZones(
        hand=["4W", "5W", "7W", "7W", "1T"],
        bottom_melds=[],
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

    reconciled = reconcile_zone_tile_limits(zones)

    assert reconciled.hand == ["4W", "5W", "7W", "1T"]


def test_diagnostics_do_not_infer_melds_from_short_dealing_hand() -> None:
    zones = TableZones(
        hand=["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T", "2T"],
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
    )

    diagnostics = diagnose_zones(zones)

    assert diagnostics.open_melds == 0
    assert diagnostics.expected_hand_counts == [13, 14]
    assert not diagnostics.valid


def test_diagnostics_accept_open_melds_from_visible_bottom_meld_area() -> None:
    tiles = [
        ZoneTile("2T", 0.9, index * 40, 0, index * 40 + 36, 52, "bottom_melds", "bottom_1")
        for index in range(3)
    ]
    zones = TableZones(
        hand=["1W", "2W", "3W", "4W", "5W", "6W", "7W", "8W", "9W", "1T"],
        bottom_melds=["2T", "2T", "2T"],
        left_melds=[],
        right_melds=[],
        top_melds=[],
        center_discards=[],
        all_tiles=[],
        my_discards=[],
        left_discards=[],
        top_discards=[],
        right_discards=[],
        meld_groups=[MeldGroup("bottom_1", "bottom_melds", MeldKind.PONG, "2T", tiles, confidence=0.9)],
    )

    diagnostics = diagnose_zones(zones)

    assert diagnostics.open_melds == 1
    assert diagnostics.expected_hand_counts == [10, 11]
    assert diagnostics.valid


def test_manual_roi_overlap_prefers_hand_over_meld() -> None:
    detections = [_det("1W", 210, 710), _det("2W", 255, 710), _det("3W", 300, 710)]
    table_roi = Roi(0, 0, 1000, 800)
    rois = {
        "my_melds": Roi(150, 680, 260, 100),
        "my_hand": Roi(180, 690, 260, 100),
    }

    zones = classify_table_zones_by_rois(detections, table_roi, rois, width=1000, height=800)

    assert zones.hand == ["1W", "2W", "3W"]
    assert zones.bottom_melds == []
    assert {tile.zone for tile in zones.zone_tiles} == {"hand"}


def test_zone_tiles_keep_coordinates_confidence_and_group_id() -> None:
    detections = [
        _det("2T", 700, 620),
        _det("2T", 745, 620),
        _det("2T", 790, 620),
        _det("1W", 150, 735),
        _det("2W", 195, 735),
    ]

    zones = classify_table_zones(detections, width=1000, height=800)
    meld_tiles = [tile for tile in zones.zone_tiles if tile.zone == "bottom_melds"]

    assert [tile.label for tile in meld_tiles] == ["2T", "2T", "2T"]
    assert all(tile.confidence == 0.9 for tile in meld_tiles)
    assert all(tile.group_id for tile in meld_tiles)
    assert meld_tiles[0].x1 == 700
