from xueliu_ai.dataset.roboflow_mahjong_converter import _build_class_map
from xueliu_ai.mahjong.tiles import TILE_TO_INDEX


def test_build_class_map_keeps_numbered_suits() -> None:
    source_names = ["1B", "1C", "1D", "1F", "EW"]
    class_map = _build_class_map(source_names)

    assert class_map[0] == TILE_TO_INDEX["1B"]
    assert class_map[1] == TILE_TO_INDEX["1W"]
    assert class_map[2] == TILE_TO_INDEX["1T"]
    assert 3 not in class_map
    assert 4 not in class_map
