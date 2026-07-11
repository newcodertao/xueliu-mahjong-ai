from xueliu_ai.mahjong.shanten import best_shanten, is_complete_hand, seven_pairs_shanten


def test_complete_normal_hand() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"]
    assert is_complete_hand(tiles)


def test_seven_pairs() -> None:
    tiles = ["1W", "1W", "2W", "2W", "3T", "3T", "4T", "4T", "5B", "5B", "6B", "6B", "9B"]
    assert seven_pairs_shanten(tiles) == 0


def test_best_shanten_number() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B"]
    assert best_shanten(tiles) <= 0


def test_open_meld_shanten_uses_fixed_melds() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "9B"]
    assert best_shanten(tiles, open_melds=1) <= 0
