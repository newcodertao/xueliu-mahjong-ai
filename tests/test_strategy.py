from xueliu_ai.strategy.discard_advisor import advise_discard
from xueliu_ai.mahjong.ukeire import effective_draws
from xueliu_ai.strategy.exchange_three_advisor import advise_exchange_three
from xueliu_ai.strategy.missing_suit_advisor import advise_missing_suit


def test_missing_suit_discard_is_prioritized() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"]
    advice = advise_discard(tiles, missing_suit="W")
    assert advice.recommended.endswith("W")


def test_advise_returns_candidates() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"]
    advice = advise_discard(tiles)
    assert advice.recommended
    assert advice.candidates


def test_advise_supports_one_open_meld() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "9B", "9B"]
    advice = advise_discard(tiles, open_melds=1)
    assert advice.recommended
    assert advice.candidates


def test_visible_counts_reduce_effective_draws() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B"]
    draws = effective_draws(tiles)
    exhausted = {tile: 4 for tile in draws}
    assert effective_draws(tiles, visible_counts=exhausted) == {}


def test_missing_suit_advisor_prefers_weakest_suit() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7W", "1T", "2T", "3T", "1B", "9B", "9B"]
    advice = advise_missing_suit(tiles)
    assert advice.suit == "B"


def test_exchange_three_returns_same_suit_tiles() -> None:
    tiles = ["1W", "9W", "2W", "4T", "5T", "6T", "1B", "2B", "3B", "4B", "5B", "6B", "7B"]
    advice = advise_exchange_three(tiles)
    assert len(advice.tiles) == 3
    assert len({tile[-1] for tile in advice.tiles}) == 1
