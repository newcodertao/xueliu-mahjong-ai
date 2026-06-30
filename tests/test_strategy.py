from xueliu_ai.strategy.discard_advisor import advise_discard


def test_missing_suit_discard_is_prioritized() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"]
    advice = advise_discard(tiles, missing_suit="W")
    assert advice.recommended.endswith("W")


def test_advise_returns_candidates() -> None:
    tiles = ["1W", "2W", "3W", "4W", "5W", "6W", "7T", "8T", "9T", "2B", "3B", "4B", "9B", "9B"]
    advice = advise_discard(tiles)
    assert advice.recommended
    assert advice.candidates
