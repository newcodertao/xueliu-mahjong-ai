from xueliu_ai.mahjong.tiles import parse_tiles, tiles_to_counts, validate_tiles


def test_parse_tiles_and_counts() -> None:
    tiles = parse_tiles("1W,2W,3T")
    counts = tiles_to_counts(tiles)
    assert tiles == ["1W", "2W", "3T"]
    assert sum(counts) == 3


def test_validate_rejects_five_copies() -> None:
    try:
        validate_tiles(["1W"] * 5)
    except ValueError as exc:
        assert "exceeds four" in str(exc)
    else:
        raise AssertionError("expected ValueError")
