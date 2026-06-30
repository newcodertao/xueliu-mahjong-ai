from __future__ import annotations

from xueliu_ai.mahjong.tiles import tile_suit, validate_tiles


def missing_suit_tiles(tiles: list[str], missing_suit: str | None) -> list[str]:
    if not missing_suit:
        return []
    suit = missing_suit.upper()
    return [tile for tile in tiles if tile_suit(tile) == suit]


def legal_discards(tiles: list[str], missing_suit: str | None = None) -> list[str]:
    validate_tiles(tiles)
    missing = missing_suit_tiles(tiles, missing_suit)
    candidates = missing if missing else tiles
    return sorted(set(candidates), key=lambda tile: (tile_suit(tile), int(tile[0])))
