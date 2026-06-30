from __future__ import annotations

from collections import Counter

SUITS = ("W", "T", "B")
RANKS = tuple(range(1, 10))
TILE_NAMES = tuple(f"{rank}{suit}" for suit in SUITS for rank in RANKS)
TILE_SET = set(TILE_NAMES)
TILE_TO_INDEX = {tile: index for index, tile in enumerate(TILE_NAMES)}
INDEX_TO_TILE = {index: tile for tile, index in TILE_TO_INDEX.items()}


def normalize_tile(tile: str) -> str:
    value = tile.strip().upper()
    if value not in TILE_SET:
        raise ValueError(f"Unknown tile: {tile!r}")
    return value


def parse_tiles(text: str) -> list[str]:
    if not text.strip():
        return []
    return [normalize_tile(part) for part in text.replace(" ", "").split(",") if part]


def tiles_to_counts(tiles: list[str]) -> list[int]:
    counts = [0] * len(TILE_NAMES)
    for tile in tiles:
        counts[TILE_TO_INDEX[normalize_tile(tile)]] += 1
    return counts


def counts_to_tiles(counts: list[int]) -> list[str]:
    result: list[str] = []
    for index, count in enumerate(counts):
        result.extend([INDEX_TO_TILE[index]] * count)
    return result


def validate_tiles(tiles: list[str], allow_13_or_14: bool = False) -> None:
    for tile in tiles:
        normalize_tile(tile)
    counts = Counter(tiles)
    over = [tile for tile, count in counts.items() if count > 4]
    if over:
        raise ValueError(f"Tile count exceeds four: {', '.join(over)}")
    if allow_13_or_14 and len(tiles) not in (13, 14):
        raise ValueError(f"Expected 13 or 14 tiles, got {len(tiles)}")


def tile_suit(tile: str) -> str:
    return normalize_tile(tile)[-1]


def tile_rank(tile: str) -> int:
    return int(normalize_tile(tile)[0])
