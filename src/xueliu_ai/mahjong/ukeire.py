from __future__ import annotations

from xueliu_ai.mahjong.shanten import best_shanten
from xueliu_ai.mahjong.tiles import TILE_NAMES, tiles_to_counts


def effective_draws(tiles: list[str], visible_counts: dict[str, int] | None = None) -> dict[str, int]:
    current = best_shanten(tiles)
    counts = tiles_to_counts(tiles)
    visible_counts = visible_counts or {}
    result: dict[str, int] = {}
    for index, tile in enumerate(TILE_NAMES):
        used = counts[index] + int(visible_counts.get(tile, 0))
        remaining = max(0, 4 - used)
        if remaining == 0:
            continue
        improved = best_shanten([*tiles, tile])
        if improved < current:
            result[tile] = remaining
    return result


def effective_draw_count(tiles: list[str], visible_counts: dict[str, int] | None = None) -> int:
    return sum(effective_draws(tiles, visible_counts).values())
