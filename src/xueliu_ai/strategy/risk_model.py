from __future__ import annotations


def simple_discard_risk(tile: str, visible_counts: dict[str, int] | None = None) -> float:
    visible_counts = visible_counts or {}
    seen = visible_counts.get(tile, 0)
    return max(0.0, 1.0 - seen * 0.2)
