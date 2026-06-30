from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GameState:
    hand: list[str] = field(default_factory=list)
    missing_suit: str | None = None
    discards: dict[str, list[str]] = field(
        default_factory=lambda: {"self": [], "left": [], "opposite": [], "right": []}
    )
    melds: dict[str, list[list[str]]] = field(
        default_factory=lambda: {"self": [], "left": [], "opposite": [], "right": []}
    )

    def visible_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pile in self.discards.values():
            for tile in pile:
                counts[tile] = counts.get(tile, 0) + 1
        for groups in self.melds.values():
            for group in groups:
                for tile in group:
                    counts[tile] = counts.get(tile, 0) + 1
        return counts
