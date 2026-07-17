from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    DISCARD = "discard"
    EXCHANGE_THREE = "exchange_three"
    CHOOSE_MISSING_SUIT = "choose_missing_suit"
    PENG = "peng"
    MING_KONG = "ming_kong"
    AN_KONG = "an_kong"
    BU_KONG = "bu_kong"
    HU = "hu"
    PASS = "pass"


@dataclass(frozen=True)
class StrategyAction:
    action_type: ActionType
    tile: str | None = None
    tiles: tuple[str, ...] = ()
    source_player: str | None = None

    @property
    def key(self) -> str:
        target = self.tile or ",".join(self.tiles)
        return f"{self.action_type.value}:{target}"
