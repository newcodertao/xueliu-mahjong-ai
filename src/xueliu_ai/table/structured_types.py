from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from xueliu_ai.vision.detection_types import Detection


class MeldKind(str, Enum):
    PONG = "pong"
    KONG = "kong"
    SUSPECTED_PONG = "suspected_pong"
    SUSPECTED_KONG = "suspected_kong"


class RegionState(str, Enum):
    CONFIRMED = "CONFIRMED"
    INFERRED_SAFE = "INFERRED_SAFE"
    UNCERTAIN = "UNCERTAIN"
    TRANSIENT = "TRANSIENT"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ZoneTile:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    zone: str
    group_id: str | None = None
    source: str = "auto"
    reason: str = ""
    track_id: int | None = None
    inferred: bool = False

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @classmethod
    def from_detection(
        cls,
        det: Detection,
        zone: str,
        group_id: str | None = None,
        source: str = "auto",
        reason: str = "",
        track_id: int | None = None,
        inferred: bool = False,
    ) -> "ZoneTile":
        return cls(
            label=det.label,
            confidence=float(det.confidence),
            x1=float(det.x1),
            y1=float(det.y1),
            x2=float(det.x2),
            y2=float(det.y2),
            zone=zone,
            group_id=group_id,
            source=source,
            reason=reason,
            track_id=track_id,
            inferred=inferred,
        )

    def with_assignment(
        self,
        zone: str | None = None,
        group_id: str | None = None,
        source: str | None = None,
        reason: str | None = None,
        confidence: float | None = None,
        inferred: bool | None = None,
    ) -> "ZoneTile":
        return ZoneTile(
            label=self.label,
            confidence=self.confidence if confidence is None else confidence,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            zone=self.zone if zone is None else zone,
            group_id=self.group_id if group_id is None else group_id,
            source=self.source if source is None else source,
            reason=self.reason if reason is None else reason,
            track_id=self.track_id,
            inferred=self.inferred if inferred is None else inferred,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "zone": self.zone,
            "group_id": self.group_id,
            "source": self.source,
            "reason": self.reason,
            "track_id": self.track_id,
            "inferred": self.inferred,
        }


@dataclass(frozen=True)
class MeldGroup:
    group_id: str
    zone: str
    kind: MeldKind
    label: str
    observed_tiles: list[ZoneTile] = field(default_factory=list)
    inferred_tiles: list[ZoneTile] = field(default_factory=list)
    confidence: float = 0.0
    axis: str = "horizontal"
    reason: str = ""

    @property
    def all_tiles(self) -> list[ZoneTile]:
        return self.logical_tiles

    @property
    def observed_count(self) -> int:
        return len(self.observed_tiles)

    @property
    def logical_count(self) -> int:
        return len(self.logical_tiles)

    @property
    def observed_only_tiles(self) -> list[ZoneTile]:
        return list(self.observed_tiles)

    @property
    def logical_tiles(self) -> list[ZoneTile]:
        return [*self.observed_tiles, *self.inferred_tiles]

    @property
    def open_meld_count(self) -> int:
        return 1

    @property
    def is_confirmed(self) -> bool:
        return self.kind in {MeldKind.PONG, MeldKind.KONG}

    @property
    def is_suspected(self) -> bool:
        return self.kind in {MeldKind.SUSPECTED_PONG, MeldKind.SUSPECTED_KONG}

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "zone": self.zone,
            "kind": self.kind.value,
            "label": self.label,
            "confidence": self.confidence,
            "axis": self.axis,
            "reason": self.reason,
            "observed_tiles": [tile.to_dict() for tile in self.observed_tiles],
            "inferred_tiles": [tile.to_dict() for tile in self.inferred_tiles],
            "open_meld_count": self.open_meld_count,
        }


@dataclass(frozen=True)
class StructuredTableState:
    zones: object
    meld_groups: list[MeldGroup]
    confirmed_open_melds: int
    suspected_open_melds: int
    observed_visible_counts: dict[str, int]
    logical_visible_counts: dict[str, int]

    @property
    def inferred_tile_count(self) -> int:
        return sum(len(group.inferred_tiles) for group in self.meld_groups) + sum(
            1 for tile in getattr(self.zones, "zone_tiles", []) if tile.zone == "hand" and tile.inferred
        )

    def consistency_errors(self) -> list[str]:
        confirmed = sum(1 for group in self.meld_groups if group.zone == "bottom_melds" and group.is_confirmed)
        suspected = sum(1 for group in self.meld_groups if group.zone == "bottom_melds" and group.is_suspected)
        errors: list[str] = []
        if self.confirmed_open_melds != confirmed:
            errors.append("confirmed_open_meld_count_mismatch")
        if self.suspected_open_melds != suspected:
            errors.append("suspected_open_meld_count_mismatch")
        for group in self.meld_groups:
            zone_labels = [
                tile.label
                for tile in getattr(self.zones, "zone_tiles", [])
                if tile.zone == group.zone and tile.group_id == group.group_id
            ]
            if sorted(zone_labels) != sorted(tile.label for tile in group.logical_tiles):
                errors.append(f"meld_zone_tiles_mismatch:{group.group_id}")
        return errors
