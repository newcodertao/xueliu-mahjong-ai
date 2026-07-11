# Structured Table Reliability Hardening

Date: 2026-07-11

## Scope

This round hardens structured table state only. It does not retrain YOLO, change Mahjong strategy, or redesign the UI.

## Implemented

- Hand slots maintain independent miss counters. Inferred tiles expire after `max_missed` and never refresh themselves as observations.
- Tile tracking matches by geometry across short class jitter and one-frame zone-boundary changes.
- Meld groups expose separate observed and inferred tiles, plus observed and logical counts.
- Strategy-visible counts contain observed tiles only. Logical counts are reserved for structure validation.
- Suspected pong/kong groups do not affect confirmed open-meld counts until stable for three frames.
- Fusion order is tracked zones, meld regrouping, cross-frame confirmation, unified state, validation, recommendation gate.
- `StructuredTableState` owns zones, meld groups, counts, observed visibility, and logical visibility from one result.
- After event classification, manual ROI overrides, and tile-limit reconciliation, the realtime pipeline rebuilds every derived field from the final zones before diagnostics or recommendation.
- Structural stability counts only consecutive safe frames. Any uncertain, transient, or invalid frame resets the counter.
- Stability signatures include track identity, inferred flags, position buckets, unknown/event/HU tiles, observed/logical counts, and diagnostic validity.
- Hard count errors use observed tiles only. Logical completion overflow is reported as uncertainty instead of an impossible observed state.
- Meld confirmation history survives a short whole-group detection gap, while recommendation remains transiently blocked.
- Consistency checks recompute visibility counts and validate group IDs, logical sizes, memberships, and zone references.
- Meld detections that cannot form a valid group are preserved as `candidate_meld_tiles`; no detection is silently discarded during final rebuilding.
- Final post-processed zones are the only source that advances meld confirmation history in realtime mode.
- Meld history uses stable IDs and nearest-center matching instead of frame-local run indexes.
- Geometrically valid 2-4 tile groups use confidence-weighted label voting. Conflicts remain visible and force a suspected group.
- A suspected bottom meld may explain a 10/11-tile hand without producing a hard diagnostic failure; recommendation remains blocked as `UNCERTAIN`.
- Structural signatures are order-independent, so list ordering alone cannot reset stability.
- The structured validator is a hard gate. Suspected melds, hand inference, unknown tiles, animations, count overflow, and inconsistency block strategy execution.
- The realtime UI shows structured state, confirmed/suspected meld counts, inferred count, and blocking reason.
- Candidate meld tiles have priority over HU classification. They cannot be absorbed by broad HU anchors and never enter strategy-visible counts.
- `center_discards` remains the compatibility field name, while overlays and UI present it as the discard area rather than an ambiguous center region.
- Active tables with incomplete recognition use `PLAYING_PARTIAL` and structured `PARTIAL`, rather than being mislabeled as dealing or animation.
- `table_decor_tiles` is reserved as an explicit non-strategy semantic zone for future layout evidence.

## State meanings

- `CONFIRMED`: stable observed state; recommendation may proceed.
- `INFERRED_SAFE`: only a previously confirmed meld has a short tracked miss; recommendation may proceed after stability.
- `PARTIAL`: the game is active but candidate melds or required table structure remain unresolved; recommendation is blocked.
- `UNCERTAIN`: suspected meld, unknown tile, or inferred hand tile; recommendation is blocked.
- `TRANSIENT`: animation or unstable phase; recommendation is blocked.
- `INVALID`: count or internal consistency failure; recommendation is blocked.

## Verification

Regression tests cover hand-slot expiry, one-frame recovery, observed/logical count separation, suspected meld promotion, meld/list consistency, class jitter, zone-boundary crossing, recommendation hard gating, final-state rebuilding, stale derived counts, safe-frame reset, short whole-meld misses, isolated candidate preservation, candidate/HU isolation, label conflicts, stable meld IDs, post-processed promotion, partial-game phase detection, and order-independent stability.

## Deferred structural work

- Explicit legal zone-transition rules for global tile tracking.
- Replay fixtures for adjacent meld and HU-display areas from real problem frames.
