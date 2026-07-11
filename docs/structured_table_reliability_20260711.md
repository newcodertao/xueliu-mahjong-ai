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
- The structured validator is a hard gate. Suspected melds, hand inference, unknown tiles, animations, count overflow, and inconsistency block strategy execution.
- The realtime UI shows structured state, confirmed/suspected meld counts, inferred count, and blocking reason.

## State meanings

- `CONFIRMED`: stable observed state; recommendation may proceed.
- `INFERRED_SAFE`: only a previously confirmed meld has a short tracked miss; recommendation may proceed after stability.
- `UNCERTAIN`: suspected meld, unknown tile, or inferred hand tile; recommendation is blocked.
- `TRANSIENT`: animation or unstable phase; recommendation is blocked.
- `INVALID`: count or internal consistency failure; recommendation is blocked.

## Verification

Regression tests cover hand-slot expiry, one-frame recovery, observed/logical count separation, suspected meld promotion, meld/list consistency, class jitter, zone-boundary crossing, and recommendation hard gating.
