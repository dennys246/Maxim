# Substrate P3.5 — Cross-session persistence + BioSystemSnapshot Protocol

**Status:** Stage 1 in progress (2026-04-14)
**Scope:** ~500 LOC across 3 stages (Stage 1: ~200, Stage 2: ~200, Stage 3: ~100)
**Target version:** 0.3-target
**Gates:** Not directly version-gating, but load-bearing for P3a round-trip tests, P4 mug-test subprocess round-trip (1.0-GATING), and P5 stress persistence.
**Depends on:** substrate_recognition ✅
**Blocks:** P3a Stage 1 round-trip test (needs P3.5 Stage 1 shell), P4 (needs full Stage 2+3), P5 (needs full Stage 2+3)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md), [substrate_p3a_episode_binding.md](substrate_p3a_episode_binding.md)

## Goal

Introduce a unified `BioSystemSnapshot` Protocol + `SessionSnapshot` composition class that lets all five bio-systems (ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer) round-trip through a single dict-shaped serialization surface. The Protocol exists so that downstream phases (P3a persistence round-trip, P4 mug-test subprocess boundary, P5 long-running stress cycles) can treat the five bio-systems uniformly rather than each hand-rolling save/load pairs with divergent signatures.

## Hypothesis (falsifiable)

A `SessionSnapshot.dump()` → disk → `SessionSnapshot.load()` round-trip preserves enough state across all five bio-systems that: (a) retrieval behavior is bit-identical on post-load probes, (b) NAc reward biases round-trip within float tolerance, (c) edge weights in ATL's `DependencyGraph` round-trip exactly, (d) schema evolution is survivable because every sub-snapshot carries an explicit `schema_version: int`.

## Dependencies — scaffolding audit

Existing state (audited 2026-04-14 in worktree):

| Surface | Status | Notes |
|---|---|---|
| `ATL.save(path)` / `ATL.load(path)` / `ATL.load_safe(path)` | ✅ exists ([atl.py:306](../../src/maxim/memory/atl.py#L306)) | Builds inline dict, calls `atomic_write_json`. Version field is hardcoded `"1.0"` **string**. |
| `Hippocampus.save/load` (via `PersistenceMixin`) | ✅ exists ([hippocampus.py:171](../../src/maxim/memory/hippocampus.py#L171)) | Version hardcoded `"3.0"` string. Has `associative_graph` key + `load_with_recovery`. |
| `NAc.save(path)` / `NAc.load(path)` / `NAc.load_safe(path)` | ✅ exists ([nac.py:1019](../../src/maxim/decisions/nac.py#L1019)) | `reward_bias` field already persisted (P2 addition). Version `"1.0"` string. |
| `SCN.save(path)` / `SCN.load(path)` | ✅ exists ([time/scn.py:640](../../src/maxim/time/scn.py#L640)) | **Signature mismatch:** `path` is required, not optional. Version `"3.0"` string. |
| `PerceptTraceBuffer` persistence | ❌ missing ([percept_trace_buffer.py:26](../../src/maxim/memory/percept_trace_buffer.py#L26)) | Has in-memory `snapshot()` returning `list[TraceEntry]` but no disk I/O. |
| `utils/atomic_io.atomic_write_json` | ✅ exists | Canonical bulk-write primitive. Reused. |
| Existing `BioSystemSnapshot` / `SessionSnapshot` type | ❌ none | Grep confirmed zero hits. |
| `memory/store.py` `EpisodicStore` / `CausalStore` / `SemanticStore` protocols | orthogonal | These are storage *targets* (where persistence writes), not snapshots. No collision. |

**Key implication.** Every bio-system that has save/load builds its state dict inline inside `save()` and then calls `atomic_write_json`. The dict-building is a zero-behavior-change extract — pull the dict literal out of `save()` into a private `_to_dict() -> dict[str, Any]` method, then `save()` becomes a two-liner. This gives us the "return a dict" half of the Protocol without introducing any new serialization format.

## Stages

### Stage 1 — Protocol shell + thin adapters + P3a round-trip unblocker

**What's built:**

1. `src/maxim/memory/snapshot.py` (new, ~250 LOC):
   - `BioSystemSnapshot` Protocol:
     ```python
     from typing import Any, Protocol, Self, runtime_checkable
     @runtime_checkable
     class BioSystemSnapshot(Protocol):
         schema_version: int
         def dump(self) -> dict[str, Any]: ...
         @classmethod
         def load(cls, state: dict[str, Any]) -> Self: ...
     ```
   - `SessionSnapshot` dataclass composing all five bio-system sub-snapshots with a top-level `{"schema_version": 1, "systems": {...}}` envelope. `dump()` orchestrates each sub-snapshot; `load()` dispatches by kind.
   - Conformance adapters — thin wrapper functions `atl_to_snapshot(atl) -> dict`, `atl_from_snapshot(state) -> ATL`, and same for Hippocampus / NAc / SCN. Each wraps the sub-dict in `{"schema_version": 1, "kind": "atl", "payload": <existing dict>}`.
   - `PerceptTraceBuffer` stub adapters: `ptb_to_snapshot` returns `{"schema_version": 1, "kind": "percept_trace_buffer", "payload": None, "_stub": True}`; `ptb_from_snapshot` raises `NotImplementedError("P3.5 Stage 2 will ship PTB save/load")`.
2. **Mechanical `_to_dict()` extraction** in the four bio-systems that have save/load:
   - [atl.py:306-322](../../src/maxim/memory/atl.py#L306-L322) `ATL.save()` body → new `ATL._to_dict() -> dict`
   - [nac.py:1019](../../src/maxim/decisions/nac.py#L1019) `NAc.save()` body → new `NAc._to_dict() -> dict`
   - [hippocampus.py](../../src/maxim/memory/hippocampus.py) `PersistenceMixin.save()` body → new `_to_dict() -> dict` (reserving an `"episodes": []` top-level key for P3a; the episode store persists its entries there once P3a ships)
   - [scn.py:640](../../src/maxim/time/scn.py#L640) `SCN.save()` body → new `SCN._to_dict() -> dict`
   - Each bio-system's existing `save(path)` refactored to one-line `atomic_write_json(path, self._to_dict())`.
   - Matching `_from_dict(data: dict) -> Self` classmethods where the existing `load()` deserialization logic lives; `load(path)` refactored to `self._from_dict(json.load(open(path)))`.
3. **No migration tooling.** No `migrate(old_state, from_v, to_v)` function. Deferred to Stage 2.
4. **No cross-layer round-trip harness.** Deferred to Stage 2.
5. **PTB save/load** is **not** implemented in Stage 1 — the stub is intentional.

**Pass gate (Stage 1):**

- All five bio-system classes pass `isinstance(sys, BioSystemSnapshot)` via `runtime_checkable` (PTB passes the structural check but its `load` raises when actually called).
- `Hippocampus.dump()` contains an `"episodes"` key whose value is `[]` (reserved for P3a).
- Round-trip test: construct an empty `Hippocampus`, call `dump()`, construct a fresh `Hippocampus`, call `load(dumped)`, assert `.memories == []` and `.associative_graph` is equal.
- Round-trip test: construct an ATL with 3 concepts, dump → load → assert concept IDs match and `graph.nodes()` matches.
- Round-trip test: construct a NAc with `reward_bias` set on one node, dump → load → assert bias value round-trips within `1e-9`.
- Round-trip test: construct a SCN with recorded ticks, dump → load → assert `circadian_bins` match.
- `SessionSnapshot` full round-trip with PTB stubbed: assert sub-snapshot envelopes all have `schema_version=1` and `kind` set correctly.
- `ruff check` + `ruff format` clean on all touched files.
- Fast suite clean (excluding the standing exclusions in CLAUDE.md).

**Tests (Stage 1):**

- `tests/unit/test_bio_system_snapshot.py` (new):
  - `TestProtocolConformance` — `runtime_checkable` check across all 5 bio-systems.
  - `TestATLRoundTrip` / `TestHippocampusRoundTrip` / `TestNAcRoundTrip` / `TestSCNRoundTrip` — one round-trip test per bio-system.
  - `TestSessionSnapshotComposition` — full 5-system compose + dump + load (PTB stubbed); assert envelope shape.
  - `TestPTBStubRaises` — explicit assertion that `ptb_from_snapshot(...)` raises `NotImplementedError` with the expected message substring.
  - `TestSchemaVersionEnvelope` — every sub-snapshot's top-level dict has `schema_version: int` (not string).
- Regression guard: grep-style invariant test that `ATL.save`, `Hippocampus.save`, `NAc.save`, `SCN.save` each contain `self._to_dict()` in the body (ensures nobody silently re-inlines the dict literal).

### Stage 2 — Full protocol + PerceptTraceBuffer + cross-layer round-trip harness

**What's built:**

- Real `PerceptTraceBuffer.save/load` using its existing `snapshot()` helper as the dict source. Lock discipline: read snapshot under its existing lock, serialize outside.
- `BioSystemSnapshot.migrate(old_state, from_v, to_v)` function — pure forward migration, one version step per call, explicit "unknown version → raise" default.
- Cross-layer round-trip harness reusing the S3 subprocess harness in `tests/substrate/persistence_harness.py`: parent dumps a `SessionSnapshot`, subprocess loads it and runs a retrieval probe, asserts probe results match parent-side expectation.
- Schema versioning hygiene pass: audit every bio-system that still has a hardcoded `"1.0"` / `"3.0"` string version, replace with `schema_version: int = 1` (a tombstone comment in each file explains the legacy strings are pinned at the envelope layer, not the payload layer).

**Pass gate (Stage 2):**

- PTB round-trip test with non-empty buffer + multi-agent tag filtering.
- Subprocess round-trip harness passes with all 5 bio-systems.
- Migration from `schema_version=0` (synthetic legacy snapshot) → `schema_version=1` green on a fixture.
- Fast suite clean, substrate subset clean.

**Tests (Stage 2):**

- `tests/unit/test_percept_trace_buffer_persistence.py` (new)
- `tests/substrate/test_snapshot_subprocess_round_trip.py` (new)
- `tests/unit/test_bio_system_snapshot.py::TestMigrationV0ToV1` (added to Stage 1 file)

### Stage 3 — real-data sweep + pre-merge review

**What's built:**

- End-to-end sweep on a real 1000+ node synthetic fixture (will reuse P3a's synthetic fixture once P3a Stage 2 ships): dump a populated 5-system state, load in a subprocess, assert retrieval F1 matches pre-dump within statistical tolerance.
- Pre-merge review round: Executor lens + Architecture lens, both flagging on the full branch tip. Review prompt templates in this file's "Review questions" section below.
- Fold all critical + important findings into one commit before PR opens.

**Pass gate (Stage 3):**

- Retrieval F1 pre-dump vs post-load: delta < 0.01 on a 1000-node fixture.
- Zero cross-confirmed review findings outstanding.
- Substrate subset + fast suite + `ruff check` all green.

## Pass criteria (maps to version gate)

Stage 1 unblocks P3a Stage 1. Stages 2+3 close the P3.5 contribution to 0.3-target. P4 (1.0-gating) depends on Stage 2+3 being fully shipped.

## Load-bearing invariants (filled in AFTER shipping Stage 1)

TODO. Populate after Stage 1 pre-merge review with the actual gotchas encountered.

## Review questions (Stage 3 reviewers — templates for later use)

**Executor lens:**
- Does every `_to_dict()` extraction preserve the pre-existing dict structure byte-for-byte? Any silent key rename re-introduces a migration problem we don't have yet.
- Does `PerceptTraceBuffer.save` hold its lock correctly during snapshot iteration?
- Is `runtime_checkable` on `BioSystemSnapshot` safe given that PTB's stub `load` raises at call time — does `isinstance` check succeed before `load` is exercised?
- Are there any thread-safety concerns with calling `_to_dict()` on a live bio-system during a running agent loop?

**Architecture lens:**
- Is `SessionSnapshot` the right shape, or should it be a Protocol itself with multiple concrete implementations (per-use-case)?
- Does the `{"schema_version": 1, "kind": "...", "payload": ...}` envelope introduce unnecessary nesting compared to a flat payload with a sibling `schema_version` key?
- When P4 ships vision nodes, does `VisionEncoder` state fit into this same Protocol, or does it need a 6th bio-system slot in `SessionSnapshot`?
- Is the `_to_dict()` + `_from_dict()` split clean, or does it reintroduce the "mutable globals + module extraction" class of bug from CLAUDE.md? (Answer should be "no, these are instance methods, not module globals" — but flag any case where it drifts.)

## Deferred follow-ups

1. **Storage compression.** 10k-node snapshots may want a compressed-on-disk form. Deferred to P5.
2. **Partial loads.** Loading just ATL without NAc/SCN. Useful for debugging; not needed for 0.3-target.
3. **Legacy version string → int migration.** The existing string versions (`"1.0"`, `"3.0"`) at the payload layer stay as-is in Stage 1. Stage 2 adds the `schema_version: int` at the envelope layer. A future cleanup pass may unify them.

## Not in this plan

- Anything requiring substrate P4/P5/P6/P8 code to exist
- Changes to `memory/store.py` storage-target protocols
- Database-backed snapshot storage (separate, post-1.0)
