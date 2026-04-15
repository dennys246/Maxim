# Substrate P3.5 — Cross-session persistence + BioSystemSnapshot Protocol

**Status:** ✅ Stages 1 + 2 SHIPPED (PRs #109, #120; 2026-04-14). Stage 3 (real-data 1000-node sweep) deferred until P3a Stage 2 fixture is reusable at scale — not version-gating for 0.3-target. Stages 1+2 together provide the full subprocess round-trip surface P4 cross-modal mug test consumes.
**Scope:** ~500 LOC across 3 stages (Stage 1: ~250, Stage 2: ~150, Stage 3: ~100)
**Target version:** 0.3-target
**Gates:** Not directly version-gating, but load-bearing for P3a round-trip tests, P4 mug-test subprocess round-trip (1.0-GATING), and P5 stress persistence.
**Depends on:** substrate_recognition ✅
**Blocks:** P3a Stage 1 round-trip test (needs P3.5 Stage 1 `_to_dict()` extraction on Hippocampus), P4 (needs full Stage 2+3), P5 (needs full Stage 2+3)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md), [substrate_p3a_episode_binding.md](substrate_p3a_episode_binding.md)

## Goal

Introduce a unified `BioSystemSnapshot` Protocol + `SessionSnapshot` composition class that lets all six bio-systems (ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer, CrossLayerGraph) round-trip through a single dict-shaped serialization surface. The Protocol exists so that downstream phases (P3a persistence round-trip, P4 mug-test subprocess boundary, P5 long-running stress cycles) can treat the six systems uniformly rather than each hand-rolling save/load pairs with divergent signatures.

## Hypothesis (falsifiable)

A `SessionSnapshot.dump()` → disk → `SessionSnapshot.load()` round-trip preserves enough state across all six systems that: (a) retrieval behavior is bit-identical on post-load probes, (b) NAc reward biases round-trip within float tolerance, (c) edge weights in ATL's `DependencyGraph` and `CrossLayerGraph`'s inter-layer edges round-trip exactly, (d) schema evolution is survivable because every sub-snapshot carries an explicit envelope-layer `schema_version: int`.

## Dependencies — scaffolding audit

Existing state (audited 2026-04-14 in worktree; Round 1 review confirmed):

| Surface | Status | Notes |
|---|---|---|
| `ATL.save(path)` / `ATL.load(path)` / `ATL.load_safe(path)` | ✅ exists ([atl.py:306](../../src/maxim/memory/atl.py#L306)) | Builds inline dict under `self._rwlock.read()`, calls `atomic_write_json`. Legacy payload version string `"1.0"`. |
| `Hippocampus.save/load` (via `PersistenceMixin`) | ✅ exists ([hippocampus_persistence.py:32](../../src/maxim/memory/hippocampus_persistence.py#L32)) | Mixin on `Hippocampus`. Legacy payload version string `"3.0"`. Has `associative_graph` key + `load_with_recovery`. No existing `episodes` key — no collision risk for P3a's reserved slot. |
| `NAc.save(path)` / `NAc.load(path)` / `NAc.load_safe(path)` | ✅ exists ([nac.py:1019](../../src/maxim/decisions/nac.py#L1019)) | `reward_bias` field already persisted (P2 addition). Builds dict under `self._lock`. Legacy payload version string `"1.0"`. `save(None)` raises `ValueError` — contract stays on `save()`, not the new `_to_dict()`. |
| `SCN.save(path)` / `SCN.load(path)` | ✅ exists ([time/scn.py:640](../../src/maxim/time/scn.py#L640)) | **`path` is REQUIRED**, no config fallback. No internal lock during dict build. Legacy payload version string `"3.0"`. Dissolved in Stage 1 because `_to_dict()` never takes a path — `SessionSnapshot` owns paths. |
| `CrossLayerGraph.save/load/to_dict` | ✅ exists ([memory/cross_layer.py:216](../../src/maxim/memory/cross_layer.py#L216)) | Owns all cross-layer edges (`DERIVED_FROM`, `INSTANCE_OF`, `INFORMS`). **Added to SessionSnapshot per Round 1 Architecture-lens critical finding** — P4 mug-test (1.0-gating) cross-modal binding lives on this graph; omitting it would silently lose vision↔concept edges across a round-trip. |
| `PerceptTraceBuffer` persistence | ❌ missing ([percept_trace_buffer.py:26](../../src/maxim/memory/percept_trace_buffer.py#L26)) | Has in-memory `snapshot(agent_id=None, min_activation=0.01) -> list[TraceEntry]` helper. Stage 1 ships **real** empty-buffer round-trip by adding thin `dump()`/`load()` methods on the class (not module functions, per Round 1 cross-confirmed finding #2). Non-empty multi-agent edge cases are Stage 2. |
| `utils/atomic_io.atomic_write_json` | ✅ exists | Canonical bulk-write primitive. Reused everywhere. |
| Existing `BioSystemSnapshot` / `SessionSnapshot` type | ❌ none | Grep confirmed zero hits. |
| `memory/store.py` `EpisodicStore` / `CausalStore` / `SemanticStore` protocols | **disambiguated below** | These are **storage-target** Protocols (Mother Maxim DB implementations, `FileEpisodicStore`, etc.); `BioSystemSnapshot` is a **serialization-shape** Protocol. See "Protocol ownership boundaries" below. |

**Key implication — dict extraction is clean.** Every bio-system that has save/load builds its state dict inline inside `save()` and then calls `atomic_write_json`. The dict-building is a zero-behavior-change extract — pull the dict literal out of `save()` into a private `_to_dict() -> dict[str, Any]` method, then `save()` becomes a two-liner. **Lock discipline preservation is explicit** — each extracted `_to_dict()` acquires the SAME lock the original `save()` held (ATL: read lock, NAc: mutex, Hippocampus: its own lock) so that concurrent writes don't race against dict construction.

**Protocol ownership boundaries (Round 1 Arch-lens important finding #6):**

`BioSystemSnapshot` and `memory/store.py::EpisodicStore` overlap visually because both produce JSON for bio-system state. The split is:

- **`BioSystemSnapshot`** (this plan) — the **serialization shape** a live, wired bio-system produces for a process-local snapshot checkpoint. Used by sim orchestrators + cross-session persistence. Always returns a dict; target is `atomic_write_json` or subprocess handoff.
- **`EpisodicStore` / `CausalStore` / `SemanticStore`** (existing) — **storage-target protocols** wired into Hippocampus/NAc/ATL as pluggable backends. `FileEpisodicStore` writes JSON; Mother Maxim's DB extras implement the same Protocol against Postgres. Used during live operation, not for snapshot checkpointing.

A user who wants SessionSnapshot state to land in Postgres writes a `SessionSnapshot` → dict → then loads it into a DB-backed `Hippocampus` constructed with a Postgres-backed `EpisodicStore`. The two protocols compose; they do NOT overlap. **Stage 1 ships a one-paragraph note in `memory/snapshot.py` docstring making this explicit.**

## Stages

### Stage 1 — Protocol + thin adapters + P3a round-trip unblocker

**What's built:**

1. **`src/maxim/memory/snapshot.py` (new, ~280 LOC):**
   - `BioSystemSnapshot` Protocol — **in-place load semantics**, NOT classmethod factory:
     ```python
     from typing import Any, Protocol, runtime_checkable

     @runtime_checkable
     class BioSystemSnapshot(Protocol):
         """Protocol for bio-systems that can dump/load their state as a dict.

         Load is INSTANCE-LEVEL and mutates self in place, preserving runtime
         wires (ATL config, NAc.ec, Hippocampus scn/callbacks). This matches
         the existing save/load shape on all bio-systems and avoids the
         PEP-544 classmethod pitfalls flagged in Round 1 review.
         """

         schema_version: int

         def dump(self) -> dict[str, Any]: ...
         def load_state(self, state: dict[str, Any]) -> None: ...
     ```
     **Rationale (Round 1 cross-confirmed finding #1):** a classmethod factory can't accept the required init params (ATL config, NAc wiring, Hippocampus config + callbacks) and can't re-establish runtime wires. Every existing `load(path)` method on bio-systems already mutates self. In-place load preserves that contract.
   - `SessionSnapshot` dataclass composing all six systems with top-level envelope:
     ```python
     {
         "schema_version": 1,
         "kind": "session",
         "systems": {
             "atl": {"schema_version": 1, "kind": "atl", "payload": {...}},
             "hippocampus": {"schema_version": 1, "kind": "hippocampus", "payload": {...}},
             "nac": {"schema_version": 1, "kind": "nac", "payload": {...}},
             "scn": {"schema_version": 1, "kind": "scn", "payload": {...}},
             "percept_trace_buffer": {"schema_version": 1, "kind": "percept_trace_buffer", "payload": {...}},
             "cross_layer_graph": {"schema_version": 1, "kind": "cross_layer_graph", "payload": {...}},
         },
     }
     ```
     `SessionSnapshot.capture(...)` orchestrates each sub-snapshot; `SessionSnapshot.write(path)` writes to disk via `atomic_write_json`; `SessionSnapshot.restore_into(atl=..., hippocampus=..., ...)` takes live instances and calls each instance's `load_state(payload)` in place. The method is named `load_state` (not `load`) on every bio-system to avoid colliding with the pre-existing `load(path: str | None)` file-I/O method; the rename of `load(path)` across ~37 call sites is out of scope for Stage 1. Both `capture` and `restore_into` accept a `strict: bool = False` flag; under `strict=True`, any partial capture or mismatched restore raises, which P4/P5 harness call sites should use.
   - **Six thin conformance adapters** — the Protocol is the consumer contract; the adapters wrap the new bio-system `_to_dict()` / `_load_from_dict()` methods with the envelope. Adapters live as module-level functions (`atl_to_snapshot(atl)`, `atl_from_snapshot(state, into)`, etc.) and serve as both the canonical call sites for `SessionSnapshot` orchestration and a stable seam for future migration functions.
   - **Envelope-authoritative versioning (Round 1 Arch critical #3):** the envelope `schema_version: int = 1` is the ONLY authoritative version. Legacy payload version strings (`"1.0"`, `"3.0"`) are **tombstoned** — a module-level docstring in `snapshot.py` explicitly states that no new payload-layer version bumps are allowed; all migration lands at the envelope layer. A one-line comment is added next to each bio-system's `save()` pointing at the tombstone rule.
2. **Mechanical `_to_dict()` + `_load_from_dict()` extraction** in all five bio-systems with existing save/load:
   - `ATL._to_dict() -> dict` — moves the inline dict literal from [atl.py:313-320](../../src/maxim/memory/atl.py#L313-L320) into its own method. **Acquires `self._rwlock.read()` internally** so the extraction is lock-equivalent to the pre-refactor code. `ATL._load_from_dict(data: dict) -> None` pulls the deserialization body from `load()`. `ATL.save(path)` / `ATL.load(path)` become thin wrappers.
   - `NAc._to_dict() -> dict` — same pattern, acquires `self._lock`. `NAc.save(None)`'s `ValueError` contract stays on `save()`, not `_to_dict()`.
   - `Hippocampus._to_dict() -> dict` (on `PersistenceMixin` in [hippocampus_persistence.py:32](../../src/maxim/memory/hippocampus_persistence.py#L32), NOT `hippocampus.py:171` — Round 1 Exec important finding). **Reserves an `"episodes": []` top-level key for P3a** — Hippocampus itself doesn't know about episodes yet, so the key is written by P3a once `EpisodeStore` lives on Hippocampus.
   - `SCN._to_dict() -> dict` — builds the same dict [scn.py:640-654](../../src/maxim/time/scn.py#L640-L654). No lock; conditional `oscillator` key preserved. The fact that `SCN.save(path)` has a required `path` is now **irrelevant** to the adapter layer because `_to_dict()` never touches paths — `SessionSnapshot` owns path orchestration. The `SCN.save(path)` signature stays unchanged for backward compatibility.
   - `CrossLayerGraph._to_dict() -> dict` — piggybacks on the existing [cross_layer.py `to_dict()`](../../src/maxim/memory/cross_layer.py) method. Thin adapter — `cross_layer.py::to_dict` already returns a dict, so `_to_dict()` is essentially an alias for Protocol conformance. `CrossLayerGraph._load_from_dict(data)` wraps the existing load deserialization.
3. **`PerceptTraceBuffer.dump()` / `PerceptTraceBuffer.load_state(state)` in Stage 1** — real methods on the class, not module stubs. Empty-buffer round-trip MUST pass in Stage 1 per Round 1 cross-confirmed finding #2. The existing `PerceptTraceBuffer.snapshot()` returns `list[TraceEntry]` — `dump()` wraps that list + ring-buffer metadata (tick counter, capacity, tau, tick_rate, min_activation). `load_state(state)` clears the buffer and replays entries, **trimming to `self._max_entries`** to preserve the ring-buffer capacity invariant, and emitting a WARN log if the dumped tuning parameters diverge from the live instance's values (Round 2 Exec critical #3 + Arch critical #2). Non-empty multi-agent edge cases (agent filter, min_activation parameter, concurrent insertion races) are deferred to Stage 2.
4. **No migration tooling** (Stage 2).
5. **No cross-layer round-trip subprocess harness** (Stage 2).

**Pass gate (Stage 1):**

- All six bio-system classes pass `isinstance(sys, BioSystemSnapshot)` via `runtime_checkable` — every one of them exposes `schema_version: int` + `dump()` + `load_state()` as instance attributes/methods.
- Every bio-system's `dump()` returns a dict with top-level `"schema_version"` key whose value is `int`, not string.
- `Hippocampus.dump()` contains an `"episodes"` key whose value is `[]` (reserved for P3a).
- Round-trip test per bio-system (6 tests): construct, populate with minimal state, `dump()` → `load()` into a fresh instance, assert state equality.
- `SessionSnapshot` round-trip test: compose 6 systems (all empty or minimal), dump → write to tempfile → load → assert equality across all six.
- **PTB empty round-trip:** `PerceptTraceBuffer() → dump() → PerceptTraceBuffer().load(dumped)` produces a buffer whose `snapshot()` returns `[]`. Non-empty case is Stage 2.
- **Lock discipline regression guards (AST-based, NOT string grep):** a test loads each bio-system's `_to_dict()` via `inspect.getsource()` and asserts the source contains the expected lock acquisition pattern (`self._rwlock.read()` for ATL, `self._lock` for NAc, etc.). Round 1 Exec minor #3 flagged simple string grep as false-positive-prone on docstrings; the AST-based guard checks the actual function body tokens.
- **Grep invariant** (separate test): `git grep` on the P3.5 diff confirms the four `save()` methods each contain `self._to_dict()` in their body via `inspect.getsource()` anchored to the function body's first non-blank line.
- `ruff check` + `ruff format` clean on all touched files.
- Fast suite clean (standing exclusions per CLAUDE.md).

**Tests (Stage 1):**

- `tests/unit/test_bio_system_snapshot.py` (new, ~350 LOC):
  - `TestProtocolConformance` — `runtime_checkable` check across all 6 bio-systems.
  - `TestATLRoundTrip` / `TestHippocampusRoundTrip` / `TestNAcRoundTrip` / `TestSCNRoundTrip` / `TestPTBRoundTrip` / `TestCrossLayerGraphRoundTrip` — one round-trip test per bio-system.
  - `TestSessionSnapshotComposition` — full 6-system compose + dump + tempfile round-trip.
  - `TestEnvelopeShape` — every sub-snapshot's envelope has `schema_version: int` + `kind: str` + `payload: dict`.
  - `TestEnvelopeVersioningAuthoritative` — modifying the envelope version number in a dumped state invalidates load (or migrates); modifying the payload-layer legacy version string has no effect on load behavior.
  - `TestLockDisciplinePreserved` — AST-based inspect.getsource check on each `_to_dict()`.
  - `TestLoadPreservesRuntimeWires` — construct an ATL with a non-default config, dump, load a new ATL pre-wired with a DIFFERENT config, assert load() mutates state WITHOUT overwriting config (i.e., load is state-only, not wire-rewriting).

### Stage 2 — Non-empty PTB + migration tooling + subprocess round-trip harness

**What's built (in feat/substrate-p3-5-stage2 worktree, 2026-04-14):**

- **Non-empty `PerceptTraceBuffer` round-trip:** new test file [tests/unit/test_percept_trace_buffer_persistence.py](../../tests/unit/test_percept_trace_buffer_persistence.py) — 9 tests covering 100+ entries, 3-agent isolation, exact activation-strength restoration (no implicit re-record at 1.0), tick counter survival, ring-buffer head/tail invariant when dump-size > live max_entries, concurrent insertion under `dump()` (proves snapshot consistency with no torn reads), tuning-drift WARN.
- **`migrate_session_envelope` + `migrate_subsystem_envelope` + decorator-style registries** in [src/maxim/memory/snapshot.py](../../src/maxim/memory/snapshot.py): `register_session_migration(from_version)` / `register_subsystem_migration(from_version)`. Pure forward migration; one envelope-version step per call; future-version refuse, unknown-source raise, broken-migration (no version progress) raise. `SessionSnapshot.from_dict` and `unwrap_envelope` both auto-walk their respective chains.
- **Session-mode subprocess harness:** [tests/substrate/persistence_harness.py::run_session_round_trip](../../tests/substrate/persistence_harness.py) — captures all 6 systems via `SessionSnapshot.capture(strict=True)`, writes ONE envelope file, spawns a child that loads via `from_file().restore_into(strict=True)` against fresh default-wired instances, runs the same probe, compares pre/post. The existing `run_round_trip` and `persistence_child.py` were extended with a `mode` field (`"per_component"` legacy vs `"session_snapshot"` new) so both paths share infra.
- **6-system probe:** [tests/substrate/probes.py::session_signature](../../tests/substrate/probes.py) returns deterministic counts/signatures for all six bio-systems.
- **Subprocess round-trip test:** [tests/substrate/test_snapshot_subprocess_round_trip.py](../../tests/substrate/test_snapshot_subprocess_round_trip.py) — 4 tests: empty round-trip, populated round-trip (15 alice + 10 bob PTB entries + 5 cross-layer DERIVED_FROM edges + 7 PTB ticks), unknown-kind rejection, state_files field shape.
- **Synthetic v0 legacy fixture for migration testing:** registered inline in `TestMigrationV0ToV1` via `register_session_migration(0)` decorator + an `_isolated_session_migrations` fixture that snapshots/restores the registry per test. 9 tests covering: v0→v1 single-step, auto-migrate via `from_dict`, unknown source raise, future-version refuse, broken migration raise, multi-step v0→v1→v2, duplicate registration raise, sub-system path independent, `unwrap_envelope` auto-runs the sub-system chain.

**Pass gate (Stage 2):**

- PTB round-trip with 100+ entries + 3 agents + concurrent insertion.
- Subprocess round-trip harness passes with all 6 systems.
- Migration `schema_version=0 → 1` green on synthetic legacy fixture.
- Fast suite clean, substrate subset clean.

**Tests (Stage 2):**

- `tests/unit/test_percept_trace_buffer_persistence.py` (new)
- `tests/substrate/test_snapshot_subprocess_round_trip.py` (new)
- `tests/unit/test_bio_system_snapshot.py::TestMigrationV0ToV1` (added to Stage 1 file)

### Stage 3 — real-data sweep + pre-merge review

**What's built:**

- End-to-end sweep on a real 1000+ node synthetic fixture (reuses P3a Stage 2 fixture once available): dump a populated 6-system state, load in subprocess, assert retrieval F1 matches pre-dump within statistical tolerance.
- Pre-merge review round: Executor lens + Architecture lens in parallel, independent; fold critical + important findings into the same branch before the PR opens.

**Pass gate (Stage 3):**

- Retrieval F1 pre-dump vs post-load: delta < 0.01 on a 1000-node fixture.
- Zero cross-confirmed review findings outstanding.
- Substrate subset + fast suite + `ruff check` all green.

## Pass criteria (maps to version gate)

Stage 1 unblocks P3a Stage 1's round-trip test. Stages 2 + 3 together close P3.5's contribution to 0.3-target. P4 (1.0-gating mug test) depends on Stage 2 + 3 being fully shipped — a subprocess round-trip with `CrossLayerGraph` carrying vision↔concept edges is literally the mug test's implementation substrate.

## Load-bearing invariants (post-Stage-2-Round-2 fold)

**Stage 2 additions (review C1 + I1+M5 + I2+I3 Arch + I3 Exec folds):**

- **`SessionSnapshot.restore_into` is all-or-nothing across sub-systems.** Two-phase: (1) snapshot every target via `target.dump()` into a rollback table BEFORE any mutation, (2) apply each adapter; on any failure, walk applied targets in reverse and `load_state` from rollback. Rollback failure is logged via `logger.exception` and the original exception re-raised. Required because P4 mug-test and P5 stress persistence both consume restore_into in hot loops where a torn restore would brick a multi-agent session. Regression guards: `TestRestoreIntoAtomicity` (3 tests).
- **`migrate_session_envelope` / `migrate_subsystem_envelope` never alias the caller's envelope.** Deep-copy at entry so callers that mutate the result post-call cannot corrupt their input or any nested payload. Migration is process-boundary, not hot-path — `copy.deepcopy` cost is fine. Regression guards: `test_migrate_does_not_mutate_caller_envelope` + `test_migrate_no_op_path_returns_independent_dict`.
- **`SessionSnapshot.dump()` returns a deep copy of the envelope.** Same reason — callers should not be able to mutate snapshot state by reference. Regression guard: `TestSessionSnapshotDumpAliasing`.
- **`isolated_migrations()` is the public test-isolation context manager.** Snapshots both registries on entry, clears them, yields, restores on exit (even on test exceptions). Tests no longer reach into private `_SESSION_MIGRATIONS` / `_SUBSYSTEM_MIGRATIONS` globals. A future refactor that puts the registries inside a class only needs to update `isolated_migrations` — tests stay stable.
- **Sub-system migration registry is kind-agnostic; `LATEST_SUBSYSTEM_SCHEMA_VERSION` bumps in lockstep across all six bio-systems.** When NAc's payload restructures and warrants a v2 sub-system envelope, `LATEST_SUBSYSTEM_SCHEMA_VERSION` bumps to 2 and EVERY sub-system's adapter functions must accept v2 (typically via no-op pass-through migrations for the systems that didn't change). The migration function can branch on `envelope["kind"]` for per-kind logic. Avoids the `(kind, version)` tuple registry at the cost of lockstep version bumps. Mother Maxim per-deployment overrides are deferred to post-1.0.
- **Migration chain validates type at every step.** A migration that returns a non-dict, or whose output `schema_version` is not `int`, or whose output version doesn't equal `from_version + 1`, all raise `ValueError` with clear error messages. Prevents broken migrations from silently no-op'ing or returning corrupted shapes. Regression guards: `test_broken_migration_*` (3 tests).

**Stage 1 invariants (carried forward, still in force):**



- **`BioSystemSnapshot.load_state` is in-place instance-mutating, NOT a classmethod factory.** Preserves runtime wires (ATL.config + semantics callbacks, NAc._ec, Hippocampus config, SCN persistence_path, CrossLayerGraph._layers). Regression guards: `TestLoadPreservesRuntimeWires` (3 tests).
- **Method name is `load_state`, NOT `load`.** Avoids colliding with the existing `load(path: str | None)` filesystem-I/O method on all four pre-existing bio-systems. The `load(path)` rename across 37+ call sites is out of scope for Stage 1.
- **Envelope `schema_version: int = 1` is the ONLY authoritative version.** Payload-layer legacy version strings (`"1.0"`, `"3.0"`) are **tombstoned** — Round 2 fold removed the payload-layer version checks from ATL / Hippocampus / NAc / SCN `load_state` for uniformity. Mutating a payload `version` string has no effect on load behavior. Regression guards: `TestEnvelopeVersioningAuthoritative` (4 tests).
- **Every `dump()` holds the same lock the original `save()` did** (ATL rwlock read, NAc mutex, Hippocampus rwlock read, PerceptTraceBuffer mutex). Regression guards: `TestLockDisciplinePreserved` (AST-based `inspect.getsource` check on each dump() method).
- **`CrossLayerGraph` is the 6th bio-system in `SessionSnapshot`.** P4 mug-test depends on vision↔concept edges surviving a subprocess round-trip; omitting it would silently lose them.
- **`PerceptTraceBuffer.load_state` trims to `self._max_entries`** on load — the ring-buffer capacity invariant cannot be exceeded post-load (Round 2 Exec critical #3). Also emits a WARN on tuning-parameter drift between dumped and live values (Round 2 Arch critical #2).
- **`SessionSnapshot.capture` and `restore_into` accept `strict: bool = False`.** Under `strict=True`, any partial capture or mismatched restore raises. P4 / P5 harness call sites should use strict mode.
- **Protocol docstring documents lock-acquisition behavior** so callers on hot-path threads know to spawn `dump()` off to a worker thread.

## Review questions (Stage 3 reviewers — templates for Round 2 code review)

**Executor lens:**
- Does every `_to_dict()` extraction preserve the pre-existing dict structure byte-for-byte? Any silent key rename re-introduces a migration problem we don't have yet.
- Does `PerceptTraceBuffer.dump` hold its lock correctly during snapshot iteration? Any race with concurrent `record()` calls?
- Is `runtime_checkable` on `BioSystemSnapshot` correctly identifying all six bio-systems? What happens if a subclass adds a new field — does the Protocol check still hold?
- Are there thread-safety concerns with calling `_to_dict()` on a live bio-system during a running agent loop?
- Does `SessionSnapshot.restore_into` handle partial failures cleanly (e.g., one bio-system's `load_state` raises mid-way)?

**Architecture lens:**
- Is `SessionSnapshot` the right shape, or should it be a Protocol itself with multiple concrete implementations?
- When P4 ships vision nodes, does a `VisionEncoder` fit into this same Protocol as a 7th system, or does it live inside ATL's snapshot?
- Do the tombstoned payload version strings create a migration footgun if someone ever needs to change a payload dict's shape without an envelope bump?
- Is the `EpisodicStore` vs `BioSystemSnapshot` disambiguation (in `memory/snapshot.py` docstring) strong enough that a Mother Maxim implementer won't accidentally conflate them?

## Deferred follow-ups

1. **Storage compression.** 10k-node snapshots may want a compressed-on-disk form. Deferred to P5.
2. **Partial loads.** Loading just ATL without NAc/SCN. Useful for debugging; not needed for 0.3-target.
3. **Vision encoder as 7th system.** If P4's `VisionEncoder` has enough state to warrant its own snapshot slot, P4 adds it. Otherwise vision nodes live inside ATL's snapshot. **Stage 2 review I1 (Arch lens):** the current `_build_fresh_instance` + 6 explicit kwargs on `capture`/`restore_into` form a 6-7-site touch surface for adding a 7th system. P4's plan-of-record should pick "extra slot" or "inside ATL" before that touch surface bites. If "extra slot," consider also pivoting `capture`/`restore_into` to `**systems: BioSystemSnapshot` with validation against `SNAPSHOT_KINDS`; defer to P4's call.
4. **Fleet compatibility windows for Mother Maxim.** Stage 2 review I4 (Arch lens) flagged that `migrate_*_envelope` future-version refusal (`is newer than this build's latest`) is wrong for a multi-version fleet — a v1 leader reading a v2 dump gets a hard `ValueError` with no compatibility window. Acceptable for 0.3 (Mother Maxim is post-1.0). When fleet semantics ship, this becomes a soft-warning + best-effort partial-load path. Filed here so it isn't re-discovered in the field.
5. **Persistence harness graduation to `src/`.** Stage 2 review I5 (Arch lens) flagged that `tests/substrate/persistence_harness.py` + `persistence_child.py` have outgrown their S3 origin and are now load-bearing for P4/P5/P6/P8. They should move to `src/maxim/test_utils/` (or `src/maxim/bench/`) so they can be imported from non-test code (e.g., a future `maxim bench persistence-cycles` subcommand). Targeted move window: after P4 ships and the harness API stabilizes (any earlier and the move risks churn against Stage 3's real-data sweep).
6. **Tuning drift WARN under P5 dump-reload-100x cadence.** Stage 2 review M3 (Arch lens) — the current `live tuning wins + log WARN` semantic is fine for one-off restore but produces a noise floor of 100 WARN logs / no alarm under P5's cadence. Revisit when P5 ships: either lift to a structured event the operator can track, or pivot to "raise on drift" per call site.

## Not in this plan

- Anything requiring substrate P4/P5/P6/P8 code to exist
- Changes to `memory/store.py` storage-target protocols (`EpisodicStore` / `CausalStore` / `SemanticStore` — those remain pluggable backends, orthogonal to snapshot shape)
- Database-backed snapshot storage (separate, post-1.0)
- Any touch to the NAc reward-bias serialization beyond what's already in place (P2 Stage 2 shipped this)
