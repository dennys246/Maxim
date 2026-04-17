# Substrate P4 Stage 1 — Cross-modal binding mechanism

**Plan:** [substrate_p4_cross_modal_binding.md](../plans/archive/substrate_p4_cross_modal_binding.md) Stage 1
**Reproduction runbook:** [protocols/p4_stage1_reproduction.md](protocols/p4_stage1_reproduction.md)
**Date:** 2026-04-15
**Status:** SHIPPED on `feat/substrate-p4`

## TL;DR

Stage 1 lands the **cross-modal binding mechanism** end-to-end on synthetic embeddings: nodes of different modalities co-occurring in the same hippocampus episode now bind via the existing Hebbian close path AND can cue each other through the new modality-filtered retrieval entry point. **No CLIP, no real images, no head-to-head sweep** — Stage 1 ships the plumbing. Stage 2 plugs in real CLIP and Oxford Flowers-102; Stage 3 runs the 20-seed three-arm head-to-head against the OpenCLIP shared-space baseline (the 1.0 gate).

34 mechanism + regression tests pass after the Round 2 pre-merge review fold. 185 tests across the broader episode / persistence / hippocampus surface stay green. Full fast suite (4702 tests) clean apart from two pre-existing worktree-environment failures in `tests/unit/test_leader_proxy.py::TestVersionInfo` (unrelated to P4 Stage 1 — `get_version_info` checks `os.path.isdir(repo_root + "/.git")` but git worktrees use a `.git` file). Zero regressions in P3a, P3b, or P3.5.

## What's wired

### Type-level changes

- `CaptureEvent.modality: SubstrateModality | None = None` — opt-in modality tagging on the existing P3a episode-event input. Default `None` keeps every legacy P3a/P3b call site backward-compatible.
- `PendingEpisodeState.node_modality_buffer: dict[str, SubstrateModality]` — per-node buffer populated as events fold into the pending episode.
- `Hippocampus._node_modality: dict[str, SubstrateModality]` — the persistent per-node sidecar. Mutated only inside `_close_pending_episode_locked` under `_episode_lock`. Persisted under the new `"node_modality"` top-level key in `Hippocampus.dump()` (additive, no migration).
- `SubstrateModality = Literal["text", "vision"]` from `agents/modality.py:73` — same literal already used by the substrate path, so mypy catches modality typos at every call site.

### Mechanism

- **Auto-tag at episode close is structural**, not a helper-discipline rule. Every node in `pending.activated_nodes` that arrived through a `CaptureEvent` carrying a non-None `modality` is tagged in the same loop iteration that appends it. There is no public `tag_node_modality` method; the only ingress is the field on `CaptureEvent`. Forgetting to tag is structurally impossible.
- **Pending-event merge is single-bottleneck.** Commit 0 extracted `Hippocampus._apply_event_to_pending(pending, event)` so both `_start_episode` (the construction seed) and `observe_episode_event` (the extend branch) route through one site. Adding a new per-event field is one edit, not N.
- **`_close_pending_episode_locked` drains the buffer BEFORE both `_episode_store.add(episode)` and `apply_hebbian_on_close`.** A partial failure in either of those subsequent steps cannot leave nodes in the binding graph without their modality entries.

### Retrieval

- **`Hippocampus.retrieve_cross_modal(cue_node_id, target_modality, limit, *, multi_hop=True)`** is the canonical cross-modal entry point.
- Snapshots the matching subset of `_node_modality` into a `frozenset[str]` under `_episode_lock` at filter-build time, releases the lock, returns a lock-free closure that does pure set-membership checking. Mirrors P3b's `episode_membership_filter` snapshot pattern exactly.
- **Defensive `ValueError`** when the cue's own modality already matches `target_modality` — surfaces the caller bug instead of silently returning empty.
- **Cue exemption is structural.** `DependencyGraph.spreading_activation` filters the source node first — for P3b's channel filter that's correct semantics, but for cross-modal the cue is in the OPPOSITE modality bucket from the target set BY DEFINITION. The closure exempts the cue (`node_id == cue_node_id or node_id in allowed`) so traversal seeds correctly while the cue is still excluded from the result ranking by `retrieve_on_cue`'s existing `node != cue_node_id` filter. Same fix is a no-op for the one-hop branch.

### Persistence

- `dump()` writes the sidecar under `"node_modality"`. Acquires `_episode_lock` inside the existing `_rwlock.read()` block — same nested-lock convention `episode_store.to_dict()` already uses. Acquisition order `_rwlock → _episode_lock` is verified deadlock-free by inspection (no `_episode_lock` holder ever acquires `_rwlock`).
- `load_state()` parses + validates outside any lock (unknown modality literals raise `ValueError` BEFORE any mutation), then **clear-then-load** the sidecar inside the existing `_rwlock.write()` block. Wholesale replacement is required for P3.5 atomic rollback semantics: a failed `restore_into` rolls back to the pre-mutation dump, and the rollback must scrub stale entries left behind by the failed attempt.

## Stage 1 test surface (34 tests after Round 2 fold)

### `tests/substrate/test_p4_00_vacuous_pass_guard.py` (13 tests)

The run-first guard. The `00` prefix forces alphabetical pytest collection so this file runs before any other `test_p4_*` file.

- `TestFixtureGeometry` (9): within-pair / cross-pair / orthogonal-centroid invariants, plus a **dim-parametrized dim-invariance test across 64 / 128 / 384 / 512 / 768** pinning the Round 2 Arch-lens fold that rescaled `noise_scale → noise_scale / sqrt(dim)` so within-pair similarity doesn't depend on the embedding dimension.
- `TestECClusteringVacuousPassGuard` (4): paired samples collapse to one EC node id; cross-pair samples land in distinct ids; text+vision for one pair land in DIFFERENT ids.

### `tests/substrate/test_p4_cross_modal_mechanism.py` (21 tests)

- `TestAutoTagAtEpisodeClose` (3): episode close drains the buffer; legacy events do NOT populate sidecar; mixed-modality episodes tag each node with its event's modality
- `TestRetrieveCrossModal` (6): forward / reverse retrieval; same-modality cue raises; untagged cue does not raise; no-cross-modal-episodes returns empty; multi-pair routing isolates partners
- `TestSnapshotPatternFilter` (2): closure has a frozenset cell AND the cell content equals `frozenset({"vision_mug"})` exactly (Round 2 Exec-lens fold — shape-only check let a regression writing `frozenset(_node_modality.keys())` slip through); side-thread holds `_episode_lock` and closure still returns
- `TestPersistence` (5): round-trip; **unknown modality literal rejected WITH prior hippocampus state preserved** (Round 2 Exec-lens fold — previous version only asserted empty state was still empty after the raise); non-dict payload rejected; legacy snapshot loads cleanly; clear-then-load replaces wholesale; **atomic rollback regression guard** monkeypatches `nac_from_snapshot` to raise AFTER verifying hippocampus has been mutated to state C (Round 2 Exec-lens fold — guards against a future `SNAPSHOT_KINDS` reordering silently trivial-passing)
- `TestCueExemptionWithInGraphUntaggedCue` (1): Round 2 Exec-lens fold. Pins the cue-exemption code path (`node_id == cue_node_id or node_id in allowed`) for a probe cue that IS in the binding graph but NOT tagged in `_node_modality` — directly exercises the exemption branch.
- `TestLastWriteWinsOnDuplicateNodeIdWithinEpisode` (1): Round 2 Arch-lens fold. Pins the degenerate-case contract (same node id, two modalities in one episode) that the drain comment described but had no test for.
- `TestStageThreeLimitation` (1): Round 2 Arch-lens fold. Pins the current single-hop-only cross-modal limitation as a regression guard. If a future refactor enables multi-hop traversal through same-modality intermediates (`text_cue → text_bridge → vision_target`), this test FAILS and forces an explicit decision. See the PR description for the Stage 2/3 design-decision note.
- `TestConcurrencyCrossLockSmoke` (1): Round 2 Exec-lens fold. Spawns concurrent `dump()` + `observe_episode_event` workers for 0.5s and asserts both make forward progress. Guards against any future `_episode_lock → _rwlock` inversion that would deadlock against `dump()`'s `_rwlock → _episode_lock` order.

## What Stage 1 deliberately does NOT include

- Real CLIP encoder geometry (Stage 2 deliverable)
- Oxford Flowers-102 fixture (Stage 2)
- VRAM audit (Stage 2)
- 20-seed three-arm sweep against OpenCLIP (Stage 3, 1.0 gate)
- Subprocess mug test on real images (Stage 2)
- Reproducible experiment results JSON (Stages 2/3 — Stage 1 is mechanism unit tests, not a sweep with reportable metrics)

## Pass gate (from substrate_p4_cross_modal_binding.md Stage 1)

1. ✅ Stage 1.5 vacuous-pass guard passes — the cluster-aware fixture lands paired items in one EC node and unpaired items in distinct nodes
2. ✅ All ~10 mechanism tests pass — actually 21 after Round 2 fold, exceeding the plan's draft count
3. ✅ Lock-inversion regression guard passes — closure does not block when `_episode_lock` is held by another thread
4. ✅ Cross-lock concurrency smoke test passes — dump + observe make forward progress under contention
5. ✅ Atomic-rollback regression guard passes — clear-then-load semantics scrub stale entries on failed restore; guarded against `SNAPSHOT_KINDS` reordering via mid-test assertion
6. ✅ Substrate slice + bio-system slice green, 0 regressions (185 tests across P3a/P3b/P3.5/P4/hippocampus)
7. ✅ Full fast suite clean (4702 passed, 1 skipped sentence-transformers path, 2 pre-existing worktree environmental failures in `TestVersionInfo` unrelated to P4)
8. ✅ `Hippocampus.dump()` / `load_state` round-trip preserves `_node_modality` exactly
9. ✅ Ruff check + format clean

## Round 2 pre-merge review fold summary

Two parallel reviewers (Executor lens + Architecture lens) each produced 10 findings. Cross-confirmed classes:

- **Consistency windows** — Arch #1 (load_state split state across `_rwlock.write()` and `_episode_lock`) + Exec #6 (drain-before-add created an asymmetric "sidecar entries without graph nodes" failure window). Both fold directions folded: `load_state` now wraps the entire episode-binding restore block in a single `with _episode_lock:` inside the existing `_rwlock.write()`; `_close_pending_episode_locked` now drains the modality buffer LAST, after both `_episode_store.add` and `apply_hebbian_on_close` succeed, so the sidecar becomes essentially-infallible and any earlier-step failure leaves all three pieces of episode-binding state consistently unmutated.
- **Cue handling** — Arch #3 (multi-hop traversal through same-modality intermediates silently blocked — Stage 3 landmine) + Exec #4 (cue-exemption code path untested for the in-graph-but-untagged case). The Stage 3 design decision is explicitly deferred — the current single-hop-only limitation is pinned in `TestStageThreeLimitation` as a regression guard so a future "fix" forces an explicit design decision rather than silent behavior change. The Exec #4 test gap is closed with a new test class that directly exercises the exemption branch.

Other IMPORTANT folds: runtime validation of `target_modality` literal (Exec #8 — mirrors `load_state` validation), frozenset content assertion (Exec #1), load_state rejects-with-prior-state-preserved (Exec #3), mid-test assertion in the rollback test guarding against `SNAPSHOT_KINDS` reordering (Exec #2), concurrency cross-lock smoke test (Exec #5), last-write-wins regression guard (Arch #7).

Other MINOR folds: dim-invariant `noise_scale` parameter (Arch #4) replacing the dim-coupled `noise_std`, cargo-cult `getattr` defenses dropped (Arch #8), `cosine_similarity` zero-norm guard removed (Exec #10 — let nan be loud), docstring updates for point-in-time read semantics (Arch #2).

Deferred to PR note, not folded: Arch #5 (error-message PII — not applicable; node ids are internal validated identifiers matching the `BackendError.fix_hint` risk profile). Arch #6 (reset-path hygiene) downgraded to PR note — the audit came back empty; no existing reset paths touch episode state, so there is nothing to wire into today. Exec #9 (`_start_episode` redundant `last_tick` set) — cosmetic.

## Open Stage 2/3 design decision

**Should Stage 3 split `retrieve_on_cue`'s `node_filter` into `traversal_filter` (always True) and `result_filter` (target-modality only)?** The current single-hop-only behavior is structurally locked by passing one `node_filter` to `spreading_activation`, which applies to both traversal seeds and returned nodes. A future split would enable multi-hop cross-modal paths through same-modality intermediates — potentially relevant for the P4 mug test if real CLIP + real episodes produce rich Hebbian chains where direct text↔vision edges are rare. The Stage 1 fixture has no such chains so this limitation does not affect Stage 1 shipment. Decision needs to happen BEFORE Stage 3 runs the head-to-head so the metric definition is stable. `TestStageThreeLimitation` is the pin — if the decision is "fix it," update the test to assert the new behavior in the same PR.

## Critical methodological constraint preserved through Stage 1

P4's central commitment is "hippocampus-only cross-modal binding without a shared embedding space beats OpenCLIP shared-space baseline by `mean + 2σ`." Stage 3 will ship **three arms** in the head-to-head, not two:

- **Arm A (substrate-native):** `paraphrase-mpnet` text + CLIP vision + hippocampus episode binding
- **Arm B (substrate-controlled — THE LOAD-BEARING CLAIM):** CLIP-text + CLIP-vision + hippocampus episode binding
- **Arm C (shared-space baseline):** CLIP-text + CLIP-vision + raw cosine similarity (no episodes, no Hebbian)

Commitment #3 is supported **iff Arm B beats Arm C**. The arxiv-comment vector "you used a better text encoder; the win is in the encoder, not the hippocampus mechanism" is structurally refuted because Arm B uses the same encoder as the baseline. **Stage 1's API surface must therefore not bake in any text encoder identity** — and it doesn't. `retrieve_cross_modal(cue_node_id, target_modality, limit, *, multi_hop=True)` takes node IDs and a literal; no embedding dim, no encoder name, no helper methods that assume the substrate's native encoder. The fixture generator parameterizes embedding dim. Stage 2 swaps encoders by configuration, not by code path.

## Reproduction

See [protocols/p4_stage1_reproduction.md](protocols/p4_stage1_reproduction.md). One-line:

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p4_00_vacuous_pass_guard.py tests/substrate/test_p4_cross_modal_mechanism.py -v
```

Expected: 25 passed, ~0.3s wall clock, no LLM, no network, no CLIP, no `semantic` extra required.
