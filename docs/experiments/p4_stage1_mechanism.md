# Substrate P4 Stage 1 — Cross-modal binding mechanism

**Plan:** [substrate_p4_cross_modal_binding.md](../plans/substrate_p4_cross_modal_binding.md) Stage 1
**Reproduction runbook:** [protocols/p4_stage1_reproduction.md](protocols/p4_stage1_reproduction.md)
**Date:** 2026-04-15
**Status:** SHIPPED on `feat/substrate-p4`

## TL;DR

Stage 1 lands the **cross-modal binding mechanism** end-to-end on synthetic embeddings: nodes of different modalities co-occurring in the same hippocampus episode now bind via the existing Hebbian close path AND can cue each other through the new modality-filtered retrieval entry point. **No CLIP, no real images, no head-to-head sweep** — Stage 1 ships the plumbing. Stage 2 plugs in real CLIP and Oxford Flowers-102; Stage 3 runs the 20-seed three-arm head-to-head against the OpenCLIP shared-space baseline (the 1.0 gate).

25 mechanism + regression tests pass. 176 tests across the broader episode / persistence / hippocampus surface stay green. Zero regressions in P3a, P3b, or P3.5.

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

## Stage 1 test surface (25 tests)

### `tests/substrate/test_p4_00_vacuous_pass_guard.py` (8 tests)

The run-first guard. The `00` prefix forces alphabetical pytest collection so this file runs before any other `test_p4_*` file. Asserts the cluster-aware fixture's mathematical property holds end-to-end against a real `EntorhinalCortex` instance. If this file fails, the rest of P4 Stage 1 is uninterpretable — random gaussians (no shared centroid) would produce GREEN binding tests on a stub mechanism because each sample would land in its own EC node.

- `TestFixtureGeometry` (4): pure numpy validation of within-pair, cross-pair, and orthogonal-centroid invariants
- `TestECClusteringVacuousPassGuard` (4): paired text/vision samples collapse to one EC node id within their modality bucket; cross-pair samples land in distinct ids; text+vision for one pair land in DIFFERENT ids (EC's modality bucket filter)

### `tests/substrate/test_p4_cross_modal_mechanism.py` (17 tests)

- `TestAutoTagAtEpisodeClose` (4): episode close drains the buffer; legacy events do NOT populate sidecar; mixed-modality episodes tag each node with its event's modality
- `TestRetrieveCrossModal` (6): forward / reverse retrieval; same-modality cue raises; untagged cue does not raise; no-cross-modal-episodes returns empty; multi-pair routing isolates partners
- `TestSnapshotPatternFilter` (2): mock-spy on `retrieve_on_cue` confirms the closure has at least one frozenset cell; side-thread holds `_episode_lock` and the closure must still return — proves the closure is lock-free post-construction (mirror of the P3b regression guard)
- `TestPersistence` (5): round-trip preserves sidecar; unknown modality literal rejected loudly; non-dict payload rejected; legacy snapshot (no `node_modality` key) loads cleanly; clear-then-load semantics replace wholesale; **atomic rollback regression guard** monkeypatches `nac_from_snapshot` to raise after the hippocampus has already been mutated, then asserts post-rollback `_node_modality` matches the pre-mutation state EXACTLY (no stale entries from the failed restore attempt)

## What Stage 1 deliberately does NOT include

- Real CLIP encoder geometry (Stage 2 deliverable)
- Oxford Flowers-102 fixture (Stage 2)
- VRAM audit (Stage 2)
- 20-seed three-arm sweep against OpenCLIP (Stage 3, 1.0 gate)
- Subprocess mug test on real images (Stage 2)
- Reproducible experiment results JSON (Stages 2/3 — Stage 1 is mechanism unit tests, not a sweep with reportable metrics)

## Pass gate (from substrate_p4_cross_modal_binding.md Stage 1)

1. ✅ Stage 1.5 vacuous-pass guard passes — the cluster-aware fixture lands paired items in one EC node and unpaired items in distinct nodes
2. ✅ All ~10 mechanism tests pass — actually 17, exceeding the plan's draft count
3. ✅ Lock-inversion regression guard passes — closure does not block when `_episode_lock` is held by another thread
4. ✅ Atomic-rollback regression guard passes — clear-then-load semantics scrub stale entries on failed restore
5. ✅ Substrate slice + bio-system slice green, 0 regressions (176 tests across P3a/P3b/P3.5/P4/hippocampus)
6. ✅ `Hippocampus.dump()` / `load_state` round-trip preserves `_node_modality` exactly
7. ✅ Ruff check + format clean

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
