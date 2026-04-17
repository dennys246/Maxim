# Substrate P4 — Cross-modal binding via hippocampus (1.0-GATING)

> **✅ P4 COMPLETE (2026-04-16).** All three stages shipped. Stage 3
> head-to-head sweep: **PASS.** Arm B (CLIP+CLIP+hippo) F1 = 1.000 vs
> Arm C (CLIP+CLIP+cosine) F1 = 0.901. Delta +0.099, both pass criteria
> met (margin + bootstrap). The hippocampus substrate adds value over raw
> CLIP cosine — Hebbian binding produces perfect retrieval while cosine
> misorders semantically close classes (water lily/lotus at 0.814).
>
> Results: [p4_cross_modal_sweep.md](../experiments/p4_cross_modal_sweep.md).
> Option 2 measurement: [p4_option2_measurement.md](../experiments/p4_option2_measurement.md) (decision: defer).
> Reproduction: [protocols/p4_cross_modal_reproduction.md](../experiments/protocols/p4_cross_modal_reproduction.md).

---

**Status:** ✅ COMPLETE (2026-04-16). All stages shipped. See banner above for results. (Original plan-draft status was OPEN 2026-04-14; Round 1 review folded 8 findings.)
**Scope:** ~500 LOC (mechanism + VisionEncoder + retrieval path) + ~100 LOC metric extractor.
**Target version:** 0.3-target. **THIS PHASE GATES 1.0** — losing the OpenCLIP head-to-head is plan-ending and re-opens the architecture's central commitment.
**Depends on:** `substrate_p3a_episode_binding.md` (Stages 1+2 SHIPPED) ✅, `substrate_p3b_channel_integration.md` (Stage 1 SHIPPED) ✅, `substrate_p3_5_persistence_snapshot.md` (Stages 1+2 SHIPPED) ✅
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Blocks:** Substrate P5 stress (P5 reuses the cross-modal mug fixture as one of its load patterns)

## Goal

Prove the **architecture's central claim**: nodes of different modalities co-occurring in the same hippocampus episode can cue each other across modality boundaries through episode reconstruction, and the resulting cross-modal retrieval beats a shared-embedding-space baseline (OpenCLIP) by a statistically meaningful margin.

The "mug test" is the canonical demonstration:

> Session 1: text "mug" + vision-mug-image co-occur in episodes. State is dumped to disk.
> Session 2 (fresh subprocess): present text "mug" alone. The vision-mug node should be retrieved with high confidence, and the same in reverse (vision cue → text node).

If this fails, commitment #3 ("hippocampus-only cross-modal binding without a shared embedding space") is wrong and the architecture pitch must change. If this succeeds against OpenCLIP by `mean + 2σ`, the central claim ships at 0.3-target.

## Hypothesis (falsifiable)

Across 20 seeds, on a fixture of `(text_concept, image)` paired episodes:

1. **Forward retrieval** (text cue → vision node): top-1 retrieval rate ≥ 0.80 mean.
2. **Reverse retrieval** (vision cue → text node): top-1 retrieval rate ≥ 0.80 mean.
3. **False-binding rate** (cue retrieves an UN-paired node from a different episode): ≤ 0.10 mean.
4. **OpenCLIP head-to-head**: Hebbian-via-hippocampus mean retrieval F1 exceeds OpenCLIP cosine-similarity baseline mean by `+ 2 × baseline_std` on the same fixture, same probe set, same seed sweep.
5. **Subprocess round-trip**: post-`SessionSnapshot.from_file().restore_into(strict=True)` retrieval F1 within 0.01 of the pre-dump value (uses the P3.5 Stage 2 harness directly).

Mean + std reported across **20 seeds** (doubled from P3a's 10) because P4 gates 1.0 and statistical power matters.

## Dependencies — scaffolding audit (2026-04-14, in worktree)

| Surface | Status | Notes |
|---|---|---|
| `Episode` dataclass with `modality: str \| None` | ❌ missing | Stage 1 adds. P3b's `episode_membership_filter` already validates against `dataclasses.fields(Episode)` so adding the field is the only new thing required to make `episode_filter(modality="vision")` work end-to-end. |
| `PendingEpisodeState.modality` + `CaptureEvent.modality` | ❌ missing | Stage 1 adds. `Hippocampus.observe_episode_event` already exists; modality propagates from event → pending → finalized episode. |
| `Hippocampus.retrieve_on_cue(multi_hop=True, node_filter=...)` | ✅ ([hippocampus.py:1229](../../src/maxim/memory/hippocampus.py#L1229)) | Multi-hop default + node_filter seam shipped in P3a Stage 2 specifically for this consumer. P4 builds the modality node-filter from `episode_filter(modality=X)` and passes it through. |
| `EpisodeStore.episode_membership_filter(membership_mode="exclusive")` | ✅ ([episode.py:414](../../src/maxim/memory/episode.py#L414)) | P3b shipped `exclusive` mode specifically for the P4 mug test bridge-node detection. Stage 1 mug test exercises `membership_mode="exclusive"` to find nodes that appear in episodes containing BOTH text and vision modalities. |
| `Hippocampus.episode_filter(**criteria)` | ✅ ([hippocampus.py:1332](../../src/maxim/memory/hippocampus.py#L1332)) | General forwarder. P4 calls `h.episode_filter(modality="vision")` once Episode has the field. |
| `DependencyGraph.spreading_activation(..., node_filter=...)` | ✅ | P3a Stage 2 seam. |
| `SessionSnapshot.capture(strict=True).write` + `from_file().restore_into(strict=True)` | ✅ ([snapshot.py](../../src/maxim/memory/snapshot.py)) | P3.5 Stage 2 shipped the atomic rollback + duck-type boundary. P4 mug test consumes it directly. |
| `tests/substrate/persistence_harness.py::run_session_round_trip` | ✅ | P3.5 Stage 2 harness. P4's subprocess mug test is one new test class on top of this. |
| `CrossLayerGraph.add_edge` + `dump`/`load_state` | ✅ ([cross_layer.py](../../src/maxim/memory/cross_layer.py)) | P3.5 Stage 1 added Protocol conformance. P4 uses cross-layer edges as the persistent record of `(text_concept ↔ vision_node)` bridges that survive a session, in addition to the hippocampus episode binding graph. **Decision deferred to Round 1 review:** does the cross-modal bridge edge live in `Hippocampus._binding_graph` (same place as P3a Hebbian edges, modality-tagged via Episode metadata) or in `CrossLayerGraph` (the existing cross-layer surface)? Both are plausible — see "Architectural decision points" below. |
| `VisionEncoder` | ❌ missing | New work. Stage 2. Wraps `sentence-transformers` `clip-ViT-B-32` model. |
| `sentence-transformers` already in `semantic` extra | ✅ | License: Apache 2.0 (verified). No new optional extra needed; `clip-ViT-B-32` is downloadable via the same path that `paraphrase-mpnet-base-v2` already uses for the substrate text encoder. |
| `scenarios/substrate/p4_mug_test.yaml` fixture | ❌ missing | Stage 2 deliverable. |
| `tests/substrate/p4_metrics.py` extractor | ❌ missing | Stage 1 stub, Stage 2 fills in real metric. |
| `docs/experiments/results/p4_cross_modal_sweep.{md,json}` | ❌ missing | Stage 3 deliverable. |
| `docs/experiments/protocols/p4_cross_modal_reproduction.md` | ❌ missing | Stage 3 deliverable. |

## Architectural decisions (locked 2026-04-14 by user signoff, pre Round 1 review)

These were decision-point questions in the plan draft; the user's calls are baked in below. Round 1 review can still pressure-test the methodology question (paraphrase-mpnet vs CLIP-text — see "subtle methodology question" at the bottom of this section), but the architectural shape is committed.

1. **Cross-modal bridge edges live in `Hippocampus._binding_graph`.** Option A — text-mug and vision-mug are both substrate nodes that co-activate in the same hippocampus episode; P3a's `apply_hebbian_on_close` already creates the `ASSOCIATES` edge between them. The "cross-modal" property is metadata on the `Episode` (`modality` field) and on the per-node sidecar (decision 2). Retrieval uses `episode_filter(modality="vision")` to gate which nodes the modality `node_filter` lets through. `CrossLayerGraph` plays NO role in P4 — it stays load-bearing for ATL↔percept-trace↔NAc cross-layer reasoning, not cross-modal retrieval mechanics. **The 1-sentence justification:** cross-modal isn't a NEW retrieval mechanism, it's the SAME mechanism with a modality filter. Zero new graph-traversal code; the entire P3a Stage 2 multi-hop seam is the mechanism.

2. **Per-node modality tagging via sidecar dict on Hippocampus, typed as `SubstrateModality`, populated automatically at episode-close (NOT via a manual `tag_node_modality` call).** `Hippocampus._node_modality: dict[str, SubstrateModality]` where `SubstrateModality = Literal["text", "vision"]` is the existing literal in `agents/modality.py:73`. Matches the standard pattern (NAc `reward_bias`). Persisted under a new `"node_modality"` top-level key in `Hippocampus.dump()` (additive — no migration needed).

   **Round 1 review folds (all cross-confirmed, all CRITICAL):**

   - **The type is `SubstrateModality`, not `str`** (Arch C2). Using a bare `str` re-introduces the silent-no-op pattern the executor-bootstrap rule pushed down into types: a typo (`"vis"` or `"text "` with trailing space) silently matches zero nodes. The literal lets mypy catch typos at the call site. `Episode.modality` is dropped entirely (see Arch M2 fold below) — the field was ill-defined for the cross-modal-by-construction use case.
   - **Auto-tag at episode-close, not via a manual entry point** (Arch I1 + Exec C2). The plan-draft's `Hippocampus.tag_node_modality(node_id, modality)` is a footgun: every caller has to remember to call it OR the sidecar drifts from the binding graph silently. Fix: `CaptureEvent` carries `modality: SubstrateModality | None`, `PendingEpisodeState.modality` flows from the first event, `_close_pending_episode_locked` automatically tags every node in `pending.activated_nodes` with the pending modality into `_node_modality` BEFORE the Hebbian close runs. This makes "you cannot bind a node into an episode without also tagging its modality" a structural invariant — forgetting becomes impossible at the API level.
   - **Lock discipline + snapshot-pattern filter callback** (Exec C1 + Arch I3, the P3b regression class). **Stage 1 implementation correction (2026-04-14):** `_close_pending_episode_locked` runs under `self._episode_lock` (a `threading.RLock`), NOT `self._rwlock.write()` — the plan draft mis-stated the lock layout. Hippocampus has two locks: `_rwlock` guards memory state (memories dict, context index, associative graph) and `_episode_lock` guards every piece of episode-binding state (pending episode, episode store, episode-id ordinal). `_node_modality` is conceptually episode-binding state and follows the same convention: **mutations** in `_close_pending_episode_locked` happen under `_episode_lock`; **reads from `dump()`** acquire `_episode_lock` inside the existing `_rwlock.read()` block (mirrors how `episode_store.to_dict()` is already called inside `dump()`'s outer `_rwlock.read()`); **`load_state` clear-then-load** acquires `_episode_lock` inside the existing `_rwlock.write()` block. Lock-acquisition order is `_rwlock → _episode_lock`; verified by inspection that no `_episode_lock` holder ever acquires `_rwlock`, so the inverse order never appears and there is no deadlock. **`retrieve_cross_modal`'s snapshot pattern** acquires `_episode_lock` (NOT `_rwlock.read()`) to snapshot the matching subset of `_node_modality` into a local `frozenset[str]` at filter-build time, releases the lock, returns a lock-free closure over the frozenset. The lock-inversion hazard (`spreading_activation` invoking the callback while holding `binding_graph._lock`) is killed by the snapshot pattern itself, independent of which lock built the snapshot — mirroring P3b's `episode_membership_filter` shape. The lambda-over-self pattern from the plan draft is forbidden.
   - **`load_state` clears before loading** (Arch C4). `_node_modality` is replaced wholesale on `load_state`, NOT merged — required for P3.5 atomic rollback semantics (the rollback table calls `load_state(pre_mutation_dump)` which must scrub stale entries from the failed mutation attempt). Regression guard: `test_node_modality_stale_entries_cleared_on_rollback` in Stage 1.

   **`Episode.modality` is dropped** (Arch M2). The field was ill-defined for the cross-modal-by-construction use case: the mug test's whole point is that ONE episode contains BOTH a text-mug and a vision-mug node co-activating, which makes per-episode modality nonsensical (`"text"` or `"vision"` or `"mixed"`?). The per-node sidecar is the source of truth. P3b's `episode_filter(modality=X)` would have been a filter nobody calls. Drop the field entirely; the only consumer of modality information is `retrieve_cross_modal` which goes through the per-node sidecar via `node_filter`.

3. **Vision encoder via `sentence-transformers`** (Apache-2.0, already in the project's `semantic` extra). Apache-2.0 license confirmed (`pip show sentence-transformers` → `License: Apache 2.0`). Uses `sentence-transformers` CLIP variant (`clip-ViT-B-32`) for both vision encoding (Stage 2) and the OpenCLIP baseline text-encode side (Stage 3). No new dep — extends the existing `semantic` extra. Single-image encode goes through `SentenceTransformer("clip-ViT-B-32").encode(image)` which returns a 512-dim numpy array directly. Cleaner than pulling `open-clip-torch` as a separate dep; same model weights under the hood.

4. **20-seed sweep** (parent plan's recommended budget for the 1.0-gating phase). At 20 seeds the standard error of the mean is ~`σ/4.5`, sensitive to a `0.45 × σ` shift. Failure-handling escalation: if the head-to-head margin is borderline at 20 seeds, escalate to 40 before declaring fail. **Round 1 fold (Arch M1, Exec I2):** the Gaussian SE assumption is wrong for bimodal per-seed distributions and breaks when `baseline_std → 0` at ceiling. Stage 3 ships TWO independent pass criteria:
   - (a) `substrate_mean ≥ baseline_mean + 2 × max(empirical_baseline_std, 0.02)` — the original `+2σ` check with a floor on effective std so a saturated baseline can't auto-fail the substrate.
   - (b) **Paired bootstrap 95% CI** on the per-seed (substrate − baseline) delta excludes zero. This is robust to bimodality and to within-seed correlation between the two arms.

   Both must hold for Stage 3 to PASS. If they disagree, ship a 40-seed rerun before any further interpretation.

6. **Stage 3 ships THREE arms, not two — the encoder confound is controlled, not hand-waved** (Arch C3 + Exec C3, cross-confirmed CRITICAL). The plan-draft's defense ("apples-to-apples would force both paths through CLIP-text, which would defeat the substrate's mechanism") is FALSE as stated and would be the single most likely arxiv-comment vector after publication. The substrate's mechanism is "text node binds to vision node via episode reconstruction," and which encoder produces the text node embedding is orthogonal to whether the episode mechanism works. Stage 3 ships:

   - **Arm A (substrate-native):** `paraphrase-mpnet-base-v2` text + `clip-ViT-B-32` vision + hippocampus episode binding.
   - **Arm B (substrate-controlled, the load-bearing claim):** `clip-ViT-B-32` text + `clip-ViT-B-32` vision + hippocampus episode binding.
   - **Arm C (shared-space baseline):** `clip-ViT-B-32` text + `clip-ViT-B-32` vision + raw cosine similarity (no episodes, no Hebbian).

   The **load-bearing publishable claim** is **Arm B beats Arm C by ≥ 2 × baseline_std AND paired-bootstrap CI excludes zero** — this isolates the hippocampus contribution from the text encoder contribution. Commitment #3 is supported iff Arm B wins. Arm A beating Arm C remains in the writeup as a secondary "the native stack does even better" finding but is NOT the gate. If Arm A wins but Arm B loses, the plan ships a partial-fail report: the mechanism alone is insufficient, the substrate's text encoder is doing load-bearing work, and commitment #3 needs to be re-stated. Running three arms on the same 20 seeds costs ~33% more wall-clock time, which is trivial compared to plan-ending credibility risk.

5. **Mug fixture: Oxford Flowers-102 (CLIP ~66% headroom), NOT imagenette.** Round 1 review (cross-confirmed by both lenses, CRITICAL severity) caught a fixture saturation hazard: imagenette's 10 classes are exactly the distribution OpenCLIP zero-shot saturates near 95-99% on, which collapses `baseline_std → 0` and renders the `+2σ` margin vacuous. **The pre-review pick was wrong.** The replacement criterion is "CLIP zero-shot performance leaves enough variance for the head-to-head margin to mean something" — published OpenCLIP numbers on Oxford Flowers-102 are around 66% top-1 (Radford et al. 2021 §3.1.4 + OpenCLIP benchmarks), giving both arms substantial headroom. Alternative considered: CUB-200 (~50% CLIP zero-shot, even more headroom) — rejected because 200 classes inflate the per-class power problem to 0.25 samples/class on a 50-pair fixture. Flowers-102 with 10-class subset (5 samples each) is the sweet spot.

   - **Oxford Flowers-102** via `torchvision.datasets.Flowers102` — 102 flower categories, 1020/1020/6149 train/val/test, same Oxford VGG source Radford et al. 2021 measured CLIP on. Cached at `~/.cache/maxim/p4_flowers/` (user-scoped, repo-agnostic).
   - **Stage 2 v1 plan amendment (2026-04-15):** the plan originally specified `datasets.load_dataset("nelorth/oxford-flowers")` with the HuggingFace `datasets` library + `HF_DATASETS_CACHE` default path, folded in Exec M2 of Round 1. **Stage 2 v1 silently switched to `torchvision.datasets.Flowers102`** because torchvision is a transitive dep of `sentence-transformers` (already present in the `semantic` extra) while the `datasets` library would have added a new optional dep. Phase 2B's calibration sweep was built and run against torchvision, the fixture descriptor is indexed against torchvision's `Flowers102.classes` ordering, and Phase 2D v1's mug test consumed the same path. The two libraries both wrap the same Oxford VGG source files but have INDEPENDENT class-label orderings — the fixture's `class_idx` fields are only valid under torchvision's ordering. Re-doing Stage 2 via `datasets` would require regenerating the fixture. The amendment records the decision: **torchvision is the pinned choice** because (a) zero additional dep cost vs the `datasets` library, (b) torchvision exposes a documented class-name list we can pin in-repo as a drift guard, (c) the Stage 2 v1 work is already indexed against it. The `nelorth/oxford-flowers` HF dataset remains a fallback if torchvision ever ships a Flowers102 reindex, in which case the loader's drift guard raises and we regenerate via either source under a `change-fixture` commit.
   - **Subset choice:** pick 10 classes whose CLIP zero-shot accuracy lands in `[0.50, 0.85]` (substantial headroom in both directions). Stage 2's first deliverable is a one-time CLIP-baseline calibration sweep over the full 102 classes to identify the 10 hardest-but-not-impossible classes — pin them by name in `scenarios/substrate/p4_mug_test.yaml`.
   - **Sample count:** 5 images per class × 10 classes = 50 pairs (matches the original budget). Per-probe metric is `pooled across all 50` (Exec I3 fold).
   - **Drift guard:** the 102 class names are pinned in-repo (`tests/substrate/p4_fixture_loader.py::FLOWERS102_CLASS_NAMES`) as a tuple, asserted against `torchvision.datasets.Flowers102.classes` at load time. If torchvision ever reorders, the assertion raises immediately instead of silently returning wrong images. Fixture SHA pin complements this: SHA covers the YAML bytes, drift guard covers the class-index-to-name mapping.

   Stage 1 uses synthetic embeddings (no real images, no `datasets` dep) so mechanism work runs without the fixture. Stage 2 ships the fixture descriptor + the 10-class calibration sweep + the bundled selection rationale.

## Stages

### Stage 1 — Mechanism (synthetic embeddings)

**Goal:** Prove the cross-modal retrieval path works end-to-end with NO CLIP dependency. **Stage 1 tests the BINDING + RETRIEVAL plumbing, not the encoder geometry.** EC clustering quality is deferred to Stage 2 — Stage 1's job is "given paired (text, vision) nodes co-activated in an episode, can the modality-filtered retrieval path return the cross-modal partner." Round 1 review (Exec C4 + Arch I2) caught a vacuous-pass risk: random gaussian embeddings would never land in the same EC cluster, the episodes would never form, and Stage 1 tests would silently pass on a stub. Fix: synthetic embeddings are deterministically constructed cluster-centric — paired (text, vision) embeddings share a tight cluster centroid + seeded noise, distinct pairs use orthogonal centroids.

**What's built:**

1. **`SubstrateModality` literal extended where needed.** Currently `Literal["text", "vision"]` in `agents/modality.py:73`. P4 uses it as-is — no extension. Future audio is a one-line literal extension.
2. **`PendingEpisodeState.modality: SubstrateModality | None` + `CaptureEvent.modality: SubstrateModality | None`** added so the field flows from incoming events through the pending state. **`Episode.modality` is NOT added** — Round 1 review (Arch M2) dropped it as ill-defined for the cross-modal-by-construction use case. The per-node sidecar is the source of truth.
3. **`Hippocampus._node_modality: dict[str, SubstrateModality]`** sidecar map. **Populated automatically inside `_close_pending_episode_locked`** — every node in `pending.activated_nodes` is tagged with `pending.modality` BEFORE the Hebbian close runs. NO public `tag_node_modality` method (Arch I1 + Exec C2 fold — manual entry was a footgun). Persisted under a new `"node_modality"` top-level key in `Hippocampus.dump()` (additive — no migration needed because P3.5 envelope is at v1 and payload-layer keys are append-only). `load_state` REPLACES the dict wholesale (clear-then-load, NOT merge — Arch C4 fold for P3.5 atomic rollback).
4. **`Hippocampus.retrieve_cross_modal(cue_node_id, target_modality: SubstrateModality, limit, *, multi_hop=True)`** — the canonical P4 entry point. Mirror P3b's snapshot-pattern filter (Exec C1 + Arch I3 fold):
   - Acquires `self._rwlock.read()` and snapshots `frozenset(node_id for node_id, m in self._node_modality.items() if m == target_modality)` into a local `allowed: frozenset[str]`.
   - Releases the rwlock.
   - Builds a lock-free closure: `lambda node_id: node_id in allowed`.
   - Calls `self.retrieve_on_cue(cue_node_id, limit, multi_hop=multi_hop, node_filter=closure)`.
   - Returns the same `list[tuple[str, float]]` shape as `retrieve_on_cue`.
   - **Defensive check:** asserts `self._node_modality.get(cue_node_id) != target_modality` at the rwlock-read step (M1 fold — catch the "passed same modality" caller bug with a clear `ValueError` instead of silently returning zero).
5. **`PersistenceMixin.dump()` and `load_state()` updated** in `hippocampus_persistence.py` to add the `"node_modality"` top-level key. Read happens under `self._rwlock.read()`, write under `self._rwlock.write()`. (Exec I4 fold — call out the module path explicitly.)
6. **Cluster-aware synthetic mug fixture** (in-test, NOT on-disk; Exec C4 + Arch I2 fold): a generator that produces N pairs where each pair has a unique `centroid: np.ndarray` of dimension `D` drawn from `random.normal()` and the (text, vision) embeddings are `centroid + small_noise`. Distinct pairs have orthogonal centroids. This guarantees paired items DO land in the same EC cluster at the configured threshold and unpaired items do NOT. Stage 1 explicitly tests this property via a Stage 1.5 test (see below).
7. **`tests/substrate/test_p4_cross_modal_mechanism.py`** — mechanism tests:
   - `test_synthetic_pair_lands_in_same_ec_cluster` — Stage 1.5 vacuous-pass guard. Asserts that paired (text, vision) synthetic embeddings ARE clustered together by `LinguisticEncoder` + EC, and unpaired items are NOT. **This test must pass before any of the retrieval tests below run** — it's the precondition that prevents Stage 1 from passing on a stub.
   - `test_episode_close_auto_tags_nodes_with_modality` — `_close_pending_episode_locked` populates `_node_modality` for every node in the closed episode.
   - `test_retrieve_cross_modal_text_to_vision` (forward path on the cluster-aware fixture)
   - `test_retrieve_cross_modal_vision_to_text` (reverse path)
   - `test_retrieve_cross_modal_excludes_same_modality_cue` — defensive ValueError when cue and target are the same modality.
   - `test_retrieve_cross_modal_no_cross_modal_episodes_returns_empty` (negative case)
   - `test_retrieve_cross_modal_filter_uses_snapshot_pattern` — mock-spy on `retrieve_on_cue` asserts `node_filter` is a closure over a frozen set, NOT a lambda over `self._node_modality`. Stops a future regression from re-introducing the lock-inversion class.
   - `test_node_modality_filter_holds_no_lock_after_construction` — spawns a side thread that grabs `self._rwlock.write()` and holds it; main thread calls the filter from inside a fake `spreading_activation` simulation. Asserts the call returns in < 0.5s. Mirror of P3b's analogous regression guard.
   - `test_node_modality_sidecar_persists_through_session_snapshot` (uses P3.5 `SessionSnapshot.capture/restore_into` in-process)
   - `test_node_modality_stale_entries_cleared_on_rollback` — Arch C4 fold. Build a Hippocampus with 3 tagged nodes, capture, rollback dump. Force a `restore_into` failure on a later sub-system (broken adapter). Assert that after rollback, `_node_modality` matches the pre-mutation state EXACTLY (no merged stale entries from the failed restore attempt).
   - `test_subprocess_mug_test_synthetic` (uses P3.5 `run_session_round_trip` with a `session_signature` probe extended for cross-modal counts)

**Pass gate (Stage 1):**

- The Stage 1.5 vacuous-pass guard (`test_synthetic_pair_lands_in_same_ec_cluster`) passes — paired items cluster together, unpaired items do not. **If this fails, the mechanism tests below are not interpretable.**
- All ~10 mechanism tests pass.
- The lock-inversion regression guard (`test_node_modality_filter_holds_no_lock_after_construction`) passes.
- The atomic rollback regression guard (`test_node_modality_stale_entries_cleared_on_rollback`) passes.
- Substrate slice + bio-system slice green, 0 regressions.
- `Hippocampus.dump()`/`load_state` round-trip preserves `_node_modality` exactly.
- The subprocess mug test (synthetic embeddings) demonstrates: parent dumps → child loads → text cue retrieves the paired vision node with non-zero weight.
- Ruff check + format clean.

**Stage 1 explicitly does NOT ship:** real CLIP encoder, real images, OpenCLIP baseline, 20-seed sweep, results JSON, EC-cluster-quality validation on real-world embeddings. Those are Stage 2 and Stage 3.

### Stage 2 — Real CLIP + Oxford Flowers-102 fixture + calibration sweep

**Goal:** Replace synthetic embeddings with real CLIP-encoded text + vision pairs sourced from HuggingFace `nelorth/oxford-flowers`. Calibrate the 10-class subset to land in CLIP's headroom band (`[0.50, 0.85]` zero-shot accuracy). Prove the mug test on real images. Audit dual-encoder VRAM footprint.

**What's built:**

1. **`src/maxim/models/vision/clip_encoder.py::VisionEncoder`** — minimal class that wraps `sentence_transformers.SentenceTransformer("clip-ViT-B-32")`, encodes a single PIL image to a 512-dim numpy array, returns a `Percept(modality="vision", embedding=...)`. Model cached at the `sentence-transformers` default path under `HF_HOME` / `~/.cache/huggingface/`. Single-object only — no detection / no multi-object handling. Lazy-imported so absence of the `semantic` extra is a clean `pytest.importorskip` skip. **Same `SentenceTransformer` instance can encode both image and text** (CLIP shares the model) — Stage 3's Arms B and C share this encoder.

2. **Calibration sweep on the full 102 flower classes** (one-shot Stage 2 deliverable, NOT a test). For each of the 102 classes: encode 5 sample images through CLIP, encode the class name through CLIP-text, compute mean cosine similarity per class, then run a zero-shot top-1 retrieval over all 102 class names. Identify the 10 classes whose CLIP zero-shot accuracy lands in `[0.50, 0.85]` — this is the headroom band. Pin the chosen 10 by name in `scenarios/substrate/p4_mug_test.yaml`. Output: `docs/experiments/p4_clip_calibration.md` with the full 102-class table + the 10-class selection rationale. **This sweep runs ONCE** — re-running is forbidden under the "no band-aid fixture tweaks" rule (see Failure handling below).

3. **`scenarios/substrate/p4_mug_test.yaml`** — fixture descriptor pinning 10 chosen flower class names + 5 sample-indices per class = 50 `(text_concept, image_idx)` pairs. The descriptor file's SHA-256 is asserted by a regression test (Exec I5 fold) — any change to the fixture without an explicit `change-fixture` commit message + a corresponding hash bump in the test will fail.

4. **`tests/substrate/p4_fixture_loader.py`** — pure loader + sample-image accessor. Uses `datasets.load_dataset("nelorth/oxford-flowers")` with the **default `HF_DATASETS_CACHE` cache path** (do NOT override — Exec M2 fold). Returns `(PIL.Image, class_name)` pairs by `(class_name, sample_index)` key. Reused by Stage 3.

5. **`tests/substrate/test_p4_fixture_validation.py`** — Round 1 review caught the naming gap (Exec M3): renamed with `test_` prefix so pytest auto-collects. Asserts:
   - The pinned 10-class subset matches the calibration sweep's headroom band.
   - The fixture descriptor SHA-256 matches the pinned hash.
   - Forward AND reverse retrieval rate ≥ 0.70 mean across 10 seeds on the real CLIP encoder (not the head-to-head sweep — that's Stage 3).
   - **Per-class retrieval rate ≥ 0.50** for every chosen class (Arch M3 fold — catches a single-class collapse hidden by the mean).
   - If any class drops below 0.50, swap it for another class from the headroom band BEFORE Stage 3 starts. This is fixture hygiene at Stage 2 validation time, NOT a band-aid at Stage 3 pass/fail time.

6. **`tests/substrate/test_p4_real_clip_mug_test.py`** — end-to-end subprocess mug test:
   - Parent: load CLIP, encode all 50 (text, image) pairs into 512-dim embeddings, build hippocampus, observe co-activation episodes for each pair, **dump pre-encoded embeddings to a temp file alongside the SessionSnapshot** (Exec I6 fold — child does NOT need the encoder).
   - Child (subprocess via `run_session_round_trip`): load the dumped substrate state, load the pre-computed embeddings file, present text cue (substrate node ID), return retrieved vision node IDs from `retrieve_cross_modal`.
   - Assert: top-1 forward retrieval rate ≥ 0.70 single-seed; per-class assertion: at least 8 of 10 classes retrieve correctly.
   - **Subprocess does NOT load CLIP** — eliminates the 30-60s cold-start cost the plan-draft would have incurred per subprocess invocation. The persistence round-trip is what's being tested; the encoder is a pure function whose output is pre-computed in the parent.

7. **VRAM audit deliverable** (Arch I4 fold). `docs/experiments/p4_stage2_vram_audit.md` — pre/post `nvidia-smi` snapshot from the RTX 5080 leader showing combined VRAM footprint of `clip-ViT-B-32` + `paraphrase-mpnet-base-v2` + Qwen-14B Q4_K_M @ 12k context. If the combined footprint breaches the 2026-04-13 spillover-detection plan's `max(1.5, 0.55 * weights_gb)` headroom budget, **Stage 3 runs on a dedicated worktree with `MAXIM_LLM_ENABLED=0`** so the LLM path doesn't compete for VRAM during the sweep.

   **Phase 2E leader audit — result + drift notes (2026-04-15, to be resolved on peer):** Audit shipped as [docs/experiments/p4_vram_audit.md](../experiments/p4_vram_audit.md). **VERDICT: WARN** at n_ctx=8096 — 3.02 GB free against 4.4 GB recommended headroom, above the 1.5 GB hard floor. Three drift notes surfaced during the run:
   - **Ctx drift.** Plan + deliverable spec both say `@ 12k context` but the leader spills over at n_ctx=12000 (user-confirmed) and runs at n_ctx=8096. The 8k measurement is the honest worst case this hardware can serve — raising to e.g. 10k would tighten the verdict but still wouldn't match the formula's 12k assumption. Stage 3 planning should use 8k as the real baseline, not the aspirational 12k.
   - **Audit script docstring drift.** [scripts/p4_vram_audit.py](../../scripts/p4_vram_audit.py) module docstring claims the mug test step covers "EC pattern-complete + hippocampus episode binding", but [`_run_mug_test_encoding`](../../scripts/p4_vram_audit.py) only encodes 50 images + 10 texts through the raw encoders — no EC, no hippocampus. Peak VRAM is under-measured by whatever those two paths add (likely <100 MB, unverified).
   - **Fixture loader pre-population assumption.** [tests/substrate/p4_fixture_loader.py::load_fixture_images](../../tests/substrate/p4_fixture_loader.py) calls `Flowers102(..., download=False)`. On a clean leader the audit runs ~60s of CLIP+mpnet loads before failing at the fixture step with torchvision's misleading "Dataset not found or corrupted" message. The runbook calls this out, but the surprise-trap cost real wall clock on first leader run. Either pre-populate via `download=True` inside the audit, or raise a clearer error upstream — peer to decide.

**Pass gate (Stage 2):**

- Calibration sweep complete; 10 chosen classes pinned in the fixture descriptor.
- `VisionEncoder` loads CLIP model on first use (network download via sentence-transformers default cache); subsequent loads are warm.
- Single-image encode latency < 200 ms on RTX 5080 leader.
- Oxford Flowers-102 downloads cleanly via `datasets` to the default HF cache.
- Fixture validation: 10-seed mean forward retrieval ≥ 0.70 AND per-class minimum ≥ 0.50.
- Subprocess mug test green; child does NOT load CLIP.
- VRAM audit committed; combined footprint within budget OR Stage 3 worktree config decision documented.
- `pytest tests/substrate/test_p4_real_clip_mug_test.py` skips cleanly when `sentence-transformers` is not installed.
- Substrate slice + bio-system slice green, 0 regressions.

### Stage 3 — 20-seed three-arm sweep + pre-merge review

**Goal:** Run the head-to-head with the encoder confound controlled. Declare pass or fail. Ship the experiment writeup + reproduction protocol. **This is the 1.0 gate.**

**What's built:**

1. **`tests/substrate/baselines/openclip_baseline.py`** — pure shared-embedding-space baseline (Arm C). For each test probe: encode the text cue with `clip-ViT-B-32` text, encode every candidate vision image with `clip-ViT-B-32` vision, return top-K by cosine similarity. No hippocampus, no episodes, no Hebbian.
2. **`tests/substrate/p4_metrics.py`** — metric extractor: `compute_cross_modal_metrics(retrieval_results, ground_truth) -> {forward_rate, reverse_rate, false_binding_rate, f1}`. **Metric definition pinned (Exec I3 fold):** rates are `pooled per-probe` (n=50 per seed), NOT `mean of per-class rates`. Documented in `docs/experiments/protocols/p4_cross_modal_reproduction.md` BEFORE any seeds are run.
3. **Three-arm sweep harness** (Arch C3 + Exec C3 fold) — parametrized test that runs ALL THREE arms against the same fixture across 20 seeds, captures per-seed metrics for each arm, computes mean + std + paired-bootstrap CI between arm pairs:

   - **Arm A:** `paraphrase-mpnet-base-v2` text + `clip-ViT-B-32` vision + `Hippocampus.retrieve_cross_modal` (substrate-native).
   - **Arm B:** `clip-ViT-B-32` text + `clip-ViT-B-32` vision + `Hippocampus.retrieve_cross_modal` (substrate-controlled — **the load-bearing claim**).
   - **Arm C:** `clip-ViT-B-32` text + `clip-ViT-B-32` vision + raw cosine similarity (shared-space baseline).

4. **`docs/experiments/p4_cross_modal_sweep.md`** — full experiment writeup: hypothesis, fixture, metric definition, all three arms, per-seed table, paired-bootstrap CI between (B vs C) and (A vs C), pass/fail call, what changed if anything mid-stage.
5. **`docs/experiments/results/p4_cross_modal_sweep.json`** — machine-readable results.
6. **`docs/experiments/protocols/p4_cross_modal_reproduction.md`** — step-by-step rerun protocol (mirrors P2/P3a's protocol shape). **Single-shot rerun rule** (Exec I5 fold): the 20-seed sweep runs ONCE under any given (fixture, encoder, mechanism) configuration. Re-running requires writing a new failure report at `docs/experiments/p4_cross_modal_failure_<date>.md` that explicitly cites the reason for re-running and what configuration changed. The protocol doc names this rule + links to it from the test docstrings.
7. **Pre-merge review round** — Executor lens + Architecture lens in parallel against the Stage 3 commit. Fold critical/important findings before opening the PR. Per the project's `feedback_review_before_ship.md` rule.

**Pass gate (Stage 3) — BOTH criteria must hold:**

- **(a) Margin criterion:** Arm B mean exceeds Arm C mean by `≥ 2 × max(empirical_std_arm_C, 0.02)` across 20 seeds. The `0.02` floor (Arch M1 + Exec I2 fold) prevents a saturated baseline from auto-failing the substrate by collapsing the effective margin to zero.
- **(b) Bootstrap criterion:** Paired bootstrap 95% CI on per-seed (Arm B − Arm C) delta excludes zero. Robust to bimodality, robust to within-seed correlation between arms.
- Forward retrieval rate ≥ 0.80, reverse ≥ 0.80, false-binding ≤ 0.10 on Arm B — all `pooled per-probe` means.
- Subprocess round-trip retrieval F1 within 0.01 of pre-dump value (verification of the P3.5 substrate underneath P4).
- Zero cross-confirmed pre-merge review findings outstanding.
- Substrate slice + fast suite + ruff check + format all green.

**If (a) and (b) disagree, ship a 40-seed rerun before any further interpretation.** Disagreement is the only sanctioned trigger for a sweep re-run within the same configuration — see the single-shot rerun rule above.

**Failure handling.** If Arm B fails to beat Arm C:

- This is **plan-ending for commitment #3** as currently stated.
- Do NOT ship band-aid fixture tweaks. Do NOT increase the seed budget hoping for variance to favor us. Do NOT cherry-pick an alternative metric. Do NOT re-run with a different fixture without a written failure report.
- **The fixture descriptor SHA-256 assertion in the test prevents fixture mutation without an explicit `change-fixture` commit.**
- File a Stage-4 failure report at `docs/experiments/p4_cross_modal_failure_<date>.md` with full per-seed data for all three arms, the configuration of each arm, and the configuration of the fixture.
- If Arm A beats Arm C but Arm B loses to Arm C: ship a partial-fail report. The mechanism alone is insufficient; the substrate's text encoder is doing load-bearing work. Commitment #3 needs to be re-stated as "the mechanism + the right text encoder beats shared space."
- Open a discussion: revisit commitment #3 (drop "without shared embedding space"?), revisit the mechanism (is per-node modality tagging too coarse?), or revisit the fixture (are the images ambiguous in a way the substrate genuinely cannot resolve but CLIP can?).
- Reproducible failure is more valuable than fake success — the P2 retrospective specifically called this out.

## Pass criteria (maps to version gate)

Stage 3 closes 0.3-target's P4 line item AND closes the 1.0 cross-modal commitment. Stages 1+2 are stepping stones with no version gate of their own.

## Load-bearing invariants (post-Round-1 fold)

- **`Episode.modality` is NOT a field** (Arch M2 fold). The mug test's whole point is that one episode contains BOTH text and vision modalities co-activating, which makes per-episode modality nonsensical. Per-node sidecar is the source of truth.
- **`Hippocampus._node_modality: dict[str, SubstrateModality]` is the single source of truth** for per-node modality. Typed as the existing `SubstrateModality = Literal["text", "vision"]` literal from `agents/modality.py:73` (Arch C2 fold) so mypy catches typos at every call site. Persisted under the `"node_modality"` top-level key in `Hippocampus.dump()`. ATL/EC remain modality-agnostic.
- **`_node_modality` is populated automatically inside `_close_pending_episode_locked`** (Arch I1 + Exec C2 fold), NOT via a manual `tag_node_modality` call. Every node in a closing episode is tagged with the pending episode's modality before the Hebbian close runs. Forgetting becomes structurally impossible — there is no API to call.
- **`_node_modality` writes happen ONLY under `self._rwlock.write()`**. Reads in `dump()` happen under `self._rwlock.read()`. `load_state` REPLACES the dict wholesale (clear-then-load, NOT merge — Arch C4 fold for P3.5 atomic rollback semantics).
- **`Hippocampus.retrieve_cross_modal` snapshots the modality node_filter** via `frozenset(node_id for node_id, m in self._node_modality.items() if m == target_modality)` under `self._rwlock.read()` at filter-build time, then closes the lock-free callback over the frozenset (Exec C1 + Arch I3 fold — mirrors P3b's `episode_membership_filter` exactly). The lambda-over-self pattern is forbidden because `spreading_activation` invokes the callback while holding `binding_graph._lock` and the AB-BA inversion is the same regression class P3b's Round 2 fold killed.
- **Defensive ValueError on same-modality cue** (Exec M1 fold). `retrieve_cross_modal(cue_node_id, target_modality)` asserts `_node_modality.get(cue_node_id) != target_modality` at the rwlock-read step. Catches the "passed same modality by mistake" caller bug with a clear error instead of silently returning zero results.
- **Stage 1 synthetic embeddings are cluster-aware, NOT random gaussians** (Exec C4 + Arch I2 fold). Paired (text, vision) embeddings share a tight cluster centroid + seeded noise; distinct pairs use orthogonal centroids. Stage 1.5 vacuous-pass guard (`test_synthetic_pair_lands_in_same_ec_cluster`) asserts paired items DO cluster together and unpaired items do NOT, BEFORE any retrieval test runs.
- **Stage 3 ships THREE arms, not two** (Arch C3 + Exec C3 fold). Arm B (CLIP-text + CLIP-vision + hippocampus) vs Arm C (CLIP-text + CLIP-vision + raw cosine) is the load-bearing publishable claim — it isolates the hippocampus contribution from the text encoder contribution. Arm A (paraphrase-mpnet text + hippocampus) is a secondary "native stack does even better" finding but is NOT the gate.
- **Stage 3 pass criterion is BOTH (a) margin floor AND (b) paired bootstrap CI** (Arch M1 + Exec I2 fold). The Gaussian SE assumption is wrong for bimodal per-seed distributions and breaks when `baseline_std → 0` at ceiling. Margin uses `max(empirical_baseline_std, 0.02)` floor; bootstrap uses 95% CI on per-seed delta. If they disagree, 40-seed rerun.
- **The fixture is Oxford Flowers-102** with a 10-class subset chosen by a one-shot calibration sweep to land in CLIP's `[0.50, 0.85]` zero-shot accuracy headroom band (Arch C1 + Exec I1 fold). Imagenette is rejected because CLIP saturates near 100% on it — vacuous head-to-head margin.
- **Fixture descriptor SHA-256 is asserted in `test_p4_fixture_validation`** (Exec I5 fold). Any change to the fixture without an explicit `change-fixture` commit + a corresponding hash bump in the test will fail. Combined with the single-shot rerun rule in the protocol doc, this is the enforcement teeth behind "do NOT ship band-aid fixture tweaks."
- **The single-shot rerun rule** — the 20-seed sweep runs ONCE under any given (fixture, encoder, mechanism) configuration. Re-running requires a written failure report at `docs/experiments/p4_cross_modal_failure_<date>.md` that names the reason. Documented in the reproduction protocol; linked from test docstrings.
- **Stage 3 metric is `pooled per-probe`, n=50 per seed** (Exec I3 fold), NOT `mean of per-class rates`. Pinned in the reproduction protocol BEFORE any seeds are run.
- **Per-class fixture validation in Stage 2** asserts every chosen class achieves ≥ 0.50 retrieval rate, not just the mean (Arch M3 fold). Catches single-class collapse hidden by the mean. Class-swap is allowed at Stage 2 validation time (NOT a band-aid because it happens before Stage 3 starts), but each swap is logged in the calibration writeup.
- **Subprocess child does NOT load CLIP** (Exec I6 fold). The parent encodes all (text, image) pairs once, dumps pre-computed embeddings to a temp file alongside the SessionSnapshot. The child loads the substrate state + pre-computed embeddings + runs `retrieve_cross_modal`. Eliminates per-iteration CLIP cold-start cost. The persistence round-trip is what's being tested; the encoder is a pure function.
- **`VisionEncoder` is stateless per session** and lives outside the bio-system family. NO `BioSystemSnapshot` adapter — no per-session state to persist. CLIP model weights are a CACHE, not state.
- **The mug test runs in a SUBPROCESS** via P3.5 Stage 2's `run_session_round_trip`. The session 1 → save → session 2 → reload pattern only proves what we want it to prove if the child interpreter has zero shared state with the parent.
- **Two sentence-transformers models loaded simultaneously requires VRAM audit** (Arch I4 fold). Stage 2 ships `docs/experiments/p4_stage2_vram_audit.md` with `nvidia-smi` snapshots from the RTX 5080 leader showing `clip-ViT-B-32` + `paraphrase-mpnet-base-v2` + Qwen-14B Q4_K_M @ 12k context combined footprint. If it breaches the spillover-detection plan's `max(1.5, 0.55 * weights_gb)` headroom budget, Stage 3 runs on a dedicated worktree with `MAXIM_LLM_ENABLED=0`.

## Round 1 plan review (COMPLETE 2026-04-14)

Two parallel reviewers (Executor lens + Architecture lens) ran against the post-decision-lock plan. Cross-confirmation across both lenses converged on the four most consequential findings:

1. **Imagenette saturates CLIP near 100% — vacuous head-to-head margin.** Both lenses CRITICAL. Folded by replacing the fixture with Oxford Flowers-102 + a one-shot calibration sweep (Stage 2 deliverable) to pick 10 classes in CLIP's `[0.50, 0.85]` zero-shot accuracy headroom band.
2. **The `paraphrase-mpnet` vs `CLIP-text` encoder confound is unrefuted by the plan-draft's "native encoder" defense.** Both lenses CRITICAL. Folded by shipping THREE arms in Stage 3 — Arm B (CLIP-text + hippocampus, the load-bearing claim) isolates the hippocampus contribution from the text-encoder contribution. Arm A (paraphrase-mpnet + hippocampus) becomes a secondary "native stack does even better" finding, NOT the gate.
3. **`_node_modality` lambda filter re-introduces the P3b lock-inversion class.** Exec CRITICAL + Arch IMPORTANT. Folded by mirroring P3b's snapshot pattern exactly — `retrieve_cross_modal` builds a `frozenset[str]` under `_rwlock.read()` and returns a lock-free closure.
4. **`tag_node_modality` is a footgun pattern (silent no-op if forgotten).** Exec CRITICAL + Arch IMPORTANT. Folded by deleting the manual entry point — `_close_pending_episode_locked` auto-tags every node in the closing episode with the pending modality. Forgetting becomes structurally impossible.

**Architecture-only criticals** the Executor missed (consistent with the P3.5 Stage 2 pattern):

5. **`Episode.modality` should be dropped entirely.** It was ill-defined for the cross-modal-by-construction use case (one episode contains BOTH modalities). Folded.
6. **`SubstrateModality` literal instead of `str`.** Folded — typed as the existing `Literal["text", "vision"]` from `agents/modality.py:73`.
7. **`load_state` clear-then-load semantics for `_node_modality`.** Folded with regression guard `test_node_modality_stale_entries_cleared_on_rollback` for P3.5 atomic rollback.

**Other important folds:**

- Stage 1 synthetic fixture must be **cluster-aware** (paired items share a centroid + noise; orthogonal centroids between pairs) — random gaussians would never form an episode and Stage 1 would pass on a stub. Stage 1.5 vacuous-pass guard added.
- Stage 3 metric pinned to `pooled per-probe (n=50)`, NOT `mean of per-class rates`.
- Stage 3 pass criterion is BOTH (a) `+2 × max(empirical_baseline_std, 0.02)` margin floor AND (b) paired-bootstrap 95% CI on per-seed delta — disagreement triggers a 40-seed rerun.
- Per-class validation at Stage 2 (≥ 0.50 per chosen class) catches single-class collapse hidden by the mean.
- Failure-handling teeth: fixture descriptor SHA-256 assertion + single-shot rerun rule documented in the protocol.
- Subprocess child does NOT load CLIP — parent dumps pre-encoded embeddings.
- VRAM audit deliverable in Stage 2 for the dual-encoder + LLM combined footprint.

All findings folded into "Architectural decisions" + "Stage X" + "Load-bearing invariants" sections above. Round 1 cross-confirmation pattern matches the project's `feedback_review_before_ship.md` rule that two-lens parallel review is non-optional and that Architecture-only criticals are real (P3.5 Stage 2 had the same pattern with the atomicity gap).

## Stage 2/3 open design decision — `node_filter` split (COMMITTED to Option 2, timing RESOLVED: defer)

**Status (2026-04-16):** Stage 2 v3 honest measurement COMPLETE. Option 2 lift = **0.0000 ± 0.0000** across 10 seeds. Decision: **defer Option 2 as post-Stage-3 cleanup.** The measurement satisfies all six post-mortem requirements (non-constructed topology via EC, real weight margin 0.7 vs 0.3, weight-aware metric via `spreading_activation` with `RetrievalConfig` defaults, 10-seed variance, build-time assertions, random-ranker swap passes at substrate 1.0 vs random 0.09). See [p4_option2_measurement.md](../experiments/p4_option2_measurement.md) for full results. The architectural commitment to Option 2 stands as the long-term answer; the timing question is resolved — same-class activation (0.490) dominates cross-class bridge activation (0.022) by 22:1, so Option 2 cannot improve top-5 precision under current `RetrievalConfig` parameters. Revisit after concept decomposition ships denser bridge topologies.

**Prior attempts:** v1 ("defer") and v2 ("ship with +96% lift") both withdrawn as tautological. See [p4_stage2_v2_post_mortem.md](../experiments/p4_stage2_v2_post_mortem.md).

### The underlying problem

`DependencyGraph.spreading_activation` applies one `node_filter: Callable[[str], bool]` to BOTH the source node AND every target visited during BFS. Rejected nodes are not enqueued, so the walk silently truncates at them.

This is correct for P3b's channel filter (`episode_filter(channel="sms")`) — you don't want the BFS wandering through non-SMS nodes to reach distant SMS destinations because each SMS thread is its own conversational context. But it's structurally wrong for P4's cross-modal filter: the cue is ALWAYS in the opposite bucket from the target set (a text cue looking for vision partners), so `retrieve_cross_modal` already has to exempt the cue itself via `node_id == cue_node_id or node_id in allowed` just to let the BFS seed at all. Same-modality INTERMEDIATES between cue and target (`text_cue → text_bridge → vision_target`) are STILL rejected, so multi-hop cross-modal paths through same-modality bridges are silently truncated.

### Why Stage 1 ships with the limitation

Stage 1's mug-test fixture has no such chains — its pairs are always direct text↔vision co-activations in one episode. So the limitation does NOT affect Stage 1 mechanism tests. Stage 3 with real CLIP embeddings may or may not be affected; we don't know yet whether real mug episodes produce rich text-text or vision-vision chains that carry load-bearing cross-modal signal.

`tests/substrate/test_p4_cross_modal_mechanism.py::TestStageThreeLimitation::test_multi_hop_through_same_modality_intermediate_is_blocked` is the forcing regression guard: it pins the current single-hop-only behavior, and if any future refactor enables chain traversal (intentional or accidental), the test fails with a message pointing to this section. **We cannot silently drift into the new semantics without an explicit decision.**

### Option 1 (shipped as Stage 1) — single-hop only, pinned

Keep `spreading_activation` and `retrieve_on_cue` unchanged. `retrieve_cross_modal`'s closure exempts only the cue. Multi-hop through same-modality intermediates is blocked. Regression-test-pinned.

- **When this wins:** if Stage 2's mug test shows healthy F1 (≥0.80) on single-hop only — real mug episodes produce direct text↔vision co-activations in CLIP space, chain traversal doesn't carry load-bearing signal, and the limitation is empirically harmless.

### Option 2 (committed target) — rename `node_filter` → `traversal_filter` + add `result_filter`

Make the two concerns explicit at the API level. `traversal_filter` controls which nodes the BFS walks through (what `node_filter` does today); `result_filter` is a post-filter applied to the ranked output. Rename in `DependencyGraph.spreading_activation` and `Hippocampus.retrieve_on_cue`. Provide a compat shim so P3b's existing single-filter calls (`episode_filter(channel="sms")` → `retrieve_on_cue(node_filter=...)`) map to both traversal + result — which is literally what P3b wants.

After Option 2 ships:

- **P3b** cross-channel filter: `traversal_filter=sms_nodes, result_filter=sms_nodes` (or via the compat shim — unchanged semantics)
- **P4** cross-modal: `traversal_filter=None, result_filter=modality_membership`. The cue-exemption hack in `retrieve_cross_modal`'s closure is DELETED because the source is no longer filtered at all.
- **Future filters compose naturally:** `traversal_filter=sms_nodes, result_filter=high_stress_nodes` for "SMS neighbors that are also high stress."

**Pros:** architecturally clean; forces every caller to think about traversal-vs-result semantics; matches the project's "push silent-no-op invariants into types" rule; P4 cue-exemption hack becomes obsolete.

**Cons:** touches `spreading_activation` (a core primitive used by P3a/P3b/P4); P3a Stage 2's 10-seed sweep must be re-run to confirm F1 numbers hold; P3b's ~25 tests need re-audit against the compat shim; ~half to full day of work including its own Round 2 pre-merge review.

### Option 3 (explicitly rejected) — add `result_filter` without renaming

Adding `result_filter` as a new optional parameter without renaming `node_filter`. Smaller blast radius but leaves the API ambiguous — future callers don't know whether to use `node_filter` (traversal) or `result_filter` (post), and the parameter names don't force the right thinking. **User rejected this as architectural debt that violates the "push invariants into types" rule.** Option 2 is the committed answer.

### Trigger for revisit

Stage 2's deliverable includes the subprocess mug test on real CLIP + Oxford Flowers-102 images. Revisit rules:

1. **If Stage 2 mug test F1 is healthy (≥0.80) on single-hop only:** Option 2 still happens before Stage 3 because the user has already committed to it, but without urgency — treat as pre-Stage-3 architectural cleanup at a convenient time.
2. **If Stage 2 mug test F1 is disappointing AND binding-graph topology inspection shows text-text or vision-vision chains dominating the retrieval path:** Option 2 becomes a BLOCKING prereq for Stage 3. Ship it BEFORE Stage 3's metric freezes so the head-to-head definition is stable.
3. **If Stage 2 mug test F1 is disappointing but NOT traceable to chain truncation:** investigate other failure modes first (CLIP encoder quality, fixture calibration, Hebbian weight parameters) before reaching for Option 2.

**Do NOT land Option 2 during Stage 3.** The decision window is Stage 2 → Stage 3, not during Stage 3 — metric freezes want stable mechanics.

### Implementation notes for when Option 2 lands

- Activation scores for chain-reached nodes are naturally ~`decay × weight` per hop lower than direct co-occurrences (~0.044 vs ~0.21 at default config). This ranking property is desirable (direct Hebbian evidence should outweigh transitive evidence) and must be preserved through the refactor.
- The `limit` + `result_filter` interaction has a footgun: `result_filter` MUST run BEFORE `limit`, otherwise the BFS may fill the limit with nodes that all fail the post-filter and return 0 hits. Unit-test this explicitly in the Option 2 PR.
- Delete the cue-exemption hack (`node_id == cue_node_id or`) from `retrieve_cross_modal`'s closure. The comment explains why the hack existed; it becomes obsolete under the split-filter model.
- `spreading_activation` signature change is the primary breaking change. Every call site needs audit: P3a (`retrieve_on_cue`), P3b (via `episode_filter` → `retrieve_on_cue`), P4 (`retrieve_cross_modal` → `retrieve_on_cue`), and any future caller.
- After the change, flip `TestStageThreeLimitation`'s assertion to verify vision_target IS retrieved and rename the test class (the "limitation" framing is obsolete).

## Deferred (filed before Stage 1)

- **Multi-object vision** — `VisionEncoder` is single-object only in P4. P4-MV (post-1.0) extends to multi-object detection + per-object cross-modal binding.
- **Audio modality** — same architectural pattern (audio encoder → SensoryTag → episode binding), deferred to P4-A (post-1.0).
- **Cross-modal binding decay** — extinction of cross-modal bindings under non-reinforcement is part of P6, not P4.
- **Vision-side replay during sleep** — part of P8.

## Not in this plan

- Anything requiring P5/P6/P8 code to exist
- Vision-encoder-as-bio-system (it's a percept producer; if state ever needs to persist it gets its own plan)
- ATL / EC modality awareness — those layers stay modality-agnostic
- A new `CROSS_MODAL` edge type on `CrossLayerGraph` unless Round 1 review picks Option B
- Dropping or modifying the `paraphrase-mpnet` text encoder for the substrate path

## Stage 2 v2 fold — WITHDRAWN (2026-04-15)

The Stage 2 v2 fold attempted to rebuild Phase 2D with distractor noise + text-text bridge topology to produce a non-tautological Option 2 measurement. The resulting +96.0% lift was itself tautological (construction-identity in a different mechanical shape) and has been withdrawn along with all v2 artifacts (sweep script, results, milestone report, fixture v2 shape).

Infrastructure improvements from the fold are preserved on main: `build_and_bind` refactor (`82da6db`), 102-class pin + `class_idx` enforcement (`8d0b92f`), YAML fallback drop (`8d0b92f`), tactical fixes bundle (`f00fc0f`).

**Authoritative post-mortem:** [../experiments/p4_stage2_v2_post_mortem.md](../experiments/p4_stage2_v2_post_mortem.md) — documents both v1 and v2 failure modes, the construction-identity pattern, and the six methodology requirements for the next attempt.
