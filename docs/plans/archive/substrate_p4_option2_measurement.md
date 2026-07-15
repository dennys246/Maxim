# Substrate P4 — Option 2 honest measurement (Stage 2 v3)

**Status:** COMPLETE (2026-04-16). **Decision: DEFER.** Option 2 lift = 0.0000 ± 0.0000 across 10 seeds. Same-class activation dominates cross-class by 22:1. Ship Stage 3 on single-hop.
**Scope:** ~350 LOC measurement script. Zero changes to production code.
**Results:** [../experiments/p4_option2_measurement.md](../../experiments/p4_option2_measurement.md)
**Parent:** [substrate_p4_cross_modal_binding.md](substrate_p4_cross_modal_binding.md), "Stage 2/3 open design decision" section.

## Context

Stage 2 v1 and v2 both attempted to measure Option 2 lift and both were caught as tautological. See [../experiments/p4_stage2_v2_post_mortem.md](../../experiments/p4_stage2_v2_post_mortem.md) for the full post-mortem. This plan designs a measurement that satisfies the six post-mortem requirements.

## The six requirements (from the post-mortem)

1. **Non-constructed topology.** The bridge structure must NOT be written by the fixture builder.
2. **Real signal-vs-noise weight margin.** Signal edges must exceed noise by a discriminable ratio.
3. **Weight-aware metric.** Must use actual `spreading_activation` with decay/threshold, not raw BFS.
4. **Multi-seed variance.** Operating-point rule applies to mean-across-seeds.
5. **Build-time assertion.** Signal weight > noise weight by the chosen ratio, checked before any sweep.
6. **Falsifiability check.** Random-ranker swap test — replacing the ranker with `random.shuffle` must degrade the metric.

## Key insight: let EC decide

The v2 failure was fixture-builder-constructed bridges. The honest alternative: feed the substrate realistic inputs and let EC's `pattern_complete_or_separate` decide whether shared-concept nodes emerge. If EC creates a shared node, the Hebbian binding produces multi-hop paths organically. If EC doesn't, there are no multi-hop paths, and Option 2 is honestly deferrable.

### Empirical calibration (paraphrase-mpnet-base-v2, TEXT_EC_THRESHOLD=0.60)

Cosine similarity of candidate bridge concepts against all 10 fixture class names:

| concept | mean | min | max | merges at 0.60 |
|---|---|---|---|---|
| "flower" | 0.503 | 0.276 | 0.753 | balloon flower, pincushion flower, orange dahlia |
| "plant" | 0.384 | 0.203 | 0.498 | none |
| "garden" | 0.390 | 0.174 | 0.557 | none |
| "blossom" | 0.392 | 0.251 | 0.564 | none |
| "botanical" | 0.436 | 0.268 | 0.525 | none |

**"flower" is broken** — it EC-merges with 3 class names (this is exactly the v2 Exec #1 bug).

**"plant", "garden", "blossom", "botanical" are clean** — all stay below 0.60 for every class name, so EC will separate them into independent nodes. These are honest bridge candidates: they represent real superordinate concepts that an agent might encounter while perceiving different flower images.

## Methodology: "Organic Shared-Concept Exposure"

### Phase 1 — Standard class binding (identical to v1)

For each of 10 classes × 5 images:
- Create an episode with `(text_classname, vision_image_i)` co-activation
- **Reinforce 5× per pair** — 5 separate episodes, each adding 0.1 to the edge weight
- Result: each class-pair Hebbian edge at weight **0.7** (init 0.3 + 4 × delta 0.1)

### Phase 2 — Shared-concept exposure (the honest bridge layer)

For each of 10 classes:
- Create an episode with `(text_classname, text_"plant")` co-activation
- **Single exposure** — 1 episode → Hebbian edge at weight **0.3**
- EC decides: does "plant" pattern-complete against any existing node, or separate?
- At threshold 0.60: "plant" separates (max cosine to any class is 0.498). A new `text_plant` node is created and Hebbian-bound to `text_classname`.

After Phase 2, the binding graph has organic 2-hop paths:
```
text_lotus (0.3)→ text_plant (0.3)→ text_azalea (0.7)→ vision_azalea_0..4
```

The path `text_lotus → text_plant → vision_azalea_0` is a real multi-hop cross-modal path that the substrate built through EC's own decisions — not something the fixture builder constructed.

### Phase 3 — Measurement

For each of 10 classes, for each text cue `text_classname`:

**(a) Single-hop filter (Option 1, current behavior):**
```python
results_single = hippocampus.retrieve_cross_modal(
    text_classname, target_modality="vision", limit=20
)
```
This blocks at `text_plant` (same modality as cue, not the cue itself → rejected by node_filter).

**(b) Simulated split filter (Option 2):**
```python
# Build a traversal_filter that allows ALL nodes (text and vision)
# but a result_filter that returns only vision nodes.
#
# Arch-review fold: using node_filter=None is strictly MORE permissive
# than the real Option 2 design (which would still block cross-class
# vision nodes as traversal intermediaries in some topologies). The
# correct simulation is: traversal_filter allows text+vision nodes
# that are NOT in the "wrong-class vision" bucket. For this fixture
# that's equivalent to "allow all text nodes + all vision nodes"
# because vision nodes don't have outgoing ASSOCIATES edges to other
# vision nodes (they're leaf nodes in the binding graph). So
# node_filter=None is mechanically equivalent here, but the plan
# must verify this assumption holds for the actual fixture topology.
results_multi = hippocampus.retrieve_on_cue(
    text_classname, limit=500, multi_hop=True, node_filter=None
)
# Post-filter to vision-only (simulates result_filter), THEN cap
# Exec-review fold: using limit=20 before post-filter silently
# truncates vision results when high-activation text intermediates
# fill the top-20. Use a large limit (500 >> total nodes) so all
# reachable nodes are returned, then post-filter, then cap at 5.
results_multi = [(nid, w) for nid, w in results_multi 
                 if hippocampus._node_modality.get(nid) == "vision"][:5]
```

**Implementation note:** before running the measurement, assert that no vision node has outgoing `ASSOCIATES` edges to other vision nodes in the binding graph. If they do (e.g., from noise edges), `node_filter=None` is no longer equivalent to the intended Option 2 `traversal_filter` and the simulation needs a proper text-or-cue-exempted filter.

**Metric per cue:** top-5 recall = fraction of this class's 5 vision nodes in the top-5 results.

**Option 2 lift per cue:** `recall_multi - recall_single`.

**Aggregate:** mean lift across 10 classes × N seeds.

### Phase 4 — Validation gates

1. **Build-time assertions:** (a) signal weight (0.7) ≥ 2× bridge weight (0.3). (b) **EC separation margin (Exec-review fold):** `max(cosine(bridge_concept, class_i)) < threshold - jitter_margin` for each bridge concept. Prevents centroid-drift + jitter from narrowing the 0.498-vs-0.60 gap below the merge point. Both checked before any measurement runs.

2. **Falsifiability (random-ranker swap test):** replace `spreading_activation` results with `random.shuffle(all_vision_nodes)[:limit]` and recompute top-5 recall. The random baseline should be **significantly worse** than the substrate's ranked results. If it's not, the metric can't distinguish working from broken. Specifically:
   - Random baseline expected recall: 5/50 = 0.10 (5 correct out of 50 total vision nodes, choosing 5)
   - Substrate recall should be ≥ 0.70 (the existing retrieval gate)
   - If substrate recall ≈ random recall → measurement is broken, abort

3. **Multi-seed:** run across 10 seeds (matching P3a's budget). Seed controls: EC threshold jitter (±0.02), episode ordering permutation, Hebbian init jitter (±0.02). Report mean ± std of Option 2 lift. **Exec-review fold:** none of these jitter sources exist in current infra — `BuildConfig`, `HebbianConfig`, and `pattern_complete_or_separate` all take fixed values. All three jitter sources are implemented as per-seed `BuildConfig` parameter variation in the measurement script (not production code changes). The script constructs a fresh `BuildConfig` per seed with `text_ec_threshold=0.60 + rng.uniform(-0.02, 0.02)`, `hebbian_init=0.3 + rng.uniform(-0.02, 0.02)`, and shuffled episode insertion order.

4. **Cross-class contamination check:** under Option 2 (no traversal filter), does `text_lotus` retrieve `vision_azalea` via the `text_plant` bridge? This is the interesting case:
   - If cross-class retrieval is negligible (activation decays below threshold at 3 hops): Option 2 lift is ~0 and Option 2 is honestly deferrable — the bridge paths exist but don't carry enough activation to matter.
   - If cross-class retrieval is non-trivial but **correctly ranked** (same-class > cross-class): Option 2 adds recall breadth without degrading precision. Ship.
   - If cross-class retrieval **pollutes** the top-5 (cross-class outranks same-class): Option 2 needs `result_filter` to be modality-only AND a re-ranking step. More complex than the original Option 2 spec.

### Phase 5 — Decision

The decision matrix:

| Outcome | Decision |
|---|---|
| Lift ≈ 0 (multi-hop paths exist but activation decays below noise) | **Defer Option 2** as cleanup. Ship Stage 3 on single-hop. Revisit after concept decomposition (which produces denser bridge topology). |
| Lift > 0, same-class recall preserved (top-5 uncontaminated) | **Ship Option 2** before Stage 3. The bridge paths carry useful signal without harming precision. |
| Lift > 0, but cross-class pollution in top-5 | **Ship Option 2 with ranked result_filter** — more complex, needs its own design pass. Still ships before Stage 3 but with additional work. |
| EC merges bridge concept with class name (at threshold or under jitter) | **Report honestly.** Note which seeds merge and which separate. If majority merge, the bridge concept is not suitable — try another concept from the candidate list before concluding "no bridge topology." |
| Random-ranker swap test fails | **Abort measurement.** Metric is broken. Redesign. |

## Why this satisfies the six requirements

1. **Non-constructed topology:** "plant" enters through EC's `pattern_complete_or_separate` — the substrate decides whether it becomes a shared node. The fixture builder feeds inputs; the substrate builds topology. **Arch-review caveat (folded):** the bridge concept was pre-screened by computing cosine similarities, which is a softer form of construction. Mitigation: run 3 concepts ("plant", "garden", "blossom") and report EC's decision for each as a finding, not an assumption. If all 3 merge or all 3 separate, the outcome is robust to concept choice. If they disagree, report the spread.

2. **Real signal-vs-noise weight margin:** class pairs at 0.7, bridge edges at 0.3, ratio 2.33:1. The ranker has genuine discrimination pressure.

3. **Weight-aware metric:** uses actual `spreading_activation` with production `RetrievalConfig` defaults (`decay=0.7, threshold=0.001, max_depth=5`) — same parameters as `retrieve_on_cue`. NOT raw BFS. **Note:** `HippocampusConfig` has a separate set (`decay=0.5, threshold=0.05, max_depth=3`) for the memory-record association graph — that is NOT what `retrieve_on_cue` uses. The binding graph path goes through `RetrievalConfig`.

4. **Multi-seed variance:** 10 seeds with controlled jitter sources.

5. **Build-time assertion:** `assert signal_weight >= 2 * bridge_weight` runs before any measurement.

6. **Falsifiability:** random-ranker swap produces recall ≈ 0.10 vs substrate ≈ 0.70+. Clear separation.

## What this plan does NOT do

- Does not change any production code. This is measurement-only.
- Does not implement Option 2. If the decision is "ship," a separate PR implements the `traversal_filter` + `result_filter` split per the existing architectural spec in `substrate_p4_cross_modal_binding.md`.
- Does not use "flower" as a bridge concept. That word EC-merges with class names — same bug class as v2 Exec #1.
- Does not claim to settle the Option 2 question permanently. If concept decomposition later ships denser bridge topologies, the measurement should be re-run with the enriched graph.

## Activation math sanity check

**Parameters:** `RetrievalConfig` defaults: `decay=0.7, threshold=0.001, max_depth=5`. (NOT the `HippocampusConfig` memory-record params of `decay=0.5, threshold=0.05, max_depth=3` — those govern a different `spreading_activation` call site. Round 1 Architecture-lens review caught this config confusion.)

**Same-class path (single-hop):** `text_lotus → vision_lotus_0`

```
vision_lotus_0: activation = 1.0 × 0.7 × 0.7 = 0.490 (decay × class_weight)
```

**Cross-class path via bridge (3-hop):** `text_lotus → text_plant → text_azalea → vision_azalea_0`

```
text_plant:      activation = 1.0 × 0.7 × 0.3 = 0.210  (decay × bridge_weight)
text_azalea:     activation = 0.210 × 0.7 × 0.3 = 0.0441 (decay × bridge_weight)
vision_azalea_0: activation = 0.0441 × 0.7 × 0.7 = 0.0216 (decay × class_weight)
```

All hops survive at `threshold=0.001`. Cross-class vision nodes ARE reachable via the bridge path.

**Ranking:** same-class `vision_lotus_0` at 0.490 vs cross-class `vision_azalea_0` at 0.0216 — ratio ~22:1. The ranker should clearly prefer same-class. But there are 9 cross-class × 5 vision-per-class = 45 cross-class vision nodes all reachable at similar activations, while there are only 5 same-class vision nodes. In a top-5 retrieval, the same-class nodes dominate (0.490 >> 0.022), but in a top-20 retrieval the cross-class nodes would fill the tail.

**Key prediction:** Option 2 lift is likely **non-zero** because cross-class paths survive the threshold gate. The interesting measurement is whether those cross-class retrievals are correctly ranked below same-class (useful breadth) or whether they pollute the top-5 (harmful). This shifts the decision matrix toward the "ship" or "pollution" rows rather than the "defer" row.

**Sensitivity:** the 3-hop bridge path's activation (0.0216) is 2 orders of magnitude above `threshold=0.001`. Even if decay were tuned down to 0.5 (the HippocampusConfig value), activation would be 0.5³ × 0.3² × 0.7 = 0.0079, still well above 0.001. The paths are robust to parameter variation.

## Round 1 review findings (folded)

**Architecture-lens (2 CRITICAL, 2 IMPORTANT, 2 MINOR):**

- **Arch C1 (CRITICAL): wrong spreading_activation parameters.** Plan originally used `decay=0.5, threshold=0.1, max_depth=3` (the HippocampusConfig memory-record params). Production `retrieve_on_cue` uses `RetrievalConfig` defaults: `decay=0.7, threshold=0.001, max_depth=5`. With correct params, cross-class vision nodes ARE reachable via bridge paths (activation 0.0216 >> threshold 0.001). Central prediction shifted from "lift ≈ 0" to "lift likely non-zero." **FOLDED:** activation math section rewritten, parameter source documented.
- **Arch C2 (CRITICAL): EC threshold 0.60 is non-default.** Production default is 0.40; 0.60 is a per-call override from the mug test. At 0.40, all candidate bridge concepts merge with class names. **FOLDED:** plan explicitly states it uses 0.60 (same as the mug test fixture); added "EC merges bridge" row to decision matrix for the case where jitter pushes below merge threshold.
- **Arch I1 (IMPORTANT): bridge concept selection is semi-constructed.** Pre-screening concepts by cosine similarity is a softer form of construction. **FOLDED:** mitigation added — run 3 concepts, report EC decision as finding not assumption.
- **Arch I2 (IMPORTANT): Phase 3(b) simulates Option 2 incorrectly.** `node_filter=None` is strictly more permissive than real Option 2. **FOLDED:** implementation note added requiring assertion that vision nodes are leaf nodes (no outgoing ASSOCIATES edges to other vision nodes) before accepting the equivalence.
- **Arch M1 (MINOR): "Two implications" lists three items.** FOLDED in activation math rewrite.
- **Arch M2 (MINOR): decision matrix missing "EC merges bridge" row.** FOLDED.

**Executor-lens (2 CRITICAL, 2 IMPORTANT, 1 MINOR):**

- **Exec C1 (CRITICAL): chain hop-count imprecise.** Original plan said "dies at hop 3" but with the wrong params it died at hop 2. **Moot after Arch C1 fold** — with correct `RetrievalConfig` params, all hops survive. Activation math section rewritten with correct values.
- **Exec C2 (CRITICAL): centroid drift + jitter could narrow EC separation margin.** Class-name centroids are updated via running mean after each `pattern_complete`. After 5 identical presentations the centroid is unchanged, but floating-point variations from re-encoding could shift it. Combined with threshold jitter (±0.02), the 0.498 max cosine of "plant" could approach 0.58. **FOLDED:** build-time assertion added: `max_cosine < threshold - jitter_margin` per bridge concept, checked before any measurement runs.
- **Exec I1 (IMPORTANT): `limit=20` truncation.** With `node_filter=None`, text intermediates compete for the top-20 slots, potentially pushing vision results out before post-filtering. **FOLDED:** limit raised to 500 (>> total nodes), post-filter applied, then capped at 5.
- **Exec I2 (IMPORTANT): jitter requires new measurement-script code.** No jitter infrastructure exists in `BuildConfig`, `HebbianConfig`, or EC. **FOLDED:** explicit callout that all jitter is per-seed `BuildConfig` variation in the script, not production code changes.
- **Exec M1 (MINOR): weight math correct.** No fence-post error in 5-episode × init+delta calculation.

## Deliverables

1. `scripts/p4_option2_measurement.py` — measurement script implementing Phases 1–5
2. `docs/experiments/p4_option2_measurement.md` — results report with decision
3. Update to `docs/plans/archive/substrate_p4_cross_modal_binding.md` "Stage 2/3 open design decision" section with the empirical answer
4. If decision is "ship": separate PR with Option 2 implementation per the existing architectural spec
5. If decision is "defer": update `TestStageThreeLimitation` docstring with the honest rationale

## Risks

1. **EC threshold sensitivity.** At threshold 0.58 instead of 0.60, "blossom" (max 0.564) or "garden" (max 0.557) might merge with a class name. The multi-seed jitter (±0.02) covers this range. If the outcome flips across the jitter band, the measurement is sensitive to threshold choice — report this honestly rather than picking the favorable seed.

2. **The activation math suggests lift ≈ 0.** If the empirical measurement confirms this, the temptation is to lower the threshold or increase max_depth to "make it work." Don't. The measurement's purpose is to determine whether Option 2 matters under current production parameters. If it doesn't, that's the answer.

3. **Bridge concept selection.** "plant" was chosen because it's the cleanest empirically. A different bridge concept might produce different results. The measurement should try 2–3 concepts ("plant", "garden", "blossom") to check robustness. If the outcome varies across concepts, report the spread.
