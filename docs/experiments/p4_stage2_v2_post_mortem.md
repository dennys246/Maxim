# Substrate P4 Stage 2 v2 — post-mortem + Option 2 decision re-re-open

**Date:** 2026-04-15
**Author:** Claude session (Maxim Mac peer)
**Status:** Post-mortem — **RESOLVED (2026-04-16).** Stage 2 v3 ran an honest measurement via organic shared-concept exposure: Option 2 lift = 0 across 10 seeds, decision is **DEFER** as post-Stage-3 cleanup. See [p4_option2_measurement.md](p4_option2_measurement.md). The v1+v2 failure analysis below is preserved as historical record.

**Why this doc exists:** a successor session reading `docs/experiments/p4_mug_test_sweep_v2.md` or the fold branch's commit history will see a "+96.0% Option 2 lift → SHIP" conclusion. That conclusion is **withdrawn**. This doc is the authoritative record of why, what we learned, and what the next attempt needs to look like.

## TL;DR

We tried twice to empirically measure whether Option 2 (the `node_filter → traversal_filter + result_filter` split in `retrieve_on_cue`) is needed for P4 cross-modal retrieval. Both attempts failed — not because Option 2 is wrong, but because the measurements were **construction-identity** rather than empirical:

- **Stage 2 v1** (PR #129) shipped a mug test with no distractors and no cross-class reachability paths. The retrieval metric was mechanically forced to 1.000 because there were no alternatives. Concluded "defer Option 2." **Caught by Round 2 Architecture-lens review as tautological.**
- **Stage 2 v2** (`fix/substrate-p4-stage2-fold` branch) rebuilt the fixture with cross-class noise + a shared-superclass text bridge, swept 12 parameter combinations, reported **+96.0% Option 2 lift** at the operating point, concluded "SHIP Option 2." **Also caught by Round 2 review — both lenses independently, cross-confirmed — as a new construction-identity in a different shape.**

The v2 number is a graph-theoretic count of "how many bridges did we write that single-hop would block" — it would reproduce at the same value under a broken ranker, zero Hebbian weights, or a randomized encoder because the measurement doesn't depend on substrate behavior. **The v1 tautology was "no alternatives → recall = 1.0." The v2 tautology is "we wrote N bridges → BFS finds N bridges."**

On top of that, Round 2 Executor-lens found **two concrete bugs** in the v2 measurement that compound the problem:

1. The bridge token `"text_flower"` EC-collapses into an existing Flowers102 class name (likely `"sunflower"` or `"passion flower"`) at the default `text_ec_threshold=0.60`. There was never a distinct bridge node — the "bridge" was just extra cross-class edges written onto one of the real class text nodes.
2. The signal/noise weight margin at the chosen operating point is **exactly zero** (both at Hebbian init=0.3). The 0.98 recall at `noise_reps=1` is sort-stability luck, not ranker capability.

**Option 2 remains committed as the long-term architectural answer** (user signoff 2026-04-15 in the original [node_filter split design doc](../plans/archive/substrate_p4_cross_modal_binding.md#stage-23-open-design-decision)). What's reopened is **the question of whether Option 2 ships BEFORE or AFTER Stage 3** — we no longer have empirical data either way, and the `TestStageThreeLimitation` regression guard from Stage 1 remains the forcing function until a non-tautological measurement lands.

## The v1 tautology (Round 2 round #1 catch)

Stage 2 v1 built a fixture where each class had:

- 1 text node (class name like `"lotus"`)
- 5 distinct vision nodes (5 real Flowers102 images, kept distinct via `VISION_EC_THRESHOLD=1.01`)
- 5 episodes pairing `(text_X, vision_X_i)` for each of the 5 images

**No cross-class reachability paths.** The binding graph was 10 disjoint stars. From `text_lotus`, spreading_activation could only reach `vision_lotus_0..4`. Top-5 retrieval was forced to `{vision_lotus_0..4}` because the set of reachable vision nodes was exactly the expected set.

The metric **could not fail** on a working substrate. It also could not fail on a broken substrate, a zero-Hebbian substrate, or an encoder that collapsed everything — because the substrate's only job in this test was "return the nodes reachable via direct Hebbian edges from the cue," which is structurally identical to "return the 5 same-class vision nodes."

Round 2 Architecture-lens review caught this as Arch #4 CRITICAL: "The reported 100% / 1.000 ± 0.000 / all-1-hop result is not measuring cross-modal binding quality — it's measuring 'do direct Hebbian edges exist after insertion.'" The Option 2 defer decision rested on unfalsifiable evidence.

## The v2 tautology (Round 2 round #2 catch — cross-confirmed)

Stage 2 v2 addressed v1's critique by building a fixture with two additional layers:

- **Noise layer:** each class gains 1 cross-class contaminant pair reinforced `noise_reps` times, creating a rotating noise ring `text_0 ↔ vision_1_0, text_1 ↔ vision_2_0, ..., text_9 ↔ vision_0_0`.
- **Bridge layer:** a shared `text_flower` superclass text node bound to each class's text (creating text-text edges) AND to each class's `vision_X_0` (creating text-vision edges on the bridge side).

The v2 sweep then measured:

- Same-class top-5 recall (signal-vs-noise ranking check)
- Cross-class single-hop reachability via `retrieve_cross_modal` (Stage 1's current filter)
- Cross-class multi-hop reachability via raw BFS, `max_depth=5`, no modality filter ("Option 2 simulation")
- Option 2 lift = `(multi_hop - single_hop) / total_pairs`

Across 12 parameter combinations the sweep reported **+96.0% Option 2 lift** at `(noise_reps=1, bridges=shared_superclass)` with 0.980 mean recall. Milestone doc concluded "Option 2 SHIP in a follow-up PR."

**Both reviewers independently caught the construction-identity pattern.**

### Arch #1 CRITICAL — the measurement is a definitional identity

The bridge pass writes `(text_X, text_flower)` episodes that create text-text Hebbian edges. Under Stage 1's `retrieve_cross_modal` single-hop filter, `text_flower` is rejected as an intermediate at traversal time (it's in the text modality bucket, not the allowed vision set), so BFS truncates at `text_flower`. Under raw BFS with no filter, `text_X → text_flower → vision_Y_0` is a 2-hop path that is **guaranteed reachable for every (X, Y) pair because we explicitly wrote those edges**.

The `n_multi_hop_reachable=450` count at the `bridges=shared` rows is therefore not a substrate property — it is the arithmetic `n_classes × (n_classes-1) × samples_per_class = 10 × 9 × 5 = 450` of the star topology the fixture constructs and then counts.

**A falsifying test:** the reviewer predicted every `n_multi_hop_reachable` value in the sweep table without running any encoder or retrieval, purely from the fixture's graph geometry. `noise=0, bridges=none → 0`. `noise=0, bridges=shared → 450`. `noise≥1, bridges=none → 210 via the noise ring`. **The numbers validate the fixture writer, not the substrate.** Running the sweep with a randomized ranker, zero Hebbian weights, or uniform embeddings would produce identical `n_multi_hop_reachable` values — because the measurement doesn't depend on any of those.

**Why this is the same bug class as v1, not a new one:** v1's tautology was "no alternatives exist → recall = 1.0." v2's tautology is "we wrote the bridges that BFS finds, then measured BFS finding them." Different mechanical shape, same logical structure: **the measurement is deducible from the fixture spec alone without running the substrate.**

### Exec #1 CRITICAL — the bridge token EC-collapses into a real class

Independent of Arch #1, the v2 fixture has a **concrete bug** in how the bridge node is created. `p4_build_and_bind.py::build_and_bind` at the bridge pass calls:

```python
bridge_text_emb = config.text_encoder.encode("text_flower")
bridge_result = ec.pattern_complete_or_separate(
    bridge_text_emb, modality="text", threshold=0.60  # config.text_ec_threshold
)
bridge_node_id = bridge_result.node_id
```

Flowers102 contains classes whose names have high paraphrase-mpnet cosine similarity to the literal string `"text_flower"`: `"sunflower"`, `"passion flower"`, `"water lily"`, `"english marigold"`, etc. At `threshold=0.60` (the v2 calibration value), EC **pattern-completes** the `text_flower` embedding into one of these existing class text nodes instead of creating a distinct bridge node. The bridge text node `bridge_node_id` IS one of the 10 class text nodes — `text_sunflower` or similar.

When the bridge pass then writes the 10 `(bridge_node_id, vision_X_0)` episodes, those become 10 direct `text_sunflower ↔ vision_X_0` cross-class edges on whichever class absorbed the bridge. **From that class's perspective, 9 of 10 other classes become direct single-hop neighbors** — no bridge traversal, no filter rejection, no multi-hop anything.

This is why the `n_single_hop_reachable=9` value at `(noise=0, bridges=shared)` confused me: my mental model said "bridges are blocked under single-hop, so cross-class single-hop count should be 0." The actual answer is "there IS no bridge — one class got 9 direct cross-class edges via EC collapse." The 9 in the sweep output is the 9 non-collapsed other classes' first vision nodes seen as direct neighbors from the collapsed class's text.

At `(noise=1, bridges=shared)`: `18` single-hop hits = the 9 collapsed-class direct edges + 9 rotating noise edges (one of which duplicates onto the collapsed class, so effectively 9 + 9 = 18).

**Build-time assertion that would have caught this pre-merge:** `assert bridge_node_id not in {c.text_node_id for c in class_results}` — same "push silent no-op into a ValueError" rule CLAUDE.md formalizes. Not added in the v2 fold because the collision wasn't anticipated.

### Exec #2 CRITICAL — zero weight margin at the operating point

The operating point at `noise_reps=1` sits on a ranker tie-break, not a ranker capability. Mechanical explanation:

- Stage 2 v2 fixture binds each class-correct `(text_X, vision_X_i)` pair with **exactly one episode**. Each of the 5 pairs per class is a distinct `(text, vision)` combination, so the Hebbian weight on each edge is `init=0.3`, **not reinforced** because each pair appears in only one episode.
- Noise at `noise_reps=1` writes `(text_X, vision_Y_0)` as one episode. Noise edge weight = `init=0.3`.
- **Signal and noise weights are equal at `noise_reps=1`.** Top-5 ranking is determined by dict iteration order + sort stability under tied activations, NOT by the substrate's ability to distinguish signal from noise.
- At `noise_reps=2`, noise reinforces to `init + delta = 0.4`, which strictly exceeds signal at 0.3. The 0.80 recall cliff is literally "noise displaces exactly one class-correct vision node per class, so every class drops from 5/5 to 4/5."

I documented the 0.80 cliff in the milestone doc as "unexplained, flagged as Stage 3 follow-up." **That was a no-band-aid violation.** The cliff is fully explained by Hebbian weight arithmetic, and the explanation shows the operating point has zero margin to the tie-break boundary.

**Why the 0.980 mean at noise_reps=1 is misleading:** it is `(9 × 1.0 + 1 × 0.8) / 10`. One class consistently drops to 0.80. Exec-lens traced which class: the same class that absorbed the `text_flower` bridge collapse in Exec #1. The bridge collapse gave that class many more same-weight neighbors to tie-break against, and the sort-stability order placed one of its class-correct vision nodes outside top-5.

So the "0.980 recall" at the operating point is:

- 9 classes scoring perfectly because they have 5 class-correct edges (weight 0.3) and 0 cross-class direct edges
- 1 class (the collapsed one) scoring 0.80 because it has 5 class-correct + 9 cross-class direct edges (all at weight 0.3), and one class-correct edge loses its top-5 slot to tie-break displacement

**Neither the 9 perfect scores nor the 1 imperfect score is a measurement of ranker capability.** The perfect classes have no competition. The imperfect class has a deterministic tie-break pattern driven by the EC collapse artifact.

## Why the fold cannot ship "Option 2 SHIP"

Any one of the three findings above would be enough on its own to invalidate the v2 SHIP conclusion. All three compound:

1. **Methodology is construction-identity** (Arch #1). Even if we fix the bridge token and the weight margin, the metric still counts "what we wrote" rather than "what the substrate did." A fixed-up sweep would still not be empirical data.
2. **Bridge token collides** (Exec #1). The specific +96% number is distorted by an EC-collapse artifact. Fixing the token gives a different number, but that different number would still be a construction identity per #1.
3. **Zero weight margin** (Exec #2). The same-class recall at the operating point doesn't reflect substrate capability. Fixing the margin gives a different recall curve, but that curve would still be a property of the fixture's noise/signal weight ratios, not the substrate's ability to learn.

**Both reviewers independently recommended HOLD the SHIP decision.** The high-trust cross-confirmation signal is the strongest possible evidence that we're about to make a plan-ending mistake if we ship.

## What gets preserved from the v2 fold

The fold is not wasted work. Several pieces are legitimate infrastructure improvements that should still land, just not under the "SHIP Option 2" framing:

- **Commit `6de09c6` (plan amendment — torchvision decision):** addresses the Round 2 Arch #1 finding about the silent switch from `datasets.load_dataset` to `torchvision.datasets.Flowers102`. Real plan-vs-implementation drift that needed documentation.
- **Commit `82da6db` (refactor `_build_and_bind`):** extracts the orchestrator from `scripts/` into `tests/substrate/p4_build_and_bind.py`, parameterizes encoder + threshold + noise + bridge config. Addresses Arch #3 (encoder-confound discipline) and Exec #4 (scripts/ layering). **The parameterization is required for Stage 3 Arms A/B/C regardless of how we answer Option 2.**
- **Commit `8d0b92f` (Flowers102 class-list pin + class_idx drift guard + YAML fallback parser drop):** addresses Arch #2 (undocumented torchvision API), Arch #10 (class_idx dead weight), Exec #1+#2 (YAML fallback silent # truncation + PyYAML split-brain). All genuine fixture-hygiene improvements.
- **Commit `f00fc0f` (tactical fixes bundle):** probe sort, headroom-band hard assert, VRAM OOM capture, canonical `torch.backends.mps.is_available()`, cosmetic `{:+.0f}` delta formatting, threshold tripwire test. All unambiguously correct.

Commits that need to be withdrawn or re-scoped:

- **Commit `5d25556` (Phase 2D v2 sweep + results + SHIP decision):** the sweep runner is preserved as a diagnostic tool, but the SHIP decision and the `docs/experiments/p4_mug_test_sweep_v2.md` conclusion are **withdrawn**. The tool can still be run to demonstrate the construction-identity pattern, but its output cannot be used as an Option 2 decision point.
- **Commit `3c3c8d9` (v2 fixture YAML + retrieval gate):** the `fixture_version: 2` bump with `build_noise_reps=1` / `build_bridges_enabled=true` / `build_text_ec_threshold=0.60` is **withdrawn**. The fixture should stay at v1 shape (no noise, no bridges) until we have a real methodology. The retrieval gate test (`TestFixtureRetrievalGate`) stays because it's valuable regardless of fixture shape.
- **Commit `cf6f485` (milestone report):** the "Option 2 SHIP" conclusion is **withdrawn**. The report is preserved as a historical record of the v2 attempt with a prominent withdrawal notice at the top.

The fold branch's final state will therefore be: infrastructure commits kept, sweep/fixture/milestone commits rewritten to preserve the work while explicitly withdrawing the SHIP conclusion. This post-mortem is the authoritative reframe.

## What the next attempt needs

The user's commitment is "not half-assing this milestone." The next attempt at answering Option 2 empirically needs to satisfy all of the following:

### 1. Non-constructed topology

The cross-class reachability paths that Option 2 would unlock must arise from a process we do NOT control — not from episodes we deliberately wrote to create those paths. Two candidate methodologies:

- **Organic topology via agent loop:** feed a real text corpus (Wikipedia flower descriptions, ImageNet captions, multi-sentence paragraphs that naturally mention multiple categories) through the agent loop. Let episode boundaries form naturally from the usual boundary rules. Measure whether text-text Hebbian edges accumulate from natural co-occurrence patterns. Then run `retrieve_cross_modal` and see whether Stage 1's filter truncates valuable paths. This is closest to "real usage" and hardest to dismiss.
- **Adversarial probe:** build a fixture deliberately designed such that a SPECIFIC set of cross-class pairs is only reachable via text-text bridges that Option 1 blocks. Then probe those specific pairs and measure whether `retrieve_cross_modal` returns them under Option 1 vs Option 2. The key difference from v2: the probe is measuring an INTENDED failure mode, not counting constructed edges. If Option 1 returns zero for the probe set and Option 2 returns them, the lift is a real capability difference, not a tautological count.

### 2. Real signal-vs-noise margin

Fixture construction must produce Hebbian weights where signal strictly exceeds noise at the operating point. The v2 had signal=0.3 noise=0.3 → 1:1 ratio → no margin. Future attempts should guarantee either:

- Signal bound with `k ≥ 2` reinforcements (weight ≥ 0.4) so at `noise_reps=1` (noise=0.3) the ratio is at least 1.33:1. Or,
- `HebbianConfig.init=0.5` so single-episode signal starts at 0.5 and noise at 0.3 gives 1.67:1. Or,
- Noise edges that are structurally weaker than signal edges by construction (e.g., noise from shorter co-activation windows that never reinforce).

### 3. Weight-aware metric, not raw BFS

The v2 sweep used `_shortest_path_hops_raw` with `max_depth=5` and no decay/threshold. Option 2's actual behavior under a split filter would be `spreading_activation` with `traversal_filter=None` and a post-hoc vision-only filter — which applies per-hop decay and a threshold check. A weight-aware metric would count "pairs whose activation is above threshold under Option 2's filter" instead of "pairs reachable by unfiltered BFS." The metric should SIMULATE the production retrieval path as closely as possible.

### 4. Multi-seed variance characterization

The v2 sweep ran every combination exactly once with `seed=0`. The "0.980 ± 0.060" std was per-class variance, not seed variance. The operating-point rule "largest noise_reps at recall ≥ 0.90" was being evaluated against a point estimate with zero characterized uncertainty. Future attempts should use ≥10 seeds per combination (matching Stage 3's 20-seed target is safest) so the gate has a real confidence interval.

### 5. Build-time assertion that catches bridge-token EC collapse

Regardless of methodology, future bridge constructions MUST assert `bridge_node_id not in {c.text_node_id for c in class_results}` immediately after the bridge is registered with EC. If the assertion fires, either the bridge token needs to change OR the EC threshold needs tightening. The assertion is the structural enforcement that would have caught Exec #1 pre-merge without any empirical run. Follows the CLAUDE.md "push silent-no-op invariants into types" rule.

### 6. Falsifiability check

Before the sweep runs, the author of the methodology must be able to describe: **"what result would convince me Option 2 is NOT needed?"** If the answer is "no result — the fixture structurally requires Option 2 to be reachable" then the methodology is construction-identity and cannot ship. A proper methodology has a clear negative outcome that would force the opposite conclusion.

## Status of Option 2 decision

- **Option 2 remains committed as the long-term architectural answer** per user signoff 2026-04-15. This has not changed.
- **The SHIP timing is RESOLVED — DEFER (2026-04-16).** Stage 2 v3 measurement found lift = 0.0000 across 10 seeds. Same-class activation dominates 22:1 under `RetrievalConfig` defaults. Option 2 deferred as post-Stage-3 cleanup. See [p4_option2_measurement.md](p4_option2_measurement.md).
- **`TestStageThreeLimitation` regression guard remains** with updated docstring reflecting the honest rationale (activation decay kills multi-hop signal, not fixture design).
- **Stage 3 shipped on single-hop `retrieve_cross_modal`** — Arm B F1 = 1.000 vs Arm C F1 = 0.901. See [p4_cross_modal_sweep.md](p4_cross_modal_sweep.md).

## Next steps

1. **Ship this post-mortem as a docs-only PR to main.** This establishes the authoritative reframe before any more code changes.
2. **Separate PR: reframe the fold branch `fix/substrate-p4-stage2-fold`.** Rewrite commits `5d25556`, `3c3c8d9`, `cf6f485` to preserve the code as infrastructure + diagnostic and withdraw the SHIP conclusion. Keep the infrastructure commits as-is.
3. **Run the Exec-lens concrete bugs through an investigation session** — empirically verify Exec #1 (encode `"text_flower"` against Flowers102 names through paraphrase-mpnet, check cosine similarity). Document the collapse as a concrete repro in this post-mortem's appendix or a companion doc.
4. **Plan the next methodology.** Write a `docs/plans/substrate_p4_option_2_measurement.md` shell plan that specifies which methodology (organic / adversarial / hybrid), what the falsifiable claim is, what the fixture construction rules are, and what the gate criteria look like. Get user signoff BEFORE implementation.
5. **Implement and run the new methodology** on a fresh branch. Ship Option 2 or re-defer based on that data.
6. **Update `TestStageThreeLimitation`** to reflect whichever direction the new data points.

## Lessons

- **Construction-identity is a failure mode that can repeat in different shapes.** v1 and v2 failed for different mechanical reasons but the same structural reason: the measurement was deducible from the fixture spec without running the substrate. Future reviewers (and future sessions) should check "would this number reproduce with a broken substrate?" as a first-pass falsifiability test.
- **Cross-confirmation between Executor and Architecture lenses is load-bearing** when a review round catches a structural issue in a single commit that affects the interpretation of all downstream work. Both v1 and v2 were caught independently by both lenses at the same severity level — if they had disagreed, the decision would have been harder, but the agreement is the strongest possible signal.
- **"No band-aid" is an operational rule, not a checklist item.** The 0.80 recall cliff at `noise_reps≥2` was flagged in the v2 milestone as "unexplained, Stage 3 follow-up." That phrasing is explicitly the shape the no-band-aid rule forbids: a symptom deferred rather than a root cause fixed. The right move at milestone-writing time was to trace the weights manually, find the tie-break, and either fix the fixture OR explicitly document the cause + its implications.
- **The user's "not half-assing this" commitment turned a scope-cut into a scope-expansion, and that's the right call.** Shipping Option 2 on flawed data would have burned a significant follow-up PR + Stage 3 metric freeze on the wrong answer. Spending another session (or two) designing a methodology that's actually falsifiable is cheap compared to the recovery cost if the wrong answer ships.

---

## Appendix A — Round 2 review round #2 findings (fold branch `fix/substrate-p4-stage2-fold`)

**Architecture lens:** 3 CRITICAL, 5 IMPORTANT, 1 MINOR, 10 INFO-verifications. Central finding: Arch #1 construction identity. Also caught Arch #2 (single-seed operating point not statistically defensible), Arch #3 (0.80 cliff as no-band-aid violation).

**Executor lens:** 2 CRITICAL, 3 IMPORTANT, 3 MINOR, 1 INFO-verification. Central finding: Exec #1 bridge token EC-collapse into Flowers102 class name. Also caught Exec #2 (zero weight margin at operating point, full mechanical explanation of the 0.80 cliff).

**Cross-confirmed:**

- **Arch #7 (unexplained `9` single-hop value) ↔ Exec #1 (bridge token EC collapse).** Same anomaly, two explanations. Exec explains it mechanically.
- **Arch #3 (0.80 cliff unexplained) ↔ Exec #2 (zero weight margin + tie-break).** Same anomaly, full mechanical explanation from Exec.
- **Arch #1 (construction identity) ↔ Exec #4 (raw BFS overstates Option 2 actual behavior).** Same structural critique of the metric from two angles.

Full review reports preserved in `/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/.../tasks/` as session artifacts.

## Appendix B — what's in scope for the docs-only PR that ships this post-mortem

This post-mortem is landing on `main` as a standalone docs PR BEFORE the fold branch itself is touched. The docs PR includes:

- This file: `docs/experiments/p4_stage2_v2_post_mortem.md`
- Withdrawal notice at the top of `docs/experiments/p4_mug_test_sweep.md` (the Stage 2 v1 report) pointing readers here
- Updates to `docs/plans/substrate_p4_cross_modal_binding.md`: top banner, Stage 2 v2 fold status section, Option 2 decision section — reflecting the reopened state

NOT in scope for the docs PR:

- Any changes to the fold branch `fix/substrate-p4-stage2-fold`. That's a separate PR.
- Any code changes. This is docs-only.
- The v2 sweep report (`p4_mug_test_sweep_v2.md`) or the v2 milestone doc — those live on the fold branch, not on main, and will be annotated in the fold branch's reframe PR.
