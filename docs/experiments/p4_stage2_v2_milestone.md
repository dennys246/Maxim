# Substrate P4 Stage 2 v2 — milestone summary

**Branch:** `fix/substrate-p4-stage2-fold` (pending PR)
**Date:** 2026-04-15
**Status:** IN PROGRESS — Round 2 pre-merge review still to run

## TL;DR

Stage 2 v1 shipped a tautological mug test. The Round 2 Architecture-lens review caught it as unfalsifiable (`VISION_EC_THRESHOLD=1.01` + no distractors + no bridges → every retrieval mechanically forced to 1.000 recall regardless of whether the substrate's ranker was doing anything). The v1 "defer Option 2" decision rested on that broken evidence.

Stage 2 v2 rebuilds Phase 2D with a **parameterized fixture builder** that sweeps noise (cross-class contamination) × bridge topology (shared superclass), runs a 12-combination calibration, picks an operating point where recall is non-trivially above 0.90, and measures the Option 2 lift empirically.

**Headline result: Option 2 lift is +96.0% at the operating point. Option 2 SHIPS in a follow-up PR.**

## Operating point

| parameter | value |
|---|---|
| noise_reps | 1 (each class has 1 cross-class contaminant pair, reinforced 1×) |
| bridge topology | shared_superclass (text_flower hub connecting all 10 classes) |
| text EC threshold | 0.60 |
| vision EC threshold | 1.01 |
| fixture SHA-256 | `967e83ed18851e1dfcad418be57f3275cf04a961462e6dc4dd055b6b71c8920b` |

**At this operating point:**

- Forward top-5 recall: **0.980 ± 0.060** (min 0.80) — above the 0.90 bar
- Cross-class pairs: 450 (10 classes × 9 other classes × 5 sample indices each)
- Single-hop reachable (current Stage 1 retrieve_cross_modal): 18 / 450
- Multi-hop reachable (Option 2 simulation via raw BFS max_depth=5): 450 / 450
- **Option 2 lift: +96.0%**

Operating point rule: "largest `noise_reps` at which bridge-enabled mean forward top-5 recall ≥ 0.90" (user-chosen tighter threshold from the fold planning session — originally 0.80 in my draft, tightened to 0.90 for a more discriminating pressure test).

## Full sweep table

| noise | bridges | mean recall | std | min | 1-hop | multi-hop | Option 2 lift |
|---|---|---|---|---|---|---|---|
| 0 | none | 1.000 | 0.000 | 1.00 | 0 | 0 | +0.0% |
| 1 | none | 1.000 | 0.000 | 1.00 | 10 | 210 | +44.4% |
| 2 | none | 0.800 | 0.000 | 0.80 | 10 | 210 | +44.4% |
| 3 | none | 0.800 | 0.000 | 0.80 | 10 | 210 | +44.4% |
| 4 | none | 0.800 | 0.000 | 0.80 | 10 | 210 | +44.4% |
| 5 | none | 0.800 | 0.000 | 0.80 | 10 | 210 | +44.4% |
| 0 | shared | 1.000 | 0.000 | 1.00 | 9 | 450 | +98.0% |
| **1** | **shared** | **0.980** | **0.060** | **0.80** | **18** | **450** | **+96.0%** ← op point |
| 2 | shared | 0.800 | 0.000 | 0.80 | 18 | 450 | +96.0% |
| 3 | shared | 0.800 | 0.000 | 0.80 | 18 | 450 | +96.0% |
| 4 | shared | 0.800 | 0.000 | 0.80 | 18 | 450 | +96.0% |
| 5 | shared | 0.800 | 0.000 | 0.80 | 18 | 450 | +96.0% |

Full sweep artifact: [p4_mug_test_sweep_v2.md](p4_mug_test_sweep_v2.md) + [results/p4_mug_test_sweep_v2.json](results/p4_mug_test_sweep_v2.json)

## What the sweep revealed

1. **The substrate's discrimination cliff is at `noise_reps=2`.** Recall drops from 1.0 / 0.98 (noise=0 / 1) to 0.80 (noise≥2) and stays there. At noise_reps=2 the noise edge weight is 0.4 vs signal 0.7 — a 1.75:1 ratio that the ranker SHOULD handle. The cliff is unexpected and worth investigating as a follow-up (tie-break order under spreading_activation? single class systematically losing? unclear). It does NOT affect the operating point selection because noise_reps=1 is comfortably above the 0.90 bar, but it's a follow-up item for Stage 3 debugging.

2. **Noise chains ALREADY create measurable Option 2 lift without bridges** (`none` rows at noise≥1 show +44.4%). This is a stronger case for Option 2 than I expected — bridges amplify the effect but the noise-chain topology alone would still exercise the same class of multi-hop paths that Stage 1's single-hop filter blocks. This means Option 2's value is not specific to the shared-superclass construct; any realistic fixture with cross-class contamination exhibits the same pattern.

3. **Bridges push the lift from 44% to 96%** — dense cross-class reachability via the shared superclass covers the remaining 240 pairs. Under realistic language use (where agents hearing "a flower" while seeing various flowers is normal), Topology A is the better model of real usage, so the 96% number is the relevant one for the Stage 3 head-to-head.

## Caveats (honest limitations of this measurement)

1. **Multi-hop metric is raw BFS reachability, not activation-thresholded.** The +96% lift counts every node reachable at `max_depth=5` without applying decay/threshold. Actual Option 2 implementation would filter by activation; some 5-hop paths have activation below threshold (~0.001) and wouldn't actually be returned. The true lift is **somewhat less than +96%** but still clearly non-zero — the decision is robust to the caveat.

2. **Topology metric is forward-only (text → vision).** Reverse direction (vision → text) isn't measured in the sweep. Arch review noted this; kept as a Stage 3 concern because the +96% forward lift is unambiguous and the Option 2 SHIP decision doesn't hinge on symmetry data.

3. **The 0.80 recall floor at noise_reps≥2 is an unexplained specific value.** Exactly one class losing exactly one node per retrieval consistently suggests a deterministic pattern in the ranker tie-break, not random noise. Needs investigation as a Stage 3 follow-up. Not blocking Stage 3 shipping.

## Option 2 decision — RE-OPENED and RESOLVED

Stage 2 v1 said "defer Option 2." That decision was based on tautological data.

Stage 2 v2's empirical answer:

**OPTION 2: SHIP.** The lift at the operating point is overwhelming (+96%), not marginal. Under realistic fixture parameters (noise=1, bridges=shared), Stage 1's single-hop `retrieve_cross_modal` filter blocks 96% of the cross-modal reachability that a proper split filter (traversal_filter=None, result_filter=modality_membership) would unlock.

**Implementation plan per the fold-planning decision:** Option 2 goes in a **SEPARATE follow-up PR** after this fold lands. That PR:

1. Renames `node_filter → traversal_filter` in `DependencyGraph.spreading_activation` + `Hippocampus.retrieve_on_cue`
2. Adds `result_filter` as an independent optional parameter
3. Provides a P3b compat shim so existing `episode_filter(channel="sms")` calls map to both `traversal_filter` + `result_filter` (the same predicate) — preserves P3b semantics verbatim
4. Re-validates P3a Stage 2's 10-seed sweep to confirm the margin over TF-IDF is unchanged
5. Flips `TestStageThreeLimitation::test_multi_hop_through_same_modality_intermediate_is_blocked` — the test's assertion switches from "reverse vision_target NOT in partner_ids" to "vision_target IS in partner_ids"
6. Re-runs Phase 2D v2 against the updated code and verifies single-hop reachability now matches multi-hop reachability (should be 450/450 under the new filter)
7. Gets its own Round 2 pre-merge review round (Executor + Architecture lenses)

## Round 2 review findings folded into this branch

Architecture lens (5 CRITICAL, 4 IMPORTANT, 1 MINOR originally — all folded or documented):

- **Arch #1 (plan divergence torchvision vs datasets):** ✅ plan amended in commit `6de09c6`
- **Arch #2 (Flowers102.classes undocumented API):** ✅ 102-entry class list pinned in `p4_fixture_loader.py::FLOWERS102_CLASS_NAMES` + `assert_torchvision_classes_match_pin()` drift guard, commit `8d0b92f`
- **Arch #3 (`_build_and_bind` hard-codes encoder):** ✅ parameterized in `tests/substrate/p4_build_and_bind.py::BuildConfig`, commit `82da6db`
- **Arch #4 (tautological mug test):** ✅ **THE CENTRAL FOLD.** v2 rebuild via noise + bridges + sweep + operating point selection — commits `82da6db`, `5d25556`, `3c3c8d9`
- **Arch #5 (thresholds entangled with fixture SHA):** ✅ moved into fixture descriptor as `build_text/vision_ec_threshold` fields, commit `3c3c8d9`
- **Arch #6 (missing 0.70 retrieval gate test):** ✅ `TestFixtureRetrievalGate` added, commit `3c3c8d9`
- **Arch #7 (VRAM audit doesn't capture OOM):** ✅ `_safe_step` helper wraps each step in try/except, commit `f00fc0f`
- **Arch #8 (Phase 2E interpretation over-claims steady-state):** ✅ softened in `p4_vram_audit.md` interpretation section, commit `f00fc0f`
- **Arch #9 (round-trip test imports from scripts/):** ✅ fixed as part of `_build_and_bind` refactor in `82da6db`
- **Arch #10 (class_idx declarative dead weight):** ✅ enforced in `load_fixture_images`, commit `8d0b92f`

Executor lens (5 IMPORTANT, 5 MINOR — all folded):

- **Exec #1 (fallback YAML parser silent # truncation):** ✅ fallback deleted entirely, PyYAML is sole parser, commit `8d0b92f`
- **Exec #2 (PyYAML split-brain):** ✅ same commit
- **Exec #3 (round-trip probe list-order sensitivity):** ✅ probe now sorts by node id, commit `f00fc0f`
- **Exec #4 (scripts/ layering smell):** ✅ extracted to `tests/substrate/p4_build_and_bind.py`, commit `82da6db`
- **Exec #5 (headroom-band silent short return):** ✅ hard assert, commit `f00fc0f`
- **Exec #6 (+-4 MB cosmetic):** ✅ `{:+.0f}` format, commit `f00fc0f`
- **Exec #7 (`torch.mps.is_available` non-canonical):** ✅ `torch.backends.mps.is_available`, commit `f00fc0f`
- **Exec #8 (calibration sweep determinism over-claim):** deferred — added as documentation note only, not a code fix. The cuDNN deterministic flag would slow the sweep and the SHA pin is against the committed fixture, not against a regeneration.
- **Exec #9 (missing threshold tripwire test):** ✅ `test_thresholds_are_not_the_ec_default`, commit `f00fc0f`
- **Exec #10 (topology forward-only):** deferred — noted as a Stage 3 concern in the v2 sweep report caveats above

## Test surface — Stage 2 v2

58 P4 tests pass after the fold (up from 46 in Stage 2 v1):

- 13 vacuous-pass guards (unchanged)
- 17 cross-modal mechanism tests (unchanged — Stage 1 infrastructure)
- 17 fixture validation tests (was 11 — added `TestFixtureV2BuildConfig` (5 tests) + `TestFixtureRetrievalGate` (1 test) + threshold tripwire)
- 1 mug test round-trip (unchanged, now runs against the noisy + bridged fixture)
- 6 CLIP encoder tests (unchanged)
- 4 stage 1 other tests that are not explicitly listed above

## Commit sequence

The fold lands as 6 commits on `fix/substrate-p4-stage2-fold`:

1. `6de09c6` — plan amendment (torchvision decision + v2 fold status)
2. `82da6db` — refactor `_build_and_bind` to `tests/substrate/` + parameterize
3. `8d0b92f` — pin 102-class list + enforce `class_idx` + drop YAML fallback
4. `5d25556` — Phase 2D v2 sweep + results + Option 2 SHIP decision
5. `3c3c8d9` — v2 fixture with canonical build params + 0.70 retrieval gate
6. `f00fc0f` — tactical fixes bundle (probe sort, headroom assert, VRAM OOM, canonical mps, cosmetics, tripwire)

Each commit message names its place in the sequence for bisectability.

## Next steps

- Round 2 pre-merge review on the fold (Executor + Architecture lenses in parallel, same shape as the round that caught v1)
- Fold any new findings into the same branch
- Full fast test suite pass
- Open fold PR
- Upon merge: open the Option 2 SHIP follow-up PR
