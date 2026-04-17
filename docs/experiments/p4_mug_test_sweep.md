# Substrate P4 Stage 2 — Mug test sweep (real CLIP + paraphrase-mpnet)

> **⚠️ v1 CONCLUSIONS WITHDRAWN (2026-04-15).** This report's "defer Option 2"
> conclusion was caught by Round 2 Architecture-lens review as **tautological**:
> the fixture had no distractors and no cross-class reachability paths, so
> `retrieve_cross_modal` was mechanically forced to return 1.000 recall
> regardless of whether the substrate was working. The Option 2 defer decision
> rested on unfalsifiable evidence.
>
> A Stage 2 v2 attempt to rebuild the fixture (on branch `fix/substrate-p4-stage2-fold`)
> **also failed Round 2 review** — for a different reason in the same bug class:
> the v2 metric measured graph-theoretic properties of the constructed topology
> rather than substrate behavior. Both v1 and v2 reports are preserved as
> historical record.
>
> **Authoritative reframe:** [p4_stage2_v2_post_mortem.md](p4_stage2_v2_post_mortem.md)
>
> **Current Option 2 status:** RESOLVED — **DEFER** (2026-04-16). Stage 2 v3
> ran an honest measurement via organic shared-concept exposure and found
> lift = 0.0000 across 10 seeds (same-class activation dominates 22:1).
> Option 2 deferred as post-Stage-3 cleanup.
> See [p4_option2_measurement.md](p4_option2_measurement.md).
>
> **Everything below this box is the v1 report as-shipped. The numbers are
> real but the interpretation is invalidated. DO NOT act on the "defer Option 2"
> conclusion.**

---

**Wall clock:** 16.4s
**Encoder arm:** Arm A (paraphrase-mpnet text + clip-ViT-B-32 vision + hippocampus)
**Fixture:** scenarios/substrate/p4_mug_test.yaml (10 classes × 5 samples)

## Summary

- Forward top-5 recall (text → vision): **1.000 ± 0.000** (min 1.00, max 1.00)
- Reverse top-5 recall (vision → text): **1.000 ± 0.000**
- Total successful forward hits: 50 / 50

## Stage 2/3 Option 2 decision data — binding-graph topology

The Stage 1 Round 2 Arch-lens review flagged that if the P4 mug test produces
retrieval paths with ≥2 hops through same-modality intermediates, the current
single-hop-only `retrieve_cross_modal` is artificially limiting Stage 3's head-
to-head and Option 2 (split `node_filter` into `traversal_filter` + `result_filter`)
becomes a blocking prereq. If all retrieval paths are 1-hop (direct text↔vision
Hebbian edges), Option 2 is purely architectural cleanup and can be deferred.

**Histogram of min-hop path lengths from text cue to retrieved vision partner:**

| hops | count | % of hits |
|---|---|---|
| 1 | 50 | 100.0% |

**100% of forward hits are direct (1-hop) retrievals.**

### Option 2 decision

- **All retrieval paths are direct AND forward recall is healthy** (≥0.80). Single-hop `retrieve_cross_modal` is empirically sufficient for Stage 3's mug test. Option 2 is deferred as architectural cleanup with no Stage 3 blocker.

## Calibration footnote — thresholds used

**Text side (paraphrase-mpnet):** EC threshold 0.60, bare class names
(not "a photo of a {X}" prompts). Rationale: paraphrase-mpnet on
`"a photo of a {flower}"` prompts produces cross-class cosine mean
0.555, max 0.721 — every class-distinct prompt is ABOVE the default
EC threshold 0.40 and collapses into a single text node. The bare
class-name distribution has cross-class mean 0.347, max 0.577, which
threshold 0.60 keeps fully separated. Stage 3 Arm B/C (CLIP-text) will
need to revisit this calibration because CLIP-text has a different
similarity distribution.

**Vision side (CLIP ViT-B-32):** EC threshold 1.01 (effectively "never
collapse"). Rationale: see the "Stage 3 concern — CLIP within/cross
similarity overlap" section below. No single threshold cleanly
separates same-class CLIP embeddings from cross-class ones on this
fixture, so the honest construction is to disable EC collapse on the
vision side entirely and let the Hebbian binding graph be the class
abstraction layer.

## Stage 3 concern — CLIP within/cross similarity overlap

**Observed on this fixture (1125 cross-class pairs, 50 within-class pairs):**

| direction | mean | min | max |
|---|---|---|---|
| within-class CLIP cosine | 0.895 | 0.762 | 0.970 |
| cross-class CLIP cosine | 0.779 | 0.659 | **0.932** |

The distributions **overlap**: cross-class max (0.932) exceeds within-
class min (0.762) by 0.17. There is no single cosine threshold that
cleanly separates same-flower from different-flower CLIP embeddings on
this fixture. Some different-flower pairs in CLIP space are more
similar than some same-flower pairs.

**Why this matters for Stage 3:**

Stage 3 Arm C is the "OpenCLIP shared-space cosine similarity" baseline
— the head-to-head opponent. Its ranking function is direct CLIP
cosine. The substrate's Arm B (CLIP-text + hippocampus) ranks via
Hebbian binding weights, which accumulate evidence from episode
co-activation and can outperform raw cosine when the underlying
geometry is noisy. This observed overlap is exactly the scenario where
the substrate's binding mechanism should add measurable lift over Arm C
— it's EVIDENCE-based rather than GEOMETRY-based.

**However**, it also means Arm C's baseline F1 on this fixture may be
lower than the published OpenCLIP Flowers-102 numbers lead us to
expect. The published numbers are measured on a different retrieval
task (zero-shot classification against all 102 class prompts). Stage 3's
mug test is measuring class-restricted cross-modal retrieval where
overlap between the 10 chosen classes dominates. Stage 3 needs to
report both the Arm B vs Arm C margin AND the Arm C absolute number
so the comparison is honest.

**Action item for Stage 3 plan:** capture Arm C baseline on this
specific fixture before running Arm B, so we know the floor the
substrate has to clear. This is a Stage 3 measurement, not a Stage 2
one — Stage 2 just surfaces the concern.

## Per-class results

| class | forward top-5 | reverse top-5 | forward hits | path hops (direct/other) |
|---|---|---|---|---|
| balloon flower | 1.00 | 1.00 | 5/5 | 5/0 |
| oxeye daisy | 1.00 | 1.00 | 5/5 | 5/0 |
| water lily | 1.00 | 1.00 | 5/5 | 5/0 |
| lotus | 1.00 | 1.00 | 5/5 | 5/0 |
| pincushion flower | 1.00 | 1.00 | 5/5 | 5/0 |
| azalea | 1.00 | 1.00 | 5/5 | 5/0 |
| fritillary | 1.00 | 1.00 | 5/5 | 5/0 |
| orange dahlia | 1.00 | 1.00 | 5/5 | 5/0 |
| morning glory | 1.00 | 1.00 | 5/5 | 5/0 |
| mexican petunia | 1.00 | 1.00 | 5/5 | 5/0 |

## How to reproduce

```bash
# Requires the 'semantic' extra (sentence-transformers + torch)
# and the Flowers102 torchvision cache populated by Phase 2B.
PYTHONPATH=src python scripts/p4_mug_test_sweep.py
```

The script is deterministic given the fixture's pinned class names and
sample indices. Re-running on the same environment produces byte-identical
recall + topology numbers.
