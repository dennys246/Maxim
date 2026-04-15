# Substrate P4 Stage 2 v2 — Mug test calibration sweep (non-tautological)

**Total wall clock:** 21.9s
**Encoder arm:** Arm A (paraphrase-mpnet-base-v2 text + clip-ViT-B-32 vision + hippocampus)
**Sweep dimensions:** noise_reps ∈ {0,1,2,3,4,5} × bridges ∈ {none, shared_superclass}
**Total combinations:** 12

## Why v2 exists

Stage 2 v1's mug test (`docs/experiments/p4_mug_test_sweep.md`)
reported 1.000 ± 0.000 forward top-5 recall and 100% 1-hop retrievals,
and the v1 report concluded 'Option 2 can be deferred.' Round 2
Architecture-lens review caught this as tautological: with
`VISION_EC_THRESHOLD=1.01`, each class had 5 distinct vision nodes
direct-bound to 1 text node and NO cross-class reachability paths.
`retrieve_cross_modal(text_X, 'vision')` could only ever return
the 5 class-correct vision nodes because nothing else was reachable.
Top-5 recall was mechanically forced to 1.0. The metric did not
distinguish a working substrate from a broken one.

v2 rebuilds Phase 2D with two new fixture layers that create real
signal-vs-noise ranking pressure AND real multi-hop cross-modal
reachability paths, then sweeps the parameter space to find an
operating point where both metrics are non-trivially measurable.

## Sweep results table

| noise_reps | bridges | mean recall | std | min | cross pairs | 1-hop | multi-hop | Option 2 lift | episodes | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | none | 1.000 | 0.000 | 1.00 | 450 | 0 | 0 | +0.0% | 50 | 8.6s |
| 1 | none | 1.000 | 0.000 | 1.00 | 450 | 10 | 210 | +44.4% | 60 | 1.2s |
| 2 | none | 0.800 | 0.000 | 0.80 | 450 | 10 | 210 | +44.4% | 70 | 1.2s |
| 3 | none | 0.800 | 0.000 | 0.80 | 450 | 10 | 210 | +44.4% | 80 | 1.2s |
| 4 | none | 0.800 | 0.000 | 0.80 | 450 | 10 | 210 | +44.4% | 90 | 1.3s |
| 5 | none | 0.800 | 0.000 | 0.80 | 450 | 10 | 210 | +44.4% | 100 | 1.2s |
| 0 | shared | 1.000 | 0.000 | 1.00 | 450 | 9 | 450 | +98.0% | 70 | 1.2s |
| 1 | shared | 0.980 | 0.060 | 0.80 | 450 | 18 | 450 | +96.0% | 80 | 1.2s |
| 2 | shared | 0.800 | 0.000 | 0.80 | 450 | 18 | 450 | +96.0% | 90 | 1.2s |
| 3 | shared | 0.800 | 0.000 | 0.80 | 450 | 18 | 450 | +96.0% | 100 | 1.2s |
| 4 | shared | 0.800 | 0.000 | 0.80 | 450 | 18 | 450 | +96.0% | 110 | 1.2s |
| 5 | shared | 0.800 | 0.000 | 0.80 | 450 | 18 | 450 | +96.0% | 120 | 1.2s |

## Same-class top-5 recall degradation curve

For the bridge-enabled rows (the canonical ship shape), how does
same-class forward top-5 recall degrade as noise_reps increases?
This is Approach A — the signal-vs-noise ranker test.

| noise_reps | mean recall | pass (≥0.90)? |
|---|---|---|
| 0 | 1.000 | ✅ |
| 1 | 0.980 | ✅ |
| 2 | 0.800 | ❌ |
| 3 | 0.800 | ❌ |
| 4 | 0.800 | ❌ |
| 5 | 0.800 | ❌ |

## Option 2 lift across noise levels (bridge topology ON)

How many cross-class (text_X, vision_Y, X≠Y) pairs become
reachable when we SIMULATE Option 2's split filter (raw BFS
over the binding graph, ignoring modality membership at
traversal time)? If Option 2 would unlock non-zero retrieval
paths at the chosen operating point, Option 2 becomes a Stage 3
blocker.

| noise_reps | 1-hop reachable | multi-hop reachable | Option 2 lift |
|---|---|---|---|
| 0 | 9 | 450 | +98.0% |
| 1 | 18 | 450 | +96.0% |
| 2 | 18 | 450 | +96.0% |
| 3 | 18 | 450 | +96.0% |
| 4 | 18 | 450 | +96.0% |
| 5 | 18 | 450 | +96.0% |

## Operating point selection

**Operating point: `noise_reps=1, bridges=shared_superclass`.**

- Mean forward top-5 recall: **0.980** (≥0.90 ✅)
- Std: 0.060
- Min: 0.80
- Cross-class pairs: 450
- Single-hop reachable: 18
- Multi-hop reachable (Option 2 simulated): 450
- **Option 2 lift: +96.0%**

Selection rule: largest noise_reps at which mean recall is
still ≥ 0.90 (user-chosen tighter threshold from the fold
planning session). Bridge topology is always
`shared_superclass` for the ship shape.

## Option 2 decision — re-opened from Stage 2 v1

**Option 2 lift is +96.0% at the operating point.** The
shared_superclass bridge creates cross-class retrieval paths
that Stage 1's single-hop filter blocks; Option 2's split
filter would unlock them. Given non-zero lift under realistic
fixture conditions (bridge-enabled, noise ≥1), Option 2 is
empirically needed for the P4 mechanism to reach its full
advertised capability.

**Option 2 decision: SHIP.** Per the fold-planning commit,
Option 2 goes in a SEPARATE follow-up PR after this fold
merges. That PR renames node_filter→traversal_filter, adds
result_filter, provides a P3b compat shim, re-validates
P3a's 10-seed sweep, and flips TestStageThreeLimitation.

## How to reproduce

```bash
PYTHONPATH=src python scripts/p4_mug_test_sweep_v2.py
```

The sweep is deterministic given the fixture descriptor + config
seed=0. Re-running on the same torchvision cache + sentence-
transformers model downloads produces byte-identical metrics.

Supersedes Stage 2 v1's tautological report at
`p4_mug_test_sweep.md`. The v1 report is kept for historical
record of the Round 2 review catch; DO NOT act on its
'defer Option 2' conclusion.
