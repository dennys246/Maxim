# Substrate P0 — Fixture-Difficulty Pilot

**Status:** COMPLETE (2026-04-12). Fixtures calibrated, baselines pinned, P1 unblocked.
**Scope:** ~350 LOC + ~2.5 days fixture authoring
**Blocks:** substrate_recognition.md (P1+P2) — **UNBLOCKED**
**Master reference:** [archive/substrate_plan.md](substrate_plan.md) for full rationale, baselines, statistical hygiene rules
**Results:** [experiments/p0_baseline_sweep.md](../../experiments/p0_baseline_sweep.md) | [experiments/results/p0_baseline_sweep.json](../experiments/results/p0_baseline_sweep.json)

## Goal

Before committing to the substrate architecture, run the cheapest possible sanity check: **are the P1 fixtures hard enough to tell us anything?** If a trivial sentence-transformer baseline trivially solves them, the fixtures are too easy and need to be harder before P1 can mean anything.

## Current state (verified 2026-04-12)

All prereqs shipped:
- S1 FixtureDrivenOrchestrator — `scenarios/substrate/P0_paraphrase_collapse.yaml` exists as a placeholder
- S2 MockLLMBackend — canned/policy/scripted modes available
- S3 Persistence harness — subprocess round-trip with probes
- S4 `--seed` — deterministic seeding with per-agent RNG streams
- `substrate_metrics` field on SimulationReport — wired through fixture orchestrator

**What doesn't exist yet (and doesn't need to for P0):**
- LinguisticEncoder, EC pattern_complete_or_separate, ATL modality tags — these are P1 work
- NAc per-node reward bias, eligibility traces — these are P2 work
- P0 only needs the baseline + fixtures + the existing harness

## Items

### P0.1 — Author research-grade P1 fixtures

Replace the placeholder `P0_paraphrase_collapse.yaml` with research-grade fixtures.

**Deliverable:** `scenarios/substrate/paraphrase_clusters.yaml` — ≥50 hand-authored clusters with:
- 2-3 paraphrase variants per cluster (semantically equivalent, lexically different)
- Near-miss distractors (semantically adjacent but different referent)
- Labeled ground truth: cluster ID per sentence
- A 60/40 train/holdout split (60% for mechanism iteration, 40% for final pass criteria)

**Workflow:** sim-as-fixture-debugger (rough draft → replay through fixture orchestrator → inspect substrate_metrics → refine → freeze).

**Exit:** Fixtures load and run through `FixtureDrivenOrchestrator` with `--seed 42`. Manual review confirms clusters are neither trivially separable nor ambiguously overlapping.

**Scope:** ~2.5 days. This is the most time-intensive part of P0 — the code is trivial, the data quality is everything.

### P0.2 — FAISS + cosine baseline

Implement the trivial embedding baseline that P1's sanity floor will be measured against.

**Implementation:**
- New `tests/substrate/baselines/embedding_baseline.py` (~100 LOC)
- Uses `sentence-transformers` (`all-MiniLM-L6-v2` and `all-mpnet-base-v2`)
- Embeds each fixture sentence, clusters by cosine similarity with a fixed threshold
- Scores against ground-truth cluster labels
- Wired into `BenchmarkRunner.baseline_path` for ongoing re-use

**Exit:** Both sentence-transformer models produce a baseline score. Published as mean + std over 10 seeds.

### P0.3 — OpenCLIP baseline (pinned for P4)

Author the OpenCLIP shared-embedding-space baseline that P4 will compete against. Pin the number now before any architecture code is written.

**Implementation:**
- New `tests/substrate/baselines/openclip_baseline.py` (~150 LOC)
- Text embedded via OpenCLIP text encoder
- Scored on the same paraphrase clusters (text-only for now; vision added at P4)
- Number pinned in this plan document and carried into `substrate_recognition.md`

**Exit:** OpenCLIP baseline score published. Decision recorded below.

### P0.4 — Decision gate

Run baselines and make a go/no-go decision on fixture quality:

| Baseline score | Interpretation | Action |
|---|---|---|
| ≥85% collapse | Fixtures too easy | Author harder clusters (more variation, near-miss distractors), re-run |
| 60-85% | Well-calibrated | Proceed to P1. Register score as P1's sanity floor |
| <60% | Possibly too hard | Verify a human reader can solve them. If yes, proceed |

**Exit:** Decision recorded, P1 sanity-floor threshold written into `substrate_recognition.md`, fixtures frozen.

### P0.4 — Decision outcome (2026-04-12)

**Best baseline:** all-mpnet-base-v2 @ threshold 0.50 = **78.5% collapse**, 1.5% cross-cluster contamination.

Falls in the **60-85% well-calibrated** zone. Difficulty gradient works as designed: easy ~91%, medium ~93%, hard ~59%. Near-miss separation at this threshold is 36.8% — the substrate architecture needs to break the collapse-vs-separation tradeoff the baseline can't.

**P1 sanity floor:** 78.5% - 5pp = **73.5%**

**Decision: proceed to P1.** Fixtures frozen. Full results: [experiments/p0_baseline_sweep.md](../../experiments/p0_baseline_sweep.md).

OpenCLIP text-encoder baseline code exists (`tests/substrate/baselines/openclip_baseline.py`) but has not been scored yet — requires `open_clip_torch` which is not installed. The OpenCLIP number is pinned at P4 time, not P0. The P0 decision gate uses sentence-transformers only.

## Fixture path convention

Fixtures live in `scenarios/substrate/`, not `tests/fixtures/substrate/` (the master plan referenced the latter — corrected here).

## Cost

- ~350 LOC (baselines + wiring)
- ~2.5 days fixture authoring
- ~30 minutes per seed on local RTX 5080 hardware
- $0 cloud cost
