# Substrate P6 — Extinction Without Reinforcement

**Status:** Draft — opens after P3a (CLOSED).
**Scope:** ~300 LOC + ~100 metric extractor
**Target version:** 0.5
**Gates:** null (not 1.0-gating)
**Depends on:** P3a (episode binding — Hebbian link formation)
**Blocks:** P8 (sleep replay needs decay mechanism)
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [behavioral_convergence_practice.md](behavioral_convergence_practice.md)

## Goal

Prove that substrate links decay appropriately without reinforcement, and that this graded decay outperforms a simple LRU eviction baseline.

## Hypothesis

Two groups of nodes: Group A (reinforced across episodes) and Group B (not reinforced after initial formation). After simulated time via SCN phase ticks, Group B retrieval drops below 20% while Group A stays above 80%. The graded decay mechanism beats an LRU baseline by `baseline_mean + 2 * baseline_std`.

## Dependencies (scaffolding audit)

| Surface | Status | Notes |
|---|---|---|
| Hebbian edge weights (P3a) | Shipped | `DependencyGraph.update_edge(weight=...)` |
| SCN phase ticks | Shipped | Simulated time progression |
| NAc reward bias (P2) | Shipped | Reinforcement signal |
| Episode binding (P3a) | Shipped | Co-occurrence → link formation |
| Persistence round-trip (P3.5) | Shipped | Save/load cycle verification |

## Stages

### Stage 1 — mechanism + metric

**What's built:**
- Decay function: edge weight decays per SCN tick without reinforcement
- Two-group fixture: Group A (reinforced every N ticks), Group B (formed once, never reinforced)
- Metric extractor: retrieval rate per group over time
- LRU baseline: evict least-recently-accessed nodes instead of graded decay

**Pass gate:** Group B retrieval <20%, Group A retrieval >80% after T ticks. Mechanism test on synthetic data.
**Tests:** `tests/substrate/test_p6_extinction.py`

### Stage 2 — two-group simulation

**What's built:**
- Realistic two-group fixture with mixed episodes
- SCN-driven time progression (not wall clock)
- Retrieval probes at regular intervals
- LRU head-to-head comparison

**Pass gate:** Graded decay beats LRU by `baseline_mean + 2*baseline_std`. Mean + std across 10 seeds.

### Stage 3 — full sweep + pre-merge review

**What's built:**
- 10-seed sweep with varied reinforcement schedules
- Persistence round-trip at each measurement point
- Pre-merge two-lens review

**Pass gate:** All Stage 2 criteria across 10 seeds. Persistence stable.
**Baseline:** LRU eviction
**Reviewers:** Executor + Architecture lenses

## Pass criteria (maps to 0.5 gate)

- Group A retrieval >80% after T ticks
- Group B retrieval <20% after T ticks
- Beats LRU baseline by `baseline_mean + 2*baseline_std`
- Persistence round-trip at every measurement point
- Mean + std across 10 seeds

## Deferred follow-ups

- Adaptive decay rates (context-dependent half-life) — practice doc territory
- Partial extinction (some links decay faster than others within an episode)

## Load-bearing invariants (filled in AFTER shipping)
