# Substrate P5 — Robust Cross-Session Persistence Under Stress

**Status:** Draft — opens after P4 (CLOSED) + concept decomposition land.
**Scope:** ~400 LOC + ~100 metric extractor
**Target version:** 0.5
**Gates:** null (not 1.0-gating, but blocks P6 and P8)
**Depends on:** P3.5 (persistence protocol), P4 (cross-modal binding)
**Blocks:** P6 (extinction), P8 (sleep replay)
**Parent:** [substrate_binding_persistence.md](archive/substrate_binding_persistence.md)
**Related:** [memory_consolidation_practice.md](memory_consolidation_practice.md)

## Goal

Prove that the bio-substrate survives realistic persistence load: 10,000+ nodes, 1,000+ episodes, mixed modalities, repeated serialize/reload cycles with no degradation.

## Hypothesis

A substrate populated with 10k+ nodes and 1k+ episodes across mixed channels, serialized every 100 episodes and reloaded, produces identical retrieval results before and after each cycle, with bounded state size and load time <5s.

## Dependencies (scaffolding audit)

| Surface | Status | Notes |
|---|---|---|
| `BioSystemSnapshot` Protocol (P3.5) | Shipped (S1+S2) | `dump()` / `load()` on all bio-systems |
| `persistence_harness.py` (S3) | Shipped | Subprocess round-trip harness |
| Hippocampus save/load | Shipped | Core persistence |
| NAc save/load (reward bias) | Shipped (P2) | `_reward_bias` fields persist |
| ATL save/load | Shipped | Concept-level persistence |
| PerceptTraceBuffer save/load | Shipped (P3.5 S2) | Buffer state round-trips |
| Episode dataclass + EpisodeStore | Shipped (P3a) | Episode-level persistence |
| Cross-modal nodes (P4) | Shipped | Vision + text nodes coexist |

## Stages

### Stage 1 — mechanism + metric

**What's built:**
- Stress fixture generator: creates 10k+ nodes across text + vision modalities with realistic episode structure
- Metric extractor: measures retrieval stability (F1 delta before/after reload), state size growth, load time
- Mechanism test: 10 serialize/reload cycles on a 1k-node substrate, verify zero F1 degradation

**Pass gate:** F1 delta = 0.0 across 10 reload cycles at 1k nodes. State size growth sub-linear.
**Tests:** `tests/substrate/test_p5_stress_persistence.py`

### Stage 2 — mid-scale validation

**What's built:**
- Scale to 10k+ nodes, 1k+ episodes
- Mixed-modality nodes (text + vision from P4)
- Serialize every 100 episodes during population
- Measure load time at each checkpoint

**Pass gate:** Load time <5s for 10k nodes. F1 delta = 0.0 across all checkpoints. Mean + std across 10 seeds.

### Stage 3 — full sweep + pre-merge review

**What's built:**
- Full 10-seed sweep with 10k+ nodes
- State size bounding verification (no unbounded growth)
- Pre-merge two-lens review (Executor + Architecture)

**Pass gate:** All Stage 2 criteria hold across 10 seeds. State size bounded.
**Reviewers:** Executor + Architecture lenses

## Pass criteria (maps to 0.5 gate)

- State size bounded (sub-linear growth with node count)
- Retrieval stable across 10+ save/reload cycles
- Load time <5s for 10k nodes
- Mean + std across 10 seeds

## Deferred follow-ups

- Concurrent-write stress (multi-agent persistence) — deferred to agent_factory_canonicalization
- Schema migration stress (version upgrade during load) — deferred

## Load-bearing invariants (filled in AFTER shipping)
