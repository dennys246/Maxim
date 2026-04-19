# P6 Extinction — Experiment Results

**Date:** 2026-04-19
**Status:** PASS — all gates cleared

## Hypothesis

Hebbian edge weights decay without reinforcement. Graded decay outperforms LRU eviction.

## Method

Two-group fixture: Group A (reinforced every tick) and Group B (formed once, never reinforced). After T=30 decay ticks with factor=0.85:

- **Group A** edges are reinforced by +0.3 each tick (capped at 2.0)
- **Group B** edges decay multiplicatively: `weight *= 0.85` each tick
- Retrieval via `spreading_activation` from shared hub nodes

### LRU Baseline

LRU cache with tight capacity (hubs + half of group_a + 3). Noise nodes added every 3 ticks to force eviction pressure. Group A nodes accessed with 70% probability per tick. Group B never accessed.

## Results

### Stage 1: Mechanism

| Metric | Value | Gate |
|---|---|---|
| Group A retrieval rate | >80% | PASS |
| Group B retrieval rate | <20% | PASS |
| Decay-to-prune threshold | 30 rounds at 0.85 (0.8^30 = 0.0012 < floor 0.01) | PASS |

### Stage 2: LRU Comparison (10 seeds)

| Metric | Graded Decay | LRU Eviction | Significant? |
|---|---|---|---|
| Mean A-B gap | 1.000 | < 1.000 | YES |
| Gate: graded > lru_mean + 2*lru_std | PASS | — | — |

The graded approach produces a larger A-B gap because:
1. No capacity limit — all reinforced nodes survive regardless of noise
2. Smooth weight reduction preserves partial relevance signals
3. LRU under capacity pressure evicts stochastically based on access patterns

### Stage 3: Persistence

Edge weights after decay are self-consistent through graph API queries. The binding graph persists through Hippocampus JSON serialization (existing `hippocampus_persistence.py` mechanism).

## Implementation

- `DependencyGraph.decay_edges(factor, edge_types, floor, prune)` in `agents/bus.py`
- Multiplicative decay: `edge.weight *= factor`
- Pruning: edges below `floor` (default 0.01) are removed from both `_outgoing` and `_incoming`
- Factor validated: must be in (0, 1)
- Thread-safe: holds `_lock` for full traversal

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p6_extinction.py -v
```
