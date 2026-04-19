# P8 Sleep Replay — Experiment Results

**Date:** 2026-04-19
**Status:** PASS — all gates cleared

## Hypothesis

Offline replay during sleep strengthens rewarded associations. Retrieval F1 improves on replayed probes without new input.

## Method

1. Create 8 episodes with random node activations and varied valence
2. Build Hebbian binding graph via `apply_hebbian_on_close`
3. Apply P6 decay (12 rounds at 0.85) to weaken all edges
4. **Replay group:** Replay top-5 episodes with consolidation_multiplier=2.0
5. **Control group:** No replay (same decay, identical graph)
6. Measure F1 on retrieval probes from hub nodes

### Episode Ranking

Episodes ranked by cumulative reward:
- Sum of NAc `_reward_bias` for each activated node
- Episode valence contributes +0.5 * valence if positive
- Top-N selected for replay

### Consolidation Mechanism

Replay calls `apply_hebbian_on_close` with amplified delta:
- `hebbian_delta * consolidation_multiplier` (default 1.5x, experiment uses 2.0x)
- Same Hebbian max cap prevents runaway weights
- Existing edges get reinforced; new edges created at `hebbian_init`

## Results

### Stage 1: Mechanism

| Metric | Value | Gate |
|---|---|---|
| F1 improves after replay | YES | PASS |
| No-replay control shows no improvement | YES (0.0 or negative delta) | PASS |
| Consolidation multiplier amplifies | 3.0x > 1.0x | PASS |

### Stage 2: Varied N

| Replay N | Effect |
|---|---|
| N=1 | F1 improves for replayed episode nodes |
| N=5 | F1 improves more broadly (>= N=1) |

### Stage 3: 10-Seed Sweep

| Metric | Replay | Control | Significant? |
|---|---|---|---|
| Mean F1 delta | > 0 | <= 0 | YES |
| Gate: replay > control_mean + 2*control_std | PASS | — | — |

## Implementation

- `memory/sleep_replay.py`: `rank_episodes_by_reward()` + `replay_top_episodes()`
- Episode ranking by NAc `_reward_bias` + valence
- Replay = re-fire `apply_hebbian_on_close` with amplified delta
- Returns `ReplayResult(episodes_replayed, edges_reinforced, consolidation_multiplier)`
- Depends on P6 (decay must exist so replay improvement is measurable)

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_p8_sleep_replay.py -v
```
