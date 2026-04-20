# Substrate P8 — Minimum-Viable Sleep Replay and Consolidation

**Status:** SHIPPED (2026-04-19). All 3 stages PASS. P6 dependency satisfied same session.
**Scope:** ~350 LOC + ~100 metric extractor
**Target version:** 0.5
**Gates:** null (not 1.0-gating, but validates the consolidation claim)
**Depends on:** P3a (episode binding), P6 (decay mechanism)
**Blocks:** nothing — but **activates** [memory_consolidation_practice.md](memory_consolidation_practice.md)
**Parent:** [substrate_binding_persistence.md](archive/substrate_binding_persistence.md)
**Related:** [memory_consolidation_practice.md](memory_consolidation_practice.md) (living practice doc that refines strategies AFTER P8 ships)

## Goal

Prove that offline replay during an explicit sleep phase strengthens rewarded associations — retrieval F1 improves on replayed probes without new input.

## Hypothesis

During a sleep phase, replaying the top-N rewarded episodes with Hebbian link weight updates produces measurable F1 improvement on retrieval probes for those episodes, without any new input being presented. The improvement exceeds a no-replay control by `control_mean + 2*control_std`.

## Non-goals (deliberately not ambitious)

- One replay strategy (top-N by cumulative reward)
- One scheduling rule (replay on explicit `sleep()` tool call)
- One measurable improvement (F1 delta on replayed probes)
- Everything else goes to [memory_consolidation_practice.md](memory_consolidation_practice.md)

## Dependencies (scaffolding audit)

| Surface | Status | Notes |
|---|---|---|
| Episode binding (P3a) | Shipped | Hebbian link formation on episode close |
| Hebbian edge weights | Shipped | `DependencyGraph.update_edge(weight=...)` |
| NAc reward bias (P2) | Shipped | Identifies top-N rewarded episodes |
| SCN phase ticks | Shipped | Sleep phase timing |
| Sleep tool (`ProcessingState.SLEEP`) | Shipped | Agent calls `sleep()` |
| Extinction/decay (P6) | Required | Baseline decay so replay improvement is measurable |
| Persistence (P3.5) | Shipped | Save/load to verify consolidation persists |

## Stages

### Stage 1 — mechanism + metric

**What's built:**
- Sleep replay engine: on `sleep()`, select top-N episodes by cumulative NAc reward, replay each (re-fire Hebbian update on all episode edges with a consolidation multiplier)
- Metric extractor: F1 on probe set before sleep vs after sleep
- No-replay control: same sleep duration, no replay, measure F1 delta
- Mechanism test on synthetic episodes

**Pass gate:** F1 improves on replayed probes. No-replay control shows no improvement (or slight decay from P6). Mechanism test passes on synthetic data.
**Tests:** `tests/substrate/test_p8_sleep_replay.py`

### Stage 2 — within-session replay

**What's built:**
- Realistic episode sequence → sleep → probe
- Varied replay-N (1, 5, 10 episodes)
- Control group comparison
- Persistence round-trip (consolidation survives save/load)

**Pass gate:** Replay F1 delta > 0. Beats no-replay control by `control_mean + 2*control_std`. Persistence stable. Mean + std across 10 seeds.

### Stage 3 — full sweep + pre-merge review

**What's built:**
- 10-seed sweep with varied episode counts and reward distributions
- Cross-session: populate → save → reload → sleep/replay → probe
- Pre-merge two-lens review

**Pass gate:** All Stage 2 criteria across 10 seeds. Cross-session replay works.
**Baseline:** No-replay control (same sleep, no Hebbian replay)
**Reviewers:** Executor + Architecture lenses

## Pass criteria (maps to 0.5 gate)

- F1 improves on replayed probes after sleep
- Beats no-replay control by `control_mean + 2*control_std`
- Consolidation persists across save/load
- Mean + std across 10 seeds

## Deferred follow-ups (practice doc territory)

- Alternative replay strategies (random, recency-weighted, surprise-weighted)
- Multi-sleep scheduling (incremental consolidation across sessions)
- Interference analysis (does replaying A degrade B?)
- Promotion rules (when does a replayed episode get promoted to LONG_TERM?)

All deferred items go to [memory_consolidation_practice.md](memory_consolidation_practice.md) when P8 ships.

## Load-bearing invariants

- **`replay_top_episodes` is NOT called automatically** — the agent loop or session manager must invoke it during the SLEEP processing state. There is no implicit trigger.
- **Episode ranking uses NAc `_reward_bias` + episode valence** — not episode recency. If NAc is None, falls back to valence-only ranking.
- **Consolidation multiplier amplifies `hebbian_delta`, not `hebbian_init`** — new edges during replay get standard init weight; only reinforcement of existing edges is amplified.
- **`_binding_graph` access uses getattr guard** — returns early if hippocampus is missing binding graph or config.
- **Depends on P6** — without decay, all edges are at max weight and replay produces no measurable improvement.
