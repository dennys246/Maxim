# G4 — Substrate-primary cluster_id reward-feedback wire

**Date:** 2026-05-11
**Plan:** [grounded_language_acquisition.md § Phase 0 G4](../plans/grounded_language_acquisition.md)
**Status:** Wire shipped; unit-verified end-to-end. Empirical re-measurement on a live Roy-0 run still pending.
**Companion:** [G3 — Roy preflight probe](14_g3_roy_preflight_probe.md) (paired PRs; G4 branched from G3).

## What was caught

Roy-0 (2026-05-10, PRs #233/#234) ran 15 min end-to-end against a healthy leader with `aut_mode=substrate-primary` and reported:

- 142 actions fired (all `sense_food_source`)
- 2001 NAc observations recorded
- 133 causal links accumulated
- 662 episodes captured
- **But:** `NAc.reward_bias` was empty (0 keys); `cluster_reward_bias` field wasn't serialised at all; pairwise `reward_bias_l2 = 0.0000` across all 3 arm pairs

Three parallel investigations (static gate trace + commit forensics + persisted-state inspection) converged on the same root cause: commit `6d0e4a7` (Track 2 of the substrate-primary series) shipped cluster-keyed action *selection* but **explicitly deferred** the cluster-keyed reward *update*. From the commit message:

> Reward update wiring (cluster_id-aware `record_outcome`) is deliberately out of Track 2's scope — the API exists and is unit-covered, but the agent_loop's `_record_outcome` chain doesn't yet plumb cluster_id.

`cluster_id` was captured at proposal time and lost before outcome. Consequence: `NAc._cluster_reward_bias` stayed empty across every session; cluster-keyed selection had no learned bias to use; recommend_action couldn't differentiate one substrate state from another.

## What shipped

Five changes form the wire end-to-end:

### 1. LLMProposal carries cluster_id

[src/maxim/agents/llm_types.py](../../src/maxim/agents/llm_types.py): added `cluster_id: str | None = None` as the last field on the frozen dataclass (CC3-compatible non-breaking optional field addition). LLM-primary proposals leave it `None`; substrate-primary proposals stash the active EC cluster.

### 2. propose_via_substrate stashes the cluster

[src/maxim/runtime/agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py): the existing `cluster_id` local (already passed to `recommend_action(current_cluster_id=...)`) is now also passed to `LLMProposal(..., cluster_id=cluster_id)` so it survives until outcome recording.

### 3. record_outcome calls update_cluster_reward

[src/maxim/runtime/tool_dispatch.py::record_outcome](../../src/maxim/runtime/tool_dispatch.py): added `cluster_id: str | None = None` kwarg. When non-empty + NAc wired, calls `nac.update_cluster_reward(agent_id, cluster_id, tool_signature, ±1.0)`. No-op for `cluster_id=None` (the LLM-primary path stays bit-identical). Wrapped in best-effort try/except so a cluster-learning failure can't crash the agent loop. `execute_parallel_actions` carries the same kwarg through to its inner `record_outcome` calls.

### 4. All 6 _record_outcome call sites + execute_parallel pass cluster_id through

Six direct `_record_outcome` sites + one `execute_parallel_actions` site in [agent_loop.py](../../src/maxim/runtime/agent_loop.py) extract the cluster from the in-scope proposal envelope:

- 5 sites use `getattr(ctrl.pending_proposal, "cluster_id", None)` (ctrl.pending_proposal may be None at error sites)
- 2 sites use `proposal.cluster_id` (local var in scope, guaranteed non-None)

### 5. Persistence + substrate_diff

- [src/maxim/decisions/nac.py](../../src/maxim/decisions/nac.py): `dump()` serialises `_cluster_reward_bias` under the new top-level `cluster_reward_bias` JSON key. Keys join the `(agent_id, cluster_id, tool_signature)` triple with `\x1f` (ASCII unit separator) so tool signatures containing `:` (e.g. `tool:use:dodge`) round-trip cleanly. `load_state()` reads the field; pre-G4 snapshots without the field load to an empty dict (backward compatible).
- [src/maxim/analysis/substrate_diff.py](../../src/maxim/analysis/substrate_diff.py): `NacDiff` carries `cluster_reward_bias_{available,l2,top_deltas}`. `nac_diff` populates them when both sides serialised the field. `substrate_diff_to_json` surfaces them under `nac.cluster_reward_bias` in `result.json` so Roy iterations can read the metric without re-running the diff.

## Result

| Metric | Value |
|---|---|
| New tests | 6 (in `TestG4ClusterRewardWire`) |
| Total substrate-primary tests | 22 (all passing) |
| Full fast suite | 6484 passed, 15 skipped, 0 failures attributable to G4 |
| Pre-existing flake | `test_context_index.py::test_similar_text_found` (unrelated, documented) |

The 6 new tests cover the chain end-to-end:

1. `propose_via_substrate` stashes cluster_id on the returned LLMProposal.
2. `record_outcome(cluster_id=X, success=True)` increments `_cluster_reward_bias[(agent, X, sig)]`.
3. `record_outcome(cluster_id=X, success=False)` decrements it.
4. `record_outcome(cluster_id=None, ...)` is a no-op (LLM-primary path stays clean).
5. Persistence roundtrip: `dump() → load_state()` preserves `_cluster_reward_bias` including tool signatures containing `:`.
6. Pre-G4 snapshot (no `cluster_reward_bias` field) loads cleanly with an empty dict.
7. `substrate_diff` surfaces non-zero L2 + the differentiating tool key when arm A learned and arm B is blank.

## What this DOES prove

- The wire exists end-to-end. Substrate-primary tool outcomes now populate `_cluster_reward_bias`.
- The dict serialises to `aut_nac.json` under the `cluster_reward_bias` JSON key.
- `substrate_diff` reads the dict and computes L2 + top deltas when both sides have it.
- Roy `result.json` will carry `nac.cluster_reward_bias.{available, l2, top_deltas}` so operators can read the metric directly.

## What this does NOT prove

- That a fresh Roy-0 run will actually populate the dict with substantive entries at sim-time. The wire fires per unit test, but the next gate is empirical: with `min_confidence=0.3` on `NAc.recommend_action` and only a few cluster updates per substrate-primary tick, the proposer may still hit the score-threshold gate before cluster bias accumulates. That's a tuning question (path-specific `min_confidence` for substrate-primary?), answered by a real Roy-0 re-measurement.
- That `reward_bias_l2` (the per-node ATL recognition bias, distinct from cluster_reward_bias) will become non-zero. That dict populates only via `credit_node` from reaction-driven `distribute_reward`, not from tool outcomes — G4 doesn't touch that path.

## Reproduction

See [protocols/15_g4_cluster_reward_wire_reproduction.md](protocols/15_g4_cluster_reward_wire_reproduction.md) for the runbook (unit suite + a real Roy-0 re-measurement against the leader).

## PR

[feat: G4 — substrate-primary cluster_id reward-feedback wire](https://github.com/dennys246/Maxim/pull/236) (branched from [PR #235](https://github.com/dennys246/Maxim/pull/235) for G3)
