# G4 — Substrate-primary cluster_id reward-feedback wire

**Date:** 2026-05-11
**Plan:** [grounded_language_acquisition.md § Phase 0 G4](../plans/grounded_language_acquisition.md)
**Status:** Wire shipped; unit-verified end-to-end; **empirically confirmed on a live Roy-0 run (2026-05-11 14:35-14:51)** — `cluster_reward_bias_l2 = 2.4587` on both A-vs-blank pairs, with the expected `sense_food_source` cluster updates at the `+1.0` per-key cap.
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

- The wire exists end-to-end. Substrate-primary tool outcomes populate `_cluster_reward_bias`.
- The dict serialises to `aut_nac.json` under the `cluster_reward_bias` JSON key.
- `substrate_diff` reads the dict and computes L2 + top deltas when both sides have it.
- Roy `result.json` carries `nac.cluster_reward_bias.{available, l2, top_deltas}` so operators can read the metric directly.

## Live Roy-0 re-measurement (empirical confirmation)

Re-ran `maxim roy run docs/plans/roy/roy_0_smoke.yaml` against the same healthy leader Roy-0 (2026-05-10) used. Wall: 926.2s (~15.4 min) — same shape as pre-G4. Priming completed 5/5 stages; all 3 arms completed at the warmup fixture's 3-percept exhaustion (`finish_reason=cancel`), unchanged from pre-G4.

**Headline:**

| Pair | `reward_bias_l2` | `cluster_reward_bias_l2` | `causal_link_count_delta` | `goal_reward_bias_l2` |
|---|---|---|---|---|
| **a_vs_b** | 0.0 | **2.4587** | +155 | 0.1918 |
| **a_vs_c** | 0.0 | **2.4587** | +155 | 0.1918 |
| b_vs_c | 0.0 | 0.2121 | 0 | 0.1918 |

**Top deltas (`a_vs_b` representative; `a_vs_c` identical shape):**

```
6× tool:sense_food_source  delta=+1.0  (at the per-key cap `max_cluster_reward_bias=1.0`)
2× tool:infant_humanoid_pick_up  delta=±0.15  (one positive, one negative — substrate is learning the affordance failed)
```

The 6 `sense_food_source` updates dominate the L2. Each comes from a distinct EC cluster id (the sensor encoder produces a fresh cluster every time drives shift past the min-delta gate, and arm A had a full 50-turn priming run); the substrate-primary path correctly accumulated cluster-keyed positive bias on the only tool it ever successfully invoked. The two `infant_humanoid_pick_up` entries differentiated because arm A's priming hit a failure case the blank arms didn't see — net signed evidence the wire propagates outcomes faithfully, not just magnitudes.

`b_vs_c` shows `cluster_reward_bias_l2 = 0.21` even though both arms started blank: each arm's 3 test-time turns produced a small number of `infant_humanoid_pick_up` updates that landed on slightly different stochastic cluster ids. Expected stochastic noise floor for blank-vs-blank under this fixture; the A-vs-blank ratio of **11.6×** (2.46 / 0.21) is the meaningful signal.

**What changed vs Roy-0 pre-G4:**

| Metric | Pre-G4 (2026-05-10) | Post-G4 (2026-05-11) |
|---|---|---|
| `cluster_reward_bias_l2` (a_vs_b) | n/a (field not serialised) | **2.4587** |
| `cluster_reward_bias.available` | `false` (field absent) | `true` |
| `causal_link_count_delta` (a_vs_b) | +133 | +155 |
| `reward_bias_l2` | 0.0 | 0.0 (expected — different code path) |
| `goal_reward_bias_l2` | 0.196 | 0.192 |
| Wall time | ~15 min | 15.4 min |

The `cluster_reward_bias` field is the headline metric the wire was built to make legible to `substrate_diff`. Pre-G4 it didn't exist in `aut_nac.json` at all (`substrate_diff` returned `available=false`); post-G4 it's serialised, populated, and differentiates arm A's primed substrate from blank arms at L2 ≈ 2.46.

## Two latent issues surfaced by the live run

These are minor and tracked as follow-ups on the same PRs:

1. **G3 preflight skipped under peer.yml.** `result.preflight = {"skipped": True, "reason": "MAXIM_LANE_LARGE_REMOTE_URL not set"}` even though `~/.config/maxim/peer.yml` carried a valid leader URL. Cause: `apply_peer_config_to_env` in [runtime/lane_backends.py:1073](../../src/maxim/runtime/lane_backends.py) only runs when lanes are first resolved — that happens after `_preflight_llm`. The preflight is conservative (skip when no URL), which protects local/cloud setups, but it means peer-with-peer.yml users get a no-op preflight. Real broken-leader failure modes are still caught when env vars are exported explicitly.
2. **`_format_summary` doesn't render `cluster_reward_bias`.** The `summary.md` only shows the old `reward_bias L2 = 0.0000` (correct, but misleading without the new metric next to it). Operators reading `summary.md` instead of `result.json` won't see the headline. Cosmetic fix.

Both follow-ups land on their respective PRs in this same session.

## What this still does NOT prove

- That the wire would still produce non-zero divergence on a held-out test fixture (Roy-0 reuses the priming arc). Roy-1 with a real holdout is the next test.
- That `min_confidence=0.3` is the right threshold for substrate-primary cold start. The current run had arm A exposing 6 distinct clusters all on `sense_food_source` — that's a single-tool monoculture, not the cluster diversity Phase 0 wants. Tuning question for the next experiment.
- That `reward_bias_l2` (the per-ATL-node recognition bias from `credit_node`) will become non-zero. That path is reaction-driven via `distribute_reward`, not tool-outcome-driven. G4 doesn't touch it; it stays 0 by design.

## Reproduction

See [protocols/15_g4_cluster_reward_wire_reproduction.md](protocols/15_g4_cluster_reward_wire_reproduction.md) for the runbook (unit suite + a real Roy-0 re-measurement against the leader).

## PR

[feat: G4 — substrate-primary cluster_id reward-feedback wire](https://github.com/dennys246/Maxim/pull/236) (branched from [PR #235](https://github.com/dennys246/Maxim/pull/235) for G3)
