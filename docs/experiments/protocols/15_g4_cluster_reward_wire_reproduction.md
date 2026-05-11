# Reproduction — G4 substrate-primary cluster_id reward-feedback wire

**Companion:** [15_g4_cluster_reward_wire.md](../15_g4_cluster_reward_wire.md)

## Prerequisites

- Maxim repo on the `feat/substrate-primary-cluster-reward-wire` branch (or `main` after merge).
- `pip install -e .` (or `PYTHONPATH=src`).
- For check (B): a healthy leader running `maxim --llm <profile>` reachable at `$MAXIM_LANE_LARGE_REMOTE_URL` with auth set. Same setup Roy-0 used.

## A. Unit-level reproduction (offline, deterministic)

The 6 new tests in `TestG4ClusterRewardWire` cover the wire end-to-end:

```bash
python -m pytest \
  tests/integration/test_substrate_primary_aut.py::TestG4ClusterRewardWire \
  -v
```

Expected: 6 passed in <2s. Each pins one piece of the chain:

| Test | What it pins |
|---|---|
| `test_propose_via_substrate_stashes_cluster_id_on_proposal` | `LLMProposal.cluster_id == encoder.encode_sensors(...)` after `propose_via_substrate`. |
| `test_record_outcome_updates_cluster_reward_bias` | `record_outcome(cluster_id=X, success=True)` → positive `_cluster_reward_bias`; `success=False` → reduces. |
| `test_record_outcome_skips_cluster_update_when_cluster_id_none` | `cluster_id=None` (LLM-primary) → no write to `_cluster_reward_bias`. |
| `test_nac_persistence_roundtrip_preserves_cluster_reward_bias` | `dump() → load_state()` preserves the dict, including `tool:use:dodge` signatures. |
| `test_pre_g4_snapshot_loads_with_empty_cluster_reward_bias` | Pre-G4 NAc JSON (no `cluster_reward_bias` field) loads cleanly. |
| `test_substrate_diff_surfaces_cluster_reward_bias_divergence` | `nac_diff(arm_a, blank_b)` returns non-zero L2 + identifies the differentiating tool. |

To run the broader substrate-primary + dispatch + diff coverage:

```bash
python -m pytest \
  tests/integration/test_substrate_primary_aut.py \
  tests/unit/test_tool_dispatch.py \
  tests/unit/test_nac_recommend_action.py \
  tests/unit/test_substrate_diff.py \
  -q
```

Expected: 68 passed in <3s.

## B. Live empirical reproduction (Roy-0 against a healthy leader)

This is the test the unit suite doesn't cover: does the wire fire in a real Roy iteration, and does `cluster_reward_bias_l2` come back non-zero on the A-vs-blank pair?

```bash
# 1. Confirm the leader is healthy.
maxim doctor --as peer

# 2. Run Roy-0 (≈15 min wall, ~50 LLM calls for priming + ~30 across 3 arms).
maxim roy run docs/plans/roy/roy_0_smoke.yaml

# 3. Read the result.
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias' \
   ~/.maxim/roy/roy-0-smoke/result.json
```

Expected `cluster_reward_bias` payload on the `a_vs_b` and `a_vs_c` pairs:

```json
{
  "available": true,
  "l2": 0.??,            // > 0 — arm A's cluster bias differs from blank
  "top_deltas": [
    {"key": "agentcluster-...tool:sense_food_source", "delta": 0.??}
  ]
}
```

Pass criteria (in priority order):

1. **`cluster_reward_bias.available == true` on every pair.** Pre-G4 this was `false` because the field didn't exist in `aut_nac.json`. After G4 the field is always serialised (even when empty).
2. **`cluster_reward_bias_l2 > 0` on `a_vs_b` and `a_vs_c`.** Arm A had priming exposure; arms B and C didn't. If A made any tool calls during priming that produced cluster updates, the L2 must be > 0.
3. **`top_deltas` lists at least one tool key** identifying which tool/cluster differentiates the arms.

If `cluster_reward_bias_l2 == 0` on a real run despite all three arms running cleanly, the diagnostic order is:

1. Check `actions.jsonl` in the priming session dir — were any tools actually called? If `len < 5` over the entire priming run, substrate-primary may be hitting the `min_confidence=0.3` gate at [nac.py:1300](../../src/maxim/decisions/nac.py). Per-path threshold tuning may be needed (out of G4 scope; tracked as the next item in the Roy-0 iteration log).
2. Check `aut_nac.json::cluster_reward_bias` directly in the priming session dir — is it populated? If the dict is empty after 100+ actions, the wire isn't firing somewhere (regression — check the G4 unit tests still pass).
3. Check `result.preflight.outcome` — if anything other than `"ok"` / `"auth_rejected"`, the preflight aborted and no real sim ran.

## C. Backward-compat check (loading pre-G4 NAc snapshots)

If you have a pre-G4 `aut_nac.json` lying around (e.g. from a Roy-0 run made before this merge):

```bash
python -c "
from maxim.decisions.nac import NAc, NACConfig
nac = NAc(NACConfig(temporal_window_seconds=60.0))
nac.load(path='/path/to/pre_g4/aut_nac.json')
print('cluster_reward_bias size:', len(nac._cluster_reward_bias))
print('Total observations:', nac._total_observations)
"
```

Expected: loads cleanly without errors, `_cluster_reward_bias` is empty (the field didn't exist pre-G4), other state loads normally. This is the regression guard for not breaking existing snapshots.

## D. Optional: tail the JSONL during a live Roy-0 run

To watch the wire fire in real time during priming:

```bash
# In one terminal, capture the JSONL.
export MAXIM_LOG_FILE=/tmp/g4_live.jsonl
export MAXIM_BACKEND_TRACE=1
maxim roy run docs/plans/roy/roy_0_smoke.yaml

# In another, watch for the cluster bias growing.
watch -n 5 'jq ".cluster_reward_bias | length" \
            ~/.maxim/sim_reports/$(ls -1t ~/.maxim/sim_reports | head -1)/aut_nac.json 2>/dev/null'
```

Expected: the count climbs from 0 toward N (where N is roughly the number of distinct `(cluster, tool)` pairs the substrate-primary AUT exercises). If it stays at 0 through priming, the proposer isn't firing — see the diagnostic order in (B) above.

## Cleanup

```bash
unset MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for diff analysis.
```
