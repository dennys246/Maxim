# The two learning channels a successful action writes

**Established:** 2026-09-12, R2 learned-bias experiment (dry-run + offline replication;
`scripts/survival_world/r2_learned_bias.py`, prereg `docs/experiments/r2_learned_bias_prereg.md`).

## The fact

When the agent performs a tool successfully and `record_outcome` runs, it writes **two independent
learning traces**, and they behave very differently:

| channel | what it is | keyed by | state-conditioned? | forms when |
|---|---|---|---|---|
| **causal link** | "tool X is a good action" | `event_signature="tool:X"` (cluster-INDEPENDENT) | **No — state-blind** | the tool SUCCEEDS (`nac.observe`) |
| **cluster reward bias** | "tool X is good in THIS drive/world state" | `(agent_id, cluster_id, "tool:X")` | **Yes — per cluster** | the reward routes to a cluster (drive-relief or the tool-success floor) |

`recommend_action` scores a tool from BOTH (plus the innate drive prior). Reasoning string shows
them separately, e.g. `causal_pos=0.89; cluster_bias[interoception]=+1.00`.

## Why it bites: behavioural credit-isolation is confounded

The causal link is **state-blind and forms on every success**, so in a single-corrective-action
world it **saturates the behavioural signal**: "eat is a good action" alone makes the agent eat, and
the drive-relief cluster bias — even maxed at `1.0` and even on the *same* cluster as training — adds
**nothing behaviourally** (the choice was already eat). Measured directly:

```
food 11: WITH clusters -> eat ;  WITHOUT clusters -> eat   (causal_pos=0.89 alone selects eat)
```

So a "did the agent learn to eat?" probe cannot tell drive-relief credit from generic tool-success —
they co-occur on every successful eat, and the state-blind causal link gets there first. The
NO-CREDIT ablation (suppress the cluster credit) does NOT suppress the causal link, so both arms
select eat → null regardless of whether the drive credit "works." (Predicted by the two-lens
review before the run; confirmed by the run.)

## How to wire an experiment that CAN isolate the drive-relief (cluster) credit

The cluster credit's real job is **state-conditioned, competitive selection** — "choose eat over
other things, *specifically* when hungry." Give it that job:

1. **Train competing successful actions** (eat AND e.g. mine/move) so eat's causal link is not
   dominant — "eat is good" can no longer carry the choice alone.
2. **Probe a real CHOICE** at a mild deficit where the innate prior is weak — the only thing that can
   tip eat over the equally-causal-linked alternatives is the cluster credit.
3. **Measure state-contingency** — P(eat | hungry) vs P(eat | satiated). The causal link is
   state-blind (flat across states); the cluster credit adds eat-preference in hungry clusters. The
   **arm × hunger-state interaction** is the isolated drive-relief effect.

## Traps

- A single-corrective-action world (only eat is trained) **cannot** behaviourally isolate the
  cluster credit — the causal link saturates. Don't try; you'll get a forced null.
- The tool-success **floor** also books to the interoception cluster (not just measured relief), so
  "cluster bias formed" ≠ "drive-relief formed" — enforce relief-only if the claim is drive-specific.
- `cluster_reward_bias` is consulted only when `current_clusters` is passed to `recommend_action`;
  `current_clusters=None` scores prior + causal link only. That difference IS the clean way to read
  the cluster channel's marginal effect (used by the R2 marginal probe).

## See also

DECISIONS.md 2026-09-12 (injected-signal lane); `docs/experiments/r2_learned_bias_prereg.md`;
`docs/experiments/r2_drive_premise_check.md`.
