# Reproduction — Roy-2c `min_confidence=0.0` probe (H1 vs H2 disambiguator)

**Companion:** [20_roy_2c.md](../20_roy_2c.md)
**Iteration spec:** [scenarios/roy/roy_2c_iteration.yaml](../../../scenarios/roy/roy_2c_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml) (reused from Roy-2pc)
**A/B partner:** [19_roy_2pc.md](../19_roy_2pc.md)
**Owning release plan:** [release_0_9_1.md § Stage 0a](../../plans/archive/release_0_9_1.md)

Roy-2c is the H1-vs-H2 disambiguator for the Roy-2pc result. **Single-variable change vs Roy-2pc:** `MAXIM_NAC_MIN_CONFIDENCE=0.0` set in the runner environment. Same priming, same fixture, same arms.

## Pre-registered diagnostic logic

| Outcome | Diagnosis |
|---|---|
| A > B > C on `sense_food_source` counts | **H2 confirmed** — gate was the block. Lower threshold rescues the wire. |
| **A ≈ B ≈ C reproduces** | **H1 confirmed** — LinguisticEncoder → EC alignment is the block. Wire-A is the only architectural fix. |
| A < C | Unexpected; investigate before Wire-A design. |

The recorded run on 2026-05-13 produced **A ≈ B ≈ C** — all three arms produced byte-identical 5× FAILED `infant_humanoid_pick_up` distributions despite the gate at 0.0. See [20_roy_2c.md § What this proves](../20_roy_2c.md) for the structural breakdown including the disjoint-cluster observation.

## Prerequisites

Same as [19_roy_2pc_reproduction.md § Prerequisites](19_roy_2pc_reproduction.md), plus:

- **0.9.1 codebase or later.** `MAXIM_NAC_MIN_CONFIDENCE` env-var override at `agent_loop._resolve_min_confidence` is introduced in 0.9.1 (Stage 0a). Older checkouts ignore the env var; default `min_confidence=0.3` applies regardless.
- `tests/unit/test_substrate_min_confidence_env.py` should pass on the local checkout — confirms the env-var resolver works.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_2c_iteration.yaml --dry-run

# 2. Env-var resolver smoke test (0.9.1+ only)
python -c "
import os
os.environ['MAXIM_NAC_MIN_CONFIDENCE'] = '0.0'
from maxim.runtime.agent_loop import _resolve_min_confidence
assert _resolve_min_confidence(None) == 0.0
print('env-var resolver OK')
"

# 3. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

## B. Roy-2c (≈20-25 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# IMPORTANT: MAXIM_NAC_MIN_CONFIDENCE=0.0 is the load-bearing variable.
# It must be set in the runner's environment — the YAML does NOT carry it.
MAXIM_NAC_MIN_CONFIDENCE=0.0 \
MAXIM_LOG_FILE=/tmp/roy_2c_live.jsonl \
MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_2c_iteration.yaml 2>&1 | tee /tmp/roy_2c_run.log
```

Expect ~21 min wall — faster than Roy-2pc's 25 min because lower gate accepts proposals faster (less wall burned on the 30s timeout).

## Expected output

Numbers below are from the recorded run on 2026-05-13 against the leader at `https://maxim.dennyschaedig.com/v1` (qwen2.5-14b-instruct via cloudflared).

### Priming (50 turns, substrate-primary, ~10 min)

Identical multi-arc mix to Roy-2 / Roy-2pc. Total priming ~605s.

### Arms (10 turns each at substrate-primary, ~225-242s each)

| Arm | Substrate | system_prompt | turns | finish_reason | duration_s (recorded) |
|---|---|---|---|---|---|
| a | from_priming | neutral | 10 | cancel | 224.93 |
| b | blank | You are a hungry infant | 10 | cancel | 241.82 |
| c | blank | neutral | 10 | cancel | 238.57 |

### Headline pairwise diffs

```bash
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias | {l2, key_count: (.top_deltas|length)}' \
   ~/.maxim/roy/roy-2c/result.json
# Expected: l2 ~2.566, 10 keys

jq '.pairwise_diffs.b_vs_c.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2c/result.json
# Expected: ~0.76 (substrate-primary stochastic-cluster floor grows with gate=0.0)
```

### Per-arm test-phase tool distribution (the headline disambiguator)

```bash
for arm in a b c; do
  sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-2c/result.json)
  echo "=== arm $arm ($sid) ==="
  jq -c '{tool, success, error}' ~/.maxim/sim_reports/$sid/actions.jsonl
done
```

Pass criteria:

1. **Every arm produces exactly 5× `infant_humanoid_pick_up` with `success: false, error: "Missing required input: object"`.** Byte-identical across arms.
2. **Zero `sense_food_source` calls in any arm.**
3. **Per-arm action count is 5 (vs Roy-2pc's 2)** — confirms the gate was filtering 3 proposals per arm in Roy-2pc.

If (1) and (2) reproduce → H1 confirmed; the recorded diagnosis holds.

If (2) FAILS (arm A produces any `sense_food_source` count > 0 and arms B/C don't) → H2 confirmed instead; flip the interpretation to "gate was the block".

If (3) fails (per-arm action count stays at 2 like Roy-2pc) → the env-var override didn't take effect. Check `MAXIM_NAC_MIN_CONFIDENCE` was set in the runner's environment, not in the YAML. The agent_loop._resolve_min_confidence path reads `os.environ` at every call.

### Cluster top deltas (the structural diagnostic)

```bash
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.top_deltas | map({tool: (.key | split("")[2]), delta})' \
   ~/.maxim/roy/roy-2c/result.json
```

Expected: 6× `tool:sense_food_source` × +1.0 (UNCHANGED from Roy-2pc — priming clusters never updated during test) + 4× `tool:infant_humanoid_pick_up` × {±0.30, ±0.45} (NEW EC clusters from test phase).

The structural finding to look for: the four test-phase cluster updates are on **entirely new** EC cluster UUIDs, disjoint from the six priming UUIDs. If the engineered percepts had pattern-completed onto priming clusters, the +1.0 entries would shift to ±0.85 / ±1.15. They don't — the clusters are disjoint.

### Roy-2pc → Roy-2c direct A/B

```bash
# Cluster wire L2 should grow (more accepted proposals → larger cluster updates)
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2pc/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2c/result.json)

# Per-arm action count
for iter in roy-2pc roy-2c; do
  for arm in a b c; do
    sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/$iter/result.json)
    n=$(wc -l < ~/.maxim/sim_reports/$sid/actions.jsonl 2>/dev/null || echo 0)
    echo "$iter arm=$arm actions=$n"
  done
done
# Expected: Roy-2pc=2 per arm, Roy-2c=5 per arm
```

## What to do if it fails

**`maxim roy run` reports `gate` value at 0.3 in any log line** — the env var didn't propagate. Verify with `env | grep MAXIM_NAC_MIN_CONFIDENCE` before the run.

**Per-arm action count stays at 2** — the env var was set but the resolver isn't being called. Check `agent_loop._resolve_min_confidence` exists and is called from `propose_via_substrate`. Run the unit tests: `pytest tests/unit/test_substrate_min_confidence_env.py -v`.

**Arm A produces non-zero `sense_food_source` count** — POSITIVE H2 result; flip the interpretation. Update [20_roy_2c.md](../20_roy_2c.md) headline to "H2 confirmed" and revisit Wire-A's design (gate-tuning becomes a viable interim alternative; Wire-A's annotation may not be strictly necessary).

**`cluster_reward_bias_l2` ~2.4678 (matches Roy-2pc)** — the gate may have been filtering and zero proposals crossed even at 0.0. Investigate whether `recommend_action` returned None for ALL turns (no candidates above 0.0) — this is a different failure mode than H1 (encoder-alignment) or H2 (gate-filter). Suggests `recommend_action`'s candidate set is empty for these percepts.

**Pre-flight aborts on stage 1** — leader unreachable. Retry with `curl`.

## Optional: tail the JSONL during the live run

```bash
tail -f /tmp/roy_2c_live.jsonl | jq -c 'select(.e=="peer_backend_call") | {ts: .t, lane: .lane, agent: .agent_id}'
```

Expected: ~20-30 `peer_backend_call` events (priming narrator + arm narrators; substrate-primary AUT doesn't hit the lane).

## Cleanup

```bash
unset MAXIM_NAC_MIN_CONFIDENCE MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for diff analysis.
```

## What changed vs Roy-2pc

Single env-var change. The YAML is functionally identical except for the `name` field (`roy-2c` vs `roy-2pc`) and the description comment.

```bash
diff scenarios/roy/roy_2pc_iteration.yaml scenarios/roy/roy_2c_iteration.yaml | head -30
```

## Related docs

- [`20_roy_2c.md`](../20_roy_2c.md) — outcome doc with the full result table + H1 confirmation
- [`19_roy_2pc.md`](../19_roy_2pc.md) — Roy-2pc's A ≈ B ≈ C outcome Roy-2c disambiguates
- [`release_0_9_1.md`](../../plans/archive/release_0_9_1.md) — owning release plan; Stage 0a is the env-var work
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — three-arm methodology + Roy-2c iteration log entry
