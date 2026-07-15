# Reproduction — Roy-1b substrate-primary on held-out fixture

**Companion:** [17_roy_1b.md](../17_roy_1b.md)
**Iteration spec:** [scenarios/roy/roy_1b_iteration.yaml](../../../scenarios/roy/roy_1b_iteration.yaml)
**Held-out fixture:** [scenarios/roy/roy_1_holdout.yaml](../../../scenarios/roy/roy_1_holdout.yaml)
**A/B partner:** [16_roy_1a.md](../16_roy_1a.md)

Roy-1b is the second methodology-validation iteration after Roy-1a. **Single-variable change vs Roy-1a:** test-time AUT mode flips from `llm-primary` to `substrate-primary`. Priming, held-out fixture, and arms are byte-identical. The diff against Roy-1a is a clean A/B of "does substrate-primary AUT exploit the cluster_reward_bias the priming wire wrote?"

## Prerequisites

Same as [16_roy_1a_reproduction.md § Prerequisites](16_roy_1a_reproduction.md):

- Maxim checkout at or after the Roy-1b merge commit (or this branch: `feat/roy-1a-llm-primary-holdout`).
- `pip install -e .` (or `PYTHONPATH=src`).
- A working LLM provider for the `large` lane (leader, cloud API, or local llama.cpp). Roy-1b's test arms use substrate-primary so the AUT loop doesn't hit the LLM — but the orchestrator narrator still calls the LLM, so a working lane is required.
- Writable `~/.maxim/`.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_1b_iteration.yaml --dry-run

# 2. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(yq .api_key ~/.config/maxim/peer.yml)" \
  "$(yq .url ~/.config/maxim/peer.yml)/models" | head -5
```

Roy-1b's fixture is identical to Roy-1a's; the fixture-activation check in [16_roy_1a_reproduction.md § A.3](16_roy_1a_reproduction.md) covers Roy-1b's fixture too — no need to re-run if Roy-1a's pre-flight cleared.

## B. Roy-1b (≈15-18 min wall)

```bash
# Kill any stale sims first.
pkill -f "maxim.*sim" 2>/dev/null

# Run, capturing both the human-readable log and the JSONL trace.
MAXIM_LOG_FILE=/tmp/roy_1b_live.jsonl MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_1b_iteration.yaml 2>&1 | tee /tmp/roy_1b_run.log
```

Substrate-primary test arms are typically faster than llm-primary because the AUT proposer doesn't hit the leader for every turn — only narration does. Expect arms in the 30-60s range each, vs Roy-1a's 60-85s.

## Expected output

Numbers below are from the recorded run on 2026-05-12 against the same leader Roy-1a used. Refer to [17_roy_1b.md § Result](../17_roy_1b.md) for the headline table.

### Priming (50 turns, substrate-primary)

Identical to Roy-1a + Roy-0 priming. 5 stages × 10 turns of `cradle_prelinguistic`. final_session_id forwards to arm A.

### Arms (10 turns each, substrate-primary)

| Arm | Substrate | system_prompt | turns | finish_reason |
|---|---|---|---|---|
| a | from_priming | neutral | 10 | cancel |
| b | blank | You are a hungry infant | 10 | cancel |
| c | blank | neutral | 10 | cancel |

### A/B against Roy-1a

The key cells of the comparison table from [17_roy_1b.md](../17_roy_1b.md):

```bash
# Roy-1a vs Roy-1b cluster_reward_bias (priming wire writes through identically)
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' \
        ~/.maxim/roy/roy-1a/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' \
        ~/.maxim/roy/roy-1b/result.json)

# Roy-1a vs Roy-1b causal_link_count_delta (priming carryover should be identical)
diff <(jq '.pairwise_diffs.a_vs_b.nac.causal_link_count_delta' \
        ~/.maxim/roy/roy-1a/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.causal_link_count_delta' \
        ~/.maxim/roy/roy-1b/result.json)
```

### Per-arm tool distribution (the key Roy-1b signal)

```bash
for arm in a b c; do
  sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-1b/result.json)
  echo "=== arm $arm ($sid) ==="
  jq -r '.tool' ~/.maxim/sim_reports/$sid/actions.jsonl | sort | uniq -c | sort -rn
done
```

Pass criteria (the key Roy-1b answer):

1. **`sense_food_source` invocations are HIGHER in arm A than arms B/C.** This is the behavioral signal that the cluster_reward_bias the priming wire wrote IS exploitable when substrate-primary's `recommend_action` consumes it.
2. **Test-time cluster updates land on `sense_food_source` (or another priming-reinforced tool) in arm A.** Arm A's test-time updates should pile onto priming-acquired clusters or land on new clusters for held-out percepts.

If both pass, Roy-1b confirms the priming-bias-exploitation hypothesis Roy-1a left open. If `sense_food_source` counts are roughly equal across arms, the bias is structurally present but the consumer doesn't read it under held-out percepts — pointing at either (a) the `min_confidence=0.3` gate at cold start (Roy-0 finding) or (b) the EC cluster representation not firing on held-out percept regimes.

## What to do if it fails

**Priming aborts within ~30s** — leader unreachable. Re-probe with `curl`.

**Every arm `turns=0, finish=error`** — AUT loop crashed. Check JSONL for `Traceback`.

**`cluster_reward_bias_l2` differs from Roy-1a's 2.4495 by > 5%** — priming-side non-determinism crept in (changes to the cradle arc, RNG seeding, or EC clustering). Investigate before drawing A/B conclusions.

**Arm A's tool distribution at test time matches arms B/C** — substrate-primary at test isn't exploiting the priming bias. See pass criteria above.

## What changed vs Roy-1a

Single-variable: `aut_mode` at test time flips from `llm-primary` to `substrate-primary`. Useful diff commands:

```bash
diff scenarios/roy/roy_1a_iteration.yaml scenarios/roy/roy_1b_iteration.yaml | head -10

# Wall time comparison (substrate-primary should be faster — no LLM in AUT loop)
jq '.total_duration_s' ~/.maxim/roy/roy-1a/result.json ~/.maxim/roy/roy-1b/result.json
```

## Cleanup

```bash
unset MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ — keep them for diff analysis.
```

## Related docs

- [`17_roy_1b.md`](../17_roy_1b.md) — outcome doc with the full result table + interpretation
- [`16_roy_1a.md`](../16_roy_1a.md) — A/B partner
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — three-arm methodology
- [`16_roy_1a_reproduction.md`](16_roy_1a_reproduction.md) — Roy-1a protocol Roy-1b mirrors
