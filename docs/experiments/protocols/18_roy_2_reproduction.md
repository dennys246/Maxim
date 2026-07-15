# Reproduction — Roy-2 multi-arc priming on held-out fixture

**Companion:** [18_roy_2.md](../18_roy_2.md)
**Iteration spec:** [scenarios/roy/roy_2_iteration.yaml](../../../scenarios/roy/roy_2_iteration.yaml)
**Held-out fixture:** [scenarios/roy/roy_1_holdout.yaml](../../../scenarios/roy/roy_1_holdout.yaml)
**Predecessor:** [17_roy_1b.md](../17_roy_1b.md) (Roy-2 tests path (a) of Roy-1b's three-pointer methodology refinement)

Roy-2 is the third methodology-validation iteration. **Single-variable change vs Roy-1a:** priming arc mix widens from 5 × `cradle_prelinguistic` to 2 × `cradle_prelinguistic` (neonatal) + 2 × `cradle` (linguistic-narrated) + 1 × `cradle_prelinguistic` (consolidation), same 50-turn budget. The held-out fixture, test-time AUT mode (`llm-primary`), and arm shapes are byte-identical to Roy-1a. Roy-2 tests whether widening priming arc *narration* widens the EC cluster *vocabulary* — Roy-1b's path (a) refinement.

## Prerequisites

Same as [16_roy_1a_reproduction.md § Prerequisites](16_roy_1a_reproduction.md):

- Maxim checkout at or after the Roy-2 merge commit (or this branch: `feat/roy-2-multi-arc-priming`).
- `pip install -e .` (or `PYTHONPATH=src`).
- A working LLM provider for the `large` lane (leader, cloud API, or local llama.cpp). Roy-2 uses `aut_mode: llm-primary` at test so test arms hit the lane on every turn that doesn't pattern-match a deterministic tool path; the priming side runs `substrate-primary` and only the narrator hits the lane there.
- Writable `~/.maxim/`.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_2_iteration.yaml --dry-run

# 2. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -5
```

Roy-2's fixture is identical to Roy-1a/1b's; the fixture-activation check in [16_roy_1a_reproduction.md § A.3](16_roy_1a_reproduction.md) covers Roy-2 too — no need to re-run if Roy-1a's pre-flight cleared.

## B. Roy-2 (≈15-18 min wall)

```bash
# Kill any stale sims first (per CLAUDE.md "Kill stale sims before running tests").
pkill -f "maxim.*sim" 2>/dev/null

# Run, capturing both the human-readable log and the JSONL trace.
MAXIM_LOG_FILE=/tmp/roy_2_live.jsonl MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_2_iteration.yaml 2>&1 | tee /tmp/roy_2_run.log
```

Expect ~10 min on priming (5 stages × ~100s each, stage 1 longer due to cold-start) + ~5 min across the three llm-primary test arms.

## Expected output

Numbers below are from the recorded run on 2026-05-12 against the leader at `https://maxim.dennyschaedig.com/v1` (qwen2.5-14b-instruct via cloudflared). Treat them as rough thresholds, not point estimates — leader latency, seed, and cold-load timing all shift the operating point.

### Priming (50 turns, substrate-primary, ~10 min)

| Stage | Arc | Duration_s (recorded) |
|---|---|---|
| act1_neonatal_a | cradle_prelinguistic | 187.2 |
| act1_neonatal_b | cradle_prelinguistic | 109.0 |
| act2_cradle_a | **cradle** | 103.3 |
| act2_cradle_b | **cradle** | 105.5 |
| act3_consolidation | cradle_prelinguistic | 104.5 |

`resume_session` chains through all 5 stages. `final_session_id` forwards to arm A. Stage 1 is the cold-start outlier (~80s longer than later stages) — typical for substrate-primary's first priming session.

### Arms (10 turns each at llm-primary, ~80-100s each)

| Arm | Substrate | system_prompt | turns | finish_reason |
|---|---|---|---|---|
| a | from_priming | neutral | 10 | cancel |
| b | blank | You are a hungry infant | 10 | cancel |
| c | blank | neutral | 10 | cancel |

`turns=10, finish_reason=cancel` reflects the 10-percept fixture being exhausted — not a failure.

### Headline pairwise diffs

See [18_roy_2.md § Headline pairwise diffs](../18_roy_2.md) for the full table. Quick read:

```bash
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias' ~/.maxim/roy/roy-2/result.json
jq '.pairwise_diffs.a_vs_b.hippocampus' ~/.maxim/roy/roy-2/result.json
jq '.pairwise_diffs.a_vs_b.nac | {reward_bias_l2, causal_link_count_delta, goal_reward_bias_l2}' \
   ~/.maxim/roy/roy-2/result.json
```

### Roy-1a vs Roy-2 cluster wire reproducibility

The load-bearing reproducibility check: cluster_reward_bias_l2 should be within 1% of Roy-1a's 2.4495 (multi-arc priming did NOT widen the cluster vocabulary in the recorded run — the same six `sense_food_source` cluster keys appear, just on different EC cluster UUIDs).

```bash
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-1a/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2/result.json)
```

### Per-arm test-phase tool distribution (the headline Roy-2 behavioral signal)

```bash
for arm in a b c; do
  sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-2/result.json)
  echo "=== arm $arm ($sid) ==="
  jq -r '.tool' ~/.maxim/sim_reports/$sid/actions.jsonl | sort | uniq -c | sort -rn
done
```

Pass criteria (the load-bearing positive-result signal):

1. **Arm A's tool distribution diverges from arm C's tool distribution** (both run with neutral system prompts; only substrate differs). In the recorded run, arm A used `sense` (3) + `pick_up` variants (3); arm C used `infant_humanoid_look` (5) + `infant_humanoid_listen` (1) + `sense_tools` (1) with zero `sense` or `pick_up` calls. The exact tool counts shift seed-to-seed but the *family* divergence should reproduce.
2. **`b_vs_c.cluster_reward_bias_l2 = 0.0`** (llm-primary at test produces zero cluster noise floor; both blank arms write nothing to the cluster wire).
3. **`a_vs_b.cluster_reward_bias_l2 ≈ 2.45`** with all top_deltas keyed to `tool:sense_food_source` (priming wire reproduces).

If (1) fails (arm A's tool distribution is identical to arm C's), the prompt-mediated behavioral signal Roy-2 reports has not reproduced — investigate whether the LLM proposer's substrate-context wiring (salience-modulated WMS, recall hints in the prompt) has regressed. If (3) fails by > 5%, priming-side determinism has drifted — investigate before drawing methodology conclusions.

### Valence_KS reproducibility

Roy-2's `valence_KS = 0.291 (p=0.023)` is the first Roy iteration with a healthy-sample valence_KS clearing α=0.05. Roy-1a missed (p=0.402); Roy-1b cleared sample-driven (1-episode blank). Reproducing Roy-2's clean read with healthy blank-arm episode counts (>20) is a methodology win:

```bash
jq '.pairwise_diffs.a_vs_b.hippocampus.valence' ~/.maxim/roy/roy-2/result.json
```

Expect `episode_count_b > 20`, `ks_pvalue < 0.05`. If `episode_count_b < 5`, the blank-arm test path didn't capture enough episodes — investigate whether the held-out fixture's percepts produced any salient-enough events for hippocampus to write episodes.

## What to do if it fails

**Priming aborts on stage 1 within ~30s** — leader unreachable. Re-probe with `curl` (per pre-flight above).

**Stage 1 cradle_prelinguistic completes but stage 3 (act2_cradle_a) fails** — the `cradle` arc may have a different world_entities expectation than `cradle_prelinguistic`. Check that `bodies/infant_humanoid` + the cradle entities (`items/cradle_*.yaml`) are present in `~/.maxim/components/` or the bundled `_data/components/`.

**Every arm `turns=0, finish_reason=error`** — AUT loop crashed. Check JSONL (`/tmp/roy_2_live.jsonl`) for `Traceback`.

**`cluster_reward_bias_l2` differs from Roy-1a's 2.4495 by > 5%** — priming-side non-determinism crept in. Investigate before drawing A/B conclusions.

**Arm A's tool distribution matches arm C's exactly** — the prompt-mediated behavioral signal didn't reproduce. Substrate carryover into the LLM proposer prompt may have regressed. Check that arm A's session resumed from the priming `final_session_id` (`.arms.a.resumed_from` in result.json) AND that `~/.maxim/sim_reports/<priming-final>/aut_nac.json` is > 0 bytes.

**`valence_KS` is 0 or p=1.0** — hippocampus didn't capture differential episodes between primed and blank arms. Either the blank arms' test path produced 0 episodes (fixture percepts not salient enough — unlikely given Roy-1a/1b worked) or the valence-annotation pipeline regressed.

## What changed vs Roy-1a

Single-variable: priming arc mix widens from 5 × `cradle_prelinguistic` to 2 × `cradle_prelinguistic` + 2 × `cradle` + 1 × `cradle_prelinguistic`. Useful diff:

```bash
diff scenarios/roy/roy_1a_iteration.yaml scenarios/roy/roy_2_iteration.yaml | grep -E '^[<>]\s+(arc|name|description)'

# Cluster wire reproducibility check (should be < 1% diff)
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-1a/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2/result.json)

# Salience_KS shrinkage check (Roy-1a 0.879 → Roy-2 0.529 — more priming diversity = smaller novelty gap)
jq '.pairwise_diffs.a_vs_b.hippocampus.salience' ~/.maxim/roy/roy-1a/result.json
jq '.pairwise_diffs.a_vs_b.hippocampus.salience' ~/.maxim/roy/roy-2/result.json
```

## Optional: tail the JSONL during the live run

```bash
# In another terminal while the iteration runs:
tail -f /tmp/roy_2_live.jsonl | jq -c 'select(.e=="peer_backend_call") | {ts: .t, lane: .lane, agent: .agent_id, latency_ms: .latency_ms}'
```

Expected: ~20-30 `peer_backend_call` events total (~7-10 per arm × 3 arms). If counts are dramatically off, the AUT proposer is either over-generating (loop) or under-generating (stalled).

## Cleanup

```bash
unset MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for diff analysis.
```

## Related docs

- [`18_roy_2.md`](../18_roy_2.md) — outcome doc with the full result table + interpretation
- [`17_roy_1b.md`](../17_roy_1b.md) — predecessor with the three-pointer methodology refinement Roy-2 tests path (a) of
- [`16_roy_1a.md`](../16_roy_1a.md) — first Roy iteration with the same llm-primary test mode + held-out fixture (Roy-2's A/B partner for the priming-arc-mix variable)
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — three-arm methodology + Roy-2 iteration log entry
- [`16_roy_1a_reproduction.md`](16_roy_1a_reproduction.md) — Roy-1a protocol Roy-2 mirrors
