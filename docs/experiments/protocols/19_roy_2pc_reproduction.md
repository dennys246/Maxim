# Reproduction — Roy-2pc positive-control on engineered-overlap fixture

**Companion:** [19_roy_2pc.md](../19_roy_2pc.md)
**Iteration spec:** [scenarios/roy/roy_2pc_iteration.yaml](../../../scenarios/roy/roy_2pc_iteration.yaml)
**Engineered fixture:** [scenarios/roy/roy_2pc_holdout.yaml](../../../scenarios/roy/roy_2pc_holdout.yaml)
**Predecessor:** [18_roy_2.md](../18_roy_2.md) (Roy-2pc reuses Roy-2's multi-arc priming) · [17_roy_1b.md](../17_roy_1b.md) (Roy-1b is the Roy-2pc A/B partner on the test-AUT-mode axis)

Roy-2pc is the first positive-control iteration in the Roy harness. **Two-variable diff vs Roy-2:** the held-out fixture changes from `roy_1_holdout.yaml` (matching/novel/unrelated, no food semantics) to `roy_2pc_holdout.yaml` (deliberately food/hunger/eating-semantic throughout) AND the test-AUT mode flips from `llm-primary` to `substrate-primary` so the `cluster_reward_bias` consumer (`recommend_action`) actually runs. Priming is identical to Roy-2 (multi-arc, 50 turns).

## Pre-registered diagnostic logic

| Outcome | Diagnosis |
|---|---|
| A > B > C on `sense_food_source` counts | Wire IS healthy + exploitable; behavioral inertness on Roy-1b/Roy-2 was a percept-overlap problem. Wire 1 escalation right for general-percept persona. |
| **A ≈ B ≈ C** | **Wire bug OR `min_confidence` gate filters even primed-cluster-matched proposals.** Roy-2c (`min_confidence=0.0` probe) becomes load-bearing before Wire 1. |
| A < C | Priming suppressed `sense_food_source` somehow. Unlikely; wire defect. |

The recorded run on 2026-05-13 produced **A ≈ B ≈ C** — all three arms produced byte-identical 2× FAILED `infant_humanoid_pick_up` distributions. See [19_roy_2pc.md § What this proves](../19_roy_2pc.md) for the H1 vs H2 hypothesis breakdown the result narrows but does not disambiguate.

## Prerequisites

Same as [17_roy_1b_reproduction.md § Prerequisites](17_roy_1b_reproduction.md):

- Maxim checkout at or after the Roy-2pc merge commit (or this branch: `feat/roy-2pc-positive-control`).
- `pip install -e .` (or `PYTHONPATH=src`).
- A working LLM provider for the `large` lane. Roy-2pc's test arms run `substrate-primary` so the AUT loop doesn't hit the lane — but the orchestrator narrator still calls the lane during priming + every test turn, so a working lane is required.
- Writable `~/.maxim/`.

## A. Pre-flight (3 min)

```bash
# 1. Spec parses?
maxim roy run scenarios/roy/roy_2pc_iteration.yaml --dry-run

# 2. Leader healthy?
curl -si --max-time 10 \
  -H "Authorization: Bearer $(awk '/api_key:/ {print $2}' ~/.config/maxim/peer.yml)" \
  "$(awk '/url:/ {print $2}' ~/.config/maxim/peer.yml)/models" | head -3
```

If `curl` returns HTTP 530, the leader's Cloudflare tunnel is cold; wait ~30s and retry. The Roy runner's pre-flight has its own retry budget but won't beat a hard tunnel-down.

## B. Roy-2pc (≈25-28 min wall)

```bash
# Kill any stale sims first (per CLAUDE.md "Kill stale sims before running tests").
pkill -f "maxim.*sim" 2>/dev/null

# Run, capturing both the human-readable log and the JSONL trace.
MAXIM_LOG_FILE=/tmp/roy_2pc_live.jsonl MAXIM_BACKEND_TRACE=1 \
  maxim roy run scenarios/roy/roy_2pc_iteration.yaml 2>&1 | tee /tmp/roy_2pc_run.log
```

Substrate-primary test arms take ~30s per turn × 10 turns each × 3 arms ≈ 15 min, plus ~10 min priming = ~25 min total. Per-arm durations are tightly clustered (Δ ≈ 6s in the recorded run) because the 30s-per-turn timeout dominates and is invariant to substrate state.

## Expected output

Numbers below are from the recorded run on 2026-05-13 against the leader at `https://maxim.dennyschaedig.com/v1` (qwen2.5-14b-instruct via cloudflared).

### Priming (50 turns, substrate-primary, ~10 min)

Identical multi-arc mix to Roy-2. final_session_id forwards to arm A.

| Stage | Arc | Duration_s (recorded) |
|---|---|---|
| act1_neonatal_a | cradle_prelinguistic | 184.5 |
| act1_neonatal_b | cradle_prelinguistic | 110.0 |
| act2_cradle_a | **cradle** | 101.6 |
| act2_cradle_b | **cradle** | 104.3 |
| act3_consolidation | cradle_prelinguistic | 105.2 |

Total priming: 605.5s (Roy-2 ran 609.5s — single-seed iterations reproduce priming wall to within 1%).

### Arms (10 turns each at substrate-primary, ~300s each)

| Arm | Substrate | system_prompt | turns | finish_reason |
|---|---|---|---|---|
| a | from_priming | neutral | 10 | cancel |
| b | blank | You are a hungry infant | 10 | cancel |
| c | blank | neutral | 10 | cancel |

### Headline pairwise diffs

The load-bearing reproducibility check: `cluster_reward_bias_l2 ≈ 2.4678` (10 keys = 6 priming + 4 test-phase pick_up), `b_vs_c.cluster_reward_bias_l2 ≈ 0.30` (4-pick_up stochastic floor).

```bash
jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias | {l2, key_count: (.top_deltas|length)}' \
   ~/.maxim/roy/roy-2pc/result.json

jq '.pairwise_diffs.b_vs_c.nac.cluster_reward_bias | {l2}' \
   ~/.maxim/roy/roy-2pc/result.json
```

### Per-arm test-phase tool distribution (the headline Roy-2pc signal)

```bash
for arm in a b c; do
  sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/roy-2pc/result.json)
  echo "=== arm $arm ($sid) ==="
  jq -c '{tool, success, error}' ~/.maxim/sim_reports/$sid/actions.jsonl
done
```

Pass criteria for the recorded result to reproduce:

1. **Every arm produces exactly 2× `infant_humanoid_pick_up` with `success: false, error: "Missing required input: object"`.** Byte-identical across arms.
2. **Zero `sense_food_source` calls in any arm.**
3. **`cluster_reward_bias_l2 ≈ 2.4678 (10 keys)`** — same shape as Roy-1b's substrate-primary-test cluster wire.

If (1) fails (any arm produces a different action distribution from another, or any arm produces a successful action), single-seed variance has crept in or the substrate-primary `recommend_action` fallback has changed. Investigate before drawing conclusions.

If (2) fails (arm A produces a non-zero `sense_food_source` count), the positive control HAS shown the wire is exploitable when percepts overlap — flip the diagnostic interpretation to "A > B > C → wire is healthy" and proceed to Wire 1 escalation without Roy-2c.

If (3) drifts > 5%, priming-side determinism regressed. Investigate before drawing conclusions.

### A/B against Roy-1b (the key positive-control diff)

```bash
# cluster_reward_bias structurally invariant across fixtures
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-1b/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2pc/result.json)
# Expected: both 2.4678 (within rounding)

# Per-arm tool distribution invariant across fixtures
for iter in roy-1b roy-2pc; do
  for arm in a b c; do
    sid=$(jq -r ".arms.$arm.session_id" ~/.maxim/roy/$iter/result.json)
    n=$(wc -l < ~/.maxim/sim_reports/$sid/actions.jsonl 2>/dev/null || echo 0)
    echo "$iter arm=$arm actions=$n"
  done
done
# Expected: 2 actions per arm in BOTH iterations (engineering percept-substrate overlap did not change tool selection)
```

## What to do if it fails

**Priming aborts on stage 1 within ~30s** — leader unreachable. Re-probe with `curl`; the Roy runner's pre-flight will report `outcome != ok` and abort cleanly.

**Stage 3 (act2_cradle_a) fails** — the `cradle` arc may have a different world_entities expectation. Check `bodies/infant_humanoid` + cradle entities are present.

**Every arm `turns=0, finish_reason=error`** — substrate-primary AUT loop crashed. Check JSONL for `Traceback`.

**Cluster_reward_bias differs from Roy-1b by > 5%** — priming-side non-determinism. Investigate.

**An arm produces non-`pick_up` actions** — the substrate-primary `recommend_action` fallback has changed since the recorded run. Useful for forward iterations but breaks the A/B comparison against Roy-1b.

**An arm produces `sense_food_source` calls** — POSITIVE result; the wire DID activate on the engineered percepts. Update [19_roy_2pc.md](../19_roy_2pc.md) headline ("A > B > C") and re-interpret the diagnostic outcome.

## Optional: tail the JSONL during the live run

```bash
tail -f /tmp/roy_2pc_live.jsonl | jq -c 'select(.e=="peer_backend_call") | {ts: .t, lane: .lane, agent: .agent_id, latency_ms: .latency_ms}'
```

Expected: ~20-30 `peer_backend_call` events total (priming narrator + arm-narrator only; substrate-primary AUT doesn't hit the lane).

## Cleanup

```bash
unset MAXIM_LOG_FILE MAXIM_BACKEND_TRACE
# Roy artifacts persist in ~/.maxim/roy/ and ~/.maxim/sim_reports/ — keep them for diff analysis.
```

## What changed vs Roy-2

Two variables changed: fixture + test-AUT-mode. Useful diff:

```bash
diff scenarios/roy/roy_2_iteration.yaml scenarios/roy/roy_2pc_iteration.yaml | grep -E '^[<>]\s+(aut_mode|fixture|name)'

# Wall time comparison (substrate-primary at test should be slower)
jq '.total_duration_s' ~/.maxim/roy/roy-2/result.json ~/.maxim/roy/roy-2pc/result.json
```

## What changed vs Roy-1b

Two variables changed: priming arc mix + fixture. Useful diff:

```bash
diff scenarios/roy/roy_1b_iteration.yaml scenarios/roy/roy_2pc_iteration.yaml | grep -E '^[<>]\s+(arc|fixture|name)'

# cluster_reward_bias structurally invariant
diff <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-1b/result.json) \
     <(jq '.pairwise_diffs.a_vs_b.nac.cluster_reward_bias.l2' ~/.maxim/roy/roy-2pc/result.json)
```

## Related docs

- [`19_roy_2pc.md`](../19_roy_2pc.md) — outcome doc with the full result table + H1/H2 hypothesis breakdown
- [`18_roy_2.md`](../18_roy_2.md) — multi-arc priming Roy-2pc reuses
- [`17_roy_1b.md`](../17_roy_1b.md) — substrate-primary at test on the original holdout (Roy-2pc's A/B partner)
- [`persona_convergence_crucible.md`](../../plans/deferred/persona_convergence_crucible.md) — three-arm methodology + Roy-2pc iteration log entry
