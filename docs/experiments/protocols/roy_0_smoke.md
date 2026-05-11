# Roy-0: Smoke — Reproduction Protocol

**Status:** Active. Drafted from a clean end-to-end run against a live leader (qwen2.5-14b-instruct via cloudflared tunnel, 15 min wall clock). Hand-edited after auto-generation; do not regenerate without `--keep-edits`.
**Purpose:** Validate that the Roy harness (R1 curriculum runner + R2 substrate diff + R3 three-arm runner + R4 iteration-log generator) boots correctly end-to-end on a tiny methodology spec. Roy-0 is a test of the HARNESS, not a persona research result.
**Expected runtime:** ~15 minutes wall clock against a healthy 14B-class leader. Add ~5-10 min if priming pushes through pre-narration buffering on a cold leader.

## Background

Roy-0 reuses the cradle_prelinguistic arc for both priming (5 stages × 10 turns = 50 turns total) and the held-out test (the 3-percept `scenarios/cradle/warmup.yaml` fixture). The deliberate overlap means arm A's substrate should carry forward into the test scenario while arms B and C see the test fixture cold. Arm B's persona prompt is deliberately vague ("You are a hungry infant"); the methodology calls for carefully-shaped prompts later (Roy-1+), but Roy-0 only exercises the prompt-injection PATH, not its content.

This is the smallest configuration that makes every Roy code path fire at least once:
- R1 chains 5 priming stages → `resume_session` handoff is exercised 4 times
- R3 dispatches arms A/B/C with `substrate=from_priming` vs `blank`
- R2 computes pairwise diff over 3 pairs
- R4 generates this protocol + the iteration-log entry in `persona_convergence_crucible.md`

## Prerequisites

- Maxim checkout at or after the R5 commit (`feat(roy): Roy-0 smoke validation + harness fixes`).
- `pymaxim` importable (`PYTHONPATH=src` or editable install).
- A writable `~/.maxim/` (or `MAXIM_DATA_HOME=<path>` override).
- **A working LLM provider for the `large` lane.** Either:
  - An exported cloud API key (`ANTHROPIC_API_KEY` is cheapest), OR
  - A reachable leader at `MAXIM_LANE_LARGE_REMOTE_URL` serving `/v1/chat/completions` (the recorded run used a cloudflared-tunnelled leader; auth-gated 401 from `/v1/models` means alive — per CLAUDE.md §"Auth in health probes"), OR
  - A working local llama.cpp lane (`maxim doctor` reports `large` healthy).
- The iteration spec YAML resolvable from the recorded path:

      docs/plans/roy/roy_0_smoke.yaml

## Running the iteration

```bash
maxim roy run docs/plans/roy/roy_0_smoke.yaml --dry-run             # parse-only sanity
MAXIM_LOG_FILE=/tmp/roy_0_smoke.jsonl maxim roy run docs/plans/roy/roy_0_smoke.yaml
```

The JSONL is what makes failures debuggable. Without it, `dispatch_exhausted` warnings hit stderr only and the orchestrator's narrator silently falls back to "The story continues to unfold around you..." with `fallback: true` and no useful signal about WHY.

## Expected output

Numbers below are from the recorded run on 2026-05-11. Treat them as rough thresholds, not point estimates — operating point depends on leader latency, seed, and any pre-narration buffering on a cold leader.

**Arms — every arm should reach `finish_reason=cancel` after ~100s:**

| Arm | Substrate | system_prompt | turns | finish_reason |
|---|---|---|---|---|
| a | from_priming | neutral | 3 | cancel |
| b | blank | You are a hungry infant | 3 | cancel |
| c | blank | neutral | 3 | cancel |

`turns=3, finish_reason=cancel` reflects the 3-percept warmup fixture being exhausted — the bridge cancels when the percept source runs out, NOT a failure.

**Priming — 5 stages should complete in sequence, each chaining to the next via `resume_session`:**

- `act1_neonatal_a` (arc `cradle_prelinguistic`) × 10 turns
- `act1_neonatal_b` (arc `cradle_prelinguistic`) × 10 turns
- `act2_primary_circular_a` (arc `cradle_prelinguistic`) × 10 turns
- `act2_primary_circular_b` (arc `cradle_prelinguistic`) × 10 turns
- `act3_secondary_circular` (arc `cradle_prelinguistic`) × 10 turns

**Divergence thresholds (recorded run; tighten with seed variation once you have it):**

- **a_vs_b:** NAc reward_bias L2 ≈ `0.0000` · causal-link Δ ≈ `+133` · Hippocampus episodes Δ ≈ `+662`
- **a_vs_c:** NAc reward_bias L2 ≈ `0.0000` · causal-link Δ ≈ `+133` · Hippocampus episodes Δ ≈ `+662`
- **b_vs_c:** NAc reward_bias L2 ≈ `0.0000` · causal-link Δ ≈ `+0` · Hippocampus episodes Δ ≈ `+0`

The interesting pieces:

- **a_vs_blank ≈ +133 causal links, +662 episodes** is the harness's pass signal. Arm A inherited priming substrate; the R1→R3 chain threaded `resume_session` correctly through 5 priming handoffs into arm A's test session.
- **NAc reward_bias L2 = 0.0000 everywhere** is a substrate-primary maturity signal, not a harness failure: substrate-primary AUT emitted 0 action proposals across the whole run, so no reward signals fired, so reward_bias never populated. Once G4 (see iteration log) closes, this number will move.
- **b_vs_c = 0 across every axis** is what tells us prompt-injection differentiation doesn't reach substrate at substrate-primary's current maturity. "Hungry infant" vs "neutral" prompts produced identical substrate state because no actions fired.

## What we hit on the recorded run

- 23 `peer_backend_call` events (all status 200, mean 1.7s latency, 12,410 input / 2,125 output tokens, 0 cached)
- 10 `narrator_generation` events (every one `fallback: false` — real cradle scenes, no static-fallback grinding)
- 0 `dispatch_exhausted` (vs the prior dev-box-only run which had 3 in the first 145s)
- 0 substrate-primary action proposals across hundreds of `Loop step N, proposal=none` heartbeats — this is the G4 finding, see iteration log
- Total wall clock: 903s (~15 min); each priming stage averaged ~100s, each arm ~100s

## What to do if it fails

**Priming aborts on stage 1 within ~30s:** the leader is unreachable, the local-LLM lane is broken, or both. Probe the leader (`curl -si --max-time 5 $MAXIM_LANE_LARGE_REMOTE_URL/v1/models` — `HTTP/2 401` is alive per the auth-gated-probe rule). Check `MAXIM_LOG_FILE` for `dispatch_exhausted` events — if every call has `attempts[*].error: RuntimeError`, the local llama.cpp lane is the failure mode. **G3 (planned):** the runner doesn't yet pre-flight this; until it does, an unhealthy LLM lane wastes ~10 min before producing a useful error.

**Priming completes but arm A's substrate diff against C shows ~0 across the board:** the `resume_session` handoff broke. Inspect the priming `final_session_id` in `result.json` and verify arm A's `resumed_from` matches it. Then check that `~/.maxim/sim_reports/<priming-final>/aut_nac.json` is > 0 bytes — substrate snapshots write at session-end only, so a stage that timed out mid-turn leaves an empty snapshot.

**Every arm shows `turns=0, finish_reason=error`:** the AUT loop crashed or never booted. Check `MAXIM_LOG_FILE` for `Traceback` / `ERROR` events; the orchestrator's startup failures typically surface as `dispatch_exhausted` cascades followed by an exception.

**reward_bias L2 > 0.0 unexpectedly:** substrate-primary actually proposed actions and the reward chain fired — G4 has closed (or partially closed) since this protocol was written. Update the threshold table and move the entry from "interesting future signal" to "current expected value."

## What changed vs prior iterations

First Roy iteration — no prior to compare against. The "What changed" framing becomes useful starting at Roy-1.

## Related docs

- [`persona_convergence_crucible.md`](../../plans/persona_convergence_crucible.md) — three-arm methodology + Roy-0 iteration log entry with full finding breakdown
- [`grounded_language_acquisition.md`](../../plans/grounded_language_acquisition.md) — Roy long-horizon harness context + substrate-primary AUT mode (G4 lives here)
- `maxim.analysis.substrate_diff` — diff library (R2)
- `maxim.simulation.roy_runner` — three-arm iteration runner (R3)
- `maxim.analysis.roy_log` — protocol + iteration-log generator (R4)
- `maxim.simulation.curriculum_runner` — chained-stage substrate priming (R1)

<!-- generated by `maxim roy log roy-0-smoke`, then hand-edited with real R5 validation findings — pass `--keep-edits` on subsequent regenerations to preserve this content -->
