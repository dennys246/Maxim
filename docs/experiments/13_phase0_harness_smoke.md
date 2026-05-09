# Phase 0 harness smoke test (cradle_prelinguistic + substrate-primary)

**Date:** 2026-05-09
**Phase:** 0 of [grounded_language_acquisition.md](../plans/grounded_language_acquisition.md)
**Status:** Harness clears the success criterion; no validation claim made.

## Command

```bash
maxim --sim cradle_prelinguistic --embodiment bodies/infant_humanoid \
      --aut-mode substrate-primary --research --interactive false \
      --sim-max-turns 5
```

## Result

Run 2026-05-09 10:25-10:28 (139.5s wall, 5 turns):

| Metric | Value |
|---|---|
| Actions | 38 |
| Causal links formed | 61 |
| Tool usage | `sense_food_source: 38 calls (100% success)` |
| Hunger drive | 0.000 → 0.650 over the run |
| Substrate telemetry rows | 195 |

The telemetry JSONL captured per-tick drives, NAc reward_bias, EC node count, and the active substrate proposal. Final-tick proposal reasoning: `causal_pos=0.99; drive:hunger(0.65) →food`.

## Surprising findings

### Substrate-primary mode owns its own clock

The first run produced 0 actions because nothing was advancing drive drift. The LLM-primary path drives the embodiment via `EmbodimentPerceptSource.next_percept()` which calls `evaluate_failures()` (and through it `tick_vital_drift`). With the LLM submit branch gated off, no other code in `run_agentic_loop` polls the embodiment, so drives sat at their YAML defaults (hunger=0.0) forever.

Fix landed in `propose_via_substrate`: call `executor.embodiment.evaluate_failures()` once per substrate tick. Mirrors what `EmbodimentPerceptSource` does on the LLM-primary path. Without this call, substrate-primary mode is a no-op when starting from baseline drives — the substrate has no opinion because no drive ever crosses 0.5.

This is structural: substrate-primary mode does not borrow ANY runtime side-effects from the LLM submit path. Anything the LLM-primary path implicitly relied on (drive drift, percept polling, telemetry hooks) has to be re-wired for substrate-primary. We've handled drive drift; future Phase 0+ work will likely surface more of these.

### Pre-linguistic ≠ silent orchestrator

The current run still has an LLM-driven orchestrator emitting English narration percepts to the AUT (e.g. *"You wake to the gentle rustle of leaves overhead..."*). The substrate-primary AUT ignores them — it acts only on its own drive state. But the percepts are still generated, still hit the AUT's text channel, and would be visible to any future component that reads them.

Phase 0 is "no English on the AUT side"; the orchestrator side staying LLM-driven is intentional per the plan, but the harness should eventually drop those narration percepts entirely (they're noise the substrate ignores). Tracked as a follow-up; out of scope for this commit.

### `sense_food_source` is a scene-tool win

The infant body did NOT directly cause the action. `sense_food_source` is registered by the `cradle_food` scene entity activated in the arc's `exploration` phase. The substrate's hunger affinity table contains "food" → matched the tool name → won out over body affordances on the cold-start drive-relevance heuristic.

That's a clean Phase -1 substrate behavior. It's also a slightly different quirk than the test suite caught: the unit test had `pick_up_food` win on `pick_up` substring + hunger affinity; this run had `sense_food_source` win on `food` substring + hunger affinity. Same mechanism, different match. Worth noting because it shows the affinity table working in the wild on an arc the test suite doesn't cover.

### `GenerativeCampaignResult.turns_completed` AttributeError

End-of-run warning: `Generative campaign failed: 'GenerativeCampaignResult' object has no attribute 'turns_completed'`. Pre-existing bug in the generative campaign runner, surfaces on every termination path. Doesn't affect telemetry or actions; flagged here for future cleanup.

## What this does NOT prove

- That the substrate forms persistent EC clusters tied to repeating sensorimotor patterns. EC `node_count` was 0 throughout the run; concept formation didn't fire. That's expected — the substrate's encoding path runs on text percepts via `LinguisticEncoder`, and substrate-primary mode bypasses the percept-text pipeline entirely. Phase 0 validation will need an additional encoding entry point that treats sensor readings as concept inputs. Tracked.
- That cross-session transfer works. Single 5-turn run.
- That the harness produces useful Phase 0 measurements. We have telemetry; we don't yet have an analysis script. Roy harness work in 1.1+.

## Files touched

- `src/maxim/simulation/arcs.py` — `cradle_prelinguistic` arc + exact-name resolution in `select_arc_for_goal`
- `src/maxim/prompts/motor_only_aut.py` — motor-only percept renderer (no narrative AUT prompt yet wired into the substrate-primary loop; the renderer is here for future hybrid modes)
- `src/maxim/simulation/substrate_telemetry.py` — JSONL writer
- `src/maxim/simulation/orchestrator.py` — `aut_mode` + `research_telemetry` parameters; telemetry construction
- `src/maxim/runtime/agent_loop.py` — `substrate_telemetry` parameter; `evaluate_failures()` call in `propose_via_substrate`
- `src/maxim/cli.py` — `--research` with `--aut-mode substrate-primary` skips `start_research_mode`
- `tests/integration/test_phase0_harness.py` — 13 tests pinning the harness contract
