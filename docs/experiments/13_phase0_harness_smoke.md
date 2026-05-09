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

- That cross-session transfer works. Single 5-turn run.
- That the harness produces useful Phase 0 measurements. We have telemetry; we don't yet have an analysis script. Roy harness work in 1.1+.

## Phase 0 sensor-encoding follow-up (2026-05-09, second commit)

The original write-up flagged the EC node-count gap as the next concrete work item. That gap is now closed. `SensorEncoder` in [src/maxim/similarity/encoder.py](../../src/maxim/similarity/encoder.py) hashes the current `{drive_name: value}` dict into a 384-dim embedding and routes it through `EC.pattern_complete_or_separate` with modality `"interoception"`. Wired into [agent_loop.py::propose_via_substrate](../../src/maxim/runtime/agent_loop.py) — fires once per substrate-primary tick before reading drives, fail-soft on encoder errors.

Re-ran the same smoke command at `--sim-max-turns 10`:

| Metric | Pre-encoding (5 turns) | Post-encoding (10 turns) |
|---|---|---|
| EC `node_count` (max) | 0 | 1 |
| EC modalities seen | (none) | `{"interoception": 1}` |
| Telemetry rows | 195 | 258 |
| Hunger drift | 0.000 → 0.650 | 0.000 → 0.831 |
| Thirst drift | (untracked) | 0.000 → 1.000 |

Phase 0 measurement gap is closed: substrate-primary now produces EC nodes, and the cluster-formation analysis the plan calls for has something to count.

### Surprising finding: smooth drive drift collapses to one cluster

Across 258 telemetry rows the run produced exactly **one** EC node, not the multiple-cluster trajectory I expected. Three things compose:

1. **Hash-based bases share structure across snapshots.** Each sensor contributes `(1-v)*basis_low + v*basis_high` where `basis_low`/`basis_high` are independent SHA-derived bases. Sensors that don't move (`arms.thermal`, `arms.pressure`, `head.thermal` stayed at 0.0 the entire run) contribute the *same* `basis_low` to every snapshot, dragging cosine similarity up.
2. **EC's `pattern_complete_or_separate` running-mean centroid update tracks the trajectory.** Each completion shifts the stored centroid toward the new embedding by `1/(n+1)`. After 250+ completions the centroid sits near the trajectory mean, and any single new sample is closer to the centroid than to either endpoint. Result: one cluster that smoothly drifts.
3. **The chosen pattern threshold (0.85) plus the embedding geometry put the trajectory inside a single completion basin.** Empirically `cos(zero_baseline, end_of_run) ≈ 0.69` (below threshold, *would* separate against a frozen prototype) — but `cos(centroid_at_step_n, snapshot_at_step_n+1) ≈ 0.99` because centroid tracking keeps the comparison local.

Net: the wiring works, but smooth continuous drift collapses to "one drifting concept" rather than producing discrete clusters. To get cluster differentiation against this embedding + EC, the substrate needs *discrete* state jumps — the agent successfully eating (hunger snaps to 0) or dropping a held entity (pressure goes to 0). The current cradle's only tool is `sense_food_source`, which doesn't reduce hunger, so the trajectory never leaves the smooth-drift regime.

This is consistent with the plan's framing of Phase 0 as a *measurement* phase: the finding "smooth drift produces one cluster, discrete jumps would produce more" is exactly the kind of result Phase 0 is supposed to surface.

### Phase 0+ refinement targets (not in this commit)

- **Disable EC centroid update for `"interoception"` modality** — preserve the original embedding as a fixed prototype so slow drift eventually crosses the threshold and creates a new cluster. Currently EC centroid update is intrinsic to `pattern_complete_or_separate`; modality-conditional behavior is the cleanest way to add this without changing text-path geometry.
- **Sensor-pattern bookkeeping vs concept formation.** "One drifting concept" may be the right biological model for proprioceptive baseline; "discrete cluster per significant drive transition" may be the right model for events worth memorizing. The split is unclear pre-Roy.
- **Threshold tuning against a Roy harness baseline.** 0.85 is a guess from offline geometry checks against the cradle's six sensors. Once we have multi-day persistent substrate runs, cluster purity + count over time will tell us whether 0.85, 0.90, or "centroid-update-disabled" is right.
- **Replace the drive-affinity heuristic with EC-similarity action selection.** Reserved per the plan — depends on knowing whether clusters are forming usefully first.

### Modality choice — `"interoception"` not `"sensor"`

Picked `"interoception"` because it matches `SensoryModality.INTEROCEPTION` in [agents/modality.py](../../src/maxim/agents/modality.py) and the cradle drives it's built for (hunger, thirst, core_temperature, arms.thermal, arms.pressure, head.thermal) are all interoceptive in the bio sense. A future `"sensor"` umbrella for exteroceptive surfaces (audio-as-pattern, raw vision-as-pattern) is a separate concern; keeping interoception distinct prevents the two cluster spaces from polluting each other when both encoders eventually run side by side. Documented in `SubstrateModality` Literal in [agents/modality.py](../../src/maxim/agents/modality.py) — adding `"sensor"` later is a one-line change.

### Files touched (this commit)

- `src/maxim/agents/modality.py` — `SubstrateModality` Literal extended with `"interoception"`
- `src/maxim/similarity/encoder.py` — new `SensorEncoder` + `_sensor_embed` (low/high basis interpolation, `_normalize_value`, min-delta gate)
- `src/maxim/runtime/agent_loop.py` — `sensor_encoder=` parameter on `propose_via_substrate`; encoder constructed once per loop in `run_agentic_loop` when `aut_mode == "substrate-primary"` and `memory_hub.ec` is reachable
- `tests/integration/test_phase0_harness.py` — `TestSensorEncodingIntoEC` class (4 tests)
- `docs/experiments/13_phase0_harness_smoke.md` — this section

## Files touched (original Phase 0 harness, prior commit)

- `src/maxim/simulation/arcs.py` — `cradle_prelinguistic` arc + exact-name resolution in `select_arc_for_goal`
- `src/maxim/prompts/motor_only_aut.py` — motor-only percept renderer (no narrative AUT prompt yet wired into the substrate-primary loop; the renderer is here for future hybrid modes)
- `src/maxim/simulation/substrate_telemetry.py` — JSONL writer
- `src/maxim/simulation/orchestrator.py` — `aut_mode` + `research_telemetry` parameters; telemetry construction
- `src/maxim/runtime/agent_loop.py` — `substrate_telemetry` parameter; `evaluate_failures()` call in `propose_via_substrate`
- `src/maxim/cli.py` — `--research` with `--aut-mode substrate-primary` skips `start_research_mode`
- `tests/integration/test_phase0_harness.py` — 13 tests pinning the harness contract
