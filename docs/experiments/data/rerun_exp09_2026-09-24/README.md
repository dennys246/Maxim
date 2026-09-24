# Exp 09 re-run, 2026-09-24 (after #870 / #871)

A re-run of behavioural-graduation **row 9** (narrative reflexes, [Exp 09](../../09_percept_reflex_poc.md))
after [#870](https://github.com/dennys246/Maxim/issues/870) (a firing records whether the body
responded) and [#871](https://github.com/dennys246/Maxim/issues/871) (sensor reflexes apply deltas)
changed what the narrative reflexes do. Protocol: [heartbeat runbook](../../protocols/heartbeat_rerun_runbook.md)
§Sim-Short 3. Verdict and walk entry: [behavioral_graduation_candidates.md](../../../plans/behavioral_graduation_candidates.md).

## Provenance

| | |
|---|---|
| Rig | big-mac-mini, worktree `.worktrees/heartbeat-exp09`, clean tree |
| Executed commit | `b9b9bca425f676912f196382e48b2706596be0ac` (`main` with #872) |
| Interpreter | imports the worktree's `src/` (checked before the run) |
| Model | `mistral-7b-instruct-v0.2.Q4_K_M.gguf`, `llm.profile mistral-7b`, `llm.n_ctx 8192` (set through `maxim config`; matches the 2026-08-18 heartbeat) |
| Box | quiet: Paper server, Minecraft bridge and ollama stopped for the run |
| Command | the runbook's §Sim-Short 3 dragon-cave sim, `--embodiment bodies/base_humanoid --interactive false --sim-max-turns 8`, `MAXIM_SUBSTRATE_PATH=1` |
| Session | `20260924_095451`: 5 turns, then `planning_failed` (mistral-7b proposed unregistered tools 4× against a limit of 3, D13) |

## Files (copied verbatim; logs gzipped)

`20260924_095451/`: `report.json`, `aut_{atl,ec,hippocampus,nac,scn}.json`, `actions.jsonl.gz`,
`bio_telemetry.jsonl.gz`, `run_log.jsonl.gz` (the `MAXIM_LOG_FILE` JSONL), `console.out.gz`.
SHA-256 of the uncompressed originals, checked against the rig after copying:
`report.json` `c3a9a82b…d2e2`; `run_log.jsonl` `54676163…e2ca`.

## Reflex events (from `run_log.jsonl`, events `sim_reflex` / `sim_sem_damage` / `sim_sensor`)

Every firing's `outcome` was `acted` (the enrichment line reads `N/N reflex(es) acted`); none
`failed`, none `suppressed`.

| t (s)* | reflex | effective intensity | effect |
|---|---|---|---|
| 33.7 | attack_flinch | 0.150 | torso.integrity 1.00 → 0.85 |
| 33.7 | fire_burn | 0.161 | torso.integrity 0.85 → 0.69 |
| 33.7 | startle | 0.200 | head.awareness 0.90 → 0.70 |
| 33.7 | environment_cold | 0.050 | stamina 1.00 → 0.95 |
| 65.1 | attack_flinch | 0.133 | torso.integrity 0.69 → 0.56 |
| 65.1 | startle | 0.154 | head.awareness 0.70 → 0.55 |
| 65.1 | environment_cold | 0.038 | stamina 0.95 → 0.91 |
| 165.9 | attack_flinch | 0.115 | torso.integrity 0.56 → 0.44 |
| 165.9 | fire_burn | 0.148 | torso.integrity 0.44 → 0.29 |
| 216.7 | attack_flinch | 0.107 | torso.integrity 0.29 → 0.19 |
| 216.7 | fire_burn | 0.132 | torso.integrity 0.19 → 0.05 |

\* Seconds since the run log's first record. The sim's own `elapsed_s` clock reads 29.1 / 60.6 /
161.4 / 212.1 for the four ticks. Values are rounded to 2 dp here; the true values are in the log
(for example torso 0.556 after the third tick, awareness 0.546, stamina 0.912).

**The modulators reproduce the formula to 3 dp.** For each firing, intensity = raw × 1/(1 + 0.3·n)
[habituation] × (1 + 0.5·(1 − torso integrity)) [sensitization], with `_HABITUATION_K = 0.3` and
`_SENSITIZATION_S = 0.5` from `embodiment/reflex.py`. The integrity is the live torso value when the
spec is evaluated, so within one tick `fire_burn` sees the torso that `attack_flinch` has just
damaged:

| reflex | n | integrity before | predicted | logged |
|---|---|---|---|---|
| attack_flinch | 0 | 1.00 | 0.150 | 0.150 |
| attack_flinch | 1 | 0.69 | 0.133 | 0.133 |
| attack_flinch | 2 | 0.556 | 0.115 | 0.115 |
| attack_flinch | 3 | 0.29 | 0.107 | 0.107 |
| fire_burn | 0 | 0.85 | 0.161 | 0.161 |
| fire_burn | 1 | 0.44 | 0.148 | 0.148 |
| fire_burn | 2 | 0.19 | 0.132 | 0.132 |

## Hypotheses

| | Result | Evidence |
|---|---|---|
| H1 reflexes fire | **PASS** | 11 `sim_reflex` events across 4 turns |
| H2 body-part targeting | **NOT MET as written** (routing shown) | Exp 09's criterion is "damage logs targeting at least 2 different components (torso + legs)". Every damage log here targets the torso. startle (`head.awareness`) and environment_cold (`stamina`) routed correctly, but those are sensor changes, not damage. `impact_brace` → legs never fired: no percept the agent received contained one of its keywords. The only "slam" is in the simulation-goal text, which goes to the orchestrator and never reaches the agent's bio-enrichment, where reflexes are evaluated. The original run's H2 also cited startle as `awareness=0.00`; per #871's analysis the old path wrote that to an orphan root key. |
| H3 reflex source on pain | **PARTIAL** | The damage carries `source=reflex_attack` / `source=reflex_fire`, and NAc holds negative reflex-keyed percept valences (`aut_nac.json` → `percept_valences`: `…reflex_attack` −0.038, `…reflex_fire` −0.041). The 4 pain REACTION events read `from pain_detector:external_signal`, so the log does not show the reflex context carried on the published pain signal itself, which is what H3 asks for. |
| H4 habituation | **PASS** | attack_flinch 0.150 → 0.133 → 0.115 → 0.107; startle 0.200 → 0.154; environment_cold 0.050 → 0.038 |
| H5 sensitization | **PASS** | fire_burn's first firing is 0.161 > raw 0.15 on a torso already at 0.85; every value above matches the formula |
| H6 no auto-damage | **PASS** | 0 `auto_attack` mentions |
| H7 telemetry | **PASS** | `sim_enrichment` reflex lines with per-firing outcomes |

**What this says about 2026-08-18.** That heartbeat logged 7 reflex events and 0 `SEM_DAMAGE`,
and attributed the missing damage to the narrator. But on `main` as of that run (`1d79787d`), reflexes
already dispatched their own tool (`ReflexRegistry.evaluate` → `execute_tool`), not through the
narrator. The dispatcher's result was ignored, and an unwired tool returned `None` at DEBUG. So
that result is **consistent with** the #870 defect: firings logged, response never applied. This
re-run cannot prove it retroactively. What it does show is that on the fixed code, every logged
firing applied its damage (11 firings, 7 `SEM_DAMAGE`; the 4 sensor firings move sensors instead).

## Caveats

- **5 of 8 turns.** The run ended at `planning_failed` (D13, a narrator/AUT tool-format failure),
  the same weak-model class as August. The reflex path does not depend on it.
- **Legs untested.** A full H2 needs a narration containing an `impact_brace` keyword.
- `report.json`'s `pain_events_count` is 0 because it counts only `subsystem == "PAIN"`. There are 0
  of those and 4 `REACTION` pain events, one per tick. That is probably a reporting gap. This log
  cannot show whether the PainBus itself published a `PAIN` record.
- **H4/H5 measure the damage the body takes.** `docs/plans/deferred/reflex_layering.md` records why
  that is the wrong layer and that route 3 would change this metric deliberately.
