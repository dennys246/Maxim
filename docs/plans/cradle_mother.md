# Cradle: the mother-taught OPERANT orient experiment

**Status:** SUPERSEDED / DORMANT (2026-07-22). The embodied `cradle_mother` sim built from this plan **measured at chance** — the sim's machinery (LLM narrator, confidence gate, 22-tool competition, turn caps) drowns the operant signal. The operant-teaching claim was validated instead on the **clean scripted substrate**: [`scripts/orient_substrate/4-7`](../../scripts/orient_substrate/) + experiment [46](../experiments/46_operant_orient_creche.md) (taught 0.90 vs chance; a crèche pools partial learners). The embodied wiring ships as a **demo only** (dormancy markers in `simulation/cradle_mother.py` + the arc). Resurrecting it needs the credit-on-progress root-cause fix ([deferred/credit_on_progress_not_execution.md](deferred/credit_on_progress_not_execution.md)). The design below is kept for that resurrection.

## Why the earlier design was scrapped (the honest part)

The first design gave the infant an intrinsic **centeredness drive** (azimuth homeostatic set_point 0) and measured whether it self-orients as a mother's *guide* fades. A pre-overnight review found the fatal flaw: **the intrinsic drive teaches orienting all by itself** — probes 1+2 (`scripts/orient_substrate/1_motor_credit_probe.py`, `2_full_path_probe.py`) show contingent **1.000** vs ~0.50 chance with *no mother anywhere*. So the drive was present in every arm and ablated by none; a rising curve wouldn't prove the mother taught anything. (Also: no exploration weight → inert run; the "scaffold_only" arm didn't disable the surface the policy actually learns on; the metric was fed-gated so the control read 0 by construction.)

## The claim (now honest)

A hungry infant with **no intrinsic orient drive** learns to orient toward a sound **purely because a mother feeds it** when its own turn moved toward the sound (operant conditioning). Remove the mother → it never learns. No LLM in the action path (substrate-primary).

## The mechanism

- **`NAc.credit_operant_reward(agent_id, reward)`** (+ `set_pending_operant_action`): an EXTERNAL, caregiver-caused drive relief reinforces the recipient's OWN recent action on `_cluster_reward_bias` (the action-SELECTION surface `recommend_action` reads) — NOT `distribute_reward`/`credit_node` (the recognition surface). This is the mirror image of `_drive_potential_diff`, which deliberately excludes caregiver `target_effect` so a caregiver's policy isn't credited by the recipient's relief. Validated in isolation by **`scripts/orient_substrate/3_operant_feed_probe.py`**: contingent **1.000**, yoked/none **~0.48** (`none` at chance = the "mother necessary" proof); with the tool-success floor ON, all arms collapse to chance (→ operant-only mode below).
- **`bodies/infant_operant`**: extends `infant_humanoid`, `azimuth: {drive: null}` (needs the spec.py "null drive = no drive" fix). Perceives direction, no innate reason to orient.
- **`MAXIM_OPERANT_ONLY_CREDIT`** (tool_dispatch): remembers each action for the mother's credit and suppresses the uniform tool-success cluster-reward floor (which otherwise saturates the cap and drowns the operant signal — probe 3's `tool_floor` arm).
- **Exteroceptive encoding** (agent_loop `_read_exteroceptive_states/_ranges`): substrate-primary now encodes the azimuth SENSOR into the cluster (perception ≠ drive), kept out of `current_drives` so affinity stays drive-only. Load-bearing: without it the driveless-azimuth infant is blind to left-vs-right.

## The per-turn loop

Each turn the reactive mother acts, then the infant takes its substrate-primary turn:
1. **Reward prior progress** — reads the azimuth the infant's *prior* turn left; if it moved TOWARD last turn's sound (`|prev_stimulus| − |az_prior| > 0`), feed (hunger relief) + `credit_operant_reward` on the infant's pending action (operant shaping). `progress` is logged in BOTH arms → the directedness metric is arm-independent.
2. **Place the stimulus** — world-set azimuth to the mother's direction this turn.
3. **(No guide)** — pure shaping. Physically turning the head and then crediting the infant's own action for the mother's guide would be dishonest; the "fade" is the EMERGENT learning curve.
4. **Speak** — motherese via the substrate-safe inject.

## Arms + metric (arc `cradle_mother`, 4 time-bins)

- **taught** — mother shapes (feeds + credits toward-turns). `MAXIM_OPERANT_ONLY_CREDIT=1`.
- **no_feed** (control) — mother places the sound but never feeds/credits (`MAXIM_CRADLE_MOTHER_DISABLE_CARE=1`). With no drive, no teacher → chance.
- Both set `MAXIM_SIM_SUBSTRATE_EXPLORE_BONUS_WEIGHT=1.5` (B1 — the infant must explore turns to bootstrap).

**Metric:** per time-bin, **directedness** = fraction of turns the infant's own turn moved TOWARD the sound (`progress > 0`). Verdict (`analyze_cradle_mother.py`): **LEARNED** (taught late ≥ 0.65 and rose ≥ 0.15) AND **MOTHER-TAUGHT** (taught late ≥ no_feed late + 0.20).

## Status / open question

Wiring smoke-verified: `credited=True`, infant moves toward the sound, agent_id alignment (`memory_hub.agent_id`) holds. **Open:** is the turn count enough for operant credit (slow) to build a directional policy? The probe needed 3000 ticks but had continuous azimuth (many clusters); the sim has ~6 discrete stimulus directions. Calibrating a 60-turn taught run; bump turns if directedness doesn't rise (operant is slow — longer sims pre-approved).

## Build pieces (all shipped)
1. `NAc.credit_operant_reward` + `set_pending_operant_action` + probe 3 + 8 unit tests — `3b478e6d`.
2. `MAXIM_OPERANT_ONLY_CREDIT` operant-only mode + conftest scrub + 4 tests — `acd6360b`.
3. `bodies/infant_operant` + `drive:null` support + 2 tests — `5072abe0`.
4. Mother shaper + credit wiring (runner→campaign→orchestrator) + exteroceptive encoding + operant arc + harness/analyzer + tests — `03ad8a0c`.
5. Calibrate → two-lens review → the mac-mini behavioral run.

---

## Archived: the original fading-scaffold design (confounded — kept for the record)

The infant had an intrinsic azimuth centeredness drive; a reactive mother guided its head toward her with a `guide_strength` that faded 1.0→0.5→0.0→0.0 across 4 acts, feeding it when oriented; the metric was the fraction it self-oriented per act, across 3 arms (taught / drive_only / scaffold_only). Scrapped because the intrinsic drive teaches orienting in every arm (no arm isolates the mother), plus three blocking harness bugs (no explore weight → inert; `MAXIM_NAC_REWARD_BIAS_DISABLED` doesn't gate the cluster-reward path; fed-gated metric reads 0 in the no-feed arm). See the memory `reference_cradle_mother_experiment_design_flaw.md`.