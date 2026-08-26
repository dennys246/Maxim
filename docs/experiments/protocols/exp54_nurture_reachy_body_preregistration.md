# Exp 54 — Nurture on the robot's own body (pre-registration)

**Status:** PRE-REGISTERED 2026-08-26, frozen before any run. **Roadmap 1.1.x item 15** —
the prerequisite for sharing the nursery-taught want (Oasis case study, 1.2) and the
first 1.1.x experiment.
**Lineage:** Exp 52 (the want is learned, on `bodies/infant_operant`) → Exp 53/53b (it
reads out on the physical Reachy, but only through an explicit δ map and the infant
body, because the learned keys are `tool:infant_operant_*` and the infant body declares
no `head_yaw`) → **this**: learn it on the robot's own body, so the files a nursery writes
are the files a user's robot reads, with nothing in between.

**Owner intent (recorded):** "make this learned orient memory the case study in sharing
through the Oasis" — a shareable want must be learned on a body whose action namespace
is the robot's. Exp 53b earned the transfer *claim*; Exp 54 earns the transfer *path*.

## The question

Does the Exp 52 result — an infant with a hunger drive and no orient drive learns to
turn toward the mother's voice because being fed relieves hunger — hold when the infant
*is* a Reachy Mini: the robot's own body component, its own four orient affordances
(normal and big turns, with the robot's own step sizes), its own tool names? And do the
files that nursery writes then drive the physical robot **through the production factory
path, with no apparatus adapter**, and get *consulted* under the plain `bodies/reachy_mini`
body a user would run?

## The body (front-gate: rides `extends` + `deep_merge`; zero new mechanism)

`src/maxim/_data/components/bodies/reachy_mini_infant.yaml`:

```yaml
component:
  name: reachy_mini_infant
  extends: bodies/reachy_mini
  category: bodies
  archetype: robot
  tags: [body, robot, reachy, infant, cradle, operant, developmental]
  description: |
    Exp 54 nursery body: the Reachy Mini's own body component — its four orient
    affordances with the robot's own head_yaw/azimuth self-effects (what the
    production ReachyOrientMotorBackend factory reads) and, load-bearing, the SAME
    entity name, so learned bias keys are tool:reachy_mini_turn_left etc. — with the
    innate azimuth centeredness drive REMOVED (drive: null, the Exp 52 rule: an
    intrinsic orient drive teaches orienting by itself) and the infant's hunger and
    thirst drives ADDED (infant_humanoid's specs) so the mother has something to relieve.
entity:
  name: reachy_mini            # NOT reachy_mini_infant — the tool prefix must match the production body
  sensors:
    azimuth:
      drive: null              # remove the innate centeredness drive (deep_merge replaces the dict with null)
    hunger:
      unit: ratio
      range: [0, 1]
      initial: 0.0
      drive: {drift_mode: entropic, drift_direction: up, drift_rate: 0.006, deprivation_threshold: 0.7, deprivation_pain: 0.3}
    thirst:
      unit: ratio
      range: [0, 1]
      initial: 0.0
      drive: {drift_mode: entropic, drift_direction: up, drift_rate: 0.008, deprivation_threshold: 0.6, deprivation_pain: 0.25}
```

`bodies/reachy_mini_infant_satiated.yaml` extends it with hunger/thirst `initial: 0.0`,
`drift_rate: 0.0` (the Exp 52 control body, verbatim pattern). Structural checks that
land with the body (unit tests, in the harness PR): the instantiated entity's `.name`
is `reachy_mini`; `generate_tools_for_entity` yields `reachy_mini_turn_left`,
`_turn_right`, `_turn_left_big`, `_turn_right_big`; `azimuth` carries no drive;
`hunger`/`thirst` carry the infant specs; the production `make_reachy_orient_factory`
returns a backend with `_deltas == {turn_left: +0.3, turn_right: −0.3, turn_left_big:
+0.9, turn_right_big: −0.9}` — **no explicit δ map anywhere in this experiment**.

**Declared differences from Exp 52 (S6):** the per-turn azimuth self-effect is the
robot's own — 0.17 (normal) or 0.50 (big) instead of the infant's 0.30 — and the
repertoire is four affordances, not two (the `turn_left,turn_right` whitelist substring-
matches the `_big` pair; that is the robot's real repertoire and stays). Stimulus band,
`oriented_threshold` 0.1, feed amount, shuffled order, relief credit, explore weight
1.5, 48 turns, 12 seeds per arm — all Exp 52 Phase B values, unchanged.

## Arms and phases

| arm | body | expectation |
|---|---|---|
| taught | `reachy_mini_infant` | learns to turn toward the voice |
| satiated | `reachy_mini_infant_satiated` | no learning (fed, never hungry) |
| no_feed | `reachy_mini_infant`, mother never feeds | no learning |

### Phase A — nursery (big-mac-mini; ~12 h for 36 runs)

`benchmark_cradle_mother.py --stimulus-order shuffled --credit relief --explore-weight 1.5`
with `--embodiment bodies/reachy_mini_infant` (satiated arm → the satiated body).
**Directedness is direction-only** (any leftward affordance for a stimulus on the left
counts), as in Exp 52. **Gate v3, unchanged constants** (`analyze_cradle_mother.py --gate
v3`): LEARNED late ≥ 0.65 and rise ≥ 0.15 from act 1 (ceiling clause S7); MOTHER-TAUGHT
≥ +0.20 over no_feed; HUNGER-NECESSARY ≥ +0.20 over satiated, satiated rise < 0.15 and
late ≤ no_feed + 0.20; L2 seed-spread apparatus gate; S3 assertions (no credit without
relief, satiated credited 0%). **Reported, not gated:** magnitude choice — the fraction
of big vs normal turns by |stimulus| bin (Exp 45c/d predict big-at-far, normal-at-near
if the credit resolves it; a coarse 3-bin representation may not).

**Stop rule A:** Phase A fails → Phases B/C do not run; the finding is recorded (the
robot's repertoire/step sizes change the learning problem — that is itself the
result), and item 15 gets a second pre-registration, not a retune.

### Phase B — readout on the physical robot, factory path (operator, ~1 h)

The Exp 53 harness with two additions made for this experiment: `--body-ref
bodies/reachy_mini_infant` and `--factory` (attach the backend through the production
`make_reachy_orient_factory`; **the `--delta` option is refused with `--factory`**). Seeds
= the first three taught, satiated, no_feed by declaration order, plus the weakest
taught seed as exploratory (as Exp 53). **Targets by declared procedure, not by number:**
before any robot record, sweep az ∈ [−1, 1] through each taught seed's loaded EC (the
amendment-1 procedure, now pre-registered); the gated targets are the centroids of the
two bins carrying the strongest left and right biases, clamped to the front hemisphere
(|az| ≤ 0.6), two magnitudes each; the two bins' boundaries are recorded and the
predicted wrong-way region (if any) is declared as exploratory placements *before*
Phase B runs. Gate I and Gate T exactly as Exp 53b — with one change that is now part
of the claim: **the step size is the agent's choice** (normal or big), so an overshoot
is a *policy* miss, not an apparatus one. The APPARATUS clause therefore narrows to
delivered-vs-chosen *direction* disagreement (sign-rule agreement ≥ 0.80, seed spread
below ceiling); a correctly-chosen big turn that overshoots counts against LEARNED-LIVE.
Primary explore 0.0; secondary explore 1.5 reported. Speech-gate floor 0.50 / 30 s. S3
SHA-256 before/after; nothing credits on the robot.

### Phase C — the user path (no motion, ~10 min)

The same taught files loaded under **plain `bodies/reachy_mini`** — the body a user's
`agentic_runtime` instantiates, innate azimuth drive present — through the harness's
`--body-ref bodies/reachy_mini`, Gate I only (probe, no motion). **Gate C:** for ≥ 2 of
3 taught seeds, at ≥ 80% of the gated placements the probe's `consulted_bias_by_modality.
audio` is non-zero and the chosen direction is correct. This is the sentence "a want
learned in the nursery is consulted on a user's robot with no remap", measured. Controls:
`consulted…audio == 0` at every placement.

## Outcome tree (decided now)

| A | B | C | verdict | consequence |
|---|---|---|---|---|
| pass | pass | pass | **EARNED — the shareable want** | item 15 DONE; the Oasis case study's "typed bundle" question becomes moot for this body (keys are the robot's); the 1.2 cross-unit experiment is pre-registered against these files |
| pass | pass | fail | learned and readable, not consulted under the innate drive | a seam finding (interoception vs audio routing under a drive-bearing azimuth — the deferred item named in `_read_exteroceptive_states`); item 15 stays open on Phase C only |
| pass | fail / APPARATUS | — | learned but not read out | recorded; Phase B re-run pre-registered separately, once |
| fail | — | — | the robot's repertoire changes the learning problem | recorded as the finding; second pre-registration, not a retune |

**Stop rules:** each phase runs once; a phase's failure ends the experiment at that
phase; one re-run of Phase B for a recorded apparatus fault only.

## What this experiment does NOT claim

Cross-**unit** transfer (a second robot — that is the 1.2 experiment); loudness or onset
salience; learning *on* the hardware; that the three-bin representation is adequate
(Phase B's exploratory placements will say where it is not); anything about the LLM
path. n = 12/arm in the nursery, 3/arm on the robot, one session each.

## Apparatus declarations (S1–S8)

S1 provenance assert + `executed_code_provenance` stamps (nursery via
`benchmark_cradle_mother.py`; robot via the harness), SDK == daemon; S3 counters + SHA;
S4 raw: `docs/experiments/data/54_phaseA_nursery.jsonl` + per-run provenance dir,
`54_agents/` (the taught/control pairs, SHA-manifested — the artifact the Oasis case study
ships), `54_phaseB_readout.jsonl`, `54_phaseC_userpath.jsonl`; S5 exposure-matched 48
turns; S6 as declared above; S7 ceiling clause; S8 the H1 pre-conditions.

## Amendment rule

Structural, pre-data amendments only, dated and appended below with what the author had
seen. Gates, margins, the declared target *procedure*, seeds and stop rules freeze at
the first Phase A record.

## Runbook

```bash
# body + harness PR first (unit tests for the body's structural checks land with it)
# Phase A (mini):
export PYTHONPATH="$PWD/src"
python scripts/benchmark_cradle_mother.py --embodiment bodies/reachy_mini_infant \
    --stimulus-order shuffled --credit relief --explore-weight 1.5 --seeds 12 \
    --out docs/experiments/data/54_phaseA_nursery.jsonl
python scripts/analyze_cradle_mother.py --gate v3 docs/experiments/data/54_phaseA_nursery.jsonl
# Phase B / C (robot):
python scripts/orient_backbone/exp53_cross_context_readout.py manifest --archive <phaseA workdir> \
    --out docs/experiments/data/54_agents_manifest.json --experiment 54
python scripts/orient_backbone/exp53_cross_context_readout.py sweep --manifest ... --out docs/experiments/data/54_targets.json
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --factory \
    --body-ref bodies/reachy_mini_infant --targets docs/experiments/data/54_targets.json --phase 1 --out docs/experiments/data/54_phaseB_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py run ... --phase 2 --condition primary ...
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --factory \
    --body-ref bodies/reachy_mini --phase 1 --out docs/experiments/data/54_phaseC_userpath.jsonl
```

## Sign-off (operator fills before Phase A)

1. Pre-registration read and frozen at commit `________` — ☐
2. Body YAML, arms, gate v3 constants, the target procedure, Phase C, stop rules accepted — ☐
3. Harness PR merged with the body's structural tests; dry run clean; amendments appended before Phase A — ☐
