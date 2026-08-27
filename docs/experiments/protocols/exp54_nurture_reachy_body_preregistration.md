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
    --arms taught,satiated,no_feed --trials 12 --seed-base 42 \
    --stimulus-order shuffled --credit relief --explore-weight 1.5 --model mistral-7b \
    --workdir ~/exp54/phaseA --out docs/experiments/data/54_phaseA_nursery.jsonl
python scripts/analyze_cradle_mother.py --gate v3 docs/experiments/data/54_phaseA_nursery.jsonl
# Phase B / C (robot):
python scripts/orient_backbone/exp53_cross_context_readout.py manifest --archive <phaseA workdir> \
    --experiment 54 --phase-a-records docs/experiments/data/54_phaseA_nursery.jsonl \
    --out docs/experiments/data/54_agents_manifest.json
python scripts/orient_backbone/exp53_cross_context_readout.py sweep \
    --manifest docs/experiments/data/54_agents_manifest.json --out docs/experiments/data/54_targets.json
# → declare the gated + exploratory placements in the results doc BEFORE Phase B
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --factory \
    --body-ref bodies/reachy_mini_infant --targets docs/experiments/data/54_targets.json --phase 1 --out docs/experiments/data/54_phaseB_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py run ... --phase 2 --condition primary ...
python scripts/orient_backbone/exp53_cross_context_readout.py run ... --phase 2 --condition secondary ...
python scripts/orient_backbone/exp53_cross_context_readout.py verdict --records docs/experiments/data/54_phaseB_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --factory \
    --body-ref bodies/reachy_mini --gate C --targets docs/experiments/data/54_targets.json \
    --manifest docs/experiments/data/54_agents_manifest.json --phase 1 \
    --out docs/experiments/data/54_phaseC_userpath.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py verdict --gate C --records docs/experiments/data/54_phaseC_userpath.jsonl
```

## Amendments

**Amendment 1 — 2026-08-26, PRE-DATA, structural (harness build + offline dry run; no
Phase A record exists).** Six items, none touching a gate, margin, seed count or stop rule:

1. **The target procedure, operationalised** (the phrase "centroids of the two bins carrying
   the strongest left and right biases … two magnitudes each" needed a procedure a script can
   execute). `sweep`: az ∈ [−1, 1] step 0.1 through each gated taught seed's loaded EC (a
   fresh load per value, explore 0, nothing saved). *Bins* = maximal runs of consecutive az
   values completing into the same `audio` cluster; a bin's left/right *strength* = the max
   persisted bias among its `turn_left*` / `turn_right*` keys (the `_big` keys count); the
   LEFT (RIGHT) bin = the bin with the strongest left (right) strength. *Eligible* left targets
   = az values that lie in the LEFT bin of a majority of the three seeds, with az < 0 and
   |az| ≤ 0.6 (right: az > 0). *Two magnitudes per direction* = the grid value nearest the
   eligible centroid and its neighbour one step further from centre (one step closer if the
   outer neighbour is not eligible; grid ties resolve toward centre). *Exploratory* = the grid
   value nearest the centroid of the predicted wrong-way region (values where a majority of
   seeds' frozen probe picks the wrong direction with |learned_margin| > 0.11; az = 0 has no
   direction and is excluded). **Validation:** run over the Exp 53 agents (infant body,
   δ map — harness verification only), the procedure reproduces Exp 53 amendment 1's
   hand-declared placements exactly: gated {−0.3, −0.2, +0.5, +0.6}, exploratory +0.2
   ([data/54_dry_run_nonfrozen/](../data/54_dry_run_nonfrozen/README.md)). The procedure
   is frozen with this amendment; the *numbers* it yields for the Exp 54 seeds are declared
   in the results doc before Phase B, as pre-registered.
2. **The exploratory agent** = the taught seed outside the gated three (42, 43, 44) with the
   lowest Phase A late-bin (act3+act4) directedness (`manifest --experiment 54
   --phase-a-records`) — Exp 53's seed-48 choice, made a rule.
3. **Gate C, operationalised.** Phase C is a Phase-1-shaped probe block (`run --phase 1
   --body-ref bodies/reachy_mini --factory --gate C --targets …`, apparatus places, the
   agent never moves). Per taught seed: the fraction of gated placements at which
   `consulted_bias_by_modality.audio ≠ 0` **and** the chosen direction is correct (direction
   only: any leftward affordance for az < 0) ≥ 0.80; PASS needs ≥ 2 of 3 seeds. Every
   control placement must show `consulted…audio == 0`. A correct direction *without* a
   consulted audio bias — the innate azimuth drive choosing — does not count, because the
   sentence being measured is "the nursery's want is consulted", not "the robot turns".
   `verdict --gate C` recomputes the verdict from the probe records.
4. **Harness facts corrected.** `benchmark_cradle_mother.py` had no `--embodiment` flag (the
   pre-registration's runbook assumed one) — added, with the satiated arm's body keyed on the
   chosen embodiment (`bodies/reachy_mini_infant` → `bodies/reachy_mini_infant_satiated`;
   `--satiated-embodiment` overrides; an unmapped embodiment refuses the satiated arm) and a
   preflight that instantiates every arm's body before the first sub-sim. Runbook `--seeds 12`
   → `--trials 12` (the flag's real name). The YAML block's `archetype: robot` is not a
   registered archetype (`_data/components/archetypes/` has none; the parent `reachy_mini`
   is `humanoid`, and `tests/unit/test_archetypes.py` rejects unregistered values) — both
   bodies carry `archetype: humanoid`; the archetype is imagination-scaffolding metadata
   and touches nothing the experiment measures. `cradle_mother.reactive_mother_tick` scores
   directedness as `progress = |stim| − |az_after| > 0` — not on tool names — and over the
   arc's stimulus band {±0.4 … ±0.9} that is exactly direction-only for both of the robot's
   steps (a 0.50 big step from the nearest stimulus, |0.4|, lands at |0.1|). Declared, not
   amended: Exp 52 Phase B ran with `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` unset (all 36 rows)
   and Phase A keeps that (S6), so a turn may hold more than one action; two same-direction
   big steps from |0.4| overshoot to |0.6| and score as not-directed although each step's
   direction was right. That is the robot's repertoire meeting this apparatus — part of what
   Phase A measures (the pre-registered "the repertoire changes the learning problem"
   outcome), not a tool-name keying to fix.
5. **Records** carry `exploratory_agent` per probe/trial (the Exp 53 records' seed-48 rule is
   the fallback), the `start` record stamps `body_ref`, `factory`, the deltas the factory read,
   the targets file and the procedure text; `--delta` is refused with `--factory`.
6. **Dry run** (the pre-registered sign-off item 3): the Reachy nursery body through the
   production factory on the dry rig — four tools, deltas {+0.3, −0.3, +0.9, −0.9} read from
   the YAML — driven by *re-keyed copies* of the Exp 53 agents (`infant_operant_turn_*` →
   `reachy_mini_turn_*`), Phase 1, Phase 2 (both blocks) and Phase C all complete with the
   Exp 53 record/verdict shape. Harness verification, not a result.

**Disclosure:** at amendment time the author had seen the harness dry run above (re-keyed Exp
53 agents: Gate I 3/3, dry Phase 2 taught 1.00 / satiated 0.00 / no_feed 0.50, dry Gate C
PASS — an offline rig with a modeled source) and the Exp 53 sweep table. **No nursery on the
Reachy body has run; no Phase A, B or C data exists.**

## Sign-off (operator fills before Phase A)

1. Pre-registration read and frozen at commit `efdf5cd5` (#558) — ☑ 2026-08-26
2. Body YAML, arms, gate v3 constants, the target procedure (amendment 1), Phase C, stop rules accepted — ☑ 2026-08-26
3. Harness PR (`feat/exp54-harness`) with the body's structural tests; dry run clean; amendment 1 appended before Phase A — ☑ 2026-08-26 (PR number recorded in the results doc)
