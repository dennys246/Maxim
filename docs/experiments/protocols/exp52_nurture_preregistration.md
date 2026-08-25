# Exp 52 — Nurture: caregiver-taught orienting through hunger relief (pre-registration)

**Status:** PRE-REGISTERED 2026-08-25, frozen before any run. Roadmap 1.1 item 17 — the
gate on `1.1.0` final ([roadmap](../../plans/roadmap_1_1_to_1_3.md)).
**Lineage:** Exp 46 (scripted mother, PASS) → Exp 48 (embodied mother, PARTIAL:
apparatus-v2 re-earned the mother effect but the instrument phase-locks and the
reward was credited by fiat) → this. Exp 48's own sanctioned next step is the randomised
stimulus order + a v3 gate frozen pre-data; this pre-registration is that, plus the
change that makes the reward *the infant's own relief*.
**Owner intent (recorded so the claim cannot drift):** "sensorimotor learning" here means
the infant **learns to want to orient** — it acquires the value of turning toward the
caregiver from a primary reward (being fed while hungry), with no hand-declared
orienting drive. Exp 45 earns orienting under a declared centering drive on real
hardware; this experiment tests the acquisition of the drive-like policy itself.

---

## The question

Does an infant that has a hunger drive and **no** orient drive learn to turn toward its
mother's voice when the only consequence of turning toward her is that she feeds it —
and when the credit the substrate receives is derived from the **relief the infant
actually experiences**, not a constant handed to the learner?

Three things must be true for the answer to be "yes, it learned to want":

1. it learns when hungry and fed contingently (**LEARNED**);
2. it does **not** learn when it is fed contingently but is not hungry — the feed event
   without need teaches nothing (**HUNGER-NECESSARY**; the arm Exp 48 never had, and the
   one that separates "learns to want" from "learns to be fed");
3. it does **not** learn when fed on the same schedule non-contingently (**MOTHER-NECESSARY**,
   the superstition control) or not at all.

## What is measured vs what is credited — the mechanism under test

Today (`simulation/cradle_mother.py::reactive_mother_tick`) the mother's feed already
writes a real hunger delta (`_apply_sensor_deltas(root, {"hunger": -feed_amount})`), and
then credits the infant's pending action with a **constant** `feed_reward=1.0` via
`NAc.credit_operant_reward`. The scripted harness (`orient_substrate/4`) credits a
constant `GAIN` and never touches hunger at all (drift 0.0). In both, the reward is
independent of the infant's state — a full infant is credited exactly as a starving one.

**Change under test (front-gate: rides existing infrastructure, one new contract):**
the operant credit's *value* becomes the sign of the drive-relief the feed produced,

```
reward = sign( Σ_{d ∈ drives the feed touched} drive_comfort_progress(spec_d, before_d, after_d) )
         with |Σ| ≤ 1e-9 → NO credit (never a fabricated ±1)
```

using the existing `embodiment/sem.py::drive_comfort_progress` (value-based, the same
signal channel 3 already uses for self-caused relief) and the existing one-step pending
operant trace (`set_pending_operant_action` at the infant's action →
`credit_operant_reward` at the mother's next tick). The credit still lands on the
direction-bearing (exteroceptive) cluster the pending action was keyed to, so the seam
routing (#411) is unchanged. **No new mechanism class:** the temporal-credit distributor
is NOT required — the contingency is one turn, which the pending-operant trace already
bridges; multi-turn delays are explicitly out of scope (roadmap item 17's earlier
wording naming the temporal-credit bridge is corrected by this section). Estimated
delta: `feed_reward` becomes computed in `reactive_mother_tick` (+ a `relief` field in its
telemetry) and the scripted harness gains hunger drift + the same computation — tens of
lines, guard tests in the same commit.

`MAXIM_OPERANT_ONLY_CREDIT=1` stays ON in every arm: it suppresses the tool-success
floor, which probe 3 showed drowns any operant signal; it does not supply reward.

## Arms (both phases)

| arm | hunger | mother feeds when… | credit |
|---|---|---|---|
| **taught** | drifts up (hungry within a few turns) | infant's prior turn moved TOWARD her sound (`progress > 0`) | sign(relief) → pending action |
| **satiated** | held at 0 (drift 0, initial 0) | same contingency as taught | relief = 0 → **no credit** (the mechanism must produce this; the arm checks that it does) |
| **yoked** | as taught | on the same *schedule* as a matched taught seed, regardless of the infant's turn | sign(relief) → whatever action happened to be pending |
| **no_feed** | as taught | never (sound still placed) | none |

`satiated` is the discriminating arm: identical feed events, identical contingency,
zero need. If it learns, the credit is not coming from relief. `yoked` uses the taught
seed's feed turn-indices (recorded in Phase A; in Phase B, the paired taught run's
`fed` telemetry), so feed *rate* is matched by construction (S5).

## Phase A — scripted (runs first; gates Phase B)

New harness `scripts/orient_substrate/9_hunger_relief_orient.py`, an extension of
`4_operant_learning_curve.py` (same body shape, same external ε-greedy, same directedness
scoring): hunger drifts `+0.05/tick` (clamped to 1.0), the mother feeds `−0.5` on the
contingency, reward = sign(relief). No LLM, no sim machinery, seconds per seed.

- **Parameters (frozen):** 8 seeds, 600 ticks, bin 50, ε = 0.2, `regime=full`
  (|az| ∈ U(0.3, 0.9)), `MIN_CONF` as in probe 4, chance ≈ 0.50.
- **Metric:** per-bin directedness (fraction of turns with `progress > 0`), as Exp 46.
- **Gates (frozen; Exp 46's LEARNED/MOTHER-TAUGHT plus the new one):**
  - **LEARNED:** taught settled (mean of last 4 bins) ≥ **0.80** AND rose ≥ **0.15** from
    the first bin.
  - **HUNGER-NECESSARY:** taught settled ≥ satiated settled + **0.20**, and satiated
    settled ≤ **0.60**.
  - **MOTHER-NECESSARY:** taught settled ≥ max(yoked, no_feed) settled + **0.20**.
  - **Mechanism sanity (must hold or the run is an apparatus failure, not a result):**
    satiated logs **zero** credits; taught logs credits only on fed ticks; every credit's
    sign is +1 (feeding a hungry infant is relief by construction).
- **Stop rule A:** if LEARNED fails here, **Phase B does not run**. The relief-sourced
  credit could not teach on the clean substrate; the audit goes to the credit path
  (magnitude/sign/threshold interaction with the ~0.11 visibility floor, L1), not to more
  embodied runs. Record the FAIL in this doc; 1.1.0 ships with it named.

## Phase B — embodied (`cradle_mother`, apparatus v3)

`scripts/benchmark_cradle_mother.py` with the arc's declared scaffold
(`feed_amount 0.5`, `stimulus_azimuths (−0.7, 0.6, −0.5, 0.8, −0.9, 0.4)`, guide 0,
4 acts × 10–12 turns), plus:

- `MAXIM_CRADLE_MOTHER_STIMULUS_ORDER=shuffled` (#514 — seeded per-block permutation;
  the sanctioned L2 phase-lock fix, never yet run at campaign scale);
- relief-sourced `feed_reward` (the change above);
- arms `taught`, `satiated`, `no_feed` (yoked is Phase-A-only unless budget allows —
  Phase A's yoked result is the superstition control of record);
- `satiated` = body `bodies/infant_operant_satiated` (extends `infant_operant`; hunger +
  thirst `drift_rate 0`, `initial 0.0`) — a body variant, not an env flag, so the arm is
  visible in provenance;
- `--trials 12` per arm, exposure-matched **48 turns/seed** (S5; the harness records
  per-arm exposure, tolerance 0.20), `explore_weight 1.5` (the v2 baseline), seed base 42,
  narrator `mistral-7b` at temperature 0 (prose-less arc), `MAXIM_OPERANT_ONLY_CREDIT=1`,
  `--workdir` on durable storage (never `/tmp`, S4), `assert_repo_interpreter` on
  (already in the harness), `executed_code_provenance` stamped per run.

**Gate v3 (frozen pre-data; v2's constants carried, one gate added, one apparatus gate
made explicit):**

| gate | rule |
|---|---|
| LEARNED | taught late (act3+act4) directedness ≥ **0.65** AND rose ≥ **0.15** from EARLY = act1 only |
| CEILING (S7) | early ≥ 0.65 → LEARNED-AT-CEILING (late ≥ 0.65 and no degradation beyond the v2 tolerance), reported as such |
| MOTHER-TAUGHT | taught late ≥ no_feed late + **0.20** |
| **HUNGER-NECESSARY** (new) | taught late ≥ satiated late + **0.20**, and satiated shows no rise ≥ 0.15 |
| **APPARATUS (L2)** | per-cell directedness must vary across seeds: if any arm's late-bin value is a seed-invariant exact fraction (the v2 signature: 8/12 → 8/12 → 4/12 with zero seed spread), the shuffle did not break phase-locking and the **agent side needs dither too** — no science verdict is issued; record as an L2 amendment |
| MARGIN (L1) | any claim of learning-without-behaviour-change, or of "no learning", must cite `learned_margin`/`explore_decisive` (#504); the argmax cannot see a bias below ~0.11 |

Verdict constants live in `scripts/analyze_cradle_mother.py` (v2) and are extended, not
retuned, for v3 — `HUNGER_MARGIN = 0.20`, `SATIATED_RISE_MAX = 0.15`, and the seed-spread
apparatus check — in the same commit as the harness change.

## Outcome tree (decided now)

| outcome | reading | action |
|---|---|---|
| A passes; B passes LEARNED + MOTHER-TAUGHT + HUNGER-NECESSARY, apparatus gate clean | **Learned to want.** Orienting acquired from hunger-relieved feeding alone; the feed without need teaches nothing. | New graduation row "caregiver-taught orienting through hunger relief" (Earned, Tier 2); Exp 48 row's PARTIAL is superseded; `1.1.0` final ships with this as the headline. |
| A passes; B: MOTHER-TAUGHT + HUNGER-NECESSARY pass, LEARNED fails (rise < 0.15 or late < 0.65, not at ceiling) | Real, hunger-dependent, mother-dependent effect that does not reach the graded-skill bar embodied. | PARTIAL, named; `1.1.0` ships with it named. The audit target is the embodied credit *magnitude* (how many fed turns reach the L1 floor), not the mechanism. |
| A passes; B: HUNGER-NECESSARY fails (satiated learns) | The credit is not coming from relief — an apparatus/plumbing leak (a non-relief credit source is alive). | STOP; not a science result. Find the leak (`MAXIM_OPERANT_ONLY_CREDIT` coverage, `credit_source` provenance), fix with a guard test, re-run B once. |
| A passes; B fails the APPARATUS gate | Seed-invariant fractions survive the shuffle: the deterministic agent still phase-locks. | No verdict; L2 amended ("agent-side dither required"); `1.1.0` ships with "embodied measurement blocked by L2" named. Agent-side dither is a new pre-registration, not an amendment. |
| A fails LEARNED | The relief-sourced credit cannot teach even on the clean substrate. | Phase B does not run. Recorded FAIL; audit the credit path; `1.1.0` ships with it named. |

**Stop rule B (pre-registered):** Phase B runs **once** (plus the single re-run the
HUNGER-NECESSARY-leak branch allows). A second divergence — a new failure mode rather
than a narrowing one — ends Exp 52 for 1.1: the result is recorded as it stands and
`1.1.0` ships. No third arm, no sweep.

## What this experiment does NOT claim

- Nothing about *magnitude* selection (that is Exp 45b–e, hardware).
- Nothing about the LLM-AUT path — this is substrate-primary with no LLM in the action
  path (the narrator drives the world only).
- Nothing about loudness or startle (item 18 / the 1.3 reflex tier).
- Nothing beyond a one-turn contingency; multi-turn credit delay is out of scope.

## Apparatus declarations (S1–S8)

- **S1 rows riding on the change:** the Exp 48 row (Operant orienting, embodied) —
  `reactive_mother_tick`'s credit value changes; Exp 46's scripted rows are re-derived by
  the new harness's `taught` arm at `GAIN`-equivalent (the constant-credit path is kept
  behind `--credit constant` for the A/B). Exp 45 (hardware) does not ride on this.
- **S3 in-sim assertions:** satiated arm credits == 0; credit sign == +1 on every fed
  turn; feed count == credit count in taught.
- **S4:** per-run JSONL committed under `docs/experiments/data/52_*` (both phases) with
  `executed_git_hash`; workdirs on durable storage.
- **S5:** exposure contract above; the mother's own turns are the declared asymmetry.
- **S6:** `MAXIM_CRADLE_MOTHER_STIMULUS_ORDER=shuffled` and the relief-sourced credit are
  fidelity changes relative to Exp 48 v2 and are declared here; neither changes between
  arms within a phase.
- **S7:** ceiling clause carried from v2.
- **S8:** one harness at a time; Phase B runs on the mini, not co-located with a leader.

## Amendment rule

Amendments after first data are permitted only for *structural invalidity* (harness bug,
degenerate metric, an apparatus-gate failure) — never for effect size — and every
amendment demotes the affected claim to exploratory unless re-run fresh. Gate constants
above are frozen at the commit that lands this file.

## Runbook

```bash
# Phase A (seconds; no robot, no LLM)
PYTHONPATH=$PWD/src python scripts/orient_substrate/9_hunger_relief_orient.py --seeds 8 \
  --json docs/experiments/data/52_phaseA_scripted.json

# Phase B (mini; ~12 h for 3 arms × 12 seeds at ~712 s/run)
export PYTHONPATH=$PWD/src MAXIM_OPERANT_ONLY_CREDIT=1 MAXIM_CRADLE_MOTHER_STIMULUS_ORDER=shuffled
python scripts/benchmark_cradle_mother.py --arms taught,satiated,no_feed --trials 12 --seed-base 42 \
  --model mistral-7b --workdir ~/exp52/phaseB --out docs/experiments/data/52_phaseB_embodied.jsonl
python scripts/analyze_cradle_mother.py --in docs/experiments/data/52_phaseB_embodied.jsonl --gate v3
```

## Sign-off (operator fills before first run)

- [ ] Gates + parameters above reviewed and FROZEN; this file committed, hash: `________`
- [ ] Harness change (relief-sourced credit + satiated body + v3 analyzer constants) merged
      with its guard tests; `--credit constant` A/B reproduces Exp 46's taught curve
- [ ] Phase A run: date `________` — LEARNED `__` HUNGER-NECESSARY `__` MOTHER-NECESSARY `__`
- [ ] Phase B started: date `________`, machine `________`, seeds `________`
