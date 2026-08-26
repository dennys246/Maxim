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
*Operationalisation, stated plainly (review fold):* the credit is the **sign** of
relief, never its magnitude (the channel-3 invariant), so what the design
discriminates is **nonzero vs exactly-zero relief** — "hungry" means hunger > 0. In
steady state the taught infant is fed at hunger ≈ 0.05–0.10 (Phase A: drift +0.05/tick
precedes each feed; Phase B: ≈ 0.08/turn of wall-clock drift against a −0.5 feed), well
below the deprivation threshold. That is the right *plumbing* test of "is the credit
sourced from relief"; a need-gated credit (mint only above the satisfaction threshold)
would be a mechanism change and its own pre-registration. Both harnesses record hunger
at feed (`hunger_at_feed_*` in Phase A; `relief=` per turn in Phase B's mother log) so
the claim is auditable.

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

**What the `--credit constant` A/B does and does not test (review fold):** on the
*taught* arm relief is > 0 on every contingent feed (drift precedes feed), so the relief
credit stream is `+1` per feed — byte-identical to `constant` with `feed_reward=1.0`, same
RNG, same trajectory. "The harness change did not move the taught curve" is therefore
true and vacuous. The entire empirical content of the change is the **satiated** arm:
under `relief` it mints nothing; under `constant` it is credited on every feed and learns.
The −1 branch is unreachable with the bundled bodies (negative feed deltas on entropic
"up" drives), so "every credit is +1" is an apparatus check, not a test of the sign logic.

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
`fed` telemetry), so feed *rate* is matched by construction (S5) — but runs on an
**independent RNG stream** (*amendment 1*): on the taught seed's stream it replays the
taught trajectory action-for-action and learns by construction.

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
    the **pre-learning baseline = the first 5 ticks of every seed, pooled** (probe 4's
    convention; *amendment 1*). Baseline ≥ 0.80 → LEARNED-AT-CEILING (settled ≥ 0.80),
    reported as such (S7).
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
| **HUNGER-NECESSARY** (new) | taught late ≥ satiated late + **0.20**, satiated shows no rise ≥ 0.15, **and satiated late ≤ no_feed late + 0.20** (*amendment 2*: a cap mirroring Phase A's `satiated ≤ 0.60`, so a slowly-rising satiated arm cannot pass on the rise term alone) |
| **APPARATUS (L2)** | per-cell directedness must vary across seeds: if any arm's late-bin value is a seed-invariant exact fraction (the v2 signature: 8/12 → 8/12 → 4/12 with zero seed spread), the shuffle did not break phase-locking and the **agent side needs dither too** — no science verdict is issued; record as an L2 amendment |
| MARGIN (L1) | any claim of learning-without-behaviour-change, or of "no learning", must cite `learned_margin`/`explore_decisive` (#504); the argmax cannot see a bias below ~0.11 |

Verdict constants live in `scripts/analyze_cradle_mother.py` (v2) and are extended, not
retuned, for v3 — `HUNGER_MARGIN = 0.20`, `SATIATED_RISE_MAX = 0.15`, and the seed-spread
apparatus check — in the same commit as the harness change.

## Phase A — RESULT (2026-08-25, frozen parameters, `main` @ `e367f526`) — PASS

Run by the operator from the merged harness commit (`executed_git_hash e367f526`,
`executed_maxim_file <repo>/src/maxim/__init__.py`), 8 seeds × 600 ticks, bin 50,
ε = 0.2 — exactly the frozen parameters. Raw record:
[`data/52_phaseA_scripted.json`](../data/52_phaseA_scripted.json).

| arm | first-5-ticks baseline | settled (last 4 bins) | feeds / credits per seed |
|---|---|---|---|
| taught | 0.650 | **0.892** | 532–544 / 532–544 (every feed credited, all +1) |
| satiated | 0.475 | **0.496** | 274–315 / **0** |
| yoked | 0.575 | **0.496** | = taught seed's feeds / all credited |
| no_feed | 0.475 | **0.496** | 0 / 0 |

- **LEARNED: PASS** — settled 0.892 ≥ 0.80, rose +0.24 ≥ 0.15 (not at ceiling).
- **HUNGER-NECESSARY: PASS** — taught − satiated = +0.40 ≥ 0.20; satiated 0.496 ≤ 0.60.
- **MOTHER-NECESSARY: PASS** — taught − max(yoked, no_feed) = +0.40 ≥ 0.20.
- **Mechanism sanity: OK** — satiated credits 0/8 seeds; taught credits == feeds
  (4305/4305), every reward +1; yoked feeds == taught feeds per seed.
- Hunger at feed (taught, median of per-seed medians): **0.050** — the steady state
  the §Arms caveat predicted; this result is "nonzero vs zero relief", as declared.

**Two things worth reading off the record.** (1) The satiated and no_feed curves are
**identical to the last digit** (`curves.satiated == curves.no_feed`): with no credit
ever minted, both arms are the same seeded random walk — the feed without need had
literally zero effect on behaviour, which is the strongest form the HUNGER-NECESSARY
claim can take on this substrate. (2) The yoked arm received the taught arm's 4305
feeds and credits — all landing on whatever action was pending — and stayed at chance
(0.496): the *contingency*, not the reward volume, carries the learning.

**Consequence per the outcome tree:** Phase A passes → **Phase B runs** (once), on the
mini, with `--stimulus-order shuffled --credit relief`, arms taught / satiated /
no_feed, n = 12/arm.

## Phase B — RESULT (2026-08-25, apparatus v3, big-mac-mini, `main` @ `60195a29`) — GRADUATE

Run once, per stop rule B, by the operator from `~/RMSrv/scripts/Maxim` on the mini:
`--arms taught,satiated,no_feed --trials 12 --seed-base 42 --stimulus-order shuffled
--credit relief --model mistral-7b`, ~663 s/run, 36/36 runs recorded, exposure-matched
48 turns/seed. Every row stamps `credit=relief`, `stimulus_order=shuffled`,
`executed_git_hash 60195a29`, the satiated body on its arm, and `relief` as the
*observed* credit mode on all 144 act records. Raw record:
[`data/52_phaseB_embodied.jsonl`](../data/52_phaseB_embodied.jsonl); per-run provenance
[`data/52_phaseB_runs/`](../data/52_phaseB_runs/README.md); full write-up
[52_nurture.md](../52_nurture.md).

| arm | act1 | late (act3+4) | per-seed late SD | fed / credited rate |
|---|---|---|---|---|
| taught | 0.614 | **0.878** | 0.130 | 0.73 / 0.73 |
| satiated | 0.341 | 0.441 | 0.079 | 0.35 / **0.00** |
| no_feed | 0.333 | 0.413 | 0.082 | 0.00 / 0.00 |

- **LEARNED: PASS** (late ≥ 0.65; rise +0.26 ≥ 0.15; not at ceiling).
- **MOTHER-TAUGHT: PASS** (+0.465 ≥ 0.20).
- **HUNGER-NECESSARY: PASS** (+0.437 ≥ 0.20; satiated rise +0.10 < 0.15; satiated late ≤
  no_feed + 0.20).
- **APPARATUS L2: clean** — per-seed SD 0.08–0.13, 6–8 distinct late values per arm; no
  seed-invariant fractions. **APPARATUS S3: OK.**
- Outcome-tree branch: **"Learned to want."** Ledger consequences in
  [52_nurture.md](../52_nurture.md) §Ledger consequences. One session, n = 12/arm —
  cross-session replication outstanding, as for every embodied row.

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
- What the operationalisation IS and IS NOT (bio-fidelity, review fold): primary
  reinforcement of an action-in-context (sound-direction cluster → turn), Thorndike-style.
  Not modeled: secondary reinforcement of the mother's voice (no value on the stimulus
  itself) and incentive salience / devaluation — a learned bias is expressed identically
  when sated (hunger does not gate readout), so "learns to want" is a state–action
  policy, not a drive. The satiated arm also differs in its pain landscape (no
  hunger/thirst PainBus events), a minor confound because the taught arm rarely reaches
  deprivation either and `recommend_action` is positive-gated.
- LEARNED's rise term is largely implied by its level term once the baseline is the
  first-5-ticks window (≈ chance): read LEARNED as a level test; do not count "rose
  +0.35" as separate evidence.

## Apparatus declarations (S1–S8)

- **S1 rows riding on the change:** the Exp 48 row (Operant orienting, embodied) —
  `reactive_mother_tick`'s credit value changes; Exp 46's scripted rows are re-derived by
  the new harness's `taught` arm at `GAIN`-equivalent (the constant-credit path is kept
  behind `--credit constant` for the A/B). Exp 45 (hardware) does not ride on this.
- **S3 in-sim assertions (enforced by `analyze_cradle_mother.py --gate v3` from the
  per-turn mother telemetry the harness aggregates — `credited_rate`, `neg_reward`,
  `credited_no_relief`):** satiated arm credits == 0; no negative reward; no credit
  without relief. "feed count == credit count in taught" holds on **shaping turns**
  (`progress > 0`) only: on turn 0 the mother's legacy fallback feeds a centered infant
  (`|az| ≤ oriented_threshold`, azimuth initial 0) whose hunger is still 0 →
  `fed=True, relief=0, credited=False`, one uncredited feed per run by construction.
  Phase A asserts the strict form in-harness (its ticks are all shaping ticks) plus
  `yoked feeds == taught feeds` (S5).
- **S4:** per-run JSONL committed under `docs/experiments/data/52_*` (both phases) with
  `executed_git_hash`; workdirs on durable storage.
- **S5:** exposure contract above; the mother's own turns are the declared asymmetry.
- **S6:** `MAXIM_CRADLE_MOTHER_STIMULUS_ORDER=shuffled` and the relief-sourced credit are
  fidelity changes relative to Exp 48 v2 and are declared here; neither changes between
  arms within a phase. The campaign harness refuses to start when an ambient env value
  disagrees with its flags (exit 3), and stamps the credit mode the sub-sim *observed*
  per act. Phase B's relief sign rides on wall-clock drift between feeds (`dt > 0`
  between mother ticks, guaranteed by the narrator call in between) — an L5-class
  timing dependency, named here.
- **S7:** ceiling clause carried from v2.
- **S8:** one harness at a time; Phase B runs on the mini, not co-located with a leader.

## Amendments

**Amendment 1 — 2026-08-25, PRE-DATA, structural (harness dry run at non-frozen
parameters: 3 seeds × 300 ticks, run to verify the harness before the pre-registered
Phase A).** Two apparatus defects in the harness/pre-registration as first written,
neither about effect size:
1. *LEARNED's rise was unattainable by construction.* The scripted mechanism learns
   within ~10 ticks (probe 4's own note), so bin 1 (50 ticks) is already learned and
   "rose ≥ 0.15 from bin 1" fails on every run that learns — the S7 ceiling defect,
   one level down. Fixed to probe 4's convention: the pre-learning baseline is the
   first 5 ticks pooled across seeds, with a LEARNED-AT-CEILING clause. Margin
   unchanged (0.15).
2. *The yoked arm replayed the taught arm.* Feeding on the taught seed's schedule
   while sharing the taught seed's RNG stream reproduced the taught trajectory
   action-for-action (same sounds, same ε draws, same empty NAc → same actions →
   same feeds), so "yoked" learned identically. Fixed: the yoked infant runs on an
   independent stream (`seed + 100_003`) with the taught seed's feed schedule.

**Disclosure (per the Exp 48 gate-v2 precedent):** at amendment time the author had read
the 3 × 300 relief-mode curves. Before the fixes: taught bin 1 = 0.85 (hence the rise
defect) and yoked settled 0.90 (the replay defect). After the fixes: taught 0.90 /
satiated 0.52 / yoked 0.47 / no_feed 0.52, all four gates PASS, sanity OK; `--credit
constant`: satiated 0.90 (constant credit ignores need — the contrast the experiment is
built on; the taught curve is identical in both modes by construction, see
§Mechanism). The dry-run record is committed as
`docs/experiments/data/52_dryrun_nonfrozen.json` (NOT the frozen run; non-frozen
parameters). No gate constant was retuned; the frozen Phase A run has not yet happened.

**Amendment 2 — 2026-08-25, PRE-DATA, structural (review round).** Phase B's
HUNGER-NECESSARY had only relative terms (margin below taught; rise < 0.15) while Phase A
has an absolute cap (satiated ≤ 0.60); a satiated arm rising slowly to 0.62 against a
0.17 control would have passed B. Added the B-form cap `satiated late ≤ no_feed late +
0.20` — the satiated infant must be indistinguishable from the teacherless control.
Conservative direction (harder to pass); no other constant changed.

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
export PYTHONPATH=$PWD/src
# The FLAGS select the apparatus (the harness overwrites the sub-sim env from them and
# refuses to start if an ambient MAXIM_CRADLE_MOTHER_* value disagrees — do not export those).
python scripts/benchmark_cradle_mother.py --arms taught,satiated,no_feed --trials 12 --seed-base 42 \
  --stimulus-order shuffled --credit relief \
  --model mistral-7b --workdir ~/exp52/phaseB --out docs/experiments/data/52_phaseB_embodied.jsonl
python scripts/analyze_cradle_mother.py --in docs/experiments/data/52_phaseB_embodied.jsonl --gate v3
```

## Sign-off (operator fills before first run)

- [x] Gates + parameters above reviewed and FROZEN at the commit that lands the harness
      (amendments 1–2 included); hash: `e367f526` (#543)
- [x] Harness change (relief-sourced credit + satiated body + v3 analyzer constants) merged
      with its guard tests (#543); under `--credit relief` the satiated arm mints zero credits and
      under `--credit constant` it is credited on every feed (the A/B's actual content)
- [x] Phase A run: date `2026-08-25` @ `e367f526` — LEARNED `PASS` HUNGER-NECESSARY `PASS` MOTHER-NECESSARY `PASS`
      (sanity OK; record `data/52_phaseA_scripted.json`)
- [x] Phase B started: date `2026-08-25`, machine `big-mac-mini`, seeds `42–53` — COMPLETE the same day, **GRADUATE** (gate v3; record `data/52_phaseB_embodied.jsonl`)
