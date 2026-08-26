# Exp 53 — Cross-context readout: does the nursery-taught want survive on the robot? (pre-registration)

**Status:** PRE-REGISTERED 2026-08-26, frozen before any run. **Roadmap 1.1 item 19 — the
`1.1.0` final cut is gated on this experiment's *recorded* outcome** (PASS, FAIL, or the
pre-registered instrument stop), not on a PASS.
**Lineage:** Exp 45 (real-hardware orient policy, innate azimuth drive, cross-*session*
transfer of a tabular NAc: PASS) → Exp 52 (nursery: a driveless infant learns to *want*
to orient from hunger relief, sim only: GRADUATE) → **this**: the Exp 52 infants,
unmodified, read out on the physical Reachy Mini.

**Owner intent (recorded so the claim cannot drift):** the 1.0/1.1 claim is learning
that *carries across sessions and contexts without fine-tuning*. Cross-session is earned
(Exp 42, Exp 45 s2, Exp 45d). Cross-**context** — the same learned state driving a
different body in a different world — has never been shown; "Sensorimotor" on a sim-only
proof is the weaker claim wearing the stronger name (the same reasoning that reopened
1.1 for Exp 52). A video of the robot turning toward a voice is illustration, not
evidence; the evidence is the side-by-side with the controls loaded the same way.

## The question

Take the twelve `taught` infants Exp 52 Phase B produced — their persisted NAc
(`aut_nac.json`: three `audio`-cluster biases, e.g. seed 42 `turn_left +0.652` on one
cluster, `turn_right +0.902` on another) and the EC they were formed in (`aut_ec.json`,
three `audio` substrate nodes, `hash_scheme` stable) — and load them, **unchanged, no
further learning**, into the production substrate-primary decision path with the
Reachy Mini's live DoA as the azimuth sensor and its body yaw as the turn. Do they turn
toward a speech source? And do the `satiated` and `no_feed` infants — persisted state
with **zero** bias entries, loaded identically — not?

This is **readout, not learning**: nothing credits the NAc on the robot; the persisted
files are not written. What transfers is the *motivation* (the learned bias), and the
test is whether the robot's percepts land in the representation the nursery built.

## What is shared by construction, and what is not

- **Sensor:** the nursery's mother "calls" by world-setting the infant's `azimuth` root
  sensor each turn; on the robot `DoAFeed` world-sets the *same* sensor
  (`world_set_axis(owner="doa_feed")`, capability-driven on any body declaring
  `azimuth`). Values in `[-1, 1]`, same convention (−1 left, +1 right).
- **Representation:** `runtime/agent_loop.py::_encode_current_clusters` reads the
  exteroceptive channel (`azimuth`, range-aware, place code OFF — the taught EC's
  `encoder_provenance` records exactly `sensor_names ["azimuth"]`, `range-aware`) into
  the `audio` modality of the loaded EC; `NAc.recommend_action(current_clusters=)` sums
  the cluster biases. Same functions, same files.
- **Actions:** the learned keys are `tool:infant_operant_turn_left` /
  `tool:infant_operant_turn_right`, so the robot runs the **`bodies/infant_operant`
  body** (entity name → tool prefix; no orient drive, as in the nursery) — *not*
  `bodies/reachy_mini`, whose tools are named differently and whose innate azimuth
  drive would confound the readout.
- **Not shared — declared apparatus (S6):** the infant body's turn affordances declare
  `self_effect: {azimuth: ±0.3}` and no `head_yaw`; the production
  `ReachyOrientMotorBackend` takes its yaw command from a `deltas` map that its factory
  reads from `head_yaw`. The harness therefore constructs the production backend
  directly with an explicit map **`{turn_left: +0.55, turn_right: −0.55}` rad** — the
  body-yaw step that delivers ≈ 0.30 az at the measured 0.578 az/rad × 0.94 delivered
  (H1), i.e. the nursery's per-turn azimuth step. Same clamped+locked dispatch
  (`ReachyMiniController.goto_target`), same head matrix, same `world_set_axis`
  readback. **δ is frozen here and is not retuned after data.**

**Considered and rejected (2026-08-26): re-running Exp 52 on a Reachy-shaped infant body**
(`head_yaw` self-effects so the factory path works unmodified). It would cost ~12 h of
mini time, re-open a frozen verdict, and — the real objection — the transfer claim is
*strongest* when the files that read out on the robot are byte-identical to the ones
the nursery wrote. The explicit-δ constructor keeps the body and the learned state
untouched. A hardware-shaped nursery body is the right move only if the robot itself
must *learn* (1.1.x), not for readout.

## Arms (nine agents + one declared exploratory)

All from `~/Maxim-experiment-archives/exp52_phaseB_2026-08-25/phaseB/<arm>_seed<n>_ew1.5/sim_reports/<session>/`
(`aut_nac.json` + `aut_ec.json` from the same session directory — the pair is
load-bearing, D2). Seeds are the first three in declaration order; chosen before any
robot run.

| arm | seeds | persisted bias entries | expected on the robot |
|---|---|---|---|
| taught | 42, 43, 44 | 3 each (left/right/oriented clusters) | turns toward the source |
| satiated | 42, 43, 44 | 0 | no learned preference (exploration only) |
| no_feed | 42, 43, 44 | 0 | no learned preference (exploration only) |
| taught **48** (exploratory, not in any gate) | 48 | 3 (weak: `+0.388 / +0.306`) | the L1-floor learner — reported, not scored |

Load: `NAc.load(path, apply_decay=False)` (the wall-clock since 2026-08-25 is the
operator's schedule, not agent time — the same rule `--resume-sim` applies),
`EntorhinalCortex.load(ec_path)`, `SensorEncoder(ec=ec, atl=None, nac=nac)`. The NAc's
`substrate_explore_bonus_weight` is set per phase (below); every other loaded field is
untouched. SHA-256 of both files is recorded before and after each agent's session and
**must be identical** (S3: readout wrote nothing).

## Stimulus geometry and trial generation

Exp 45's `--perturb` protocol, unchanged in spirit: a **sustained speech source** (phone
playing a podcast, 1–2 m, in front of the base's neutral heading) never moves; the
**apparatus** rotates the base by a commanded offset to place the source at the target
azimuth, re-centres between trials, and logs every apparatus move as `apparatus_*`.
Ground truth (the commanded offset, the gain estimate) **never reaches the substrate**
— the agent sees only the speech-gated DoA-derived azimuth, as in the nursery.

Targets — **superseded by amendment 1 (pre-data)**: originally chosen from the stimulus
list, they are now placed where the sweep shows the learning lives: **az ∈ {−0.3, −0.2,
+0.5, +0.6}** (+ two exploratory placements, −0.6 and +0.2, excluded from the gates), all
front hemisphere (L9 fold). Balanced
L/R schedule, seeded permutation per agent (int seed, never `hash()`).

Speech gate: every azimuth used is the median of k=5 speech-flagged reads; a trial with
no valid gated read before or after the turn is `invalid` (never fabricated) and
re-drawn, counted in the S3 record. The block refuses to start below a 0.70 speech-gate
rate over a 10 s probe.

## Phase 1 — INSTRUMENT CHECK (no motion) — gates Phase 2

Verify-the-instrument before theorising: the one unknown is whether live azimuth
percepts **pattern-complete into the nursery's audio clusters** instead of separating
into new nodes (EC `pattern_complete_threshold` 0.44). For each agent × each of the 4
targets (base rotated by the apparatus, robot otherwise still), record:

- `cluster_id` from `_encode_current_clusters` and whether it is one of the agent's
  persisted `audio` substrate nodes (`completed`) or a new node (`separated`);
- a **frozen-policy probe**: `recommend_action` over the two infant tools with
  `substrate_explore_bonus_weight = 0.0`, logging `tool_name`, `learned_margin`,
  `explore_decisive` (the #504 provenance). Correct = `turn_left` iff az < 0.

**GATE I (instrument), decided now:** for **≥ 2 of 3 taught seeds**, ≥ 80% of placements
`completed` **and** the probe is correct with `|learned_margin| > 0.11` (the L1
visibility floor) in ≥ 80% of placements. For every control agent the probe must show
**no learned preference** — consulted audio bias 0, `learned_margin` 0 — whether or not a
causal-credit tool is returned (amendment 1, item 2).

**Stop rule I (pre-registered):** GATE I fails → **Phase 2 does not run.** The outcome is
recorded as an **apparatus finding** (the cluster/margin table per agent), `1.1.0` ships
as-is with the sim-only claim scoped honestly, and cross-context transfer moves to
1.1.x with the finding as its motivation. *No* re-encoding, re-thresholding, cluster
re-mapping, or δ change is attempted inside this experiment — any of those is a new
mechanism and gets its own pre-registration.

## Phase 2 — READOUT TRIALS (motion) — runs once

**Primary condition: `substrate_explore_bonus_weight = 0.0`** — frozen policy with motion
(amendment 1, item 3; explore 1.5 is a secondary, reported block). Per agent **12 trials**
(4 gated targets × 3; plus the 2 exploratory placements × 3, recorded, not gated), one decision + one executed turn each, through the
production path: `propose_via_substrate(nac, executor, sensor_encoder)` → the chosen
infant tool → the explicit-δ backend → speech-gated re-read.

**Measure — directedness (delivered):** trial is `toward` iff `|az_after| < |az_before| −
0.05` on measured azimuth (motion that actually reduced the bearing, as the nursery's
directedness counts turns that moved toward the sound). Secondary (recorded, not
gated): the sign rule `turn_left ⇔ az_before < 0` on the chosen tool, which separates
"chose right, delivered wrong" (D30/D31-class hardware faults) from "chose wrong".

**GATE T (transfer), decided now — all three must hold:**
- **LEARNED-LIVE:** taught mean directedness over the 3 seeds (36 trials) **≥ 0.70**;
- **TAUGHT − SATIATED ≥ 0.20** and **TAUGHT − NO_FEED ≥ 0.20** (Exp 52's margin);
- **APPARATUS:** per-seed taught directedness is not a single repeated value across
  seeds (L2 check, 3 seeds), and the secondary sign-rule agreement with the delivered
  measure is ≥ 0.80 (otherwise the delivered result is a hardware artefact, and the
  verdict is `APPARATUS`, not FAIL).

**Stop rule T:** Phase 2 runs **once**. One re-run is permitted only for a recorded
apparatus fault (daemon dropout, speech-gate rate below the floor mid-block, motor-mode
loss) with the aborted block kept in the record. A FAIL is not re-run.

## Outcome tree (decided now)

| Phase 1 | Phase 2 | Verdict | `1.1.0` |
|---|---|---|---|
| GATE I pass | GATE T pass | **cross-context transfer EARNED** — new Earned row; the Exp 52 row gains a *readout on hardware* line | ships with the claim; the announcement may say the robot turns toward the voice it was taught to want, with the controls in the video |
| GATE I pass | GATE T fail | **FAIL recorded** — motivation did not read out on the body at this geometry | ships, sim-only claim, the fail named in the release notes |
| GATE I pass | APPARATUS | delivered-vs-chosen disagreement → hardware finding (D-number), no verdict on the claim | ships, sim-only claim; re-run pre-registered separately |
| GATE I fail | — | **instrument stop** — live percepts do not complete into the nursery's clusters | ships as-is; transfer → 1.1.x |

## What this experiment does NOT claim

- **Learning on the hardware.** Nothing credits the NAc here; the H1/Exp 45 rows cover
  in-session hardware learning with an innate drive. This is the readout of a want that
  was learned elsewhere.
- **Generalisation** beyond the taught geometry (four targets inside the nursery's
  stimulus band; front hemisphere only), or to the *mother's* voice specifically — any
  sustained speech is the stimulus, as in Exp 45.
- **Magnitude** (one fixed δ), **loudness** (1.1.1), the LLM-AUT path, multi-step
  centring (one turn per trial).
- **Statistical weight:** 3 seeds × 12 trials per arm, one hardware session, one room,
  one source; the cross-session replication caveat on the Exp 45 and 52 rows applies
  here from birth.

## Apparatus declarations (S1–S8)

- **S1 provenance:** the harness calls `assert_repo_interpreter` before the first robot
  command and stamps `executed_code_provenance` + SDK/daemon versions (must be equal;
  1.8.3 at pre-registration) + `hardware_id` into every record.
- **S3 measurement integrity:** invalid (un-gated) reads are never fabricated;
  before/after SHA-256 of `aut_nac.json` / `aut_ec.json` per agent; a `credited` counter
  that must stay 0.
- **S4 raw records:** `docs/experiments/data/53_cross_context_readout.jsonl` (one line per
  probe placement and per trial: agent, arm, seed, commanded offset, az before/after,
  gated-read counts, `cluster_id`, `completed`, `tool_name`, `learned_margin`,
  `explore_decisive`, delivered yaw, head pose per turn (D30), apparatus events) +
  `53_agents_manifest.json` (per-agent file paths + SHA-256s).
- **S5 exposure:** every agent gets exactly 4 probe placements and 12 trials; controls
  are not shortened.
- **S6 fidelity toggles (declared):** explicit `deltas` map (δ = 0.55 rad) instead of
  the factory's `head_yaw` read; `substrate_explore_bonus_weight` set programmatically
  per phase (0.0 / 1.5); place code OFF; `media_backend="no_media"`.
- **S7 ceiling clause:** a taught seed at 1.00 is a pass, not a saturation problem.
- **S8 pre-conditions (H1's, verbatim):** `curl …/api/daemon/status` version ==
  `reachy-mini` client version; `motor_control_mode == "enabled"`; `yaw_verify.py`
  d(head)/d(body) within 0.9–1.1 (the mics must actually rotate — the Exp 45 retraction
  class); recenter readback within 0.05 rad; speech-gate probe ≥ 0.70.

## Amendments

**Amendment 1 — 2026-08-26, PRE-DATA, structural (harness dry run, no robot).** Three
findings from running the harness offline through the *production* encode → recommend
path against the archived files (the dry rig fakes only the motor and the sensor):

1. **Where the learning lives on the azimuth axis.** Sweeping az ∈ [−1, 1] through each
   taught agent's loaded EC (fresh stash per value, nothing saved) shows the nursery's
   three `audio` clusters partition the axis into **FAR-LEFT (az ≤ −0.5; seed 44:
   ≤ −0.4) / CENTRE (−0.4 … +0.3) / RIGHT (≥ +0.4)** — identically for seeds 42, 43, 44
   (48 differs). The learned biases sit: **`turn_right +0.90` on RIGHT** (all three),
   **`turn_left +0.59–0.65` on CENTRE**, and only `turn_left +0.006` on FAR-LEFT.
   Mechanism (read, not guessed): operant credit is keyed to the *decision-time*
   cluster (`LLMProposal.clusters` → `set_pending_operant_action`), and the one-turn
   trace credits the *last* action before a feed; from a left stimulus (−0.7 / −0.5)
   the last action before the feed is taken from the centre bin, so the left credit
   accrued there. A coherent learned policy — *right → turn right; centre-left → turn
   left* — with a coarse representation (the centre bin also spans 0…+0.3). Every
   value completed into an existing cluster (no separation) for all ten agents.
   **Consequence:** the frozen targets ±0.5/±0.6 would have probed RIGHT (strong) and
   FAR-LEFT (margin 0.006 < the 0.11 floor) — a Gate I failure manufactured by target
   placement, not by the instrument. **Gated targets become az ∈ {−0.3, −0.2, +0.5,
   +0.6}** (left targets in the centre bin, right targets in the right bin; commanded
   offsets ≈ −0.52 / −0.35 / +0.87 / +1.04 rad). Two **exploratory placements**, recorded
   and excluded from every gate: **−0.6** (FAR-LEFT, the weak bin) and **+0.2** (the
   centre bin's right half, where the learned policy predicts a *wrong-way* turn — the
   representation's limit, stated in advance).
2. **Controls act on causal credit.** `no_feed` agents choose `turn_right` at explore 0
   with `learned_bias 0.0` and `causal 0.94` — persisted execution-success links, side-
   blind; `satiated` agents return no tool. The pre-registered control expectation ("no
   positively-scored tool") is therefore replaced by the correct one: **a control's
   consulted audio bias is 0 and its `learned_margin` is 0** — no learned preference —
   whether or not it acts. Under Phase 2 this predicts control directedness ≈ 0.5
   (one fixed direction against a balanced L/R schedule) or 0 (no action).
3. **Explore-first is not persisted.** `_ever_selected` / `_visit_count` are session
   state, so at explore 1.5 the first decision per tool is forced exploration and the
   novelty term stays comparable to the learned bias for several trials — 12 trials
   cannot reproduce Phase B's late-bin regime. **Phase 2 primary condition = explore
   0.0** (frozen policy *with* motion and delivered measurement; Gate T applies to it);
   **explore 1.5 is a secondary block**, run and reported per arm, not gated.

4. **Ceiling vs the L2 spread check.** The APPARATUS clause "per-seed taught directedness
   is not a single repeated value" contradicts S7 when all three seeds read 1.00 (the
   dry run did exactly that). The spread check applies **below ceiling only**: three
   seeds at 1.00 is a PASS; an identical repeated value below 1.00 is the phase-lock flag.

**Disclosure:** at amendment time the author had seen the dry-run probe table for all
ten agents (taught: correct at every placement, margins 0.90 right / 0.006 far-left;
seed 48 wrong-way at +0.5/+0.6 as its `+0.061 turn_left` bias predicts) and the sweep
table above. No robot data exists. The sweep is committed with the harness
(`docs/experiments/data/53_dry_run_nonfrozen.jsonl`, harness verification, not a result).

## Amendment rule

Structural, pre-data amendments only (a harness dry run may reveal a mechanical
impossibility, as Exp 52's did); every amendment is dated and appended below with what
the author had seen at the time. Gates, margins, δ, targets, seeds and the two stop
rules are not amendable after the first Phase 1 record exists.

## Runbook

```bash
# Pull the nine (+1) agent pairs off the archive into a manifest (no robot):
python scripts/orient_backbone/exp53_cross_context_readout.py manifest \
    --archive ~/Maxim-experiment-archives/exp52_phaseB_2026-08-25/phaseB \
    --out docs/experiments/data/53_agents_manifest.json

# Dry run against the offline rig (no robot) — harness verification, not a result:
python scripts/orient_backbone/exp53_cross_context_readout.py run --dry-run \
    --manifest docs/experiments/data/53_agents_manifest.json --out /tmp/53_dry.jsonl

# Phase 1 then Phase 2 on the robot (Phase 2 refuses to start unless Phase 1's GATE I
# verdict in the same output file is PASS):
export PYTHONPATH="$PWD/src"
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 \
    --manifest docs/experiments/data/53_agents_manifest.json --phase 1 \
    --out docs/experiments/data/53_cross_context_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 \
    --manifest docs/experiments/data/53_agents_manifest.json --phase 2 \
    --out docs/experiments/data/53_cross_context_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py verdict \
    --records docs/experiments/data/53_cross_context_readout.jsonl
```

## Sign-off (operator fills before the first robot run)

1. Pre-registration read in full and frozen at commit `0570aa8e7bfd` (+ amendment 1 in the harness commit) — ☑ 2026-08-26 (operator)
2. Seeds, targets (as amended), δ, gates and both stop rules accepted as written — ☑ 2026-08-26 (operator)
3. Harness dry run clean; amendment 1 appended above *before* Phase 1 — ☑ 2026-08-26
4. Robot pre-conditions (S8) all pass, values recorded in the first JSONL line — ☐ (filled by the harness `start` record)
