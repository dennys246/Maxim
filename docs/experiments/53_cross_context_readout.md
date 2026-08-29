# Exp 53 — Cross-context readout: the nursery-taught want on the physical Reachy Mini

**Status:** COMPLETE 2026-08-26. Exp 53 — pre-registered verdict **APPARATUS** (direction
transferred 36/36; the declared step size did not fit the small target). **Exp 53b** (same
session, one declared change: δ = the robot's own step) — **PASS: cross-context transfer
EARNED.** The nursery-taught want reads out on the physical robot.
**Replication R1, 2026-08-28** — the same protocol re-run at tag `v1.1.0` from a CLEAN
worktree after the 1.1.0 re-score found the original runs' provenance unestablishable (see
§Replication R1): **PASS — taught 1.00 / 1.00 / 1.00, satiated 0.00, no_feed 0.50**. The
EARNED status now rests on R1; the original 53/53b records stay as recorded.
**Pre-registrations (frozen):**
[protocols/exp53_cross_context_readout_preregistration.md](protocols/exp53_cross_context_readout_preregistration.md)
(amendments 1–2, pre-data) ·
[protocols/exp53b_cross_context_readout_delta_preregistration.md](protocols/exp53b_cross_context_readout_delta_preregistration.md).
**Harness:** [scripts/orient_backbone/exp53_cross_context_readout.py](../../scripts/orient_backbone/exp53_cross_context_readout.py).
**Raw data (S4):** [data/53_cross_context_readout.jsonl](data/53_cross_context_readout.jsonl)
(Exp 53: Phase 1 + Phase 2 primary) · [data/53b_cross_context_readout.jsonl](data/53b_cross_context_readout.jsonl)
(Exp 53b: Phase 1 + Phase 2 primary + secondary) — both `executed_git_hash 68f9026e268f`
**with `working_tree_dirty_src_scripts: true` on every start record, and `68f9026e` is a
branch commit NOT reachable from `main` or `v1.1.0` (squashed into `617b1625`, #551) — the
code those two runs executed cannot be established from the repository** (disclosed
2026-08-28; the 1.1.0 re-score's finding, [lesson](../lessons/experiment-prereg-precedes-data.md)) ·
**R1 (2026-08-28):** [data/53b_cross_context_readout_replication_2026-08-28.jsonl](data/53b_cross_context_readout_replication_2026-08-28.jsonl) — `executed_git_hash df881b87f305` (= tag
`v1.1.0`), `working_tree_dirty_src_scripts: false` on all 5 start records ·
SDK == daemon 1.8.3, hardware `1c5d3b8f935996af`, operator Mac on the robot's LAN · [data/53_agents/](data/53_agents/) — the ten agents'
`aut_nac.json` + `aut_ec.json`, byte-for-byte from the Exp 52 Phase B archive, SHA-256 in
[data/53_agents_manifest.json](data/53_agents_manifest.json) · dry run (harness
verification, not a result): [data/53_dry_run_nonfrozen.jsonl](data/53_dry_run_nonfrozen.jsonl).

## The question

The 1.1 claim is learning that carries across sessions **and contexts** without
fine-tuning. Cross-session was earned in sim (Exp 42, 45); cross-context — the same
learned state driving a different body in a different world — had never been shown.
So: take the Exp 52 infants' persisted NAc + EC, load them **unchanged** (no credit, no
decay, files SHA-verified before and after) into the production substrate-primary path
with the robot's live DoA as the azimuth sensor and its body yaw as the turn, and ask
whether the taught infants turn toward a speech source while the zero-bias controls,
loaded identically, do not.

## What the instrument check found (Phase 1, no motion) — GATE I PASS

- **Pattern completion 60/60.** Every live azimuth, for all ten agents, completed into
  a cluster the nursery had built; none separated into a new node.
- **The nursery's learned map** (found in the pre-data dry run, amendment 1): three
  `audio` clusters partition the axis FAR-LEFT (≤ −0.5) / CENTRE (−0.4…+0.3) / RIGHT
  (≥ +0.4), identically for seeds 42/43/44, with `turn_right +0.90` on RIGHT and
  `turn_left +0.59–0.65` on CENTRE (operant credit keys on the decision-time cluster;
  the last action before a feed from a left stimulus is taken from the centre bin). The
  original ±0.5/±0.6 targets would have probed the weak far-left bin (`+0.006`); the
  gated targets moved to {−0.3, −0.2, +0.5, +0.6}, with −0.6 and +0.2 kept as
  exploratory placements.
- Taught 43 and 44: 4/4 correct with margin (0.59–0.90); taught 42: 3/4 — the miss was
  a dropped goto ack (measured az +0.13 against a +0.5 target, centre bin → `turn_left`),
  scored as a miss because the pre-registration has no clause to excuse it. 2/3 ≥ the
  gate. Controls: satiated 42/43/44 return no tool at every placement; no_feed 42/43/44
  return `turn_right` everywhere with learned bias 0 (persisted causal-link credit,
  side-blind). Seed 48 (exploratory): wrong-way at 4/6 — its nursery map is different
  (`turn_right +0.31` on CENTRE); the weak learner mis-learned, not under-learned.
- Amendment 2 (pre-data, robot session): the S8 speech-gate floor moved from 0.70/10 s
  to H1's 0.50/30 s after the chip's own speech energy was shown non-zero 0.86 of the
  time while the VAD flag ran 0.39–0.73 across three continuous sources.

## Phase 2, primary block (explore 0.0, 12 gated trials per agent) — APPARATUS

| arm (seeds 42/43/44) | delivered directedness | chosen direction correct | no-action |
|---|---|---|---|
| taught | **0.75 / 0.75 / 0.75** | **36/36** | 0 |
| satiated | 0.00 | — | 36/36 |
| no_feed | 0.50 | `turn_right` 54/54 (side-blind) | 0 |

By gated target, taught: −0.3 → 9/9, **−0.2 → 0/9**, +0.5 → 9/9, +0.6 → 9/9. Every miss
is the same event: the −0.2 target delivers at −0.12…−0.13, the declared δ = 0.55 rad
step moves ≈ +0.30 az (achieved 0.964 of command), and the head lands at +0.17…+0.19 —
right direction, `|after| > |before|`. Exploratory placements: −0.6 → 9/9 toward (the
weak bin still turns left), +0.2 → 9/9 wrong-way, exactly as amendment 1 predicted.

Gate T: LEARNED-LIVE 0.75 ≥ 0.70 ✓; taught − satiated +0.75 ✓; taught − no_feed +0.25 ✓;
**sign-rule agreement 27/36 = 0.75 < 0.80 ✗** and three seeds at an identical 0.75 ✗ →
the pre-registered **APPARATUS** verdict: the delivered result is a geometry artefact
of the declared step, and no verdict on the claim is taken from this block. The 36/36
direction readout is the measured fact that motivated 53b; it is not argued into a PASS.

S3: `credited` 0 for every agent; every agent's files unchanged (SHA-256). Hardware:
head-pose drift under repeated body commands (D30) larger than H1 measured — roll
14–19°, pitch 3–11° — with the controller's divergence warning on nearly every goto;
it did not disturb pattern completion. Two placement misses in 60 probes (one dropped
ack, one sign flip — D31 class).

## Exp 53b — δ = 0.30 rad (the `reachy_mini` body's own step) — PASS

Same session, same ten files (SHA-verified), same targets and gates; the one
pre-registered change is the step size. Speech-gate probes 0.64 (Phase 1) / 0.55 (Phase 2).

**Gate I: PASS 3/3** — 60/60 pattern completion again; every taught seed 4/4 correct with
margin; controls no learned preference (satiated: no tool; no_feed: `turn_right`, bias 0).

| arm (seeds 42/43/44) | delivered directedness | chosen direction correct | no-action |
|---|---|---|---|
| taught | **1.00 / 1.00 / 1.00** | **36/36** | 0 |
| satiated | 0.00 | — | 36/36 |
| no_feed | 0.50 | `turn_right` 54/54 (side-blind) | 0 |

By gated target, taught (median az before → after): −0.3: 9/9 (−0.22 → −0.04); **−0.2:
9/9 (−0.13 → 0.00)**; +0.5: 9/9 (+0.56 → +0.37); +0.6: 9/9 (+0.66 → +0.44). Exploratory:
−0.6 → 9/9 toward (−0.43 → −0.31); **+0.2 → 0/9, `turn_left` 9/9 (+0.23 → +0.41)** — the
centre bin's right half turns the wrong way, as amendment 1 predicted before any robot
data. Achieved step 0.296 rad (0.987 of command). 180 trials, 0 invalid reads.

**Gate T: PASS** — LEARNED-LIVE 1.00 ≥ 0.70; taught − satiated +1.00 ≥ 0.20; taught −
no_feed +0.50 ≥ 0.20; sign-rule agreement 36/36 = 1.00 ≥ 0.80; three seeds at ceiling
(S7). S3: `credited` 0 everywhere, every agent's files unchanged. Seed 48 (exploratory):
0/12 — the mis-learned map reads out as mis-learned, which is itself evidence the readout
is faithful. Head drift (D30): roll 13–16°, pitch 6–9°. **Secondary block (explore 1.5,
reported, not gated) — recorded:** taught 0.75 / 0.75 / 0.75, satiated 0.28, no_feed 0.42
(180 trials, 0 invalid, files unchanged). The taught arm's chosen direction was still
36/36 with `explore_decisive` 0/36 — the learned bias decided every trial even at
explore 1.5 — and all nine misses are again the −0.2 target, but for a different reason
than Exp 53's: over the session the delivered geometry drifted ≈ +0.07 az (the −0.2
target read −0.06 median in this block vs −0.13 in the primary; every other target
shifted the same way — D30 head drift and/or source creep), so the source was already
inside the ±0.05 band and any step counted as overshoot. The controls now act (the
exploration term admits their tools) and random-walk at 0.28 / 0.42, as amendment 1
predicted.

**Verdict per the 53b outcome tree: cross-context transfer EARNED**, with Exp 53
recorded beside it as the apparatus finding that motivated the step size.

## Replication R1 — 2026-08-28, tag `v1.1.0`, clean tree — PASS

**Why.** The 1.1.0 repository re-score ([score card](../limits/score_cards/2026-08-27-claude.md))
found that the Exp 53 and 53b records above were executed from a dirty working tree at a
commit unreachable from the tag, with the 53b pre-registration reaching `main` only in the
same squash as its data — by the Exp 42b standard, a result whose code-under-test cannot be
established is not a validation. R1 is the same frozen protocol re-run where every provenance
question has a git answer: worktree `.worktrees/exp53b-replication` at `v1.1.0` (`df881b87`),
`git status --short src scripts` empty, `PYTHONPATH` absolute, the harness's own in-process
provenance check passing (it exits 3 if the imported `maxim` is not that tree's `src/`).

**What did not change.** Agents (the ten `53_agents/` pairs, SHA-256 re-verified before and
after every load), δ = 0.30 rad, gated targets {−0.3, −0.2, +0.5, +0.6} + exploratory
{−0.6, +0.2}, 12 gated + 6 exploratory trials per agent, `--condition primary`, speech-gate
floor 0.50 / 30 s, the harness at the tag. Source: phone playing a podcast, 1–2 m, in front
of the base's neutral heading, taped down. Nothing was tuned.

**Deviation, disclosed.** The 53b pre-registration says "53b runs once"; R1 was run under the
row's Re-run triggers without an amendment merged beforehand (the prereg-precedes-data rule
landed the same day, #566). Protocol, gates, agents, δ and targets are the frozen ones; the
only new declared requirement is clean-tree provenance. Future replications merge a
one-paragraph amendment first.

**Phase 1 — GATE I PASS** (run `20260828T144251Z`, speech-gate 0.65 over 177 reads): 3/3
taught seeds complete + correct with margin (60/60 placements pattern-complete into the
nursery's clusters); satiated 0/18 acted; no_feed acted but `no_learned_preference` on
every seed; seed 48 (exploratory) wrong as its own map predicts. The +0.2 probe turned
`turn_left` for every taught seed — the predicted CENTRE-bin wrong-way, excluded from the gate.

**Phase 2 — GATE T PASS** (final complete run `20260829T011136Z`, speech-gate 0.63 over 176
reads, 180 trials, 01:12–01:50 UTC):

| arm (seeds 42/43/44) | gated trials toward | sign rule | exploratory +0.2 | exploratory −0.6 |
|---|---|---|---|---|
| taught | **36/36 = 1.00** (12/12 per seed) | 36/36 | `turn_left` 9/9 → wrong-way 9/9 | toward 9/9 (margin 0.006–0.007) |
| satiated | 0/36 = **0.00** (no action 54/54) | — | no action | no action |
| no_feed | 18/36 = **0.50** (`turn_right` 54/54, margin 0.0) | — | toward 3/3 | wrong 3/3 |
| taught seed 48 (exploratory) | 0/12 | — | toward 3/3 | toward 3/3 |

Delivered geometry: targets −0.3 / −0.2 / +0.5 / +0.6 read −0.22 / −0.13 / +0.57 / +0.67 at
every placement — the same ≈ +0.07 az offset the original 53b block noted, present from the
first trial and **stable across the block** (no drift; the original's secondary block drifted
to −0.06). Every taught `turn_left` from −0.13 landed at +0.00 / −0.02; every `turn_right` from
+0.67 at +0.46. Verdict JSON: `learned_live`, `taught_minus_satiated_ok`,
`taught_minus_no_feed_ok`, `taught_seed_spread` all true (three seeds at 1.00 = PASS under
amendment 4's ceiling clause); `taught_sign_rule_agreement` 1.0.

**The session record, in full** (S3 — nothing deleted; every start carries the same
provenance). The JSONL holds five `start` records: a partial Phase 1 (`20260828T143101Z`,
2 agents / 9 probes, stopped), the complete Phase 1 above, a partial Phase 2
(`20260828T145949Z`, 3 trials, stopped), a partial Phase 2 (`20260828T181211Z`, seed 42
complete + seed 43 to trial 17, ended by `Motion command rejected by controller`; one
trial with `arm: null` recorded as an error), and the complete Phase 2 above. One further
Phase 2 attempt was refused before its start record at a speech-gate rate of 0.49
(podcast pause; re-run unchanged at 0.63). **`verdict` pools every primary trial in the
file** — there is no run-id filter — so the printed 1.00 / 0.00 / 0.50 aggregates 219 trials
across the three Phase 2 starts; the final complete run alone is 36/36 · 0/36 · 18/36, the
same three numbers. Phase 1 and the final Phase 2 are ~10.5 h apart on the same local day
(the pre-registration says "one session"; the harness gates Phase 2 on the same file, not
on time); deliveries shifted +0.03…+0.06 between phases, the phone untouched.

**Apparatus notes.** During Phase 1 the first `goto` was rejected (`time value is out of
range [0,1]`), re-issued once, and the controller then reported a `body_yaw −57.1°`
achieved-vs-commanded divergence (H1 F1 early warning) — a one-time event; no `body_yaw`
divergence recurred in Phase 2. `head_roll −7°` / `head_pitch −9°` divergence warnings fired
on nearly every apparatus move, **constant, not progressive** — D30 (head drifts under
body-only turns), not degradation. These warnings are console-only; the JSONL carries
`head_pre` / `head_post` per trial but not the F1 verdict (harness gap, below).

**Harness gaps found by R1** (not defects in the result): (1) `verdict` should select the
last complete run or refuse on multiple `start` records — pooling partial starts happens to
be harmless here and would not be in general; (2) the H1 F1 early warning should be a
record, not a print; (3) a partial Phase 2 start leaves no `agent_done` for its last agent,
so the file cannot say whether a run was interrupted or crashed. Filed against the harness
rather than the experiment.

**What R1 changes.** The ledger row's EARNED status now rests on a record whose code is the
released 1.1.0 tag and whose tree was clean. The original 53/53b records are retained as the
first observation with their provenance caveat stated. Scope is unchanged: readout, not
learning; one room, one source, n = 3/arm, fixed step, front hemisphere.

## Demo (not evidence)

[Video](https://youtu.be/lLoPM2EkbPU) — the operator sneaks up on the robot and speaks;
`taught_seed43` glances toward the voice, `satiated_seed43` loaded identically sits still.
Shot with [scripts/orient_backbone/exp53_demo_readout.py](../../scripts/orient_backbone/exp53_demo_readout.py)
(same loaded files and production path, no apparatus, one glance per speech onset,
δ = 0.9 rad; records stamped `evidence: false`). The numbers above come from the
pre-registered blocks, not from this footage.

## What this shows — and does not

- **Shows:** learned state written by a nursery sim — two JSON files, loaded unchanged,
  never credited on the robot — drives a physical body toward a speech source on 36/36
  trials, while the never-hungry and never-fed infants loaded identically do nothing or
  turn one way regardless of side. The representation completes on live percepts
  (120/120 across both experiments) and the learned direction reads out at every gated
  placement. That is the cross-context half of the 1.1 claim: learning that carries
  across sessions AND contexts without fine-tuning. Exp 53 shows the same direction
  readout (36/36) with a step size that overshot the smallest target — the miss was a
  declared apparatus parameter, and 53b changed only that.
- **Does not show:** learning on the hardware (nothing credits the NAc here); anything
  about magnitude selection or loudness; generalisation beyond the taught geometry —
  the centre bin's right half (+0.2) turns the *wrong* way, the representation's stated
  limit. One session, one room, one source, 3 seeds per arm.
