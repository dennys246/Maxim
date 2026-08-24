# H1 pre-registration — the first honest hardware measurement of the 1.0+ era

**Status:** PRE-REGISTERED 2026-08-07, before any hardware contact.
**Authority:** [roadmap_1_1_to_1_3.md](../../plans/roadmap_1_1_to_1_3.md) (1.1 item 6 + hardware note).
**This document must not be edited after the session starts** except to append results.

## Why this session is different

Motors in Stewart positions 2 and 3 were broken for essentially the entire
1.0+ era (replaced + reflashed ~2026-08-05). **Every live-hardware measurement
in that era was taken on a degraded platform** — including the 2026-07-16
"TRUE characterization" (0.57 az/rad, R²=0.9982), the 2026-08-05 contested
sweep (~0.19 az/rad), and Exp 45/45b/45c/45d/45e/46/48. The best current
hypothesis for the contested 3× gain discrepancy is **progressive mechanical
degradation**: 0.57 (July) and 0.19 (August) are both real measurements of one
platform in two states of decline.

**H1 is therefore not a re-run; it is the first baseline.** It buys four
things at once: Exp 45 staleness resolution, 1.3's Stage 0a, motor-binding
Phase 3's gain calibration, and the reconciliation of the CONTESTED banner in
[audio_localization.md](../../embodiment/reachy_mini/audio_localization.md).

## Hard preconditions (abort if any fails)

1. **The workspace-safety fold is MERGED to main and running on the session's
   checkout** (`fix/orient-workspace-safety`: controller clamps + motion lock,
   `turn_around`/`Selfy.move` bypass fixes). The destruction root cause must be
   closed before the platform is trusted with another session. Verify:
   `git log --oneline -5` includes the safety commit, and
   `python -c "from maxim.hardware.reachy.controller import ReachyMiniController as C; print(C._MAX_BODY_YAW_RAD)"` prints ≈2.79.
2. **Full motor health confirmed by the operator** — all six Stewart legs
   move cleanly after the motor-2+3 replacement/reflash. TWO motors were found
   broken; "one spare swapped" is not the same claim as "all healthy."
3. **Version match, recorded in the run artifacts** (the 2026-08-05 skew
   produced plausible-looking garbage on BOTH sensing and control):
   `curl http://<robot>:8000/api/daemon/status` version ==
   `python -c "from importlib.metadata import version; print(version('reachy-mini'))"`.
   Any mismatch → STOP, pin the client, re-verify.
4. **`automatic_body_yaw` off** (daemon must not rotate the frame behind the
   runtime; the controller asserts this on wake but verify in the log line).
5. **Provenance stamps in every run record:** `executed_git_hash`, SDK
   version, daemon version, source geometry label, room label. A measurement
   whose code-under-test cannot be established is not a measurement
   (Exp 42b lesson).

**Abort criteria during the session:** any motion-command rejection, any SDK
version RuntimeWarning, any audible/visible motor glitch or snap → power down,
stop the session, do not "push through."

## Protocol (the 2026-07-16 protocol, verbatim where possible)

### Part A — actuation verification FIRST (verify-the-instrument)

`scripts/orient_backbone/yaw_verify.py` with daemon-side pose readback:

- commanded→actual body travel ratio (07-16 healthy-era-belief value: 0.955;
  record small-step and large-step separately)
- `d(head)/d(body)` — must be ≈ +1.0 (head rides the body; the 2026-07-16
  head-frame fix). If this is not ≈1, STOP — every downstream number is
  frame-corrupted.
- **Head-relative yaw envelope** (added by the safety-fold review round):
  step the head-relative yaw command up from 0° in 5° increments (body held)
  and record the achieved relative yaw at each step until it stops tracking.
  This settles the 65°-vs-22° provenance conflict (vendor docs say 65°;
  Exp 49's ~22° was measured on the degraded platform) and calibrates the
  controller's `_MAX_HEAD_REL_YAW_RAD` capability clamp. If the healthy
  envelope is materially below 65°, tighten the constant citing this
  measurement.

### Part B — DoA sweep, ≥2 source geometries

`scripts/orient_backbone/doa_sweep.py`: body yaw swept ±1.4 rad in 0.1 rad
steps, head riding along, fixed sustained speech source, 5 speech-gated reads
per pose, ascending + descending passes.

- **Geometry 1:** source ~1–2 m in front of neutral heading (the 07-16 and
  08-05 geometry — required for comparability).
- **Geometry 2:** source displaced ≥45° in azimuth and/or a different range
  (separates sensor shape from room/source geometry — the 08-05 left/right
  asymmetry was never disambiguated).
- Optional Geometry 3 if time allows: different room or source height.

### Part C — motor-bound delivered-shift measurement

With the SEM orient affordances live (`turn_left/right[_big]` through
`ReachyOrientMotorBackend`): command each affordance N≥10 times from
centered pose, record commanded vs achieved body yaw (controller readback)
AND the measured azimuth transition per turn. This is motor-binding Phase 3's
gain-calibration input and 1.3's Stage 0a — collected in the same session.

## Pre-registered hypotheses and decision rules

**H1a (gain):** healthy hardware measures central gain **≥ 0.57 az/rad**,
plausibly nearer the geometric **0.637**.
**H1b (linearity):** R² ≥ 0.99, monotone across ±1.4 rad, both directions,
both geometries.

### Outcome tree (decided now, not after seeing data)

| Outcome | Reading | Action |
|---|---|---|
| gain ≥ 0.57, linear, both geometries | **Progressive-degradation hypothesis SUPPORTED.** 0.57 and 0.19 were two decline states. | Resolve the CONTESTED banner: annotate the 08-05 staircase as degraded-platform artifact (keep data, mark resolved). Update the magnitude boundary from the NEW gain (see H2 trigger). |
| gain ≈ 0.57 ± 0.05, linear | July was near-healthy; August badly degraded. Same resolution as above; boundary stands. | Close CONTESTED; magnitude claims move from "provisional" to "replicated at n=2 platform states" only where direction-based. |
| staircase (~13° sectors) reproduces at BOTH geometries on healthy motors | The staircase is real chip behavior, not degradation. | CONTESTED banner resolves the OTHER way; investigate firmware/shell/mounting delta since 07-16; magnitude policy redesign becomes 1.3 work. |
| gain < 0.57, no staircase, geometry-dependent | Room/source confound. | Third geometry mandatory; no doc updates until reconciled. |
| Part A fails (travel ratio « 0.95 or d(head)/d(body) ≉ 1) | Platform still not healthy or new regression. | STOP. No sweep data is interpretable. Hardware session ends. |

### Contingent H2 branch — pre-registered trigger (runs ONLY if fired)

The big-step decision boundary |az| ≈ 0.33 was derived from the 0.55–0.57
gain (`az_boundary` = nearest-neighbour quantization boundary of the available
shifts under that gain).

**Trigger:** the healthy-hardware gain implies a boundary that moves by
**> 0.03 az** from 0.33 (i.e. measured gain outside ≈ [0.52, 0.62]).

- **If fired:** run the magnitude re-probe (Exp 45b protocol) against the new
  boundary in the same or next session, and re-evaluate the Exp 45c flip-bins
  design against the new bin edges. YAML magnitudes may then be retuned — with
  the new sweep as the cited source.
- **If not fired:** H2 does NOT run. The boundary stands; Exp 45b/45c
  magnitude rows in the graduation walk stay **provisional** (their data came
  from the degraded platform) but are not re-measured in 1.1.

**No other analysis may be promoted to a claim from this session** — anything
interesting-but-unregistered spawns a new pre-registered follow-up (post-hoc
finding discipline, CLAUDE.md working principles).

## Graduation-walk consequence (item 5 of the 1.1 cut line)

Whatever the outcome: every **magnitude** claim in the walk is marked
provisional-pending-this-session; **direction** findings (sign-based, robust
to proportional gain error) are evaluated on their own evidence. H1's result
is the citation the Exp 45-family rows need to move at all.

---

# RESULTS — appended 2026-08-08 (session 1: Parts A + B complete; envelope + Part C pending recalibration)

**Provenance:** `executed_git_hash 38aaddea` (main incl. the #472 safety fold), SDK 1.8.3 == daemon 1.8.3 (verified), `automatic_body_yaw` off, operator-confirmed motor swap. Raw data: [data/h1_doa_sweep.jsonl](../data/h1_doa_sweep.jsonl) (3 runs x 58 points) + [data/h1_daemon_state_precal.json](../data/h1_daemon_state_precal.json).

## Part A — actuation verification: PASS (with one flagged figure)

- **d(head)/d(body) = +1.024** — the mics ride the body; the head-frame fix holds on the repaired platform. THE gate, passed decisively.
- Travel ratio 0.934 mean — **0.951 small steps / 0.865 large steps**, the duration-limit signature. Small-step figure matches the 08-05 matched-stack value (0.961); sweep steps are small-regime, so Part B is uncontaminated. The large-step figure is now understood as an early symptom of the calibration finding below.

## Part B — DoA sweep, two geometries + an accidental replication: PASS, top branch of the outcome tree

Fold-model fit (az = g*fold(psi − psi0)) on medians:

| run | source psi0 | gain g | R^2 | per-side g (right/left) |
|---|---|---|---|---|
| geom 1 (front) | −0.14 | **0.578** | 0.9944 | 0.586 / 0.564 |
| front replication* | −0.12 | **0.575** | 0.9957 | 0.585 / 0.559 |
| geom 2 (~63° right; fold captured) | −1.09 | **0.645** | 0.9843 | — (fold-region leverage) |

*The first "geom2-displaced"-labelled run was a placement misunderstanding — the source stayed in front. Re-classified here as a same-geometry replication (the label in the JSONL is wrong; the data is what it is and the zero crossing proves placement). The true displaced run follows it in the file.

**Outcome-tree verdict — the TOP branch fires:** gain >= 0.57 at both geometries, R^2 0.98–0.996, monotone in the fold domain, **no staircase at either geometry** (the geom-2 curve folds at rel. bearing ±pi/2 — linear-array physics, matching the sim's honest-physics model — which geometry 1 could never expose).

- **Progressive-degradation hypothesis SUPPORTED.** 0.57 (2026-07-16) and 0.19 (2026-08-05) were two decline states of the failing motors 2+3. The 08-05 staircase was a degraded-platform artifact. The CONTESTED section in [audio_localization.md](../../embodiment/reachy_mini/audio_localization.md) is resolved accordingly (annotated in place, data retained).
- **H2 does NOT fire.** Policy-relevant (front-geometry, matching the conditions the 0.33 boundary was derived under) gain 0.578 ∈ [0.52, 0.62]; implied boundary shift ~0.01 az < 0.03. Magnitude YAML stays frozen. The geom-2 fit at 0.645 (~geometric 0.637) is recorded as a geometry/fold-region dependence observation — post-hoc, spawns nothing without its own pre-registration.
- Per-side gains near-symmetric (~4% right-favoring, both front runs, same direction with the source moved → leans sensor/mounting, mild). The fitted psi0 ≈ −0.13 rad found a likely cause the same evening — see finding F2.

## Two unplanned findings (post-hoc — each spawns follow-up work, neither is promoted to a claim)

**F1 — retained-axes ratchet (controller behavior, reproducible).** `goto_target` fills unspecified axes from the CURRENT POSE READBACK ("don't recenter the others", pre-#472 behavior). Under any achieved-vs-commanded bias this is positive feedback: ~20 yaw-only probe commands ratcheted roll +~1.3°/command up to +38° (pitch drifted too), invisibly contaminating the first envelope measurements. The orient scripts dodge it by pinning the full head matrix each command; the #472 ±40° roll clamp bounds it. **Fix direction (own PR): retain the last COMMANDED value, not the readback.** The first envelope numbers (right side "~22° at 65° cmd") are RETRACTED as roll-contaminated; envelope re-measurement pends recalibration. *(Fix LANDED 2026-08-09 — see the status ledger row below for the mechanism + regression guard.)*

**F2 — motor-zero miscalibration on the repaired platform (the session's operative discovery).** Six consecutive explicit all-zero commands plateau at **yaw +6.2°, roll +18.2°, pitch +9.2°** (stable to ±0.2°); the daemon's OWN state agrees exactly (roll 0.3197 rad — [data/h1_daemon_state_precal.json](../data/h1_daemon_state_precal.json)), ruling out any Maxim-side readback error. The platform has a fixed attractor that is not the commanded zero — consistent with the replacement motors 2+3 having been reflashed from MOTOR-1's config (zero-offsets are per-motor calibration; copying skips it). Constant offsets shift curves but not slopes, so Part A/B conclusions stand (and the fitted psi0 ≈ −0.13 rad ≈ −7° now has a candidate cause: this attractor's yaw component). **Operator action: per-motor zero calibration for motors 2+3, then re-verify zero, then envelope + Part C.**

## Status ledger

| item | state |
|---|---|
| Part A (yaw_verify) | **PASS** |
| Part B (2 geometries + replication) | **PASS — top branch; CONTESTED resolved; H2 does not fire** |
| Neck envelope | PENDING recalibration (first attempt retracted per F1) |
| Part C (delivered shift) | PENDING recalibration |
| Exp 45 graduation row | stays **Stale** until calibration + Part C complete; sweep-half evidence recorded on the row |
| Exp 50 amendment | g_H = 0.578 fillable now; envelope constant pends |


---

# RESULTS — session 2 appended 2026-08-08 (recalibration + envelope + Part C: H1 COMPLETE)

**Provenance:** same stack as session 1 (`38aaddea`, SDK==daemon 1.8.3). Between
sessions the operator ran the per-motor zero calibration for the replaced
motors 2+3 (the F2 fix). Part C ran on big-mac-mini (48GB) with qwen2.5-32b
local; distilled data: [data/h1_partc_summary.json](../data/h1_partc_summary.json)
(39 turns + 12 measured-credit events; raw 55MB JSONL retained on the mini).

## F2 recalibration: CONFIRMED FIXED

Explicit all-zero command post-recal lands at **rel yaw −0.7°, roll −2.3°,
pitch +2.9°** (was a stable +6.2/+18.2/+9.2 attractor). `yaw_verify` improved
exactly as F2 predicted: travel ratio 0.934 → **0.951**, small-step 0.951 →
**0.975** (large-step ~unchanged 0.855, the duration-limit component).
`d(head)/d(body) = +1.018` — holds.

## Neck envelope (roll/pitch pinned, level platform): ~±50° delivered

| cmd | achieved (right) | cmd | achieved (left) |
|---|---|---|---|
| 15° | 10.0° | −15° | −6.9° |
| 30° | 24.2° | −30° | −21.5° |
| 45° | 42.9° | −45° | −36.7° |
| 65° | **54.8°** | −65° | **−48.0°** |

Roll stayed within ±4° the whole run (no ratchet under explicit pinning). The
**session-1 retraction is superseded with an explanation**: the "~22° envelope"
was measured on the pre-recal miscalibrated platform under accumulated roll —
NOT a capability. Resolution of the 65°-vs-22° conflict: **65° = vendor
command ceiling (daemon soft-saturates gracefully, verified live); ~±50° =
true delivered envelope on a healthy level platform; the Exp 49 sim limit of
22° reflects the miscalibrated platform and under-states healthy hardware**
(sim-config revisit noted, not churned here). Mild right>left extreme
asymmetry (54.8 vs 48.0) and ~7-8° came-from-side return lag (backlash-
flavored) recorded. Controller clamp constants unchanged — the 65° command
ceiling is inside the daemon's gracefully-handled range.

## Part C — motor-bound delivered shift: MEASURED (normal arms n=37; _big n=8/side — block appended below)

39 turns through the PRODUCTION affordance path (SEM motor binding →
controller → daemon), LLM-driven (qwen2.5-32b), 12 measured-credit events:

| affordance | n | delivered ratio | toward-center | away-center |
|---|---|---|---|---|
| turn_left | 19 | 0.890 ± 0.042 | 0.923 | 0.867 |
| turn_right | 18 | 0.937 ± 0.075 | 0.984 | 0.899 |
| turn_left_big | 1 (session 2) → **8 (block, 2026-08-24)** | 0.979 → **0.943 ± 0.001** | — | — |
| turn_right_big | 1 (session 2) → **8 (block, 2026-08-24)** | 0.932 → **0.942 ± 0.003** | — | — |

- **Position-dependent load confirmed statistically:** turns toward center
  deliver ~0.92–0.98 of commanded; away from center ~0.87–0.90. This is the
  structure motor-binding Phase 3's gain calibration should absorb (a single
  scalar gain under-fits it).
- **Right>left actuation asymmetry ~4%** (0.936 vs 0.895) — same direction
  and similar magnitude as the sweep's per-side sensor tilt; whether they are
  one phenomenon (actuation) or two is a post-hoc question for a
  pre-registered follow-up, not claimed here.
- **The measured-credit chain behaved correctly on hard cases:** negative
  credit for azimuth-worsening turns (−0.018/−0.035/−0.040), and a
  fold-crossing transition (az −0.374 → +0.367) credited +0.007 — near-zero,
  honest — instead of a fabricated large relief. Zero events
  `nulled_by_collateral`. Credit sparsity as designed: 12 measured events for
  39 turns, the gaps being speech-silence windows (source had dead air).
- **The `_big` arms are n=1 each** (both fired and measured — turn_right_big
  Δaz −0.373 for one ~48° delivered swing). A dedicated ~8-rep-per-side block
  with continuous audio is the remaining follow-up; NOT blocking (the normal
  arms carry the Phase-3 calibration need; the _big YAML magnitudes stay
  frozen per H2-did-not-fire).

## Part C follow-up — the `_big` delivered-shift block: MEASURED 2026-08-24 (n=8/side)

**Provenance:** operator Mac, `main` at `b01a6589` (+ the untracked block harness),
SDK == daemon **1.8.3**, hardware_id `1c5d3b8f935996af`. Gate step 1 re-verified
first: `yaw_verify` travel ratio **0.950**, d(head)/d(body) **+1.005**. Harness:
[`scripts/orient_backbone/delivered_shift_block.py`](../../../scripts/orient_backbone/delivered_shift_block.py)
— drives the PRODUCTION affordance path (`SpecModulator.execute` →
`ReachyOrientMotorBackend` → `ReachyMiniController.goto_target`), reads the
achieved rotation from the daemon's own `/api/state/full` body_yaw (never the
commanded value), and records TWO azimuth transitions per turn: the backend's own
frame-corrected before/after pair (the number the credit path uses) and an
independent later-window median-of-5. Continuous speech source dead ahead
(az ≈ +0.09), speech-gate rate 0.80–0.83. Raw: `data/h1_partc_big_block.jsonl`,
admitted `run_id 20260824T213553Z-79752` (16/16 turns, all `success`/`reached`,
none clamped; an earlier `20260824T213320Z-76884` aborted before its third turn on
a daemon ack timeout and carries a `block_aborted` marker that was appended BY
HAND before run 2 — the harness's retry-once recenter and generic-abort paths
were added after that run and have not yet been exercised on hardware — exclude
it). Gate-step-1 numbers (`yaw_verify`) are from the console, unarchived.

| affordance | n | commanded | achieved (daemon) | delivered ratio | Δaz backend | Δaz script | az/rad (backend) |
|---|---|---|---|---|---|---|---|
| turn_left_big | 8 | +51.6° | **+48.63° ± 0.04** | **0.943 ± 0.001** | **+0.489 ± 0.069** | +0.515 ± 0.049 | 0.576 ± 0.081 |
| turn_right_big | 8 | −51.6° | **−48.57° ± 0.14** | **0.942 ± 0.003** | **−0.400 ± 0.078** | −0.399 ± 0.063 | 0.472 ± 0.093 |

- **Delivered shift MEASURED at n=8/side in ONE session:** 0.943 of command on BOTH
  sides — the `_big` step delivers ~48.6° for a 51.6° command, the same ~0.94–0.95
  travel ratio `yaw_verify` reports. The ± values in this section are population
  SD (n=8) of 8 autocorrelated re-reads of one platform state from a centered pose
  — a within-session spread, NOT a cross-session estimate (L8 applies; cross-
  session replication is outstanding). Session 2's n=1 values (0.979 / 0.932)
  were LLM-driven turns from arbitrary poses under position-dependent load — a
  different condition, not draws of this distribution.
- **The YAML `_big` magnitude (`azimuth: ±0.50`) holds on the LEFT and over-states
  the RIGHT by ~20%:** measured +0.49 vs −0.40. Both estimators agree (backend and
  script, sign agreement 16/16). Delivered BASE rotation is symmetric (48.63° vs
  48.57°), which rules out base actuation — and nothing more: head-pose drift
  during the block (D30, roll to −8.7°), head ride-along on the production path
  (NOT measured in-block — the harness recorded body yaw only; it now records
  world head yaw/roll per turn for the next run), sensor tilt (session 2's ~4%
  right>left, the sweep's per-side tilt) and source placement (az_pre ≈ +0.08,
  range 0.01–0.13 — slightly right of center) all remain candidates. **YAML
  magnitudes stay frozen:** H2 did not fire, retuning is reserved for a fired H2
  with a sweep as source, and this is a post-hoc observation — a mirrored-source
  repeat that records head pose per turn is **to be pre-registered** (not yet
  filed), not a retune.
- **Instrument finding 1 — a folded reading reached the production credit path
  once in 18 turns.** In the aborted run's record `i=1` (0-based turn index; a
  right turn, body −0.84 rad) the backend's first post-settle window returned
  +0.289 while the later median-of-5 read −0.289 — an exact mirror, i.e. an L10
  sign-flip landing in `measured_drive_transitions` and therefore in the credit
  sign. 0/16 in the admitted run. Recorded under [L10](../../limits/README.md)
  (the observation extends L10's onset from `|psi| ≳ 1.0` down to `|body| = 0.84`
  rad with the head riding along) and filed as **D31** — the credit path has no
  fold guard.
- **Instrument finding 2 — head roll drifts under repeated body-only `_big`
  commands.** The controller's F1 early-warning fired repeatedly on recenters:
  achieved head_roll diverged from the last commanded by −5.3° → −8.7° and stayed
  there. The backend commands `body_yaw` only, so roll is not pinned (session 2
  pinned roll/pitch explicitly for the envelope test and saw ±4°). Magnitudes are
  from the operator console — the harness did not yet record head pose per turn
  (it does now). Filed as bugs-ledger D30; not a Part C result.

## Session-2 operational lessons (recorded for the runbook)

Live-session bring-up on a NEW machine hit, in order: reachy-mini SDK absent
(pin EXACTLY to the daemon version — a loose `>=` re-opens the 08-05 skew);
`robots.yaml` absent (each machine's `~/.maxim` is its own); macOS Local
Network TCC denying python with a masquerading `EHOSTUNREACH` while
Apple-signed curl passes (grant via a GUI terminal once); `media_backend:
"default"` in ad-hoc controller constructions connecting media channels that
refuse under no_media daemons (pass `no_media` explicitly, per #456); bare
`maxim` menu option 1 is the SIM stack, not the live runtime (`maxim --llm
<model>` is the live path); and multi-line pastes into `maxim>` queue as
separate inputs and scramble tool selection.

## H1 status: COMPLETE

| item | verdict |
|---|---|
| Part A | PASS (both sessions; post-recal improved) |
| Part B | PASS — top branch; CONTESTED resolved; H2 does not fire |
| Neck envelope | **~±50° delivered / 65° command ceiling** — conflict resolved |
| Part C | Normal arms MEASURED (n=37); **_big MEASURED n=8/side (2026-08-24 block, below)** |
| F1 retained-axes ratchet | **FIXED** (2026-08-09, `fix/goto-target-retained-axes-ratchet`): `goto_target` retains the last COMMANDED value per axis; readback seeds once; raw movers invalidate via `note_external_head_motion()`. Guard: [tests/unit/test_reachy_retained_axes.py](../../../tests/unit/test_reachy_retained_axes.py) (fails on the pre-fix controller) |
| F2 motor-zero miscalibration | FIXED by per-motor recalibration, verified |
| Exp 45 graduation row | **UN-STALED** (see row for the _big caveat) |
| Exp 50 amendment | FILLED (g_H=0.578, envelope ±50°, boundary unchanged) |
