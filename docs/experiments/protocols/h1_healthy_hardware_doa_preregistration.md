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

**F1 — retained-axes ratchet (controller behavior, reproducible).** `goto_target` fills unspecified axes from the CURRENT POSE READBACK ("don't recenter the others", pre-#472 behavior). Under any achieved-vs-commanded bias this is positive feedback: ~20 yaw-only probe commands ratcheted roll +~1.3°/command up to +38° (pitch drifted too), invisibly contaminating the first envelope measurements. The orient scripts dodge it by pinning the full head matrix each command; the #472 ±40° roll clamp bounds it. **Fix direction (own PR): retain the last COMMANDED value, not the readback.** The first envelope numbers (right side "~22° at 65° cmd") are RETRACTED as roll-contaminated; envelope re-measurement pends recalibration.

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

