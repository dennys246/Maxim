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
