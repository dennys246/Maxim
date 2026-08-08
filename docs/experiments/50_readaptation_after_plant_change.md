# Experiment 50 — Re-adaptation after plant change (savings + learned non-use)

**Status:** PRE-REGISTERED 2026-08-07, before any hardware contact. **Runs
STRICTLY AFTER [H1](protocols/h1_healthy_hardware_doa_preregistration.md)** —
the healthy-hardware baseline (gain `g_H`, neck envelope) is this experiment's
reference frame; running before H1 re-creates the exact "policy re-adapting vs
sensor reads differently" confound the phantom-DoA retraction taught us to kill.
**This document must not be edited after trial 1** except (a) the single
pre-declared H1-constants amendment (§Amendment slot) and (b) appended results.

## The question, framed honestly

Motors 2+3 were degraded for the entire 1.0+ era and have been replaced. The
NAc orient policies trained in that era (Exp 45/45b/45c/45d/45e) did not
"forget" anything — **the policy is frozen-perfect; the PLANT it was calibrated
to is gone.** The correct bio-analog is therefore NOT memory loss from injury;
it is the motor-adaptation literature's core paradigm — internal-model
recalibration after altered limb dynamics (force-field / prism adaptation) —
plus one structural bonus: the bounds learner tightened workspace limits in
response to pain on the degraded platform, which is **learned non-use** (the
phenomenon constraint-induced movement therapy treats: protecting a limb that
has already healed).

Predictions come from that literature, not from vibes:

- **Savings:** relearning from the degraded-era policy is faster than learning
  from scratch (retention of structure survives plant change).
- **Direction transfers; magnitude arrives miscalibrated in a KNOWN direction:**
  direction is sign-based and survives a gain change; magnitude bin boundaries
  were derived from the degraded gain, so with healthy gain `g_H` a given
  physical offset reads a different azimuth and the loaded policy's step-size
  choices misfire predictably (if `g_H > g_D`, offsets read LARGER → excess
  big-turns → overshoot).
- **Learned non-use persists absent a recovery mechanism** — an honest
  structural finding either way (see P4).

## Front-gate (Principle 3)

No new mechanism. Rides, unchanged: NAc persistence + `cluster_reward_bias` /
`recommend_action`, `maxim substrate merge-nac`, the Exp 45 live harness
([scripts/orient_backbone/live_3_learn.py](../../scripts/orient_backbone/live_3_learn.py)
with its `--perturb` apparatus + contamination guard), the saved degraded-era
artifacts, and the bounds learner as a read-only measurement surface. If P4
exposes a bounds-recovery gap, that is RECORDED as a mechanism gap with a
revival trigger — not fixed mid-experiment (1.1 zero-new-mechanisms rule;
fixing it would also unblind the arm that motivated it).

## Hard preconditions

1. **H1 COMPLETE with Part A passed** (travel ratio, `d(head)/d(body)` ≈ +1.0,
   neck envelope measured) and the sweep reconciled per H1's outcome tree.
   `g_H` = H1's measured central gain.
2. Workspace-safety fold (PR #472) on the running checkout; version match
   recorded; `automatic_body_yaw` off — same checks as H1, same abort criteria
   (any motion rejection / version warning / motor glitch → stop).
3. **Degraded-era artifacts verified present + provenance-checked before any
   session:** the Exp 45d seed-2 NAc (the 1.00/1.00 direction+magnitude
   policy) and its `*.meta.json` policy sidecar (bin boundary 0.330, gain
   0.55, action deltas — the state-space definition MUST travel with the
   policy per the merge-nac sidecar rule), plus optionally
   `queen_mind_orient_v0_1.zip`. Record their file hashes in the run log.
4. Provenance stamps per run: `executed_git_hash`, SDK + daemon versions,
   arm, session dir, source-NAc hash. Each arm runs in a FRESH session dir.

## Design — three arms on healthy hardware, same room/geometry as H1 Geometry 1

| Arm | NAc at trial 0 | What it measures |
|---|---|---|
| **A — fresh** | empty (`--fresh`) | learning-from-scratch baseline on the healthy plant: trials-to-criterion `T_A` |
| **B — injured baseline** | Exp 45d seed-2 NAc loaded (the degraded-era 1.00/1.00 policy) | trial-0 retention + re-adaptation curve + trials-to-criterion `T_B` |
| **C — merged bundle (optional, run if time allows)** | `queen_mind_orient_v0_1.zip` via `merge-nac` | does the fleet-merged policy re-adapt like B (bundle-level savings)? |

**n ≥ 3 hardware sessions per arm run** (the 45d replication standard; the
n=1 lesson is written into the graduation row this experiment feeds).
**Probe conventions are Exp 45d's verbatim:** frozen-policy probes (greedy,
no learning) at fixed trial indices scoring direction and
magnitude-appropriateness over the 4 bins; ground truth never reaches NAc
(`--perturb` apparatus). **Trials-to-criterion** := first probe index at which
direction ≥ 0.9 AND magnitude ≥ 0.75, sustained for 2 consecutive probes.
**Nothing is retuned mid-experiment:** YAML magnitudes, bin boundary, and
`--az-gain` stay frozen at the values recorded in the loaded policy's sidecar
for arm B/C, and at the H1-derived values for arm A (see Amendment slot).

## Pre-registered hypotheses + gates

- **P1 — direction retention (arm B, trial 0):** direction probe **≥ 0.75**
  before any new learning. (On the SAME plant this probed 1.00; a plant change
  should not flip signs. Below 0.75 → direction did not transfer → the
  savings framing weakens and P3 is interpreted cautiously.)
- **P2 — magnitude miscalibration in the PREDICTED direction (arm B, trial 0):**
  magnitude probe **< direction probe**, AND the error direction matches the
  sign of `g_H − g_D` (overshoot if `g_H > g_D`, undershoot if `g_H < g_D`;
  the concrete expected-error statement is filled by the Amendment from H1's
  measured `g_H` BEFORE trial 1). If H1 lands `g_H ≈ g_D` (boundary unmoved),
  P2 converts to a null prediction: trial-0 magnitude ≈ 1.00 — either outcome
  is informative, neither is failure.
- **P3 — savings (the core gate):** median `T_B` **<** median `T_A`, each arm
  n ≥ 3. Ties or inversions → NO savings claim; report honestly.
- **P4 — learned non-use (measurement, not a pass/fail gate):** read the
  bounds learner's persisted limits before arm B; prediction: at least one
  axis sits BELOW the H1-measured healthy envelope. Then measure whether it
  re-expands over the arm-B sessions. **Pre-declared interpretation:** if the
  learner has no recovery path and the limit persists indefinitely, that is a
  FINDING (structural learned non-use) and spawns a deferred-plan entry with
  a revival trigger — it does not fail the experiment and must not be patched
  mid-run.

**Verdicts:** PASS = P1 ∧ P3 (P2 in predicted direction strengthens; P4 either
way is reported). PARTIAL = P1 without P3, or P3 with P1 marginal. VOID = an
instrument-level confound (e.g., Part-A-class actuation fault discovered
mid-run) — declared as VOID, never massaged. **A PASS adds a new
graduation-candidates row** ("Re-adaptation after plant change — savings +
retention under altered dynamics") as Earned with this doc as citation and
re-run triggers: NAc persistence/decay change, `merge-nac` semantics change,
any motor/shell hardware change, orient YAML change. No existing row's status
moves on this experiment's outcome (Exp 45's row un-stales on H1, not on this).

## Amendment slot (the ONLY permitted pre-results edit)

After H1, before trial 1, fill in — dated, one commit:
`g_H = ___ az/rad` (H1 central gain), `g_D = 0.19–0.57` (the degraded-era
bracket; state which comparison value P2 uses and why), healthy neck envelope
`= ___°`, arm-A `--az-gain = ___`, P2's concrete expected-error direction.

## Post-hoc discipline

Findings not covered by P1–P4 spawn new pre-registered follow-ups (CLAUDE.md
post-hoc rule); they are not promoted to claims from this run. Two consecutive
divergence-shaped surprises → bird's-eye audit before a third hypothesis, per
the debugging-divergence trigger — including its checklist question: *did the
action we commanded actually happen?* (re-run `yaw_verify.py` before
theorizing about any sensor.)
