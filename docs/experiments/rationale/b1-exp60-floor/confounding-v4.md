# B1 v4 delta: confounding lens

**Verdict: no DO-NOT-BUILD.** The v4 change does not tilt the FEAR-vs-floor comparison toward FEAR. It makes the floor stronger, which is the conservative direction. There are three SHOULD-FIX items and three NITs.

## What I checked

- **Ran the toy model.** `learner_sim.py` reproduces exactly: v3 explore=0.10, prior=0.64, raw=0.07; v4 1.00 for all three Q arms; RANDOM 0.15.
- **The run has no data.** The claim holds. There is no b1 output under `docs/experiments/data` or `results`. The range 632cee32..76b3c499 is a single build commit. Rules 16–17 come from the scripted (fake) bridge. The toy's timings come from Exp 60's existing records, not from B1.
- **Rule 17 is in the code.** `FloorWaterTrial.before_teleport` arms with `water_only=True`. Once armed, `ArmedPolicy.select` returns None while the observation reads dry, before `inner.select` is called. So no floor pick is made on the shore.

I also ran a sensitivity sweep. It is a scratch copy driving the real `TabularQPolicy`, with 300 seeds per arm and the escape time jittered ±20%.

| Variation | explore / prior / raw |
|---|---|
| Q₀ = −0.5 | 0.93 / 0.98 / 0.94 |
| Q₀ = −1 | 0.99 / 0.99 / 1.00 |
| Q₀ = −3, −4.4, −10 | 1.00 / 1.00 / 1.00 |
| **K = 0 (no training)** | **1.00 / 1.00 / 1.00** |
| No pre-probe | 0.83 / 1.00 / 0.93 |
| 2 or 3 water keys (split by time) | ≥ 0.99 |
| One tool blocks 1.5 s | ≥ 0.99 |
| Escape takes 3.2 s | 1.00 |
| v3 constants with 3 keys | 0.01 / 0.82 / 0.10 |

## (a) Q₀ = −2 and forking paths

**Fair, not tuned to a knife edge.** The learner works for any Q₀ from about −1 down to −10 once γ = 1. The real condition is that Q₀ sits below the escape's own return (≈ −0.45) and γ = 1. Then every tried failing action's target, −c + Q₀, falls below Q₀, so greedy picks work through the untried actions in turn. That is ordinary pessimistic initialisation. It does not encode which act wins. FEAR gets nothing comparable, but FEAR's act choice already comes from the designer's prior, and that is disclosed.

**Forking paths.** The constants were picked after the toy, with no B1 data, and in the direction that makes the floor harder to beat. That lowers the chance of `FLOORS_BEATEN` and makes the expected tie more likely. It is not a garden of forking paths in the claim's favour.

The toy's assumptions (one key, a fixed 2.6 s escape, < 0.5 s failures, no same-tool cap) flatter the floor. If they are wrong on the rig, the floor gets weaker. That is the remaining confound, handled under S2 below. The same-tool cap is moot here: pessimistic greedy never repeats a failed tool, because a tried tool's Q drops below Q₀.

**SHOULD-FIX S1 (mislabel).** Rule 15 and OD5 call −2 "the worst terminal return … assumed to end in the pain". That is wrong. With γ = 1 the real return of a pain episode includes the oxygen cost: ∫₀⁶ (t/15) dt = 1.2 s, which is 2.4 cost units, plus the −1 pain intensity and the −1 terminal, giving **≈ −4.4**.
- **Fix:** relabel Q₀ = −2 as "the pain components only (terminal + intensity)". Add the measured robustness range (Q₀ between −1 and −10 all give ≥ 0.99) so a reader can see −2 is not a tuned value.

**SHOULD-FIX S2 (K = 0 gives 1.0: the pre-probe does the learning).** In the toy, the Q arms learn `escape_water` entirely in the six greedy pre-probe placements (about 48 picks working through 26 tools). The 10 executed training episodes add nothing. FEAR, by contrast, is refused if it forms a positive escape link before the probe. Rule 4 already counts pre-probe actions, so this is accounted for. But H2's text ("with … 10 executed episodes, Q_EXPLORE finds `escape_water`") and the training learning-curve DV describe a route the toy says will not happen.
- **Fix:** state in H2 that the toy predicts criterion during the pre-probe.
- Report each Q arm's surfacing per pre-probe placement as a secondary DV.

## (b) Water-only arming (rule 17)

**Primary DV: no asymmetry, provided one assumption holds.** Rule 17 says FEAR proposes nothing on the shore at the first post placement. That is argued from FEAR's preflight (no positive escape link) and from fear being zero on the shore cluster. It is not checked.

**SHOULD-FIX S3 (make it a check).** Turn that assumption into a check using rule 14's call times.
- **Fix:** a FEAR first-post success whose surfacing call STARTED before the first in-water observation refuses that arm-seed. That treats FEAR the same way the floors' gate treats them.

**NIT N1 (secondary DV).** Over all 6 post placements the comparison is now built to favour FEAR. FEAR's placements 2–6 shore `escape_water` lands about 0.7 s before the teleport. The floors are gated off the shore.
- **Fix:** label that secondary "not comparable across arms (rule 17)".
- Also: the floors' gate reads `is_in_water` (eye height), while FEAR keys on the EC world cluster. If the two flip on different ticks, latency differs by up to 0.5 s. That costs at most one draw for RANDOM and leaves FEAR's slack intact. Report the first-water-select latency per arm.

## (c) Credit leak (rule 16)

**Closed, but not by the path rule 16 describes.** Under rule 17 the inner policy never sees an air select inside a placement or training window, so rule 16's select-side "first air select is terminal" never runs there. It runs only in the ungated shore roam, where every pick is dropped as `made_out_of_water`.

The real closure is the harness's 4 Hz `end_episode("surfaced")`. In `index.js`, `escape_water` holds `jump` for 600 ms after the head clears. The harness therefore reads surfaced while the escape is still in flight and the loop is blocked. The pending pick is the escape, credited terminal 0. I found no residual leak.

**NIT N2.**
- Rewrite rule 16 to name the path that actually closes the episode.
- Add a deletion-proven test that the in-placement surfacing credit goes to the escape pick.
- `idle_dry` mixes pre-teleport skips with post-surfacing skips; split the count.

## (d) H1/H2 and the verdict rules

**RANDOM ≈ 0.15 re-derived.** 1 − (25/26)⁴ = 0.145. If the 2.6 s already includes the 600 ms post-clear hold, the DV's time-to-clear is ≈ 2.0 s. That allows 5 draws, giving 0.178 (v3's figure).

**NIT N3.** State the range 0.11–0.18 and which timing the 2.6 s refers to. The toy treats 2.6 s as time-to-surface, which is conservative for the floors.

**Power is unaffected.** Exact 8-v-8 test with Holm's worst-case α of 0.025 for H1:

| FEAR rate | RANDOM rate | Power |
|---|---|---|
| 1.0 | 0.25 | 1.00 |
| 0.9 | 0.15 | 0.87 |
| 0.9 | 0.18 | 0.84 |
| 0.8 | 0.15 | 0.66 |

H2 with 8/8 vs 8/8 gives p = 1, which is the expected tie, so the verdict rules stay consistent.

**One gap, not introduced by v4.** FEAR can pass the ≥ 6/8 sanity gate and still fail H1. For example, 6/8 vs 2/8 gives p = 0.066, which yields `FLOOR_MATCHED_CHANCE` on a weak FEAR. That is an honest result, but the prereg should say it can happen.

**Overall:** no confound in the claim's favour. The open risk is the opposite one: a `FLOORS_BEATEN` produced by a floor that never explored properly on the rig. S2's pre-probe reporting plus a competence condition tied to rule 13's key-coverage diagnostic would close it. That condition: `FLOORS_BEATEN` stands only if Q_EXPLORE has tried `escape_water` in the key it reads at the first post placement; otherwise annotate the verdict "floor under-explored". (Recommended addition to S2.)
