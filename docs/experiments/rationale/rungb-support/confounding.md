# Rung B SUPPORT — confounding lens (design review, 2026-09-25)

Reviewed: `docs/experiments/protocols/rungb_support_preregistration.md`, `scripts/survival_world/rungb_support.py`, `tests/unit/test_rungb_support.py`. Numbers below are an offline sweep using the `embed`/`cos` from `exp62_cross_pool_replay.py` (shipped gain law, `_stable_basis`) on the Exp 60 live gate vectors. They are predictions, not trace data.

## DO-NOT-BUILD

**D1. The open-spawn reference measures a different geometry, and the predicted result comes out the opposite way to the fear state.** (Prereg §Protocol, §Limits: "the question is the same … at a different place".) It is not the same question. The bridge's `perceivedLight` is `max(block, sky − darkness(t))`. At an open spawn, `light_level` therefore ramps 15 → 4 through the night. In the sealed shell, sky light is 0, so light stays at 0 all day and only `time_of_day` moves. The two references also differ in mass: the shell vector carries `is_in_water` = 1.0 and `distance_from_spawn`/`y_altitude` off neutral, while the open-spawn vector is almost entirely light plus time. Sweeping the whole day circle from tod 0.0417:
- shell, submerged (the fear-learning state): exact 0.947 · middle **0.053** · beyond 0. **No SUPPORT.**
- open spawn (other sensors at neutral): exact 0.26 · middle **0.27** · beyond 0.47. **SUPPORT.**

So the rig run would probably report SUPPORT for a reason that does not apply to the fear state.

**D2. The measurement has nothing left to discover, and the prereg misreads the 0.799.** At a fixed place with an idle bot, light is a deterministic function of time inside the bridge, so the trace is a known curve. Corollary 3 of `cosine-separation-is-directional.md` applies: replay it offline. The replay also changes the premise. For pool 1 submerged: cos(0.75, midnight) = 0.903 and cos(0.5) = 0.892, both exact; cos(0.95) = 0.847 and cos(0.99) = 0.799. The "night" miss is the **wrap** of a linearly encoded `time_of_day`: 0.99 and 0.04 are 1.2 real minutes apart but sit at opposite extremes. The honest reading is the prereg's own "keying" outcome (circular time encoding), not a graded read. The interpretation "mostly exact ⇒ 0.799 is shell geometry, not night" is backwards: the shell is mostly exact, and 0.799 is the wrap.
*Fix:* replace the rig run with an offline sweep, with the shell-submerged gate vector as reference and light pinned at the shell's value. Route the wrap finding to Phase 5 keying. If a sky-exposed fear state is ever wanted, first name an apparatus that learns fear there.

## SHOULD-FIX (if any rig variant survives)

**S1. The "invalidates the run" rule is only reported, not enforced** (`analyze_snapshots`: `varying` is computed, but `support` is still emitted). *Fix:* exit 4 when `varying ⊄ {light_level, time_of_day}`.

**S2. A sensor missing from a later snapshot is silently dropped** (`states` filters to keys that are present, and only the first snapshot is checked). The embedding shifts without any error. *Fix:* refuse any snapshot that lacks a declared sensor.

**S3. The 24-minute capture double-counts about 0.2 of the day**, from tod 0.04 to ~0.24, which is the exact-rich start. This biases the time-weighted fraction toward `exact`. *Fix:* truncate at the first return to the reference tod, or weight per day-circle bin.

**S4. Stale snapshots are counted.** The recorder writes `state_age_s`, but the analyzer ignores it, so a bridge stall re-weights one state many times. *Fix:* refuse, or drop, snapshots whose age exceeds a stated bound.

**S5. The 0.75 floor was chosen to include a known point (0.799).** *Fix:* report middle fractions at 0.70 / 0.75 / 0.80 as sensitivity, so a verdict that hangs on the floor is visible.

**S6. The decision check binds only against a one-node EC.** Live, a snapshot in the middle band may complete to a *different* existing node (the shore cluster), which is the case a graded read actually faces. *Fix:* state this as a limit, or also check against the Exp 60 shore node.

## NIT

- `test_the_reference_scores_exactly_one` never asserts cos == 1. No test proves the known-answer refusals fire: corrupt `_sensor_embed`, or change the threshold, and the tests still pass. Add a prove-by-deletion case for both refusals.
- The dry-run disclosure is adequate as written. The prereg should also say that the outcome can be derived offline, since that bears directly on "before any SUPPORT data exists".
- The embedding known-answer check is sound: it goes through a fresh EC with the same `ranges`, at 1e-9.
