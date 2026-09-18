# An instrument band must be calibrated on the SHORTEST window any arm will produce

**Where it bit (R3, 2026-09-18).** The R3 gauntlet froze a loop-cadence band as the median ± 2 IQR of
the in-window tick periods of twelve floor-arm events — each ≈ 28 s long, ≈ 46 periods per row. The
bench then refused seven carried-fear rows for "tick period median above the band". Those events
last ≈ 3 s: their in-window ticks are at most one idle tick, the `flee` tick and the `escape_water`
tick — ONE or TWO periods — and one of them is always the `flee` tie-break dispatch, 0.70–0.79 s in
EVERY arm. A 46-period median and a 1-period "median" are different statistics; the band read the
tie-break interval and refused rows by where the loop's tick fell relative to the teleport. The loop's
idle cadence was identical across arms to ± 15 ms (C vs A, p = 0.76). The first draft of the
amendment even explained it as "trained agents tick slower" — a mechanism the rows refuted (the
trained-but-detached arm ticked like the fresh one). Two review lenses caught it from the rows.

**The rule.** When a per-event statistic is frozen as a refusal band:
- calibrate it on the SHORTEST window any arm will produce, or measure it on a like-for-like window
  every arm shares (the shore liveness window before the event; in-water intervals that do not span
  a blocking tool call);
- refuse only when enough periods sit behind the median (record the count per row; an IQR of `None`
  is the tell — R3's escape rows had `tick_period_iqr_s: None` on 24 of 24);
- report the like-for-like cadence as the covariate (`idle_tick_period_median_s` in `r3_run.report`),
  and label an in-window number for what it is.

**Why it matters.** The band did not detect an instrument difference; it flagged its own
inapplicability, and the refusals were correlated with the DV's own timing (a one-period row is one
whose `flee` fired on the first in-water tick — the fast tail). A rule that refuses by phase is a
selection on the outcome. Regression guard: `tests/unit/test_r3_run.py::test_idle_tick_period_ignores_proposing_ticks`
and the amendment tests; the design lesson lives in the R3 prereg §Amendments.

See also: [harness-loop-must-be-proven-live.md](harness-loop-must-be-proven-live.md);
[sensor-range-clamps.md](sensor-range-clamps.md) (the reservoir behind a clamp, the same campaign).
