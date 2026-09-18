# R3 survival benchmark — bio-faithful lens on the 2026-09-18 POST-DATA amendments

Charter: does the amended instrument still measure the mechanism's real job, and does the measured
picture across the five arms make bio-faithful sense? Read against the prereg (§Arms, §Dependent
measures, §Calibration result, §Pilot, §Amendments), the v1 and v3 bio-faithful reports (not
re-derived), `docs/agents/bio-memory.md` §Wire 4, the 60 bench rows in
`docs/experiments/data/r3_bench.jsonl` tick by tick, the frozen and amended reports, and the
amendment code (`scripts/survival_world/r3_run.py::reclassify_under_amendments`,
`harness_unchanged_between`; `water_trial.py::lethal_event` for how `tick_period_median_s` is made).

The short version: **the amendment's ACTION is right and its EXPLANATION is wrong.** The seven
refused rows should be counted — but not because "agents that carry substrate tick slower". They do
not. The loop's idle cadence is the same in every arm to ±15 ms (C 0.584 s, D 0.566, A 0.581,
B 0.578, E 0.581, pooled idle periods; C vs A Mann–Whitney p = 0.76). The number the band was applied
to is not a loop cadence at all on a 3-second event: it is the median of ONE or TWO periods, one of
which is always the `flee` tie-break dispatch (0.70–0.79 s in every arm). Whether a C/D row was
refused was decided by the phase of the loop's tick relative to the teleport — a coin flip, not a
workload. Writing "the cost of carrying a fear" into §Outcome as a per-arm covariate would ship a
false mechanism claim on top of a correct recount.

---

## SHOULD-FIX

### SF-1 — Amendment 2's stated cause ("per-tick workload is larger with a populated EC and a booked fear") is refuted by the rows; the "slower loop" is a window-composition artifact of the tie-break tick

**Failure scenario.** §Outcome reports `tick_period_median_s` C 0.684 / D 0.673 vs A/B/E 0.583 as
"the loop ticks ≈ 17 % slower carrying a fear", a reader takes it as a measured cost of the
mechanism (bio: vigilance has a metabolic price), and the next design budgets for it — when no such
cost exists in the data and the number is the price of `flee`, paid identically in every arm.

**Evidence.**
- `water_trial.py::lethal_event` computes `tick_period_median_s` over IN-WINDOW ticks only
  (`0 ≤ t ≤ t_end`). A C/D event lasts ≈ 3.2 s and the tick after the blocking escape call lands at
  t ≈ 3.5 s, outside the window. So the in-window ticks are: at most one idle tick (t ∈ [0, 0.29]),
  the `flee` tick, and the `escape_water` tick — **1 or 2 periods**, never more. The band
  ([0.394, 0.768]) was frozen as median ± 2·IQR over ≈ 46 periods per floor-arm row. A 46-period
  median and a 1-period "median" are different statistics; the band was applied across that gap.
- Per row (all 24 C/D rows read): **every one of the seven refused rows has exactly ONE in-window
  period, and that period is the `flee`→`escape` dispatch** — D 360 0.769, C 342 0.786, D 364 0.778,
  D 366 0.787, D 368 0.793, C 349 0.786, C 351 0.773. Four other one-period rows passed only because
  their tie-break tick happened to run under 0.768 (D 365 0.707, D 369 0.606, C 350 0.655,
  D 370 0.652). The thirteen two-period rows report the MEAN of one idle period (0.51–0.69) and the
  flee period (0.65–0.79) — e.g. C 340 = (0.525 + 0.787)/2 = 0.656 exactly. A row was refused iff the
  tick before its `flee` tick fell at t < 0 (pre-teleport); that is tick phase, not workload.
- The tie-break tax is arm-independent: the period following the `flee` proposal is
  A 0.745 (0.547–0.799), B 0.765, C 0.786, D 0.767, E 0.736 (0.642–0.797) — fully overlapping.
  It is the synchronous `flee` dispatch inside the loop (the prereg's own "0.70–0.77 s in EVERY arm").
  In A/B/E it is one period among 14–46 and cannot move the median; in C/D it is the median.
- The idle cadence (periods between two non-proposing ticks, over each row's whole telemetry
  window): A n = 516 median 0.581, B n = 135 0.578, C n = 23 0.584, D n = 24 0.566, E n = 515 0.581.
  Mann–Whitney C vs A p = 0.76; D vs A p = 0.029 in the FASTER direction (15 ms). C∪D vs A∪B∪E
  p = 0.08, medians 0.575 vs 0.581. There is no slow loop.
- The mechanism reading agrees: what the substrate branch does per tick that scales with carried
  state is `anticipatory_threat_need` (a `min` over ≤ 2 active clusters of a dict scan over
  `_cluster_fear`, which holds ONE entry in C/D) and `recommend_action` over 2 candidates; the
  per-tick encodes (`encode_sensors` per channel) and `evaluate_failures` run identically in every
  arm; the telemetry snapshot dumps EC/NAc summaries, and E's EC is as populated as C's (10
  training episodes on one cluster each) yet E ticks like A — which the amendment cites as support
  for "the booked fear" being the cost. It is not: E ticks like A because E's window holds 43 idle
  periods, not because E lacks a fear. The amendment read the right control the wrong way round.

**Fix (prereg text + report, no harness data change).** (i) Replace the causal sentence in
§Amendment 2 with the artifact: "the in-window tick-period median degenerates to 1–2 periods on a
≈ 3 s event, one of which is always the `flee` tie-break dispatch (0.70–0.79 s in every arm); the
band, frozen on ≈ 46-period medians, refused rows by tick phase". (ii) Do NOT carry
`tick_period_median_s` as "the cost of carrying a fear" into §Outcome. Either report it labelled
"in-window median, n = 1–2 periods for C/D, dominated by the tie-break — not a cadence", or —
better, and the same class of change as `reclassify_under_amendments` (a pure function over the
rows' `ticks`) — add an `idle_tick_period_median_s` covariate computed from non-proposing tick
pairs and report THAT per arm (C 0.584 / D 0.566 / A 0.581 / B 0.578 / E 0.581). (iii) The lesson
for `docs/wiring/` is "check the n behind a statistic before applying a band frozen on a different
n; a cadence band belongs on idle ticks", not "calibrate the band per arm kind" — a per-arm-kind
band of ≈ 0.65–0.79 for C/D would enshrine the tie-break tax as a property of trained agents.

### SF-2 — The honesty note is fair on admissibility, over-stated on mechanism, and under-states what the band could and could not do

**Evidence.** Fair: neither amendment touches an outcome — the recounted rows are indistinguishable
from their arms' clean rows on every DV (below), and the hash rule was unsatisfiable by
construction. Over-stated: "the band did the job it was frozen for — it flagged an instrument
difference" credits the rule with detecting a workload difference that does not exist; what it
flagged was its own inapplicability. Under-stated: the note says the band was "frozen on the FRESH
floor arm", implying the arm was the problem; the problem is the EVENT LENGTH — the same band would
refuse a fresh arm-B agent if B's event were 3 s long, and would pass a trained agent whose event
lasted 28 s. Say so; it changes what the next campaign must fix.

---

## NIT

### N-1 (Amendment 1) — `HARNESS_PATHS` = (`scripts/`, `src/`, `tests/`) does not cover `pyproject.toml` or lockfiles
A dependency bump between the calibration and bench hashes would pass `harness_unchanged_between`
silently. Between `6b16bbe9` and `4cca5524` nothing but data and this prereg moved (verified with
`git diff --name-only`), so the bench is clean; the rule as written should name the exclusion or add
the manifest. Not a bio matter — flagged for the confounding lens.

### N-2 — The prereg's C/D onset "3.15 ± 0.20 s" decomposes into three named parts, two of which are the body, not the drive
From the rows: first proposing tick C 0.99 s (0.53–1.38) / D 0.95 s (0.86–1.05) — the tick before
it sits at t ∈ [−0.43, 0.29] and never sees water (bridge state lag ≈ 0.1 s + tick phase); `flee`
tie-break 0.77 s (0.58–0.79); ascent 1.43 s (1.21–1.65). The fear acts at the FIRST tick that
sees water; ≈ 1.0 s of the 3.2 s is sensing + phase, 0.77 s (24 %) the tie-break tax, 1.43 s
(45 %) the ascent. Worth one sentence in §Outcome so "3.2 s" is not read as the drive's reaction time.

### N-3 — One E row fired 13 ms BEFORE the sampler's < 14 hp read (E 380: cross 25.384, `flee` 25.371)
The loop's own drive read and the harness sampler are two clocks on the same bridge; the innate
route saw 13.x hp one bridge interval before the sampler did. Reassuring (the harness's read is
independent of the agent's), not a defect; the lag distribution over A/E is 0.0–0.93 s = one tick
phase, as the pilot said.

---

## Answers to the five charter questions

1. **The tick-period finding.** Not a mechanism property. C and D do not tick slower; the
   in-window statistic degenerates to the `flee` dispatch period on a 3 s event (SF-1). What C
   carries that E does not — a booked fear on the water cluster (`fear_before` −1.0 / D −0.75; E
   `cluster_fear_dump` {}) — changes WHICH tick proposes (the first in-water tick vs the health
   crossing at 25 s), not how long a tick takes. "A populated EC and a booked fear" is the wrong
   explanation; E is the control that refutes it (same EC population, same cadence, no fear). A
   per-arm "reported covariate" is faithful only if it is the idle cadence; as computed it is not.
2. **Does it bias the DVs?** No, because there is no slower loop. The one real phase term — the
   first in-water tick's offset from the teleport, uniform over one idle period (≤ 0.6 s, mean
   ≈ 0.3 s) — is common to every arm (A/E pay it at the health crossing, B at the θ-clearing
   publish, C/D at the teleport) and cannot shift a 12-row median by more than ≈ 0.1 s: C − B is
   −5.4 s with both medians' 95 % intervals ≈ 0.3 s wide; nothing crosses. For D beside C
   (−0.049 s, p = 0.84) the phase term IS the dominant noise, which is exactly what the p says.
3. **The five-arm picture.** Matches the v1/v3 description of the job to the decimal: innate
   route A 27.995 s / 10.7 hp / 22.5 s of oxygen pain + 2.1 s of health pain; in-situ B 8.575 s /
   0 hp / 3.0 s of oxygen pain with min oxygen 9 in 12/12 rows; carried C 3.18 s / 0 / 0; D 3.13 s
   / 0 / 0; E 28.07 s / 10.8 hp / 22.5 s. **E ≡ A to 0.075 s on the median** (and on health lost,
   pain-seconds, escape score 0.7 = innate threat 1.0 × affinity, crossing lag): ten training
   episodes without the subscriber left a water cluster the agent re-entered (`live_g2` pass, need
   0.0) and NOTHING that helps — the representation (Wire 2) is behaviourally inert without the
   valence (Wire 4). That is the cleanest statement the benchmark makes and §Outcome should make
   it. **D ≡ C** (−0.049 s, p = 0.84): the 0.75 discount is visible on the executed-escape score
   (D 0.525 in 12/12 vs C 0.7 in 12/12) and invisible on the DV, as the prereg predicted — the
   argmax and the tick are the same; B acts on the same 0.525 in 10/12 rows (2 rows reached −1.0
   before the escape tick). B's acquisition is a clock: 0.5 at 5.1–5.6 s (−0.25), 1.0 at
   5.9–6.3 s (−0.75 clears θ), `flee` 0.3–0.4 s later, SD ≈ 0.15 s on the first call. "What the
   drive buys", in the mechanism's currency (negative reinforcement NOT paid): against the innate
   floor 24.8 s of latency, 10.7 hp, 22.5 s of oxygen pain and 2.1 s of health pain; against
   learning it in place 5.4 s and 3.0 s of oxygen pain (11 bubbles of it). The residual 3.2 s is
   the body — sensing + phase, tie-break, ascent (N-2) — not the drive.
4. **The amendments' faithfulness.** Neither changes what the benchmark measures. Amendment 1 is
   provenance (N-1). Amendment 2's seven rows: `end` surface, `pain_publishes` [] in all seven
   (surfaced at 2.6–3.4 s; the pain edge is ≥ 5.15 s — there is no pain edge to mis-time),
   drive-decisive with causal 0 / learned 0 (drive 0.7 C, 0.525 D; runner-up 0.30 / 0.13),
   `escaped_before_damage` true, health lost 0, min oxygen 16–17, `max_state_age_s` 0.074–0.102
   (all under 0.15), `guard_breach` null, escape `detail` "surfaced", 3 positive links all booked at
   call RETURN after the window, fear unchanged. A loop "that slow" could in principle miss a
   publish — but the loop was not slow (SF-1) and these events contain no publish to miss. No bio
   reason to exclude any of them; if anything the recount is stronger than the amendment claims,
   because the refusals were phase noise rather than a real instrument difference being waived.
5. **The honesty note.** Fair on admissibility; over-stated on mechanism; under-stated on the
   rule's structural failure (SF-2).

---

## What I verified

- Prereg §Arms, §Dependent measures, §The Goldilocks calibration, §Calibration result, §Pilot,
  §Amendments (working-tree copy, `git diff` against `b99ddf92`); the v1 and v3 bio-faithful
  reports; `docs/agents/bio-memory.md` §Wire 4.
- `scripts/survival_world/r3_run.py` diff: `reclassify_under_amendments` (pure; recounts only a
  refusal that is exactly `TICK_BAND_REFUSAL` with no `;` and `end` ∈ {surface, death}),
  `harness_unchanged_between` (`git diff --name-only`, prefix filter), the `--amended` report path;
  `tests/unit/test_r3_run.py` additions; `gauntlet_drift` in `r3_run.py` (band check on
  `tick_period_median_s`); `water_trial.py::lethal_event` (`in_window` = ticks with
  `0 ≤ t ≤ t_end`; `periods` sorted; median/IQR) and `_telemetry_ticks`.
- All 60 event rows of `r3_bench.jsonl`: per row `ticks` (t, proposal), the in-window period
  count and values, idle periods pooled per arm (n = 516/135/23/24/515), the post-`flee` period per
  arm, `calls` start/return/detail, `pain_publishes`, `executed_escape_event.score_components`,
  `decisive`, `max_state_age_s`, `guard_breach`, `fear_before`/`fear_after`,
  `positive_escape_links_after`, `training`/`ingest`/`live_g2`/`representation_gate`, the
  `samples` health series (first < 14 hp read per A/E row vs `t_first_call`). Mann–Whitney via
  scipy on the pooled idle periods.
- `agent_loop.py::propose_via_substrate` (per-channel encode → `note_active_clusters` →
  `evaluate_failures` → `anticipatory_threat_need` (min over active clusters) → max-combine →
  `recommend_action`), the substrate tick site and `substrate_telemetry.snapshot`;
  `nac.py::cluster_fear` / `anticipatory_threat_need` (dict scan over `_cluster_fear`, one entry in
  C/D).
- `r3_gauntlet.json` (`tick_period_band_s` [0.394, 0.768], `cal_code_hash 6b16bbe9b32f`); the
  frozen vs amended report deltas (C n 9 → 12, D 8 → 12, E 11; C − B −5.30 → −5.40; D beside C
  −0.146 → −0.049).
- Not verified: anything live; the dependency manifest between the two hashes beyond
  `git diff --name-only` (N-1).

## Verdict

**FIX-THEN-AMEND.** Count the seven rows — the data support it more strongly than the amendment
says. But before §Outcome is written, replace Amendment 2's explanation and its covariate: the
loop does not tick slower with a carried fear (idle cadence identical across arms, C vs A
p = 0.76); the refused number is the `flee` tie-break period standing alone in a 1–2-period window,
and the refusals fell by tick phase. Report the idle-tick cadence as the covariate (or label the
in-window number for what it is), state E ≡ A and D ≡ C in the mechanism's terms, and record the
wiring lesson as "match the band's statistic to the row's n", not "calibrate per arm kind".
