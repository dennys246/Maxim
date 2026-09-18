# R3 survival benchmark — CONFOUNDING lens on the two POST-DATA amendments (2026-09-18)

Reviewed: `docs/experiments/r3_survival_benchmark_prereg.md` §Amendments (uncommitted, branch
`r3/amendment`), their implementation in `scripts/survival_world/r3_run.py`
(`harness_unchanged_between`, `reclassify_under_amendments`, the `report --amended` branch) and
`tests/unit/test_r3_run.py::test_amendment_*`, against the rows `docs/experiments/data/r3_bench.jsonl`
(60 events, hash `4cca5524`), the gauntlet `r3_gauntlet.json` (cal hash `6b16bbe9`), the frozen
report `r3_report.json` (INCOMPLETE) and the amended report `r3_report_amended.json` (INCOMPLETE by E
at 11/12). Charter: a rule loosened after the data is the most dangerous move in this repo — are these
two honest instrument corrections or outcome-shaped? Everything below is computed from the rows, not
read from the prereg's summary of them.

**Short version.** Amendment 1 is an honest correction with two implementation gaps (ancestry is
claimed but not checked; the path filter misses `pyproject.toml`/`data/`). Amendment 2 reaches the
right decision — the seven rows are refused by an instrument artefact, and recounting them is MORE
honest than re-running them — **but for a stated reason that is false**: the loop is NOT slower on
the trained/ingested arms. The pre-window idle cadence is arm-invariant (C 0.581, D 0.557, A 0.549,
E 0.554 s); what the band read on every C/D row is a "median" over ONE or TWO in-water intervals, one
of which is the interval spanning the `flee` call (0.77–0.79 s — the prereg's own "tie-break cost"
plus a tick), because the escape call blocks the loop until the head is clear. Every refused row is a
1-period row whose single period is the flee-spanning one; `tick_period_iqr_s` is `None` on 24/24
C/D rows. So the amendment's evidence paragraph, its "per-arm cadence covariate", and its lesson
("calibrate the band per arm kind") all describe a mechanism that does not exist and would put a
false claim about the substrate into §Outcome. The recount also does not leave the DVs "identical":
the frozen rule selectively refused C's fast tail (the 1-period rows are the rows where `flee` fired
on the first in-water tick; seed 351 is the fastest C row of all). The direction of that bias is
AGAINST the claim, so the recount removes ≈ 0.1 s of conservative bias from C's median and changes no
contrast — but the note must say that, not "identical". Verdict at the bottom: FIX-THEN-AMEND.

## What I verified (from the rows and the code)

- **V1 — `git diff --name-only 6b16bbe9 4cca5524`** touches exactly `docs/experiments/data/r3_cal.jsonl`,
  `docs/experiments/data/r3_gauntlet.json`, `docs/experiments/r3_survival_benchmark_prereg.md`.
  `git merge-base --is-ancestor 6b16bbe9 4cca5524` → true. `origin/main` is at `4cca5524` now. All 73
  bench rows (1 apparatus + 60 events + 12 donors) carry `executed_git_hash 4cca5524a686`.
- **V2 — the frozen hash rule is unsatisfiable under the runbook.** The gauntlet file is produced AT
  the cal hash and committed AFTER it; runbook step 1 requires a clean `main` checkout containing the
  gauntlet, so the bench hash is necessarily a descendant of the cal commit, never the cal hash. Nuance:
  `scripts/_provenance.py::DIRTY_SCOPE = ("src", "scripts")`, so an UNTRACKED gauntlet under `docs/` at
  the cal hash would technically pass the dirty gate — the rule is unsatisfiable under the runbook +
  the data-provenance discipline (gauntlet on main before the bench), not "by construction". Say so.
- **V3 — `harness_unchanged_between` is fail-CLOSED**: `OSError`/`SubprocessError` (incl.
  `TimeoutExpired`) → `(False, [...])`; `returncode != 0` → `(False, [...])`; an unresolvable hash
  (report run without `--gauntlet` gives `cal_hash = "None"`) → rc 128 → INCOMPLETE. The test covers
  rc 128. It does NOT check ancestry (only a two-tree diff) — see F-A1.1.
- **V4 — the tick statistic on the escape arms is degenerate.** `water_trial.lethal_event`
  (`in_window = ticks with 0 ≤ t ≤ t_end`; `periods` = consecutive differences; `_median`; IQR only
  when ≥ 4 periods). Reproduced every reported median exactly from that definition. Per C/D row
  (n_periods → outcome): **all 7 refused rows have n_periods = 1** (the single period is the interval
  spanning the `flee` call: 0.769–0.793 s); 4 rows with n_periods = 1 passed because their flee-spanning
  interval was 0.606–0.707; **all 13 rows with n_periods = 2 passed** (the mean of one idle tick
  ≈ 0.53–0.65 and the flee tick ≈ 0.75–0.79 → 0.61–0.72). Whether a row has 1 or 2 periods is the
  teleport's phase against the loop (first in-water tick at 0.51–0.65 s → 1 period; at 0.01–0.29 s →
  2 periods). A/E rows have 46–48 in-water ticks; B 15–16.
- **V5 — the idle cadence is arm-invariant.** Median of the PRE-window (shore, idle) intervals recorded
  on the same rows: C 0.581 (n = 15), D 0.557 (19), A 0.549 (19), E 0.554 (18); A's in-water median
  over 552 intervals 0.583; every arm's shore liveness window shows 5–6 ticks in 3.0 s. E trains (a
  populated EC, `TRAINED["E_exposed_ablated"] = "ablated"`) and ticks 0.58 in water. The amendment's
  "per-tick workload is larger with a populated EC and a booked fear" is contradicted by its own rows.
- **V6 — the refused rows are not "identical in outcome" to their arms' clean rows; they are the fast
  tail of C.** C refused `t_surface` {2.587, 2.997, 3.101} (median 2.997) vs clean {2.992 … 3.439}
  (median 3.277), Mann–Whitney p = 0.10 (3 vs 9); 351 is the fastest C row of the campaign (below the
  clean minimum). D refused {3.036, 3.112, 3.293, 3.395} (3.203) vs clean (3.131), p = 0.93. Pooled
  C+D refused vs clean p = 0.35. Within C, r(tick median, t_surface) = −0.37: the 1-period rows are the
  rows where `flee` fired on the first in-water tick, i.e. the early escapes. On every DV that
  separates arms the seven are at the arm's ceiling: pain-seconds 0/0, health lost 0, `escaped_before_damage`
  true, `decisive` true, `max_state_age_s` 0.074–0.102 ≤ 0.15, no guard breach, `fear_after` present.
- **V7 — effect of the recount on the contrasts:** C median 3.277 → 3.18 (Δ −0.10 s), D 3.131 →
  3.131. C vs B is complete separation either way (C max 3.439 < B min 8.314; p = 3.7e-5 at n = 9 or
  12). No contrast changes sign, size class or significance. The recount's entire effect is on the
  campaign STATUS (three arms 9/8/11 → 12/12/11), which is exactly where the incentive sits.
- **V8 — `reclassify_under_amendments` scope, against the code.** `_finish_event` raises the gauntlet
  drift Refusal LAST, after the stale-sample, guard-breach, decisive, no-escape, DETACHED-fear and B-fear
  checks, so a band-only string proves every earlier refusal passed; `gauntlet_drift` joins with `"; "`
  and no reason text contains `;` (bands render as `[a, b]`), so `";" not in core` is currently exact;
  `event_row` writes `"Refusal: <msg>"`, `shared_row` writes `str(exc)` — `removeprefix` handles both
  (the bench rows show both spellings). A cap row cannot reach the band: `lethal_event` raises the cap
  Refusal itself (`partial={"event": row}`, so `end == "cap"` sits on a non-band string). A death row
  with a band-only refusal WOULD be recounted (`end in ("surface", "death")`) — correct: death is a
  complete event and the frozen DV is censored at death. See F-A2.3 for what the code lets through
  that the text does not.
- **V9 — `report()` does not dedupe by (arm, seed)**; the frozen "refused rows re-run and supersede"
  is implemented only as "refused rows do not count". `_existing_clean` marks the seven band-refused
  rows as NOT done, so a `--resume` WITHOUT `--only E_exposed_ablated` re-runs them. See F-A2.4.
- **V10 — tests:** `python -m pytest tests/unit/test_r3_run.py -q` → 7 passed (114 s). The amendment-2
  test covers band-only / two-reason / stale / cap; it does not cover below-band, `None`, or a
  superseded row.
- **V11 — the amended report** reads `status INCOMPLETE — E_exposed_ablated: 11 clean rows < 12`,
  `amendment_1.harness_unchanged {4cca5524a686: true}`, `amendment_2.recounted` = the seven, `refused`
  = E 388 only; the frozen report's four causes are intact in `r3_report.json`.
- **V12 — precedent:** Exp 60 carries post-data Amendments 5/6 in the same "POST-DATA, cause named by
  measurement" format, so the form is established; the bar it sets is that the cause be NAMED BY
  MEASUREMENT — which is where Amendment 2 currently fails.
- **V13 — no `docs/wiring/` entry exists in the diff** for the tick lesson the honesty note says is
  "recorded in `docs/wiring/`" (`git diff --stat -- docs/wiring/` is empty; no wiring file mentions the
  band).

## Findings

### DO-NOT-AMEND

None. Both amendments are admissible in kind: neither changes a DV, an arm, a contrast or the
stale-sample rule; the amended rule is outcome-SYMMETRIC (it would recount a band-only death row just
as it recounts a band-only surface row), which is the property that separates an instrument correction
from an outcome-shaped one. What is not admissible as written is the JUSTIFICATION of Amendment 2 and
three code seams — hence FIX-THEN-AMEND, not DO-NOT-AMEND.

### SHOULD-FIX (each blocks writing §Outcome from the amended report)

**F-A2.1 — Amendment 2 names a mechanism its own rows refute; rewrite the evidence paragraph so the
cause is the one measured.** The text: "agents that trained or ingested tick slower — the per-tick
workload is larger with a populated EC and a booked fear; measured medians … C 0.684, D 0.673". Those
medians are not loop cadences (V4): on every C/D row the statistic is one or two in-water intervals
and on every refused row it is the single interval spanning the `flee` call. The loop's idle cadence
is the same in every arm (V5). The honest statement is: *the band's statistic is undefined on the
escape arms — a 3 s window holds 1–2 in-water periods, IQR `None` on 24/24 rows, and the period it
reads is the flee-spanning one (0.77–0.79 s, i.e. the tie-break cost + a tick), so a refusal is the
teleport's phase, not the loop* — and the like-for-like cadence (shore/pre-window ticks) is
arm-invariant. This is a STRONGER case for the recount than the workload story (the refusal is
orthogonal to the agent, not a property of it), and it is the only one the rows support. The
"F22 band cannot tell a loop regression from an arm's workload" sentence goes with it; F22's
regression scenario is still caught on A/B/E (46–48 and 15–16 periods) and by Amendment 1 for code.

**F-A2.2 — "identical in outcome" / "indistinguishable" is an overclaim and the wrong KIND of
justification; replace with the numbers and justify by the instrument, not by the rows' DVs.** V6:
the frozen band refused C's fast tail (p = 0.10, 351 the fastest row of the arm) because a 1-period
window is one where `flee` fired on the first in-water tick. Say: "on every arm-separating DV the seven
sit at the arm's ceiling (0 pain-s, 0 hp, decisive); on `t_surface` they are C's fast tail (2.59–3.10
vs clean median 3.28, p = 0.10) and D's middle (p = 0.93); the frozen exclusion biased C's median UP by
≈ 0.1 s, against the claim; the recount removes that bias and changes no contrast (C vs B complete
separation at n = 9 or 12)". An amendment that is admissible only because the recounted rows look like
the clean rows is outcome-conditioned; this one is admissible because the refusal never measured the
agent — lead with that.

**F-A2.3 — the code recounts MORE than the text: any band refusal, including BELOW the band and a
`None` median.** `core.startswith(TICK_BAND_REFUSAL)` matches "tick period median 0.30 outside …" and
"tick period median None outside …". The text says the rows "fell ABOVE the band" and recounts on the
workload premise; a loop ticking FASTER than 0.394 s at the same code is precisely F22's
regression signature, and a `None` median is a window with no in-water ticks. Fix in
`reclassify_under_amendments`: parse `tick_period_median_s` from the event, require it numeric and
`> band_hi` (read `band_hi` from the gauntlet passed in, or from the refusal string's `[lo, hi]`);
add the two negative cases to `test_amendment_2_*`. Narrow to exactly what the text promises.

**F-A2.4 — composition defect with the frozen supersede rule: a re-run band-refused row and its
recounted original BOTH count.** V9: `report()` counts every non-refused event row; `_existing_clean`
treats the seven as not done; the amendment text itself prescribes a `--resume` and relies on the
operator adding `--only E_exposed_ablated`. Without `--only` (or on any later resume), C/D would read
n = 13+ and the "one clean row per seed" invariant is silently broken. Fix: in
`reclassify_under_amendments`, do NOT recount a row if a LATER row for the same (arm, seed) is clean
(the frozen supersede wins; record `amended.superseded_by_ts`); add a test with a recounted row
followed by a clean re-run of the same seed asserting n does not grow. (Note for the frozen harness,
out of this amendment's scope: `report()` never deduped by seed — two clean rows for one seed would
count twice even without the amendment; a latent defect the recount makes reachable.)

**F-A1.1 — Amendment 1's text claims ancestry is checked; the code checks a two-tree diff only.**
"the gauntlet's calibration hash is an ANCESTOR of the bench hash and … no harness … file changed …
The report's `--amended` path checks exactly that." `harness_unchanged_between` runs `git diff
--name-only cal bench`, which is symmetric and ancestry-blind (the bench-time
`_is_ancestor_of_main(cal)` checks cal-on-main, not cal-under-bench). Tree-equality on the three
prefixes is the invariant D5 wanted (the gauntlet's code IS the bench's code), so the diff is the
load-bearing half; but the sentence must match the code. Fix: add `git merge-base --is-ancestor cal
bench` inside `harness_unchanged_between` (fail-closed on rc ≠ 0) and record `ancestor: bool` under
`amendment_1`; or strike "ANCESTOR" from the text. Prefer the check — one subprocess line.

**F-A1.2 — `HARNESS_PATHS = ("scripts/", "src/", "tests/")` misses tracked inputs that change
behaviour, and the text should name what NO git rule covers.** Tracked and uncovered: `pyproject.toml`
(dependency pins, pytest config), `data/` (`data/util/*.json|yaml`, `robots.yaml` read by
`hardware/config.py`), `scenarios/`, `.python-version`. None changed between the two hashes (V1), so
the campaign is unaffected; the rule is what is under review. Add `pyproject.toml`, `data/`,
`scenarios/` to the tuple (a doc-only change under `docs/` cannot alter behaviour: the apparatus and
anchor records are pinned per row by `apparatus_record_ts` / `anchor_measured` equality in
`gauntlet_drift`, and `~/.maxim/exp60_water_classroom.json` is outside the repo but pinned the same
way). Then state in the amendment what the frozen rule never covered either: the bridge process, the
Minecraft server, the venv — per-row `bridge_state_interval_s` and the fingerprint are their only
guards. Not widening a hole is not the same as having named it.

**F-A1.3 — the multi-hash fallback makes the E resume a trap; the amendment must say where the E row
may run.** `g["cal_code_hash"] = bench_hashes[0] if len(bench_hashes) == 1 else cal_hash`: with two
bench hashes the old "rows at hash(es) other than the gauntlet's" fires again AND the UNAMENDED
"rows span 2 code hashes" rule fires. The amendment PR itself touches `scripts/survival_world/r3_run.py`
and `tests/unit/test_r3_run.py`, so an E resume at any commit AFTER it merges reads INCOMPLETE under
Amendment 1's own rule. Therefore: the E 388 re-run must execute at `4cca5524` (current `origin/main`
— true today; confirm on the box with `git rev-parse --short HEAD` before trusting the row), and its
data commit must land BEFORE the amendment PR merges (or in the same merge-commit PR). Write this into
the amendment; add a `report` assertion that with `--amended` and > 1 bench hash the status names the
spanning rule explicitly rather than falling back to the frozen wording. Also prefer passing a flag
into `report()` (e.g. `hash_rule="ancestry"`) over rewriting the gauntlet's `cal_code_hash` in memory —
a reader of `report()` today sees the cal hash silently replaced by the bench hash.

**F-3 — §Outcome must LEAD with the frozen status, and the honesty note under-delivers on two
promises.** (a) The frozen report is the pre-registered result: §Outcome opens with "FROZEN:
INCOMPLETE — C 9/12, D 8/12, E 11/12, hash ≠ gauntlet" and its four causes verbatim, then "AMENDED
(post-data, §Amendments): …", then the contrasts, which are the same under both. Reporting "both
statuses" side-by-side is not sufficient if the amended one comes first or is the one quoted in the
headline. (b) "recorded in `docs/wiring/`" — no such entry exists (V13); either land it in this PR or
change the tense. (c) "For the NEXT campaign, the band is calibrated per arm kind" — unbuildable with
the statistic as built: IQR is `None` on every C/D row and a per-arm band from 1–2 periods is a band on
the flee cost. The lesson that the rows support: *measure the cadence on a like-for-like window (the
3 s shore liveness window whose stamps are already in `telemetry_liveness.jsonl`, or in-water intervals
that do not span a tool call), refuse only when the window holds ≥ k periods, and report the tick
count per row beside the median.* (d) The per-arm `tick_period_median_s` in `arms.*` must not be
presented as a cadence covariate in §Outcome; if a covariate is wanted, it is the pre-window idle
median (V5), which the rows already carry.

### NIT

- `";" not in core` is a substring test standing in for "exactly one drift reason". Correct today
  (V8); state the intent: `len(core.split("; ")) == 1`. Same behaviour, self-documenting.
- `out.stdout.split()` splits on any whitespace; `splitlines()` is the path-safe form.
- `test_amendment_1_*` patches `subprocess.run` globally — fine for a unit, but a second test that
  runs the real `git diff 6b16bbe9 4cca5524` in-repo and asserts `(True, [])` would pin V1 as a
  regression guard for the actual campaign.
- The `Refusal: ` prefix asymmetry between `event_row` (`f"{type(exc).__name__}: {exc}"`) and
  `shared_row` (`str(exc)`) is cosmetic but is why the frozen `refused` list shows two spellings;
  worth one line in the harness, not this amendment.

## Charter questions, answered in order

1. **Amendment 1.** Unsatisfiable under the runbook + provenance discipline (V2, with the
   DIRTY_SCOPE nuance). The amended precondition IS D5's invariant (tree-equality on the harness paths),
   but the code implements only the diff half (F-A1.1) and the path tuple is one line short (F-A1.2).
   `harness_unchanged_between` is fail-closed (V3). Nothing under `docs/` can alter behaviour; the
   things that can and are not in git are named in F-A1.2.
2. **Amendment 2.** (a) There is no per-arm cadence difference to confound anything (V5); the
   ≈ 0.05 s expected first-tick shift the workload story implies does not exist. What the frozen rule
   did do is refuse C's fast tail — bias AGAINST the claim, ≈ 0.1 s on C's median, no contrast moved
   (V6, V7). (b) The function is narrower than a hand recount but WIDER than the text: it also
   recounts below-band and `None` medians (F-A2.3); the join/prefix logic is exact today (V8). (c) A
   death row: recounted, correctly (censored DV). A cap row: unreachable (V8). A row later superseded
   by a clean re-run: DOUBLE-COUNTED (F-A2.4). (d) A per-arm-kind band cannot be computed from this
   statistic (IQR `None`, n = 1–2). Re-running C/D is materially WORSE than recounting: it replays the
   phase lottery (expect ≈ 7/24 refusals again), and re-run-until-pass selects rows whose `flee` fired
   on a later tick — a selection on the DV's own timing. Recount is the honest option; say why.
3. **Honesty note / reporting.** Keep both files; §Outcome leads with FROZEN (F-3a). Over-claims:
   "identical in outcome", the workload mechanism, "checks exactly that" (ancestry), "by construction".
   Under-delivered: the `docs/wiring/` entry, the "per arm kind" lesson (F-3b–d).
4. **The E resume.** Under the frozen supersede rule a clean 388 re-run counts and the stale row stays
   listed — fine. Watch: it must be at `4cca5524` and committed before the amendment PR (F-A1.3); it
   must have been launched with `--only E_exposed_ablated` or the seven are re-run and double-counted
   (F-A2.4) — check the resume's stdout for `=== C_self_learned seed 342 ===` lines.
5. **Else.** The v3.1 design assumed "the loop's tick-period DISTRIBUTION per event"; on the escape
   arms there is none, and an A-only calibration could not reveal it. That is a design gap of the
   frozen instrument, worth its own line in the amendment and the promised wiring entry: *a per-event
   instrument statistic must be calibrated on the SHORTEST window any arm will produce.*

## Verdict

**FIX-THEN-AMEND.** Amendment 1: amend after adding the ancestry check and the missing paths, and
after fixing the E-resume ordering in the text. Amendment 2: the DECISION (recount the seven) stands
and is the more honest of the two options, but the amendment may not be written into §Outcome with its
current evidence paragraph, covariate or lesson — replace the workload mechanism with the measured one
(a degenerate 1–2-period statistic reading the flee-spanning interval; idle cadence arm-invariant),
replace "identical" with the numbers and the bias direction, narrow the code to above-band numeric
medians, guard against the double count, and lead §Outcome with the frozen status.
