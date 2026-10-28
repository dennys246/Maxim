# M60 adversarial design pass: the coverage ratchet fails a fall, not a rise

> Reviewer: adversarial gate lens. The work was read-only. It covers `docs/plans/coverage_ratchet_m60_approach.md`
> (DRAFT v1) against `scripts/lint_coverage.py` (docstring, `floor_head_rules`, `floor_diff_rules::changed_entry`,
> `main`), `.github/workflows/test.yml` (triggers, concurrency, `unit-tests`, `release-build`),
> `scripts/_lint_git.py` (`base_ref`, `push_base`, `must_not_skip`), `scripts/check_nightlies.py`,
> `docs/publication_guide.md` §2b, `docs/agents/ci-testing.md`, `docs/lessons/ci-efficiency-2026-10-16.md` decision 2,
> and outstanding.md M22/M52/M60. Date: 2026-10-27.

**Verdict: DO NOT BUILD as written.** Two fail-open findings (D1, D2) break the owner's condition that a real fall
keeps failing. Today a fall can hide inside the 1-point band. After this change it can also hide inside every rise
that nobody has pinned yet. The pins that the LOWERING rule reads go stale, so a floor can also be lowered past a
real fall. The fix is small and reuses existing plumbing: keep this design for the rise half, and add the
merge-base measurement (a cut-down Alternative B) for the fall half and for the base `missing` pin. With that in
place, the nightly guard is a convenience and no longer the only thing holding the line. The SHOULD-FIX items below
apply either way.

---

## DO-NOT-BUILD

### D1. Fail-open: a rise followed by a fall is never seen, and the lag bound is false

- **Attack.** PR A adds tests and lifts `maxim/memory` from 80.0 to 83.0. On a PR that is now only a note. PR B
  merges the same day. It deletes or skip-marks tests (diff coverage reads `src/maxim` only, so a test-only change is
  invisible to it) and brings the package back to 80.0. B passes, because it sits on the floor. The nightly measures
  80.0 and stays green. **No run ever observes the fall.** It is permanent, not lagged.
- **The claimed bound, "one nightly's merges", is false.** The slack lasts until a pin PR merges. Nothing forces that
  merge (see S1). Pin PRs need the owner's GPG signature, they fail on noise (S2), and they cannot merge during a
  release freeze (N2). The slack is the sum of every rise since the last merged pin PR, and it has no upper bound.
- **Compared with today:** the slack is the band (1.0 pt) plus k. The note says "today band alone allows 1 pt". The
  new slack is unbounded, so the change strictly widens a real fall's escape.
- **Fix (required).** Add a fall check against the base's measured coverage. On `pull_request` and `push`:
  - Fetch the `coverage-json` artifact of the newest green `push` run at or before the merge-base. Reuse
    `_lint_git.push_base`'s walk: first-parent ancestors, `actions: read`, `GH_TOKEN`. The job already has all three.
  - Fail a scope whose measured `missing` exceeds the base's measured `missing` by more than k, unless that scope's
    statement count fell (covered code was deleted). That is the LOWERING test, applied against the measurement.
  - Leave the committed floors as the absolute backstop, unchanged.
  - The escape per PR is then k statements, not the band or the accumulated rise.
  - k is cumulative across PRs (a walk-down of up to k per PR). The nightly pin cuts that walk off. Document it as a
    residual: it is noise-scale, and a deliberate walk costs real deletions.
  - A PR cannot forge the input, because it comes from a `main` push run.
  - When no artifact exists within `PUSH_BASE_SEARCH_LIMIT` commits or inside the 14-day retention, fail closed
    (exit 2) and print "merge main", which moves the merge-base to a recent commit.
  - Pending `main` runs get cancelled (test.yml's concurrency comment), which is why the fetch must take the nearest
    green ancestor, not the exact merge-base.

### D2. Fail-open: a stale `missing` pin lets a floor be LOWERED past a real fall

- **Attack.** Base pin: `maxim/X` `{80.0, missing 200}` over 1,000 statements. Tests added during the week bring main
  to 85.0, missing 150, never pinned (the rise is only a note). PR C deletes 100 covered statements and removes
  tests, giving 900 statements, missing 195, 78.3%. It lowers the floor to 78.3 with missing 195.
- `changed_entry` checks `m.missing (195) <= b.missing (200)` and `h.missing <= min(195, 200)`, and both pass.
  Uncovered statements actually grew by 45 against main.
- **Today** the rise forces a pin of missing 150, so the LOWERING rule refuses ("cover 45 statement(s) first").
- **The same stale pin feeds other rules:**
  - "a pinned missing never rises";
  - `pin_rises`;
  - the SF-2 split SUM, which charges successors against the source's base `missing` pin, so a stale-high
    file-floor pin lets a split lose that headroom;
  - the package rename/split sum.
- The docstring already lists stale `missing` as a residual. This change makes it the steady state. The design said
  it would not touch the residual. It does.
- **Fix (required).** Wherever a rule reads `b.missing`, use `min(b.missing, base_measured.missing)`, taking
  `base_measured` from the D1 artifact. That covers `changed_entry`, the split sum, the package sum and the LOWERING
  test. A file floor reads the artifact's per-file entry. If the D1 fetch is not built, the alternative is to refuse
  every LOWERING and every split while any floor at base is more than the band below main's last measurement. That
  needs the same artifact, so it is no cheaper.

---

## SHOULD-FIX

### S1. Fail-open: the nightly tightness guard fires, but nothing reads it

- It does fire. `unit-tests` has no job-level `if:` (`ci_step_rules` bans one), so it runs on `schedule`. The lint
  then takes the `else` branch ("head rules only"), and the new rise rule makes the job red.
- Nobody acts on the red:
  - `check_nightlies.py` reads only jobs whose name contains `(nightly)`, and `unit-tests` has none.
  - `release-build` and `release.yml --at` read the same filter.
  - `push_base` reads `push` runs only.
  - So a red tightness guard blocks no merge and no release. Its only audience is whoever reads the Actions tab, or
    the scheduled-run e-mail that goes to the last person to edit the cron line.
- The note's own figures (7 of 40 PRs, and H folding a pin into about every second PR) imply the guard will be red
  most nights. A permanently red job trains people to ignore it, and it hides a real red in the same job.
- **Fix:**
  - With D1 and D2 in place, keep the nightly guard as it is and call it a convenience.
  - Without D1, make it blocking: `check_nightlies` also requires the `unit-tests` job of the nightly run to
    succeed. Name it in `NIGHTLY_MARKER` handling, or add a second list.
  - Either way, reword "Floors therefore lag main by at most one nightly plus the time to merge that PR". The time
    to merge has no bound, and per D1 a same-day rise and fall never lag; they vanish.

### S2. Fail-closed: a pin taken from the nightly run usually fails the pin PR's own CHANGED rule

- `changed_entry` requires `h.missing <= min(m.missing, b.missing)`, where `m` is the pin PR's own run.
- CI noise is about ±2 statements: memory measured 760 then 762, retrieval 59 against 61.
- A batch of N scopes pinned from run A fails against run B whenever any scope measures lower `missing` in B. With
  dozens of scopes that is close to certain.
- Any merge between the nightly and the pin PR that adds tests fails it as well.
- Result: the "one command" becomes command, CI, re-pin, CI. That is the bookkeeping loop M60 exists to end.
- **Fix:** M60's own row says "pins only from the uploaded CI `coverage.json` artifact of the PR's head commit".
  1. Open the pin PR from the nightly's names.
  2. Run `--pins-from-run` on the pin PR's own `unit-tests` run.
  3. Push.
- The pinner should refuse a run whose head SHA is not the current HEAD, unless `--from-nightly` is passed
  explicitly as a draft.

### S3. Pinner input and refusals (fail-closed today, but it misleads)

**Use the artifact, not the log.**
- Read it with `gh run download <id> -n coverage-json`, then compute pins with the lint's own `measure_scopes`,
  `file_scopes` and the hint rule (`min(m.missing, f.missing)`).
- That avoids parsing text, binds the input to the run id, and makes per-file pins exact.

**Why `--merge-floors` refuses the artifact.** No rationale is recorded: not in the docstring, not in the
`7a476685` message, not in the lessons. The refusal reads as a format choice (the input is the printed floors
block), not a trust boundary. M52 itself proposes "a CI step re-derives the pin from the job's own artifacts". It
does not apply here.

**Validate the run through `gh api repos/{repo}/actions/runs/<id>` and its jobs, and refuse when any of these
fail:**
- the workflow is not `test.yml`;
- `head_sha` is not the expected commit: the PR's HEAD (S2), or `main` for a draft;
- the `unit-tests` job's "Run required fast suite" step concluded other than `success`. The upload step is
  `if: always()`, so after a `-x` failure the artifact is partial: the very data the ratchet refuses ("after a failed
  `-x` run the data is partial");
- the run attempt is not the latest one (`--attempt`);
- the artifact has expired (14 days).

**If the log is kept anyway:**
- strip gh's `<job>\t<step>\t<ISO timestamp> ` prefix and a leading BOM;
- do not filter on the step name, because gh has printed `UNKNOWN STEP` for every step at times;
- match the job name exactly and the `coverage (pin available):` prefix;
- treat zero pin lines as an error (exit 2: "the ratchet step did not run, or the log is truncated"), never as
  "nothing to pin";
- refuse a duplicate scope that appears with two values.

**Scope.** Keep refusing NEW floors (two runs, `--merge-floors`). Refuse runs whose event is not
`pull_request`/`push`/`schedule`/`workflow_dispatch` on this repo.

### S4. Fail-open (pre-existing): input 1's claim is incomplete, and unknown events skip the diff rules

- The note says a PR cannot set `GITHUB_EVENT_NAME` because `ci_step_rules` pins the step with no `env:`. That check
  covers the step only. `ci_step_rules` reads neither workflow- nor job-level `env:` nor `>> $GITHUB_ENV` writes in
  earlier steps of the same job, and a PR's run uses the PR's own test.yml.
- GitHub documents that `GITHUB_*` defaults cannot be overwritten. I have not verified that for `$GITHUB_ENV` writes.
  The gate itself checks nothing.
- The payoff is not the rise: any event outside `(None, "", "pull_request", "push")` makes `main()` print
  "head rules only" and skip NEW/CHANGED/LOWERING, diff coverage and the ledger diff.
- `merge_group` (a merge queue, if one is ever enabled) and `pull_request_target` would silently take that branch.
- **Fix (cheap, and M60 already edits this branch):**
  - under `GITHUB_ACTIONS`, cross-check the event against `GITHUB_EVENT_PATH`'s payload (`pull_request` key
    present ⇔ PR);
  - make every event other than `schedule`/`workflow_dispatch` run the diff rules, with an unknown event exiting 2
    (fail closed);
  - ban `GITHUB_` keys in any `env:` of the job or workflow inside `ci_step_rules`.
- Fix input 1's wording either way.

### S5. Fail-open: nothing pins the schedule trigger

- A PR can delete `schedule:` from `on:`, or move the cron, and the tightness guard vanishes with no failure.
  `ci_step_rules` checks the job, not the triggers.
- **Fix:** `ci_step_rules` asserts `schedule` is present in the workflow's triggers.
- PyYAML gotcha: the `on:` key parses as the boolean `True`, so read `doc.get("on", doc.get(True))`.
- Also add a positive control in `test_lint_coverage.py`.

### S6. Fail-open (lag): concurrency cancels the nightly

- The scheduled run shares `group: Tests-refs/heads/main` with push runs.
- GitHub cancels a pending run in the group when a newer one queues, as test.yml's own comment says. A nightly that
  queues behind a running push and is then overtaken by a later push never runs, so there is no tightness check that
  night.
- `check_nightlies` copes ("newest not cancelled"). The guard just skips a day without saying so.
- **Fix:** put `github.event_name` in the concurrency group for `schedule`, or have the pin-PR flow read the newest
  non-cancelled scheduled run and print its age.

### S7. The xdist statement needs two more clauses (fail-closed: red runs)

Under `--dist loadfile` (±16 statements per package):
- a single-run pin lands about 16 statements high about half the time, and the next run then fails the fall check
  (k = 5);
- noise rises above the band (16 statements is more than 1 pt for any package under 1,600 statements) flap the
  nightly.

"M22 unblocked for the rise class" therefore also requires that pins under xdist are the minimum of two or more runs
(`--merge-floors`' spread rule, applied to CHANGED pins). With D1, the base-measurement comparison must use a
tolerance that covers order noise too. Say so in M22's row, not only "the fall tolerance".

---

## NIT

- **N1. `--local` to notes loses no real catch.**
  - #1201 and #1240 were rises ("raise it"), which are notes in CI too.
  - #1270 (`pin_rises`) stays gated.
  - #1213 and #1232 were diff coverage, which stays gated.
  - The only loss is a real fall caught one CI cycle earlier, and macOS drift (1–2 pt) made that gate unreliable
    anyway.
  - Fix:
    - print a local fall as a distinct `WARNING coverage fall (verify in CI):` line, not mixed into notes;
    - update ci-testing.md §4 ("`PASSED` also means the coverage ratchet passed") and the `suite_at_commit.sh`
      summary text, because `PASSED` will no longer cover floors.
- **N2.** The release step "a nightly on the merge commit" (publication guide step 4) dispatches on `main`. The
  dispatch is `workflow_dispatch`, so `unit-tests` goes red whenever pins lag, and a pin PR cannot merge during the
  freeze. It is harmless, because `check_nightlies` ignores `unit-tests`, but say so in §2b so an operator does not
  chase it.
- **N3.** A red ratchet step on a nightly skips the "Release gates, runtime twin" step that follows it (no
  `if: always()`). The design makes that red regular. Today that does not matter, because the twin runs on PR and
  push. Note it in the note's "gives up" list.
- **N4. In-flight R/H/S branches.**
  - Their rise pins stay valid, because raising is allowed and CHANGED is verified against their own run.
  - They will conflict with the pin PR on `coverage_floors.json`, and a CONFLICTING PR fires no Tests
    (green-pr lesson).
  - Advise: on the first merge of `main` after M60, drop rise-only pin edits, and keep voluntary raises only where
    the branch wants them.
  - A hand-resolved conflict that takes the higher number is caught by the HEAD k rule (fail-closed).
- **N5.** `workflow_dispatch` on a branch: rises fail and the diff rules are skipped. It is not a merge gate, and per
  S2 its pins are refused unless HEAD matches. Fine once S3's run validation is in.
- **N6.** The note's "Today band alone allows 1 pt" should read "band + k". Its count "7 of the 9 ratchet failures"
  should cite the 40-PR window (2026-10-16), as the lesson does.

---

## Inputs the gate consults after the change (the reviewer's list, with D1/D2 applied)

| Input | How it could lie | Status |
|---|---|---|
| `GITHUB_EVENT_NAME` | job/workflow `env:`, `$GITHUB_ENV` from an earlier step, a new event type → diff rules skipped | S4 |
| `coverage_floors.json` (HEAD + base via `git show`) | stale `missing` pins read as current | D2 |
| `coverage.json` (this job) | pre-suite tampering (documented residual) | unchanged |
| base measurement artifact (new, D1) | missing/expired/cancelled run → fail closed, nearest green ancestor; partial (suite failed) → refuse | D1 |
| run log / artifact for the pinner | other commit/branch/attempt, partial data, truncated log, `UNKNOWN STEP` | S2, S3 |
| the nightly itself | cancelled by concurrency, trigger deleted by a PR, red read by nobody | S1, S5, S6 |
| `--local` working tree | macOS drift, no extras → notes only (correct) | N1 |
