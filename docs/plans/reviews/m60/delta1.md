# M60 delta adversarial pass 1: DRAFT v2

> Reviewer: a delta adversarial reader, different from the v1 pass. The work was read-only. It covers
> `docs/plans/coverage_ratchet_m60_approach.md` (DRAFT v2) against `scripts/lint_coverage.py` (docstring,
> `floor_head_rules`, `floor_diff_rules::changed_entry`, `pin_rises`, `changed_lines`, `_line_statements`,
> `diff_coverage`, `load_coverage`, `ci_step_rules`), `scripts/_lint_git.py` (`base_ref`, `must_not_skip`,
> `push_base`, `_push_event`, `_accepted`), `.github/workflows/test.yml` (triggers, concurrency, `unit-tests`,
> `release-build`, the `coverage-json` upload) and `scripts/check_nightlies.py`. Date: 2026-10-27. No tests were run;
> every finding is from reading the code.

**Verdict: DO NOT BUILD as written.** The fall rule in v2 subtracts changed statements on the HEAD side only:
- That one-sided subtraction reopens D1 for the cost of a reformat.
- The base side must be treated the same way, and the diff must be taken against the baseline commit, not the
  merge-base.

Two fail-open SHOULD-FIX items also block the build:
- an artifact from an earlier attempt of the base run;
- a no-baseline fallback that a PR can trigger.

The remaining findings are fail-closed or NITs.

A fact the note misses, and several findings depend on it: on `pull_request`, `actions/checkout` checks out
`refs/pull/N/merge`. That is the PR merged into main's tip at the time of the event. So `base_ref`, which computes
`merge-base(HEAD, origin/main)`, returns **main's tip T**, not the PR's branch point.
- The head measurement includes every commit on main up to T.
- T's own push run is usually still running (about 40 min) or was cancelled while pending. So the newest green
  baseline B is usually older than T.
- The note's "newer than the merge-base" case cannot arise on a fresh run. The reverse case, B older than T, is the
  common one.
- "Merging main gives a fresh baseline" is a no-op, because every run already re-merges main.
- 14-day expiry matters only when main has had no green push for 14 days.

---

## DO-NOT-BUILD

### X1. Fail-open: the fall rule is asymmetric, so touching uncovered lines buys headroom (reopens D1)

**Why it fails open.** The rule is `unchanged_missing(head) <= base.missing + k`:
- The left side drops every uncovered statement on a line the diff adds or modifies.
- The right side still counts the base versions of those same statements.
- So every uncovered statement a PR *touches* is pure headroom.

**Attack.** One PR does three things:
1. It reformats or renames a local in 100 uncovered statements of `maxim/X`.
2. It touches 400 covered statements, for example by renaming a variable in a tested function. The diff coverage of
   the whole diff is then 400/500 = 80%, which passes. Diff coverage is pooled across files, not per file.
3. It deletes or skip-marks tests covering 100 untouched statements elsewhere in `maxim/X`. Diff coverage reads
   `src/maxim` only, so this change is invisible to it.

**Result.**
- `unchanged_missing(head)` = base − 100 + 100 = base, so the PR passes.
- The real fall is 100 statements.
- Only the floor backstop sees it, and v2 lets floors go stale (rises are notes, and the nightly blocks release, not
  merges). So this is D1's unbounded slack again, at the price of a cosmetic edit.

**The same asymmetry shows up in other forms:**
- **Moving a file between packages.** The diff is `--no-renames`, so the whole file counts as changed at its new
  path. The source package and `overall` gain the file's uncovered count as headroom. No code is deleted, which
  contradicts the residual "it costs a real deletion".
- **Deleting a file.** This is the documented residual, but it is not needed.
- **Adding a file to the exclusion list.** It drops out of the head `coverage.json` but stays in base `missing`.

**Line shifts are not a problem:** the rule compares counts, so shifted lines are neither missed nor double-counted.

**Fix: compare like with like.**
- Define `unchanged_missing(base)` as base-artifact uncovered statements of files present and measured on both sides,
  minus the statements that the diff's `-` hunks (deleted or modified lines, in base line numbers) map to.
- Use the base-side mapping the split rule already uses: `parse_unified_diff` old-side hunks →
  `_line_statements(base_src, …)`.
- The rule becomes `unchanged_missing(head) <= unchanged_missing(base) + k`.
- Files that are deleted, moved, newly excluded or newly added then contribute zero on both sides, so the
  deletion residual closes.
- Keep it net (a scope can trade tests between untouched statements), not a one-sided count of flips: one-sided
  noise would accumulate.
- **Deletion probe:** run the reformat attack above and a cross-package `git mv`. Both must fail.

### X2. The diff must be taken against the baseline commit B, not the merge-base T (fail-closed and fail-open)

**Why.** Because of the merge ref, the head contains the commits from B to T, which were merged by other PRs. Those
commits' lines are "unchanged" in the PR's diff from T.

**Fail-closed.** Suppose a PR merged between B and T at 85% diff coverage, leaving 15 uncovered statements. Every PR
then fails the fall rule in that package (k = 5). The cause is someone else's code, and the failure lasts until a
newer main run goes green.

**Fail-open.** If a commit between B and T deleted uncovered code, that deletion is headroom.

**Fix.**
- Compute the fall rule's changed-line sets (both sides, per X1) from `git diff B -- src/maxim`.
- Keep diff coverage on the merge-base T.
- `push_base` already gives B on a push. On a PR it needs a `start=` parameter, because `push_base` hard-requires a
  push event: `_push_event` reads `before`/`after` and raises on a PR.

---

## SHOULD-FIX

### S1. Fail-open: the artifact is not bound to the attempt whose suite succeeded

**The gap.** The design checks two things: the latest attempt's suite step concluded `success`, and the run's
`head_sha`. Two points from the code and workflow:
- The artifacts API lists a run's artifacts with no attempt number.
- The upload step is `if: always()` and named `coverage-json`.

After a re-run of a flaky main run, there are two possibilities, and I have not verified which applies:
- **The run holds two `coverage-json` artifacts.** One of them is attempt 1's partial `-x` data.
- **The v4 same-name 409 turns the re-run red.** This is the known v4 behavior, upload-artifact issue #478.

**Why partial data fails open.** A partial artifact has *higher* `missing`, which loosens the fall rule. The cap
`min(...)` is unaffected, but the fall rule's right side is not. The note's input 4 says "at worst a forged baseline
is LOWER than the truth"; that is the wrong direction for partial data.

**Fix.**
- Name the artifact `coverage-json-${{ github.run_attempt }}`, and select the name that matches the jobs API's
  `run_attempt`.
- Refuse an ambiguous selection, which means no baseline.
- Pin the name in `ci_step_rules`.

### S2. Fail-open: a PR can trigger "no baseline", which silently restores today's band+k slack

The fallback is fail-closed relative to today, but a PR can choose it, and it disables the guarantee v2 adds. Three
ways in:

1. **Reusing `load_coverage` for the base artifact.** It raises `MeasurementError` when a named file does not exist
   at HEAD (`(root / rel).is_file()`). So any PR that deletes or moves one file would make the baseline unreadable.
   It also raises when the PR bumps `COVERAGE_VERSION`.
2. **Renaming the suite step.** The lint matches the step by its name. A PR that renames "Run required fast suite"
   has no matching step in main's runs.
3. **The fallback is quiet.** It is one print line on a green job.

**Fix.**
- Write a separate base loader that does not check HEAD paths. Files gone at HEAD are simply excluded, per X1.
- Treat a version mismatch, or a step-name change in this diff, as no baseline, and print it as a `::warning::`
  annotation.
- Treat every non-2xx API response as exit 2. "Missing" may only mean a successful listing that does not contain the
  artifact, or one that shows `expired: true`. A 403 or 404 caused by a dropped `actions: read` must not read as
  missing.
- Pin `actions: read` on `unit-tests` in `ci_step_rules`.

### S3. Fail-closed: the pinner's "holds by construction" is false

**Why.**
- `--pins-from-run` reads run 1, which is of the PR head merged with T₁.
- Committing the pins creates a new HEAD, and CI verifies it on run 2, which is of the PR head merged with T₂.
- `changed_entry` requires `h.missing <= min(m₂.missing, …)`.
- With ±2 statements of noise, and with any tests merged into main between the runs, `m₂ < m₁` for some scope is
  close to certain when many scopes are pinned. This is S2 from the v1 pass, unchanged.

**Fix.** Since v2's measured cap now bounds stale-high pins, relax only the measurement side:
`h.missing <= min(m.missing + k, b.missing, bm.missing)`.
- The pin still never rises above the base or main's measurement.
- The next LOWERING reads `bm`, so the k slack does not compound.

### S4. Fail-closed: the cap makes legitimate lowerings unpassable

The cap is `min(b.missing, bm.missing)`. The LOWERING rule has no tolerance, so there are three problems:

1. **Noise on main.** If main measured `bm` 2 statements noise-low, a PR that deletes covered code needs its own run
   to be noise-low by 2 as well. Re-running the PR helps only by chance, because `bm` is fixed.
2. **Commits between B and T.** These add legitimate uncovered code (per X2). Then `m.missing > bm.missing` by
   someone else's statements, and the author is told to cover them.
3. **Undefined cases.** The cap is not defined for a scope `bm` lacks: a package created after B, or a rename
   target. It must fall back to `b`.

**Fix.**
- Apply `bm` only when B == T, or when B..T touches none of the scope's files. Otherwise use `b`.
- Use `bm.missing + 2` (the observed noise), or re-measure. Do not use a bare `bm`.
- Add a test for both cases.

---

## NIT

- **N1. Diff coverage gives no per-scope counts today.** `diff_coverage` returns only message lists. Its per-file
  `seen` map (changed statement start lines, covered or not) is local. Return `{rel: set(changed stmt lines)}` from a
  shared helper, and add the base-side twin for X1.
- **N2. Acceptances end the walk.** `push_base` returns an accepted sha (from `push_base_accepts.json`) before any
  older green run. That commit's job was red, so the step check is load-bearing there and must stay.
- **N3. The tightness job.** `check_nightlies` reads it correctly:
  `release-build` (PR) and `release.yml --at` → `latest_nightly`/`nightly_at` → `nightly_verdict`, where every
  `(nightly)` job must be `success` → exit 1.
  - With `needs: unit-tests`, a red suite *skips* the tightness job. That still blocks the release, but it prints
    "Coverage tightness (nightly): skipped". Say so in §2b.
  - `ci_step_rules` must also pin the job's `needs`, `if`, and the absence of `continue-on-error`.
  - §8 also has `unit-tests` running tightness on `schedule`, so tightness runs twice. Pick one.
- **N4. Concurrency.** No test pins test.yml's group, and no workflow `env:` holds a `GITHUB_*` key, so the ban and
  the group change break nothing. Two side effects:
  - `workflow_dispatch` (the release "nightly on the merge commit") still shares the push group.
  - Scheduled runs now run in parallel with push runs on main. That adds runner demand; see the 2026-08-06
    incident.
- **N5. Missing residual.** Restore the v1 pass's residual: a walk-down of up to k per PR is cumulative, and only the
  (possibly stale) floor bounds it.
- **N6. Wrong reasoning in input 4.** Correct the line "at worst LOWER → stricter" (see S1).
- **N7. Wrong expiry wording.** Correct the "Expiry"/"merge main" wording per the merge-ref fact above.

**Not STOP.** X1 (DO-NOT-BUILD) and S1/S2 (fail-open) must be folded first, followed by a delta round on X1/X2.
