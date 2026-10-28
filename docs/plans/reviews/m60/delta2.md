# M60 delta adversarial pass 2: v3 "M60a"

> Reviewer: a third reader, different from the v1 and v2 passes. Read-only. Covers
> `docs/plans/coverage_ratchet_m60_approach.md` (v3) against `scripts/lint_coverage.py` (module docstring,
> `missing_slack`, `floor_head_rules`, `_sum_rule`, `pin_rises`, `floor_diff_rules::changed_entry` and `namespace`,
> `load_coverage`, `ci_step_rules`, `local_touched_packages`, `main()`), `scripts/suite_at_commit.sh`,
> `.github/workflows/test.yml` (`unit-tests` steps, the upload step) and `scripts/coverage_floors.json` (to size k).
> Date: 2026-10-27. No tests run; findings are from the code.

**Verdict: not STOP.** No DO-NOT-BUILD. One fail-open SHOULD-FIX (P1, a one-token fix), so P1 must be folded first.
The other findings are fail-closed or NITs. Change 1 and change 3 are sound as designed.

---

## SHOULD-FIX

### P1. Fail-open, bounded: the +k cap uses `missing_slack(m)`, which is 96 statements on `overall`, and it feeds LOWERING through `b.missing`

**The claim that fails.** The note says "LOWERING stays exact". The LOWERING test (`m.missing > b.missing` fails) is
unchanged, but its right side `b.missing` is now a pin that may sit k above the measurement it was verified against.

**Walk (any scope, per the current code plus change 2):**
1. PR1 re-pins, by hand, with `missing = m1 + k`. That passes `min(m.missing + k, b.missing)` whenever the base pin
   allowed it. The pinner is not needed; any hand pin may do this.
2. PR2 deletes or skip-marks tests, adding k uncovered statements. The HEAD percent rule passes, because of its own k
   tolerance. In the same PR, it LOWERS the percent floor to `round_down(measured)`. The check `m2 = m1 + k <= b` passes.
3. PR3 drops another k under the lowered floor, and the HEAD tolerance passes it.
4. PR4 cannot lower again, because `m1 + 2k > b`.

**The size of the hole:**
- Today the escape is k. PR2's lowering is refused ("cover k statement(s) first").
- With change 2, the escape is 2k.
- It is non-cumulative: the cap's base side keeps a pinned `missing` monotone non-increasing. So the note's "not
  cumulative" claim holds, but "bounded at k" and "LOWERING stays exact" do not.

**How big k is:** `missing_slack` is the 0.1-pt tick, so it grows with scope size. Measured from today's floors file:
- `overall` ≈ 95,800 statements, so k = **96**;
- `maxim/runtime` k = 14;
- `maxim/simulation` k = 11.

The noise this slack exists to absorb is ±2 statements whatever the size, which is the docstring's own reason for
`K_MIN`. So a 96-statement real fall on `overall` becomes lowerable, to absorb 2 statements of noise.

**The same inflated `b.missing` reaches two other rules:**
- **The SUM rules.** Both the package rename/split SUM and the SF-2 file split SUM take their budget from
  `base_missing` and `base_map[t].missing`. A group's budget therefore grows by the slack of every pin in it. That is
  bounded and non-cumulative, because successors' NEW pins stay exact and pins never rise. But a split of two large
  packages carries about 25 statements of slack.
- **`pin_rises`.** It is unaffected; it compares two pins.

**Fix.**
- Use an absolute noise constant, not the tick: `cap = min(m.missing + K_MIN, b.missing)`, where `K_MIN` = 5 is
  already defined as "twice the observed ±2".
- Today a pin hand-taken from a noise-high run already carries about +2, so this widens the LOWERING escape from about
  k + 2 to k + 5, which is noise-scale.
- Rewrite the note's "Bound" and "Unchanged" bullets. The honest statement is: "a pinned `missing` may sit up to
  `K_MIN` above the measurement it was verified on; LOWERING and the split sums read that pin, so their escape grows
  by `K_MIN` per pin; pins never rise, so it does not accumulate."
- Add a test: lowering at `b.missing = m1 + K_MIN` with measured `m1 + K_MIN` passes, and with `m1 + K_MIN + 1` it
  fails.
- Add a deletion probe: replace `K_MIN` by `missing_slack(m)` on a 10,000-statement scope, and the test must go red.

### P2. Fail-closed: "measured file missing from the checkout → refuse" fires on the common case

**Why it fires so often.**
- On `pull_request`, the run's `head_sha` is the PR head commit, so binding it to `HEAD` is correct.
- The data, however, is of `refs/pull/N/merge` at main's tip T1.
- Every file main added after the branch point is therefore "missing from the checkout".
- "Merge main first" makes a new HEAD, which needs a new run, and that run merges a newer T. On a busy main the loop
  need not converge.

**Why refusing is unnecessary.** Path normalisation does not need the file to exist. The pins are of the merge, which
is what lands on main, and CI re-verifies them on the pin commit's own merge run.

**Fix.**
- Do not reuse `load_coverage`'s existence check in the pinner.
- Normalise the paths, check `meta.version == COVERAGE_VERSION` and `branch_coverage`, and compute the scopes from
  every measured file.
- Refuse only when a key being rewritten (a file floor) is absent from the checkout.

### P3. Fail-closed: the pinner skips the other common CHANGED failure

The pinner rewrites only scopes that are more than band above their floor. A branch pin whose `missing` exceeds
`min(m.missing + slack, b.missing)` on the latest run is reported nowhere by the pinner. That case is the S3 noise
case, or a hand pin.

**Fix:** also print, without rewriting, every scope whose current branch pin would fail `changed_entry` against the
artifact's measurement, with the pin to use. The cleanest way is to call `changed_entry`'s logic and not duplicate it.

---

## NIT

- **N1. Docstrings become false.** Three places need updating in the same PR:
  - The module docstring says of the k tolerance: "never applies to the pin rules below, and pins never move with it".
  - The `missing_slack` docstring says the same.
  - `pin_rises`' docstring says its printed `min(measured, base)` is "the cap `changed_entry` enforces".

  Its printed pin still passes, so only the wording is stale.
- **N2. File floors.** `changed_entry` is shared by the packages and files namespaces, and by rename carry-over
  targets (`"t (carrying k)"`). Change 2 therefore applies to all three automatically. That is right, because the
  pinner rewrites file floors too, and with P1's `K_MIN` a file gets the same 5 its HEAD tolerance already allows. Say
  so in the note. "In the CHANGED rule" reads as packages only.
- **N3. Change 1 keeps gating only "format and fixed fields".** Today `floor_head_rules` mixes the format checks with
  per-scope checks. Several per-scope checks read no measurement:
  - orphan (no directory or file at HEAD);
  - a symlinked file-floor key;
  - the case-exact listing;
  - a `null` floor;
  - a package of N or more statements with no floor. Statement counts come from the AST and coverage's `exclude_lines`,
    and are platform-stable.

  By the `pin_rises` rationale (#1270), these can keep gating locally. Demoting them loses no pre-merge catch, because
  CI's `floor_head_rules` is unchanged, so this is not fail-open; it is a CI round trip saved. Either split them out
  of `floor_head_rules` as "measurement-free" and gate them, or say they become notes.

  Nothing is caught nowhere: CI runs every floor rule as today, and `local_touched_packages` has no caller outside
  `main()` and its tests.

  Also update `suite_at_commit.sh` lines 9–10 and ci-testing.md line 85 ("`PASSED` also means the coverage ratchet
  passed"), plus the v1 N1 `WARNING coverage fall (verify in CI)` line.
- **N4. Run binding: add the cheap checks.** Also refuse:
  - a run whose `path` is not `.github/workflows/test.yml`;
  - a run whose `event` is not `pull_request`, `push`, `schedule` or `workflow_dispatch`;
  - an artifact listing with more than one name matching `coverage-json-<attempt>`;
  - an artifact with `expired: true`;
  - any non-2xx response, which exits 2 and never counts as "not found".

  All of these fail closed today anyway, because CI re-verifies the pin.
- **N5. A merge-ref pin is valid.** Main's merge carries the measured content, so it is the scope that lands. Its
  dependence on T1 is fail-closed: if main moved and lowered coverage, CI on the pin commit refuses. The note's input
  1 says this; keep it.
- **N6. The artifact rename.** It breaks no consumer. A grep of `scripts/`, `tests/`, `docs/` and `.github/` finds
  `coverage-json` only in the test.yml upload step. `ci_step_rules` does not read the upload step, and nothing
  downloads the artifact.

  The legacy name on attempt 1 only is correct: a re-run reuses the original workflow definition, so an old-workflow
  attempt 2 would carry the old name (and hit the v4 duplicate-name conflict). Refusing it is right.

  Optionally pin the upload step's `name` in a workflow-shape test, so a rename shows up there rather than as a
  pinner failure.
- **N7. Pinner formula against CI after the change.**
  - The pin is `{round_down(m_A.percent), min(m_A.missing, f.missing)}`. On run B it passes when:
    - `m_A ≤ m_B + slack`, which holds within noise;
    - `f.missing ≤ b.missing`, which holds unless the branch already broke `pin_rises`, a fail-closed failure CI
      reports;
    - the percent is at least `lowest_B` (the CHANGED rule) and at least `m_B.percent` − band (the HEAD rise
      check). Both hold within noise.
  - So a pinner pin passes within noise, but only when the pinned scope's `f` is not a branch-LOWERED floor. Then
    `h.percent < b.percent` takes the exact LOWERING branch, which is fail-closed. Have the pinner skip scopes where
    `f.percent < b.percent`, and report them.

**Not STOP.** Fold P1, which is fail-open. P2, P3 and the NITs can be folded in the same pass or filed. After P1 is
folded, the round owed is delta-scoped to `changed_entry`'s cap and its test only.
