# M60a pre-merge code review: Architecture lens

Diff: `git diff origin/main...HEAD` at 587a25e7 (branch `ci/coverage-ratchet-one-way`). Read-only review; narrow
pytest `-k "pins or local or changed"` on `tests/unit/test_lint_coverage.py`: 39 passed.

**Verdict: no BLOCKER.** The mode fits the module, and every new input fails closed. Five SHOULD-FIX items are
all the same kind: the rule book and the docs around it are stale or contradict the new behaviour.

## SHOULD-FIX

**A1. The rule book (module docstring) does not describe `--pins-from-run`.** `--merge-floors` has its own
paragraph (section "Bootstrap"). The new mode is described only in the `load_coverage` and `run_measurement`
docstrings and in ci-testing.md. Its refusals (SHA, workflow and event, latest attempt, step `success`, the
attempt-bound artifact with legacy fallback on attempt 1) and its pin formula (`round_down(m.percent)`,
`min(m.missing, f.missing)`; nulls, falls and NEW floors are left alone) are rules. The footer "Stdlib plus `pyyaml`"
also needs "`--pins-from-run` shells out to `gh` (exit 2 without it)".
*Fix:* add a short paragraph next to the `--merge-floors` one, and amend the footer.

**A2. The k-tolerance paragraph now contradicts the CHANGED rule.** Section 1 says the tolerance "applies only to
this HEAD comparison against the committed pin, never to the pin rules below". The CHANGED pin rule now carries
`K_MIN` slack, and `K_MIN` is k's minimum.
*Fix:* add "except the CHANGED cap's `K_MIN` (M60a, below), which is bounded by the base pin".

**A3. The slack weakens "every new or changed floor is verified data", and its cost is documented only in the
approach note.** The K_MIN slack does keep the owner's "a pinned missing never rises": the base side stays exact,
and pins are non-increasing, so the slack does not accumulate. It does let a CHANGED `missing` sit up to 5 over the
measurement. Every rule that reads a base `missing` pin then inherits that headroom:
- LOWERING (`m.missing <= b.missing`, "no tolerance");
- the package rename/split sum;
- the SF-2 file-split sum, also "no tolerance".

A sum over N pins can therefore absorb up to 5·N statements. That is more than the approach note's "a 5-statement
escape in total".

The docstring also cites this as an "owner decision 2026-10-27". The approach note's recorded owner decisions are
"do M60" and "ship the safe half". The slack itself came from delta1 S3 and delta2 P1.

*Fix:*
- Add a Residuals bullet: the slack costs at most `K_MIN` per pin, against LOWERING and both sum rules, and does not
  accumulate because pins never rise.
- Correct the approach note's bound to cover the sums.
- Either get the owner to confirm the slack explicitly, or attribute it to "M60a design rounds (delta1 S3), under
  the owner's ship-the-safe-half decision".

**A4. `docs/plans/outstanding.md` row M47 still describes the deleted behaviour.** It cites
`local_touched_packages` and says "the floors of TOUCHED packages gate". Its "Residue" also calls the overall floor
"never gated locally" as if that were a gap, but under M60a that is now the design.
*Fix:* rewrite the row to say `--local` gates diff coverage, `pin_rises` and the format, and that no floor gates
(M60a).

**A5. `scripts/suite_at_commit.sh` header comment (lines 9–12) is stale.** It says the floors of the packages
the diff touches "fail HERE" and that the untouched ones are only noted.
*Fix:* say that diff coverage and pin rises fail here, and that every floor is a note (CI is the floor authority).

No code, test or script calls `local_touched_packages` or `_package_of_module`. Outside the review files, only
M47 (A4) and the approach note mention them, so their deletion is otherwise clean.

## NIT

**N1. The `gh` subprocess helper is in the wrong module.** `_download_artifact` duplicates `_lint_git.gh_api`'s
subprocess pattern (timeout, `GitUnavailable`). *Fix:* move it into `_lint_git.py` as `gh_run_download`, so the
`gh` calls live in one module. The `subprocess` and `gh` dependency is otherwise acceptable: `push_base` already
uses `gh_api` in CI, and this mode fails closed with exit 2.

**N2. Workflow drift breaks the pinner silently.** `ci_step_rules` does not need the new mode: CI can never reach
it, because the gate step is pinned to the bare command, and the pinner gates nothing. But the pinner depends on
two workflow strings that nothing pins: the step name `RUN_SUITE_STEP` and the upload name
`coverage-json-${{ github.run_attempt }}`. If either drifts, the pinner refuses every run. That is fail-closed, but
nobody learns of it until the pinner is used.
*Fix:* add a workflow-shape test (`tests/unit/test_ci_workflow_shape.py` or test_lint_coverage) that asserts:
- the measuring step's `name` equals `L.RUN_SUITE_STEP`;
- the upload step's `with.name` equals `f"{L.COVERAGE_ARTIFACT}-${{{{ github.run_attempt }}}}"`.

**N3. `--pins-from-run` does not refuse under `GITHUB_ACTIONS`.** Nothing reaches it in CI, and it writes only a
tracked file. A refusal would mirror `--local` and keep a future workflow step from rewriting floors mid-job.
*Fix:* a one-line refusal plus a test (optional).

**N4. The approach note's "Inputs" item 4 still says "+k slack, bounded at k".** That predates delta2 P1.
*Fix:* make it `K_MIN`.

**N5. A function-local `import tempfile`.** *Fix:* hoist it to the module imports.

## Placement and process

- **Placement.** `--pins-from-run` belongs in `lint_coverage.py`. It reuses `load_coverage`, `measure_scopes`,
  `file_scopes`, `Floor`, `round_down`, `BAND` and `K_MIN`, mirrors `--merge-floors`, and must print exactly the CI
  hint's pin. A separate script would duplicate those, or import them anyway.
- **CHANGELOG.** None is owed (no `src/` change).
- **M60b.** Its requirements are durable: the M60 row links `reviews/m60/` as the requirements list and names
  #1313 and #1103, and M22 carries the order-noise prerequisites.
- **"Enforced, or on the backlog".** "Pin only from CI's run" is enforced by CI's own CHANGED rule on the pin
  commit, so no new backlog row is owed.
- **ci-testing.md §4.** Its text matches the code.
