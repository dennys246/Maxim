# M60a code review: Wire integrity lens

Diff: `git diff origin/main...HEAD` at 587a25e7 (branch `ci/coverage-ratchet-one-way`). Read-only review.
Narrow check run: `pytest tests/unit/test_lint_coverage.py -k "pins or run_measure or measure or load_coverage"`, 31 passed.

**Verdict: no BLOCKER.** Every machine consumer outside the diff still holds. What is left is stale prose that
describes the removed `--local` behaviour, plus two names that nothing pins to `test.yml`.

## 1. Artifact name `coverage-json` → `coverage-json-${{ github.run_attempt }}`

| Consumer (outside diff) | Holds? |
|---|---|
| Other workflows (`.github/`) | Yes. Nothing else downloads or names the artifact (grep `coverage-json` over the repo). |
| `scripts/pr_merge_readiness.py`, `scripts/check_nightlies.py`, `scripts/lint_release_gates.py` | Yes. None of them reference coverage or the artifact. |
| Runbooks/docs saying `gh run download -n coverage-json` | Yes. There are none: the only `gh run download` in the repo is the new `_download_artifact`. |
| `docs/lessons/development-flow-speed.md` ("Reproduce where it fails", #1103) | Yes. It says "two CI runs of one commit, diffed by `executed_lines`" and names no artifact, so it stays true. |
| Memory `project_session_v_atonia.md` line 104 ("merge-base's green push run `coverage-json` artifact") | This describes the rejected v2 design, so it is history, not an instruction. No action needed. |
| `test.yml` upload step: name "Upload coverage.json (#1103, diagnostic)", with the comment "Diagnostic only" | **SHOULD-FIX.** The artifact is now load-bearing, because `--pins-from-run` consumes it, but the step's name and comment still call it diagnostic-only. A future cleanup could delete the step, or set `if-no-files-found: ignore`, believing it is unused. **Fix:** rename the step to something like "Upload coverage.json (#1103 diagnostic; M60a --pins-from-run input)" and add one comment line that names `lint_coverage.py::COVERAGE_ARTIFACT` as the consumer. |
| Pin between `COVERAGE_ARTIFACT` and `test.yml` | **SHOULD-FIX (see §6).** No test asserts that the workflow uploads `f"{COVERAGE_ARTIFACT}-${{{{ github.run_attempt }}}}"`. The unit tests feed the name from the constant itself (`_api(..., artifacts=("coverage-json-1",))`), which is self-referential. A rename in `test.yml` would make every `--pins-from-run` refuse with "no unexpired artifact", and no CI check would catch it. It fails closed, but it is silent until someone uses the tool. |
| Re-run of failed jobs only: attempt N may have no `unit-tests` job or no `coverage-json-N` | NIT. It fails closed: the tool refuses with a named reason. You could mention "re-run the whole workflow" in the refusal text. |

## 2. `--local` output and exit code

- `scripts/suite_at_commit.sh` reads only the exit status (`python scripts/lint_coverage.py --local || status=$?`) and parses no text. **Holds.** The exit-2 path for "no base" is gone from the floor branch. The diff-coverage path still needs a base, so a missing base still gives exit 2 through that path, which is the same contract.
- `scripts/prepr.sh` does not run `lint_coverage.py`. **Holds.**
- The old note text "untouched scope, measured in CI's environment, not gated" is grepped by no test or tool (zero hits outside the diff). **Holds.**
- **SHOULD-FIX: stale prose consumers of the old contract.**
  - `scripts/suite_at_commit.sh` header, lines 9–12. It still says "the floors of the packages the diff touches fail HERE … The untouched packages' floors are only noted". Under M60a no floor gates locally. **Fix:** rewrite it as "diff coverage and the measurement-free pin rule (`pin_rises`) fail HERE; every floor result is a note (M60a): floors are CI's Linux measurement, pinned with `--pins-from-run`."
  - `docs/plans/outstanding.md` row M47. It cites the deleted symbol `local_touched_packages` and says touched-package floors "gate as in CI". That is now a dangling citation and a false description. **Fix:** update the row to M60a's behaviour. Its "Residue" also changes: no floor is gated locally now, not only the overall floor.
  - `docs/agents/ci-testing.md` line 24 (the inventory row). It lists `--local`, `--merge-floors` but not `--pins-from-run`. NIT: add it.

## 3. CHANGED-rule message "min(measured X + 5 of noise, base Y)"

- Readers: humans in CI logs and `tests/unit/test_lint_coverage.py` (updated in the diff). `pr_merge_readiness.py` and the docs do not parse "floor pins missing" or quote the old `min(measured X, base Y)` form. The `pin_rises` message (line 931, "above the base's pinned") is unchanged. **Holds.**
- The printed `pin {pin}` is `min(m.missing, b.missing)`, which is consistent with `pins_from_measurement` and the HEAD hint (#1409). **Holds.**
- NIT: the CI "raise it to {…}" hint (`floor_head_rules`, line 673) does not mention `--pins-from-run <this run id>`, even though the brief now names it as THE fix. A one-line suffix ("or: `python scripts/lint_coverage.py --pins-from-run $GITHUB_RUN_ID` on this commit") would close the loop from failure to tool.

## 4. `load_coverage(root, path=None)`

The callers are `main()` (line 2057, positional `root` only), `run_measurement` (new), and tests at lines 2446/2452/2467 (`L.load_coverage(repo)`). No other script imports it: `suite_at_commit.sh` imports `lint_coverage` only for `COVERAGE_VERSION`. The default keeps the old behaviour, including the "stale file?" existence check. **Holds.**

## 5. `coverage_floors.json` and the `apply_pins` writer

- Other readers and writers: `lint_coverage.py` itself (`parse_floors`, `pin_rises`, `floor_diff_rules`, the base read via `show`) and `test.yml`, whose only use is the step comment. `--merge-floors` and `--suggest` print JSON to stdout and never write the file. `roadmap_1_3_x.md` and the brief are prose. **Holds.**
- Format: the committed file round-trips byte-exactly through `json.dumps(json.loads(t), indent=2) + "\n"` (checked: True). Key order is preserved: dict insertion order, and pins only overwrite existing keys, because `pins_from_measurement` emits only floored scopes. `percent` is a float from `round_down`, so it is written as e.g. `60.0`, and the file already uses float literals. `parse_floors` checks that the top-level keys are exactly the expected set, and `apply_pins` adds none. **Holds.**
- NIT: `apply_pins` routes by `name.startswith("src/maxim/")`, which is correct today (file keys are `src/maxim/…`, package keys `maxim/…`). An `assert name in slot` before writing would make a future key-shape drift fail loudly instead of adding a new entry.

## 6. Step name "Run required fast suite" (`RUN_SUITE_STEP`) and job `unit-tests`

- Pinned elsewhere? **No.** `ci_step_rules` finds the measuring step by `--cov` in its `run`, not by name. `test_lint_release_gates.py` uses the literal only in synthetic YAML fixtures. The unit test builds its fake API response from `L.RUN_SUITE_STEP`, which is self-referential. The job has no `name:` key and no matrix, so the API job name is `unit-tests` today. Adding a `name:` or a matrix would also break the lookup.
- A rename of the step, the job or the artifact makes `--pins-from-run` refuse every run ("step … is absent", "no unit-tests job", "no unexpired artifact"). It fails closed, but CI is green and the breakage shows only when a human needs the tool.
- **SHOULD-FIX: add one test** in `tests/unit/test_lint_coverage.py` that parses the real `.github/workflows/test.yml` and asserts:
  1. `jobs[CI_JOB]` has no `name:` and no `strategy.matrix`, so the API job name equals `CI_JOB`;
  2. exactly one step in `jobs[CI_JOB]` has `name == RUN_SUITE_STEP`, and it is the step whose `run` contains `--cov`, the same index `ci_step_rules` finds;
  3. a step with `uses: actions/upload-artifact@…` has `with.name == f"{COVERAGE_ARTIFACT}-${{{{ github.run_attempt }}}}"`, `with.path == COVERAGE_JSON` and `if: always()`.

  Alternatively, fold these three checks into `ci_step_rules`, so the lint job enforces them on every PR. That is stronger, because it fails the PR that renames the step.

## Summary

| # | Class | Finding |
|---|---|---|
| 6 / 1 | SHOULD-FIX | Step name, job name and artifact name are not pinned to `test.yml`; add the parse-the-workflow test (or extend `ci_step_rules`). |
| 2 | SHOULD-FIX | Stale prose about touched-package floors: the `suite_at_commit.sh` header and the `outstanding.md` M47 row (which cites the deleted `local_touched_packages`). |
| 1 | SHOULD-FIX | The upload step is still labelled "diagnostic only" but is now a tool input; rename it and add a comment naming the consumer. |
| 2 | NIT | `ci-testing.md` inventory row lacks `--pins-from-run`. |
| 3 | NIT | The CI "raise it to" hint could name `--pins-from-run $GITHUB_RUN_ID`. |
| 5 | NIT | `apply_pins` should assert the key exists before writing. |
| 1 | NIT | The refusal on a re-run-failed-jobs attempt could suggest re-running the whole workflow. |
