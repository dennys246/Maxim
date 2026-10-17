# CI & testing infrastructure — working brief

> Part of the CLAUDE.md satellite layer. Read this whole file before editing `.github/workflows/`, `scripts/lint_*.py`, `scripts/suite_at_commit.sh`, `scripts/pr_merge_readiness.py`, `scripts/coverage_*.json`, `tests/conftest.py`, `tests/network_guard.py`, or the `[tool.pytest.ini_options]` / `[tool.coverage.*]` tables in `pyproject.toml`. The slim CLAUDE.md core + this brief are intended to be sufficient context for work in this area. Measurements and owner decisions behind it: [docs/lessons/ci-efficiency-2026-10-16.md](../lessons/ci-efficiency-2026-10-16.md).

## 1. Mental model

**Where the time goes (measured 2026-10-16).** The suite is 15,427 tests (unit 14,508, substrate 352, integration 336, behavioral 173, other 58). CI's `unit-tests` job takes a median 38.6 min: the fast suite step alone is ~34.7 min because it runs serially under `--cov --cov-branch`, and install is ~3.1 min, of which ~2m50s is `llama-cpp-python` building from sdist (there is no pip cache). A local `scripts/suite_at_commit.sh` run is estimated at 50–60 min. `pytest-xdist` is installed (the `[test]` extra and CI's pip line) but no run uses it. A handful of slow tests dominate: 11 tests at ≥5 s were 44% of a serial sample's time.

**What CI catches that local runs do not.** In the last 40 merged PRs, CI caught **zero functional test failures**: 9 of its 10 failures were the coverage ratchet (7 exact-pin or rise-above-floor bookkeeping, often after merging `main` or because macOS and Linux measure differently; 2 real gaps) and one was the PR-title lint. So the local suite is where defects are found; CI is where the coverage measurement, the Linux environment, the network namespace and the diff-scoped lints are authoritative.

**The gating surfaces.** A PR merges on the required checks `unit-tests`, `lint` and `Release build (wheel contents + version)` (`scripts/repo_settings_expected.json`), plus ruleset gates (CodeQL alerts) that `gh pr checks` does not render. `scripts/pr_merge_readiness.py <N>` reads all of them. A check that never ran looks exactly like one that passed (CLAUDE.md's green-PR lesson). Every merge re-runs the full workflow on push to `main`. The PR trigger includes `edited` (for base retargets), so a title or body edit also restarts a ~38-min run.

**Final full-suite run (owner decision 2026-10-16, not yet in force).** The owner adopted "CI's green, non-cancelled `unit-tests` run on the PR's head SHA counts as the final full-suite run". It takes effect when backlog row M64 makes `pr_merge_readiness.py` assert it. Until then CLAUDE.md's rule stands: the last full local suite runs on the exact commit pushed.

## 2. Key files

| Area | Key files |
|---|---|
| Workflow | `.github/workflows/test.yml`: `unit-tests` (mypy public surface → fast suite under coverage inside `scripts/ci_netns.sh` → `test_memory_hub.py` → `scripts/lint_coverage.py` → `scripts/lint_release_gates.py --junit fast-suite.xml`); `lint` (ruff, the CI greps, every `scripts/lint_*.py`); `release-build`; nightly `slow-tests`, `model-cache-tests` (shared `.github/actions/model-cache-setup`, the only cache in the workflow: HuggingFace models), `repo-settings`, `secret-scan`; `release-audit` on non-PR events |
| Workflow shape pins | `tests/unit/test_ci_workflow_shape.py`, `scripts/check_repo_settings.py --static` (required contexts map to unconditional jobs) |
| Pre-push | `scripts/suite_at_commit.sh` (§4), `scripts/pr_merge_readiness.py` (merge readiness) |
| Coverage ratchet | `scripts/lint_coverage.py` (floors, diff coverage, the pinned `[tool.coverage]` table; `--local`, `--merge-floors`), `scripts/coverage_floors.json`, `scripts/coverage_exclusions.json`; coverage + pytest-cov versions are pinned exactly in test.yml |
| Diff-scoped lints | judge a PR against its merge-base with `origin/main`, a push to `main` against `scripts/_lint_git.py::push_base` (last push whose job passed): `lint_coverage`, `lint_no_silent_swallows`, `lint_function_length`, `lint_mypy_ratchet`, `lint_atomic_io_ratchet`, `lint_fix_touches_tests` (needs `PR_TITLE`), `lint_unreleased_on_src_change`, `lint_unreleased_declared`, `lint_config_schema_append_only`, `lint_multi_agent_marker`, `lint_body_rest_neutral`, `lint_secrets`, `lint_ledger_format`, `lint_evidence_gate`, `lint_bf2_obligation`, `lint_release_gates` |
| Tree-wide lints | `lint_claude_md_invariants`, `lint_backlog_ids`, `lint_version_sync`, `lint_claims_sync`, `lint_orphan_modules`, `lint_harness_provenance`, `lint_prereg_precedes_data`, `lint_loop_mode_reads` |
| Test harness | `tests/conftest.py` (autouse env scrubs, `--require-extras`, the `requires_model_cache` skip), `tests/network_guard.py` (in-process network block; `allow_network` marker), `pyproject.toml` `[tool.pytest.ini_options]` (`testpaths`, `norecursedirs` excludes `.worktrees`, markers) |

## 3. Invariants & lessons

**[engineering] Confirm the expected checks are PRESENT, not just green** (core rule in CLAUDE.md). Read a failing check's own output before applying a remembered remedy. Regression guard: [scripts/pr_merge_readiness.py](../../scripts/pr_merge_readiness.py) + [tests/unit/test_pr_merge_readiness.py](../../tests/unit/test_pr_merge_readiness.py).

**[engineering] Touch it, shrink it: a change that edits lines inside a function pinned over 200 lines must leave it smaller by at least that many lines** (owner decisions 2026-10-16). It applies to every pinned function; the reason is to keep god functions shrinking in the moment, not only in decomposition slices. Move the code you touched out (a helper, a phase, a module), so the extraction carries the edit. An urgent fix that cannot carries a `touch_exemptions` record in `src/maxim/utils/function_length_baseline.json` whose `touched`/`from`/`to` match the lint's printed numbers exactly. Never shrink by deleting comments or docstrings (counted, but review refuses it). A repo-wide rename or a formatter upgrade may need one record per pinned function: that is expected, not a defect. Regression guard: [scripts/lint_function_length.py](../../scripts/lint_function_length.py)::touch_rule + [tests/unit/test_lint_function_length.py](../../tests/unit/test_lint_function_length.py) (the `test_t_*` tests, each proven by a deletion probe).

**[engineering] Coverage is measured once, in CI's fast-suite run, by the exact-pinned coverage version, and the local pre-check uses CI's own command.** Nothing may sit between the measuring suite and its gate except the MemoryHub step. Regression guard: [scripts/lint_coverage.py](../../scripts/lint_coverage.py)::ci_step_rules + [tests/unit/test_lint_coverage.py](../../tests/unit/test_lint_coverage.py)::test_the_suite_script_measures_with_cis_own_command.

**[engineering] Every CI test run is inside the loopback-only network namespace, and no install step swallows its failure.** The package installs WITH its core dependencies (#1188: a transitive-only dependency vanished and `main` went red at an unchanged commit). Regression guard: [tests/unit/test_ci_workflow_shape.py](../../tests/unit/test_ci_workflow_shape.py)::test_every_test_run_is_inside_the_loopback_only_network_namespace, ::test_no_install_step_swallows_its_failure, ::test_no_job_installs_the_package_without_its_core_dependencies; positive control [tests/unit/test_network_boundary.py](../../tests/unit/test_network_boundary.py).

**[engineering] A skip for a missing REQUIRED extra is a failure** (`--require-extras=console,sign`), so the console and signed-bundle tests cannot silently skip on every lane. Regression guard: [tests/conftest.py](../../tests/conftest.py)::_fail_required_extra_skip.

**[engineering] Each required status check maps to exactly one job with no job-level `if:`** (a skipped required check counts as passing). Regression guard: [scripts/check_repo_settings.py](../../scripts/check_repo_settings.py) `--static` + [tests/unit/test_check_repo_settings.py](../../tests/unit/test_check_repo_settings.py).

**[engineering] A push to `main` is judged by the diff-scoped lints against the last push whose job passed**, not `before` (a cancelled pending run would otherwise skip a range). Regression guard: [scripts/_lint_git.py](../../scripts/_lint_git.py)::push_base + [tests/unit/test_lint_git_push.py](../../tests/unit/test_lint_git_push.py).

## 4. Live gotchas / known gaps

- **A PR that fails `scripts/lint_coverage.py`:** raise the floor it names (`scripts/coverage_floors.json`), cover the changed lines it lists, or append a ref'd ledger entry (`scripts/coverage_exclusions.json`); the rules are in the lint's docstring. A rise above an exact pin after merging `main` is the bookkeeping class being redesigned (M60); do not hand-re-pin from a macOS number.
- **Per-file coverage floors (format v2, 2026-10-13):** `coverage_floors.json` also carries an opt-in `files` map (`"src/maxim/....py": {percent, missing}`) enforced beside the package floor, so a PR can have to edit a file pin as well as a package pin. Full rules (normalization, hysteresis, renames, the v1 → v2 change): section 1b of the `scripts/lint_coverage.py` module docstring.
- **Decomposing a function over 200 lines (`run_agentic_loop`, `start_simulation_mode`, `_main_impl`, …):** each slice edits `src/maxim/utils/function_length_baseline.json` in the same commit — the procedure, with a worked example, is the "Baseline edits per decomposition slice" paragraph of `scripts/lint_function_length.py`'s docstring. The swallow lint credits moves the same way: a `try` moved verbatim is free, and one edited on the way needs a `scripts/swallow_moves.json` record naming its source (`scripts/lint_no_silent_swallows.py`, move credit). For `run_agentic_loop` (1.3.2), the gates every slice must keep green unchanged, the golden regeneration rule and the function-specific pins are in `docs/plans/roadmap_1_3_x.md` §"The decomposition"; "the loop contains X" pins read `tests/unit/_loop_source.py`.
- **A test that needs a credential-shaped fake** (an API key, a `Bearer` header, a PEM header) trips `scripts/lint_secrets.py` on every path: build the fake at runtime from pieces (the `fake()` helper in `tests/unit/test_lint_secrets.py`), or land an entry in `scripts/secret_scan_allowlist.json` on `main` first; the rules are in the lint's docstring and in docs/agents/simulation-experiments.md.
- **New opt-in env vars in hot startup paths need autouse conftest scrubs in the same commit** — a test that sets the var leaks the side effect into every later runtime-constructing test (once cost a 9-minute hang on a real 1 GB model download). Template: the existing `_isolate_maxim_*` fixtures in [tests/conftest.py](../../tests/conftest.py). The ~54 autouse fixtures cost under a minute of suite time in total (2026-10-16). Full history: [docs/lessons/env-var-autouse-scrubs.md](../lessons/env-var-autouse-scrubs.md).
- **A tree-scanning lint is tested with its own new files tracked:** run it after `git add -N <new files>` (#1175 was green only because its test file was untracked).
- **Run `scripts/prepr.sh` on the last commit before pushing, and let it finish.** It runs every `jobs.lint` step plus the targeted mypy step, read from test.yml, and scans the PR title/body as CI does. Exit codes: 0 PASS · 1 FAIL · 2 REFUSED (it cannot reproduce CI's verdict: behind or stale origin/main, dirty tree, pin or python mismatch, PR-text mismatch, an unreproducible workflow construct) · 3 TAINTED-PASS (passed under `--allow-*`, `--no-fetch`, `--no-pr` or typed PR text). Only 0 clears a push; a `NOTE:` platform line does not change the code. Before a PR exists, pass `--title` (taints) or `--no-pr`. `--list` prints the plan (M59).
- **Stale timing comments in test.yml:** the concurrency block's "~12 min of real work" and the `unit-tests` timeout comment's 18:26/26:46 figures predate the 15k-test suite; the 2026-10-16 numbers are in §1. The `timeout-minutes: 60` has ~20 min of headroom over the median.
- **The mypy file list lives only in `scripts/run_mypy_public.sh`** (M65); CI and CLAUDE.md call it, and `test_ci_workflow_shape.py` fails if a second copy appears.

### `scripts/suite_at_commit.sh` (moved from docs/lessons/development-flow-speed.md)

```bash
scripts/suite_at_commit.sh                     # the suite + the coverage ratchet at HEAD, in a throwaway worktree
scripts/suite_at_commit.sh <commit>            # ... at any commit
scripts/suite_at_commit.sh --no-cov <commit>   # the suite only: quicker, NOT the pre-push check
scripts/suite_at_commit.sh "$(git stash create)"  # uncommitted tracked work (no ref moved, no signature)
```

It creates a uniquely named `.worktrees/suite-<sha>.XXXXXX` under the MAIN checkout (so concurrent runs on one commit never share or delete each other's tree), prunes stale worktree records first, runs the fast suite and `test_memory_hub.py` there with an absolute `PYTHONPATH` (so the installed package cannot shadow it), prints the commit next to the result, and removes the worktree. No `-x`, deliberately: the full failure list is worth more than the first. It tests a commit by design: untracked files are exactly what it keeps out of the run. The result names the commit it tested; that is not evidence of which diff was REVIEWED. To clean up a stopped run, kill only the pytest whose cwd is your own `suite-<sha>` tree.

**Since M47 (2026-10-11) it is also the coverage pre-check.** By default the fast suite runs with CI's exact measuring command and is followed by `scripts/lint_coverage.py --local`, so `PASSED` also means the coverage ratchet passed. `--local` gates diff coverage (>= 80% of changed `src/maxim` lines) and the floors of the TOUCHED packages (changed `src` files, deletions and moves included, plus the packages the changed tests import); the overall floor, untouched packages and a PR's own floor-pin edits are printed as notes, because floors are measured in CI's environment. A touched floor breached within noise needs CI's number. The script checks the installed coverage version before the suite runs. #1201, #1213, #1232 and #1240 each failed CI on the ratchet because no local run measured coverage.

## 5. Env vars owned

| Var | Purpose | Owner |
|---|---|---|
| `MAXIM_RUN_MODEL_TESTS` | Opt in to `requires_model_cache` tests (skipped by default; set by the nightly model-cache and slow lanes) | `tests/conftest.py` |
| `MAXIM_EXPECT_NETNS` | Set by `scripts/ci_netns.sh`; arms the network-namespace positive control | `tests/unit/test_network_boundary.py` |

## 6. Lesson archive

- [docs/lessons/ci-efficiency-2026-10-16.md](../lessons/ci-efficiency-2026-10-16.md) — the 2026-10-16 measurements, owner decisions and backlog rows M59–M65 (M59/M65 shipped with prepr.sh; M66 tracks running it)
- [docs/lessons/green-pr-with-no-tests-run.md](../lessons/green-pr-with-no-tests-run.md) (core rule in CLAUDE.md)
- [docs/lessons/env-var-autouse-scrubs.md](../lessons/env-var-autouse-scrubs.md) (core rule in CLAUDE.md)
- [docs/lessons/development-flow-speed.md](../lessons/development-flow-speed.md) — snapshot suites, fold batches, the M22 xdist trial

Cross-refs: docs/agents/runtime-tools.md (optional-deps surface, decomposition targets); docs/agents/simulation-experiments.md (secret-scan rules for campaign data); docs/CODE_REVIEW.md (the pre-merge review round).
