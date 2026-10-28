# M60a code review: delta round (Executor, Architecture, Wire)

Scope: exactly the fold commit 4afcedd0 (`git show 4afcedd0`), answering `code_executor.md`, `code_architecture.md`
and `code_wire.md`. Read-only review, 2026-10-27. Probes ran in a scratch copy (`git archive 4afcedd0`).

## Verdict

**No BLOCKER. No fail-open SHOULD-FIX.** One fail-closed SHOULD-FIX (an attribution in the docstring) and three NITs.

**STOP: no further round owed.**

## What was run

- In the scratch copy, `tests/unit/test_lint_coverage.py` + `tests/unit/test_lint_git_push.py` +
  `tests/unit/test_ci_workflow_shape.py`: 310 passed, 1 failed. The failure is
  `test_no_other_file_carries_the_mypy_file_list` ("exemptions that match no tracked file"), an artifact of the
  scratch copy having no tracked files. It passes in the worktree.
- `tests/unit/test_check_release_tag.py` passed. That file has its own `gh_api` and is untouched.
- Probes in the scratch copy. Each was reverted, and the copy was restored from the archive.

| Probe | Result |
|---|---|
| Rename the step `Run required fast suite` in test.yml | `test_the_pinner_names_match_the_real_workflow` fails |
| Rename the upload to `coverage-json` (drop `-${{ github.run_attempt }}`) | the same test fails |
| Add `name: Unit tests` to the `unit-tests` job | the same test fails |
| Drop `p.is_absolute() and` from the re-rooting condition | `test_pins_from_run_reroots_only_an_absolute_runner_key` fails |
| Remove the `GITHUB_ACTIONS` refusal | `test_pins_from_run_is_refused_inside_github_actions` fails |
| Remove both `cwd=` (`gh_api(cwd=root)` lambda → bare `gh_api`; `cwd=REPO_ROOT` on download) | **all `-k pins` tests still pass** (see NIT 1) |

## 1. Each finding against the fold

| Finding | Fixed? |
|---|---|
| E-SF1: re-root only absolute keys; fixture emits CI's relative keys; parametrize the absolute case | **Yes.** `p.is_absolute()` guards the re-rooting. `_ci_artifact` is relative by default and takes `absolute=True`. The new test covers both shapes, plus a relative key with `/src/maxim/` twice that is left intact. The comment example still shows the absolute path, but it now labels the absolute branch, which is correct. |
| E-SF2: `gh` cwd bound to the checkout | **Yes in code.** The default `api` is `lambda path: gh_api(path, cwd=root)`, and `_download_artifact` runs with `cwd=REPO_ROOT`. No test asserts either `cwd` (NIT 1). |
| A1: docstring paragraph for `--pins-from-run` | **Yes.** Its refusals, the latest attempt, the legacy name only on attempt 1, foreign files accepted, what it reports versus writes, the `gh` dependency, exit 2, and the Actions refusal all match the code. The "Stdlib plus `pyyaml`" footer was not amended, but the new paragraph names `gh` (NIT 3). |
| A2: the k-tolerance paragraph contradicted the CHANGED cap | **Yes.** It now has a parenthetical naming the CHANGED cap's separate `K_MIN` allowance. |
| A3: the `K_MIN`×N residual and its attribution | **Residual: yes.** There is a new Residuals bullet, and the approach note's item 4 is corrected. **Attribution: no** (SHOULD-FIX 1). The approach note's §2 "What it costs" still says "a 5-statement escape in total" (NIT 2). |
| A4 / A5 / W2: stale prose in the M47 row and the `suite_at_commit.sh` header | **Yes.** Both now say no floor gates locally (M60a). The M47 "Residue" still frames the ungated overall floor as the residue, when no floor is gated locally at all. That is a docs-only nit and folds into NIT 2. |
| A-N3: refuse inside Actions | **Yes.** It returns 2 before anything else, and a test covers it. |
| A-N5: hoist `import tempfile` | **Yes.** |
| W1: upload step names its consumer | **Yes.** The step name and comment name `--pins-from-run`. |
| W6 / A-N2: pin the job, step and artifact names to the real test.yml | **Yes.** There is a test, not a `ci_step_rules` extension. The probes above show it fails on each of the three renames. Being a unit test, it fails the PR that renames anything, because the fast suite runs on every PR. |
| W2-NIT: ci-testing.md inventory | **Yes.** |
| W3-NIT: the hint names the tool | **Yes.** |

Findings left unfolded, all previously NIT: E-NIT1 (noisy notes), E-NIT2 / W "re-run the whole workflow" refusal text,
E-NIT3 (`__kwdefaults__`), A-N1 (`_download_artifact` in `_lint_git`), W5 (an `apply_pins` assert). These are acceptable
as NITs.

## 2. Regressions from the fold

- **`gh_api(path, *, cwd=None)`.** The new parameter is keyword-only with a default, so every existing positional
  call `gh_api(path)` holds. That covers `_lint_git.push_base` (`api = api or gh_api`) and the second `_lint_git`
  site. `tests/unit/test_lint_git_push.py` passed 38 of 38. The test helpers that monkeypatch `gh_api`
  (`_push_event_helpers.py`, `test_lint_git_push.py`) substitute one-argument fakes. `push_base` calls them with
  one argument, so they still fit. `check_release_tag.py` and `check_repo_settings.py` define their own `gh_api`, so
  they are unaffected.
  - One latent edge: the `run_measurement` default lambda calls the module global `L.gh_api` with `cwd=`. A
    one-argument fake patched over `L.gh_api` would now raise `TypeError`. The only such patch,
    `test_pins_from_run_main_writes_the_file_and_ci_accepts_it`, also replaces `__kwdefaults__`, so the lambda never
    runs and that patch is dead (E-NIT3, not a regression).
- **The hint text between "floor X%" and "— raise it to".** Machine consumers, grepped over `scripts/`, `docs/`,
  `.github/` and `tests/`: the only parser is `tests/unit/test_lint_coverage.py`'s regex
  `^\s*(?:- )?(\S+): .* raise it to (\{.*\})$`. Its `.*` absorbs the new text, and the test passes. Nothing else
  parses the line: `pr_merge_readiness.py`, the workflows and `suite_at_commit.sh` read only exit status. The doc
  hits are prose ("raise it to …").
- **The workflow-name test** passes on the real test.yml and fails on each of the three renames.
- **Is the docstring accurate against the code?**
  - "Files this branch lacks are accepted": yes, and only for a foreign file. The existence check is
    `if not foreign and not (root / rel).is_file()`, so the local `coverage.json` path still refuses a missing
    file.
  - "Re-rooted only if absolute": yes.
  - "Falls, null floors, unmeasured scopes and branch pins more than `K_MIN` above the run are reported" is accurate
    in substance. However, the note fires for ANY pin more than `K_MIN` above the run, not only pins this branch
    changed (E-NIT1). "Branch pins" slightly overstates the filtering (NIT 2).
  - "It binds the run before reading anything" is true for run data. The floors file is read first, which is
    harmless.

## Findings

### SHOULD-FIX 1 (fail-closed): the residual quotes an owner decision that was never recorded

`lint_coverage.py`'s Residuals bullet attributes the slack to "the owner decision of 2026-10-27, 'ship the safe half
with a noise allowance'". The recorded owner words (approach note, lines 7–9) are "Do M60" and "ship the safe half
now and make the 'compare to main' fall check its own item". Nothing in the repo says "noise allowance". The
CHANGED-rule bullet in section 2 also still says "(M60a, owner decision 2026-10-27)" for the slack. A3 asked for an
explicit owner confirmation, or an attribution to the design rounds.

Fix: either get the owner's explicit confirmation of the slack and record it, or reword both sites, for example:
"from the M60a design rounds (delta1 S3, sized at `K_MIN` by delta2 P1), under the owner's 2026-10-27
ship-the-safe-half decision". It is fail-closed because no gate's behaviour depends on the prose. It matters because
an invented owner quote in a rule book is the claim-attribution class the project's lessons guard against.

### NIT 1: no test pins the `cwd` of either `gh` call

Deletion probe: removing both `cwd=` leaves every test green. A test that stubs `subprocess.run` (or `L.gh_api`
taking `cwd`) and asserts `cwd == root` / `REPO_ROOT` would close it. It fails closed in practice: run from another
checkout, the HEAD binding refuses.

A related nit: `_download_artifact` binds to `REPO_ROOT`, while the API binds to `root`. They are identical through
`main()`, but taking `root` explicitly would make them one binding.

### NIT 2: leftover prose

- `docs/plans/coverage_ratchet_m60_approach.md` §2 "What it costs" still says "a 5-statement escape in total". That
  contradicts the corrected item 4 (`K_MIN` × N for the sums).
- `outstanding.md` M47's "Residue" names only the overall floor, when under M60a no floor gates locally, by design.
- The docstring's "branch pins more than `K_MIN` above the run" should say "pins".

### NIT 3: the footer "Stdlib plus `pyyaml`" omits that `--pins-from-run` shells out to `gh`

The A1 footer half was not folded. The new paragraph does state it.

## 3. Diff coverage of the scripts change (not gated; reported)

| New branch | Test? |
|---|---|
| `p.is_absolute()` re-root, both arms (absolute and relative) | Yes: `test_pins_from_run_reroots_only_an_absolute_runner_key` |
| `GITHUB_ACTIONS` refusal | Yes: `test_pins_from_run_is_refused_inside_github_actions` |
| `api or (lambda …: gh_api(path, cwd=root))` default lambda | **No.** Every test injects `api` (NIT 1). |
| `_download_artifact` `cwd=REPO_ROOT` | **No.** The real downloader is never called in tests (NIT 1). |
| `gh_api(cwd=)` passthrough | **No.** The `test_lint_git_push` real-`gh_api` tests call it without `cwd`. |
| The hint's new text | Yes: the two updated hint assertions and the regex test |
| Workflow names | Yes: `test_the_pinner_names_match_the_real_workflow`, proved by the three rename probes |

STOP: no further round owed.
