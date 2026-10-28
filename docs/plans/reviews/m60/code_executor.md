# M60a code review: Executor lens

Diff: `git diff origin/main...HEAD` at 587a25e7 (branch `ci/coverage-ratchet-one-way`). Reviewer: Executor lens
(docs/CODE_REVIEW.md), read-only, 2026-10-27. Nothing in the worktree was modified except this file. Probes ran in a
scratch copy (`git archive HEAD`).

## Verdict

**No BLOCKER.** The code does what it claims on real inputs. I ran the whole pipeline offline against real GitHub
payloads and a real CI artifact, and it worked. Every deletion probe I tried turned a test red. There are two
SHOULD-FIX items and three NITs.

## What was run

- `tests/unit/test_lint_coverage.py`: **246 passed** (99.8 s).
- Real API shapes, read-only `gh api`, run 40549504653 (Tests, `pull_request`, attempt 1, success):
  - `actions/runs/{id}` has `path` = `.github/workflows/test.yml`, `event` = `pull_request`, `run_attempt` = 1 (an
    int), and `head_sha` = the PR **head** commit (`8cb08d04…`), not the merge ref. That is what the HEAD binding
    needs.
  - `runs/{id}/attempts/1/jobs?per_page=100` has 18 jobs, and one is named exactly `unit-tests`. Its steps include
    `Run required fast suite`, whose `conclusion` is `success`. The conclusion values are the documented strings
    (`success`/`failure`/`cancelled`). In run 40350416806, attempt 1 is `failure` and attempt 2 is `cancelled`, and
    the step check refuses both.
  - Re-run 40022134226 (attempt 2): the attempts-2 jobs endpoint returns `unit-tests` with `run_attempt: 2`. The
    artifacts endpoint lists the run's artifacts across all attempts. Under the old fixed name it held only ONE
    `coverage-json`, which confirms why the name must be bound to the attempt.
  - `gh api` expands `{owner}/{repo}` from the **cwd's** git remote. It works inside the repo. Outside it fails with
    "unable to expand placeholder" (see SHOULD-FIX 2).
- `gh run download 40549504653 -n coverage-json -D <scratch>/dl` puts `coverage.json` **directly** in the directory,
  so `dest / COVERAGE_JSON` is correct.
- End to end, offline: I called `run_measurement` with stubs that return the three real API payloads (with
  `head_sha` set to the local HEAD) and the real downloaded artifact, then ran `measure_scopes`,
  `pins_from_measurement` and `apply_pins` against the committed floors file. It loaded 537 files. It returned no pins
  (no scope is past its band) and 13 notes. `apply_pins(text, {})` and same-value pins (one package key and the file
  key `src/maxim/simulation/orchestrator.py`) leave the file **byte-identical**, and the output parses again.
- Deletion probes, each reverted in a scratch copy. Every one turned at least one test red:
  - `--local` gating the touched packages again;
  - CHANGED cap without `+ K_MIN`, and with `+ missing_slack(m)` instead;
  - no re-rooting;
  - a foreign file required to exist;
  - the legacy name preferred over the attempt-bound name;
  - the legacy name accepted on any attempt;
  - no suite-step check;
  - no HEAD check;
  - no workflow/event check;
  - jobs read from attempt 1;
  - `apply_pins` routing every key to `packages`;
  - the pinned missing taken from the measurement instead of `min(m, f)`;
  - no write in `pins_from_run_main`.

## Findings

### SHOULD-FIX 1: the test fixture's key shape is one CI never produces; the re-rooting branch is stub-only

The real artifact's keys are **already relative** (`src/maxim/__init__.py`; 537 of 537 keys, coverage 7.13.3, run
from the checkout root with `source = ["src/maxim"]`). `_ci_artifact` rewrites them to
`/home/runner/work/Maxim/Maxim/src/maxim/...`. So:

- the line `rel = rel[rel.index("/src/maxim/") + 1:]` is exercised only by the stub;
- the real shape, relative keys through the `foreign=True` path, has no test (my offline run is the only evidence it
  works);
- the re-rooting also runs on a relative key. A relative key `src/maxim/a/src/maxim/b.py` would silently become
  `src/maxim/b.py`. An absolute runner path whose checkout directory contains `/src/maxim/` would re-root at the wrong
  occurrence, and the result still passes `startswith(SCOPE + "/")`. That is hypothetical today, but it is a silent
  mis-attribution, not a refusal.

Fix:
- Re-root only when `p.is_absolute()`. Then a key containing `/src/maxim/` twice should either use the occurrence
  that leaves an existing-in-SCOPE suffix or refuse.
- Make `_ci_artifact` emit the real relative keys by default, and keep the absolute shape as a second parametrized
  case.
- Fix the docstring/comment example (`/home/runner/work/...`), which describes a shape CI does not write.

### SHOULD-FIX 2: `gh` resolves the repo from the process cwd, but HEAD is read from `REPO_ROOT`

`pins_from_run_main(REPO_ROOT, ...)` binds HEAD through `git(root, ...)`. Both `gh_api` and `_download_artifact`,
however, run `gh` with no `cwd`, so `{owner}/{repo}` and `gh run download`'s repo come from wherever the operator
stands. Outside a git directory this fails loudly (rc 2, "unable to expand placeholder"), which is safe. Run from
inside a different git checkout, `gh` would query that checkout's GitHub repo. The HEAD check would almost certainly
refuse, but the two inputs are bound to different repositories.

Fix: pass `cwd=root` to both subprocess calls. For `gh_api` in `_lint_git`, add an optional `cwd`, or pass
`-R owner/repo` derived from `root`'s remote. Add a test that asserts `cwd`.

### NIT 1: the "more than K_MIN above" notes are noisy and phrased conditionally

On the real run, 13 of 13 notes are "its pinned missing N is more than 5 above the run's M; if this branch changed
that pin, CI refuses it". `overall` alone is 900 statements of slack. These scopes were not changed by the branch, so
the note fires on every scope whose missing has fallen since its last pin. That is most of them.

Fix: emit the note only for scopes whose pin differs from `origin/main`'s (the same base `floor_diff_rules` uses), or
collapse them into one summary line.

### NIT 2: the refusal for re-run-failed-jobs and old re-runs could say what to do

A run re-run before this change (for example 40022134226, attempt 2, legacy name only) is refused with "no unexpired
artifact 'coverage-json-2'". That is correct and fails closed, but it is opaque. Append "re-run the whole workflow, or
push a new commit".

### NIT 3: the main-path test patches `run_measurement.__kwdefaults__`

`test_pins_from_run_main_writes_the_file_and_ci_accepts_it` works because the defaults are bound at definition time.
The extra `monkeypatch.setattr(L, "gh_api", ...)` and `_download_artifact` lines are dead. The test is fine, but it
would be clearer if `pins_from_run_main` took `api=`/`download=` keywords and passed them through.

## Checked and fine

- `apply_pins` sends a key under `src/maxim/` to `files`, `overall` to the root, and everything else to `packages`. A
  new key would be appended at the end of the section. That cannot happen here, because only floored scopes are
  pinned.
- The committed floors file round-trips byte-for-byte, and a test guards that.
- The HEAD floor check never gates `missing`, so a pin of `min(m, f)` with `f < m` is consistent with CI's own
  "raise it" hint.
- `--local` with `only=set()` still runs the format/version/band checks, and the touched-package probe shows the
  tests pin it.
- `test.yml`: nothing else consumes the artifact name. `ci_step_rules` allows the upload step after the gate, as
  before.
