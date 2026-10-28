# pr_overlap.py: Wire integrity review (2026-10-27)

Scope: the staged diff (scripts/pr_overlap.py, tests/unit/test_pr_overlap.py, ci-testing.md §4 bullet, outstanding.md M83).
Narrow test: `tests/unit/test_pr_overlap.py` 8 passed. `scripts/lint_claude_md_invariants.py`: PASS.

## 1. `refs/pr-overlap/<n>` in the shared .git

Consumers checked: every `for-each-ref`/`refs/` user in `scripts/` and `.github/workflows/` (`o19_verdict.py` reads only
`refs/tags/o19/*`; `check_nightlies.py`/`check_release_tag.py`/`vendor_console_ui.py`/`prepr.py` read tags, `refs/heads/main`
or remote refs only); `suite_at_commit.sh` runs `git worktree prune`, which touches worktree admin dirs, not refs; no lint scans refs.
No gc/fetch config beyond the default `remote.origin.fetch`. **Holds:** nothing lists or prunes this namespace, and the explicit
refspecs do not touch `refs/remotes/origin/*`.

- **SHOULD-FIX: refs accumulate forever and pin objects.** Three already exist (`refs/pr-overlap/1465`, `1484`, `1507`). Each closed
  PR's ref stays, keeping its commits reachable from gc in the shared object store. Fix: after `pairs()`, delete them
  (`git update-ref -d` per ref, in a `finally`), or at least delete any `refs/pr-overlap/*` whose number is not in the current open list.
- **NIT: concurrent runs.** Two sessions running it at once fetch into the same ref names; a ref-lock collision makes `git fetch`
  fail, so the script exits 2. That fails loudly, but a per-run namespace (`refs/pr-overlap/<pid>/<n>`) plus deletion removes the collision.
- **NIT: head race.** `merge-tree` is given `headRefOid` from `gh pr list`, but the fetch pulls whatever `pull/N/head` is now. A push
  between the two leaves the listed SHA possibly absent locally, and the script exits 2. Fix: merge-tree the fetched ref
  (`refs/pr-overlap/<n>`), or warn when `rev-parse` of the ref differs from `headRefOid`.

## 2. `gh pr list --json` fields vs pr_merge_readiness.py

`mergeStateStatus` is read the same way (`pr_merge_readiness.py` reads it from `gh pr view`). `files` is not used there (the 100-file cap is
documented in the docstring). **Divergence on checks:** readiness uses the check-runs API and `_latest_run_per_check`, which collapses
duplicate runs by `started_at` because superseded runs are not failures. `_unit_tests` returns the FIRST rollup entry named
`unit-tests`. Re-runs and workflow re-triggers (`edited` retarget, re-run of failed jobs) can leave several entries.
- **SHOULD-FIX:** choose the `unit-tests` entry with the greatest `startedAt` (the rollup's CheckRun carries it; treat a missing value
  as newest, matching readiness), and add a test with two entries (old FAILURE, newer SUCCESS). Without this the order can rank
  a green PR as red, which is the same class of error readiness already fixed. It is advisory only, so it is not a BLOCKER.

## 3. Hot-file constants

All three paths exist. The last 40 first-parent merges on origin/main touched: CHANGELOG.md 24, coverage_floors.json 15,
function_length_baseline.json 14, **docs/plans/roadmap_1_3_x.md 13**, **docs/plans/outstanding.md 9**, cli.py 6; everything else 6 or fewer.
- **NIT:** roadmap_1_3_x.md and outstanding.md are as hot as the pinned files. They are line-specific docs, and real conflicts in them
  are already caught by `merge-tree`, so `HOT_FILES` only changes the ordering key and the label. Optionally add outstanding.md
  (it is long-lived), or say in the docstring that the list holds files every PR bookkeeps, not files that happen to be busy.

## 4. `--pins-from-run` advice

`--pins-from-run` exists only on the LOCAL branch `ci/coverage-ratchet-one-way` (13 hits in its `lint_coverage.py`). The branch is not
pushed and has no open PR. On origin/main the string appears nowhere except this diff.
- **SHOULD-FIX:** do not print a command that does not exist on `main`. Pick one: (a) probe at runtime
  (`"--pins-from-run" in (REPO_ROOT/"scripts/lint_coverage.py").read_text()`) and fall back to "re-pin per `scripts/lint_coverage.py`'s
  docstring (M60); do not hand-re-pin from a macOS number", which matches the existing ci-testing.md §4 bullet; or (b) print the generic
  text now, and have the M60a PR switch it to `--pins-from-run` and update `test_pr_overlap.py:82-83`. (b) is simpler and puts the
  wording change in the PR that adds the flag. Either way, the test must stop asserting the flag before M60a lands.

## 5. Docs

The ci-testing.md §4 bullet links `../plans/outstanding.md` (resolves); the M83 row is in the Mechanization backlog table, in sequence after M82;
the lint passes. The script is cited correctly as `scripts/pr_overlap.py`.
- **NIT:** the bullet says the follow-ups are "in outstanding.md". Name the row ("row M83") so a reader can find it.

## Verdict

No BLOCKER. Three SHOULD-FIX: ref cleanup, latest-run `unit-tests` selection, and the `--pins-from-run` text. Four NITs.
