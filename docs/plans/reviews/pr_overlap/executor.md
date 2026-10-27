# pr_overlap.py — Executor lens review (2026-10-27)

Diff reviewed: staged `scripts/pr_overlap.py`, `tests/unit/test_pr_overlap.py`, ci-testing.md §4 bullet, outstanding.md M83.
Ran for real (text + `--json`, both exit 0; 3 open PRs: #1507, #1484 draft, #1465 draft), spot-checked with
`git merge-tree` / `git commit-tree` (objects only), and ran deletion probes in a scratch copy.

## Real-run spot check

Tool output: every pair "CONFLICTS in src/maxim/utils/function_length_baseline.json", and "if #1507 lands first,
#1484/#1465 need: resolve function_length_baseline.json".

Ground truth:
- `merge-tree origin/main <head>`: #1507 clean (0 behind); **#1484 and #1465 each already conflict with main in
  function_length_baseline.json** (76 behind, both DIRTY).
- Simulated "#1507 lands first" (`commit-tree` of merge(main, #1507), then `merge-tree` with each other head): #1484
  and #1465 conflict in the baseline, which is the SAME conflict they already have with main. #1507's landing adds
  nothing to what they owe.

So the conflict lines are right that the file conflicts, but they blame the wrong PR, and they will keep doing so
whenever a PR is behind.

## Findings

### SHOULD-FIX 1: the pairwise heads predictor confuses "conflicts with main" with "conflicts with the other PR"
`conflicting_files(a.head, b.head)` merges over merge-base(a, b), an old main. Each head's diff against that base
includes all of main's commits that the head has merged in, so (i) a PR's own conflict with main is reported as
a conflict with whichever PR shares the file (seen today, above), and (ii) the "owed if X lands first" text lists
work that is owed regardless. The opposite case, a false "clean", is mostly semantic rather than textual (two
non-adjacent baseline/CHANGELOG entries that each re-pin the same function). I could not build a textual false
clean that does not also show the PR as DIRTY against main, so the practical damage is misattribution, not
missed conflicts.
**Fix:** fetch `origin/main` too. Per PR, compute `vs_main = conflicting_files(origin/main, head)` and print it as
"conflicts with main now: ..." (merge_state DIRTY says only that something conflicts). Per ordered pair (a first),
build `C_a = commit-tree(merge-tree(main, a), -p main -p a)` when a merges cleanly with main, then
`conflicting_files(C_a, b)` minus `vs_main[b]` = what a's landing adds. That is the simulation the "owed" line
claims to describe, and it is asymmetric, which the current output hides. It writes only objects, so the script
stays read-only. Add a fixture test where main moves and B is behind: the test should fail on today's code.

### SHOULD-FIX 2: silent file-list truncation at 100
Confirmed: `gh pr view 772 --json files,changedFiles` gives 100 files against changedFiles 194 (#394 is also over
100). The truncation is only a docstring residual, and the output does not say it happened, so a missed overlap
looks like "no overlap". **Fix:** since the heads are fetched anyway, take the file set locally with
`git diff --name-only $(git merge-base origin/main <head>) <head>` (no cap, and it matches what the PR changes
against today's main). At minimum, request `changedFiles` and print a WARNING line on any PR where
`len(files) < changedFiles`.

### SHOULD-FIX 3: deletion probes, survivors (tests stay green when the code is broken)
| Mutation | Result |
|---|---|
| `n_conf = 0` (order ignores conflicts) | 8 passed |
| `n_hot = 0` | 8 passed |
| `bool(mine)` → `False` | 8 passed (n_hot masks it in the order test) |
| `--json` `"order": []` | 8 passed (`--json` path untested) |
| `fetch_heads` no-op | 8 passed |
| `returncode not in (0,1)` removed | 8 passed (the tree-shape check covers it; fine) |
| `status != COMPLETED` branch removed | 8 passed (redundant: in-progress runs have `conclusion: ""`, see NIT 1) |
| tree-shape check, base filter, draft key, hot report | caught |

**Fix:** an order test where two non-isolated PRs differ only by conflict count (and one only by hot overlap); a
`main(["--json"])` test with `list_prs`/`fetch_heads`/`pairs` monkeypatched, asserting the keys and order; a
`fetch_heads` test asserting the refspecs (monkeypatch `_run`).

### NIT 1: `_unit_tests` against the real rollup
I checked against the real shape (#1507: `CheckRun unit-tests IN_PROGRESS ""`; #1465: `COMPLETED FAILURE`): the
parsing is correct. Residuals: (a) a `StatusContext` carries `context`/`state`, not `name`/`status`/`conclusion`,
so a status-API "unit-tests" would read "absent" (no workflow posts one today; match on
`c.get("name") or c.get("context")` and fall back to `state`). (b) If the rollup ever lists two `unit-tests` runs, the
first one wins. (c) The `status` branch is redundant with `conclusion or "pending"`.

### NIT 2: order key on today's data
Today's result, `#1507 → #1484 → #1465`, is sensible, but only because drafts sort last. Two weak points:
`merge_state` is unused, so a DIRTY or BEHIND-by-76 PR ranks the same as a clean, up-to-date one (suggest sorting
DIRTY after the others, or "conflicts with main" from SHOULD-FIX 1); and every `src/` PR touches CHANGELOG.md
(the unreleased lint), so `bool(mine)` and `n_hot` barely discriminate. "unit-tests success" on a head 76 commits
behind is stale evidence, yet it outranks a fresh "pending".

### NIT 3: robustness
- Race: `headRefOid` comes from `gh pr list`, but the merge uses the fetched ref. A push in between leaves the OID
  unfetched, `merge-tree` fails, and the run exits 2. Use `refs/pr-overlap/<n>`, or compare the fetched ref's OID and
  warn.
- `refs/pr-overlap/*` is never pruned (closed PRs accumulate). Delete the stale ones at the start.
- `--limit 100` on open PRs is silent too. Fine today; warn if `len == 100`.
- `merge-tree --write-tree` needs git 2.38 or later. An older git exits 2 loudly, which is acceptable.

## Verdict
Not a BLOCKER: the tool is advisory, read-only (only `refs/pr-overlap/*` plus loose objects), and fails closed
(exit 2) on tool errors. SHOULD-FIX 1 matters for its main claim ("what each PR owes if the other lands first"):
on today's real data that line is wrong for both draft PRs.
