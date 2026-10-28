# pr_overlap: delta review (Executor + Architecture + Wire integrity), 2026-10-27

Scope: staged `scripts/pr_overlap.py`, `tests/unit/test_pr_overlap.py`, the ci-testing.md §4 bullet and outstanding.md row M83, which fold executor.md / architecture.md / wire.md. Read-only. Narrow suite: 17 passed.

## 1. The folds

| Finding | Status |
|---|---|
| Main conflicts reported per PR and never blamed on another (simulated landing via merge-tree + commit-tree, minus the PR's own main conflicts) | Fixed: `analyse` + `land`; pinned by `test_a_conflict_with_main_is_the_prs_own_and_never_blamed_on_another_pr` (the author's probe, and my P2 below, both turn it red) |
| Files come from git, not gh's 100-file cap | Fixed: `changed_files`, merge-base diff; 150-file test |
| Newest unit-tests run; StatusContext read | Fixed: `_unit_tests`; both orders tested, queued counts as newest, StatusContext case |
| Per-run ref namespace, dropped in `finally` | Fixed, except one path: see S2 |
| Advice no longer names `--pins-from-run` | Fixed; asserted absent |
| Order key covers main conflicts, DIRTY, red CI | In the code. The DIRTY term is untested: see S1 |
| Each order clause tested | All except DIRTY (S1) |
| M83 says attention-only | Yes, and so does the §4 bullet |

## 2. New questions

- **`land()`'s commit-tree:** it writes one loose commit object and no ref (scratch repo: `for-each-ref` empty afterwards). GPG: this user's config has `commit.gpgsign=true` and a signing key. In a scratch repo with `commit.gpgsign=true` and `gpg.program=/usr/bin/false`, git 2.50.1 `commit-tree` succeeded both with and without the `-c` override, so commit-tree does not read `commit.gpgSign` here. The `-c commit.gpgsign=false` costs nothing and covers older git. Safe.
- **`fetch_heads` overwriting `refs/remotes/origin/main`:** this is harmless. It is the same forced update the default refspec `+refs/heads/*:refs/remotes/origin/*` makes on every `git fetch`. It moves the "behind N" counts other worktrees show, as any fetch does. It touches no branch, index or HEAD. The only interaction is ref-lock contention with another session's concurrent fetch: one of the two fails with "cannot lock ref". For this script that means exit 2. That path is also where S2 lives.
- **Live run** (`python scripts/pr_overlap.py`, 2.8 s, exit 0): 3 PRs, 3 pairs. #1484 and #1465 are both DIRTY and both conflict with main in `function_length_baseline.json`. Suggested order: #1507 → #1465 → #1484. **Spot-check by hand:** I used `merge-tree` on main + #1507 (clean), then `commit-tree`, then `merge-tree` of that commit with #1465. The only conflict is `src/maxim/utils/function_length_baseline.json`, the same file #1465 already conflicts with on main. So the reported "if #1507 lands first, #1465 needs: merge main" with no new resolves is correct. The order also matches the key: #1507 is the only unblocked PR, and the two drafts tie, so the older one goes first.
- **After the run**, `git for-each-ref refs/pr-overlap/` is empty (0 refs).

## 3. Probes (scratch copy, never the worktree file)

- P1: deleted `or p.merge_state == "DIRTY"` from `merge_order` and **all 17 tests still pass**. Recorded as S1.
- P2: made `land()` always simulate, even when the PR conflicts with main. 1 test fails. Guarded.
- P3: replaced `permutations` with `combinations` in `analyse`, so only one landing direction is simulated. 2 tests fail. Guarded.

## Findings

- **S1, SHOULD-FIX, fail-closed.** The DIRTY term of the order key has no test (P1 survives). The fold claimed that every order clause is tested. In practice the term is mostly redundant with `main_conflicts`, and the per-PR line still prints DIRTY, so the gap misorders at worst and never gives a false all-clear. Fix: add a `blocked_after_ready` variant with `state="DIRTY"` and no main conflicts.
- **S2, SHOULD-FIX, fail-closed.** Refs can leak when `fetch_heads` fails part-way. The namespace is created inside `fetch_heads` and handed back only on success. If `git fetch` returns non-zero after writing some refs, or a later `rev-parse` raises, `main`'s `ns` is still `None`, and `finally` drops nothing. Partial writes are possible, for example when it loses the `origin/main` lock to another session's concurrent fetch. The leaked refs then sit under `refs/pr-overlap/<pid>/`. Fix: compute `ns` in `main()` before the call and pass it in. Add a test where a monkeypatched `fetch_heads` raises after the refs are named.
- **N1, NIT, fail-open (minor).** The blame subtraction compares file names. A landing that adds a NEW hunk conflict inside a file the PR already conflicts in on main is not reported. The PR must resolve that file anyway, and its own line says so.
- **N2, NIT.** Two docs say nothing else is written: the ci-testing.md bullet ("writes only loose objects and per-run refs") and the docstring ("Nothing else is written"). Both omit that the run also updates `refs/remotes/origin/main` (and FETCH_HEAD). It is harmless, but the docs should state it.
- **N3, NIT, fail-closed.** In `_unit_tests`, a StatusContext has no `startedAt`, so it always counts as newest when mixed with CheckRuns of the same name. `"absent"`, `"skipped"` and `"neutral"` count as blocked. Both are conservative.
- **N4, NIT.** `analyse` runs `merge_tree(base, head)` twice per PR: once for `main_conflicts` and again inside `land()`. This costs only speed.

No BLOCKER, and no fail-open SHOULD-FIX.

**STOP: no further round owed.** S1 and S2 are fail-closed; fold them or file them as a follow-up.
