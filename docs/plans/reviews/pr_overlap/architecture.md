# Architecture review: scripts/pr_overlap.py (staged diff, 2026-10-27)

Lens: Architecture (docs/CODE_REVIEW.md). Read: the staged diff, `scripts/pr_merge_readiness.py`, `scripts/_lint_git.py`,
`scripts/lint_orphan_modules.py`, `scripts/lint_backlog_ids.py`, `scripts/lint_coverage.py::main`, the ci-testing.md §4
bullet and the M83 row. Narrow test run: `tests/unit/test_pr_overlap.py` 8 passed.

Verdict: no BLOCKER. Two SHOULD-FIX items are correctness of what the tool tells a session to do; fold them before the PR.

## Placement: new script or a mode of pr_merge_readiness.py?

- For a mode: both read `gh pr` state, both parse the check rollup, and "prefer editing existing modules" is the house rule.
- Against: the readiness script is a per-PR verdict with a carefully argued exit contract (0 ready / 1 blocked /
  2 unsettled / 3 tool error) and two design rules about negatives over unsettled snapshots. This tool is cross-PR,
  advisory, always exits 0. A `--overlap` mode would need its own exit semantics inside that contract, takes no PR number,
  and would put `git fetch` + `merge-tree` in a script that today only makes `gh` reads. `scripts/` already holds many
  single-purpose tools; the "prefer editing" rule is mostly about `src/` bloat.
- Recommendation: keep it a separate script, but share the one piece of logic that is genuinely the same (the check-run
  selection, SF-1). Step 2 (`merge_train.py`) will need PR listing, rollup reading and the readiness verdict together; when it
  lands, extract a small `scripts/_gh_pr.py` (list/rollup/latest-run) used by all three, rather than a third copy.

## Findings

### SF-1 (SHOULD-FIX): `_unit_tests` takes the FIRST `unit-tests` entry in the rollup, the #1442 bug class
A head commit can carry several `unit-tests` runs (a description edit starts a second workflow run; `concurrency`
cancels the first). `pr_merge_readiness.py::_latest_run_per_check` exists precisely because the first/oldest run is
superseded, not failed. Here a superseded `cancelled` run can be reported and, through `merge_order`'s
`unit_tests != "success"` term, demote a green PR. Fix: pick the newest entry (rollup items carry `startedAt`; an unsettled one
with no start counts as newest), ideally by importing `_latest_run_per_check` (adapting key names) or moving it into a shared
helper; add a test with two `unit-tests` entries, older `CANCELLED`, newer `SUCCESS`.

### SF-2 (SHOULD-FIX): `owed_if_first` cites a flag that does not exist
`lint_coverage.py --pins-from-run` is not an option: `lint_coverage.py::main` knows `--merge-floors [--all-scopes]` only and
does not reject unknown arguments, so a session following the advice runs the plain lint and believes it re-pinned. It also
cuts against ci-testing.md §4's next bullet ("do not hand-re-pin from a macOS number"; M60). Fix: say "re-pin coverage
floors from its own CI run's printed pins (see ci-testing.md §4 / M60)", or name `--merge-floors` only if that is the
intended procedure; add a test asserting every `scripts/...py --flag` string in the advice is accepted by its script (or at
least that the string is not fabricated).

### SF-3 (SHOULD-FIX): "Enforced, or on the backlog": the new process rule is attention-only, and M83 does not say so
The ci-testing.md bullet adds a rule ("Before asking the owner to merge, run `pr_overlap.py` ... give that order in chat ...
message the sessions"). Nothing checks it. M83 describes the CI-invalidation problem and steps 2/3 but does not state that
the run-it rule is attention-only or what mechanizes it. Exit-0 advisory is the right contract for this tool (ordering is
the owner's call; a gate on "you ran an advisory script" would be theatre, cf. M6's closed-differently reasoning). Fix: one
sentence in M83's middle cell: "Running it before a merge request is attention-only until step 2, where the train computes
the order itself and the rule is retired"; and cite `M83` by ID in the ci-testing.md bullet (it links outstanding.md, so
`lint_backlog_ids.py` will check the citation).

### NIT-1: the order ignores mergeability
`merge_order` can put a DIRTY, BLOCKED or red-CI PR first because it overlaps nothing. Docstring and code agree, so this is a
design choice, but the "suggested order" is read as "merge this next". Fix: sort non-mergeable (`DIRTY`, failing
`unit-tests`, draft) after mergeable ones, or print them as "not mergeable yet" outside the order.

### NIT-2: "read-only" is slightly overstated
`fetch_heads` writes `refs/pr-overlap/<n>` (shared by every worktree of the repo) and `merge-tree --write-tree` writes loose
objects; refs for closed PRs are never pruned. The docstring says no branch/index/worktree is touched, which is true; the
ci-testing bullet says "read-only". Fix: say "writes only private refs under `refs/pr-overlap/`", and delete refs whose PR is
no longer open (`git update-ref -d`) at the start of a run.

### NIT-3: the 100-file truncation is silent at run time
The residual is documented, but a PR at exactly 100 files is not flagged in the output. Fix: mark `files (truncated)` when
`len(files) == 100`.

### NIT-4: small helper duplication
`_run` duplicates `pr_merge_readiness._gh_json`'s subprocess handling and `_lint_git.git`. Acceptable for a standalone
script today; fold into the shared helper proposed above at step 2, not now.

### NIT-5: wording
The bullet says "message the sessions it listed"; the script lists PRs, not sessions (fine in practice: "the sessions owning
the PRs it listed"). M83's axis "Test/CI truthfulness" reads more like "Engineering process" (throughput, not truthfulness).

## Wiring checks (no finding)
- `lint_orphan_modules.py` scans only `src/maxim/**`; a new `scripts/` file is not a candidate. Unaffected.
- `run_mypy_public.sh` lists src files only; `pr_merge_readiness.py` is not listed there either; nothing owed.
- No scripts index requires registration; the sibling is referenced from CLAUDE.md + ci-testing.md + its test, and this tool
  is referenced from ci-testing.md, M83 and its test, the right level for an advisory tool (no CLAUDE.md line needed).
- The test imports via `sys.path.insert` of `scripts/` like its siblings and replaces no module (conftest rule).

## Docstring vs code
Matches: fields printed, pairwise `merge-tree` over heads, order key (isolated, conflicts, hot overlaps, green CI, oldest;
drafts last), exit 0/2. The residuals (pairwise only, `main` moving, 100-file page) are honest; add NIT-2's ref writes and
note that heads behind `main` are compared over their own merge-base, so a conflict already resolved on `main` can be
reported (covered in spirit by "once main moves", but worth one clause).
