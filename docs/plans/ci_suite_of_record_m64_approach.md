# M64: CI's `unit-tests` run is the final full suite (approach note)

Status: BUILT 2026-11-25 after the adversarial design pass (CLAUDE.md "Faster without lowering the bar" item 1: this is a gate). The pass found no DO-NOT-BUILD; its findings are folded below and listed in §Design review.
Owner decisions: 2026-10-16 (M64 adopted, waiting on this check); 2026-11-25 (push a draft PR when local
testing starts, and stop running a full local suite before the push).

## What changes for a developer

| Step | Before | After |
|---|---|---|
| Before the first push | narrow tests, a full local suite on the exact commit, prepr PASS | narrow tests on the touched modules + deletion probes, prepr PASS |
| Opening the PR | after the local suite passes | right after prepr passes, as a **draft** (CI runs only on `pull_request`, so a bare branch push tests nothing) |
| A fix | re-run the local suite, then push | push it; CI re-runs, and `concurrency` cancels the superseded run |
| The final full suite | the local suite on the exact commit pushed | CI's latest `unit-tests` run on the PR's head SHA, asserted by `pr_merge_readiness.py` |
| Marking ready | after the push | when CI is green on the reviewed head and the review round is folded |

Unchanged: prepr PASS clears every push; review rounds and delta rounds; the merged diff is the reviewed diff;
`box_quiet.py` for any local suite still run (a local suite stays useful for a slow-to-diagnose failure, it is
just no longer the gate). **Exception:** an unpublished security fix (GHSA work) never goes to a public PR
early; it stays on the private advisory fork and keeps the local full suite as its gate.

## The check

`evaluate()` gains a step "suite of record" (`_suite_of_record_findings`), plus two fixes the design pass found it depends on:

1. Pick the runs named exactly `unit-tests` whose `app_id` equals the app pinned for that context in
   `scripts/repo_settings_expected.json` (`protection.required_status_checks.checks`), read from the PR's BASE branch
   through the API (`_base_suite_app_id`; review round: the running checkout could be the PR's own, supplying its own
   pin). A same-named check from
   another app is not the suite. The step reads the raw run list and takes the **newest by check-run id**
   (`suite_of_record_run`): ids are assigned in creation order, so a re-run cancelled before it started still
   outranks the older success it replaced (design finding 6), and the existing `_latest_run_per_check` order for
   the other checks is untouched.
2. None, with nothing in flight → BLOCKING `suite-of-record-absent`. None, with checks in flight → UNSETTLED
   `suite-of-record-not-yet-present` (rule 1 of the script: no negative over an unsettled snapshot). This step
   does not read the ruleset, so a ruleset that stops requiring `unit-tests` cannot make READY suite-free.
3. In flight → UNSETTLED `suite-of-record-running`.
4. Settled with any conclusion other than exactly `success` → BLOCKING `suite-of-record-not-green` (closes
   `neutral`, which is INFO only for other checks).
5. Success → INFO `suite-of-record` naming the run id and the head SHA, so the readiness output IS the record.
6. A floors-only push re-used an earlier measurement (#1640) → INFO `suite-of-record-reused`. Read from the
   run's own job (`GET /actions/jobs/<id>`, the id from the check run's `details_url`): the `Run required fast
   suite` step is `skipped` on re-use. Exact per attempt, one call, and a failed read is INFO
   `suite-of-record-reuse-unknown`, never a blocker (design finding 3: the artifact list keeps an attempt-1
   `coverage-reused-from-1` after an attempt-2 re-run that measured). Accepting re-use rests on the workflow's
   own guarantees (`lint_coverage.py::reuse_decision`), which the design pass checked.
7. **Head moved during the read** (design finding 1): `_fetch` re-reads `headRefOid` after the check runs; a
   change is a fetch error, so UNSETTLED. (Comparing each run's `head_sha` to the head would be vacuous: the
   runs are fetched by that SHA.)
8. **`UNKNOWN` merge state** (design finding 2): `mergeStateStatus` or `mergeable` `UNKNOWN` on an open PR is
   UNSETTLED `merge-state-unknown`, because a BEHIND PR reads that way while GitHub computes it; before, it
   counted as mergeable.

`evaluate()` takes `suite_app_id` keyword-only with no default, so a caller that forgets the step gets a
`TypeError`, never a silently skipped check (CLAUDE.md "push silent-no-op invariants into types").

## Inputs, and how each could lie

| Input | Lie | Handling |
|---|---|---|
| check runs for `headRefOid` | a push lands between `gh pr view` and the merge | head re-read after the checks (7) |
| check name | another app posts `unit-tests` | app id pinned from the expected-settings file (1) |
| expected-settings file | missing, or no `unit-tests` entry | UNSETTLED `surface-unverified`: the app cannot be pinned, so no verdict |
| expected-settings vs live protection | drift | the nightly `scripts/check_repo_settings.py` fails on drift (reading live protection needs admin scope) |
| several runs on one commit | an old success hides a new cancel | newest by id (1); cancelled is BLOCKING |
| `neutral` / `skipped` conclusion | looks non-failing | only `success` counts (4) |
| job lookup for re-use | API error, wrong attempt | per-job read, exact per attempt; failure is INFO only (6) |
| merge ref vs head | CI tests head merged with an old base | `strict: true` protection makes a behind PR BEHIND, BLOCKING; `UNKNOWN` now UNSETTLED (8) |
| `workflow_dispatch` run on the branch | posts a same-app `unit-tests` on the head, testing the head not the merge ref | accepted: under strict protection the head already contains the base, so the tree is the same (design finding 5) |
| draft PR | merged before CI finishes | DRAFT is already BLOCKING; GitHub refuses to merge a draft |
| ruleset drops `unit-tests` | readiness goes READY with no suite | the step does not read the ruleset (2) |
| M61, if it lands | an `edited` run yields a newest `skipped` `unit-tests` | BLOCKING, as M61's row wants; cost today: a title edit after green re-opens a ~25-min UNSETTLED window, and a flake in it blocks (design finding 7) |

## What CI's `unit-tests` does not cover (accepted, design finding 4)

Dropping the local full suite as the gate drops: macOS-only tests, and tests that `importorskip` a
developer-only dependency outside `--require-extras=console,sign` (reachy_mini, e.g.
`test_reachy_speaker_sink.py`, runs in no CI lane; sentence_transformers/torchvision tests such as
`test_clip_encoder` and `tests/substrate/*` run nightly in the model-cache lane). A change touching those runs
its narrow tests locally, where the dependency is installed; that is already the narrow step.

## Early drafts are public (design finding 9)

Never pushed early: an unpublished security fix (the call between the private GHSA fork and a public PR is made
before the FIRST push of any change touching a security boundary, since a `security`/`silent-wrong` label is
often only assigned later), and experiment data or prereg artefacts behind a blind-grading firewall. More open
drafts mean more runner contention (the 2026-08-06 starvation): the expected wait is queue time, not suite time.

## Out of scope

Running readiness automatically before a merge (M6, closed: GitHub's strict required checks already refuse it).
A second machine's suite (`rig_suite.sh`) stays available but is no longer a gate.

## Design review

One adversarial pass, 2026-11-25: no DO-NOT-BUILD; SHOULD-FIX 1, 2, 3, 4, 9 and NIT 5, 6, 7 folded above; NIT 8
(app pin source) confirmed the design. Findings 1, 2, 4 and 5 were fail-open or a coverage loss, so they were
folded before the build rather than filed.

## Docs that change in the same PR

CLAUDE.md "Faster without lowering the bar" item (2); `docs/agents/ci-testing.md` §1 (the "not yet in force"
paragraph) and §4 (the pre-push flow); `docs/lessons/development-flow-speed.md` §2 note; `docs/plans/outstanding.md`
row M64 → shipped; `docs/agents/ci-testing.md` §3 gains the invariant with its regression guard.
