# A PR can show every check green while the test suite never ran

**Date:** 2026-08-30 (PR #576, the 1.1.2 cycle)
**Class:** the D37 shape one layer up — a guard that is *absent* reads as a guard
that *passed*.

## What happened

PR #576 was opened after a full local verification (9,710 passed, 0 failed).
`gh pr checks 576` returned:

```
CodeQL             pass  2s
Analyze (actions)  pass  40s
Analyze (python)   pass  1m25s
```

Three green rows, nothing red, nothing pending. It reads as a passing PR. It was
not: **the entire `Tests` workflow never fired.** No `unit-tests`, no `lint`, no
`release-build`, no compatibility matrix. Every check in that list came from
CodeQL, which runs off a different trigger.

## Cause

`main` moved while the branch was in review (#574 and #575 merged within an hour
of each other). GitHub marked the PR `mergeable: CONFLICTING`, and a
`pull_request`-triggered workflow **cannot run without a computable merge
commit** — there is nothing to check out. GitHub does not surface this as a
failed check; the jobs simply never exist.

`gh run list --workflow=Tests` confirmed it: runs for `main` (push) and for the
*other* open PR, none for this branch.

## Why it is dangerous

The failure is **indistinguishable from success in the same view an author uses
to decide the PR is ready.** A red check is a stop sign; an absent check is
nothing at all. Reopening the PR does not help — `reopened` still cannot compute
a merge commit. Only resolving the conflict and pushing does.

Had this merged, `main` would have taken 10 commits — including an
`agent_loop.py` extraction and a new CI job — with zero test evidence, while the
PR page showed all-green.

## The rule

**Before trusting a PR's checks, confirm the checks you expect are PRESENT, not
just that the visible ones are green.**

```bash
gh pr view <N> --json mergeable -q .mergeable      # must be MERGEABLE
gh pr checks <N>                                    # must LIST unit-tests + lint
gh run list --workflow=Tests --limit 3              # your branch, event=pull_request
```

For this repo the expected set on a PR is: `unit-tests`, `lint`,
`release-build`, four `Python N.NN compatibility` lanes, and `aarch64 resolve`.
`Model-cache tests`, `Slow tests`, `Release-object audit` and `aarch64 real
install smoke` correctly show `skipping` on PRs — that is their `if:` condition,
not an absence.

## Three variants, all of which look like a healthy PR (2026-08-31)

The 1.1.2 stack hit all three in one day. They are distinguishable, and the
distinguishing question is always *did the mechanism run?* — never *is anything
red?*

| symptom | cause | fix |
|---|---|---|
| 3 green checks, no `unit-tests` in the list | PR is `CONFLICTING`; no merge commit can be computed, so `pull_request` workflows cannot run. CodeQL uses a different trigger and runs anyway | resolve the conflict and push. **Reopening does not help** |
| **no checks at all** | the PR was opened against a sibling branch, so `branches: [main]` filtered it out; retargeting to main fires `edited`, not in the default `[opened, synchronize, reopened]` | close + reopen (fires `reopened`), or push. **Fixed at source 2026-08-31** by adding `edited` to `types:` |
| all required checks green, merge still **BLOCKED** | a repository RULESET — separate from classic branch protection — requires a CodeQL analysis (`alerts_threshold: all`), and CodeQL's default setup wants a **push**; a reopen does not give it one | push a commit (an empty one is enough) |

The third deserves dwelling on. `gh api .../branches/main/protection` reported
required contexts `["unit-tests", "lint"]` and both were green, yet
`mergeStateStatus` was `BLOCKED`. **Classic branch protection and rulesets are two
different surfaces**, and `gh pr checks` shows neither the ruleset nor the absent
CodeQL analysis — `gh api repos/<o>/<r>/rulesets` is where the answer lives. The
temptation at that moment is to merge from the CLI to route around a UI that
"won't let you"; that would have merged code with no code-scanning analysis, on
the reasoning that the guard's silence meant approval.

## Generalisation

This is the same family as the vacuous-guard findings the 1.1.2 review round
produced (`fail_loud_stage2.py check` passing on an empty capture;
`pytest -m slow` exiting 0 having collected nothing): **an enforcement mechanism
that does not run looks exactly like one that ran and found nothing.** The
counter is always the same — assert that the mechanism *executed*, not merely
that it did not complain.

Mechanically checkable and not yet mechanized: a step asserting that the PR's
check list contains the required job names.
