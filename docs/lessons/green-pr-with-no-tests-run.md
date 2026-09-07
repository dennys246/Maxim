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

Mechanically checkable, and now mechanized — see the section below:
`scripts/pr_merge_readiness.py` asserts that the required contexts are PRESENT
(and reports every other gating surface at the same time).

---

## Fourth variant (2026-09-06, PR #654) — and the mechanization

**Symptom, identical to the others:** `mergeable: MERGEABLE`, `mergeStateStatus:
BLOCKED`, twelve green rows in `gh pr checks`. **Cause, new:** not a workflow that
never fired, but a **code-scanning ALERT**. The `main-protection` ruleset carries a
`code_scanning` rule (`alerts_threshold: all`), so two `py/clear-text-logging-
sensitive-data` findings in the PR's own diff blocked the merge while appearing
nowhere as a check row.

**The misdiagnosis is the actual lesson.** The gating surface was found correctly —
the rulesets were read on the first attempt. What went wrong was the next step:

1. The CodeQL check read `neutral — "1 configuration not found"`. That matches this
   document's own documented variant ("default setup needs a PUSH"), so a push was
   made — **while `Analyze (python)` was still `in_progress`**. An aggregate check
   was diagnosed mid-flight.
2. The push produced `"2 configurations not found"` — *more* jobs in flight — which
   was read as deterioration rather than as "still running", and prompted a second
   wasted cycle.
3. Once settled, the check's own `output.summary` said exactly what was wrong:
   *"2 new alerts including 2 high severity security vulnerabilities"*. That text was
   one `gh api .../check-runs --jq .output` call away the entire time.

So the failure was **reaching for a remembered remedy before reading the
instrument** — the same shape as `verify-the-instrument` and
`diagnose-from-structured-signals-not-substrings`. Adding "and sometimes it is an
alert" to a list of known causes would not have helped: the list is always missing
the next variant.

**Resolution of the alerts themselves:** both were false positives and were dismissed
with a written justification. Queen keys are *public* ed25519 verification anchors,
not secrets; neither flagged statement emitted key material (`hive list` printed
`len(queen_keys)`, an integer; `hive trust` printed contributor ids). CodeQL taints
the whole registry dict because it contains a key-named field, so *any* read from it
reaching a `print` is flagged — including `o.get("name")`, which is why restructuring
the key handling did not clear it. The hygiene fix was kept anyway (`847f46fb`):
keys are reduced to a count before output and the printed policy is derived from
policy fields only, per the `leader_proxy._check_auth` house rule that key-shaped
values never flow toward output.

**Postscript, and the sharpest part.** The first version of that instrument was reviewed
before merge and found to commit *the same error it was built to prevent*: moments after a
push — the most common moment anyone would run it — the required checks do not exist yet,
and it asserted `required-check-absent` while **naming a cause it had not established**
("a CONFLICTING PR produces no merge commit"). It also returned exit 0 = "ready to merge"
for `DRAFT` / `CONFLICTING` / `BEHIND` PRs, because it consulted `mergeStateStatus` only for
the literal value `BLOCKED` and never read `mergeable` at all — meaning this document's own
headline case (PR #576, a CONFLICTING PR) would have received a machine-authoritative
"READY". A live run then found two more: `/rules/branch/{branch}` 404s on this repo (so the
"precise" ruleset read had to fall back to enumerating active rulesets — a 404 is not "no
rules"), and a merged PR reported an unsettled non-answer.

The durable rule extracted from that, now stated in the script's own docstring: **never
assert a NEGATIVE over an unsettled snapshot.** A settled *failure* is a positive fact and
legitimately outranks in-flight work; an *absence* is a claim about something not existing
and is only sound once nothing is still moving. Writing a tool to enforce a discipline does
not exempt the tool from it.

**Mechanization (closes the "not yet mechanized" note above).**
[scripts/pr_merge_readiness.py](../../scripts/pr_merge_readiness.py) answers "why is
this PR not mergeable?" in one command, deliberately WITHOUT encoding a list of known
causes. It reports what each surface currently says — required checks *present* (not
merely green), each failing check's own `output.title`, open code-scanning alerts
scoped to the PR, and ruleset rules that gate the merge without rendering as check
rows — and it **withholds a verdict entirely while anything is `in_progress`** (exit
code 2), which is precisely the step that was skipped above. Guard:
[tests/unit/test_pr_merge_readiness.py](../../tests/unit/test_pr_merge_readiness.py),
whose `TestReplaysPr654` replays this incident stage by stage: at the moment the first
push was made, the tool must return IN-FLIGHT with zero BLOCKING findings.
