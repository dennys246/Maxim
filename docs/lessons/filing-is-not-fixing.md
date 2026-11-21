# Filing is not fixing: the no-band-aid rule covers deferral too

**Adopted 2026-11-21** (owner decision). It comes from an audit of all 236 open issues against `main` at
043eab8f, and of the 341 PRs merged since 2026-10-01. The CLAUDE.md guardrail "No band-aid fixes" was being
kept to the letter and missed in its intent. Very few fixes hid a symptom. What happened instead: the root
cause was found, written up accurately in a new issue, and left there. A well-written issue had come to
count as handling the problem.

## What the audit found

Figures marked *(audit)* come from the audit session's six per-batch classifications, each a reader's
judgment against the code. They have not been recomputed and are not a measurement. The rest can be
recomputed with the command or commit given.

- **Flow.** `gh issue list --state all --search "created:2026-11-01..2026-11-21"` gives 88 opened. The same
  window gives 40 closed before the 2026-11-21 sweep below. All 236 open issues were created after
  2026-09-01. Per week since 2026-09-21, about twice as many were opened as closed *(audit)*.
- **Resolvable but sitting.** About 108 of the 236 (46%) had a known root cause and a fix of about a day or
  less, needing no owner decision, hardware or experiment *(audit)*. Ten more were already fixed on `main`
  and still open. They were closed in the 2026-11-21 sweep, each with a comment naming the fixing commit:
  #1391, #1392, #1394, #1395 (PR #1503), #1267, #1366, #1435, #1567, #1663 and #1677.
- **PR-filed follow-ups.** 39 merged PRs filed 56 follow-up issues, and 28 of those were still open
  *(audit)*. About 17 of the 28 were found by the filing PR's own review round, and several shared that PR's
  root cause.
- **Ratchets that only loosened.** Growth `exceptions` in `src/maxim/utils/function_length_baseline.json`
  went from 0 (at c4962838, the last commit before 2026-10-01) to 7, with none removed. Three cite #1150 for
  the split ("splitting execute_and_learn stays #1150's"). #1150 stayed open the whole time, from
  2026-10-07, while the function grew under it. `xfail` markers went from 0 to 37 over the same span. Most
  are sanctioned strict red gates, but about 12 are parked on open defects *(audit)*.
- **Dangerous items in the pile.** Each had its fix written out in the issue body.
  - #1185: `runtime/approval.py::detect_approval_intent` read "stop, don't run it", "not ok" and "no good"
    as approval of a plan, on live input (reproduced 2026-11-21). It had sat six weeks inside a plan issue.
  - #1519: untrusted comms text reached the prompt unframed.
  - #1680: `write_paper` had never worked, because of a one-keyword mismatch.
  - #1647: the nightly slow lane had been red for three weeks, with both causes diagnosed.
  - #1228 and #1225: the agent-under-test's safety reviewer ran on the narrator's model, and `learns=False`
    was silently ignored.

## The shapes, with examples

1. **The same root cause, filed instead of folded.**
   - PR #1701 fixed one of four consumers of a model-writable field and filed the other three as #1700.
   - The fix for #971 shipped a pairing rule as three copies; the fourth site became #981.
   - The fix for #1411 built a ledger, a lock and a generation check around a config→env projection. It
     then filed the migration (#1417), plus a symptom of not migrating (#1415).

   This is the "fix that would need to be repeated elsewhere" case the guardrail already names. Filing the
   siblings does not change that.
2. **Reader-by-reader patching.** The fixes for #1414 and #1465 each patched one reader of per-tick input,
   instead of making `RuntimeState.update` clear its event keys. Each review filed the next fallout: #1422,
   #1423 and #1475. #835 has the same shape: each new consumer adds its own `startswith("[ACTION_FOLLOWUP")`
   special case.
3. **Pin-and-file during verbatim moves.** Decomposition slices moved known-buggy code unchanged, which is
   correct for a pure move. They pinned the bug and filed it: #1145, #1146, #1147 and #1202. Nothing
   scheduled the repayment, so the docstrings now list their own known bugs.
4. **A hand-made exception instead of the fix.** #1557's known wall-clock flake broke `main`. The coverage
   gate it tripped then blocked every later merge, and was cleared by a hand entry in
   `push_base_accepts.json` (#1558). The next flake repeats the cycle.
5. **Fixed, never closed.** The ten closed in the sweep above inflated the count and hid the real queue.

## Why it happened

- **Filing was always allowed and always cheaper.** The 2026-09-29 gate stop rule
  ([development-flow-speed.md](development-flow-speed.md) §1) sends a gate's fail-closed review findings to a
  follow-up issue. In practice that spread to every NIT-labelled finding, including correctness findings
  that shared the PR's root cause. Nothing asked "why can this not be fixed here?"
- **A fix costs a whole round.** That means three lenses, a ledger walk, and a full-suite slot on a shared
  box. A five-line fix costs the same as a feature, so it gets filed.
- **Nothing closed issues.** No sweep caught issues fixed by a PR that lacked `Closes #N`.
- **Fences.** File ownership between sessions stalled ready fixes, and nobody asked the fence holder.

## The rule

1. **Fold the root cause; don't file it.** A finding is fixed IN the PR when any of these holds:
   - it comes from review or from the work itself, and shares the PR's root cause;
   - it is another site of the same pattern;
   - it is fallout the PR exposes.

   File it only when the fix needs something the session cannot supply. Its `Deferred because:` line names
   one of these reasons:
   - `owner-decision`
   - `hardware` (robot or rig)
   - `experiment` (a run or re-run)
   - `upstream`
   - `fenced:<who>` (another session holds the file; ask them first)
   - `design-pass:<link>` (the issue or note where the gate's approach note is owed)
   - `too-large:<estimated src lines>`
   - `gate-stop-rule` (the 2026-09-29 rule's own case: fail-closed findings, in an already-covered class, on
     a gate, so classified by the reviewer)

   "Found in review" says where it came from, not why it waits.
2. **Small and known means now.** A defect with a known cause, no decision owed, and an estimated fix of
   about 50 `src/` lines or fewer goes in this PR, plus its test. Any extraction rule T forces does not
   count toward the 50. If it really cannot go in this PR (a fence, or the PR is already in review), file
   it with the `small-fix` label; M92 lists `small-fix` issues open for more than seven days.
3. **Silent-wrong, safety and security defects jump the queue.** That means any of:
   - output reported as right when it is wrong;
   - an approval, permission or safety check that passes bad input;
   - untrusted text reaching a prompt;
   - a defect a ledger claim rests on.

   Each is labelled `silent-wrong` or `security` when filed, and is fixed before new feature work in the
   same area. Past seven days, it needs an entry in a committed exceptions file naming the owner decision,
   as "Weak evidence never gates" requires for any override. A GitHub comment does not count.
4. **A pinned bug is a loan.** A verbatim-move slice that pins a known bug names the PR that repays it. That
   repayment lands as its own commit or PR before the next slice in that function moves more code. It stays
   separate so that slices remain pure moves, keeping their swallow-move credit and goldens. A new growth
   allowance carries a `repaid_by: #N` naming the split that retires it, and a second allowance with the
   same `repaid_by` is refused: do the split.
5. **Close what is fixed, the same session.** Every issue a PR fixes gets its own `Closes #N`. An issue found
   already fixed is closed with a comment naming the fixing commit. A session's wrap-up states issues filed
   and issues closed.

**Stated limits.** Whether a finding "shares the root cause" is judged by the reviewer during a review
round (the disposition below). Outside review, the author judges it. That half is enforced only by the
reason vocabulary and by M92's per-reason counts, which make drift onto one reason visible.

## How it is enforced

Until these checks land, the rule is a process invariant. Each is a row in
[../plans/outstanding.md](../plans/outstanding.md) §Mechanization backlog.

- **M92:** the issue-hygiene nightly. It refuses a session-filed issue that lacks a valid `Deferred because:`
  line, lists fixed-but-open and stale `small-fix` issues, and prints per-reason counts and the flow.
- **M93:** `repaid_by` on growth allowances, with stacking refused.
- **M94:** the release gate reads the age of `silent-wrong` and `security` issues.

The review charter ([../CODE_REVIEW.md](../CODE_REVIEW.md)) has each lens give every finding a disposition,
`FOLD` or `FILE: <reason>`, so for review findings the call is a reader's, not the author's.
