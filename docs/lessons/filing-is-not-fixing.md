# Filing is not fixing: the no-band-aid rule covers deferral too

**Adopted 2026-11-21** (owner decision), after an audit of all 236 open issues against `main` at 043eab8f
and of the 341 PRs merged since 2026-10-01. The CLAUDE.md guardrail "No band-aid fixes" was being kept to the
letter and missed in its intent. Very few fixes hid a symptom. Instead, the root cause was found, written up
accurately in a new issue, and left there. A well-written issue had come to count as handling the problem.

## What the audit found

- **Flow.** Every week since 2026-09-21, about twice as many issues were opened as closed. In the three weeks
  to 2026-11-21 that was 82 opened and 35 closed. All 236 open issues were created after 2026-09-01.
- **Resolvable but sitting.** About 108 of the 236 (46%) had a known root cause and a fix of about a day or
  less, with no owner decision, hardware or experiment needed. A further ~13 were already fixed on `main`
  and never closed.
- **PR-filed follow-ups.** 39 merged PRs filed 56 follow-up issues; 28 were still open. About 17 of those 28
  were found by the filing PR's own review round, and several shared that PR's root cause.
- **Ratchets that only loosened.** Growth `exceptions` in `src/maxim/utils/function_length_baseline.json` went
  from 0 to 7 between 2026-10-01 and 2026-11-21, none removed. Six said "splitting stays #1150's"; #1150 was
  open the whole time while the function grew under it. Expected-failure markers went from about 2 to 37;
  about 12 were parked on open defects.
- **Dangerous items in the pile**, each with its fix written out in the issue body:
  - #1185: `runtime/approval.py::detect_approval_intent` read "stop, don't run it", "not ok" and "no good"
    as approval on live plan-approval input. It had sat six weeks inside a plan issue.
  - #1519: untrusted comms text reached the prompt with no framing.
  - #1680: `write_paper` had never worked because of a one-keyword mismatch.
  - #1647: the nightly slow lane was red for about three weeks, with both causes diagnosed.
  - #1228 / #1225: the safety reviewer of the agent under test ran on the narrator's model, and
    `learns=False` was silently ignored.

## The shapes, with examples

1. **The same root cause, filed instead of folded.**
   - PR #1701 fixed one of four consumers of a model-writable field and filed the other three as #1700.
   - #971 shipped a pairing rule as three copies; the fourth site became #981.
   - #1411 built a ledger, a lock and a generation check around a config→env projection, then filed the
     migration (#1417) and a symptom of not migrating (#1415).

   This is the "fix that would need to be repeated elsewhere" case the guardrail already names. Filing the
   siblings does not change that.
2. **Reader-by-reader patching.**
   - #1414 and #1465 each patched one reader of per-tick input instead of making
     `RuntimeState.update` clear its event keys.
   - Each review filed the next fallout: #1422, #1423, #1475.
   - #835 has the same shape: each new consumer adds its own `startswith("[ACTION_FOLLOWUP")` special case.
3. **Pin-and-file during verbatim moves.** Decomposition slices moved known-buggy code unchanged (correct for a
   pure move), pinned the bug, and filed it: #1145, #1146, #1147, #1202. Nothing scheduled the repayment, so
   the docstrings now list their own known bugs.
4. **A hand-made exception instead of the fix.** #1557's known wall-clock flake broke main. The coverage gate
   it tripped blocked every later merge, and a hand entry in `push_base_accepts.json` cleared it (#1558). The
   next flake repeats the cycle.
5. **Fixed, never closed.** For example, #1391, #1392, #1394 and #1395 were all fixed by PR #1503 and left
   open. This inflates the count and hides the real queue. Release-gate wording can also drift away from
   the shipped fix (#1267).

## Why it happened

- **Filing was always allowed and always cheaper.** The 2026-09-29 stop rule
  ([development-flow-speed.md](development-flow-speed.md) §1) lets a gate's fail-closed review findings go to
  a follow-up issue. In practice that spread to every NIT-labelled finding, including correctness findings
  that shared the PR's root cause. Nothing asked "why can this not be fixed here?"
- **A fix costs a whole round.** Three lenses, a ledger walk, and a full-suite slot on a shared box. A
  five-line fix cost the same as a feature, so it was filed.
- **Nothing closed issues.** No sweep caught issues fixed by a PR that lacked `Closes #N`, or titles that
  drifted from the code.
- **Fences.** File ownership between sessions (Session A holds the loop file, Session V holds `api.py`)
  stalled ready fixes, and nobody asked the fence holder.

## The rule

The CLAUDE.md guardrail now reads "No band-aid fixes, and filing an issue is not a fix". In full:

1. **Fold the root cause; don't file it.** Fix these in the PR that found them:
   - a finding, from review or from the work itself, that shares the PR's root cause;
   - another site of the same pattern;
   - fallout the PR itself exposes.

   Filing is allowed only when the fix needs something the session cannot supply:
   - `owner-decision`
   - `hardware` (robot or rig)
   - `experiment` (a run or re-run)
   - `upstream`
   - `fenced:<who>` (another session holds the file; ask them first)
   - `design-pass` (a gate or new mechanism that owes its approach note)
   - `too-large` (an estimate over about a day)

   The issue body carries a `Deferred because: <reason>` line from that list. "Found in review" is where it
   came from, not a reason.
2. **Small and known means now.** A defect with a known cause, no decision owed, and a fix of about 50 `src/`
   lines or fewer plus its test goes in the current PR or the session's next one. It is not filed. The
   gate stop rule still stands, but only for its own case: fail-closed findings, in an already-covered class,
   on a gate.
3. **Silent-wrong, safety and security defects jump the queue.** That covers four kinds:
   - output reported as right when it is wrong;
   - an approval, permission or safety check that passes bad input;
   - untrusted text reaching a prompt;
   - a defect a ledger claim rests on.

   These are fixed before new feature work in the same area. One that sits more than a week needs an
   owner-recorded reason, and is labelled `silent-wrong` or `security` when filed.
4. **A pinned bug is a loan.** A verbatim-move slice that pins a known bug names the slice that repays it.
   The next slice touching that function repays it before moving more code. A growth allowance that cites an
   issue that already backs another allowance is refused; do the split.
5. **Close what is fixed, the same session.** Every issue a PR fixes gets its own `Closes #N`. An issue found
   already fixed is closed with a comment naming the commit that fixed it. A session's wrap-up states issues
   filed and issues closed; a session that filed more than it closed says why.

## How it is enforced

Until the checks below land, the rule is a process invariant:

- **M92:** the issue-hygiene nightly. It refuses a new issue without a `Deferred because:` line, and lists
  open issues cited as fixed by a merged commit.
- **M93:** `lint_function_length.py` refuses a stacked growth allowance.
- **M94:** the release gate reads `silent-wrong` and `security` issue age.

All three are rows in [../plans/outstanding.md](../plans/outstanding.md) §Mechanization backlog. The review
charter ([../CODE_REVIEW.md](../CODE_REVIEW.md)) has each lens give every finding a disposition (FOLD, or
FILE with a reason from the list above), so the decision is a reader's, not the author's.
