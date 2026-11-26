# Exp 63c: the build's rulings on the frozen text (S1–S5)

The harness's three-lens review round (2026-11-10) found five places where the frozen prereg
([exp63c_scripted_recall_prereg.md](../../exp63c_scripted_recall_prereg.md), PR #1625) can be read more than one way.
The process session (maxim-4d) ruled on each one, on 2026-11-10. These are rulings on the frozen text, not edits to
it. Each ruling takes the literal or the stricter reading, and none weakens a gate. The owner can overrule any of them
at merge. The harness (`scripts/o19_verdict.py`, `scripts/o19_rerun.py`) implements them as stated here.

## S1. The smoke and the checks that need R3d

**Question.** §Smoke's heading says the smoke "never computes R3a or R3d". 63c's C5 adds (k), "an armed control
reads decisive", and (q), "a decisive pair shares a cue non-goal key". Both need R3d's decisive computation.

**Ruling: (a).**
- The smoke skips (k) and (q). They are defined under §Gates and run at verdict time, where they gate the
  complete-attempt check.
- The smoke prints `C5(k)/(q): verdict-time only (§Smoke heading)`, so nobody reads their absence as a pass.

In the code, `exp63c_c5_problems(..., verdict_time=False)` never calls the turn decision with `decide=True`.

## S2. What "echo load" counts

**Question.** §Report says "echo load: new records holding each planted key at that turn". Which keys: the answer
keys, the cue's keys, or every key of the fact?

**Ruling.** The frozen replay settles it. Its `echo_load` counted the cue's non-goal keys
(`replay/replay.py`: `k in sorted(d["ng"])`), and that is the echo load the freeze tables and the forecast rest on.
- "Echo load" means the new records holding each of the cue's non-goal keys.
- The answer-key count is reported beside it as a separate descriptive item, "answer-key load".

The report never gates.

## S3. "Contested", and the PASS ledger wording

**Question.** §Report defines a decisive win as contested when the best pair's displaced id held one of the cue's
non-goal keys. Two things are unstated:
- which tie order applies, since decisive is computed over every valid order;
- whether "uncontested: every displaced record shared only the goal word" may follow from "no best pair contested".

**Ruling: fail-safe.**
- (i) A turn's `contested` flag is true when ANY valid order's best pair is contested. The report lists every
  order's best pair and its scores.
- (ii) The PASS ledger wording "uncontested: every displaced record shared only the goal word" is used only when its
  literal statement holds: no displaced id, in any valid order, on any decisive turn, holds a cue non-goal key.
  Otherwise the wording names the contested turns.

In the code, (ii) is `uncontested_literal` per turn, and `uncontested` and `contested_turns` in the descriptive block.

## S4. A manifest that touches a control

**Question.** §Smoke says a control that the run's LLM-made manifest touches "reads contaminated by the armed
rule". The armed rule reads only carried records, and the manifest text is not logged.

**Ruling.** Armed versus contaminated is the carried-store rule only. This is consistent with the text:
- the manifest reaches the ranker only through a record (for example, sense outputs carrying its entities);
- any carried record that holds a control key is already caught by the armed rule.

The smoke REPORTS imagination and design lines that hold a control or planted key. It never sets C5.

## S5. Which harness reads S2′ judges

**Question.** OD-c4 says S2′ judges reads "after the first OK one", while the field-by-field text says "from the
first OK read to the last OK read".

**Ruling: the field-by-field text** (delta wiring 1–2) is the specification.
- Reads are judged from the first OK read to the last OK read.
- Failed reads after the last OK one are reported.
- An attempt with ZERO OK reads fails S2′ (it is incomplete); it is never a vacuous pass.
- A non-OK read BETWEEN the first OK read and the last is judged, and so it fails.

## Also decided

**Red gates (build order step 4).** No exception. The harness branch's history has a commit of the 63c tests marked
`xfail(strict=True)`, with the suite green while they xfail, before the commit that lands the harness and removes the
markers.

**The peak-prompt check (environment N3, a spec note).** §Smoke says "the peak prompt size against `n_ctx`". The
rationale asks for `input_tokens + max_tokens < n_ctx`, but `max_tokens` is not logged on `peer_backend_call`. So the
smoke checks `input_tokens < n_ctx` and the PR says so. Tightening it needs the call's budget in the log, which is a
`src/` change outside this build.


## 2026-11-21: the shipped frame rule changed after the run (#1519); the verdict keeps the executed rule

#1519 changed the shipped frame-pairing rule after this experiment ran. All frame consumers now pair through
`content_safety.frame_spans`, which also refuses a pair whose span crosses a section header. One edge case changes
how stored text is keyed: text holding repeated same-id openers. `memory/text.py::text_values('a <<X…>> b <<X…>> c
<</X>> d')` gave `['a  d']` at e841abca, 9176025a and 043eab8f, and gives `['a ']` after #1519.

Owner decision 2026-11-21: the verdict port follows the EXECUTED rule. `scripts/o19_verdict.py`'s frozen port
(`_P_FRAMED_REGION`, `_p_unbalanced_markers`) is unchanged. Its pins
(`test_exp63b_the_port_is_the_executed_keying`, `test_exp63c_the_port_reads_records_as_the_shipped_store`) now
compare it with the shipped code AT THE EXECUTED COMMITS. The executed frame rule is vendored, byte-checked against
e841abca and 9176025a, at `tests/fixtures/exp63b/executed_content_safety.py` (the `executed_frames` fixture).
Ranking is unchanged; nothing is re-run.
