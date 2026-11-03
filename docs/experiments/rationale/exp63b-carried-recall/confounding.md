# Exp 63b: confounding lens (design review, 2026-11-02)

Reviewed: `docs/experiments/exp63b_carried_recall_prereg.md` (DRAFT v1; nothing built, nothing run), against
`docs/experiments/exp63_carried_recall_prereg.md` and the base design's
[confounding review](../exp63-carried-recall/confounding.md).

Charter question: could a PASS, a NOT SHOWN or a FAIL arise for a reason other than the claim?

Short answer: **G1 (the recomputation over the cue path) and I1 are the right core, and R3′ keeps real power. Three
things are wrong before a harness is built.** The evidence table omits the own-capture skip, and that skip removes one
of the two candidate decisive turns and refutes G4's premise. The table's volumes are a defect that #1414 removed, and
capture volume alone moves R3d from near-certain to unlikely. R3a's `min(3, E)` reads FAIL on correct code, and the
verdict table can raise the row on a run in which no carried record ever mattered.

All numbers below were recomputed on 2026-11-02 from Exp 63's committed stores and logs. They are **old-code bytes,
design use only, and move nothing**. The throwaway scripts are in the session scratchpad (`common.py`, `conf1.py` to
`conf5.py`); nothing tracked was changed but this file. The Monte Carlo figures are a crude lexical model (the 16
committed narrator lines reshuffled, the 15 committed records with no input text, one narrator line per turn, no AUT reply records).
They show how the gates behave as a function of a variable. They do not predict mistral's text.

---

## What the replay shows once the skip is applied

The draft's table is arithmetically right: I reproduced every cell, and no `query_text` in the log reaches the 120-character
cut, so the replay's query is whole. Its method is the problem. `_query_hippocampus_traced` skips a record whose
`_same_words(input_text)` equals the turn's words on **every path, whatever its origin**, and the skip runs AFTER
`recall(query=text, limit=5)` has taken its five slots. Applying it (`common.py::surface`, a line-for-line port of
`_add_memory` and the path order, no graph or substring):

| Turn | Line | same-words records (carried) | cue's 5 slots | paths shown | carried shown | decisive (G4) |
|---|---|---|---|---|---|---|
| 1 | Can you tell me where the sleeping guard is located in the dungeon? | 0 | 5 carried | cue ×3 | 3 | R3a's turn |
| 2 | What is your next step toward: escape a dungeon with a sleeping guard? | 13 (13) | **all skipped** | **goal ×3** | 3 | yes, **on the goal path** |
| 3 | What is the best strategy to escape a dungeon with a sleeping guard? | 0 | 5 new | cue ×3 | 0 | no |
| 4 | What is your next step toward: escape a dungeon with a sleeping guard? | 60 (13) | **all skipped** | **goal ×3** | 0 | no |
| 5 | Can you describe the sleeping guard's position and any nearby obstacles? | 0 | 3 new, 2 carried | cue ×3 | 0 | no |
| 6 | Can you tell me the location of the sleeping guard? | 0 | 5 new | cue ×3 | 0 | no |
| 7 | Which direction should I turn to reach the exit? | 0 | 3 new, 2 carried | cue ×3 | 0 | no |
| 8 | Can you describe the sleeping guard's location and appearance? | 9 (9) | **all skipped** | **goal ×3** | 0 | **no** |

The line on turns 2 and 4 is `stall_detector.py`'s generic nudge (`What is your next step toward: {goal_short}?`), a
fixed template. It also appears on phase 1's turns 6 and 8, so it is a same-words hit on carried records.

---

## DO-NOT-BUILD

### D1. G4's premise and its power statement are false once the own-capture skip is applied

The draft states: "on every replayed turn it holds at least 3 records, so the goal path never runs and Exp 63's R3d
would read `NOT SHOWN` by construction", and "The replay gives turns 2 and 8 as candidates, so the gate has power".

- **The goal path runs on turns 2, 4 and 8** (3 of 8), because all five cue slots go to records the skip then removes.
- **Turn 8 is not a candidate.** Its "3 carried at 1.00" are the 9 carried records holding the turn's exact words. They
  are skipped, and the goal path shows 3 new records.
- **Turn 2 survives, but through the goal path, not the cue path.** So on these bytes the cue path never yields a
  decisive turn, and G4's title ("the ranking the claim is about now runs on the cue path") has no replay evidence.
- **Exp 63's R3d would not be NOT SHOWN "by construction"** on these bytes: turn 2 is a goal-path decisive turn. The
  case for changing R3d is still right (a cue-decided turn must be able to count), but this argument for it is not.
- **Why it matters beyond the prose.** G1 reads correctly and would catch this at build, but the prereg would freeze a
  power statement no recomputation supports, and the owner would be asked to approve a gate on that statement.

**Fix.** Re-derive the table with the skip applied, and add columns for same-words records, the paths shown and
decisive-under-G4. Delete "the goal path never runs" and "turns 2 and 8". Then apply D2: state that the power on the
new code is unknown.

### D2. R3d's outcome is decided mainly by capture volume, a variable the design does not control, and the table's volume is a removed defect

The committed store's size is the #1414 defect. The same-words new records arrive at a median gap of **0.55–0.60 s** (121
copies of one line, 82 of another, 20, 19, 14): `propose_intent` re-published the input every loop tick. 63fd32c0
(#1414, 2026-10-22, `loop_perception.py::take_user_input`) made an input live for the tick that delivers it, and
`publish_arrival` publishes once. So the 84 + 272 records behind the draft's table should not exist on the subject code.
No committed AUT store postdates the fix (the newest are 2026-10-03), so the real volume is unknown until a run.

It is the dominant variable. Holding the 16 lines and the plan records fixed, and varying only captures per input
(`conf5.py`, 500–1,500 shuffles; the draft's rule):

| Captures per input | P(a decisive turn after turn 1) | R3a false FAIL (D3) |
|---|---|---|
| 1 (what #1414 should give) | **0.999** | 0.106 |
| 3 | 0.815 | 0.495 |
| 20 (the committed run's order of magnitude) | 0.258 | 0.500 |

So R3d goes from **automatic** to **unlikely** on a variable unrelated to the claim. At one capture per input a NOT SHOWN
would be a 1-in-1,000 lexical event and a signal to investigate, not a mild narrowing. At the committed volume a NOT
SHOWN is the expected outcome. "The gate has power on this protocol but is not automatic" is unsupported in both
directions.

**Fix.**
1. Say in the prereg that R3d is a **non-vacuity condition on R3′** ("this run contained a turn where carried records
   changed what surfaced"), and not an independent test (S5 gives the reason).
2. Stamp new records per input as a descriptive field, so the reader sees the regime.
3. The build-order step 3 real short sim reads the capture count from its store and reports it in the PR, before the
   first attempt, on its own throwaway data home. It reads no gate. If the count is far from one per input, the
   #1414 fix did not reach the sim path, which is a finding to stop on.

### D3. R3a's `min(3, E)` reads FAIL on correct code, and the new text drops Exp 63's race rule

**`min(3, E)` ignores the skip and the limits.** E counts "the carried records the cue or the goal scores above zero".
Surfaced ids are bounded by the cue's 5 and the goal's 3, and the skip removes same-words records from both. If turn 1's
line repeats a phase-1 line, E is large and fewer than `min(3, E)` surface, with every line of code correct. In the
model that is **10.6% of runs at one capture per input and 50% at three or more** (`conf5.py`). Phase 1 and phase 2 share
the stall template and near-identical narrator lines, so a repeat on turn 1 is not exotic. The table sends that to
**BROKEN**, which blocks the release.

**The race rule went missing.** Exp 63 says an id outside C on the R3a trace, or no qualifying trace at all, makes the
attempt **incomplete**: a record landing between the horizon read and the recall is not a defect. G3 replaces the
section and restates neither branch, and "Everything Exp 63 states stands" cannot rescue a passage G3 rewrote. Read
literally, "the first … trace" is undefined when no trace qualifies, and a late id is a FAIL.

**Fix.**
- R3a: on the first qualifying trace, every surfaced id is in C and the count is **the recomputed count** (R3′'s own
  computation at that trace, skip and limits included). E is that recomputed count.
- Restate both branches: no qualifying trace, or a surfaced id outside C that is not in V0 (a late arrival), makes the
  attempt incomplete, while P1 and P2 hold. A surfaced NON-late id outside C stays a FAIL: that is the skip failing.
- State what R3a then is: given R3′ at that trace, it adds only "the own capture did not surface", which is worth
  keeping, but it is not a second independent recall test.

### D4. The verdict table can raise the row on a run in which no carried record mattered

Asked: is NOT SHOWN → PARTIAL sound under "weak evidence never gates"? **In one branch arguably, in the other no.**
The row lumps two different states.

- **R3a unobservable (E = 0).** No surfaced set on the first turn involved a carried record. R3′ then has no power over
  carried records on that turn. The mutation run (C below) shows R3′ sees a carried-record defect ONLY on turns where a
  carried record reaches the recomputed result, so a run with no such turn would conform with carried records dropped
  entirely. P0–P3 still pass, and they compare phase 1's SAVE with phase 2's, so a defect that loses the text at save
  is invisible to them: every carried record holds nothing, E = 0, the verdict reads NOT SHOWN, and the row rises to
  PARTIAL. With both phases on one goal, E = 0 is itself near-impossible on working code (83 of 84 carried records hit
  on turn 1, and 83 of 84 hold a goal key). So the likeliest cause of this branch is the defect the row exists to catch,
  and the draft pays it with a raise from rank 0 to rank 2 that clears the release.
- **R3a held, no decisive turn.** The first-turn clause is shown and the run is complete, so this is not weak evidence in
  the CLAUDE.md sense: no typed abort, nothing operator-attested. But it reverses Exp 63's own rule (D2 of 2026-10-03:
  NOT SHOWN "could not test the claim", no status change), it does so for a release that is waiting, and D2 above shows
  the branch's meaning depends on volume.

**Fix.**
- Required: **R3a unobservable never raises.** Add a power floor: any raise needs at least one turn on which the
  recomputation surfaces a carried id. Below it the verdict is `NOT SHOWN`, T1-16 stays STALE, and the failing floor is
  investigated.
- Owner decision for the second branch, strict option first: **(a) no raise on any NOT SHOWN**, Exp 63's rule, and the
  release waits (the default recommendation); or (b) R3a held and no decisive turn → PARTIAL, with its narrow text naming
  what was pulled from release framing. (b) is defensible only with the floor above, and the prereg must say it reverses
  D2 of 2026-10-03 knowingly. Either way the mapping is frozen before data, which it already is.

---

## SHOULD-FIX

### S1. The new C5(L) turns the central defect into an ABORT

G5's C5(L) is "each turn's `memories` equals the recomputed count", and C5 failures abort. Before, it could only fire
on a swallowed exception. Now it fires on any count mismatch, and a count mismatch is also what the claim's central
defect looks like: in the mutation run, carried records with empty tokens and carried records dropped from recall both
give a count mismatch on turn 1. Those attempts read incomplete, never FAIL, so after 3 attempts the result is ABORT
and the row stays STALE. It fails closed, but it files a recall defect as an instrument fault, and it burns attempts.

**Fix.** Log what the check wants to know. I1 gains `recall_failed: bool`, true when the outer `except` in
`_query_hippocampus_traced` fired. C5(L) reads that flag and nothing else. A count mismatch with the flag false goes to
R3′ and FAILs. (That handler is a `log.debug` today, so the same change can move it to the Stage-1 form.)

### S2. G1's R3′ can fail, but it only sees defects that reach a top-3

Does R3′ remain capable of failing? **Yes.** Mutating the live side while the judge recomputes from the saved bytes
(`conf3.py`, committed store, skip applied):

| Live defect on carried records | Turns logged ≠ recomputed | Count-mismatch turns |
|---|---|---|
| dropped from recall | 1, 2, 8 | 1 |
| all tokens lost | 1, 2, 8 | 1 |
| heard text not restored | 1, 2, 8 | none |
| outcome fields not restored | **none** | none |
| `active_goal` label not restored | **none** | none |

So it detects the loud defects, and it misses a partial field loss, because which fields carry a win is lexical. It
also misses anything on turns where carried records are outside the top: on these bytes, 5 of 8 turns say nothing about
carried records, and those conformance passes should not be read as "every turn".

**Fix (cheap, and it removes the lottery from this half).** One logging line after the resume restore: for the loaded
store, a digest over sorted `(id, sorted(content_tokens()))` of every record, which the judge recomputes from phase 1's
saved bytes. That is a live-side identity on the ranker's whole input for every carried record, whatever the turns'
wording, and it answers "full candidates" directly. A new gate (R2′), failing to FAIL.

### S3. `cue_text` and `own_words` are self-reported, so a wrong query conforms

R3′ recomputes from the query the subject says it used. If `enrich` ran on the wrong text, or `_turn_input` held a
stale line, the live call and the judge agree and R3′ passes. `own_words = ""` when an own capture should have been
skipped would pass the same way, and that is the case R3a's skip check most needs. `self._turn_input` is sticky by
design, so a stale value is reachable.

**Fix.** C5 addition: each turn's trace has `own_words` equal to the `_same_words` of a saved non-carried record's input
in that turn, and `len(cue_text)` agrees with the `text_len` on the bridge's `Bridge.send_and_wait ENTER` line when the
trace is the percept-path one (the short sim confirms the two are the same string). Both sources are independent of the
pipeline.

### S4. R3d's competitor wording: the floor is right, "returned" is wrong, and the own capture must be excluded

Asked: is requiring the competitor to score above zero the right strictness? **Yes, and it is the part of G4 that
carries the meaning.** A carried record above a score-0 record competed with nothing, since D1 excludes that record.
With the floor, "ranks above a new record" is close to the counterfactual "remove every carried record and a new record
enters what is shown". Two defects in the text:

- **"that the ranker returned, i.e. scores above zero" is two different sets.** `recall` returns the top 5, and a
  record at rank 200 was not returned. The literal reading changes the outcome: P(decisive) 0.825 / 0.42 / 0.00 at
  1 / 3 / 20 captures per input, against 0.999 / 0.815 / 0.258 for the score floor. Pick one.
- **The own capture is counted as a competitor.** It is skipped and never competed. On the goal path it can sit below a
  carried record (a line with few goal keys), so it can supply the only "competitor". The effect is small in the model
  (0.999 / 0.811 / 0.252 with it excluded) but it is a false positive by definition.

**Fix.** Define it as the counterfactual, the natural successor of `o19_verdict.py::forces_carried`: a turn is decisive
when, in every valid tie ordering, a carried id is shown from the cue or goal path, and recomputing the same turn with
all carried ids removed shows a non-carried id that was not shown (0.999 / 0.813 / 0.192). If the draft's form is kept,
delete "returned", and exclude same-words records from the competitors.

### S5. R3d is origin-blind and does not test persistence, which is fine only if it is not sold as doing so

Asked: would a store with the carried records' origin scrambled pass identically, and is that fine?

- **Yes, it would pass.** The ranker reads no origin, so "carried" is only a label on a set of ids. A control with no
  session boundary at all: take phase 1 alone and call the records visible at turn 4 "carried", then apply G4 to turns
  5–8. **Turns 5 and 6 are decisive** (`conf3.py`; split at turn 5: turn 6). Nothing was saved or reloaded.
- **That is fine**, and it is the claim: taking part "as full candidates" MEANS origin does not matter. Persistence
  evidence comes from P1–P3 and from R3′ conforming on ids that came from disk. But then R3d carries no evidence about
  persistence, only about whether the run exercised it. The draft promotes it into the claim ("a carried memory outranks
  a new memory that competed with it"), a sentence T1-16's ledger text does not have.
- **What explains a decisive turn.** Word overlap between two sessions narrated from one goal, not anything specific to
  carried memory. On these bytes 66 of 84 carried records hold all four goal keys, because the narrator restates the
  goal and the stall template quotes it. Turn 2's decisive win exists because turn 1's line held 3 of 4 goal keys
  (`escape` missing), so those 15 new records score below the carried records on the goal path. The next line to hold
  all four keys puts new records at 1.0, recency then beats carried records on the goal path for good, and that path can
  no longer be decisive: it is monotone, as Exp 63's was. Those are carried bytes, so the win is genuine, but it is
  lexical.
- **A carried record holding the turn's exact words cannot itself be decisive:** it is skipped. Turn 2 is an exact
  repeat and is decisive only through other carried records on the goal path. A near-repeat is not skipped, so it can.

**Fix.** Keep "outranks" out of the claim sentence and state R3d as the non-vacuity condition (D2). Add to "Not claimed":
recall of carried memory under a different goal, and any origin-specific effect. Say in the prereg that R3d passes on an
origin-scrambled store by design. Report per turn which tokens decided each decisive turn.

### S6. "The only moved variable is the subject code" is true of the argv and hides what moved

The sentence is accurate, and because every gate compares a run with itself, the moves below do not confound a verdict.
They do confound any reading of Exp 63's numbers or the replay as a forecast. 281 files under `src/maxim/` changed since
52653848, and the ledger already says the turn content is not comparable. Named moves that reach these gates:

- **#1414** (D2): store volume.
- **`publish_arrival`** (Session R): the own capture now precedes enrichment, which is what makes G3 necessary.
- **#1260, #1261:** the narrator's tools and what it sees, so its lines, and so all the overlap R3d reads.
- **`stall_detector.py`** was extracted, and #1488 touched the stall arm's test floor. The template's line appeared on 2 of 8
  phase-2 turns here, and its rate decides how often the skip eats the cue. I did not establish whether its firing changed.
- **`content_tokens` scope:** `text_fields` now holds outcome results and reasoning, so an AUT's own words enter the
  pool, a term the model below leaves out.
- **Prompt size at `n_ctx 8192`:** enrichment renders differently. C1's run-to-cap condition has not been shown on this
  code with mistral-7b, so ABORT is a live outcome; the short sim checks it.

**Fix.** Replace the sentence with this list, and add: "no number from Exp 63's run forecasts this one". `--sim-run-full-turns`
still exists (`cli_parser.py`), and no sim-end consolidation runs without a persistent agent
(`orchestrator.py`, `consolidation="full" if persistent_agent is not None else None`), so those two assumptions hold.

### S7. Serialization: C5(d) and the G2 pin still have the old ranker's scope

- **C5(d)** checks `perception.observations.text` because the JSON writer stringifies a non-JSON value
  (`atomic_write_json(..., default=str)`, confirmed) that the live ranker would ignore. The ranker now walks all of
  `text_fields`, including whole `observations`, `outcome.result` and `decision.reasoning`, to depth 3. A non-JSON value
  anywhere there is skipped live and becomes a string in the saved bytes, so the recomputation differs from the logged
  ids. That is a false R3′ FAIL, and BROKEN. `Outcome.__post_init__` covers tool results (`_stored_result`, #1163), but not
  the other fields.
- **The G2 pin is "over real committed records"**, which are old-shape bytes. Their `outcome.result` is `{}`, so the pin
  would never exercise the fields the new ranker added.
- **G2's port list omits** what the ranker's inputs now pass through: `_METADATA_KEYS`, the `TOOL_OUTPUT` frame removal
  in `text_values` (`utils/content_safety.py`), and, for the skip, `input_text`'s fallback order and `_same_words`'s NFKC.

**Fix.** Widen C5(d): every value `text_fields` walks is JSON-native in the saved store, any `default=str`-shaped value
making the attempt incomplete. Pin the port against `content_tokens(from_dict(to_dict(record)))` over the short sim's
real new-code store as well, and over hand-built records with results, frames and metadata keys.

### S8. The build order has no ablation

R3′'s failure paths are unproven until a mock shows them. **Fix.** Before any rig run the mock must show, as a
deletion-style probe: carried dropped from recall → FAIL; carried tokens lost → FAIL (not ABORT, S1); own-capture skip
disabled → FAIL on R3a; a repeated turn-1 line → PASS (D3); a late id on the R3a trace → incomplete; and an
origin-scrambled store → the same verdict (S5).

---

## NIT

- **N1. A shipped-path observation for the owner, not an experiment fault.** The skip is documented as "this turn's own
  capture", yet it removes ANY record with the same words, of any age and session (13 carried on turn 2, 9 on turn 8).
  Because `limit=5` applies before the skip, a repeated input gets nothing from the cue path at all (3 of 8 turns here).
  A question re-asked in a later session therefore loses its own earlier record, and five slots with it. It is a
  candidate issue, and a fix is a re-run trigger for this row. Under one capture per input the saturation needs five
  repeats, so it may be mild on the new code. I did not assess the product impact.
- **N2. C5(s).** On the committed store no non-own record contains any turn's line (0 of 8 turns), so the substring path
  stayed dry. Under-3 turns will be common at low volume (84–92% of model runs have one), so the path will run. I1's
  `cue_text` makes substring ids recomputable, so "instrument limit" could be retired by porting
  `Hippocampus._memory_matches_query`; keeping the abort is acceptable, since it fails closed.
- **N3. Re-run trigger.** "`memory/text.py`'s keying" and "the memory record shape" do not obviously cover
  `types.py::text_fields` and `content_tokens`, which decide what the ranker reads without changing the shape. Name them.
- **N4. The verdict table's FAIL row** says "R3a" while the NOT SHOWN row says "R3a unobservable". After D3 and D4, write the
  three R3a states (held, failed, unobservable) out once.

---

## What did not fail

- **The R3a view does not let a contaminated view through.** It requires empty holes and every visible non-carried
  record to be an own capture, and non-carried records only accumulate, so it can only be the first trace of turn 1. A
  reserved number above the horizon is not a hole (`stored_capture_seqs` ranges floor..horizon), as G3 says. The remaining
  contamination routes are a late arrival (D3's branch) and a stale `own_words` (S3). A narrator line that repeats is
  fine, since the live call skips it too. One check for the short sim: if the resumed loop captures anything non-own
  before turn 1's first trace (a scenario text, a startup record), no trace qualifies and every attempt aborts.
- **Ties.** No two of the 356 records share a timestamp, so "every valid tie ordering" costs nothing on real data.
- **The own-capture logic in G1** matches `_add_memory`, including marking a skipped id seen on every path. The early
  `break` in the cue loop changes nothing the `[:3]` does not already.
- **The carried recency handicap** is as the draft says: a carried record wins only on a strictly higher score.
- **Exp 63's C5(c) rendering identity** still holds structurally: `_activate_rendered` activates `memories[:3]`, and the
  loop's percept path (the D5 recall-only case included), its deliberation cycles, the exec agent and the think tool all
  render through `format_thought_response`.

---

## Verified

Against the worktree at origin/main (480ba23c) with `PYTHONPATH` set absolute; `maxim.__file__` read back as the worktree.

- **Skip semantics:** `src/maxim/integration/bio_enrichment.py::_query_hippocampus_traced` (`_add_memory`; cue `limit=5`;
  goal `goal_results[:3]`), `::_same_words`, `::_input_text`.
- **Ranker:** `src/maxim/memory/hippocampus_retrieval.py::_rank_by_relevance` (D1; recency tiebreak; `limit`), `::recall`
  (every record a candidate; no other filter on a query path).
- **Tokens:** `src/maxim/memory/text.py::text_values`, `::key_tokens`, `::content_tokens`;
  `src/maxim/memory/types.py::EpisodicMemory.text_fields`, `::content_tokens`, `::input_text`, `::Outcome.__post_init__`.
- **Goal keys:** `key_tokens(goal)` = `escape, dungeon, sleep, guard`.
- **Replay:** the supplied `replay.py` output equals the draft's table cell for cell; `conf1.py` adds the skip.
- **Volume:** 0.55–0.60 s median gap on the five largest same-words groups (`conf2.py`); 63fd32c0;
  `src/maxim/runtime/loop_perception.py::take_user_input`; `src/maxim/agents/perception_agent.py::publish_arrival`. No
  `aut_hippocampus.json` was added under `docs/experiments/data/` since 2026-10-22; the newest session is 2026-10-03.
- **Carried content:** 76 of 84 carried records heard a line, 8 are plan records; 66 of 84 hold all four goal keys.
- **Holes:** `src/maxim/memory/hippocampus.py::stored_capture_seqs`.
- **Substring:** `src/maxim/memory/hippocampus.py::search_by_content`, `::_memory_matches_query`.
- **Writer:** `src/maxim/utils/atomic_io.py::atomic_write_json` (`default=str`).
- **Judge:** `scripts/o19_verdict.py::exp63_gates`, `::exp63_decisive`, `::forces_carried`, `::turn_traces`, `::visible_ids`,
  `::phase_argv`.
- **Ledger:** `docs/plans/behavioral_graduation_candidates.md` (T1-16's text; `STALE` rank 0, `PARTIAL` rank 2, "both block");
  `scripts/_evidence_records.py::o19_table_problems`.
- **Template:** `src/maxim/simulation/stall_detector.py` (the generic fallback line).

**Not verified, and why.** What the subject code actually writes per input in a sim: nothing post-#1414 is committed and
no sim was run (out of bounds). All of D2's and S4's percentages are conditional on the crude model above, and their
direction and scale are the finding, not their digits. Whether the stall template's firing rate changed since 52653848.
