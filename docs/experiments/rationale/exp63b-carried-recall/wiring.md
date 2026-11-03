# Exp 63b: WIRING lens (design review, 2026-11-02)

Reviewed: `docs/experiments/exp63b_carried_recall_prereg.md` DRAFT v1, against the code at `480ba23c`. Question: will the
gates read what the shipped path does, and will the harness and the evidence gate accept it? Exp 63 last ran on
52653848, so every assumption was treated as drifted and read again. I did not read the other lenses' files.

Probes ran against the real `src/` with no tracked file changed. Scripts are in
`/private/tmp/claude-501/-Users-dennyschaedig-Scripts-Maxim/342dd0e2-83af-477e-a802-1b4ef4302dee/scratchpad/wiring_probe/`:
`probe_path.py`, `probe_port.py`, `probe_replay.py`, `probe_gates.py`, `probe_r3a.py`. All numbers from old stores are
design arithmetic on old-code bytes and move nothing.

**Go/no-go: fold before building.** The design's shape holds: a new key, an additive judge, a stdlib port and an I1 of
one or two lines all work. Five gate-level defects must be folded first. Three of them make a verdict read the wrong
way (a false FAIL, a masked FAIL, a false PASS). Two of them can strand the campaign or prove nothing.

---

## DO-NOT-BUILD

### W1. R3a's `min(3, E)` falsely FAILs a conforming turn, and two R3a outcomes are undefined

G3 says the R3a trace "surfaces `min(3, E)` ids, all in C", with E = the carried records the cue or the goal scores
above zero. That count is not what the pipeline can return. `_query_hippocampus_traced` reads the cue's top 5 and only
`goal_results[:3]` of the goal path, and the own capture takes a place in each. A record ranked fourth on the goal path
is scored and unreachable.

Reproduced on the real pipeline (`probe_r3a.py`): query "where did the rusty lantern go near the sleeping guard",
carried A, B, C. A and B score on the cue, and all three score on the goal, so E = 3 and `min(3, E)` = 3. The goal
ranking is `[A, B, own, C]`, so `goal_results[:3]` holds nothing new. The pipeline surfaces `['A', 'B']`, conforming
exactly. G3 would FAIL it, and FAIL means BROKEN on the ledger.

E also ignores two things: a carried record with the same words as the turn is skipped (the narrator repeats itself; in
Exp 63 "What is your next step toward: …" was asked on several turns), and the top-5 limit applies.

Two outcomes have no rule:
- **No trace has the view.** Exp 63 made this C5-incomplete. G3 names only E = 0 (NOT SHOWN). If NOT SHOWN also absorbs
  "no view", a startup capture regression reads PARTIAL and moves the row.
- **The view's trace surfaced an id outside C** (a late arrival). Exp 63 made this C5-incomplete. G3 dropped the clause
  by replacing the paragraph, while "Every other gate stands" does not reach it.

**Fix.** R3a passes when that trace's logged sequence equals the recomputation (R3′ on that trace), is non-empty, and
lies in C. The recomputed empty sequence is the only "unobservable" case: NOT SHOWN. No trace with the view, or a logged
id outside C, or a view contaminated by a late arrival: C5(R3a), incomplete, as in Exp 63 (keeping A2: this abort applies
only while P1 and P2 hold).

### W2. The new C5(L) turns a real R3′ failure into an ABORT, so FAIL is unreachable for any defect that changes the count

G5: "each turn's `memories` equals the recomputed count, so a swallowed exception still aborts". C5 is an abort. R3′ is
the FAIL. They now compare the same number, and C5 runs first (`judge`: an incomplete attempt never reaches the gates).
So the defect R3′ exists for ("a carried record did not take part in recall as its saved bytes say it should") shows up
as a short count, and it ABORTs: three attempts, no FAIL, row left STALE. The old C5(L) did not overlap R3′, because it
compared against `hippocampus_size`.

The event C5(L) wants is one exact line: `_query_hippocampus_traced`'s outer `except` returns `[], [], horizon, holes`.

**Fix (in order of preference).** Add to I1 a third logging-only field, `recall_failed` (true on that except path, proven
by deletion), and make C5(L) read it. Note the same outer `try` covers the cue, goal and substring calls; the inner graph
`except` keeps going and does not set it. Weaker alternative with no new field: C5(L) fires only on `memories == 0` while
the recomputation surfaces some. That leaves an all-carried-lost case reading ABORT, so state it.

### W3. R3d's "visible non-carried record the ranker returned" counts the own capture, a record that never competed

G4 adds D1's zero-score exclusion but not the own-capture skip. The own capture is visible (it is in V0, see S1) and
scores above zero, and the ranker returns it. `_add_memory` then discards it. A carried id ranking above it shows
nothing, because the skip guarantees it could never have been shown.

Reproduced (`probe_r3a.py`): carried A, own capture "the sleeping guard". The trace surfaces `['A']` by `goal`, the goal
ranking is `['A', 'own']`, and as drafted A "outranks a visible non-carried record that scored above zero". That is a
decisive turn with no competition. It is most likely exactly when the claim is weakest: an early turn, on the goal path,
when almost no new record exists. It routes to PASS, so to EARNED.

**Fix.** R3d's competitor set is non-carried, scoring above zero on that path's query, and **not skipped as own words**.
Carry the same exclusion into every tie ordering. Also tighten the gloss "that the ranker returned, i.e. scores above
zero": `limit=5` means those are not the same set. Say "scores above zero" only.

### W4. The evidence gate's B0 checks are not in the harness's C1–C5, and a run that fails only them strands the campaign

`_evidence_records.py::judge_sim_fields` calls `judge_b0_fields` on every sim echoed in a row. When the executed code
writes `PROVENANCE_SCHEMA >= 2`, as `simulation/report.py` does, it requires `provenance_schema`, `concurrent_sims is
False`, `sampling_complete is True`, `model_changed_during_run is False`, `aut_model_attribution == "not_established"`,
a `scenario_sha256`, and `llm_calls` with each role's `<role>_router_serial` in it (each router's counts checked by
`judge_b0_router`). `o19_verdict.py::complete_problems` reads none of them.

- `git grep '"provenance_schema"' -- 'docs/experiments/data/*'` returns nothing. **Exp 63b would be the first committed
  sim evidence through these checks.** The draft's "nothing has exercised this harness" is true one layer further down
  than it says: nothing has exercised the gate's B0 half on a real `--sim` record either.
- The failure is unrecoverable. An attempt complete under C1–C5 decides (`judge`: "the first complete attempt decides"),
  `prepare` refuses any further attempt ("the rows file already holds a complete attempt"), the verdict may read PASS,
  and the gate then refuses it. The row cannot move, and there is no attempt left.

**Fix.** Add a C4″ to `complete_problems` for `63b` mirroring these checks on each phase's report, so a run that the
gate would refuse is an ABORT that may be retried. Do not import the gate into the judge: restate the rule and pin the two equal by a test,
as `SUBJECT_PATHS` is pinned. The smoke (W5) must also push its row through `judge_sim_fields` for real.

### W5. The Exp 63 mock cannot be extended, and there is no smoke tool for build step 3

`tests/unit/test_o19_rerun.py -k "exp63 or 63"`: 64 passed on this tree. That says the judge agrees with the mock.
`o19_rerun.py::mock63_session` no longer models the path, and it never tested the ranker:

| The mock | The shipped path |
|---|---|
| ids chosen by `v.rank_by_relevance`, **the judge's own frozen ranker**, then re-judged by it: circular | ids chosen by `_rank_by_relevance` over `content_tokens` |
| captures land AFTER each trace | the own capture lands BEFORE the trace, every time (S1) |
| one path, `goal` | graph, `cue`, `goal`, `substring`, and cue fills first |
| no own capture, no skip | skip on every path |
| text in `observations["text"]`, which no capture writes (`text_fields`' docstring says so) | `cli_input`, `transcript`, `observations.cli_input` |
| store `_format_version` "1.0" | "1.2" |

A `63b` mock on this pattern would pass G1 by construction while the real path differs. That is the failure in
`docs/wiring/harness-loop-must-be-proven-live.md` ("a fake that emits what the live world does not hides the defect it
should expose") and in reproduction.md §8.

Step 3's "one real short sim" has no mechanism. `o19_rerun.py` has `run` and `preflight`; the real `run` pushes a marker
and writes rows. M26 (the smoke mode) is unbuilt. "Short" would not work: `EXP63_TURNS = 8` is fixed, so R1 fails a short
run, and §8 says a smoke uses the protocol's own settings.

**Fix, two parts.**
1. **The `63b` mock runs the real seams, with only the LLM stubbed.** In process: real `Hippocampus`, `MemoryAgent`,
   `PerceptionAgent`, `take_user_input`, `BioEnrichmentPipeline.enrich`, `format_thought_response`, `Hippocampus.save`,
   and a resume through `Hippocampus.load`, emitting the trace through `StructuredFormatter`. `probe_path.py` shows this
   is about forty lines. The shipped ranker then chooses the ids and the judge's port recomputes them, which is the test
   G1 and G2 need. A deletion test (remove the skip, or move a capture after the trace) must turn it red.
2. **Build M26 as `o19_rerun.py smoke --exp 63b`.** Protocol argv, a throwaway home, no marker, no rows under
   `docs/experiments/data/`, both phases at 8 turns (415 s for phase 2 in the Exp 63 run). State in the prereg: one smoke
   per candidate commit, it is not an attempt, and it computes C1–C4″, C5 and R3′ as instrument checks. It does **not**
   compute or print R3a or R3d, so it cannot serve as a free look at the outcome. Whether R3′ on a smoke is acceptable is
   the confounding lens's call; I recommend yes, because it is the only end-to-end check of the port on real records.

**What the smoke must show before a marker is pushed:**
- `select_arc_for_goal(goal)` is None (read today: None), and the report's route is the one Exp 63 took.
- C1–C4 and C4″ hold; phase 2's `resume.stores.hippocampus == "loaded"`.
- `exp63b_c5_problems == []`, including the identity checks in S8.
- Every turn has a percept trace with `cue_text` and `own_words`, and its own capture is in V0.
- Phase 2's first trace reads `hippocampus_size == N1 + 1` and horizon `seq_C + 1`.
- R3′ conforms on all 8 turns.
- N1 and records per turn are of the post-#1414 order (S4). If they read 84 and 272, the re-capture is back.
- The row passes `judge_sim_fields`, and `scripts/lint_secrets.py` is clean on the smoke's run log and stores.
- Then `o19_rerun.py preflight --exp 63b` on the merged commit.

---

## SHOULD-FIX

### S1. G3's mechanism is misdescribed: it is not a race, and the fix should say so and check it

The draft: "the first trace races its own capture: if the capture lands first…". Traced, it is synchronous on the loop
thread: `agent_loop.py` §1.2 → `loop_perception.py::take_user_input` → `MaximAgent.publish_arrival` →
`PerceptionAgent.publish_arrival` → `AgentBus.publish` (a synchronous fan-out) → `MemoryAgent._on_percept` →
`_begin_memory_formation` → `_capture_to_hippocampus` → `Hippocampus.capture_from_loop` → `capture`, where
`_stamp_encoding_strength` takes `capture_seq` at insert. `enrich` runs after that returns.

`probe_path.py` confirms it: `store 3->4 on take_user_input (sync)`, then the trace reads `horizon=3` with `next_seq`
having been 3. The own capture is in V0 on **every** addressed turn. So Exp 63's view is unavailable on 100% of attempts,
not on the runs where a race goes one way. The draft's conclusion stands; "A reserved number above the horizon is not a
hole" is also right (`stored_capture_seqs`). The async path (`capture_from_loop_async`, reserving at enqueue) is the
ACTION capture, and it is what produces holes on later turns: a sync own capture jumps past an in-flight number.

This matters because a capture that lands first by construction can be asserted, and a race cannot. **Fix:** reword, and
add the check in S8(b).

### S2. G2's port list omits what the recomputation needs, and its pin over committed records cannot pass

The port can be exact. `probe_port.py` wrote one from the list below: over 440 committed records rebuilt to 1.2 shape,
0 mismatches against the shipped `content_tokens`; over 4,000 fuzzed values with frames, cut frames, deep nesting and
metadata keys, 0 mismatches. It needs more than G2 lists:

- **Frames (missing).** `text_values` calls `outside_tool_output` and `drop_unbalanced_frames` when a string holds
  `TOOL_OUTPUT`. So the port needs `_FRAMED_REGION`, `_FRAME_MARKER`, `_unbalanced_markers`, `_partial_marker_start` and
  `drop_unbalanced_frames`' cut rules, from `utils/content_safety.py`. An action record's result is exactly where a frame
  would sit, so leaving this out is a false R3′ FAIL.
- **`input_text` and `_same_words` (missing).** G1 and G3 need them. `EpisodicMemory.input_text` is `cli_input`, else
  `transcript`, else `observations.{cli_input, transcript, text}`, each only if a non-blank `str`. `_same_words` is NFKC,
  casefold, whitespace collapse.
- **`_METADATA_KEYS`** (including `bio_response`, the echo guard) and `depth=3` restarting **per top-level value**.
- **Tuples.** Live `text_values` walks tuples; JSON has only lists. No port change, just know it.
- **The field list.** `content_tokens` is `text_fields()` (`cli_input`, `transcript`, `observations`,
  `decision_rationale`, `decision.reasoning`, `outcome.result`, `outcome.error`) plus `context.active_goal`,
  `action.tool_name`, `detected_objects`, `detected_people`. `CompressedMemory` has its own list; C5(d) keeps refusing
  compressed records, so the port can refuse them too.
- **The pin.** "Pinned behaviourally to the shipped code by a test over real committed records" cannot pass as written:
  21 of the 440 committed records hold a legacy `ToolOutput(...)` repr, which `Outcome.from_dict::_loaded_result` parses
  and the saved dict does not, so the port differs on exactly those 21. **Fix:** pin over records put through one 1.2
  round trip plus a fixture from the smoke's store, the way `test_the_frozen_ranker_is_the_shipped_ranker` rebuilds
  through `from_dict`, and add a frame fuzz. Have C5(d) refuse any `outcome.result` string starting `ToolOutput(`.
- **The pin's target.** G2's "the shipped code" should be the executed commit's, vendored and byte-checked, as
  `tests/fixtures/exp63/executed_ranker.py` is, or the first `memory/text.py` change breaks a pin the committed verdict
  rests on. Vendor `content_tokens`' closure the same way.

### S3. One class of field cannot be reproduced from JSON bytes, and it would read as a false R3′ FAIL

`Hippocampus.save` calls `atomic_write_json(path, payload)`, whose default is `default=str`. A non-JSON value inside a
walked field is ignored by the live `text_values` (it is not a `str`, `dict`, `list` or `tuple`) and is a string in the
saved bytes. Reproduced: an observation `{'tags': {'alpha', 'beta'}}` saves as `"{'alpha', 'beta'}"`, and the reloaded
record's tokens gain `alpha` and `beta` (`probe_path.py`). Such a record is scored differently live than the judge scores
it, and nothing in the saved bytes marks it. Exp 63's C5(d) `text` check covered one key of this.

Where it can and cannot occur, read from the capture sites:
- **Safe.** `Outcome.result` from a tool: `_stored_result` turns a `ToolOutput` into plain `_json_native` data at
  `__post_init__`, so live and saved agree (`{'x'}` became `['x']` in the probe). `MemoryAgent` percept observations and
  the sim adapter's observation dict hold only strings, `None` and numbers.
- **Plain today, unguarded.** Dict results bypass `_stored_result`, but the two I read hold strings and numbers
  (`tool_pain_bridge.py`'s `{"reflection": …}`, `cerebellum.py`'s `{"valence", "pain_step"}`).
- **Not closed.** `pain_bus.py`'s `**signal.context` spreads an arbitrary mapping into `observations`. And a `result`
  string starting `ToolOutput(` parses on reload, so a carried record's live tokens differ from its saved ones.

Exp 63's two runs had no pain events (`pain_events_count` 0 in both reports), but a `FAIL` is terminal and reads BROKEN, so
it should not rest on that.

**Fix.** (a) A parametrized unit test, one case per capture site with a realistic payload, asserting
`live.content_tokens() == from_dict(json_round_trip(to_dict())).content_tokens()`. (b) When a turn fails R3′, the judge
prints the first differing record and its token diff, so the investigation starts from the instrument. (c) A record
whose live tokens differ from its saved tokens is an instrument fault, not the claim: the prereg should say so, and if
the diagnostic shows it, the attempt reads incomplete. A saved string cannot prove it was stringified, so a practical
C5(d) form is a narrow shape test on walked strings (`^\{'`, `^frozenset\(`, `^<[A-Za-z.]+ object at`), named as a
heuristic. Do not widen it past that.

### S4. The replay table describes a store that no longer exists

"What changes" is computed over Exp 63's phase-2 store: 356 records. 341 of them are `memory_agent` percept captures of
**12 distinct utterances** ("What is your next step toward: …" 134 times, "Which direction should I turn…" 82 times).
That was the re-capture per tick that #1414 fixed: `take_user_input` now clears `state.data["cli_input"]`, and
`propose_intent` no longer re-publishes it. So N1 will not be 84 and turn 8 will not see 356.

`probe_replay.py` collapses each run of identical words to one arrival and runs the real pipeline, own capture present
(design arithmetic only):

| Turn | new visible | paths | carried shown |
|---|---|---|---|
| 1 | 0 | cue, cue, cue | 3 |
| 2 | 2 | cue, cue, cue | 3 |
| 3 | 4 | cue, cue, cue | 2 |
| 4 | 6 | cue, cue, **goal** | 1 |
| 5 | 8 | cue, cue, cue | 0 |
| 6 | 10 | cue, cue, cue | 1 |
| 7 | 12 | cue, cue, cue | 0 |
| 8 | 14 | cue, cue, cue | 0 |

N1 = 14 and 14 new. Turns 2, 3, 4 and 6 are decisive on W3's corrected competitor set, using the lowest scoring
competitor and ignoring ties. So the power claim survives in direction with different turns, while G4's "on every
replayed turn it holds at least 3 records, so the goal path never runs" does not (turn 4). G4's reason still holds; the
goal path is simply no longer dead.

**Fix.** Redo the table on the collapsed store with the own-capture skip applied, state that this approximates #1414's
effect on old-code bytes, and drop "own-capture skip not applied". Say power is unknown until the first attempt. The
same size change makes D1's under-3 case and so the substring path ordinary (S9).

### S5. G1's multi-path tie semantics: use one simulator, and state the empty-`own_words` case

G1's formula is right, and I checked it against `_query_hippocampus_traced` line by line:
- Cue reads up to 5 and breaks at 3 surfaced; goal runs only when under 3 and reads `goal_results[:3]` with no refill.
- A skipped own capture IS added to `seen_ids` and consumes a slot in the cue's 5 and in the goal's 3, so the goal path
  can return fewer than 3 while more goal-scoring records exist. A record on two paths takes the first and does not
  count twice. All of this is what `dedup(...)[:3]` plus a skip gives, provided the skip happens during the dedup and
  the top 5 and 3 are cut **before** it. Say that order explicitly.
- A graph id skipped as own is not logged, but it would be skipped on the cue too, so no information is lost.
- Late arrivals: `recall` is called twice, so the two rankings are separate calls over one pool V0 ∪ L. A late arrival
  with the turn's own words would take a slot and leave no trace. It needs identical words landing inside a query
  window; state it as a known limit rather than building for it.

Gaps:
- **`own_words == ""`.** The code's guard is `if own_words and …`. Without it the judge skips every record with empty
  `input_text`. G1 must state it; action records have no words, so this is not rare in the port.
- **Multi-path ties.** `goal_ids_conform`, `shown_goal_ids` and `exp63_decisive` are one-path reasoning, and extending
  them by hand for two paths and a skip is where a subtle judge bug would land. **Fix:** a small `simulate(cue_order,
  goal_order, graph, own)`, a line-for-line copy of the loop, plus an enumerator of tie linearisations of the two top
  lists. R3′ is `any(simulate(...) == logged)`; R3d is `all(decisive(simulate(...)))`, so one mechanism decides both. Tie
  groups are small: exact `(score, timestamp)` equality. Past a stated bound on the product, abort as C5 rather than
  guess. Pin `simulate` against the real `_query_hippocampus_traced`.

### S6. Everything a new key needs: the additive claim holds, but only with every site done

**Does "additive, `63` byte-identical in behaviour" hold? Yes, if `63b` is a new experiment id and new functions, and
shared helpers are not edited.** Confirmed: `judge("63", …)` on the current judge gives PASS with `gates` equal to the
committed verdict and no `within_campaign_leaks` key (`probe_gates.py`). Half B (`o19_judge_edit_problems`) re-judges with
HEAD, so an edit to `EXP63_PATHS`, `exp63_c5_problems`, `ranker_tokens`, `visible_ids` or `turn_traces` fails "a judge
edit changes an existing O19 verdict". `EXP63_PATHS` must not gain `"cue"`; make new constants. History check
(`o19_history_problems`) is unaffected as long as **`O19_INTERFACE` is not extended**.

**`"experiment": "63"` is refused:** `protocol_problems` gives `experiment 63 has 2 open campaigns`, and a kind must have
one experiment. So `"experiment": "63b"`, not a `supersedes`. `_KEY`, `_SCOPE`, `lint_prereg`'s token and the tag name all
accept `63b`.

**The trap: adding only the table entry fails quietly at three sites.** With the entry, `EXPERIMENTS` and `LEAK_GATES`
added, `protocol_problems()` returns `[]`, and then:
- `required_files("63b")` returns no `aut_hippocampus.json`, so C2 is not required.
- `judge` skips C5 (its `== "63"` guard), then its `else` calls **`exp09_gates`**.
- `_gate_passes` falls through to **`exp63_gates`**, the old judge.

So:

| Site | Change |
|---|---|
| `o19_verdict.py::PROTOCOL` | new entry: `experiment`, `scope "63b_carried_recall"`, `kind "exp63b_verdict"`, `prereg`, the same `phases` |
| `::EXPERIMENTS` | add `"63b"`. Keep the literal `EXPERIMENTS = frozenset(`: `test_lint_evidence_gate.py` edits the judge by that anchor |
| `::LEAK_GATES` | add `"63b"`, phase counts 1 and 2 |
| `::PRE_1079_KEYS` | **do not touch**, and keep the line byte-identical (`PRE_1079` in the same test). `63b` gets the within-campaign bar by not being listed |
| `::required_files`, `::complete_problems` (C4′ and the new C4″), `::judge` (C5 and the gate dispatch, two sites), `::_gate_passes` | dispatch on `63b`, ideally through one `is_carried_recall(exp)` so a future id cannot fall through |
| `::protocol_problems` | the message "is not 10, 09 or 63" |
| `o19_rerun.py::mock_phase`, `::prepare` (`check_memory_strategy`), `::_run_phases` (`exp63_attempt_problems`, itself calling `exp63_c5_problems`) | dispatch on `63b`; `EXP63_TOP`, `EXP63_GOAL_LIMIT` may be shared |
| `_evidence_records.py::O19_KINDS` | add `exp63b_verdict` (`test_lint_evidence_gate.py` asserts every campaign kind is in it) |
| `docs/experiments/evidence_pass_table.json` | see S7. Read from the merge-base, so it must land before the ledger PR; the build order has that right |
| `tests/unit/test_o19_rerun.py` | `sorted(v.PROTOCOL)` already parametrizes argv parsing; `test_goals_caps_and_scope_are_the_preregs` excludes only `experiment_of(k) == "63"`, so `63b` falls into the Exp 10/09 prereg-text loop unless it gets its own `test_exp63_protocol_*`-style case |

Clean: `MAX_CAMPAIGNS` (63b is its own root, one campaign), `o19_table_problems` (D1 append-only, new key on a table whose
`63` is frozen), `_o19_campaign_count_problems`, `bound_judge` (`bound_files` is exactly judge, harness, prereg),
`lint_ledger_format`, `check_model_config` and every symbol it imports (all import today), `check_argv_parses` (both
phases parse), C4 on the current report (`_endpoint_origin` reduces both sides, so #997's origin-only endpoint compares
equal).

### S7. The verdict table is not yet what the gate reads

`"targets": {"EARNED": ["PASS"], "PARTIAL": ["NOT SHOWN"]}` works: `PARTIAL` is in `L.RANK`, values are free strings, and
for a root campaign `o19_succession_problems` lets any non-REPRODUCED token through. T1-16 under two kinds is fine
(T1-13 has two). Do not list `REPRODUCED` for a root: it is unreachable. But:

- **Scope.** `support_problem` matches the row's head qualifier exactly. With `"scopes": ["narrow"]`, the draft's "EARNED
  at the run date" (no qualifier) is refused; with `[null]`, the draft's "PARTIAL (narrow…)" is. Pick one. Exp 63 and the
  claim's own scope say `narrow`: `"scopes": ["narrow"]`, and write `EARNED <date> (narrow: …)`. README and the index must
  show the same word (`lint_claims_sync` R3). The fold is one row of the table.
- **R3a needs three states.** `exp63_gates` keys off `out["R3a"]["pass"]`. Under the draft, R3a unobservable is NOT SHOWN
  and R3a false is FAIL. Add a `status`, keep `gating` free of an unobservable R3a, and decide what `_gate_passes` reports
  for the leaked-gate bar. I recommend `pass: False` for unobservable: LEAK_GATES is fail-closed by design.
- **R3d ordering.** "After R3a's turn" has no meaning when R3a is unobservable. NOT SHOWN then applies and R3d is not
  read; say so.

### S8. I1: where it goes, and the identity checks it needs

**Smallest edit.** `enrich` (145 lines) and `_query_hippocampus_traced` (146) are not pinned in
`function_length_baseline.json`, so Rule T does not fire. `_run_agentic_loop_body` is pinned at 2164 and I1 does not touch
it. The trace dict is in `enrich`.
- `cue_text`: one line, `"cue_text": text`. Keep `query_text`.
- `own_words`: **return it** as a fifth element, not re-derived in `enrich`. `_same_words(self._turn_input)` after the
  call would usually match, but a diagnostic must report the value the code compared, not a second computation of it,
  and `_turn_input` is shared with the think tool. Callers: `enrich`; `_query_hippocampus` takes `[0]`; the two test call
  sites unpack with `*_`; three return statements. `test_interoceptive_outcome.py` reads these functions' source for
  `extra`, `interoception` and `.encoding`, so avoid those tokens (an `extra=` belongs in `enrich`, not there).

**Behaviour.** Logging only: no branch reads either field, `to_compact` only serializes, and `enrichment_trace` is a plain
logger event. It never reaches `sim_log`, the abstraction buffer's `log_agentic` or a prompt. Empty `""` survives
`to_compact`'s None filter.

**Privacy.** No rule forbids it. #997 is endpoints and argv. F18 forbids framed text in the sim activity stream, which
this is not. The same words are already in `aut_hippocampus.json`, so a sim commits nothing new. `lint_secrets` and
`copy_session`'s host check run over the run log as they do over the store. One change is real: on every runtime with
`MAXIM_LOG_FILE`, an input's logged length goes from 120 characters to all of it. Say that in the changelog fragment. Do
not bound `cue_text`: a cut query keys differently, which breaks G1. Do not gate it behind an env var: that would be a
subject difference and owe a conftest scrub.

**Each logged field needs an identity check** (the rule from `feedback_diagnostic_fields_need_an_identity_check`). Add to
C5(b), for the percept trace of each turn:
- (a) `isinstance(cue_text, str)`, `query_text == cue_text[:120]`, `keywords[:8]` reproduced by the port's `key_tokens`,
  and `own_words == same_words(cue_text)`.
- (b) A record with `same_words(input_text) == own_words` is in V0 (S1): this asserts capture before trace, so a #1414
  regression or a moved publish aborts.

**"A turn's trace" should use these.** `§1.2` now emits one trace per narrator input whether or not the PFC gate passes
(D5; Exp 63's 8-per-phase was gate-dependent). Cycle 2+ (`_run_deliberation_cycles`, `user_input` empty, inheriting
`_turn_input`) and `exec_agent.py`'s enrich add more, goal-bearing, and the think tool adds empty-goal ones.
`turn_traces`' "first goal-bearing AUT trace" still lands on §1.2 in the normal case, and `Bridge.send_and_wait ENTER
turn=N` (`simulation/bridge.py`) still fires before `inject_cli`. But a slow deliberation from turn N can log after ENTER
N+1. **Fix:** for `63b`, the turn's trace is the first goal-bearing AUT trace in the window with `own_words ==
same_words(cue_text)`. G1 recomputes correctly from any trace, since each carries its own inputs, but R3a and the
identity checks need the right one.

### S9. C5(s) and G1's substring term no longer match the store they will see

G5 keeps "a goal-bearing substring id makes the attempt incomplete, as an instrument limit". Fine, and the path is now
reachable: on the collapsed store, anything under 3 after the goal path runs `search_by_content(text)`. G1's formula
still includes "substring ids", which R3′ will never see. Two fixes: drop that term from G1, and scope C5(s) to the turns'
traces. As written (`exp63_c5_problems`' `shaped`), it reads every goal-bearing trace, so a deliberation trace with a
substring hit would burn an attempt on a trace no gate reads. C5(b) must also accept the new `cue` path on all traces.

### S10. C5(c): holds, with two caveats to state

Holds through D5: the loop calls `format_thought_response` whenever `enrich` returned a result, `evaluate_reflexes` only
gates reflexes, and all four callers render (`agent_loop.py` twice, `exec_agent.py`, `tools/narrative.py`).
`_activate_rendered` is the only `source="enrichment"` writer for the Hippocampus. Exp 63's parenthesis said "Every enrich
caller renders", and there is now a fourth caller to name.

Caveat: `_activate_rendered` runs only past `if not lines: return ""`, and a surfaced memory always makes a line, so this
is safe. And an exception between the trace and the render (`_evaluate_reflexes`, `_log_enrichment_contributions`) leaves
a logged id unrendered, which C5(c) reads as an instrument fault. That is the right reading.

---

## NIT

- `o19_verdict.py`'s module docstring and exit-code comment, and `experiment_of`'s docstring, name "10", "09", "63".
- Keep a `check_bound` freeze in mind: no PR may touch the two O19 scripts or the prereg between the first marker and the
  verdict, or the verdict refuses. With parallel sessions active, say so on the PR.
- `docs/experiments/README.md`: a new marked row for 63b must show the same token and date as the T1-16 row it shares
  (`lint_claims_sync` R2–R3). Re-pointing the existing row is simpler.
- `lint_prereg_precedes_data`: `63b` is governed by both preregs (`parent_chain`). Exp 63's PRE-DATA amendment is older,
  so it passes, but any new unscoped amendment on Exp 63's prereg would be judged against 63b's data.
- Any amendment to this prereg after it lands needs the `**Amendment N — <date>, PRE-DATA, …**` header, not "DRAFT v2".
- `_same_words` uses Unicode tables, so the rig's and the gate's Python could differ on unassigned code points. Immaterial
  for this text; one line in the prereg's instrument notes covers it.
- C5(a) needs a `sim_deliberation` line naming `sim_aut`. Exp 63 had 27. `sim_pre_deliberation` logs as `THOUGHT`, so an
  all-gate-closed run would have none and abort. Unlikely, since every Exp 63 turn passed; accept `sim_thought` too.
- G4's "Note the asymmetry" holds: recency sorts new first, so a carried record needs a strictly higher score. Fine.

---

## Verified

**The sim path.** Order on a phase-2 turn: `simulation/bridge.py::send_and_wait` logs ENTER, then `inject_cli`; the loop
gets `{source: "cli", cli_input, ...}` from `sim_adapter.py::next_observation`; `take_user_input` publishes; capture is
synchronous; `percept_text` gives `text`; `enrich` runs with `user_input=_addressed`; the trace logs; the result renders.
`context.user_input` on §1.2 is the stripped addressed text. On a cycle 2+, `EnrichmentContext(active_goal=…)` has no
`user_input`, so the skip inherits `self._turn_input` as G1 needs, and `exec_agent.py` and `tools/narrative.py` inherit
the same way. `[ACTION_FOLLOWUP` and a loop reply give `addressed == ""`, so they inherit too.

**`stored_capture_seqs`** behaves as G3 says. `next_capture_seq` resumes past the saved maximum, and the floor moves at
load (`next seq 3 floor 3` after a 3-record store). Holes are below the horizon only.

**P3 against the current record.** `probe_path.py`: capture, save (format 1.2), load in a fresh `Hippocampus`, enrich,
render, repeated `recall` to force promotion, save. Fields moved on carried records: exactly `access_contexts`,
`access_count`, `accessed_at`, `activation_count`, `activation_sources`, `consolidated_at`, `last_scored_at`, `long_term`,
`promotion_pressure`. All inside `EXP63_MUTABLE_FIELDS`, nothing left over. `situation`, `encoding`, `metadata`,
`retro_tag`, `storage_strength` and `retrievability_anchor_us` did not move. `_promote_to_long_term` touches only those
fields. `retro_tag` moves only in `_resolve_retro_tags_locked`, which runs at sleep. No in-session write to any field
`content_tokens` reads turned up in `memory/`, `memory_agent.py`, `bio_integration.py` or `integration/`
(`observations["_skill_name"]` is on the staged copy), so the identity-keyed token cache cannot go stale against the saved
bytes.

**"No `sleep()` runs" still holds on `--sim`.** `bio_integration.py` takes `on_session_end_lightweight` in sim mode; only
the full end consolidates (`session_end.py::_consolidate`). `aut_hippocampus.json` comes from `report.py` via
`hippocampus.save`.

**`store_records`** reads a 1.2 store: `memories` is still a list of dicts by `id` (`_dump_locked`), and `_compressed` is
still the marker.

**C5(a).** `bio_stack.py` wires `trace_agent_id=agent_id`, and `SIM_AUT_AGENT_ID` is `"sim_aut"`. **C5(d).** Compression
needs consolidation; `capture_seq` is stamped on every `capture`, since every capture site goes through it.

**C4 and C4′.** Report fields are all still written (`report.py`); `resolve_memory_strategy()` reads `access_based`;
`select_arc_for_goal(goal)` is None; `HARNESS_ENV`'s `MAXIM_LOG_FILE_MAX_BYTES` is still read (`utils/logging.py`).
`read_served` after #1314: `candidate_server_keys` reads the host's key files, not `MAXIM_DATA_HOME`, so a fresh home does
not hide them.

**Markers.** Ruleset 24345734 (`o19-markers`, read with `gh api`): tag target, `include: ["refs/tags/o19/**"]`, active,
`deletion` and `update` rules, no bypass actors, `current_user_can_bypass: never`, created and updated 2026-10-01, one
history version. It covers `refs/tags/o19/63b/` and predates any first marker. `ruleset_problems` will accept it.
`ls-remote` shows no `63b` tag.

**Prereg lint.** `token_of` gives `63b` for both the file and the data directory, `parent_chain` gives `["63"]`, and
`rerun_by_name` is False, so no Scope line is needed.

**The draft's route.** `successor_problems` (only an ABORT may be succeeded) and `o19_table_problems` ("a re-run is a new
experiment id, owner D4") say what the draft says.
