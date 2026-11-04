# Exp 63b: carried memory takes part in recall on the content-token ranker and the cue path (pre-registered 2026-11-02)

> **Status: frozen. Design review folded ([confounding](rationale/exp63b-carried-recall/confounding.md),
> [bio-faithful](rationale/exp63b-carried-recall/bio-faithful.md), [wiring](rationale/exp63b-carried-recall/wiring.md));
> owner decisions OD1 and OD2 taken 2026-11-02; re-read against #1573's code on `main`. No
> harness code and no data exist.** It re-tests ledger row T1-16
> after 1.3.4's recall fix fired the row's re-run trigger
> ([#1476](https://github.com/dennys246/Maxim/issues/1476)). The row has been STALE since 2026-10-25 and blocks the
> 1.3.4 cut; the owner declined a release exception on 2026-11-02, so the result is reported whatever it reads. Base
> design: [exp63_carried_recall_prereg.md](exp63_carried_recall_prereg.md) ("Exp 63" below). Everything Exp 63 states
> stands here unless this file changes it.

## Why a new experiment and not a new attempt

`"63"` closed with a PASS. Only an ABORT may be succeeded (`o19_verdict.py::successor_problems`), a campaign table
entry with data is frozen (`_evidence_records.py::o19_table_problems`: "a re-run is a new experiment id, owner D4"), and
Exp 63 names the route itself: "a re-run as a new experiment on the fixed ranker". The subject code changed, so a
successor could not support the row anyway (reproduction.md §13).

| | |
|---|---|
| Campaign key / experiment | `63b` |
| Verdict kind | `exp63b_verdict` (new; `O19_KINDS` and `evidence_pass_table.json` gain it) |
| Data directory | `docs/experiments/data/63b_carried_recall/` |
| Markers | `refs/tags/o19/63b/` (ruleset 24345734 covers them) |
| Judge | additive: `"63"`'s gates, ranker and table entry keep their behaviour, so the committed 52653848 verdict still reproduces (gate half B). Every `experiment_of` site is handled; wiring S6 lists them |

## Claim

Suggested wording (bio-faithful S4):

> After a resume, the earlier session's Hippocampus store is back with its content unchanged. Text arriving on the
> addressed path (here a simulation narrator's, not a person's) ranks carried and new memories alike: by how many of its
> content words each holds, then by recency, with no reading of a memory's origin; an episode stored many times takes one place, the newer copy on a tie. On the first resume turn what surfaces is carried. On at least one
> later turn a carried memory takes a place on the cue path, displacing a new memory, because it holds more of the
> cue's words outside the goal. This is one 8-turn resume on one model.

- **Mechanism:** Hippocampus persistence + `BioEnrichmentPipeline._query_hippocampus_traced` (graph, inert in production;
  cue; goal; substring) over `_rank_by_relevance` (`key_tokens` over `content_tokens`; a zero score is excluded, D1).
- **Not claimed:** that the agent remembers what a person told it; that it uses what surfaced (surfaced is not rendered,
  and rendered is not acted on: T1-2); that a cue selects the RIGHT carried memory (a planted-cue experiment is a new
  claim and owes all four lenses); recall across different goals; survival through consolidation; "RECALL
  reconsolidation". R3d is blind to origin and is a non-vacuity condition on R3′, not a test of persistence.

## Protocol and apparatus

Exp 63's: the same two phases, goal `escape a dungeon with a sleeping guard`, 8 turns each, `--interactive false`,
`--sim-max-turns 8` and `--sim-run-full-turns`, phase 2 resuming phase 1, a fresh data home;
`mistral-7b-instruct-v0.2.Q4_K_M.gguf` (`llm.profile mistral-7b`) at `llm.n_ctx 8192`, set through `maxim config`; big-mac-mini, quiet (no second LLM consumer, no suite, no Minecraft stack); the rig stays at the first attempt's
commit. The argv is unchanged, but the subject is not: #1414, `publish_arrival`, #1260/#1261, the stall template and
`content_tokens`' scope all moved since 52653848. They do not confound the within-run gates, and they rule out any
forecast from Exp 63's numbers. **The narrator is a stand-in stimulus:** it takes the same path as a person's input, but
it holds the goal, mostly asks questions and repeats a stall-template line.

## OD1: the own-capture skip (owner decision 2026-11-02: option A, fix first)

`_query_hippocampus_traced` skipped any record holding the turn's exact words, from any session, after `recall(limit=5)`
has cut the list. Measured on the shipped pipeline (`tests/unit/test_recall_memory_seams_1444.py`'s helpers): an earlier
identical ask surfaces 0 records, and a fact followed by the question stored 5 or more times is lost (it survives at 4).
So "full candidates" was false for a memory holding the turn's words, and repeated questions could starve the cue on the
1.3.4 goal path itself. Widening the limit only moves the threshold; keying the skip on capture identity alone lets
identical earlier asks fill every slot. That behaviour was a brief invariant; the brief now states #1573's.

**Decided: option A (strict).** The root cause is fixed under its own issue and review, and this experiment runs on the
fixed code: a later fix would fire this row's own `Re-run on:`, so running first would spend the rig twice. Not taken:
option B, running on shipped code with a narrowed claim. The fix ([#1573](https://github.com/dennys246/Maxim/issues/1573)):
the turn's own records are excluded by identity before every path's limit, and an episode stored many times (the same
words, tool and outcome, `copy_key`) takes one place. Stated limits: records of one line that are not copies each hold a
place, and a tool whose output differs on every call (`scene_summary`) never makes a copy (#1575). The gates below read this.

## What changes from Exp 63, and why

Replay of the ranker with the pre-#1573 skip applied over Exp 63's COMMITTED phase-2 store and traces. Design use only:
old-code bytes, and a pre-#1414 store (356 records for 12 distinct heard lines). **No power statement rests on this
table:** a shipped store should be far smaller, and the real volume is unknown until the smoke.

| Turn | new visible | same-words records (old skip) | shown (path, carried or new) |
|---|---|---|---|
| 1 | 0 | 0 | cue: 3 carried |
| 2 | 15 | 13 | goal: 3 carried |
| 3 | 63 | 0 | cue: 3 new |
| 4 | 84 | 60 | goal: 3 new |
| 5 | 159 | 0 | cue: 3 new |
| 6 | 169 | 0 | cue: 3 new |
| 7 | 189 | 0 | cue: 3 new |
| 8 | 272 | 9 | goal: 3 new |

Exp 63's goal-only R3d no longer describes what surfaces, because the cue path runs first. That, and not any
impossibility, is why it changes.

### I1 (instrumentation, logging only, each proven by deletion)

`enrichment_trace` already carries `own_ids` (#1573: the turn's own records, excluded on every path). It gains `cue_text`
(the full `text` queried) and `recall_failed` (true when `_query_hippocampus_traced`'s outer `except` fired). Two new log lines carry one digest, SHA-256 over the sorted `(id, sorted(content_tokens()), repr(copy_key()))` of live records: after the resume restore (every loaded record), and in each phase as the store is saved (every record that save writes). The turn's trace is the first goal-bearing AUT trace in its window whose non-empty `own_ids[0]` differs from the previous turn's trace's (the id is inherited by later traces of a turn).

### G1 (R3′): one simulator

A line-for-line stdlib `simulate`, over the pool V0 ∪ L less the trace's `own_ids`. On each of the cue and goal paths the
ranking is collapsed to the first of each set of copies (`distinct_copies`) BEFORE the cut: cue top 5, stopping at 3
surfaced; goal top 3, run only under 3, with no refill. Ids dedup across paths, and copies do not. Logged paths read graph,
cue, goal. R3′ holds when SOME linearisation of the exact (score, timestamp) ties reproduces the log. R3′ and R3d share
this one mechanism; a tie product past 10,000 orderings is C5. A late id logged on the goal path is not in the cue's pool. `simulate` is pinned against the real function.

### G2 (the frozen port)

Stdlib copies of `key_tokens` (with `normalize_tokens`, `_lemmatize`, `settle_key`, stop words, punctuation),
`text_values` (depth, `_METADATA_KEYS`, frame removal), `text_fields`, `content_tokens`, `input_text`'s fallback order,
`same_words`, `copy_key` (the `repr` of tool, success, result and error, both record kinds), `distinct_copies` and
`_rank_by_relevance` over saved-record dicts. A design probe found 0 mismatches over 440 records rebuilt
to the 1.2 shape and 4,000 fuzzed values. The pin runs against the shipped code over hand-built records (results, frames,
metadata keys) and the smoke's real store, and once the run lands the pin moves to a vendored copy of the executed
closure, as `tests/fixtures/exp63/executed_ranker.py` does. `"63"` keeps its own frozen copy.

### R2′ (new gate): every carried record is a full candidate

The resume digest (I1) equals the digest the judge recomputes from phase 1's saved bytes; a mismatch FAILs; a missing digest line is C5(b). R3′ alone
misses a partial field loss and says nothing on turns where no carried record reaches the top: this is the part of "full
candidates" that does not depend on wording.

### G3 (R3a)

Since Session R the own capture lands synchronously, on the loop thread, before enrich, on every addressed turn, so Exp
63's view (horizon `seq_C`, nothing new) never exists. The view: the first turn's trace, in turn order, with empty holes whose visible non-carried records are all in its `own_ids`. Four outcomes:
- **Holds:** R3′ conforms on that trace, it surfaced at least one id, and every id is in C.
- **Incomplete (C5), while P1 and P2 hold:** no trace has the view, or a surfaced id outside C is a late arrival outside
  V0.
- **Fails:** a surfaced non-late id outside C (the exclusion failing).
- If the recomputation surfaces nothing on that trace, R3a reads unobservable, and the verdict table decides.

Given R3′ on that trace, R3a adds only that the own capture did not surface; it is not a second recall test.

### G4 (R3d, a non-vacuity condition)

A turn after R3a's is decisive when, in EVERY valid tie ordering:
1. a carried id is shown **by the cue path**;
2. recomputing the turn with all carried ids removed shows, **by the cue path**, a non-carried id that was not shown;
3. the carried id scores strictly higher than that displaced id on the cue's keys outside `key_tokens(goal)`.

A turn whose cue has no key outside the goal cannot be decisive. A trace's `own_ids` are never competitors. A carried
graph id does not count, and a goal-path win is reported as `goal_fill_win`, never counted. R3d credits a shared word, not
a shared meaning. "After R3a's turn" needs R3a's turn, so an unobservable R3a means no decisive turn.

### G5 (C5 and completeness)

- **C5(L)** reads `recall_failed` and nothing else. A count mismatch with the flag false goes to R3′ and FAILs.
- **C5(b)** accepts the `cue` path on all traces and requires I1's fields.
- **C5(s)** applies to the turns' traces only: the substring path is reachable now, and it is not ported.
- **C5(d) widens:** the save digest (I1) equals the digest the judge recomputes from that phase's saved bytes with G2's
  port, in both phases, and no `outcome.result` string starts `ToolOutput(`. A mismatch makes the attempt incomplete and
  prints the first differing record: a value the JSON writer changed (a tuple, a set, a numpy scalar, a str enum) reads
  differently live and saved, and that is an instrument fault, not a false R3′ FAIL. Saved bytes alone cannot show this.
- **C5(i), new, identity:** each turn's `own_ids[0]` is a saved non-carried record in that trace's V0 whose `same_words(input_text)` equals `same_words(cue_text)`, and `len(cue_text)` equals the `text_len` on that turn's `Bridge.send_and_wait ENTER` line. Both sources are independent of the
  pipeline, so a wrong query cannot conform.
- **C4″, new:** `complete_problems` restates the evidence gate's B0 checks (`judge_b0_fields`) and is pinned equal to it
  by a test. Otherwise a complete attempt the gate refuses decides, and strands the campaign. No committed sim record has
  yet been through those checks.

Every other gate (P0–P3, R1, C1–C4, C4′, C5(a), (c), (g)) stands. C5(c) holds through the recall-only enrich.

**Descriptive, per turn (never gating):** paths shown, cue hits on non-goal keys, what the cue added over the goal,
distinct heard texts among surfaced ids, carried share, new records per input, `goal_fill_win`, carried records hidden by a newer copy.

## Attempts and the verdict

Declared markers, attempts on `main` before the next, at most 3 attempts, the first complete attempt decides. Keys outside `PRE_1079_KEYS` get the within-campaign leak bar. `LEAK_GATES["63b"]` is `P0` for phase 1 and `P0, P1, P2, P3, R1, R2′, R3a, R3′, R3d` for phase 2. A gate reads passed only in its held state: R3a unobservable, no view or a late id, and no decisive turn each read not passed. An incomplete attempt showing any of these ends the campaign, because the harness refuses another attempt.

| Verdict | Condition | Ledger T1-16 |
|---|---|---|
| `PASS` | complete; P0–P3, R1, R2′, R3a, R3′ hold and R3d finds a decisive turn | **EARNED** at the run date (narrow) |
| `NOT SHOWN` | complete; P0–P3, R1, R2′, R3′ hold; R3a unobservable, or no decisive turn | no raise: STALE stays (OD2) |
| `FAIL` | complete; any of P0–P3, R1, R2′, R3a or R3′ fails | **BROKEN**; investigated |
| `ABORT` | no complete attempt within 3, or the leak bar refuses the next | STALE stays; the release decision returns to the owner |

T1-1 stays SUPERSEDED under every outcome.

### OD2: what NOT SHOWN does to the row (owner decision 2026-11-02: no raise)

**No raise on any NOT SHOWN**, Exp 63's rule of 2026-10-03. The row stays STALE, so the cut waits on a further owner
decision. An R3a that is unobservable is also investigated: no carried record reached a result, so R3′ had no power over
carried records, and a save-side text loss could reach that branch. Not taken: PARTIAL (narrow) on R3a held with no
decisive turn. Pass sets per target status (for the M1b evidence gate): `exp63b_verdict` → `{PASS}` supports EARNED on T1-16. The table entry is `{"rows": ["T1-16"], "scopes": ["narrow"], "targets": {"EARNED": ["PASS"]}, "require": {"apparatus_checked": true}}`, and the ledger head reads `EARNED <date> (narrow: …)`.

**Re-run on:** a change to `_rank_by_relevance`, `_query_hippocampus_traced`, the own-record exclusion, `copy_key` or `distinct_copies`, `memory/text.py`'s
keying, `types.py::text_fields` or `content_tokens`, `take_user_input`, `_addresses` or `ADDRESSED_SOURCES`, Hippocampus
save/restore or the resume path, or the memory record shape.

## Build order

1. The skip fix (OD1, #1573), its own PR, reviewed and merged.
2. This prereg merged, on its own PR, frozen before any data.
3. `src/` instrumentation, the judge, the pass table and red gates (`xfail(strict=True)`) first, with a `changelog.d/`
   fragment and the three-lens code round. The `63b` mock runs the REAL seams with only the LLM stubbed, because the Exp 63
   mock takes its ids from the judge's own ranker; deletion probes include removing the exclusion, a repeated turn-1 line and
   a scrambled-origin store.
4. `o19_rerun.py smoke --exp 63b` (M26): protocol argv at 8 turns, a throwaway home, no marker, no rows under
   `docs/experiments/data/`, one per candidate commit, not an attempt. It computes C1–C4″, C5 and R3′ as instrument
   checks and **never computes or prints R3a or R3d**. Before any marker it must show: phase 2's first trace reads
   `hippocampus_size == N1 + 1` and horizon `seq_C + 1`; every turn has a trace with I1's fields; R3′ conforms on 8
   turns; records per input are of the post-#1414 order, reported in the PR (far from one per input is a finding to stop
   on); `select_arc_for_goal(goal)` is None and phase 2's `resume.stores.hippocampus` reads `loaded`; the row passes `judge_sim_fields`; `lint_secrets.py` is clean on the outputs. Then `preflight` at the merged commit.
5. The run on the rig at that commit.
6. Data PR, never squashed.
7. Interpretation PR: the verdict in the ledger with README, `docs/experiments/README.md` and the prereg header moving
   together (`lint_claims_sync`), read by a different reader.
