# Exp 63: carried memory takes part in recall across sessions (pre-registered 2026-10-03)

> The design review (confounding + bio-faithful lenses, `docs/experiments/DESIGN_REVIEW.md`, a derivative of Exp 10)
> and its delta round are in [rationale/exp63-carried-recall/](rationale/exp63-carried-recall/); their findings are
> folded below with the owner's decisions of 2026-10-03. This prereg lands on `main` before any harness code or data.
> Build order: the instrumentation, the harness key (after #1050), a code review, a dry run, then the run.

## Amendments

**Amendment 1 — 2026-10-03, PRE-DATA, the harness design pass, its code review and the owner's decisions D1–D3 of
2026-10-03, all before any Exp 63 data.**
Every change to this file since it landed on `main` (#1065), each marked in place "(Amendment 1)"; the instrumentation
PR's own in-place notes "(amended 2026-10-03, before any data …)" (#1067: `goal_path_holes`, the missing-horizon
read) stand as they were and are part of this amendment's record:
1. **R3a's capture race aborts.** No goal-bearing trace with the carried-only view, or one that surfaced an id
   outside C, makes the attempt incomplete (C5), never a FAIL; the old `+2` window is gone.
2. **The substring rule is gone** from R3′ (substring ids cannot be recomputed); C5(s) forbids them on goal-bearing
   traces instead.
3. **The superset rule:** R3′ recomputes over **V0 ∪ L** (L: logged ids outside V0 that are in the saved store).
4. **P3:** `storage_strength` and `retrievability_anchor_us` are always byte-equal (the strategy is pinned by C4′).
5. **C4′** (the retention model is pinned, read at preflight, stamped in every row).
6. **C5 additions:** (a) the AUT is `sim_aut`; (d) observation text is a string, timestamps are numbers, and every
   carried record carries an integer `capture_seq` (strict: without it V0 and seq_C are undefined); (g) graph ids;
   (s) no substring ids.
7. **The `NOT SHOWN` verdict row**, and **D2:** NOT SHOWN is terminal, like PASS and FAIL; only an ABORT may be
   succeeded.
8. **D1:** R3d is decided on the three memories actually shown.
9. **Empty-goal traces are allowed:** C5(b) checks the goal of goal-bearing traces only, and R3a reads goal-bearing
   traces only (E3).
10. **A2:** R3a's unobservability aborts only while P1 and P2 hold; after a total restore failure the attempt is
    complete and FAILs on P1/P2.
11. **E4:** a graph id that landed after its trace's horizon read (in the saved store) is allowed, as a goal id is,
    and joins L.
12. **C5 means incomplete:** a C5 failure makes the attempt incomplete (an aborted attempt, never a FAIL), and the
    judge decides C5 and R3a's observability from the committed traces and stores (the harness reads the same).
13. **C5(b):** every field of a phase-2 AUT trace is type-checked (E5), and every AUT trace, not only a turn's,
    carries an integer `hippocampus_size` (P1 reads the first one).
14. **C5(c)** is scoped to the ids in phase 2's saved store (a carried id missing there is P2's FAIL), and states
    that every enrich caller renders through `format_thought_response`.
15. **The visible set V0** is defined once, under Gates.
16. **Turns are attributed by line order** in the run log, not by its `t` (rounded to 0.01 s).
17. **R3′:** "8 of 8 turns, in order" is restated as "all 8 turns must conform", with exact-tie groups in any order;
    the hole-inclusion note (amended 2026-10-03) now names L.
18. **PASS requires R3d's decisive turn**; R3d's "visible non-carried" counts V0 ∪ L, and an exact tie does not rank
    above.
19. **Descriptive:** whether each turn is decisive, and, under ties, the turn the goal top 3 no longer MUST hold a
    carried id and the turn it no longer CAN (A8).
20. **The ledger:** the new row lands at `SETUP` before the first attempt; T1-1 → SUPERSEDED lands in the same diff
    that earns the new row, and the successor names T1-1 (D3, `scripts/lint_ledger_format.py`).

## Why this experiment exists

- **Ledger row it replaces:** T1-1, "Cross-session memory persistence (substrate carries memory across sessions)",
  STALE since 2026-09-30; it blocks the 1.3.2 cut.
- **Why not a third re-run.** The O19 re-runs of Exp 10 (campaigns 1 and 2) both aborted on the simulation narrator
  (#1042, #1052). Their fixes (#1045/#1047, #1056, #1058) also changed the agent-under-test's prompt. Owner decision
  2026-10-03 (strict): a successor campaign on changed subject code cannot support the row
  ([reproduction.md](reproduction.md) §13), so the claim is re-tested as a new experiment on current code (#1060).
- **What the design review found.** Exp 10's recall gate (≥ 3 memories surfaced per resume turn) is implied by the
  store being non-empty, and a gate that a carried memory *surfaces* on every turn is a lexical lottery. Recall's goal
  path ranks every memory against the fixed goal string by plain-word overlap. It counts stop words, keeps
  punctuation and reads a field the loop's captures never write, so ~90–95% of memories score 0 and recency breaks
  the ties against carried memories ([#1064](https://github.com/dennys246/Maxim/issues/1064)). Owner decision
  2026-10-03: run on the **current** ranker and gate on **conformance**, that carried memories take part in recall
  exactly as the shipped ranker dictates. The ranker fix is its own issue and a re-run trigger for this row.

## Claim (the new row)

> The carried store is saved and reloaded exactly, with every carried memory's content unchanged. On every resume
> turn the carried memories take part in recall as full candidates (scored by the executed ranker, which reads no
> memory's origin or age except its recency tiebreak): the surfaced set is exactly what the shipped ranking over
> carried and new memories dictates. On the first resume turn, the memories surfaced are carried ones.

- **Mechanism:** Hippocampus persistence (`aut_hippocampus.json` save and resume restore) + `BioEnrichmentPipeline`
  recall (`_query_hippocampus`: graph path, then the goal path, then substring).
- **What "recall" means here, stated plainly:** the goal path ranks memories by word overlap with the session goal,
  with recency breaking ties and no minimum score (#1064). The claim is about storage persistence into that recall,
  not about cue-dependent retrieval, and not about survival through consolidation: no `sleep()` runs and nothing can
  be evicted at this store size.
- **Not claimed:** "RECALL reconsolidation" (dropped: recall never changes a memory's content, only its access and
  activation bookkeeping; the old EC reconsolidation on recall was removed in D8); negative transfer (not tested);
  a behavioral difference from carried memory (T1-2's claim).

## Apparatus (owner decision 2026-10-03: as O19)

- **Box:** big-mac-mini, quiet (Paper servers, Minecraft bridges and ollama stopped; operator-attested, the harness
  stamps the hostname). One harness, no second LLM consumer.
- **Model:** `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (`llm.profile mistral-7b`) at `llm.n_ctx 8192`, set through
  `maxim config` and checked from the reports.
- **Code:** a commit on `main` with this prereg, the instrumentation and the harness merged; clean tree; the rig
  stays at the first attempt's commit.

## Instrumentation (subject code, logging only; lands before any data, each field proven by deletion)

`enrichment_trace` gains:
- `agent_id`: the agent whose pipeline emitted the trace (the narrator also owns a bio stack, though nothing calls
  its pipeline today; nothing structural guarantees that).
- `memory_ids`: the ids `_query_hippocampus` returned, in order.
- `memory_paths`: a parallel list, each `"graph"`, `"goal"` or `"substring"`.
- `goal_path_horizon`: the store's highest stored `capture_seq` (-1 when empty), read at entry to
  `_query_hippocampus` and logged on every trace (the goal path is skipped when the graph path already returns 3, and
  the memory-agent thread can capture between the query and the `hippocampus_size` read).
- `goal_path_holes` (amended 2026-10-03, before any data; the instrumentation's review): the numbers from the
  process's first reservable one (the store's load watermark) up to the horizon that were not stored at that moment,
  read with the horizon (the floor moves only at a load, which precedes every capture), as inclusive `[start, end]`
  ranges. Numbers below the watermark were loaded or are gone
  for good, and are never in a later save either. A capture reserves its number before it
  lands (the loop's async capture at enqueue, a direct capture at insert), so a later number can be stored while an
  earlier one is in flight; dropped and evicted numbers are holes too.

The trace is logged at enrich time; rendering activates the first 3 ids (`_activate_rendered`).

## Protocol (two phases; phase 3 dropped, owner decision 2026-10-03)

From one fresh data home, each with `--interactive false`, `--sim-max-turns 8` and `--sim-run-full-turns`:

| Phase | Goal (verbatim) | Turn cap | Resumes |
|---|---|---|---|
| 1, baseline | `escape a dungeon with a sleeping guard` | 8 | none (a fresh data home) |
| 2, the gate | `escape a dungeon with a sleeping guard` | 8 | phase 1 |

## Complete-attempt condition

As the O19 Exp 10 prereg's C1–C4 ([exp10_rerun_2026-10-02_preregistration.md](protocols/exp10_rerun_2026-10-02_preregistration.md)),
for two phases, with `--sim-run-full-turns` in the recorded argv, plus (Amendment 1):

- **C4′ (the retention model is pinned).** The harness reads `memory.strategy` at preflight in the attempt's fresh
  data home and stamps it in every row; it must read `access_based` (the default; the harness drops every operator
  `MAXIM_*` key, so `MAXIM_MEMORY_STRATEGY` cannot reach the sim).
- **C5 (the instrument reads true).** Each failure makes the attempt incomplete (an aborted attempt, never a FAIL):
  an instrument fault is not data. The judge decides C5 and R3a's observability from the committed traces and stores
  before an attempt counts as complete.
  - **(a) Agent.** The AUT's agent id is `sim_aut` (`config_loader.SIM_AUT_AGENT_ID`; C4's pinned argv rules out
    adopting a persistent agent), cross-checked against the AUT's own `sim_deliberation` log lines. Every
    `enrichment_trace` carries a non-empty `agent_id`; the gates read the traces whose `agent_id` is `sim_aut`.
  - **(b) Shape.** Every phase-2 AUT trace has `memory_ids` and `memory_paths` of equal length, equal to its
    `memories` count, and every id is in phase 2's saved store. It carries `goal_path_horizon` and
    `goal_path_holes` (a missing horizon means its read failed: amended 2026-10-03, before any data). A non-empty
    `goal` is the protocol goal; a trace with an empty goal ran no goal path and is allowed (Amendment 1). Every
    field is type-checked: a corrupt trace makes the attempt incomplete, never a judge crash.
  - **(L) Liveness.** Each turn's trace (below) reads `memories == min(3, hippocampus_size)`. It can fail only when
    `_query_hippocampus` swallows an exception to `[]`, an instrument fault.
  - **(d) Record shape.** Phase 2's saved store holds no `CompressedMemory` (compression would change the ranking
    tokens after a trace was logged; nothing compresses in-session today), every record's `perception.observations.text` is a string or absent (the
    store's JSON writer stringifies any other value, which the live ranker would have ignored) and its `timestamp` a
    number, and every record in phase 1's store (C) carries an integer `capture_seq` (Amendment 1).
  - **(c) Rendering identity.** For every memory id in phase 2's saved store (a carried id missing there is P2's
    FAIL), `activation_sources.enrichment` in phase 2's saved store minus
    phase 1's (0 for a new id) equals the number of times that id appears in `memory_ids[:3]` across all phase-2 AUT
    traces. (Every enrich caller renders through `format_thought_response`, which activates exactly those ids.)
  - **(g) Graph ids.** Every `graph`-labelled id has a non-empty `perception.detected_objects` in the saved store
    (`recall(object_detected=…)` returns no other record). A graph id outside its trace's V0 landed between the
    horizon read and the recall, as a goal id may, and joins L (Amendment 1, E4).
  - **(s) No substring ids.** No goal-bearing trace carries a `substring` id: with three or more visible records the
    goal path always fills the cap first, so a substring id means the instrument or the path order is wrong.

## Gates

Let C be the ids in phase 1's saved store, N1 = |C|, and `seq_C` = the highest `capture_seq` in C. For a trace with
horizon h and holes H, its **visible set** V0 is the phase-2 saved-store records with `capture_seq ≤ h` and not in H.

**Persistence**
- **P0.** N1 ≥ 3.
- **P1.** Phase 2's first AUT `enrichment_trace` reads `hippocampus_size >= N1`.
- **P2.** Every id in C is in phase 2's saved store.
- **P3 (content identity).** For every id in C, phase 2's record equals phase 1's field for field, except the
  mutable access fields `access_count`, `accessed_at`, `last_scored_at`, `activation_count`, `activation_sources`,
  `promotion_pressure`, `access_contexts`, `long_term` and `consolidated_at`. Every other field is byte-equal,
  including `storage_strength`, `retrievability_anchor_us`, `encoding_tag` and `retro_tag` (the retention model is
  pinned to `access_based` by C4′, which never moves them, and `sleep()` never runs here). Every id not in C has
  `capture_seq > seq_C`.

**Recall.** Turns are attributed by the order of lines in the run log, between successive `sim_exec`
`Bridge.send_and_wait ENTER turn=N` lines (the log's `t` is rounded to 0.01 s, so time is not used); "a turn's
trace" is its first AUT `enrichment_trace` with a non-empty `goal`.
- **R1.** Each of phase 2's turns 1–8 has such a trace.
- **R3a (carried memories are reachable).** The first goal-bearing phase-2 AUT trace whose
  `goal_path_horizon == seq_C` and whose holes are empty (no new memory yet) surfaces min(3, N1) ids, all in C. If no
  such trace has that view (a capture landed first), or that trace surfaced an id outside C (a record that landed
  between its horizon read and the recall: only carried records were visible at the read), the attempt is incomplete
  (C5): reachability is then not observable (Amendment 1). This holds only while P1 and P2 hold: after a total
  restore failure no trace can see seq_C, and the attempt is complete and FAILs on P1/P2, which read independent
  bytes (Amendment 1, A2).
- **R3′ (conformance, owner decision 2026-10-03).** For each turn's trace, the goal-path ids are recomputed with the executed commit's ranker
  (`recall(query=goal, limit=5)`: every record a candidate, compressed included, ranked by `_rank_by_relevance`,
  then the first 3), carried by the judge as a frozen copy over the saved-store records and pinned by a test to the
  ranker in the executed commit. Recall's view V satisfies V0 ⊆ V ⊆ the saved store, and a record that landed after
  the horizon read can only push others down, so the recomputation runs over **V0 ∪ L**, where L is the logged
  ids (goal or graph path) outside V0 that are in the saved store (records that landed between the horizon read and
  the recall; Amendment 1). A logged id in one of the trace's holes whose record is in the saved store is in L (amended
  2026-10-03, before any data: a capture that landed between the holes read and the recall). A record visible at a
  trace but evicted before the save would make R3′ unreproducible; no eviction happens at this run's store size (the
  cap is 10,000 records). The trace's logged sequence must equal `dedup(logged graph ids + recomputed goal top 3)[:3]`. Records
  tied exactly on (score, timestamp) may appear in either order (`recall` builds candidates from a set, whose order
  is not reproducible); ties are by exact equality only. All 8 turns must conform.
- **R3d (the test has power; owner decisions 2026-10-03, D1 in Amendment 1).** R3d is decided on the three memories
  actually SHOWN, `dedup(logged graph ids + goal top 3)[:3]`, recomputed over V0 ∪ L. A turn is **decisive** when, in
  EVERY valid tie ordering, a carried id is among the shown ids from the goal path AND ranks strictly above at least
  one visible non-carried record ("visible" is V0 ∪ L: a late arrival was in recall's view, so it is a real
  competitor; a carried record tied exactly with it does not rank above it). A carried id that only fills a slot because fewer than 3 new records are visible
  does not count; a carried graph id (the cue path, not the ranking) does not count; and a turn whose graph ids fill
  the slots (the goal path never ran) is not decisive. At least one turn after R3a's turn must be decisive. With
  none, the attempt could not test the claim, and the verdict is `NOT SHOWN` (no status change).

A FAIL of R3′ means a carried record did not take part in recall as its saved bytes say it should (an index or field
not restored, a compressed form treated differently): the persistence-into-recall defect P2 cannot see.

**Descriptive (never gating).** Per phase-2 turn: `|memory_ids ∩ C|` observed and recomputed, whether the turn is
decisive, the best carried goal-path score and how many new memories score at or above it, the turn at which carried
memories leave the goal-path top 3 (if any; under ties, both the turn it no longer MUST hold one and the turn it no
longer CAN), the graph-path id count (the only cue-dependent channel), and
`memory_recall` tool calls on carried ids (a second, ungated route).

## Attempts and the verdict

Declared markers, attempts on `main` before the next, at most 3 attempts, the first complete attempt decides, and the
rig stays at the first attempt's commit, as the O19 preregs state. The harness is the O19 pair extended with this
experiment's key, after #1050. The new row lands at `SETUP` before the first attempt, so the verdict can raise it.

| Verdict | Condition | Ledger |
|---|---|---|
| `PASS` | complete; P0–P3, R1, R3a, R3′ hold and R3d finds a decisive turn | the new row **EARNED** at the run date; T1-1 → **SUPERSEDED** by it |
| `NOT SHOWN` | complete; P0–P3, R1, R3a and R3′ hold, but no decisive turn | no change: the row stays `SETUP`, T1-1 stays STALE; terminal (D2) |
| `FAIL` | complete; any of P0–P3, R1, R3a or R3′ fails | the new row is not earned; T1-1 stays STALE and the failing gate is investigated |
| `ABORT` | no complete attempt within 3 | no change: T1-1 stays STALE |

Pass sets per target status (for the M1b evidence gate): `exp63_verdict` → `{PASS}` supports EARNED on the new row.

**NOT SHOWN is terminal (owner decision 2026-10-03, D2; Amendment 1).** Like PASS and FAIL, it closes the experiment:
no successor campaign may open after it (only an ABORT may be succeeded; `o19_verdict.successor_problems`). T1-1 then
stays STALE, and the next route is #1064's ranker fix and a re-run as a new experiment on the fixed ranker.

**The ledger move on PASS (D3).** T1-1 → `SUPERSEDED` by the new row lands in the same diff that raises the new row
to EARNED: `scripts/lint_ledger_format.py` refuses a row entering SUPERSEDED unless its successor reaches a positive
status in that diff and its row names the retired row's id (the new row names T1-1), or a committed
`superseded` exception clause on main names the transition.

**Re-run on:** a change to `_rank_by_relevance` or `_query_hippocampus` (including #1064's fix); Hippocampus
save/restore or the resume path (`simulation/report.py` RESUME_STORES); the memory record shape.
