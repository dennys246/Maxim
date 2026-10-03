# Exp 63: carried memory takes part in recall across sessions (pre-registered 2026-10-03)

> The design review (confounding + bio-faithful lenses, `docs/experiments/DESIGN_REVIEW.md`, a derivative of Exp 10)
> and its delta round are in [rationale/exp63-carried-recall/](rationale/exp63-carried-recall/); their findings are
> folded below with the owner's decisions of 2026-10-03. This prereg lands on `main` before any harness code or data.
> Build order: the instrumentation, the harness key (after #1050), a code review, a dry run, then the run.

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
- `goal_path_horizon`: the store's highest `capture_seq`, read at entry to `_query_hippocampus` and logged on every
  trace (the goal path is skipped when the graph path already returns 3, and the memory-agent thread can capture
  between the query and the `hippocampus_size` read).

The trace is logged at enrich time; rendering activates the first 3 ids (`_activate_rendered`).

## Protocol (two phases; phase 3 dropped, owner decision 2026-10-03)

From one fresh data home, each with `--interactive false`, `--sim-max-turns 8` and `--sim-run-full-turns`:

| Phase | Goal (verbatim) | Turn cap | Resumes |
|---|---|---|---|
| 1, baseline | `escape a dungeon with a sleeping guard` | 8 | none (a fresh data home) |
| 2, the gate | `escape a dungeon with a sleeping guard` | 8 | phase 1 |

## Complete-attempt condition

As the O19 Exp 10 prereg's C1–C4 ([exp10_rerun_2026-10-02_preregistration.md](protocols/exp10_rerun_2026-10-02_preregistration.md)),
for two phases, with `--sim-run-full-turns` in the recorded argv, plus:

- **C5 (the instrument reads true).** Each failure aborts the attempt: an instrument fault is not data.
  - **(a) Agent.** Every phase-2 trace the gates read has `agent_id` equal to the AUT's.
  - **(b) Shape.** Every phase-2 AUT trace has `memory_ids` and `memory_paths` of equal length, equal to its
    `memories` count, and every id is in phase 2's saved store.
  - **(L) Liveness.** Each turn's trace (below) reads `memories == min(3, hippocampus_size)`. It can fail only when
    `_query_hippocampus` swallows an exception to `[]`, an instrument fault.
  - **(d) No compressed record.** Phase 2's saved store holds no `CompressedMemory` (compression would change the
    ranking tokens after a trace was logged; nothing compresses in-session today).
  - **(c) Rendering identity.** For every memory id, `activation_sources.enrichment` in phase 2's saved store minus
    phase 1's (0 for a new id) equals the number of times that id appears in `memory_ids[:3]` across all phase-2 AUT
    traces.

## Gates

Let C be the ids in phase 1's saved store, N1 = |C|, and `seq_C` = the highest `capture_seq` in C.

**Persistence**
- **P0.** N1 ≥ 3.
- **P1.** Phase 2's first AUT `enrichment_trace` reads `hippocampus_size >= N1`.
- **P2.** Every id in C is in phase 2's saved store.
- **P3 (content identity).** For every id in C, phase 2's record equals phase 1's field for field, except the
  mutable access fields `access_count`, `accessed_at`, `last_scored_at`, `activation_count`, `activation_sources`,
  `promotion_pressure`, `access_contexts`, `long_term` and `consolidated_at`, and, only when the reports show
  `memory.strategy=strength`, `storage_strength` and `retrievability_anchor_us`. `encoding_tag` (stamped at capture,
  never recomputed) and `retro_tag` (written only by `sleep()`, which never runs here) are byte-equal like the rest.
  Every id not in C has `capture_seq > seq_C`.

**Recall.** Turns are attributed by the `sim_exec` `Bridge.send_and_wait ENTER turn=N` lines; "a turn's trace" is
its first AUT `enrichment_trace` with a non-empty `goal`.
- **R1.** Each of phase 2's turns 1–8 has such a trace.
- **R3a (carried memories are reachable).** The first phase-2 AUT trace whose `goal_path_horizon == seq_C` (no new
  memory yet) surfaces min(3, N1) ids, all in C, or any id outside C is one R3′'s recomputation reproduces within its
  `+2` window (a capture that landed between the horizon read and the scan). If no phase-2 trace has that horizon (a capture landed first), the
  attempt aborts: reachability is then not observable.
- **R3′ (conformance, owner decision 2026-10-03).** For each turn's trace, recompute
  `dedup(the logged graph-path ids + the goal-path top 3 + the logged substring ids)[:3]`, where the goal-path
  top 3 is the executed commit's own `_rank_by_relevance(candidates, goal, 5)[:3]` over the phase-2 saved-store
  records with `capture_seq ≤ goal_path_horizon` (tolerance: any horizon in `[goal_path_horizon, goal_path_horizon + 2]`
  that reproduces the logged ids, for the capture race). Substring ids cannot be recomputed (the path matches the full
  percept text, and the trace logs 120 characters), so each logged substring id must only be a candidate
  (`capture_seq ≤ goal_path_horizon + 2`), and they must appear in recency order. The logged `memory_ids` must equal the recomputation, in
  order, on **8 of 8** turns. A FAIL means a carried record did not take part in recall as its saved bytes say it
  should (an index or field not restored, a compressed form treated differently): the persistence-into-recall defect
  P2 cannot see.

**Descriptive (never gating).** Per phase-2 turn: `|memory_ids ∩ C|` observed and recomputed, the best carried
goal-path score and how many new memories score at or above it, the turn at which carried memories leave the goal-path
top 3 (if any), the graph-path id count (the only cue-dependent channel), and `memory_recall` tool calls on carried ids
(a second, ungated route).

## Attempts and the verdict

Declared markers, attempts on `main` before the next, at most 3 attempts, the first complete attempt decides, and the
rig stays at the first attempt's commit, as the O19 preregs state. The harness is the O19 pair extended with this
experiment's key, after #1050.

| Verdict | Condition | Ledger |
|---|---|---|
| `PASS` | complete; P0–P3, R1, R3a and R3′ hold | a new row (this claim) **EARNED** at the run date; T1-1 → **SUPERSEDED** by it |
| `FAIL` | complete; any of P0–P3, R1, R3a or R3′ fails | the new row is not earned; T1-1 stays STALE and the failing gate is investigated |
| `ABORT` | no complete attempt within 3 | no change: T1-1 stays STALE |

Pass sets per target status (for the M1b evidence gate): `exp63_verdict` → `{PASS}` supports EARNED on the new row.

**Re-run on:** a change to `_rank_by_relevance` or `_query_hippocampus` (including #1064's fix); Hippocampus
save/restore or the resume path (`simulation/report.py` RESUME_STORES); the memory record shape.
