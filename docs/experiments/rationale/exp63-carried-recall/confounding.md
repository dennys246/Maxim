# Exp 63: confounding lens (design review, 2026-10-03)

Reviewed: `docs/experiments/exp63_carried_recall_prereg.md` (DRAFT; nothing built, nothing run).
Consulted: `docs/experiments/DESIGN_REVIEW.md`, `docs/experiments/10_cross_session_enrichment.md`,
`docs/experiments/protocols/exp10_rerun_2026-10-02_preregistration.md`, `docs/experiments/reproduction.md`,
`src/maxim/integration/bio_enrichment.py` (`enrich`, `_query_hippocampus`, `format_thought_response`,
`_activate_rendered`), `src/maxim/memory/hippocampus_retrieval.py` (`recall`, `_rank_by_relevance`),
`src/maxim/memory/hippocampus.py` (`capture`, `_insert_and_index_locked`, `store`),
`src/maxim/simulation/orchestrator.py` (`_restore_aut_from_session`, the AUT and narrator `AgentConfig`s, both
`run_agentic_loop` calls), `src/maxim/simulation/report.py::RESUME_STORES`, `scripts/o19_verdict.py`
(`turn_windows`, `exp10_gates`).
Data used to ground every claim below: the committed O19 logs and stores. The main source is
`data/rerun_exp10_o19c2/` (attempt 1: phase 1 `20261002_143553` and phase 2 `20261002_144108` ran to
`max_turns`; phase 3 `20261002_144808` ended `planning_failed`). Phase 1 of `data/rerun_exp10_o19/20261001_215255`
was also read.

Charter question: could a positive or a null on these gates arise for a reason other than the claim ("a resumed
session's recall surfaces memories carried from the earlier session on every resume turn, and the carried store is
reloaded exactly")?

Short answer: **the persistence half (P0–P2) is sound, and two cheap additions make it tighter. The recall half is
not a test as drafted.** R2 is implied by P1. R3's outcome is decided by function-word overlap between the LLM's
plan sentences and the goal string, plus a recency tie-break that always works against carried memories. Its 8
"every turn" observations are one binary event. A PASS and a FAIL are both mostly lexical luck.

---

## How recall actually ranks (the fact every finding below rests on)

Phase-2's first trace per turn comes from the AUT's percept-path enrich (`agent_loop.py` ~L3102), called with
`active_goal` = the sim goal. Inside `_query_hippocampus`:

1. **Graph path** (cue-dependent: the narrator's text → EC read-only completion → `retrieve_on_cue` → ATL names →
   `recall(object_detected=…)`). It runs first, and runs only if the encoder and EC are wired.
2. **Goal path**, only if fewer than 3 so far: `recall(query=goal, limit=5)`, then the top 3 are added. **The query is
   the constant sim goal, not the turn's text.** `_rank_by_relevance` scores each memory as the fraction of
   `goal.lower().split()` tokens found in its `context.active_goal` (the LLM's plan sentence), its tool name, its
   detected objects or people, `perception.observations["text"]`, and its `decision_rationale`. **There is no score
   floor**: every memory is a candidate. Ties go to the **newest timestamp**.
3. **Substring path**, only if still fewer than 3. Since (2) has no floor, it never runs once the store holds 3 or
   more memories.

Measured on the O19 c2 stores:

- The goal `escape a dungeon with a sleeping guard` tokenizes to `{escape, a, dungeon, with, sleeping, guard}`. Stop
  words count, and punctuation is not stripped, so `guard's` and `dungeon's` never match.
- In phase 1's 72 memories, **65 are percept captures (`encoding.site == "memory_agent"`) that always score 0**.
  Their text lives in `observations["transcript"]` / `["cli_input"]`, which the ranker does not read. The other 7
  are loop memories carrying a plan sentence.
- In phase 1, only 5 memories score above 0. Four of them score 1/6 from the word **`a`** alone. One, `d4b1f0ba`
  ("I need to gather more information about **the guard** before deciding on **a** course of action"), scores 2/6.
  Phase 2's 262 new memories: 4 score 1/6 (`a`, or `with`), and none scores 2/6.
- Carried memories keep their phase-1 timestamps (verified: across the 72 carried records, phase 1 and phase 2
  differ only in access and activation fields). **The tie-break always ranks a carried memory below an equal-score
  phase-2 memory.**

A replay of `_rank_by_relevance` over the phase-2 store, filtered to memories captured before each turn's trace,
predicts:

- turn 1: 3 carried (`d4b1f0ba`, `847149e3`, `a1233073`);
- turn 2: 2 carried;
- turns 3–8: exactly 1 carried each, always `d4b1f0ba`.

The saved stores confirm the prediction exactly. Phase 2 adds `activation_sources.enrichment` +8 to `d4b1f0ba`, +2
to `847149e3` and +1 to `a1233073`: 11 carried activations, plus 13 on new memories, which is 24 = 8 traces × 3
rendered. So in c2 the graph path contributed nothing that could be told apart from the goal path, and **R3 would
have passed 8/8 because of one memory, which ranked first for containing the words "a" and "guard"**.

---

## DO-NOT-BUILD

### D1. R3 ("≥1 carried id in the first trace of every phase-2 turn") is a lexical lottery, and its 8 turns are one observation

- **Why a PASS is not evidence for the claim.** R3 holds on a turn exactly when fewer than 3 phase-2 memories
  captured before that turn score at least as high as the best carried memory. Ties count against carried memories,
  by recency. The score is the share of goal tokens, stop words included, found in a plan sentence the LLM happened
  to write. A PASS therefore means "phase 1's LLM once wrote a sentence with more goal function words than phase 2's
  LLM wrote, at most twice". It says nothing about carried memories being recalled because they persisted. They
  are candidates either way (P2 already proves they are in the store).
- **Why a FAIL is not evidence against it.** Suppose phase 1's best plan scores 1/6 (just the word `a`), as 4 of 5
  did in c2. Then 3 phase-2 plans with an `a` (4 of 8 in c2) push every carried memory out by recency, and R3
  fails with persistence and recall both working exactly as built.
- **"Every turn" is not 8 tests.**
  - The goal-path query is constant, and the candidate set only grows. The best carried score is fixed, while the
    count of phase-2 memories at or above it only rises. So once R3 fails on turn t, it fails on every later turn.
    R3 is effectively "R3 at turn 8", one Bernoulli draw.
  - From c2's rates (1 of 15 plan sentences reached 2/6), P(PASS) is roughly 0.3–0.4. That is a rough estimate from
    one attempt, but it is far from "near-certain by construction" and far from "near-impossible". It is a coin
    flip on wording.
  - Only the graph path, which is cue-dependent, could break the monotonicity. Nothing in the draft logs which path
    supplied an id.
- **The narrator's wording does enter (open question 2), but indirectly.** It shapes the AUT's plan sentences,
  which are the only text the ranker scores. It also changed between Exp 10, O19 and now (#1047), so a PASS or FAIL
  cannot be compared across those runs.
- **No threshold fixes this (open question 1).** "At least one", "a majority" and "a turn fraction" all read the
  same monotone, lexically decided ranking. A majority would fail on turn 3 in c2. A turn fraction gives up the
  owner's "every turn" while still measuring wording.

**Fix (pre-build, numeric).** Replace R3 with a **recall-conformance gate**. It tests what persistence-into-recall
can actually break: carried memories must be full, correctly deserialized candidates of recall over the merged
store.

- **Instrument.** `enrichment_trace` gains three fields:
  - `memory_ids`, as drafted;
  - a parallel `memory_paths` list, each entry `"graph"`, `"goal"` or `"substring"`;
  - `goal_path_horizon`: the store's highest `capture_seq` (or its size), read immediately before the
    `recall(query=goal)` call. (`hippocampus_size` is read after the query, and the memory-agent thread can capture
    in between.)
- **R3′ (gating).** On the first trace of each phase-2 turn, recompute `dedup(graph ids as logged + goal top-3 +
  substring results if still under 3)[:3]`.
  - Compute the goal top 3 with the executed commit's own `_rank_by_relevance(candidates, goal, 5)[:3]`. The
    candidates are the phase-2 saved-store records with `capture_seq ≤ goal_path_horizon`, read from the bytes.
  - The logged `memory_ids` must equal the recomputed list, in order, on **8 of 8 turns**.
  - Allow a tolerance window of horizon to horizon + 2 for the capture race, stated in advance.
  - What a FAIL means: a carried record did not take part in recall as its saved bytes say it should (for example,
    an index or field not restored, or a compressed form treated differently). That is a real persistence-to-recall
    defect, and it is the failure mode P2 cannot see.
- **R3 as drafted becomes descriptive.** The verdict reports, per turn:
  - observed `|memory_ids ∩ C|`;
  - the recomputed count;
  - the best carried score and the number of phase-2 memories at or above it;
  - the turn at which carried memories leave the goal-path top 3, if any.
- **The claim text must follow.** "Surfaces carried memories on every resume turn" is not something this mechanism
  does. It is something its ranking may or may not dictate. Suggested wording: "a resumed session's recall ranks
  carried memories as full candidates on every resume turn (the surfaced set equals what the ranking over carried
  and new memories dictates), carried memories surface on the first resume turn, and the carried store is reloaded
  exactly." Whether a carried memory surfaces on later turns is reported, with the lexical cause shown.

If the owner wants a gate that carried memories *surface* on later turns, that needs a manipulation the ranking
responds to, and it belongs in a different claim. One option is a fixed-wording probe that the harness can control.
The other is the graph path, which is cue-dependent and is the bio-faithful channel. It is not a threshold on R3.

---

## SHOULD-FIX

### S1. R2 is implied by P1, so it is not a recall gate

The goal path has no score floor and always returns `min(5, size)` candidates. So the first trace reads
`memories == 3` whenever `hippocampus_size ≥ 3`, `goal` is non-empty, and no exception fires. P0 and P1 give
size ≥ N1 ≥ 3, and R1 requires a non-empty goal.

- Data: all 21 O19 traces with size ≥ 3 read `memories == 3`. The one trace at size 0 read 0.
- R2 can only fail by an exception, since `_query_hippocampus` swallows exceptions to `[]`. That is an instrument
  fault, not a recall result.

**Fix.** Relabel R2 as a liveness/instrument check (`memories == min(3, hippocampus_size)` on every first trace).
Say in the prereg that it is implied by P1 and R1 except for the exception path. Do not count it toward the
"recall" half of the claim.

### S2. C5's known-answer leg is weak: any valid id passes it

"On phase 2's first trace reading `hippocampus_size == N1`, every surfaced id is a phase-1 id" holds for any id the
instrument could log, because only phase-1 ids exist at that point. An instrument that logged the wrong 3 carried
ids would pass. Also, if a capture lands before the first trace, no trace reads `== N1` and the leg is vacuous. The
O19 prereg itself notes this can happen.

**Fix: two known-answer legs, both computable offline, both already shown to hold on c2.**

- **(a) Rendering identity.** For each memory id, `activation_sources.enrichment(phase-2 store) −
  activation_sources.enrichment(phase-1 store)` must equal the number of times that id appears in `memory_ids[:3]`
  across **all** phase-2 traces. Every caller renders through `format_thought_response`, which calls
  `_activate_rendered`. For new ids the phase-1 value is 0.
  - On c2 this holds exactly (carried 11 + new 13 = 24 = 8 × 3).
  - Prove it by deletion: a test with `memory_ids` deliberately shuffled must fail.
- **(b) Exact re-rank on turn 1.** This is R3′'s computation on the first trace. If no trace reads size == N1, the
  leg still applies, because the horizon is logged.

A failure of either leg aborts the attempt (an instrument fault), as drafted.

### S3. Trace attribution: the narrator has a bio stack, and the trace carries no agent id

- The narrator is built with `AgentConfig(agent_id="sim_orchestrator", with_bio_stack=True,
  load_persisted=False)`, so it owns a hippocampus and a `BioEnrichmentPipeline`.
- Today nothing calls it:
  - its `run_agentic_loop` receives no `bio_enrichment_pipeline`, and its `hippocampus=` is commented out;
  - `ExecAgent.wire_bio_enrichment` has no caller;
  - `ThinkTool(pipeline=…)` is registered only on the AUT registry.
- In every O19 log, phases 1 and 2 show exactly one trace per turn, and its `hippocampus_size` tracks the AUT
  store. So **all observed traces are the AUT's**.
- Nothing structural holds this. The `enrichment_trace` line has no `agent_id` (the `ENTER` lines get
  `sim_orchestrator` from the log context; the traces get nothing).
- The pipeline's `_agent_id` is set only inside the `aut_component_registry is not None` block (embodiment runs),
  so even reading it would give `""` for both agents in this protocol.
- The risk if a future change wires the narrator: its trace could become a turn's "first trace with a non-empty
  goal". It would carry ids from a fresh store, which C5 catches. But if it read 0 memories (an empty list passes C5
  vacuously), it would fail R2 as a FAIL, not an abort.

**Fix.**
- Pass `agent_id` into `BioEnrichmentPipeline(...)` in `build_bio_stack`; the constructor already takes it.
- Emit it in `enrichment_trace`.
- The verdict counts only `agent_id == "sim_aut"` traces. A first-per-turn trace with any other agent id aborts
  the attempt.

This is one line of subject code, logging only, and it lands with the `memory_ids` instrumentation before any data.

### S4. P2 checks id presence, not that the record is the carried one

- Ids come from `uuid4()` in `capture`. But `capture(record=…)` (and `Hippocampus.store`) keeps a pre-built record's
  own id and does `self._memories[memory_id] = memory` with no collision check.
- A phase-2 path that re-stored a phase-1 record would therefore overwrite it under the same id. P2 would still
  pass, and R3 (or R3′) would count a phase-2 capture as "carried".
- Nothing in the c2 data shows this happening. Across all 72 carried records, phases 1 and 2 differ only in
  `access_count`, `accessed_at`, `last_scored_at`, `activation_count`, `activation_sources`, `promotion_pressure` and
  `access_contexts`. Every new record has `capture_seq ≥ 72` (phase 1's maximum is 71), and its `run_id` differs.
  But no check enforces any of this.

**Fix: P2′.**
- For every id in C, the phase-2 and phase-3 record equals the phase-1 record, field for field, except an
  enumerated set of mutable access and strength fields. That set is the 7 above plus `long_term`, `consolidated_at`,
  `storage_strength`, `retrievability_anchor_us`, `encoding_tag` and `retro_tag`, pinned in the prereg.
- `timestamp`, `created_at`, `capture_seq`, `run_id`, `perception`, `context`, `decision`, `action` and `outcome`
  must be byte-equal.
- Every non-carried id has `capture_seq > max(capture_seq over C)`.

This is also the "carried store is reloaded **exactly**" half of the claim, which presence alone does not show.

### S5. Phase 3 (descriptive) holds a veto over the gating phases, and its metric measures stop words

- **Veto.** C1 requires every phase to reach its cap. O19 c2's only attempt **aborted on phase 3** (garden,
  `planning_failed`), and phases 1 and 2 had run cleanly to `max_turns`. A phase that cannot gate should not be able
  to ABORT the experiment.
- **Metric.** The garden goal tokenizes to `{you, are, in, a, peaceful, garden,, enjoy, the, flowers}`. Note the
  trailing comma: `garden,` never matches `garden`.
  - A dungeon plan like "I need to understand **the** dungeon's structure before making **a** move" scores 2/9 on
    `the` and `a` alone.
  - Any garden plan such as "enjoy the flowers in the garden" scores at least 4/9 and is newer.
  - So the "carried share" reports how many plan sentences phase 3 writes, and how fast. It is not negative
    transfer.
- **The 0-entity garden (open question 5) does not confound the memory metric.** `_ENTITY_INDICATORS`
  (`imagination/trigger.py`) feeds `resolved_entities`, which reaches affordances and imagination. It never reaches
  `_query_hippocampus`.

**Fix (pick one).**
- **(a)** Drop phase 3. Recommended: it buys no evidence and adds one more abort path.
- **(b)** Run phase 3 after the verdict-deciding phases, and remove it from C1–C5, P1 and P2, so it can neither
  abort nor fail the attempt. If kept, report its carried share next to the re-rank prediction, with the stop-word
  cause stated, and keep "negative transfer not measured" in the row.

### S6. "RECALL reconsolidation" (open question 4) is not on the measured path

- `recall(working_memory=…)` adds RECALL entries only when a working-memory set is passed. `_query_hippocampus`
  passes none.
- The old EC reconsolidation on recall was removed (D8: the graph path now uses `pattern_complete_readonly`).
- The only recall-time write on this path is `_activate_rendered` → `activate_after_use` (activation counts), plus
  `_touch_internal` and promotion scoring inside `recall`. None of these changes ranking.

**Fix.** Drop "RECALL reconsolidation" from the new row's mechanism. Optionally, report S2(a)'s activation deltas on
carried ids as a descriptive "recall-time strengthening of carried memories" (for example, `d4b1f0ba` +8 in c2).
Do not gate on it.

### S7. File the ranker defect separately, and do not fix it inside this experiment

`_rank_by_relevance` counts stop words (`a`, `with`, `the`, `in`, `you`), keeps punctuation (`guard's`, `garden,`),
and never reads percept text (`observations["transcript"]` / `["cli_input"]`). That puts ~90% of the store at score
0. This is the root cause of D1's lottery and a real recall defect.

- Fixing it before the run changes the subject. That is allowed, since Exp 63 runs on current code, but it is an
  owner decision, and the prereg must say which ranker it tests.
- Either way it gets an issue. The confounding lens's recommendation: **run on the current ranker with R3′**, which
  is valid whichever ranker runs, and file the issue.

---

## NIT

- **N1. Turn attribution under lag.** In O19 c1 phase 1, turn 4's first trace came 25 s after `ENTER turn=4`. Since
  the goal path ignores the trace text, a trace left over from the previous turn ranks identically (same query,
  slightly smaller horizon). R3′ handles this because it uses each trace's own horizon. State it in the prereg so
  the verdict does not try to match `query_text` to the turn.
- **N2. The graph path is unobserved.** In c2 the re-rank explains every activation, so the graph path supplied
  nothing distinguishable. With `memory_paths` logged, report the per-turn graph-path count. A non-zero count is the
  only cue-dependent cross-session recall in this design, and worth a sentence in the outcome even though it does
  not gate.
- **N3. The AUT's `memory_recall` tool is a second channel to carried memories, not gated.** In c2 it added 5
  `tool` activations on carried ids. Report it; do not mix it into R3′.
- **N4. P1 reads the first `enrichment_trace` of the whole log** (`exp10_gates.first_trace`, no goal or turn
  filter). That is fine today, since only the AUT emits traces. With S3, filter it to the AUT too.

---

## What I verified (and how)

- **Only one trace per turn.** O19 c2 phase 2 has 8 `ENTER turn=N` lines and 8 `enrichment_trace` lines, one per
  turn: sizes 72, 89, 105, 109, 115, 222, 262, 334, all with `memories=3` and goal set. Phase 1 has 8 and 8 (the
  first at size 0 reads `memories=0`). c1 phase 1 shows the same pattern.
- **The first phase-2 trace reads `hippocampus_size == N1 == 72`.** All 72 carried ids are in the phase-2 store,
  with unchanged content apart from access and strength fields, and every new memory has `capture_seq ≥ 72`.
- **Carried timestamps < new timestamps** (phase 1 max 1790973655.999, phase 2 new min 1790973735.085).
- **Score distribution** under the ranker's own tokenization: phase 1 {0: 67, 1/6: 4, 2/6: 1}; phase 2 new
  {0: 258, 1/6: 4}. The overlapping tokens are listed per memory above (mostly the token `a`).
- **The offline re-rank reproduces the observed activation deltas exactly** (carried enrichment 11 = 3 + 2 + 6 × 1;
  new 13 = 0 + 1 + 6 × 2; `d4b1f0ba` +8 over 8 traces). This is evidence that R3′ and S2(a) can be computed, and
  that they hold on real data.
- **Code paths.**
  - Narrator wiring: `orchestrator.py` narrator `AgentConfig(with_bio_stack=True, load_persisted=False)`; its
    `run_agentic_loop` passes no pipeline; no caller of `ExecAgent.wire_bio_enrichment`.
  - Pipeline `_agent_id` is set only under `aut_component_registry is not None`.
  - `_query_hippocampus` path order and caps; `_rank_by_relevance` has no floor and breaks ties by recency;
    `capture(record=…)` keeps the record's id with no collision check.
  - The resume restore loads `aut_{hippocampus,nac,ec,atl}.json` (`RESUME_STORES`).

---

## Delta round (v2)

Read: `exp63_carried_recall_prereg.md` DRAFT v2. Under the stop rule, only issues that v2 introduced are listed.

**The folds are correct.** D1 is now R3′, S1 is L, S2 is C5(c), S3 is `agent_id` plus C5(a), S4 is P3, S5 and S6 are
dropped, and N2 and N3 are descriptive.

### The three questions asked

- **R3a's abort path is not likely to burn attempts.**
  - In every observed phase start, the first enrich came before any capture, with a margin of 20 s or more:
    - c2 phase 2: the trace at 1714.93 read size 72 = N1, and the first new capture came at 1735.09;
    - c2 phase 1: the trace at 1400.4 read size 0, and the first capture came at 1422.9;
    - c1 phase 1: the first trace read size 0.
  - The memory-agent captures of percepts lag the enrich by tens of seconds.
  - `capture_seq` is contiguous (0..N−1 in both c2 stores, with no gaps or nulls), so `seq_C = N1 − 1` and the
    equality is well-defined.
  - The real abort risk is the horizon spec gap in S-v2-1 below, not timing.
- **R3′ is not fully specified.** See S-v2-1 and S-v2-2.
- **C5(c) holds with two phases.** It holds exactly on c2 (phase 2's file minus phase 1's file gives 11 carried + 13
  new = 24 = 8 × 3). Every enrich caller renders through `format_thought_response`, and no other code activates
  with `source="enrichment"`. A render that throws after the trace is logged makes C5(c) fail, which aborts the
  attempt (fail-closed).

### SHOULD-FIX (introduced by v2)

- **S-v2-1. `goal_path_horizon` is undefined when the goal path does not run.**
  - It is read "immediately before the goal-path `recall`", but the goal path runs only when the graph path
    returned fewer than 3. Phase 2 loads EC and ATL, and c2 showed `concepts=1` on turns 1–2, so a graph-filled
    trace is possible.
  - On such a trace there is no horizon. If it is the only trace still at `seq_C`, R3a aborts for a reason that is
    not timing.
  - **Fix.** Read the horizon at the entry of `_query_hippocampus`, before the graph path, and log it on every
    trace. Keep the +2 tolerance.
- **S-v2-2. Substring ids cannot be recomputed from the log.**
  - `search_by_content` matches on the full percept text, but the trace logs `query_text[:120]`.
  - The substring path runs only when graph ids overlap the goal path's top 3, which leaves the deduplicated list
    under 3. That is rare, but it is reachable.
  - **Fix (either one).**
    - (a) Take substring ids as logged, as the graph ids are taken. Check only that each is a candidate (seq ≤
      horizon + 2) and that they appear in recency order.
    - (b) Log the full query text.
- **S-v2-3. An L failure has no verdict.** PASS requires L, but neither the FAIL row nor ABORT lists it.
  - L can fail only through a swallowed exception, which is an instrument fault.
  - **Fix.** An L failure aborts the attempt: move it into C5.

### NIT

- **R3a race.** R3a requires all surfaced ids to be in C, so a capture landing between the horizon read and the
  recall's scan would make it FAIL on a race. Judge R3a's ids against the same +2 window as R3′, or say that R3′
  already covers this.
- **Compressed records.** The verdict should refuse if any saved record is a `CompressedMemory`. Compression would
  change the ranking tokens after the trace, making the recomputation invalid. Nothing in-session compresses today
  (no `sleep()`).
