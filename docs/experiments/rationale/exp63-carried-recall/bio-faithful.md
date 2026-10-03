# Exp 63: BIO-FAITHFUL lens (design review, 2026-10-03)

Charter: [DESIGN_REVIEW.md](../../DESIGN_REVIEW.md). Question: does the design test the mechanism's real
job, not a caricature? Do the gates credit only the named mechanism, and do they demand only what the
biology they invoke predicts? I reviewed `docs/experiments/exp63_carried_recall_prereg.md` (DRAFT) against
`docs/agents/bio-memory.md`, the original `10_cross_session_enrichment.md`, the T1-1 ledger row, the code,
and the committed Exp 10 stores. I made no code changes.

**On the evidence used here.** Two sets of committed bytes characterize the mechanism below:
`data/rerun_exp10_2026-09-27/` (typed aborts) and `data/rerun_exp10_o19c2/` (an ABORTED attempt). They
are used ONLY to show what the shipped retrieval code does with real sim stores. They are not evidence for or
against the claim, and nothing below asks them to move a row (weak-evidence rule).

## Verified first (file::symbol)

**The retrieval path the gates read.** `integration/bio_enrichment.py::BioEnrichmentPipeline._query_hippocampus`
tries three paths in order and returns `summaries[:3]`:
1. **Graph path.** `EC.pattern_complete_readonly` → `Hippocampus.retrieve_on_cue` → ATL concept name →
   `Hippocampus.recall(object_detected=name, limit=2)`. This path is dead in this sim: `detected_objects` is
   empty on every memory in every committed Exp 10 store I checked (0 of 334 in c2 phase 2).
2. **Goal path**, run if fewer than 3 were found: `Hippocampus.recall(query=goal, limit=5)`, then the first
   3 are kept.
3. **Substring path**, run if still fewer than 3.

**The goal-path ranker.** `memory/hippocampus_retrieval.py::_rank_by_relevance` scores each memory by
`|query_tokens ∩ memory_tokens| / |query_tokens|`. Tokens come from a whitespace split with no punctuation
strip and no stopwords, so "a" and "with" count and "guard's" does not match "guard". Memory tokens come
from `context.active_goal`, `action.tool_name`, detected objects and people, `perception.observations["text"]`
and `decision_rationale`. Ties sort by **recency**, and there is **no relevance floor**: score-0 memories are
returned.

**Most of the store is invisible to the cue.** Loop captures put their text in `observations["transcript"]` /
`["cli_input"]`. The ranker reads `observations["text"]`, and only `Hippocampus.store_observation` writes
that key. In every committed Exp 10 store, `"text" in observations` is false for every memory. Only the
few action memories that carry an LLM plan as `active_goal` can score above 0 (2 of 100 in the 09-27
baseline, 15 of 334 in c2 phase 2). Everything else ties at 0.0 and is ordered by recency. I recomputed
the ranker over the saved c2 phase-2 store (334 memories, 72 carried) for the dungeon goal:

| score | carried | own |
|---|---|---|
| 0.33 | 1 | 0 |
| 0.17 | 4 | 4 |
| 0.0 | 67 | 258 |

The top carried memory scores 0.33 on "a" + "with" + "guard"-style overlap, not on content. The saved
bytes agree: across that phase, `activation_sources["enrichment"]` grew on just 3 of 72 carried memories
(+8, +2, +1), while own memories got 13 renders. One weakly matching carried plan memory took the slot
turn after turn.

**What recall does to a memory (the "reconsolidation" question).** `RetrievalMixin.recall`:
- Under the write lock it calls `_touch_internal` on EVERY result, including the 2 of 5 the goal path
  never renders. That sets `accessed_at = time.time()` and does `access_count += 1`.
- For a `query` recall, `_score_and_maybe_promote_batch` decays `promotion_pressure` by wall-clock time,
  appends the query's hash to `access_contexts` (a query already in the deque is not credited again), adds
  `_compute_access_score` (salience + frequency), and promotes SHORT_TERM → LONG_TERM at 3.0.
- Separately, `BioEnrichmentPipeline._activate_rendered` runs at **format** time, not at enrich time. It
  calls `activate_after_use(..., source="enrichment")` on the 3 rendered ids, which bumps
  `activation_count` / `activation_sources`. Under `memory.strategy=strength` only, it also updates
  `(storage_strength, retrievability_anchor_us)`.
- **Content is never mutated.** The brief says so (`Hippocampus.recall()` invariant), and the 09-27 README
  confirmed by diff that on all 100 carried memories only access bookkeeping changed.

All of those fields persist in `aut_hippocampus.json`. I diffed the 09-27 baseline against its phase-2
resume. Exactly 5 carried memories changed: the limit-5 goal recall touched them, and
`activation_sources.enrichment` grew on 3 of them, the 3 rendered. Every change was in `accessed_at`,
`access_count`, `access_contexts`, `promotion_pressure`, `last_scored_at`, `activation_*`.

**No forgetting pass runs.** The generic `--sim` loop ends with `on_session_end_lightweight` (no `sleep()`;
brief, "NAc decay is tick-anchored" bullet). `HippocampusConfig.max_nodes` is 10,000, so store-time
eviction cannot fire at these sizes. P2 ("every carried id survives") is therefore guaranteed by
construction whenever the load works. That makes it an engineering check of the load, not a prediction
about memory surviving consolidation.

## DO-NOT-BUILD

### B1. R3 "≥1 carried memory on every resume turn" asks for something neither the biology nor the code predicts, so its outcome does not measure persistence (answers open question 3)

**Biology.** Hippocampal retrieval depends on the cue (encoding specificity; pattern completion from a
partial cue). New episodes formed in the same context compete with old ones (retroactive interference,
recency). The faithful prediction for a resumed session in the same context is:
- carried memories are **available**: stored and reloaded;
- they are **accessible when the cue favours them**;
- their share of what is retrieved **falls** as same-context competitors build up.

Nothing in that biology predicts that an old episode wins a retrieval slot on every turn. Requiring it
over-demands.

**Code.** Here the "cue" reaches 2–5% of the store. The rest ties at 0 and is ranked by recency, and
recency favours phase 2's own memories by definition. So R3 on turns 2–8 passes or fails on one thing:
does some carried plan-text memory beat every newer plan-text memory on an overlap score made mostly of
function words? That depends on the LLM's wording and on whitespace tokenization, not on persistence.

- A **FAIL** would be read as "carried recall failed" while persistence and retrieval both worked to spec.
- A **PASS** credits a function-word coincidence: one carried memory scoring 0.33 on "a"/"with" (the c2
  picture above).

**Turn 1 is different.** On the first trace, which C5 already pins as `hippocampus_size == N1`, only
carried memories exist. Surfacing ≥3 of them there is the known-answer test that the resumed store is
reachable by the live recall path. That is the faithful core of the claim, and it should gate.

**Proposed replacement**, all computable from the logged `memory_ids` plus committed bytes:
- **R3a (reachability, gating).** Phase 2's first trace surfaces ≥ 3 ids, all in C. This merges C5's
  known answer into a gate: it shows the reloaded memories are retrievable, not only present.
- **R3b (carried memories compete on the same terms as new ones, gating).** On every phase-2 turn whose
  first trace surfaces no id in C, every surfaced goal-path id must score ≥ `max_{c ∈ C} score(c, goal)`,
  with equality allowed (the declared recency tiebreak hands ties to newer memories).
  - The score is `_rank_by_relevance`'s overlap fraction. The harness must call the shipped function, or
    pass the logged score through; it must not re-implement the ranker.
  - `max_{c∈C} score` is a static property of phase 1's saved store, because carried content is immutable
    (see B2 / P3).
  - This is the faithful form of "carried memory is recalled". A persisted memory loses a slot only to a
    memory at least as relevant to the cue, never because it came from an earlier session. A broken load,
    a second-class carried memory, or an origin-dependent penalty all fail it. Losing to a better or newer
    match does not.
  - It needs one field beyond `memory_ids`: the path (`graph` / `goal` / `substring`) and the ranker score
    per surfaced id. That is logging, not a pipeline change.
- **R3c (descriptive).** Per turn, `|memory_ids ∩ C|` and the carried share (already planned), plus the
  turn on which the carried share first reaches 0. Report it; it must never gate.

If the owner wants a simpler gate than R3b, the faithful fallback is R3a alone plus R3c descriptive. Keep
R3 as drafted only if the claim is narrowed to "goal-keyword recall with recency fill", and state that.

## SHOULD-FIX

### S1. Drop "RECALL reconsolidation" from the mechanism. Measure what recall really does, and name it honestly (answers open question 4)

**Biology.** Reconsolidation (Nader et al. 2000) means a retrieved memory becomes labile and is
re-stabilized, and its content can be updated or disrupted in that window.

**Code.** None of that exists. Content is immutable on access, and the brief states it as an invariant.
What recall does is **retrieval bookkeeping**:
- `access_count` and `accessed_at` on every returned result;
- `activation_*` on rendered ones;
- **use-dependent promotion pressure** that can move SHORT_TERM → LONG_TERM. That is closer to
  retrieval-practice strengthening and systems-consolidation tagging than to reconsolidation.

Naming it "reconsolidation" credits a mechanism the code deliberately does not have. Do not gate on it
under that name. A new row claiming "retrieval strengthens a carried memory" would be a new mechanism
claim and needs its own experiment.

What committed bytes CAN show, as gates that check the claim's "reloaded exactly" half:
- **P3 (content identity, gating).** For every id in C, every field outside a declared
  retrieval-bookkeeping set is byte-identical between phase 1's saved store and phase 2's (and phase 3's).
  The bookkeeping set is `accessed_at`, `access_count`, `access_contexts`, `promotion_pressure`,
  `last_scored_at`, `long_term`, `consolidated_at`, `activation_count`, `activation_sources`, and under
  `strength` only `storage_strength` / `retrievability_anchor_us`.
  - This makes "reloaded exactly" precise.
  - It observes the "content is not mutated on access" invariant on the real path.
  - B1/R3b relies on it.
  - The 09-27 README did this by hand. Make it a gate.
- **P4 (bookkeeping only where retrieval happened; descriptive, or an instrument check folded into C5).**
  A carried id whose bookkeeping changed must have been returned by some `recall()`. And a carried id's
  `activation_sources["enrichment"]` delta must be ≤ the number of phase-2 traces whose `memory_ids` list
  it. The bound is ≤ and not =, because activation happens at format time: a surfaced but unrendered id
  is not counted, and that difference is the brief's "surfaced-but-unshown is not a use". This gives a
  second, independent instrument for the logged ids, from bytes the logger did not write.

**Wording.** The T1-1 mechanism "Hippocampus persistence + RECALL reconsolidation" becomes:
> Hippocampus persistence (save/restore of `aut_hippocampus.json`) + goal-keyword retrieval
> (`_query_hippocampus` → `recall(query=goal)`); retrieval updates access bookkeeping and never content.

### S2. R2 (≥3 memories every turn) is guaranteed by construction, so it credits nothing

The goal path has no relevance floor and returns the top 3 of any store with ≥ 3 memories, all scores
included. So P0 (N1 ≥ 3) + R1 (a non-empty goal) ⇒ R2, unless an exception is swallowed (the `except` in
`_query_hippocampus` returns `[]`). R2 therefore tests "no exception", not recall. Biology would expect
retrieval to fail to fire under a poor cue, so "always 3" is cap-filling, not a sign of healthy recall.
- Keep R2 only as an instrument or liveness check: name it that way and move it next to C5.
- Or drop it as a gate.
- Do not cite it as recall evidence in the ledger row.

### S3. The claim's word "recall" needs its operational meaning stated, including the two retrieval gaps the review found

The prereg should say plainly that retrieval in this sim is goal-keyword overlap with a recency fill.
- The graph path is dead (no `detected_objects`).
- About 95%+ of captures cannot be reached by the cue, because the ranker reads `observations["text"]`
  and loop captures write `transcript`/`cli_input`.

Otherwise a PASS reads as evidence of cue-dependent, pattern-completion recall, which is not what ran.

That second gap looks like a real defect: a key mismatch between the capture and retrieval sides. Per the
no-band-aid rule it is **not** this experiment's to fix. It should go to the owner as an issue, root
cause first: which side is wrong, and does any other `observations["text"]` reader have the same gap?
The prereg should add `_rank_by_relevance` / the capture-side observation keys to the new row's
`Re-run on:` triggers, because fixing them will change which memories surface.

### S4. Phase 3 as designed cannot show negative transfer or cue specificity (answers open question 5 from this lens)

Biology predicts that a garden cue should disfavour dungeon episodes. The code cannot express that:
- On garden turn 1 only carried (dungeon) memories exist and there is no floor, so the carried share is
  100% by construction.
- On later turns everything ties near 0 ("a", "the", "in", "you" overlap), so recency decides and the
  share is ~0 by construction.

The planned `|memory_ids ∩ C| / |memory_ids|` therefore measures turn order, not transfer. If phase 3
stays, report each surfaced carried memory's overlap score against the garden query next to it. That
makes the missing relevance floor visible, which is the honest descriptive finding. Otherwise cut the
phase and save the run time. I lean to cutting it: it gates nothing, and its headline number is
pre-determined.

### S5. The claim sentence should say storage persistence, not survival through consolidation

No `sleep()` runs between or within these sessions, and nothing can be evicted at this store size. So the
experiment shows that the store is saved and reloaded and the live path can reach it. It does not show
that a memory survives a consolidation or forgetting pass, which is what "the hippocampus carries memory
across sessions" means biologically.

Suggested claim:
> A resumed session reloads the earlier session's hippocampal store with content intact, the live
> enrichment path retrieves those memories when they are the best match for its cue, and a carried memory
> loses a retrieval slot only to an equally or more relevant memory.

Survival through `sleep()` belongs to the memory-strength line, not this row.

## NIT

- **N1.** "in the order the prompt receives them" (Instrumentation) slightly overstates. The trace is
  emitted in `enrich()` and the prompt receives the result at `format` (`_activate_rendered`), where the
  section can still be dropped. Say "the order `_query_hippocampus` returned them".
- **N2.** `accessed_at` and `promotion_pressure` decay run on wall-clock time, so the downtime between
  phase 1 and phase 2 counts as disuse. With no `sleep()` this has no effect here. Note it so a later
  variant that adds consolidation does not mistake operator scheduling for a memory effect.
- **N3.** Phase 1's store already has many LONG_TERM memories (46 of 100 in the 09-27 baseline), from
  in-session recall pressure. Report the LONG_TERM count among C. It explains which carried memories a
  future consolidation pass would protect, and costs nothing.
- **N4.** The C5 known answer ("on phase 2's first trace … every surfaced id is a phase-1 id") is sound and
  bio-neutral. Under B1 it becomes R3a's gate, so keep C5 as the instrument check (length, membership) and
  move the known answer to R3a. Then an instrument fault and a claim failure stay distinguishable.

## Summary of what changes in the prereg if all are folded

- Claim narrowed (S5).
- Mechanism renamed: no "reconsolidation" (S1).
- Gates: P0–P2 kept; **P3 content identity** added; R1 kept; R2 demoted to liveness (S2); **R3 replaced
  by R3a (turn-1 reachability) + R3b (origin-blind competition) + R3c descriptive** (B1).
- Instrumentation: per-id path + ranker score added beside `memory_ids` (B1); `activation_sources` delta
  added as a byte-level cross-check (S1/P4).
- Phase 3 cut or reframed (S4).
- The capture/retrieval observation-key mismatch filed to the owner as a separate issue (S3).

## Delta round (v2)

Scope: prereg v2 against this file's findings, with the owner decisions of 2026-10-03 (R3′ conformance in
place of R3b, #1064 a re-run trigger, phase 3 dropped). Stop rule: only findings caused by v2, plus NITs.

**The folds check out.**
- The claim is narrowed to storage persistence into recall, and it states the recall it means (S5/S3).
- Reconsolidation is dropped from the claim and the reason is stated (S1).
- R2 is now liveness L, labelled as implied (S2).
- R3a is the turn-1 known answer. It is keyed on `goal_path_horizon == seq_C`, which is stricter than my
  `hippocampus_size` version (B1/N4).
- "Logged at enrich time" is in (N1).
- C5(c) is my P4, made an equality. That is correct: rendering activates exactly `memory_ids[:3]` through
  `_activate_rendered`. It assumes every AUT trace is formatted exactly once. If the budgeter drops the section,
  C5(c) aborts the run rather than wrongly failing the claim, which is the right direction.

**R3′ catches everything R3b did, with one stated exception.**
- R3′ recomputes from saved bytes. P3 pins those bytes to phase 1's content. So if a carried record was
  degraded at load, R3′ (or P3) fails.
- The ranker reads only `context.active_goal`, `tool_name`, the observation fields and `timestamp`. All of
  those are immutable under P3, so ranking the end-of-phase store does not drift.
- The exception is a rule inside the ranker that favours or penalises a memory by origin. That would conform,
  so R3′ would pass it, where R3b would have failed it. At the executed code there is no such rule (the score
  ignores time except the recency tiebreak), and ranker changes are re-run triggers. **NIT D1:** add one
  sentence: "full candidates" means scored by the executed ranker, which reads no origin and no age except
  the recency tiebreak. The claim then states what R3′ does not test.

**SHOULD-FIX D2 (caused by v2): P3's exemption list includes two fields that cannot legitimately change in
this run.**
- `encoding_tag` is stamped at capture and never recomputed (`Hippocampus._stamp_encoding_strength`
  docstring: "the record is history").
- `retro_tag` is written only by `_resolve_retro_tags_locked`, which runs only in `sleep`, and no `sleep`
  runs here.
- Exempting them lets a load that drops either one pass "reloaded exactly". Move both into the
  byte-equal set.
- `storage_strength` / `retrievability_anchor_us` change only under `memory.strategy=strength`. Exempting
  them only when the reports show that strategy is a NIT (D3).

**The claim text is faithful,** with D1 added.
