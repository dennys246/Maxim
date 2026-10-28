# Cross-session recall (1.3.4, Session R): approach note

**Status:** v4, as built. Design rounds 1 and 2 are folded (§9). §10 records what building found and the owner
decisions it took.

**Issues:** #1444, #1163, #1064 and #1447, shipped as ONE change, plus the first-turn gate (F1) and the
tool-result channel (F3).

**Red gate:** `tests/unit/test_cross_session_recall_1444.py`, a strict xfail on main since `a59a9e42`.

**Goal:** a fact the user states in session N is answerable in session N+1, on the shipped path, and stored text
that came from outside the agent never reaches a prompt outside an untrusted-content frame.

## Owner decisions (2026-10-24)

- **D1, the ranker floor.** A memory that shares no content token with the query is never returned on a query
  path. Recency breaks ties among scores above zero.
- **D2, legacy reprs on load.** A `ToolOutput(...)` repr is parsed with `ast`, and only `success`, `output` and
  `error` are literal-evaluated. Anything else is dropped, counted and logged once.
- **D3, one tokenizer for all three stores.** Hippocampus, NAc and ATL memorize and recall text through one standard.
- **D4, #1137** ships as its own PR afterwards, with its own minor bump.
- **D5 (F1), user input always passes the PFC gate.** Being addressed is salient, and this is an innate prior.
- **D6 (F3), the tool-result channel is fixed in this PR.** The property holds everywhere, not only in the
  enrichment section.
- **D7 (F10), #1137's PR also carries the user's words on `CompressedMemory`.** It is bounded, and it is one shape
  change with one bump. Until then, a fact survives only until its record is compressed (`compression_age`, 24 h).
  That limit is stated here and in the brief.

## 1. The first turn (D5, F1)

`runtime/thought_gate.py::ThoughtGate.should_think` rejects with "empty working memory" before it scores anything,
and every session's first user input meets an empty working memory. So the restart-then-ask case never enriched.

The fix gives `should_think` a required keyword-only `user_input: bool`.
- When it is true, the gate passes with reason `user input` (innate prior). Refractory still applies, energy still
  applies, and the decision is logged.
- **Call sites:**
  - `agent_loop.py`: a one-line hunk on Session A's file, which is a named hunk. It passes whether
    `_percept_text_for_cycle` came from `cli_input` or a transcript.
  - `exec_agent.py`, two sites: they pass `False`, which leaves them unchanged (Dormant paths).
- Required keyword-only means forgetting it raises `TypeError` (CLAUDE.md, "invariants into types").

## 2. One text-keying standard (D3): `memory/text.py::normalize_tokens`

This needs no new mechanism. `normalize_tokens` is already the ATL's tokenizer on both sides.
- **It is frozen in this PR.** ConceptExtractor persists concept names derived from it, so its quirks are not
  "fixed" here (`used→us`, `status→statu`, `len < 3` dropping "Al" and "TV"). Fixing them would orphan stored
  concepts. The quirks are documented in its docstring.
- **A new helper, `memory/text.py::content_text_tokens(value, *, depth=3)`.** It walks str, dict and list values to
  a bounded depth, skipping the keys `bio_response` and `_skill_name`. It removes frame header and closer lines
  before tokenizing, so frame boilerplate never becomes content, and a past think that recalled an utterance does
  not re-surface it (F8 echo).

**Census, with each site's disposition.** "→ std" means the site moves to `normalize_tokens`.

| Site | After |
|---|---|
| `bio_enrichment._extract_keywords`, which feeds the NAc, the ATL and the hippocampus cue | → std. One cue per enrich |
| `hippocampus_retrieval._rank_by_relevance` | → std over the record's content tokens (§3) |
| `hippocampus_retrieval._compute_association_weight`, goal overlap | → std. Changes the weights of NEW persisted edges only; walked |
| `EpisodicMemory.keywords`, `CompressedMemory.keywords` | → std. No production consumer today (F9). Kept consistent |
| `memory/association_index.py::_extract_keywords` | → std |
| ATL enrichment query | a token index (§4) |
| `NAc.scan_links_for_keywords`, whose only caller is enrichment | the signature side is tokenized at query time: split on `:`, `_` and `-`, then `normalize_tokens`, then intersect. Replaces substring containment, which lemmas break (`berry` vs `tool:eat_berries`, F9). A named hunk in `decisions/nac.py`, a read helper outside selection |

**Excluded on purpose, because they are persisted or hashed keys:**
- the LSH shingles in `context_index.py`;
- `hippocampus_consolidation._context_signature`;
- NAc's `SituationSignature(goal_keywords=…)`.

**Excluded because they are not a store:**
- `runtime/gating._match_learned_aversion` (Wire 2's salience). It has its own ledger rows; a follow-up issue;
- `exec_agent`'s `goal_keywords` (Dormant).

## 3. Hippocampus: record shape, content tokens and the query

**The seam (#1163), in `memory/types.py::Outcome.__post_init__`.**
- A tool-result object (not a Mapping, with `success` and `output` attributes) becomes
  `{"success", "output", "error"}`.
- `output` is kept only if it is JSON-native after a recursive check (sets, tuples and bytes become lists or str),
  else `str(output)`.
- The executor's stamps are not copied: they ride on `EpisodicMemory.encoding`.
- No reader in `src/`, `scripts/` or `docs/experiments` uses the object (grep, confirmed by review F14).

**Load (D2).**
- `Outcome.from_dict` turns a legacy `ToolOutput(` string, capped at 16 KB, into the same shape.
- The parse reads only a top-level `Call` named `ToolOutput`, and only its three keywords.
- On a `SyntaxError` (an `error_kind=<ToolErrorKind.X: 'x'>` or `<object at 0x…>`), it uses `tokenize` to cut at
  the top-level `error_kind=` keyword and retries once.
- A `ValueError` on `output` only (PosixPath, datetime, nan) drops just `output`.
- `RecursionError`, `MemoryError` and the int-digit `ValueError` drop the result.
- The review's census found 40,337 reprs across 162 local stores, all of which parse.

**The format bump.**
- Hippocampus `_format_version` goes from `"1.0"` to `"1.1"`, with `version=` passed in every Hippocampus writer:
  `hippocampus_persistence.save`, and the snapshot path through `wrap_envelope` (F17).
- The loader accepts both.
- A compat fixture holds a legacy-repr record; its test is part of D2's test.

**Matching.** `Hippocampus._memory_matches_query` matches the substring against `content_text_tokens`' text (the
string values, recursively), never `str(dict)`. Dict keys stop matching ("error", "salience", `pain=`). The changed
consumers, `tools/narrative.py`, `simulation/introspection.py` (its `"pain"` count), `simulation/validation.py`
and `exec_agent`, go into the ledger walk.

**Content tokens.** A record method returns `content_text_tokens` over:
- `perception.cli_input`, `transcript` and `observations`;
- the active goal (the intent goal on a compressed record);
- the tool, detected objects and people;
- `decision_rationale`;
- the outcome's `output`/`error`.

It is not memoized at first. A 2,000-record latency probe decides; if a memo is needed, it is `init=False`,
`repr=False`, `compare=False`, and invalidated where records mutate (F14).

**The ranker (#1064, D1).** The score is `|query ∩ record| / |query|`, and a score of 0 is excluded. Recency breaks
ties. An empty query returns nothing.

**The query (#1447).** In `_query_hippocampus_traced`:
1. the graph path;
2. **the cue path**, new: `recall(query=<input text>)`;
3. the goal path;
4. the substring path.

Each memory is listed once, and the cap stays at 3. **Self-recall (F11):** the cue and substring paths skip a
record whose user text normalizes to the current input. The turn's own capture is asynchronous, so it may already
be stored, and it would score 1.0.

**Stated limit (F4):** under a constant goal (Talk's "converse with the user"), the goal path still fills slots
by recency among Talk records, because every one of them scores above 0 on {converse, user}. D1 removes fill for
zero scores only.

## 4. ATL: a token index (D3)

- `ATL._index_concept` also indexes `name_token:<t>`. It is rebuilt on load, so there is no shape change.
- A new filter, `recall(name_token=)`. Enrichment uses it, excluding substrate categories (cluster concepts named
  by EC ids, F16).
- The exact `name:` key and its callers are unchanged.

## 5. The boundary: recalled text is framed where it is produced, and no prompt carries a broken frame

**Rule 1: frame at production.**
- `_summarize_episode` renders the episode's input text (`cli_input`, else `transcript`, else the observation's
  text values).
  - The text is NFKC-normalized and whitespace-collapsed, THEN bounded to 240 (now `memory/text.py::bounded_words`), so
    expansion cannot exceed the bound (F7).
  - It is framed with `frame_tool_output("recalled_input", text, external=True)` and labelled `earlier input:`.
    The label claims no speaker, because sim narration and room speech arrive on the same fields (F12).
  - The outcome's text is framed the same way, at 80 characters.
  - An unrenderable episode still renders "past experience", counted and logged once.
- ExamineTool's `You recall: <sentence>` frames the sentence the same way (D6; a named hunk in
  `tools/narrative.py`, Session V's file).
- **`Tool.untrusted_output`**, a new class attribute (default `False`, declared on the tool, like
  `speech_params`). It is set on ExamineTool and ThinkTool. `_followup_synthetic_input` then frames their follow-ups
  as external (named hunks: `tools/base.py` and `tools/narrative.py`, Session V's; `agent_loop.py`, Session A's).
- `_log_enrichment_contributions` logs memory ids and counts, not frame headers (F18).

**Rule 2: no prompt carries an unbalanced frame (F2, F5).**
- **`content_safety.py::drop_unbalanced_frames(text)`.** It matches hex-id markers only, so `TOOL_OUTPUT_RULE`'s
  literal `id=...` never matches (F5).
  - An opener with no closer: drop from the opener to the end of the text.
  - A closer with no opener: drop from the start of the text through the closer.
  - It returns `(text, n_dropped)`.
- **Applied per item** where a renderer joins items:
  - PromptBuilder's working-memory thoughts, deliberation transcript entries, "Recent Decisions" carryover lines
    (`llm_fallback.py::ReasoningCarryover.to_prompt_line`) and context-pool outcomes
    (`context_pool.py::add_outcome`).
  - So one cut thought never drops the other five.
- **Applied per section** in `PromptBudgeter._fit`, which covers `build` and `build_segmented`, after its own
  truncators. That includes the front-cutters `_truncate_conversation`, `_truncate_context_pool`,
  `_truncate_reasoning_carryover` and `_compact_conversation`.
- **Backstop:** `LLMRouter.generate_json` applies it before the `\x1e` split, and counts and logs any drop. It is
  the real choke point for every model call, including the ACTION_FOLLOWUP f-string branch, exec_agent's
  `generate_json_direct` and `simulation/report.py`.
- **The instruction rule.** When an included section holds a frame, the budgeter prepends `TOOL_OUTPUT_RULE` to it,
  once. It is not baked into the stored enrichment text (F5).

**Inputs the boundary consults:**
1. **The stored text.**
   - Already defanged: the forged closer or marker, zero-width characters, `\x1e`, fullwidth.
   - Confusables, bidi overrides, tag characters and U+2028/2029 survive. The nonce still makes structural forgery
     impossible, so this is a follow-up issue (F7: fail-closed, in the #823 class).
2. **The nonce** comes from `secrets` and is fresh per call.
3. **Cuts at every site**, front or back: rule 2.
4. **Routes around PromptBuilder:** the router backstop.
5. **The legacy parse:** §3.

**Activation (F6).** `_activate_rendered` counts at `format_thought_response`, before any prompt-side cut. This
gap predates this change: truncation already cut rendered memories. Rule 2 widens it, and a follow-up issue
records the gap.

## 6. The gate, strengthened (F4)

- **The fillers go through `talk`** under the same Talk goal, so the goal path's recency favours them and only the
  cue can surface the fact. The bridge's `settle_s` is patched down in the test: timing only.
- **Unit tests for each rule:**
  - each cut site cuts a frame and asserts nothing outside a complete frame survives;
  - the router backstop;
  - D2's compat fixture;
  - the first-turn gate (D5);
  - the tokenizer sites;
  - self-recall.
- Each mechanism is proven by deletion.

## 7. Ledger

- T1-16's named trigger fires: the ranker, `_query_hippocampus` and the record shape. The walk ships with the PR.
  `test_o19_rerun.py::test_the_frozen_ranker_is_the_shipped_ranker` is re-pointed at `52653848`'s ranker (via
  `git show`), never deleted.
- D3 widens the walk to every row that reads:
  - enrichment predictions (NAc);
  - concepts (ATL);
  - hippocampal edge weights;
  - `search_by_content`.
- D5 changes when the PFC gate passes. That reaches every LLM-primary row with user input.

## 8. Not in this change

These become follow-up issues:
- ATL forming concepts from what the user said: a new formation source, on the engram line.
- Spelling variants (colour vs color).
- F7's Unicode hardening.
- F6's activation gap.
- Wire 2's tokenizer.
- F15's raw channels:
  - `_summarize_episode`'s `goal[:40]` (LLM text that can echo the user);
  - ATL names in "Related concepts";
  - PromptBuilder's Dormant `transcript[:80]`.
- Talk's `CWD Context` listing.

## 9. Round 2 folds (v2 verdict SOUND WITH CONDITIONS; these supersede the text above where they differ)

- **N1 (D5, faithfully).**
  - `user_input=True` passes before the refractory and energy checks: "always" means always.
  - The innate-prior branch neither calls `record_escalation` nor moves `_last_pass_tick`, so an unscored pass never
    trains the adaptive threshold.
- **N2: user input comes from the percept's SOURCE, never from which field carries the text.**
  - Text arriving with a cli or voice source is user input.
  - Sim narration that `sim_adapter` puts in `cli_input` is not, so the gate is not disabled for sims.
  - The classification lives in `thought_gate.py` (one function), and the call site passes the observation's
    source.
- **N3: what rule 2 may delete is bounded.**
  - Per-item and per-section drops stay. They are where cut recalled text is removed, inside one item or one
    section.
  - The router backstop moves to `LLMRouter._complete_text`, so every router path is covered, `route()` included.
    It never deletes and never crosses `\x1e` or the system message. It neutralizes stray markers in place (the
    `_FRAME_TOKEN` defang), and counts and logs each one.
  - Raw user-facing inputs are defanged where PromptBuilder renders them: `=== User Request ===`, deferred inputs
    and the follow-up `Original request`. A pasted closer then cannot cut a section.
  - The claim is corrected to "every LLMRouter call".
  - `agents/llm_agent.py::LLMAgent.generate` callers are out of scope: the narrator, `simulation_generator`,
    interactive continuation and the `tool_pain_bridge` reflection. They never carry recalled text, and the
    narrator's raw view of Maxim's reply is the existing LLM-echo class.
- **N4.** `TOOL_OUTPUT_RULE` is added when a framed section is added (`PromptBudgeter.add`), so `_fit` budgets it.
- **N5.**
  - The executed tool's `untrusted_output` is stamped onto the `ActionFollowup` at dispatch, after alias
    resolution, so `reflect`/`research` → `think` carry it.
  - It is set on `think`, `examine` and `memory_recall`.
  - The default stays `False`, so other tools keep their current label. That is the #823 behaviour, and only the
    label differs.
  - The carryover and context-pool `[:100]` cuts also fall under rule 2's per-item drop, so their safety does not
    depend on header length.
- **N6.**
  - `content_text_tokens` removes whole frame regions (`_FRAMED_REGION`, then any residual markers), because every
    frame in a record is derived text.
  - It skips metadata keys (`source`, `modality`, `has_maxim`, `detections_count`, `_skill_name`, `bio_response`).
  - **The self-recall skip** applies in `_add_memory` on every path. It is keyed on `EnrichmentContext.user_input`
    (new; the turn's user text, set at the agent_loop call site; empty in deliberation and ThinkTool calls): a
    record whose input text normalizes to it is skipped.
- **N7 (rule T).** `run_agentic_loop` is pinned at 2,411 lines. The touched gate block (the `should_think` call and
  its decision bookkeeping) moves into `thought_gate.py::gate_percept(...)`, so the function shrinks by more than
  the lines touched. No touch exemption is needed.
- **NITs:**
  - N9: the NAc signature tokens drop the namespace segment before the first `:`, and are memoized per signature.
  - N10: the 24 test call sites of `should_think` add the keyword.
  - N11: `--goal` text stored raw goes on the F15 follow-up list.
  - F6: the brief's activation invariant names the known gap in the same commit.

## 10. What building found (2026-10-24), and the owner decisions it took

- **D5, refined (supersedes §1 and N1).** A pass of the PFC gate also starts multi-cycle deliberation, up to 3
  extra LLM calls. Sims tag narrator and scenario text `cli`, so "a user input always passes" would have multiplied
  LLM calls in every LLM-primary sim.
  - Owner decision: **an addressed input is always ENRICHED.** Recall runs on every user turn, the first after a
    restart included.
  - **Deliberation keeps its scored gate.** `ThoughtGate` is unchanged.
  - The call site logs the gate's real decision.
- **A sixth defect: the user's words were never memorized in Talk.** It is measured and fixed here (owner decision
  D8, folded into this PR so the red gate flips in place).
  - §3's `propose_intent` alone published the input to `MemoryAgent`, and the loop skips §3 while a proposal is
    pending.
  - The single `state.data["cli_input"]` slot was overwritten by the next input, or erased by a non-text sim
    percept.
  - On the robot, the CaptureManager made `propose_intent` return early.
  - The fix: `loop_perception.py::take_user_input` publishes every real turn from every channel once, at arrival.
    This is the experience stream's first producer (docs/plans/experience_stream.md, the 1.4 plan, drafted in
    parallel).
  - `AgentBus.publish` re-raises capture-contract breaks.
- **The deeper direction (owner, 2026-10-24).** Memory should be a free-flowing component, not tied to loop state or
  stages. A four-lens parallel review (write side, read side, bio-faithful, architecture) shaped the 1.4 plan. This
  PR is phase 0 of it.
- **Found, not R's, to be filed:** a sixth identical Talk reply is suppressed by a repeat guard, and that turn then
  times out silently after the full timeout.

## 11. The three-lens review (2026-10-25), and where the build differs from §1–§9

The verdicts were:
- Executor: SOUND WITH CONDITIONS.
- Wire: SOUND WITH CONDITIONS.
- Architecture: NOT READY, on T1-16.

All findings were folded into this branch.

**Owner decisions:**
- **T1-16 goes STALE**, re-run owed. This is the strict option. The judge must be rebuilt for the cue path first.
  A STALE row blocks the next release (1.3.4) until that re-run, or until a committed machine-readable exception
  at the cut (M25).
- **A person's own "yes" is memorized.** The loop's own replies are not: they move to `LOOP_REPLY_SLOT`.

**How the build differs from the text above:**
- **The input is read once, at the top of §1.2** (`take_user_input`), not in §1.6. §1.2 enriches the same text, so the
  pending voice, CLI and API channels are recalled for, not only memorized. The classifier is
  `loop_perception.py::_addresses`, not one in `thought_gate.py`, which is untouched (N2/N7). Rule T is paid by
  extractions in `agent_loop.py` and `tool_dispatch.py`.
- **The runtime clears the input slot**, not the agent (ARCHITECTURE.md: agents do not mutate state).
  `agents/base.py::Agent` declares a default `publish_arrival`. `maxim.api` runs duck-typed agents, so the call is a
  capability probe.
- **Keys are idempotent.** `key_tokens` and `settle_key` sit on top of the frozen `normalize_tokens`, which is not
  idempotent. NAc settles the keys it is handed, and does not re-tokenize them. NAc keeps the namespace segment (N9's
  drop would lose `look_at`), and its docstring says so.
- **There is one text-field list, `EpisodicMemory.text_fields`**, used by search, `matching_sentence` and the ranker.
  Content keys are computed once per record, at capture and load, outside the lock. On a 2,000-record store the first
  enrichment takes 7 ms; it used to take about 900 ms per query.
- **The ATL substrate exclusion keys on `category == "substrate"`**, which compression keeps.
- **Self-recall compares NFKC case-folded text.** Deliberation and thoughts inherit the turn's words.
- **The P3.5 envelope stays at schema 1** (§3 and F17 said `version=` passes through `wrap_envelope`). Its payload reads
  back through `Outcome.from_dict`'s parse, as `hippocampus_persistence.py` documents.
- **The legacy-repr compat test lives in `tests/unit/test_recall_memory_seams_1444.py`.**
  `tests/integration/test_persistence_compat.py` pins that the 0.8 fixture re-saves as 1.1.
- **New guards** close the deletion probes that stayed green:
  - the router seam;
  - the follow-up consumer of `untrusted_output`;
  - the transcript per-item drop;
  - the three defang call sites;
  - a carryover cut that really splits a frame;
  - frozen-tokenizer golden vectors.

  The recall-only half of D5 is pinned by #1178's strict L5 gate, which XPASSes if reflexes run.

**Delta round 1 (2026-10-25)** found more, all folded with guards:
- **B1:** the mypy ratchet: `content_tokens` returns a frozenset.
- **B2:** the one text-field list must be the superset. The whole outcome result is searched: Reflexion
  reflections, `stdout`, pain contexts.
- **S1:** the controller's confirmation and timeout pops dropped a person's input that was waiting behind a loop
  reply. They are removed, because `take_user_input` has already read and popped the tick's slots.
- **S2:** the run mode is seeded before the arrival percept.
- **S2b:** the PFC gate reads only the observation's text, as before. A mailbox input is enriched for recall only,
  so no new deliberation starts (D5).
- **S3:** memory hears the `cli_input` the loop processes, not a transcript beside it.
- **N1:** the content cache holds the field objects, never their `id()`.

## 12. #1137 (owner decision 2026-10-25: in 1.3.4, Session R's next PR after #1465)

Without this change the 1.3.4 recall fix stopped working once a record was compressed, which for a user's words
under `access_based` is 3 × `compression_age`, about 3 days. It ships as a release condition for the recall claim.

- **`CompressedMemory.active_goal`** keeps the episode's context goal beside the intent `goal`. `record_active_goal`
  reads it on both kinds. `_memory_summary` and `_compute_association_weight` now read the active goal.
- **`CompressedMemory.heard`** keeps what the episode heard: NFKC-normalized, whitespace-collapsed and bounded to 240
  characters, the render bound. It is stored raw, and `input_text` returns it, so recall frames it at render exactly
  as it frames an episode's words.
- **The format:** `hippocampus.json` is 1.2. A 1.1 file loads, and a compressed record keeps neither field
  (fixture: `tests/fixtures/persistence_1_1/hippocampus_1_1.json`).
- **The red gate:** `tests/unit/test_compressed_recall_1137.py` (strict xfail at `c79ef22b`, flipped in place). It
  extends #1444's gate with forced compression, and a non-vacuity test pins that the record is compressed. Deleting
  `heard` or `input_text` turns it red.
- **Not changed:** `bridges/planning_bridge.py` still skips compressed records (its template key needs an action),
  and the Dormant `integration/recall.py::EpisodicRecallSource` reads `record_active_goal` as before.

**#1137's three-lens review (2026-10-25), folded.** Executor+Architecture found one blocker; Wire found no blocker.
- **The blocker:** `CompressedMemory.content_tokens` was uncached. An old store is mostly compressed records, so on
  2,000 of them each query took about 0.53 s, the SF5 regression again. They are now cached, and keyed at
  compression and at load: about 5 ms per query.
- **SF:** compressed words reached the ranker but not substring search or ExamineTool. `CompressedMemory.text_fields`
  is now `[heard]`.
- **SF:** `active_goal=None` meant both "pre-1.2" and "the episode had none". The second case is now stored as `""`.
- **SF:** the new lines had no tests (round trip, bound, the two readers). They are tested now.
- **NIT:** one bound, `memory/text.py::bounded_words` / `RECALLED_WORDS_MAX_CHARS`, shared by the render and
  compression.
- **NIT:** `heard` is kept out of the repr.
- **NIT:** the stale size claim and `to_context_dict`'s goal are fixed, and the downgrade note is in
  `hippocampus_persistence.py` and `docs/user/upgrading.md`.
