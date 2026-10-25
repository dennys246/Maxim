# Cross-session recall (1.3.4, Session R): approach note

**Status:** DRAFT v2. The adversarial design pass, round 1, is folded (verdict on v1: NOT SOUND on F1). This version
goes to the delta round.

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
  - The text is NFKC-normalized and whitespace-collapsed, THEN bounded to `_RECALLED_TEXT_MAX_CHARS = 240`, so
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
