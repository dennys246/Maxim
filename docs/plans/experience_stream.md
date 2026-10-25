# Experience stream: memory that listens instead of being polled (1.4 plan)

> **Status: DRAFT v1, written 2026-10-24.**
> - **Owner decision, 2026-10-24: two phases.** 1.3.4 ships Session R's recall fix together with "publish the
>   user's input at arrival", which becomes this stream's first producer (Phase 0, below). This plan covers the
>   rest.
> - **No code before the four-lens design review** (§11).
> - Every mechanism here enters as `[engineering]`, behind a flag that defaults OFF, with today's goldens pinned
>   byte-identical while the flag is off.
> - Later phases are gated on Session A's 1.3.5/1.3.6 `agent_loop` slices.

**Sources.** This plan stands on four read-only reviews run on 2026-10-24 against the Session R worktree (HEAD
`9f9545cb` plus uncommitted edits): a write-side map, a read-side map, a bio-faithful lens and an architecture
lens. It also builds on the recall work's approach note, [recall_1_3_4.md](recall_1_3_4.md).

**Companion plans:**
- [thalamic_relay.md](thalamic_relay.md) (GL3): the relay this plan rides.
- [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md): the encoding tag and the activation rule.
- [grounding.md](grounding.md): tracks, bio-mapped.
- [lookback_primitive.md](lookback_primitive.md): "no new store".
- [latent_forward_model.md](latent_forward_model.md): the Hippocampus trace is the replay buffer.

## 1. Why this plan exists

**The owner's goal.** Memory should not depend on loop state or on a single stage of the loop. Information
should flow into memory as it happens, and recall should flow out wherever it is cued.

Today memory is the opposite. It is read and written at fixed sections of
`runtime/agent_loop.py::run_agentic_loop`, so whether something is remembered depends on which section happened
to run on which tick.

### 1.1 Measured evidence

**Write side.** The probe was a real `MaximHandle` Talk loop with a fake model and four user utterances
(scratch script `probe_talk_writes5.py`).
- The run made 4 Hippocampus captures, all of them `respond` loop traces.
- **None of the 4 utterances landed in `perception.cli_input`.**
- `PerceptionAgent.propose_intent`, the only producer that publishes the user's words to `MemoryAgent`, was
  called 0 times.
- The words survived only inside `context.state_ref`. That is a deep copy held in `Hippocampus._state_store`
  (`memory/state_store.py::StateStore`), an in-RAM LRU that nothing saves and nothing indexes.

**Read side.** The red gate `tests/unit/test_cross_session_recall_1444.py` showed that a fact stated in session
N never reaches the prompt in session N+1. Four defects in the recall path compose into that one symptom
(#1444, #1163, #1064, #1447). Session R's design pass found two more:
- the first turn after a restart is never enriched;
- the user's words are never captured (§1.1, above).

### 1.2 The stage and state couplings (write side)

| Coupling | Where | What it loses |
|---|---|---|
| A percept reaches `MemoryAgent` only when §3 runs | `run_agentic_loop` §3: `ctrl.pending_proposal is None and not has_pending_llm_input` | In Talk every turn has a pending proposal, so no utterance is ever published |
| One input slot | `state.data["cli_input"]`, merged by `runtime/state.py::RuntimeState.update` | The next input overwrites it. A non-text sim percept writes `cli_input: None` over it (`runtime/sim_adapter.py::SimulationAdapter.next_observation`) |
| The capture manager returns early | `agents/perception_agent.py::PerceptionAgent.propose_intent` returns when `_capture_manager is not None` | On the robot (vision or audio) typed input never becomes a Percept. *Read from the code, not probed* |
| The action capture records the wrong tick | `runtime/bio_integration.py::capture_loop_action` uses the tick's `observation`, which is `{}` on the proposal tick | Action records do not carry the words that caused them |
| Single-slot substrate nodes | `bio_integration._latest_substrate_nodes` | Each percept overwrites the last; only a tool execution consumes it |
| Reactions need an open episode | `Hippocampus.capture_reaction` needs `_pending_episode` | Reactions before the first action are dropped |
| Encoders ignore words | `memory/concept_extractor.py::ConceptExtractor._process_capture`; `MemoryHub._on_memory_captured` (EC semantic) | The user's words and the agent's replies never become concepts or embeddings, even when captured (W12/W13: gaps by design) |
| The bus swallows errors | `agents/bus.py::AgentBus.publish` logs handler errors at DEBUG | `EncodingContractError` raised in a bus handler is silently dropped, which breaks "the capture contract fails loud" |

### 1.3 The couplings on the read side

- **One cue per tick.** Recall cues on `loop_perception.py::percept_text(observation)` at §1.2, before the model
  answers. Text that arrives later in the tick never cues recall: tool results, follow-ups, and the model's own
  reply.
- **One overwritten prompt slot.** `StructuredContext.bio_enrichment_context` is rebuilt every tick. Recalled
  text survives only as truncated copies: the working-memory THOUGHT entry (`[:500]`), the deliberation
  transcript (`[:800]`), the thoughts list (`[:200]`) and the carryover (`[:100]`).
- **Stores are queried side by side, never associatively.** Hippocampus, NAc and ATL each receive the same
  keyword list, and one store's results never cue another. Graph spreading is inert, because
  `memory/episode.py::apply_hebbian_on_close` is Dormant.
- **Producers with no consumer.**
  - The 2S-d situation cue (`memory/pattern_completer.py::PatternCompleter.cue_situation`) runs, but its ids are
    discarded.
  - `PatternCompleter.complete` runs and records an activation, but only Dormant readers consume its output.
- **Tools are the only other cue.** `ThinkTool`, `ExamineTool` and `memory_recall` cue on whatever the model
  chooses to recall. Nothing else does.

## 2. Front gate: ride what exists

**Does this need to be its own mechanism? Mostly no.**
- **Transport: the AgentBus** (`agents/bus.py::AgentBus`). It already carries `Percept`, `ProposedGoal` and
  `GoalCompleted`. `comms/conversation.py` already publishes its archive Percept without the loop, which is the
  working model for "free-flowing".
- **Source of afferent events: GL3's relay** ([thalamic_relay.md](thalamic_relay.md)). It already turns receptor
  output into `AfferentEvent`s carrying a deterministic `PhysicalEventId`. **The Hippocampus is not one of its
  destinations today, and adding it is this plan's main contribution to GL3.**
- **Encoding door: the Hippocampus async capture path** (`memory/hippocampus.py::Hippocampus.capture_from_loop_async`).
  It already stamps time and sequence at enqueue, is bounded, has a `flush` barrier, and runs on one worker
  thread.
- **The post-capture fan-out stays as it is**: `MemoryHub` capture callbacks feed SCN, ConceptExtractor and EC.
- **Recall protocol: `integration/recall.py::RecallSource`**, the existing Console protocol, with
  `curate_recall`.
- **Active-memory buffer: `agents/working_memory.py`'s `WorkingMemorySet`**, with its existing `RECALL` kind.
  Today `RECALL` is written by `Hippocampus.recall(working_memory=)` and no prompt reads it.

**Not used:**
- **No new bus.** The archived unified event bus plan was absorbed into ReactionBus.
- **No revival of `memory/percept_trace_buffer.py`.** It stays Dormant, and CI forbids constructing it in
  production.
- **ReactionBus and PainBus stay evaluative-only.** Their refractory windows would drop a repeated "yes", and
  any change to them fires T3-9.

**Two things are genuinely new.**
1. **The `ExperienceEvent` type.** Percept is mutable, has no id and is built three times from one observation
   (`agent_loop` §1.5's `make_text_percept`, `PerceptionAgent.process_observation`, and MemoryAgent's own
   dict). None of those can carry `caused_by` or measured encoding signals.
2. **The cue-driven recall service.** Call-and-return enrichment can only cue on the one text it is handed.

## 3. Biological grounding (what is faithful, what is contested)

- **Encoding is continuous and indexical.** The hippocampus indexes cortical activity (Teyler & DiScenna; Teyler
  & Rudy). Encoding is the automatic recording of *attended* experience (Morris), chunked at event boundaries
  (Zacks; Ben-Yakov & Henson).
- **Encoding is graded, not gated.** Subsequent-memory effects (Wagner; Brewer), divided attention (Craik),
  neuromodulatory tagging (Lisman & Grace; McGaugh), and synaptic tagging and capture (Frey & Morris) all show
  graded strength. Truly unattended content lands near a floor. **"Graded, with a floor" is the honest model.**
  `memory/encoding.py::EncodingSignals` and the strength plan's `S0 = s_base·(1+k·tag)` already have this shape.
- **Retrieval is cue-driven.** CA3 pattern completion works from any overlapping active representation (Marr;
  McNaughton & Morris; Rolls; Tulving's ecphory). The PFC *monitors* and *searches* strategically (Moscovitch);
  it does not decide whether automatic retrieval happens.
- **The gate belongs at working-memory entry** (PBWM; O'Reilly & Frank), not at encoding or retrieval. Working
  memory holds about four items (Cowan).
- **Consolidation stays separate from encoding.** Complementary learning systems (McClelland, McNaughton &
  O'Reilly) and sleep replay (Wilson & McNaughton; Girardeau) keep a fast, labile tier apart from the slow one.
  - **Keep FORMING as the labile tier.** Do not flow past it (#817).
- **Contested, and not leaned on:**
  - the boundary conditions of reconsolidation;
  - prioritized replay (Mattar & Daw);
  - predictive processing as a whole framework. The narrow claim stands: prediction error raises encoding.

## 4. The target design

### 4.1 `ExperienceEvent`

This is a frozen dataclass on CC3 path (a): every field has a default, plus an
`extra: dict = field(default_factory=dict, hash=False, compare=False)`.

| Field | Meaning |
|---|---|
| `experience_id` | From `utils/seeding.py::stable_hash_64_signed` over (agent, session, seq), never builtin `hash()`. **It joins to GL3's `PhysicalEventId` when one exists**; the open decision is which identity wins (§10) |
| `kind` | `utterance_in`, `utterance_out`, `percept`, `action_outcome`, `thought`, `reaction` |
| `source` / `addressed` | The percept source. `addressed` uses Session R's one classifier (`runtime/loop_perception.py::addressed_text` / `ADDRESSED_SOURCES`) |
| `provenance` | `experienced` / `narrated` / `imagined` / `apparatus`, GL3's G6/G16 kinds. This is source monitoring |
| `text` | Bounded. External text is stored raw but is only ever rendered framed (§6) |
| `encoding: EncodingSignals` | **Required.** What the source measured. No subscriber may invent a constant (the existing capture contract, moved onto the event) |
| `experience_us` / `capture_seq` | Stamped at emit, on the experience clock |
| `caused_by` | `experience_id \| None`. An action outcome points at the input that triggered it, and a reply points at its question. This is the structural fix for "action records lack the user's words" |

### 4.2 Publish once, at the source

Every producer emits exactly once, where the information arises:

| Producer | Emit site |
|---|---|
| The user's input | Phase 0: where the observation is taken (`loop_perception.py::take_observation`, or `simulation/conversational_source.py::ConversationalSource`). It is published once, then consumed |
| The agent's utterance | The executed speech tool (read `Tool.speech_params` off the tool the call ran, as `InstrumentedExecutor` does) |
| Action and outcome | `runtime/tool_dispatch.py::execute_and_learn`: one emit line beside the existing capture. The #1133 capture order and the timing of `situation=` stay put |
| Thought | Deliberation and `ThinkTool`, at the point where the text becomes prompt input |
| Reaction / pain | PainBus and ReactionBus subscribers emit an event; the buses themselves are unchanged |

Under the flag, §3's `propose_intent` publish, the duplicate Percept built at §1.5 and
`MemoryAgent._on_tool_result` are superseded (§9).

### 4.3 Encoding subscribers

- `MemoryAgent` becomes the encoding subscriber. It calls `capture_from_loop_async`, never the synchronous door,
  and deduplicates on `experience_id`.
- **Graded:** the `EncodingSignals` tag sets the initial strength. **Bounded:** encode at event boundaries (a new
  user turn, an outcome, a situation change, pain or relief), not on every tick.
- **Deduplicated through EC.** Pattern completion versus separation (`similarity/ec.py`): a low-surprise repeat
  reinforces the existing trace, and only a separated event forms a new one.
- **ATL and EC learn from words.** ConceptExtractor and EC semantic embedding read `utterance_in`/`utterance_out`
  text. This is a new formation source on the engram-formation line, and it gets its own stage and its own ledger
  walk.
- `MemoryHub` capture callbacks fan out after encoding, as today.

### 4.4 The cue-driven recall service

- `integration/cue_recall.py::CueRecallService` (new) implements `RecallSource`.
- **Cues: anything active.** Addressed utterances, thoughts, tool results, the situation cluster, the current
  percept's EC node, and the goal as one cue among many. This generalizes 2S-d.
- **A pure read.** It uses non-touching reads (`recall_by_ids`, `search_by_content`, `cue_situation`-style),
  never `Hippocampus.recall`, which touches every result and adds promotion pressure.
- **Off the loop thread, with coalescing.** The latest cue wins. It never blocks the loop and never activates.
- **Results go into the active-memory buffer** (§4.5) as structured RECALL entries (memory id, kind, provenance,
  score), not as cut text.
- Session R's #1447 cue path, the content-token ranker (`memory/text.py::normalize_tokens`) and the self-recall
  skip are the first ranking it uses.

### 4.5 The active-memory buffer and activation

- `WorkingMemorySet` reserves slots for RECALL entries. Today the ring of 64 evicts the oldest entry, so percepts
  at loop rate can evict a recall before it is ever rendered.
- **PromptBuilder renders the RECALL entries** in place of `bio_enrichment_context`. The text is framed (§6).
- **Activation is counted once, at first surfacing.** `memory/layer.py::activate_after_use` is called after
  `PromptBudgeter._fit` decides what the prompt includes. That closes Session R's F6 gap: today activation is
  counted inside `format_thought_response`, before any cut.
- **A memory that stays in the buffer is not re-credited on every render.** Otherwise "staying active" would
  strengthen it, which rebuilds the `access_count` immortality the strength plan removes.

### 4.6 `ThoughtGate` moves

- `runtime/thought_gate.py::ThoughtGate` stops deciding whether recall happens.
- It decides two things:
  1. what is admitted from the buffer into working memory and the prompt;
  2. whether multi-cycle deliberation runs.
- Session R's D5 (an addressed input is always enriched, deliberation keeps its scored gate) is the 1.3.4 special
  case of this move.
- §1.2's `BioEnrichmentPipeline.enrich` shrinks to the enrichment for deliberation cycles 2 and later. It stays;
  it is not deleted.

## 5. Hazards and their guards

| Hazard | Guard |
|---|---|
| Flooding | Encode at boundaries; EC dedup; the async queue's existing bound (100, drops oldest with a WARNING), with the drop count reported per run; reserved buffer slots |
| Touch/rehearsal inflation | Cue reads never touch; activation happens once, at first render; a test asserts `access_count` and promotion pressure are unchanged by cueing |
| Echo and self-recall | Exclude the current `experience_id` and its `caused_by` chain. A refractory window per id. A record's derived (framed) text is never content (`memory/text.py::text_values`) |
| Stored injection | Replay and consolidation work on ids, situations and strengths, never re-rendering text to a model. Recalled text is rendered only inside `utils/content_safety.py::frame_tool_output`, and no prompt carries a broken frame (Session R's rule 2) |
| Laundering into semantic memory | ATL formation from words takes provenance into account. Narrated and external text is never promoted as an `experienced` concept |
| Reconsolidation mutation | Content is never mutated on access (the bio-memory `recall` invariant stays) |
| Lock discipline | The recall worker collects ids, releases the store, then calls `activate`. It never calls activate under a store lock or inside `for r in store:`. The SCN leaf-lock rule holds |
| Determinism | A settle mode: `flush()` on the capture worker and the recall service, so harnesses and goldens are reproducible |
| Silent bus failures | Prerequisite, Phase 0: `AgentBus.publish` re-raises `EncodingContractError` |

## 6. A decision record owed (DECISIONS.md)

ARCHITECTURE.md says "state is the single source of truth" and "no component may bypass state". This plan needs
an entry stating that **experience is an append-only log, distinct from control state**:
- `state.data` keeps holding control state: mode, goal, pending proposals.
- Experience events flow past it.

Without that entry, every option beyond Phase 0 violates an absolute rule. The architecture audit
(ARCHITECTURE.md) must encode the distinction in the same change.

## 7. Phases

| Phase | What | Gate |
|---|---|---|
| **0 (1.3.4, Session R)** | Publish the user's input once, at arrival, with one source classifier; `AgentBus` contract errors fail loud; this lands in R's recall PR. *Referenced here, not designed here* | R's red gates (overwrite, non-text erase, robot path, bus contract) |
| 1 | The `ExperienceEvent` type and emits from every producer (§4.2), behind a default-off flag; no consumer change | Goldens byte-identical with the flag off; emit-once tests per producer, each proven by deletion |
| 2 | Encoding subscribers on the async door; `caused_by` on loop captures | Session A's 1.3.5 slices (the action-capture seam) |
| 3 | `CueRecallService`, the reserved buffer slots, PromptBuilder rendering, activation at render | Session A's 1.3.6 slices (§1.2 lives inside `run_agentic_loop`) |
| 4 | The ThoughtGate move (§4.6) | Phase 3 |
| 5 | ATL and EC learn from words | Its own engram-formation design pass |
| 6 | GL3 integration: the Hippocampus as a relay destination | GL3.B5 (text receptors) |

**Option C, memory as a running process with its own clock, is a design note, not a phase.** It would keep a
continuously updated recall field and feed familiarity and surprise into the gate. It enters only if an E1/B4
style experiment needs ambient recall.

## 8. Ledger walk (behavioral_graduation_candidates.md)

| Row | Status |
|---|---|
| T1-16 (Exp 63) | **Fires by its letter** at Phase 3: the recall query moves, and records gain `caused_by` (record shape). A re-run or a structural walk is owed |
| T1-2 | Walked at Phases 1–3: what reaches the prompt changes. It is already STALE, with a re-fire owed |
| Every LLM-primary row with user input | Walked at Phases 0, 3 and 4 |
| Body-world rows (T1-6, T1-13, T1-14, T1-15) | **Walked, not fired**, provided `execute_and_learn`'s capture stays where it is and its order and `situation=` are unchanged. Their paths are substrate-primary and never read enrichment |
| Wire 2 (gating) and T3-9 (pain cascade) | Untouched: the ReactionBus and PainBus are unchanged |

## 9. Dormant, never deleted

These become Dormant at the phase that supersedes them, each marked `Dormant since <date>: <reason>`:
- `run_agentic_loop` §1.5's duplicate `make_text_percept` (Phase 1);
- §3's `PerceptionAgent.propose_intent` input publish (Phase 0);
- `agents/memory_agent.py::MemoryAgent._on_tool_result`;
- the two `ToolResult` publishers with no callers, `MaximAgent.publish_tool_result` and
  `AgenticGoalAgent.report_tool_result` (marked in Phase 0's PR);
- `PatternCompleter.complete`'s activation until it has a consumer (open decision §10).

## 10. Open owner decisions

1. **Identity:** does `experience_id` become GL3's `PhysicalEventId`, or wrap it?
2. **Event boundaries:** which kinds start a new encoded event, and is a long utterance stream chunked by turn or
   by time?
3. **ATL from words:** which provenance kinds may form concepts? Do `narrated` and external text form none, or
   only at a discount?
4. **Buffer size:** how many reserved RECALL slots (the Cowan figure is about 4), and what is the decay?
5. **What the gate reads after Phase 4:** the buffer's familiarity and surprise, or the current
   working-memory-head score?
6. **PatternCompleter:** stop it activating with no consumer, or wire its predictions into the buffer?
7. **The StateStore:** save the in-RAM snapshots, or declare them ephemeral once `caused_by` carries the input?
8. **DECISIONS.md wording** for §6, and whether ARCHITECTURE.md's audit gains an "experience log" layer.

## 11. Required next step

A **four-lens design review** before any build, with the charters from
[docs/experiments/DESIGN_REVIEW.md](../experiments/DESIGN_REVIEW.md) adapted to a mechanism plan:

| Lens | Question |
|---|---|
| Confounding | Which ledger claims would change meaning? |
| Bio-faithful | §3's claims and the gate placement |
| Wiring | Every producer and consumer outside the diff, the fences, the lock order |
| Environment | Sims, Talk, the robot capture-manager path, the CLI REPL |

Each lens returns DO-NOT-BUILD / SHOULD-FIX / NIT into `docs/plans/rationale/experience_stream/<lens>.md`. The
folds go to the owner as one plan.

## 12. Side findings (to file as issues, independent of this plan)

1. `memory/hippocampus.py::Hippocampus.capture_from_loop` reads `state_snapshot.get("mode")`, but
   `RuntimeState.snapshot` nests the data under `"data"`, so every loop capture records `active_mode='observe'`.
2. `math/angular_gyrus.py::AngularGyrus.store` returns `record.id` at capacity without storing (WARNING only).
3. `memory/pattern_completer.py::PatternCompleter.complete` records activations (`source="prediction"`) with no
   live consumer of its output.
4. The CLI REPL path (`pending_cli_input`, `input_handlers.py`) puts the input in the loop's local `cli_input`,
   never in `state.data["cli_input"]`, so the §3 publish may never fire there. *Unverified; needs a probe.*
5. `proprioception/pain_bus.py`'s `_suppress_bridge` is an instance flag, not thread-local, so a Reaction
   published by another thread during the window skips the bridge. *Possible race; not probed.*
6. `memory/state_store.py::StateStore` (the per-capture `state_ref` snapshots) is in RAM only and never saved,
   which is the only place Talk's words currently survive.
