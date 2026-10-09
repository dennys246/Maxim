# Latent forward model — action-conditioned consequence prediction (1.4 grounding line, GL4–GL5)

> **PROPOSED — opened 2026-10-07 by owner decision G2, reversing roadmap 1.4 decision 3 of
> 2026-09-18 ("the deferred JEPA plan is re-pointed, not revived").** This is the plan that
> [roadmap_1_4.md](roadmap_1_4.md) §Phase 5 "A graded predictor (anticipation)" reserves by name:
> "a new plan (`latent_forward_model.md`) … where the §JEPA predictive idea and the §Pressure candidate
> input would live". Phase 5 makes that plan conditional on an audit of `anticipatory_pre_activate` and
> `embodiment/cerebellum.py`. That audit is **Stage S0a** here and runs first: it is S0's companion and
> its written verdict decides how this plan's predictor divides the work with the Cerebellum (§7, S0a).
> The line's umbrella is [grounding.md](grounding.md). The target this plan learns is produced by
> [autonomic_layer.md](autonomic_layer.md) (GL2), and its inputs arrive through
> [thalamic_relay.md](thalamic_relay.md) (GL3).
>
> **Placement (owner decision G1).** S0–S4 are grounding stage **GL4**; S5 is **GL5**, the first grounding claim,
> held by release threshold **T9 (conditional, never co-headlined with E3)**. This plan's design review
> and its S0 outputs count toward **T8** (grounding truth and contracts, engineering only, gates 1.4.0).
> **Body world first:** the cradle under substrate-primary selection and Minecraft. The word world enters
> later as the innate-prior tier. **Timing:** GL4 S0a (audit) and the paper run now, in parallel with
> 1.3.2 (the S0b prereg and script are paper too); S0b needs a fresh capture (a sim run). S1 is
> record-only `src/` inside G1's exempt file set, but since it waits for the post-fence resume stage
> (G15, G17) it lands after the fence anyway, so that exemption is moot. Its files, exactly: `embodiment/tool_bridge.py` (the
> `ActionContext` assembly and the observe call), `embodiment/cerebellum.py` (payload `"1.2"`, the new
> payload key, `import_state` refusing newer versions), `runtime/executor.py::Executor._stamp_invocation`,
> `runtime/bio_integration.py` (`EncodingSignals.extra["context"]` at the loop capture) and the new leaf
> `embodiment/action_context.py` (the `ActionContext` type). (GL2a's own list is `embodiment/body.py`,
> `embodiment/sem.py`,
> `runtime/executor.py::Executor._run_started` + `::Executor._stamp_invocation` (widened by owner decision
> G14, 2026-10-09), `tools/base.py::ToolOutput`, `runtime/bio_integration.py`; `embodiment/event_id.py`
> left it by G17.) GL4 S1 waits for GL2a **and for the post-fence resume stage** (owner decisions G15
> and G17, 2026-10-09: GL2a mints no pid at all; the `PhysicalEventId` type, the per-agent sequencer,
> its session-id source and the cross-session resume land together after the fence and before S1, the
> first stage that persists a pid across sessions; GL2a records carry no pid and are never training
> data). S3 and S4 are `src/` and wait for the agent_loop/orchestrator
> decomposition fence; S2 is offline (`scripts/`) and waits for S0b's data.
>
> **GL1 four-lens design review (2026-10-09).** The umbrella and `autonomic_layer.md` were reviewed by the
> four lenses ([confounding](reviews/grounding_gl1/confounding.md),
> [bio-faithful](reviews/grounding_gl1/bio-faithful.md), [wiring](reviews/grounding_gl1/wiring.md),
> [environment](reviews/grounding_gl1/environment.md)); the findings that land on this plan's stages
> (S0b, S2, S5 = GL5) are folded here, with owner decisions G13, G15 and G16. **This plan gets its own
> four-lens design review before GL4 starts** (S0b's prereg is its first reader); S5 gets a further one on
> its prereg before any harness. GL1's measurement is the census
> ([scripts/grounding_census.py](../../scripts/grounding_census.py),
> [docs/experiments/grounding_census.md](../experiments/grounding_census.md)), which S0b's floors are set
> against.
>
> **Every mechanism here enters as `[engineering]`.** Each behavioural change ships opt-in and default-OFF,
> with today's behaviour pinned byte-identical. The one exception is S3's cradle seam (the `--sim` shutdown
> reaching `BioStack.on_session_end`), which corrects a skipped session-end call rather than adding a
> mechanism, and carries its own test and ledger walk. Nothing here powers an E1–E3 arm unless the arm declares it
> (M10, extended per grounding.md).

**Owns (proposed):** `src/maxim/memory/consequence_predictor.py` (new, S3), the `ActionContext` type in the
new leaf `embodiment/action_context.py` (S1) and its assembly in `embodiment/tool_bridge.py` (S1), the Cerebellum's observe call site and payload `"1.2"` (S1), the
`"consequence"` EC modality registration (S4), `scripts/consequence_pair_audit.py` (S0b),
`scripts/train_consequence_predictor.py` + `scripts/eval_consequence_predictor.py` (S2),
`<home>/consequence_predictor.json` (S3).
**Companion plans:** [grounding.md](grounding.md) (umbrella, GL0–GL6) ·
[autonomic_layer.md](autonomic_layer.md) (the target, `InteroceptiveOutcome`) ·
[thalamic_relay.md](thalamic_relay.md) (`Receptor`, `AfferentTrack`, `AfferentEvent`; it uses
`PhysicalEventId`, which lives in the leaf module `maxim/embodiment/event_id.py`, built at the autonomic
plan's post-fence resume stage, G17) ·
[engram_formation.md](engram_formation.md) E7 (the Cerebellum read-side resurrection route) ·
[lookback_primitive.md](lookback_primitive.md) (the Hippocampus is the replay buffer; no new store) ·
[three_factor_credit_assignment.md](three_factor_credit_assignment.md) (R4's map) ·
[grounded_language_acquisition.md](grounded_language_acquisition.md) (SUBSUMED for grounding by
grounding.md, owner decision G5; its Phase 2 word side becomes this plan's innate-prior tier).
**Subsumes:** [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) (SUBSUMED by
this plan, owner decision G5; §2). Its banner is re-pointed here and its four rules are carried over.

---

## 1. Why this plan exists

The 2026-10-07 audit (#1120) found that Maxim's concepts are similar by **name**, never by **what they do
to the body**:
- `touch` ×16 share one node: the blanket (safe), the fire pit (burn) and the sharp rock. (The GL1
  census, [grounding_census.md](../experiments/grounding_census.md), found the sharp rock's declared +0.4
  pressure sits inside its 0.6 comfort band, so its YAML does not model a cut; the safe-vs-harmful
  collision is blanket vs fire pit.)
- `warm_self`-safe and `warm_self`-harmful share one node, and `turn_left`/`turn_right` collapse.
- Only 42 of 405 shipped affordances declare a `self_effect`/`target_effect` (verified by a walk of the
  shipped YAML; GL1's census re-measures them as its baseline).

The body world learns per situation cluster and per tool, and no EARNED row depends on the word (EC
text) path. It has no
function that answers *"if I do a to e in situation s, what happens to my body?"* for a key it has not
seen. The word world answers "what is this like?" from mpnet geometry alone.

This plan's thesis is one sentence: **an agent that predicts the bodily consequence of acting on a thing
can call two things similar when their consequences are, whatever their names say.** The word embedding
stays as the prior an agent starts with. Experience hands similarity over to consequence. The phenomenon has
a name and a standard design: **acquired equivalence and distinctiveness** (Honey & Hall 1989; in humans,
Myers, Shohamy et al. 2003). Its test (§6) requires that two items share a consequence while their sensed
readings disagree, so that "similar by consequence" cannot be produced by "similar by what the agent
senses" (owner decision G13, 2026-10-09).

**What kind of prior the word embedding is.** mpnet's geometry is learned from human text, so for the agent
it is an **inherited cultural (linguistic) prior**, not a biologically innate one. Its biological comparator
is instructed and observational learning set against first-hand experience (instructed fear, Phelps et al.
2001; observational fear, Olsson & Phelps 2007), and α's fall (§4.2 item 6) is read as precision weighting
of instruction against experience. In the code's behaviour-tier vocabulary it is still the **innate prior**
tier: hard-coded by us, not learned by the agent. Both statements hold; this plan uses "inherited prior"
for the biology and "innate-prior tier" for the code tier. (Truly innate priors, such as Garcia's prepared
taste aversions, resist being overridden; that is not the dynamics claimed here.)

The roadmap's E3 needs the same instrument read differently. "How far is pain" is a *graded* prediction of
Δoxygen/Δhealth by situation, and Wire-4's θ-gated step on cluster identity does not carry it (roadmap
§Thesis, "Anticipation — the prediction is a step, not a slope"). One predictor gives two readouts. The
graded readout is E3's to claim and the similarity readout is GL5's. S0a decides which model each readout
rides (§7).

---

## 2. What this is not: the deferred JEPA plan, and what carries over

[deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) solves a different
problem.

| | Deferred JEPA plan | This plan |
|---|---|---|
| Question | "Which word goes with this sensor pattern?" | "If I do *a* to *e* in situation *s*, what happens to my body?" |
| Training pair | (sensor_t, narrator_t), co-occurring within 1 s, with an NAc reward in the window (its §Pairing rule) | (`ActionContext` at invocation, `InteroceptiveOutcome` of the same physical event), joined on `PhysicalEventId` |
| Mechanism | **Co-occurrence alignment**: two MLP heads project the 384-d SensorEncoder and the 768-d LinguisticEncoder into a shared 256-d latent, trained with InfoNCE | **Action-conditioned prediction**: a fixed context encoder feeds a predictor (kernel ridge first), which regresses onto a fixed target encoding of the body consequence |
| Language's role | One side of the alignment | One INPUT block of the context: the innate-prior tier. Words become similar as a side effect, when the consequences of the actions they name are |
| Collapse risk | Real (needs a contrastive term) | None, because the target encoder is fixed |
| Data | Roy-5b cradle co-firing (never audited) | Executed invocations through `Executor`, in any world |
| Persistence | PyTorch `.pt` (contradicts the no-pickle rule and the Pi constraint) | JSON through `atomic_write_json` + `with_format_version` |

**Why co-occurrence alignment is not this.** Co-occurrence binds what *appears together*: the word
"hungry" with a hunger reading. It cannot separate the blanket from the fire pit. Both are touched, both
are named while the infant is cold, and both co-occur with the same narrator text. What separates them is
what touching them *does*, and that is an action-conditioned quantity no co-occurrence window measures.
The archived [cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) ("DO NOT
RESURRECT") bound by temporal co-activation, and it stays archived. This plan binds by predicted
consequence, not by co-activation.

**What carries over.** These are the four rules, carried over as owner decision G2 requires (rule 2's
handling of narrated consequences is as owner decision G6 set it):
1. **No pretrained cross-modal weights.** No CLIP, ImageBind or any imported alignment. Sentence-transformers
   stays runtime infrastructure for the word block only.
2. **The contamination guard is a CI test.** A training pair whose `pid` was not minted for a physical
   event in the run, a pair with no provenance, and a curated pair each raise at construction. A
   `narrated` pair (a consequence written by the narrator's tools, §3.6) is constructible only with its
   `narrated` stamp and the declared narrated discount applied to its weight; it can never be relabelled
   `experienced`. There is no `--manual-pairs` surface anywhere
   (`tests/unit/test_consequence_no_contamination.py`, the analogue of the deferred plan's
   `test_jepa_no_contamination.py`).
3. **Opt-in.** Every stage that changes a runtime path ships behind a `maxim config` key, default OFF.
4. **Existing encoders untouched.** `SensorEncoder`, `LinguisticEncoder`, `_sensor_embed` and the EC's
   existing modalities are not edited. The predictor reads their outputs; it never re-weights them.

Also kept as an option: the deferred plan's projection can come back as a **learned target encoder**
(below). Not kept: its architecture, its pairing rule and its Stage 0.

**Honest naming statement.** While the target encoder is fixed, which it is through S5, this is
**action-conditioned supervised regression in a JEPA shape. It is not a JEPA in the I-JEPA sense.** The
joint-embedding part earns the name only when the target becomes a **learned embedding of a rich percept**,
such as vision or audio onsets arriving through a `Receptor`, rather than the fixed autonomic code. The
interface is kept JEPA-shaped (`encode_context` / `predict_latent` / `encode_target`) so that a learned
target encoder can slot in later. Until then, no ledger row, CHANGELOG line, plan title, release claim or
code identifier says "JEPA". The working name is "the consequence predictor".

---

## 3. Current state in code (origin/main f1c833ab)

Claims checked at a `file::symbol` are stated as fact. Anything else is marked **UNVERIFIED**.

### 3.1 The Cerebellum: a forward model that already exists and learns the wrong target

| Fact | Evidence | Status |
|---|---|---|
| Key = `ModelKey(entity_path, modulator, affordance, param_bucket)`. Params are bucketed at 10% of the sensor range, or step 5.0 | `embodiment/cerebellum.py::ModelKey`, `::bucket_params` | live |
| Per key, `ForwardModel` keeps a per-sensor running mean and variance (single-cue Rescorla-Wagner, lr 0.2). Confidence = `1 − exp(−0.3·n)`, capped at 0.95 | `::ForwardModel.update`, `::ForwardModel.confidence` | live |
| **The write is LIVE.** Every SEM affordance call reaches `observe_from_action(entity=self._entity.full_path, …, actual=entity_state)` | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` | live (it was a silent TypeError from 2026-04-07 to 2026-09-01; guard `tests/unit/test_cerebellum_wiring.py`) |
| **It learns the owning entity's readings, under exact keys.** `entity_state` holds the sensors of the entity that OWNS the modulator, read after `modulator.execute` and before `_apply_sensor_deltas(self._embodiment.root, _self_effect)` writes to the body. In the cradle, `fire_pit.touch` teaches it the fire pit's `heat_output`/`fuel` and never the infant's `arms.thermal +0.6`. In Minecraft the `avatar` modulator sits on the player root, so it learns ABSOLUTE post-action readings, not deltas, and their meaning differs per world | same function, the order of the `entity_state` loop and the self_effect block; `_data/components/items/cradle_fire_pit.yaml`; `_data/components/bodies/minecraft_player.yaml` | live defect for this purpose |
| The read side is Dormant: `predict`, `observe_action_sequence`, `query_engrams`, `cleanup_program`, and the `CerebellumModulator` backend | "Dormant since 2026-10-04 … #909" docstrings; `backends/cerebellum_modulator.py`; AST guard `tests/unit/test_cerebellum_dormant_909.py` (it scans **`src/` and `scripts/`**, so even an offline script that calls `cerebellum.predict` fails it) | dormant; resurrection is routed through `engram_formation.md` E7 |
| It is persisted (since #908): `<home>/cerebellum.json`, payload `"version": "1.1"` + `_format_version`, key string `a\|b\|c\|d` | `runtime/bio_stack.py::_build_cerebellum`, `BioStack.save_cerebellum` | live |
| **A second live reader of the key:** right after the observe, `get_confidence(self._entity.full_path, …)` is logged through `sim_cerebellum`. It is keyed on the OWNER, so it must move with the key | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`; `simulation/sim_logger.py::sim_cerebellum` | live |
| **`import_state` loads an unknown version with only a warning.** Any version other than `"1.0"`/`"1.1"` logs "Unknown Cerebellum state version" and then reads `models` as usual, so a newer file would be misread rather than refused (a downgrade hazard) | `embodiment/cerebellum.py::Cerebellum.import_state` | live |
| **Param bucketing uses the OWNER's sensor ranges.** The call site builds `sensor_ranges` from `self._entity.sensors` (matched to param names) and passes it to `bucket_params` | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`; `embodiment/cerebellum.py::bucket_params` | live |
| The same file carries motor **programs**, read by `planning/adaptive_planner.py` (`find_related_programs`) and rendered into prompts as `motor_programs` (`prompts/acting_coach.py::_compose_cerebellum_predictions`, `agents/prompt_builder.py`). Program crystallisation is Dormant (#909), so in practice they are empty | `embodiment/motor.py`; `embodiment/cerebellum.py` Dormant docstrings | live readers, dormant writer |
| It has no similarity. `predict` is an exact dict lookup, and the key carries no situation, so `escape_water` in water and on land share one mean | `::Cerebellum.predict` | structural |

### 3.2 The timed predictor

`decisions/temporal_credit.py::TemporalCreditDistributor.anticipatory_pre_activate` is "Dormant since
2026-05-26 … the production agent loop never registered the per-tick caller". It predicts *when* an
oscillator-tracked event recurs, so that eligibility can be primed before `distribute()`. It has no input
for an action or an object, and its output is a list of (event, activation) pairs, not a body consequence.
Ledger T3-4 is DORMANT. `docs/agents/bio-memory.md`'s SCN invariant still says it runs once per tick. That
is false, and it is corrected in GL0 (grounding truth item 14).

### 3.3 The per-invocation body record and the replay buffer

- `runtime/executor.py::Executor._stamp_invocation` writes `rpe`, `drive_pressure_before`, `drive_relief`
  and `pain` onto every `ToolOutput`. `Executor._drive_relief` normalises `drive_progress_by_drive`, and
  **a drive moved away from comfort is recorded as 0.0**, so the record carries no signed harm.
- `runtime/bio_integration.py::capture_loop_action` saves these as `memory/encoding.py::EncodingSignals`
  (`site="loop"`, with a CC3 `extra`) on a Hippocampus trace, together with the action, its params and the
  2S-d situation cue. **This is the replay buffer.** Using it follows `lookback_primitive.md`'s decision 6:
  no new store.
- The capture queue drops the oldest entry when full (`memory/hippocampus.py::capture_from_loop_async`
  docstring). How often that happens is **UNVERIFIED**.
- **The executor's invocation id is `uuid.uuid4()`** (`runtime/executor.py::Executor.execute`). It cannot be
  a persisted join key: a training set keyed on it differs between two identical runs, and the two-process
  determinism guard would fail. The join key is the deterministic `PhysicalEventId(agent_id, session_id, seq)` (owner
  decision G3; one frozen, SHAPE-FROZEN type in the leaf module `maxim/embodiment/event_id.py`),
  carried by `InteroceptiveOutcome.pid`. `seq` comes from one per-agent `EventSequencer` held by
  the agent's primary Embodiment. **Amended 2026-10-09 by owner decisions G15 and G17:** GL2a mints no
  pid; GL2a's `InteroceptiveOutcome` has no event id, and its records are never training data. The type,
  the sequencer, its session-id source and the cross-session resume land together as one stage, after
  the fence and **before GL4 S1** (the first stage that persists a pid across sessions, in
  `extra["context"]` and the Cerebellum payload): the sequencer persists its own
  high-water mark and resumes past the maximum of that mark and every pid-bearing store, and it is tested
  through **both** real load seams, `runtime/bio_stack.py::build_bio_stack` and
  `simulation/orchestrator.py::_restore_aut_from_session` (`--resume-sim`), never by hand-seeding a
  sequencer (wiring lens D2). Resuming from the Hippocampus alone (the
  `memory/hippocampus.py::Hippocampus._resume_capture_seq` rule) is not enough, because that store loses
  pids: queue drops, eviction, and executor paths that never reach `capture_loop_action`. Ephemeral
  wrappers (`agent_id == ""`, the `simulation/tools.py` scene embodiments, the `foundry.py` wrappers) mint
  no ids. The tool path stamps the `pid` on `ToolOutput` alongside the invocation id. The uuid stays where
  it is, in the executor's and `ToolPainBridge`'s bookkeeping, and is never a join key.
- **`ToolOutput`'s repr is persisted and substring-searched** (#1189): `Outcome.result` stores the
  `ToolOutput` object and it is stringified at save, so `search_by_content` matches its field names. Every
  field this line adds to `ToolOutput` (GL2a's record, S1's `ActionContext`) is declared `repr=False`, and a
  guard pins `str(ToolOutput)` byte-identical with and without it (wiring lens D3). The pre-existing leak
  of the 2b-ii / 2S-c stamps is #1189's, not this plan's.
- Credit-side neighbours this plan must NOT modify: `tool_bridge._drive_potential_diff` and
  `sem.drive_comfort_progress` (named in T1-9's `Re-run on:`), and the sign-only booking in `tool_dispatch`.

### 3.4 The other consequence stores, all of them lookup tables

`decisions/causal_link.py::CausalLink` holds a scalar valence per event signature. `NAc.cluster_reward_bias`
is keyed by (cluster, tool), and `NAc.record_cluster_fear` by world cluster. None of them computes a value
for a key it has not seen, except where EC similarity of the SITUATION happens to line up. An
(object, action) pair is not a situation.

### 3.5 No usable training data exists today: the evidence

| Source | Per-invocation (context, action, consequence)? | Volume |
|---|---|---|
| `actions.jsonl` (`simulation/report.py::save_action_log`) | Tool, params, success and `str(output)[:1000]` only. **No `side_effects`, no encoding** | one row per call |
| Hippocampus `"loop"` traces | **Yes**: action, params, situation cue, pain/relief/pressure/rpe. Lossy (queue, consolidation) and unsigned | one per captured invocation |
| Cerebellum | Aggregate means per key: no samples, no situation | — |
| Committed `docs/experiments/data/` survival records (exp60_trials 20 rows, exp61_pairs 121, exp62_rows 31, r3_bench 75) | **No.** Per-trial summaries; `post.placements[].actions` are tool-name lists | — |
| Committed world traces (`l11_world_trace_2026-09-04.jsonl` 1,193 snapshots; `language_trace_1204_2026-09-21.jsonl` 984) | **No actions** | — |
| Committed cradle results (e.g. `42d53_results.jsonl`, `n_actions=420`) | Summary rows only; the per-invocation data was not kept | — |
| Operator-local `~/.maxim/sim_reports/` (40 newest `aut_hippocampus.json` sampled) | 5,994 traces, **0 with encoding signals** (all predate Phase 2b); embodied affordances rare (`hearth_warm_self` 119 vs `sense_food_source` 2,761) | unstamped and operator-local, so it **cannot be gating evidence** (weak-evidence rule) |

**Conclusion:** no committed or local record can train or test this predictor. Stage S0b is therefore a
fresh, provenance-stamped capture whose audit criteria are frozen beforehand. It is not the "audit over
committed records" that an earlier draft of the stage map assumed.

**Volume estimate (UNVERIFIED until S0b):** a scripted-substrate cradle run gives about 10²–10³ affordance
invocations. An LLM cradle run gives about 10¹. A Minecraft water-classroom run gives about 10¹–10²
invocations over fewer than 10 distinct affordances.

### 3.6 Provenance today

No experienced/imagined/narrated field exists on `ToolOutput`, on percepts or on Hippocampus traces.
`imagined` exists only on causal links and episodes that involve session-scoped entities
(`NAc.tag_imagined_links`, `memory/episode.py`).

**Narrator writers to the AUT's body exist.** (An earlier version of this section said a grep found
none; that was wrong.) `cradle_fire_pit.yaml` says "proximity effects are handled by orchestrator sensor
writes", and `simulation/orchestrator.py` registers the three tools that make them, all in
`simulation/tools.py`:
- `SetEntitySensorTool` sets or adjusts (`delta`) an AUT body sensor, then calls `evaluate_failures`.
  Sensor reflexes dispatch through its `delta` mode (#871).
- `DamageComponentTool` publishes a `PainSignal` straight to the AUT's PainBus, then calls
  `evaluate_failures`.
- `OrchestratorActorTool` applies a scene entity's affordance to the AUT through its `target_effect`, then
  calls `_aut_embodiment.evaluate_failures()`.

The orchestrator registers them on `orch_registry` and runs them inside the orchestrator agent's
`run_agentic_loop`, on the orchestrator thread (the `start_simulation_mode` caller running the
orchestrator agent's loop; not `sim.dm`, which only interactive DM campaigns use), while the AUT loop
runs on `sim.aut`; the identity contract treats that thread as a declared edge (out-of-band ids go
through the lock-guarded sequencer, from the post-fence resume stage, G17). The reflex dispatch builds separate instances of the same classes
and runs them inside the AUT's `enrich`, on the loop thread. Their consequences are
**narrated**, not experienced. Owner decision G6: they are stamped `narrated`, never `experienced`, and
they are usable for training and for credit at a **declared discount**, because excluding them would mute
the world the LLM's language priors simulate. The discount's value is an owner decision at GL4 start
(§10). The provenance kinds are `experienced` / `narrated` (discounted, G6) / `imagined` /
**`apparatus`** (excluded; owner decision G16, 2026-10-09), with `declared`/`reported` still open at GL3.B1,
the registry+provenance stage.

**Apparatus writes (G16).** A harness that writes a drive directly is neither the agent's action nor the
narrated world: the Exp 60/61 water-trial rescue (`scripts/survival_world/water_trial.py::WaterTrial.rescue`:
RCON teleport, `/effect instant_health` + `saturation`), the R3 respawn, the Exp 56/57 teacher writes
(`scripts/exp56/common.py`, `_apply_sensor_deltas` then `NAc.credit_operant_reward`) and the Exp 52 mother
feeds (`simulation/cradle_mother.py::reactive_mother_tick`). Their consequences are stamped `apparatus`:
recorded, **never a training target and never credit**. A pair carrying an `apparatus` outcome cannot be
constructed (§4.1), and S0b counts these writes by harness and cause (environment lens DNB-2).

### 3.7 The word world's affordance concepts, and the probe that defines the test

`imagination/trigger.py::_make_aff_encoder` / `encode_entity_affordances` encode the affordance NAME
(`AffordanceDecompositionStrategy`) into EC `"text"` at 0.44 with a running mean, and only with
`MAXIM_SUBSTRATE_PATH=1`. The entity is not part of what gets encoded. Ledger T1-5 ("Affordance concept
transfer") is PARTIAL, and its disposition is an owner decision at GL0 start (strict recommendation:
DROPPED; §10).

An offline probe on `paraphrase-mpnet-base-v2` (a measurement, not a gate):

| pair | cosine | at 0.44 |
|---|---|---|
| fire breath ↔ flame jet | 0.601 | bind |
| fire breath ↔ water jet | 0.298 | separate |
| **flame jet ↔ water jet** | **0.664** | **bind, and closer than flame jet ↔ fire breath** |
| fire ↔ flame | 0.785 | bind |
| touch blanket ↔ touch fire pit | 0.469 | bind |
| turn left ↔ turn right | 0.874 | bind |
| escape water ↔ flee | 0.568 | bind |

### 3.8 Dependencies and compute

Core dependencies are `numpy` + `scipy` (`pyproject.toml` `dependencies`). `torch` enters only through the
`llm-torch`/`semantic` extras, which are excluded from `pi`. Without sentence-transformers,
`similarity/encoder.py::_fallback_embed` returns a 384-d bag of words, **so the word prior's geometry
differs by install**. Every learned map must carry the context encoder's `encoding_geometry_tag` and refuse
to load or predict on a mismatch.

---

## 4. Design

### 4.1 The contracts

**Target: the autonomic code, not a new dataclass.** The target is
`InteroceptiveOutcome.as_vector(schema_id="ans-v1")` from [autonomic_layer.md](autonomic_layer.md) (GL2a). It has
two blocks:
- **Core (fixed 6-d, body-agnostic):** `[valence, nociception, drive_pain, relief, harm, urgency]`, with
  `valence = clip(relief − harm − nociception, −1, 1)`, **unweighted** (owner decision G11, 2026-10-09, an
  innate prior), and urgency v1 = pressure only (G12). Under G11's definition fix, `drive:health` contributes
  **once**: harm and nociception are defined in autonomic_layer.md so that one injury is not counted in
  both. Relief and harm come from the per-drive change in `drive_pressure` (0 inside the comfort band:
  alliesthesia, bio lens DNB-1), not from `drive_comfort_progress`.
- **Per-body block:** `deviation_after ‖ drive_delta` in the body's declared drive order. Its dimension
  differs between bodies.

**The forward-model target is the change-only subset of the record** (bio lens SF-3). The record mixes
changes (relief, harm, caused nociception, `drive_delta`) with levels (urgency = max pressure after,
`drive_pain`, `deviation_after`). Levels are largely predictable from the context, which already carries
the sensed readings, so a predictor could pass every gate by copying state, and "similar by consequence"
would become "similar by the situation it was used in". The target projection therefore takes the
**phasic** dimensions only (relief, harm, caused nociception, and the per-body `drive_delta`), and the
level terms enter as **context** (`situation` block). The record keeps every field; only the projection
differs, and it carries its own target-schema id. Every predictor gate is read against a **context-copy
baseline**, a model that sees the context but not the action (§7, S2), which must lose on the phasic
dimensions.

**Caused, not felt.** `ToolPainBridge.pop_invocation_pain` returns caused pain if any, else the pain the
body merely felt while the invocation ran. A forward model predicts reafference, the consequence of one's
own act; felt-but-not-caused pain is contiguity without contingency. GL2a records the caused/felt flag
(autonomic_layer.md); the target trains on **caused** nociception, S0b counts both, and felt-only pain is
reported, never trained on (strict default; confirmed at S2 start).

**One schema id per training set.** GL2b(ii) changes what `nociception` means for the same event (the
infant burn moves from drive pain to nociceptive pain), and bumps the schema id (`ans-v2`). A training or
evaluation set never spans a schema id: S0b, S2 and S3 refuse a mixed set (confounding lens SF-2). Because
GL2a persists the per-drive block beside the core, a later projection can be recomputed from the replay
buffer.

**Gates read per-dimension signs, never valence.** S0b, S2 and S5 gate on `nociception`, `harm` and
`relief` separately; valence is reported beside them as the innate-prior summary (confounding lens SF-2).

The predictor trains on the **core first, across bodies**. A cradle burn and a Minecraft drowning then land
in one space. The per-body block is trained **per body** (keyed on `body_path`), only where a body has at
least the S0b floor of pairs. An earlier draft defined a separate `ConsequenceCode`. It is dropped ("merge
before multiplying"), because a second record of the same consequence would let the two drift apart. The
**target encoder** is that change-only vector plus a fixed per-dimension standardisation computed once from
S0b's capture and pinned. It is not fitted again.

**Context:**

```python
@dataclass(frozen=True)
class ActionContext:
    """CC3 path (a): defaults on every field + extra (JSON-safe, no key collisions).
    Persists inside the predictor's training set, so the forward-compat path is declared."""
    pid: PhysicalEventId | None = None   # the ONLY join key, shared with the InteroceptiveOutcome.pid of
                                         # the same event (G3; no uuid, no wall time); exists from the
                                         # post-fence resume stage (G17); a training pair refuses None
    entity_path: str = ""                # the TARGET entity acted on (not the modulator's owner)
    affordance: str = ""
    param_bucket: str = ""               # reuses cerebellum.bucket_params
    blocks: tuple[tuple[str, tuple[float, ...]], ...] = ()
        # "identity":  LinguisticEncoder raw embed of "<affordance> <entity name>"  (inherited prior;
        #              innate-prior tier)
        # "object":    the Receptor embedding of the TARGET entity's own sensed readings
        # "situation": the active body-world Receptor vector(s) at decision time (world / interoception)
        # "params":    normalised numeric action params
    geometry: tuple[tuple[str, str], ...] = ()   # block -> encoding_geometry_tag; mismatch => refuse
    provenance: str = ""                         # REQUIRED: experienced | narrated | imagined | apparatus.
                                                 # No default to "experienced": __post_init__ rejects "".
                                                 # An apparatus pair is never constructible (G16)
    extra: dict = field(default_factory=dict, hash=False, compare=False)
```

On `ToolOutput` the `ActionContext` field is declared `repr=False`, and `str(ToolOutput)` is pinned
byte-identical with and without it (#1189, §3.3).

This maps directly onto the requested context. The **Receptor embedding** is the `object` and `situation`
blocks. Before GL3.B1 (the registry+provenance stage) lands, these are the bytes `_sensor_embed` already
produces. After it they arrive through the registered `Receptor`, and the registry is byte-identical by
its own gate, so S2 does not wait for GL3.
**Entity identity** is the `identity` block: how the agent perceives *which thing* through its name, which
is the inherited prior (the innate-prior tier, §1). **Action params** are the `params` block plus
`param_bucket`.

A training example is `(ActionContext, InteroceptiveOutcome)` joined on `pid`, and on nothing else.
**The join is the contamination guard's anchor.** A pair whose `pid` was not minted by the agent's
`EventSequencer` in a provenance-stamped run (the M1 stamp: executed maxim file, git hash, clean tree)
cannot be constructed. Each pair carries its provenance and a weight: `experienced` pairs weigh 1;
`narrated` pairs weigh the declared narrated discount (G6, value set at GL4 start); `imagined` pairs follow
Q3; an `apparatus` outcome builds no pair at all (G16). **Every gate in this plan reads experienced-only
data**; with-narrated numbers are reported beside it and never gated, because a narrated target is written
by an LLM reading entity and affordance names, so it carries the very word prior the claim says
experience overrides (confounding lens SF-7; G6's training discount is not reopened). A narrated write that adjudicates the AUT's own action is a separate physical event with its own
`pid`; how such an outcome is paired with that action's context, without joining on anything but `pid`,
is settled in S0b's prereg, which counts these writes first.

### 4.2 Where it hooks in

1. **Pair assembly (S1, record only).** `ModulatorAffordanceTool.execute` already knows the entity, the
   affordance, the params and the target's sensors, so it builds the `ActionContext`. `Executor._stamp_invocation`,
   the single per-invocation writer, attaches it beside GL2a's `ToolOutput.interoceptive_outcome`. The pair
   rides the existing Hippocampus `"loop"` capture as `EncodingSignals.extra["context"]`, next to GL2a's
   `extra["interoception"]`. No `EncodingSignals` field changes, and no new store is created. It does
   change the persisted record: `EncodingSignals.to_dict` flattens `extra` into the trace that
   `memory/types.py` saves, so each loop trace grows by the context blocks (a 768-d identity vector plus
   384-d object and situation vectors, as JSON text). The growth per trace and per file is measured in S1;
   the ledger consequence is in S1's blast radius.
2. **The online, key-specific model is the Cerebellum, with its target corrected (S1).** At the existing
   `observe_from_action` call site, `actual` becomes the consequence dims instead of the owner entity's
   absolute readings, and the key uses the TARGET entity where the affordance's object is not its owner.
   The single-cue RW rule, the confidence curve and persistence stay. The Cerebellum then has one meaning
   across worlds. **This is a write-side change only.** The read side stays Dormant, and
   `test_cerebellum_dormant_909.py` stays green, until S5 or E3 earns it through the E7 route.
3. **The slow generalising model is retrained at session end (S3).** `BioStack.on_session_end` (which calls
   `save_cerebellum` and `distributor.cleanup_session`) replays the session's `"loop"` traces that carry a
   pair. Together with the persisted training set, it refits `ConsequencePredictor`. This is the
   complementary-learning-systems split: the Hippocampus is the fast, episodic store, and a cortex-like
   model learns slowly and generalises by replaying its traces offline. The Cerebellum is not one of the
   two CLS systems: it is a separate, error-driven forward-model learner, online and key-specific here.
   Training never runs on the hot path.
   **`BioStack.on_session_end` is not reached on every path today.** Its only `src/` caller is
   `simulation/minecraft_harness.py`; the survival-world and Exp 56 scripts call it too. On the `--sim`
   orchestrator path, which the cradle runs on, the Cerebellum is saved by
   `runtime/agent_factory.py`'s shutdown, which calls `bio_stack.save_cerebellum()` directly and never runs
   `BioStack.on_session_end`, so `distributor.cleanup_session` is skipped there too (the documented
   `shutdown() ≠ on_session_end()` gotcha). S3 therefore names the **cradle seam**: the agent_factory shutdown calls
   `BioStack.on_session_end` instead of `save_cerebellum` alone, as its own wiring commit with its own test
   (§7, S3).
4. **The predictor (S2 offline, S3 in `src/`).** Kernel ridge in dual form, with a cosine kernel per block
   and fixed block weights, in **numpy only**:
   - It is closed-form and **deterministic**. Training pairs are ordered by `pid` before the solve, so
     neither the dict order nor `PYTHONHASHSEED` reaches the arithmetic.
   - It fits about 10³ pairs × about 1.5k dims as a single n×n solve. That is under a second on a laptop CPU
     (**UNVERIFIED on a Pi**; S2 measures it on the Pi profile).
   - It persists as JSON through `atomic_write_json` + `with_format_version` at
     `<home>/consequence_predictor.json`. The file carries `hash_scheme: "stable-sha256-v1"`, the per-block
     geometry tags, the target schema id, the training `pid` list and the pair count. No pickle and no `.pt`.
   - It is **Pi-capable**: no torch, no scipy beyond what core already has. A Pi install's word block uses
     the fallback geometry, so it trains its own map and refuses a map trained under mpnet.
   - An MLP enters only if S2's pre-registered gate fails for ridge AND the failure is shown to be
     non-linear. It would be numpy with manual gradients, seeded from `utils/seeding.py::stable_hash_32`.
5. **The EC consumes it through an opt-in `"consequence"` modality (S4).** For each (entity, affordance) the
   agent can act on, encode its PREDICTED core with the existing `_sensor_embed`. This is read-only reuse
   with the A4 gain, so "nothing happens" maps to the zero vector and forms no node. The modality has a
   frozen centroid, threshold 0.85 (body-world semantics, pinned, not fitted) and its own geometry tag.
   EC matrices are already per modality (`similarity/ec.py::EntorhinalCortex._matrix_for`), so the
   interoception, audio, world and text scans stay **byte-identical**. The hivemind does not filter by
   modality: `hivemind/bundle.py` exports every `substrate_nodes` entry (filtered by domain only), and
   `hivemind/merge.py` folds a modality missing from `SENSOR_MODALITY_THRESHOLDS` at the default cosine
   threshold. Keeping consequence nodes local is therefore a bundle edit (S4). Rejected alternative: a projection flag
   on `pattern_complete_or_separate`, which edits the shared encode path that every earned row runs through.
   Cosine sees direction, not magnitude (`docs/wiring/cosine-separation-is-directional.md`). A mild burn and
   a severe burn share a direction, so **magnitude is read from the predicted code, never from node identity.**
6. **Concept similarity tiers from word to consequence (S4).** A consumer asking "are *x* and *y* the same
   kind of thing to act on?" reads
   `sim(x, y) = α·cos_identity(x, y) + (1 − α)·cos(pred(x), pred(y))`, with `α = 1 − conf(n_eff)`, where
   `conf` reuses `ForwardModel.confidence`'s curve and `n_eff` is the kernel-weighted count of training
   contexts near *x* and *y*.
   - Before any experience α = 1 and the word prior rules.
   - α decays toward the learned tier as nearby experience accumulates. It stays high for contexts unlike
     anything experienced.
   - Because `conf` caps at 0.95, α never falls below 0.05: the prior is never fully switched off.
   - This is a read-time blend. Nothing overwrites a word node, and T1-5's decomposer is not edited.
7. **Shadow consumer first (S3).** For every candidate tool at decision time, the predicted core, its
   consequence cluster and α are logged. The log has **no selection effect**. A selection effect enters only
   as S5's declared experimental arm (§8).

### 4.3 Behaviour tiers (every automatic behaviour declares one)

| Behaviour | Tier |
|---|---|
| A pair must join on a minted `PhysicalEventId`; a `narrated` pair is never relabelled `experienced` and always carries the declared discount; an `apparatus` outcome builds no pair; a geometry mismatch refuses; a training set never spans a target schema id | **invariant** (`[engineering]`, enforced in the type and the loader) |
| The narrated discount's value | **innate prior**: an owner-set constant (GL4 start), pinned before S2's data |
| Word-embedding (identity-block) similarity before experience, α = 1 | **innate prior** tier (biologically an inherited cultural prior, §1). Follow-up trigger: S5's result |
| Block weights, kernel, the 0.85 consequence threshold, the α curve, the target standardisation, the change-only target projection, the core valence formula (unweighted, G11) | **innate prior**: pinned constants, fixed before the S5 data. Hard-coded priors get a follow-up issue with a trigger (behaviour-tiers rule) |
| The Cerebellum's per-key consequence means (after S1) | **learned** (online, key-specific) |
| Predicted consequence, consequence clusters, α decay with experience | **learned** (slow, generalising) |
| Imagined-entity pairs | Owner decision at S3 start. The recommendation is to exclude them (the strict option) |

### 4.4 Invariants this plan introduces (all `[engineering]`; none behavioural until S5 earns it)

| Invariant | Regression guard |
|---|---|
| A training pair cannot be constructed without a minted `PhysicalEventId`, an explicit provenance and matching geometry tags; a `narrated` pair cannot be relabelled `experienced` and always carries the declared discount; an `apparatus` outcome cannot be paired (G16) | outstanding.md M41 (backlog row until S3 lands; structural thereafter: the pair type's constructor + `tests/unit/test_consequence_no_contamination.py`); the test lands in S3 with the first training set |
| Training and the persisted predictor are byte-identical across processes with differing `PYTHONHASHSEED` | `tests/unit/test_consequence_predictor_two_process.py` (S3), on the `test_stable_hash_two_process.py` pattern |
| With the consequence modality off, every existing EC modality dump is byte-identical | `tests/unit/test_ec_consequence_modality_golden.py` (S4) |
| The Cerebellum learns the body consequence of the acted-on target, through the production call path | extension of `tests/unit/test_cerebellum_wiring.py` (S1) |
| No `"consequence"` node crosses a hivemind bundle | `tests/unit/test_hivemind_excludes_consequence.py` (S4) |

---

## 5. Front-gate scope pressure

| Existing infrastructure | Can it carry this? | Used here as |
|---|---|---|
| `Cerebellum` (per-key RW forward model, persisted) | **Partly.** It is the right home for the ONLINE, key-specific model and for the training-signal site. It cannot generalise: an exact-key lookup with no context embedding and no situation in the key | **Rides on it**: target fix + target-entity key; online model unchanged |
| `anticipatory_pre_activate` (timed predictor) | **No.** It predicts WHEN an event recurs, not what an action does to the body; it has no action or object input | untouched; audited in S0a |
| `Executor._stamp_invocation` + `ToolOutput` | Yes, for the record: it is already the single per-invocation writer | **Rides on it**: pair assembly |
| Hippocampus `"loop"` traces + `EncodingSignals.extra` | Yes, as the replay buffer (persisted, run provenance, situation cue) | **Rides on it**: no new store |
| `BioStack.on_session_end` | Yes, as the consolidation seam, once the `--sim` path reaches it (§4.2 item 3) | **Rides on it**: offline refit, plus the cradle seam wiring in S3 |
| EC per-modality matrices, `_sensor_embed`, geometry tags | Yes, for node formation over a predicted vector | **Rides on it**: a new modality, not a new similarity engine |
| `CausalLink` / `cluster_reward_bias` / `cluster_fear` | **No.** Scalars per seen key; no function from a context embedding to a vector consequence | untouched |
| LinguisticEncoder name nodes (word world) | **No.** Its similarity is fixed by mpnet, and §3.7 shows it orders flame/water the wrong way | becomes the prior tier |

**Why existing infrastructure cannot do this.** Every surface above that stores a consequence stores it
*per key it has seen*. The only generalisation in the codebase is EC similarity of a SITUATION. An
(object, action) pair is not a situation, and the one model that is keyed on (object, action), the
Cerebellum, matches keys exactly. What has to be new is one function approximator, `ConsequencePredictor`,
that maps a context embedding to a consequence vector and returns a value for an (entity, affordance,
situation) it has never executed. The other new pieces are assembly or registration only: the
`ActionContext` builder, the `"consequence"` modality registration, and the train/eval scripts.

**Input-shape check first (the Roy-4 lesson).** S0b must show that executed pairs exist in volume AND that
name and consequence *disagree* somewhere in them, counted over non-null consequences only (§7, S0b). If
every pair of affordances that share a name also share a consequence, the predictor cannot be told apart
from the word prior, and the line stops at S0. A second shape check follows from G13: the predictor must
also be told apart from **sensed similarity**, so the claim needs items whose sensed readings and
consequences disagree (§6), and these are authored for S5, not found in the shipped YAML.

---

## 6. The falsifiable signature: acquired equivalence against a no-learning sensed-similarity baseline

**Redesigned 2026-10-09 by owner decision G13** (confounding lens DNB-2, bio lens SF-8). An earlier draft
made the jet-triad reversal the test. It cannot be: in every authored item, what the agent senses about an
object (the `object` block) and what the object does to the body are written together, so the cosine
between object blocks already orders `fire_breath > water_jet` **before any consequence is learned**.
Ordinary nearest-neighbour generalisation over sensed readings, with no forward model, produces the
reversal. That is "sensed properties beat names", not "similar by consequence".

**The test is acquired equivalence** (Honey & Hall 1989), on authored items whose **sensed readings and
consequences disagree**:
1. **Stage 1 (experience of every item is required).** Items A and B share a consequence O1 while their
   sensed readings differ; item C reads like A (object-block cosine C↔A above B↔A) but has consequence
   O2. Names are chosen so that neither the identity block nor the drive-keyword path (§7, S5) points to
   the shared consequence. Each item is acted on in Stage 1; an item never acted on is not part of the test.
2. **Stage 2.** A alone is retrained to a new consequence O3.
3. **Test.** B (consequence-equivalent, sensed-unlike) must inherit O3 more than C (sensed-like,
   consequence-unlike) does, by a margin pre-registered in S5's prereg. The readout through which Stage-2
   value reaches B (the S4 consequence-cluster node, or the α-blended similarity of §4.2 item 6) is named in
   the prereg. A fixed-kernel ridge over sensed features cannot produce this ordering by itself; whether the
   shipped model can is the experiment's question, and a null ships as a null.

**The gate is against a no-learning sensed-similarity baseline**, not only against identity-only: spread the
trained consequence to the probe items by object-block cosine alone, with no fitted map. That baseline
predicts C > B. The model passes only by reversing it. A model that merely matches the baseline has learned
nothing about consequence. Pairs must include both kinds of disagreement: items whose readings match but
whose consequences differ (a hidden property, such as an insulated object that reads hot and does not burn),
and items whose readings differ but whose consequences match. T9 keeps its claim ("similar by consequence")
under this design (G13); without it the claim would have to shrink to "sensed properties override the name
prior".

**The jet triad becomes a control: generalisation from sensed features.** Under the word prior an aversion
learned on `flame_jet` leaks more to `water_jet` (0.664) than to `fire_breath` (0.601). Ordering them
`fire_breath > water_jet` after training on `flame_jet` is expected from sensed similarity alone, and is
reported as that measure, never as the test. "`water_jet` must not inherit the aversion" is dropped as a
pass criterion: with no differential experience of `water_jet`, generalisation is what biology predicts
(category-based fear generalisation, Dunsmoor & Murphy 2015). `dragon.yaml::fire_breath` exists, with no
`self_effect`/`target_effect`; `flame_jet`, `water_jet` and a `fountain` entity are **not shipped
components** (fixtures in `tests/integration/test_affordance_transfer.py` and two archived plans). The
triad therefore needs **new AUT-invoked entities** with a `self_effect` on the AUT, so that its pairs are
`experienced`; delivered through an actor (`OrchestratorActorTool`) every pair would be `narrated` and the
gated experienced-only number would be empty (environment lens SF-8). They are authored blind under S5's
rules and counted in S0b's keys.

**The shipped collisions** are where the cradle names lie, and they are corrected against the pressure-based
relief/harm of §4.1 (bio lens N5), using the collisions the GL1 census
([grounding_census.md](../experiments/grounding_census.md)) actually finds. Every harm-class collision in
its primary walk is a **same-name** pair on one node: `blanket.touch` (`arms.thermal +0.1`, 0 at rest) vs
`fire_pit.touch` (`arms.thermal +0.6`, harm), and `warm_self` on a `*_safe` warmth item (`arms.thermal
+0.05`, `cold −0.3`) vs on a `*_harm` one (`arms.thermal +0.6`, `cold −0.3`). Both differ **neutral versus
negative** on `harm`, not by opposite sign (for a chilled infant both `warm_self` variants relieve `cold`
alike; only the harmful one adds harm). `blanket.wrap` is **not** a collision partner of `fire_pit.touch`:
the two names land on different nodes (cosine 0.21 for "wrap"/"touch", 0.26 for "blanket wrap"/"fire pit touch", both below 0.44), and `wrap` (`core_temperature +0.1`)
gives 0 relief at rest. `turn_left` vs `turn_right` (0.874) collide by name with opposite signs on
`azimuth`, both classed harmful by the orienting comfort band. Each sign is per pre-state band (§7, S0b).

---

## 7. Stages

The stage IDs S0–S5 are this plan's (their place in the umbrella's map is stated once, in the header).
GL6 (a selection consumer that is not an experimental arm) is the umbrella's and is not planned here. Each stage lists its
gate, its guards, its blast radius and the owner decisions to ask together at its start
(development-flow rule 3).

| Stage | Kind | Depends on | Fence |
|---|---|---|---|
| S0a | paper + offline read of persisted JSON | none | runs now |
| S0b | prereg + script + one fresh scripted capture | GL1 census (prediction of the collision and convergence counts); M1 stamps; this plan's own four-lens review; a named capture runtime (below) | the prereg and script are paper and run now; the capture is a fresh sim run, its mechanism settled by the prereg (below); if the runtime needs `src/` record fields, the capture waits for GL2a/S1, and for pid-bearing records, the post-fence resume stage (G17) |
| S1 | `src/`, record only | GL2a (`InteroceptiveOutcome` on the tool path, no event id) **and** the post-fence resume stage that builds `PhysicalEventId`, the sequencer and the cross-session resume (G15, G17) | after the fence: the resume stage lands after it, so S1 does too, and G1's record-only exemption is moot (header) |
| S2 | `scripts/`, offline | S0b PASS | runs as soon as S0b's data exists |
| S3 | `src/` | S1, S2 PASS | fence |
| S4 | `src/`, opt-in | S3 | fence |
| S5 | experiment (T9) | S4; R4's routing audit has decided the selection surface; the authored acquired-equivalence items (§6) | four-lens design review before any harness |

### S0a — the Phase 5 audit: Cerebellum + `anticipatory_pre_activate` (S0's companion)

**What.** This is the audit roadmap Phase 5 makes mandatory before this plan opens. It is a written verdict,
committed as `docs/plans/rationale/latent-forward-model/s0a_phase5_audit.md`, on one question: **can either
shipped piece carry "how far pain is"**, a graded prediction of Δoxygen/Δhealth by situation, and the
consequence of acting on an unseen object? It reads:
- `embodiment/cerebellum.py`: the key, the update rule, what the call site passes, and the persisted
  `cerebellum.json` from a post-#908 home. **It must not read a pre-#908 fresh-each-session model as "cannot
  learn"** (roadmap Phase 5).
- `anticipatory_pre_activate` and the drive TemporalEvents: whether a per-tick caller and an
  oscillator-tracked event could ever yield a graded time-to-pain, and what consumer it would need.
- A Minecraft post-#908 `cerebellum.json`, if one exists on the rig, read as JSON. **It is not read through
  `Cerebellum.predict`**: `test_cerebellum_dormant_909.py` scans `scripts/`, so an offline caller would trip
  it. Reading the persisted means directly keeps the Dormant marker honest.

**Gate (falsifiable).** The verdict names one of three branches, each with its evidence:
- **(i) The Cerebellum can carry the graded read once S1 corrects its target**, possibly with the situation
  cluster in its key. Then E3's graded predictor resurrects the Cerebellum's read side through the E7 route
  (with `test_cerebellum_dormant_909.py` and E7 edited in the same PR, as that test demands), and this plan's
  predictor keeps the **generalisation** role only.
- **(ii) Neither can carry it.** This plan's predictor is also E3's graded-predictor candidate, via the
  Minecraft readout in S5.
- **(iii) The timed predictor carries the *when* and the Cerebellum the *what*.** The verdict says which
  consumer composes them and why that composition is not a new mechanism.

Expected from the code read (to be confirmed, not assumed): the Cerebellum as shipped cannot carry it,
because its key has no situation and its target is the owner's absolute readings. `anticipatory_pre_activate`
cannot, because it has no action input. Those facts point to branch (ii) or to branch (i) after S1, which is
exactly what the audit must settle with evidence.

**Guards.** None (paper). The verdict is read by the Architecture lens in this plan's design review.
**Blast radius.** None.
**Owner decision at S0a start (GL4 start):** the narrated discount's value (G6; strict default: a small
discount, with every S2 and S5 result reported with AND without narrated pairs, and every gate read on
the experienced-only number). Owner decision G2 already
opened the plan; S0a's verdict decides scope, not existence.

### S0b — the pre-registered paired-data capture audit

**What.** `docs/experiments/consequence_pair_audit_prereg.md` lands on `main` (merge commit) before the
capture's first data timestamp (`lint_prereg_precedes_data.py`). `scripts/consequence_pair_audit.py`
(read-only) then runs over one fresh capture through the real `Executor`, carrying M1 provenance stamps and
run with `--interactive false`:
- **(a) Cradle.** One scripted-substrate campaign in a **purpose-built all-items scene** (environment lens
  SF-9). All cradle-family items together declare 34 consequence-bearing (entity, affordance) keys
  (blanket, cool_air, false_hearth, fire_pit, food, sharp_rock, the green/purple hearth/flame items and
  their `_b` twins, warmth_alpha/beta safe/harm); the arc cradle scene (`simulation/arcs.py`: fire_pit,
  food, cool_air) has 5. "≥ 30 keys" therefore needs nearly every item in one scene, which brings the
  tool-name collision that `infant_humanoid_chilled.yaml` warns about (the `_b` twins exist for it), and
  the `cold` classes need a body with a `cold` sensor (`infant_humanoid_chilled`), because `warm_self`'s
  `cold: −0.3` changes nothing on `infant_humanoid`. The scene and body are named at GL4 start, and the
  reachable keys and classes are counted per body before the capture. The triad and acquired-equivalence
  items (§6), once authored, are counted among the keys.
- **(b) Minecraft.** One water-classroom campaign on the rig, through the production loop
  (`simulation/minecraft_harness.py::run_minecraft_aut` → `run_agentic_loop`).

**The capture runtime is named per arm, and only invocations that reach `capture_loop_action` count**
(wiring lens S10). No runtime today produces pid-bearing, captured cradle records without an LLM: Exp 42
runs `scripts/benchmark_exp42_preference.py` → a `maxim --sim` subprocess with an LLM narrator on the
orchestrator thread (so its body writes are `narrated`), and the `scripts/orient_substrate/*` probes build
`Embodiment(root=...)` with `agent_id ""`, which mints no ids. The candidate cradle runtime is the fixture
orchestrator (`simulation/fixture_orchestrator.py`, `--sim scenarios/substrate/*.yaml`, no narrator LLM)
with a substrate-primary AUT (`--aut-mode substrate-primary`); **UNVERIFIED** that this path builds the
AUT's SEM affordance tools with a non-empty `agent_id` and captures through
`tool_dispatch.execute_and_learn`. If it does not, the cradle capture waits for a production runtime that
does, wired in `src/` after the fence; it is never a hand-composed harness that calls the factory directly
(D43). The PLANNING-approved, parallel and retry `executor.execute` sites do not capture, so their
invocations are excluded and counted as such.

The capture must record, per invocation, the body readings before and after, the target entity's own
sensed readings and the situation vector. Today's `actions.jsonl` keeps none of these. The prereg names the
capture mechanism; a `scripts/` harness that drives the `Executor` directly, the way
`test_cerebellum_wiring.py` does, is a known-answer fixture for the audit script, not the capture. If the
named runtime needs `src/` record fields, S0b's capture waits for GL2a (and S1 for the context), and the
prereg says so. The consequence is computed **offline** from the before/after snapshots by GL2a's pure
function (`sem.py::interoceptive_outcome`, or its frozen copy in the script if GL2a has not merged),
including GL2a's net-of-drift report (G14), so S0b does not depend on GL2a's `src/` landing. The prereg
states the proxy's known gaps if today's `ToolOutput` fields are used instead: harm recorded as 0 relief,
and the infant thermal burn maxing at 0.2, classed as a drive.

**Gate (frozen in the prereg before the capture).** Reconciliation note: the deferred stage map applied
key-count floors to both worlds, but Minecraft has fewer than 10 distinct affordances, so the key floors
apply to the cradle and Minecraft gets a graded-consequence floor.
- **Every count reads experienced-only records** (narrated and apparatus records are counted and reported,
  never gated; §4.1), on one target schema id.
- **Sign is defined per (key, pre-state band)** (confounding lens SF-3). A key's consequence depends on the
  body's state (`fire_pit.touch` harms `arms.thermal` and relieves a cold core), on repetition (the
  nociceptor reads state, so repeated `warm_self` stacks) and on saturation (a touch on an arm already at
  1.0 gives `drive_delta` 0). The bands are frozen in the prereg; records whose pre-state sits at a range
  edge are reported separately and excluded from sign counts; key order in the scripted capture is
  counterbalanced. Sign is read per dimension (`nociception`, `harm`, `relief`), never from valence.
- **Cradle PASS** requires all of:
  - ≥ 500 executed invocations with a computable consequence;
  - ≥ 30 distinct (target entity, affordance) keys;
  - ≥ 3 consequence classes (e.g. burn, relief, nothing);
  - **name–consequence dissociations, over non-null consequences only, counted in two kinds, each with
    its own frozen floor** (confounding lens DNB-3). A null-consequence key (`observe`/`look`-type, or a
    band in which nothing changes) is in neither kind, because a null/null pair says nothing about
    grounding:
    - **collisions**: identity-block cosine ≥ 0.44 and opposite non-null sign in the same band: the
      "name similarity blocks it" half of T9. The census's shipped collisions are same-name pairs whose
      consequences differ **neutral versus negative** (bio lens N5): `touch` on the blanket vs the fire
      pit, and `warm_self` on a `*_safe` vs a `*_harm` warmth item. The prereg decides, before the capture,
      whether a neutral-versus-negative pair counts as a collision (and on which dimension) or is
      reported beside the opposite-sign count; it never borrows `blanket.wrap` / `fire_pit.touch`, which
      sit on different nodes;
    - **convergences**: identity-block cosine < 0.44 and the same non-null sign in the same band: the
      "transfers despite the name" half.

    The floors are set in the prereg against GL1's census prediction of both counts from the YAML
    declarations ([docs/experiments/grounding_census.md](../experiments/grounding_census.md)); the
    census statistic is pairwise cosine (order-free), with node co-membership under the production
    encoding order as a second column. S0b confirms them on *executed* records;
  - 100 % of counted invocations joinable on `PhysicalEventId` with a stamp. Pids exist only from the
    post-fence resume stage (G17), so a capture held to this floor runs after that stage; capturing
    earlier needs another deterministic join named in the prereg (an open point for GL4's start, not
    decided here).
- **Cradle FAIL** on any of: < 200 invocations; either dissociation kind below its frozen FAIL floor;
  < 2 consequence classes. A FAIL stops the line. Any later re-sourcing of the data is recorded as a
  post-null change of source (the 2026-09-20 precedent).
- **Minecraft: re-scoped, because the floor as first written was unreachable** (environment lens SF-2,
  confounding lens SF-5). Pathfinding is dead in water, so `move_to` underwater fails fast with no Δoxygen
  spread; the tool-path record sees the world at actuator return, so `escape_water`'s record holds the
  swim-up, not the refill (the live bridge returns 600 ms, `SURFACE_HOLD_MS`, after the eyes clear, before
  the oxygen latch clears; the scripted bridge returns "surfaced" while still submerged, so its record shows
  harm); and underwater oxygen falls whatever the action, so a spread by band arises from the situation
  alone. The Minecraft half therefore:
  - **depends on the out-of-band producer** and on the resume stage that adds `pid` / `cause_pid`
    (both after the fence; G9, G17): the consequence of an invocation is joined
    over a frozen consequence window by the next out-of-band record carrying `cause_pid` = the invocation's
    `pid` (`CauseRef.cause_pid`, autonomic_layer.md). Until that producer lands, the Minecraft half is
    recorded **"not run"**, never as passed;
  - **is action-contrastive within a band:** PASS needs ≥ 100 executed `escape_water` invocations spanning
    ≥ 3 oxygen bands, and a non-zero, pre-registered difference Δoxygen(`escape_water`) − Δoxygen(control)
    within each band, where the control is a no-op pass in the same band plus an affordance that executes
    underwater (`turn` is the candidate; **UNVERIFIED** that it acts underwater). A gate on Δoxygen by band
    alone is refused. The band edges and the window are frozen in the prereg;
  - reads rig records only; the scripted bridge's records are not used for the floor.

  A Minecraft FAIL or "not run" does not stop S1–S4; it removes S5's Minecraft arm and branch (ii)'s E3
  readout, and is recorded as such.
- **Known-answer identity check** (`feedback_diagnostic_fields_need_an_identity_check`): for each authored
  cradle affordance, the recorded body change, **net of declared drift over the window** (G14), equals the
  declared `self_effect` (clamped to range). If it does not, the instrument is wrong, and no count is read.
  The expected table is computed independently of the helpers it checks (from the YAML deltas and specs),
  never by calling `sem.drive_comfort_progress` / `drive_span`.
- **Reported, not gated:** queue-drop count, the per-world split, how many shipped affordances have no
  consequence at all, the count of narrated body writes by tool (`SetEntitySensorTool`,
  `DamageComponentTool`, `OrchestratorActorTool`, reflex dispatch), each stamped `narrated` (Q8), the count
  of apparatus writes by harness and cause (G16), caused versus felt-only nociception, the with-narrated
  version of every gated count, and the excluded non-capturing invocations.

**Guards.** The audit script's own known-answer test, against a hand-built three-invocation fixture.
**Blast radius.** None: no `src/`, no ledger row.
**Owner decisions at S0b start:** Q7 and Q8; the capture scene, body and runtime (Q9); whether the
Minecraft capture rides the rig in
an existing Track C slot or the offline scripted bridge (recommended: the cradle capture now; Minecraft on
the rig in a slot that is never stacked on an E-rung campaign, because the scripted bridge's physics is not
game-native and the arm exists to close the circularity of authored physics); and the 0.44 identity-block
threshold used to count dissociations (recommended: production 0.44, never the retired 0.40).

### S1 — Cerebellum target fix + pair record (engineering, record only)

**What.**
- `tool_bridge`'s observe call passes `actual =` the consequence dims (the 6-d core plus the body's
  per-drive block) and keys on the target entity. The `get_confidence` → `sim_cerebellum` log line right
  after it moves to the same key, so it never reports a key nothing writes.
- Param bucketing keeps the `sensor_ranges` it has today (the modulator owner's sensors, matched to param
  names): the params are the affordance's arguments, so only the key's entity field changes.
- The `ActionContext` is assembled in `tool_bridge` and attached at `Executor._stamp_invocation`, beside the
  `pid` the tool path stamps on `ToolOutput` from the post-fence resume stage (G17; GL2a stamps none). It is captured into `EncodingSignals.extra["context"]`. Its
  `provenance` is set explicitly at assembly (`experienced` on the AUT's own executed invocation); there is
  no default. The `ToolOutput` field is `repr=False` (#1189, §3.3).
- The `pid` S1 persists (in `extra["context"]` and the Cerebellum payload) is the cross-session one: S1
  lands only after the resume stage (G15, G17), which builds the pid and its resume together, so no pid
  is ever persisted without its resume.
- The Cerebellum payload version moves to `"1.2"`. A `"1.1"` file loads with its consequence statistics
  empty and logs one warning (`check_format_version` pattern). It is never a silent no-op, and the old
  absolute-reading means are not reinterpreted as consequences. **`import_state` refuses a version newer
  than it knows** instead of warning and loading. The consequence means are written under a new payload
  key, not under `models`, so a pre-S1 install that reads a `"1.2"` file (a downgrade) finds no models
  rather than misreading consequence means as absolute readings. The `programs` half of the file is
  byte-identical.
- `ActionContext` gets its CC3 docstring (path a).

**Gate.**
- **Zero selection-path diffs:** grep the diff for `recommend_action`, `credit_node`,
  `TemporalCreditDistributor`, EC encode and `_sensor_embed` (no hits).
- `test_agent_loop_selection_golden.py`, `test_decision_provenance.py` and `test_encoder_golden_v1.py` are
  byte-identical, and a check that EXECUTES the survival path through the real loop and capture stays
  green, with a positive assertion that the `escape_water` invocation's `ToolOutput` carries an
  `ActionContext` with a `pid` (so the deletion probe re-reds it):
  `tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear`,
  once #954 has moved it onto `StepClock`; until then its wall-time margins are stated beside the result
  (environment lens SF-1). `tests/unit/test_exp61_run.py` is **not** an executing check (it runs no loop)
  and is not cited. (A re-run of the Exp 60/61/R3 verdict scripts is not a check here either:
  `compute_verdict` re-reads committed JSONL and never executes the producer.)
- The per-trace and per-file size growth of `aut_hippocampus.json` on one scripted cradle capture is
  measured and recorded in the PR.

**Guards.**
- The extended `test_cerebellum_wiring.py` drives THIS call path and asserts that the infant's
  `arms.thermal` change reaches the Cerebellum for `fire_pit.touch`, and that `get_confidence` is read on
  the same key.
- Deletion probe: removing the observe call, or reverting `actual` to `entity_state`, fails the test.
- A format round-trip in `tests/integration/test_persistence_compat.py`, covering `"1.1"` → `"1.2"`, plus a
  refusal test for an unknown newer version.
- `test_cerebellum_dormant_909.py` stays green, unedited: the read side is untouched.
- `str(ToolOutput)` is byte-identical with and without the `ActionContext` (the `repr=False` guard), and
  GL2a's T1-16 ranking test is extended to the substring path (`Hippocampus.search_by_content` with a query
  that would match a context token) for `extra["context"]` too (#1189).

**Blast radius.**
- No earned row's `Re-run on:` names the Cerebellum or `tool_bridge`'s observe call. The only mention
  found is T1-13's #888 discharge text ("the only positive Reaction constructor is `CerebellumModulator`"),
  which S1 does not wire.
- **T1-16 fires by wording** ("the memory record shape"): `EncodingSignals.to_dict` flattens `extra` into
  the persisted trace (`memory/types.py`), so `extra["context"]` changes the saved record, as GL2a's
  `extra["interoception"]` does. S1 takes the same ruling GL2a gets on that row (one ruling covers both
  keys; if it is a re-run, S1's key joins GL2a's batched re-run rather than spending its own). The
  structural evidence offered with it: `memory/hippocampus_retrieval.py::_rank_by_relevance` and
  `integration/bio_enrichment.py::_query_hippocampus` read no `encoding` key, the save/restore
  round-trip is lossless, and the substring recall path is unchanged because the field is `repr=False`
  (the guard above; without it the discharge would be false, wiring lens D3). T1-1 (Exp 10) carries the older "hippocampus persistence schema change" wording,
  but it is SUPERSEDED by T1-16 and no longer gates.
- `[Unreleased]` grows (`lint_unreleased_on_src_change.py`).

**Owner decisions at S1 start:** Q6 (target-entity key: yes; situation in the key: no, because the
predictor carries the situation and the key format stays stable).

### S2 — offline predictor and evaluation (`scripts/`, numpy kernel ridge)

**What.** `scripts/train_consequence_predictor.py` and `scripts/eval_consequence_predictor.py`, trained on
S0b's cradle capture. Held-out splits are **by key and by entity**, never by invocation, which would leak.
`docs/experiments/consequence_predictor_offline_prereg.md` lands on `main` before the eval reads the data.

**Gate (pre-registered).** It covers the full model (identity + object + situation + params) against
three baselines and one ablation: **identity-only**; the **no-learning sensed-similarity baseline**
(the consequence of the nearest trained items by object-block cosine, with no fitted map; G13); the
**context-copy baseline** (a model that sees the context but not the action; bio lens SF-3); and the
object-only ablation. Every number is read on the change-only target, per dimension (`nociception`,
`harm`, `relief`), on experienced-only pairs and one target schema id:
- on the held-out **collision and convergence** keys (S0b's non-null dissociations), full-model
  consequence-sign accuracy ≥ 0.8;
- the full model beats identity-only by ≥ 0.2 there, **and beats the sensed-similarity baseline** by a
  pre-registered margin (gating against identity-only alone would let sensed-feature generalisation pass
  as consequence learning; confounding lens DNB-2);
- the full model beats the context-copy baseline on the phasic dimensions by a pre-registered margin;
- on non-dissociation keys it is no worse than identity-only by more than 0.05;
- the **acquired-equivalence probe** (§6), if its items are authored and captured: B inherits more than C,
  against the sensed-similarity baseline. If not yet authored, reported as "not run", never as passed.

The jet triad is reported as the sensed-feature generalisation control (§6), not gated.

FAIL means ridge cannot ground with these inputs. Then either try the MLP once, pre-registered as a second
iteration, or stop. **If identity-only, or the sensed-similarity baseline, already scores ≥ 0.8 on the
dissociation keys, the split is mis-built for this claim: refuse the result.** Every gate reads
experienced-only pairs; the with-narrated number is reported beside it and never gated (§4.1;
confounding lens SF-7, G6 not reopened). Also measured and reported: the solve time on the Pi profile.

**Guards.**
- The eval's own anti-vacuity row: a shuffled-target control must score at chance. The shuffle **permutes
  consequences across keys**, keeping each key's invocations together; a within-key shuffle leaves key
  identity predictive and can sit above chance for a reason unrelated to the model (confounding lens NIT-4).
- A two-process determinism check on the script's JSON output (the precursor of S3's test).

**Blast radius.** None. No ledger row moves on an offline result (weak-evidence rule: this is capability,
not a claim).
**Owner decisions at S2 start:** the kernel choice (open; recommended: a cosine kernel per block with fixed
block weights, ridge λ chosen by nested CV on the training keys only, before the held-out keys are read);
confirming caused-only nociception in the target (§4.1; strict default: caused only, felt reported). The
core valence formula (unweighted `relief − harm − nociception`, G11) and urgency v1 (pressure only, G12)
were decided 2026-10-09 and are not re-asked.

### S3 — session-end retrain, persistence, guards, shadow consumer (`src/`)

**What.**
- `memory/consequence_predictor.py::ConsequencePredictor` (`encode_context` / `predict_latent` /
  `encode_target`). The refit runs at `BioStack.on_session_end`, opt-in via a `maxim config` key (prefer
  config over a new env var; an unavoidable env var gets an autouse conftest scrub in the same commit).
- **The cradle seam, as its own commit with its own test:** `runtime/agent_factory.py`'s shutdown calls
  `BioStack.on_session_end` (which saves the Cerebellum and runs `distributor.cleanup_session`) instead of
  `save_cerebellum` alone, so the `--sim` orchestrator path reaches the refit (§4.2 item 3). The test drives
  the shutdown and asserts `on_session_end` ran; a deletion probe (revert to `save_cerebellum`) fails it.
  This seam is not behind the predictor's flag: it also starts running `cleanup_session` (the #888
  temporal-anchor clearing) on the `--sim` path, so it carries its own ledger walk (blast radius below).
- JSON persistence as in §4.2 item 4, with `check_format_version` on load and geometry refusal.
- The shadow log of predicted core, cluster and α per candidate tool, written to the existing JSONL
  trace surface only.

**Gate.**
- All guards are red before the change and green after, each proven by deletion (delete the join check,
  the provenance check or the geometry check, and the matching test must fail).
- A caller grep shows the refit is reached from `on_session_end` on both the cradle (`--sim`, through the
  agent_factory seam) and the Minecraft harness paths ("a fix ships with a caller").
- Three-lens code review, whose Wire-integrity lens writes the producer → consumer table for the pair,
  the persisted file and the shadow log.

**Guards.**
- `tests/unit/test_consequence_no_contamination.py`: a hand-built pair without a minted
  `PhysicalEventId`, a pair with no provenance, and a geometry-mismatched pair each RAISE at construction
  or at `add_pair`; a `narrated` pair relabelled `experienced` RAISES; a `narrated` pair's training weight
  equals the declared discount (deletion probe: drop the discount and the test fails).
- `tests/unit/test_consequence_predictor_two_process.py`: two processes with differing `PYTHONHASHSEED`
  train on the same capture and write byte-identical JSON. It must fail against a version that orders pairs
  by `dict` iteration.
- `tests/integration/test_persistence_compat.py` round-trip.
- The selection golden is byte-identical with the flag on (the shadow log has no selection effect).

**Blast radius.**
- The predictor: none by wording, since no selection, credit or EC symbol is touched.
- The cradle seam: it adds `cleanup_session` to the `--sim` path. T1-13's `Re-run on:` names
  "temporal-anchor pruning / `TemporalCreditDistributor` credit-path change". Its earned path (the
  Minecraft harness and the survival scripts) already calls `on_session_end`, so that path does not
  change. The walk is recorded on T1-13 and lists every earned row whose evidence ran on `--sim`; if one
  is affected, the owner rules on it before merge (strict default: a re-run).
- **T4a applies:** the predictor is a mechanism with no selection consumer until S5. If S5 has not earned
  by the 1.4.0 transaction, `ConsequencePredictor` is marked `Dormant since <date>` in its docstring and the
  release PR says so.

**Owner decisions at S3 start:** Q3 (imagined pairs); the config key name.

### S4 — opt-in `"consequence"` EC modality + similarity tiering

**What.** Registration of the modality (§4.2 item 5) and the α-blended similarity read (§4.2 item 6). Both
are behind the S3 config key, default OFF.

**Gate.**
- With the flag OFF, the EC dumps for interoception, audio, world and text are byte-identical before and
  after (golden).
- With the flag ON, consequence nodes form only from experienced-provenance predictions.
- `turn_left`/`turn_right` and the `touch` ×16 collisions separate by consequence on the S0b data where the
  identity block collapses them. This is an offline check on the recorded capture, and it is reported, not
  claimed.

**The hivemind edit.** Local-only does not come for free: `hivemind/bundle.py` exports every
`substrate_nodes` entry whatever its modality, so keeping `"consequence"` nodes out of a bundle IS a
`bundle.py` export-filter edit. On the receiving side, `hivemind/merge.py::SENSOR_MODALITY_THRESHOLDS` has
no `"consequence"` entry, so a foreign bundle carrying one would fold at the default cosine threshold, not
at 0.85. S4 therefore either adds `"consequence"` to `SENSOR_MODALITY_THRESHOLDS` or refuses the modality
at merge/ingest (Q4; strict: refuse it).

**Guards.** `test_ec_consequence_modality_golden.py`; `test_hivemind_excludes_consequence.py` (no
`"consequence"` node in an exported bundle, and a bundle carrying one is refused at merge, or folded at 0.85
if Q4 takes the threshold route).

**Blast radius.**
- Structural discharge of **T1-3** ("EC threshold or centroid-update change") and of the EC
  world-modality triggers on **T1-6, T1-10–T1-15**, via the byte-identical goldens.
- **T1-14 fires by wording** ("`hivemind/bundle.py` scrub … change"): the export filter is a bundle.py
  edit. **T1-11 and T1-12 fire by wording** ("`hivemind/merge.py` change", "`ec_merge_aligned` change") if the
  merge-side handling edits `merge.py`. Discharge is an owner ruling at S4 start (strict default: the
  offline guards re-run, plus a re-run of the rows' rig evidence batched with the next scheduled survival
  re-run, never a silent structural discharge).
- **Caveat:** adding the modality to `ECConfig.frozen_centroid_modalities` breaks its pinned equality with
  `hivemind/merge.py::DEFAULT_FROZEN_CENTROID_MODALITIES`, and that edit could be read as an "EC change".
  The recommended form keeps the pin untouched (Q4).

**Owner decisions at S4 start:** Q4, and the T1-11/T1-12/T1-14 discharge ruling.

### S5 — the behavioural experiment (the first grounding claim, T9)

**What.** Working title "consequence transfer". The **four-lens experiment DESIGN review**
(`docs/experiments/DESIGN_REVIEW.md`: confounding, bio-faithful, wiring, environment; all four, because it
is a new claim) runs on the prereg **before any harness is built**. Reports go verbatim into
`docs/experiments/rationale/consequence-transfer/`. Then the prereg lands on `main` before data, the harness
gets its code review, a dry run, and then the run. Substrate-primary selection, no LLM in the action path.
**The capture runtime is named per arm in the prereg** (the S0b rule: a production entry point whose
invocations reach `capture_loop_action`, never a hand-composed harness; wiring lens S10). Its arms
(redesigned by owner decision G13, 2026-10-09):
- **Cradle, authored physics: acquired equivalence (the test, §6).** New AUT-invoked items with a
  `self_effect` on the AUT, whose **sensed readings and consequences disagree**: pairs that read alike and
  act differently (a hidden property) and pairs that read differently and act alike, with names that point
  neither way. Stage 1 gives **experience of every item**; Stage 2 retrains one; the test reads which
  unretrained item inherits. Primary measure: the inheritance ordering and the first-contact choice after
  Stage 2, versus three comparators: the **no-learning sensed-similarity baseline** (object-block cosine,
  no fitted map; the gate is read against it), an identity-only (word prior) arm, and a yoked
  no-predictor arm. **Authoring rule:** the physics author is blind to the arm design, and every
  consequence must follow from each object's own `self_effect`/`target_effect` declaration, with no
  per-experiment tuning. The blind author now also writes items whose readings and effects disagree, so the
  rule no longer guarantees the sensed/consequence confound it used to strengthen.
- **The jet triad (§6), a control, not the test:** it measures generalisation from sensed features. It
  needs new AUT-invoked entities (§6); its result is reported, not gated, and "`water_jet` must not inherit
  the aversion" is no longer a criterion.
- **Minecraft, game-native physics: repositioned honestly.** Cradle consequences are YAML, so cradle
  results may only recover the author's regularities; in Minecraft the game supplies the consequences. But
  this arm, as designed, tests **consequence prediction, not transfer**: it is a graded, action-contrastive
  prediction of Δoxygen/Δhealth within a situation band for `escape_water` against a same-band control
  (S0b), the roadmap's "how far is pain" readout and, under S0a branch (ii), the bridge to E3. It is **not
  evidence for T9's transfer claim** unless a game-native transfer contrast (affordances whose names
  differ or collide, with game consequences) is designed into the prereg. It depends on the out-of-band
  producer (the post-return refill is outside the tool-path record) and runs only if S0b's Minecraft half
  passed. The cradle arm therefore carries T9 alone, and risk 1 (§9) stays open for it.

**Held-out ("untried") items.** This definition governs the zero-shot readouts (first contact in Stage 1,
the jet-triad control) and the prior term. In the acquired-equivalence test every item is experienced by
design, so the test reads each probe item's **change** from the end of Stage 1 to the end of Stage 2; the
probe items are not acted on in Stage 2, and the recorded components must show their per-tool channels
below unchanged over Stage 2. An item counts as untried only if **all** of these hold (confounding lens
SF-8): zero `reward_bias`; zero cluster history;
**no causal link on its `tool:<name>` signature** (`nac.observe` forms one on any success and
`ToolPainBridge` on negatives; `recommend_action` reads them as `causal_pos` / `causal_neg`); **no shared EC
text node** with any trained item at the production 0.44 (with `MAXIM_SUBSTRATE_PATH=1` the #1181 widening
spreads credit across names that share a node; a share that cannot be avoided is reported as the measured
word-prior channel); and **no Wire-4 situation-fear term** on it (Wire 4 is limited to `drive:health` /
`drive:oxygen`, so it matters on bodies with a health drive and on all of Minecraft). Every one of these
channels is kept identical across the predictor arm and the yoked no-predictor arm, and the gate is read
on decisions where the recorded `components` show `causal == 0` and `learned_bias == 0` for the probed
tools.

**The word prior's second path into selection: the tool-name keyword match (a confound to control).**
`decisions/nac.py::NAc.recommend_action` Component 3 adds `drive_value` when a drive name is a substring of
the tool name, and `drive_value × 0.7` when a `_DRIVE_TOOL_AFFINITIES` keyword is (`cold`/`thermal` →
`warm`, `fire`, `blanket`, `huddle`; `thirst` → `water`). `runtime/substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS`
maps `temp`/`thermal` to `cold`, so these rows fire on cradle bodies, and tool names are
`<entity>_<affordance>` (`embodiment/tool_bridge.py`), so a cold infant's `warm_rug_touch` and
`fire_stone_touch` both get the bonus, and `water_jet` matches `thirst`. The path is the same in every arm,
so it cannot by itself make a difference between arms, but it can make a **null** (a bonus of up to 0.7 on
a misleading item swamps a predicted aversion at first contact). The prereg controls it by one of: (a)
fixture names that contain no drive name and no affinity keyword, checked by a test over
`_DRIVE_TOOL_AFFINITIES` and `_DRIVE_CORRECTIVE_NEEDS` (recommended); (b) every drive held at or below 0.5 at
the probe; or (c) Component 3 counted as part of the word-prior baseline and the gate read on the recorded
`components`, net of `drive`. GL2b(i)'s keyword list (its new `"heat"` row) is frozen before the fixtures
are named.

**Gate.** Pre-registered, per arm, with margins fixed in the prereg. A recorded null ships as a null (T9).
The gate reads **experienced-only** pairs, per dimension, on one target schema id; the with-narrated
result is reported beside it and never gated (§4.1; G6 not reopened), and a result that holds only with
narrated pairs is reported as such. Apparatus outcomes are excluded (G16).
On EARNED: a **new** Tier-1 row with `Re-run on:` and `Regression guard:`. GL0 DROPPED T1-5 (2026-10-08)
and kept its ID for this link: the GL5 successor is linked `SUPERSEDED by` (T1-5 →
`SUPERSEDED <date> by T1-<new>`), and the new row cites T1-5 in its history. It is never a re-label of T1-5. A Tier-3 seed row at `SETUP` may be added
when the prereg merges.

**Guards.** The harness inherits S3–S4's guards, plus M10's fingerprint extended with the grounding flags
(grounding.md), so no E-rung arm runs with them set.

**Blast radius: the selection term.** The cradle arm's "untried-tool prior" is a selection effect, and it
applies **only to untried tools** (the full definition above), so experienced tools keep their earned
tables. The Component 3 keyword path is named here as part of the selection surface the arm runs on. Where it is placed decides what fires:
- **Inside `NAc.recommend_action` (`decisions/nac.py::NAc.recommend_action`).** This fires the
  "`recommend_action` change" trigger **by wording, even while default-OFF**, on **T1-7, T1-11, T1-12,
  T1-13, T1-14, T1-15** (Exp 45/56/57/60/61/62), and it sits inside the 1.3.2 fence.
- **The strict alternative (recommended):** a composing step **before** `recommend_action`, which re-orders
  or annotates the untried candidates it is handed and leaves the function untouched. No trigger fires by
  wording. A structural walk is still recorded on each of those rows, showing selection byte-identical with
  the arm off. Whether this "fires by effect" is the Architecture lens's question, and the walk is the
  answer.

Making the prior a default, non-experimental consumer is **GL6**, with its wide re-runs, and is not this
plan's.

**Owner decisions at S5 start (asked together):** Q5 (seam vs term); the acquired-equivalence item set
and its readout; the keyword-path control (a, b or c above); the capture runtime per arm; whether a
game-native transfer contrast is added to the Minecraft arm; the arms, n and margins;
the ledger row this would earn; whether T9's claim waits on E3's recorded outcome (G1: never
co-headlined).

---

## 8. Blast radius by stage

| Stage | Rows whose `Re-run on:` fires | How it is discharged |
|---|---|---|
| S0a, S0b, S2 | none | no `src/` |
| S1 | **T1-16 by wording** ("the memory record shape": `extra["context"]` is flattened into the persisted trace). T1-1 (Exp 10) carries the older schema wording but is SUPERSEDED | the same ruling GL2a's `extra["interoception"]` gets (one ruling for both keys); structural evidence: the ranker reads no `encoding` key, the round-trip is lossless; goldens byte-identical; the executing survival checks green |
| S3 | the predictor: none by wording. The cradle seam: a walk on T1-13 ("temporal-anchor pruning"), since `cleanup_session` starts running on `--sim` | goldens byte-identical with the shadow log on; the walk lists earned rows whose evidence ran on `--sim` (owner ruling if any); T4a Dormant marking if S5 has not earned by 1.4.0 |
| S4 | T1-3; EC world-modality triggers on T1-6, T1-10–T1-15 (structural); **T1-14 by wording** (the `bundle.py` export filter); **T1-11, T1-12 by wording** if the merge-side handling edits `merge.py` | structural, via the byte-identical golden with the flag off, the hivemind pin untouched; the by-wording rows take the owner's S4-start ruling (strict default: guards re-run plus a batched rig re-run) |
| S5 (strict seam) | none by wording; structural walk on T1-7, T1-11–T1-15 | byte-identical with the arm off |
| S5 (term inside `recommend_action`) | **T1-7, T1-11, T1-12, T1-13, T1-14, T1-15 by wording** | re-runs budgeted into S5, rig slot after E3 (Track C ordering) |

Not touched by any stage: `credit_node`, `TemporalCreditDistributor`'s credit path (S3's cradle seam
only adds an existing `cleanup_session` call to the `--sim` path), the shared EC encode path,
`_sensor_embed`/`gain_exponent` (Exp 62), `SensorEncoder` (Exp 56), `_drive_potential_diff` /
`drive_comfort_progress` (T1-9), the affordance decomposer (T1-5), and the `substrate_merge` fold (Exp 56).
The hivemind is touched once, in S4: the `bundle.py` export filter and the merge-side handling of the
`"consequence"` modality (Exp 61's T1-14 wording, above). The positive Reaction that lapses T1-13's #888 discharge is GL2c's, in
[autonomic_layer.md](autonomic_layer.md). This plan adds no Reaction.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Circular authored physics.** Cradle "transfer" may only recover the YAML author's regularities | High (confounding lens) | Blind physics author; consequences derived from each object's own declaration. The Minecraft arm is game-native but, as designed, tests consequence prediction, not transfer (§7, S5), so it does **not** close this risk for T9 unless a game-native transfer contrast is added; stated in the result |
| **Sensed similarity passes as consequence similarity.** Authored items' sensed readings and effects move together, so object-block cosine alone reproduces a "reversal" | Critical (confounding lens DNB-2) | G13: acquired-equivalence items whose readings and consequences disagree; every gate read against a no-learning sensed-similarity baseline; the jet triad demoted to a control |
| **Level terms let the predictor copy context** | High (bio lens SF-3) | Change-only target projection; a context-copy baseline in S2 and S5 |
| **The word prior's keyword path** (`recommend_action` Component 3) produces a null at first contact | Medium (confounding lens SF-6) | Keyword-free fixture names checked by a test, or a held drive level, or a components-net gate (§7, S5) |
| **Apparatus writes trained or credited as experience** (rescue, respawn, teacher, mother) | High (environment lens DNB-2) | G16: `apparatus` provenance, never a pair, never credit; S0b counts them |
| Contamination by curated pairs, or narrated consequences passing as experienced | Critical (fails the thesis) | Join invariant on the minted `PhysicalEventId` + required provenance + CI test (narrated never relabelled, discount always applied); every gate reads experienced-only data, with-narrated reported beside it; no manual surface |
| The narrator's world dominates training (the cradle's consequences are largely narrator-written) | High | The declared discount (G6); the with/without report; S0b counts narrated writes by tool before S2 |
| Too little data, or overfitting on hundreds of pairs | High | S0b's hard floors; strong ridge regularisation; held out by entity; a shuffled-target control |
| A non-deterministic join key (the executor's uuid4) leaks into persistence | High (silent) | `PhysicalEventId` only; the two-process test fails on uuid- or dict-order-dependent output |
| Word-prior geometry differs on a Pi (384-d fallback) | Medium | Per-block geometry tags; refuse on mismatch; train and evaluate on one declared encoder |
| The dormancy rule on the Cerebellum read side | Process | S1 touches only the write. The read resurrects only through S0a branch (i) + E7, or S5, with `test_cerebellum_dormant_909.py` edited in the same PR |
| The replay buffer is lossy (queue drops, consolidation pruning) | Medium | S0b counts drops; the persisted training set keeps `pid`s and vectors at each refit |
| Magnitude is invisible to cosine | Medium | Nodes carry kind; magnitude is read from the predicted code |
| Most affordances (363 of 405, by the YAML walk) have no declared consequence | Medium | Stated, never imputed: those keys have no predicted consequence, and α stays at the prior for them |
| Two may-fail lines in one release | Medium | T9 conditional, never co-headlined with E3 (G1); grounding flags excluded from E-arms (M10) |
| Divergence across S2 iterations | Process | One pre-registered MLP retry at most. Two divergent iterations → stop and audit the layer beneath (the autonomic target and the capture), per the cycle-divergence rule |

---

## 10. Owner decisions

**Decided 2026-10-07 (not re-opened here):**
- **G1:** placement in Phase 5's slot; T8 engineering-only, T9 conditional and never co-headlined with E3;
  body world first; S0a and the paper run now, S0b needs a fresh capture (a sim run), S1 waits only for
  GL2a (G1's record-only fence exemption; its exempt file set is in the header; **amended 2026-10-09 by
  G15**: S1 also waits for the cross-session pid resume, which lands after the fence), S2 is offline, and
  S3 and S4 wait for the fence.
- **G2:** the predictor enters as this plan; the four rules are carried over; it is not called JEPA until
  its target is a learned rich-percept embedding. It also settles that the Phase 5 audit becomes S0a, S0's
  companion, whose verdict decides scope, not existence, and that the deferred JEPA banner is re-pointed
  here without the line being called "JEPA".
- **G3:** `Receptor`, `AfferentTrack`, `AfferentEvent`, and `PhysicalEventId(agent_id, session_id, seq)` (G15) as the join
  key: one frozen type in the leaf module `maxim/embodiment/event_id.py`, one per-agent
  `EventSequencer`; pairs join on `pid` only. (`seq` "persisted from GL2a" is **amended 2026-10-09 by
  G15 and G17**, below: the type and its sequencer are built at the post-fence resume stage.)
- **G5:** `deferred/jepa_cross_modal_alignment.md` is SUBSUMED by this plan;
  `grounded_language_acquisition.md` is SUBSUMED for grounding by grounding.md.
- **G6:** narrator-written consequences are stamped `narrated`, never `experienced`, and train at a
  declared discount (§3.6, §4.1); S5 reports with and without them.
- **G4:** Minecraft satisfies the perception abstraction. This plan's Minecraft arm needs no robot trigger.

**Decided 2026-10-09 (GL1 design-review decisions; DECISIONS.md "2026-10-09 — Grounding GL1
design-review decisions (G9–G20)"):**
- **G11 / G12:** valence = relief − harm − nociception, unweighted (innate prior), with `drive:health`
  contributing once by definition; urgency v1 = pressure only.
- **G13:** GL5 = an acquired-equivalence design: authored items whose sensed readings and consequences
  disagree, experience of each item required, the model gated against a no-learning sensed-similarity
  baseline; the jet triad becomes a generalisation-from-sensed-features control. T9 keeps its claim (§6,
  §7 S5).
- **G14:** GL2a's exempt set includes `Executor._run_started`; the record is scoped to the invoked
  affordance's declared drives and reported net of declared drift; the existing trio stays byte-identical.
- **G15:** GL2a mints session-unique pids and persists nothing; the cross-session resume (own high-water
  mark, both load seams tested) lands after the fence and before GL4 S1 (§3.3). *Its GL2a half is
  superseded by G17.*
- **G16:** a harness-scoped `apparatus` provenance (rescue/teleport/heal, respawn, teacher and mother
  writes): recorded, never a training target, never credit (§3.6).
- **G17:** GL2a's `InteroceptiveOutcome` ships without an event id. `PhysicalEventId`, the per-agent
  sequencer, its session-id source and the cross-session resume land together at the post-fence resume
  stage, before S1 (the first joiner); records written before it carry no pid and are never training
  data; `CauseRef.cause_pid` lands then too (no deterministic session id exists today, §3.3).
- **G18:** S0a **is** the audit roadmap 1.4 Phase 5 requires: one audit, one verdict.
- **G19:** T1-5's successor (the GL5 claim, this plan's S5) lives in the **cradle**, where every
  harm-class collision the GL1 census found sits; Minecraft stays a consequence-prediction arm.
- **G20:** the L0 gate is a measurement only; PASS or FAIL moves nothing.

**Open, each asked at the start of the stage named (recommendations in bold are the strict option where
one is offered):**

| # | Decision | Ask at | Recommendation |
|---|---|---|---|
| Q3 | Imagined-entity pairs: exclude, or down-weight ×0.5 (mirroring `decay_imagined_links`)? | S3 start | **Exclude** |
| Q4 | The consequence modality and the hivemind: local-only with a separate per-modality freeze field, or update the frozen-modality pin now? | S4 start | **Local-only; the pin is untouched** |
| Q5 | S5's selection effect: a composing step before `recommend_action` (no trigger by wording), or a term inside it (fires T1-7, T1-11–T1-15)? | S5 start | **The composing step** |
| Q6 | Cerebellum key on the TARGET entity? Add the situation cluster to the key? | S1 start | Target entity yes; situation no (the predictor carries it, and the key format stays) |
| Q7 | Does S0b's capture include an LLM-driven cradle arm? | S0b start | **No.** About 10× fewer invocations and a selection-bias confound; scripted-substrate + Minecraft only |
| — | The narrated discount's value (G6) | GL4 start (with S0a) | **A small discount**, pinned before S2's data; every S2 and S5 result reported with and without narrated pairs |
| Q8 | How does a narrated write that adjudicates the AUT's own action pair with that action's context (separate `pid`s)? The writers exist (§3.6) | S0b start | Settled in S0b's prereg after the narrated-write count; never a join on anything but `pid`; stamped `narrated` before any pair is built |
| — | Minecraft capture: a rig slot in Track C, or the offline scripted bridge? | S0b start | The rig, in a slot never stacked on an E-rung campaign |
| — | Kernel and block weights | S2 start | Cosine kernel per block, fixed weights, ridge λ by nested CV on training keys only |
| Q9 | S0b's capture scene, body and runtime (a purpose-built all-items scene; the fixture-orchestrator substrate-primary runtime if it captures, else a production runtime after the fence) | GL4 start (with S0a) | **Name all three before the capture**; count reachable keys and classes per body first |
| — | Caused-only nociception in the target | S2 start | **Caused only**; felt-only reported |
| — | T1-5's status (decided at GL0 start) | GL0 | **DROPPED** (strict). Non-strict alternative: `PARTIAL <date> (narrow: compound-name level)`, whose reason is to keep the ID live for the eventual `SUPERSEDED by` link from S5's new row |
| — | S5's arms, n, margins and the ledger row it would earn | S5 start | A new row, never a re-label of T1-5 |

---

## 11. Record edits this plan requires (made in the PR that merges it)

- `roadmap_1_4.md` §Phase 5 "A graded predictor": the pointer to `latent_forward_model.md` becomes a link,
  with a dated note that the plan opened on owner decision G2 and that the mandated audit is its S0a.
- `roadmap_1_4.md` §JEPA: reduced to a three-line pointer here, recording the reversal of decision 3.
- `deferred/jepa_cross_modal_alignment.md`: a dated banner, "SUBSUMED 2026-10-07 by
  `latent_forward_model.md` (owner decision G5; owner reversed roadmap 1.4 decision 3); its four rules are
  carried over there".
  The file stays in place, because three `src/` files cite its path.
- `DECISIONS.md`: covered by the umbrella's 2026-10-07 grounding-line entry. This plan adds the honest-naming
  rule to it.
- `engram_formation.md` E7: owner pointer → this plan's S0a.
- Ledger T3-4 (`anticipatory_pre_activate`, DORMANT): resurrection trigger re-pointed to S0a's verdict.
  This edit lands in **GL0's PR**, which owns the ledger, not in the PR that merges this plan.
- `docs/plans/README.md`: listed under the grounding entry in §Active.
- Mechanization: the provenance rule on training pairs is outstanding.md M41 (backlog row until S3 lands; structural thereafter: the pair type's constructor + `tests/unit/test_consequence_no_contamination.py`). The guard lands in S3 together
  with the first training set.
