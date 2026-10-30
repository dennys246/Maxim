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
> later as the innate-prior tier. **Timing (status 2026-10-24):** **T8 (b), this plan's four-lens design
> review, is DONE (2026-10-24) and folded** ([§12](#the-t8-b-four-lens-design-review-2026-10-24); owner
> decisions G21–G27, §10). GL4 S0a (audit) has not started; it and the paper (the S0b prereg and script)
> stay homed in Session E (paper only; roadmap_1_3_x.md); S0b needs fresh captures (sim runs, G23). S1 is
> record-only `src/` inside G1's exempt file set, but since it waits for the post-fence resume stage
> (G15, G17) it lands after the fence anyway, so that exemption is moot. Its files, exactly (amended
> 2026-10-24 by G22 and G25): `embodiment/tool_bridge.py` (the observe call and the `get_confidence` →
> `sim_cerebellum` line are **deleted** there; wiring lens D1), `embodiment/cerebellum.py` (payload `"1.2"`,
> the new reafference payload key beside `models`, `import_state` refusing newer versions with a typed
> error, the Bio-mapping docstring), `runtime/executor.py::Executor._stamp_invocation` (the observe call,
> moved here, and the `ActionContext` assembly), `runtime/executor.py::Executor._run_started` and
> `::Executor._outcome_start`, which `_run_started` calls (the pre-action body read widened to the body's
> sensors, inside G14's scope; wiring lens D3; the pid minted before `tool.run`, §4.1; pre-merge review
> Wire N5),
> `tools/base.py::ToolOutput` (the `repr=False` `action_context` field; wiring lens S3),
> `runtime/bio_integration.py` (`EncodingSignals.extra["context"]` at the loop capture) and the new leaf
> `embodiment/action_context.py` (the `ActionContext` type). (GL2a's own list is `embodiment/body.py`,
> `embodiment/sem.py`,
> `runtime/executor.py::Executor._run_started` + `::Executor._stamp_invocation` (widened by owner decision
> G14, 2026-10-09), `tools/base.py::ToolOutput`, `runtime/bio_integration.py`; `embodiment/event_id.py`
> left it by G17.) GL4 S1 waits for GL2a **and for the post-fence resume stage** (owner decisions G15
> and G17, 2026-10-09: GL2a mints no pid at all; the `PhysicalEventId` type, the per-agent sequencer,
> its session-id source and the cross-session resume land together after the fence and before S1, the
> first stage that persists a pid across sessions; GL2a records carry no pid and are never training
> data; GL2a merged as `660d2806`/`485225d7`). S3 and S4 are `src/` and wait for the agent_loop/orchestrator
> decomposition fence; S3's cradle seam sits inside the fenced `start_simulation_mode` /
> `run_agentic_loop` and lands after 1.3.6 (G25, wiring lens D2); S2 is offline (`scripts/`) and waits for
> S0b's data.
>
> **GL1 four-lens design review (2026-10-09).** The umbrella and `autonomic_layer.md` were reviewed by the
> four lenses ([confounding](reviews/grounding_gl1/confounding.md),
> [bio-faithful](reviews/grounding_gl1/bio-faithful.md), [wiring](reviews/grounding_gl1/wiring.md),
> [environment](reviews/grounding_gl1/environment.md)); the findings that land on this plan's stages
> (S0b, S2, S5 = GL5) are folded here, with owner decisions G13, G15 and G16. **This plan's own four-lens
> design review (T8 (b)) ran 2026-10-24 at `d9dffe6e` and is folded** (§12; G21–G27); S0b's prereg is its
> first reader, and S5 gets a further one on its prereg before any harness. GL1's measurement is the census
> ([scripts/grounding_census.py](../../scripts/grounding_census.py),
> [docs/experiments/grounding_census.md](../experiments/grounding_census.md)), which S0b's floors are set
> against.
>
> **Every mechanism here enters as `[engineering]`.** Each behavioural change ships opt-in and default-OFF,
> with today's behaviour pinned byte-identical. Two named exceptions are not behind a flag (pre-merge
> review, Architecture S6 and N2). (1) The `--sim` **AUT's** teardown reaching its own
> `BioStack.on_session_end` (wiring lens D2, #1449). It corrects a skipped session-end call rather than
> adding a mechanism, and it is **its own Session-A slice after the fence**, with its own test and ledger
> walk; S3 depends on it and does not contain it. (2) S1's write-side changes: the Cerebellum's reafference
> target and the observe's move into the executor. The move narrows Cerebellum training to executors that
> mint a record, and widens it to the tools that were generated without `cerebellum=` (S1). Both are
> harmless to selection, because the Cerebellum's read side is Dormant. Nothing here powers an E1–E3 arm
> unless the arm declares it (M10, extended per grounding.md).

**Owns (proposed):** `src/maxim/memory/consequence_predictor.py` (new, S3), the `ActionContext` type in the
new leaf `embodiment/action_context.py` (S1) and its assembly in `runtime/executor.py::Executor._stamp_invocation`
(S1; G25, wiring lens S3), the Cerebellum's observe call site, moved into the executor, and payload `"1.2"`
(S1; G22), the `"consequence"` EC modality registration, its EC-only encode function and the node-keyed
outcome learner with its per-item membership map and harness-only freeze trigger (S4; G21; the learner's
home is S4's front-gate answer, Q11), `scripts/consequence_pair_audit.py` (S0b),
`scripts/train_consequence_predictor.py` + `scripts/eval_consequence_predictor.py` (S2),
`<home>/consequence_predictor.json` (S3).
**Companion plans:** [grounding.md](grounding.md) (umbrella, GL0–GL6) ·
[autonomic_layer.md](autonomic_layer.md) (the target, `InteroceptiveOutcome`) ·
[thalamic_relay.md](thalamic_relay.md) (`Receptor`, `AfferentTrack`, `AfferentEvent`; it uses
`PhysicalEventId`, which lives in the leaf module `maxim/embodiment/event_id.py`, built at the autonomic
plan's post-fence resume stage, G17) ·
[engram_formation.md](engram_formation.md) E7 (the Cerebellum read-side resurrection route) ·
[lookback_primitive.md](lookback_primitive.md) (the Hippocampus is the replay buffer; no new store) ·
[deferred/three_factor_credit_assignment.md](deferred/three_factor_credit_assignment.md) (R4's map) ·
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
senses" (owner decision G13, 2026-10-09). Its mechanism is **outcome-mediated**: items that predict the
same outcome come to share a representation (representational compression, Gluck & Myers 1993), held in
the hippocampal region and entorhinal cortex (Coutureau et al. 2002), and a fast learner keyed on that
shared representation carries later learning from one member to the others (bio lens N2, D1). That is why
S4 builds the shared representation in the EC and a learner on it (G21, §4.2 item 5).

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

## 3. Current state in code (written at origin/main f1c833ab; re-read at d9dffe6e by the T8 (b) review)

Claims checked at a `file::symbol` are stated as fact. Anything else is marked **UNVERIFIED**. The four
T8 (b) lenses re-read every §3 claim at `d9dffe6e` and found them holding, with three changes since
`f1c833ab` (wiring lens N1, environment lens N1): GL2a has merged (`660d2806`/`485225d7`; §3.3), the
record's shape differs from what §4.1 first assumed (§3.3), and the `--sim` session-end path differs from
what §4.2 item 3 first said (§4.2 item 3).

### 3.1 The Cerebellum: a forward model that already exists, and learns the object's response, never the body's

A forward model of an object's state after manipulation is a real cerebellar job (tool and object internal
models, Imamizu et al. 2000). The defect is that the body's change never reaches the Cerebellum, not that
the object model is wrong; S1 **adds** the body's reafference and keeps the object model (bio lens N1, D2;
G22).

| Fact | Evidence | Status |
|---|---|---|
| Key = `ModelKey(entity_path, modulator, affordance, param_bucket)`. Params are bucketed at 10% of the sensor range, or step 5.0 | `embodiment/cerebellum.py::ModelKey`, `::bucket_params` | live |
| Per key, `ForwardModel` keeps a per-sensor running mean and variance (single-cue Rescorla-Wagner, lr 0.2). Confidence = `1 − exp(−0.3·n)`, capped at 0.95 | `::ForwardModel.update`, `::ForwardModel.confidence` | live |
| **The write is LIVE, on tools built with a Cerebellum.** Every SEM affordance call on a tool generated with `cerebellum=` (`runtime/bootstrap.py::build_executor` → `Executor.generate_entity_tools`) reaches `observe_from_action(entity=self._entity.full_path, …, actual=entity_state)`. Tools generated without it never train. These are the `--sim` generative cradle path's scene tools (`simulation/generative_runner.py::_load_world_entities` / `_activate_phase_entities`), the DM `SceneState` tools under `campaign_runner.py` (which passes none), and the ephemeral `OrchestratorActorTool` tool (pre-merge review, Wire SF-3) | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` (`self._cerebellum is None` skips it) | live (it was a silent TypeError from 2026-04-07 to 2026-09-01; guard `tests/unit/test_cerebellum_wiring.py`) |
| **It learns the owning entity's readings, under exact keys.** `entity_state` holds the sensors of the entity that OWNS the modulator, read after `modulator.execute` and before `_apply_sensor_deltas(self._embodiment.root, _self_effect)` writes to the body. On a cradle path whose tools carry a Cerebellum, `fire_pit.touch` teaches it the fire pit's `heat_output`/`fuel` and never the infant's `arms.thermal +0.6`. On the `--sim` generative cradle path it teaches nothing today, because those scene tools carry no Cerebellum (row above). In Minecraft the `avatar` modulator sits on the player root, so it learns ABSOLUTE post-action readings, not deltas, and their meaning differs per world | same function, the order of the `entity_state` loop and the self_effect block; `_data/components/items/cradle_fire_pit.yaml`; `_data/components/bodies/minecraft_player.yaml` | live defect for this purpose |
| The read side is Dormant: `predict`, `observe_action_sequence`, `query_engrams`, `cleanup_program`, and the `CerebellumModulator` backend | "Dormant since 2026-10-04 … #909" docstrings; `backends/cerebellum_modulator.py`; AST guard `tests/unit/test_cerebellum_dormant_909.py` (it scans **`src/` and `scripts/`**, so even an offline script that calls `cerebellum.predict` fails it) | dormant; resurrection is routed through `engram_formation.md` E7 |
| It is persisted (since #908): `<home>/cerebellum.json`, payload `"version": "1.1"` + `_format_version`, key string `a\|b\|c\|d` | `runtime/bio_stack.py::_build_cerebellum`, `BioStack.save_cerebellum` | live |
| **A second live reader of the key:** right after the observe, `get_confidence(self._entity.full_path, …)` is logged through `sim_cerebellum`. It is keyed on the OWNER, so it must move with the key | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`; `simulation/sim_logger.py::sim_cerebellum` | live |
| **`import_state` loads an unknown version with only a warning.** Any version other than `"1.0"`/`"1.1"` logs "Unknown Cerebellum state version" and then reads `models` as usual, so a newer file would be misread rather than refused (a downgrade hazard) | `embodiment/cerebellum.py::Cerebellum.import_state` | live |
| **`load_safe` treats an unreadable store as corrupt and replaces it.** An exception in `UNREADABLE_STORE_ERRORS` (`ValueError`, `KeyError`, `TypeError`, …) makes it copy the file aside and start fresh, so the next save overwrites it. A refusal raised as one of those would set a newer file aside and overwrite it (wiring lens S4) | `embodiment/cerebellum.py::Cerebellum.load_safe`; `utils/store_ownership.py::UNREADABLE_STORE_ERRORS`; `runtime/bio_stack.py::_build_cerebellum` | live |
| **The observe runs inside the tool, before the body record exists.** `ModulatorAffordanceTool.execute` observes while `tool.run` is still running; the invocation's pain, drift window and `InteroceptiveOutcome` are built after the tool returns, in `Executor._stamp_invocation` → `Executor._interoceptive_outcome` (wiring lens D1). `ModulatorAffordanceTool.execute` is pinned at 502 lines (Rule T applies) | `runtime/executor.py::Executor._stamp_invocation`, `::Executor._interoceptive_outcome`; `utils/function_length_baseline.json` | live |
| The first observation sets the mean to `actual` with error 0 (a one-shot copy, lr = 1), so the docstring's "IS the Rescorla-Wagner delta rule" is wrong on trial 1 and first-contact surprise is lost (bio lens N3). It matters only if a reader of the error resurrects (S0a, E7) | `embodiment/cerebellum.py::ForwardModel.update` | live, unread |
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
  **a drive moved away from comfort is recorded as 0.0**, so that trio carries no signed harm.
- **GL2a has merged** (`660d2806`/`485225d7`). `embodiment/sem.py::InteroceptiveOutcome` is on
  `ToolOutput` (`repr=False`) and in `EncodingSignals.extra["interoception"]`. Its shape, as shipped, differs
  from §4.1's first draft (wiring lens S1 and N1, environment lens N1):
  - the core is **five** fields (`sem.py::_OUTCOME_CORE`: nociception, drive_pain, relief, harm,
    urgency); there is no `valence`, no `as_vector` and no `schema_id` on the record;
  - the per-drive block covers **only the invoked affordance's declared drives** (G14;
    `sem.py::affordance_declared_drives`, `Executor._outcome_start`), so it is ragged across keys, and
    absent drives are never imputed;
  - nociception is one value, `extra["nociception_basis"] = "caused_or_felt"`: the split waits for the
    `ToolPainBridge` accessor at the out-of-band producer stage, after the fence. On the cradle tool path
    every covered drive is `caused=True`;
  - `_outcome_start` returns `None` for a body with `agent_id == ""`, so no record is built there;
  - `invocation_id` is `compare=False` and not persisted (`to_dict`).
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
  `extra["context"]`; the Cerebellum's per-key means are aggregates and hold no per-sample pid, wiring
  lens N4): the sequencer persists its own
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
(object, action) pair is not a situation. None of them is keyed on a **consequence**: value learned on
one tool stays on that tool's entry, so no item can inherit it from another that shares its outcome
(bio lens D1; why S4 adds a node-keyed learner, §5). One table *is* keyed on a node:
`NAc._reward_bias`, keyed `(agent_id, node_id)` and written by `NAc.credit_node`. §5 says why it cannot
carry a consequence value (pre-merge review, Architecture B1).

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

**The shipped record is not yet this vector** (wiring lens S1, environment lens N1; §3.3). It has a
five-field core with no `valence`, no `as_vector` and no schema id, and its drive block covers only the
invoked affordance's declared drives (G14). So:
- **The projection is a pure function owned here**: written in S2's script, moved into
  `memory/consequence_predictor.py` in S3. It is never a method added to the shape of the GL2a record
  without GL2's own review. Valence, where reported, is computed by it from G11's formula.
- **Absent drives are a mask, not a zero.** A zero would claim "no change" for a drive the action never
  touched. The body-order block is filled from the record's declared drives, with the rest masked.
- **The projection stamps the target schema id (`ans-v1`) on every pair**, since the record carries none
  (or GL2 adds `extra["schema_id"]`, its own decision).

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
reported, never trained on (strict default; confirmed at S2 start). **Felt-only invocations are masked on
the nociception dimension**, not labelled 0, because a 0 label would teach "no pain" where pain
co-occurs (confounding lens N6). **The split does not exist yet** (wiring lens S2, environment lens S4):
the shipped record holds one `caused_or_felt` value, and `ToolPainBridge.pop_invocation_pain` returns
caused, else felt, so neither S0b's frozen script nor S2 can separate them. Until the split lands at the
out-of-band producer stage (after the fence), **`nociception` is reported, never gated**, and S0b's and S2's
gates read `harm` and `relief`. In the cradle it would be vacuous anyway: `proprioception/pain.py::TISSUE_DAMAGE_DRIVES`
is `{"drive:health"}`, the infant's `health` is derived, and the only nociceptive cradle pain
(`damage_component`) is narrated, so every experienced cradle record has nociception 0 under `ans-v1`.

**One schema id per training set.** GL2b(ii) changes what `nociception` means for the same event (the
infant burn moves from drive pain to nociceptive pain), and bumps the schema id (`ans-v2`). A training or
evaluation set never spans a schema id: S0b, S2 and S3 refuse a mixed set (confounding lens SF-2). Because
GL2a persists the per-drive block beside the core, a later projection can be recomputed from the replay
buffer.

**Gates read per-dimension signs, never valence.** S0b, S2 and S5 gate on `nociception`, `harm` and
`relief` separately; valence is reported beside them as the innate-prior summary (confounding lens SF-2).
`nociception` joins the gated set only once the caused/felt split has landed (above; wiring lens S2).

The predictor trains on the **core first, across bodies**. A cradle burn and a Minecraft drowning then land
in one space. The per-body block is trained **per body** (keyed on `body_path`), only where a body has at
least the S0b floor of pairs. An earlier draft defined a separate `ConsequenceCode`. It is dropped ("merge
before multiplying"), because a second record of the same consequence would let the two drift apart. The
**target encoder** is that change-only vector plus a per-dimension standardisation. **The procedure is
pinned before S2 reads data; its numbers are fitted on training folds only**, like λ, never over held-out
keys or invocations (confounding lens S5, D3). The standardisation **scales without centring**, so a zero
change stays the zero vector after every transform; the S4 modality encodes the raw (unstandardised)
predicted or observed change (confounding lens S7).

**Context: raw inputs recorded, blocks computed offline** (owner decision G25, wiring lens D3). An earlier
draft had S1 record embedded blocks. It cannot: no situation vector reaches the tool path (the decision-time
encode is in the fenced `runtime/substrate_proposal.py::propose_via_substrate` / `_encode_current_clusters`,
and only cluster ids and margins travel on `LLMProposal`); the only identity embed door,
`similarity/encoder.py::LinguisticEncoder.embed`, writes `ec.record_encoder_provenance("linguistic", …)`
into the persisted `ec.json` and loads mpnet on a hot path that never loads it; no registered `Receptor`
carries a scene entity's own sensors (GL3.B1 registers `body.drives`, `audio.doa`, `world.state`); and
`EncodingSignals.__post_init__` runs `json.dumps(extra)` on the loop thread, about 1.5k floats per
invocation. So S1 records **raw inputs** only, and S2 (offline) and S3 (at the refit) embed them, each block
stamped with the geometry tag it was computed under. `_sensor_embed` is pure and deterministic, so the
offline encode is byte-identical to a live one under the same tag. No hot-path embed, no `ec.json` side
effect, no fenced edit, and the T1-16 trace growth drops by about two orders of magnitude.

```python
@dataclass(frozen=True)
class ActionContext:
    """CC3 path (a): defaults on every field + extra (JSON-safe, no key collisions).
    Persists inside the predictor's training set, so the forward-compat path is declared.
    Raw inputs only; embedding happens offline (S2) or at the refit (S3)."""
    pid: PhysicalEventId | None = None   # the ONLY join key, shared with the InteroceptiveOutcome.pid of
                                         # the same event (G3; no uuid, no wall time); exists from the
                                         # post-fence resume stage (G17); a training pair refuses None
    entity_path: str = ""                # := GL2a's CauseRef.entity for the same event (the acting entity,
                                         # "" for the body's own modulators); wiring lens S3. A resolved
                                         # target= is recorded separately in extra if one is ever needed
    entity_name: str = ""                # feeds "identity" (with affordance)
    affordance: str = ""
    param_bucket: str = ""               # reuses cerebellum.bucket_params
    params: tuple[tuple[str, float], ...] = ()          # feeds "params"
    object_readings: tuple[tuple[str, float, float, float], ...] | None = None
        # (sensor, value, lo, hi) of what a receptor DELIVERED to the AUT for this entity before the
        # decision (environment lens D2); None = absent (nothing sensed), never an empty tuple / zero vector
    body_pre: tuple[tuple[str, float, float, float], ...] = ()
        # the AUT's pre-action body read from Executor._outcome_start's snapshot, widened to the body's
        # sensors inside G14's _run_started scope: "situation" AT EXECUTION TIME, kept distinct from the
        # credit key's decision-time cluster ids, which are already on the trace
    provenance: str = ""                         # REQUIRED: experienced | narrated | imagined | apparatus.
                                                 # No default to "experienced": __post_init__ rejects "".
                                                 # An apparatus pair is never constructible (G16)
    extra: dict = field(default_factory=dict, hash=False, compare=False)
```

The blocks the predictor reads are built from these fields, offline, under recorded geometry tags (a
mismatch refuses):
- **`identity`**: an embed of `"<affordance> <entity name>"` (inherited prior; innate-prior tier), through
  an **EC-free embed door** (one that writes no `record_encoder_provenance`), named in S2 and reused in S3.
  GL3.B1 (thalamic_relay.md §5.7 item 3) will make `provenance=` a required keyword on
  `LinguisticEncoder.encode*`; the door follows it.
- **`object`**: `_sensor_embed` over `object_readings`. **Absent** when nothing was sensed; S2 gates only on
  keys with a block (environment lens D2, G25). In Minecraft every affordance sits on the `avatar`
  modulator of the player root, so the "object" would duplicate the body: **the block is absent there**.
- **`situation`**: `_sensor_embed` over `body_pre`.
- **`params`**: normalised numeric params plus `param_bucket`.

**What the object block is** (environment lens D2, G25). Shipped cradle item sensors are constants nothing
writes, and the AUT perceives them only through a deliberate `sense` act
(`tools/discovery.py::UniversalSenseTool`; the orchestrator deregisters the per-entity sensor tools), as
text. Reading `tool_bridge`'s `entity_state` as the object block would hand the learner ground truth no
receptor delivered (a synthetic sensor). So the block is **what a receptor delivers to the AUT**: before
GL3.B1, the AUT's own `sense` result for that entity, captured in its percept stream, and only if it was
sensed before the decision; after GL3.B1, an exteroceptive `Receptor` (none is registered yet for scene
entities, wiring lens D3). "Reads alike" means **shared sensor names with near values under
`_sensor_embed`**, which hashes per sensor name (`similarity/encoder.py::_sensor_embed`).

On `ToolOutput` the `ActionContext` field is declared `repr=False` (an edit to `tools/base.py::ToolOutput`,
in S1's file set), and `str(ToolOutput)` is pinned byte-identical with and without it (#1189, §3.3).
**Entity identity** is how the agent perceives *which thing* through its name, which is the inherited prior
(the innate-prior tier, §1). **Action params** are the `params` block plus `param_bucket`.

A training example is `(ActionContext, InteroceptiveOutcome)` joined on `pid`, and on nothing else.
**The join is the contamination guard's anchor.** A pair whose `pid` was not minted by the agent's
`EventSequencer` in a provenance-stamped run (the M1 stamp: executed maxim file, git hash, clean tree)
cannot be constructed. `ActionContext.provenance` and `InteroceptiveOutcome.provenance` are one fact
recorded twice: the pair constructor refuses a mismatch (wiring lens N3; M41's test carries the case).
Each pair carries its provenance and a weight: `experienced` pairs weigh 1;
`narrated` pairs weigh the declared narrated discount (G6, value set at GL4 start); `imagined` pairs follow
Q3; an `apparatus` outcome builds no pair at all (G16). **Every gate in this plan reads experienced-only
data**; with-narrated numbers are reported beside it and never gated, because a narrated target is written
by an LLM reading entity and affordance names, so it carries the very word prior the claim says
experience overrides (confounding lens SF-7; G6's training discount is not reopened). A narrated write that adjudicates the AUT's own action is a separate physical event with its own
`pid`; how such an outcome is paired with that action's context, without joining on anything but `pid`,
is settled in S0b's prereg, which counts these writes first.

**The GL3.B2 handoff, this plan's side** (wiring lens S8; strict option (a); the pid seam settled by the
pre-merge review, Wire B3). The pair's `pid` is the **tool-path invocation pid**: **one pid per
invocation, minted at `Executor._run_started` before `tool.run`** (once the post-fence resume stage has
built the type and the sequencer, G17), and stamped as `InteroceptiveOutcome.pid` at
`Executor._stamp_invocation`, which reuses it and never mints a second. The context is assembled per
invocation in the executor, not per registry pass. The same pid is held for the duration of `tool.run`
in an execution-context value on the thread that runs the tool. GL3.B3's H3t emit (the burn transduced
inside `ModulatorAffordanceTool.execute`, during `tool.run`) **reuses** it for that burn's track copies;
it does not draw a new one. So one tool-path physical event carries one pid across its pair, its
`InteroceptiveOutcome` and its H3t track copies, as thalamic_relay.md §5.10 ("One physical event, one
pid") and autonomic_layer.md §3.1.2 require. Minting after `tool.run` returned would put the H3t emit
ahead of its own pid. A receptor pass that is not on the tool path (a pull read, the drift tick) draws its
own pid from the same per-agent `EventSequencer` (G3), so such an `AfferentEvent` pid is never a tool-path
pid. GL3.B2's gate therefore reads "every row's pid resolves to exactly one physical event: one tool-path
invocation (its `InteroceptiveOutcome` and any H3t track copies) or one receptor pass". Its fence row reads
"GL4 S1" (S2 is offline `scripts/`). The matching GL3.B2 / GL3.B3 and autonomic §3.1.1–§3.1.2 wording is
those plans' to amend. The
`object` block does not arrive through a GL3.B1 receptor until an exteroceptive one for scene entities is
registered (wiring lens D3). Whether S3's contamination guard is GL3.B1's landing consumer (G8), or that is
left to S5, is asked at S3 start (§10). The matching edit in thalamic_relay.md §GL3.B2 and §8 is that
plan's.

### 4.2 Where it hooks in

1. **Pair assembly (S1, record only), in the executor** (G25, wiring lens S3). Executor-stamped `ToolOutput`
   fields are documented "Tools never set it" (`tools/base.py::ToolOutput`), and no channel exists from the
   tool to the executor. `Executor._outcome_start` already reads everything the context needs through the
   #1195 read-only accessors (`affordance_name`, `acting_entity`, `modulator`, `affordance_schema`), so the
   `ActionContext` is assembled there and in `Executor._stamp_invocation`, the single per-invocation writer,
   symmetric with GL2a's `ToolOutput.interoceptive_outcome`. It records raw inputs only (§4.1). The pair
   rides the existing Hippocampus `"loop"` capture as `EncodingSignals.extra["context"]`, next to GL2a's
   `extra["interoception"]`. No `EncodingSignals` field changes, and no new store is created. It does
   change the persisted record: `EncodingSignals.to_dict` flattens `extra` into the trace that
   `memory/types.py` saves, so each loop trace grows by the raw context (strings, a few dozen readings and
   ranges, params; no embedded vectors). The growth per trace and per file is measured in S1; the ledger
   consequence is in S1's blast radius.
2. **The online, key-specific model is the Cerebellum, learning the body's signed reafference (S1; owner
   decision G22, bio lens D2, wiring lens D1).** A cerebellar forward model predicts the **sensory**
   consequence of a motor command (Wolpert, Miall & Kawato 1998); it does not value the outcome against the
   body's need, which is orbitofrontal, amygdalar and insular work. The first draft's target (the 6-d core
   plus the per-drive block) mixed levels (`drive_pain`, `urgency`, `deviation_after`), a derived summary
   (`valence`) and need-gated changes (relief and harm under alliesthesia) under a key with no internal
   state, so the per-key mean would have tracked the body's recent state history, the defect §3.1's last
   row names. So:
   - **Target:** the per-sensor change of the invoked affordance's declared **body** sensors (the sufferer
     body), net of declared drift, on caused entries only: the record's `drive_delta` plus raw deltas for
     body sensors without a declared drive. To first order it is state-independent physics (touching the
     fire pit raises `arms.thermal` by 0.6 whatever the core temperature is); clamping at range edges is the
     one remaining state dependence, and S0b reports range-edge records apart. `valence`, `drive_pain`,
     `urgency` and `deviation_after` are **not** in the Cerebellum target.
   - **Add, not replace:** the existing object-readings model stays under its existing `models` key. The
     reafference means go under a new payload key.
   - **Alliesthesia at read time:** predicted relief = the drop in `drive_pressure` from now to now + Δ.
     Evaluation happens where the read happens, with the current state, so the Cerebellum keeps one meaning
     across states and worlds.
   - **The call moves into the executor.** The consequence exists only after `tool.run` returns (§3.1), so
     the observe and the `get_confidence` → `sim_cerebellum` line move out of
     `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` into `Executor._stamp_invocation` (right
     after `Executor._interoceptive_outcome` returns a record), which holds the Cerebellum already
     (`Executor._cerebellum`, threaded through `Executor.generate_entity_tools`). A second consequence
     computation in `tool_bridge` is never an option: it would be a second record of one event, without the
     caused/felt pain or the drift netting (G14). **Scope change:** `_outcome_start` returns `None` for
     `agent_id == ""`, so the Cerebellum stops training wherever no record is minted; today it trains on
     every SEM call whose tool was generated with one. S1 counts the affected executors (the ephemeral
     `OrchestratorActorTool` tool is built without `cerebellum=`, so it is not one). The move also **turns
     training on** where it is off today: tools generated without `cerebellum=` but run by an executor that
     holds `_cerebellum` and mints a record (the `--sim` generative cradle scene tools, the DM `SceneState`
     tools). That is intended, because training follows the record and not how the tool was built. S1
     counts these gains beside the losses (pre-merge review, Wire SF-3).
   - **Key:** unchanged. On every shipped cradle and Minecraft item the acted-on object owns the modulator,
     and the body's own modulators have no object (wiring lens S3), so "key on the target entity" is a no-op.
     No situation in the key: safe under this target (G22; Q6 decided).
   The single-cue RW rule, the confidence curve and persistence stay. **This is a write-side change only.**
   The read side stays Dormant, and `test_cerebellum_dormant_909.py` stays green, until S5 or E3 earns it
   through the E7 route. The bio-mapping says "reafference prediction without attenuation", since nothing
   cancels self-caused input with the prediction (Blakemore, Wolpert & Frith 1998; bio lens N4).
3. **The generalising model is refit offline at session end (S3).** `BioStack.on_session_end` (which calls
   `save_cerebellum` and `distributor.cleanup_session`) replays the session's `"loop"` traces that carry a
   pair. Together with the persisted training set, it refits `ConsequencePredictor`. This is **an offline
   batch refit over the replay buffer; a FUNCTIONAL analogy to complementary learning systems only, NOT CLS
   interleaved learning** (bio lens SF-6): a dual-form kernel ridge keeps every pair and re-solves exactly,
   so it is an exemplar store, with nothing slow, no interference to avoid and no consolidation. The
   Hippocampus stays the episodic store it replays. The Cerebellum is a separate, error-driven forward-model
   learner, online and key-specific. Training never runs on the hot path.
   **The order is named** (wiring lens S6): on the Minecraft harness `aut.bio.on_session_end()` runs after
   `run_agentic_loop`'s `end_bio_session` has flushed, stopped the capture worker, saved and run MemoryHub
   consolidation (`simulation/minecraft_harness.py::run_minecraft_aut`'s `finally`). So the refit reads traces after
   consolidation; it selects them by `run_id`, and counts the ones it cannot find beside S0b's queue-drop
   count. Whether consolidation prunes or merges `"loop"` traces or their `extra` is **UNVERIFIED** and is
   measured in S3.
   **The `--sim` AUT never reaches `BioStack.on_session_end` today** (wiring lens D2; corrected 2026-10-24,
   G25). Its `src/` caller is `simulation/minecraft_harness.py`; the survival-world and Exp 56 scripts call it
   too. On `maxim --sim`, `simulation/orchestrator.py::start_simulation_mode` calls `_orch_instance.shutdown()`
   and nothing else: that is the **narrator's** instance (`agent_id="sim_orchestrator"`,
   `load_persisted=False`), whose Cerebellum never trains. The AUT (`_aut_instance`, `agent_id="sim_aut"`, a
   per-run `tempfile.mkdtemp` home) is never shut down. Its session end is
   `runtime/agent_loop.py::run_agentic_loop` → `runtime/bio_integration.py::end_bio_session`, which flushes
   and saves the Hippocampus and ends the MemoryHub session and never touches the BioStack. So on `--sim`
   the AUT's Cerebellum is **never saved** and its `distributor.cleanup_session` never runs, and
   `simulation/orchestrator.py::_restore_aut_from_session` (`--resume-sim`) restores
   `aut_hippocampus`/`aut_nac`/`aut_ec`/`aut_atl` only, **neither the Cerebellum nor the predictor**. The
   earlier text here said the agent_factory shutdown saves it; that edit would refit the narrator's
   untrained stack and pass a caller grep (the D43 shape).
   **No `AgentInstance` path reaches it either** (pre-merge review, Wire SF-1).
   `runtime/agent_factory.py::AgentInstance.shutdown` runs `memory_hub.on_session_end()`, the Hippocampus
   and NAc saves and `bio_stack.save_cerebellum()`. It never calls `BioStack.on_session_end`, so
   `distributor.cleanup_session` never runs there. The only `src/` caller of `BioStack.on_session_end` is
   `simulation/minecraft_harness.py`. **Where the refit runs is therefore named per runtime:**
   - **Runs:** the Minecraft harness (`run_minecraft_aut`'s `finally`), the survival-world scripts and
     `scripts/exp56/common.py` (each calls `aut.bio.on_session_end()`), and the `--sim` AUT once #1449's
     slice lands (below).
   - **Does not run, and S3 does not claim it:** the `--sim` narrator, the console HANDLE
     (`console/handle.py` → `AgentInstance.shutdown`), `api.py` agents and `maxim.create.agent()` agents
     (`nac.save()` + `bio_stack.save_cerebellum()` only). A persistent agent's predictor is therefore not
     refit on those runtimes. Making `AgentInstance.shutdown` reach `on_session_end` is a separate change
     with its own reason and its own walk, not S3's.

   **The `--sim` AUT's seam is #1449's own Session-A slice, after the fence; S3 depends on it and cites it**
   (pre-merge review, Architecture S6). It is a defect independent of the predictor. Homed inside S3, it
   would land only if S0b and S2 pass, and a FAIL that stops the line would leave it unfixed. The slice:
   the orchestrator's AUT teardown calls `_aut_instance.bio_stack.on_session_end()` after the AUT thread
   joins and the Hippocampus has flushed. Its two branches are named (Wire SF-2):
   - **The adopted (HANDLE) branch** (`_aut_instance = _adopted.instance`): the side that owns the
     invocation owns its end (code-review probe 10), and under adoption that is the HANDLE. So the seam does
     **not** run on the adopted instance inside the sim. That avoids a session end inside the sim followed
     by the HANDLE's own shutdown, a double end. The HANDLE path then reaches no `on_session_end`, as
     above.
   - **The post-sim `/new` recursion** (`aut_memory_hub.on_session_end()`, then a recursive
     `start_simulation_mode`): each recursion level ends its own AUT instance exactly once, before it
     recurses, so no AUT is ended twice and none is skipped. **Under adoption the persistent agent is
     re-adopted, not rebuilt** (delta round, Wire N-1: `persistent_agent` is threaded through the
     `/new` return in `orchestrator.py`): every level then takes the adopted HANDLE branch above, so no
     level runs `on_session_end` on it, and the HANDLE's own shutdown is the only end.

   The slice's test drives both branches. The slice sits inside the fenced `start_simulation_mode` (pinned
   by the function-length baseline, so Rule T applies), so it lands after 1.3.6. It carries its own ledger
   walk: T1-13's anchor pruning (`cleanup_session` starts running on the AUT's distributor on `--sim`) and
   the `--sim` / `--resume-sim` rows. **S3's own additions stay in S3:** the refit at `on_session_end`, and
   `_restore_aut_from_session` plus the `aut_*.json` save gaining the Cerebellum and the predictor (§7,
   S3).
4. **The predictor (S2 offline, S3 in `src/`).** Kernel ridge in dual form, with a cosine kernel per block
   and fixed block weights, in **numpy only**:
   - It is closed-form and **deterministic**. Training pairs are ordered by `pid` before the solve, so
     neither the dict order nor `PYTHONHASHSEED` reaches the arithmetic.
   - It fits about 10³ pairs × about 1.5k dims as a single n×n solve. That is under a second on a laptop CPU
     (arithmetic, not measured). S2 measures solve time, peak memory and file size **on the named machines**
     (the owner's leader, Mac peer and rig) at n = 10³ and 10⁴; no Pi is on hand, so the Pi is
     **UNVERIFIED** and the "Pi profile" is a proxy (environment lens S8). The solve is O(n³), and the n²
     float64 kernel is about 800 MB at n = 10⁴.
   - **It is a store, and it holds the training set.** A dual-form model predicts from all n training
     contexts, so `<home>/consequence_predictor.json` *is* a second copy of the pairs (about 35–40 MB of
     JSON at 10³ pairs × about 1.5–2k dims, arithmetic; wiring lens S5, environment lens S8). It is written
     through `atomic_write_json` + `with_format_version` and carries `hash_scheme: "stable-sha256-v1"`, the
     per-block geometry tags, the target schema id, the training `pid` list and the pair count. No pickle and
     no `.pt`. It also takes the memory-store contracts (bio-memory brief): `StoreFileOwnership` with a
     `_reset_store_state` (never save over a file it did not read; #939, #971), `load_persisted` honoured
     (#972), a corrupt file copied aside, a **cap on n with an eviction rule** (or a Nyström/primal fallback
     declared as its own stage) and a file-size budget, both frozen in S3 and counted in S0b. The non-strict
     alternative, keeping only the dual coefficients plus `pid` pointers into the Hippocampus, holds only if
     trace eviction keeps `pid`-addressable pairs (**UNVERIFIED**).
   - It needs no torch and no scipy beyond what core already has. A fallback-geometry install's word block
     trains its own map and refuses a map trained under mpnet.
   - An MLP enters only if S2's pre-registered gate fails for ridge AND the failure is shown to be
     non-linear. It would be numpy with manual gradients, seeded from `utils/seeding.py::stable_hash_32`.
5. **The consequence representation: an opt-in `"consequence"` EC modality and a learner keyed on it (S4;
   owner decision G21, bio lens D1, confounding lens D1).** The first draft encoded each item's PREDICTED
   core into the EC and named two readouts, the node and the α blend. Neither can produce acquired
   equivalence. A fixed-kernel ridge moves a prediction in proportion to its kernel value to A's *inputs*, so
   retraining A moves C (sensed-like A) more than B; after the refit A leaves the node it shared with B; and
   nothing learns *on* a node (the T8 (b) toy, `docs/plans/reviews/grounding_gl4/lfm_toy.py`: neutral
   situation, Δf(B)[O3] = −0.005, Δf(C)[O3] = +0.297). The mediating path is therefore designed here, in
   three parts:
   1. **The representation is formed from experienced outcomes, and frozen.** Each item's node is formed
      from its **experienced** Stage-1 outcome records (never from predictions; bio lens SF-4: writing
      expectations into the store of experienced episodes is a reality-monitoring failure, Johnson & Raye
      1981), and its membership is **frozen per item at the end of Stage 1**. Predictions stay read-time
      only; a prediction-formed node, if ever wanted, is stamped `imagined` and excluded from every memory
      consumer, retro-tagging and the hivemind.
      **What the EC freezes, and what it does not** (pre-merge review, Architecture B1 = Executor S10).
      `ECConfig.frozen_centroid_modalities` freezes a node's **prototype** only: in
      `similarity/ec.py::EntorhinalCortex.pattern_complete_or_separate`, the first embedding to reach a node
      stays its centroid. The EC has no notion of an item. Every encode is matched by cosine, so a Stage-2
      encode of A's O3 outcome joins or creates a node by similarity whatever "frozen" says. The modality is
      listed there (under Q4's local-only form) so that prototypes stay put. **Membership needs two new,
      named pieces** (§5 front-gate, §8 touch list):
      - **(a) A per-item → consequence-node membership map.** Each item gets one node: the
        `"consequence"` node whose prototype is nearest (cosine) to the encoded **mean** of the item's
        experienced Stage-1 outcome vectors. Ties are broken deterministically, by node creation order and
        then node id, never by dict or hash order. An item whose Stage-1 outcomes land on several nodes
        (clamping, band variation) still gets one node by this rule, and the spread is reported. The map
        is new S4 state, persisted under the S4 contract (§7 S4).
      - **(b) A freeze trigger** that builds the map and fixes it. **Only the S5 harness calls it, at its
        declared Stage-1 boundary.** It is tiered as an **experimenter control** (apparatus-like; §4.3),
        not `learned`, and it has no production caller. With no freeze, the map is empty and the node-keyed
        learner books nothing. So S5's acquired-equivalence result holds for the harness configuration with
        its declared boundary, and it is stated that way. A production freeze trigger would be a GL6
        question, and T4a / Dormant applies to the map, the trigger and the learner if S5 has not earned by
        1.4.0 (§7 S3).
   2. **A fast outcome/value learner keyed on the consequence node.** It is updated when a member item is
      acted on in Stage 2, through the membership map. It books on the **outcome record's** provenance
      (`experienced`), not on a situation encode's. Candidates (S4's front-gate answer, §5; Q11): a **new**
      NAc table keyed on the node id, holding a signed per-dimension value, or a node → outcome running mean
      on the Cerebellum's delta rule. Two existing tables cannot carry it (§5). `cluster_reward_bias`, keyed
      by (cluster, tool), is tool-specific, so B cannot inherit A's entry (§3.4). `NAc._reward_bias`, keyed
      `(agent_id, node_id)`, is a non-negative scalar that only widens a recognition radius.
   3. **The readout is through membership.** B's Stage-2 value is read through its frozen node membership.
   This lifts §8's `credit_node` exclusion for that path; §8 and S5's blast radius carry it. Under §5's
   front-gate answer, the learner does not write through `credit_node` (its only table, `_reward_bias`,
   cannot carry the value), so the lift is held unused unless S4 finds otherwise. The
   `ConsequencePredictor` keeps the job it can do: zero-shot prediction for **untried** items (S5's
   untried-item arm, §7).

   **The encoded vector** (bio lens SF-1, SF-2; wiring lens S7; confounding lens S7). The modality encodes
   the change-only projection plus the per-drive **signed** `drive_delta` in the body's drive order, raw
   (unstandardised), with absent drives masked. That carries outcome identity as well as sign, so a cold-core
   relief and a hunger relief do not share a node, and value is applied at read time (G22), so one item
   gets one node whatever its need state. Every dimension is declared on a **centred range [−1, 1]**, so
   "nothing happens" is the zero vector. On native [0, 1] ranges it is not: `_sensor_embed` sets the A4
   neutral point at the range midpoint, so 0 is an extreme at full weight, and the bio lens measured
   cos(nothing, harm 0.2) = 0.959 and cos(nothing, relief 0.3) = 0.902, one node at 0.85. The **A4 gain is
   documented as wrong at N ≤ 8** (stability 0.62 at N = 6 against A1's 0.88,
   `docs/limits/l11_sensor_dilution.md`), and this vector has 3 to about 8 dimensions, so S4 uses the
   ungained embed unless a measurement justifies A4 at this N. The 0.85 threshold was calibrated on the
   world modality, so the consequence modality gets its own pinned, measured value with a follow-up trigger
   (behaviour-tiers rule). `_sensor_embed` *rotates* direction with value by design, so "a mild and a severe
   burn share a direction" is not given: S4's gate is an offline cosine replay of S0b's vectors through the
   real encoder (`docs/wiring/cosine-separation-is-directional.md` corollary 3).

   **The encode door is new and EC-only** (wiring lens S7). `similarity/encoder.py::SensorEncoder.encode_sensors`
   calls `ec.pattern_complete_or_separate`, then `atl.activate_substrate_node`, then `nac.update_eligibility`,
   so reusing it would make every consequence node an ATL activation and a creditable NAc eligibility key.
   S4 names a new function, `_sensor_embed` → `pattern_complete_or_separate`, with no ATL activation and no
   eligibility; the node-keyed learner (part 2) is the only writer of value on a consequence node. It
   passes no `threshold_override`, so no reward bias widens a consequence node's radius. It calls
   `pattern_complete_or_separate` outside `perception/`, so it needs a named entry, with its reason, on
   GL3.B1's M39 lint allowlist (`scripts/lint_relay_only_encode.py`; pre-merge review, Executor N2).

   The modality has a frozen centroid and its own geometry tag.
   EC matrices are already per modality (`similarity/ec.py::EntorhinalCortex._matrix_for`), so the
   interoception, audio, world and text scans stay **byte-identical**. The hivemind does not filter by
   modality: `hivemind/bundle.py` exports every `substrate_nodes` entry (filtered by domain only), and
   `hivemind/merge.py` folds a modality missing from `SENSOR_MODALITY_THRESHOLDS` at the default cosine
   threshold. Keeping consequence nodes local is therefore a bundle edit (S4). Rejected alternative: a projection flag
   on `pattern_complete_or_separate`, which edits the shared encode path that every earned row runs through.
   Cosine sees direction, not magnitude (`docs/wiring/cosine-separation-is-directional.md`), so
   **magnitude is read from the code, never from node identity**, and S5's O1, O2 and O3 differ in
   direction (confounding lens S7). An existing never-exported path,
   `hivemind/bundle.py::_filter_ec_nodes_by_domain` (it drops `domain == IDENTITY_DOMAIN_MARKER` nodes
   unconditionally), would keep consequence nodes local without a `bundle.py` edit; it overloads the
   "identity" meaning, so it is offered under Q4 as an option, not a recommendation, and the receiver-side
   refusal is owed either way (wiring lens N5).
6. **Concept similarity tiers from word to consequence (S4): a reported read, never a gated readout**
   (G21, confounding lens D1 item 3). A consumer asking "are *x* and *y* the same kind of thing to act on?"
   reads `sim(x, y) = α·cos_identity(x, y) + (1 − α)·cos(c(x), c(y))`, where `c` is the item's consequence
   code (its experienced code once tried, its prediction while untried).
   - **α is precision-weighted** (bio lens SF-5): it is set from the ridge's predictive variance,
     `k(x,x) − k_xᵀ(K+λI)⁻¹k_x` (dual-form kernel ridge is the posterior mean of a Gaussian process), combined
     with the consequence residual variance near *x*, as a pinned innate-prior formula, floored at 0.05 so the
     prior is never fully switched off. A count curve (`1 − conf(n_eff)`, the first draft) would drive α down
     as fast for inconsistent consequences as for consistent ones, which is not precision weighting.
   - Before any experience α ≈ 1 and the word prior rules. α falls where nearby experience is consistent,
     and stays high for contexts unlike anything experienced.
   - **It is not a gated readout in S5.** Extra Stage-2 trials on A raise experience near C (sensed-like A)
     more than near B, so α(A,C) falls further than α(A,B), and with neutral names sim(A,B) > sim(A,C)
     appears purely because B kept more of the identity prior: "B inherits more" from experience density,
     with no consequence equivalence. S5 reports α per pair beside equalised experience, and gates on the
     node-keyed learner (item 5).
   - This is a read-time blend. Nothing overwrites a word node, and T1-5's decomposer is not edited.
7. **Shadow consumer first (S3).** For every candidate tool at decision time, the predicted core, its
   consequence cluster and α are logged. The log has **no selection effect**. A selection effect enters only
   as S5's declared experimental arm (§8). "At decision time" is `runtime/substrate_proposal.py::propose_via_substrate`
   / the loop's choose step, which is fenced: the shadow log's site lands with the 1.3.6 `agent_loop` phase-2
   slices (wiring lens S6), written to the existing JSONL trace surface only.

### 4.3 Behaviour tiers (every automatic behaviour declares one)

| Behaviour | Tier |
|---|---|
| A pair must join on a minted `PhysicalEventId`; a `narrated` pair is never relabelled `experienced` and always carries the declared discount; an `apparatus` outcome builds no pair; a geometry mismatch refuses; a training set never spans a target schema id | **invariant** (`[engineering]`, enforced in the type and the loader) |
| The narrated discount's value | **innate prior**: an owner-set constant (GL4 start), pinned before S2's data |
| Word-embedding (identity-block) similarity before experience, α = 1 | **innate prior** tier (biologically an inherited cultural prior, §1). Follow-up trigger: S5's result |
| Block weights, kernel, the consequence modality's own threshold (measured, then pinned; bio lens SF-1), the α formula (predictive variance, floor 0.05), the target standardisation procedure (scale, no centring), the change-only target projection, the centred [−1, 1] ranges, the core valence formula (unweighted, G11), the 0.44 identity threshold (bio lens SF-8) | **innate prior**: pinned constants, fixed before S0b's data is read (confounding lens N4), with a sensitivity row beside the S5 gate. Hard-coded priors get a follow-up issue with a trigger (behaviour-tiers rule) |
| Alliesthesia at read time (predicted relief = the drop in `drive_pressure` from now to now + Δ; G22) | **innate prior** (G11's formula, applied where the read happens) |
| Which entity was acted on (`ActionContext.entity_path`, from the YAML), caused vs felt pain (from the affordance's declaration, not an efference-copy comparison), the provenance labels | **engineering prior (oracle attribution)**, as GL1 SF-6 re-tiered cause attribution; never cited as learned reafference identification or source monitoring; each with a follow-up trigger (bio lens SF-8) |
| The Cerebellum's per-key reafference means (after S1) and its object-readings model (kept) | **learned** (online, key-specific) |
| Consequence-node formation from experienced outcomes, and the node-keyed outcome learner's values (S4; G21) | **learned** (fast, keyed on the shared representation) |
| The membership map's assignment rule (the node nearest the item's mean experienced Stage-1 outcome, with a deterministic tie-break) | **innate prior**: a pinned rule; the map's contents are computed from experience |
| The freeze trigger (when membership is fixed) | **experimenter control** (apparatus-like, a declared harness control; only the S5 harness calls it, with no production caller; pre-merge review, Architecture B1). Never cited as learned |
| The matched-class floor learners (S4; the same learner, map rule, freeze and Stage-2 booking, keyed on identity-similarity text nodes and sensed-reading object-block nodes; §6) | **experimenter control** (the S5 gate's floors; only the S5 harness composes them, with no production caller; delta round, Architecture N2). Their values are computed from experience, but they are never cited as the agent's learned behaviour |
| Predicted consequence for untried items, α's fall with consistent experience | **learned** (offline batch refit; "slow" here means offline, not CLS-slow, bio lens SF-6) |
| Imagined-entity pairs | Owner decision at S3 start. The recommendation is to exclude them (the strict option) |

### 4.4 Invariants this plan introduces (all `[engineering]`; none behavioural until S5 earns it)

| Invariant | Regression guard |
|---|---|
| A training pair cannot be constructed without a minted `PhysicalEventId`, an explicit provenance and matching geometry tags; a `narrated` pair cannot be relabelled `experienced` and always carries the declared discount; an `apparatus` outcome cannot be paired (G16) | outstanding.md M41 (backlog row until S3 lands; structural thereafter: the pair type's constructor + `tests/unit/test_consequence_no_contamination.py`); the test lands in S3 with the first training set |
| Training and the persisted predictor are byte-identical across processes with differing `PYTHONHASHSEED` | `tests/unit/test_consequence_predictor_two_process.py` (S3), on the `test_stable_hash_two_process.py` pattern |
| With the consequence modality off, every existing EC modality dump is byte-identical | `tests/unit/test_ec_consequence_modality_golden.py` (S4) |
| The Cerebellum learns the body's signed reafference through the production call path (the executor), and keeps its object model | `tests/unit/test_cerebellum_wiring.py`, re-pointed in S1 to drive `build_executor` → `Executor.execute`, with its AST call-site pin moved to `Executor._stamp_invocation` (G22) |
| A Cerebellum file newer than the code survives a session end byte-identical | a refusal test in S1 (the typed error sits outside `UNREADABLE_STORE_ERRORS`; wiring lens S4) |
| A null consequence encodes to the zero vector; relief and harm are near-orthogonal; no `"consequence"` node id appears in NAc eligibility, `reward_bias` or ATL activations | S4 guards, each proven by deletion (bio lens SF-1, wiring lens S7) |
| No `"consequence"` node crosses a hivemind bundle, and neither does any NAc field keyed on a consequence node id | `tests/unit/test_hivemind_excludes_consequence.py` (S4), extended to `scrub_nac_state_for_bundle`'s output (pre-merge review, Wire SF-8) |
| Membership is read only through the frozen per-item map; with the map deleted, B's readout does not move | the membership map's deletion probe (S4; pre-merge review, Executor S10) |

---

## 5. Front-gate scope pressure

| Existing infrastructure | Can it carry this? | Used here as |
|---|---|---|
| `Cerebellum` (per-key RW forward model, persisted) | **Partly.** It is the right home for the ONLINE, key-specific model of the body's reafference. It cannot generalise: an exact-key lookup with no context embedding and no situation in the key | **Rides on it**: a reafference target added beside the object model (G22), the observe moved into the executor; key unchanged |
| `anticipatory_pre_activate` (timed predictor) | **No.** It predicts WHEN an event recurs, not what an action does to the body; it has no action or object input | untouched; audited in S0a |
| `Executor._stamp_invocation` + `ToolOutput` | Yes, for the record: it is already the single per-invocation writer | **Rides on it**: pair assembly |
| Hippocampus `"loop"` traces + `EncodingSignals.extra` | Yes, as the replay buffer (persisted, run provenance, situation cue) | **Rides on it**: no new store |
| `BioStack.on_session_end` | Yes, as the refit seam, where it is reached: the Minecraft harness and survival scripts today, the `--sim` **AUT** after #1449's slice (§4.2 item 3; fenced). **Not** on any `AgentInstance.shutdown` path (HANDLE, `api.py`, `create.agent()`), which reaches `save_cerebellum()` only (Wire SF-1) | **Rides on it**: offline refit (S3), after #1449's slice |
| EC per-modality matrices, `_sensor_embed`, geometry tags, the frozen-centroid mechanism | Yes, for node formation over experienced consequence codes, and for frozen **prototypes**. **No, for per-item membership**: `frozen_centroid_modalities` pins a node's centroid, the EC has no notion of an item, and every encode is matched by cosine (pre-merge review, Architecture B1) | **Rides on it**: a new modality and a new EC-only encode function (not `SensorEncoder.encode_sensors`, which activates ATL and NAc eligibility), not a new similarity engine |
| A per-item → consequence-node membership map | **Does not exist** (row above) | **New, small** (S4): one node per item, the node whose prototype is nearest the encoded mean of the item's experienced Stage-1 outcomes, ties broken by node creation order then node id (§4.2 item 5 part 1 (a)) |
| A freeze trigger that fixes membership at a declared boundary | **Does not exist.** The stage boundary exists only in the S5 harness | **New, small** (S4): called only by the S5 harness; tiered experimenter control, no production caller (§4.3); T4a applies |
| `CausalLink` / `cluster_reward_bias` / `cluster_fear` | **No.** Scalars per seen key; no function from a context embedding to a vector consequence. And no key is a consequence: `cluster_reward_bias` is keyed by (cluster, tool), so value learned on A's tool stays on A's entry and B cannot inherit it (bio lens D1) | untouched |
| `NAc._reward_bias` (keyed `(agent_id, node_id)`, written by `NAc.credit_node`) | **No, even though it is keyed on a node** (pre-merge review, Architecture B1; read at `d9dffe6e`). (1) It is a **scalar** per node, and the AE gate reads a **per-dimension signed** Δ (§6). (2) `credit_node` clamps it to `[0, max_reward_bias]` (0.20; "bias only widens, never inverts"), so an aversive O3 cannot be stored, and a bias clamped to 0 is removed. §6 requires O1/O2/O3 to differ in sign pattern, so no stated restriction rescues it. (3) Its meaning is recognition: `get_threshold_overrides` turns it into a per-node `threshold_override` that widens the node's EC radius, so a value written there would also move membership. (4) It is ablated by `MAXIM_NAC_REWARD_BIAS_DISABLED`, decayed by `decay_reward_biases`, and shipped in hivemind bundles with NAc state | untouched; listed under Q11 as a rejected option |
| An outcome/value learner keyed on a consequence node | **Does not exist** as a table that can hold a signed per-dimension value (the two rows above). It is the "fast associative learner on the shared representation" acquired equivalence needs (§1) | **New, small** (S4; G21): a **new** NAc table keyed on the node id (signed, per dimension, not `_reward_bias`), or a node → outcome running mean on the Cerebellum's delta rule; S4's front-gate answer picks one (Q11) and names why the other cannot carry it |
| LinguisticEncoder name nodes (word world) | **No.** Its similarity is fixed by mpnet, and §3.7 shows it orders flame/water the wrong way | becomes the prior tier |

**Why existing infrastructure cannot do this.** Every surface above that stores a consequence stores it
*per key it has seen*. The only generalisation in the codebase is EC similarity of a SITUATION. An
(object, action) pair is not a situation, and the one model that is keyed on (object, action), the
Cerebellum, matches keys exactly. What has to be new is one function approximator, `ConsequencePredictor`,
that maps a context embedding to a consequence vector and returns a value for an (entity, affordance,
situation) it has never executed. That is its job: **untried** items (G21). The second new piece is the
node-keyed outcome learner (row above): the tables that store a consequence key it per tool, so none can
carry value from one item to another that shares its outcome, which is what acquired equivalence needs on
**tried** items. Two more small pieces are new because the EC cannot carry them: the per-item membership
map and its freeze trigger (the EC freezes prototypes, not membership; the trigger is an experimenter
control with no production caller). The other new pieces are assembly or registration only: the
`ActionContext` builder, the `"consequence"` modality registration and its EC-only encode function, and
the train/eval scripts.

**The mechanism twin is the mechanism on tried items** (G21, confounding lens D2). A per-item twin (the S1
Cerebellum means + S4 clustering + the same Stage-2 booking, no predictor) gives A, B and C the same
consequence representation as the full arm does once every item is experienced. So credit is split: the
acquired-equivalence gate credits the **consequence representation** (frozen nodes + node-keyed learner),
and the untried-item arm credits the **`ConsequencePredictor`**, which must beat that twin there (§6, §7 S5).

**Input-shape check first (the Roy-4 lesson).** S0b must show that executed pairs exist in volume AND that
name and consequence *disagree* somewhere in them, counted over non-null consequences only (§7, S0b). If
every pair of affordances that share a name also share a consequence, the predictor cannot be told apart
from the word prior, and the line stops at S0. A second shape check follows from G13: the predictor must
also be told apart from **sensed similarity**, so the claim needs items whose sensed readings and
consequences disagree (§6), and these are authored for S5, not found in the shipped YAML. The shipped YAML
has one hidden-property pair, and its sensors are byte-identical (`cradle_false_hearth.yaml` is modelled
on `cradle_fire_pit.yaml`, `heat_output` 0.9 and `fuel` 0.8 in both, with `warm_self` safe on the fire pit
and harmful on the false hearth), and 24 of the 34 consequence-bearing cradle keys sit on twelve items with
no sensors at all (confounding lens D3, environment lens D2). So the shipped data cannot test G13, and S2
is a capability stage (§7, S2).

---

## 6. The falsifiable signature: acquired equivalence through a designed mediating path, gated against matched-class floors that could win

**Redesigned 2026-10-09 by owner decision G13** (confounding lens DNB-2, bio lens SF-8); **the mediating
path, the floors and the gate statistic amended 2026-10-24 by owner decision G21** (T8 (b): bio lens D1,
confounding lens D1 and D2). An earlier draft
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
   the shared consequence. Each item is acted on in Stage 1, an **equal number of times**; an item never
   acted on is not part of the test. At the end of Stage 1, each item's consequence-node membership is
   **frozen** (§4.2 item 5).
2. **Stage 2.** A alone is retrained to a new consequence O3. The node-keyed learner books it on A's frozen
   node, which B shares and C does not.
3. **Test.** B (consequence-equivalent, sensed-unlike) must inherit O3 more than C (sensed-like,
   consequence-unlike) does. **The readout is named here, not left to the prereg** (G21): B's and C's value
   read through their frozen node membership from the node-keyed learner (§4.2 item 5 part 3). The
   first draft's readouts cannot produce the ordering, by algebra, not by experiment: a fixed-kernel ridge
   moves C's prediction toward O3 more than B's, the refit moves A off B's node, and the α blend favours B
   only through experience density (§4.2 items 5–6; the toy at
   [reviews/grounding_gl4/lfm_toy.py](reviews/grounding_gl4/lfm_toy.py)). A null ships as a null.

**The gate statistic is matched to the null floor** (G21, confounding lens D2). Per dimension,
Δ_B − Δ_C (each from Stage-1 end to Stage-2 end) must exceed a pre-registered margin **with its interval
excluding 0**, against the no-revaluation control (§7, S5).
**Matched-class floors that could win** (pre-merge review, Architecture S1 and Executor N1; the strict
reading of G21; **confirmed by the owner as binding floors of the T9/AE gate now, DECISIONS.md G30,
2026-10-24: only their margins are set at S5 start**). The arm's Δ_B − Δ_C must also exceed, by the margin and with its interval excluding 0,
the Δ_B − Δ_C of **the same node-keyed learner keyed on other nodes**, with the same freeze, the same
membership rule and the same Stage-2 booking:
- **identity-similarity nodes**: EC `"text"` nodes of `"<affordance> <entity name>"` at the production
  0.44;
- **sensed-reading nodes**: object-block nodes under `_sensor_embed`, at the threshold the consequence
  modality pins.

**The membership rule's analogue for the floors (delta round, Executor N-4):** each item's floor node is
the node nearest that item's identity-block encode (identity floor) or object-block encode (sensed
floor), from its Stage-1 encodes, with the same tie-break and the same freeze at Stage-1 end as the
consequence rule (§4.2 item 5). Their node formation needs an M39 route: they reuse the registered
receptors' encode route (the `imagination.affordance` / own-body affordance receptors for the text
node, the object-block receptor for the sensed node) rather than adding allowlist entries; if a route
is missing at S4, the S4 PR adds a named allowlist entry with its reason.

Either can win if names or readings leak into B (a shared text node, or B's readings near A's), and by the
item rules (cos_id(B,A) ≤ cos_id(C,A), cos_obj(C,A) > cos_obj(B,A)) both are expected to predict C ≥ B.
So beating them is evidence about the **key**, and the learner class is held fixed.
**The BF2 floors stay as declared, and they are labelled as floors that cannot win on this readout**
([baseline_floors.md](baseline_floors.md)). On a learner-state readout, `tabular_q` on (EC cluster × tool)
predicts Δ_B = Δ_C = 0 by construction. B and C are not acted on in Stage 2, so their Q-values do not move,
and the tools are distinct. `random` has no learner state, and its Δ is defined as 0 in the prereg. So
"beat `tabular_q` and `random`" is passed by the margin alone. Both are run and reported for BF2
compliance, and neither is evidence by itself. The prereg writes every floor's learner-state readout
explicitly, so each comparison is computable.
**What a PASS can and cannot mean.** Given shared nodes, the gate's only failure routes are node
formation (A and B not landing on one node, or C landing on it) and the learner's wiring. A PASS says
that experienced outcomes grouped A with B and that value booked on A reached B through the group. It says
nothing more: nothing about choice (G27), and nothing about the predictor.
**The no-learning sensed-similarity baseline is REPORTED, not the gate**: it spreads the trained consequence
by object-block cosine alone and predicts C > B, so a learner that transfers nothing (B = C) "beats" it by
the baseline's own error. Beating a signed-wrong baseline is cheap; that was the strawman. Beating a floor
that is zero by construction is just as cheap, which is why the matched-class floors gate. This gate
credits **only the consequence representation** (frozen nodes + node-keyed learner), not the
`ConsequencePredictor`, because on tried items the per-item twin *is* that mechanism (§5). The predictor is
credited by a separate **untried-item arm** (§7, S5), where it must beat the per-item twin. A floor match on
any gated arm (a matched-class floor on the AE or AD arm, the twin on the untried-item arm) takes
**`PARTIAL (narrow: floor matched)`**, stated in the prereg in advance (BF2).

**The acquired-distinctiveness (AD) arm is gated too** (pre-merge review, Architecture S3; strict). G27 has
it carry T9's "blocks it where names collide" half, so it gets a gate, not a report. Items D1 and D2 have
the same affordance name, identity blocks on one text node at 0.44, equal sensed readings and differing
consequences. Stage 1 gives equal experience of both, and membership freezes at its end. Stage 2 retrains
D1 to a new outcome O3. **Readout:** D2's change in learner value from Stage-1 end to Stage-2 end (Δ_D2),
per dimension, read from the node-keyed learner through D2's frozen membership, net of the no-revaluation
control. **Statistic:** the identity-keyed floor's Δ_D2 minus the consequence learner's Δ_D2 exceeds the
margin, interval excluding 0, and likewise for the sensed-keyed floor. Both floors transfer by
construction here, because D1 and D2 share a text node and read alike, so they are the floors that could
win. In addition, the consequence learner's Δ_D2 lies within a pre-registered equivalence bound of the
no-revaluation control. **Floors:** the two matched-class floors (gated, as above). The per-item twin and
`tabular_q` (zero transfer by construction) and `random` are reported. **PARTIAL rule, stated in
advance:** no separation from a matched-class floor gives `PARTIAL (narrow: floor matched)`. Separation
from the floors but a Δ_D2 outside the equivalence bound gives `PARTIAL (narrow: partial transfer)`. A
PASS says only that the consequence nodes separated D1 from D2 where the names collide, measured as learner
state. On this path that is the S4 separation check, read on the authored items.

**A composed-loop toy comes before S5's prereg** (G21, confounding lens D1; the E1/B4 rule that toys must
be the composed loop). It runs the S3 refit, the S4 node, the membership read, the S5 seam and every arm on
a scripted A/B/C world, and must show that the mechanism predicts B > C and that every floor does not.
That includes the matched-class floors, which can win, and not only the BF2 floors, which are zero by
construction. It also runs the AD items (D2 not inheriting where the floors do).
Commit target: `docs/plans/rationale/latent-forward-model/s5_toy.md` (to be built). The confounding lens's
toy, which shows the first draft's readouts fixed by algebra and schedule, is committed at
`docs/plans/reviews/grounding_gl4/lfm_toy.py`.

**The artifact routes are removed** (G21, confounding lens D1 item 4). In the toy, a blocked Stage-1
schedule with habitual-state probes flips the ordering through the `situation` block (block weights
(1, 1, 3), λ = 1: Δf(B) = +0.198, Δf(C) = +0.102), which is co-occurrence binding by the trial schedule. So:
the body is reset to a frozen pre-state band before every Stage-1 and Stage-2 trial, and B and C are probed
in the same state; trial order is interleaved and counterbalanced; Stage-1 counts are equal per item; the
session and refit boundaries are pinned identically across arms; α is not a gated readout (§4.2 item 6:
reported per pair with n_eff equalised); a situation-block ablation is reported beside the gate. The reset
is a harness write, stamped `apparatus` (G16), and **the reset forms no training pair**. The trial that
follows is the AUT's own experienced act, taken from a harness-set pre-state (G26, narrowed by the pre-merge
review, Executor S11). This is a **forced schedule**, which the `--sim` runtime cannot produce. §7 S5 names
the driver and says how GL3's last-writer slot stamps the reset.

**The item set** (G21, confounding lens D2; environment lens D2):
- **One shared sensor schema** across the AE items (the same sensor names, values differ), so object-block
  cosine measures readings, not schema (`_sensor_embed` sums SHA bases per sensor **name**, so items with
  disjoint names are near-orthogonal whatever their values). **No item has an empty object block.**
- **The prereg tabulates cos_obj(A,B), cos_obj(A,C) and cos_id for every triple.**
- **"Reads alike" means shared sensor names with near values under `_sensor_embed`**, and each item's
  readings reach the AUT through the receptor path of §4.1 before its first Stage-1 contact.
- **Names are numerically neutral** (confounding lens S8): per triple cos_id(B,A) ≤ cos_id(C,A) (the word
  prior favours C, the conservative direction), no A/B shared text node at 0.44 under the production
  encoder with `MAXIM_SUBSTRATE_PATH=1`, and the keyword test (§7, S5 option (a)) run over the **full tool
  name** `<entity>_<affordance>`, which rules out `warm_self` as an AE affordance.
- **O1, O2 and O3 differ in direction** (sign pattern across dimensions), none is null, O3 sits on a
  dimension where B and C start at the same level, and the prereg records cos(O1,O3), cos(O1,O2),
  cos(O2,O3) (confounding lens S7, S6).
- **O3 is authored with the item set, blind**, before the design is known; A's sensors and name are
  unchanged across stages (confounding lens N3, bio lens SF-3).

Pairs must include both kinds of disagreement: items whose readings match but whose consequences differ (a
hidden property, such as an insulated object that reads hot and does not burn), and items whose readings
differ but whose consequences match. T9 keeps its claim ("similar by consequence") under this design (G13);
without it the claim would have to shrink to "sensed properties override the name prior". As worded by G27,
T9 claims that the consequence representation carries learned value from A to B, items that share
consequences where names differ, **measured as learner state, with no claim about choice**. The
"blocks it between affordances whose names collide" half is tested by the acquired-distinctiveness arm
(§7, S5; confounding lens S1; kept by G27).

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
Read per target dimension, none of these is an "opposite non-null sign" pair (the azimuth flip is on a
sensor, not a target dimension), so the collision definition is frozen in the plan, in S0b, before any
count is seen (confounding lens S2).

---

## 7. Stages

The stage IDs S0–S5 are this plan's (their place in the umbrella's map is stated once, in the header).
GL6 (a selection consumer that is not an experimental arm) is the umbrella's and is not planned here. Each stage lists its
gate, its guards, its blast radius and the owner decisions to ask together at its start
(development-flow rule 3).

| Stage | Kind | Depends on | Fence |
|---|---|---|---|
| S0a | paper + offline read of persisted JSON | none | can run now (not started; Session E; T8 (b) done 2026-10-24) |
| S0b | prereg + script + N short fresh-body captures (G23) | GL1 census, recomputed on the identity-block strings (prediction of the collision and convergence counts); M1 stamps plus the encoder stamp; this plan's own four-lens review (done, §12); the named capture runtime (below) | the prereg and script are paper and run now; the captures are fresh sim runs; S1's context fields need S1, and pid-bearing records need the post-fence resume stage (G17); the `nociception` dimension waits for the caused/felt split (§4.1) |
| S1 | `src/`, record only | GL2a (merged) **and** the post-fence resume stage that builds `PhysicalEventId`, the sequencer and the cross-session resume (G15, G17) | after the fence: the resume stage lands after it, so S1 does too, and G1's record-only exemption is moot (header) |
| S2 | `scripts/`, offline | S0b PASS; for gating `nociception`, the caused/felt split (wiring lens S2) | runs as soon as S0b's data exists |
| #1449 | `src/`, defect fix (Session A slice, not a GL4 stage) | the 1.3.6 `start_simulation_mode` slices (G25, wiring lens D2) | after the fence; independent of S0b/S2 (pre-merge review, Architecture S6) |
| S3 | `src/` | S1, S2 PASS; **#1449's slice** (the `--sim` AUT reaching `BioStack.on_session_end`), which S3 cites and does not contain | fence |
| S4 | `src/`, opt-in | S3 | fence |
| S5 | experiment (T9) | S4; **GL3.B1** (the last-writer map, its kind-taking setter and the apparatus window, TR22: S5's reset stamping needs them; delta round, Architecture SF-3; strict: S5 waits for it); R4's routing audit has decided the selection surface; the authored acquired-equivalence items (§6); **the composed-loop toy** (`docs/plans/rationale/latent-forward-model/s5_toy.md`, G21) before the prereg | four-lens design review before any harness |

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
  Minecraft readout in S5, tested there against its own floor (the B1 mechanism twin: TD state value on the
  cluster + the same prior), never inferred from S0b's data-existence contrast (confounding lens S9).
- **(iii) The timed predictor carries the *when* and the Cerebellum the *what*.** The verdict says which
  consumer composes them and why that composition is not a new mechanism.
- **For whichever branch it names, the bio mapping it implies** (bio lens SF-7): MECHANISM, FUNCTIONAL or
  none, with a check that no bio-named class gets a job outside its mapping. The cerebellum predicts
  short-latency sensory consequences and sub-second-to-about-a-second timed aversive responses
  (eyeblink, interpositus); interval timing over the seconds to tens of seconds that Minecraft drowning
  spans is cortico-striatal (Buhusi & Meck 2005), and long-interval aversive anticipation is hippocampal;
  `anticipatory_pre_activate` is oscillator- and SCN-bin based, and circadian machinery does not time
  seconds-scale threat. A seconds-scale graded predictor is named functionally, not "cerebellar" or "SCN".
  If branch (i) reads the Cerebellum's error, `ForwardModel.update` initialises at 0 (no consequence) so
  trial 1 carries the largest error (bio lens N3).

Expected from the code read (to be confirmed, not assumed): the Cerebellum as shipped cannot carry it,
because its key has no situation and its target is the owner's absolute readings. After S1 it predicts the
body's reafference Δ, which is state-independent to first order; the graded "how far" read would apply
alliesthesia at read time (G22). `anticipatory_pre_activate`
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
(read-only) then runs over fresh captures through the real `Executor`, carrying M1 provenance stamps plus
the encoder's id, revision and dimension (a run on the 384-d hash fallback is refused; environment lens S5),
and run with `--interactive false`:
- **(a) Cradle: N short fresh-body sessions, shipped bodies only** (owner decision G23, environment lens D1).
  The first draft's "one campaign" cannot reach band-clean floors. On `bodies/infant_humanoid.yaml`,
  `arms.thermal` recovers at 0.0008/s toward set point 0 with a 0.5 comfort band
  (`embodiment/sem.py::drift_step`); one harm contact (+0.6) takes 125 s to re-enter the band and 750 s to
  reach rest, and eight safe warms (+0.05 each) from 0.6 clamp the arm at 1.0, while drift removes about
  0.0004 per 0.5 s loop tick. Substrate selection then concentrates: in
  `docs/experiments/data/42d53_results.jsonl` `safe_pref` is 0.996 and `n_exploit` = `n_contact` − 3, so
  within about ten contacts nearly every record warms an arm at the range edge, which the sign rule excludes.
  `NAc.recommend_action`'s explore-first gate gives each untried scored tool one trial, in (score, name)
  order, so order cannot be counterbalanced inside a session. So:
  1. **N short sessions, each on a fresh body** (a fresh `--sim` subprocess, as
     `scripts/benchmark_exp42_preference.py` does), with the scene's entity order and the item names'
     alphabetical order **permuted across sessions**: the only counterbalance the explore-first tie-break
     allows.
  2. **The prereg computes N before any capture**: from the YAML deltas and `drift_step`, the expected
     number of non-edge records per key and per pre-state band per session, by the same independent
     arithmetic as the known-answer identity check. The floors are frozen in those per-key terms. **If the
     arithmetic gives too few, the prereg restates the floors**; it never adds a capture-only body, and
     never uses apparatus resets as training pre-states (G23).
  3. The exploration weight is set through `maxim config set sim.substrate_explore_bonus_weight` (default
     0.0, so explore-first is off without it), and `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` is frozen; unset,
     actions per turn = narrator wall-clock ÷ 0.5 s, which varies by box (`simulation/bridge.py`).
  4. **The body is named, then the prereg tabulates which keys are non-null on it** (Q9). On
     `infant_humanoid_chilled`: cold relief and thermal harm, and `food_source.eat` null at the edge (hunger
     stays below 0.5 by design). On `infant_humanoid`: no `cold` sensor, so the 24 glow-item keys reduce to
     null (+0.05 inside the band) or harm (+0.6). No shipped body makes all 34 keys consequence-bearing.

  The scene is still a **purpose-built all-items scene** (environment lens SF-9). All cradle-family items
  together declare 34 consequence-bearing (entity, affordance) keys (blanket, cool_air, false_hearth,
  fire_pit, food, sharp_rock, the green/purple hearth/flame items and their `_b` twins, warmth_alpha/beta
  safe/harm); the arc cradle scene (`simulation/arcs.py`: fire_pit, food, cool_air) has 5. That brings the
  tool-name collision `infant_humanoid_chilled.yaml` warns about (the `_b` twins exist for it), and a
  preflight refuses a scene whose entity names collide (environment lens S6). **Only about 10 of the 34
  keys carry an object block**: the twelve glow items declare `sensors: []` (environment lens D2), so S0b
  reports per key whether a block exists and how often it was sensed, and keys with a block are counted
  apart (gate below). Any new items (sensors on new glow items, the triad, the acquired-equivalence items)
  are new YAMLs authored blind, never edits to shipped items, which Exp 42/44 cite; once authored they are
  counted among the keys. The `warmth_*_{harm,safe}` items put the answer in their names and are flagged or
  excluded (environment lens S5); the purple/green families are the neutral-name ones.
- **(b) Minecraft.** A water-classroom campaign on the rig, through the production loop
  (`simulation/minecraft_harness.py::run_minecraft_aut` → `run_agentic_loop`).
- **Reachy is out of scope** (G4), not "not run": `bodies/reachy_mini_infant.yaml` sets the azimuth drive to
  `null`, its hunger and thirst are relieved only by mother feeds (apparatus, G16), and on the live robot
  azimuth is live-owned, so `tool_bridge` withholds the modeled self_effect (environment lens N4).

**The capture runtime is named per arm, and only invocations that reach `capture_loop_action` count**
(wiring lens S10). The cradle runtime is **`maxim --sim --arc <yaml> --aut-mode substrate-primary`**, the
Exp 42 recipe (environment lens S1, which resolves the earlier UNVERIFIED). The fixture-orchestrator
candidate is struck: `simulation/fixture_orchestrator.py::FixtureDrivenOrchestrator.run` drives only the
fixture's text and sensor percepts, `simulation/campaign_runner.py::run_fixture_campaign` threads no
`embodiment`/`entity_map`, the `scenarios/substrate/*.yaml` fixtures carry no cradle entities, and scene items
are activated only by `simulation/generative_runner.py::_activate_phase_entities` inside
`run_generative_campaign`, whose `Narrator` is an LLM. On the named runtime the AUT's `agent_id` is
`"sim_aut"`, so pids can be minted, and autonomous dispatch reaches `tool_dispatch.execute_and_learn` →
`capture_loop_action` (that substrate proposals take this branch end to end is consistent with
`docs/wiring/harness-loop-must-be-proven-live.md` but **UNVERIFIED**, and the prereg's dry run traces it).
**Selection has no LLM, but the narrator does**, so:
- the arc YAML's phase `instruction` strings are empty, as `arcs.py::_make_pref_phases` does, so the
  narrator is never told to write the AUT's sensors;
- the narrator model, its box and the sim box are named (never the leader; the no-co-location rule), and the
  AUT turn's 30 s wall window (`generative_runner.py::_aut_turn_timeout_s`) is stated;
- a sentinel refuses a session whose narrated body-write count is not zero, or reports it as
  narrated-contaminated.
The `scripts/orient_substrate/*` probes build `Embodiment(root=...)` with `agent_id ""` and mint nothing; a
hand-composed harness that calls the factory directly is never the capture (D43). The PLANNING-approved,
parallel and retry `executor.execute` sites do not capture, so their invocations are excluded and counted.
The `--sim` AUT's Cerebellum is never saved on this runtime (§4.2 item 3), which S0b does not need.

**Who chooses the actions** (confounding lens S4). The agent chooses them: the runtime cannot force a
schedule, and G23 forbids apparatus resets as training pre-states. So learned avoidance shapes coverage
(harmful keys are sampled rarely and early), and S0b reports per-key n and band coverage **stratified by
first-burn time**, with the cross-session permutation as the counterbalance.

The capture records, per invocation, the body readings before and after, the target entity's readings as
sensed by the AUT (or their absence), and the pre-action body read (the raw `ActionContext` fields, §4.1).
Today's `actions.jsonl` keeps none of these, so the capture with context needs S1. A `scripts/` harness that
drives the `Executor` directly, the way `test_cerebellum_wiring.py` does, is a known-answer fixture for the
audit script, not the capture. The consequence is computed **offline** from the before/after snapshots by
GL2a's shipped pure function (`sem.py::interoceptive_outcome`; GL2a has merged, so no frozen copy is needed;
environment lens N1), including its net-of-drift report (G14). **"Computable consequence"** is defined in
the prereg: whether a record covering no drive (`sense`, `listen`, `look`, between a third and two thirds of
Exp 42's actions) counts toward the invocation floor (environment lens N5).

**Gate (frozen in the prereg before the capture).** Reconciliation note: the deferred stage map applied
key-count floors to both worlds, but Minecraft has fewer than 10 distinct affordances, so the key floors
apply to the cradle and Minecraft gets a graded-consequence floor.
- **Every count reads experienced-only records** (narrated and apparatus records are counted and reported,
  never gated; §4.1), on one target schema id.
- **Sign is defined per (key, pre-state band)** (confounding lens SF-3). A key's consequence depends on the
  body's state (`fire_pit.touch` harms `arms.thermal` and relieves a cold core), on repetition (the
  nociceptor reads state, so repeated `warm_self` stacks) and on saturation (a touch on an arm already at
  1.0 gives `drive_delta` 0). The bands are frozen in the prereg; records whose pre-state sits at a range
  edge are reported separately and excluded from sign counts; first-trial order is counterbalanced across
  sessions (above). Sign is read per dimension (`harm`, `relief`; `nociception` reported until the
  caused/felt split, §4.1), never from valence. **The cradle gates read `harm` and `relief` only**, on the
  named target schema (`ans-v1` now, or `ans-v2` after GL2b(ii), never both; environment lens S4).
- **Cradle PASS** requires all of (per-key floors restated from the G23 arithmetic where needed):
  - ≥ 500 executed invocations with a computable consequence;
  - ≥ 30 distinct (target entity, affordance) keys, **of which the count with an object block is a
    separate frozen floor** (environment lens D2: about 10 shipped keys carry one, so either the floor is
    restated over keys with a block or new blind-authored items supply them);
  - ≥ 3 consequence classes (e.g. burn, relief, nothing);
  - **name–consequence dissociations, over non-null consequences only, counted in two kinds, each with
    its own frozen floor** (confounding lens DNB-3). A null-consequence key (`observe`/`look`-type, or a
    band in which nothing changes) is in neither kind, because a null/null pair says nothing about
    grounding. **The collision definition is frozen here, in the plan** (confounding lens S2), because under
    "opposite non-null sign" the census's shipped count is plausibly zero (every primary-walk harm flip is a
    same-name **neutral-versus-harmful** pair, bio lens N5, and the azimuth turns are "opposite signs, both
    harmful" on a sensor, not a target dimension), and choosing a definition after seeing its count is the
    post-hoc class:
    - **collisions**: identity-block cosine ≥ 0.44 and, **per dimension**, one key non-null and harmful
      while the other is null or of opposite sign **on that dimension**, provided the union is non-null:
      the "name similarity blocks it" half of T9. "Opposite non-null sign" is reported as a sub-count.
      `touch` on the blanket vs the fire pit, and `warm_self` on a `*_safe` vs a `*_harm` warmth item, are
      collisions on `harm` under this definition. It never borrows `blanket.wrap` / `fire_pit.touch`, which
      sit on different nodes;
    - **convergences**: identity-block cosine < 0.44 and the same non-null sign in the same band: the
      "transfers despite the name" half.

    **Counted as distinct keys and components, not pairs** (confounding lens S3): k mutually convergent
    keys make k(k−1)/2 pairs, and the twelve warmth/hearth/flame items share one effect template, so the
    floors are on the number of **distinct keys participating** and the number of **connected components
    of the dissociation graph**, with effect-template duplicates (the `_b` and α/β twins) collapsed into one
    unit; pair counts are reported beside them.

    The floors are set in the prereg against a census prediction of both counts **on the identity-block
    statistic S0b uses** (cosine of `"<affordance> <entity name>"`, not the census's affordance-name
    decomposition cosine, which gives `touch`↔`touch` = 1.000 where the identity block gives 0.469;
    confounding lens S2, environment lens S5), recomputed by a census revision or by the prereg before the
    capture ([docs/experiments/grounding_census.md](../experiments/grounding_census.md)); the statistic is
    pairwise cosine (order-free), with node co-membership under the production encoding order as a second
    column. S0b confirms them on *executed* records;
  - 100 % of counted invocations joinable on `PhysicalEventId` with a stamp. Pids exist only from the
    post-fence resume stage (G17), so a capture held to this floor runs after that stage; capturing
    earlier needs another deterministic join named in the prereg (an open point for GL4's start, not
    decided here).
- **Cradle FAIL** on any of: < 200 invocations; either dissociation kind below its frozen FAIL floor;
  < 2 consequence classes. A FAIL stops the line; because the floors are now derived from the G23
  arithmetic, a FAIL is evidence about the data, not a world artefact of one saturating campaign. Any later
  re-sourcing of the data is recorded as a post-null change of source (the 2026-09-20 precedent).
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
  - **is action-contrastive within a band, and is a data-existence check** (confounding lens S9): with a
    window that includes the refill, game physics guarantees the difference, so it tests that the producer
    and the join ran. PASS needs ≥ 100 executed `escape_water` invocations spanning ≥ 3 oxygen bands, and
    a difference Δoxygen(`escape_water`) − Δoxygen(`turn`) within each band whose **sign and minimum
    magnitude per band are pre-registered**. `turn` is the executing control: the bridge's `turn` is
    `bot.look(yaw, pitch, true)` (`scripts/minecraft_bridge/index.js`, `case "turn"`), which runs submerged
    and has no oxygen effect (environment lens S3, N2; resolves the earlier UNVERIFIED). The "no-op pass" is
    dropped as a control (a non-act has no `pid` and cannot carry a `cause_pid` join); the same agent's
    drowning trace in the same band is reported as a comparator. A gate on Δoxygen by band alone is refused.
    The graded "how far is pain" readout is a predictor property, tested in S5's Minecraft arm against its own
    floor, never inferred from this contrast;
  - **bands are filled within one agent type, by placement depth/delay** (environment lens S3): in
    `r3_bench.jsonl` surface times cluster at about 3.0 s, 8.3–9.1 s and 27.5–28.5 s, and the modes track the
    arm (fear-learned agents escape early, innate-only late), so mixing arms would confound band with learner
    history. Each invocation's pre-state band is recorded. Band edges are frozen in **sensed units**
    (`oxygen` is declared on [0, 40] while the game value is 0–20; read through
    `substrate_proposal._read_world_ranges`, `docs/wiring/sensor-range-clamps.md`; environment lens N3);
  - **needs an apparatus that leaves the bot surfaced without a teleport** (environment lens S2; strict
    option (a)): on every shipped water harness the refill happens after an apparatus teleport
    (`scripts/survival_world/water_trial.py` teleports at surface, `t_end` − `t_surface` ≈ 0.002 s in
    `r3_bench.jsonl`; `WaterTrial.rescue` teleports, settles, then heals), and without it the live bridge
    holds jump only 600 ms after the eyes clear while the measured sinkback is 2.119 s. So the prereg builds
    a pool with a reachable ledge or air pocket, checked by a `check_surface_cell_air`-style preflight and a
    measured sinkback time, and freezes the window shorter than the sinkback. (Non-strict (b): end the window
    at the first harness write and accept the short in-water refill.) `water_trial.py`'s exit is never
    edited: it is a cited guard of the Exp 60/61/62/R3 rows;
  - **budget:** 100 escapes at R3's rate (about 30 events per hour) is about 3.5 h of rig time plus setup,
    in a Track C slot never stacked on an E-rung; Exp 60's probe regime (65 `escape_water` calls, all below
    the 4.3 s probe cap, one band) cannot reach the floor;
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
  of apparatus writes by harness and cause (G16), caused versus felt-only nociception (unavailable before
  the split; on the cradle tool path every entry is caused, so the split matters only in Minecraft), the
  with-narrated version of every gated count, the excluded non-capturing invocations, per-key object-block
  presence and sense counts, per-key n and band coverage by first-burn time, and capture-queue drops per
  session (`HippocampusConfig.capture_queue_size` 100, drop-oldest; `max_nodes` 10,000; environment lens N6).

**Guards.** The audit script's own known-answer test, against a hand-built three-invocation fixture.
**Blast radius.** None: no `src/`, no ledger row.
**Owner decisions at S0b start:** Q7 and Q8; the capture scene, body, runtime and N (Q9; the runtime is now
named, G23 sets the shape); whether the Minecraft apparatus is the new surfaced pool (a) or the
first-write window (b) (environment lens S2; strict (a)); whether the
Minecraft capture rides the rig in
an existing Track C slot or the offline scripted bridge (recommended: the cradle capture now; Minecraft on
the rig in a slot that is never stacked on an E-rung campaign, because the scripted bridge's physics is not
game-native and the arm exists to close the circularity of authored physics); and the 0.44 identity-block
threshold used to count dissociations (recommended: production 0.44, never the retired 0.40).

### S1 — Cerebellum target fix + pair record (engineering, record only)

**What** (amended 2026-10-24 by owner decisions G22 and G25; §4.2 items 1–2).
- **The observe moves into the executor** (G22, wiring lens D1). The `observe_from_action` call and the
  `get_confidence` → `sim_cerebellum` line are deleted from `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute`
  (which shrinks a 502-line pinned function, satisfying Rule T; wiring lens N6) and re-made in
  `Executor._stamp_invocation`, right after `Executor._interoceptive_outcome` returns a record, reaching the
  key through the #1195 accessors. `get_confidence` reads the same key, so it never reports a key nothing
  writes. The moved block carries two `except Exception: log_swallowed_exception()` handlers. Moving them is
  an **edited** move, so S1 adds the `scripts/swallow_moves.json` record (pre-merge review, Executor N7,
  Wire N2).
- **The tool-level `cerebellum=` argument goes** (pre-merge review, Wire SF-3; strict). Once the observe
  is in the executor, `ModulatorAffordanceTool(cerebellum=)`, `generate_tools_for_entity(cerebellum=)`,
  and the threading in `Executor.generate_entity_tools` and `simulation/dm_runtime.py::SceneState` would
  hold a collaborator the tool never uses. A kept-but-unused collaborator makes its pin vacuous, so S1
  removes the argument at each of those sites. The Dormant
  `embodiment/backends/cerebellum_modulator.py` is **not** one of them: its `cerebellum=` is passed to
  `CerebellumModulator`, the read-side backend's own collaborator, not to a tool, so it stays unedited
  (checked at `d9dffe6e`; the Wire lens's C1 table listed it as a tool site). The D79 pins in
  `tests/unit/test_build_executor.py` change with it: the `executor._cerebellum is cerebellum` pin stays,
  and the `kwargs["cerebellum"] is cerebellum` pin on generated tools is replaced by a pin that the
  executor's `_stamp_invocation` observe reaches the same Cerebellum. Intended behaviour: training follows
  the record. Every tool an executor holding `_cerebellum` runs, on a body that mints a record, trains,
  whatever built the tool. So the `--sim` generative cradle scene tools and the DM `SceneState` tools
  **start** training. S1 lists these gains beside the losses (next bullet). **Imagined entities among
  them (delta round, Wire N-2):** the generative runner's `_imagine_entity_from_ref` entities
  (`metadata={"imagined": True}`) are among those scene tools, so after S1 their tools train the
  Cerebellum too, and their writes carry `imagined` (thalamic_relay.md §5.7 item 1, not `narrated`). S1
  states this gain; whether imagined-entity outcomes may train is the S1-time analogue of Q3, and the
  strict default is Q3's: they are excluded from training (the record carries the kind, so S1 can refuse
  them at the observe).
- **Two targets on one key.** The new **reafference** target (the declared body sensors' signed change, net
  of declared drift, caused entries only: the record's `drive_delta` plus raw deltas for body sensors
  without a declared drive) is written under a new payload key; the existing object-readings model keeps
  learning under `models` from the tool's `entity_state`, which the tool already returns in
  `output["entity_state"]`. `valence`, `drive_pain`, `urgency` and `deviation_after` are not in either
  target. Alliesthesia is applied at read time by whatever reads it (none until E7 or S5).
- **The key is unchanged** (wiring lens S3; G22): owner = acted-on object on every shipped item, so a
  target-entity key would be a no-op, and no situation enters it. Param bucketing keeps the `sensor_ranges`
  it has today (the modulator owner's sensors, matched to param names).
- **The scope change is counted:** the Cerebellum stops training where `_outcome_start` mints no record
  (`agent_id == ""`). S1 lists which executors carry a Cerebellum and lose it (UNVERIFIED today), and
  which tool sets start training (the generative cradle scene tools, the DM `SceneState` tools; above).
- **The `ActionContext` is assembled in the executor** (`Executor._outcome_start` / `_stamp_invocation`,
  through the #1195 accessors; G25, wiring lens S3), beside the `pid` the tool path stamps on `ToolOutput`
  from the post-fence resume stage (G17; GL2a stamps none). It records **raw inputs** (§4.1; wiring lens
  D3): the entity name and affordance, params, the AUT's pre-action body read (from `_outcome_start`'s
  snapshot widened to the body's sensors, inside G14's `_run_started` scope), and the target's readings as
  the AUT sensed them before the decision, or `None`. `entity_path := CauseRef.entity`. It is captured into
  `EncodingSignals.extra["context"]`. Its `provenance` is set explicitly at assembly (`experienced` on the
  AUT's own executed invocation); there is no default. The `ToolOutput` field is `repr=False`, an edit to
  `tools/base.py::ToolOutput` (#1189, §3.3).
- **Where "sensed before the decision" comes from is named at S1 start.** The AUT's `sense` results live
  in its percept stream, not in the executor; S1 names the read (and, if it needs a fenced edit, the
  object block waits for the fence's slice and is recorded absent until then).
- The `pid` S1 persists in `extra["context"]` is the cross-session one: S1 lands only after the resume
  stage (G15, G17), which builds the pid and its resume together, so no pid is ever persisted without its
  resume. The Cerebellum payload carries no per-sample pid (its means are aggregates; wiring lens N4).
- The Cerebellum payload version moves to `"1.2"`. It is the payload-layer `"version"` string that moves,
  not `_format_version`: the persistence brief tombstones payload `"version"` strings, but the Cerebellum is
  not one of the envelope-conformant stores and `Cerebellum.import_state` still reads its `"version"`
  (wiring lens N2). A `"1.1"` file's `models` **are kept as the object model** (G22: add, not replace), its
  reafference statistics start empty, and one warning is logged (`check_format_version` pattern). It is
  never a silent no-op, and the old absolute-reading means are never reinterpreted as reafference.
  **`import_state` refuses a version newer than it knows** with a **typed exception outside
  `UNREADABLE_STORE_ERRORS`** (or `load_safe` handles it as it handles `OSError`), so the newer file is left
  untouched and saves over it are refused; a `ValueError` would make `load_safe` set it aside and overwrite
  it (wiring lens S4). The reafference means sit under a new payload key, so a pre-S1 install that reads a
  `"1.2"` file finds its object model where it expects it. The `programs` half of the file is byte-identical.
  `embodiment/cerebellum.py::Cerebellum._reset_store_state` empties the new payload key too, so a reset or
  refused store keeps no reafference means (pre-merge review, Wire SF-8; the reset test covers it).
- **The Bio-mapping docstring of `embodiment/cerebellum.py` is rewritten** (G22, bio lens D2, N4): FUNCTIONAL:
  body reafference Δ plus the object's response; NOT valuation; reafference prediction without
  attenuation; the trial-1 copy (§3.1) noted.
- `ActionContext` gets its CC3 docstring (path a).
- **The claim surfaces that say the Cerebellum trains in `tool_bridge.py` are corrected in the same PR**
  (pre-merge review, Wire N1): `docs/wiring/engram-formation.md` §1 and §6,
  `docs/wiring/body-and-word-worlds.md` §3 step 5, the `embodiment/cerebellum.py` docstrings, the
  `tool_bridge.py` comment block, and `tests/unit/test_cerebellum_dormant_909.py`'s module docstring (its
  scan stays green, since `executor.py` is in `src/`).
- **Coordination:** `runtime/executor.py` and `tools/base.py` are Session V's files in roadmap_1_3_x.md, so
  the S1 PR names the coordination (pre-merge review, Architecture N4). There is no fence conflict, since S1
  lands after 1.3.6.

**Gate.**
- **Zero selection-path diffs:** grep the diff for `recommend_action`, `credit_node`,
  `TemporalCreditDistributor`, EC encode, `_sensor_embed`, `LinguisticEncoder.embed` and
  `record_encoder_provenance` (no hits), and `ec.json` is byte-identical on a scripted substrate-primary
  capture (no hot-path embed; wiring lens D3).
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
- **`test_cerebellum_wiring.py` is re-pointed** (G22, wiring lens D1): its `_rig` drives
  `ModulatorAffordanceTool.execute` directly with no executor, so it could never see an
  `InteroceptiveOutcome`. It now drives `build_executor` → `Executor.execute`, and its AST call-site check
  (`test_every_call_site_kwarg_exists_in_the_signature`) points at `Executor._stamp_invocation`. It asserts
  that the infant's `arms.thermal` +0.6 reaches the Cerebellum's reafference model for `fire_pit.touch`,
  that the object model still updates under `models`, and that `get_confidence` is read on the same key.
- Deletion probe: removing the observe call, or feeding it `entity_state` as the reafference target, fails
  the test.
- A format round-trip in `tests/integration/test_persistence_compat.py`, covering `"1.1"` → `"1.2"` (the
  object model kept), plus a refusal test: a newer file survives a session end byte-identical.
- `test_cerebellum_dormant_909.py` stays green; only its module docstring's claim is corrected (above). The
  read side is untouched.
- The re-pointed D79 pins in `tests/unit/test_build_executor.py` (above), and a gain-site test: a scene
  tool generated without `cerebellum=` and run by a record-minting executor reaches the Cerebellum's
  reafference model.
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
  but it is SUPERSEDED by T1-16 and no longer gates. The trace now grows by raw inputs only (§4.2 item 1).
- A `changelog.d/` fragment is added (`scripts/changelog_fragments.py check`; before M83 step 3 this was `[Unreleased]` growth).

**Owner decisions at S1 start:** Q6 is **decided** (G22): no key change, no situation in the key. Asked at
S1 start: the source of the "sensed before the decision" flag (above); a resolved `target=` case, only if
the owner names one.

### S2 — offline predictor and evaluation (`scripts/`, numpy kernel ridge)

**What.** `scripts/train_consequence_predictor.py` and `scripts/eval_consequence_predictor.py`, trained on
S0b's cradle capture. They own the target projection (§4.1) and embed the raw `ActionContext` fields into
blocks offline, under recorded geometry tags (§4.1). `docs/experiments/consequence_predictor_offline_prereg.md`
lands on `main` before the eval reads the data.

**S2 is a capability stage, stated as such** (owner decision G25, confounding lens D3). The first draft
gated on held-out **collision and convergence keys**, which has no admissible route to PASS on the shipped
data: a held-out collision key is one whose name misleads, so a zero-shot prediction for it can come only
from the object block; 24 of the 34 keys have none (sensorless items, zero vector), so full ≈ identity-only
there by construction and the 0.2 margin is unreachable; the one shipped hidden-property pair has
byte-identical sensors; and where it could pass, the margin over the sensed baseline is a ridge-vs-1-NN
model-class difference on the same features. A FAIL would have stopped the line for a reason unrelated to
ridge. G13's test is S5's authored set.

**Gate (pre-registered).** Every number is read on the change-only target, per dimension (`harm`,
`relief`; `nociception` reported until the caused/felt split, §4.1), on experienced-only pairs and one
target schema id. The target standardisation and λ are fitted **on training folds only**, with the
procedure pinned before the data is read (confounding lens S5):
- **(i) Capability: within-key held-out invocations.** Split by invocation **blocks** within a key and
  pre-state band. The full model beats the **context-copy baseline** on the phasic dimensions by a
  pre-registered margin. Context-copy is defined as the **situation block only** (the levels): a copy model
  that kept the identity or object block would nearly know the key (confounding lens S5). A **key-mean**
  baseline (the training mean per key, what the Cerebellum would say) is reported beside it. This answers
  "does it learn per-key consequences and not copy state".
- **(ii) Across keys: matched-model-class ablations.** The full ridge against identity-only ridge and
  object-only ridge, with the same λ procedure, **gated only on keys with a non-empty object block**;
  sensorless keys are reported as their own "identity-only by construction" row (environment lens D2).
  Held out by key, by entity and by **twin group** (`green_hearth` + `green_hearth_b`, `warmth_alpha_*` +
  `warmth_beta_*`), so a twin in training cannot make a same-effect held-out key trivially predictable
  (confounding lens N5).
- **Reported, not gated:** the no-learning sensed-similarity baseline (the consequence of the nearest
  trained items by object-block cosine, with no fitted map; G13), with its degenerate cases stated
  (disjoint sensor names give near-orthogonal blocks whatever the values; empty blocks have undefined
  cosine); the held-out collision and convergence keys, with full, identity-only and sensed-baseline sign
  accuracy; the acquired-equivalence probe, which is S5's (§6) and is "not run" here; the jet triad, as the
  sensed-feature generalisation control (§6).

**What FAIL means.** Only a FAIL of (i) stops the line: then try the MLP once, pre-registered as a second
iteration, or stop. A FAIL on the dissociation keys or on (ii) does not stop it by itself; it is recorded
(§9's divergence row). Every gate reads experienced-only pairs; the with-narrated number is reported beside
it and never gated (§4.1; confounding lens SF-7, G6 not reopened). Also measured and reported (environment
lens S8): solve time, peak memory and file size on the named machines at n = 10³ and 10⁴ (Pi UNVERIFIED).

**Guards.**
- The eval's own anti-vacuity row: a shuffled-target control must score at chance. The shuffle **permutes
  consequences across keys**, keeping each key's invocations together; a within-key shuffle leaves key
  identity predictive and can sit above chance for a reason unrelated to the model (confounding lens NIT-4).
- A two-process determinism check on the script's JSON output (the precursor of S3's test).

**Blast radius.** None. No ledger row moves on an offline result (weak-evidence rule: this is capability,
not a claim).
**Owner decisions at S2 start:** the kernel choice (open; recommended: a cosine kernel per block with fixed
block weights pinned before S0b's data is read, ridge λ chosen by nested CV on the training folds only,
before the held-out data is read; confounding lens N4); confirming caused-only nociception in the target
(§4.1; strict default: caused only, felt-only masked; gated only once the split has landed, wiring lens
S2; non-strict: gate on the `caused_or_felt` mix and say so). The
core valence formula (unweighted `relief − harm − nociception`, G11) and urgency v1 (pressure only, G12)
were decided 2026-10-09 and are not re-asked.

### S3 — session-end retrain, persistence, guards, shadow consumer (`src/`)

**What.**
- `memory/consequence_predictor.py::ConsequencePredictor` (`encode_context` / `predict_latent` /
  `encode_target`). The module docstring describes it as "an offline batch refit over the replay buffer;
  FUNCTIONAL analogy only, NOT CLS interleaved learning" (bio lens SF-6). It owns the target projection and
  the offline block embedding moved from S2's script (§4.1). The refit runs at `BioStack.on_session_end`, in
  the order §4.2 item 3 names, selecting traces by `run_id`, opt-in via a `maxim config` key (prefer
  config over a new env var; an unavoidable env var gets an autouse conftest scrub in the same commit).
- **The cradle seam is #1449's own Session-A slice, which S3 depends on and cites** (G25, wiring lens D2;
  strict option (a); split out of S3 by the pre-merge review, Architecture S6; §4.2 item 3). That slice
  makes the orchestrator's AUT teardown, in `simulation/orchestrator.py::start_simulation_mode` after the
  AUT thread joins and the Hippocampus has flushed, call `_aut_instance.bio_stack.on_session_end()` (which
  saves the Cerebellum and runs `distributor.cleanup_session`). It skips the adopted HANDLE branch, ends
  each `/new` recursion level's AUT once, and carries its own test (it drives `start_simulation_mode`'s
  teardown on a fixture runtime with a mocked LLM, asserts that the **AUT's** `BioStack.on_session_end`
  ran on both branches, and a deletion probe fails it), Rule T and its own ledger walk (T1-13's anchor
  pruning, the `--sim` / `--resume-sim` rows). It is not behind the predictor's flag. The first draft's edit
  at `runtime/agent_factory.py::AgentInstance.shutdown` is **struck**: on `--sim` that is the narrator's
  instance, so it would refit an untrained stack and pass a caller grep. `AgentInstance.shutdown` reaches
  `save_cerebellum()` only, never `on_session_end` (Wire SF-1). So S3's refit runs on the Minecraft harness,
  the survival-world and Exp 56 scripts, and the `--sim` AUT after #1449. It does **not** run on the
  console HANDLE, `api.py` agents or `create.agent()` agents, and S3 claims none of them (§4.2 item 3).
- **S3's own resume edit:** `_restore_aut_from_session` (`--resume-sim`) and the `aut_*.json` save it reads
  gain the Cerebellum and the predictor, each with a `test_persistence_compat.py` round-trip. This sits in
  the fenced `start_simulation_mode`, so it is a Session-A-coordinated edit after 1.3.6, with Rule T
  accounted for. It changes the resume contract that `simulation/report.py::RESUME_STORES` / `resume_stamp`
  / `_STORE_OK` and `scripts/o19_rerun.py` / `scripts/o19_verdict.py` read. **T1-16's `Re-run on:` fires by
  wording** ("the resume path (`simulation/report.py` RESUME_STORES)"). Whether the new stores join
  `RESUME_STORES` (so `resume_stamp` reports them) is stated in the S3 PR with T1-16's walk (pre-merge
  review, Wire SF-8; strict: they join, and the o19 scripts' pinned store tuple is re-checked).
- JSON persistence as in §4.2 item 4, with `check_format_version` on load, geometry refusal and the store
  contracts (`StoreFileOwnership`, `load_persisted`, corrupt-copy, the n cap and size budget; wiring lens
  S5, environment lens S8). The predictor file's entries in the agent-store registries **land here in S3**,
  with the file. They are written out once, with the learner's and the map's, in S4's persistence, resume
  and exchange contract (§7 S4).
- The shadow log of predicted core, cluster and α per candidate tool, written to the existing JSONL
  trace surface only, at the fenced decision site §4.2 item 7 names.
- Whether this stage's contamination guard is GL3.B1's landing consumer (G8), or that is left to S5, is
  stated at S3 start (wiring lens S8).

**Gate.**
- All guards are red before the change and green after, each proven by deletion (delete the join check,
  the provenance check or the geometry check, and the matching test must fail).
- **A test, not a grep,** shows the refit is reached from `on_session_end` on both the cradle (the `--sim`
  AUT teardown, through #1449's seam) and the Minecraft harness paths ("a fix ships with a caller"; a grep
  passes on the narrator's instance while the AUT never reaches it).
- Three-lens code review, whose Wire-integrity lens writes the producer → consumer table for the pair,
  the persisted file and the shadow log.

**Guards.**
- `tests/unit/test_consequence_no_contamination.py`: a hand-built pair without a minted
  `PhysicalEventId`, a pair with no provenance, and a geometry-mismatched pair each RAISE at construction
  or at `add_pair`; a `narrated` pair relabelled `experienced` RAISES; a `narrated` pair's training weight
  equals the declared discount (deletion probe: drop the discount and the test fails).
- `tests/unit/test_consequence_predictor_two_process.py`: two processes with differing `PYTHONHASHSEED`
  train on the same capture and write byte-identical JSON. It must fail against a version that orders pairs
  by `dict` iteration. The BLAS thread count is pinned (or the persisted floats rounded), because a numpy
  solve is not guaranteed bit-stable across thread counts (wiring lens S5).
- `tests/integration/test_persistence_compat.py` round-trip, plus a corrupt-copy test.
- The selection golden is byte-identical with the flag on (the shadow log has no selection effect).

**Blast radius.**
- The predictor: none by wording, since no selection, credit or EC symbol is touched.
- #1449's cradle seam (its own slice, walked there; recorded here because S3 depends on it): it adds
  `cleanup_session` to the **AUT's** distributor on the `--sim` path (wiring lens D2 item 4). T1-13's
  `Re-run on:` names "temporal-anchor pruning / `TemporalCreditDistributor` credit-path change". Its earned
  path (the Minecraft harness and the survival scripts) already calls `on_session_end`, so that path does
  not change. The walk is recorded on T1-13 and lists every earned row whose evidence ran on `--sim`; if one
  is affected, the owner rules on it before merge (strict default: a re-run).
- S3's resume edit: `--resume-sim` newly restoring the Cerebellum and the predictor changes what a resumed
  `--sim` AUT carries. The walk covers rows whose evidence used `--resume-sim`
  (`docs/wiring/cradle-sim-arms-and-resume.md`), and **T1-16 by wording** (RESUME_STORES, above).
- **T4a applies:** the predictor is a mechanism with no selection consumer until S5. If S5 has not earned
  by the 1.4.0 transaction, `ConsequencePredictor`, the `"consequence"` modality, the node-keyed learner, the
  membership map, its freeze trigger (which has no production caller at all; §4.2 item 5), the
  matched-class floor learners (built in S4 with only the S5 harness as caller; delta round, Architecture
  N2, Executor N-4) and the α read (S4) are marked `Dormant since <date>` in their docstrings and the release PR says so (wiring lens N6).

**Owner decisions at S3 start:** Q3 (imagined pairs); the config key name; whether S3's contamination guard
carries GL3.B1 (Q10).

### S4 — the consequence representation: opt-in `"consequence"` EC modality, node-keyed learner, similarity tiering

**What** (amended 2026-10-24 by owner decision G21; §4.2 items 5–6). All behind the S3 config key, default
OFF:
- the modality's registration, with nodes formed from **experienced outcome records** only, through a new
  **EC-only encode function** (`_sensor_embed` → `pattern_complete_or_separate`; no ATL activation, no NAc
  eligibility, no `threshold_override`; wiring lens S7) with its M39 allowlist entry (Executor N2), and
  the modality listed in `frozen_centroid_modalities` for its **prototypes** only;
- **the per-item → consequence-node membership map** and **its freeze trigger** (§4.2 item 5 part 1;
  pre-merge review, Architecture B1 = Executor S10). The map gives one node per item, the node nearest the
  encoded mean of the item's experienced Stage-1 outcomes, with ties broken by node creation order then
  node id. The trigger builds and fixes the map at a declared boundary (Stage-1 end in S5). It is called
  only by the S5 harness and tiered as an experimenter control. It has no production caller, and S4's PR
  says so;
- the encoded vector exactly: the raw change-only projection plus the per-drive signed `drive_delta`,
  centred [−1, 1] ranges, absent drives masked, the ungained embed unless A4 is justified by a measurement
  at this N (bio lens SF-1, SF-2);
- the **node-keyed outcome learner**, updated through the map when a member item is acted on, and its
  readout through membership (G21); **S4's front-gate answer** picks its home (Q11: a **new** NAc table
  keyed on the node id holding a signed per-dimension value, or a node → outcome running mean on the
  Cerebellum's delta rule). It also says why `cluster_reward_bias` keyed by (cluster, tool) cannot carry it
  (tool-specific: B cannot inherit A's entry), and why `NAc._reward_bias` cannot either (a non-negative
  scalar clamped to `[0, max_reward_bias]` that widens recognition radii; §5);
- the **matched-class floor learners** S5 gates against (§6): the same learner, map rule, freeze and
  Stage-2 booking keyed on identity-similarity (EC `"text"`) nodes and on sensed-reading (object-block)
  nodes. They are built here so that S5's harness composes them and does not hand-roll them;
- the α-blended similarity read (§4.2 item 6), precision-weighted, reported only.

**The persistence, resume and exchange contract** (pre-merge review, Architecture S9, Wire SF-8). It covers
the learner's table, the membership map and S3's `<home>/consequence_predictor.json`. The predictor's
entries land in S3 with the file; the learner's and the map's land here.
- **Writing:** `with_format_version` + `atomic_write_json`; `check_format_version` on load; a
  `hash_scheme` stamp wherever a hashed value persists.
- **Store contracts:** `StoreFileOwnership` with a `_reset_store_state` that empties every surface
  `import_state` / `load` fills, `load_persisted` honoured, and a corrupt file copied aside. If the learner
  lives on NAc, `nac.json`'s persisted shape changes, so `NAc.dump` / `load_state` gain the field with a
  `test_persistence_compat.py` round-trip, and the NAc persisted-shape trigger joins S4's blast radius.
- **Both real load seams, with round-trip tests:** `runtime/bio_stack.py::build_bio_stack` and
  `simulation/orchestrator.py::_restore_aut_from_session` (`--resume-sim`). If the map or a non-NAc
  learner is a new file, it also joins S3's `aut_*.json` save and `RESUME_STORES` (T1-16's trigger, §7 S3).
- **Agent-store registries:** `runtime/agent_factory.py::AGENT_STATE_FILES` (so `create.agent`'s #939
  refusal sees the new files and `migrate_agent_state` moves them); `cli_utils.py::_AGENT_FILES` /
  `MEMORY_PATHS`; and the D2 co-invalidation groups. The map and the learner are keyed on EC node ids, so
  they are invalidated together with `ec`. The predictor's training set is pid-keyed to Hippocampus traces,
  so S3 states its group (strict: with `ec` and `hippocampus`).
- **`maxim substrate prune` (delta round, Wire N-4).** `hivemind/cli.py` → `merge.py::prune_nac_cluster_biases`
  removes EC nodes and prunes only `_CLUSTER_ROW_FIELDS` + `reward_bias`. The contract covers it: when a
  pruned node is a consequence node, its learner rows and its membership-map entries are dropped in the
  same prune (the floor learners' rows with their text / object-block nodes likewise), so no row points at
  a removed or reused id. A prune round-trip test (prune a consequence node, assert no learner or map row
  names it) joins S4's guards.
- **The hivemind:** `hivemind/bundle.py::scrub_nac_state_for_bundle` ends with the `_BUNDLE_NAC_FIELDS`
  allowlist (read at `d9dffe6e`), so a **new** NAc field is dropped at export by default. The Wire lens's
  "an unknown field ships" no longer holds there. The contract is therefore: the learner's NAc field is
  never added to `_BUNDLE_NAC_FIELDS`. A learner written into an **existing** exported field (for example
  `reward_bias`, which the front-gate rejects anyway) is refused. `test_hivemind_excludes_consequence.py`
  is extended from EC nodes to NAc fields keyed on consequence node ids, proven by deletion (add the field
  to the allowlist and the test fails). The membership map and a non-NAc learner are not bundle surfaces;
  the test pins that too.
- **The Cerebellum:** `Cerebellum._reset_store_state` clears S1's reafference payload key (S1).

**Gate.**
- With the flag OFF, the EC dumps for interoception, audio, world and text are byte-identical before and
  after (golden).
- With the flag ON, consequence nodes form only from experienced outcome records, never from predictions
  (bio lens SF-4).
- **An offline cosine replay of S0b's vectors through the real encoder** (wiring lens S7;
  `docs/wiring/cosine-separation-is-directional.md` corollary 3): a null consequence is the zero vector, mild
  and severe of one kind co-cluster at the modality's own pinned threshold, burn and relief separate, and
  outcomes on different drives do not share a node. The threshold is measured here and then pinned, with a
  follow-up trigger (bio lens SF-1).
- `turn_left`/`turn_right` and the `touch` ×16 collisions separate by consequence on the S0b data where the
  identity block collapses them, read on **held-out invocation blocks** (on trained keys it would follow from
  the per-key means, a tautology; confounding lens N1). It is reported, not claimed.

**The hivemind edit.** Local-only does not come for free: `hivemind/bundle.py` exports every
`substrate_nodes` entry whatever its modality, so keeping `"consequence"` nodes out of a bundle IS a
`bundle.py` export-filter edit. On the receiving side, `hivemind/merge.py::SENSOR_MODALITY_THRESHOLDS` has
no `"consequence"` entry, so a foreign bundle carrying one would fold at the default cosine threshold, not
at the modality's threshold. S4 therefore either adds `"consequence"` to `SENSOR_MODALITY_THRESHOLDS` or
refuses the modality at merge/ingest (Q4; strict: refuse it). The `IDENTITY_DOMAIN_MARKER` route (§4.2
item 5; wiring lens N5) is a Q4 option, not a recommendation.

**Guards.** `test_ec_consequence_modality_golden.py`; `test_hivemind_excludes_consequence.py` (no
`"consequence"` node in an exported bundle, and a bundle carrying one is refused at merge, or folded at the
modality's threshold if Q4 takes the threshold route); a null-is-zero and relief-⊥-harm guard; a guard that
no `"consequence"` node id ever appears in NAc eligibility, `reward_bias` (other than the node-keyed
learner's own table, if it lives on NAc) or ATL activations, proven by deletion (wiring lens S7); the
node-keyed learner's own known-answer test (a member's update reaches another member's readout and not a
non-member's), with a deletion probe; **the membership map's deletion probe** (delete the map, or let a
Stage-2 encode re-assign an item by cosine, and the known-answer test fails; Executor S10); a
known-answer test of the map's assignment rule and tie-break, run in two processes with differing
`PYTHONHASHSEED`; the round-trip tests through both load seams; the registry and bundle guards of the
contract above.

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
- **The node-keyed learner is a new credit surface** (G21 lifts §8's `credit_node` exclusion for this
  path; under §5's answer it does not write through `credit_node` or `_reward_bias`, so the lift is held
  unused). If it lives on NAc (a new node-id table), every row whose
  `Re-run on:` names an NAc credit-path change is in scope (an earlier trigger walk in
  behavioral_graduation_candidates.md cleared T1-13, T1-14 and T1-15 because `credit_node` was unedited);
  if it lives outside NAc (the Cerebellum-delta-rule form), the walk is structural. Either way, with the flag off, the selection, decision
  provenance and NAc goldens are byte-identical. The walk is owed at S4 start, with the home.
- **The new persisted state** (the contract above): if the learner lives on NAc, `nac.json`'s shape
  changes, which fires every row whose `Re-run on:` names an NAc persistence or a hivemind NAc-field change.
  The registry and resume additions are walked against T1-16 (RESUME_STORES) and the #939 refusal.

**Owner decisions at S4 start:** Q4; the node-keyed learner's home (Q11); the T1-11/T1-12/T1-14 discharge
ruling, and the walk for the learner's credit surface.

### S5 — the behavioural experiment (the first grounding claim, T9)

**What.** Working title "consequence transfer". The **four-lens experiment DESIGN review**
(`docs/experiments/DESIGN_REVIEW.md`: confounding, bio-faithful, wiring, environment; all four, because it
is a new claim) runs on the prereg **before any harness is built**. Reports go verbatim into
`docs/experiments/rationale/consequence-transfer/`. Then the prereg lands on `main` before data, the harness
gets its code review, a dry run, and then the run. Substrate-primary selection, no LLM in the action path.
**The capture runtime is named per arm in the prereg** (the S0b rule: invocations that reach
`capture_loop_action` through the real loop, never a hand-composed harness; wiring lens S10).
**The AE and AD arms need a forced schedule, and `--sim` cannot force one** (pre-merge review,
Architecture S11, Executor S11). They need equal Stage-1 counts, an interleaved counterbalanced order and
a per-trial reset. The `--sim` runtime's agent chooses its own actions (S0b, "Who chooses the actions"), and
explore-first fixes first-trial order. **So these arms' driver is a scripted sequence through the real
loop and capture, to be built** on the `tests/unit/_cradle_loop_driver.py` pattern that GL2a and GL3's TR24
use. That driver runs `run_agentic_loop` substrate-primary on the canonical builders (`build_bio_stack` +
`build_executor` with the AUT's `agent_id`), with the substrate proposer replaced by one that returns the
script. Every action then goes `run_agentic_loop` → `tool_dispatch.execute_and_learn` →
`capture_loop_action`, and the session ends through the real `BioStack.on_session_end` (so S3's refit and
the Stage-1 / Stage-2 boundaries are the shipped ones). S5's driver lives in `scripts/` with its own
known-answer test, and the prereg names it. **How it replaces the proposer outside pytest (delta round,
Architecture N4):** `tests/unit/_cradle_loop_driver.py` does it with pytest's `monkeypatch`; the
`scripts/` driver uses `unittest.mock.patch.object(substrate_proposal, "propose_via_substrate", …)` as a
context manager around the run. That is the same single module attribute, because
`runtime/loop_substrate.py` calls `_sp.propose_via_substrate(...)` through the module reference so one
patch covers it. It adds **no `src/` seam**; a production injection point would be a new mechanism and
would need a front-gate answer, so it is not taken. **Why it counts as the composed loop and not a hand-composed
shortcut:** only *selection* is scripted, and these arms' readout is learner state with no choice claim
(G27), so selection is not under test. Every component under test (the executor's record, the capture,
the EC encode door, the map, the learner's booking, the refit) runs on its production path. A hand-composed
harness that calls the learner or the factory directly is still refused (D43). The untried-item arm
reaches selection through the Q5 seam, so its probe phase runs on the unscripted loop, and the prereg names
which.
**How the reset is stamped** (pre-merge review, Architecture S11, Executor S11). The reset is a harness
write through GL3.B1's one kind-taking setter with the keyword-only kind `apparatus` (the last-writer
slot, thalamic_relay.md §5.7 item 1, TR22; never a raw `vital_metrics[...] =`, which M78's lint refuses),
so **S5 depends on GL3.B1** (§7 table; delta round, Architecture SF-3). The driver opens GL3's apparatus window around it, closes it, and takes
one settle snapshot before the trial's decision, so the window's "until the first snapshot taken after it
closes" ends before the AUT acts. **The reset forms no training pair** (G26, narrowed). The trial's pair is
the AUT's own experienced act. Its outcome is the caused change the record computes, stamped
`experienced`. Its `body_pre` was set by apparatus, and that is recorded beside the pair
(`ActionContext.extra["pre_state_writer"] = "apparatus"`), never as the pair's provenance. Under GL3's
strict slot rule ("narrated/apparatus whenever any declared sensor's last writer was"), the post-reset
**situation encodes** read `apparatus`. That does not empty S5's gated data, because consequence nodes are
formed from outcome records and the learner books on the outcome's provenance (§4.2 item 5), not on a
situation encode's. It does mean that, under GL3.B1's credit flag, the trial's cluster credit is refused.
The prereg states this, and the matching statement for GL3's window rule is made in thalamic_relay.md
§5.7 item 1 ("GL4 S5's harness resets": the reset write holds `apparatus` in the slot, inside the
window, and a post-reset situation encode reads `apparatus` until an AUT writer touches the sensor;
delta round, Executor N-3). **Before the
prereg, the composed-loop toy** (`docs/plans/rationale/latent-forward-model/s5_toy.md`; G21) shows that the
mechanism predicts B > C and that every floor does not (§6). Its arms (redesigned by owner decision G13,
2026-10-09; amended by G21, 2026-10-24):
- **Cradle, authored physics: acquired equivalence (the test, §6). Credits the consequence
  representation** (frozen nodes + node-keyed learner), not the predictor. New AUT-invoked items with a
  `self_effect` on the AUT, whose **sensed readings and consequences disagree**: pairs that read alike and
  act differently (a hidden property) and pairs that read differently and act alike, with names that point
  neither way, built to §6's item-set rules (one sensor schema, no empty object block, numeric name
  neutrality, directional O1/O2/O3). Stage 1 gives **equal experience of every item**; membership freezes at
  Stage-1 end; Stage 2 retrains A; the test reads which unretrained item inherits. **Primary measure:**
  per dimension, Δ_B − Δ_C from Stage-1 end to Stage-2 end, read from the node-keyed learner through
  frozen membership, net of the no-revaluation control. **Gated floors:** the matched-class floors (the
  same node-keyed learner keyed on identity-similarity nodes and on sensed-reading nodes, same freeze and
  Stage-2 booking; §6), which can win. `tabular_q` and `random` on the same state keys stay as BF2's
  declared floors, with exploration declared in the prereg, labelled as zero by construction on this
  readout (§6). **Reported comparators:** the
  no-learning sensed-similarity baseline (object-block cosine, no fitted map), an identity-only (word
  prior) arm, a yoked no-predictor arm, the per-item twin (which on tried items is the mechanism, §5), α
  per pair with experience equalised, a situation-block ablation, and a block-weight/λ sensitivity row
  (confounding lens N4). **Authoring rule:** the physics author is blind to the arm design, and every
  consequence, O3 included, must follow from each object's own `self_effect`/`target_effect` declaration,
  with no per-experiment tuning. The blind author now also writes items whose readings and effects
  disagree, so the rule no longer guarantees the sensed/consequence confound it used to strengthen.
- **The untried-item arm (new, G21). Credits the `ConsequencePredictor`.** Items never acted on, under the
  full "untried" definition below, probed at first contact after training on others. The predictor's
  zero-shot prediction for them, through the Q5 seam (which acts on untried candidates only), must beat the
  **per-item twin** (S1 Cerebellum means + S4 clustering + the same Stage-2 booking, no predictor), and, as
  BF2 requires of any new substrate claim, `tabular_q` and `random` on the same state keys. A twin match is
  `PARTIAL (narrow: floor matched)`, stated in advance.
- **The acquired-distinctiveness arm (confounding lens S1; kept by G27).** T9's "blocks it between affordances whose names collide" half otherwise has no gated
  test: the AE names point neither way, and the jet triad and S4's separation check are reported only. Two
  items with the same affordance name, near-identical identity blocks, equal sensed readings, and differing
  consequences. After experience, value learned on one must not transfer to the other. **It is gated, not
  reported** (pre-merge review, Architecture S3; strict): §6's AD paragraph gives the readout (Δ_D2
  through frozen membership), the statistic (below each matched-class floor's Δ_D2 by the margin, and
  within an equivalence bound of the no-revaluation control), the floors (the matched-class floors gated;
  the per-item twin, `tabular_q` and `random` reported, all zero-transfer by construction) and the PARTIAL
  rule. It runs on the same forced-schedule driver as the AE arm. The shipped `false_hearth`/`fire_pit`
  `warm_self` pair and the census's same-name flips are templates (new YAMLs, never edits).
- **Controls the AE arm needs** (confounding lens S6): a **no-revaluation control** (Stage 2 retrains A to
  its own O1, or there is no Stage 2, with the same elapsed passes and session boundaries), so drift,
  causal-link decay, consolidation, refits and the newly running `cleanup_session` cannot pass as
  inheritance; items **rotated through the A/B/C roles** across replicates, with which outcome is O1 vs O2
  counterbalanced. **The unit of replication is the authored triple**, and n is stated in triples: the
  predictor is closed-form and the schedule scripted, so repeated seeds on one triple are not independent.
- **The Stage-2 manipulation is named in the prereg** (bio lens SF-3, environment lens S6, confounding lens
  N3). An item's consequence *is* its YAML, so "retrain A to O3" is a physics change. The strict form is a
  contingency switch declared in A's own YAML before the run (a declared staged effect, authored blind with
  the set), so it is the object's physics, not tuning. A drive-state lever is acceptable only on a drive A
  alone affects (otherwise it is outcome devaluation, a different phenomenon). Implementation candidates:
  a resumed session (`--resume-sim`) under a second arc whose A-named item carries O3, the swap declared an
  apparatus event (G16), with a preflight refusing colliding entity names (scene activation is additive,
  `generative_runner.py::_activate_phase_entities`, and `tools/registry.py::ToolRegistry.register` silently
  overwrites by name). A resume carries hippocampus, NAc, EC and ATL only until S3's resume edit lands, so
  the Cerebellum, the predictor, the node-keyed learner and the membership map reach Stage 2 only through
  S3's resume edit and S4's contract (both load seams). If the scripted driver keeps one process across
  stages, the prereg says so and the session boundary is still the real `on_session_end`. B's and C's
  declarations and the body states at test are confirmed identical across stages.
- **Artifact routes removed** (G21; §6): frozen pre-state band before every trial, B and C probed in the
  same state, interleaved counterbalanced order, equal Stage-1 counts, session and refit boundaries pinned
  identically across arms. The reset write is stamped `apparatus` and forms no pair. The trial that follows
  is the AUT's own experienced act from a harness-set pre-state (G26, narrowed by the pre-merge review,
  Executor S11; the stamping is above). G23's "never apparatus resets as training pre-states" is S0b's
  rule, and the prereg restates the scope against G23.
- **The jet triad (§6), a control, not the test:** it measures generalisation from sensed features. It
  needs new AUT-invoked entities (§6); its result is reported, not gated, and "`water_jet` must not inherit
  the aversion" is no longer a criterion.
- **Minecraft, game-native physics: repositioned honestly.** Cradle consequences are YAML, so cradle
  results may only recover the author's regularities; in Minecraft the game supplies the consequences. But
  this arm, as designed, tests **consequence prediction, not transfer**: it is a graded, action-contrastive
  prediction of Δoxygen/Δhealth within a situation band for `escape_water` against a same-band control
  (S0b), the roadmap's "how far is pain" readout and, under S0a branch (ii), the bridge to E3, gated against
  its own floor, the B1 mechanism twin (TD state value on the cluster + the same prior; confounding lens
  S9). It is **not evidence for T9's transfer claim**. A game-native transfer contrast is **not available
  on today's bridge** (environment lens S7): the avatar has no object-directed contact act, the sensors carry
  no per-object property, `damage` has no source, and food cannot drain natively in campaign time; any
  contrast (magma vs campfire, berry bush vs cactus) needs new bridge affordances and sensors, a protocol
  change that fires the re-run trigger on four EARNED/PARTIAL rows (`scripted_water.py`'s note). Strict
  default: no contrast. It depends on the out-of-band producer (the post-return refill is outside the
  tool-path record) and on S0b's surfaced-pool apparatus, and runs only if S0b's Minecraft half passed. The
  cradle arm therefore carries T9 alone, and risk 1 (§9) stays open for it.

**The AE readout does not go through the Q5 seam** (G21, bio lens D1 bullet 4). The strict seam re-orders
or annotates **untried** candidates only, and B and C are tried in Stage 1 by design, so the seam never
touches them. S5's acquired-equivalence measure is therefore the node-keyed learner's recorded value
through frozen membership, which has no selection effect. A behavioural B-vs-C choice readout
(confounding lens N2: "first B-vs-C choice after Stage 2", defined against the Stage-1 choice baseline; it
is not "first contact") would need a selection seam acting on **tried** items, which Q5 does not cover and
which would reach the earned tables the strict seam protects. **Decided by G27 (Q12: none):** no such
seam, the choice is not measured, and T9 claims learned value carried from A to B as learner state, with
no choice claim. The untried-item arm is the one that reaches selection, through the Q5 seam.

**Held-out ("untried") items.** This definition governs the zero-shot readouts (the untried-item arm, the
jet-triad control) and the prior term. In the acquired-equivalence test every item is experienced by
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
`_DRIVE_TOOL_AFFINITIES` and `_DRIVE_CORRECTIVE_NEEDS` over the **full tool name** (recommended; it rules
out `warm_self` as an AE affordance, confounding lens S8); (b) every drive held at or below 0.5 at
the probe; or (c) Component 3 counted as part of the word-prior baseline and the gate read on the recorded
`components`, net of `drive`. GL2b(i)'s keyword list (its new `"heat"` row) is frozen before the fixtures
are named.

**Gate.** Pre-registered, per arm, with margins fixed in the prereg. A recorded null ships as a null (T9).
The AE gate is §6's (Δ_B − Δ_C per dimension above a margin, interval excluding 0, net of the
no-revaluation control, beating the matched-class floors; `tabular_q` and `random` run as BF2's declared
floors, zero by construction on this readout). The AD gate is §6's AD paragraph. The untried-item gate
beats the per-item twin, `tabular_q` and `random`. **A floor match on any gated arm takes
`PARTIAL (narrow: floor matched)`**, stated in the prereg in advance (BF2). Gated dimensions follow §4.1 (`nociception` only after the caused/felt split).
The gate reads **experienced-only** pairs, per dimension, on one target schema id; the with-narrated
result is reported beside it and never gated (§4.1; G6 not reopened), and a result that holds only with
narrated pairs is reported as such. Apparatus outcomes are excluded (G16).
**On EARNED, the row is an owner decision at S5 start** (pre-merge review, Architecture S2): which row, its
tier, and whether it supersedes T1-5. After G27 the claim is learner state with no claim about choice. A
`[behavioral]` invariant needs measurable behavioural weight, and T1-5 was a behavioural transfer claim.
**The strict default, offered first:** a learner-state result cannot by itself graduate a `[behavioral]`
invariant or supersede T1-5. It earns a Tier-3 mechanism row, or a row carrying an explicit
`(narrow: learner state)` qualifier, with `Re-run on:` and `Regression guard:`. The row cites T1-5
(DROPPED by GL0, 2026-10-08, its ID kept) in its history, and a `SUPERSEDED by` link from T1-5 is made only
if the owner rules the result carries T1-5's claim. It is never a re-label of T1-5. A Tier-3 seed row at
`SETUP` may be added when the prereg merges.

**Guards.** The harness inherits S3–S4's guards, plus M10's fingerprint extended with the grounding flags
(grounding.md), so no E-rung arm runs with them set.

**Blast radius: the credit write.** The node-keyed learner (S4) is a credit surface, and its Stage-2
booking is what the AE arm measures; its walk is S4's (§7, S4; §8). It carries no selection effect in S5
(Q12 decided: no tried-item seam, G27).

**Blast radius: the selection term.** The untried-item arm's "untried-tool prior" is a selection effect,
and it applies **only to untried tools** (the full definition above), so experienced tools keep their earned
tables. The Component 3 keyword path is named here as part of the selection surface the arm runs on. Where it is placed decides what fires:
- **Inside `NAc.recommend_action` (`decisions/nac.py::NAc.recommend_action`).** This fires the
  "`recommend_action` change" trigger **by wording, even while default-OFF**, on **T1-7, T1-11, T1-12,
  T1-13, T1-14, T1-15** (Exp 45/56/57/60/61/62), and it sits inside the decomposition fence (the fence lifts when 1.3.6's decomposition slices land — the point at which T0 can pass).
- **The strict alternative (recommended):** a composing step **before** `recommend_action`, which re-orders
  or annotates the untried candidates it is handed and leaves the function untouched. No trigger fires by
  wording. A structural walk is still recorded on each of those rows, showing selection byte-identical with
  the arm off. Whether this "fires by effect" is the Architecture lens's question, and the walk is the
  answer.

Making the prior a default, non-experimental consumer is **GL6**, with its wide re-runs, and is not this
plan's.

**Owner decisions at S5 start (asked together):** Q5 (seam vs term, for the untried-item arm); the
acquired-equivalence item set (the readout is decided, G21 and G27; Q12 and the acquired-distinctiveness arm
are decided by G27: no tried-item seam, keep the arm); the Stage-2 manipulation; the keyword-path
control (a, b or c above); the capture runtime per arm (the scripted forced-schedule driver for the AE and
AD arms, above); whether a game-native transfer contrast is added to the Minecraft arm (strict: no;
environment lens S7 prices it); the arms, n (in triples) and margins, including the margin over each
matched-class floor (the floors themselves are not asked: they bind now, G30) and the AD arm's
equivalence bound; the ledger row this would earn, its tier and
whether it supersedes T1-5 (strict: a Tier-3 mechanism row or a `(narrow: learner state)` row; it never
supersedes T1-5 by itself); whether T9's claim waits on E3's recorded outcome (G1: never co-headlined).

---

## 8. Blast radius by stage

| Stage | Rows whose `Re-run on:` fires | How it is discharged |
|---|---|---|
| S0a, S0b, S2 | none | no `src/` |
| S1 | **T1-16 by wording** ("the memory record shape": `extra["context"]` is flattened into the persisted trace). T1-1 (Exp 10) carries the older schema wording but is SUPERSEDED. The observe's move into the executor (G22) changes where the Cerebellum trains: it stops where no record is minted, and starts on tools generated without `cerebellum=` (the generative cradle scene tools, DM `SceneState`). The tool-level `cerebellum=` argument is removed (S1). No earned row's `Re-run on:` names either | the same ruling GL2a's `extra["interoception"]` gets (one ruling for both keys); structural evidence: the ranker reads no `encoding` key, the round-trip is lossless; goldens byte-identical; `ec.json` byte-identical (no hot-path embed); the executing survival checks green; the executors that lose Cerebellum training and the tool sets that gain it, both recorded in the PR; the D79 pins re-pointed; the `swallow_moves.json` record |
| #1449 (its own Session-A slice, not a GL4 stage) | a walk on T1-13 ("temporal-anchor pruning"), since `cleanup_session` starts running on the AUT's distributor on `--sim`; the `--sim` rows | its own teardown test (adopted and `/new` branches) and deletion probe; the walk lists earned rows whose evidence ran on `--sim` (owner ruling if any) |
| S3 | the predictor: none by wording. The resume edit: `--resume-sim` newly restores the Cerebellum and the predictor; **T1-16 by wording** (RESUME_STORES / the resume path) | goldens byte-identical with the shadow log on; the walk lists earned rows whose evidence used `--resume-sim` (owner ruling if any); T4a Dormant marking if S5 has not earned by 1.4.0 |
| S4 | T1-3; EC world-modality triggers on T1-6, T1-10–T1-15 (structural); **T1-14 by wording** (the `bundle.py` export filter); **T1-11, T1-12 by wording** if the merge-side handling edits `merge.py`; **the node-keyed learner's credit surface** (G21): the NAc credit-path rows if it lives on NAc, a structural walk if not; the NAc persisted shape if the learner lives on NAc; T1-16 if a new file joins `RESUME_STORES` | structural, via the byte-identical golden with the flag off, the hivemind pin untouched; the by-wording rows take the owner's S4-start ruling (strict default: guards re-run plus a batched rig re-run); the learner's walk is owed at S4 start with its home. **Touch list** (pre-merge review, Architecture B1, S9, Wire SF-8, Executor N2): the new membership map and its freeze trigger (S5 harness caller only); the learner's table; the EC-only encode door with its M39 allowlist entry; `AGENT_STATE_FILES`, `cli_utils._AGENT_FILES` / `MEMORY_PATHS` / the D2 groups; both load seams; `hivemind/bundle.py` (the export filter, the NAc-field guard) |
| S5 (strict seam) | none by wording; structural walk on T1-7, T1-11–T1-15 | byte-identical with the arm off |
| S5 (term inside `recommend_action`) | **T1-7, T1-11, T1-12, T1-13, T1-14, T1-15 by wording** | re-runs budgeted into S5, rig slot after E3 (Track C ordering) |

Not touched by any stage: `credit_node` **except along the node-keyed learner's path in S4, if it lives on
NAc** (G21 lifts this exclusion for that path only; §5's answer holds the lift unused), `NAc._reward_bias`
(rejected as the learner's home, §5), `TemporalCreditDistributor`'s credit path (#1449's seam only adds an existing `cleanup_session` call
to the `--sim` AUT's path), the shared EC encode path
(S4 adds its own EC-only function), `_sensor_embed`/`gain_exponent` (Exp 62), `SensorEncoder` (Exp 56;
S4 does not route through `encode_sensors`), `_drive_potential_diff` /
`drive_comfort_progress` (T1-9), the affordance decomposer (T1-5), and the `substrate_merge` fold (Exp 56).
The hivemind is touched once, in S4: the `bundle.py` export filter and the merge-side handling of the
`"consequence"` modality (Exp 61's T1-14 wording, above). The positive Reaction that lapses T1-13's #888 discharge is GL2c's, in
[autonomic_layer.md](autonomic_layer.md). This plan adds no Reaction.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Circular authored physics.** Cradle "transfer" may only recover the YAML author's regularities | High (confounding lens) | Blind physics author; consequences derived from each object's own declaration. The Minecraft arm is game-native but, as designed, tests consequence prediction, not transfer (§7, S5), so it does **not** close this risk for T9 unless a game-native transfer contrast is added; stated in the result |
| **Sensed similarity passes as consequence similarity.** Authored items' sensed readings and effects move together, so object-block cosine alone reproduces a "reversal" | Critical (confounding lens DNB-2) | G13: acquired-equivalence items whose readings and consequences disagree, on one shared sensor schema with no empty object block; the sensed-similarity baseline reported beside every result; the jet triad demoted to a control |
| **The architecture has no path for the effect it tests**: a fixed-kernel ridge, a re-derived node and the α blend all predict C ≥ B, so a positive is an artifact (α density, schedule leakage, refit timing) and a null is uninformative | Critical (T8 (b) bio lens D1, confounding lens D1) | G21: the mediating path designed in S4 (frozen experienced-outcome nodes + a node-keyed learner + a membership readout); the composed-loop toy before the prereg; the artifact routes removed (§6) |
| **A floor that could win is never run**, and the gate beats a signed-wrong baseline | Critical (confounding lens D2; pre-merge review, Architecture S1) | G21, strict reading: matched-class floors that could win (the same node-keyed learner keyed on identity-similarity nodes and on sensed-reading nodes) gate the AE and AD arms; `tabular_q` and `random` run as BF2's declared floors and are labelled zero by construction on a learner-state readout; the per-item twin is the mechanism on tried items and the floor on untried ones; the statistic is Δ_B − Δ_C with an interval excluding 0; `PARTIAL (narrow: floor matched)` stated in advance |
| **The membership freeze holds only in the harness** (no production caller) | Medium (pre-merge review, Architecture B1) | The freeze is tiered experimenter control; the claim is stated for the harness configuration with its declared Stage-1 boundary; T4a Dormant marking at 1.4.0 if S5 has not earned |
| **The world saturates the body**, so one campaign's records sit at range edges and S0b fails as a world artefact | High (environment lens D1) | G23: N short fresh-body sessions, order permuted; N and the floors derived from the YAML deltas and `drift_step` before capture |
| **The object block is ground truth no receptor delivered** | High (environment lens D2) | G25: the block is what a receptor delivers, absent when nothing was sensed; S2 gates only on keys with a block |
| **The predictor's file is an unbounded second store** | Medium (wiring lens S5, environment lens S8) | Store contracts, a cap on n with eviction, a file-size budget; measured on named machines |
| **Level terms let the predictor copy context** | High (bio lens SF-3) | Change-only target projection; a context-copy baseline (situation block only) and a key-mean baseline in S2 and S5 |
| **The word prior's keyword path** (`recommend_action` Component 3) produces a null at first contact | Medium (confounding lens SF-6) | Keyword-free fixture names checked by a test, or a held drive level, or a components-net gate (§7, S5) |
| **Apparatus writes trained or credited as experience** (rescue, respawn, teacher, mother) | High (environment lens DNB-2) | G16: `apparatus` provenance, never a pair, never credit; S0b counts them |
| Contamination by curated pairs, or narrated consequences passing as experienced | Critical (fails the thesis) | Join invariant on the minted `PhysicalEventId` + required provenance + CI test (narrated never relabelled, discount always applied); every gate reads experienced-only data, with-narrated reported beside it; no manual surface |
| The narrator's world dominates training (the cradle's consequences are largely narrator-written) | High | The declared discount (G6); the with/without report; S0b counts narrated writes by tool before S2 |
| Too little data, or overfitting on hundreds of pairs | High | S0b's hard floors; strong ridge regularisation; held out by entity; a shuffled-target control |
| A non-deterministic join key (the executor's uuid4) leaks into persistence | High (silent) | `PhysicalEventId` only; the two-process test fails on uuid- or dict-order-dependent output |
| Word-prior geometry differs on a Pi (384-d fallback) | Medium | Per-block geometry tags; refuse on mismatch; train and evaluate on one declared encoder |
| The dormancy rule on the Cerebellum read side | Process | S1 touches only the write. The read resurrects only through S0a branch (i) + E7, or S5, with `test_cerebellum_dormant_909.py` edited in the same PR |
| The replay buffer is lossy (queue drops, consolidation pruning) | Medium | S0b counts drops; the persisted training set keeps `pid`s and vectors at each refit |
| Magnitude is invisible to cosine | Medium | Nodes carry kind; magnitude is read from the code; S5's O1/O2/O3 differ in direction |
| "Nothing happened" forms a node (native [0, 1] ranges, or a centred standardisation) | High (bio lens SF-1, confounding lens S7) | Centred [−1, 1] ranges, raw unstandardised encode, scaling without centring; a null-is-zero guard (S4) |
| Most affordances (363 of 405, by the YAML walk) have no declared consequence | Medium | Stated, never imputed: those keys have no predicted consequence, and α stays at the prior for them |
| Two may-fail lines in one release | Medium | T9 conditional, never co-headlined with E3 (G1); grounding flags excluded from E-arms (M10) |
| Divergence across S2 iterations | Process | One pre-registered MLP retry at most, and only on a FAIL of S2's capability gate (i); a FAIL on the dissociation keys or on (ii) alone is recorded and does not stop or divert the line (G25, confounding lens D3). Two divergent iterations → stop and audit the layer beneath (the autonomic target and the capture), per the cycle-divergence rule |

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

**Decided 2026-10-24 (T8 (b) design-review decisions; DECISIONS.md G21–G27)** (the review: §12; not
re-opened here; G24 is GL3's, recorded in [thalamic_relay.md](thalamic_relay.md)):
- **G21 (bio D1, confounding D1 + D2; S4, S5, §6, T9). Strict: the mediating path is designed in S4.**
  (1) The consequence representation is formed from each item's **experienced** Stage-1 outcomes, with
  membership frozen per item at the end of Stage 1 (the decision named the EC frozen-centroid mechanism;
  *the pre-merge review found that it freezes prototypes only, so membership is carried by a new per-item
  map and a harness-only freeze trigger, §4.2 item 5, §5. That corrects the mechanism the decision named,
  not the decision*). (2) A fast
  outcome/value learner keyed on the consequence node, updated when a member item is acted on in Stage 2;
  §5 says why `cluster_reward_bias` keyed by (cluster, tool) cannot carry it. (3) B's Stage-2 readout is read
  through node membership. This lifts §8's `credit_node` exclusion for that path. **Credit is split** (after
  it was shown that the per-item-outcome twin *is* the mechanism on tried items): the acquired-equivalence
  gate (per dimension Δ_B − Δ_C, Stage-1 end → Stage-2 end, above a pre-registered margin with its interval
  excluding 0) credits **only the consequence representation** and must beat `tabular_q` and `random` on
  the same state keys (BF2; *under the pre-merge review's strict reading, Architecture S1, both are zero by
  construction on the learner-state readout, so matched-class floors that could win also gate, §6*); the
  sensed-similarity baseline is reported, not the gate; a separate
  **untried-item arm** credits the `ConsequencePredictor`, which must beat the per-item twin there. A floor
  match on either → `PARTIAL (narrow: floor matched)`, stated in advance. The `ConsequencePredictor` keeps
  the zero-shot-to-untried job. Also: a composed-loop toy before S5's prereg
  (`docs/plans/rationale/latent-forward-model/s5_toy.md`); the artifact routes removed; one shared sensor
  schema, no empty object block, cos_obj / cos_id tabulated per triple; the AE readout is not via the Q5
  seam (§7, S5).
- **G22 (bio D2, wiring D1; S1, §4.2 item 2, Q6). Strict:** the Cerebellum learns the **signed sensory
  reafference on the sufferer body** (the declared body sensors' change, net of declared drift, caused
  entries only: the record's `drive_delta` plus raw deltas for sensors without a declared drive), beside
  the existing object-readings model under its existing key (add, not replace); alliesthesia is applied at
  read time. `valence`, `drive_pain`, `urgency` and `deviation_after` are dropped from the Cerebellum
  target; the `cerebellum.py` Bio-mapping docstring is S1 work. Q6's "no situation in the key" is safe and
  decided. The observe call and the `get_confidence` → `sim_cerebellum` line move out of
  `tool_bridge.py::ModulatorAffordanceTool.execute` into the executor (`Executor._stamp_invocation`, after
  `Executor._interoceptive_outcome`), and `test_cerebellum_wiring.py`'s AST pin is re-pointed.
- **G23 (environment D1; S0b). Strict:** shipped bodies only; "one campaign" becomes N short fresh-body
  sessions with scene and name order permuted; the prereg computes, from the YAML deltas and `drift_step`,
  the expected non-edge records per key and band before capture, and if too few, restates the floors; never
  a capture-only body, never apparatus resets as training pre-states. Q9's per-body key list follows.
- **G25 (everything else, strict):** the remaining DO-NOT-BUILDs fold as the reviewers proposed.
  Confounding D3: S2 re-scoped as capability (within-key held-out invocation blocks vs the context-copy
  baseline, standardisation and λ fitted on training folds only; matched-model-class ablations on keys with
  a non-empty object block; the sensed-baseline comparison reported, not gated; an S2 dissociation FAIL
  alone does not stop the line). Wiring D2: S3 names the `--sim` AUT's real session-end seam, fenced in
  `start_simulation_mode` / `run_agentic_loop` after 1.3.6; today the AUT never reaches
  `BioStack.on_session_end` and `--resume-sim` restores neither the Cerebellum nor the predictor (follow-up
  issue #1449, "--sim AUT never reaches BioStack.on_session_end"). Wiring D3: S1 records raw inputs, and S2/S3
  embed them offline with geometry tags (no hot-path embed, no `ec.json` side effect). Environment D2: the
  object block is what a receptor delivers to the AUT, marked absent when nothing was sensed; S2 gates only
  on keys with a block; the Minecraft object block is absent; "reads alike" = shared sensor names with near
  values under `_sensor_embed`. Bio SF-1 / wiring S7: centred [−1, 1] ranges, so "nothing happens" is the
  zero vector; the A4 gain documented as wrong at N ≤ 8.
- **G26 (asked after the fold; S0b, S5). G23 is S0b's; S5's per-trial reset is a control.** S5 resets the
  body to a frozen pre-state band before every trial, as G21's artifact-route fold requires. The reset is
  stamped `apparatus` (G16) and never trains the predictor. G23's "never apparatus resets as training
  pre-states" governs the S0b capture, not S5's control; the S5 prereg restates this. *Wording narrowed by
  the pre-merge review (Executor S11): the reset **forms no training pair**. The trial after it is the AUT's
  own experienced act from a harness-set pre-state, and its pair does train (§7 S5).*
- **G27 (asked after the fold; S5, T9). The acquired-equivalence readout is learner state.** There is no
  selection seam on tried items (**Q12 decided: none**). The AE gate reads the node-keyed learner's value
  through node membership, so T9 claims that the consequence representation carries learned value from A to
  B (items that share consequences while their names and readings differ), measured as learner state. It
  makes no claim about choice. The "blocks it where names collide" half is carried by the
  acquired-distinctiveness arm (§7 S5), so that arm is kept and T9 is not re-worded to one half. *The
  pre-merge review (Architecture S3) gives that arm its gate (§6), since a kept half with no gate would
  not be claimed.*

**Open, each asked at the start of the stage named (recommendations in bold are the strict option where
one is offered):**

| # | Decision | Ask at | Recommendation |
|---|---|---|---|
| Q3 | Imagined-entity pairs: exclude, or down-weight ×0.5 (mirroring `decay_imagined_links`)? | S3 start | **Exclude** |
| Q4 | The consequence modality and the hivemind: local-only with a separate per-modality freeze field, or update the frozen-modality pin now? (Option: the `IDENTITY_DOMAIN_MARKER` never-exported path, wiring lens N5) | S4 start | **Local-only; the pin is untouched**; receiver-side refusal either way |
| Q5 | S5's selection effect for the untried-item arm: a composing step before `recommend_action` (no trigger by wording), or a term inside it (fires T1-7, T1-11–T1-15)? | S5 start | **The composing step** |
| Q6 | ~~Cerebellum key on the TARGET entity? Add the situation cluster to the key?~~ | — | **Decided by G22** (2026-10-24): no key change (owner = target on every shipped item; wiring lens S3); no situation in the key |
| Q7 | Does S0b's capture include an LLM-driven cradle arm? | S0b start | **No** (LLM-driven *selection*). About 10× fewer invocations and a selection-bias confound. The named runtime still has an LLM **narrator** (environment lens S1): empty phase instructions, named narrator box, narrated-write sentinel; substrate-primary selection + Minecraft only |
| — | The narrated discount's value (G6) | GL4 start (with S0a) | **A small discount**, pinned before S2's data; every S2 and S5 result reported with and without narrated pairs |
| Q8 | How does a narrated write that adjudicates the AUT's own action pair with that action's context (separate `pid`s)? The writers exist (§3.6) | S0b start | Settled in S0b's prereg after the narrated-write count; never a join on anything but `pid`; stamped `narrated` before any pair is built |
| — | Minecraft capture: a rig slot in Track C, or the offline scripted bridge? | S0b start | The rig, in a slot never stacked on an E-rung campaign |
| — | Kernel and block weights | S2 start | Cosine kernel per block, fixed weights pinned before S0b's data, ridge λ by nested CV on training folds only |
| Q9 | S0b's capture scene, body and N (the runtime is now named: `maxim --sim --arc <yaml> --aut-mode substrate-primary`; the fixture-orchestrator candidate is struck, environment lens S1; the shape is G23's) | GL4 start (with S0a) | **Name scene and body before the capture**, tabulate the non-null keys per body (G23), derive N from the YAML/`drift_step` arithmetic |
| — | Minecraft apparatus: a new surfaced pool with no teleport, or a window ending at the first harness write? | S0b start | **The surfaced pool** (environment lens S2) |
| — | Caused-only nociception in the target | S2 start | **Caused only**, felt-only masked, gated only after the caused/felt split (wiring lens S2) |
| Q10 | Is S3's contamination guard GL3.B1's landing consumer (G8), or is that left to S5? | S3 start | Stated at S3 start (wiring lens S8); no recommendation yet |
| Q11 | The node-keyed outcome learner's home. Options: (a) the existing `NAc._reward_bias` (keyed `(agent_id, node_id)`, written by `credit_node`), the ride-existing option; (b) a **new** NAc table keyed on the node id, signed and per dimension; (c) a node → outcome running mean on the Cerebellum's delta rule | S4 start | **(a) is rejected by the front-gate** (§5: a non-negative scalar clamped to `[0, 0.20]`, so an aversive O3 cannot be stored; it widens the node's recognition radius and so moves membership; it is ablated, decayed and bundled). Between (b) and (c): S4's front-gate answer, with its walk; no recommendation yet (bio lens D1 names both) |
| Q12 | ~~A selection seam acting on **tried** items, for a behavioural B-vs-C choice readout in the AE arm?~~ | — | **Decided by G27** (2026-10-24): none; the AE gate reads the learner through membership, and T9 makes no choice claim |
| — | ~~Keep the acquired-distinctiveness arm, or re-word T9 to the half the design tests?~~ | — | **Decided by G27** (2026-10-24): keep the arm (confounding lens S1); it carries T9's "blocks it where names collide" half |
| — | The source of the "sensed before the decision" flag for the object block | S1 start | Named at S1 start; recorded absent until it exists |
| — | T1-5's status (decided at GL0 start) | GL0 | **DROPPED** (strict). Non-strict alternative: `PARTIAL <date> (narrow: compound-name level)`, whose reason is to keep the ID live for the eventual `SUPERSEDED by` link from S5's new row |
| — | S5's arms, n, margins (incl. over each matched-class floor, and the AD equivalence bound) and the ledger row it would earn, its tier and whether it supersedes T1-5 | S5 start | **A learner-state result does not by itself graduate a `[behavioral]` invariant or supersede T1-5**: a Tier-3 mechanism row, or a row with `(narrow: learner state)`; never a re-label of T1-5 (pre-merge review, Architecture S2) |
| — | The matched-class floors as the strict reading of G21 (gate against the same learner keyed on identity-similarity and sensed-reading nodes) | **Decided 2026-10-24 (owner; DECISIONS.md G30)** | **Confirmed as binding floors of the T9/AE gate now** (and of the AD arm, §6); only their margins are set at S5 start (the row below). Proposed by the pre-merge review, Architecture S1; `tabular_q`/`random` stay as BF2's declared floors |

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
- **T8 (b) fold (2026-10-24), made by the main session, not in this file:** DECISIONS.md G21–G27;
  grounding.md's GL4 and GL5 rows (and its banner's T8 (b) status); roadmap_1_4.md's T9 wording;
  thalamic_relay.md §GL3.B2 and §8 (this plan's side of the handoff is §4.1); the confounding lens's toy
  copied to `docs/plans/reviews/grounding_gl4/lfm_toy.py`; the follow-up issue #1449 ("--sim AUT never
  reaches BioStack.on_session_end"); the wiring lens's proposed
  [docs/wiring/session-end-seams-per-runtime.md](../wiring/session-end-seams-per-runtime.md) entry.
- **Pre-merge review fold (2026-10-24), owed by the main session in the same branch** (§12, "Pre-merge
  review round"):
  - grounding.md: GL4 S4 and GL5 rows (the EC freezes prototypes only; the membership map and harness-only
    freeze trigger; matched-class floors; the AD gate; the softened EARNED wording; #1449 as its own slice
    that S3 depends on); the G1 exempt set naming `Executor._outcome_start` (Wire N5) and Architecture N1's
    "consequent on G22/G25; moot" wording; the GL3 row's GL3.B2 clause (one pid per tool-path invocation,
    minted at `_run_started`, reused by H3t; Executor S8, Wire SF-9, B3).
  - roadmap_1_4.md: T9 and the GL5 row (the matched-class floors, the AD arm's gate, the untried-item arm's
    own floors (Executor N7), and EARNED softened to a learner-state row decided at S5 start).
  - DECISIONS.md: a dated note under G21 (the frozen-centroid correction; the matched-class floors as G21's
    strict reading, to be told to the owner), G26 (narrowed to "the reset forms no training pair") and G27
    (the AD arm gated).
  - docs/wiring/body-and-word-worlds.md (Architecture S4): its GL5 sentence ("that reversal is the
    falsifiable signature the grounding line tests") and item 6 ("the jet ordering inverts … tested in
    GL5") still state the pre-G13/G21 claim. They become AE through consequence nodes, measured as learner
    state, with the jet triad a control, plus a change-log line. Its §3 step 5 ("trains through
    `tool_bridge.py`") is S1's edit set (Wire N1).
  - docs/wiring/session-end-seams-per-runtime.md: the narrator and console HANDLE rows become "Cerebellum
    saved via `save_cerebellum()`; `on_session_end` / `cleanup_session`: No". Add `api.py` /
    `create.agent()` and the adopted-HANDLE branch (Executor S9, Wire SF-1, SF-2).
  - #1449's body: it is its own Session-A slice after the fence, with the adopted and `/new` branches named
    (Architecture S6, Wire SF-2).

---

## 12. Review record

### The T8 (b) four-lens design review (2026-10-24)

The four lenses read this plan at `d9dffe6e` (worktree `.worktrees/e-t8b`):
[confounding](reviews/grounding_gl4/confounding.md) (3 DO-NOT-BUILD / 9 SHOULD-FIX / 6 NIT),
[bio-faithful](reviews/grounding_gl4/bio-faithful.md) (2 / 8 / 4),
[wiring](reviews/grounding_gl4/wiring.md) (3 / 8 / 6) and
[environment](reviews/grounding_gl4/environment.md) (2 / 8 / 6). The confounding lens's numpy toy (a fixed
cosine-kernel ridge in dual form on the §6 A/B/C design) is committed at
[reviews/grounding_gl4/lfm_toy.py](reviews/grounding_gl4/lfm_toy.py). It is a toy, not the composed loop;
the composed-loop toy S5 owes is `docs/plans/rationale/latent-forward-model/s5_toy.md`. Owner decisions
G21–G25 settle the ten DO-NOT-BUILDs; each is folded where it lives (G26 and G27, asked after the fold, settle the
S5 reset's scope and the AE readout, §10):

| DO-NOT-BUILD | Resolved by | Where |
|---|---|---|
| Bio D1: no part of the design can produce acquired equivalence; every named readout predicts C ≥ B | G21 | §1 (mechanism); §4.2 items 5–6; §5 (the node-keyed learner row; the twin); §6; §7 S4, S5; §8; §9; §4.3 |
| Confounding D1: the named readouts cannot produce B > C, so a positive is an artifact (α density, schedule leakage, refit timing) | G21 | §4.2 items 5–6; §6 (readout named, toy, artifact routes removed); §7 S5 |
| Confounding D2: no reasonable floor that could win; the gate is set against a signed-wrong baseline | G21 (credit split) | §5 ("the mechanism twin"); §6 (gate statistic, item set); §7 S5 (arms, gate); §9 |
| Confounding D3: S2's held-out dissociation gate has no admissible route to PASS | G25 | §5 (input-shape check); §7 S2; §9 (divergence row) |
| Bio D2: the Cerebellum's "corrected" target is an evaluative, need-gated code with level terms | G22 | §3.1; §4.2 item 2; §4.3; §7 S0a, S1; §10 Q6 |
| Wiring D1: the consequence does not exist at the tool's observe call site | G22 | header file set; §3.1; §4.2 item 2; §4.4; §7 S1; §8 |
| Wiring D2: the cradle seam edits the orchestrator's instance; the `--sim` AUT never reaches `on_session_end`; `--resume-sim` restores neither file | G25 | header; §4.2 item 3; §5; §7 table, S3; §8 |
| Wiring D3: two of the four `ActionContext` blocks cannot be produced at the assembly site without a fenced edit or an EC side effect | G25 | §4.1 (raw inputs; the dataclass); §4.2 item 1; §7 S1 |
| Environment D1: the cradle body saturates and selection concentrates, so one campaign cannot reach the band-clean floors | G23 | §7 S0b (a); §9; §10 Q9 |
| Environment D2: the `object` block has no game-native source in either world | G25 | §4.1 (what the object block is); §5; §6 (item set); §7 S0b, S2 |

**SHOULD-FIX dispositions.** Folded, except where a reason is given:
- **Confounding** S1 (the name-collision half of T9 untested): folded as the acquired-distinctiveness arm in
  §7 S5, with re-wording T9 kept as the owner's alternative at S5 start (since settled by G27: the arm is kept). S2 (collision definition and the
  census statistic): folded in §6 and S0b. S3 (pair counts not independent): folded in S0b. S4 (who chooses
  the actions): **partly folded**: the forced Latin-square schedule with a body reset before each invocation
  is refused by G23 (no apparatus resets as training pre-states) and by the explore-first gate, so S0b
  takes the lens's own fallback (per-key n and band coverage stratified by first-burn time) plus G23's
  cross-session permutation. S5 (context-copy ill-defined; standardisation leak): folded in §4.1 and S2. S6
  (no-revaluation control, role rotation, triple as unit): folded in §7 S5. S7 (centred standardisation
  makes "nothing" a direction): folded in §4.1 and §4.2 item 5. S8 (numeric name neutrality, full tool
  name): folded in §6 and S5. S9 (the Minecraft contrast cannot fail): folded in S0b and S0a (ii).
- **Bio-faithful** SF-1 (null is not zero on [0, 1]; A4 at small N): folded (G25) in §4.2 item 5, S4. SF-2
  (outcome identity; state-independent nodes): folded in §4.2 item 5, S4. SF-3 (Stage-2 manipulation):
  folded in §6 and S5. SF-4 (predictions written as nodes): folded in §4.2 item 5 (nodes from experienced
  outcomes, G21). SF-5 (α is a count, not precision): folded by setting α from the ridge's predictive
  variance (§4.2 item 6), so §1's "precision weighting" now holds. SF-6 (CLS decorative): folded in §4.2
  item 3, §4.3, S3. SF-7 (bio criterion in S0a; Q6 to wait for it): first half folded in S0a; the second
  half is **superseded by G22**, which decides Q6 under the reafference target. SF-8 (oracle attributions in
  the tier table): folded in §4.3.
- **Wiring** S1 (projection not in the shipped record; per-affordance drive block): folded in §3.3 and
  §4.1. S2 (caused-only nociception before the split): folded in §4.1, §7 table, S0b, S2, §10 (strict (a)).
  S3 (assemble in the executor; `entity_path := CauseRef.entity`; `tools/base.py`): folded in the header,
  §4.1, §4.2 items 1–2, S1, Q6. S4 (`load_safe` set-aside-and-overwrite): folded in §3.1, S1, §4.4. S5 (the
  predictor is a store): folded in §4.2 item 4 and S3. S6 (session-end order; shadow log site): folded in
  §4.2 items 3 and 7, S3. S7 (encode door credits and activates; ranges): folded (G25) in §4.2 item 5, S4,
  §4.4. S8 (GL3 handoff): folded on this plan's side with strict option (a) in §4.1; the matching
  thalamic_relay.md edit is that plan's, and whether S3 carries GL3.B1 is Q10.
- **Environment** S1 (the fixture orchestrator cannot activate cradle items): folded in S0b and Q7/Q9. S2
  (the refill happens after an apparatus teleport): folded in S0b (strict (a)) and §10. S3 (bands by learner
  state; no-op control has no pid; `turn` acts underwater): folded in S0b. S4 (nociception structurally zero
  in the cradle; caused vs felt): folded in §4.1 and S0b. S5 (identity geometry and leaky names): folded in
  S0b. S6 (no mechanism to retrain A): folded in S5's Stage-2 manipulation. S7 (no game-native transfer
  contrast on today's bridge): folded in S5 (strict default: none). S8 (dual-form ridge stores its training
  set; n grows): folded in §4.2 item 4, S2, S3.

**NIT dispositions.** All folded: confounding N1 (S4 separation on held-out blocks), N2 ("first B-vs-C
choice", §7 S5), N3 (O3 authored blind, §6), N4 (pin weights and λ before S0b's data; sensitivity row,
§4.3, S2, S5), N5 (hold out by twin group, S2), N6 (mask felt-only, §4.1); bio N1 (§3.1 title and framing),
N2 (Gluck & Myers; Coutureau, §1), N3 (trial-1 copy, §3.1, S0a), N4 (reafference without attenuation,
§4.2 item 2, S1); wiring N1 (§3 re-dated, GL2a merged), N2 (why the payload `"version"` moves, S1), N3
(two provenance fields, §4.1), N4 (no pid in the Cerebellum payload, §3.3, S1), N5 (the identity-domain
path, a Q4 option), N6 (T4a covers S4; Rule T on the pinned functions, S1, S3); environment N1 (GL2a merged,
S0b), N2 (`turn` acts underwater, S0b), N3 (band edges in sensed units, S0b), N4 (Reachy out of scope,
S0b), N5 ("computable consequence", S0b), N6 (queue drops per session, S0b).

**Literature** cited by the bio-faithful lens (Gluck & Myers 1993; Coutureau et al. 2002; Wolpert, Miall &
Kawato 1998; Imamizu et al. 2000; Buhusi & Meck 2005; Blakemore, Wolpert & Frith 1998; Johnson & Raye 1981;
McClelland, McNaughton & O'Reilly 1995) is from the reviewer's own knowledge and was **UNVERIFIED** in that
session.

### Pre-merge review round (2026-10-24)

The three pre-merge lenses (Executor, Architecture, Wire integrity; docs/CODE_REVIEW.md) read the staged
T8 (b) fold against `d9dffe6e`. Their reports cover both plans and the record edits. **Counts for this
plan's findings:**
- Architecture: 1 BLOCKER, 6 SHOULD-FIX, 2 NIT (of 1 / 12 / 5 overall).
- Executor: 0 BLOCKER, 2 SHOULD-FIX, 3 NIT (of 2 / 11 / 7).
- Wire integrity: 1 BLOCKER, 4 SHOULD-FIX, 3 NIT (of 3 / 9 / 7).

The rest land on thalamic_relay.md, autonomic_layer.md, grounding.md, the roadmaps, outstanding.md or
docs/wiring/, and are folded or owed there (§11).

| Finding | Where folded |
|---|---|
| **Architecture B1 = Executor S10** (BLOCKER): the frozen-centroid mechanism freezes prototypes, not item membership; the map and the freeze trigger are unlisted; the freeze is mis-tiered; `NAc._reward_bias` is not answered | §3.4; §4.2 item 5 (parts 1 (a) map and rule, (b) harness-only trigger; part 2 the two rejected tables); §4.3 (rule = innate prior; trigger = experimenter control); §4.4 (map deletion probe); §5 (rows for the map, the trigger and `_reward_bias`, with the four reasons; read at `d9dffe6e`, no restriction rescues it, so the strict choice is a new table); §7 S3 (T4a list), S4 (What, Guards); §8 (S4 touch list); §9 (new risk row); §10 G21 note, Q11 (option (a) added and rejected) |
| **Wire B3** (BLOCKER; GL4 side): the pid seam contradicts itself across plans, and H3t runs before the pid exists | §4.1 GL3.B2 handoff: one pid minted at `Executor._run_started` before `tool.run`, reused by `_stamp_invocation` and by GL3.B3's H3t emit; the gate wording restated; header file set. GL3/autonomic wording is the other plans' |
| Architecture S1 (+ Executor N1): BF2 floors zero by construction | §6 (matched-class floors; BF2 floors labelled; what a PASS means; heading renamed); §7 S4 (floors built), S5 (AE arm, Gate); §9; §10 G21 note and new row (the strict reading of G21, for the owner) |
| Architecture S2: EARNED overstates a learner-state claim | §7 S5 Gate ("On EARNED" rewritten; strict default first); §10 last rows |
| Architecture S3: the AD arm has no gate | §6 (AD gate paragraph: readout, statistic, floors, PARTIAL rule); §7 S5 (AD bullet, Gate); §10 G27 note |
| Architecture S6: #1449 tied to the forward model's chain | header; §4.2 item 3; §7 table (#1449 row), S3 (What, Gate, Blast radius); §8 |
| Architecture S9 / Wire SF-8: no persistence / resume / bundle contract | §7 S1 (`Cerebellum._reset_store_state`), S3 (registries land with the file; RESUME_STORES and T1-16), S4 (the contract: format, store contracts, both load seams, `AGENT_STATE_FILES`, `_AGENT_FILES` / `MEMORY_PATHS` / D2, `scrub_nac_state_for_bundle`); §4.4; §8. *Corrected on reading:* `scrub_nac_state_for_bundle` already ends with the `_BUNDLE_NAC_FIELDS` allowlist, so a new NAc field is dropped at export. The contract is that the field is never allowlisted, pinned by deletion |
| Architecture S11 + Executor S11: a forced schedule the `--sim` runtime cannot produce; reset stamping; G26 wording | §6 (artifact routes); §7 S5 (the scripted driver through the real loop on the `_cradle_loop_driver.py` pattern, justified as the composed loop; reset stamping through GL3's slot and window; the pair's provenance), Artifact-routes bullet, Stage-2 manipulation; §10 G26 note |
| Wire SF-1 / SF-2: `AgentInstance.shutdown` never reaches `on_session_end`; HANDLE branch; `/new` recursion | §4.2 item 3 (per-runtime list; both branches); §5 row; §7 S3 |
| Wire SF-3: the move turns training on for tools built without a Cerebellum; fate of `cerebellum=` and the D79 pins | §3.1 (rows corrected for the generative path); §4.2 item 2; §7 S1 (the argument removed; D79 pins re-pointed; gain-site test); §8. *Corrected on reading:* `cerebellum_modulator.py`'s `cerebellum=` goes to `CerebellumModulator`, not to a tool, so it stays |
| Executor N2: S4's encode door vs the M39 lint | §4.2 item 5; §7 S4; §8 |
| Executor N7 (second half) / Wire N2: `swallow_moves.json` | §7 S1 |
| Wire N1: claim surfaces saying the Cerebellum trains in `tool_bridge.py` | §7 S1 (edit set) |
| Wire N5: `_outcome_start` vs `_run_started` | header file set (the grounding.md half is owed, §11) |
| Architecture N2: the unflagged-exception list | header |
| Architecture N4: Session V coordination | §7 S1 (the GL3 stages' half is thalamic_relay.md's) |

**Not folded here, with the reason:**
- Architecture S4 (body-and-word-worlds.md states the old GL5 claim) is a docs/wiring/ edit, owed by the
  main session (§11).
- Executor N7's first half (roadmap T9 omits the untried-item arm's floors) is roadmap_1_4.md's, owed (§11).
- Executor S8, Wire SF-9 and Wire N5's grounding.md half are owed by the main session (§11).
- Executor S9 (the session-end wiring table rows) is a docs/wiring/ edit, owed (§11).

The three folds that read code were checked at `d9dffe6e`: the frozen-centroid branch, `credit_node`'s
clamp and `get_threshold_overrides`, and the bundle allowlist. Two corrections to the reviewers' facts
came out of that check: the bundle allowlist, and `cerebellum_modulator.py`. Those two are noted in the
table above.

**Delta round (2026-10-24).** The same three lenses read the fold's unstaged delta only (stop rule):
Executor 0 BLOCKER / 6 SHOULD-FIX / 6 NIT, Architecture 0 / 5 / 5, Wire integrity 0 / 4 / 5, all folded
(most land on thalamic_relay.md). Owner decisions G28 (#1176 unflagged), G29 (no `unknown` kind) and G30
settle the decision-shaped findings. On this plan: the matched-class floors are **binding floors of the
T9/AE gate now**, with only their margins set at S5 start (G30; §6, §7 S5 asks, §10); S5 depends on
GL3.B1 and its reset goes through the kind-taking setter, with the window statement made in
thalamic_relay.md §5.7 item 1 (Architecture SF-3, Executor N-3; §7 table and S5); the floor learners are
tiered as an experimenter control and on the T4a Dormant list, with their membership analogue and M39
route (Architecture N2, Executor N-4; §4.3, §6, §7 S3); the scripted driver replaces the proposer with
`unittest.mock.patch.object` at the one module attribute, no `src/` seam (Architecture N4; §7 S5); the S4
contract covers `maxim substrate prune` (Wire N-4; §7 S4); the `/new` recursion under adoption
re-adopts the persistent agent (Wire N-1; §4.2 item 3). **No further round (owner, stop rule;
converged).**
