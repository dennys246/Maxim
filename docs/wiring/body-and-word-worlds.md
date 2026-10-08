# The body world and the word world: two disconnected chains

**Established:** 2026-10-07, from the five-angle audit on
[#1120](https://github.com/dennys246/Maxim/issues/1120) and the four grounding planning reads that
followed it (thalamic contract, autonomic layer, afferent tracks, action-conditioned prediction).
Every claim was read in `src/` at `f1c833ab`, not run, unless it cites a measurement. **UNVERIFIED**
marks a claim taken from the audit that no planning read re-checked in code. **Update this page**
whenever a mechanism it names changes, an issue it links closes, or a ledger row it cites moves. The
tables in §1 and §2 are the one-glance answer and must not drift from the sections under them.

**What the page is for.** Anyone designing a mechanism that would let a word, a percept or a
learned value "mean" a body consequence reads this first. The short version: Maxim has two
representational worlds. Only one of them acts without the LLM, no EARNED ledger row depends on the
other (the word, EC `text`, path), and nothing joins the two.

Fix plan: [docs/plans/grounding.md](../plans/grounding.md). Mental model of the substrate chain:
[docs/agents/bio-memory.md](../agents/bio-memory.md) §1 (whose single-chain diagram this page
corrects). Sibling page: [engram-formation.md](engram-formation.md).

## 1. The two worlds at a glance

```
BODY WORLD (live by default; the only path that acts without the LLM)
  sensor reading ─► SensorEncoder._sensor_embed (384-d hashed sums; A4 gain on `world`)
                 ─► EC sensor modality (interoception / audio / world) @ 0.85, frozen centroid
                 ─► NAc cluster_fear (Wire 4) / cluster_reward_bias
                 ─► NAc.recommend_action  (substrate-primary only)  ─► the body acts

WORD WORLD (only with MAXIM_SUBSTRATE_PATH=1; affordance NAMES only from the --sim orchestrator)
  text / affordance NAME ─► LinguisticEncoder (paraphrase-mpnet 768-d; decomposer per surface)
                         ─► EC `text` @ 0.44, running-mean centroid
                         ─► ATL concept by NAME ─► reward_bias / prompt annotation ─► the LLM reads it

                      no edge joins the two chains (§3)
```

| Modality | Encoder / space | Threshold, centroid | Producer(s) | Default? | Acts without the LLM? | Earned rows on it |
|---|---|---|---|---|---|---|
| `interoception` | `SensorEncoder`, 384-d hashed sum, ungained | 0.85, frozen | `runtime/substrate_proposal.py::_read_drive_states` (+ the derived `cold` need) | live | yes (substrate-primary) | T1-6 (Exp 42), T1-9 (Exp 52) |
| `audio` | `SensorEncoder`, 384-d, ungained | 0.85, frozen | `embodiment/audio_localization.py::DoAFeed` (live Reachy), `AzimuthDoASource` (sim) | live | yes | T1-7 (orient 45), T1-10 (Exp 53) |
| `world` | `SensorEncoder`, 384-d, **gained** (A4 p=3.0) | 0.85, frozen | `embodiment/backends/minecraft.py::MinecraftWorldBackend.sync_world_sensors` → `world_set_axis` | live | yes | T1-11 (Exp 56), T1-13 (Exp 60), T1-14 (Exp 61), T1-15 (Exp 62); T1-12 (Exp 57) is PARTIAL, not EARNED |
| `text` | `LinguisticEncoder`, paraphrase-mpnet 768-d (384-d bag-of-words fallback without sentence-transformers) | 0.44, running mean | percepts on **any runtime**: `agents/memory_agent.py` → `integration/memory_hub.py::MemoryHub.on_percept_received` → `LinguisticEncoder.encode`; affordance NAMES only via `imagination/trigger.py::encode_entity_affordances` (only production caller: `simulation/orchestrator.py`) | `MAXIM_SUBSTRATE_PATH=1` only | no: prompt text and a `tool:*` nudge that is 0 on every recorded run (§5) | none (T1-5 is PARTIAL; §5) |
| `vision` | `LinguisticEncoder` over the percept's **text content**, under a `vision` tag (code-read) | 0.44 | `agents/perception_agent.py` (DN / robot), through the same `on_percept_received` encode | robot path, and `MAXIM_SUBSTRATE_PATH=1` only | no | none |

Notes on the table:
- `ECConfig.frozen_centroid_modalities` defaults to `{"interoception", "audio", "world"}`
  (`similarity/ec.py::ECConfig`; `world` was added in 1.1.4 PR 1). `bio-memory.md`'s EC table still
  lists two.
- **The same reading can enter twice.** Minecraft `health`/`food`/`oxygen` carry both a `modality:
  world` declaration and a `drive:` block (`_data/components/bodies/minecraft_player.yaml`), so one
  sensor reading enters both the world channel and the interoception channel.
- **Interoception is represented twice, unlinked.** A sim pain injection
  (`simulation/conversational_source.py::inject_pain` → `make_intero_percept`) maps through
  `agents/modality.py::_SUBSTRATE_MAP` (`INTEROCEPTION → "text"`) and embeds **as language**, while
  the body's drives embed through `SensorEncoder`. Nothing joins the two.
- **Three modality vocabularies** coexist: `SensoryModality` (7 values), `Percept.modality`
  (`text`/`vision`/`audio`/`intero`) and open EC tag strings, joined by the lossy `_SUBSTRATE_MAP`
  (SOUND/TOUCH/SMELL/INTEROCEPTION → `text`).
- **Affordance concepts are the embedding of the affordance NAME.**
  `similarity/decomposer.py`'s `AffordanceDecompositionStrategy` encodes the compound plus its
  components. The entity is not part of what gets encoded.

## 2. Every would-be bridge between the worlds

A store is a mechanism only when it is **written → keyed → read → acts**
([CODE_REVIEW.md](../CODE_REVIEW.md)). Scored that way, no cross-modal link exists.

| Would-be link | Written? | Keyed on | Read? | Acts? | Status |
|---|---|---|---|---|---|
| Hebbian episode binding (`memory/episode.py::apply_hebbian_on_close`) | no on the production percept path (one node stashed per percept) | node pair | — | — | **Dormant** since 2026-08-29 (D6); T3-7 DORMANT |
| Graph path in enrichment (`integration/bio_enrichment.py::BioEnrichmentPipeline._query_hippocampus_traced`, docstring "the associative 'fire → pain' path") | walks `Hippocampus._binding_graph`, which does not grow (D6) | — | yes | returns nothing in production | **inert**; docstring overclaims |
| `memory/hippocampus.py::Hippocampus.retrieve_cross_modal` | n/a | — | **no `src/` caller** (only `scripts/p4_*.py`) | — | uncalled, no dormancy marker yet |
| Concept-decomposition Stage-2 relation metadata (`memory/episode.py`, `memory/hippocampus.py` "Concept decomposition Stage 2") | accumulated | node pair | **UNVERIFIED** that any reader exists (audit: "unread") | no | unread |
| Naming events (`embodiment/naming_events.py`) | drive-utterance co-firing scaffold | — | — | — | **Dormant** since 2026-05-29 |
| ATL situation link (`memory/concept_extractor.py::ConceptExtractor._link_situation`, 2S-b) | yes | trace ↔ situation-cluster concept | recall only | no | live, but situation concepts are kept out of co-occurrence ("No inline relationships are formed for situation concepts") |
| `archive/cross_modal_substrate_binding.md` | — | — | — | — | ARCHIVED, "DO NOT RESURRECT" |
| `deferred/jepa_cross_modal_alignment.md` (384 ↔ 768 projection) | — | — | — | — | SUBSUMED 2026-10-07 by `docs/plans/latent_forward_model.md`; zero code |
| Cerebellum forward model (`embodiment/cerebellum.py::Cerebellum`) | yes, every SEM affordance call | exact `ModelKey(entity_path, modulator, affordance, param_bucket)` | read side **Dormant** (#909) | no | live write, wrong target (§4) |

**Provenance is not on any EC/ATL/NAc write.** An imagined affordance
(`imagination/trigger.py::ImaginationTrigger._encode_entity_affordances`) and the agent's own body
affordance (`encode_entity_affordances`) produce the same node, and
`similarity/encoder.py::LinguisticEncoder.encode_decomposed` calls `NAc.update_eligibility`, so credit
can land on it. `imagined` exists only on `ComponentRegistry`, `ImaginationResult`, `Episode.imagined`
and CausalLinks tagged retroactively at session end (`NAc.tag_imagined_links`). The only per-node
"source" is `EntorhinalCortex._substrate_node_sources`, which records hivemind origin, not
experienced/imagined. (`bio-memory.md`'s gotcha "LLM-imagined entities skip substrate encoding" is
false: they are encoded, unflagged.)

## 3. One burn, traced: fire → pain, and where it stops

The cradle fire pit is the cleanest case: the name says fire, the physics says burn, and the two
never meet.

| Step | Where | What happens | Reaches learning? |
|---|---|---|---|
| 1. Act | `_data/components/items/cradle_fire_pit.yaml::touch` | writes `arms.thermal +0.6` on the infant | — |
| 2. Transduce | `embodiment/body.py::Embodiment.evaluate_failures` → `_publish_drive_pain` | infant `arms.thermal`: range `[-1,1]`, homeostatic, `comfort_band 0.5`, `pain_scale 0.4` (`infant_humanoid.yaml`) ⇒ max drive pain **0.2**; from rest the +0.6 touch gives **(0.6 − 0.5)·0.4 = 0.04** | — |
| 3. Classify | `proprioception/pain.py::classify_pain`, `failure_pain_kind` | `source="drive:arms.thermal"` is not in `TISSUE_DAMAGE_DRIVES` ⇒ `PainKind.DRIVE`, **not nociceptive** | — |
| 4a. Pain memory | `create_pain_memory_subscriber` (threshold 0.4) | below threshold | **no** |
| 4b. Wire 2 percept valence | `create_percept_valence_subscriber` (0.3) | below threshold; would key on the **sufferer** (`entity_name` = the infant's body), not the fire pit | **no** |
| 4c. NAc pain outcome | `create_pain_nac_subscriber` (0.3) | below threshold | **no** |
| 4d. Wire 4 cluster fear | `create_pain_cluster_fear_subscriber` (0.3 + allowlist `{drive:health, drive:oxygen}`) | below threshold and not allowlisted | **no** |
| 4e. Felt pain on the record | `ToolOutput.pain` (2S-c) | NOCICEPTIVE only; this is DRIVE | **no** |
| 4f. Channel 1 | `bridges/tool_pain_bridge.py::ToolPainBridge.record_tool_embodiment_failure` (no threshold) | `nac.record_outcome(..., NEGATIVE)` on `tool:<name>`, intensity-blind | **yes**: the action-level causal link learns "this tool hurt" |
| 4g. ReactionBus | `runtime/bio_stack.py::_distribute_reward_from_reaction` (no threshold) | `credit_node(-0.04)`, which the ≥ 0 clamp turns into "erode a positive bias or do nothing" | effectively no |
| 5. Forward model | `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` → `Cerebellum.observe_from_action` | learns the **fire pit's own** `heat_output`/`fuel`, read before the self_effect reaches the infant; the burn never reaches it | **no** |
| 6. Word side | `encode_entity_affordances` (substrate path only) | `fire breath`/`touch` encode by name; `fire` and `breath` never get their own nodes (§5) | — |
| 7. Join | the graph path (§2) | would link a fire concept to pain; the binding graph does not grow (D6) | **no** |

**Where it stops:** at step 3 the burn is classed as discomfort; at step 4 every PainBus learner is
above it; at step 5 the predictor watches the wrong entity; at step 7 there is no edge. The only
learning that survives is a negative causal link on the tool name (4f). The cradle text in
`docs/simulation.md` ("all three layers converge on … PainBus → NAc") is true of where these three
layers publish and false of the learning, for this body. (**UNVERIFIED** in a run; the arithmetic and
thresholds are verified in code.)

**A second burn path, outside the learning trace.** The keyword reflex `thermal_contact`
(`_data/reflexes/infant.yaml`) fires on narrative text ("touch fire", "searing heat", …), not on the
sensor, and dispatches `damage_component` on the arms. That pain is narrator-driven (`narrated` under owner decision G6, DECISIONS.md 2026-10-07).
`DamageComponentTool.execute` does call `evaluate_failures`, but it writes component integrity, not
`arms.thermal`, so it skips step 2's `arms.thermal` transduction, and nothing in the table above records
it as the fire pit's consequence.

**Pain ingress is not one convergence point.** Besides `evaluate_failures` → `_publish_drive_pain` /
`_publish_pain`, these producers publish pain or a pain Reaction without passing it:
- `simulation/tools.py::DamageComponentTool` builds a `PainSignal` directly (the narrator's tool, and
  the `thermal_contact` reflex above);
- `runtime/sim_adapter.py` (`next_observation`): a `proprioception`/`pain_signal` sim percept becomes a
  `Reaction` on the ReactionBus;
- `simulation/sandbox.py::PainTriggerLayer` (sensitive file access);
- `runtime/pain_interceptor.py` and `proprioception/perceived_pain.py` (assessed / perceived pain,
  published as Reactions);
- `bridges/pain_bridge.py::PainCircuitBridge` (DN, robot).
`api.py` also exposes a public `pain_signal` subscription (a consumer, listed so the census is
complete; it observes only `PainSignal`s on the agent bus, so the Reactions put straight onto the
ReactionBus above never reach it). Any claim of the form "all pain converges at one point" is false;
GL3.B0's census inventories them.

## 4. The word world's learning loop has four breaks

| # | Break | Evidence |
|---|---|---|
| 1 | **Outcomes do not land on concept nodes.** NAc outcome records key on `tool:<name>`, sensor clusters and goals. The one path onto a text node is eligibility → `TemporalCreditDistributor.distribute` → `NAc.credit_node`, paid only by Reactions. | audit; `decisions/temporal_credit.py`, `decisions/nac.py::NAc.credit_node` |
| 2 | **No positive producer.** Every live Reaction is `kind="pain"`, `Valence.NEGATIVE` (`reactions/compat.py::pain_signal_to_reaction`). The only positive constructor, `embodiment/backends/cerebellum_modulator.py::_emit_success_reaction`, has no production caller and no `reaction_bus`. `_reward_bias` is clamped `[0, max]`, so pain can only erase a bias. The satiation Reaction `embodiment/sem.py::EntropicDriveSpec` promises was never built: `evaluate_failures` detects the crossing (`elif cleared: breach_latch.pop(...)`) and discards it; `ReactionKind` reserves `"satiation"` with no producer. | `reactions/types.py::ReactionKind`; T1-13's own #888 discharge text |
| 3 | **Annotators look concepts up by exact name.** `_annotate_affordance_valence` calls `atl.recall(name=chunk.text)`. A component close to its compound has no node, a paraphrase merged under another node's name is invisible, and shared words (`jet`) link unrelated affordances. | `integration/bio_enrichment.py::_annotate_affordance_valence` |
| 4 | **No selection reader of concept bias.** `recommend_action` reads `reward_bias` only for `tool:*` keys; concept-node bias reaches the LLM prompt and EC threshold widening only. Because of break 2, the `tool:*` nudge is **0 on every recorded run**: T1-13's annotation reports the scored `reward_bias` term at 0.0 in all 108 decisions in `exp61_pairs` + `exp62_rows`. | `decisions/nac.py::NAc.recommend_action` |

Positive learning does exist, on other surfaces: the cluster-keyed motor credit
(`tool_bridge.py::_drive_potential_diff` → `drive_comfort_progress` → `NAc.update_cluster_reward`,
EARNED in T1-6/T1-7/T1-9) and positive causal links from tool success. Neither touches a concept
node.

## 5. Names versus consequences: measurements

**Name collisions on shipped components** (audit #1120; counts re-verified 2026-10-07 by a walk of
every `affordances:` block under `_data/components/`):
- `touch` ×16 share one node: a blanket (safe), the fire pit (burn) and a sharp rock (cut) all
  complete into the same `touch`.
- `warm_self` ×14, in safe and harmful variants, shares one node.
- `turn_left` and `turn_right` collapse into one node.
- Only **42 of 405** shipped affordances declare `self_effect`/`target_effect`, so most (entity,
  affordance) keys have no body consequence that anything could learn.

**mpnet cosines on compound names** (`paraphrase-mpnet-base-v2`, one offline probe 2026-10-07 on the
operator's installed sentence-transformers; a measurement, not a gate):

| Pair | Cosine | At the 0.44 text threshold |
|---|---|---|
| fire breath ↔ flame jet | 0.601 | bind |
| fire breath ↔ water jet | 0.298 | separate |
| **flame jet ↔ water jet** | **0.664** | **bind, and closer than flame jet ↔ fire breath** |
| fire ↔ flame | 0.785 | bind (the T1-5 PoC number) |
| touch blanket ↔ touch fire pit | 0.469 | bind |
| turn left ↔ turn right | 0.874 | bind |
| escape water ↔ flee | 0.568 | bind |

Under the word prior, an aversion learned on `flame_jet` leaks **more** to `water_jet` than to
`fire_breath`. A consequence-grounded representation has to reverse that ordering; that reversal is
the falsifiable signature the grounding line tests (GL5).

**What #1120 established about the transfer test** (real encoder, 2026-10-07):
1. The negative controls looked up `atl.recall(name="water")` by exact name and missed `water jet`,
   which does form. In the dragon-only scenario EC does not merge water into fire
   (`water jet` ~ `fire breath` = 0.298).
2. Both controls' `reward_bias >= 0` assertions cannot fail (#910).
3. False transfer appears once a mage is encoded first: at the retired **0.40**, running-mean drift
   pulls everything into `fire breath`; at **0.44**, `water jet` lands on the mage's `jet` node
   (0.785).
4. **Components close to their compound (~0.73) never get their own node**, at 0.40 or 0.44
   (`fire` ~ `breath` 0.297 to each other). There is no component path.
5. IT-1's positives hold at 0.44, but at the compound level (`flame jet` → `fire breath`, 0.601).

So T1-5 ("Affordance concept transfer", PARTIAL 2026-06-15) is name similarity at the compound
level; its behavioural claim was never measured, and its Regression guard cites a lesson that left
CLAUDE.md in the 2026-08-13 diet, a string splitter, and the never-run
`docs/experiments/temporal_credit_validation.md`.

**E4's text-drift measurement** (2026-10-06, `engram_formation.md` E4,
`docs/experiments/data/e4_text_widening_drift/diagnosis.json`): NO COLLAPSE, but the 0.44 text
radius **already spans concepts**: thermal and texture pairs share nodes at bias 0. Its widening
overreach set (13 strings) is the required input for whoever builds the first positive text-credit
producer.

## 6. The body is under-wired too

| Defect | Evidence | Status |
|---|---|---|
| Infant burn is too weak and the wrong kind (max 0.2, DRIVE, under every learning threshold) | §3 | verified in code; issue to file |
| **Heat yields no corrective need** (cold does). `sem.py::corrective_need_intensity` returns a value only for homeostatic deficits below `set_point − comfort_band` and for entropic drives; above the set point it returns `None`. `substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS` maps `temp`/`thermal` → `"cold"` only | docstring: "the 1.3 survival-loop scope (break 1)" | latent: no shipped experiment measures an overheated agent (**UNVERIFIED** that none does) |
| **Wire-2 percept valence keyed on the SUFFERER, not the cause.** `create_percept_valence_subscriber` keys `(agent, entity_name or entity_type, failure_mode)`; `body.py::_publish_pain` sets `entity_name` to the body node that owns the failure. `simulation/tools.py` builds `side_effects["actor_invocation"] = {"source_entity", "source_affordance", ...}` and drops it | the NAc docstring example ("a dragon that burned the agent once…") cannot be produced by any `body.py` path | verified |
| **LLM-primary never calls `NAc.note_active_clusters`** (only caller: `runtime/substrate_proposal.py`), so Wire 4 books no situation fear there | audit | re-read before treating as a defect; not the autonomic plan's: GL3 or its own defect issue |
| **SCN drive events Dormant** (`body.py::_emit_drive_temporal_event`, D9) | docstring | Dormant |
| `EntropicDriveSpec.coupled_to` / `HomeostaticDriveSpec.modulated_by` parsed, never read | both docstrings: "1.0 interface, deferred" | reserved slots, not a defect today |
| `SensoryTag.perceived_intensity` / `modulated_by` ("set by SensoryGate"): no `SensoryGate` class exists; nothing sets the field | grep | dead field |
| Out-of-band body change (world writes, the Minecraft sync, an actor hitting the AUT) produces no consequence record; `drive_pressure_before`/`drive_relief`/`pain` ride `ToolOutput` only. Relief that the agent did not cause with a tool reaches nothing except `cradle_mother.reactive_mother_tick` (Exp 52). Harm is recorded as 0.0 relief (`sem.py::relief_fraction_from_progress` keeps the positive part) | `runtime/executor.py::Executor._stamp_invocation` | verified |
| **Affordance credit is blind to modulator drives**, and Exp 42's discrimination depends on that blindness | [#1161](https://github.com/dennys246/Maxim/issues/1161) | open |
| `anticipatory_pre_activate` Dormant since 2026-05-26 (`decisions/temporal_credit.py::TemporalCreditDistributor`), zero `src/` callers; `bio-memory.md`'s SCN invariant says it runs every tick | T3-4 DORMANT | docs overclaim |

## 7. Timing: fast signals wait on slow ones

`runtime/agent_loop.py::run_agentic_loop` is one thread: one percept per pass, execute the pending
proposal (§4), substrate tick (§6b), decay (§8.5), sleep (§9). Clocks: wall time (loop period,
substrate cadence, drift, the PainBus and ReactionBus refractories), the experience clock (integer
µs, read only by memory strength today), pass count (NAc decay) and Hippocampus `capture_seq`.
`time/temporal_event.py::TemporalEvent`'s producers mint `uuid4` ids
(`bridges/tool_pain_bridge.py`, `embodiment/body.py::_emit_drive_temporal_event`), and its signatures are
wall time: neither is deterministic.

| # | Finding | Evidence |
|---|---|---|
| L1 | **Substrate-primary acts on stale state.** §6b sets `ctrl.pending_proposal` at the end of pass N; §4 executes it in pass N+1 after the sleep with no re-evaluation. `loop_state.py::_substrate_tick_due` requires `pending_proposal is None`, so `evaluate_failures` cannot run in between. `MinecraftClient.call_action` blocks, so during a held primitive no pain is evaluated. Roadmap budget: "0.58 s tick + its blocking hold" per primitive. | worst-case breach→pain and breach→protective-action latencies in passes are **UNVERIFIED as measured numbers** |
| L2 | **The turn budget defers pain.** When `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` denies a tick (`runtime/loop_substrate.py::substrate_tick`, `_substrate_gate_denied`), `propose_via_substrate` is skipped, and with it `note_active_clusters` and `evaluate_failures`. Drift catches up lazily; for **drift-driven** breaches pain publishes late, stamped late, which misaligns eligibility (narrator writes call `evaluate_failures` immediately, so they are not delayed). A slow signal (the narrator turn) gates the fastest one. | code-read |
| L3 | **LLM-primary cannot be preempted.** `WorkerPool.cancel_pending` drains only queued jobs; a pending LLM proposal is not invalidated by pain; `_drop_stale_proposal` drops only proposals older than 35 s; `_wait_for_proposal` blocks the loop thread up to 300 s per cycle, with no live tick, no percept intake and no decay. The only live preemption is a non-follow-up CLI input cancelling a `multi_step`/`fallback` proposal: triggered by text, not by the body. | code-read |
| L4 | **Sim percepts are one FIFO, one per pass.** `ConversationalSource._queue` is a `deque`; `CompositePerceptSource.next_percept` takes the first non-None child. A pain percept waits behind earlier text. Priority was deliberately deferred (`archive/thalamus_relay_design_pass.md`). | code-read |
| L5 | **Innate reflexes sit behind the cortical gate.** `BioEnrichmentPipeline.enrich` (which runs `_evaluate_reflexes`) runs only if `runtime/thought_gate.py::ThoughtGate.should_think` passes: refractory 2 passes, ≥ 0.15 of the LLM **token** budget, non-empty working memory, and a learned threshold that `goal_reward_bias` moves. A rejected percept is already popped, so its reflex never fires. That inverts the biology. | code-read; frequency in a campaign **UNVERIFIED** |
| L6 | `default_network/gate.py::ThalamicGate` is DN-only and vision-shaped; DN is skipped in sim; `DefaultNetwork.add_escalation_callback` has no caller. | grep |
| L7 | **Dead preemption scaffolding.** `runtime/preemption.py::PreemptionCircuit` ("Pain is the first registered source") is constructed nowhere in `src/`, `tests/` or `scripts/`; nothing publishes a `PreemptionSignal` or calls `MaximAgent.wire_preemption`; it survives the orphan lint only through the `runtime/__init__.py` re-export. `agent_loop.py` §3's `hasattr(agent.goal, "check_hold")` branch is dead (no `check_hold` in `src/`). `tool_dispatch.py`'s `_execution_tracker.capture_before` guard can never be true. Recommended disposition: a `Dormant since` marker (dormancy over deletion; removal touches the `maxim.runtime` re-exports). | grep |

L1, L2, L5 and L7 are **to file as defect issues** (owner decision G8, 2026-10-07); each is fixable
without the grounding line's registry.

What does already bypass slow processing: tool-coupled pain inside `execute` (`ToolPainBridge`,
T1-4); same-tick fear inside `propose_via_substrate` (encode → `note_active_clusters` →
`evaluate_failures` → `anticipatory_threat_need` → `recommend_action`, load-bearing since Exp 58
W-4); the sim-only audio orienting reflex; DN behaviours on the live robot.

Cross-thread notes:
- The `mc-sync-<agent>` pump (`simulation/minecraft_harness.py::MinecraftSyncPump._run`) writes
  `vital_metrics` while the loop reads them in `propose_via_substrate`, with no snapshot lock (each
  per-key write is GIL-atomic). It never calls `evaluate_failures`. Whether one encode has ever mixed
  two snapshots is **UNVERIFIED**.
- In `--sim`, `evaluate_failures` has a second caller thread. `simulation/orchestrator.py` registers
  `OrchestratorActorTool`, `DamageComponentTool` and `SetEntitySensorTool` on the orchestrator's
  registry (`orch_registry`); they run inside the orchestrator agent's `run_agentic_loop` on the
  orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop) and
  call `_aut_embodiment.evaluate_failures()` (or the embodiment's) while the AUT loop runs on
  `sim.aut`. (`sim.dm` exists only for interactive DM campaigns and never touches these tools; the
  reflex dispatch's separate instances of the same classes run inside the AUT's `enrich`, on the loop
  thread.) Both threads mutate `Entity.drive_breach_severity` (the breach latch) with no lock. The
  grounding line declares this an edge (a lock from GL2a, an inbox from GL3.B3).

## 8. What depends on what: earned rows and blast radius

**No EARNED row depends on the word (EC `text`) path** (#1120). Most ride the body world's sensor
chain; T1-16 (Exp 63) is Hippocampus recall into the prompt and T1-4 rides PainBus /
`ToolPainBridge`. So the cost of grounding lands where that shared code is touched:

| Touching | Fires by its Re-run-on wording |
|---|---|
| `TemporalCreditDistributor` / `credit_node` write path; any positive Reaction emitter | T1-13 (Exp 60): its #888 discharge says "If a positive Reaction emitter is ever wired, this reasoning lapses and the trigger applies"; T1-14/T1-15 by inheritance; T1-14 also functionally (the staged-donor sanity assumes `_reward_bias == {}`) |
| `recommend_action` / `cluster_reward_bias` / the drive-activation floor | T1-7, T1-11, T1-12 (PARTIAL), T1-13 (and T1-14/15) |
| `SensorEncoder` / `_sensor_embed` / `_encode_current_clusters` / EC world modality | T1-6, T1-10, T1-11, T1-12 (PARTIAL), T1-15 (and T1-13/14) |
| Any positive event when a drive's breach latch clears (`minecraft_player` `oxygen`/`health` clear on every surfacing after a latched breach; `food` and the `d1` of `minecraft_bench` and `minecraft_bench57` are entropic) | T1-11, T1-12, T1-13, T1-14, T1-15, whatever the routing (owner decision G7; T1-12 added 2026-10-08) |
| `EncodingSignals.extra` on the loop capture (flattened into the persisted trace by `EncodingSignals.to_dict`) | T1-16 ("the memory record shape"); Exp 10's memory-shape wording |
| EC persistence (`ec.json`) / bundle manifest / `_BUNDLE_EC_NODE_FIELDS` | T1-10, T1-14 |
| Shared EC scan / threshold / centroid rule | T1-3, T1-10, T1-15 |
| `infant_humanoid` (every infant variant `extends:` it: `_chilled`, `_cold`, `_naming_v1`, `infant_operant*`) | T1-6, T1-9/T1-10, T3-9 (T3-6, T3-10 and T3-15 are DROPPED and carry no `Re-run on:`) |
| PainBus / ReactionBus / NAc reward pipeline | T1-4, T3-9 |
| `run_agentic_loop` idle gate or autonomy handling | T1-13 (and T1-14/15) |
| Affordance decomposition | T1-5 |

Rows whose stated rationale cites a dead path: T3-19 (`_reward_bias` clamp) says "pain-avoidance
routed via valence instead", but edge valence is T3-7 DORMANT. In production pain avoidance runs
through Wire-4 `cluster_fear`, negative NAc causal links (`causal_neg` in `recommend_action`;
`sense_tools` `caution:`) and `percept_valences`. T3-7's resurrection trigger ("the 1.3 fabric
stage") points at a plan that was deferred on a second robot body and is now re-keyed (2026-10-07) to
the grounding line's GL3 registry+provenance stage plus a rung needing cross-modal binding; GL0
re-points T3-7 to GL4.

## 9. What would have to be true

For a word, a percept or a learned value to mean a body consequence, in order (each depends on the
ones above it):

1. **The body says what happened, signed and attributed.** One record per body-consequence event
   (deviation, pain, relief, urgency), produced at `evaluate_failures` and the tool path, keyed on
   the **cause** as well as the sufferer. A burn counts as nociception; heat yields a need; recovery
   from deprivation is a positive event. (Autonomic layer, GL2: the record is `InteroceptiveOutcome`.)
2. **One physical event has one identity across every channel that carries it**, deterministic
   under the step clock (no uuid, no wall time), so a fast pain, a slow ache and a world-sensor change
   join as the same burn. (`PhysicalEventId` and the per-agent sequencer at GL2a; the afferent-track
   scheduler takes the seq over at GL3.B3.)
3. **Every substrate write carries its provenance** (experienced / narrated / imagined), never
   defaulted; narrated consequences earn credit and train the predictor only at a declared discount
   and are never relabelled experienced (owner decision G6); whether imagined nodes are refused or
   discounted is decided at GL3.B1. (GL2a for the record, GL3.B1 for EC/NAc writes.)
4. **Fast signals are not gated by slow ones:** pain transduction does not wait for a turn budget,
   a reflex does not wait for the token budget, and a decided action can be dropped when the body
   changed after the decision. (The L1/L2/L5 defect issues, fixable without the registry; GL3's
   tracks, from GL3.B3, for one scheduling model; preemption is GL3.B4.)
5. **Something predicts consequence from (percept, action) and generalises to an unseen pair**,
   trained on experienced pairs, plus narrated pairs at the declared discount (owner decision G6),
   with the Cerebellum watching the body that changed
   rather than the entity that owns the modulator. (Latent forward model, GL4.)
6. **Concept similarity reads that prediction**, with the word embedding as the prior that experience
   overrides: the jet ordering inverts, `touch`-on-fire separates from `touch`-on-blanket. (GL4 S4,
   tested in GL5.)
7. **Selection reads it**, as a declared arm, never as a silent default. (GL6.)

The plan that sequences these, with gates, reviews and blast radius per stage, is
[docs/plans/grounding.md](../plans/grounding.md).

## Change log

- 2026-10-07: created from the #1120 audit and the four grounding planning reads (GL0 truth item 22).
- 2026-10-07 (review round 2): earned-row wording (no EARNED row depends on the word path); percept
  encoding is any-runtime under the flag, only affordance names are orchestrator-only; affordance
  counts re-verified (UNVERIFIED removed); the `thermal_contact` reflex and the pain-ingress list
  (§3); drift-only scope of L2; the orchestrator-thread `evaluate_failures` caller (§7); T1-12 PARTIAL, DROPPED
  rows removed and new trigger rows (§8); narrated provenance (§9).
- 2026-10-08 (fold 2): the narrator tools run on the orchestrator thread, not `sim.dm`, and the reflex
  dispatch on the loop thread (§7); the `thermal_contact` reflex does call `evaluate_failures` (§3);
  GL3 stage IDs prefixed; T1-12 joins the satiation rows (§8); §9 item 5 admits narrated pairs at the
  declared discount.
