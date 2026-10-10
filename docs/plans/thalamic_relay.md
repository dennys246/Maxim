# Thalamic relay: the Receptor registry and AfferentTracks (grounding line, GL3)

> **PROPOSED, 2026-10-07.** **Sub-plan of the 1.4 grounding line ([grounding.md](grounding.md)),
> stage GL3.** Not a rung and not a behavioural claim. Every mechanism here enters `[engineering]`.
> Every behaviour change ships **behind a flag that defaults OFF**, with today's goldens pinned
> byte-identical while the flag is off. **`src/` work waits for the 1.3.2 decomposition fence**:
> `runtime/agent_loop.py`, `runtime/loop_*.py`, `runtime/substrate_proposal.py`'s call path and
> `simulation/orchestrator.py` belong to Session A until their slices land. Stage GL3.B0 below (tests and
> census only) can run now. This plan's four-lens design review is part of release threshold **T8**
> (grounding truth and contracts, engineering only, gates 1.4.0). Building GL3 is not required for 1.4.0.
> **Stage IDs are always written with the `GL3.` prefix** (`GL3.B0` … `GL3.B8`): a bare "B8" in the
> ledger is T1-6's "B8 delta-attribution change" trigger, which is unrelated to this plan.

**Owner decisions G1, G3, G4, G6 and G8, already taken (2026-10-07), recorded here, not re-opened**
(owner decision G2, the predictor, is GL4's; all are in [grounding.md](grounding.md)):
- **G1** The grounding line fills `roadmap_1_4.md` Phase 5's slots. The body world comes first (Minecraft
  plus cradle, substrate-primary). The word world comes later, as the innate-prior tier.
- **G3** Code names are `Receptor` (a receptor class or percept source; prose may say "engine"
  informally) and `AfferentTrack`. The handoff record is `AfferentEvent`, and it carries a deterministic
  shared `PhysicalEventId` = (agent, seq), with no uuid and no wall time (amended 2026-10-09 by G15:
  (agent, session id, seq); and by G17: the type is built at the post-fence resume stage, not GL2a).
  No existing bio class is renamed. `PhysicalEventId` is **not defined here**: the autonomic plan's
  post-fence resume stage builds it in the leaf module `maxim/embodiment/event_id.py` (no maxim
  imports), and this plan **uses** it (§5.2).
- **G4** Minecraft satisfies the **perception** abstraction. The perception plans' "second body" triggers
  are re-keyed to capability triggers that fire at GL3 (§9). The robot hardware factory keeps its
  physical-robot trigger.
- **A Receptor is not a track.** A Receptor emits onto one or more AfferentTracks. One physical event fans
  out under one `PhysicalEventId`. Tracks are **logical channels scheduled on the loop clock**. They are
  never OS threads, except at hardware edges, which post into an inbox that the loop drains.
- **G6** Consequences written by the narrator's tools (`simulation/tools.py::SetEntitySensorTool`,
  `DamageComponentTool`, `OrchestratorActorTool`, and the reflex dispatch that goes through them) are
  stamped `narrated`, never `experienced`. They are usable for forward-model training and for credit at a
  **declared discount** (value: owner decision at GL4 start; strict default for that later decision: a
  small discount). Provenance kinds were `experienced` / `narrated` / `imagined`; `declared` and `reported`
  stay open at GL3.B1 (§11). A narrated record can never be relabelled experienced. *Amended 2026-10-09
  by G16:* a fourth kind, `apparatus` (harness writes to a drive: rescue, heal, respawn, teacher and
  mother feeds), recorded and never trained on or credited.
- **G8** The Receptor registry lands **with** provenance (one stage, GL3.B1), whose consumers are GL5's
  experiment and the forward model's contamination guard. The nociceptive-fast preemption slice (GL3.B4)
  becomes a declared 1.4 rung arm, or waits for one (the owner names the rung later). The timing defects
  L1, L2, L5 and the dead preemption scaffolding L7 (§3.4, §3.5) are **filed as defect issues** (#1176 L1, #1177 L2, #1178 L5, #1179 L7);
  each is fixable without the registry.

**Owns (proposed):** a new `src/maxim/perception/` package (the name `archive/thalamus_relay_design_pass.md`
design-pass decision 1, DP1, reserved): `ReceptorSpec`, `AfferentTrackSpec`, `AfferentEvent`,
`ReceptorRegistry`, `TrackScheduler`. It also owns a per-node provenance dict on
`similarity/ec.py::EntorhinalCortex` (GL3.B1). **Uses, does not own:** `PhysicalEventId` and the per-agent
`EventSequencer` (`maxim/embodiment/event_id.py`, built at the autonomic plan's post-fence resume stage,
G17; the sequencer is held by the agent's primary `Embodiment`). Seq authority moves to this plan's scheduler at the GL3.B3 stage gate (§5.5).
**Companion plans:** [autonomic_layer.md](autonomic_layer.md) (GL2: `InteroceptiveOutcome`,
`NociceptorSpec`; it supplies the first dual-track receptor); [latent_forward_model.md](latent_forward_model.md)
(GL4: it consumes `AfferentEvent`s joined on `pid`, never on the executor's `uuid4` invocation id,
which is not persisted-stable; the tool path stamps the pid on `ToolOutput` alongside the invocation id).
The state page is
[docs/wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md).
**Prior art, cited as decided:** [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md)
(the organizing frame), [archive/thalamus_relay_design_pass.md](archive/thalamus_relay_design_pass.md)
(SUBSUME), [archive/percept_testbed_audit.md](archive/percept_testbed_audit.md) ("do not build a
percept-channel manifest as conceived", answered in §4.1).
**Inputs, not plans of record:** [deferred/cross_modal_perception_fabric.md](deferred/cross_modal_perception_fabric.md),
[deferred/perception_pipeline_placement.md](deferred/perception_pipeline_placement.md),
[deferred/modality_resolution_and_alignment.md](deferred/modality_resolution_and_alignment.md),
[deferred/mesh_perception_transport.md](deferred/mesh_perception_transport.md),
[deferred/reflex_layering.md](deferred/reflex_layering.md), [deferred/nociception_layer.md](deferred/nociception_layer.md).

Evidence: everything below was read at `origin/main f1c833ab` and is cited as `file::symbol` (line
numbers drift). Facts are from code reads, not runtime probes, unless stated. **UNVERIFIED** marks an
inference that the drafts could not confirm.

## Why this plan exists

The relay the project actually has is not where the 2026-07 documents put it. For the body world the
de facto thalamus is `runtime/substrate_proposal.py::_SUBSTRATE_CHANNELS`, a tuple of `ModalityChannel`s.
The earned survival and orient rows ride on it, and no EARNED row depends on the word (EC text) path.
Around it there are six direct encoder call sites, a percept FIFO that moves one item per loop pass, and
pain ingress through several producers (§3.3) whose timing depends on whichever caller happens to run
`evaluate_failures`, on which thread. No percept carries **who produced it**, **what kind of truth it is**
(experienced, imagined or narrated), or **which physical event it belongs to**. As a result, an imagined
affordance can receive the same NAc credit as an experienced one. A burn cannot be told apart from
"the same burn, felt again later". A breach that begins after the substrate has selected an action cannot
stop that action. This plan adds the three missing identities at one registration point, and adds timing
as logical tracks, both at the smallest size that carries the identity.

---

## 1. What the 2026-07 thalamus framing decided, and what was built

| Design-pass decision DP1–DP4, plus framing (source) | Built? | Evidence |
|---|---|---|
| Frame: thalamus = exteroceptive relay, hypothalamus = drive integration; the two-organ split is correct anatomy, not a bug (`archive/thalamus_hypothalamus_framing.md`) | Framing only | n/a |
| **DP1** SUBSUME the fragmented relays (`ThalamicGate`, `BioEnrichmentPipeline`, the exec-agent call site); do not grow `ThalamicGate`. The coordinator does not exist until a second need forces it. The `perception/` name is reserved for it (`archive/thalamus_relay_design_pass.md`) | Direction adopted. **No coordinator and no `perception/` package exist** | `src/maxim/` has no `perception/` |
| **DP2** Un-flatten the sim percept through a typed side-channel, not a dict key | **BUILT** (PR #402) | `runtime/sim_adapter.py::SimulationAdapter.current_percept`, `NullSimulationAdapter.carry_percept`; guard `tests/unit/test_sim_adapter_unflatten.py` |
| First slice: `CompositePerceptSource`, an ordered first-non-None multiplexer that raises on a second injection implementer | **BUILT**, live only behind `MAXIM_SIM_AUDIO_ORIENT` | `simulation/composite_source.py::CompositePerceptSource`, `simulation/audio_orient_wiring.py`; guard `tests/unit/test_composite_source.py` |
| **DP3** `enabled` = one per-channel gate at the routing fork; `gain` is per-route and deferred | **NOT BUILT**: no channel gate exists | the only per-channel knobs are env (`MAXIM_SIM_AUDIO_SALIENCE` / `_NOVELTY`) |
| **DP4** Audio azimuth gets its own `"audio"` EC route, de-bundled from interoception | **BUILT** as the extero/intero seam (PR #411) | `embodiment/sensory_streams.py::ModalityChannel`, `::DECLARABLE_MODALITY_TAGS`; `substrate_proposal.py::_SUBSTRATE_CHANNELS` |
| Testbed audit M1 (flag consolidation), M2 (per-run active-config record), M3 (per-channel telemetry) | M1 and M3 not built as a channel surface. M2 **UNVERIFIED** (not traced) | the ablation flags are still per-env |
| Fabric (`deferred/cross_modal_perception_fabric.md`) | Zero code. DEFERRED 2026-09-19 on "a second robot body physically exists" | file header |
| Mesh percept transport | Wire prep only. `MeshMessageType.PERCEPT_PUSH` has **no producer** | `mesh/message.py`; `simulation/sources.py` docstring (D46) |

**What this means.** After PR #411 and 1.1.4 PR 2, the body-world relay has the right shape: one
`ModalityChannel(tag, read_values, read_ranges)` per channel, membership declared per sensor in body
YAML (`modality: world|audio`), one encode per channel and one EC cluster space per channel, with late
convergence in `NAc.recommend_action`. The "fragments" the July documents named (`ThalamicGate`,
`BioEnrichmentPipeline`) are both LLM-facing, and neither is on that path. **This plan therefore
subsumes from the body path outward.** `_SUBSTRATE_CHANNELS` becomes the core of the registry (GL3.B1). The
LLM-facing fragments are wrapped last, and only where a consumer needs them.

---

## 2. Receptor census: every live percept source

"EC route" means the substrate path. "LLM route" means the prompt or the percept bus.

| Receptor-to-be | Producer (file::symbol) | How it enters | EC modality / space | `agent_id` | Provenance | Clock |
|---|---|---|---|---|---|---|
| Body drives / interoception | `substrate_proposal.py::_read_drive_states` (plus the derived `cold` need) | `SensorEncoder` channel | `"interoception"`, 384-d hashed sum, ungained, frozen centroid, 0.85 | passed by the caller to `encode_sensors(agent_id=)`, not carried by the reading | none (implicitly experienced) | none on the reading; per tick |
| Minecraft world state | `simulation/minecraft.py::MinecraftClient.latest_state` → `embodiment/backends/minecraft.py::MinecraftWorldBackend.sync_world_sensors` → `world_set_axis(owner="minecraft_bridge")` → `vital_metrics` | `SensorEncoder` channel (`_read_world_states`, declared `modality: world`) | `"world"`, 384-d, **gained** (A4 p3.0), frozen | caller-passed | none | `state_age_s` exists on the client but is not carried |
| Minecraft health / food / oxygen | the same sensors **also** declare `drive:` (`_data/components/bodies/minecraft_player.yaml`) | **the same reading enters twice**: the world channel and the interoception channel | world + interoception | caller | none | none |
| Minecraft game events (chat / damage / death / block) | `simulation/minecraft.py::MinecraftPerceptSource.next_percept` → `make_text_percept("[minecraft:damage] …", channel="narrative")` | text percept; `SensoryTag(NARRATIVE)` by default | `"text"` (768-d mpnet, 0.44), **only with `MAXIM_SUBSTRATE_PATH=1`** | **None** (the factory is called without `agent_id`; `MemoryHub` falls back to the hub's agent) | none. **A damage event is tagged NARRATIVE, not nociceptive** | `Percept.timestamp` (wall) |
| Reachy DoA (live) | `embodiment/audio_localization.py::DoAFeed` | **two lanes**: the sensor lane via `world_set_azimuth` → audio channel, and the percept lane via `make_audio_percept` → `adapter.carry_percept` → the §1.16 prompt fold (`loop_perception.py::orient_to_audio`) | `"audio"` (ungained, frozen; optional place code `MAXIM_PLACE_CODE_EXTEROCEPTION`) | set on the percept | none | `(azimuth, timestamp)` cached; head/body yaw stamped at capture. **The two lanes share no event id** |
| Sim DoA | `AzimuthDoASource` inside a `CompositePerceptSource` | same as live, through `SimulationAdapter.current_percept` | `"audio"` | per the factory call | none | wall |
| Narrator / DM / CLI text | `simulation/conversational_source.py::ConversationalSource.inject_cli` → `make_text_percept(NARRATIVE/cli)` | LLM route; EC `"text"` only with `MAXIM_SUBSTRATE_PATH` | `"text"` | often None | none | wall |
| Sim pain injection | `ConversationalSource.inject_pain` → `make_intero_percept` | a text percept with an INTEROCEPTION tag, published as a `Reaction` straight onto `reaction_bus` | `agents/modality.py::_SUBSTRATE_MAP[INTEROCEPTION] = "text"`, so it **embeds as language** | caller | none | wall |
| Narrator consequence writes (`--sim`) | `simulation/tools.py::SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool` (registered on `orch_registry` against `_aut_embodiment`), and the reflex dispatch, which builds separate instances of the same classes (`BioEnrichmentPipeline`'s `_reflex_damage_tool` / `_reflex_sensor_tool`) | writes the AUT's body sensors, then calls `_aut_embodiment.evaluate_failures()`: the narrator's tools **on the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop)**, the reflex dispatch inside the AUT's `enrich` on the loop thread (`sim.aut`); `DamageComponentTool` also publishes a `PainSignal` directly | reaches EC later, through the body channels' pull read of the written sensors | the AUT's | **`narrated` (owner decision G6)**; today none | wall |
| Vision (DN / robot) | `agents/perception_agent.py` raw `Percept(modality="vision")` | DN `ThalamicGate`, then the LLM. On the EC route (`memory_agent` → `MemoryHub.on_percept_received`), `LinguisticEncoder.encode` embeds the percept's **text content** under modality `"vision"` (code-read), **only with `MAXIM_SUBSTRATE_PATH=1`** | a `"vision"` tag over a **text** embedding | via `_build_context()` | none | wall |
| Imagined entity affordances | `imagination/trigger.py::ImaginationTrigger._encode_entity_affordances` → `encode_decomposed(aff_name, "text", agent_id)` | EC `"text"` + ATL + **`NAc.update_eligibility`** | `"text"` | yes | **None on the node.** `register_substrate_node(source="local")`. `imagined` lives only on `ComponentRegistry` (`provenance="imagined"`), `ImaginationResult`, `Episode.imagined`, and on CausalLinks tagged **retroactively at session end** (orchestrator, "Tag + decay causal links from imagined entities") | `TemporalSignature.now()` (wall) |
| Own-body affordance names | `simulation/orchestrator.py` → `imagination/trigger.py::encode_entity_affordances` | the same encode path as the imagined case | `"text"` | yes | none; **indistinguishable from imagined** | wall |
| Mesh peer percepts | none (`PERCEPT_PUSH` has no producer) | n/a | n/a | n/a | n/a | n/a |

**GL3.B0 census (2026-10-09; corrected after its review).** The checked-in table is
`tests/unit/test_receptor_census.py::CENSUS`: every site with its liveness and the threads that run it;
the scan fails on any new or vanished site (it checks sites; the liveness and thread columns are
code-read, and a new caller one hop up is invisible to it). It adds producers the table above missed or
misdescribed:
- percepts: the messaging channels (`comms/`, on a webhook thread); `AgentPool.run_turn` (a pool worker
  in a concurrent round); the scenario fixture source (`simulation/scenario_source.py`, a raw `Percept`);
  the Dormant `EmbodimentPerceptSource`; and above all `agents/perception_agent.py::PerceptionAgent.process_observation`,
  **the per-pass percept of every `MaximAgent` loop** (CLI / transcript text, vision only when detections
  exist), published synchronously to the memory agent, on the loop thread (the "Vision" row understates it);
- `ConversationalSource.inject_cli` runs on six threads: the orchestrator, `sim.dm` (through
  `send_and_wait`), `sim.stdin` (human free text straight into the AUT's percept source), `sim.stall`
  (nudges, into the orchestrator's own source), a console request, and `main` (the cradle mother, and the
  generative, fixture, pre-campaign and non-interactive DM sends);
- `inject_pain`'s direct-Reaction branch needs `pain_bus=`, which no caller passes: the live path is the
  INTEROCEPTION percept, turned into a Reaction on the loop by `SimulationAdapter.next_observation` (so
  the row above, "published as a Reaction straight onto reaction_bus", is wrong); `inject_sensor` has no caller;
- body writers outside the narrator row: the **cradle reactive mother** (`simulation/cradle_mother.py::reactive_mother_tick`)
  writes the AUT's hunger and azimuth from `main` while `sim.aut` runs, and the DM cascade
  (`dm_runtime.py::CascadeResolver.resolve`) writes campaign entities on `sim.dm` (on `main` when the
  campaign runs non-interactively, after `sim.aut` starts), the AUT's body when a role resolves to it; the
  narrator's `DamageComponentTool` also writes the body through `apply_damage`;
- the imagined-affordance encode is also reached from the fixture runner's manifest on `main` while the
  AUT loop runs; `ComponentIndex._embed` is its own sentence model, not the EC.

**Gaps the census exposes** (all code-read):

1. **Identity.** No reading says which receptor produced it. The DoA feed is already a one-receptor,
   two-lane source built by hand, and its lanes share only a wall timestamp.
2. **`agent_id`.** `PerceptContext.agent_id` is documented as "until then, producers leave it None".
   Minecraft events and narrator percepts do leave it None. Sensor channels get `agent_id` from the
   caller, not from the reading. F0.5 is unfinished.
3. **Provenance.** No EC, ATL or NAc write carries a provenance kind. The only per-node provenance is
   `EntorhinalCortex._substrate_node_sources` ("local" or a hivemind contributor id, which records origin,
   not experienced vs imagined) and `_encoder_provenance` (encoder realized-state per recorder). An
   imagined affordance and the agent's own affordance land on the **same node with eligibility
   updated**, so credit can reach it. Confirmed at `similarity/encoder.py::LinguisticEncoder.encode_decomposed`
   (calls `self._nac.update_eligibility`) and `similarity/ec.py::EntorhinalCortex.register_substrate_node`
   (default `source="local"`).
4. **Misfiled pain.** A Minecraft `[minecraft:damage]` event is NARRATIVE text. A sim pain injection
   embeds as language. Neither enters as nociception. Body-sensor nociception reaches the body through the
   health sensor and `evaluate_failures`, but several producers publish pain without it (the ingress
   census in §3.3).
5. **Double entry.** Minecraft health, food and oxygen sit in both the world and the interoception
   channel. Under this plan that is legal (two receptors reading one sensor), but it must be named, and
   GL3.B1 must not de-duplicate it: doing so moves the geometry and fires Exp 60–62.
6. **Three clocks.** Percepts carry wall `timestamp`. Eligibility anchors use `TemporalSignature.now()`
   (wall) at encode. Hippocampus captures use the agent's experience clock (`experience_us`,
   `capture_seq`; `runtime/experience_time.py`). On turn-based worlds the eligibility anchor and the
   capture time are in different units. **UNVERIFIED** whether this mis-credits anything today.
7. **Dead gain field.** `agents/modality.py::SensoryTag.perceived_intensity` / `modulated_by` are
   documented as "set by SensoryGate". No `SensoryGate` class exists. The only assignment is
   `SensoryTag.from_dict` deserialization. No producer sets it.
8. **Vision as text.** A `"vision"` EC tag is applied over a sentence embedding of the percept text
   (code-read). Leave it alone until a vision receptor exists, and name it on the wiring page.
9. **Three modality vocabularies.** `SensoryModality` (7 values), `Percept.modality`
   (`Literal["text","vision","audio","intero"]`) and the open-string EC tags
   (`interoception/audio/world/text/vision`) are joined by the lossy `agents/modality.py::_SUBSTRATE_MAP`
   (SOUND / TOUCH / SMELL / INTEROCEPTION → `"text"`). The July testbed audit flagged this, and nothing
   changed.
10. **Embedding-space identity already exists, and must not be duplicated.**
    `similarity/encoder.py::encoding_geometry_tag` and `sensor_geometry_fields`, plus `geometry=` on
    `pattern_complete_or_separate`, already name the encoding space (encoder, modality, declared sensor
    set, normalization, dim, gain, ranges). A Receptor's embedding space is **derived** from these, never
    authored.

---

## 3. Timing findings: where a fast signal waits today

### 3.1 The loop pass and its clocks

`runtime/agent_loop.py::run_agentic_loop` runs on one thread. A "pass" is one iteration of its loop:

| § | What runs | Where |
|---|---|---|
| 0–0.6 | `pre_tick_gate`: stop/pause checks, the **live tick** (`_loop_live_tick` → `tick_embodiment_drift` [llm-primary only] + `ExperienceClockDriver.on_live_pass`), then the idle gate | `runtime/loop_gates.py::pre_tick_gate`, `::_loop_live_tick` |
| 1 | `sim.next_observation`: **one** percept per pass | `runtime/sim_adapter.py::SimulationAdapter.next_observation` |
| 1.1 / 1.15 / 1.16 | imagination, auto-sense, audio orientation (with the sim-only orienting reflex) | `loop_perception.py::perceive` (1.3.2 slice 5; `imagine`, `auto_sense`, `orient_to_audio` → `PerceptionOutcome`) |
| 1.2 | ThoughtGate, then `BioEnrichmentPipeline.enrich` (which runs the **percept reflexes**) | inline; `integration/bio_enrichment.py::_evaluate_reflexes` |
| 2 / 3 | poll the LLM worker (non-blocking); `agent.propose_intent` fallback | inline |
| 4 | **execute `ctrl.pending_proposal`** | inline → `runtime/tool_dispatch.py` |
| 6b | substrate tick (substrate-primary): `propose_via_substrate` sets `ctrl.pending_proposal` | `runtime/loop_substrate.py::substrate_tick` |
| 6 | LLM submit (llm-primary) | inline |
| 8.5 | NAc per-pass decay | `agent_loop.py::_loop_bio_tick_maintenance` |
| 9 | `time.sleep(target_period - elapsed)` | inline |

Four clocks are in use. **Wall time** drives the loop period, the substrate cadence
(`loop_state.py::_substrate_tick_due`, `loop_controller.py::LoopController.llm_submit_interval = 0.5`),
drive drift (`embodiment/body.py::evaluate_failures`, `drift_dt = now - _last_poll`), and the PainBus
(0.5 s per `(entity, failure_mode)`) and ReactionBus refractories. **Experience time**
(`memory/experience_clock.py::ExperienceClock`, integer µs, advanced once per live pass by
`runtime/experience_time.py::ExperienceClockDriver`) is read only by memory strength today. **Pass
count** is the clock of NAc decay (§8.5), which is why `_submitted_recently` waking the loop was left
alone (changing it changes Exp 56/57 measurements; `_substrate_tick_due` docstring). **Sequence numbers**:
`Hippocampus` gives each capture a per-agent `capture_seq`, resumed past the saved maximum on load
(`memory/hippocampus.py::Hippocampus._resume_capture_seq`, called from the load path in
`memory/hippocampus_persistence.py`). `time/temporal_event.py::TemporalEvent` only carries an `event_id`;
its producers mint it as `uuid4().hex` (`bridges/tool_pain_bridge.py::ToolPainBridge`,
`embodiment/body.py::_emit_drive_temporal_event`) with wall `TemporalSignature.now()`, which is not
deterministic.

**Determinism infrastructure.** `tests/unit/_loop_harness.py::_StepClock` patches the global `time`
module; only the loop thread advances it. `_ScriptedWorld` is an in-process Minecraft client that is a
pure function of the step clock. `test_agent_loop_selection_golden.py::test_golden_is_identical_across_processes_and_hash_seeds`
pins the selection trace across processes and `PYTHONHASHSEED`. **Every scheduler gate in this plan runs
through this harness.**

### 3.2 Where real concurrency exists

| Thread | Owner | Touches body or bus? |
|---|---|---|
| loop (`sim.aut` under `--sim`) | `run_agentic_loop` | yes; calls `evaluate_failures` from `propose_via_substrate` (substrate-primary) and `tick_embodiment_drift` (llm-primary), and, under `--sim`, through the reflex dispatch's `DamageComponentTool` / `SetEntitySensorTool` instances inside the AUT's `enrich`. **Not the only caller** (next row) |
| the orchestrator thread (`--sim` only: the `start_simulation_mode` caller running the orchestrator agent's loop) | `simulation/orchestrator.py` registers `OrchestratorActorTool` / `DamageComponentTool` / `SetEntitySensorTool` on `orch_registry` against `_aut_embodiment`, run inside `run_agentic_loop(orch_agent, …)` | **yes**: the narrator's tools write the AUT's sensors and call `_aut_embodiment.evaluate_failures()` on this thread while the AUT loop runs on `sim.aut`; `DamageComponentTool` publishes a `PainSignal` directly |
| `sim.dm` (interactive DM campaigns; a non-interactive campaign runs the same code on `main`) | `simulation/orchestrator.py` → `campaign_runner.run_dm_campaign` | never touches the narrator tools above, but it is **not** body-silent: its sends reach `PerceivedPainAssessor.assess_text` (pain into the AUT's buses), `inject_cli` writes the AUT's percept source, and the DM cascade can write the AUT's sensors (GL3.B0 census, §2 and §3.3) |
| `mc-sync-<agent>` (0.5 s) | `simulation/minecraft_harness.py::MinecraftSyncPump._run` | writes `vital_metrics` via `world_set_axis`; publishes no pain and **never calls `evaluate_failures`** |
| `minecraft-bridge-reader` | `simulation/minecraft.py` | fills the snapshot and event queue; `call_action` **blocks** the loop thread until `action_result` |
| `doa-feed` | `embodied_runtime/agentic_runtime.py` → `DoAFeed.run` | `world_set` of azimuth plus `carry_percept` (one slot, latest wins) |
| DefaultNetwork `_run_loop` (30 Hz) | `default_network/network.py::DefaultNetwork` | behaviours, `PriorityArbiter`, ThalamicGate. Its thread also starts under `--sim` (the AUT's DN is built enabled, `loop_setup.py` starts it); with no robot it appears to publish nothing on these buses (**UNVERIFIED**, GL3.B0 review) |
| WorkerPool lanes | `runtime/worker_pool.py` | LLM jobs; `cancel_pending` drains queued jobs only |
| Hippocampus capture worker; `sim.stdin`, `sim.stall` | `memory/hippocampus.py`; `simulation/orchestrator.py` | capture; human edge, watchdog |

The pump writes `vital_metrics` while the loop reads channel values. Nothing guarantees that one encode
sees one snapshot (per-key writes are GIL-atomic; there is no snapshot lock). **UNVERIFIED** whether this
has ever mixed two snapshots in one encode. Tracks fix ordering, not tearing.

**The orchestrator thread is a declared edge.** Two threads run `evaluate_failures` on the AUT's body
under `--sim` (`sim.aut` and the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop)), so the shared mutable state it touches is written from both:
`Entity.drive_breach_severity` (the breach latch dict, `embodiment/sem.py`) and GL2a's per-entity
previous-snapshot slot. No lock guards either today (code-read: `embodiment/body.py` and
`embodiment/sem.py` take none). The rule, in steps (amended 2026-10-09 by G9 and G17): **GL2a** mints
nothing and adds no lock over that state; **from the post-fence resume stage** the tool path mints on the
executor's thread, under the `EventSequencer`'s own private lock;
**with the out-of-band producer, after the fence,** records minted off the loop thread go through the
sequencer's lock, so ids never collide, and one `Embodiment`-owned lock guards the latch dict and the
snapshot slot; **from GL3.B3**, orchestrator-thread
transduction no longer calls `evaluate_failures` directly: it posts into the edge inbox, which the loop drains once per
pass (H0, §5.5), so the latch and the slot are touched from the loop thread only.

### 3.3 The pain path

Body-sensor transduction happens in `embodiment/body.py::evaluate_failures`, which applies drift, latches
the breach on band entry, then `_publish_pain` → `proprioception/pain_bus.py::PainBus.publish`. Callers:
`propose_via_substrate` (substrate-primary), `loop_gates.py::tick_embodiment_drift` (llm-primary, once
per live pass), the reflex dispatch inside the AUT's `enrich` (loop thread), and the narrator's tools on
the orchestrator thread (§3.2). **It is not the only pain
ingress.** Producers that bypass it (code-read census, for GL3.B0's test):

| Producer | Path |
|---|---|
| `simulation/tools.py::DamageComponentTool` | constructs a `PainSignal` and publishes it directly, then also calls `evaluate_failures` |
| `runtime/sim_adapter.py::SimulationAdapter.next_observation` | a `proprioception` / `pain_signal` sim percept becomes a `Reaction` (source `sim_adapter:<pain_type>`) on the ReactionBus |
| `simulation/sandbox.py::PainTriggerLayer` | publishes a `Reaction` and a `PainSignal` onto the PainBus |
| `runtime/pain_interceptor.py::PainInterceptorExecutor` | publishes a `Reaction` onto the ReactionBus |
| `proprioception/perceived_pain.py::PerceivedPainAssessor` | constructs `PainSignal`s and publishes `Reaction`s |
| `bridges/pain_bridge.py::PainCircuitBridge` | the DefaultNetwork pain circuit (robot only) |
| `bridges/tool_pain_bridge.py::ToolPainBridge` | tool-coupled pain inside `execute` (T1-4) |

**GL3.B0 census corrections (2026-10-09; corrected after its review; `tests/unit/test_receptor_census.py::CENSUS`).**
- `ToolPainBridge` publishes no `PainSignal` or `Reaction` (it books attribution into NAc and emits SCN
  `TemporalEvent`s); `PainCircuitBridge` subscribes, and its `record_action_start` arms
  `PainDetector._check_movement_failure` (half of the robot's motion-pain wiring).
- **`PerceivedPainAssessor.assess_text` is a live, undeclared cross-thread pain ingress:** the
  orchestrator assigns it to `bridge.percept_anxiety_hook`, and `SimulationBridge.send_and_wait` calls it
  on every non-substrate-primary send, so it publishes into the AUT's PainBus / ReactionBus from whichever
  thread sends: the orchestrator, **`sim.dm`**, or `main`. `assess` runs inside
  `runtime/pain_interceptor.py::AnticipatoryPainExecutor` (before execute), not `PainInterceptorExecutor`.
- **Tool-failure pain is de-wired:** `PainDetector.record_tool_error` is reached only through
  `Executor._report_failure` when the executor has a `pain_detector`, and no `build_executor` caller passes
  one ([#1200](https://github.com/dennys246/Maxim/issues/1200); not fixed by GL3.B0). `record_tool_running` has no caller.
- `CerebellumModulator`'s prediction `Reaction`s are Dormant: `cerebellum_modulator_factory` has no caller.
- `PainTriggerLayer` publishes a `Reaction` when a ReactionBus exists, else a `PainSignal` (either, not both).
- `evaluate_failures` has three callers the list above omits: `ModulatorAffordanceTool.execute` (the
  AUT's tools on the loop; `OrchestratorActorTool`'s ephemeral tools on the orchestrator thread),
  `simulation/foundry.py::run_gauntlet` (no agent) and the Dormant `EmbodimentPerceptSource`.

`api.py`'s `on("pain_signal")` subscriber is a consumer, not a producer, but it subscribes to the bus
outside `build_pain_bus`'s ordered list. `PainBus.publish` is **synchronous on the publisher's thread**. Subscribers run in `build_pain_bus` order (memory → NAc outcome → Wire-2 percept
valence → Wire-4 cluster fear → extras), then a `Reaction` is forwarded to `ReactionBus`.
`proprioception/pain.py::PainKind` (NOCICEPTIVE / DRIVE / ANTICIPATORY / FRUSTRATION / EXHAUSTION) is the
closest existing thing to "which track", and today it decides memory encoding only.

**The fast paths that exist:** (1) tool-coupled pain, synchronous inside `execute` through `ToolPainBridge`
(row T1-4); (2) **same-call fear in substrate-primary**: inside `propose_via_substrate` the order is
encode channels → `nac.note_active_clusters` → `evaluate_failures` (pain → Wire-4 fear write) → re-read
drives → `anticipatory_threat_need` → `recommend_action`. **This order is load-bearing (Exp 58 W-4: the
encode must precede the pain tick, or the fear lands on the wrong cluster).** (3) The sim-only audio
orienting reflex (§1.16, excluded from substrate-primary). (4) The percept reflex registry
(`embodiment/reflex.py::ReflexRegistry`, `_data/reflexes/*.yaml`), wired only on the orchestrator's AUT.
(5) DefaultNetwork behaviours (robot only). (6) **CLI preemption**: in §1.5 a new non-follow-up CLI input
cancels `pending_action_followup` and any pending proposal whose `strategy_used` is `multi_step` or
`fallback`. **This is the only live preemption rule in the loop, and text triggers it, not the body.**

### 3.4 The delays, L1–L6 (code-read; magnitudes UNVERIFIED until GL3.B0 characterizes them)

**L1, L2 and L5 (and L7, §3.5) are filed as defect issues (owner decision G8): #1176 L1, #1177 L2, #1178 L5, #1179 L7.** Each is
fixable without the registry. Each issue names the root-cause seam its fix belongs in, and the issue's
review checks that the fix is not one of the special cases §4.2 warns about. Whether L1's fix is itself a
preemption check, and so the GL3.B4 arm, is the L1 issue's first question. The tracks (§5) later re-house
the timing these fixes repair; they do not wait for it.

- **Delay L1. Substrate-primary executes a proposal decided on the previous world state.** §6b sets
  `ctrl.pending_proposal` in pass N, and §4 executes it in pass N+1 after the sleep, with no
  re-evaluation. `_substrate_tick_due` requires `pending_proposal is None`, so no pain tick runs in
  between. A breach that starts after selection is not transduced until the next substrate tick (≥ 0.5 s
  later), and it cannot stop the stale action. During a held Minecraft primitive (`call_action` blocks),
  the pump keeps writing world state, but **no pain is evaluated until the hold returns**
  (`roadmap_1_4.md` §E budget: "0.58 s tick + its blocking hold").
- **Delay L2. In the orchestrator, the turn budget also gates nociception.** When
  `MAXIM_SUBSTRATE_ACTIONS_PER_TURN` denies a tick (`loop_substrate.py::substrate_tick`,
  `_substrate_gate_denied`), `propose_via_substrate` is skipped entirely, and with it
  `note_active_clusters` and `evaluate_failures`. For **drift-driven** breaches the band-entry latch then
  publishes late, stamped late, which misaligns NAc's temporal window (a narrator write is not delayed:
  its tool calls `evaluate_failures` immediately, on the orchestrator thread). A slow signal (the narrator
  turn) gates the fastest one.
- **Delay L3. LLM-primary cannot preempt deliberation.** A running LLM job cannot be cancelled. Pain does not
  invalidate a pending LLM proposal (`_drop_stale_proposal` drops only proposals older than
  `_STALE_PROPOSAL_AGE_S = 35 s`). `_run_deliberation_cycles` → `_wait_for_proposal` blocks the loop thread
  for up to 300 s per cycle, with no live tick, no intake and no decay during that time.
- **Delay L4. Sim percepts are one FIFO, one per pass.** `ConversationalSource._queue` is a `deque` with
  `popleft`, so a pain percept waits behind earlier text. `CompositePerceptSource.next_percept` takes the
  first non-None child. Priority was **deliberately deferred** by the design pass ("priority is exactly the
  N=1 policy-bake that earns the coordinator").
- **Delay L5. The "spinal" reflex sits behind the cortical gate.** `enrich` (and so `_evaluate_reflexes`) runs
  only if `_pfc_gate_passed`. `runtime/thought_gate.py::ThoughtGate.should_think` rejects on refractory
  (2 passes), on LLM **token** energy (< 0.15 of budget), on an empty working memory, or on an adaptive
  threshold that `goal_reward_bias` moves. A rejected percept has already been popped, so its reflex never
  fires. An innate reflex therefore depends on the token budget. That inverts the biology. How often it
  happens in a campaign is **UNVERIFIED**.
- **Delay L6. `ThalamicGate` is DN-only and vision-shaped.** `default_network/gate.py::ThalamicGate.evaluate`
  runs in `DefaultNetwork._process_tick` only when there are detections. There is no sim path.
  `DefaultNetwork.add_escalation_callback` has no caller.

### 3.5 L7: dead preemption scaffolding (verified by grep)

- `runtime/preemption.py::PreemptionCircuit` ("generalized interrupt system … Pain is the first registered
  source"), with `SourceConfig{threshold, cooldown, priority}`, `PreemptionSignal`, `ExecutionTracker` and
  `TOOL_REVERSALS`. **Nothing in `src/`, `tests/` or `scripts/` constructs it.** Nothing publishes a
  `PreemptionSignal`, and nothing calls `MaximAgent.wire_preemption`. It survives the orphan lint only
  because `runtime/__init__.py` re-exports `PreemptionCircuit` and `ExecutionTracker`.
- The `agent_loop.py` §3 branch `if hasattr(agent.goal, "check_hold")` is dead: no `check_hold` exists in
  `src/`.
- `runtime/tool_dispatch.py`: the `agent._execution_tracker.capture_before(...)` guard can never be true,
  because only `wire_preemption` sets it.

Its vocabulary (per-source threshold, cooldown, priority; graded withdrawal; a reversal per tool) is
close to what a track needs. Its mechanics (wall-clock cooldowns, AgentBus pub/sub, no producer, no
ordering) are not. **Recommendation (strict, dormancy over deletion):** mark it `Dormant since
2026-10-07: never wired; superseded in vocabulary by AfferentTrackSpec` in its module docstring, filed as
the L7 defect issue. Deletion is the non-strict alternative: it removes the `PreemptionCircuit` /
`ExecutionTracker` re-exports from `maxim.runtime` (a CHANGELOG line). The scheduler does **not** move into
`runtime/preemption.py`; it lives in `perception/` (§5.3).

---

## 4. Front-gate scope pressure

### 4.1 Why this is not the manifest the 2026-07 audit rejected

`archive/percept_testbed_audit.md` rejected "a percept-channel manifest that rides `PerceptSource`" on
four counts. Each count is answered by construction, not by assertion:

| Audit count | How this plan avoids it |
|---|---|
| (1) Two sources of truth: it collided with `perception_pipeline_placement.md`'s config surface and with the body YAML | **Body-sensor `ReceptorSpec`s are derived from body YAML at bind time** (the `modality:` / `drive:` keys that `_SUBSTRATE_CHANNELS` already reads), never authored in a parallel file. The embedding space is derived from `encoding_geometry_tag`. The new YAML keys (`tracks:`, `relay_gain:`) each get their reader in the same PR (the #1120 lesson: `coupled_to` / `modulated_by` are parsed and never read). |
| (2) `PerceptSource` is not a uniform seam, and substrate-primary reads only the body path, so a `PerceptSource` manifest is inert in the LLM-free mode | The registry's **core is the body path**, `_SUBSTRATE_CHANNELS`, not `PerceptSource`. `PerceptSource`-shaped producers are wrapped later (GL3.B5) and stay behind `CompositePerceptSource` as an edge multiplexer. |
| (3) Audio azimuth is double-represented by design (signed extero cluster + folded intero drive), so a flat `{enabled, gain, noise}` entry is ill-defined | The straddle stays two representations. It becomes **one Receptor emitting onto two tracks under one `PhysicalEventId`**. Gain stays per-route (design pass DP3): `relay_gain` on the relay route, motivational gain in the drive's `pain_scale`. |
| (4) Per-channel attribution could not be measured | The registry stamps every route it takes or gates, per receptor (`ec.record_encoder_provenance(f"receptor:{id}", …)`), which is the testbed audit's M3 telemetry. |

It also keeps SUBSUME: the coordinator is introduced now because the second need the design pass waited
for has arrived. **Provenance and event identity must be stamped in one place, or the next producer
forgets them.** Three silent misses have already happened: imagined writes are unflagged, `agent_id` is
None, and `perceived_intensity` is never set. Under CLAUDE.md's silent-no-op rule, that count calls for
structure, not another helper.

### 4.2 Per mechanism: why existing infrastructure cannot do this

**`ReceptorRegistry` + `ReceptorSpec`.** `ModalityChannel` already gives one channel, one encode and one
EC space, but it has no identity, no provenance and no `agent_id` of its own. Six sites call encoders
directly (`_encode_current_clusters`, `propose_via_substrate`, `MemoryHub.on_percept_received`, the two
`imagination/trigger.py` affordance paths, and the orchestrator self-entity encode via
`encode_entity_affordances`), so there is no single place to stamp them. `CompositePerceptSource` is
deliberately priority-free and identity-free. **It rides on:** `ModalityChannel` (becomes the
`sensor_sum` adapter), `SensorEncoder.encode_sensors(modality=)`, `LinguisticEncoder`,
`encoding_geometry_tag` / `record_encoder_provenance`, `DECLARABLE_MODALITY_TAGS`, the body-YAML
declarations, and `PerceptSource` (CC8).

**`AfferentEvent` (on the autonomic plan's `PhysicalEventId`, built at its post-fence resume stage, G17).** No existing record joins two lanes of one physical
event. DoAFeed's lanes share only a wall timestamp, which is not a join key under turn-based or lockstep
clocks. `TemporalEvent` is the right credit envelope, but the uuid its producers mint and its wall-clock
signature are nondeterministic, and it is coupled to SCN registration. It is kept as the envelope, and it
carries `str(pid)` in `context` instead of changing its frozen shape. **It rides on:** the resume stage's
`PhysicalEventId` and per-agent `EventSequencer` (which follow the `Hippocampus.capture_seq` pattern,
including its resume-past-max-on-load rule) and `ExperienceClock`. This plan adds no second identity type.

**`TrackScheduler` + `AfferentTrackSpec`.** PainBus and ReactionBus are synchronous pub/sub on the
publisher's thread with wall-clock refractories. They have no latency, no per-signal destination subset,
no preemption right, and no way to show one physical event on two channels at different times. Adding
that to PainBus would turn a transport into a scheduler for every publisher at once (the T1-4 and T3-9
blast radius). `PreemptionCircuit` has the vocabulary but no producer, no consumer, wall-clock cooldowns
and no ordering. `ThalamicGate` runs on the DN thread, is vision-shaped, and does not run in sim. **It
rides on:** `ExperienceClock` (stamps), `loop_setup.py::LoopRun` / `build_loop_run` (one build site),
`pre_tick_gate`'s live-tick slot (the drain point), PainBus subscribers (destinations), `PainKind` (track
membership), `ReflexRegistry` maths, and `_loop_harness` (lockstep gates).

**EC per-node provenance (GL3.B1).** `Episode.imagined` is episode-level, CausalLink `imagined` is set
retroactively at session end, and EC `source` records hivemind origin. None reaches
`register_substrate_node` / `update_eligibility` at write time.

**What the defects need, and what they do not.** Delays L1, L2 and L5 are defects with root causes of
their own, and owner decision G8 files them as issues now, fixable without the registry (§3.4). What the
issues must avoid is the special-case shape: re-running `evaluate_failures` in a second spot before §4,
an "always run pain" bypass beside the turn gate, a reflex call duplicated outside the gate. Each issue
fixes its seam once (L2, for example, by separating transduction from selection inside the substrate tick,
so the turn budget gates action only). What the defect fixes cannot give, and the reason the registry and
tracks still exist, is the shared identity of one physical event seen on fast and slow channels, with its
provenance, for GL4 and the credit line.

**Count:** one new package with a thin coordinator and a scheduler, three types (`ReceptorSpec`,
`AfferentTrackSpec`, `AfferentEvent`; `PhysicalEventId` is the autonomic plan's, built at its post-fence
resume stage, G17), and one new EC parallel dict. No new
encoder, no new EC modality, no new bus, no class per modality. `ModalityChannel` stops being the registry
and becomes one encoder adapter.

---

## 5. Design

### 5.1 Principles

1. **Receptors declare; the registry routes.** A Receptor turns a world or body change into a reading and
   declares what it is. The registry is the single registration and routing point: it applies the gate
   and relay gain, stamps identity and provenance, and routes to EC, the LLM and tracks. **Receptors never
   call `encode_sensors`, `LinguisticEncoder.encode*` or `NAc.*` themselves.** Nuclei stay declarations.
2. **Transport stays; timing moves.** PainBus stays the subscriber registry and transport.
   `build_pain_bus`'s required-subscriber invariant is untouched. A track decides *when* a publish
   happens and *to which destinations*.
3. **Tracks are logical, on the loop's clock.** Real concurrency stays at the hardware edges that already
   exist (pump, bridge reader, DoA feed, stdin, WorkerPool, DN). Edges post into a lock-guarded **edge
   inbox**. The loop thread drains it at exactly one point per pass and assigns sequence numbers there.
   That single drain point is what makes ordering deterministic.
4. **The relay stays thin.** Multiplex, gate and gain, route, stamp. Nothing else. Stop rule: if registry
   code grows behaviour that belongs in a nucleus, push it back down (framing-doc rule).

### 5.2 Types (one definition each; the drafts' `PerceptEngineSpec` → `ReceptorSpec`, `TrackSpec` → `AfferentTrackSpec`)

```python
# Imported, not defined here. The leaf module (no maxim imports), built at the autonomic plan's
# post-fence resume stage (G17), not GL2a:
#   from maxim.embodiment.event_id import PhysicalEventId
# PhysicalEventId(agent_id: str, session_id: str, seq: int): frozen, SHAPE-FROZEN at 1.0 (CC3 path b);
# __post_init__ rejects an empty agent_id, an empty session_id and a negative seq;
# __str__ is "{agent_id}:{session_id}:{seq}" (G15).
# ONE join rule: InteroceptiveOutcome.pid, CauseRef.cause_pid, AfferentEvent.pid,
# TemporalEvent.context["pid"] and GL4 training rows all join on the pid, never on the
# executor's uuid4 invocation id (not persisted-stable).

ProvenanceKind = Literal["experienced", "narrated", "imagined", "apparatus"]   # apparatus: G16 (harness writes;
                                                   # never trained on or credited); + "declared"/"reported":
                                                   # owner decision (§11, GL3.B1)
ReceptorClass  = Literal["exteroceptive", "interoceptive", "proprioceptive",
                         "nociceptive", "linguistic", "efference"]
EncoderKind    = Literal["sensor_sum", "linguistic", "linguistic_affordance", "precomputed"]

@dataclass(frozen=True)
class ReceptorSpec:
    """One registered receptor. CC3 path (a): defaults on every field + extra, because its stamp
    (receptor_id, receptor_class, provenance_kinds, encoder, derived geometry) is persisted into
    EC encoder provenance (ec.json) and the bundle manifest's encoder_provenance."""
    receptor_id: str = ""                 # stable, dotted: "minecraft.world", "body.drives",
                                          # "reachy.doa", "sim.narrator", "imagination.affordance"
    receptor_class: ReceptorClass = "exteroceptive"
    modality_tag: str = ""                # EC namespace; validated against ONE registry
    encoder: EncoderKind = "sensor_sum"
    provenance_kinds: frozenset[str] = frozenset()   # what it MAY emit. The empty default is a
                                          # SENTINEL that __post_init__ rejects: no type defaults
                                          # provenance to "experienced"; every spec declares it
    tracks: tuple[str, ...] = ()          # validated against the AfferentTrack registry
    clock: Literal["realtime", "turn"] = "realtime"
    relay_gain: float = 1.0               # per-ROUTE relay gain (design pass DP3); motivational
                                          # gain stays in the drive's pain_scale (hypothalamus side)
    enabled: bool = True                  # design pass DP3: one gate at the routing fork
    extra: dict = field(default_factory=dict, hash=False, compare=False)
    def __post_init__(self): ...          # non-empty id/tag; tag registered; provenance_kinds
                                          # non-empty and within ProvenanceKind; relay_gain finite
                                          # >= 0; extra keys must not collide with fields

@dataclass(frozen=True)
class AfferentTrackSpec:
    """One logical afferent pathway. CC3 path (a): it is recorded in the run fingerprint
    (M10 as amended 2026-10-07 / testbed-audit M2), so it persists."""
    name: str = ""
    bio_map: str = ""                     # e.g. "A-delta / spinothalamic (first pain)"
    latency_passes: int = 0               # 0 = same drain phase; k = deliverable k passes later
    priority: int = 0                     # higher drains first within a phase
    preempts: frozenset[str] = frozenset()     # {"pending_proposal","in_flight_plan","held_action"}
    preempt_threshold: float = 1.0        # intensity at/above which preemption fires
    gain: float = 1.0                     # innate prior; adaptive later (adaptive_nociception.md)
    destinations: frozenset[str] = frozenset() # {"reflex","autonomic","relay","credit","prompt","pain_bus"}
    refractory_passes: int = 0            # per (agent, receptor, locus, track), counted in PASSES
    coalesce: Literal["latest", "max", "sum"] = "latest"
    extra: dict = field(default_factory=dict, hash=False, compare=False)

@dataclass(frozen=True)
class AfferentEvent:
    """The receptor→registry handoff: ONE physical event. The scheduler delivers the SAME
    object on every track the receptor declares; a delivery is (AfferentTrackSpec, AfferentEvent),
    and track gain is applied by the destination from the spec. CC3 path (a)."""
    pid: PhysicalEventId | None = None    # required in practice: __post_init__ raises if None
    receptor_id: str = ""                 # required: non-empty, registered
    provenance: str = ""                  # required: the "" sentinel is rejected; must be in the
                                          # spec's provenance_kinds (never defaulted to "experienced")
    locus: str = ""                       # entity path / body part / sensor name
    intensity: float = 0.0                # raw, before track gain
    payload: Mapping[str, Any] = ...      # tagged by EncoderKind: sensor values+ranges | text; JSON-safe
    experience_us: int = 0                # ExperienceClock stamp at ingest (the Hippocampus unit)
    pass_index: int = 0
    salience: float = 0.0                 # level-1 attention [0,1] (fabric §C convention)
    novelty: float = 0.0
    caused_by: PhysicalEventId | None = None   # e.g. "[minecraft:damage]" text → the health-drop event
    extra: dict = field(default_factory=dict, hash=False, compare=False)
```

**Reconciliation choices (put to the four-lens review):**
- **One `PhysicalEventId`, the autonomic plan's.** One planning draft hashed `(engine_id, agent_id,
  engine_seq)` and another used `(agent, seq)`. Owner decision G3 was `(agent, seq)`, amended by G15 to
  `(agent, session id, seq)`; the type lives once, in `maxim/embodiment/event_id.py`, built at the
  post-fence resume stage (G17). The receptor is a field of the event, not of its identity, so one burn
  seen by two receptors (thermal sensor plus the derived drive) can still share or link ids through
  `caused_by`.
- **One seq authority per agent at a time.** The per-agent `EventSequencer` (built at the post-fence
  resume stage, G17), held by the agent's primary `Embodiment`, assigns seq until GL3.B3; from GL3.B3 the scheduler's drain point assigns it and
  the sequencer becomes the scheduler's backing store (persisted high-water). Ephemeral, scene and foundry
  wrappers (`agent_id == ""`, `simulation/tools.py`'s `scene_emb`, the `simulation/foundry.py` wrappers)
  are not the AUT: they mint no records and no ids. (Amended 2026-10-09 by G15 and G17:) GL2a mints no
  pid; the type, the sequencer, its session-id source and the cross-session resume past the sequencer's
  own high-water mark (the `Hippocampus._resume_capture_seq` rule), wired at both load seams
  (`bio_stack.build_bio_stack`, `orchestrator._restore_aut_from_session`), land together after the fence
  and before GL4 S1, so they are in place before GL3.B3's handover.
- **`agent_id` lives once**, on the pid. The event carries no second copy that could disagree.
- **No wall time on the event.** One draft had `wall_ts`. It is dropped: `experience_us` is the analysis
  clock, `Percept.timestamp` still exists for the LLM route, and a wall field would break byte-identical
  traces.
- **CC3 vs the silent-no-op rule.** CC3 path (a) wants defaults on every field. The silent-no-op rule
  wants the identity core to be impossible to omit. The recommendation is sentinel defaults that
  `__post_init__` rejects with `ValueError` (a loud failure at construction), which satisfies both. The
  alternative is keyword-only required fields (a `TypeError`) plus path (b). This is a review item, not an
  owner decision. Either way, **no provenance field has a usable default**: `ReceptorSpec.provenance_kinds`,
  `AfferentEvent.provenance`, GL2a's `InteroceptiveOutcome` and GL4's `ActionContext` (latent_forward_model.md) each reject a
  missing or sentinel provenance in `__post_init__`.

### 5.3 The registry

`ReceptorRegistry` (in `perception/`, built once per agent, carried on `LoopRun` beside the scheduler):

- `register(spec, receptor)`: the `Receptor` is a protocol with `spec` plus either a **pull** reader
  (`read(executor) -> payload | None`, which is today's `ModalityChannel` reader) or a **push** adapter for
  hardware edges, which posts into the scheduler's edge inbox. A duplicate `receptor_id` raises. An
  unregistered `modality_tag` raises. Registration order is preserved and **equals today's code order**,
  `_SUBSTRATE_CHANNELS` = (interoception, audio, world), because that order and the number of active
  channels set the summed cluster term in `recommend_action`.
- `tick(agent_id, executor)`: pulls every pull receptor, keeps the **empty-read → no-encode** rule, builds
  `AfferentEvent`s, and routes:
  - **EC route** per `encoder`: `sensor_sum` → `SensorEncoder.encode_sensors(modality=spec.modality_tag,
    ranges=…)`, unchanged; `linguistic*` → `LinguisticEncoder`. It returns `{modality: cluster_id}`, the
    same dict `_encode_current_clusters` returns today.
  - **LLM route:** the existing nuclei, unchanged: `BioEnrichmentPipeline` for text, `ThalamicGate` for
    DN vision, the §1.16 fold for audio.
  - **Track route:** one scheduler emit per event, fanned out to `spec.tracks` (GL3.B3 onward).
  - **Stamp:** `ec.record_encoder_provenance(f"receptor:{receptor_id}", {...spec stamp, derived geometry})`,
    which reuses the existing per-recorder stamp and its merge rules. **This stamp is a wire format, not a
    log line:** `EntorhinalCortex._encoder_provenance` is saved into `ec.json`'s `encoder_provenance` key,
    and `hivemind/cli.py` carries that key into `hivemind/bundle.py`'s `manifest["encoder_provenance"]`.
    GL3.B1's gate therefore covers the EC save bytes and the bundle-manifest bytes (§6), and its blast
    radius includes T1-10 and T1-14 (§7).
- `enabled=False` drops the event at the fork for every route and records which routes were gated.

### 5.4 The seven AfferentTracks

Every number is an **innate prior**: hard-coded on purpose, reviewed at GL3.B3 start, and owed a follow-up
with a trigger (`adaptive_nociception.md` for gain, `nociception_layer.md` step 5 for membership). Latency
is in **loop passes** and is ordinal, not metric: passes are irregular in wall time, so the Aδ-before-C
ordering is preserved and no millisecond realism is claimed. Docstrings must say "FUNCTIONAL, not fibre
dynamics", as `PainBus` already does. **Only `nociceptive_fast` and `affective_slow` are built, at
GL3.B3, as a fan-out with no preemption; `nociceptive_fast`'s preemption right is exercised only at
GL3.B4. The other five are proposals that enter only with a consumer.** Gain is 1.0 on every track at
entry, so no intensity changes until a stage earns a different value.

**Bio mapping notes.** Thermoreception and thermal pain run on the anterolateral (spinothalamic) system,
not the dorsal column, so a thermal receptor emits on `nociceptive_fast` / `affective_slow`, never on
`mechano_proprio`; its innocuous reading stays in the body channel through the registry's EC route.
Sound direction (DoA) is auditory (brainstem → MGN → auditory cortex), so it rides `extero_detail`, with
its orienting response on `reflex` (collicular), not `mechano_proprio`. `affective_slow` carrying DRIVE
breaches (hunger, air hunger) is justified **only** as Craig's lamina-I homeostatic pathway (lamina I →
spinothalamic / parabrachial → insula), which carries homeostatic as well as nociceptive afferents; it is
not a C-fibre pain claim about hunger.

| Track | Bio mapping | Latency (passes) | Priority | Preemption rights | Gain | Destinations | Refractory / coalesce | Emitters today → subsumes |
|---|---|---|---|---|---|---|---|---|
| `reflex` | spinal reflex arc (withdrawal, flexor); superior colliculus orienting | 0 | 100 | pending proposal (held action later) | 1.0 | reflex, autonomic | `ReflexRegistry`'s own habituation maths; track refractory 0 / latest | narrative keyword reflexes, the audio orienting reflex → `ReflexRegistry` evaluation moves **out of** `enrich` / ThoughtGate (L5); the §1.16 reflex tier |
| `nociceptive_fast` | Aδ, spinothalamic "first pain": sharp, localized (incl. noxious heat) | 0 | 90 | pending proposal, in-flight plan (exercised only from GL3.B4) | 1.0 | pain_bus (the learning subscribers, unchanged), credit (eligibility stamp), relay (situation) | 1 pass per (receptor, locus); at most one preemption per pid / max | `body.evaluate_failures` for `PainKind.NOCICEPTIVE` (incl. `drive:health`), tool damage, `inject_pain` → PainBus *timing* for NOCICEPTIVE; the dead `PreemptionCircuit`'s pain source |
| `affective_slow` | C fibre, spinoreticular / spinoparabrachial "second pain": diffuse, lingering, summing; for DRIVE breaches, Craig's lamina-I homeostatic pathway (see the note above) | 2 | 50 | none | 1.0 | autonomic (GL2's `InteroceptiveOutcome`), credit (cluster fear, memory tag), prompt ("it still hurts") | 0 / sum | the same physical event as `nociceptive_fast`, plus `PainKind.DRIVE` breaches → nothing today; this is GL2's input path |
| `mechano_proprio` | dorsal column / medial lemniscus: touch, proprioception (not thermal, not auditory) | 0 | 60 | none | 1.0 | relay (body schema / interoception channel), credit (Cerebellum forward-model error) | 0 / latest | Minecraft position/contact world sensors via the pump, `world_set_axis`, SEM `self_effect` deltas on mechanical/postural sensors → the pump's untimed `vital_metrics` writes become ingested events |
| `threat_low_road` | thalamo-amygdala "low road": coarse, fast | 0 | 80 | pending proposal (only at/above threshold) | 1.0 | autonomic, selection (`anticipatory_threat_need`) | 1 pass per cluster / max | a situation-cluster change whose learned fear clears θ → the Wire-4 READ moves from "next substrate tick" to "on cluster change"; `ExecAgent` urgent flag (**UNVERIFIED** live) |
| `extero_detail` | cortical route (V1→IT; auditory brainstem → MGN → A1→belt): detailed, slower | 1 | 30 | none | 1.0 | relay (EC / GL4, incl. the `audio` channel), prompt (gated) | 0 / latest | vision detections, DoA azimuth and audio percepts, Minecraft events → the ordered scan in `CompositePerceptSource` gains a priority; `ThalamicGate` becomes this track's prompt gate |
| `language` | language cortex route (heard or read words) | 1 | 20 | pending proposal **only** for `multi_step` / `fallback` strategies | 1.0 | prompt, relay (word world, under `MAXIM_SUBSTRATE_PATH`) | 0 / latest | CLI, narrator, transcript → the §1.5 CLI-preemption rule, **declared, not changed** |

**Fan-out example (one burn, after GL2b's `NociceptorSpec`).** `arms.thermal` crosses its noxious
threshold. The thermal receptor emits one `PhysicalEventId("aut", "<session>", 41)`, delivered on `reflex` (withdraw),
`nociceptive_fast` (locus `arms`, delivered now, published onto PainBus under the nociceptor's failure mode,
`drive:arms.thermal:noxious` as GL2b specifies it) and `affective_slow` (two passes later, summed with any
follow-on heat, feeding GL2's deviation/urgency). The thermal value itself stays in the body channel
through the registry's EC route; it does not ride `mechano_proprio`. On PainBus, the memory and NAc
outcome subscribers see the burn subject to their learning thresholds. **Wire-4 cluster fear does not
fire:** its allowlist (`decisions/nac.py::DEFAULT_CLUSTER_FEAR_FAILURE_MODES` = `drive:health`,
`drive:oxygen`) excludes the burn's failure mode, and no stage edits it (autonomic_layer.md §1.4: a hivemind wire
boundary). GL4 and credit join on `aut:41`, so "the sharp pain and the lingering ache are the same burn" is a
fact of the data, not an inference.

### 5.5 The deterministic `TrackScheduler`

Built once per agent in `runtime/loop_setup.py::build_loop_run` and carried on `LoopRun`. Once live, it
becomes a **required keyword** on `build_loop_run` (silent-no-op-into-types rule).

- `emit(receptor_id, *, locus, intensity, payload, provenance, caused_by=None) -> PhysicalEventId`: loop
  thread only. It assigns `seq` and enqueues one delivery per declared track with
  `due_pass = pass_index + spec.latency_passes`.
- **Seq authority handover (GL3.B3's stage gate).** Before GL3.B3, the per-agent `EventSequencer` built
  at the post-fence resume stage (G17; GL2a mints nothing) (held
  by the primary `Embodiment`, lock-guarded for out-of-band records minted off the loop thread) is the only
  seq authority. At GL3.B3, seq assignment moves to the drain point: `emit` draws from the same per-agent
  counter, which becomes the scheduler's backing store, so the persisted high-water and its
  resume-past-max-on-load rule (both from the post-fence resume stage, G15) carry over unchanged. One authority at a time: a guard test asserts that,
  with the scheduler live, the `Embodiment` never assigns a seq itself, and that seqs stay unique and
  monotonic across the handover and across a save/load.
- `edge_inbox.put(receptor_id, edge_local_seq, ...)`: **the only call legal from another thread.** It is a
  lock-guarded list. Under `--sim`, the orchestrator thread's narrator-tool transduction posts here (§3.2),
  as do the hardware edges.
- `ingest_edges()`: loop thread, **once per pass**. It sorts the inbox by `(receptor_id, edge_local_seq)`,
  which removes any dependence on arrival interleaving, then calls `emit` for each entry.
- `drain(phase) -> list[tuple[AfferentTrackSpec, AfferentEvent]]`: every due delivery, ordered by
  `(due_pass, -priority, seq)`. That is a total order, so delivery is deterministic.
- `preempted_since(basis_seq, right) -> AfferentEvent | None`: has any event on a track holding that right,
  at or above its threshold, arrived after `basis_seq`?
- Refractory is counted in passes per `(agent, receptor, locus, track)`. **Wall time never touches a track
  decision.** The PainBus wall refractory (0.5 s per `(entity, failure_mode)`) still applies to non-track
  publishers until they migrate, and so does the ReactionBus refractory, which keys on
  `f"{kind}:{source}"` (`reactions/bus.py`); `reactions/compat.py::pain_signal_to_reaction` gives every
  body `PainSignal` the source `pain_detector:external_signal` (`embodiment/body.py` publishes every body
  pain as `PainType.EXTERNAL_SIGNAL`), so a drive Reaction and a nociceptor Reaction from one
  `evaluate_failures` call coalesce there within its refractory. The GL3.B3 PR must name every
  publisher's path (the §3.3 census), because the refractories coexist. (`preempted_since` is used only
  from GL3.B4.)

**Hook points in the pass:**

| Hook | Where | What |
|---|---|---|
| H0 `ingest_edges()` | in `pre_tick_gate`, right after `_loop_live_tick` (the slot the experience driver uses) | edge data (hardware edges and the orchestrator-thread declared edge) enters the run in a fixed order: **the edge inbox is drained exactly once per pass** |
| H1 percept → events | §1, `next_observation` | a sim pain percept becomes a `nociceptive_fast` / `affective_slow` emit and no longer waits in the FIFO; text becomes `language` |
| H2 fast drain + preemption check | **immediately before §4 executes `ctrl.pending_proposal`** | deliver due `reflex` / `nociceptive_fast` / `threat_low_road`; from GL3.B4 only: if `preempted_since(proposal.basis_seq, "pending_proposal")` → typed drop (§5.6) |
| H3 transduction | inside `propose_via_substrate`: `evaluate_failures` emits through the scheduler, and latency-0 tracks are **drained in place** | **the Exp 58 W-4 order (encode → `note_active_clusters` → pain → Wire-4 fear write → re-read drives → `anticipatory_threat_need` → `recommend_action`) stays byte-identical** |
| H4 slow drain | §8.5, beside `_loop_bio_tick_maintenance` | deliver due `affective_slow`, `extero_detail` and the slow credit destinations |
| H5 wake source | the idle gate (`_loop_is_idle`) | a due event wakes the loop. **This fires T1-13/14/15's idle-gate trigger by its wording**, so it is not in GL3.B3/B4 (§6 GL3.B7) |

**Determinism.** `seq` is assigned only on the loop thread. Edge entries are sorted at the one H0 point.
No uuid and no wall time enter any decision. `experience_us` is recorded for analysis and for GL4, and it
is never used to schedule. With the flag off, the existing goldens stay byte-identical. With the flag on,
a new arm and golden are pinned through `_loop_harness` (`_ScriptedWorld` is single-threaded, so H0 sees
no edges and the order is a pure function of the step clock). Real worlds stay non-deterministic at their
edges. Tracks make the post-edge order deterministic; they do not change the edges.

### 5.6 Preemption semantics (deterministic, never idle; GL3.B4 only)

Nothing in this subsection is built before GL3.B4, which is a declared 1.4 rung arm or waits for one
(owner decision G8).

- **Basis stamp.** Every proposal carries `basis_seq`, the scheduler's high-water seq when its context was
  built: in `propose_via_substrate` for substrate proposals, and at submit time in §6 for LLM proposals
  (`LLMProposal` gains a defaulted field; it is runtime-ephemeral).
- **Pending proposal.** At H2: drop with reason `preempted_by_track:<track>:<pid>`, logged as a sim EXEC
  line plus a structured event. Then, **in the same pass**, substrate-primary re-runs selection (so the
  protective action happens this pass), and llm-primary re-arms submission with the preempting event in
  `auto_sense_context`. **Named for the gate design pass:** the same-pass reselect re-runs
  `propose_via_substrate`, which calls `nac.note_active_clusters` and runs the Wire-4 fear write again, so
  one pass would note active clusters twice and could write fear twice. The design pass must decide
  whether the reselect re-enters after the transduction step (selection only) or re-runs the whole
  function, and pin the answer with a count test.
- **A preemption drop is not a planning failure.** It does not count against the D13 budget, and it never
  falls through to idle. The D13 invariant (`docs/agents/runtime-tools.md`) requires every drop site to
  re-arm, so this drop gets its own typed handler next to `_handle_planning_failure`.
- **In-flight plan.** The HTTP call is never aborted (WorkerPool cannot). When the job's proposal is
  consumed in §2, the same `basis_seq` check routes it through the same handler.
- **Held action.** Only if a backend declares `interruptible`. Minecraft would need a bridge-side `stop`
  (**UNVERIFIED** that the JS bridge can cancel a hold). Not in GL3.B4.
- **Livelock guard (invariant).** One physical event preempts **at most once**, and refractory applies per
  locus. Without this, a standing breach would re-preempt every reselection.
- **Tool-coupled pain stays synchronous.** `ToolPainBridge`'s direct attribution inside `execute` is not
  deferred by any track (T1-4). Tracks add a timed delivery beside it; they never replace it.

### 5.7 Provenance on the EC / NAc write path (GL3.B1)

Rule (`[engineering]`, enforced in the type): **every EC write carries a provenance kind; `experienced`
creates NAc eligibility at full weight, `narrated` only at the declared discount (owner decision G6), and
`imagined` creates none (strict default, TR12).** A narrated record can never be relabelled experienced.

1. `AfferentEvent.provenance` is required, and the registry passes it to the EC route. Records minted by
   the narrator's tools (`SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool`, and the
   reflex dispatch that goes through them) carry `narrated` (G6). A pull receptor's later read of a sensor
   the narrator wrote is a mixed state; how it is stamped (per-sensor last-writer provenance, or `narrated`
   whenever any declared sensor's last writer was the narrator, the strict default) is a GL3.B1
   gate-design item.
2. EC gains a per-node provenance set, a parallel dict like `_substrate_node_sources`, recording which kinds
   have ever *reinforced* the node (a node first imagined and later experienced shows both). It must enter
   **every** EC persistence and exchange path, or it is silently dropped:
   - `similarity/ec.py::EntorhinalCortex.save` / `load` (`with_format_version` + `atomic_write_json`).
     Files written before GL3.B1 load with an explicit `legacy` marker and one warning: never relabelled
     `experienced` (no type and no loader defaults provenance to `experienced`). How the credit check
     treats `legacy` nodes is part of TR13's flip (strict default: credit as today, because every earned
     row's nodes predate the stamp and nothing reads the set while the flag is OFF).
   - `hivemind/bundle.py::_BUNDLE_EC_NODE_FIELDS`, which today keeps only `embedding`, `modality`,
     `count`, `member_count`, `geometry`, `domain`, `source`, `contributors`: a field not listed there is
     scrubbed from every exported node. Adding it may need a bundle format version bump (T1-14 wording).
   - `hivemind/merge.py::ec_merge_aligned`'s node fold: the union (OR) of both sides' sets, so a narrated
     or imagined kind can never be lost or overwritten by `experienced` in a merge.
   - `hivemind/ingest.py`'s admission pass over a foreign `ec.json` nodes slice: validate the field
     (known kinds only, bounded size), refuse otherwise.
3. `LinguisticEncoder.encode*` takes `provenance=` **keyword-only with no default**, so forgetting it is a
   `TypeError`. `update_eligibility` is called at full weight for `experienced`, at the declared discount
   for `narrated`, and not at all for `imagined`. `imagined` and `narrated` still reach EC and ATL
   (recognition, and the innate word prior that GL4 treats as the prior tier).
4. Credit readers (`NAc.credit_node`, the `TemporalCreditDistributor` fan-out) check the node's
   provenance set: they refuse an EC node whose set holds no creditable kind (`experienced`, `narrated`,
   or `legacy` as TR13 rules), and cap a narrated-only node at the discount. This second, independent lock keeps a stale eligibility from an
   older save from carrying full credit. **Scope: EC node ids only.** Eligibility keys that are not EC
   nodes are out of scope and pass unchanged; the load-bearing case is `tool:<name>`, which
   `bridges/tool_pain_bridge.py::ToolPainBridge` emits as a `TemporalEvent` →
   `decisions/temporal_credit.py::TemporalCreditDistributor.record_event` → `NAc.update_eligibility`
   (refusing it would kill tool credit, T1-4). **The check ships behind a flag that defaults OFF.**
   Flipping it is a separate, owned decision with its own re-runs (§7), and it cannot flip before the G6
   discount value exists (decided at GL4 start).

The contamination guard (latent_forward_model.md's CI test, and this plan's `tests/unit/test_ec_provenance.py`)
checks both halves of G6: a narrated record can never be relabelled `experienced` (through the registry,
a save/load, a merge or an ingest), and the discount is applied wherever narrated data trains or credits.

This tags provenance **at write time**. The end-of-session retroactive CausalLink tagging in the
orchestrator stays as a belt (it is a no-op when the tag is already set) until a deletion probe proves it
redundant.

**Kinds and policy still open** (§11): whether the agent's own-body affordance names (YAML self-knowledge,
neither experienced nor imagined) get a `declared` kind; whether game-log text about a real event
(`[minecraft:damage]`) is `narrated` with `caused_by` or a fourth kind `reported`; and whether imagined
nodes are refused credit (invariant, the strict default) or credited at a discount (an innate prior, like
`FOREIGN_FEAR_DISCOUNT` and the 0.5 imagined-link decay). Narrated is decided (G6: discounted, value at
GL4 start). The namespace alternative (a separate `"text.imagined"` EC modality) is simpler, but it stops
an imagined name from completing onto an experienced node, which is exactly what GL5 needs to measure. The
provenance dict is recommended.

### 5.8 What it subsumes, and what it leaves alone

| Existing piece | Fate |
|---|---|
| `_SUBSTRATE_CHANNELS` + `ModalityChannel` | **The GL3.B1 core.** Three registered receptors in today's code order (`body.drives` for interoception, then `audio.doa` for audio, then `world.state` for world, matching §5.3) whose readers *are* today's functions. `ModalityChannel` survives as the `sensor_sum` adapter. `_encode_current_clusters` and `propose_via_substrate` call `registry.tick`. |
| `CompositePerceptSource` | Stays as an **edge** multiplexer. Priority lives in the scheduler, which is the coordinator the design pass said N ≥ 2 would earn. Its single-injection-implementer error becomes routing addressed by `receptor_id` (GL3.B5). |
| `DoAFeed`'s hand-built two lanes | One push receptor `reachy.doa` on `extero_detail` (auditory; both lanes, the `audio` EC channel and the prompt fold, carry one pid), with its orienting response on `reflex`; never `mechano_proprio` (§5.4 bio note). `world_set_azimuth` stays the body-write side effect (one writer per sensor). Rig-only stage (GL3.B8), T1-7 / T1-10 apply. |
| **Reflex timing** (`ReflexRegistry`) | Kept: habituation, sensitization, pre-emption maths, `ReflexFiring.outcome`. **Evaluation moves to the `reflex` track** (GL3.B6). L5 itself (the reflex behind ThoughtGate) is a defect issue fixed before that, without the registry (§3.4); the track re-houses the fixed timing. `enrich` reads the firing results for its latent affordances. §1.16's orienting reflex moves too; the sim-only rule stays (head-frame lesson). |
| **PainBus** / `build_pain_bus` | **Kept as transport** and subscriber registry: the `pain_bus` destination of `nociceptive_fast` (NOCICEPTIVE) and `affective_slow` (DRIVE). Track-delivered publishes bypass its wall refractory, which the pass refractory replaces. |
| `ReactionBus` | Kept (typed surface). The `inject_pain` → `reaction_bus` shortcut becomes an H1 emit. |
| **CLI preemption rule** (§1.5) | Declared as the `language` track's preemption right, unchanged in behaviour (characterized, GL3.B7). |
| **`ThalamicGate`** | Unchanged DN nucleus. **Later** (GL3.B7) it becomes the prompt gate of `extero_detail` / `language`. The registry consults it only for `exteroceptive` / `vision` on the LLM route. |
| `BioEnrichmentPipeline` / `ExecAgent._run_pre_deliberation` | Unchanged relay-to-PFC nucleus. The call-site redirect named in design pass DP1 comes last (it is a call site, not a component). |
| Imagination and own-body affordance encodes | Two receptors: `imagination.affordance` (`provenance_kinds={"imagined"}`) and `body.affordance_names` (kind per §11) (GL3.B1). |
| `MinecraftPerceptSource`, narrator, CLI | Text receptors with `provenance_kinds={"narrated"}` (or `reported`, §11) and a non-empty agent (GL3.B5). |
| `TemporalEvent` | Kept as the credit envelope. It carries `str(pid)` in `context`, so its frozen shape does not change. The uuid its producers mint stays a log id only. |
| `ExperienceClock` | Ridden on, not replaced. |
| DefaultNetwork 30 Hz thread | **Left alone.** It is a hardware-edge reflex loop on the robot. Under G4, `deferred/hybrid_substrate_reflex_runtime.md` keeps its robot trigger. "DN consumes the `reflex` track through an outbox" is recorded there as a note, not as a trigger change. |
| `embodiment/percepts.py::EmbodimentPerceptSource` (Dormant since 2026-07-14) | **Stays Dormant.** The registry makes it unnecessary. It is not resurrected. |
| `SensoryTag.perceived_intensity` / `modulated_by` (no producer) | `relay_gain` is the real per-route gain. Owner decision: mark the field Dormant/dead, or wire it from `relay_gain`. Not a stage gate. |
| **`runtime/preemption.py::PreemptionCircuit`, `ExecutionTracker`, `wire_preemption`, the `check_hold` branch, the `capture_before` guard** | **Dead (never ran); defect L7, [#1179](https://github.com/dennys246/Maxim/issues/1179).** Owner decision at GL3.B0 (TR1). **Recommended (strict, dormancy over deletion):** mark them `Dormant since 2026-10-07` in the module docstring; callers and re-exports stay; `AfferentTrackSpec` borrows `SourceConfig`'s vocabulary without importing it. Non-strict alternative: delete them, which removes the `PreemptionCircuit` / `ExecutionTracker` re-exports from `maxim.runtime` (a CHANGELOG line). The scheduler lives in `perception/`, not in this module. |

### 5.9 Behaviour tiers

| Behaviour | Tier |
|---|---|
| Registration validation (unique id, registered tag, non-empty agent, provenance within declared kinds); one EC writer per receptor; YAML-derived specs | **invariant** (`[engineering]`) |
| The scheduler's total order; one `PhysicalEventId` across the fan-out; edge inbox drained once per pass; no wall time in track decisions | **invariant** |
| A preemption drop re-arms and never idles; at most one preemption per physical event | **invariant** |
| The Exp 58 W-4 order inside `propose_via_substrate` | **invariant** (already pinned by the selection golden) |
| Provenance on every EC write; a narrated record is never relabelled experienced; `imagined` creates no eligibility (GL3.B1) | **invariant** |
| `narrated` eligibility and credit at the declared discount (G6; value at GL4 start); any imagined discount, if TR12 ever chose one | **innate prior** |
| The credit check on EC nodes (behind its flag) | **invariant** once flipped; non-EC keys (`tool:<name>`) out of scope |
| Every `AfferentTrackSpec` number (latency, priority, gain, thresholds, refractory, coalesce); PainKind → track membership; which tracks a receptor declares; `relay_gain` defaults | **innate prior**: hard-coded on purpose, with follow-ups owed (`adaptive_nociception.md`, `nociception_layer.md` step 5; the `behavior_tiers.md` M4 migration trigger) |
| What delivered events teach (cluster fear, bias, GL4 mappings) | **learned**, by existing mechanisms; nothing in this plan learns |

### 5.10 New `[engineering]` invariants and their guards

| Invariant | Regression guard |
|---|---|
| Every percept enters EC through the registry | `scripts/lint_relay_only_encode.py` (M39: an AST lint allowing `SensorEncoder.encode_*` / `LinguisticEncoder.encode*` / `EC.pattern_complete_or_separate` only from `perception/` plus an allowlist naming each current caller with its reason). Until GL3.B1 ships: **process invariant, mechanization backlog M39**. |
| Tracks are logical, never OS threads | M40: a grep/AST lint, no `threading.Thread` / `asyncio.create_task` in `perception/` except allowlisted hardware-edge adapters. Process invariant until built. |
| One physical event, one pid, across every track; a consequence carries its cause | Structural guard: `AfferentEvent.__post_init__` rejects a missing pid / receptor / provenance, and `PhysicalEventId.__post_init__` (from the post-fence resume stage, G17) rejects an empty agent, an empty session id or a negative seq; `tests/unit/test_afferent_tracks.py` pins the fan-out. |
| One seq authority per agent at a time; seqs unique and monotonic across the GL3.B3 handover and a save/load | `tests/unit/test_afferent_tracks.py` (handover arm) + the resume stage's test through both real load paths (G15, G17) |
| No provenance default; a narrated record is never relabelled experienced, through the registry, save/load, merge or ingest | Structural guard: the sentinel rejections in `ReceptorSpec` / `AfferentEvent` / `InteroceptiveOutcome` / `ActionContext` `__post_init__`; `tests/unit/test_ec_provenance.py` |
| Total order, no wall time, cross-process determinism | `tests/unit/test_afferent_tracks.py` + the flag-on arm of `test_agent_loop_selection_golden.py` across two processes and two `PYTHONHASHSEED`s |
| Preemption re-arms, at most once per pid | `tests/unit/test_tracks_preemption_loop.py` (livelock probe) + a `TestLoopWiringPins`-style pin on the handler |
| `provenance=` cannot be omitted on a text encode | Structural: keyword-only, no default (`TypeError`) on `LinguisticEncoder.encode*` |

---

## 6. Stages

**Renumbered 2026-10-07 (owner decision G8).** Old → new: B0 → GL3.B0; B1 + B3 → GL3.B1 (the registry
lands with provenance); B5 → GL3.B2; B2b → GL3.B3 (the fan-out, now the first track slice); B2a → GL3.B4
(preemption, a declared rung arm or waits); B4 → GL3.B5; B6 → GL3.B6; B7 → GL3.B7; B8 → GL3.B8.

Every `src/` stage gets the three-lens pre-merge review (Executor, Architecture, Wire integrity;
`docs/CODE_REVIEW.md`). This plan gets a **four-lens design review** before GL3.B1 (it is part of T8), and
it must answer the testbed audit and SUBSUME in writing (§4.1). GL3.B4 also gets a **gate design pass** on
a one-page approach note, because the preemption check is a gate. **Fence:** GL3.B1 and GL3.B3–B7 change
code Session A owns (`agent_loop.py`, `loop_*.py`, `substrate_proposal.py`'s call path,
`orchestrator.py`). They start only after the relevant decomposition slices merge, and they must not
reopen a slice's characterization tests. The defect issues for L1, L2, L5 and L7 (§3.4, §3.5) are not
stages of this plan: they are filed now and fixed in their own seams, under the same fence.

### GL3.B0: census and red gates (tests only; runnable now, inside the fence)

> **BUILT 2026-10-09** (tests only; branch `docs/gl3-b0-afferent-census`). The census is
> `tests/unit/test_receptor_census.py` (its corrections are folded into §2 and §3.3); the red gates are
> `tests/unit/test_afferent_red_gates_units.py` (a, b, c, g) and `tests/unit/test_afferent_red_gates_loop.py`
> (d, e, f; (f) in a refractory and an energy-exhausted variant), each `xfail(strict=True,
> raises=AssertionError)` so a broken fixture fails loudly instead of counting as red. TR1 and TR2 are
> decided (§11, DECISIONS.md 2026-10-09).
>
> **The read surfaces are B0's choice, and the flipping stage must flip them as written** or record an
> owner-approved change (never a silent re-point; a red gate that does not flip is data): (a) reads NAc
> eligibility above zero on the imagined nodes and assumes TR12's strict "refuse"; (b) drives the real
> composition (`build_minecraft_aut(..., client=)` → `aut.percept_source.next_percept()`) and reads the
> percept's `agent_id`, so passing the AUT's `agent_id` through `build_minecraft_aut` flips it; (d) matches
> the stale execution by proposal identity AND by (the next pass, the same tool), so a fix that copies the
> action dict cannot flip it falsely; (c) reads a `pid=` keyword on
> `world_set_azimuth` and the percept's `.pid` / `context.pid` / `metadata["pid"]` (GL3.B8 decides where
> the pid is readable, explicitly); (g) reads a `"provenance"` key on
> `EntorhinalCortex.substrate_node_metadata` and assumes the strict mixed-state rule (§5.7 item 1:
> `narrated` whenever any declared sensor's last writer was the narrator).
>
> **The frozen characterization** (`tests/unit/test_afferent_latency_characterization.py`, fixture
> `tests/fixtures/afferent_latency_characterization_v1.json`, measured at 7d18347e with `src/` equal to
> `main`). It is a frozen RECORD, not a golden: it is never regenerated once `src/` moves (the L1/L2 fixes
> and GL3.B4 will move the numbers; GL3.B4 compares against this record). Its test checks equality only
> while `src/` is the measured tree, and the shape otherwise; the instrument's blob hashes are recorded
> as provenance and never asserted, so the shared `_loop_harness.py` stays free to change. "Both arms" are read as the two `_loop_harness` worlds, substrate-primary, in loop passes:
> **damage** (shore world, health 20 → 8 landing while a proposal is pending: the L1 case GL3.B4
> compares against) — pain publish +2 passes, first protective action (`flee`) +3 passes, and the
> stale pre-damage proposal is dispatched; **drowning** (the fear_water world without seeded fear, so
> the breach itself writes the Wire-4 fear) — `drive:oxygen` publish +0 passes, first successful
> protective action (`escape_water`) +25 passes. llm-primary is not characterized: its protective action
> cannot be measured deterministically without an LLM. The loop driver duplicates `_loop_harness.run_arm`'s
> setup in the gate file because `run_arm` takes no world, gate or hook arguments and the harness is
> Session A's; it can fold into the harness if those arguments are added there.

- **Build:** `tests/unit/test_receptor_census.py` enumerates every `make_*_percept` caller, every raw
  `Percept(` site, every `encode_sensors` / `LinguisticEncoder.encode*` caller and `_SUBSTRATE_CHANNELS`,
  against a checked-in table matching §2, **plus** every pain producer in §3.3's ingress census and every
  `evaluate_failures` caller with the thread it runs on (§3.2: the narrator tools on the orchestrator
  thread, the reflex dispatch on the loop thread, never `sim.dm`). A new unregistered producer fails it. This
  is the check behind the census, so no backlog row is needed. Strict red gates (`xfail(strict=True)`):
  (a) an imagined affordance encode produces no eligibility on its node;
  (b) a Minecraft event percept carries a non-empty `agent_id`;
  (c) DoAFeed's two lanes share a pid;
  (d) `test_tracks_d1_stale_substrate_proposal`: through `_loop_harness`, health drops between passes and
  the pre-damage proposal is **not** executed;
  (e) `test_tracks_d2_turn_gate_skips_pain`: with a denying `substrate_action_gate`, a drift-driven breach
  is still published on the breach pass;
  (f) `test_tracks_d5_reflex_behind_thought_gate`: with a refractory or energy-exhausted gate, a matching
  reflex still fires;
  (g) the EC node a narrator-tool consequence (`SetEntitySensorTool` / `DamageComponentTool`) reaches
  carries per-node provenance `narrated`, never `experienced` (the EC-node provenance write, which flips
  with GL3.B1; the record half, `InteroceptiveOutcome` stamped `narrated`, is GL2a's own guard, not
  this red gate).
  Which change flips each: (e) and (f) flip with the L2 and L5 defect issues; (d) with the L1 issue's fix
  or with GL3.B4, whichever the L1 issue decides; (a), (b) and (g) with GL3.B1; (c) with GL3.B8.
  Plus a **characterization** (not a gate): breach → publish and breach → protective-action latency, in
  passes, on both arms. **It is frozen**: committed, with the commit it measured, before GL3.B4's gate
  design pass begins. GL3.B4's latency gate compares against this frozen number, never a re-measurement.
- **Falsifiable gate:** the census equals §2's and §3.3's tables. All seven red gates fail today, which
  proves none is vacuous. Coordinate the harness use with Session A (shared `_loop_harness`).
- **Blast radius:** none (tests only).
- **Owner decisions at start:** TR1 (L7: Dormant, recommended, or delete); TR2 (L2's semantics: may the
  turn budget delay *nociception*, or only *action*? Recommendation: only action). Filing L1, L2, L5 and L7
  as issues is already decided (G8).

### GL3.B1: the Receptor registry with provenance on the write path (lands with its consumer)

- **When:** only together with a consumer of provenance (owner decision G8): GL5's experiment, or the
  forward model's contamination guard (latent_forward_model.md). The registry does not land alone.
- **Build:** `ReceptorSpec`, `Receptor`, `ReceptorRegistry.register/tick`, `AfferentEvent` (pids drawn
  from the resume stage's `EventSequencer`, G17; the seq handover is GL3.B3's). The three `ModalityChannel`s registered as
  receptors in today's code order (interoception, audio, world), with specs derived from body YAML;
  `_encode_current_clusters` and `propose_via_substrate` call `registry.tick`. Provenance per §5.7: the
  imagination and own-body affordance sites become receptors; `provenance=` is required on
  `LinguisticEncoder.encode*`; the narrator tools' records carry `narrated`; the EC per-node provenance set
  goes through `EC.save/load`, `_BUNDLE_EC_NODE_FIELDS`, `ec_merge_aligned` and ingest; the credit check
  ships behind a flag that defaults OFF and covers EC node ids only. No tracks, no new YAML keys.
- **Falsifiable gate (all must hold):**
  1. **Selection path, byte-identical:** a golden fixture over the `minecraft_player`, `infant_operant` and
     `reachy_mini` bodies across a scripted sensor trajectory; the sequence of
     `(modality, cluster_id, geometry_tag, margin)` is byte-identical before and after. The selection
     golden (both arms, cross-process, cross-hash-seed), `test_encoder_golden_v1.py`,
     `test_place_code_wiring.py::test_value_and_range_walks_stay_in_lockstep` and
     `scripts/selection_dynamics_rebaseline.py` output are unchanged.
  2. **Persistence and wire bytes:** a saved `ec.json` and an exported bundle manifest differ from the
     pre-stage output **only** by the added `receptor:*` keys under `encoder_provenance` and the per-node
     provenance field, pinned by a golden diff. A pre-GL3.B1 `ec.json` loads with `legacy` and one warning
     (`tests/integration/test_persistence_compat.py`). A bundle round-trip (export → ingest → merge) keeps
     every node's provenance set, and a narrated kind survives a merge against an experienced node.
  3. Red gates (a), (b) and (g) flip.
  4. Flag on: `tests/integration/test_memory_hub.py` passes; an imagined-only node receives zero credit; a
     narrated-only node receives exactly the declared discount; `tool:<name>` credit is unchanged.
  5. **Deletion probes:** bypass the registry and gate 1's new golden fails; remove the credit check and an
     imagined-only node gets credit; widen the check to non-EC keys and the `tool:<name>` credit test
     fails; drop the field from `_BUNDLE_EC_NODE_FIELDS` and gate 2's round-trip fails. M39's lint passes
     with its allowlist.
- **Guard tests:** `tests/unit/test_receptor_registry.py` (validation, order, empty-read rule),
  `tests/unit/test_receptor_registry_golden.py`, `tests/unit/test_ec_provenance.py`, an addition to
  `tests/integration/test_persistence_compat.py`, and a two-process test that the persisted provenance and
  pid strings are identical under differing `PYTHONHASHSEED`.
- **Blast radius (strict reading of the trigger wording):** **T1-6** (Exp 42) through its
  "SensorEncoder / EC-interoception change" trigger: the drives channel becomes a registered receptor, so
  it fires functionally through the encoder call site, not through any body wording (no body changes
  here). **T1-10** (Exp 53) twice: "`_encode_current_clusters` / … change", and "NAc/EC persistence format
  change", because the receptor stamp and the provenance field change `ec.json` **even with the flag
  off**. **T1-11** (Exp 56) and **T1-12** (Exp 57, PARTIAL): "`SensorEncoder` / EC world-modality change";
  T1-12 also through "`ec_merge_aligned` … change" (the provenance fold). **T1-15** (Exp 62): "the
  `minecraft_player` world-sensor roster" and "`_sensor_embed` / `gain_exponent` / `gain_modalities`",
  whose call site moves. **T1-14** (Exp 61): "`hivemind/bundle.py` scrub" (`_BUNDLE_EC_NODE_FIELDS` and
  the manifest's `encoder_provenance`), "`ingest.py` bounds", "bundle format version bump" if one is taken,
  "`NAc.credit_node` write-path change" when the check is on, and every Exp 60 trigger by inheritance.
  **T1-13** (Exp 60) by its world-modality wording, and through "`TemporalCreditDistributor` credit-path
  change" when the check is on. Offline guards re-run at GL3.B1; the rig re-runs for credit belong to the
  flag flip, not the build.
- **Owner decisions at start:** which consumer it lands with; TR3 (does a byte-identity-proven registry
  fire these rows? Strict default: yes; re-run each row's offline / dry-run guard, and record a committed
  machine-readable exception for the rig-only re-runs if no rig time is spent. The alternative, that the
  goldens discharge the triggers structurally without firing them, is **open, not settled**; the #888
  owner-decision-C precedent is its argument); TR4 home; TR5 derived specs; TR11 kinds; TR12 imagined
  refuse vs discount; TR13 credit-check flag default and `legacy` handling; TR14 dict vs namespace.

### GL3.B2: handoff to the latent forward model (GL4)

- **Build (in `latent_forward_model.md`, not here):** the registry exposes
  `(pid, receptor context embedding, action)` per pass, and GL2's autonomic target joins on `pid` /
  `caused_by`. The tool path stamps the pid on `ToolOutput` alongside the executor's invocation id. This
  plan's obligation is only that the join key exists and is deterministic.
- **Falsifiable gate:** on a recorded cradle and survival trace, ≥ 1 joined
  `(context, action, consequence)` row exists per scripted consequence, every row's pid resolves to
  exactly one `AfferentEvent`, and no row joins on an invocation id. Row counts are reported, with no
  threshold.
- **Blast radius:** none (no consumer; GL4's T4a rule applies).

### GL3.B3: AfferentTracks, the first slice: the thermal dual-track fan-out (no preemption; flag default OFF)

- **Needs:** GL3.B1, GL2a's `InteroceptiveOutcome` and GL2b's `NociceptorSpec`.
- **Build:** `AfferentTrackSpec` and `TrackScheduler` with **two** tracks, `nociceptive_fast` and
  `affective_slow`. H0 (the edge inbox, drained once per pass; under `--sim` the orchestrator thread's
  narrator-tool transduction posts there, §3.2), H3 (transduction emits through the scheduler and is
  drained in place, so the W-4 order is unchanged) and H4. NOCICEPTIVE fans out to fast + slow, DRIVE goes
  to slow; `str(pid)` is carried into `TemporalEvent.context`, and `InteroceptiveOutcome.pid` is the same
  pid. The **seq authority handover** (§5.5). No preemption right is exercised and no `basis_seq` exists
  yet. Flag: `maxim config set runtime.afferent_tracks true` (config over env; if an env var is
  unavoidable, add an autouse conftest scrub in the same commit).
- **Falsifiable gate (all must hold):**
  1. Flag off: both existing goldens are byte-identical, cross-process and cross-hash-seed.
  2. One scripted burn yields exactly two track deliveries with one pid; the slow delivery lands
     `latency_passes` later; the autonomic record and the credit record join on the pid.
  3. **Deletion probe on the slow track:** it removes the affective reading only, and the fast path's
     goldens are unchanged.
  4. **Handover:** with the scheduler live, the `Embodiment` never assigns a seq itself; seqs stay unique
     and monotonic across the handover and across a save/load.
  5. Flag on: a new golden is identical across two processes and two `PYTHONHASHSEED` values.
  6. Under a scripted `--sim` narrator write (a narrator tool run on the orchestrator thread, not on
     `sim.dm`), the breach latch and the snapshot slot are touched from the loop thread only (a
     thread-identity assertion), and the record carries `narrated`.
- **Guard tests:** `tests/unit/test_afferent_tracks.py` (order, identity, pass refractory, inbox sort,
  handover).
- **Blast radius:** with the flag **off**, nothing fires, and the golden proves it. With the flag **on**,
  any validation run is a new condition, and no earned row may be cited under it without a re-run; the
  orchestrator-thread move also changes when narrator-written pain publishes (at the next drain, not inside the tool
  call). No idle gate moves (H5 is not built), so T1-13's idle-gate wording does not fire. **T3-9** (Exp 09:
  "PainBus / ReactionBus / NAc reward pipeline change") arguably fires once a track-routed publish exists:
  walk it before building. **T1-4**: tool-coupled pain stays synchronous inside `execute`, pinned by the
  existing T1-4 tests (`tests/substrate/test_sem_execution_production.py`,
  `tests/unit/test_tool_pain_pending_leak.py`).
- **Owner decisions at start:** TR6 (the two built tracks' numbers; recommendation as tabled in §5.4);
  TR7 (latency in passes, ordinal, recommended, vs experience µs, which is deterministic under `_StepClock`
  but couples scheduling to world kind); TR10 (does it fire T3-9? strict default: yes, walk and re-run its
  offline guard).

### GL3.B4: `nociceptive_fast` preempts a pending substrate proposal (a declared rung arm, or it waits)

- **Entry:** only as an arm a 1.4 rung declares (owner decision G8; the owner names the rung later), and
  only after GL3.B0's characterization is frozen. Outside its declaring arm the flag stays off, asserted by
  M10 as amended 2026-10-07. A gate design pass on a one-page approach note comes first; it must settle
  the same-pass reselect's double `note_active_clusters` / Wire-4 write (§5.6).
- **Build:** `basis_seq` on substrate proposals, H2 (the preemption check before §4, with an in-pass
  reselect), `nociceptive_fast`'s `pending_proposal` preemption right, and the typed preemption handler.
- **Falsifiable gate (all seven must hold, or the stage fails):**
  1. Flag off: both existing goldens are byte-identical, cross-process and cross-hash-seed.
  2. Flag on: red gate (d) flips (if the L1 issue has not already flipped it). The pre-damage proposal is
     never executed, the pain event's `seq > basis_seq`, and reselection runs in the same pass.
  3. Flag on: breach → protective-action latency, in passes, is **strictly lower** than GL3.B0's
     **frozen** characterization on the damage arm. The comparison is pre-registered in the arm's prereg.
  4. Flag on: a new golden is identical across two processes and two `PYTHONHASHSEED` values.
  5. **Deletion probe:** remove the H2 check and (2) and (3) go red; remove the in-pass reselect and (3)
     goes red.
  6. **Livelock probe:** a standing breach over 20 passes preempts at most once per physical event.
  7. A count test pins `note_active_clusters` calls and Wire-4 writes per pass at the number the design
     pass chose.

  These engineering gates are tests. Any behavioural claim belongs to the declaring rung arm's prereg,
  under the weak-evidence rule.
- **Guard tests:** `tests/unit/test_tracks_preemption_loop.py` (the `_loop_harness` arms above).
- **Blast radius:** with the flag off, nothing fires. With the flag on, it is the declared arm's
  condition; T3-9 as at GL3.B3; T1-4 pinned unchanged. If the owner ever turns the flag on by default,
  T1-13/14/15 fire and need the batched rig slot with GL2c's Exp 60 re-run.
- **Owner decisions at start:** which rung declares it; TR6 (`nociceptive_fast`'s preemption right and
  threshold); TR8 (the protective action after a preemption: same-pass reselect, recommended, vs the
  `reflex` track's motor program first).

### GL3.B5: text receptors and the multiplex (after the orchestrator slices)

- **Build:** `minecraft.events`, `sim.narrator` and `cli` register. `CompositePerceptSource` routing is
  addressed by `receptor_id`. Minecraft `[damage]` events carry `caused_by` = the health-drop sensor
  event, which is the first cross-lane join. This is the word world's entry point, so under G1 it follows
  the body-world stages.
- **Falsifiable gate:** the interactive log check (`MAXIM_LOG_FILE` + `--interactive false`, 3 turns) shows
  the same percepts and followups as before. Every text percept's agent is non-empty. A `caused_by` join
  rate is measured on a recorded Minecraft trace and reported; no threshold is claimed.
- **Blast radius:** no T1 row (no EARNED row depends on the word path, #1120). T3-9's trigger wording does
  not name percept factories, so it fires only if `inject_pain`'s `reaction_bus` path moves, and that move
  is GL3.B6's.
- **Owner decision at start:** TR15 (whether the Dormant `perceived_intensity` field is marked dead or
  wired from `relay_gain`).

### Later track slices (not scheduled in 1.4; each enters only when a rung names it)

- **GL3.B6 `reflex` track** (re-houses L5's timing, which the L5 defect issue has already fixed, and fixes
  the L4 pain half): `ReflexRegistry` evaluation moves to the reflex destination, H1 for `inject_pain`, and
  the §1.16 reflex. **Gate:** red gate (f) stays green, and on an Exp 09-style offline fixture the firing
  counts with the gate passing are unchanged. **Blast radius: fires T3-9** (it changes H1 firing counts
  directly). Decide before building.
- **GL3.B7 LLM-primary and the sensory tracks:** in-flight-plan preemption (part of L3; the 300 s blocking
  deliberation stays a decomposition item), `extero_detail` / `language` with ThalamicGate as their prompt
  gate, the CLI rule characterized as unchanged, and **H5 (idle-loop wake)**. **Blast radius: H5 fires
  T1-13/14/15** ("`run_agentic_loop` idle-gate or autonomy handling change"). NAc decays per pass, so a
  woken loop changes eligibility and bias magnitudes. Schedule it with their rig re-runs.
- **GL3.B8 hardware edges:** the Minecraft pump, DoA and stdin go through `edge_inbox`, plus held-action
  preemption where a backend can interrupt. Rig only. Red gate (c) flips here. **Blast radius: the DoA
  feed change fires T1-7** ("DoA front-end / Reachy transport change") **and T1-10** ("`DoAFeed` …
  change"). Verify actuation with `yaw_verify.py` first (embodiment brief §1).

---

## 7. Blast radius summary (rows by ID)

| Stage | Rows that fire by wording | Mitigation |
|---|---|---|
| GL3.B0 | none | tests only |
| GL3.B1 (registry + provenance) | strict reading: T1-6 (functionally, "SensorEncoder / EC-interoception"), T1-10 (encode call site **and** EC persistence format, even flag off), T1-11, T1-12 (world-modality wording + `ec_merge_aligned`), T1-15, T1-14 (bundle scrub / manifest `encoder_provenance` / ingest bounds; `credit_node` when on; Exp 60 by inheritance), T1-13 (world-modality wording; credit path when on) | selection golden byte-identical; persistence/manifest golden diff limited to the added keys; owner ruling TR3 (strict default: fire, re-run offline guards, committed exception for rig re-runs); credit check OFF, rig re-runs tied to the flip |
| GL3.B2 | none | no consumer |
| GL3.B3, flag off | none | goldens prove it |
| GL3.B3, flag on | new condition (incl. narrator-pain timing under `--sim`); T3-9 arguably; T1-4 pinned unchanged | default OFF; walk T3-9 before building |
| GL3.B4 | flag off: none; flag on: the declaring arm's condition, T3-9; default-on would fire T1-13/14/15 | declared arm only (M10); batched rig slot with GL2c if ever default-on |
| GL3.B5 | none of T1; T3-9 only if the reaction path moves | log check |
| GL3.B6 reflex track | T3-9 | Exp 09 offline fixture |
| GL3.B7 idle-loop wake (H5) | T1-13, T1-14, T1-15 | batched rig slot after the 1.3.2 live Exp 60 re-run |
| GL3.B8 DoA feed | T1-7, T1-10 | rig; yaw verification |
| L1 / L2 / L5 / L7 defect issues | walked in each issue (L2 changes when drift-driven pain publishes under the orchestrator's turn gate, so T3-9's wording applies; L1's fix may touch the substrate selection path) | each issue walks its rows before building |

The 1.4.0 minor-version heartbeat walks every Tier-1 row regardless (T4).

**Other risks.** *God-object drift* (the stop rule in §5.1). *A selection-dynamics change hidden inside a
refactor*: channel order and the active-channel count set `recommend_action`'s summed cluster term, so
GL3.B1 preserves both, plus the empty-read rule. *Cross-thread `vital_metrics` tearing*: tracks fix
ordering, not snapshots (**UNVERIFIED** impact) until GL3.B8. *Two threads in `evaluate_failures` under
`--sim`* (§3.2): unlocked today; the resume stage's sequencer lock (G17) guards ids only, and GL3.B3 moves the orchestrator
thread behind the inbox. *Bio over-claim*: pass latency is ordinal only. *Divergence*: if two consecutive stages
each surface a new failure mode, stop and audit the layer beneath (the body sensor resolvers, #1124 /
#1156 / #1159, are the likely layer).

---

## 8. Dependency on the decomposition fence

`runtime/agent_loop.py`, `runtime/loop_*.py` and `simulation/orchestrator.py` belong to **Session A**'s
1.3.2 decomposition (`agent_loop`'s phase-1 slices are built: slice 4 merged as #1187, slice 5, the perception
sections → `loop_perception.py`, built 2026-10-10; `start_simulation_mode` remains). Under G1, src
work waits for the fence, with one exception, GL2a, which lives outside it
(`maxim/embodiment/event_id.py`, which this plan imports, is built at the autonomic plan's post-fence
resume stage, G17, not GL2a). For this plan:

| Stage | Touches | Waits for |
|---|---|---|
| GL3.B0 | tests + `_loop_harness` only | nothing; coordinate harness changes with Session A |
| GL3.B1 | `substrate_proposal.py::_encode_current_clusters` / `propose_via_substrate` (the body path); `imagination/trigger.py`, `orchestrator.py` self-entity encode; EC / NAc / `hivemind/bundle.py`, `merge.py`, `ingest.py` | `agent_loop`'s phase-2 slices (the LLM-primary sections the body path crosses, per `roadmap_1_3_x.md`) **and** the `start_simulation_mode` slices; GL2a (the pid); a provenance consumer (G8) |
| GL3.B2 | none here | GL4 Stage 2 |
| GL3.B3 | `loop_setup.py::build_loop_run`, `loop_substrate.py`, `loop_gates.py::pre_tick_gate`, `simulation/tools.py` (the narrator tools post to the inbox) | the orchestrator slices; GL2b. (`agent_loop` slice 4's PLANNING arm, #1187, and slice 5's imagination / auto-sense / audio / `state.update` arm, `tests/unit/test_loop_perception_characterization.py`, 2026-10-10, are in place.) |
| GL3.B4 | `agent_loop.py` §4, `substrate_proposal.py` | as GL3.B3, plus a declaring rung arm and the frozen GL3.B0 characterization |
| GL3.B5 | `orchestrator.py`, `conversational_source.py`, `minecraft.py` | the orchestrator slices |

The behaviour-preservation gates the slices already pin (`test_agent_loop_selection_golden.py`,
`test_decision_provenance.py`, `test_encoder_golden_v1.py`) are exactly the gates GL3.B1 and GL3.B3's
flag-off arm reuse. For the survival rows, the check must **execute the producer**: the scripted
water-trial smoke (`scripts/survival_world/scripted_water.py`) and `tests/unit/test_exp61_run.py`. A
reproduction of the Exp 60/61/R3 verdicts through `compute_verdict` is not such a check: it re-reads the
committed JSONL and never runs the changed code. Nothing in this plan edits these gates.

---

## 9. Second-body re-keying (G4)

The "deferred until a second body exists" gate mixed two arguments. **Robot-factory honesty**
(`deferred/second_body_staging.md` Stage B: do not design `hardware/controller.py::RobotController` and the
`maxim.robots` entry point from one robot) is about motion and hardware. Minecraft does not satisfy it,
because it is a SEM modulator backend (`MinecraftWorldBackend` via `minecraft_modulator_factory`), not a
`maxim.robots` controller. **That trigger stays.** **Perception-abstraction honesty** is already satisfied:
at least five live sources with different receptor classes, encoders, spaces and clocks (drives,
Minecraft world state, Reachy DoA, narrator and Minecraft text, DN vision). The fabric's own Stage 0b/0c
never needed a second body either; they run on the Reachy.

Re-keyed triggers. This wording is the one used everywhere (banners, README, roadmap, grounding.md):

| Plan | New trigger (capability, fires at GL3) |
|---|---|
| `deferred/cross_modal_perception_fabric.md` | Revives when **GL3's registry+provenance stage ships AND a 1.4 rung needs cross-modal binding** (GL3.B1 carries the event identity and provenance the binding's join key needs). Until then it is an input to this plan (binding convention, two-level attention, artifact contract), not a plan of record. |
| `deferred/perception_pipeline_placement.md` + `runtime/perception_placement.py` (Dormant type layer) | Revives when **GL3's registry+provenance stage ships AND a stage is placed across a wire** (GL3.B1). The `src` Dormant docstring's trigger changes in the same PR (a CHANGELOG line). |
| `deferred/modality_resolution_and_alignment.md` | Revives when **GL3's registry+provenance stage ships** (GL3.B1). Its discriminability facts are inputs. |
| `deferred/second_body_staging.md`, the microduck, the orient line (incl. `hybrid_substrate_reflex_runtime.md`) | **Unchanged:** physical-robot trigger. One added line: "the grounding line's Receptor is a percept receptor, not this plan's robot 'engine seam'." |

The naming follows: `Receptor`, never a bare `Engine`, so it cannot collide with the robot "engine seam",
`VisionEngine` or `TTSEngine`.

---

## 10. Mechanization rows this plan relies on

| Row | Rule | Check |
|---|---|---|
| M39 | Every percept enters EC through the registry | AST lint with an allowlist naming each current caller and its reason (the M38 pattern). Lands with GL3.B1. |
| M40 | Tracks are logical, never OS threads | grep/AST lint over `perception/`, hardware-edge adapters allowlisted. Lands with GL3.B3. |
| M10 (amended 2026-10-07) | No grounding flag is active in an E1–E3 arm unless the arm declares it | M10's list already names GL3 track preemption (`runtime.afferent_tracks`); the GL3.B1 PR adds the credit-check flag to it. |

The pid-and-cause rule and the no-provenance-default rule need no backlog row: they are **structural
guards** (`AfferentEvent.__post_init__`, `PhysicalEventId.__post_init__` from the resume stage, the sentinel rejections),
cited as such in §5.10.

---

## 11. Open owner decisions (each asked at the named stage's start; strict option is the default recommendation where one exists)

Decided, not re-opened: G6 (narrated is discounted, never experienced; the discount value is GL4's
start decision), G8 (registry with provenance; GL3.B4 a declared arm or waits; L1/L2/L5/L7 filed as
#1176–#1179), and the identity contract (`PhysicalEventId` (agent, session id, seq), built with its
sequencer, session-id source and cross-session resume at the post-fence resume stage, G15 and G17; one
seq authority per agent; the orchestrator thread a declared edge).

| # | Decision | At | Recommendation |
|---|---|---|---|
| TR1 | Dead `PreemptionCircuit` / `ExecutionTracker` / `wire_preemption` / `check_hold` / `capture_before` (L7): `Dormant since` or delete | GL3.B0 (via the L7 issue) | **Decided 2026-10-09 (owner): Dormant** (`Dormant since 2026-10-07`; callers and re-exports stay), carried out by #1179. |
| TR2 | L2: may the turn budget delay nociception, or only action? | GL3.B0 (via the L2 issue) | **Decided 2026-10-09 (owner): only action.** Red gate (e) encodes it; #1177 fixes it. |
| TR3 | Does a byte-identity-proven GL3.B1 fire T1-6/10/11/12/15 (+13/14)? | GL3.B1 | Strict: yes, re-run offline guards, committed exception for rig re-runs. The structural discharge is the open alternative. |
| TR4 | Home and names: `perception/`, `ReceptorRegistry`, `TrackScheduler` | GL3.B1 | Yes (the design pass reserved `perception/`). |
| TR5 | Body-sensor specs derived from YAML (recommended) or authored; each new YAML key ships with its reader | GL3.B1 | Derived. |
| TR6 | Track numbers: latency, priority, preemption rights, gain, refractory | GL3.B3, GL3.B4 | As tabled in §5.4; only `nociceptive_fast` / `affective_slow` built; the preemption right only at GL3.B4. |
| TR7 | Latency in passes (ordinal) or experience µs | GL3.B3 | Passes. |
| TR8 | Protective action after preemption: same-pass reselect, or reflex motor program first | GL3.B4 | Same-pass reselect, with the double-`note_active_clusters` question settled by the gate design pass. |
| TR9 | (Settled by the identity contract as amended by G15 and G17: no pid at GL2a; the type, the sequencer and its session-id source are built, persisted per agent and resumed past the saved maximum at the post-fence resume stage, before GL4 S1.) | n/a | n/a |
| TR10 | Do GL3.B3, GL3.B4 and GL3.B6 fire T3-9? | GL3.B3, GL3.B4, GL3.B6 | Strict: yes; walk and re-run Exp 09's offline guard. |
| TR11 | Provenance kinds beyond experienced / narrated / imagined: add `declared`? `reported`, or `narrated` + `caused_by`? | GL3.B1 | `declared` yes (the planning drafts' recommendation); `narrated` + `caused_by` (this plan's lean; no draft recommendation). |
| TR12 | Refuse vs discount credit for **imagined** nodes (narrated is decided: G6) | GL3.B1 | Strict: refuse (invariant). |
| TR13 | Credit-check flag default; how `legacy` (pre-GL3.B1) nodes are treated when it flips | GL3.B1 | OFF until a separate, owned flip with its re-runs, after the G6 discount value exists; `legacy` credited as today (refusing it would silently change every earned agent's credit at the flip). |
| TR14 | Provenance dict vs `"text.imagined"` EC namespace | GL3.B1 | Dict. |
| TR15 | `SensoryTag.perceived_intensity`: mark dead, or wire from `relay_gain` | GL3.B5 | Owner's pick; not a gate. |
| TR16 | Minecraft health/food/oxygen double entry: keep both, and record the drive side as derived (`caused_by` the world event) once GL2 owns the drive receptor | GL3.B1 / GL2 | Keep both. Never de-duplicate inside GL3.B1. |
| TR17 | Which 1.4 rung declares GL3.B4, and which consumer GL3.B1 lands with | GL3.B4, GL3.B1 | The owner names them (G8). |
