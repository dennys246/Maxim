# Grounding: concepts become similar by what they do to the body (the 1.4 grounding line)

> **ACTIVE, 2026-10-07.** Owner decisions taken; GL0 and GL1 start now; the three sub-plans stay
> PROPOSED pending their design reviews. A 1.4 line that fills existing slots in [roadmap_1_4.md](roadmap_1_4.md)
> §Phase 5 (the relief store, the graded predictor / `latent_forward_model.md`); it creates no second
> store and no second predictor. Owner decisions **G1–G4** (below) were taken in conversation on
> 2026-10-07, and **G5–G8** in the review round the same day; all are recorded here as decided. Every
> mechanism enters as `[engineering]`; nothing here claims behaviour until GL5 records an outcome.
> **Body world first; the word world follows as the innate-prior tier.** GL0, GL1, GL4 S0a (audit)
> and the paper run now, in parallel with 1.3.2; GL4 S0b needs a fresh capture (a sim run). GL3.B0
> (tests only) is runnable now, inside the fence. GL2a and GL4 S1 are `src/` inside the fence exemption
> (§G1's exempt file set); GL4 S1 waits only for GL2a; GL4 S3 and S4 are `src/` and wait for the 1.3.2
> decomposition fence like every other `src/` stage; GL4 S2 is offline (`scripts/`).
>
> **Stage IDs.** `GL0`–`GL6` are this line's stages. Sub-plans number their own internal stages
> (autonomic `GL2a`–`GL2c`; the forward model `S0`–`S5`, where its **S5 is GL5**; the relay's
> `GL3.B0`–`GL3.B8`, always written with the `GL3.` prefix because a bare `B<n>` collides with the **B8
> delta-attribution invariant**). The relay's map ([thalamic_relay.md](thalamic_relay.md) §6) is
> canonical: **GL3.B0** census + red gates (tests only; runnable now, inside the fence); **GL3.B1** the
> registry+provenance stage (lands only with its consumer: GL5's experiment or the forward model's
> contamination guard); **GL3.B2** the handoff to GL4; **GL3.B3** the first track slice, the thermal
> dual-track fan-out (`nociceptive_fast` + `affective_slow`, the shared pid reaching credit; no
> preemption; needs GL2b's `NociceptorSpec`; carries the seq-authority handover as its stage gate);
> **GL3.B4** nociceptive-fast preemption (a declared 1.4 rung arm, or it waits, G8); **GL3.B5** text
> receptors; **GL3.B6** the `reflex` track; **GL3.B7** LLM-primary + the sensory tracks and the wake
> hook; **GL3.B8** hardware edges.

**State page (read first):** [../wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md):
the evidence for everything in §The problem. This plan is the fix list; that page is the state.
Update both when a stage lands.
**Sub-plans:** [autonomic_layer.md](autonomic_layer.md) (GL2) ·
[thalamic_relay.md](thalamic_relay.md) (GL3) · [latent_forward_model.md](latent_forward_model.md)
(GL4). All three are PROPOSED and do not exist on `main` until GL1 merges them.
**Owns (proposed):** the consequence record and its producers in `embodiment/sem.py` /
`embodiment/body.py`; the `Receptor` registry and `AfferentTrack` scheduler (home decided at GL3.B1's
start); the consequence predictor and its scripts; the GL5/GL6 preregs.
**Companion plans:** [engram_formation.md](engram_formation.md) (T7 sibling; E7 binds to GL4 S0a),
[memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) (consumer of the consequence
record via `EncodingSignals`), [lookback_primitive.md](lookback_primitive.md) and R4 (credit routing
decides the selection surface), [fear_learning.md](fear_learning.md) Exp A and
[coding_world.md](coding_world.md) C3 (relief-store co-consumers),
[deferred/behavior_tiers.md](deferred/behavior_tiers.md).

## Owner decisions recorded 2026-10-07

| # | Decision |
|---|---|
| **G1 Placement** | The line maps onto roadmap 1.4 §Phase 5's existing slots (the relief store; the latent forward model). New release threshold **T8** (engineering only) gates 1.4.0, like T7. New **T9** is the first grounding claim, CONDITIONAL, and is never co-headlined with E3. **Body world first** (Minecraft + cradle, substrate-primary); the word world comes later as the innate-prior tier. GL0, GL1, GL4 S0a (audit) and the paper run now in parallel with 1.3.2; GL4 S0b needs a fresh capture (a sim run). `src/` waits for the `agent_loop` / orchestrator decomposition fence, except record-only GL2a and GL4 S1. **The exempt file set, exactly:** GL2a edits `embodiment/body.py`, `embodiment/sem.py`, the new leaf `embodiment/event_id.py`, `runtime/executor.py` (`Executor._stamp_invocation` only), `tools/base.py::ToolOutput` and `runtime/bio_integration.py`; GL4 S1 edits `embodiment/tool_bridge.py` (the `ActionContext` assembly and the observe call), `embodiment/cerebellum.py` (payload `"1.2"`, the new payload key, `import_state` refusing newer versions), `runtime/executor.py` (`Executor._stamp_invocation` only), `runtime/bio_integration.py` (`EncodingSignals.extra["context"]` at the loop capture) and the new leaf `embodiment/action_context.py` (the `ActionContext` type), under the same exemption. GL4 S1 waits only for GL2a; GL4 S3 and S4 are `src/` and wait for the fence; GL4 S2 is offline (`scripts/`). Anything outside that set waits for the fence. |
| **G2 Predictor** | Reverses roadmap 1.4's 2026-09-18 decision 3 ("JEPA re-pointed, not revived"). The predictor enters as [latent_forward_model.md](latent_forward_model.md), the name Phase 5 already reserves, and inherits the JEPA plan's four rules: no pretrained cross-modal weights; the contamination guard is a CI test; opt-in; existing encoders untouched. It is **not called JEPA** until its target becomes a learned embedding of a rich percept. With a fixed autonomic target it is supervised regression in a JEPA shape, and the plan says so. Needs a DECISIONS.md entry (GL1). |
| **G3 Naming** | Code names `Receptor` (a receptor class / percept source; prose may say "engine" informally) and `AfferentTrack`. The handoff record is `AfferentEvent`, carrying a deterministic shared `PhysicalEventId(agent_id, seq)`: no uuid, no wall time. Stage IDs are GL0–GL6 (`GL` is unused in `docs/plans/**`). No existing bio class is renamed. |
| **G4 Second body** | Minecraft satisfies the PERCEPTION abstraction. `cross_modal_perception_fabric.md`, `perception_pipeline_placement.md` and `modality_resolution_and_alignment.md` re-key from body count to capability triggers that fire at GL3. The robot hardware factory (`second_body_staging.md` Stage B, the orient line, microduck) keeps its physical-robot trigger. |
| **G5 Dispositions** | Taken now (§Consolidates). [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) is **SUBSUMED** by `latent_forward_model.md` (its four rules carried over). [grounded_language_acquisition.md](grounded_language_acquisition.md) is **SUBSUMED** for grounding by this plan (its own scope note kept). [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md)'s L0 gate is kept as GL1's innate-prior measurement, and its old "PASS → candidate 1.5 headline" consequence is **RETIRED**. [deferred/nociception_layer.md](deferred/nociception_layer.md) is **REVIVED** into `autonomic_layer.md`. These status words are used everywhere. |
| **G6 Narrated provenance** | Consequences written by the narrator's tools (`simulation/tools.py::SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool`, and the reflex dispatch that goes through them) are stamped `narrated`, never `experienced`. They are usable for forward-model training and for credit at a **declared discount**; its value is an owner decision at GL4 start (strict default: a small discount), and GL5 reports its results with AND without narrated data. Reason (owner): excluding them would mute the world the LLM's language priors simulate. Provenance kinds are `experienced` / `narrated` / `imagined`; whether to add `declared` and `reported` stays open at the start of GL3.B1 (the registry+provenance stage). The contamination guard checks that a narrated record can never be relabelled `experienced` and that the discount is applied. |
| **G7 Satiation and the survival rows** | GL2c fires T1-11, T1-12, T1-13, T1-14 and T1-15 **whatever its routing** (T1-12 added by the owner 2026-10-08: `minecraft_bench57`'s `d1` is the same entropic-up spec as Exp 56's, `satisfaction_threshold: 0.3`, and T1-12's `Re-run on:` matches T1-11's). On `minecraft_player` the homeostatic `oxygen` and `health` breach latches clear on every surfacing or regeneration (`embodiment/body.py::Embodiment.evaluate_failures`, `elif cleared: breach_latch.pop(...)`), which is Exp 60's own `escape_water` contingency; `minecraft_player`'s `food` is entropic with `satisfaction_threshold: 16`; `minecraft_bench`'s `d1` (T1-11) and `minecraft_bench57`'s `d1` (T1-12) are entropic upward with `satisfaction_threshold: 0.3`. So GL2c ships **off by default** and lands only with a batched live re-run of Exp 60/61/62 (plus the T1-11 and T1-12 arguments or re-runs), and with the staged-donor refusals **restated, not deleted**: `scripts/survival_world/exp61_run.py::donor_sanity_staged` (refuses non-empty `reward_bias`, `cluster_reward_bias`, `links`, `event_outcome_welford`) and `scripts/survival_world/r3_run.py::_R3._boundary` (`reward_bias`, `links`). Routing (relief store, distributor, or both) stays an open GL2c decision; no routing avoids the lapse. |
| **G8 The registry enters with its consumer** | GL3's `Receptor` registry lands **together with** provenance on the write path (one stage, GL3.B1, the registry+provenance stage), whose consumers are GL5's experiment and the forward model's contamination guard. The first track slice (GL3.B3) is the thermal dual-track fan-out with no preemption; nociceptive-fast preemption (GL3.B4) becomes a **declared 1.4 rung arm**, or waits for one; the owner names the rung later. The timing defects L1, L2, L5 and the dead preemption scaffolding L7 ([state page §7](../wiring/body-and-word-worlds.md)) are **filed as defect issues** in GL0(d) (#1176 L1, #1177 L2, #1178 L5, #1179 L7); each is fixable without the registry. GL3's census and red gates (GL3.B0) stay first and are runnable now, inside the fence. |

**The event-identity contract** (stated once here; the type is defined in
[autonomic_layer.md](autonomic_layer.md), which builds it at GL2a, and GL3 imports it):
- One type, `PhysicalEventId(agent_id: str, seq: int)`: frozen, SHAPE-FROZEN at 1.0 (CC3 path b),
  `__post_init__` rejects an empty `agent_id` and a negative `seq`, `__str__` is `"{agent_id}:{seq}"`.
  It lives in one leaf module with no `maxim` imports, `maxim/embodiment/event_id.py`.
- One seq authority per agent: a per-agent `EventSequencer` held by the agent's primary `Embodiment`.
  Ephemeral, scene and foundry wrappers (`agent_id == ""`, the `scene_emb` wrapper in
  `simulation/tools.py::OrchestratorActorTool`, the `simulation/foundry.py` wrappers) are not the AUT
  and mint no records and no ids. `seq` persists per agent from GL2a and resumes past the saved
  maximum on load (the `memory/hippocampus.py::Hippocampus._resume_capture_seq` rule). At GL3, seq
  assignment moves to the track scheduler's drain point and the GL2a counter becomes its backing
  store: one authority at a time, and the handover is GL3.B3's stage gate.
- The orchestrator thread is a **declared edge**. In `--sim`, `simulation/orchestrator.py` registers
  `OrchestratorActorTool`, `DamageComponentTool` and `SetEntitySensorTool` on the orchestrator's tool
  registry (`orch_registry`); they run inside the orchestrator agent's `run_agentic_loop` on the
  orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop) and
  call `_aut_embodiment.evaluate_failures()` while the AUT loop runs on `sim.aut`. (`sim.dm` is a
  different thread: it exists only for interactive DM campaigns and never touches these tools.) The
  reflex dispatch builds separate instances of the same classes (`BioEnrichmentPipeline`'s
  `_reflex_damage_tool` / `_reflex_sensor_tool`) and calls them inside the AUT's `enrich`, on the loop
  thread, not the edge. The narrator tools touch shared mutable state from a second thread
  (`Entity.drive_breach_severity`, the breach latch, and the per-entity previous-snapshot slot GL2a adds
  beside it). From GL2a, one lock guards that state and the sequencer, so out-of-band records minted
  off the loop thread are serialized; from GL3.B3, orchestrator-thread transduction posts into the edge
  inbox, drained once per pass. GL3.B0's census and GL3.B3's gate 6 assert this thread identity. (The Minecraft sync pump, `MinecraftSyncPump._run`, writes vitals but never calls
  `evaluate_failures`.)
- Field names: `InteroceptiveOutcome.pid`, `CauseRef.cause_pid` (`PhysicalEventId | None`),
  `AfferentEvent.pid`. **One join rule everywhere:** forward-model pairs join on `pid`, never on the
  executor's `uuid4` invocation id, which is not persisted-stable; the tool path stamps the pid on
  `ToolOutput` beside the invocation id.
- No type defaults provenance to `experienced`: `ActionContext`, `ReceptorSpec.provenance_kinds` and
  `InteroceptiveOutcome` require it explicitly and reject a sentinel in `__post_init__`.
- `Embodiment.drain_outcomes()` has named production drainers (the loop capture in substrate-primary;
  in `--sim`, the AUT loop's `capture_loop_action` on `sim.aut`, which also drains the records the
  narrator's tools queued from the orchestrator thread), a bound (drop-oldest, each drop counted and logged), and
  ephemeral wrappers never queue.

Two architectural rules came with these decisions and are binding on every sub-plan:
- **Engine ≠ track.** A `Receptor` emits onto one or more `AfferentTrack`s; one physical event fans
  out with one shared `PhysicalEventId`.
- **Tracks are logical scheduled channels** on the loop/tick clock: deterministic and
  lockstep-testable. They are never OS threads, except at hardware edges, which post into an inbox
  the loop thread drains at one fixed point.

## The problem

The 2026-10-07 five-angle audit on [#1120](https://github.com/dennys246/Maxim/issues/1120) found that
the EC holds **two disconnected worlds**. Evidence, with `file::symbol` citations, is on the state
page; in brief:

- **The body world** (`SensorEncoder` → EC interoception/audio/world at 0.85, frozen →
  `cluster_fear` / `cluster_reward_bias` → `NAc.recommend_action`) is live by default and is the only
  path that acts without the LLM. **No EARNED row depends on the word (EC `text`) path**: Exp
  56/60/61/62 and orient 45/52/53 ride this chain, T1-16 (Exp 63) is Hippocampus recall into the
  prompt, and T1-4 rides PainBus / `ToolPainBridge`.
- **The word world** (`LinguisticEncoder` → EC `text` at 0.44, running mean → ATL by name) runs only
  with `MAXIM_SUBSTRATE_PATH=1`. Percept encoding (`agents/memory_agent.py` →
  `MemoryHub.on_percept_received` → `LinguisticEncoder.encode`) writes EC `text` / `vision` on any
  runtime with that flag; only **affordance-name** encoding (`imagination/trigger.py::encode_entity_affordances`)
  is orchestrator-only. Affordance concepts are embeddings of the affordance NAME; components close
  to their compound never form nodes.
- **No cross-modal link exists.** Hebbian binding is Dormant (D6), `retrieve_cross_modal` is
  uncalled, naming events are Dormant, the binding plan is archived "DO NOT RESURRECT", the JEPA plan
  has zero code, and no write carries experienced/imagined provenance.
- **The word world cannot learn.** Concept nodes are paid only by Reactions, every live Reaction is
  negative, `_reward_bias` is clamped ≥ 0, annotators look concepts up by exact name, and selection
  never reads concept bias.
- **Names and consequences disagree** on shipped components (`touch` ×16; `warm_self` safe/harmful;
  `turn_left`/`turn_right` at 0.874; `flame jet` is closer to `water jet`, 0.664, than to `fire
  breath`, 0.601). Only 42/405 affordances declare a body effect.
- **The body is under-wired.** The infant burn peaks at 0.2 and is classed DRIVE, below every PainBus
  learner; heat yields no corrective need; Wire-2 valence is keyed on the sufferer, not the cause;
  there is no positive producer.
- **Fast signals wait on slow ones.** Substrate-primary executes a proposal decided on the previous
  state; the turn budget defers drift-driven pain; LLM-primary cannot be preempted; innate reflexes sit behind
  the ThoughtGate; the preemption scaffolding is dead code. (L1, L2, L5 and L7 on the state page are
  filed as defect issues, G8: #1176 L1, #1177 L2, #1178 L5, #1179 L7.)

**Thesis.** Concepts should become similar by what they do to the body. The word embedding stays,
as the innate prior; experience of consequences is the learned tier that can override it.

## The organizing model (biology)

The archived frame [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md)
("percept = thalamus, drives = hypothalamus") is the organizing model, extended by a binding layer:

| Biology | Role in the brain | In this line | Code names |
|---|---|---|---|
| Thalamus (relay nuclei, TRN gating) | Every sensory stream except smell passes a relay that gates, weights and routes it | **The `Receptor` registry + `AfferentTrack` relay**: one registration point that stamps identity, provenance and gain, and routes to EC, the LLM and tracks | `Receptor`, `AfferentTrack`, `AfferentEvent`, `PhysicalEventId` |
| Spinal and brainstem afferent pathways | Parallel channels with different speeds and destinations | **The tracks** (table below) | `AfferentTrack` specs |
| Insula (interoceptive representation) | "What happened to my body, and was it good" | **The consequence record**: signed deviation, pain, relief, urgency, cause | `InteroceptiveOutcome` (GL2) |
| Hypothalamus (set points, corrective drives); autonomic output | Two-sided regulation; corrective needs | **Regulatory needs**: heat as well as cold; satiation as a positive event | GL2b, GL2c |
| Cerebellum (error-driven forward model); hippocampus + neocortex (complementary learning systems: the hippocampus learns fast and specific, the neocortex slow and general by replay) | Predict the sensory consequence of an action; consolidate episodes into general knowledge | **The latent forward model**: a cortex-like slow learner trained by replaying Hippocampus traces, beside the Cerebellum's online error-driven model; the binding layer that makes concepts similar by consequence | GL4 (`latent_forward_model.md`) |

**Tracks, bio-mapped** (the proposal in [thalamic_relay.md](thalamic_relay.md); every number is an
innate prior to be reviewed; latency is ordinal, in loop passes, "FUNCTIONAL, not fibre dynamics"):

| Track | Bio mapping | Character | First engine |
|---|---|---|---|
| `nociceptive_fast` | Aδ fibres, spinothalamic "first pain" (anterolateral system) | sharp, localized, fast; may preempt a pending proposal (only as a declared rung arm, G8) | `evaluate_failures` NOCICEPTIVE events (incl. `drive:health`), the thermal nociceptor (GL2b), tool damage, sim pain injection |
| `affective_slow` | C fibres, spinoreticular / spinoparabrachial "second pain"; for DRIVE breaches, Craig's lamina-I homeostatic pathway (interoceptive afferents → parabrachial → insula) | diffuse, lingering, summing; feeds the autonomic code | the same physical event as `nociceptive_fast` (fan-out); thermal; DRIVE breaches (hunger, air hunger), which ride here only under the lamina-I reading, not as pain fibres |
| `mechano_proprio` | dorsal column / medial lemniscus | touch, proprioception (not temperature, not pain) | Minecraft world sensors via the pump, SEM `self_effect` deltas on touch / position sensors |
| `reflex` | spinal reflex arc; superior colliculus orienting | innate motor program, no deliberation | narrative keyword reflexes; the audio orienting reflex |
| `threat_low_road` | thalamo-amygdala "low road" | coarse and fast | a situation change whose learned fear clears threshold |
| `extero_detail`, `language` | cortical routes (V1→IT; auditory brainstem → inferior colliculus → A1 → belt; language cortex) | detailed, slower; reach the prompt | vision, audio percepts and DoA azimuth (auditory, not somatosensory), Minecraft events; CLI, narrator |

## The architecture

```
 PERCEPT RECEPTORS                    AFFERENT TRACKS                 THALAMIC RELAY
 body drives ─────┐               ┌─ nociceptive_fast (Aδ) ──┐
 Minecraft world ─┤  AfferentEvent│─ affective_slow  (C)  ───┤   gate · gain · provenance
 DoA / audio ─────┼─(PhysicalEvent┼─ mechano_proprio  ───────┼──► stamp · route ──► EC (per modality)
 narrator / CLI ──┤   Id: agent,  │─ reflex ─────────────────┤                  ──► forward-model context
 imagination ─────┘   seq)        └─ threat_low_road / … ────┘                  ──► LLM prompt (existing nuclei)
                                        │ preempts a stale pending proposal
                                        ▼
 SEM effects / drives ──► AUTONOMIC LAYER: consequence record (deviation, pain, relief, urgency, cause)
                          │                                   │
                          ▼                                   ▼
          forward-model TARGET (GL4)              regulatory needs (heat/cold, satiation)
                          │                                   │
                          ▼                                   ▼
  LATENT FORWARD MODEL: (context, action) → predicted consequence ──► consequence similarity
                          │                              (word prior α → learned 1−α)
                          ▼
             NAc.recommend_action (declared arm, GL6) ──► action without the LLM
```

The two inputs meet only in the forward model: the relay supplies the context (what I perceive, in
which situation, with which provenance); the autonomic layer supplies the target (what happened to
my body). Credit and the predictor join a fast pain and a slow ache on the same `PhysicalEventId`.

**Engine ≠ track, concretely.** The thermal receptor sees `arms.thermal` cross its noxious threshold.
It produces one `PhysicalEventId(agent_id, 41)` and one event on each declared track: `reflex`
(withdraw), `nociceptive_fast` (delivered now; publishes onto PainBus, where the pain-memory, Wire-2
and NAc-outcome subscribers see it once it clears their thresholds; Wire-4 cluster fear does **not**,
because its allowlist is `{drive:health, drive:oxygen}`) and `affective_slow` (delivered later,
summed, feeding the consequence record). Temperature is anterolateral, so the burn does not ride
`mechano_proprio`. "The sharp pain and the lingering ache are the same burn" is then a fact of the
data, not an inference. A `Receptor` declares *what* it is (receptor class, modality tag, encoder,
provenance kinds it may emit, gain, tracks); an `AfferentTrack` declares *when and where* its events
go (latency, priority, preemption rights, destinations, refractory in passes).

**What this line does not do.** It does not resurrect
[archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md): it binds by
**predicted consequence**, not by temporal co-activation. It does not edit the affordance
decomposer or the word encoders (an edit would fire T1-5's trigger and break the innate prior). It
does not build a percept-channel manifest "as conceived" (rejected in
[archive/percept_testbed_audit.md](archive/percept_testbed_audit.md); GL3's front-gate answers in
writing why a registry with a byte-identical first slice is not that manifest). It does not touch
`DEFAULT_CLUSTER_FEAR_FAILURE_MODES` (a hivemind wire boundary).

## The three sub-plans

### [autonomic_layer.md](autonomic_layer.md): the signed body-consequence code (GL2)

One frozen record, `InteroceptiveOutcome`, per body-consequence event (a tool invocation, or an out-of-band change detected
at `embodiment/body.py::Embodiment.evaluate_failures`): signed per-drive deviation and delta, the
unsigned pressure already recorded, drives satiated in this event, and a body-agnostic core
(nociception, drive pain, relief, harm, urgency, valence) with a `cause` and the shared event id. It
is computed by one pure function beside the existing helpers in `embodiment/sem.py` (`drive_span`,
`drive_pressure`, `drive_comfort_progress`, `classify_pain`), so no formula is derived twice; a guard
pins `relief == max(drive_relief)`. The record carries the event-identity contract above (its type is
built here, at GL2a) and an explicit provenance. On top of the record, the regulatory fixes: a
**noxious receptor** declared beside a drive (`NociceptorSpec`, a second receptor on the same sensor,
never a `pain_scale` edit on the shipped body); a **two-sided corrective need** (heat as well as
cold, with `corrective_need_intensity` kept byte-identical); **cause-keyed valence** rows in their own
key namespace (a live LLM-path change, not an additive no-op: GL2b(iii) below); and the **positive
producer** (satiation when a breach latch clears, graded relief), designed jointly with Phase 5's
relief store. It **revives** [deferred/nociception_layer.md](deferred/nociception_layer.md) (G5),
carrying its steps 2–4 rather than multiplying plans.

**Why existing infrastructure cannot do this.** `memory/encoding.py::EncodingSignals` cannot be the
record: it is a memory-encoding contract whose `__post_init__` rejects negatives, so signed deviation
cannot live there, and it exists only at capture sites, not at out-of-band body change. The tool-path
trio (`drive_pressure_before`, `drive_relief`, `pain`) rides `ToolOutput` only and records harm as
0.0. The drive specs are SHAPE-FROZEN at 1.0 (CC3), so a nociceptor cannot be a new drive field, and a
standard `failure_modes:` entry would flood PainBus every tick and bypass B8. Everything else rides:
ReactionBus, the reserved `"satiation"` `ReactionKind`, the distributor, the entity-owned breach
latch, `classify_pain`, B8's one parser. No new bus.

### [thalamic_relay.md](thalamic_relay.md): `Receptor` registry + `AfferentTrack`s (GL3)

One registration point for every percept source. A `Receptor` declares its receptor class, modality
tag (validated against one registry), encoder kind, the provenance kinds it may emit, its tracks,
clock and relay gain; the embedding space is **derived** from the existing
`similarity/encoder.py::encoding_geometry_tag`, never authored. The relay pulls/drains receptors,
builds `AfferentEvent`s (required non-empty `agent_id`, required provenance), routes to EC through
the unchanged encoders, to the LLM through the existing nuclei (`BioEnrichmentPipeline`,
`ThalamicGate`, the audio fold), and to tracks, and stamps EC encoder provenance. **G8: the registry
enters with its consumer.** It lands in one stage with provenance on the write path, whose consumers
are GL5's experiment and the forward model's contamination guard; the existing body channels
(`runtime/substrate_proposal.py::_SUBSTRATE_CHANNELS`, the de facto body-world thalamus today,
registered in code order: interoception, audio, world) are wrapped byte-identically inside that
stage. The receptor stamp (`EntorhinalCortex.record_encoder_provenance("receptor:…")`) persists to
`ec.json` and, through `hivemind/cli.py`, into bundle manifests (`hivemind/bundle.py`
`manifest["encoder_provenance"]`): a wire format, so the stage's gate covers EC save and bundle
manifest bytes. Per-node provenance must enter `hivemind/bundle.py::_BUNDLE_EC_NODE_FIELDS` (else the
export scrubs it), the `hivemind/merge.py::ec_merge_aligned` node fold, ingest node validation and
`EC.save`/`load`; a credit refusal for non-`experienced` nodes leaves non-EC eligibility ids
(`tool:<name>` from `ToolPainBridge` → `TemporalCreditDistributor.record_event` →
`update_eligibility`) out of scope, or tool credit dies. The tracks are a per-agent scheduler on the
loop's own clock: a total order `(due_pass, −priority, seq)`, `seq` assigned at the scheduler's drain
point (the event-identity contract), refractory counted in passes, and a typed preemption handler
that drops a stale pending proposal and re-arms in the same pass (never idles, never counts against
D13). The first track slice (GL3.B3) is the thermal dual-track fan-out, with no preemption, built on
GL2b's `NociceptorSpec`; nociceptive-fast preemption (GL3.B4) enters only as a declared 1.4 rung arm
(G8).

**Why existing infrastructure cannot do this.** PainBus and ReactionBus are synchronous pub/sub on
the publisher's thread with wall-clock refractories: no latency, no per-signal destination subset,
no preemption rights, no single event on two channels at different times; teaching PainBus those
would turn a transport into a scheduler for every publisher at once (T1-4/T3-9 blast radius).
`runtime/preemption.py::PreemptionCircuit` has the right vocabulary but wall-clock cooldowns, no
producer and no consumer. `CompositePerceptSource` is deliberately priority-free and moves one
percept per pass. `ThalamicGate` is DN-only and vision-shaped. `TemporalEvent` is the right credit
envelope but its uuid identity and wall-clock signature are nondeterministic. No existing carrier
puts provenance on `register_substrate_node` / `update_eligibility` at write time; six sites call
encoders directly, and three silent misses (imagined-unflagged, `agent_id` None,
`perceived_intensity` never set) are the rule's threshold for pushing the invariant into a type.
The timing defects themselves (L1 stale proposal, L2 turn budget defers drift-driven pain, L5
reflexes behind the ThoughtGate, L7 dead scaffolding) are filed as defect issues and are fixable
without the registry (G8); what the tracks add is the shared identity and one scheduling model, which
three point fixes in three files cannot give.

### [latent_forward_model.md](latent_forward_model.md): the binding layer (GL4)

Given what I perceive and what I am about to do, what will happen to my body? A training example is
`(context, consequence)` joined on the event's `PhysicalEventId` (`pid`), which the tool path stamps
on `ToolOutput` beside the executor's invocation id (that `uuid4` is not persisted-stable and is never
the join key): context blocks are the word embedding of the affordance and entity (innate prior), the
sensed readings of the TARGET entity, the active situation vectors and normalised params; the target
is the GL2 consequence record, `InteroceptiveOutcome.as_vector(schema_id="ans-v1")`. Narrated pairs
(G6) train at the declared discount and are never relabelled `experienced`. The **Cerebellum** stays
the online, error-driven forward model on exact keys, with its target fixed (it currently learns the
modulator owner's absolute readings, read before the self-effect reaches the body). A slow,
generalising `ConsequencePredictor` is the cortex-like half of complementary learning systems: it
refits from replayed Hippocampus `"loop"` traces (the fast, specific store; no new store) at session
end. `BioStack.on_session_end` is today called only by `simulation/minecraft_harness.py` and scripts;
the orchestrator / cradle path saves through the `runtime/agent_factory.py` shutdown
(`bio_stack.save_cerebellum()`) and never runs it (so `distributor.cleanup_session` is skipped there
too), so S3 wires that seam with its own test. Kernel ridge first (closed form,
deterministic, numpy + JSON via `atomic_write_json`; no pickle, no `.pt`). EC consumes predictions
through an opt-in `consequence` modality; concept similarity becomes a read-time blend
`α·cos_word + (1−α)·cos_predicted`, with α falling as nearby experience accumulates. **Honest
naming (G2):** with a fixed target this is supervised regression in a JEPA shape; the interface stays
JEPA-shaped so a learned target encoder can slot in later, and no ledger row says "JEPA" until it
does.

**Why existing infrastructure cannot do this.** `Cerebellum.predict` is an exact dict lookup on
`ModelKey`: it cannot generalise to an unseen (entity, affordance, situation). `CausalLink`,
`cluster_reward_bias` and `cluster_fear` hold scalars per seen key; none maps a context embedding to
a vector consequence. `anticipatory_pre_activate` predicts *when* an event recurs, not what an action
does. The word nodes' similarity is fixed by mpnet, which orders flame/water wrongly. The only thing
that generalises to an unseen key today is EC similarity of a SITUATION, and an action-object pairing
is not a situation. What rides: the Cerebellum (training-signal site), `Executor._stamp_invocation`
(stamps the pid for pair assembly), Hippocampus traces (replay buffer), `on_session_end`
(consolidation seam, once wired on the cradle path), EC
per-modality matrices + `_sensor_embed` + geometry tags (node formation). New: one function
approximator, the context builder, the modality registration, and train/eval scripts.

## Stage map

Legend: **[3L]** three-lens code review (Executor, Architecture, Wire integrity; [CODE_REVIEW.md](../CODE_REVIEW.md))
on every sub-plan and every `src/` change; **[4L]** four-lens design review, verbatim into
`docs/plans/rationale/grounding-<part>/` for a mechanism plan or `docs/experiments/rationale/<slug>/`
for a prereg; **[⟲]** ledger trigger walk; **[rig]** live re-run; **[fence]** waits for the 1.3.2
decomposition slices. Owner decisions for each stage are asked together at its start (§Open owner
decisions).

| Stage | What | Depends on | Reviews / runs | Falsifiable exit gate | Guard tests |
|---|---|---|---|---|---|
| **GL0 Truth** (now; docs, tests, docstrings) | (a) the #1120 honest test PR; (b) the truth pass (§GL0 below) with one `[Unreleased]` docstring line; T1-5 re-scope + `scripts/lint_claims_sync.py::NO_INDEX_ENTRY` in the same diff; dormancy markers and trigger re-points; (c) the state page [../wiring/body-and-word-worlds.md](../wiring/body-and-word-worlds.md) + a `docs/wiring/README.md` line; (d) a GitHub issue per verified audit defect not yet filed (already filed: #1161, #1125, #1124, #1156, #1159, #1074), including the timing defects L1, L2, L5 and L7 (G8; filed as #1176 L1, #1177 L2, #1178 L5 and #1179 L7, each fixable without the registry). Its own PR(s). | none | [3L] on the `src/` docstrings; one Executor + Wire-integrity read on (a). No [⟲]. | The two old strict xfails flip for the named cause (commit 1); the four new strict red gates are red (three in `TestIssue1120RedGates`, one of them documentary at the retired 0.40, plus `TestIT2NoFalseTransfer::test_credit_widening_does_not_absorb_water`, #1181); `lint_claims_sync`, `lint_ledger_format`, `lint_claude_md_invariants`, `test_lane_rosters` green; every truth item fixed or explicitly deferred. | `tests/integration/test_affordance_transfer.py` (slow lane; `scripts/lane_rosters/slow.json` regenerated) |
| **GL1 Paper** (now ‖ GL0) | This umbrella + three sub-plans, each with its written front-gate answer; DECISIONS.md `2026-10-07 — The grounding line` (G1–G8, the event-identity contract, engine ≠ track, logical tracks); roadmap §"The grounding line" between §Phase 5 and §JEPA (§JEPA becomes a pointer), T8/T9; README; banners (§Consolidates). Offline: the **L0 gate** from `grounded_word_binding.md` (innate-prior quality; its old PASS consequence RETIRED, G5) and a **name-vs-consequence collision census** over all shipped affordances (committed script + JSON). | GL0(c) | [4L] on **this umbrella** and on each sub-plan (the umbrella's own four-lens review is a T8 item); [3L] on each doc. | No unfolded DO-NOT-BUILD in any of the four-lens reports, the umbrella's included; the census quantifies collisions (pairs sharing a 0.44 node with opposite declared effect); L0 recorded PASS/FAIL against its frozen criterion. | the census script's known-answer check (`touch` on blanket vs fire pit) |
| **GL2a Autonomic, record-only** | The consequence record + pure function; the leaf `embodiment/event_id.py` (`PhysicalEventId`) and the per-agent `EventSequencer` (persisted, resumed past the saved maximum); one additive `ToolOutput` field (the record, with its `pid` beside the invocation id); `Embodiment.drain_outcomes()` for out-of-band change, latched like channel 2 (entry/exit/deepen, no per-tick flood), bounded drop-oldest with counted drops, ephemeral wrappers never queue; the lock for the orchestrator-thread edge. Written to a trace and to `EncodingSignals.extra["interoception"]` only; **no reader acts on it**. Files: exactly G1's exempt set. | GL1 [4L] on `autonomic_layer.md`; **#1125**: land after it, or carry its fix as commit 1 (never ship a record that omits `arms.thermal`) | [3L]; mypy on `runtime/executor.py`; [⟲] on T1-16 (below). | Scripted cradle sequence (cool_air ×2, warm_self ×2, touch) matches a hand-computed table; deletion probe: removing the producer changes only the trace and the persisted `extra`; selection golden and encoder golden byte-identical; an **executing** survival check (`tests/unit/test_water_trial_smoke.py` and `tests/unit/test_exp61_run.py`) passes unchanged. (A verdict reproduction is not a gate here: `compute_verdict` re-reads committed JSONL and never runs the producer.) | `relief == max(drive_relief)`; outcome present iff a body is attached; latch semantics (reuse `tests/unit/test_transition_drive_pain.py` shapes); `test_agent_loop_selection_golden.py`, `test_encoder_golden_v1.py`, `test_decision_provenance.py` unchanged |
| **GL2b Regulatory fixes** | (i) heat corrective need (direction-aware wrapper; `corrective_need_intensity` byte-identical). (ii) burn = pain via `NociceptorSpec` + a `failure_pain_kind` rule (band `noxious` ⇒ NOCICEPTIVE; a `noxious` band-grammar entry in `docs/user/tool_side_effects.md`); the nociceptor's PainBus `failure_mode` is specified as `drive:<sensor>:noxious`. **Refractory collision:** ReactionBus keys its refractory on `f"{kind}:{source}"` and `reactions/compat.py::pain_signal_to_reaction` sets `pain_detector:external_signal` for all body pain, so the drive Reaction (0.04) and the nociceptor Reaction from one `evaluate_failures` call coalesce within 0.5 s; PainBus keys on `(entity, failure_mode)`. (iii) cause-keyed Wire-2 rows, in a **distinct key namespace** (cause and sufferer keys share a shape today and would collide, e.g. `rusty_sword`); not additive at the reader (below). The cause is attached per sensor, only where B8 marks that sensor harmed by this call: `evaluate_failures(*, cause=)` is body-wide, and a lingering breach on another sensor never inherits it; actor-path B8 gating is specified in `autonomic_layer.md`. (LLM-primary `note_active_clusters` is not this plan's: it moves to GL3 or is filed as its own defect.) One issue + PR each. | GL2a; **#1161 resolved first** for (ii) | [3L] each; [⟲] each; Exp 42 re-run as a sim if (ii) or #1161 lands. | Strict red gates flip: a hot infant emits a `heat` need; fire-pit `touch` yields a NOCICEPTIVE PainSignal at or above the learning threshold **as defined by the stage-start calibration decision** (autonomic_layer.md proposes ≥ 0.4, every PainBus learner) while `warm_self` and `*_safe` yield zero nociception; an actor affordance with a `target_effect` on the AUT writes a Wire-2 cause row keyed on the actor's noun (a **new fixture authored in GL2b**: no shipped actor affordance can hurt the AUT; the dragon's `fire_breath` declares no `target_effect`); a bystander affordance never stamps a cause. Deletion probe per fix. | B8 causer-vs-bystander test extended to `noxious`; (ii)'s golden table covers **both** refractories (ReactionBus and PainBus) for one call; `corrective_need_intensity` grid equality on every shipped body; Minecraft bodies emit identical `_read_drive_states` |
| **GL2c Positive producer** | Satiation (latch clears after deprivation; once per breach episode) and graded relief, as a positive event; **one joint four-lens review** with Phase 5's relief store (+ `fear_learning.md` Exp A, `coding_world.md` C3's reserved sign). Off by default behind a `maxim config` switch. | GL2a; R4's routing audit has decided the selection surface; E4's overreach set read | [4L] joint; [3L]; **[⟲] [rig]** (G7): fires T1-11, T1-12, T1-13 (its #888 discharge lapses), T1-14 and T1-15 **whatever the routing**, because `minecraft_player`'s `oxygen`/`health` latches clear on every surfacing or regeneration after a latched breach (Exp 60's own `escape_water` contingency) and `food` (`satisfaction_threshold: 16`) and the `d1` of `minecraft_bench` and `minecraft_bench57` (0.3) are entropic. Lands only with a **batched live re-run of Exp 60/61/62** plus the T1-11 and T1-12 arguments or re-runs, in its own rig slot after the 1.3.2 live Exp 60 re-run, never stacked on it. | Flag off (the default): the executing survival checks (`test_water_trial_smoke.py`, `test_exp61_run.py`, `test_r3_run.py`) pass unchanged. Flag on: exactly one satiation event per breach episode on a scripted Minecraft eat-after-starvation sequence; Exp 60/61/62's frozen gates still PASS on the rig, else the rows are recorded BROKEN and block the release. | strict red gate "deprive→satisfy publishes one positive satiation event"; existing zero-bias guards in `test_nac.py`; the staged-donor refusals **restated, not deleted**: `exp61_run.py::donor_sanity_staged` (`reward_bias`, `cluster_reward_bias`, `links`, `event_outcome_welford`) and `r3_run.py::_R3._boundary` (`reward_bias`, `links`) |
| **GL3 Relay + tracks** (G8 order; stage IDs per [thalamic_relay.md](thalamic_relay.md) §6) | **GL3.B0, census + red gates** (tests only; runnable now, inside the fence): offline inventory of every ingress into the state page (including every pain producer that bypasses `evaluate_failures`, state page §3, and every `evaluate_failures` caller with its thread) plus the timing red gates, and a frozen characterization of breach→protective-action latency in passes (GL3.B4's latency gate needs it frozen first). **GL3.B1, registry+provenance** (lands only with its consumer): the `Receptor` registry, existing receptors as byte-identical wrappers registered in code order (interoception, audio, world: `_SUBSTRATE_CHANNELS`), landing **with** provenance on the EC / NAc write path, whose consumers are GL5's experiment and the forward model's contamination guard. **GL3.B2, handoff to GL4:** the registry exposes `(pid, context, action)` per pass for the forward model's join on `pid`. **GL3.B3, the first track slice:** the thermal dual-track fan-out (`nociceptive_fast` + `affective_slow`, one shared pid reaching credit and GL2's record), behind a flag, **no preemption**; needs GL2b's `NociceptorSpec`; the seq handover from GL2a's sequencer to the scheduler's drain point is its stage gate. **GL3.B4, preemption** (a declared 1.4 rung arm, or it waits for one, G8): `nociceptive_fast` preempts a pending substrate proposal in substrate-primary, behind a flag, with transduction drained in place so the Exp 58 W-4 order (encode → `note_active_clusters` → pain → fear read → select) is unchanged; its same-pass reselect re-runs `propose_via_substrate` (a second `note_active_clusters` / Wire-4 read in one pass), which the gate design pass must settle. Later, each only when a rung names it: GL3.B5 text receptors; GL3.B6 the `reflex` track out of the ThoughtGate; GL3.B7 LLM-primary + the sensory tracks and the wake hook (LLM-primary `note_active_clusters`, from GL2b, if not filed as its own defect); GL3.B8 hardware edges. | GL3.B0: none (runnable now). GL3.B1 onward: GL2a; agent_loop slices for tick scheduling and the orchestrator slices for word-path receptors [fence]; GL3.B3 also GL2b(ii) | [4L] on `thalamic_relay.md` (must answer the "no manifest" and SUBSUME decisions in writing); gate design pass before GL3.B4 (preemption is a gate); [3L]; [⟲] per stage. | GL3.B1: `(modality, cluster_id, geometry_tag, margin)` sequence byte-identical over `minecraft_player`, `infant_operant`, `reachy_mini` trajectories, **and** EC save (`ec.json`) and bundle-manifest bytes identical apart from the declared receptor stamp and per-node provenance fields; deletion probe (bypass the relay) fails the golden; a narrated record cannot be relabelled `experienced`; `tool:<name>` eligibility still credits. GL3.B3, flag off: both goldens byte-identical cross-process and cross-`PYTHONHASHSEED`; flag on: one scripted burn yields two track deliveries with one pid, joined by the autonomic and credit records; seqs unique and monotonic across the handover and a save/load; narrator writes touch the latch from the loop thread only. GL3.B4, flag on: the stale-proposal red gate flips, the preempting event's `seq > basis_seq`, reselection in the same pass, breach→protective-action latency in passes strictly below the frozen GL3.B0 characterization; livelock probe (≤ 1 preemption per physical event over 20 passes); deletion probes on the check and the reselect. | `tests/unit/test_afferent_tracks.py` (order, identity, pass refractory, inbox sort); `_loop_harness` arms; the timing red gates (stale proposal, turn gate skips drift-driven pain, reflex behind ThoughtGate), which flip with the L1/L2/L5 defect fixes or with the tracks, whichever lands first |
| **GL4 Latent forward model** | **S0a** (paper, runs now): Phase 5's mandated audit (can the Cerebellum or `anticipatory_pre_activate` carry "how far pain is"?). **S0b**: a consequence-pair data audit with prereg'd floors over a fresh capture (a sim run). **S1**: Cerebellum target fix (target = consequence dims, key = target entity; payload `"1.2"`; write side only; its other readers named and kept consistent: `tool_bridge.py`'s `get_confidence(self._entity.full_path…)` on `sim_cerebellum` (owner-keyed), `Cerebellum.import_state` (loads an unknown version with a warning today: refuse a newer one instead), param bucketing on the owner's sensor ranges, and the `motor.py` / acting-coach programs). **S2**: offline predictor + eval scripts (kernel ridge; held out by key and by entity). **S3**: session-end refit + persistence + contamination guard + shadow log, including the cradle seam (an `on_session_end` call at the `agent_factory.py` shutdown, its own wiring + test; this also starts running `distributor.cleanup_session` on that path, which the ledger walk must cover). **S4**: opt-in `consequence` EC modality + similarity tiering; `hivemind/bundle.py` exports every modality, so excluding `consequence` **is** a bundle.py edit (fires T1-14's bundle-scrub wording), and `merge.py::SENSOR_MODALITY_THRESHOLDS` gains `consequence` or merge refuses it. (LFM's S5 is GL5.) | S0 ‖ GL2 (proxy fields, gaps stated); S1 after GL2a only (G1's exempt set); S2 after S0 PASS; S3 after S1 + S2 [fence]; S4 after S3 [fence] | [4L] on `latent_forward_model.md`; [3L] on S1, S3, S4. [⟲]: S1 walks the Cerebellum readers; S1's `extra["context"]` on the loop capture fires T1-16's memory-record-shape wording (as GL2a's does); S4 fires T1-14 by wording. | S0: written audit verdict; S0b's key/class floors apply to the cradle: PASS needs ≥ 500 executed invocations with a consequence, ≥ 30 keys, ≥ 3 consequence classes, ≥ 5 name–consequence dissociation pairs, 100 % joinable; FAIL (< 200, < 2 dissociations or < 2 classes) **stops the line**. Minecraft gets the graded-oxygen floor instead (≥ 100 executed `escape_water` / `move_to` invocations over ≥ 3 oxygen bands with a non-zero Δoxygen spread); its FAIL removes the Minecraft arm, not the line. S2: on held-out dissociation keys, full model sign accuracy ≥ 0.8 and ≥ 0.2 above word-only, and ≤ 0.05 worse than word-only elsewhere; word-only ≥ 0.8 there means the split is mis-built (refuse). S4: flag off, EC dumps for all existing modalities byte-identical. | `test_consequence_no_contamination.py` (a pair without a `pid`, a narrated record relabelled `experienced`, a narrated pair without the declared discount, and a geometry mismatch each RAISE; backlog row M41 until S3 lands); two-process determinism with differing `PYTHONHASHSEED`; Cerebellum wiring test through the real call path + deletion probe; format round-trip in `test_persistence_compat.py`; no `consequence` nodes in an exported bundle |
| **GL5 First claim (T9)** | Prereg: "consequence similarity transfers learned value between affordances whose names differ and blocks it between affordances whose names collide". Substrate-primary, no LLM in the action path. Three arms closing each other's confounds: cradle name-contradicts-physics (a "warm rug" that burns, a "fire stone" that is cold; physics author blind to arms; effects derivable from each object's own declaration); the jet triad (train `flame_jet`, probe `fire_breath` and `water_jet`: the ordering must invert, `water_jet` must not inherit the aversion); Minecraft graded Δoxygen/health (game-native physics). Results are reported with AND without narrated data (G6). Successor candidate to T1-5; it is LFM's S5. | GL4 S2 (S4 for the prior term); R4's surface | [4L] on the prereg (all four: a new claim); [3L] on the harness; prereg on `main` before data; dry run before run. | Pre-registered gates. EARNED → a new Tier-1 row; T1-5 → `SUPERSEDED by` it. A recorded null ships as a null. | per prereg |
| **GL6 Selection consumer** | Only if GL5 earns: the predicted consequence enters selection as a **declared arm** (M10 fingerprint), never a silent default; untried tools only. | GL5 EARNED; E-ladder state | [4L] + [3L]; **[⟲] wide**; [rig] re-runs of Exp 56/60/61/62 after E3's campaign. | The rows the walk names are re-run and MAINTAINED, or the arm stays off by default and that is recorded. | per stage |

**Critical path.** GL0 → GL1 → GL2a → {GL2b ‖ GL4 S0 → GL4 S1 (after GL2a only) → GL4 S2 (after
S0b's data)} → GL2c (+ R4 audit) → {GL3.B1 registry+provenance ‖ GL4 S3 (after S1 + S2) → GL4 S4}, both
after the fence → GL5 (needs both: provenance is its input) → GL6. GL3.B0 (census + red gates, tests
only) is not gated by any of these: it runs now, inside the fence, and must be done before GL3.B1.
GL3.B3 (the fan-out, after GL3.B1 and GL2b(ii)) and GL3.B4 (preemption) are off the critical path:
GL3.B4 enters as a declared 1.4 rung arm when a rung names it (G8), and the L1/L2/L5/L7 defect issues
are fixable without it. GL0, GL1, GL3.B0, GL4 S0a (audit) and the paper run now; GL4 S0b needs a fresh capture (a sim
run). GL2a and GL4 S1 may run before 1.3.2 ships (G1's exempt file set); every other `src/` stage
waits for the decomposition slices that touch its files.

**Rig budget (Track C, inserted as its own items).** GL2c's batched live re-run of Exp 60/61/62
(+ the T1-11 and T1-12 arguments or re-runs) takes one slot, after the 1.3.2 live Exp 60 re-run and
between E-rung campaigns, never during one (divergence rule). GL3.B4 (preemption), as a declared rung
arm, runs inside that rung's own campaign; GL3.B3 needs no rig slot (flag off by default). GL2b(ii)'s Exp 42 re-run is a sim. GL6's re-runs come
after E3.

### GL0 detail: the truth pass

Each correction is a claim fix with no behaviour change. "src" items need the `[Unreleased]` line;
the PR title is `docs(...)`/`chore(truth): …` so `lint_fix_touches_tests` does not apply.

| # | Where | Correction | Kind |
|---|---|---|---|
| 1 | `integration/bio_enrichment.py::BioEnrichmentPipeline._query_hippocampus_traced` docstring | the "fire → pain" graph path is the intended path, inert in production (D6, T3-7) | src |
| 2 | `archive/affordance_concept_transfer.md` | dated CORRECTED banner: components never form nodes; transfer is compound-to-compound by name (0.601); `[DANGEROUS]` could never fire (#910); no caution behaviour was measured; 0.40 retired, production 0.44 | docs |
| 3 | `docs/plans/README.md` Archive entry for that plan | "cross-entity transfer is compound-name level only (#1120); behavioural claim pulled (T1-5)" | docs |
| 4–5 | `docs/wiring/engram-formation.md` §1/§3 (+ `roadmap_1_4.md` §Phase 5 R4 "≤0.20 nudge") | the `tool:*` read exists, but no production path writes a positive value: 0 on all 108 decisions in `exp61_pairs` + `exp62_rows`; E4 drift was measured 2026-10-06, NO COLLAPSE, 0.44 already spans concepts | docs |
| 6–7 | `docs/agents/bio-memory.md` gotchas, EC table | imagined entities ARE encoded, without provenance; frozen modalities are `{interoception, audio, world}` | docs |
| 8 | ledger T1-5 claim + Regression guard | compound-name level, no component path, no behavioural transfer; guard → `tests/integration/test_affordance_transfer.py` after the #1120 fix; status per owner decision; edit `NO_INDEX_ENTRY` in the same diff | docs + `scripts/` constant |
| 9–10 | `imagination/trigger.py::encode_entity_affordances`, `ImaginationTrigger._encode_entity_affordances` docstrings | only production caller is the orchestrator; "shared components enable cross-entity transfer" is false | src |
| 11 | `memory/atl.py`, `memory/semantic_types.py` module docstrings | "in the brain the ATL is a cross-modal hub; here it is not" (keep `Bio-mapping: FUNCTIONAL`) | src |
| 12 | `embodiment/sem.py::EntropicDriveSpec` docstring + field comment | no positive Reaction is emitted; `satisfaction_threshold` only clears the latch (docstring only; the class is SHAPE-FROZEN) | src |
| 13 | `bio-memory.md` `_reward_bias` clamp invariant; ledger T3-19 rationale | pain avoidance runs through Wire-4 `cluster_fear`, negative causal links and `percept_valences`; edge valence is Dormant | docs |
| 14 | `bio-memory.md` SCN invariant | `anticipatory_pre_activate` is Dormant since 2026-05-26, zero `src/` callers | docs |
| 15 | `bio-memory.md` §1 mental model | two disconnected chains; link to the state page | docs |
| 16 | `memory/hippocampus.py::Hippocampus.retrieve_cross_modal` | `Dormant since 2026-10-07`, resurrection trigger: this line's binding stage | src |
| 17 | `memory/episode.py::apply_hebbian_on_close`; `docs/bugs/README.md` D6; ledger T3-7 | re-point the resurrection trigger from "the 1.3 fabric stage" to this line's GL4 (or the fabric if it revives first) | src + docs |
| 18–19 | `README.md` bio-systems table (also PyPI long_description); `docs/plans/glossary.md` | SCN: time-anchored credit fallback, not anticipatory credit; Angular Gyrus: math layer, not cross-modal binding | docs |
| 20 | `docs/simulation.md` Cradle three-layer sensation | the three cradle layers do reach PainBus, but PainBus is not one convergence point (several pain producers bypass `evaluate_failures`, state page §3); reaching NAc learning depends on intensity and class; the infant burn (max 0.2) does not; cite #1161 | docs |
| 21 | `docs/lessons/scn-temporal-coupling-eligibility.md` | dated note: no cross-entity transfer was measured; the cited validation never ran | docs |
| 22 | new `docs/wiring/body-and-word-worlds.md` | the state page | docs |
| 23 | `integration/bio_enrichment.py::_annotate_affordance_valence` docstring | lookup is by exact chunk name; `[effective]` is unreachable in practice | src |
| 24 | `similarity/ec.py` module + `EntorhinalCortex` docstrings (NIT) | "multi-dimensional signature similarity; completion is per modality, no cross-modality comparison" | src |

Also: correct the "deferred until a second body" lines for the perception half (G4) in the roadmap
and CLAUDE.md's Active-initiatives line (token budget permitting; add no CLAUDE.md routing row).
External (owner action, Maxim-web kickoff): pymaxim.bio `systems/concept-decomposition` and
`concepts/architecture` likely carry the transfer framing (**UNVERIFIED**).

**The #1120 test fix.** Commit 1 is the lookup fix only (the controls read the node ids returned by
encoding the fountain, not `atl.recall(name="water")`); the two strict xfails XPASS, so their markers
are removed in that commit. That is the flip, for the named cause. Commit 2 pins the production
threshold (`ECConfig()`), replaces `_find_substrate_concept` with a helper over
the production `_make_aff_encoder(...).encode_decomposed`, makes IT-1's transfer read go through the
mage's own encoding result (the fountain anti-vacuity arm was removed: under credit the fountain does
share, see the IT-1 docstring and #1181), removes the two `>= 0` assertions that
cannot fail (#910), and adds four strict red gates: `water jet` and `flame jet` share the mage's `jet`
node at 0.44 (flips only if word-level sharing stops); running-mean drift at the retired 0.40
collapses the fountain into `fire breath` (documentary: pins why 0.40 was retired; no fix is planned);
components never get their own nodes (that flip is an affordance-decomposition change and fires the
GL5 successor's re-run trigger); and credit widening lets `water jet` complete into `fire breath`
(`TestIT2NoFalseTransfer::test_credit_widening_does_not_absorb_water`, #1181). IT-3's name, IT-5
and IT-6's encoder dependence and the module docstring are made honest. Then `#1120` closes and the
gate defects move under this line.

## Blast radius

**No EARNED row depends on the word path.** The line's cost lands where it touches body-world
shared code. Rows by stage, fired by their `Re-run on:` wording
([behavioral_graduation_candidates.md](behavioral_graduation_candidates.md)):

| Stage | Rows that fire | Handling |
|---|---|---|
| GL0 | none (docs, docstrings, tests) | none |
| GL1 | none (paper, offline) | none |
| GL2a | **T1-16** ("the memory record shape") by wording: GL2a adds `EncodingSignals.extra["interoception"]` to every loop capture, and `EncodingSignals.to_dict` flattens `extra` into the persisted trace; Exp 10's memory-shape wording likewise | structural walk on T1-16 (`_rank_by_relevance` reads no `extra` key) and on Exp 10's memory-shape wording; byte-identity goldens for selection and encoding; grep in the PR showing no other named symbol edited |
| GL2b | (i) T1-6 functionally (a new need changes the interoception encode and B8's attribution inputs on hot cradle bodies), T1-13 "drive-activation floor" wording, T3-9; (ii) T1-6 (Exp 42), T1-9/T1-10 only if a shared `infant_humanoid*` body changes (a variant body avoids it), T1-4, T3-9; (iii) **a live LLM-path change**: negative cause rows reach `NAc.get_percept_aversions` → `GatingContext.learned_aversions` → `TextSalienceScorer` → ThoughtGate and enrichment (`integration/bio_enrichment.py`); consumers to walk: `nac.json` save/load, `hivemind/bundle.py` scrub (`_IDENTIFIER_TOKEN`), `hivemind/merge.py` `_merge_mean_clamped` / `_TIGHTEN_ONLY_FIELDS`, `hivemind/ingest.py` bounds / report / `keep_agent_rows`, `analysis/substrate_diff.py`, the Exp 61 harness's `FEAR_MODE in key` check, and `record_outcome_full`'s context → link `context_factors`; fires T1-14 (bundle scrub) by wording, T1-2 (already STALE) and Exp 37 prompt text | per-fix walk written on the rows; Exp 42 sim re-run |
| GL2c | **T1-13 by its explicit lapse, T1-11, T1-12, T1-14 and T1-15, whatever the routing** (G7), T1-4, T3-9 | off by default; batched live re-run of Exp 60/61/62 + the T1-11 and T1-12 arguments or re-runs in its own rig slot; staged-donor refusals restated |
| GL3.B0 census | none (tests and docs) | none |
| GL3.B1 registry+provenance | by wording, even byte-identical: T1-6 ("SensorEncoder / EC-interoception change"), T1-10 (EC persistence: the receptor stamp and per-node provenance reach `ec.json`), T1-11 and T1-12 ("SensorEncoder / EC world-modality change"), T1-15, T1-14 (bundle manifest `encoder_provenance` and the `_BUNDLE_EC_NODE_FIELDS` scrub), T1-13 (credit path) | **owner decision at the stage's start**; strict default: the rows fire, offline guards re-run, a committed machine-readable exception for rig re-runs; credit refusal ships default-OFF and never refuses non-EC (`tool:<name>`) eligibility |
| GL3.B2 handoff | none (no consumer) | none |
| GL3.B3 fan-out | flag off: nothing (goldens prove it); flag on: a new condition no earned row may be cited under (incl. narrator-pain timing under `--sim`); T3-9 arguably; T1-4 pinned unchanged | default OFF; walk T3-9 before building |
| GL3.B4 preemption | flag off: nothing; flag on: the declaring arm's condition, T3-9; default-on would fire T1-13/14/15 | declared rung arm only (M10); its re-run is the declaring rung's campaign; batched with GL2c's rig slot if ever default-on |
| GL3.B6 `reflex` track | T3-9 (changes its firing counts) | walk before building |
| GL3.B7 LLM-primary + wake hook | the wake-source hook fires T1-13/14/15's idle-gate wording | scheduled with their rig re-runs |
| GL3.B8 hardware edges | touching `DoAFeed` fires T1-7 and T1-10 | rig only; verify actuation first |
| GL4 S0–S2 | none as designed (no consumer) | none |
| GL4 S1 | T1-16 by wording (`extra["context"]` on the loop capture, with file growth from 768-d + 384-d vectors per trace) | as GL2a |
| GL4 S3 | the cradle `on_session_end` seam starts running `distributor.cleanup_session` on the orchestrator path: walk the cradle rows (T1-6, T1-9) | its own wiring test + walk |
| GL4 S4 | T1-14 (bundle scrub wording: excluding `consequence` is a `bundle.py` edit); S4's frozen-modality pin could be read as an EC change | keep consequence nodes out of the shared `frozen_centroid_modalities` default and out of bundles; `SENSOR_MODALITY_THRESHOLDS` entry or merge refusal |
| GL5 | a prior term inside `recommend_action` fires Exp 45/56/60/61/62 even default-off | owner decision: a composing seam before `recommend_action` (no trigger) or accept and budget the re-runs |
| GL6 | T1-3, T1-7, T1-10–T1-15 | re-runs after E3 |

The 1.4.0 heartbeat walk (T4) covers every Tier-1 row regardless. A new trigger-table category,
**"Consequence-code / autonomic producer change"** (naming T1-11, T1-12, T1-13/14/15, T1-6, T1-9 and T3-9;
DROPPED rows T3-6, T3-10 and T3-15 carry no `Re-run on:` clause and cannot fire), is added in GL1 so
an autonomic change does not match only by wording.

## Release thresholds this line adds

- **T8: grounding truth and contracts** (engineering only, no behavioural claim, gates 1.4.0,
  modelled on T7). GL0 merged (with the L1, L2, L5 and L7 defect issues filed, G8); this umbrella
  **and** the three sub-plans each four-lens design-reviewed (the umbrella's own review included,
  GL1) with no unfolded DO-NOT-BUILD; the dormancy markers and trigger re-points done.
- **T9 (conditional): the first grounding claim** (GL5, and GL6 if it enters) under the T5/T6 rule. It
  ships in 1.4.0 only if recorded by the time T1–T4, T7 and T8 hold; a recorded failure ships as a
  failure; it is **never co-headlined with E3** (two may-fail lines in one headline make a null in
  either unreadable).

## Behaviour tiers

The line's tier statement: **the word embedding is the innate prior; consequence is learned.** Every
GL stage declares its tiers in its PR body ([deferred/behavior_tiers.md](deferred/behavior_tiers.md)).

| Behaviour | Tier |
|---|---|
| Cause attribution on a consequence event (the producer knows who did it); a bystander never becomes a cause | invariant |
| Receptor registration validation (unique id, registered tag, non-empty `agent_id`, provenance within the declared kinds) | invariant |
| One physical event, one `PhysicalEventId` across its tracks; the scheduler's total order; no wall time in track decisions | invariant |
| A preemption drop re-arms and never idles; at most one preemption per physical event | invariant |
| Predictor pairs join on the event's `pid`, never the invocation id; a `narrated` record is never relabelled `experienced` and trains only at the declared discount; geometry mismatch refuses | invariant |
| Narrated consequences receive credit only at the declared discount (G6); imagined nodes: refuse or discount, decided at GL3.B1's start; non-EC eligibility ids (`tool:<name>`) are outside the refusal | invariant |
| The narrated discount's value | innate prior (owner decision at GL4 start; strict default a small discount) |
| Word-embedding similarity before experience (α = 1) | innate prior (follow-up trigger: the GL5 result) |
| Nociceptor thresholds; the need→affordance affinity table; the core valence weighting; every `AfferentTrack` number (latency, priority, gain, thresholds, refractory) and track membership per receptor; receptor gains; kernel, block weights, the 0.85 consequence threshold, the α curve | innate prior (pinned, declared, follow-ups filed with triggers: `adaptive_nociception.md` for gains, `nociception_layer.md` step 5) |
| Predicted consequences, consequence clusters, α decay; what tracks deliver (fear, bias) | learned |

New invariants enter as `[engineering]`. None is `[behavioral]` until GL5 earns it.

## Rules this line creates, and what enforces each

Per "Enforced, or on the backlog", each rule names its check or takes a row in
[outstanding.md](outstanding.md) §Mechanization backlog in the same commit. This line adds M39, M40 and
M41 and amends M10; a structural guard is cited as "structural guard: <type or test>", never with an
M-number.

| Rule | Check |
|---|---|
| Every percept enters EC through the relay | **M39**: an AST lint that EC encode / `SensorEncoder.encode_*` / `LinguisticEncoder.encode*` are called only from the relay plus an allowlist naming each current caller with its reason (the M38 pattern) |
| Tracks are logical channels, never OS threads | **M40**: no `threading.Thread` / `asyncio.create_task` in the tracks module; hardware-edge adapters allowlisted |
| A consequence event carries its cause; tracks share one event id | structural guard: `PhysicalEventId` and the frozen record types with required keyword-only fields, so forgetting is a `TypeError`; no row |
| Predictor rows are executed and provenanced; narrated is discounted and never relabelled; no curated pairs | **M41** until GL4 S3 lands; then structural guard: the inherited contamination CI test |
| No grounding stage is active in an E1–E3 arm unless declared | amend **M10** (the parallel-line toggle in the E-rung harness fingerprint) with the grounding flags; no new row |
| Every engine census entry is registered | the GL3.B0 census test (`tests/unit/test_receptor_census.py`: a new unregistered producer fails it) |

## What this line consolidates, subsumes or re-points

Dispositions decided by the owner (G5, 2026-10-07). Every plan below that carries a banner got a short
dated 2026-10-07 banner in this PR; **no file moves** (`grounded_language_acquisition.md` is cited by
21 `src/` files, the CLI help, `analysis/roy_log.py` and 4 tests; `jepa_cross_modal_alignment.md` by 3
`src/` files). Status words are exactly SUBSUMED / REVIVED / RE-KEYED / CORRECTED / RETIRED.

**Revive triggers, one wording everywhere:**
- the perception fabric revives when **GL3's registry+provenance stage ships AND a 1.4 rung needs
  cross-modal binding**;
- perception-pipeline placement revives when **GL3's registry+provenance stage ships AND a stage is
  placed across a wire**;
- modality resolution revives when **GL3's registry+provenance stage ships** (its discriminability
  facts are inputs).

| Plan | Disposition | Banner |
|---|---|---|
| [deferred/jepa_cross_modal_alignment.md](deferred/jepa_cross_modal_alignment.md) | **SUBSUMED** by `latent_forward_model.md`: an action-conditioned predictor replaces the 384↔768 projection; its four rules carry over verbatim. Its projection stays a future option for a learned target encoder. | "SUBSUMED 2026-10-07 by `latent_forward_model.md`" + DECISIONS.md |
| [grounded_language_acquisition.md](grounded_language_acquisition.md) | **SUBSUMED for grounding** by this plan (its own scope note kept). Its Phase 2 symbol-binding layer becomes the word side of the forward model; its paired-data audit is answered by changing the pair to (context, *consequence*). README §Active entry folded into this line's. | "SUBSUMED FOR GROUNDING 2026-10-07 by `grounding.md`; kept at this path because `src/` and the CLI cite it." |
| [deferred/grounded_word_binding.md](deferred/grounded_word_binding.md) | **L0 gate kept** as GL1's innate-prior measurement; its old "PASS → candidate 1.5 headline" consequence is **RETIRED**. | banner |
| [deferred/cross_modal_perception_fabric.md](deferred/cross_modal_perception_fabric.md) | **RE-KEYED (G4)** to the fabric trigger above. Binding convention, two-level attention and artifact contract are inputs to `thalamic_relay.md`; not revived wholesale (its Stage 0c is a may-fail experiment of its own). | banner |
| [deferred/perception_pipeline_placement.md](deferred/perception_pipeline_placement.md) (+ the Dormant `runtime/perception_placement.py`) | **RE-KEYED (G4)** to the placement trigger above; the `src` Dormant docstring's trigger changes too (CHANGELOG), in GL0 | banner |
| [deferred/modality_resolution_and_alignment.md](deferred/modality_resolution_and_alignment.md) | **RE-KEYED (G4)** to the modality-resolution trigger above | banner |
| [deferred/second_body_staging.md](deferred/second_body_staging.md) | **Unchanged trigger (G4)**; one line: its "engine seam" is the robot backend, not a percept `Receptor` | one line |
| [deferred/nociception_layer.md](deferred/nociception_layer.md) | **REVIVED** into `autonomic_layer.md` (G5): its triggers (c) and (d) fire; the consequence code's pain half is its vocabulary; `autonomic_layer.md` carries steps 2–4 | "REVIVED 2026-10-07 into `autonomic_layer.md`" |
| [deferred/reflex_layering.md](deferred/reflex_layering.md) | Its routes 2/3 (stimulus vs felt pain vs response) are GL3's fast/slow split; added trigger "GL3's two-track thermal receptor opens" | short dated banner |
| [deferred/hybrid_substrate_reflex_runtime.md](deferred/hybrid_substrate_reflex_runtime.md) | Physical-robot trigger unchanged (G4); a note that its reflex tier is not GL3's `reflex` track; whether to re-read its "second robot body" gate is a GL3.B0 start question | short dated banner |
| [archive/affordance_concept_transfer.md](archive/affordance_concept_transfer.md) | **CORRECTED** (GL0 item 2) | correction banner |
| [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md) | **The organizing frame** for GL2 + GL3 | "Revisited 2026-10-07 by the grounding line." |
| [archive/thalamus_relay_design_pass.md](archive/thalamus_relay_design_pass.md), [archive/percept_testbed_audit.md](archive/percept_testbed_audit.md) | **Decided prior art.** SUBSUME holds; GL3's front-gate answers "do not build a percept-channel manifest as conceived" in writing | no edit |
| [archive/cross_modal_substrate_binding.md](archive/cross_modal_substrate_binding.md) | **Not resurrected** (consequence, not co-activation) | no edit |
| [deferred/adaptive_nociception.md](deferred/adaptive_nociception.md), [deferred/setpoint_aware_neutral.md](deferred/setpoint_aware_neutral.md), [deferred/transfer_non_situation_nac_rows.md](deferred/transfer_non_situation_nac_rows.md) | inputs, triggers unchanged; GL2b must not reopen setpoint-aware neutral's DO-NOT-BUILD | none |
| [engram_formation.md](engram_formation.md) | sibling; E7's owner pointer → `latent_forward_model.md` GL4 S0a | dated pointer line |
| Orient line plans (`cradle_mother`, `live_audio_orient_wiring`, `reachy_orient_live`, `sem_motor_binding`, `substrate_native_orienting`, `microduck_intent_layer`) | unchanged; their trigger is the robot | none |

**Ledger edits.** T1-5 re-judged now (GL0; DROPPED recommended); `SUPERSEDED by T1-<new>` only after
a successor is at a positive status. T3-7, D6 and `episode.py` re-pointed to GL4; T3-4
(`anticipatory_pre_activate`) re-pointed to GL4 S0a; that ledger edit lands in GL0's PR (the ledger is
GL0's), not in this one. No new row at plan time; a Tier-3 `SETUP` seed row per claim-bearing
mechanism is allowed (e.g. "consequence similarity separates `warm_self`-safe from `warm_self`-harmful where
name similarity collapses them"). Tier-1 rows only at graduation.

## Open owner decisions, by stage

Each is asked together at the start of its stage (development-flow rule 3). Where a draft offered a
strict option, it is the default recommendation. G1–G8 are decided and are not re-asked.

**GL0 start** (decided 2026-10-08, owner)
- T1-5 status: **DROPPED** (strict), the ID kept so the GL5 successor is linked `SUPERSEDED by` it.
- #1120: the strict red gate at the retired 0.40 is **kept** (documentary).
- The `bio-memory.md` `[behavioral]` SCN-coupling stub: **neither demoted nor kept as earned**: the owner
  wants the experiment run. It is marked evidence PENDING, with the redesigned run owed
  ([#1180](https://github.com/dennys246/Maxim/issues/1180)) and a lint to tell pending from earned
  (`outstanding.md` M42). (Its cited validation did run about 19 times in 2026-04 but recorded no evidence.)
- External pymaxim.bio pages: **corrected via a Maxim-web kickoff** in the same window.
- Reward-driven widening absorbing an unrelated name: **a defect** ([#1181](https://github.com/dennys246/Maxim/issues/1181), strict red gate).

**GL1 start**
- Is `latent_forward_model.md` itself Phase 5's Cerebellum/timed-predictor audit, or must that audit
  run first and name the gap? *Recommend: the audit is its S0a and opens the plan.*
- Where does the T1-5 successor live: the body world or the cradle (where `touch` ×16 and `warm_self`
  ×14 are)? *Recommend: let the GL1 census decide.*
- Does an L0 FAIL still archive `grounded_word_binding.md`? *Strict default: the frozen "fail →
  archive".*

**GL2a start**
- Core valence formula: `relief − harm − nociception`, or weighted (negativity bias)? Innate prior
  either way.
- Urgency v1 = pressure only, slope deferred to the experience-clock/tracks work?
- Does `ActionContext` ride `ToolOutput.side_effects` (a row in `docs/user/tool_side_effects.md`) or a
  `ToolOutput` field the forward model owns?
- Narrated records vs the fence (autonomic_layer.md §8): stamping narrator consequences `narrated` (G6)
  needs the three narrator tools' `execute` bodies in `simulation/tools.py`, outside G1's exempt set, and
  thread identity cannot stand in (the reflex dispatch calls separate instances of the same tool
  classes on the loop thread). Add them
  to the exempt set, or ship the tool-path record only and land the out-of-band producer after the fence
  with the narrated scope? *Recommend the latter (strict).*

**GL2b start**
- Burn calibration (noxious threshold, gain) and **where** it is declared: on `infant_humanoid`
  (fixes every variant, fires T1-6/T1-9) or on a new variant body (fires neither, fixes nothing
  shipped). *Recommend the variant body (strict), with values meeting autonomic_layer.md's
  constraint: a +0.6 contact from rest reaches ≥ 0.4 and a single `warm_self` or `*_safe` produces
  0.* (A 0.55 threshold with a +0.6 contact at ≥ 0.5 was one drafted option.)
- Heat-need naming (`heat` vs `overheat`) and affinity keywords (note `withdraw` is already the
  `pain` and `threat` keyword); does Stage (i) wait for a shipped cooling affordance?
- #1161 order: resolve option A (the modulator read + narrow the collateral gate) before burn-as-pain.
  *Recommend yes; Exp 42 must be re-run either way.*
- The cause-row key namespace and the walk of its consumers (§Blast radius, GL2b(iii)) before (iii)
  merges.
- LLM-primary `note_active_clusters`: file as its own defect, or carry into GL3?

**GL2c start**
- Satiation routing: through the distributor onto `_reward_bias`, only into the Phase 5 relief store
  (world-cluster keyed, never the `_reward_bias` surface), or both. *The relief store alone is the
  strict option.* Every routing fires T1-11, T1-12 and T1-13/14/15 (G7).
- Double credit when a tool caused the satiation and channel 3 already credited the cluster.
  *Recommend one credit per event, keyed by `pid`.*
- The joint four-lens review for the producer + relief store (+ `fear_learning` Exp A, `coding_world`
  C3). *Recommend yes: the store's own rule says no second store is ever created.*

**GL3.B0 start** (census + red gates; the later GL3 stages' questions are asked at each stage's own
start, listed below under its heading)
- Dead preemption scaffolding (`runtime/preemption.py`, `wire_preemption`, `check_hold`, the
  `ExecutionTracker` capture; L7, via its defect issue): *Recommend a `Dormant since 2026-10-07`
  marker (strict, dormancy over deletion; removal touches the `maxim.runtime` re-exports).* Deletion,
  folding `SourceConfig`'s fields into the track spec, is the non-strict alternative. The track
  scheduler is a new module either way; it does not build on the dormant one.
- L2 (via its defect issue): may the turn budget delay *nociception*, or only *action*? *Recommend:
  only action.*
- Re-read `deferred/hybrid_substrate_reflex_runtime.md`'s "second robot body" gate under G4?

**GL3.B1 start** (registry+provenance; asked at GL3.B1's start, not GL3.B0's)
- Which consumer it lands with: GL5's experiment or the forward model's contamination guard (G8).
- Provenance kinds beyond `experienced` / `narrated` / `imagined` (G6): add `declared` (own-body
  affordance names from YAML) and `reported` (game-log text about a real event)?
- Refuse vs discount credit for **imagined** nodes (narrated is decided: discount, G6). *Strict:
  refuse.*
- Provenance as a per-node dict vs a separate `text.imagined` modality. *Recommend the dict.*
- Registry+provenance trigger ruling: does a byte-identity-proven registry fire
  T1-6/10/11/12/13/14/15? *Strict: yes, offline guards re-run, committed exception for rig re-runs.*
- The relay's home and coordinator name (the reserved `perception/` package was proposed).
- Receptor specs for body sensors derived from YAML at bind time (recommended) or authored; each new
  YAML key gets a reader in the same PR (unlike `coupled_to` / `modulated_by`).

**GL3.B3 start** (the thermal dual-track fan-out)
- Latency in passes (ordinal, deterministic; proposed) vs experience µs.
- The two built tracks' latencies, priorities and refractories (innate priors in the proposed table).
- Does the fan-out fire T3-9 by wording? *Strict: yes, walk and re-run Exp 09's offline guard.*

**GL3.B4 start** (preemption)
- Which 1.4 rung declares GL3.B4 as its arm (G8: the owner names it later)?
- After a preemption: reselect in the same pass (proposed) or let the `reflex` track's program act
  first?
- `nociceptive_fast`'s preemption right and threshold (an innate prior).
- Does GL3.B4 fire T3-9 and the Exp 60 family by wording? *Strict: yes.*

**GL3.B5 start** (text receptors)
- `SensoryTag.perceived_intensity`: mark dead or wire from the relay gain.

**GL4 start**
- The narrated discount's value (G6). *Strict default: a small discount;* GL5 reports with and
  without narrated data either way.
- Kernel choice: kernel ridge first; an MLP only as a pre-registered second iteration after a ridge
  FAIL shown to be non-linear.
- Imagined-entity pairs: exclude (strict, recommended) or down-weight ×0.5.
- The `consequence` modality and hivemind: local-only (recommended) or update the frozen-modality
  pin now.
- Cerebellum key: the TARGET entity; add the situation cluster (changes the key format) or let the
  predictor carry situation?
- S0 capture: scripted-substrate cradle + Minecraft only, or also an LLM cradle arm (~10× fewer
  invocations plus a selection-bias confound)? Offline scripted bridge or rig for the Minecraft arm?
  The orchestrator's sensor writes (`SetEntitySensorTool`; `cradle_fire_pit.yaml`'s "proximity
  effects are handled by orchestrator sensor writes") exist and are narrated consequences (G6), so
  S0b counts them separately.

**GL5 start**
- Arms, n, margins; which new ledger row it would earn (never a re-label of T1-5).
- Prior term placement: a composing seam before `recommend_action` (no trigger; strict) or a term
  inside it (fires five rows' re-runs).

**GL6 start**
- Whether the arm may enter at all before E3 records, and its default (off unless the walk's re-runs
  are MAINTAINED).

## Risks

1. **The positive producer lapses a written discharge.** T1-13's #888 discharge rests on "the only
   positive Reaction constructor is `CerebellumModulator` … never given a `reaction_bus`". GL2c makes
   that false, and under any routing it also fires T1-11, T1-12, T1-14 and T1-15, because the Minecraft
   breach latches clear at Exp 60's own `escape_water` contingency (G7). Mitigation: off-by-default
   flag, executing survival checks unchanged with it off, the staged-donor refusals restated, and one
   batched live re-run of Exp 60/61/62 (+ T1-11 and T1-12) in its own rig slot.
2. **Shared bodies.** Every infant variant `extends: bodies/infant_humanoid`; editing `arms.thermal`
   there fires T1-6, T1-9, T1-10 and T3-9 at once. Variant bodies are the shield, as the
   roadmap already uses for `minecraft_player`; a `classify_pain` change fires only the PainBus
   category.
3. **#1161: Exp 42's discrimination depends on a blind read.** Making arm thermal "accounted" can
   invert `warmth_alpha_harm`'s credit at saturation. GL2b(ii) does not land before #1161's decision
   and an Exp 42 re-run plan.
4. **The first positive text-credit producer.** E4 found the 0.44 radius already spans concepts; if
   GL2c's producer reaches text nodes, E4's 13-string overreach set is read first.
5. **The relay as a framework.** Rejected before ("no manifest as conceived"; roadmap decision 5: no
   instrument ahead of its consumer). Mitigation (G8): the registry enters only with provenance and
   its consumers (GL5's experiment, the forward model's contamination guard), byte-identical; the
   first track slice (GL3.B3) is a flag-off fan-out with no preemption, and preemption (GL3.B4)
   enters only as a declared rung arm; the timing defects are fixed as defects,
   without the registry; no speculative receptors; stop rule: behaviour that belongs in a nucleus is
   pushed back down.
6. **Hidden selection change in a refactor.** `_SUBSTRATE_CHANNELS` order and active-channel count
   set the summed cluster term in `recommend_action`; the registry preserves the code order
   (interoception, audio, world) and the empty-read → no-encode rule. The double representation of Minecraft vitals is named, not de-duplicated (that
   moves geometry and fires Exp 60–62).
7. **Determinism.** Tracks as threads would break the step-clock / lockstep tests (#951). M40 and the
   cross-process, cross-hash-seed golden keep them logical.
8. **Livelock and the D13 interplay.** A drop site that forgets to re-arm recreates the D13 livelock;
   repeated preemption could thrash. Mitigation: one typed handler, at most one preemption per
   physical event, refractory in passes, livelock and deletion probes.
9. **Circular authored physics.** Cradle consequences are YAML, so transfer may only recover the
   author's regularities. Mitigation: blind physics author, effects derived from each object's own
   declaration, and the game-native Minecraft arm.
10. **Data volume and contamination.** S0's hard floors stop the line on too little data or no
    dissociation; pairs join only on executed, provenanced invocations; the word prior's geometry
    differs by install (mpnet vs the 384-d fallback), so every learned map carries geometry tags and
    refuses on mismatch.
11. **Two may-fail lines in one release.** T9 is conditional and never co-headlined with E3; M10
    keeps grounding flags out of E-arms.
12. **Divergence.** If GL2b's fixes each surface a new failure mode for two iterations running, stop
    and audit the body layer beneath; the sensor-resolver disagreements (#1124, #1156, #1159) are the
    likely layer.
13. **Claim drift.** CHANGELOG claim lines and CLAUDE.md's Active initiatives are not linted (M36).
    Keep grounding wording out of release-claim prose until GL5 records.
14. **Bio over-claim.** Track latency is ordinal, in passes; docstrings say "FUNCTIONAL, not fibre
    dynamics", as `PainBus` already does. The predictor is not called JEPA (G2).

## Review record

- 2026-10-07: drafted from the #1120 audit and four planning reads (thalamic contract, autonomic
  layer, afferent tracks, action-conditioned prediction) plus a roadmap-and-truth read. Owner
  decisions G1–G4 recorded. The four-lens design review of each sub-plan and the three-lens review of
  this document are owed (GL1).
- 2026-10-07 (review round 2): owner decisions G5–G8 recorded (dispositions; narrated provenance,
  discounted; GL2c re-runs Exp 60/61/62 + T1-11 under any routing; the registry enters with its
  consumer). The event-identity contract stated once; cross-document findings folded (the word path
  and earned rows, the narrator's sensor writes, GL2a's blast radius and #1125, GL2b's fixture,
  refractory collision and cause-row consumers, the pain-ingress census, bio mappings, stage-ID
  prefixes, mechanization numbering, fence footprint, revive-trigger wording).
