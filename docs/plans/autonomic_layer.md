# The autonomic layer — a signed body-consequence record and the missing regulatory signals (grounding GL2)

> **PROPOSED 2026-10-07 — grounding-line sub-plan, stages GL2a / GL2b / GL2c** (umbrella:
> [grounding.md](grounding.md); owner decisions G1–G8 of 2026-10-07 are recorded there and in
> [DECISIONS.md](../../DECISIONS.md) and are not re-opened here). It fills no new roadmap slot of its own:
> its positive producer is the **input** to [roadmap_1_4.md](roadmap_1_4.md) §Phase 5's cluster-keyed
> relief store, and its record is the **target** of [latent_forward_model.md](latent_forward_model.md).
> Every mechanism enters as `[engineering]`; every behavioural change ships **off by default, with
> today's selection pinned byte-identical**. **Nothing here is built until this plan's four-lens design
> review (GL1) has no unfolded DO-NOT-BUILD.** `src/` work waits for the 1.3.2 decomposition fence
> **except GL2a**, which is record-only (owner decision G1). GL2a's exempt file set is exactly:
> `embodiment/body.py`, `embodiment/sem.py`,
> `runtime/executor.py` (`_run_started` and `_stamp_invocation`, widened by G14 on 2026-10-09),
> `tools/base.py::ToolOutput` and `runtime/bio_integration.py` (the leaf `embodiment/event_id.py` left
> the set by G17: it lands at the post-fence resume stage). Anything outside that set waits for the
> fence (§5.2 says what that means for narrated records).
>
> **GL1 four-lens review folded 2026-10-09.** Reports:
> [confounding](reviews/grounding_gl1/confounding.md) · [bio-faithful](reviews/grounding_gl1/bio-faithful.md) ·
> [wiring](reviews/grounding_gl1/wiring.md) · [environment](reviews/grounding_gl1/environment.md). Owner
> decisions G9–G16 (DECISIONS.md 2026-10-09) settle their nine DO-NOT-BUILDs, and
> [grounding.md](grounding.md) §GL1 names the sentence that resolves each; G17–G20 came with the GL1
> code review the same day. In short: GL2a is the
> tool-path record only (G9); it is action-scoped, read through the #1125 resolver and netted of drift
> (G14); its relief and harm follow need (alliesthesia, §3.1); it carries **no event id**: the
> `PhysicalEventId` type, the per-agent sequencer, its session-id source and the cross-session resume
> land together at the post-fence resume stage, before GL4 S1 (G17, superseding G15's GL2a half); its
> field is `repr=False` (#1189); and harness writes are `apparatus` (G16).

**Owns (proposed):** the consequence record and its pure computation (`embodiment/sem.py`, beside the
existing drive helpers); the identity type `PhysicalEventId` in the leaf module `embodiment/event_id.py`
and the per-agent `EventSequencer` (built at the post-fence resume stage, G17; GL3 imports both); the out-of-band record at
`embodiment/body.py::Embodiment.evaluate_failures`; one additive record-only field on
`tools/base.py::ToolOutput`; the `NociceptorSpec` producer; the direction-aware corrective need; the cause
stamp on PainBus context; the satiation producer.
**REVIVED here (owner decision G5):** [deferred/nociception_layer.md](deferred/nociception_layer.md) —
its triggers (c) and (d) fire on GL2, so its steps 2–4 and the F1 fix are carried here as stage
prerequisites (§7), not as a parallel plan. **Companion plans:** [roadmap_1_4.md](roadmap_1_4.md) §Phase 5 (R4 routing; the relief store —
one joint review, §5.4); [latent_forward_model.md](latent_forward_model.md) (consumes the record);
[thalamic_relay.md](thalamic_relay.md) (owns `AfferentTrack`s; GL3.B3's thermal dual-track fan-out, its first track slice, is
built on this plan's nociceptor); [memory_strength_and_forgetting.md](memory_strength_and_forgetting.md) (a consumer of
the relief/pain fields through `EncodingSignals`, not subsumed); [fear_learning.md](fear_learning.md)
Exp A and [coding_world.md](coding_world.md) C3 (the relief store's other candidate writers).
**Inputs, triggers unchanged:** [deferred/adaptive_nociception.md](deferred/adaptive_nociception.md),
[deferred/reflex_layering.md](deferred/reflex_layering.md),
[deferred/setpoint_aware_neutral.md](deferred/setpoint_aware_neutral.md) (§4 says why GL2 does not reopen
its DO-NOT-BUILD), [deferred/behavior_tiers.md](deferred/behavior_tiers.md).
**Organizing frame:** [archive/thalamus_hypothalamus_framing.md](archive/thalamus_hypothalamus_framing.md)
("percept = thalamus, drives = hypothalamus"). This plan is the hypothalamic / insular side.

## Why this plan exists

The 2026-10-07 audit (#1120) found the body world under-wired in four concrete ways: a burn on the infant
is filed as mild drive discomfort, not pain; heat above the set point produces no corrective need while
cold does; the learned percept valence is keyed on the body that suffered, not the thing that caused it;
and nothing in the codebase ever emits a positive Reaction, so the one surface built for positive credit
(`NAc._reward_bias`) has never grown on a recorded run. Separately, the latent forward model needs a
**target**: a small, signed, body-agnostic description of "what happened to my body, and was it good".
Most of that record already exists as scattered, tool-path-only fields. This plan makes it one record,
fixes the four defects behind it, and adds the missing positive signal last and behind a switch, because
that is the change that reaches the earned survival rows.

## 1. Current state (verified at `origin/main` f1c833ab unless marked UNVERIFIED)

### 1.1 What already exists — record-only, and only on the tool-invoked path

| Piece | Where | What it is | Status |
|---|---|---|---|
| Drive pressure before the action | `embodiment/sem.py::drive_pressure`, stamped by `runtime/executor.py::Executor._drive_pressure_snapshot` | per drive, `[0,1]`, **unsigned**, both directions, normalised by the declared range | LIVE, record-only (`ToolOutput.drive_pressure_before`) |
| Relief per drive | `sem.py::relief_fraction_from_progress` ← `embodiment/tool_bridge.py::_drive_progress_by_drive` ← `executor.py::Executor._drive_relief` | positive part of `drive_comfort_progress` over `drive_span`; movement **away** from comfort is recorded as 0.0 | LIVE, record-only (`ToolOutput.drive_relief`) |
| Felt nociceptive pain of this invocation | `bridges/tool_pain_bridge.py::_note_caused_pain` / `pop_invocation_pain` | peak NOCICEPTIVE intensity of the delta-attributed failures | LIVE, record-only (`ToolOutput.pain`) |
| Memory encoding record | `memory/encoding.py::EncodingSignals` | `pain` (nociceptive only), `drive_pressure_before`, `drive_relief`, `extra` (CC3 path (a); `__post_init__` rejects colliding keys and non-JSON values) | LIVE |
| Pain classification | `proprioception/pain.py::PainKind`, `classify_pain`, `failure_pain_kind`, `drive_failure_sensor`, `TISSUE_DAMAGE_DRIVES = {"drive:health"}` | NOCICEPTIVE / DRIVE / ANTICIPATORY / FRUSTRATION / EXHAUSTION | LIVE (nociception_layer step 1, shipped in 2S-c) |
| Motor credit (the positive learning that does exist) | `tool_bridge.py::_drive_potential_diff` → `drive_comfort_progress` (sign only, ±1) → `tool_dispatch` → `NAc.update_cluster_reward` | value-progress toward comfort, credited to the INTEROCEPTION cluster | LIVE, earned (T1-6, T1-7, T1-9) |
| Cerebellar forward model | `embodiment/cerebellum.py` (`observe_from_action` from `tool_bridge.py`) | predicts raw sensor deltas per (entity, modulator, affordance, bucket); saved since #908, read side Dormant (#909) | write LIVE, read Dormant |

**One defect still sits under this record, and one is closed.** #1125 is CLOSED (2026-10-08, #1164):
`Executor._drive_pressure_snapshot` now reads modulator-qualified drives (`arms.thermal`, `head.thermal`,
`arms.pressure`) through `embodiment/sem.py::_read_sensor_value` / `_resolve_sensor_slot` (the one
resolver; `tool_bridge.py` only imports it). #1161 (OPEN): the
affordance-credit reads in `tool_bridge.py` (`pre_values`, `_drive_potential_diff`,
`_drive_progress_by_drive`) read drives by bare name, so a qualified drive's progress is never measured —
and Exp 42's safe-vs-harm discrimination (T1-6) depends on that blindness (#1161's own measurement: with
the read resolved and no gate change, `warmth_alpha_harm` `warm_self` flips from −1 to +1 on warms 3–5).
Consequence for this plan: **today's `ToolOutput.drive_relief` omits exactly the drive a burn moves**, so
the record reads drives itself (§3.1.4) and never refills that field.

### 1.2 What does not exist

1. **No positive `Reaction` reaches `TemporalCreditDistributor`.** `runtime/bio_stack.py::_distribute_reward_from_reaction`
   (nested in `build_bio_stack`) pays `+intensity` for a POSITIVE Reaction, but the only positive
   constructor in `src/` is `embodiment/backends/cerebellum_modulator.py::_emit_success_reaction`
   (`kind="reward"`), whose factory has no production caller and which is never given a `reaction_bus`.
   Every live Reaction is `kind="pain"`, `Valence.NEGATIVE` (`reactions/compat.py::pain_signal_to_reaction`).
   T1-13's annotation records the scored `reward_bias` term at 0.0 at all 108 scored decisions in
   `exp61_pairs.jsonl` + `exp62_rows.jsonl`.
2. **The satiation Reaction promised by `sem.py::EntropicDriveSpec` was never built.** The docstring said a
   positive Reaction fires when the value crosses back below `satisfaction_threshold` (corrected in GL0); in
   `body.py::Embodiment.evaluate_failures` the crossing is detected (`elif cleared:
   breach_latch.pop(ds_name, None)`, both the homeostatic and entropic branches) and discarded.
   `reactions/types.py::ReactionKind` already reserves `"satiation"`, with no producer. (GL0 corrected the
   docstring and field comment; the class is SHAPE-FROZEN at 1.0 (CC3), so no field changed.) **Where that crossing
   happens on the earned bodies** (corrected 2026-10-09, GL1 environment review DNB-2; G7's decision
   stands): on `minecraft_player` the homeostatic `oxygen` and `health` latches clear on any recovery
   **after a latched breach** (the homeostatic `elif cleared:` branch; `oxygen` breaches below 14 and
   clears at ≥ 15.2, `_BREACH_HYSTERESIS` 0.2). On the earned campaigns that recovery is made by the
   **apparatus**, not by the agent's `escape_water`. Exp 60/61 training is propose-only
   (`scripts/survival_world/water_trial.py::WaterTrial.train`), and every episode ends in
   `WaterTrial.rescue`: an RCON teleport to shore, a settle until oxygen ≥ `RECOVER_OXYGEN_MIN` (19), then
   `heal()` (`/effect instant_health` + `/effect saturation`). Exp 60's probes are capped before the
   breach (`probe_cap_s` 4.335 < `pain_edge_min_s` 5.085 in `docs/experiments/data/exp60_trials.jsonl`),
   so a FEAR-arm escape mints no crossing. R3's respawns reset health and oxygen to 20. `minecraft_player`
   `food` is entropic with `satisfaction_threshold: 16.0` and is not reachable natively in campaign time.
   `minecraft_bench` `d1` (Exp 56, T1-11) and `minecraft_bench57` `d1` (Exp 57, T1-12; the same spec,
   copied verbatim) are entropic, drifting up, with `satisfaction_threshold: 0.3`, and Exp 56's teacher
   (`scripts/exp56/common.py`) and Exp 52's mother (`simulation/cradle_mother.py::reactive_mother_tick`)
   write that drive directly before `NAc.credit_operant_reward`. Any producer at this site therefore lands
   on the earned survival and transfer rows' own measured events (§3.5), mostly as **apparatus** writes
   (G16, §3.1.4).
3. **No record of out-of-band body change** (narrator writes, the Minecraft bridge's sensor writes,
   drift, an actor hitting the AUT). The Minecraft writes are seen only at the next loop-thread
   `evaluate_failures` call: `simulation/minecraft_harness.py::MinecraftSyncPump._run` writes sensors and
   never calls `evaluate_failures`. The only out-of-band relief consumer is the cradle-mother special
   case (`simulation/cradle_mother.py::reactive_mother_tick`, Exp 52).
4. **Relief from leaving a harm** (`escape_water`, `flee`) records no relief key (memory plan §2b-ii).
5. **Harm is dropped from the record:** `relief_fraction_from_progress` keeps the positive part only; no
   signed per-drive delta exists on any record.

### 1.2a Who calls `evaluate_failures`, and the pain that bypasses it

`evaluate_failures` is **not** a single convergence point, and it is not called from one thread.

| Caller | Thread | Notes |
|---|---|---|
| `runtime/substrate_proposal.py` substrate-primary tick; `runtime/loop_gates.py` LLM-primary embodiment tick | the agent loop (`sim.aut` worker in `--sim`) | drift and lingering breaches are detected here |
| `embodiment/tool_bridge.py::ModulatorAffordanceTool.execute` | the agent loop | post-action; B8 applies to its `embodiment_failures` |
| `simulation/tools.py`: `DamageComponentTool`, `OrchestratorActorTool` (on the AUT body), `SetEntitySensorTool` (two sites), as the narrator's tools | the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop): `simulation/orchestrator.py` registers them on `orch_registry` and runs them inside `run_agentic_loop(orch_agent, …)` while the AUT loop runs on `sim.aut` (`sim.dm` is a different thread, used only by interactive DM campaigns, and never touches these tools) | a **declared cross-thread edge** (§3.1.2); their consequences are **narrated** (G6, §3.1.4) |
| the reflex dispatch (`integration/bio_enrichment.py`'s `_reflex_damage_tool` / `_reflex_sensor_tool`: separate instances of the same `DamageComponentTool` / `SetEntitySensorTool` classes, wired into the AUT's `BioEnrichmentPipeline`) | the agent loop (`sim.aut`): it runs inside the AUT's `enrich` | the loop thread, not the edge; its consequences are still **narrated** (G6) |
| `simulation/foundry.py`; `simulation/tools.py` `scene_emb` (an ephemeral `Embodiment` around a scene entity) | either | not the AUT: these wrappers mint no record and no id |
| `embodiment/percepts.py::EmbodimentPerceptSource` | — | no production constructor |
| `simulation/fixture_orchestrator.py` (`--sim scenarios/substrate/*.yaml`) | — | calls no `evaluate_failures` (GL1 wiring review N3) |

Two pieces of mutable state are touched from both threads today, with no lock in `embodiment/body.py`:
the entity-owned breach latch `Entity.drive_breach_severity` and (once the out-of-band producer adds it)
the per-entity previous-snapshot slot. The lock over them lands **with the out-of-band producer, after
the fence** (G9, §3.1.2); GL2a takes no lock (it mints nothing, G17), and the sequencer the post-fence
resume stage builds takes only its own private lock. `PainBus._suppress_bridge`
is a plain instance flag, not thread-local, so with two publishing threads one thread's flag can
suppress, or fail to suppress, the other's bridge dispatch: pre-existing, and added to GL3.B0's
two-thread census (wiring N2).

**Pain ingress census.** Body pain also enters PainBus / ReactionBus without passing
`evaluate_failures`: `DamageComponentTool` publishes a `PainSignal` directly, proportional to damage,
before it calls `evaluate_failures`; `simulation/tools.py::InjectPainTool` (→
`simulation/conversational_source.py`'s `inject_pain`, a `Reaction(kind="pain")` straight onto
ReactionBus); `runtime/sim_adapter.py::next_observation` turns a `pain_signal` percept into a `Reaction`
(`source="sim_adapter:…"`); `simulation/sandbox.py::PainTriggerLayer` and `runtime/pain_interceptor.py`
(sensitive-path pain); `proprioception/perceived_pain.py` (anticipated pain, F1); and the Default
Network's `bridges/pain_bridge.py::PainCircuitBridge` (robot). The public `api.py` `"pain_signal"` event
subscribers observe only `PainSignal`s on the agent bus (`bus.subscribe(PainSignal, _on_pain)`); the
Reactions put straight onto ReactionBus (`InjectPainTool` → `conversational_source.inject_pain`,
`sim_adapter.next_observation`) never become a `PainSignal`, so those `api.py` subscribers never see
them. PainBus's own direct subscribers (memory, NAc, Wire 2, Wire 4) do receive such pain Reactions,
lossily, through `PainBus._bridge_reaction_to_pain_subs`, whenever they are published on that
`pain_bus.reaction_bus` (wiring N1). The record built here sees only what reaches `evaluate_failures` or
the tool path; every other ingress is listed, not covered, and the census is re-checked in GL1.

### 1.3 The regulatory defects

**R-1 — a burn is drive discomfort, not pain, and too weak for every PainBus learner.**
`_data/components/bodies/infant_humanoid.yaml` `arms.thermal`: range `[-1,1]`, homeostatic,
`comfort_band 0.5`, `pain_scale 0.4` ⇒ `drive_pain_for_value` max = (1.0−0.5)·0.4 = **0.2**. The cradle
fire pit's `touch` writes `arms.thermal +0.6` ⇒ from rest, (0.6−0.5)·0.4 = **0.04**. It is published by
`body.py::Embodiment._publish_drive_pain` with `source="drive:arms.thermal"` ⇒ `classify_pain` → DRIVE;
the FailureEvent `drive:arms.thermal:discomfort` ⇒ `failure_pain_kind` → DRIVE.

| Consumer (`proprioception/pain_bus.py`) | Threshold | Sees the burn? |
|---|---|---|
| `create_pain_memory_subscriber` | 0.4 | no |
| `create_percept_valence_subscriber` (Wire 2) | 0.3 | no |
| `create_pain_nac_subscriber` | 0.3 | no |
| `create_pain_cluster_fear_subscriber` (Wire 4) | 0.3 + allowlist `{drive:health, drive:oxygen}` | no |
| `ToolPainBridge.record_tool_embodiment_failure` (channel 1) | none | yes — `nac.record_outcome(..., NEGATIVE)` on `tool:<name>`, intensity-blind |
| 2S-c felt pain (`ToolOutput.pain`) | NOCICEPTIVE only | no (DRIVE) |
| `_distribute_reward_from_reaction` | none | yes: `credit_node(-0.04)`, which the ≥0 clamp turns into "erode a positive bias or do nothing" |

So the burn is not invisible (the action-level causal link learns it), but memory never tags it, Wire 2
never keys it and the record calls it "not pain". Bio: noxious heat is transduced by a separate receptor
population (TRPV1-class nociceptors) from innocuous warm/cool thermoreceptors at the same skin site; the
YAML models only the thermoreceptor.

**R-2 — heat yields no corrective need.** `sem.py::corrective_need_intensity` returns a value only for
homeostatic deficits below `set_point − comfort_band` and for entropic "down" drives; above the set point
it returns `None`. `runtime/substrate_proposal.py::_DRIVE_CORRECTIVE_NEEDS` maps `temp`/`thermal` →
`"cold"` only. Hypothalamic thermoregulation is two-sided. Impact today: no prereg names an overheated
agent (UNVERIFIED that none measures one), but the need is **not latent** in the cradle scene: a shipped
cooling act exists (`items/cradle_cool_air.yaml`: `draft.feel` takes core −0.2 and arms −0.15; its
`shelter` warms, +0.05), and `simulation/arcs.py` activates `cradle_cool_air` beside the fire pit in the
`exploration` **phase** of the `cradle` and `cradle_prelinguistic` arcs (and the `_deceptive` arcs built
from their phases); entities persist across phases, so it stays in the scene after that. Whether a heat need gets a consumer there is decided by its affinity keywords
(a `"cool"` keyword would match the `cool_air_*` tool names), so the keyword choice is the switch between
latent and live (GL1 environment review SF-4).

**R-3 — Wire-2 percept valence is keyed on the SUFFERER.** `pain_bus.py::create_percept_valence_subscriber`
keys `(agent_id, context["entity_name"] or entity_type, failure_mode)`; `_publish_pain` /
`_publish_drive_pain` set `entity_name` to the entity in `self.root.walk()` that owns the failure — the
agent's own body or an acquired item. The cause is known at `simulation/tools.py`'s actor invocation
(`side_effects["actor_invocation"] = {"source_entity", "source_affordance", ...}`) and dropped. The NAc
docstring's "a dragon that burned the agent once" cannot be produced by any `body.py` path.
`NAc.record_percept_valence` already accepts signed values; its only reader, `NAc.get_percept_aversions`,
treats non-negatives as zero and feeds `GatingContext.learned_aversions` → `TextSalienceScorer`.

**R-4 — reserved, not defects:** `EntropicDriveSpec.coupled_to` / `HomeostaticDriveSpec.modulated_by` are
parsed and never read ("1.0 interface, deferred"); they stay reserved for drive interaction and are not
repurposed here. Drive-event SCN emission (`body.py::_emit_drive_temporal_event`) is Dormant (D9); this
plan's events are its natural producer if it is ever resurrected, which is not proposed. LLM-primary never
calls `NAc.note_active_clusters` (only caller `runtime/substrate_proposal.py`); the record must not assume
a noted situation exists. That defect is fenced behind the agent_loop slices and is not owned here.

### 1.4 Invariants this plan must not break

- **Channel split** (`docs/agents/embodiment.md` §2): channel 1 state-based; channel 2 severity-latched on
  `Entity.drive_breach_severity`; channel 3 value-progress, sign only. Guard:
  `tests/unit/test_transition_drive_pain.py`.
- **B8 delta-attribution** (`tool_bridge.py::_intrinsically_harmful_sensors`); `drive_failure_sensor` is the
  one parser.
- **Reward-bias clamp** (`NAc.credit_node`, T3-19): `_reward_bias ∈ [0, max_reward_bias]` (§5.6).
- **Drive specs SHAPE-FROZEN at 1.0 (CC3);** `Reaction` SHAPE-FROZEN with additive-defaulted fields allowed
  after an isolation review.
- **Named by ledger triggers — compute beside, never edit:** `drive_comfort_progress`,
  `_drive_potential_diff` (T1-9 by name), `corrective_need_intensity` (T1-6 via the interoception channel).
- **Wire-4 allowlist** `DEFAULT_CLUSTER_FEAR_FAILURE_MODES` is a hivemind wire boundary (Exp 60/61/62). No
  stage edits it; a burn is **not** added to the fear set.
- **No new bus** (nociception_layer principle 5).
- **A constant is never recorded as a measurement** (memory-strength invariant): no body ⇒ no record, never
  a record of zeros.

## 2. Front-gate: why this rides on existing infrastructure, and the four things it must add

The layer adds **no bus, no store and no subscriber**. It rides on four seams that already exist:
`Embodiment.evaluate_failures` (where drive and failure-mode state is evaluated for the body, with the
entity-owned latch that already stops per-tick floods; not the only pain ingress, §1.2a),
`Executor._stamp_invocation` (the single
writer of the per-invocation body record, already stamping pressure, relief and pain onto `ToolOutput`),
PainBus / ReactionBus (pain and the reserved `"satiation"` kind), and the existing pure drive helpers in
`sem.py` (no second derivation of any formula). What it must add, and why the existing piece cannot do it:

| Need | Rides on | Must add — and why existing infrastructure cannot do it |
|---|---|---|
| Body-consequence record | `drive_pressure`, `drive_span`, `drive_comfort_progress`, `classify_pain`, `EncodingSignals.extra`, `ToolOutput` additive fields | One frozen record + one pure function. `EncodingSignals` cannot be the record: it validates per-drive values as unsigned `[0,1]`, so signed delta and deviation cannot live there, and it exists only at capture sites, not at out-of-band body change. |
| Out-of-band record (after the fence, G9) | `evaluate_failures`; the entity-owned latch pattern | A per-entity previous-snapshot slot (`__slots__`, never serialized, like `drive_breach_severity`), a bounded queue and a pull, `Embodiment.drain_outcomes()`. No bus: the loop pulls. |
| Event identity | `memory/hippocampus.py::Hippocampus._resume_capture_seq` (the resume-past-the-saved-maximum rule, reused by the post-fence resume stage, G15) | A per-agent `EventSequencer` and `PhysicalEventId`, built at that resume stage (G17), not GL2a. Hippocampus `capture_seq` numbers memories, not physical events (one event can yield no capture or several), and the executor's `uuid4` invocation id is neither deterministic nor persisted-stable. |
| Burn nociception | channel-1/2 drive machinery, B8, the one parser, `classify_pain` | `NociceptorSpec`, because drive specs are SHAPE-FROZEN and a standard `failure_modes:` entry floods PainBus and bypasses B8 (§3.2). |
| Heat need | `_read_drive_states`, `_DRIVE_TOOL_AFFINITIES` | A direction-aware wrapper; `corrective_need_intensity` must stay byte-identical (it is named by T1-6). |
| Positive producer | ReactionBus, `"satiation"` `ReactionKind`, the body latch, the Phase 5 relief store | Structurally nothing — one emission at the latch-clear site. That it needs no mechanism is exactly why its blast radius is the whole credit path (§5). |
| Cause keying | `record_percept_valence` (already signed), PainSignal context | A keyword `cause=` on `evaluate_failures` and two context keys. It cannot ride the existing key: the sufferer row means "which of my parts hurts", and persisted rows would silently change meaning. |

**Why GL2 does not reopen `setpoint_aware_neutral.md`'s DO-NOT-BUILD.** That review rejected a change to
`similarity/encoder.py::_sensor_embed`. GL2 computes a signed deviation **in a record**; it changes no
encoder and writes nothing into EC. Encoding the record into EC is `latent_forward_model.md` S4, opt-in,
and must answer that review in its own front-gate.

## 3. Design

### 3.1 `InteroceptiveOutcome` — one signed body-consequence record

One record per body-consequence **event**: a tool invocation, or an out-of-band change detected at
`evaluate_failures`. It is the insular "what happened to my body, and was it good". It is both a learning
signal (later stages) and the forward model's target.

```python
@dataclass(frozen=True, slots=True)
class InteroceptiveOutcome:
    """CC3 path (a): defaults on every field + extra (JSON-only; __post_init__ rejects collisions).
    The defaults exist only to satisfy path (a); __post_init__ REJECTS the sentinel provenance "",
    so no record can exist without provenance. Built ONLY through sem.interoceptive_outcome(...),
    whose identity keywords are required (§3.1.4). GL2a's record has NO pid field (G17); the
    post-fence resume stage adds `pid` (additive under path (a)) and from then rejects pid None."""
    # pid: PhysicalEventId | None = None    # added at the post-fence resume stage (G17), REQUIRED from
    #                                       # then: the join key (agent_id, session_id, seq; G15)
    invocation_id: str = ""                 # executor uuid4, in-process diagnostic only; NEVER a join key
    agent_id: str = ""
    body_path: str = ""                     # whose body (drive names collide across bodies)
    provenance: str = ""                    # REQUIRED: experienced | narrated | imagined | apparatus (G16)
    sufferer: str = ""                      # entity path whose body changed (today's Wire-2 key)
    cause: CauseRef | None = None           # who/what did it; None = unknown / world
    # The PER-DRIVE BLOCK, sorted by name, only drives with a declared range (never imputed). On the
    # tool path ONLY the invoked affordance's own declared drives (G14). Persisted in full beside the
    # core (§3.1.4), so a later schema can be recomputed from the replay buffer.
    pressure_before: tuple[tuple[str, float], ...] = ()  # == existing drive_pressure (unsigned [0,1])
    pressure_after: tuple[tuple[str, float], ...] = ()   # drive_pressure after, NET of declared drift
    drive_delta: tuple[tuple[str, float], ...] = ()      # SIGNED [-1,1]: drive_comfort_progress / drive_span,
                                                         # net of declared drift: the physical description,
                                                         # not the valence (+ toward the set point)
    deviation_after: tuple[tuple[str, float], ...] = ()  # SIGNED [-1,1]; homeostatic (v - set_point)/span
                                                         # (+ above, - below); entropic: drive_pressure (>=0)
    caused: tuple[tuple[str, bool], ...] = ()            # per drive: True = this action's declared effect
                                                         # (every tool-path entry); False = felt (out-of-band)
    satiated: tuple[str, ...] = ()          # declared drives whose latch cleared in THIS invocation's own
                                            # evaluation (the two `elif cleared:` sites only)
    # body-agnostic core, each [0,1]:
    nociception: float = 0.0                # max PainKind.NOCICEPTIVE intensity (ANTICIPATORY excluded);
                                            # for drive:health, THIS event's injury (G11; see below)
    drive_pain: float = 0.0                 # max PainKind.DRIVE intensity
    relief: float = 0.0                     # max over drives of the DROP in drive_pressure (alliesthesia)
    harm: float = 0.0                       # max over drives of the RISE in drive_pressure;
                                            # TISSUE_DAMAGE_DRIVES excluded (health counts once, G11)
    urgency: float = 0.0                    # v1: max pressure AFTER (G12, §3.1.3)
    extra: dict = field(default_factory=dict, hash=False, compare=False)
    # extra on the tool path: drift_dt_s, the per-drive drift netted out, the observed (un-netted) delta,
    # nociception_basis ("caused_or_felt" until the bridge split lands, below)

@dataclass(frozen=True, slots=True)
class CauseRef:
    """CC3 path (a)."""
    entity: str = ""        # YAML noun of the causing entity ("fire_pit", "zombie"); never the sufferer
    affordance: str = ""    # "touch", "fire_breath"
    tool: str = ""          # tool signature when a tool call caused it
    # cause_pid: PhysicalEventId | None = None   # added at the post-fence resume stage (G17): the
    #                                            # physical event that caused it, when one did
    extra: dict = field(default_factory=dict, hash=False, compare=False)

# embodiment/event_id.py — a leaf module with no maxim imports, built at the post-fence resume stage
# (G17), not GL2a; GL3 imports it
@dataclass(frozen=True, slots=True)
class PhysicalEventId:
    """CC3 path (b): SHAPE-FROZEN at 1.0 (CC3) — an identity; a new field would change equality and
    break every persisted join. Deterministic: no uuid, no wall time (G3). __post_init__ rejects an
    empty agent_id, an empty session_id and a negative seq. str() is the join form
    "{agent_id}:{session_id}:{seq}" (G15 shape; the type, the sequencer and the resume land together,
    G17)."""
    agent_id: str
    session_id: str         # G15; its deterministic source is asked at the resume stage's start (G17)
    seq: int                # monotonic within the session, minted only by that agent's EventSequencer
```

**Computation is pure and lives beside the existing helpers** in `embodiment/sem.py`
(`interoceptive_outcome(specs, ranges, before, after, drift, pains, cleared, *, cause, provenance)`;
`pid` joins the required keywords at the post-fence resume stage, G17),
reusing `drive_span`, `drive_pressure`, `drive_comfort_progress` and `classify_pain`, plus one pure drift
helper factored from `embodiment/body.py::Embodiment.tick_vital_drift`'s arithmetic (G14), which
`tick_vital_drift` is reimplemented on, byte-identical over a value × `dt` grid on every shipped body.

**Relief and harm follow need (alliesthesia; GL1 bio-faithful review DNB-1).** A body-consequence code
that scores the same physical change the same way whatever the body's state is the caricature: warmth on
the skin is pleasant when the core is cool, neutral when the body is comfortable, and food is rewarding
when hungry and neutral when sated (Cabanac 1971). So the core reads the **change in `drive_pressure`**
per drive, `relief = max_d (p_before − p_after)⁺` and `harm = max_d (p_after − p_before)⁺`.
`sem.py::drive_pressure` is already 0 inside a homeostatic comfort band and on the satisfied side of an
entropic drive, so this is first-order alliesthesia with no new formula. `drive_delta`, the raw signed
progress over span, stays in the per-drive block as the physical description. Worked through the shipped
YAML: `fire_pit.warm_self` after two `cool_air.feel` gives harm 0 and relief ≈ 0.27; `blanket.touch` at
rest gives 0; sated eating gives 0.

**The relief pin, restated so it can fail** (wiring D1, confounding DNB-1). The existing trio is never
refilled (§3.1.4). The guard is: **the positive part of the record's `drive_delta` equals
`ToolOutput.drive_relief` for every drive present in `drive_relief`** (within float tolerance). The two are
computed by different code from different reads: the trio diffs around `_apply_sensor_deltas` inside the
tool, by bare name, before drift; the record diffs the executor's `_run_started` snapshot against the
post-run read through the resolver, net of the drift the body applied. So the pin fails if the drift
netting is wrong, if the resolver reads another slot, or if a foreign write lands inside the window, and
its deletion probe (remove the netting) re-reds it on the `_StepClock` 20 s case (§5.2). Qualified drives
are absent from `drive_relief` while #1161 is open: that difference is a named, #1161-scoped divergence
with its own `xfail(strict=True)` gate ("the record carries `arms.thermal`; `drive_relief` does not"),
which flips when #1161 lands.

**Health counts once (owner decision G11).** `drive:health` is carried as nociception, consistent with
`TISSUE_DAMAGE_DRIVES`, and is excluded from `harm`, so one transduction enters valence once. Its
nociception is **this event's injury**, the normalised Δhealth loss, not `drive_pain_for_value`'s level
(which codes the accumulated deficit: losing 1 hp at 11 hp would read 1.0, at 15 hp 0). The record's
docstring says it is an engineering proxy for injury (hypoxic health loss is near-painless); when the
damage source is unknown, `extra["injury_cause_unknown"]` is set so GL4 can separate drowning from contact
damage (the Minecraft bridge's `damage` event carries no source). **ANTICIPATORY pain is excluded** from
every field: a prediction is not an outcome (nociception_layer principle 4).

**Caused vs felt pain** (bio-faithful SF-2). `ToolPainBridge.pop_invocation_pain` returns the pain the
action caused when it caused any, else the pain the body felt while it ran. A forward model predicts
reafference, so the two must be told apart. Splitting them needs an additive accessor in
`bridges/tool_pain_bridge.py`, which is outside G1's exempt set, so it lands at the out-of-band producer
stage (after the fence, with the narrated and apparatus scopes, G9). Until then
the record carries `extra["nociception_basis"] = "caused_or_felt"`, and GL4 S0b counts those records
separately and trains on them only after the split.

#### 3.1.1 The forward-model projection (reconciled with `latent_forward_model.md`)

There is **one** record; the forward model's target is its projection, not a second record (an earlier
draft's separate `ConsequenceCode` is dropped).
`InteroceptiveOutcome.as_vector(schema_id="ans-v1")` returns:

- **Core, fixed 6-d, body-agnostic:** `[valence, nociception, drive_pain, relief, harm, urgency]`, with
  `valence = clip(relief − harm − nociception, −1, 1)`, unweighted, the v1 innate prior (owner decision
  G11). This is what lets a Minecraft burn and a cradle burn land near each other.
- **Per-body drive block:** `deviation_after ‖ drive_delta` in the body's declared drive order. Its
  dimension varies by body, so the predictor either trains per body or predicts the core only;
  `schema_id` + `body_path` fix the order and a mismatch refuses.
- **The forward-model target is the change-only subset** (bio-faithful SF-3). The core mixes changes
  (relief, harm, nociception) with levels (urgency and `drive_pain` are levels; so is `deviation_after`).
  Levels are largely predictable from the sensed context, so a predictor could score by copying the
  situation. The target projection is therefore the phasic dimensions; the levels are passed in as
  context. The record keeps every field; only the projection changes, and `latent_forward_model.md`
  carries the matching context-copy baseline.
- **Gates read per-dimension signs** (`nociception`, `harm`, `relief`), never `valence` alone; valence is
  reported beside them as the innate-prior summary (confounding SF-2).
- **The schema id moves with the target's meaning** (confounding SF-2). GL2b(ii) changes what
  `nociception` means for the same contact (the infant burn goes from DRIVE 0.04 to NOCICEPTIVE), so it
  bumps the schema id (`ans-v2`), and no training or evaluation set spans two schema ids (S0b and S2
  refuse a mixed set). Because the per-drive block is persisted, an `ans-v1` capture can be re-projected
  under `ans-v2`.

**One join rule everywhere:** a training pair joins on `pid`, on the tool path and out-of-band alike,
and on nothing else. The executor's `uuid4` invocation id is not persisted-stable and is never a join
key; from the post-fence resume stage (G17; GL2a records carry no `pid` and never train), on the
tool path `Executor._stamp_invocation` mints the `pid` and stamps it on the `ToolOutput`
(inside `ToolOutput.interoceptive_outcome`, beside the in-process invocation id) so the forward model's
`ActionContext` and this record carry the same one. While the target is this fixed code, the predictor
is supervised regression in a JEPA shape (owner decision G2).

#### 3.1.2 Identity: `PhysicalEventId`, one sequencer per agent, one declared cross-thread edge

A burn is one physical event seen on more than one `AfferentTrack` (GL3: nociceptive-fast and
affective-slow). Every track copy and this record carry the same `PhysicalEventId(agent_id, session_id,
seq)`. The contract (canonical across the grounding plans; amended 2026-10-09 by G9, G15 and G17).
**Everything in this subsection lands at the post-fence resume stage (G17), not GL2a**: GL2a's record
carries no event id, and records written before that stage are never forward-model training data.

- **One type, one leaf module, built at the resume stage.** `PhysicalEventId` lives in
  `embodiment/event_id.py`, a leaf with no `maxim` imports. GL3's relay imports it; it does not define it.
- **One seq authority per agent.** A per-agent `EventSequencer`, held by the agent's **primary**
  `Embodiment`, is the only minter. "Primary" means the `Embodiment` that
  `runtime/bootstrap.py::build_executor` constructs with a non-empty `agent_id`, and at most one live
  sequencer per `agent_id` per process is asserted, so a harness that builds two AUTs (or rebuilds one)
  fails loudly instead of running two seq authorities (wiring N4). The stage must define **release**
  (bio session end and executor shutdown release the agent's sequencer), name the two harnesses that
  rebuild one `agent_id` in one process today (`scripts/survival_world/r3_run.py`,
  `scripts/survival_world/exp61_run.py`) and how each releases before it rebuilds, and add a
  `tests/conftest.py` autouse reset of the live-sequencer registry (G17). Ephemeral, scene and foundry wrappers
  (`agent_id == ""`, the `simulation/tools.py` `scene_emb`, the `simulation/foundry.py` wrappers,
  `maxim.create.embodiment()`, the `scripts/orient_substrate/*` probes' `Embodiment(root=...)`) are not
  the AUT: they mint no record and no id, so public-API users of `create.embodiment()` get no record
  (stated, wiring S3). `seq` is assigned only when a record is **emitted**, so it is stable under
  `_StepClock` lockstep tests. The sequencer serialises minting with its own private lock (G9).
- **When a pid is minted** (wiring S3). Only when the invocation reached `tool.run` **and**
  `self.embodiment.agent_id` is non-empty. `_stamp_invocation` also runs on the inactive-scene,
  unregistered-tool and exception paths, and those mint nothing. A synthetic `ToolOutput` built by a
  wrapper (`runtime/fear_gate.py::FearGatedExecutor` returns a new one on a block) carries no record,
  which is correct.
- **No pid at GL2a; the identity and its resume land together (owner decision G17, superseding G15's
  GL2a half).** No deterministic session id exists today: every session id is wall-clock
  `time.strftime` (`simulation/orchestrator.py`, `simulation/research_orchestrator.py`,
  `simulation/report.py`, `runtime/loop_setup.py`, `runtime/agent_loop.py`), the minting sites have none
  in scope (`runtime/bootstrap.py::build_executor` takes no session parameter), and R3 and Exp 61
  rebuild the same `agent_id` in one process. So GL2a mints nothing, and **the type, the sequencer, its
  session-id source and the cross-session resume land together after the fence and before GL4 S1**,
  the first stage that persists pids where they are joined across sessions (the Cerebellum payload,
  `ActionContext`). The session-id source is asked at that stage's start: deterministic under
  `_StepClock` (no uuid, no wall time, G3); who mints it; its scope; `--resume-sim` behaviour; and a new
  required keyword on `build_executor`, with its callers. From that stage each record's `str(pid)`
  rides into the persisted trace through `EncodingSignals.extra["interoception"]`. The resume resumes from the sequencer's **own** high-water mark (a top-level key in
  an existing per-agent persisted file that carries `_format_version`, written at every save), taken as
  the max of that mark and any pid seq seen in a pid-bearing store. It does not resume from the
  Hippocampus traces alone, which lose pids: the capture queue drops its oldest entry when full, the
  store evicts and compresses, the PLANNING-approved, parallel and retry `executor.execute` paths never
  capture, and `load_persisted=False` agents never resume (wiring D2). It is wired at **both** load
  seams, `runtime/bio_stack.py::build_bio_stack` (which restores the Hippocampus before
  `build_executor` builds the `Embodiment`) and `simulation/orchestrator.py::_restore_aut_from_session`
  (which runs on `--resume-sim` after the AUT executor exists). Its gate drives both real load paths and
  never hand-seeds the sequencer; it fires T1-16's "Hippocampus save/restore or the resume path"
  wording if it touches the Hippocampus load or `RESUME_STORES` (wiring S9).
- **GL3 handover.** At GL3.B3, seq assignment moves to the relay scheduler's drain point and the
  resume stage's counter becomes the scheduler's backing store. One authority at a time: the handover is GL3.B3's stage
  gate, with a test that the two never both assign.
- **The orchestrator thread is a declared edge.** The narrator tools (`OrchestratorActorTool`,
  `DamageComponentTool`, `SetEntitySensorTool`, registered on `orch_registry`) call
  `_aut_embodiment.evaluate_failures()` on the orchestrator thread (the `start_simulation_mode` caller
  running the orchestrator agent's loop) while the AUT loop runs on `sim.aut` (§1.2a). The reflex
  dispatch's instances of the same classes run inside the AUT's `enrich`, on the loop thread. The agent loop is therefore **not** the only caller, and
  the Minecraft bridge is not the off-thread one (`MinecraftSyncPump._run` never calls
  `evaluate_failures`). **GL2a adds no lock over `evaluate_failures`** (owner decision G9) and mints
  nothing (G17). From the resume stage, tool-path minting runs wherever the AUT executor runs, and
  narrator calls mint nothing, so the sequencer's own private lock is enough. **With the out-of-band producer, after the fence,** `evaluate_failures`'
  latch, snapshot and mint section runs under one `Embodiment`-owned `threading.RLock` (re-entrant, per
  the CLAUDE.md threading rule: the section publishes pain synchronously, and whether any subscriber
  re-enters the body is checked in that stage's design pass, not assumed), which guards the state both
  threads touch (`Entity.drive_breach_severity` and the per-entity snapshot slot). No lock exists in
  `embodiment/body.py` today; adding it serialises an existing race and changes how narrator-thread pain
  publication interleaves, so it is called out in that stage's PR, which also names the out-of-band
  producer its two-thread test exercises (wiring S2).
  From GL3.B3, orchestrator-thread transduction posts into the relay's edge inbox, drained once per
  pass; GL3.B0's census and GL3.B3's gate 6 assert this thread identity.

#### 3.1.3 Urgency

Bio: urgency is need × imminence. v1 = `max(pressure_after)` (computable today, no clock; owner decision
G12). The slope term
(Δpressure per experience-clock µs, or time-to-deprivation for drifting entropic drives) waits for GL3's
timing work: world-owned Minecraft drives declare `drift_rate: 0` (`minecraft_player.yaml`), so imminence
must be measured, and the experience clock is the only legitimate clock (bio-memory brief).

#### 3.1.4 Where it is produced, and the provenance and forward-compat path

- **Tool path (owner decision G14; GL1 wiring D1, confounding DNB-1, environment DNB-1).** One additive
  field, `ToolOutput.interoceptive_outcome`, record-only, declared
  `field(default=None, repr=False)` (#1189, below). Its window and drive set are fixed:
  - **The window** runs from the raw before-snapshot `Executor._run_started` takes just before
    `tool.run` (beside today's `pressure_before`) to the read `Executor._stamp_invocation` makes after
    `tool.run` returns.
  - **The drives** are only the invoked affordance's own declared drives (the key set
    `tool_bridge.py::_drive_progress_by_drive` iterates), read through
    `embodiment/sem.py::_resolve_sensor_slot` / `_read_sensor_value`, so `arms.thermal` is never omitted.
    Body-wide change outside that set is not on the tool-path record; it belongs to the out-of-band
    producer. On `minecraft_player` the bridge affordances declare no `self_effect` (`escape_water`
    declares none; `eat` declares a stub `food` that the live readback owns), so a Minecraft tool-path
    record carries an essentially empty drive block, and every Minecraft body consequence, the
    post-return oxygen refill included, waits for the out-of-band producer (wiring S5, environment SF-2).
  - **Drift is netted out.** `evaluate_failures` applies `tick_vital_drift(now − _last_poll)` on entry,
    and on LLM-primary the previous evaluation is the loop tick before the LLM call, so a raw diff would
    book the whole LLM turn's drift as the action's consequence (a 20 s turn adds +1.6 `cold` on
    `infant_humanoid_chilled`, clamped at 1.0, against `warm_self`'s −0.3). The body records, for the
    calling thread's evaluation inside this invocation, the drift it applied per declared drive (the
    pure helper above) and the interval (`extra["drift_dt_s"]`); the record reports values net of it
    and keeps the observed delta in `extra`. Drift is never advanced early from the executor: that
    would move pain-publication cadence.
  - **Satiation** is detected only at the two `elif cleared:` sites, recorded by the body for the
    calling thread's evaluation inside this invocation. Never from a latch diff, and never from the
    silent pop on an unreadable sensor (`ent.drive_breach_severity.pop(ds_name, None)` in the
    `current is None` branch).
  - **The existing trio stays byte-identical.** `drive_pressure_before`, `drive_relief` and `pain` are
    computed exactly as today and are never refilled from the record: `drive_relief` reaches
    `EncodingSignals.drive_relief` → `memory/encoding.py::encoding_tag` → `storage_strength`, so
    refilling it would be a memory-strength change with its own T1-16 walk. A golden taken from the
    pre-change commit pins the trio, `encoding_tag` and `storage_strength` for every capture of the
    scripted cradle sequence and the Minecraft `fear_water` arm; it is byte-identical after the change,
    and it still passes with the record reverted (the reverse deletion probe: the record never fed the
    trio).
  - **The residual a narrator write leaves** (environment SF-6, confounding SF-1). In `--sim` a
    narrator write to a declared drive that lands inside the window (the arc tells the narrator to raise
    `arms.thermal` toward 0.8 on approach to the fire pit) is in an `experienced` tool-path record. The
    relief pin catches it on drives the trio sees. The structural fix, a per-body write epoch bumped by
    any write outside the executor's bracket that marks the record `mixed`, needs `_write_sensor`
    (outside the exempt set), so it lands with the narrated scope. Until then §10's narrator invariant is
    scoped to out-of-band records, and S0b/S2 count `--sim` tool-path records separately.

  It is a `ToolOutput` **field**, stamped by the executor like the trio, not a `side_effects` key:
  `docs/user/tool_side_effects.md` registers what a tool's `execute()` reports, and tools never set this,
  so it needs no registry row. The forward model's `ActionContext` is built inside the tool but attached
  by the same executor stamp, rides the same way (a field, `repr=False`), and that decision is
  `latent_forward_model.md` S1's.
- **Out-of-band path (after the fence, owner decision G9):** `evaluate_failures` compares against the
  previous evaluation's per-entity snapshot and queues a record; `Embodiment.drain_outcomes()` returns
  them. Emission is **latched like channel 2**: band entry/exit, deepening past `_BREACH_DEEPEN_FRACTION`,
  a nociceptive event, or a satiation — never per tick. The snapshot stores the values the latch was
  evaluated on, never a re-read, and a Minecraft record carries the bridge's `state_age_s` at both ends,
  since the sync pump writes on its own thread every 0.5 s (environment NIT-4). A tool's delayed
  consequence (oxygen recovering after `escape_water` returns) joins its invocation through
  `CauseRef.cause_pid`: the next out-of-band record within a declared horizon carries the invocation's
  `pid` (confounding SF-1, environment SF-2). `cause_pid` and `pid` exist from the post-fence resume
  stage (G17), so this join needs that stage as well as this one.
- **`drain_outcomes()` lifecycle (with the out-of-band producer, after the fence).** The production
  drainer is the loop capture (`runtime/tool_dispatch.py` →
  `runtime/bio_integration.py::capture_loop_action`) on the agent-loop thread, in substrate-primary and
  LLM-primary alike; in `--sim` that is the AUT loop on `sim.aut`, which also drains the records the
  narrator's tools queued from the orchestrator thread. The queue is **bounded** (owner decision G10):
  the bound is measured from a scripted session's peak, and on overflow it drops the oldest record,
  counts the drop and logs it (never silent). Ephemeral wrappers never queue (§3.1.2). A body attached
  where no loop capture runs queues up to the bound and then drops with the count; that stage's design
  pass lists such runtimes. GL2a builds none of this: under G9 the queue would have no producer, so its
  bound would measure 0.
- **Consumers in GL2a:** a trace line, and the loop capture writing the core, the per-drive block
  and `provenance` (no `pid` until the resume stage, G17) into `EncodingSignals.extra["interoception"]` (no `EncodingSignals` field
  change; `encoding_tag` reads declared fields only, never `extra`). It is still a change to the
  **persisted memory record**: `EncodingSignals.to_dict` flattens `extra` into the trace the Hippocampus
  saves, so every loop capture's saved shape grows by this key (blast radius in §5.2).
- **"No reader acts" holds only because the field is `repr=False`** (GL1 wiring D3; issue
  [#1189](https://github.com/dennys246/Maxim/issues/1189)). `Hippocampus.capture_from_loop` stores the
  `ToolOutput` object in `Outcome.result`, `atomic_write_json` serialises it with `default=str`, and
  `Hippocampus.search_by_content` substring-matches `str(value)`. That is Path 3 of
  `integration/bio_enrichment.py::_query_hippocampus`, and `tools/narrative.py`,
  `agents/exec_agent.py` and `simulation/introspection.py` call it too. A default-repr record would put
  `relief`, `harm`, `oxygen`, `thermal` and cause nouns into the searchable text of every captured action.
  So the field is `repr=False`, a guard pins `str(ToolOutput)` byte-identical with and without the
  record, and the T1-16 discharge covers the substring path (§5.2). The existing stamps (`pain=`,
  `drive_relief=`) already leak this way; the root fix (a declared projection at capture) changes the
  persisted shape and is #1189's, an owner decision, not a GL2a fold.
- **Provenance (owner decisions G6, G16).** Four kinds: `experienced`, `narrated` (discounted, G6),
  `imagined`, and `apparatus` (excluded, G16); `declared` / `reported` stay open at GL3.B1, the
  registry+provenance stage. **`apparatus`** is harness-scoped: writes the experimenter makes to a drive,
  namely the water trial's rescue teleport and `/effect` heal (`WaterTrial.rescue` / `heal`), R3's
  respawn, Exp 56's teacher feed (`scripts/exp56/common.py`) and Exp 52's mother feed
  (`simulation/cradle_mother.py::reactive_mother_tick`), and any other harness write to a drive. An
  apparatus record is kept for the report, is never a forward-model training target and never pays
  credit. The apparatus scope is a **window**, not a thread-local: a harness write opens an apparatus
  epoch on the body, and the evaluation(s) that next observe that write are stamped `apparatus` until
  the epoch closes. A thread-local scope cannot work here, because the write and the evaluation that
  sees it run on different threads: the rescue's oxygen change arrives on the `mc-sync` thread
  (`MinecraftSyncPump`), and the latch clears on a later loop-thread `evaluate_failures`. The epoch's
  exact extent (how many evaluations, or until which state) is that stage's design-pass item. It lands
  with the out-of-band producer; until then no out-of-band record is minted, so none can be mislabelled.
  GL2c's "a rescue-caused crossing delivers nothing" gate drives the **real** rescue path,
  `scripts/survival_world/water_trial.py::WaterTrial.rescue`, never a hand-written drive write. Consequences written by the narrator's tools
  (`simulation/tools.py::SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool`, and the
  reflex dispatch, which uses separate instances of the first two classes on the loop thread) are stamped **`narrated`, never
  `experienced`**. A narrated record is usable for forward-model training and for credit at a **declared
  discount** (value: owner decision at GL4 start; the strict default for that decision is a small
  discount), and S5/GL5 report every result with and without narrated data. Excluding narrated records
  outright would mute the world the LLM's language priors simulate (owner rationale). `imagined` records
  are never produced by GL2. Mechanism: the narrator tools run their `evaluate_failures` call inside an
  `Embodiment` narrated scope (thread-local, so a concurrent loop-thread evaluation stays
  `experienced`); the Embodiment passes the scope's kind to the factory explicitly, and tool-path records
  minted by the agent's own executor are `experienced`. The scope lands after the fence with the
  out-of-band producer (owner decision G9); until it lands, no out-of-band record is minted. The forward
  model's gates read experienced data only; narrated and apparatus numbers are reported, never gated
  (confounding SF-7: a narrated target is itself generated from names; this does not reopen G6's
  discounted use in training). Credit already paid
  today by PainBus for narrator pain is unchanged in GL2 (changing it would be its own ledger walk); the
  discount applies to this record's consumers. The forward model's contamination guard checks that a
  narrated record can never be relabelled `experienced` and that the discount is applied. Frozen does
  not prevent a relabel: `dataclasses.replace(rec, provenance="experienced")` builds a new record, so
  the no-relabel guard tests that call too, and asserts that the factory / validator path refuses it or
  that the contamination test catches it (no relabelling helper exists either).
- **Forward-compat:** `InteroceptiveOutcome` and `CauseRef` take CC3 path (a), which is what lets the
  resume stage add `pid` / `cause_pid` as defaulted fields (G17); `PhysicalEventId` takes path
  (b) with the rationale above. Serialised through `to_dict`/`from_dict` inside files that already carry
  `_format_version` (traces, Hippocampus); no new persisted file in GL2. No value is hashed; if a later
  stage hashes an id across processes it uses `utils/seeding.py::stable_hash_64_signed`.
- **Structural enforcement:** the factory's `cause=` and `provenance=` are **required keyword-only**
  (and `pid=` from the post-fence resume stage, G17), so forgetting them is a `TypeError`, and
  `__post_init__` rejects the `provenance=""` sentinel (and, from that stage, `pid=None`), so a record
  without them cannot be constructed at all. No type in the
  grounding line defaults provenance to `"experienced"`. Structural guard: the factory signature plus
  the stage's AST test that `InteroceptiveOutcome(` is never constructed directly in `src/`.

**#1125 is merged; #1161 is the only relief-side blindness left.** #1125 closed 2026-10-08 (#1164): the
records read modulator drives through the one resolver, `embodiment/sem.py::_resolve_sensor_slot` /
`_read_sensor_value`, and GL2a reads its drives through it, so it never ships a record that omits
`arms.thermal`. It must not derive `drive_delta` from `side_effects["drive_progress_by_drive"]` while
#1161 keeps that side effect blind (`_drive_progress_by_drive` still reads `vital_metrics` by bare
name).

### 3.2 Burn as nociception: `NociceptorSpec`, a second receptor on the same sensor

Do **not** reclassify `arms.thermal` as a tissue-damage drive and do **not** raise its `pain_scale`. That
conflates thermoreception with nociception, makes innocuous warmth painful, and moves the Exp 37
calibration the YAML comment says those values were tuned for. Instead a sensor may declare a nociceptor
beside its drive:

```yaml
arms:
  sensors:
    thermal:
      drive: {drift_mode: homeostatic, set_point: 0.0, comfort_band: 0.5, pain_scale: 0.4, ...}  # unchanged
      nociceptor:            # NEW, optional
        modality: heat       # heat | cold | mechanical | chemical; VALIDATED (heat ⇒ above, cold ⇒ below)
        direction: above     # above | below
        threshold: <owner>   # noxious onset, sensor units
        pain_scale: <owner>  # intensity per unit past threshold, clamped [0,1]
```

Parsed into a new frozen `NociceptorSpec` (CC3 path (a)) on `Entity.nociceptor_specs` — not a field on the
frozen drive specs. `modality` gets a reader, not decoration: the parser validates it against
`direction` (heat ⇒ `above`, cold ⇒ `below`), and GL3.B3 uses it for track routing (bio-faithful N3). Evaluated in `evaluate_failures` inside the drive loop's structure:

- **Channel 1:** a FailureEvent named `drive:<sensor>:noxious`, state-based like every drive failure.
  `drive_failure_sensor` parses it unchanged (sensor = `arms.thermal`), so **B8 applies with no parser
  change** and a bystander affordance is never blamed for a lingering burn.
- **Channel 2:** published once on threshold entry and on material deepening, through the same
  entity-owned severity latch (key `noxious:<sensor>`), with `context["source"] = "nociceptor:<sensor>"`
  and `context["failure_mode"] = "drive:<sensor>:noxious"` (the drive pain uses
  `failure_mode = "drive:<sensor>"`, so PainBus's `(entity, failure_mode)` refractory keeps the two
  apart).
- **Classification:** `classify_pain` no longer matches `drive:` and falls to the type (EXTERNAL_SIGNAL →
  NOCICEPTIVE). `failure_pain_kind` gains one rule: band `noxious` ⇒ NOCICEPTIVE (it returns DRIVE today
  for any non-health drive band). That is the only classification edit.
- **The ReactionBus refractory collision.** `reactions/bus.py` keys its refractory on
  `f"{kind}:{source}"`, and `reactions/compat.py::pain_signal_to_reaction` sets
  `source=f"pain_detector:{signal.pain_type.value}"` — `pain_detector:external_signal` for **all** body
  pain, drive and nociceptor alike. So the drive Reaction (0.04) and the nociceptor Reaction from one
  `evaluate_failures` call coalesce inside the 0.5 s pain refractory, and whichever publishes first wins:
  the burn can reach `_distribute_reward_from_reaction` as 0.04. The fix is nociception step 2 (carry the
  stored source/kind onto the `Reaction`, F3), which therefore lands with or before this stage; the
  stage's golden table covers **both** refractories (PainBus per `(entity, failure_mode)`, ReactionBus
  per `kind:source`) and a test asserts both Reactions reach the distributor.
- **Side-effects registry.** `docs/user/tool_side_effects.md`'s delta-attribution note names the drive
  bands `discomfort` / `deprived`; the `noxious` band is added to that grammar in the same PR (the
  registry is append-only, so this is an append).

**Why not a standard `failure_modes:` entry** (which would need zero code): standard failure modes are
state-based on **both** channels (`FailureMode.evaluate` fires on every call while triggered unless
`persistent`), bypass B8 ("standard failure modes pass through unchanged") and are not latched on PainBus.
A YAML burn would flood PainBus every tick and blame every bystander while `arms.thermal` drifts back —
the Exp 42 pollution B8 exists to prevent. The channel split and B8 are preserved exactly because the
nociceptor rides the drive loop's machinery, not the failure-mode list.

**Calibration constraint (values are the owner's, decided at GL2b(ii)'s start, §8):** the learning
threshold this stage must clear is whatever that stage-start calibration decision fixes; the proposal is
that a +0.6 contact from rest reaches ≥ 0.4 (the highest PainBus learner threshold, so every learner
sees it), and that a single `warm_self` (+0.2) or any `*_safe` / `green_hearth` (+0.05) application
produces 0. Note the nociceptor reads sensor **state**, not the affordance: repeated `warm_self` that stacks
`arms.thermal` past threshold is noxious, which is bio-faithful, and the GL1 census should report which
shipped sequences stack past it. **The shipped scenes rarely touch from rest** (GL1 environment review
SF-3): SEM effects are additive deltas clamped to range, `cool_air.feel` takes `arms.thermal` −0.15 per
call (GL2a's own sequence reaches −0.30, so a touch lands at 0.30), and the arc tells the narrator to raise
`arms.thermal` toward 0.8 on approach (a touch from 0.8 clamps at 1.0, delta +0.2). For a touch from a
chilled arm to be noxious the threshold must be < 0.3; for a single `warm_self` from rest to give 0 it
must be > 0.2; in that band two stacked `warm_self` are noxious and noxious onset sits below the thermal
comfort band (0.5), which inverts the biology. So the stage-start calibration reads the `arms.thermal`
distribution at `touch` / `warm_self` time over the committed cradle data first, then either states the
gate over those starting states ("noxious from ≥ X") or gives the variant body a contact affordance that
is not additive (a new authored `touch` that sets a floor, declared on the variant, never an edit to a
shipped one). It also names the narrated share: the burns the arc produces through
`SetEntitySensorTool`. The gate adds a "`warm_self` after `touch`, arm still hot" row beside the
from-rest rows (confounding NIT-3). **Body placement:** declare it on a **variant body**, never by editing
`infant_humanoid.yaml`, which `infant_humanoid_chilled` (Exp 42), `_cold`, `_naming_v1` and
`infant_operant*` all extend; Exp 42's `warmth_alpha_harm` drives `arms.thermal` to 1.0, so a nociceptor on
the base body changes T1-6's fixture directly.

Tier: the threshold and gain are an **innate prior** (tier 2). Habituation and sensitisation stay deferred
to `adaptive_nociception.md`, at this producer, never in a consumer. The nociceptor is also GL3's first
two-track receptor: one thermal stimulus, a nociceptive-fast and an affective-slow event, one
`PhysicalEventId`. That fan-out is GL3.B3 (the first track slice, no preemption), built with this
stage's `NociceptorSpec`; only nociceptive-fast **preemption**, GL3.B4, is a declared 1.4 rung arm (owner
decision G8). Bio mapping for those tracks: thermoreception and thermal pain both ascend the
anterolateral (spinothalamic) system — Aδ "first pain" on the fast lateral route, C-fibre "second pain"
on the slow spinoreticular/affective route — never the dorsal-column `mechano_proprio` track. First and
second pain come from two **fibre populations** with different thresholds and kinetics (Aδ and C), not one
receptor fanned onto two tracks; the fan-out's docstring says "FUNCTIONAL: one threshold for both fibre
classes" (bio-faithful N4). That the slow affective track also carries **drive** breaches is justified
only as **homeostatic afferents**: lamina-I spinal afferents for the infant's thermal discomfort, and the
vagal / NTS cranial route for air hunger, which is chemoreceptive (carotid and medullary chemoreceptors
→ NTS). Hunger is mostly humoral state acting on the hypothalamus, not an afferent event (bio-faithful
N2). Outside that reading, drive breaches have no claim on a pain track.

### 3.3 Two-sided corrective needs (cold path byte-identical)

Add `sem.corrective_need(spec, value) -> tuple[str, float] | None` returning a direction (`"below"` /
`"above"`) and an intensity. **`corrective_need_intensity` stays byte-identical and is reimplemented as the
`"below"` projection** of the new function, so the cold path cannot drift: a guard compares the two over a
value grid on every shipped body. In `substrate_proposal.py` the need map becomes direction-aware:
`thermal`/`temp`: below → `"cold"`, above → `"heat"`; every other entry (`food` → hunger, `health` →
threat) keeps no above branch. `NAc._DRIVE_TOOL_AFFINITIES` gains a `"heat"` entry (keywords are the
owner's; `"withdraw"` is already the `pain` and `threat` keyword). Minecraft bodies declare no `temp` /
`thermal` drive (`minecraft_player`, `minecraft_bench*`, verified by YAML grep), so their
`_read_drive_states` output is identical.

Tier: **innate prior** (the need → affordance affinity table is the cold-start prior). Behaviour-tiers
rule: the hard-coded prior gets a follow-up issue with its migration trigger in the same PR.

### 3.4 Valence keyed on the CAUSE, with sign, beside the sufferer rows

- `Embodiment.evaluate_failures(*, cause: CauseRef | None = None, cause_sensors: frozenset[str] =
  frozenset())` — keyword-only, defaults = today's behaviour. The cause is attached **per sensor**: only
  to the failures on sensors in `cause_sensors`, the set B8 (`tool_bridge.py::_intrinsically_harmful_sensors`)
  marks as harmed by **this call's** own delta. Channel 2 publishes inside `evaluate_failures`, but
  today `ModulatorAffordanceTool.execute` computes B8's set **after** `evaluate_failures()` returns; B8
  depends only on the effect dicts and the specs, so the caller computes it first and passes it in (GL1
  wiring review S6). A breach lingering on another sensor never inherits the cause. PainBus's
  `(entity, failure_mode)` refractory ignores the cause, so a second cause on the same mode within 0.5 s
  is dropped (stated). Callers that know the cause pass
  it: `ModulatorAffordanceTool.execute` (cause = the affordance's own entity + affordance, gated by its
  `self_effect` / `target_effect` harmful set) and `simulation/tools.py::OrchestratorActorTool`
  (`source_entity`, `source_affordance`), which today runs no B8 at all: the stage computes the same
  harmful set over the actor's `target_effect` on the AUT root and gates the cause on it (the actor
  path is narrated, §3.1.4). `_publish_pain` / `_publish_drive_pain` add `context["cause_entity"]`,
  `context["cause_affordance"]` on the gated failures only.
- `create_percept_valence_subscriber` writes an **additional** row when a cause is present. **The
  sufferer row stays exactly as today** (it means "which of my parts hurts", and persisted rows keep their
  meaning). Sign: pain → negative, as today; satiation or relief caused by an entity (blanket, mother,
  food) → positive, GL2c only.
- **Cause rows need their own key namespace.** Sufferer and cause keys share the shape
  `{agent}\x1f{entity_class}\x1f{failure_mode}`, and the same noun can be both (an acquired
  `rusty_sword` is a sufferer when it is damaged and a cause when it cuts). The cause row's middle field
  is namespaced, and the namespace must survive `hivemind/bundle.py`'s scrub, which keeps a row only if
  that field matches `_IDENTIFIER_TOKEN` (`^[A-Za-z0-9_.-]{1,64}$`): a `cause:` prefix would be dropped
  silently on export, `cause.` would ship. Every consumer of `percept_valences` is walked in the stage:
  `nac.json` save/load; the bundle scrub; `hivemind/merge.py` (`_merge_mean_clamped` and
  `_TIGHTEN_ONLY_FIELDS`, where `percept_valences` is tighten-only); `hivemind/ingest.py` (bounds,
  report, `entry_index.keep_agent_rows`); `analysis/substrate_diff.py`; the Exp 61 donor check
  `scripts/survival_world/exp61_run.py::donor_sanity_staged` (`FEAR_MODE in str(k)` matches cause rows
  too); and the pain→NAc subscriber, which passes the whole `signal.context` to `record_outcome_full`, so
  the two new keys also enter causal-link `context_factors`.
- **The reader is not additive.** `get_percept_aversions` keeps negative rows of every key, so negative
  cause rows reach `GatingContext.learned_aversions` → `TextSalienceScorer` → the ThoughtGate /
  enrichment path (`integration/bio_enrichment.py`'s `_snapshot_learned_aversions` injection) — a live
  LLM-path behaviour change on any runtime that learns aversions. No ledger row's trigger names that
  reader (grep of `behavioral_graduation_candidates.md`), so nothing fires by wording, but the change is
  real. (A namespaced key would also match oddly: `runtime/gating.py::_match_learned_aversion` splits keys
  on `_`, `-`, `:` and whitespace but not `.`, so `cause.fire_pit` yields the fragments `cause.fire` and
  `pit`.) Strict default (owner decision at GL2b(iii) start): cause rows are written and the aversion
  reader excludes the cause namespace, so the aversion map stays byte-identical until a declared arm
  turns it on; and cause rows are excluded from bundle export, with a count, until a transfer experiment
  declares them.
- Positive rows are invisible to that reader (it treats ≥ 0 as zero). An approach reader is a separate
  mechanism that must be earned; it is not proposed. **Positive cause rows (GL2c) get their own key**, so
  appetitive and aversive values are never netted into one scalar (biology keeps them in partly separate
  populations, basolateral amygdala positive and negative neurons; Namburi et al. 2015).
- **Cause rows are cradle / `--sim` only.** The Minecraft bridge's damage signal (`entityHurt` →
  `event("damage", ...)`) carries no attacker or source, and the snapshot carries no cause, so every
  Minecraft world consequence has `cause=None`. A Minecraft cause needs a bridge protocol change, which
  fires the re-run trigger on four EARNED rows and gets its own ledger walk (GL1 environment review SF-7). On substrate-primary the negative rows have no
  selection reader either (the reader is the text salience scorer) — stated, not fixed here.

Tier (re-tiered 2026-10-09, GL1 bio-faithful review SF-6): cause attribution is an **engineering prior
(oracle attribution)**, not an invariant. Pavlovian learning attaches value to the *perceived* cue by
contingency, with cues competing through prediction error (Rescorla–Wagner, Kamin blocking); handing the
learner the producer's YAML noun as ground truth means only one cue can ever be credited, and
`NAc.record_percept_valence` (`current + α·v`, clamped) has no prediction error and no extinction. So
cause rows can never be cited as learned stimulus valence, in GL5 or in any claim, and GL2b(iii)'s PR
files a follow-up issue with a trigger for RW-style updating (prediction error, extinction) on the cause
namespace. The
valence values are **learned**; that a bystander never becomes a cause stays an invariant.

### 3.5 The positive producer: satiation and relief

Three positive kinds, kept distinct because biology keeps them distinct (GL1 bio-faithful review SF-5;
the crossing rule is an innate-prior simplification, since biological satiety is partly pre-absorptive):

- **Satiation (a consummatory or corrective act):** the breach latch clears with a latched severity
  present **and** a tool cause (`cause` not None): eating, warming. The event the `EntropicDriveSpec`
  docstring promised; the homeostatic equivalent is a latched breach clearing back inside the hysteresis
  band. Emitted **once per breach episode**, never per tick. The `"satiation"` `ReactionKind` is used only
  for this kind.
- **Relief at the offset of an aversive state:** nociception or air hunger stops (negative reinforcement;
  pain-relief learning gives cues present at pain offset a positive value, Tanimoto, Heisenberg & Gerber
  2004). Oxygen recovery is this kind: air hunger is relieved, not sated. It is recorded at the event where
  the aversive input stops.
- **Recovery (passive or world):** a latch clears with `cause` None, e.g. `minecraft_player` `health`
  regenerating past its band long after the harm stopped. It is recorded and pays **no** credit unless an
  arm declares it; routed through the distributor it would credit whatever action happened to be
  eligible.
- **Graded relief** (on the record): the drop in `drive_pressure` on any event (§3.1).

**Intensity and expectation, inputs to GL2c's joint review** (bio-faithful SF-4). A satiation's intensity
is the episode's latched severity (`Entity.drive_breach_severity`, the deprivation depth), never the delta
of the step that crossed the threshold, which can be a last drift across the hysteresis band. And it is
delivered as a prediction error against the relief store's expectation for that cluster: dopaminergic
reward is a prediction error, and a fully predicted reward produces none (Schultz 1997). Without that, a
repeated surfacing saturates the bias, which then measures how often the act happened, not its value. The
store is the natural home for the expectation, which is a bio argument for option B or C over A; it does
not change G7.

**Record stage (GL2a):** `InteroceptiveOutcome.satiated` + `relief`. Zero consumers; its only ledger
consequence is GL2a's record-shape walk (§5.2).

**Why every routing reaches the earned survival rows (owner decision G7; its description corrected
2026-10-09 by G16).** The crossing is not a rare event on the bodies the earned rows ran on (§1.2 item 2),
but on the earned campaigns the **apparatus** makes it, not the agent. Exp 60/61 training is
propose-only and every episode ends in `WaterTrial.rescue`, so the `oxygen` latch clears during the
rescue's teleport and settle (clear at ≥ 15.2; breach below 14); Exp 60's probes are capped before the
breach and never latch; R3's respawns reset health and oxygen to 20; `food` (`satisfaction_threshold:
16.0`) is not reachable natively in campaign time; and `minecraft_bench` `d1` (Exp 56) and
`minecraft_bench57` `d1` (Exp 57, the same spec, `satisfaction_threshold: 0.3`) are written by the
teacher, as Exp 52's mother writes `hunger`. Those are `apparatus` records (G16), never credit. That
matters because censoring is more common in the arm that did not escape, so an apparatus-caused positive
would be **differential by arm** on the comparison T1-13 measures. The rows still fire: any delivery of a
positive signal at that site — through a `Reaction`,
through the relief store, or both — feeds a new positive value into the place the survival and transfer
claims were measured. **GL2c therefore fires T1-11, T1-12, T1-13, T1-14 and T1-15 whatever the
routing** (T1-12 added by the owner 2026-10-08; its `Re-run on:` matches T1-11's). No
routing "avoids" the T1-13 lapse; the routing only decides which surface the positive value lands on.

**Delivery (GL2c) — three routing options; the routing stays an open GL2c decision:**

| Option | Route | Reaches |
|---|---|---|
| B — relief store only | the satiation/relief record is the input to roadmap Phase 5's cluster-keyed relief store (world-cluster keyed), and nothing else. No `Reaction` is emitted; `_reward_bias` is never written by this producer. | the relief store; its reader is rung E2's, designed in the joint review |
| A — Reaction stage | `Reaction(kind="satiation", valence=POSITIVE, intensity=<the episode's latched severity>, context=ReactionContext(agent_id=<body agent>), source="drive:<name>:satiation")` on `pain_bus.reaction_bus` | with no other wiring: `hippocampus.capture_reaction` (subscribed to every Reaction; it appends to the pending episode, so satiation changes persisted episode valence and content) and `_distribute_reward_from_reaction` → `TemporalCreditDistributor.distribute` → `NAc.credit_node(+)` on **every** `(agent, node)` in `NAc._eligibility`: EC **text** nodes (`LinguisticEncoder` paths), **sensor-cluster** nodes (`SensorEncoder`; their `reward_bias` has no reader, #911, but trips `donor_sanity_staged` / `_R3._boundary`) and whatever bystander `tool:<name>` keys are still eligible (positive-direction B8 pollution the pid-keyed dedupe does not cover) → `_reward_bias` → `recommend_action` Component 2 (`reward_bias(agent_id, "tool:<name>")`) and EC text-threshold widening. The first time `_reward_bias` can grow in production. The ReactionBus default refractory (0.5 s, keyed `satiation:drive:<name>:satiation`) drops a second crossing of one drive inside the window (stated). |
| C — both, behind one switch | A and B | union |

Rows fired, every option: T1-13 (by its explicit lapse clause under A; by the measured contingency under
any option), T1-14 and T1-15 (by inheritance, and T1-14 functionally through the donor sanity check),
and T1-11 and T1-12 (Exp 56's and Exp 57's `d1` is an entropic drive with a satisfaction crossing,
so it can cross back inside a campaign; whether it does is the written argument, or the re-run, owed
with the batch, one per row). The
earlier recommendation was B, for one reason that survives G7: the relief
store's rule that "no second store is ever created" gives the positive write one home. It is **not** a
reason about blast radius.

**The committed refusals are RESTATED, not deleted.** Two frozen instruments refuse exactly the state a
positive producer creates:

- `scripts/survival_world/exp61_run.py::donor_sanity_staged` refuses a donor whose `links`,
  `event_outcome_welford` or `cluster_reward_bias` is non-empty ("a probe or an execution happened
  before export"), and whose node-level `reward_bias` is non-EMPTY: any non-zero bias ("a positive
  reaction was credited") and any zero-valued node key alike (the zero-key refusal sits beside it in
  the same function).
- `scripts/survival_world/r3_run.py::_R3._boundary` refuses when `reward_bias` or `links` is non-empty
  at the gauntlet boundary.

A satiation during donor training or before an R3 boundary (surfacing, regeneration, eating) would trip
them under option A, and under option B wherever the relief store writes into a refused field. GL2c
re-states each refusal for the new producer (e.g. "no non-zero bias from a non-satiation source",
"satiation entries are counted and reported, never silently admitted"), never deletes one, and the
re-statement is a change to a frozen instrument with its own review.

**GL2c ships off by default and lands only with a batched live re-run.** The switch is a declared
`maxim config` key, default OFF, added to M10's harness fingerprint (amended 2026-10-07: no
grounding-line flag is set in an E1–E3 arm unless the arm declares it). The PR that lets it be turned on
lands only with a batched live re-run of Exp 60, 61 and 62 (T1-13 / T1-14 / T1-15) plus the T1-11
and T1-12 arguments or the Exp 56 / Exp 57 re-runs, all under the restated refusals. The rig slot is its own Track C item, after
the 1.3.2 live Exp 60 re-run, never stacked on it or on an E-rung campaign.

**T1-13's lapse clause, explicitly.** T1-13's discharge (#888, restated in the #851 walk) rests on "the only
positive Reaction constructor in `src/` is `CerebellumModulator` … never given a `reaction_bus`" and ends:
"**If a positive Reaction emitter is ever wired, this reasoning lapses and the trigger applies.**" Under
option A that sentence fires by its letter. Under B or C it fires in substance: a positive producer
exists on the body T1-13 ran on, and whether any agent-caused (non-apparatus) crossing reaches the
`escape_water` contingency the discharge argued nothing could reward is exactly what the batched re-run's
per-arm, per-cause satiation counts show (G16). GL2c's PR records the walk entry on T1-13 either way.

**Narrated and apparatus satiation.** A crossing caused by a narrator write (e.g. a narrated feeding
through `SetEntitySensorTool`) is a `narrated` record (§3.1.4) and is delivered at the declared narrated
discount, never at full weight. A crossing caused by the apparatus (a rescue, a heal, a respawn, a
teacher or mother feed) is an `apparatus` record and delivers **nothing** (G16). The harnesses enter the
apparatus scope around `rescue` / `heal` / respawn and the feeds, and GL2c's flag-on gate asserts that a
rescue-caused crossing delivers nothing.

**Double credit (option A or C; extended by G16).** A tool-caused satiation is also credited by channel 3
(`update_cluster_reward`, a different map) — the nociception plan's F2 shape — and a teacher or mother
feed is already credited by `NAc.credit_operant_reward`. Bio: phasic dopamine fires once per unexpected
reward. Recommendation: one credit per physical event, keyed by `pid`; the distributor skips a satiation
whose cause is a tool channel 3 already credited, or a feed `credit_operant_reward` already paid. Named
in the Wire-integrity review.

**engram_formation.md E4's overreach set is mandatory under A or C** (GL1 wiring review S7). The
distributor's eligible set includes EC text nodes, so option A pays positive `_reward_bias` to them, and
E4's 13-string widening overreach set is read before A or C is chosen (roadmap T7). Either satiation
credit is gated to the causing `tool:` key, or the eligible-set composition is recorded as an input to
the joint review. Option B does not reach text nodes.

### 3.6 The `≥ 0` reward-bias clamp is kept

`NAc.credit_node` clamps `_reward_bias ∈ [0, max_reward_bias]`, zero meaning absent (T3-19, Tier 2). This
plan keeps it, for three reasons. (1) A positive producer is exactly what a `[0, max]` surface was built
for; the defect was the missing producer, not the clamp. (2) Opening the clamp would let every negative
Reaction — pain, and today also F1's anticipated pain — write a negative bias straight into
`recommend_action` Component 2: a new aversive selection path that fires T1-7, T1-11, T1-12 and T1-13 at
once, with no experiment asking for it. (3) The negative half of the signed record already has homes that
earned rows depend on: situation fear (Wire 4) and percept valence (Wire 2, now cause-keyed). One known
weakness is recorded, not fixed: T3-19's rationale cites "pain avoidance via valence on edges", which is
the Dormant T3-7 path (GL0 truth item 13 corrects the wording).

## 4. Behaviour tiers

| Behaviour | Tier | Follow-up / migration trigger |
|---|---|---|
| Computing the record from declared specs (deviation, delta, pressure, satiation crossing, drift netting) | invariant | — |
| Relief and harm as the change in `drive_pressure` (first-order alliesthesia) | innate prior | learned only via the forward model |
| Core valence weighting (`relief − harm − nociception`, unweighted; health counted once) | innate prior (owner decision G11) | learned only via the forward model |
| Urgency v1 = max pressure after | innate prior (owner decision G12) | slope term with GL3 timing |
| `apparatus` records excluded from training and credit | invariant (G16) | — |
| Nociceptor threshold and gain | innate prior | `adaptive_nociception.md` revive triggers |
| Heat need + affinity keywords | innate prior | follow-up issue in the GL2b(i) PR |
| Cause attribution (who did it; the producer's YAML noun) | engineering prior (oracle attribution; never cited as learned stimulus valence) | follow-up issue: RW-style updating on the cause namespace |
| A bystander never becomes a cause | invariant | — |
| Percept valence values, credit from satiation | learned | — |
| Narrated-record discount | innate prior (a declared constant) | owner decision at GL4 start; S5/GL5 report with and without narrated data |

## 5. Stages

Each stage is one issue + PR (GL2b is three), with a three-lens code review (Executor / Architecture /
Wire integrity). **This plan's four-lens design review runs in GL1**, before GL2a; GL2c gets its own
**joint** four-lens review with the relief store. Owner decisions for each stage are asked together at its
start (§8). No stage carries a timeline.

### 5.1 GL0 items owned here (truth, docs + red gates, no behaviour)

- Correct the `EntropicDriveSpec` docstring and field comment ("satiation Reaction: not built; see
  `autonomic_layer.md` GL2c"). Docstring-only, class shape unchanged; grows `## [Unreleased]`.
- Strict red gates (`xfail(strict=True)`), each failing on `main` for the stated reason:
  (a) a fire-pit `touch` on the nociceptor variant body produces a NOCICEPTIVE PainSignal ≥ 0.4 (today:
  DRIVE, 0.04 — the gate is written against the variant fixture that GL2b(ii) adds, so in GL0 it targets a
  test-local body declaration); (b) `_read_drive_states` on a hot infant emits a `heat` need (today: none);
  (c) an actor affordance that declares a `target_effect` harmful to the AUT writes a Wire-2 cause row
  keyed on the actor's noun (today: only the AUT-body sufferer row). No shipped affordance can serve: the
  dragon's `fire_breath` (`_data/components/creatures/dragon.yaml`) declares no `target_effect`, so it
  cannot hurt the AUT, and there is no `breathe_fire`. The gate is written against a **new fixture**
  authored in GL2b(iii) (a test-local actor in GL0);
  (d) a deprive → satisfy cycle produces exactly one satiation event on the record (today: none). GL2a's
  own scripted sequence already contains a **homeostatic** crossing on the tool path: `core_temperature`'s
  latch clears on the second `warm_self` (deviation 0.15 ≤ 0.25·0.8); the entropic case is the cradle
  hunger / `cradle_food.eat` cycle (hunger −0.4, satisfaction 0.3, so one feed clears only from
  hunger ≤ 0.7).
- File issues for the verified defects not yet filed (heat need, satiation producer, cause keying), linked
  from here.

### 5.2 GL2a — the record, record-only (outside the fence; owner decisions G1, G9, G14, G15, G17)

> **BUILT 2026-10-09** (branch `feat/gl2a-interoceptive-outcome`). Guards:
> `tests/unit/test_interoceptive_outcome.py`, `tests/unit/test_gl2a_trio_golden.py` (fixture
> `tests/fixtures/gl2a_trio_golden_v1.json`, generated at d9e89f5a) and
> `tests/unit/test_drift_step_byte_identical.py`; the cradle driver is `tests/unit/_cradle_loop_driver.py`.
> Two gates stay red by design: the #1161-scoped `arms.thermal` gate and gate (d)'s drift-only crossing
> (the out-of-band producer). The entropic half of gate (d) needed two feeds, not one: a few ticks of
> hunger drift between the latch-setting evaluation and the feed lift the post-feed value just past 0.3
> (`satisfaction_threshold`), so one feed from 0.7 never clears. #954 is still open; the PR states the
> water-trial margins it measured.

- **Build:** `CauseRef` (without `cause_pid`), `InteroceptiveOutcome` (without `pid`),
  `sem.interoceptive_outcome` and the pure drift helper; `ToolOutput.interoceptive_outcome`
  (`repr=False`); `Executor._stamp_invocation` sets it, and its short-circuit equality tuple (today
  `(rpe, drive_pressure_before, drive_relief, pain)`, which returns the input `ToolOutput` unchanged when
  every stamp already matches) gains `interoceptive_outcome`, or a record on an otherwise unchanged
  output is silently dropped; in `body.py`, the per-call record of the drift applied and the latch
  cleared at the two `elif cleared:` sites; the raw before-snapshot in `Executor._run_started`; the trace
  line and `EncodingSignals.extra["interoception"]`; and **the cradle scripted-sequence loop driver** the
  gate below needs (test-side, not `src/`): no harness today drives a fixed `feel` ×2 / `warm_self` ×2 /
  `touch` sequence through `run_agentic_loop` on a cradle body (`tests/unit/_loop_harness.py` drives the
  scripted Minecraft world), so GL2a builds one. Files: exactly the exempt set in the header
  (`embodiment/body.py`, `embodiment/sem.py`, `runtime/executor.py` `_run_started` +
  `_stamp_invocation` (G14), `tools/base.py::ToolOutput`, `runtime/bio_integration.py`). **Not in GL2a**
  (G9): the per-entity snapshot, the `RLock` over `evaluate_failures`, `Embodiment.drain_outcomes()` and
  its bound; they land with the out-of-band producer. **Not in GL2a** (G17): `embodiment/event_id.py`,
  `PhysicalEventId`, the per-agent sequencer, its session-id source, `pid` and `cause_pid`; they land at
  the post-fence resume stage. GL2a records carry no pid and are never forward-model training data.
- **Narrated records and the fence (decided 2026-10-09, owner decision G9: tool-path only).** Stamping
  narrator consequences `narrated` (§3.1.4) needs the three narrator tools' `execute` bodies in
  `simulation/tools.py` (four `evaluate_failures` call sites) to enter the narrated scope, and that file
  is not in the exempt set. Thread identity cannot stand in for the scope: the narrator tools run on the
  orchestrator thread, but the reflex dispatch calls separate instances of the same
  `DamageComponentTool` / `SetEntitySensorTool` classes from the AUT's enrichment pipeline, on the loop
  thread (`sim.aut`). So GL2a ships the tool-path record only, minted solely by the executor whose
  embodiment is the agent's primary `Embodiment` (a narrator tool run by the orchestrator's executor
  mints nothing), and the out-of-band producer lands after the fence together with the narrated and
  apparatus scopes, so no out-of-band record's provenance is ever guessed.
- **After the fence, before GL4 S1 (G15, G17): the resume stage** (§3.1.2). It builds the identity and
  its resume together: `embodiment/event_id.py` (`PhysicalEventId`, rejecting an empty `agent_id` or
  `session_id` and a negative `seq`), the per-agent `EventSequencer` with its own private lock, the
  session-id source (asked at the stage's start: deterministic, who mints it, its scope, `--resume-sim`
  behaviour, a new required keyword on `build_executor` and its callers), `InteroceptiveOutcome.pid`,
  `CauseRef.cause_pid`, the one-live-sequencer assertion with its release (bio session end, executor
  shutdown), the two same-process rebuild harnesses (`scripts/survival_world/r3_run.py`,
  `scripts/survival_world/exp61_run.py`) and a conftest autouse reset; then the sequencer's own
  high-water mark, both load seams, and its gate through the real load paths. GL4 S1 depends on it.
- **Depends on:** GL1's four-lens review of this plan (folded 2026-10-09); #1125 is merged (#1164);
  **#954** for the survival check below, or a stated margin.
- **Falsifiable gate**, driven **through the real loop and capture** (`run_agentic_loop` →
  `tool_dispatch.execute_and_learn` → `capture_loop_action`, as `tests/unit/test_water_trial_smoke.py`
  does), with the record read back from the Hippocampus, never by calling `executor.execute` directly
  (wiring S4):
  - on the scripted cradle sequence (`cool_air`'s `draft.feel` ×2, `fire_pit.warm_self` ×2,
    `fire_pit.touch`, on `infant_humanoid`) the records match a committed table computed **by hand from
    the YAML deltas and specs**, never by calling the `sem.py` helpers it checks (confounding NIT-1):
    `arms.thermal` 0 → −0.15 → −0.30 → −0.10 → +0.10 → +0.70, `core_temperature` −0.15 → −0.35 →
    −0.55 → −0.35 → −0.15 → 0.00, with the `core_temperature` satiation crossing on the second
    `warm_self` asserted (environment NIT-2);
  - a `_StepClock` case (`tests/unit/_loop_harness.py::_StepClock` patches the global `time`, so
    `body.py` sees it) advances 20 s between the loop tick and the tool call on
    `infant_humanoid_chilled` (entropic `cold`, 0.08/s), and its hand-computed table shows a warmth item's
    `warm_self` `cold` change (−0.3) net of drift (environment DNB-1). The order is fixed: the tool
    applies its delta first, then `evaluate_failures` drifts the body (`tick_vital_drift`), and the drift
    is **clamped** to the sensor's range. The hand table: `cold` 0.6 → 0.3 after the delta → 0.3 + 20 ×
    0.08 = 1.9, clamped to 1.0; the applied drift is 0.7 (not the declared 1.6), the observed change is
    +0.4, and the net is +0.4 − 0.7 = −0.3. So the helper nets the **applied** (clamped) drift computed
    on the post-delta value, never the declared `drift_rate × dt`;
  - red gate (d) flips for the record half on the tool path (the homeostatic crossing above; the entropic
    cradle feed), and through drift only once the out-of-band producer ships;
  - the deletion probe (remove the producer) changes only the trace and the persisted `extra` key: the
    trio / `encoding_tag` / `storage_strength` golden, the selection golden and the encoder golden stay
    byte-identical, and `str(ToolOutput)` is byte-identical with and without the record (#1189).
- **Guards:**
  - the relief pin as restated in §3.1 (positive part of `drive_delta` == `drive_relief` on every drive
    present in `drive_relief`), its deletion probe (remove the drift netting → the `_StepClock` case
    re-reds), and the #1161-scoped `xfail(strict=True)` gate for `arms.thermal`;
  - the pre-change golden of the trio, `encoding_tag` and `storage_strength` over the scripted cradle
    sequence and the Minecraft `fear_water` arm, byte-identical, plus its reverse probe (revert the
    record, the golden still passes);
  - record present **iff an agent-bound body is attached and the tool ran** (no zero records; wiring S3);
  - `str(ToolOutput)` byte-identical with and without the record;
  - satiation detected only at the `elif cleared:` sites (a test that the unreadable-sensor pop records
    nothing);
  - the required-keyword factory and the sentinel rejection (`provenance=""`); the record has no `pid`
    field at GL2a (G17). The identity guards (`PhysicalEventId` sequence deterministic under
    `_StepClock`, unique within a session, at most one live sequencer per `agent_id` per process with its
    release, ephemeral wrappers mint nothing, the empty `agent_id` / `session_id` and negative `seq`
    rejections) belong to the post-fence resume stage;
  - behaviour preservation byte-identical: `test_agent_loop_selection_golden.py`,
    `test_decision_provenance.py`, `test_encoder_golden_v1.py`.

  The latch-semantics, two-thread and queue-bound guards belong to the out-of-band stage (G9): in GL2a
  narrator calls mint nothing and the queue has no producer, so those tests would be vacuous.
  **Survival harnesses.** The executing check is
  `tests/unit/test_water_trial_smoke.py::test_water_trial_ticks_acts_and_the_staging_close_persists_fear`,
  the one test there that executes `escape_water` through the executor, and it asserts that the
  `escape_water` `ToolOutput` carries a record (no `pid` at GL2a, G17; the deletion probe re-reds it) and **no**
  relief (the refill is out-of-band). It is still on wall time (`train_cap_s=8.0`, `t_surface < 2.5`),
  named in open [#954](https://github.com/dennys246/Maxim/issues/954), and GL2a adds per-invocation work
  that narrows those margins; so GL2a lands after #954 moves it onto `StepClock`, or the PR states its
  margins (environment SF-1). `tests/unit/test_exp61_run.py` runs no loop (verdict, donor-sanity and
  statistics over hand-built files), so it is **not** an executing check and is not cited as one. (The
  committed Exp 60/61/R3 verdict reproductions are not a gate here either: `compute_verdict` re-reads
  committed JSONL and never executes the code under test.)
- **Rows fired by wording: T1-16, not none.** GL2a adds `extra["interoception"]` to every loop capture,
  and `memory/encoding.py::EncodingSignals.to_dict` flattens `extra` into the persisted trace, so the
  saved memory record's shape changes and T1-16's "the memory record shape" trigger fires by its letter
  (T1-1, Exp 10, carries the older "hippocampus persistence schema change" wording but is SUPERSEDED by
  T1-16). The PR records the walk: either a structural discharge or the Exp 63 re-run. The structural
  discharge has three legs: `_rank_by_relevance` and `_query_hippocampus` read no `extra` key (grep);
  a byte-identical ranking test over a store with and without the key; and the same test on the
  **substring path** (`Hippocampus.search_by_content`, Path 3 of `_query_hippocampus`) with queries that
  would match a record token (`"relief"`, `"oxygen"`, `"thermal"`), which holds only because the field
  is `repr=False` (wiring D3, #1189). `runtime/executor.py` is touched: run the CLAUDE.md mypy invocation.

### 5.3 GL2b — the regulatory fixes (one issue + PR each)

**(i) Heat corrective need** (§3.3). Fenced: `substrate_proposal.py` and `nac.py`'s affinity table are on the
body selection path, so it waits for the remaining agent_loop slices.
- Gate: red gate (b) flips; `corrective_need_intensity` byte-identical over a grid on every shipped body;
  Minecraft `_read_drive_states` dicts identical. The walk includes the `simulation/arcs.py` cradle scene,
  where `cradle_cool_air` is a shipped cooling act: the affinity keywords decide whether the need is latent
  or live there (§1.3 R-2). The keyword list is frozen before GL5's fixtures are named, because tool names
  carry the entity name and the word prior reaches selection through these keywords (§GL5 in
  `grounding.md`; confounding SF-6).
- Rows fired: T1-13 ("`recommend_action` drive-activation floor" by its wording); T1-6 (the derived need is
  encoded into the interoception channel whenever a cradle body goes above set point — Exp 42's harm item
  does exactly that); T3-9 (PARTIAL; "Cradle / drive / SEM body change"). A re-run or a written discharge
  per row. (T3-6, T3-10 and T3-15 are DROPPED and carry no `Re-run on:` clause, so they cannot fire.)

**(ii) Burn = pain via `NociceptorSpec` on a variant body** (§3.2).
- Depends on: **#1161 resolved first** (owner decision at stage start; the recommendation is #1161
  option A before this): burn-as-pain cannot be measured honestly while the affordance-credit read is
  blind, and Exp 42 must be re-run either way. Nociception_layer step 2 (store `kind`, put it on
  `Reaction`, fix F3) lands **with or before** it: without it the nociceptor's Reaction shares the
  ReactionBus refractory key `pain:pain_detector:external_signal` with the drive Reaction from the same
  call (§3.2) and the burn can be delivered as 0.04.
- Gate: red gate (a) flips on the variant, with the learning threshold fixed by the stage-start
  calibration decision over the measured starting states (§3.2); single `warm_self` / `*_safe`
  applications produce zero nociception, from rest and from the post-`touch` state; the B8
  causer-vs-bystander test on the chilled body extended to `noxious`; both the drive Reaction and the
  nociceptor Reaction of one contact reach `_distribute_reward_from_reaction`; deletion probe (remove
  the `noxious` rule in `failure_pain_kind` → the gate re-reds).
- It bumps the record's schema id to `ans-v2` (§3.1.1): the same contact's `nociception` changes meaning.
- Guard: nociception_layer step 4's **golden table**, generated from pre-change code over every `PainType`
  × source (including `nociceptor:*`) × origin × `agent_id`, recording each consumer's accept flag and
  delivered value **through both refractories** (PainBus per `(entity, failure_mode)`, ReactionBus per
  `kind:source`); only the nociceptor rows may differ. The `noxious` band is appended to
  `docs/user/tool_side_effects.md`'s drive-band grammar in the same PR.
- Rows fired: T1-6 (Exp 42 re-run is owed by #1161 anyway; [sim], not rig); T3-9 (Cradle / SEM body
  change); T1-4 (SEM cascade: ToolPainBridge sees a new failure). T1-9 / T1-10 only if a shared body
  changes, which the variant avoids. Not 56 / 60 / 61 / 62 (no Minecraft body declares a nociceptor).
  Exp 37 / other cradle row ids: UNVERIFIED.

**(iii) Cause-keyed valence** (§3.4).
- Builds the red-gate (c) fixture: an actor affordance with a `target_effect` harmful to the AUT.
- Gate: red gate (c) flips; sufferer rows byte-identical; a bystander affordance never stamps a cause; a
  breach lingering on another sensor never inherits the cause (B8's set is computed before
  `evaluate_failures` and passed as `cause_sensors`); cause rows are cradle / `--sim` only (no Minecraft
  source exists); the actor path's B8 gating; the aversion
  map byte-identical under the strict default (§3.4); cause rows survive or are counted out of the
  bundle scrub exactly as decided, never silently dropped.
- Rows fired — **not none, and not additive at the reader** (§3.4): no ledger row's trigger cites
  `percept_valence`, Wire 2 or the aversion reader (grep of `behavioral_graduation_candidates.md`), but
  negative cause rows reach the live LLM path through `get_percept_aversions` unless the reader excludes
  them; T1-14 fires by its wording ("`hivemind/bundle.py` scrub"), because the strict default's
  exclusion of cause rows from bundle export is a `bundle.py` scrub edit, and it is also walked because
  `donor_sanity_staged`'s `FEAR_MODE in str(k)` check now also matches cause rows and `hivemind/`
  merge/ingest handle the new keys (Exp 56/61 donors are fresh, and ingest keeps receiver rows only via
  `entry_index.keep_agent_rows`, so the walk is expected to discharge structurally); T1-2 (already STALE) and Exp 37's prompt text per the GL0 trigger walk. The
  pain-memory capture copies `**signal.context` into observations and the pain→NAc subscriber passes it
  to `record_outcome_full` (link `context_factors`), so the two new keys enter pain episodes and links:
  T1-16's "the memory record shape" fires by its letter and is walked like GL2a's.

**Divergence rule.** If GL2b's fixes each surface a new failure mode for two iterations running, stop and
audit the body layer beneath; the sensor-resolver disagreements (#1124, #1156, #1159) are the likely layer.

### 5.4 GL2c — the positive producer (last, deliberately)

> **Also unblocks [#1180](https://github.com/dennys246/Maxim/issues/1180)** (PARKED 2026-10-10): the temporal-credit
> fallback's behavioural test needs a live positive reward to reach `TemporalCreditDistributor.distribute`,
> and none exists until this stage's producer (`docs/experiments/temporal_credit_validation.md`, PARKED block).

- **Depends on:** GL2a; **the out-of-band producer with the narrated and apparatus scopes** (after the
  fence; G9, G16), because on Minecraft every oxygen crossing is out-of-band and a narrator-caused cradle
  satiation would otherwise be delivered at full weight (environment SF-2); R4's routing audit has
  decided the selection surface (Phase 5 "first"); **one joint
  four-lens design review with Phase 5's relief store** (plus `fear_learning.md` Exp A and `coding_world.md`
  C3's reserved opposite sign), owner decision at stage start, recommended yes; **nociception_layer step 3
  (the F1 / F1b fix) landed before or with it** (§7); `engram_formation.md` E4's overreach set read
  (mandatory under A or C, §3.5).
- **Build:** the satiation emission at the latch-clear site, routed per the owner's option (§3.5), behind a
  declared `maxim config` switch, default OFF, in M10's harness fingerprint, enabled only by a
  pre-registered experiment. (If an env var is unavoidable it needs an autouse conftest scrub in the same
  commit.) The restated refusals (§3.5) are part of the build.
- **Gate (flag off):** every golden byte-identical, and the executing survival check (the
  `test_water_trial_smoke.py` test named in §5.2, on `StepClock` after #954, or with stated margins)
  unchanged; `test_exp61_run.py` and `test_r3_run.py` are run but not cited as executing the producer
  (environment SF-1). **Gate (flag on):** red gate (d) flips for delivery; exactly one positive event per
  breach episode on a scripted **cradle** deprive → satisfy sequence (`infant_humanoid` hunger,
  `cradle_food.eat`; Minecraft `food` deprivation is not reachable natively, and `ScriptedWaterBridge`
  serves `food` 20 and a no-op `eat`, so a Minecraft eat case needs a **new** scripted fixture class,
  never an edit to that cited guard; environment SF-5) and on a scripted Minecraft
  surfacing-after-submersion (relief at offset, §3.5); a rescue-caused crossing delivers nothing (G16);
  passive recovery delivers nothing; under option B this producer never writes `_reward_bias`; under
  option A `_reward_bias` becomes non-empty **only** from satiation; the zero-bias guards in
  `test_nac.py` and the refusals in `donor_sanity_staged` and `_R3._boundary` are restated, not deleted;
  narrated satiations carry the declared discount.
- **Lands only with the batched live re-run** of Exp 60, 61 and 62 plus the T1-11 and T1-12 arguments
  or the Exp 56 / Exp 57 re-runs (G7): Exp 60's frozen gates still PASS on the rig, or the row is recorded BROKEN and blocks the
  release. The re-run reports satiation counts **per arm and per cause** (agent action / apparatus /
  respawn; G16). **MAINTAINED needs more than a PASS** (confounding SF-4): with a positive value on the
  surfacing act, the agent could surface on time because surfacing is rewarded rather than because of the
  situation fear T1-13 claims. So the re-run records the per-component score at every scored decision
  (`NAc.recommend_action` already emits `components`: `causal`, `reward_bias`, `learned_bias`, `drive`,
  `explore`), and MAINTAINED requires the frozen gate to PASS **and** the decision to be unchanged with
  the satiation term zeroed (a counterfactual recomputed offline from the recorded components). A PASS
  that holds only with the term is recorded as the new cause, not as MAINTAINED. The same requirement
  goes into the joint four-lens review's brief.
- **Rows fired, every routing (G7):** T1-11, T1-12, T1-13, T1-14 (also functionally: the donor sanity)
  and T1-15;
  plus, by the "PainBus / ReactionBus / NAc reward pipeline change" wording, T1-4 and T3-9 under options
  A and C; **T1-16** under A and C, because `hippocampus.capture_reaction` (subscribed to every Reaction,
  `runtime/bio_stack.py`) appends satiation Reactions to the pending episode and so changes the persisted
  episode's valence and content, "the memory record shape" (wiring S8); T1-6 / T1-9 for cradle hunger/thirst satiation, per wording; and the relief store's own walk
  (rung E2's) under options B and C.

### 5.5 Hand-off to the latent forward model

`latent_forward_model.md` consumes the tool-path record (and, after the fence, `drain_outcomes()`) as its
target, the change-only subset first (§3.1.1), on experienced records only.
Its S1 changes the Cerebellum's observe call to these consequence dims (target entity keyed); this plan
only guarantees the record exists and is honest. Affordances with no declared `self_effect` (363 of the
405 shipped, per the #1120 audit) have no predicted consequence: stated, not imputed.

## 6. Ledger blast radius

| Stage | Symbols touched | Rows whose trigger fires (by wording unless noted) |
|---|---|---|
| GL0 | docstrings, red gates | none |
| GL2a | new symbols (no event id, G17); `ToolOutput` additive field (`repr=False`); the before-snapshot in `_run_started`; the per-call drift and cleared record in `evaluate_failures`; `extra["interoception"]` on every loop capture | **T1-16** ("the memory record shape": `EncodingSignals.to_dict` flattens `extra` into the persisted trace) — walk or structural discharge, including the substring path (#1189) |
| Out-of-band producer + scopes (after the fence, G9) | snapshot, lock, `drain_outcomes()`, narrated and apparatus scopes in `simulation/tools.py` and the harnesses | T1-16 (more captures carry the key); walked in its own PR |
| Resume stage (after the fence, before GL4 S1, G15, G17) | `embodiment/event_id.py`, the sequencer and its session-id source (a new required `build_executor` keyword and its callers), `pid` / `cause_pid`; the sequencer's high-water mark; `build_bio_stack`, `_restore_aut_from_session` | **T1-16** by its "Hippocampus save/restore or the resume path (`RESUME_STORES`)" wording if it touches them |
| GL2b(i) | `substrate_proposal` need map, `_DRIVE_TOOL_AFFINITIES` | T1-13 (floor), T1-6, T3-9 |
| GL2b(ii) | `failure_pain_kind`, `evaluate_failures`, a variant body YAML, the side-effects registry grammar | T1-6 (with #1161), T1-4, T3-9; T1-9/T1-10 only if a shared body changes |
| GL2b(iii) | percept-valence subscriber (cause namespace), PainSignal context, the aversion reader's exclusion, bundle scrub | T1-16 (pain episodes' shape); T1-14 by wording (the strict-default exclusion is a `bundle.py` scrub edit) and walked (donor check, hivemind keys); T1-2 (STALE); a live LLM-path change via `learned_aversions` unless excluded |
| GL2c (any routing) | the latch-clear emission and its route; the restated refusals | **T1-11 / T1-12 / T1-13 / T1-14 / T1-15 (G7: a positive producer on the earned bodies, whose crossings the apparatus mostly makes, G16; T1-14 and R3 refusals functionally)**; T1-6 / T1-9 per wording |
| GL2c (A or C) | `_distribute_reward_from_reaction` input, `credit_node` writes, `capture_reaction` | adds T1-4, T3-9, T1-16 (episode valence) |
| GL2c (B or C) | relief store input | adds the relief store's own walk |
| Wire 4 | — | must not change in any stage |

DROPPED rows (T3-6, T3-10, T3-15) have no `Re-run on:` clause and fire on no stage.

Ledger edit owed with GL1 (a ledger PR, not this plan's): a trigger-table category "**Consequence-code /
autonomic producer change**" naming T1-11, T1-12, T1-13/14/15, T1-6, T1-9, T1-16 and T3-9, so an autonomic
change matches by category and not only by wording.

## 7. What this plan absorbs from `nociception_layer.md`

`nociception_layer.md` is REVIVED into GL2 (owner decision G5): its trigger (d) ("any R4 credit-routing PR opens") fires with
GL2c's joint review, and (c) fires when GL2b(ii) adds a producer consumers must classify. Its still-valid
content is carried here as stage prerequisites, not a parallel plan:

| Its item | Status | Where it lands here |
|---|---|---|
| Step 1 — `PainKind`, `classify_pain`, `failure_pain_kind` | shipped (2S-c) | GL2b(ii) adds one `failure_pain_kind` rule |
| Step 2 — store `kind`, put it on `Reaction` (additive field + isolation review), fix F3 | open | with or before GL2b(ii) |
| **Step 3 — F1 / F1b: anticipated pain is paid as real negative reward** (`perceived_pain.py` publishes `Reaction(kind="pain", NEGATIVE)` with a real `agent_id`; `_distribute_reward_from_reaction` pays it; live on the `--sim` orchestrator AUT path; [#880](https://github.com/dennys246/Maxim/issues/880)) | open, near-inert **only because** the clamped surface holds no positive bias | **must land before or with GL2c.** Once GL2c can make a bias grow (option A), a self-confirming prediction erodes real reward; under B its trigger (d) still fires with the joint routing review. Replacement signal: the signed change in anticipated pain across an action (fear reduction), or a recorded known gap. Keep DRIVE pain paying reward unless the owner decides otherwise (excluding it re-runs the survival rows). |
| Step 4 — consumers declare their rule; golden-table guard | open | the golden table is GL2b(ii)'s guard; the full migration stays its own step |
| Step 5 — producer-side adaptation | deferred | `adaptive_nociception.md`, at the `NociceptorSpec` producer |
| Step 6 — bridge boundaries in docstrings; F2 overlap | open | named in GL2c's double-credit review |
| Principle 4 — anticipation is not an outcome | adopted | ANTICIPATORY excluded from every record field |
| Taxonomy — `drive:health` is an injury proxy | adopted | carried as nociception and labelled a proxy (§3.1) |

`deferred/nociception_layer.md` carries the banner "REVIVED 2026-10-07 into `autonomic_layer.md`" (owner
decision G5), added in this plan's PR.

## 8. Owner decisions, by stage (asked together at each stage's start)

| Stage | Decision | Recommendation (strict option first where offered) |
|---|---|---|
| GL2a | #1125 | **Closed:** merged 2026-10-08 (#1164); GL2a reads through its resolver. |
| GL2a | Narrated records vs the fence | **Decided 2026-10-09 (G9):** tool-path only; the out-of-band producer lands after the fence with the narrated scope. |
| GL2a | `drain_outcomes()` queue bound | **Decided 2026-10-09 (G10):** measured from a scripted session's peak, drop-oldest, counted and logged; applies once the out-of-band producer exists. |
| GL2a | The load seam for the restored `seq` | **Decided 2026-10-09 (G15, superseded in part by G17):** no pid at GL2a; the type, the sequencer, its session-id source and the resume (own high-water mark, both seams, real load paths) land together after the fence, before GL4 S1. |
| GL2a | Executor scope and the record's window | **Decided 2026-10-09 (G14):** `_run_started` + `_stamp_invocation`; action-scoped, resolver-read, net of drift; trio byte-identical. |
| GL2a | Core valence formula | **Decided 2026-10-09 (G11):** unweighted `relief − harm − nociception`, an innate prior; health counted once. |
| GL2a | Urgency v1 | **Decided 2026-10-09 (G12):** pressure only; slope with GL3 timing. |
| GL2a | Session id source (G15) | **Decided 2026-10-09 (G17):** none exists (every session id is wall-clock `time.strftime`, and `build_executor` has no session parameter), so GL2a mints no id. Moved to the resume stage (next row). |
| Resume stage (after the fence, before GL4 S1) | Session id source; sequencer release | Asked at the stage's start (G17): deterministic (no uuid, no wall time); who mints it; its scope; `--resume-sim` behaviour; a new required keyword on `build_executor` and its callers; release at bio session end and executor shutdown; the `r3_run.py` / `exp61_run.py` same-process rebuilds; a conftest autouse reset. |
| GL2b(i) | Heat-need name (`heat` vs `overheat`) and affinity keywords | `heat`; keywords chosen to avoid `withdraw`. The keywords are the switch between latent and live in the cradle arc scene, which has a cooling act (`cradle_cool_air`, §1.3 R-2); say which in the PR, and freeze the list before GL5's fixtures are named. |
| GL2b(ii) | **#1161 order:** option A (resolve the modulator read + narrow the collateral gate) before burn-as-pain? | Yes, #1161 first, with its Exp 42 re-run plan and the T1-9 call and safe-warm-when-satiated flip it raises. |
| GL2b(ii) | Burn calibration: the learning threshold this stage must clear, the nociceptor threshold and gain, and body placement | Threshold ≥ 0.4 (every PainBus learner); values satisfying §3.2's constraint; variant body only, the shipped base body waits for a post-#1161 Exp 42 re-run. |
| GL2b(ii) | Nociception step 2 before or with this stage? | With or before (the ReactionBus refractory collision, §3.2, makes it a prerequisite in practice). |
| GL2b(iii) | Cause-row namespace; does the aversion reader read cause rows; do cause rows ship in bundles? | A `_IDENTIFIER_TOKEN`-safe namespace (e.g. `cause.<noun>`); **strict:** the reader excludes cause rows (aversion map byte-identical) and the bundle export excludes them with a count, until a declared arm or transfer experiment needs them. |
| GL2c | **Satiation routing:** B (relief store only), A (Reaction → distributor → `_reward_bias`), or C (both)? Open (G7). | No routing avoids the T1-13 lapse or the T1-11/12/14/15 walks (§3.5). B gives the positive write one home (the relief store's "no second store" rule); whichever is chosen ships default OFF and lands only with the batched live re-run of Exp 60/61/62 (+ the T1-11 and T1-12 arguments or re-runs) and the restated refusals. |
| GL2c | One joint four-lens review with the relief store, `fear_learning` Exp A and `coding_world` C3? | Yes — the store's "no second store is ever created" rule requires it. |
| GL2c | Double credit with channel 3 (options A/C) | One credit per physical event, keyed by `pid`. |
| GL2c | F1 fix: replacement signal (fear reduction) or a recorded gap; does DRIVE pain keep paying reward? | F1 fixed before or with GL2c; DRIVE keeps paying. |
| GL2c | Rig slot | Its own Track C item after the 1.3.2 live Exp 60 re-run; batchable with an Exp 60 re-run owed by GL3.B4 (nociceptive-fast preemption), which is owed only if GL3.B4's flag ever defaults on; never during an E-rung campaign. |
| GL4 (recorded here, decided there) | The narrated discount on records this plan produces | A small discount (strict default); S5/GL5 report with and without narrated data. |

## 9. Risks

- **Silent double attribution** (GL2c option A vs channel 3) — named in the Wire-integrity review.
- **Flooding:** any out-of-band emission without the latch re-creates the per-tick pain flood the channel
  split fixed. The latch reuse is the guard.
- **Bystander causes:** a cause not gated by B8's harmful set reintroduces the Exp 42 pollution on the
  Wire-2 map.
- **Units:** `deviation_after` is normalised by `drive_span`, which needs a declared range; a drive without
  one yields no entry. The units invariant in `_read_drive_ranges` applies.
- **One-sided signs:** Minecraft health/oxygen cannot exceed the set point (max 20 = set point), so the
  signed deviation is one-sided there; the forward model must not read "never positive" as learned.
- **The record inherits upstream blindness** (#1161) unless it resolves sensors itself (§3.1.4); it reads
  through the #1125 resolver and never refills the blind trio.
- **Two writers on one body:** the narrator's orchestrator thread and the AUT loop both run
  `evaluate_failures` on the AUT body; once the out-of-band producer mints there, a seq could be minted
  twice or a latch read half-updated without its lock. That stage's two-thread test is the guard; GL2a
  mints nothing (G17), and from the resume stage tool-path minting runs on the executor's thread, under
  the sequencer's own lock.
- **Mislabelled provenance:** a narrator consequence recorded as `experienced` would hand language-prior
  physics to the forward model at full weight. The G9 default (tool-path only until the narrated scope
  lands), the relief pin (it catches a foreign write on a drive the trio sees), the `--sim` tool-path
  records counted separately, and the contamination guard's no-relabel check are the guards. The
  remaining hole, a narrator write to a declared drive inside the window, closes with the write epoch.
- **Apparatus counted as experience:** a rescue, heal, respawn or teacher feed stamped `experienced`
  would be a synthetic reward delivered as world-native (D1), and in Exp 60 it would fall unevenly
  across arms. `apparatus` (G16) and the per-arm, per-cause counts are the guards.
- **Satiation lands near the measured contingency:** a positive producer can reinforce the act the
  survival claims measured (§3.5). Off by default, `apparatus` excluded, the batched re-run, and
  MAINTAINED only with the satiation term zeroed are the guards.
- **Claim drift:** keep autonomic wording out of release-claim prose until a GL5 record exists (M36 does
  not lint CHANGELOG claim lines).

## 10. New `[engineering]` invariants this plan would add (with their guards)

Each enters `docs/agents/embodiment.md` in the stage that builds it, with its `Regression guard:` line.

- The record is built only through the factory with required keyword-only `cause=`, `provenance=`
  (and `pid=` from the post-fence resume stage, G17). Regression guard: `embodiment/sem.py::interoceptive_outcome` signature (structural) + the
  stage's AST test that `InteroceptiveOutcome(` is not constructed elsewhere in `src/`.
- The positive part of the record's `drive_delta` equals `ToolOutput.drive_relief` on every drive present
  in `drive_relief`, and the trio, `encoding_tag` and `storage_strength` are byte-identical to the
  pre-change golden. Regression guard: the GL2a unit test (proposed
  `tests/unit/test_interoceptive_outcome.py`) with its drift-netting deletion probe.
- `str(ToolOutput)` is byte-identical with and without the record (`repr=False`, #1189). Regression
  guard: the same test file.
- `corrective_need_intensity` is the `"below"` projection of `corrective_need`, byte-identical on every
  shipped body. Regression guard: the GL2b(i) grid test (proposed `tests/unit/test_corrective_need_two_sided.py`).
- A cause is stamped only for sensors in the affordance's B8 harmful set. Regression guard: the GL2b(iii)
  bystander test.
- (Enters at the post-fence resume stage, G17.) `PhysicalEventId` carries no uuid and no wall time,
  rejects an empty `agent_id`, an empty `session_id` and a negative `seq`, and is minted only by the
  agent's `EventSequencer` (ephemeral wrappers mint none; one live sequencer per `agent_id`, released at
  bio session end and executor shutdown). Regression guard: the frozen dataclass in
  `embodiment/event_id.py` (structural, SHAPE-FROZEN) + the lockstep determinism test + its test through
  both real load paths.
- No record carries a defaulted provenance; an out-of-band narrator consequence is never `experienced`,
  and an apparatus write is never trained on or credited. Regression guard:
  `InteroceptiveOutcome.__post_init__` sentinel rejection (structural) + the post-fence narrated- and
  apparatus-scope tests (every narrator tool's out-of-band record is `narrated`; a rescue-caused crossing
  delivers nothing). Until the write epoch lands, the invariant is scoped to out-of-band records (§3.1.4).
- No stage edits `DEFAULT_CLUSTER_FEAR_FAILURE_MODES`. Regression guard: the existing Wire-4 guards
  (`tests/unit/test_cluster_fear.py`).

No new mechanization row is needed: each rule above is a structural guard (a typed constructor, a
`__post_init__` rejection, or a named test), cited as such and not by a backlog number. If a stage drops
one of those tests, the rule gets a row in `outstanding.md` §Mechanization backlog in the same commit, at
the next free number. The GL2c switch rides the existing M10 row (amended 2026-10-07).

## Links

- [grounding.md](grounding.md) (umbrella, stage map GL0–GL6, owner decisions G1–G8)
- [thalamic_relay.md](thalamic_relay.md) (`Receptor`, `AfferentTrack`, `AfferentEvent`)
- [latent_forward_model.md](latent_forward_model.md) (the record's consumer)
- [roadmap_1_4.md](roadmap_1_4.md) §Phase 5 (R4, the relief store), T8 / T9
- [deferred/nociception_layer.md](deferred/nociception_layer.md) and its reviews in
  [reviews/nociception_layer/](reviews/nociception_layer/)
- `docs/agents/embodiment.md` §2 (the three pain/credit channels, B8, the drive protocol)
- `docs/agents/bio-memory.md` (reward-bias clamp, Wire 4)
- Issues: #1120 (audit), #1125 (closed, #1164), #1161, #880 (F1), #888 / #889 (R4 defects), #908 / #909
  (Cerebellum), #954 (wall-time water-trial tests), #1189 (`ToolOutput` repr in recall)
- GL1 four-lens review (2026-10-09): [reviews/grounding_gl1/](reviews/grounding_gl1/)
