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
> `embodiment/body.py`, `embodiment/sem.py`, the new leaf `embodiment/event_id.py`,
> `runtime/executor.py` (`_stamp_invocation`), `tools/base.py::ToolOutput` and
> `runtime/bio_integration.py`. Anything outside that set waits for the fence (§5.2 says what that
> means for narrated records).

**Owns (proposed):** the consequence record and its pure computation (`embodiment/sem.py`, beside the
existing drive helpers); the identity type `PhysicalEventId` in the leaf module `embodiment/event_id.py`
and the per-agent `EventSequencer` (GL2a; GL3 imports both); the out-of-band record at
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

**Two open defects already sit under this record.** #1125 (OPEN): `Executor._drive_pressure_snapshot`
misses modulator-qualified drives (`arms.thermal`, `head.thermal`, `arms.pressure`). #1161 (OPEN): the
affordance-credit reads in `tool_bridge.py` (`pre_values`, `_drive_potential_diff`,
`_drive_progress_by_drive`) read drives by bare name, so a qualified drive's progress is never measured —
and Exp 42's safe-vs-harm discrimination (T1-6) depends on that blindness (#1161's own measurement: with
the read resolved and no gate change, `warmth_alpha_harm` `warm_self` flips from −1 to +1 on warms 3–5).
Consequence for this plan: **the tool-path record today omits exactly the drive a burn moves.**

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
   happens on the earned bodies:** on `minecraft_player` the homeostatic `oxygen` and `health` latches
   clear on every surfacing and every regeneration **after a latched breach** (the homeostatic `elif
   cleared:` branch; `oxygen` breaches below 14 bubbles), i.e. at Exp 60's own `escape_water`
   contingency, since its training trials breach; `minecraft_player` `food` is entropic with
   `satisfaction_threshold: 16.0`; `minecraft_bench` `d1` (Exp 56, T1-11) and `minecraft_bench57` `d1`
   (Exp 57, T1-12; the same spec, copied verbatim) are entropic, drifting up, with
   `satisfaction_threshold: 0.3`. Any producer at this site therefore lands on the earned survival and
   transfer rows' own measured events (§3.5).
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

Two pieces of mutable state are touched from both threads today, with no lock in `embodiment/body.py`:
the entity-owned breach latch `Entity.drive_breach_severity` and (once GL2a adds it) the per-entity
previous-snapshot slot. §3.1.2 adds the lock.

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
`sim_adapter.next_observation`) never become a `PainSignal`, so those subscribers never see them. The record built here sees only what reaches `evaluate_failures` or
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
`"cold"` only. Hypothalamic thermoregulation is two-sided. Impact today is **latent**: I found no prereg
naming an overheated agent (UNVERIFIED that none measures one).

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
| Out-of-band record | `evaluate_failures`; the entity-owned latch pattern | A per-entity previous-snapshot slot (`__slots__`, never serialized, like `drive_breach_severity`), a bounded queue and a pull, `Embodiment.drain_outcomes()`. No bus: the loop pulls. |
| Event identity | `memory/hippocampus.py::Hippocampus._resume_capture_seq` (the resume-past-the-saved-maximum rule) | A per-agent `EventSequencer` and `PhysicalEventId`. Hippocampus `capture_seq` numbers memories, not physical events (one event can yield no capture or several), and the executor's `uuid4` invocation id is neither deterministic nor persisted-stable. |
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
    The defaults exist only to satisfy path (a); __post_init__ REJECTS the sentinels (pid None,
    provenance ""), so no record can exist without identity or provenance. Built ONLY through
    sem.interoceptive_outcome(...), whose identity keywords are required (§3.1.4)."""
    pid: PhysicalEventId | None = None      # REQUIRED: the join key (agent_id, seq); tracks fan out on it
    invocation_id: str = ""                 # executor uuid4, in-process diagnostic only; NEVER a join key
    agent_id: str = ""
    body_path: str = ""                     # whose body (drive names collide across bodies)
    provenance: str = ""                    # REQUIRED: experienced | narrated | imagined; no default
    sufferer: str = ""                      # entity path whose body changed (today's Wire-2 key)
    cause: CauseRef | None = None           # who/what did it; None = unknown / world
    # per drive, sorted by name, only drives with a declared range (never imputed):
    deviation_after: tuple[tuple[str, float], ...] = ()  # SIGNED [-1,1]; homeostatic (v - set_point)/span
                                                         # (+ above, - below); entropic: drive_pressure (>=0)
    drive_delta: tuple[tuple[str, float], ...] = ()      # SIGNED [-1,1]: drive_comfort_progress / drive_span
                                                         # (+ toward comfort = relief, - away = harm)
    pressure_before: tuple[tuple[str, float], ...] = ()  # == existing drive_pressure (unsigned)
    satiated: tuple[str, ...] = ()          # drives whose breach latch CLEARED in this event
    # body-agnostic core, each [0,1]:
    nociception: float = 0.0                # max PainKind.NOCICEPTIVE intensity (ANTICIPATORY excluded)
    drive_pain: float = 0.0                 # max PainKind.DRIVE intensity
    relief: float = 0.0                     # max positive drive_delta
    harm: float = 0.0                       # max |negative drive_delta|
    urgency: float = 0.0                    # v1: max pressure AFTER (§3.1.3)
    extra: dict = field(default_factory=dict, hash=False, compare=False)

@dataclass(frozen=True, slots=True)
class CauseRef:
    """CC3 path (a)."""
    entity: str = ""        # YAML noun of the causing entity ("fire_pit", "zombie"); never the sufferer
    affordance: str = ""    # "touch", "fire_breath"
    tool: str = ""          # tool signature when a tool call caused it
    cause_pid: PhysicalEventId | None = None   # the physical event that caused it, when one did
    extra: dict = field(default_factory=dict, hash=False, compare=False)

# embodiment/event_id.py — a leaf module with no maxim imports, built at GL2a; GL3 imports it
@dataclass(frozen=True, slots=True)
class PhysicalEventId:
    """CC3 path (b): SHAPE-FROZEN at 1.0 (CC3) — an identity; a new field would change equality and
    break every persisted join. Deterministic: no uuid, no wall time (G3). __post_init__ rejects an
    empty agent_id and a negative seq. str() is the join form "{agent_id}:{seq}"."""
    agent_id: str
    seq: int                # per-agent monotonic, minted only by that agent's EventSequencer
```

**Computation is pure and lives beside the existing helpers** in `embodiment/sem.py`
(`interoceptive_outcome(specs, ranges, before, after, pains, latch_cleared, *, pid, cause, provenance)`),
reusing `drive_span`, `drive_pressure`, `drive_comfort_progress` and `classify_pain`. The positive part of
`drive_delta` equals the existing `relief_fraction_from_progress` by construction; a guard pins `relief ==
max(drive_relief)` so the two cannot diverge (the memory-strength `sum(dict) == scalar` pattern).

**`drive:health` is carried as nociception**, consistent with `TISSUE_DAMAGE_DRIVES`, and the record's
docstring says it is an engineering proxy for injury (nociception_layer taxonomy note: hypoxic health loss
is near-painless). **ANTICIPATORY pain is excluded** from every field: a prediction is not an outcome
(nociception_layer principle 4).

#### 3.1.1 The forward-model projection (reconciled with `latent_forward_model.md`)

There is **one** record; the forward model's target is its projection, not a second record (an earlier
draft's separate `ConsequenceCode` is dropped).
`InteroceptiveOutcome.as_vector(schema_id="ans-v1")` returns:

- **Core, fixed 6-d, body-agnostic:** `[valence, nociception, drive_pain, relief, harm, urgency]`, with
  `valence = clip(relief − harm − nociception, −1, 1)` as the v1 innate prior (weighting is an owner
  decision, §8). This is what lets a Minecraft burn and a cradle burn land near each other.
- **Per-body drive block:** `deviation_after ‖ drive_delta` in the body's declared drive order. Its
  dimension varies by body, so the predictor either trains per body or predicts the core only;
  `schema_id` + `body_path` fix the order and a mismatch refuses.

**One join rule everywhere:** a training pair joins on `pid`, on the tool path and out-of-band alike,
and on nothing else. The executor's `uuid4` invocation id is not persisted-stable and is never a join
key; on the tool path `Executor._stamp_invocation` mints the `pid` and stamps it on the `ToolOutput`
(inside `ToolOutput.interoceptive_outcome`, beside the in-process invocation id) so the forward model's
`ActionContext` and this record carry the same one. While the target is this fixed code, the predictor
is supervised regression in a JEPA shape (owner decision G2).

#### 3.1.2 Identity: `PhysicalEventId`, one sequencer per agent, one declared cross-thread edge

A burn is one physical event seen on more than one `AfferentTrack` (GL3: nociceptive-fast and
affective-slow). Every track copy and this record carry the same `PhysicalEventId(agent_id, seq)`. The
contract (canonical across the grounding plans):

- **One type, one leaf module, built at GL2a.** `PhysicalEventId` lives in `embodiment/event_id.py`, a
  leaf with no `maxim` imports. GL3's relay imports it; it does not define it.
- **One seq authority per agent.** A per-agent `EventSequencer`, held by the agent's **primary**
  `Embodiment`, is the only minter. Ephemeral, scene and foundry wrappers (`agent_id == ""`, the
  `simulation/tools.py` `scene_emb`, the `simulation/foundry.py` wrappers) are not the AUT: they mint no
  record and no id. `seq` is assigned only when a record is **emitted** (latched), so it is stable under
  `_StepClock` lockstep tests.
- **seq persists per agent from GL2a.** Each record's `str(pid)` rides into the persisted trace through
  `EncodingSignals.extra["interoception"]`; on load the sequencer resumes past the maximum `seq` found
  there, the rule `memory/hippocampus.py::Hippocampus._resume_capture_seq` already applies to
  `capture_seq`. The load seam that hands the restored maximum to the attached `Embodiment` is named in
  GL2a's design pass.
- **GL3 handover.** At GL3.B3, seq assignment moves to the relay scheduler's drain point and the GL2a
  counter becomes the scheduler's backing store. One authority at a time: the handover is GL3.B3's stage
  gate, with a test that the two never both assign.
- **The orchestrator thread is a declared edge.** The narrator tools (`OrchestratorActorTool`,
  `DamageComponentTool`, `SetEntitySensorTool`, registered on `orch_registry`) call
  `_aut_embodiment.evaluate_failures()` on the orchestrator thread (the `start_simulation_mode` caller
  running the orchestrator agent's loop) while the AUT loop runs on `sim.aut` (§1.2a). The reflex
  dispatch's instances of the same classes run inside the AUT's `enrich`, on the loop thread. The agent loop is therefore **not** the only caller, and
  the Minecraft bridge is not the off-thread one (`MinecraftSyncPump._run` never calls
  `evaluate_failures`). From GL2a, `evaluate_failures`' latch, snapshot and mint section runs under one
  `Embodiment`-owned `threading.RLock` (re-entrant, per the CLAUDE.md threading rule: the section
  publishes pain synchronously, and whether any subscriber re-enters the body is checked in the design
  pass, not assumed), which guards the state both threads touch —
  `Entity.drive_breach_severity` and the per-entity snapshot slot — and the sequencer. No lock exists in
  `embodiment/body.py` today; adding it serialises an existing race and is called out in the GL2a PR.
  From GL3.B3, orchestrator-thread transduction posts into the relay's edge inbox, drained once per
  pass; GL3.B0's census and GL3.B3's gate 6 assert this thread identity.

#### 3.1.3 Urgency

Bio: urgency is need × imminence. v1 = `max(pressure_after)` (computable today, no clock). The slope term
(Δpressure per experience-clock µs, or time-to-deprivation for drifting entropic drives) waits for GL3's
timing work: world-owned Minecraft drives declare `drift_rate: 0` (`minecraft_player.yaml`), so imminence
must be measured, and the experience clock is the only legitimate clock (bio-memory brief).

#### 3.1.4 Where it is produced, and the provenance and forward-compat path

- **Tool path:** `Executor._stamp_invocation` already holds pressure-before, relief and pain. One additive
  field, `ToolOutput.interoceptive_outcome`, record-only. The existing trio
  (`drive_pressure_before`, `drive_relief`, `pain`) is filled from the same computation and stays (CC3;
  consumers read them). It is a `ToolOutput` **field**, stamped by the executor like that trio, not a
  `side_effects` key: `docs/user/tool_side_effects.md` registers what a tool's `execute()` reports, and
  tools never set this. So it needs no registry row. The forward model's `ActionContext` is built inside
  the tool but attached by the same executor stamp; this plan recommends it ride the same way (a field,
  not a registry key), and that decision is `latent_forward_model.md` S1's.
- **Out-of-band path:** `evaluate_failures` compares against the previous evaluation's per-entity snapshot
  and queues a record; `Embodiment.drain_outcomes()` returns them. Emission is **latched like channel 2**:
  band entry/exit, deepening past `_BREACH_DEEPEN_FRACTION`, a nociceptive event, or a satiation — never
  per tick.
- **`drain_outcomes()` lifecycle.** The production drainer is the loop capture
  (`runtime/tool_dispatch.py` → `runtime/bio_integration.py::capture_loop_action`) on the agent-loop
  thread, in substrate-primary and LLM-primary alike; in `--sim` that is the AUT loop on `sim.aut`, which
  also drains the records the narrator's tools queued from the orchestrator thread. The queue is **bounded**: on
  overflow it drops the oldest record, counts the drop and logs it (never silent). Ephemeral wrappers
  never queue (§3.1.2). A body attached where no loop capture runs queues up to the bound and then
  drops with the count; the GL2a design pass names the bound and lists such runtimes.
- **Consumers in GL2a:** a trace line, and the loop capture writing the core plus `str(pid)` and
  `provenance` into `EncodingSignals.extra["interoception"]` (no `EncodingSignals` field change). No
  reader acts. It is still a change to the **persisted memory record**: `EncodingSignals.to_dict`
  flattens `extra` into the trace the Hippocampus saves, so every loop capture's saved shape grows by
  this key (blast radius in §5.2).
- **Provenance (owner decision G6).** Three kinds: `experienced`, `narrated`, `imagined` (`declared` /
  `reported` stay open at GL3.B1, the registry+provenance stage). Consequences written by the narrator's tools
  (`simulation/tools.py::SetEntitySensorTool`, `DamageComponentTool`, `OrchestratorActorTool`, and the
  reflex dispatch, which uses separate instances of the first two classes on the loop thread) are stamped **`narrated`, never
  `experienced`**. A narrated record is usable for forward-model training and for credit at a **declared
  discount** (value: owner decision at GL4 start; the strict default for that decision is a small
  discount), and S5/GL5 report every result with and without narrated data. Excluding narrated records
  outright would mute the world the LLM's language priors simulate (owner rationale). `imagined` records
  are never produced by GL2. Mechanism: the narrator tools run their `evaluate_failures` call inside an
  `Embodiment` narrated scope (thread-local, so a concurrent loop-thread evaluation stays
  `experienced`); the Embodiment passes the scope's kind to the factory explicitly, and tool-path records
  minted by the agent's own executor are `experienced`. Whether the scope lands in GL2a or after the
  fence is §5.2's decision; until it lands, no out-of-band record is minted. Credit already paid
  today by PainBus for narrator pain is unchanged in GL2 (changing it would be its own ledger walk); the
  discount applies to this record's consumers. The forward model's contamination guard checks that a
  narrated record can never be relabelled `experienced` and that the discount is applied. Frozen does
  not prevent a relabel: `dataclasses.replace(rec, provenance="experienced")` builds a new record, so
  the no-relabel guard tests that call too, and asserts that the factory / validator path refuses it or
  that the contamination test catches it (no relabelling helper exists either).
- **Forward-compat:** `InteroceptiveOutcome` and `CauseRef` take CC3 path (a); `PhysicalEventId` takes path
  (b) with the rationale above. Serialised through `to_dict`/`from_dict` inside files that already carry
  `_format_version` (traces, Hippocampus); no new persisted file in GL2. No value is hashed; if a later
  stage hashes an id across processes it uses `utils/seeding.py::stable_hash_64_signed`.
- **Structural enforcement:** the factory's `pid=`, `cause=` and `provenance=` are **required
  keyword-only**, so forgetting identity is a `TypeError`, and `__post_init__` rejects the `pid=None` and
  `provenance=""` sentinels, so a record without them cannot be constructed at all. No type in the
  grounding line defaults provenance to `"experienced"`. Structural guard: the factory signature plus
  the stage's AST test that `InteroceptiveOutcome(` is never constructed directly in `src/`.

**Dependency on #1125.** The tool-path record must not inherit #1125's blindness: on the infant, a record
that omits `arms.thermal` omits the burn. GL2a therefore reads drives through
`tool_bridge.py::_resolve_sensor_slot` (the resolution #1125 applies to the records) and must not derive
`drive_delta` from `side_effects["drive_progress_by_drive"]` while #1161 keeps that side effect blind.
GL2a lands after #1125, or carries #1125's fix as its first commit (which of the two: §8); it never
ships a record that omits `arms.thermal`.

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
        modality: heat       # heat | cold | mechanical | chemical
        direction: above     # above | below
        threshold: <owner>   # noxious onset, sensor units
        pain_scale: <owner>  # intensity per unit past threshold, clamped [0,1]
```

Parsed into a new frozen `NociceptorSpec` (CC3 path (a)) on `Entity.nociceptor_specs` — not a field on the
frozen drive specs. Evaluated in `evaluate_failures` inside the drive loop's structure:

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
shipped sequences stack past it. **Body placement:** declare it on a **variant body**, never by editing
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
on the slow spinoreticular/affective route — never the dorsal-column `mechano_proprio` track. That the
slow affective track also carries **drive** breaches (air hunger, hunger, the infant's thermal
discomfort) is justified only as Craig's lamina-I homeostatic pathway (lamina-I spinothalamic afferents
carrying the body's physiological condition to the insula); outside that reading, drive breaches have no
claim on a pain track.

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

- `Embodiment.evaluate_failures(*, cause: CauseRef | None = None)` — keyword-only, default `None` = today's
  behaviour. The keyword is **body-wide**, but the cause is attached **per sensor**: only to the failures
  on sensors that B8 (`tool_bridge.py::_intrinsically_harmful_sensors`) marks as harmed by **this call's**
  own delta. A breach lingering on another sensor never inherits it. Callers that know the cause pass
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
  mechanism that must be earned; it is not proposed. On substrate-primary the negative rows have no
  selection reader either (the reader is the text salience scorer) — stated, not fixed here.

Tier: cause attribution is an **invariant** (who did it is a fact the producer knows); the valence is
**learned**.

### 3.5 The positive producer: satiation and relief

Two signals, kept distinct because biology keeps them distinct:

- **Satiation (drive-reduction crossing):** the breach latch clears with a latched severity present — the
  event the `EntropicDriveSpec` docstring promised; the homeostatic equivalent is a latched breach clearing
  back inside the hysteresis band. Emitted **once per breach episode**, never per tick.
- **Graded relief:** the positive part of `drive_delta` on any event, already on the record.

**Record stage (GL2a):** `InteroceptiveOutcome.satiated` + `relief`. Zero consumers; its only ledger
consequence is GL2a's record-shape walk (§5.2).

**Why every routing reaches the earned survival rows (owner decision G7).** The satiation crossing is
not a rare event on the bodies the earned rows ran on (§1.2 item 2): on `minecraft_player` the `oxygen`
and `health` latches clear on every surfacing and every regeneration after a latched breach (`oxygen`
below 14 bubbles); Exp 60's training trials breach, so the crossing fires **at Exp 60's own
`escape_water` contingency**, the very act whose anticipatory timing T1-13 measures; `food` there is
entropic (`satisfaction_threshold: 16.0`), and `minecraft_bench` `d1` (Exp 56) and `minecraft_bench57`
`d1` (Exp 57, the same spec) are entropic with `satisfaction_threshold: 0.3`. Any delivery of a positive signal at that site — through a `Reaction`,
through the relief store, or both — feeds a new positive value into the place the survival and transfer
claims were measured. **GL2c therefore fires T1-11, T1-12, T1-13, T1-14 and T1-15 whatever the
routing** (T1-12 added by the owner 2026-10-08; its `Re-run on:` matches T1-11's). No
routing "avoids" the T1-13 lapse; the routing only decides which surface the positive value lands on.

**Delivery (GL2c) — three routing options; the routing stays an open GL2c decision:**

| Option | Route | Reaches |
|---|---|---|
| B — relief store only | the satiation/relief record is the input to roadmap Phase 5's cluster-keyed relief store (world-cluster keyed), and nothing else. No `Reaction` is emitted; `_reward_bias` is never written by this producer. | the relief store; its reader is rung E2's, designed in the joint review |
| A — Reaction stage | `Reaction(kind="satiation", valence=POSITIVE, intensity=<relief of the satiated drive>, context=ReactionContext(agent_id=<body agent>), source="drive:<name>:satiation")` on `pain_bus.reaction_bus` | with no other wiring: `hippocampus.capture_reaction` (episode net valence) and `_distribute_reward_from_reaction` → `TemporalCreditDistributor.distribute` → `NAc.credit_node(+)` → `_reward_bias` → `recommend_action` Component 2 (`reward_bias(agent_id, "tool:<name>")`) and EC text-threshold widening. The first time `_reward_bias` can grow in production. |
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
option A that sentence fires by its letter. Under B or C it fires in substance, because the positive
value lands on the `escape_water` contingency the discharge argued nothing could reward. GL2c's PR
records the walk entry on T1-13 either way.

**Narrated satiation.** A crossing caused by a narrator write (e.g. a narrated feeding through
`SetEntitySensorTool`) is a `narrated` record (§3.1.4) and is delivered at the declared narrated
discount, never at full weight.

**Double credit (option A or C).** A tool-caused satiation is also credited by channel 3
(`update_cluster_reward`, a different map) — the nociception plan's F2 shape. Bio: phasic dopamine fires
once per unexpected reward. Recommendation: one credit per physical event, keyed by `pid`; the distributor
skips a satiation whose cause is a tool channel 3 already credited. Named in the Wire-integrity review.

**engram_formation.md E4's overreach set.** If any option ever lets this producer credit text nodes,
E4's 13-string widening overreach set must be read first (roadmap T7). Option B does not reach text
nodes.

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
| Computing the record from declared specs (deviation, delta, pressure, satiation crossing) | invariant | — |
| Core valence weighting (`relief − harm − nociception`) | innate prior | owner decision; learned only via the forward model |
| Urgency v1 = max pressure after | innate prior | slope term with GL3 timing |
| Nociceptor threshold and gain | innate prior | `adaptive_nociception.md` revive triggers |
| Heat need + affinity keywords | innate prior | follow-up issue in the GL2b(i) PR |
| Cause attribution (who did it) | invariant | — |
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
  (d) an entropic deprive → satisfy cycle produces exactly one satiation event on the record (today: none).
- File issues for the verified defects not yet filed (heat need, satiation producer, cause keying), linked
  from here.

### 5.2 GL2a — the record, record-only (outside the fence; owner decision G1)

- **Build:** `embodiment/event_id.py` (`PhysicalEventId`), the per-agent `EventSequencer` with its
  persisted resume, `CauseRef`, `InteroceptiveOutcome`, `sem.interoceptive_outcome`;
  `ToolOutput.interoceptive_outcome` (carrying the `pid`); the per-entity snapshot, the `RLock` and the
  bounded `Embodiment.drain_outcomes()`; the trace line and `EncodingSignals.extra["interoception"]`.
  Files: exactly the exempt set in the header (`embodiment/body.py`, `embodiment/sem.py`,
  `embodiment/event_id.py`, `runtime/executor.py` `_stamp_invocation`, `tools/base.py::ToolOutput`,
  `runtime/bio_integration.py`).
- **Narrated records and the fence.** Stamping narrator consequences `narrated` (§3.1.4) needs the three
  narrator tools' `execute` bodies in `simulation/tools.py` (four `evaluate_failures` call sites) to
  enter the narrated scope, and that file is not in the exempt set. An out-of-band record minted from a narrator call without the scope would be labelled
  `experienced`, which G6 forbids. Thread identity cannot stand in for the scope: the narrator tools run on
  the orchestrator thread (the `start_simulation_mode` caller running the orchestrator agent's loop),
  but the reflex dispatch calls separate instances of the same `DamageComponentTool` /
  `SetEntitySensorTool` classes from the AUT's enrichment pipeline, on the loop thread (`sim.aut`). Owner decision
  at GL2a start: **strict default** — GL2a ships the tool-path record only (minted solely by the
  executor whose embodiment is the agent's primary `Embodiment`; a narrator tool run by the orchestrator's executor
  mints nothing), and the out-of-band producer lands after the fence together with the narrated scope, so
  no out-of-band record's provenance is ever guessed; the non-strict alternative adds those three
  `execute` bodies (no orchestrator edit) to the exempt set and ships both halves now. The identity
  contract, the sequencer and the lock land in GL2a either way.
- **Depends on:** GL1's four-lens review of this plan; **#1125** — GL2a lands after #1125 or carries its
  fix as commit 1; it never ships a record that omits `arms.thermal`.
- **Falsifiable gate:** on a scripted cradle sequence (`cool_air` ×2, `warm_self` ×2, `touch`) the records
  match a committed hand-computed table, **including `arms.thermal`**; red gate (d) flips for the record
  half (through a satisfying tool action under the strict default; through drift as well once the
  out-of-band producer ships); the deletion probe (remove the producer) changes only the trace and the
  `extra` key.
- **Guards:** `relief == max(drive_relief)`; record present iff a body is attached (no zero records);
  latch semantics (entry / exit / deepen, no per-tick flood — reuse `test_transition_drive_pain.py`
  shapes); `PhysicalEventId` sequence deterministic under `_StepClock`, and resumed past the saved
  maximum after a save/load round trip; ephemeral wrappers mint nothing; `PhysicalEventId` rejects an
  empty `agent_id` and a negative `seq`; the required-keyword factory and the sentinel rejection; a
  two-thread test (loop thread + a narrator-thread caller) that seq stays unique and the latch
  consistent; the queue bound drops oldest with a count. Behaviour preservation byte-identical:
  `test_agent_loop_selection_golden.py`, `test_decision_provenance.py`, `test_encoder_golden_v1.py`.
  For the survival harnesses the gate **executes** the producer: the scripted water-trial smoke
  (`tests/unit/test_water_trial_smoke.py`) and `tests/unit/test_exp61_run.py` pass unchanged. (The
  committed Exp 60/61/R3 verdict reproductions are not a gate here: `compute_verdict` re-reads committed
  JSONL and never executes the code under test.)
- **Rows fired by wording: T1-16, not none.** GL2a adds `extra["interoception"]` to every loop capture,
  and `memory/encoding.py::EncodingSignals.to_dict` flattens `extra` into the persisted trace, so the
  saved memory record's shape changes and T1-16's "the memory record shape" trigger fires by its letter
  (T1-1, Exp 10, carries the older "hippocampus persistence schema change" wording but is SUPERSEDED by
  T1-16). The PR records the walk: either a structural discharge (`_rank_by_relevance` and
  `_query_hippocampus` read no `extra` key, shown by grep and a byte-identical ranking test over a
  store with and without the key) or the Exp 63 re-run. `runtime/executor.py` is touched: run the
  CLAUDE.md mypy invocation.

### 5.3 GL2b — the regulatory fixes (one issue + PR each)

**(i) Heat corrective need** (§3.3). Fenced: `substrate_proposal.py` and `nac.py`'s affinity table are on the
body selection path, so it waits for the remaining agent_loop slices.
- Gate: red gate (b) flips; `corrective_need_intensity` byte-identical over a grid on every shipped body;
  Minecraft `_read_drive_states` dicts identical.
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
  calibration decision; single `warm_self` / `*_safe` applications produce zero nociception; the B8
  causer-vs-bystander test on the chilled body extended to `noxious`; both the drive Reaction and the
  nociceptor Reaction of one contact reach `_distribute_reward_from_reaction`; deletion probe (remove
  the `noxious` rule in `failure_pain_kind` → the gate re-reds).
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
  breach lingering on another sensor never inherits the cause; the actor path's B8 gating; the aversion
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

- **Depends on:** GL2a; R4's routing audit has decided the selection surface (Phase 5 "first"); **one joint
  four-lens design review with Phase 5's relief store** (plus `fear_learning.md` Exp A and `coding_world.md`
  C3's reserved opposite sign), owner decision at stage start, recommended yes; **nociception_layer step 3
  (the F1 / F1b fix) landed before or with it** (§7); `engram_formation.md` E4's overreach set read if
  text nodes can be reached.
- **Build:** the satiation emission at the latch-clear site, routed per the owner's option (§3.5), behind a
  declared `maxim config` switch, default OFF, in M10's harness fingerprint, enabled only by a
  pre-registered experiment. (If an env var is unavoidable it needs an autouse conftest scrub in the same
  commit.) The restated refusals (§3.5) are part of the build.
- **Gate (flag off):** every golden byte-identical, and the executing survival checks
  (`test_water_trial_smoke.py`, `test_exp61_run.py`) unchanged. **Gate (flag on):** red gate (d) flips
  for delivery; on a scripted Minecraft surfacing-after-submersion and eat-after-hunger sequence, exactly
  one satiation per breach episode; under option B this producer never writes `_reward_bias`; under
  option A `_reward_bias` becomes non-empty **only** from satiation; the zero-bias guards in
  `test_nac.py` and the refusals in `donor_sanity_staged` and `_R3._boundary` are restated, not deleted;
  narrated satiations carry the declared discount.
- **Lands only with the batched live re-run** of Exp 60, 61 and 62 plus the T1-11 and T1-12 arguments
  or the Exp 56 / Exp 57 re-runs (G7): Exp 60's frozen gates still PASS on the rig, or the row is recorded BROKEN and blocks the
  release.
- **Rows fired, every routing (G7):** T1-11, T1-12, T1-13, T1-14 (also functionally: the donor sanity)
  and T1-15;
  plus, by the "PainBus / ReactionBus / NAc reward pipeline change" wording, T1-4 and T3-9 under options
  A and C; T1-6 / T1-9 for cradle hunger/thirst satiation, per wording; and the relief store's own walk
  (rung E2's) under options B and C.

### 5.5 Hand-off to the latent forward model

`latent_forward_model.md` consumes `drain_outcomes()` and the tool-path record as its target, core first.
Its S1 changes the Cerebellum's observe call to these consequence dims (target entity keyed); this plan
only guarantees the record exists and is honest. Affordances with no declared `self_effect` (363 of the
405 shipped, per the #1120 audit) have no predicted consequence: stated, not imputed.

## 6. Ledger blast radius

| Stage | Symbols touched | Rows whose trigger fires (by wording unless noted) |
|---|---|---|
| GL0 | docstrings, red gates | none |
| GL2a | new symbols and `embodiment/event_id.py`; `ToolOutput` additive field; lock, sequencer and snapshot in `evaluate_failures`; `extra["interoception"]` on every loop capture | **T1-16** ("the memory record shape": `EncodingSignals.to_dict` flattens `extra` into the persisted trace) — walk or structural discharge |
| GL2b(i) | `substrate_proposal` need map, `_DRIVE_TOOL_AFFINITIES` | T1-13 (floor), T1-6, T3-9 |
| GL2b(ii) | `failure_pain_kind`, `evaluate_failures`, a variant body YAML, the side-effects registry grammar | T1-6 (with #1161), T1-4, T3-9; T1-9/T1-10 only if a shared body changes |
| GL2b(iii) | percept-valence subscriber (cause namespace), PainSignal context, the aversion reader's exclusion, bundle scrub | T1-16 (pain episodes' shape); T1-14 by wording (the strict-default exclusion is a `bundle.py` scrub edit) and walked (donor check, hivemind keys); T1-2 (STALE); a live LLM-path change via `learned_aversions` unless excluded |
| GL2c (any routing) | the latch-clear emission and its route; the restated refusals | **T1-11 / T1-12 / T1-13 / T1-14 / T1-15 (G7: the crossing is Exp 60's `escape_water` contingency; T1-14 and R3 refusals functionally)**; T1-6 / T1-9 per wording |
| GL2c (A or C) | `_distribute_reward_from_reaction` input, `credit_node` writes | adds T1-4, T3-9 |
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
| GL2a | #1125: land after it, or carry its fix as commit 1? (A dependency either way, not an option to skip.) | After #1125, or its fix as commit 1; never ship a record that omits `arms.thermal`. |
| GL2a | Narrated records vs the fence: add the three narrator tools' `execute` bodies in `simulation/tools.py` to the exempt set, or ship the tool-path record only and land the out-of-band producer with the narrated scope after the fence (§5.2)? | **Tool-path only (strict)**: no out-of-band record's provenance is ever guessed. |
| GL2a | `drain_outcomes()` queue bound, and the load seam that hands the restored maximum `seq` to the `Embodiment` | A bound sized from a scripted session's peak, drop-oldest with a counted, logged drop; the seam named in the design pass. |
| GL2a | Core valence formula: `relief − harm − nociception`, or weighted (e.g. nociception ×2, a negativity bias)? | Unweighted v1; innate prior either way; revisit only with forward-model evidence. |
| GL2a | Urgency v1 pressure-only, slope with GL3 timing? | Yes. |
| GL2b(i) | Heat-need name (`heat` vs `overheat`) and affinity keywords; does it wait for a cooling affordance in a shipped scene? | `heat`; keywords chosen to avoid `withdraw`; build the need, and say in the PR that no shipped scene yet offers a cooling act. |
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
- **The record inherits upstream blindness** (#1125/#1161) unless it resolves sensors itself (§3.1.4).
- **Two writers on one body:** the narrator's orchestrator thread and the AUT loop both run
  `evaluate_failures` on the AUT body; without the GL2a lock, a seq could be minted twice or a latch read
  half-updated. The two-thread test is the guard.
- **Mislabelled provenance:** a narrator consequence recorded as `experienced` would hand language-prior
  physics to the forward model at full weight. The strict GL2a default (tool-path only until the narrated
  scope lands) and the contamination guard's no-relabel check are the guards.
- **Satiation lands on the measured contingency:** the oxygen/health crossing fires at Exp 60's
  `escape_water`, so a positive producer can reinforce exactly the act the survival claims measured
  (§3.5). Off by default and the batched re-run are the guards.
- **Claim drift:** keep autonomic wording out of release-claim prose until a GL5 record exists (M36 does
  not lint CHANGELOG claim lines).

## 10. New `[engineering]` invariants this plan would add (with their guards)

Each enters `docs/agents/embodiment.md` in the stage that builds it, with its `Regression guard:` line.

- The record is built only through the factory with required keyword-only `pid=`, `cause=`,
  `provenance=`. Regression guard: `embodiment/sem.py::interoceptive_outcome` signature (structural) + the
  stage's AST test that `InteroceptiveOutcome(` is not constructed elsewhere in `src/`.
- `relief == max(drive_relief)` on every tool-path record. Regression guard: the GL2a unit test (proposed
  `tests/unit/test_interoceptive_outcome.py`).
- `corrective_need_intensity` is the `"below"` projection of `corrective_need`, byte-identical on every
  shipped body. Regression guard: the GL2b(i) grid test (proposed `tests/unit/test_corrective_need_two_sided.py`).
- A cause is stamped only for sensors in the affordance's B8 harmful set. Regression guard: the GL2b(iii)
  bystander test.
- `PhysicalEventId` carries no uuid and no wall time, rejects an empty `agent_id` and a negative `seq`,
  and is minted only by the agent's `EventSequencer` (ephemeral wrappers mint none). Regression guard:
  the frozen dataclass in `embodiment/event_id.py` (structural, SHAPE-FROZEN) + the lockstep determinism
  test + the resume-past-saved-maximum round-trip test.
- No record carries a defaulted provenance; a narrator consequence is never `experienced`. Regression
  guard: `InteroceptiveOutcome.__post_init__` sentinel rejection (structural) + the GL2a (or post-fence
  narrated-scope) test that every narrator tool's record is `narrated`.
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
- Issues: #1120 (audit), #1125, #1161, #880 (F1), #888 / #889 (R4 defects), #908 / #909 (Cerebellum)
