# Reaction Abstraction Plan

> **Status:** Active — Phase 1 sequences inside the foundations wave; Phases 2–5 execute post-foundations.
>
> **Origin:** Emerged from the F0.4 architectural review. While deciding whether `Percept.metadata` was appropriately abstracted for pain signals, the investigation revealed that pain is already a reactive signal (not a percept) and that the codebase has no typed abstraction for evaluative signals that drive learning. This plan fills that gap.
>
> **Depends on:** F0.4 (PerceptContext), F0.5 (agent_id threading). Phase 2 additionally depends on F0.2 (PerceptTraceBuffer).
>
> **Supersedes:** F0.R1 and F0.R2 in `foundations_plan.md` are absorbed into Phases 1–2 of this plan.

## The sandwich: Percept → Reaction → Bio-systems

Maxim's agent loop needs two distinct input surfaces:

| Surface | What it represents | Example | Drives |
|---|---|---|---|
| **Percept** | "I observed X" — sensory/environmental input | "I see a mug on the table" | Memory formation (Hippocampus), concept extraction (ATL), temporal indexing (SCN) |
| **Reaction** | "X caused an internal state change that should drive learning" — evaluative/motivational signal | "Grasping the hot mug caused pain" | Valence/reward learning (NAc), avoidance gating (fear circuit), goal interruption |

Percepts are the bottom layer (input from world). Reactions are the middle layer (interpretation/valuation). Bio-systems are the top layer (memory, learning, decision). Every producer writes to one of the two typed surfaces; every consumer reads from one or both.

**Why not one surface?** Pain, fear, hunger, surprise, and fatigue are semantically distinct from observations. A vision percept ("I see a stove") should not carry the same schema as a pain reaction ("touching the stove burned me"). The stove percept builds a memory; the burn reaction drives avoidance learning. Conflating them in one type (e.g., pain-as-Percept with metadata) led to the exact untyped-dict problem that F0.4 solved for messaging framing — and the architectural discussion confirmed that pain [already flows through its own reactive pathway](../../src/maxim/proprioception/pain_bus.py) (PainBus → PainSignal → NAc bridges) rather than through the Percept pipeline.

**Why now?** The substrate roadmap promises at least five reaction types (pain, hunger, fear, surprise, fatigue). Without an abstraction, each gets its own bus, schema, NAc bridge, isolation check, and persistence surface. PainBus is already the template, and it's already bending (the "tuple[str, ...] is too flat" observation from the F0.R2 refinement). Defining the pattern once, before the second reaction type ships, is cheaper than migrating five ad-hoc implementations.

## Design decisions (locked in)

These were resolved in discussion before this plan was written. They are not open questions.

1. **`Reaction` is a single type with a `kind` field**, not a subclass hierarchy. Consistent with how Percept handles `modality` as a field. Less boilerplate, easier to extend, serializes uniformly.

2. **Separate `ReactionBus`**, not a unified event bus. Typed subscribers are statically discoverable ("who listens to pain?"). PainBus's existing refractory-period and attribution semantics carry over. A unified bus would conflate percepts, reactions, and everything else into string-tagged dispatch.

3. **SEM facilitates but doesn't know the Reaction type.** Modulators define failure modes, effort costs, and success/fail hooks in their YAML specs — that's pure SEM with no `Reaction` import. [Cerebellum](../../src/maxim/embodiment/cerebellum.py) (or a thin `ReactionBridge` adapter) reads modulator outcomes and translates: failure mode triggered → `Reaction(kind="pain")`, motor effort exceeded → `Reaction(kind="fatigue")`, unexpected success → `Reaction(kind="surprise")`. The SEM protocol's job is to expose the hooks cleanly enough that the translation layer has everything it needs. Generated SEM components (Asset Foundry) work automatically because the translation reads from the spec, not from hand-wired code.

4. **Phase 1 sequences inside the foundations wave** so the type exists when F0.R1 and the remaining F0.x items need it. ~200 LOC, types-only, no runtime changes.

## Core types

### `Reaction`

```python
@dataclass(frozen=True)
class Reaction:
    kind: ReactionKind            # "pain", "fear", "hunger", "surprise", "fatigue", ...
    intensity: float              # 0.0–1.0, same scale as PainSignal.intensity
    valence: Valence              # POSITIVE / NEGATIVE / NEUTRAL (from NAc)
    timestamp: float
    context: ReactionContext
    source: str                   # Producer identifier (e.g., "cerebellum:motor_failure", "pain_detector:velocity")
```

Frozen so producers can't mutate in-flight. `kind` is a string (not an Enum) for extensibility — new reaction kinds don't require code changes, just documentation. The isolation hygiene rule (below) constrains what fields may be added.

### `ReactionKind`

```python
ReactionKind = Literal["pain", "fear", "hunger", "surprise", "fatigue", "satiation"]
```

Extensible via Literal union growth. Kept as a type alias, not an Enum, to match the Percept pattern where `Modality` and `Channel` are Literal aliases.

### `ReactionContext`

Parallel to `PerceptContext`. Carries the binding information that lets consumers attribute the reaction to specific percepts and actions.

```python
@dataclass(frozen=True)
class ReactionContext:
    agent_id: str | None = None
    timestamp: float | None = None
    scn_tag: CircadianContext | None = None

    # What triggered this reaction — role-keyed for flexibility.
    # Different reaction kinds use different binding roles:
    #   pain:     {"trigger": ..., "action": ...}
    #   fear:     {"trigger": ...}
    #   surprise: {"trigger": ..., "expectation": ...}
    #   hunger:   {"absence_window": ...}
    bindings: dict[str, TraceSnapshot] = field(default_factory=dict)
```

The `bindings` dict is the key extensibility surface. Each reaction kind documents which roles it populates; consumers can switch on `kind` to interpret them. Adding a new reaction kind with a new binding role is a documentation change, not a schema migration.

### `TraceSnapshot`

A typed reference to a percept that was active when the reaction fired. Populated from the `PerceptTraceBuffer` (F0.2) at emission time.

```python
@dataclass(frozen=True)
class TraceSnapshot:
    percept_id: str
    activation_strength: float    # From the trace buffer's τ-decay
    content_hash: str | None = None  # For future verification / dedup
    decay_factor: float = 1.0     # How much the trace had decayed at snapshot time
```

Before F0.2 lands, `TraceSnapshot` can be constructed manually by the emitter with `activation_strength=1.0` and `decay_factor=1.0` (full-strength, no decay information). Once `PerceptTraceBuffer` exists, emitters call `trace_buffer.snapshot(agent_id)` to get real values.

## ReactionBus

Generalization of [PainBus](../../src/maxim/proprioception/pain_bus.py). PainBus today has:

- `publish(signal: PainSignal)` with refractory period (0.5s per type+entity)
- `subscribe(callback)` for typed subscriber registration
- History ring buffer for recent signals
- `route_pain_percept()` helper that converts Percept→PainSignal (the F0.R1 detour we're removing)

ReactionBus keeps all of these semantics but generalizes:

- `publish(reaction: Reaction)` — dispatches by `reaction.kind` to per-kind subscriber lists
- `subscribe(kind: ReactionKind, callback)` — typed subscription filtered by kind
- `subscribe_all(callback)` — for consumers that want every reaction (e.g., logging, research capture)
- Refractory period configurable per-kind (pain keeps 0.5s; hunger might use 60s)
- History ring buffer shared across kinds, queryable by kind

**Migration path for PainBus:** PainSignal becomes `Reaction(kind="pain")`. Existing PainBus subscribers are re-registered as `reaction_bus.subscribe("pain", callback)`. The `PainSignal` class is retained as a type alias or thin factory function for backward compatibility during the transition, then deprecated. `route_pain_percept()` is deleted (F0.R1).

## Producer protocols

```python
class PerceptProducer(Protocol):
    """Produces Percepts from a sensory or environmental source."""
    @property
    def name(self) -> str: ...
    def produce(self) -> Percept | None: ...

class ReactionProducer(Protocol):
    """Produces Reactions from an evaluative or motivational source."""
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> ReactionKind: ...
    def produce(self) -> Reaction | None: ...
```

These are structural (Protocol-based), not nominal (no base class to inherit). Any object that implements the interface is a valid producer. This matches Python's duck-typing culture and avoids forcing SEM types to inherit from agent-layer abstractions.

### Who implements what

| Producer | Protocol | Source |
|---|---|---|
| SEM sensors (via [EmbodimentPerceptSource](../../src/maxim/embodiment/percepts.py)) | `PerceptProducer` | Already designed, needs instantiation wiring |
| [CommsGateway](../../src/maxim/comms/gateway.py) | `PerceptProducer` | F0.4 migrated channel/sender to PerceptContext |
| [PerceptionAgent](../../src/maxim/agents/perception_agent.py) | `PerceptProducer` | Vision, transcript, CLI — already produces Percepts |
| `make_text_percept` factory (F0.6) | `PerceptProducer` | Text-in for AgentPool runtime unification |
| [PainDetector](../../src/maxim/proprioception/pain.py) (movement metrics) | `ReactionProducer` | Currently emits PainSignal; migrates to Reaction(kind="pain") |
| [ToolPainBridge](../../src/maxim/decisions/tool_pain_bridge.py) (tool errors) | `ReactionProducer` | Currently emits PainSignal; migrates |
| [PerceivedPainAssessor](../../src/maxim/proprioception/perceived_pain.py) (anticipatory) | `ReactionProducer` | Currently emits PainSignal; migrates |
| [CerebellumModulator](../../src/maxim/embodiment/backends/cerebellum_modulator.py) (on behalf of SEM modulators) | `ReactionProducer` | **New** — translates modulator outcomes into Reactions |
| Future: FearPredictor, HungerIntegrator, SurpriseDetector | `ReactionProducer` | Each ~50–100 LOC when needed |

## SEM integration: facilitates, doesn't own

### How it works today

The [SEM protocol](../../src/maxim/embodiment/sem.py) defines:
- **Sensors** (lines 71–106): `read()` → `SensorReading`. No Percept knowledge.
- **Modulators** (lines 110–136): `execute()` → `ModulatorResult` (success/error/metadata). No Reaction knowledge.
- **FailureModes** (lines 488–530): Declarative triggers (sensor field, op, threshold, pain_intensity). Evaluated by [Embodiment.evaluate_failures()](../../src/maxim/embodiment/body.py).

The translation happens outside SEM:
- [EmbodimentPerceptSource](../../src/maxim/embodiment/percepts.py) reads sensors → bundles into Percept (designed but unused)
- [Embodiment._publish_pain()](../../src/maxim/embodiment/body.py) reads failure mode triggers → publishes PainSignal to PainBus
- [CerebellumModulator](../../src/maxim/embodiment/backends/cerebellum_modulator.py) wraps modulator execution → trains Cerebellum on outcomes
- [tool_bridge.ModulatorAffordanceTool](../../src/maxim/embodiment/tool_bridge.py) calls modulator → evaluates failures → feeds Cerebellum

### What this plan adds

**No changes to SEM types.** Sensor, Modulator, FailureMode, Entity stay as-is. The SEM YAML spec schema is unchanged.

**CerebellumModulator gains ReactionProducer behavior.** When `execute()` completes:
- If the modulator result is a failure AND the entity has matching failure modes → emit `Reaction(kind="pain", intensity=failure.pain_intensity, context=ReactionContext(bindings={"action": ..., "trigger": ...}))`
- If the modulator result required excessive effort (motor program exceeded predicted cost) → emit `Reaction(kind="fatigue", ...)`
- If the modulator result was surprisingly good (Cerebellum predicted failure, got success) → emit `Reaction(kind="surprise", valence=POSITIVE, ...)`

**Embodiment._publish_pain() migrates to emit Reaction instead of PainSignal.** This is the F0.R1 work: the sim-layer `inject_pain` detour through Percept.metadata is replaced by direct Reaction emission. `route_pain_percept()` is deleted.

**Generated SEM components (Asset Foundry) work automatically** because the Cerebellum translation reads from the spec's failure mode declarations. A foundry-generated weapon with `failure_modes: [{trigger: charge < 0.05, pain: 0.9}]` produces `Reaction(kind="pain", intensity=0.9)` through the same Cerebellum path as a hand-authored one. The [foundry gauntlet's](deferred/asset_foundry_plan.md) scoring dimension "pain/failure activation" becomes: "did the generated entity produce Reactions of kind pain?"

## Isolation hygiene for reactions

Parallel to [F0.4's PerceptContext isolation rule](../../src/maxim/agents/percept_context.py):

**A Reaction produced by Agent A must be safe to deliver to Agent B without changing B's learning trajectory beyond what a real evaluative signal on that kind would do.**

Concretely, `ReactionContext` MUST NOT carry:

- **Cross-agent intent.** No `mother_lesson`, no `narrator_desired_avoidance`. If Mother wants Baby to learn to avoid fire, Mother has to produce a percept of fire; Baby's own pain system generates the avoidance Reaction. The Reaction is a function of Baby's internal state + recent percepts, not Mother's goals.
- **Private state of another agent.** No `sender_reward_history`, no `peer_pain_threshold`. Agents see each other as black boxes across both the Percept and Reaction surfaces.
- **Scenario/test oracles.** No `expected_reaction_kind`, no `correct_avoidance_target`. Scenario tagging for post-hoc analysis belongs in Percept.metadata (the escape hatch), not in ReactionContext.
- **Learned-policy hints.** No `suggested_response`, no `optimal_action_from_nac`. The receiving agent's NAc computes its own policy from its own causal links.

This rule is the Reaction-side complement to PerceptContext's rule. Together they define the **information barrier** for the [deferred mother_npc_stimulus_plan](deferred/mother_npc_stimulus_plan.md): Mother communicates with Baby through Percept content only; Baby generates Reactions from its own bio-stack only. No back-channel through either typed surface.

## NAc integration path

### Short-term (Phases 1–2): hashed context, no schema change

NAc's current causal link key is `(event_sig, outcome_sig, context_hash)`. When a Reaction arrives via an NAc bridge, the bridge computes `context_hash` from the ReactionContext bindings — `hash(str(sorted(bindings.items())))`. This is opaque but sufficient for pattern-matching: "did this exact binding pattern happen before?"

### Medium-term (substrate P2): structured access

Substrate P2 introduces per-node reward bias keyed by `(agent_id, node_id)` where `node_id` is an ATL concept ID. At that point, NAc gains a `percept_refs: tuple[TraceSnapshot, ...]` column on its causal link table so queries can run by percept involvement, not just by hash match. This is a NAc schema migration that lands in substrate P2, not in this plan's phases.

The hashed-context approach is explicitly a bridge, not a permanent design. It lets Reactions flow through NAc's existing learning path without blocking on a schema change.

## Runtime unification connection

Maxim has two parallel runtimes:

1. **MaximAgent** (single-agent): produces `Percept` objects via PerceptionAgent, consumes via MemoryAgent. Full bio-stack pipeline.
2. **AgentPool** (multi-agent): takes string percepts via `run_turn(agent_id, percept: str)`, calls `hippocampus.store_observation(text=..., metadata={"agent_id": ...})` directly. No Percept objects.

The Percept/Reaction producer protocols unify them: both runtimes register the same producers against the same buses, and the difference between them becomes **which producers to register** rather than **whether to use typed data at all**.

Specifically, Phase 4 introduces a `make_text_percept` factory (from F0.6) that implements `PerceptProducer`. AgentPool.run_turn wraps its string percept in `make_text_percept(text, agent_id=agent_id)` and the result flows through the same Percept surface that MaximAgent uses. The bio-systems all see the same data type regardless of runtime.

This is a prerequisite for the [deferred mother_npc_stimulus_plan](deferred/mother_npc_stimulus_plan.md): Mother (an AgentInstance-shaped thing) needs to produce Percept objects that Baby (a MaximAgent-shaped thing) consumes.

## Phasing

| Phase | Scope | LOC est. | Dependencies | Absorbs |
|---|---|---|---|---|
| **1 — Types** | Define `Reaction`, `ReactionContext`, `TraceSnapshot`, `ReactionKind`. Isolation-hygiene docstring. Module: `maxim/reactions/types.py`. No runtime changes. | ~200 | F0.4, F0.5 | — |
| **2 — ReactionBus + PainBus migration** | `ReactionBus` class. PainSignal → Reaction(kind="pain"). Migrate existing PainBus subscribers. Delete `route_pain_percept`. Rewrite `inject_pain` to emit Reaction directly. | ~300 | Phase 1, F0.2 (for TraceSnapshot population) | F0.R1, F0.R2 |
| **3 — SEM producer protocols** | `PerceptProducer` / `ReactionProducer` Protocol types. CerebellumModulator gains ReactionProducer behavior. Wire [EmbodimentPerceptSource](../../src/maxim/embodiment/percepts.py) into the agent loop (currently unused). SEM sensors adopt PerceptProducer. | ~200 | Phase 2 | F0.8 |
| **4 — Factory + runtime unification** | `make_text_percept` factory implements PerceptProducer. AgentPool.run_turn uses it. Percept factory consolidation across remaining call sites. | ~150 | Phase 3, F0.6 | F0.6 runtime-unification piece |
| **5 — NAc structured access** | NAc causal link table gains `percept_refs` column. Queries by percept involvement. Per-node reward bias keys off `(agent_id, node_id)`. | ~150 | Phase 4, substrate P2 | — |
| **Total** | | **~1,000** | | |

### Sequencing against the foundations wave

- **Phase 1** lands inside the foundations wave, after F0.5 merges. It's types-only (~200 LOC, no runtime changes) and becomes a prerequisite for Phase 2 and for the remaining F0.x items that touch reaction/pain paths.
- **Phases 2–4** execute post-foundations, absorbing F0.R1/F0.R2/F0.8/F0.6's unification piece so those items don't ship separately.
- **Phase 5** lands in substrate P2 where per-node reward bias is already scoped.

### What each phase unblocks

- After Phase 1: the Reaction type exists. Plan docs can reference it. F0.R1's "drop inject_pain detour" can be scoped against the new type.
- After Phase 2: PainBus is gone, ReactionBus exists. The second reaction kind (hunger, fear) can be added by implementing ReactionProducer + registering on the bus. ~50 LOC per new kind.
- After Phase 3: SEM components automatically produce Percepts and Reactions. The [Asset Foundry's](deferred/asset_foundry_plan.md) gauntlet scoring becomes generic ("did the entity produce useful Percepts + Reactions?").
- After Phase 4: both runtimes share the Percept surface. Mother NPC stimulus plan has its prerequisite.
- After Phase 5: NAc can query "which percepts were involved in this causal link?" — substrate P2's per-node reward bias works correctly.

## What exists today (audit results)

| Component | Status | Location |
|---|---|---|
| SEM Entity/Sensor/Modulator/FailureMode | **Complete** | [sem.py](../../src/maxim/embodiment/sem.py) |
| Cerebellum (forward models, motor programs) | **Complete** | [cerebellum.py](../../src/maxim/embodiment/cerebellum.py) |
| CerebellumModulator (adapter) | **Complete** | [cerebellum_modulator.py](../../src/maxim/embodiment/backends/cerebellum_modulator.py) |
| ComponentRegistry (YAML discovery + instantiation) | **Complete** | [component_registry.py](../../src/maxim/embodiment/component_registry.py) |
| Tool bridge (affordance → callable tool) | **Complete** | [tool_bridge.py](../../src/maxim/embodiment/tool_bridge.py) |
| Embodiment (entity tree, failure eval, vital drift) | **Complete** | [body.py](../../src/maxim/embodiment/body.py) |
| EmbodimentPerceptSource (sensor → Percept bridge) | **Designed, unused** | [percepts.py](../../src/maxim/embodiment/percepts.py) |
| PainBus + PainSignal | **Complete** | [pain_bus.py](../../src/maxim/proprioception/pain_bus.py) |
| PainDetector (movement → pain) | **Complete** | [pain.py](../../src/maxim/proprioception/pain.py) |
| PerceivedPainAssessor (anticipatory pain) | **Complete** | [perceived_pain.py](../../src/maxim/proprioception/perceived_pain.py) |
| Percept + PerceptContext (F0.4) | **Complete** | [bus.py](../../src/maxim/agents/bus.py), [percept_context.py](../../src/maxim/agents/percept_context.py) |
| Reaction type | **Not exists** | This plan, Phase 1 |
| ReactionBus | **Not exists** | This plan, Phase 2 |
| PerceptProducer / ReactionProducer protocols | **Not exists** | This plan, Phase 3 |

## Open items (not blocking Phase 1, but worth tracking)

1. **Should ReactionBus live in `maxim/reactions/` (new module) or `maxim/proprioception/` (where PainBus lives)?** Lean new module — reactions are broader than proprioception. Decision deferred to Phase 2 implementation.

2. **Cerebellum → Reaction emission: which specific modulator outcomes map to which reaction kinds?** Phase 3 implementation will need a concrete mapping table. Start with: failure → pain, timeout → fatigue, prediction_error → surprise. Hunger and satiation come from temporal integrators (non-SEM), not from modulator outcomes.

3. **EmbodimentPerceptSource is designed but unused.** Phase 3 wires it into the agent loop — but it's unclear who currently calls it. Likely wired through MaximAgent's perception pipeline or through a new embodiment-percept subscription on the bus. Needs investigation at Phase 3 implementation time.

4. **Asset Foundry's SEM protocol tests (8 structural tests)** reference the current SEM types only. After Phase 3 adds PerceptProducer/ReactionProducer protocols, the foundry should gain 2 additional tests: "generated entity's sensors satisfy PerceptProducer" and "generated entity's failure modes emit Reactions through CerebellumModulator." This is a ~40 LOC addition to the foundry's F-2 phase, noted here for cross-referencing.
