# Embodiment Core Plan

> **Status:** ALL PHASES SHIPPED (2026-04-07). Phase 0: SEM protocol, YAML loader, auto-tool generation, LLM/narrative backends, body runtime, percept source, ATL integration. Phase 1a: Cerebellum forward models, R-W learning, CerebellumModulator. Phase 1b: Motor programs, ProgramRegistry (triple-indexed), program executor with pain gates, engram system, AdaptivePlanner + PromptBuilder wiring. Phase 2: Composed failures, persistent failures with recovery, ToolPainBridge embodiment integration, failure persistence. Phase 3 (Hardware) moved to future_plans.md. Total: 164 embodiment tests.
> **Depends on:** ATL semantic memory (exists), PainBus (exists), Hippocampus (exists), NAc (exists), ToolPainBridge (exists), PerceptSource protocol (exists), LLMRouter (exists), EnergySignal (exists), Tool base class + ToolRegistry (exists).
> **Related plans:** `dungeon_master_persona.md` (downstream consumer — DM's `CharacterState` mirrors body-state primitives), `agent_mesh.md` (adds EmbodimentCapability), `benchmark_plan.md` (Tier 3 metrics defined there, implementation details here).
> **Includes:** Hardware adapter (formerly `embodiment_hardware_adapter_plan.md`) merged as Phase 3.
>
> **Benchmark integration:** When each phase ships, add the corresponding Tier 3 metric computation to `AUTIntrospector.embodiment_stats()` and create benchmark scenarios in `scenarios/benchmarks/`. See `benchmark_plan.md` Tier 3 section for the metric interface.

---

## Core Insight

Maxim's ATL and proprioception subsystems were built for grounded math about a body. ATL holds canonical knowledge (joint ranges, torque curves, pain thresholds) as semantic concepts with IPS statistics and Angular Gyrus geometry. A new **Cerebellum** module stores lightweight forward models that predict sensory consequences deterministically. The LLM is consulted **only when no forward model exists** — it teaches the cerebellum, then fades out.

This fixes the biggest risk of LLM-imagined percepts (inconsistent physics at runtime) by making the LLM a **teacher**, not a per-tick oracle.

**The composability insight:** Hardware varies wildly — cameras, joints, wheels, grippers, IMUs, force sensors. The system must treat each as a self-describing unit that registers its own capabilities. The **Sensor-Entity-Modulator (SEM)** protocol makes every hardware component a composable triple: an Entity (the thing), its Sensors (how you read it), and its Modulators (how you change it). Each combo auto-registers agent tools, Cerebellum model keys, ATL concepts, and pain triggers — no hand-wiring.

---

## Architecture

```
             Agent proposes action
                      |
            SEM layer resolves entity + modulator
                      |
              +-------------------+
              |   Cerebellum      |  <-- forward models (fast, deterministic)
              |   (predict)       |
              +---------+---------+
                        |
           +------------+------------+
           |                         |
    Model exists?              No model?
           |                         |
           v                         v
   Predict percepts         Consult LLM (ATL-grounded)
           |                         |
           |                         +-->  train Cerebellum on result
           v                         v
           +-------------+-----------+
                         |
                  Predicted percepts
                         |
               Dispatch to backend (sim/hardware/rule)
                         |
                  Actual percepts (read from Sensors)
                         |
              Error = predicted - actual
                         |
              Cerebellum learns (R-W update)
              PainBus fires via ToolPainBridge (if threshold)
                         |
              Percepts -> MemoryAgent -> Hippocampus -> NAc
```

### Layer Responsibilities

| Layer | Role | Backed By |
|-------|------|-----------|
| **SEM Protocol** | Composable sensor/entity/modulator triples; auto-tool generation | New protocols in `embodiment/sem.py` |
| **EmbodimentSpec** | Declarative body (entities, sensors, modulators, failures) loaded from YAML | YAML files + dataclasses in `embodiment/spec.py` |
| **ATL Body Concepts** | Canonical physical knowledge (ranges, torque, wear, pain) | New `body_part` concept category |
| **Cerebellum** | Learned forward models per (entity, modulator, affordance, param_bucket) | New module, Rescorla-Wagner-style predictors |
| **LLMBackend** | Generates percepts for novel situations; teaches Cerebellum | Existing LLMRouter |
| **RuleBackend** | Deterministic physics for simple cases (joint limits, damping) | Pure Python |
| **Embodiment** | Runtime: holds entity tree, dispatches to backends, tracks failures | New class |

---

## Sensor-Entity-Modulator (SEM) Protocol

The SEM protocol is the composability foundation. Every piece of hardware — real or simulated — is described as a triple:

- **Entity**: the physical thing (a joint, a camera, a wheel, a gripper)
- **Sensor**: reads state from the entity (angle, temperature, frame, force)
- **Modulator**: changes state of the entity (rotate, set_torque, capture, restart)

Each is a small protocol class. Entities compose into trees (arm → elbow → wrist → gripper). The system auto-generates agent tools, Cerebellum keys, and pain triggers from the registered SEM graph.

### Protocol Definitions

```python
# --- embodiment/sem.py ---

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SensorReading:
    """One reading from a sensor."""
    sensor_name: str
    entity_name: str
    value: Any          # float, dict, ndarray — depends on sensor
    unit: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class ModulatorResult:
    """Outcome of a modulator action."""
    success: bool
    modulator_name: str
    entity_name: str
    affordance: str
    params: dict[str, Any]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Sensor(Protocol):
    """Reads state from an entity. One sensor = one readable quantity."""

    @property
    def name(self) -> str:
        """Sensor identifier, unique within its entity. e.g. 'angle', 'frame', 'temperature'."""

    @property
    def unit(self) -> str:
        """Human-readable unit. e.g. 'degrees', 'celsius', 'rgb_frame'."""

    @property
    def reading_schema(self) -> dict[str, Any]:
        """Describes the value shape for tool generation.
        e.g. {"type": "float", "range": [0, 360]}
        e.g. {"type": "ndarray", "shape": [480, 640, 3], "dtype": "uint8"}
        """

    def read(self) -> SensorReading:
        """Take a reading. Non-blocking for hardware; may block briefly for frame capture."""


@runtime_checkable
class Modulator(Protocol):
    """Changes state of an entity. One modulator = one controllable axis."""

    @property
    def name(self) -> str:
        """Modulator identifier, unique within its entity. e.g. 'rotate', 'pan_tilt', 'brake'."""

    @property
    def affordances(self) -> dict[str, AffordanceSchema]:
        """Named actions this modulator can perform.
        e.g. {"rotate_angle": AffordanceSchema(params={"degrees": float, "speed": float})}
        """

    def execute(self, affordance: str, params: dict[str, Any]) -> ModulatorResult:
        """Execute an affordance. Returns structured result."""


@dataclass(frozen=True, slots=True)
class AffordanceSchema:
    """Describes one named action a modulator can perform."""
    params: dict[str, type | tuple[type, Any]]   # Same format as Tool.input_schema
    description: str = ""
    timeout: float = 30.0


class Entity:
    """A physical thing with sensors and modulators.

    Entities compose into trees: arm -> elbow -> wrist -> gripper.
    Each entity is self-describing — its sensors, modulators, and
    failure modes are introspectable at runtime.
    """

    def __init__(
        self,
        name: str,
        entity_type: str,
        *,
        sensors: dict[str, Sensor] | None = None,
        modulators: dict[str, Modulator] | None = None,
        parent: Entity | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.entity_type = entity_type          # "joint", "camera", "wheel", etc.
        self.sensors: dict[str, Sensor] = sensors or {}
        self.modulators: dict[str, Modulator] = modulators or {}
        self.parent = parent
        self.children: list[Entity] = []
        self.metadata = metadata or {}          # free-form (joint limits, etc.)

        # Vital metrics (drift linearly, used by failure triggers)
        self.vital_metrics: dict[str, float] = {}

        if parent is not None:
            parent.children.append(self)

    @property
    def full_path(self) -> str:
        """Dot-separated path from root. e.g. 'left_arm.elbow.rotate'."""
        if self.parent is None:
            return self.name
        return f"{self.parent.full_path}.{self.name}"

    def walk(self) -> Iterator[Entity]:
        """Depth-first traversal of this entity and all descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def read_all_sensors(self) -> dict[str, SensorReading]:
        """Read every sensor on this entity. Returns {sensor_name: reading}."""
        return {name: sensor.read() for name, sensor in self.sensors.items()}

    def find(self, path: str) -> Entity | None:
        """Find a descendant by dot-path relative to this entity."""
        parts = path.split(".", 1)
        for child in self.children:
            if child.name == parts[0]:
                return child.find(parts[1]) if len(parts) > 1 else child
        return None
```

### Auto-Tool Generation

When entities register with the Embodiment runtime, tools are auto-generated and injected into the `ToolRegistry`. No hand-written tool classes per hardware component.

#### Tool Naming: Collision-Detecting Progressive Prefixing

Tool names must be LLM-friendly (semantic, not opaque IDs) and unique across the registry. Strategy:

1. **Default:** `{entity.name}_{affordance}` → e.g., `shoulder_rotate_angle`
2. **On collision:** prepend parent name → `head_shoulder_rotate_angle`
3. **Still collides:** use full path → `reachy_head_shoulder_rotate_angle`
4. **Detection is mechanical:** `ToolRegistry.register()` already raises `KeyError` on duplicate names. `generate_tools_for_entity()` catches this and retries with progressively longer prefixes.

This keeps names short and readable for the common case (single robot), and handles multi-robot without opaque IDs that would hurt LLM tool selection.

```python
# --- embodiment/tool_bridge.py ---

from maxim.tools.base import Tool, ToolOutput


def _resolve_tool_name(
    base_name: str,
    entity: Entity,
    registry: ToolRegistry,
) -> str:
    """Find a unique tool name by progressively prepending parent names.

    Tries: base_name → parent_base_name → grandparent_parent_base_name → full_path_base_name.
    """
    candidate = base_name
    ancestor = entity.parent
    while candidate in registry._tools:
        if ancestor is None:
            # Full path exhausted — should not happen with well-formed trees
            raise ValueError(f"Cannot resolve unique tool name for {base_name} on {entity.full_path}")
        candidate = f"{ancestor.name}_{candidate}"
        ancestor = ancestor.parent
    return candidate


class SensorReadTool(Tool):
    """Auto-generated tool: read a sensor on an entity."""

    def __init__(self, entity: Entity, sensor: Sensor, registry: ToolRegistry) -> None:
        self.name = _resolve_tool_name(f"read_{entity.name}_{sensor.name}", entity, registry)
        self.description = (
            f"Read the {sensor.name} sensor on {entity.name} "
            f"({entity.entity_type}). Returns value in {sensor.unit}."
        )
        self.input_schema = {}  # No params — just read
        self._entity = entity
        self._sensor = sensor
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        reading = self._sensor.read()
        return {
            "entity": reading.entity_name,
            "sensor": reading.sensor_name,
            "value": reading.value,
            "unit": reading.unit,
        }


class ModulatorAffordanceTool(Tool):
    """Auto-generated tool: execute one affordance on a modulator."""

    def __init__(
        self, entity: Entity, modulator: Modulator, affordance_name: str,
        schema: AffordanceSchema, registry: ToolRegistry,
    ) -> None:
        self.name = _resolve_tool_name(f"{entity.name}_{affordance_name}", entity, registry)
        self.description = (
            schema.description
            or f"Execute {affordance_name} on {entity.name} via {modulator.name} modulator."
        )
        self.input_schema = dict(schema.params)
        self.timeout = schema.timeout
        self._entity = entity
        self._modulator = modulator
        self._affordance_name = affordance_name
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        result = self._modulator.execute(self._affordance_name, kwargs)
        if not result.success:
            return ToolOutput(success=False, error=result.error)
        return {
            "entity": result.entity_name,
            "affordance": result.affordance,
            "success": True,
            **result.metadata,
        }


class EntitySenseTool(Tool):
    """Auto-generated tool: read ALL sensors on an entity at once."""

    def __init__(self, entity: Entity, registry: ToolRegistry) -> None:
        self.name = _resolve_tool_name(f"sense_{entity.name}", entity, registry)
        self.description = (
            f"Read all sensors on {entity.name} ({entity.entity_type}). "
            f"Sensors: {', '.join(entity.sensors.keys())}."
        )
        self.input_schema = {}
        self._entity = entity
        super().__init__()

    def execute(self, **kwargs: Any) -> Any:
        readings = self._entity.read_all_sensors()
        return {
            name: {"value": r.value, "unit": r.unit}
            for name, r in readings.items()
        }


def generate_tools_for_entity(entity: Entity, registry: ToolRegistry) -> list[Tool]:
    """Generate all tools for an entity and its descendants.

    Returns a flat list of Tool instances ready for ToolRegistry.register().
    Names are resolved against the registry to avoid collisions — if two
    robots both have a 'shoulder', the second gets prefixed automatically.
    """
    tools: list[Tool] = []
    for ent in entity.walk():
        # Bulk sensor read
        if ent.sensors:
            tool = EntitySenseTool(ent, registry)
            registry.register(tool)
            tools.append(tool)

        # Individual sensor reads
        for sensor in ent.sensors.values():
            tool = SensorReadTool(ent, sensor, registry)
            registry.register(tool)
            tools.append(tool)

        # Modulator affordances
        for modulator in ent.modulators.values():
            for aff_name, aff_schema in modulator.affordances.items():
                tool = ModulatorAffordanceTool(ent, modulator, aff_name, aff_schema, registry)
                registry.register(tool)
                tools.append(tool)

    return tools
```

### Example: How a Robot Arm Looks

```yaml
# scenarios/embodiment/robot_arm_3dof.yaml
body:
  name: robot_arm
  entity_type: arm
  children:
    - name: shoulder
      entity_type: joint
      sensors:
        angle: {unit: degrees, range: [-180, 180]}
        temperature: {unit: celsius, range: [20, 80]}
      modulators:
        motor:
          affordances:
            rotate_angle: {params: {degrees: float, speed: float}, description: "Rotate shoulder joint"}
            brake: {params: {}, description: "Engage shoulder brake"}
      failure_modes:
        - name: overextension
          trigger: {field: angle, op: ">", value: 175, pain: 0.8}
        - name: overheating
          trigger: {field: temperature, op: ">", value: 70, pain: 0.6}

    - name: elbow
      entity_type: joint
      sensors:
        angle: {unit: degrees, range: [0, 150]}
        strain: {unit: ratio, range: [0, 1]}
      modulators:
        motor:
          affordances:
            rotate_angle: {params: {degrees: float, speed: float}, description: "Rotate elbow joint"}

    - name: wrist_camera
      entity_type: camera
      sensors:
        frame: {unit: rgb_frame, shape: [480, 640, 3]}
      modulators:
        lifecycle:
          affordances:
            capture_frame: {params: {}, description: "Capture a single frame from wrist camera", timeout: 5.0}
            restart: {params: {}, description: "Restart the wrist camera"}
```

**Auto-generated tools from this YAML** (agent sees all of these):

| Tool Name | Type | Description |
|-----------|------|-------------|
| `sense_shoulder` | EntitySense | Read all sensors on shoulder |
| `read_shoulder_angle` | SensorRead | Read angle sensor on shoulder |
| `read_shoulder_temperature` | SensorRead | Read temperature sensor on shoulder |
| `shoulder_rotate_angle` | ModulatorAffordance | Rotate shoulder joint |
| `shoulder_brake` | ModulatorAffordance | Engage shoulder brake |
| `sense_elbow` | EntitySense | Read all sensors on elbow |
| `read_elbow_angle` | SensorRead | Read angle sensor on elbow |
| `read_elbow_strain` | SensorRead | Read strain sensor on elbow |
| `elbow_rotate_angle` | ModulatorAffordance | Rotate elbow joint |
| `sense_wrist_camera` | EntitySense | Read all sensors on wrist camera |
| `read_wrist_camera_frame` | SensorRead | Read frame sensor on wrist camera |
| `wrist_camera_capture_frame` | ModulatorAffordance | Capture a single frame |
| `wrist_camera_restart` | ModulatorAffordance | Restart the wrist camera |

### Backend Dispatch

Modulators don't know *how* they're backed. The same `Modulator` protocol works for:

1. **Simulated (Phases 0-1):** `LLMModulator` calls LLMRouter for novel actions, `CerebellumModulator` uses cached forward models
2. **Rule-based:** `RuleModulator` applies deterministic physics (joint clamping, damping)
3. **Hardware (Phase 3):** `HardwareModulator` wraps real SDK calls (Reachy Mini motor commands, camera capture)

```python
# Backend implementations — all satisfy the Modulator protocol

class LLMModulator:
    """Generates percept predictions via LLM for novel situations."""
    ...

class CerebellumModulator:
    """Uses learned forward models for fast deterministic prediction."""
    ...

class RuleModulator:
    """Deterministic physics — joint clamping, damping, thermal models."""
    ...

class HardwareModulator:
    """Wraps real hardware SDK calls. Phase 3."""
    ...
```

Similarly for sensors:

```python
class SimulatedSensor:
    """Returns state from the Embodiment's internal model."""
    ...

class HardwareSensor:
    """Reads from real hardware (encoder, thermometer, camera)."""
    ...
```

### Cerebellum Key Derivation

Cerebellum model keys derive naturally from the SEM triple:

```python
# Key = (entity.full_path, modulator.name, affordance_name, param_bucket)
# e.g. ("left_arm.elbow", "motor", "rotate_angle", "fast_90deg")
```

This replaces the plan's previous `(component, affordance, param_bucket)` with the richer SEM path, giving the Cerebellum per-entity-per-modulator resolution.

### Pain Trigger Wiring

Failure modes are declared per-entity in YAML (same structured trigger format as before). When a sensor reading crosses a trigger threshold:

1. Entity evaluates trigger condition against latest `SensorReading`
2. Fires `PainSignal` through existing `PainBus` with `source="embodiment"`, `context={"entity": entity.full_path, "sensor": sensor.name}`
3. `ToolPainBridge` captures it (existing subscriber) — NAc learns `(modulator_affordance, entity_state) -> pain`
4. Hippocampus captures episodic memory tagged with entity + failure mode

No new bus infrastructure. The SEM layer publishes to the same `PainBus` everything else uses.

### ATL Concept Auto-Registration

When an entity tree loads, each entity registers an ATL `body_part` concept with:
- Sensor ranges (from YAML `range` fields) → IPS statistics (min/max/expected)
- Modulator affordances → Angular Gyrus geometry (parameter space dimensions)
- Failure modes → pain threshold associations

This is the ATL grounding from the original plan, but driven by the SEM spec rather than hand-coded per body part.

---

## Context Injection + Motor Programs (Cross-Cutting, Spans Phases 0-3)

Three problems that must be solved together:
1. **How does the agent know which SEM tools to use and when?** (prompt injection)
2. **How does the system learn multi-step movement sequences?** (motor programs)
3. **How do we pre-train these without burning hardware time?** (simulation)

### Problem 1: SEM Context Injection into Agent Prompts

The existing `PromptBuilder` (`agents/prompt_builder.py`) already has a priority-based section system with `PromptBudgeter`. SEM context injects at three existing points:

**Tool listing (CRITICAL priority, line ~746):**
Auto-generated SEM tools appear in the tool list like any other tool. The existing `ToolIndex` learns which tools are relevant to the current goal and surfaces them first. No special handling — SEM tools are just tools.

**Sensor-driven percept context (NICE_TO_HAVE priority, observation section ~884):**
`EmbodimentPerceptSource` feeds sensor readings into the normal percept pipeline. The agent sees "shoulder angle: 172deg, temperature: 65C" as an observation, same as it sees "user said hello." When readings approach failure thresholds, pain colocation makes this urgent:

```
=== Body State (pain-relevant) ===
- shoulder.angle: 172° (WARN: overextension threshold at 175°, pain 0.8)
- shoulder.temperature: 65°C (approaching overheating threshold at 70°C)
- elbow.strain: 0.58 (nominal)
```

Pain-proximate readings get promoted from NICE_TO_HAVE to IMPORTANT priority. The agent doesn't need to be told "don't overextend" — it sees the warning and has NAc's causal prediction telling it what happens next.

**NAc causal context (IMPORTANT priority, line ~987):**
Already injected as `=== Causal Predictions ===`. After a few pain events, NAc predictions appear naturally:

```
=== Causal Predictions (learned from experience) ===
- shoulder_rotate_angle(degrees=180) → overextension_pain (valence=NEGATIVE, confidence=0.85)
- wrist_camera_restart() → frame_restored (valence=POSITIVE, confidence=0.72)
```

**SCN temporal context (via existing threshold adjustment):**
SCN's `get_threshold_adjustment()` modulates pain thresholds by time-of-day. If the robot has learned that joints stiffen after 30min idle, SCN lowers the overextension threshold during those periods. The agent sees tighter warnings without knowing why — the temporal pattern is baked into the numbers.

**ATL concept context (NICE_TO_HAVE priority, line ~949):**
Active body-part concepts surface when relevant:

```
=== Active Concepts ===
- fast-joint-flex: rapid rotation > 90°/s on any joint. Associated outcomes: strain (0.4), overheating (0.2)
```

**No new prompt sections needed.** SEM tools and body state flow through existing channels. The only new code is `EmbodimentPerceptSource` (already in Phase 0) and a pain-proximity promoter (~20 LOC in `prompt_builder.py` to check sensor readings against failure thresholds and bump priority).

### Problem 2: Motor Programs (Learned SEM Sequences)

A reach-and-grasp isn't three separate LLM decisions. It's one learned program: rotate_shoulder → rotate_elbow → close_gripper, with specific parameters tuned from experience. The Cerebellum is the right home for this — biological cerebellum stores exactly these coordinated motor sequences.

#### Data Structure

```python
@dataclass
class MotorProgram:
    """A learned sequence of SEM actions that achieves a named outcome."""

    name: str                          # e.g. "reach_forward", "look_at_sound"
    goal_signature: str                # What outcome this program achieves
    steps: list[MotorStep]             # Ordered SEM actions
    confidence: float                  # R-W learned reliability (0-1)
    total_executions: int
    total_successes: int
    avg_duration_s: float              # Learned from observation
    last_used: float                   # Timestamp — for SCN temporal patterns
    context_requirements: dict         # Preconditions (e.g. {"shoulder.angle": "<90"})
    known_failure_modes: list[str]     # Pain types this program has triggered


@dataclass
class MotorStep:
    """One step in a motor program — a single SEM affordance invocation."""

    entity_path: str                   # e.g. "left_arm.shoulder"
    modulator: str                     # e.g. "motor"
    affordance: str                    # e.g. "rotate_angle"
    params: dict[str, Any]             # e.g. {"degrees": 45, "speed": 1.0}
    expected_duration_s: float         # Learned
    expected_sensor_state: dict        # Cerebellum's predicted sensor readings after this step
    pain_gate: dict | None             # If set, abort program when sensor crosses threshold
```

#### Lifecycle: Discovery → Learning → Suggestion → Refinement

**1. Discovery (LLM-driven, Phase 0):**
When the agent executes a sequence of SEM tools to achieve a goal, the Cerebellum observes the sequence. If the same goal leads to the same (or similar) sequence 3+ times, the Cerebellum proposes crystallizing it as a `MotorProgram`:

```python
class Cerebellum:
    def observe_action_sequence(
        self,
        goal: str,
        steps: list[tuple[str, str, str, dict]],  # (entity, modulator, affordance, params)
        outcome_valence: Valence,
        duration_s: float,
    ) -> MotorProgram | None:
        """If this goal→sequence pattern recurs, crystallize as motor program."""
```

**2. Learning (R-W update, Phase 1):**
Each execution updates the program's confidence via Rescorla-Wagner:
- Success → confidence increases (capped at 0.95)
- Failure → confidence decreases
- Pain during execution → step-level `pain_gate` thresholds tighten
- Parameter drift → step params adjust toward observed successful values (running mean)

NAc observes the program-level outcome: `(motor_program_name, context) → outcome`. This is coarser than step-level learning and captures "reach_forward works in context X but fails in context Y."

**3. Suggestion (prompt injection, Phase 1+):**
When the agent proposes an action that matches a motor program's `goal_signature`, the system suggests the program. This happens at two levels:

**AdaptivePlanner level** (before LLM call):
```python
# In adaptive_planner.py propose_plans():
matching_programs = cerebellum.find_programs_for_goal(goal_signature)
if matching_programs:
    # Rank by: confidence × NAc_predicted_value × SCN_temporal_relevance
    best = rank_programs(matching_programs, nac, scn, current_time)
    if best.confidence > 0.6:
        # Return as a PlanCandidate with pre-filled steps
        return PlanCandidate(
            source="motor_program",
            program=best,
            confidence=best.confidence,
        )
```

**Prompt context level** (in LLM call, if planner doesn't short-circuit):
```
=== Available Motor Programs ===
- reach_forward (confidence: 0.82, 12 successes / 15 attempts)
  Steps: shoulder_rotate_angle(45°) → elbow_rotate_angle(30°) → ...
  Known risks: overextension if shoulder.angle > 160° at start
  Last used: 2min ago, typically works at this time of day
```

The agent can choose to invoke the program, modify it, or ignore it. High-confidence programs can execute without LLM consultation (the AdaptivePlanner short-circuits).

**4. Refinement (pain-driven, ongoing):**
When a motor program triggers pain mid-execution:

1. PainBus fires → program execution halts at the painful step
2. Cerebellum records which step caused pain and the sensor state at that moment
3. The step's `pain_gate` is added/tightened: "abort if shoulder.angle > 168 before this step"
4. NAc records `(program_name, pre-step_state) → pain` — learns *when* the program fails
5. Next execution: Cerebellum checks pain gates before each step. If gate would fire, it modifies params (reduce speed, reduce range) or aborts early and returns control to the LLM

This is the pain colocation you mentioned — sensors don't just report state, they gate motor programs in real time.

#### Subsystem Roles

| Subsystem | Role in Motor Programs | Interface Used |
|-----------|----------------------|----------------|
| **Cerebellum** | Stores programs, predicts step outcomes, manages pain gates, tunes params | `observe_action_sequence()`, `find_programs_for_goal()`, per-step `predict()` |
| **NAc** | Learns (program, context) → outcome valence. Vetoes programs with high failure prediction | `predict(event_type="motor_program", event_signature=name)` |
| **SCN** | Temporal relevance — programs used at similar times of day get priority. Detects rhythmic patterns (e.g., "camera restart needed every 2h") | `query_similar_time()`, `find_rhythmic_patterns()` |
| **ATL** | Generalizes programs into movement concepts. "reach_forward" and "reach_up" are both "reaching" — if one fails due to strain, the other gets a warning | Concept clustering on program metadata |
| **PainBus** | Real-time interrupt. Pain during step N halts the program and triggers refinement | `subscribe()` in Cerebellum's program executor |
| **PerceptSource** | Sensor readings feed pain gate evaluation before each step | `EmbodimentPerceptSource.next_percept()` |
| **Hippocampus** | Contextual engrams — links motor programs to situational context via short-lived cross-system traces | `capture()`, associative graph `add_bidirectional()`, `recall_associated()` |

### Motor Engrams: Cerebellum ↔ Hippocampus Cross-System Traces

The Cerebellum stores the **how** (motor program steps and forward models). But it doesn't know **where, when, or what was happening** when a program succeeded or failed. The Hippocampus does — that's what episodic memory is for. Motor engrams are short-lived associative traces that link the two, so contextual information can modulate motor execution.

#### What Is a Motor Engram

A motor engram is an `EpisodicMemory` in the Hippocampus that is:
1. Tagged with motor program metadata (`perception.observations["motor_program"] = program.name`)
2. Linked to the Cerebellum's motor program via the associative graph (`EdgeType.ASSOCIATES`)
3. Stored at **SHORT_TERM** tier by default — decays in ~2 days unless reinforced
4. Contains the contextual snapshot: what entities were in what state, what goal was active, what sensory environment was present, what time of day it was

This is *not* storing the motor program in the Hippocampus. The program lives in the Cerebellum. The engram is a contextual annotation that decays unless it proves useful.

#### Engram Formation

When a motor program executes and produces a **significant outcome** (pain intensity > 0.3, reward prediction error > 0.3, or novelty > 0.7), the Cerebellum requests an engram capture:

```python
class Cerebellum:
    def _form_engram(
        self,
        program: MotorProgram,
        entity_states: dict[str, dict[str, SensorReading]],
        outcome_valence: Valence,
        pain_step: int | None,           # Which step triggered pain, if any
        hippocampus: Hippocampus,
    ) -> str | None:
        """Capture a contextual engram in Hippocampus linked to this motor program.

        Only fires on significant outcomes — routine successes don't need
        episodic context (the Cerebellum's R-W update handles those).
        """
        salience = _compute_engram_salience(outcome_valence, program.confidence)

        memory_id = hippocampus.capture(
            perception=Perception(
                observations={
                    "motor_program": program.name,
                    "goal": program.goal_signature,
                    "entity_states": _snapshot_entity_states(entity_states),
                    "pain_step": pain_step,
                    "program_confidence": program.confidence,
                },
                salience=salience,
                novelty=1.0 - program.confidence,  # Novel if program is unconfident
            ),
            context=Context(
                active_goal=program.goal_signature,
                active_mode="motor_execution",
            ),
            action=Action(
                tool_name=f"motor_program:{program.name}",
                tool_params={"steps": len(program.steps)},
            ),
            outcome=Outcome(
                success=outcome_valence != Valence.NEGATIVE,
                result={"valence": outcome_valence.value, "pain_step": pain_step},
            ),
        )

        if memory_id:
            # Bidirectional associative edge: engram ↔ program
            # Weight is proportional to outcome significance
            hippocampus.graph.add_bidirectional(
                memory_id,
                f"cerebellum:program:{program.name}",
                EdgeType.ASSOCIATES,
                weight=salience,
            )

        return memory_id
```

**Significance threshold for engram formation:**
- Pain intensity > 0.3 during execution → always capture (the context of failure matters)
- RPE magnitude > 0.3 (surprising outcome, either direction) → capture
- Novelty > 0.7 (first few executions of a new program) → capture
- Routine success with confidence > 0.8 → do NOT capture (Cerebellum's R-W handles this)

This keeps engram volume low — only contextually informative episodes get stored.

#### Engram Recall: Context-Dependent Motor Program Selection

Before executing a motor program, the Cerebellum queries the Hippocampus for relevant engrams via spreading activation. This answers: "what happened *in similar contexts* when this program ran before?"

```python
class Cerebellum:
    def _query_engrams(
        self,
        program: MotorProgram,
        current_entity_states: dict[str, dict[str, SensorReading]],
        hippocampus: Hippocampus,
    ) -> list[MotorEngram]:
        """Retrieve contextual engrams relevant to executing this program now."""

        # 1. Get engram IDs linked to this program via associative graph
        program_node = f"cerebellum:program:{program.name}"
        associated = hippocampus.graph.get_associated(
            program_node,
            edge_types={EdgeType.ASSOCIATES},
        )
        engram_ids = [node_id for node_id, weight in associated if weight > 0.2]

        if not engram_ids:
            return []

        # 2. Spread activation from those engrams to find contextually similar ones
        activated = hippocampus.recall_via_spreading_activation(
            seed_ids=engram_ids,
            max_depth=2,      # Shallow — we want direct context, not distant associations
            decay=0.5,
            threshold=0.1,
        )

        # 3. Filter to motor engrams only, rank by context similarity
        engrams = []
        for memory, activation in activated:
            if memory.action.tool_name.startswith("motor_program:"):
                context_sim = _entity_state_similarity(
                    memory.perception.observations.get("entity_states", {}),
                    current_entity_states,
                )
                engrams.append(MotorEngram(
                    memory=memory,
                    activation=activation,
                    context_similarity=context_sim,
                    outcome_valence=memory.outcome.result.get("valence"),
                    pain_step=memory.outcome.result.get("pain_step"),
                ))

        return sorted(engrams, key=lambda e: e.activation * e.context_similarity, reverse=True)
```

**How engrams modulate motor execution:**

| Engram Signal | Effect on Motor Program |
|---------------|------------------------|
| Similar context + NEGATIVE outcome | Tighten pain gates for this execution. If confidence low enough, fall back to LLM |
| Similar context + POSITIVE outcome | Boost confidence for this execution. AdaptivePlanner short-circuits more readily |
| Similar context + pain at step N | Pre-emptively modify step N params (reduce speed/range) before executing |
| Different context + any outcome | Weak influence — engram similarity is too low to override Cerebellum's own model |
| No engrams found | Pure Cerebellum — execute based on forward models and R-W confidence alone |

#### Engram Lifecycle: Formation → Reinforcement → Consolidation or Decay

```
Program executes → significant outcome?
    |                    |
    No                   Yes
    |                    |
    R-W update only      Capture engram (SHORT_TERM)
                              |
                         Next execution in similar context
                              |
                    +--------------------+
                    |                    |
              Same outcome          Different outcome
                    |                    |
              Touch engram          Capture new engram
              (access_count++)      (associative edge to old one)
                    |                    |
              After 5+ touches      Old engram decays (2 days)
                    |
              Consolidation candidate
                    |
              Sleep cycle evaluates:
                - NAc corroboration (0.20)
                - SCN temporal recurrence (0.20)
                - Percept recurrence (0.12)
                - Base significance (0.30)
                    |
              Score > 0.6? → LONG_TERM
              Score < 0.3? → Remove
              Else → stays SHORT_TERM, re-evaluated next sleep
```

**The cyclical feedback loop:**

1. **Cerebellum → Hippocampus:** Motor program execution produces significant outcome → engram captured with contextual snapshot
2. **Hippocampus → Cerebellum:** Before next execution, Cerebellum queries engrams → context-dependent gating/modulation of the program
3. **Hippocampus → NAc:** Engram capture triggers NAc observation → causal link `(program, context) → outcome` strengthened
4. **NAc → Cerebellum:** Before execution, AdaptivePlanner checks NAc prediction → vetoes or greenlights based on learned valence
5. **Hippocampus → SCN:** Engram registered in temporal index → SCN detects rhythmic patterns (e.g., "reaching fails in the evening" when joints are warm/fatigued)
6. **SCN → Cerebellum:** Temporal threshold adjustment modulates pain gates by time of day

This creates a multi-system memory loop where each execution potentially updates three subsystems (Cerebellum, Hippocampus, NAc), and each subsystem's state influences the next execution differently. The engrams are the ephemeral glue — they live for ~2 days and either prove their value through reinforcement (consolidate to LONG_TERM, becoming permanent contextual knowledge) or decay away (the motor program keeps working fine without them, proving the context wasn't load-bearing).

#### What This Enables Experimentally

The engram system is explicitly designed for experimentation. Questions we can answer:

1. **Does contextual memory improve motor performance?** Disable engram formation, run the same motor programs. Compare success rates. If engrams help, programs in context-dependent tasks (reaching around obstacles, time-of-day-dependent movements) should degrade.

2. **What's the optimal engram TTL?** The 2-day SHORT_TERM default is a hypothesis. Try 6 hours, 1 day, 1 week. Measure motor program success rates as a function of engram retention. Too short = lost context. Too long = stale context interfering.

3. **Do cross-system cycles create emergent behavior?** Track the Cerebellum→Hippocampus→NAc→Cerebellum loop over 100+ program executions. Do patterns emerge that no single subsystem would produce? E.g., does the system learn "reach slowly in the morning, fast in the afternoon" without being explicitly told about time-varying joint stiffness?

4. **Engram consolidation selectivity:** Which motor engrams survive to LONG_TERM? Hypothesis: pain-associated engrams consolidate more readily than success-associated ones (negativity bias in biological memory). Test by comparing consolidation rates for positive vs negative motor engrams.

5. **Sim-to-real engram transfer:** Pre-train motor programs in simulation (engrams capture sim context). On hardware, do sim-trained engrams help or hurt? They might provide useful structural context ("reach_forward needs shoulder at <90 first") but misleading sensory context (sim sensor values don't match real ones). The confidence discount on sim-trained programs (0.5x) should protect against this, but measure it.

Benchmark scenarios in `scenarios/benchmarks/` should include motor-program-specific metrics: program success rate with/without engrams, engram consolidation rate, cross-system loop latency.

### Problem 3: Simulation-Driven Motor Program Pre-Training

Motor programs can be pre-trained in simulation before hardware deployment. This is the cheapest way to bootstrap a movement repertoire.

#### Flow

```
1. Load body YAML (e.g. robot_arm_3dof.yaml)
2. Build entity tree with SimulatedSensor + LLMModulator backends
3. Run maxim --sim agent --goal "learn to reach objects at various positions"
4. Agent discovers movement sequences via LLM-driven exploration
5. Cerebellum crystallizes successful sequences as MotorPrograms
6. Export: data/embodiment/cerebellum.json (includes motor programs)
7. Load on real hardware — programs transfer, Cerebellum refines from real sensors
```

**Key insight:** The LLM-backed simulation generates *plausible* sequences. The Cerebellum stores them. On real hardware, the forward models update from real sensor feedback (prediction error), but the program *structure* (which steps in which order) transfers directly. Pain gates are reset on transfer — they must be learned from real sensors.

#### Campaign-Based Training

Reuse the existing campaign YAML infrastructure from the research protocol:

```yaml
# scenarios/embodiment/motor_program_training.yaml
campaign:
  name: motor_program_bootstrap
  goal: "Learn reliable motor programs for basic manipulation"
  model: mistral-7b           # Cheap model for exploration
  runs: 5                     # Repeat to build confidence

  training_sequence:
    - goal: "reach forward to position (0.3, 0, 0.2)"
      repeat: 5
      success_criterion: {entity: "wrist", sensor: "position", target: [0.3, 0, 0.2], tolerance: 0.05}

    - goal: "reach up to position (0, 0, 0.4)"
      repeat: 5
      success_criterion: {entity: "wrist", sensor: "position", target: [0, 0, 0.4], tolerance: 0.05}

    - goal: "look at sound source"
      repeat: 3
      success_criterion: {entity: "head.camera", sensor: "frame", contains: "sound_source"}
```

#### Sim-to-Real Transfer Protocol

1. **Export** cerebellum state after simulation: `cerebellum.export_state() → JSON`
2. **Mark programs as sim-trained:** `program.source = "simulation"`, `program.confidence *= 0.5` (discount — sim physics aren't perfect)
3. **On hardware load:** Cerebellum loads programs but resets per-step `expected_sensor_state` to `None` (must be re-learned from real sensors)
4. **First N hardware executions:** Cerebellum runs programs slowly (reduced speed params) and observes real sensor feedback
5. **Confidence recovery:** After 3-5 successful hardware executions, confidence returns to normal levels
6. **Pain gate bootstrapping:** Gates start permissive on hardware, tighten from real pain signals

This is the same Rescorla-Wagner update loop that already exists — the only new thing is the confidence discount on sim-trained programs and the sensor-state reset.

#### What This Enables

- **Overnight pre-training:** Run sim campaigns on cheap local models while you sleep. Wake up with a robot that already knows 20 motor programs.
- **Safe exploration:** Novel sequences explored in sim first. Only validated programs execute on hardware.
- **Transfer across bodies:** Sim-train on `robot_arm_3dof.yaml`, transfer programs to a real arm with different joint ranges. Cerebellum adapts params from real sensor feedback. ATL provides the semantic bridge ("this is still a reaching motion, just with different limits").
- **Regression testing:** Run the training campaign after code changes. If programs degrade, the benchmark catches it before hardware deployment.

---

## Phase 0 — SEM Protocol + ATL-Grounded MVP (Gate) (~500 LOC)

**Goal:** Ship the SEM protocol, YAML loader, auto-tool generation, and LLM backend — prove the core loop produces stable, learnable (action -> pain) pairs with ATL grounding active from day one.

**Why SEM is in Phase 0:** Without the composable protocol, Phases 1-3 would build on a monolithic foundation and require rework. SEM is the load-bearing abstraction — it must exist before anything builds on top.

### Deliverables

- `src/maxim/embodiment/sem.py` — `Sensor`, `Modulator`, `Entity`, `SensorReading`, `ModulatorResult`, `AffordanceSchema` protocols and classes (motor program dataclasses deferred to Phase 1b)
- `src/maxim/embodiment/spec.py` — `EmbodimentSpec` YAML loader: parses body YAML into Entity tree with `SimulatedSensor`/`LLMModulator` backends
- `src/maxim/embodiment/tool_bridge.py` — `SensorReadTool`, `ModulatorAffordanceTool`, `EntitySenseTool`, `generate_tools_for_entity()`
- `src/maxim/embodiment/body.py` — `Embodiment` runtime: holds entity tree, evaluates failure triggers, emits PainSignals, manages vital-metric drift
- `src/maxim/embodiment/percepts.py` — `EmbodimentPerceptSource(PerceptSource)` adapter + pain-proximity priority promoter (~20 LOC in `prompt_builder.py`: readings near failure thresholds get bumped from NICE_TO_HAVE to IMPORTANT)
- `src/maxim/embodiment/llm_backend.py` — `LLMSensor` and `LLMModulator` implementations using LLMRouter with ATL-injected context
- `src/maxim/embodiment/atl_integration.py` — auto-register `body_part` ATL concepts from Entity tree
- `scenarios/embodiment/robot_arm_3dof.yaml` — demo spec
- `scenarios/embodiment/embodiment_baseline.yaml` — regression test scenario
- `tests/unit/test_embodiment_sem.py` — SEM protocol + tool generation tests
- `tests/unit/test_embodiment_mvp.py` — integration: YAML -> entity tree -> tools -> LLM percepts -> pain

### Hard Constraints

- **Fixed failure vocabulary.** 6 base modes only: `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.
- **Structured trigger format, no eval.** `{field: "angle", op: ">", value: 175, pain: 0.8}`.
- **No homeostasis.** Vital metrics drift linearly only.
- **LLM called once per action**, not per tick.
- **Failures route through existing `ToolPainBridge`** (extended to accept embodiment-sourced failures — ~20 LOC change).
- **Tool names are deterministic** from entity + sensor/affordance names, with progressive prefixing on collision. No dynamic renaming at runtime.
- **Sensor polling at 1Hz default.** `EmbodimentPerceptSource` reads sensors every 1 second. Pain-relevant sensors (those within 20% of a failure threshold) are promoted to every-tick reading. Polling rate adjustable via `MAXIM_EMBODIMENT_POLL_HZ` env var or user command (`maxim embodiment poll-rate 5`). System demand can increase rate temporarily (e.g., during active motor program execution, poll at tick rate).

### Success Criteria (Relative, Not Arbitrary)

1. **Grounding A/B comparison:** pain intensity sigma across 10 repetitions of same action must be >=50% lower with ATL grounding enabled compared to ungrounded baseline. Measured on `embodiment_baseline.yaml`.
2. **NAc learning:** after running `embodiment_baseline.yaml`'s forced-bounds-violation sequence, NAc must learn (action -> pain) link with confidence > 0.5 within 3 repetitions.
3. **Latency budget:** percept generation p95 < 2s per action (verified via `response_latency_ms` expectation type from refinement harness).
4. **Tool generation:** loading `robot_arm_3dof.yaml` must produce exactly the expected tool set (count + names match). No manual registration.

### Validation Approach

Use the existing `scenarios/refinement_baseline.yaml` pattern. Add embodiment-specific scenarios that use the **same metric expectation types** (`action_count_range`, `tool_success_rate`, `response_latency_ms`) already wired in `validation.py`. Add one new expectation type: `nac_convergence` (asserts causal link confidence >= threshold within N repetitions).

**If MVP fails** (sigma reduction < 50% OR NAc doesn't converge), stop and revisit architecture. The rest of the plan is contingent on this working.

---

## Phase 1a — Cerebellum Forward Models (~400 LOC, ~550 with tests/edge cases)

**Goal:** Replace LLM percept calls with learned deterministic predictors. Ship a stable checkpoint before adding motor programs and engrams on top.

### Design

```python
class Cerebellum:
    """Forward models for predicting sensory consequences of actions.

    Stores lightweight predictors per (entity_path, modulator, affordance, param_bucket).
    Each predictor learns via prediction-error feedback (Rescorla-Wagner style,
    mirroring NAc but for sensory prediction instead of reward).
    """
    def predict(
        self,
        entity: Entity,
        modulator: str,
        affordance: str,
        params: dict,
    ) -> dict[str, SensorReading] | None:
        """Return predicted sensor readings or None if no model exists."""

    def observe(
        self,
        key: ModelKey,
        predicted: dict[str, SensorReading],
        actual: dict[str, SensorReading],
    ) -> None:
        """Update forward model from prediction error."""

    def has_model(self, entity: Entity, modulator: str, affordance: str, params: dict) -> bool

    def get_confidence(self, entity: Entity, modulator: str, affordance: str, params: dict) -> float
    def get_variance(self, entity: Entity, modulator: str, affordance: str, params: dict) -> dict
    def prune_stale_models(self, max_age_s: float) -> int
    def export_state(self) -> dict      # for persistence
    def import_state(self, data: dict) -> None
```

### Model Structure (per key)

- Expected sensor values (mean, variance) per sensor on the entity
- Expected failure probabilities per failure mode
- Confidence (grows with observations)
- Last-observed timestamp (for pruning)

### Prediction Policy

- Confidence < 0.3 -> use LLM, observe, train Cerebellum
- Confidence >= 0.3 -> use Cerebellum prediction
- High-variance models -> fall back to LLM (uncertain predictions need grounding)

### Bucket Granularity Decision

**Highly specific (entity_path, modulator, affordance, param_bucket) keys.** No generalization at the cerebellum layer. Generalization happens at ATL: it clusters specific cerebellum models into concepts like "fast-elbow-flex" that span multiple buckets. Clean separation:
- **Cerebellum:** specific, deterministic, fast
- **ATL:** general, symbolic, slow

When Cerebellum has no model for the current bucket, it falls back to ATL concept prediction *before* calling the LLM.

### CerebellumModulator

A new `Modulator` backend that wraps the Cerebellum prediction pipeline:

```python
class CerebellumModulator:
    """Modulator that uses Cerebellum for prediction, LLM as fallback.

    On execute():
    1. Check Cerebellum confidence for (entity, affordance, params)
    2. If confident: return predicted sensor readings
    3. If not: delegate to LLMModulator, then train Cerebellum on result
    4. After actual readings arrive, update via observe()
    """
```

This slots into the existing SEM protocol — the Entity doesn't know or care whether its modulator is backed by Cerebellum, LLM, rules, or hardware.

### Thread Safety

Cerebellum is accessed from both agent loop (read during predict) and training path (write during observe). Use **per-key locks** (read-heavy pattern; per-key granularity avoids global contention). Document this in the class docstring.

### Deliverables

- `src/maxim/embodiment/cerebellum.py` — `Cerebellum` class with forward models, R-W update, per-key locks, persistence
- `src/maxim/embodiment/backends/cerebellum_modulator.py` — `CerebellumModulator` wrapping predict/fallback/train loop
- `tests/unit/test_cerebellum.py` — forward model accuracy, LLM-skip rate, persistence round-trip, thread safety

### Success Criteria

1. **Reproducible LLM-skip rate:** over a fixed replay of 20 action patterns x 5 reps = 100 actions, LLM calls drop from 100 to <=40.
2. **Prediction accuracy:** MAE on held-out actions < 20% of full-scale sensor range.
3. **Persistence:** Cerebellum state serializes to `data/embodiment/cerebellum.json` and reloads correctly after restart.
4. **Thread safety:** 8 threads x 100 concurrent predict/observe calls complete without deadlock or data corruption.

### Why Biologically

Biological cerebellum does exactly this — forward models predict sensory consequences of motor commands; climbing-fiber complex spikes carry prediction error; massive microcircuit specialization. This is a crude but functional version for synthetic sensors.

**Phase 1a is a stable checkpoint.** If it works, proceed to 1b. If forward models don't converge, debug here before adding motor programs on top.

---

## Phase 1b — Motor Programs + Engrams (~450 LOC, ~600 with tests/edge cases)

**Goal:** Crystallize recurring SEM action sequences into reusable motor programs. Wire Cerebellum ↔ Hippocampus engrams for context-dependent motor execution. Enable sim-driven pre-training.

**Depends on:** Phase 1a (Cerebellum forward models must be working).

### Motor Program Dataclasses

```python
# --- embodiment/motor.py ---

@dataclass
class MotorProgram:
    """A learned sequence of SEM actions that achieves a named outcome."""

    name: str                          # e.g. "reach_forward", "look_at_sound"
    goal_signature: str                # What outcome this program achieves
    steps: list[MotorStep]             # Ordered SEM actions
    confidence: float                  # R-W learned reliability (0-1)
    total_executions: int
    total_successes: int
    avg_duration_s: float              # Learned from observation
    last_used: float                   # Timestamp — for SCN temporal patterns
    context_requirements: dict         # Preconditions (e.g. {"shoulder.angle": "<90"})
    known_failure_modes: list[str]     # Pain types this program has triggered
    source: str = "discovered"         # "discovered" | "simulation" — for sim-to-real discount


@dataclass
class MotorStep:
    """One step in a motor program — a single SEM affordance invocation."""

    entity_path: str                   # e.g. "left_arm.shoulder"
    modulator: str                     # e.g. "motor"
    affordance: str                    # e.g. "rotate_angle"
    params: dict[str, Any]             # e.g. {"degrees": 45, "speed": 1.0}
    expected_duration_s: float         # Learned
    expected_sensor_state: dict        # Cerebellum's predicted sensor readings after this step
    pain_gate: dict | None             # If set, abort program when sensor crosses threshold
```

### Motor Program Similarity: Scoped Per-SEM Triple

**Similarity is only meaningful within the same `(entity_type, modulator_name, affordance_name)` tuple.** Comparing a `rotate_angle` on an elbow to a `capture_frame` on a camera is meaningless — different sensors, different parameter spaces, different physics.

**Sequence similarity** (for crystallization — "is this the same motor program?"):
- Two sequences match if they have the same ordered list of `(entity_path, modulator, affordance)` tuples
- Params are bucketed, not exact-matched: each float param is rounded to the nearest bucket (configurable, default: 10% of the sensor's declared range, or nearest 5 for unbounded params)
- A sequence seen 3+ times with the same structure (same SEM triples in same order) but different param buckets is still one program — the params are averaged across observations

**Context similarity** (for engram recall — "is this the same situation?"):
- Only compares **scalar sensors** on entities involved in the program
- Frame sensors, audio sensors, and other high-dimensional readings are **excluded** from similarity computation (they'd need embeddings, which is out of scope)
- Normalized per-sensor: `|current - remembered| / sensor_range` for each scalar sensor on each entity in the program
- Overall similarity = mean of per-sensor similarities, weighted by pain-proximity (sensors near thresholds get 2x weight)

```python
def entity_state_similarity(
    remembered: dict[str, dict[str, float]],    # {entity_path: {sensor_name: value}}
    current: dict[str, dict[str, float]],
    sensor_ranges: dict[str, dict[str, tuple[float, float]]],  # from YAML spec
    pain_thresholds: dict[str, dict[str, float]] | None = None,
) -> float:
    """Compare scalar sensor states between remembered and current context.

    Returns 0.0 (completely different) to 1.0 (identical).
    Only compares sensors that exist in both snapshots.
    Non-scalar sensors (frames, audio) are excluded — caller filters before passing.
    """
    similarities = []
    weights = []
    for entity_path in remembered:
        if entity_path not in current:
            continue
        for sensor_name, remembered_val in remembered[entity_path].items():
            if sensor_name not in current[entity_path]:
                continue
            current_val = current[entity_path][sensor_name]
            lo, hi = sensor_ranges.get(entity_path, {}).get(sensor_name, (0, 1))
            range_size = max(hi - lo, 1e-6)
            sim = 1.0 - min(abs(current_val - remembered_val) / range_size, 1.0)
            similarities.append(sim)

            # Pain-proximate sensors get 2x weight
            weight = 1.0
            if pain_thresholds:
                threshold = pain_thresholds.get(entity_path, {}).get(sensor_name)
                if threshold is not None:
                    proximity = 1.0 - min(abs(current_val - threshold) / range_size, 1.0)
                    if proximity > 0.8:  # Within 20% of threshold
                        weight = 2.0
            weights.append(weight)

    if not similarities:
        return 0.0
    return sum(s * w for s, w in zip(similarities, weights)) / sum(weights)
```

### Motor Program Crystallization

The Cerebellum gains a sequence observer that watches for recurring goal -> SEM-action-sequence patterns:

```python
class Cerebellum:
    # ... forward model methods from Phase 1a, plus:

    def observe_action_sequence(
        self,
        goal: str,
        steps: list[tuple[str, str, str, dict]],  # (entity_path, modulator, affordance, params)
        outcome_valence: Valence,
        duration_s: float,
    ) -> MotorProgram | None:
        """If this goal->sequence pattern recurs 3+ times with same SEM structure,
        crystallize as MotorProgram. Params are averaged across observations."""

    def find_programs_for_goal(self, goal_signature: str) -> list[MotorProgram]:
        """Return motor programs whose goal_signature matches."""

    def execute_program(
        self,
        program: MotorProgram,
        embodiment: Embodiment,
        pain_bus: PainBus,
        hippocampus: Hippocampus | None = None,
    ) -> ModulatorResult:
        """Execute a motor program step by step with pain-gate checks.
        Halts and refines if pain fires mid-sequence.
        If hippocampus provided, queries engrams before execution and
        captures new engram on significant outcomes."""

    def cleanup_program(self, program_name: str, hippocampus: Hippocampus | None = None) -> None:
        """Remove a motor program and its orphan graph nodes.
        Deletes the synthetic anchor node 'cerebellum:program:{name}'
        and all associated edges from the hippocampal graph."""
```

**AdaptivePlanner integration** (~30 LOC): Before LLM decomposition, check `cerebellum.find_programs_for_goal()`. If a high-confidence program exists and NAc doesn't veto it (`predict()` returns non-negative valence), short-circuit to program execution.

**Prompt injection** (~20 LOC in `prompt_builder.py`): When motor programs match the current goal, inject them as an `=== Available Motor Programs ===` section at IMPORTANT priority. The agent can invoke, modify, or ignore them.

### Orphan Graph Node Cleanup

When the Cerebellum prunes a stale motor program (via `prune_stale_models()` or explicit `cleanup_program()`), it must also clean up the synthetic anchor node in the hippocampal associative graph:

1. Remove `cerebellum:program:{program.name}` node from the graph
2. Remove all edges (both directions) connected to that node
3. Orphaned engram memories in the Hippocampus are left alone — they decay naturally via the normal SHORT_TERM → eviction lifecycle

This prevents graph pollution from deleted programs. ~15 LOC in `cerebellum.py`, called from `prune_stale_models()`.

### Deliverables

- `src/maxim/embodiment/motor.py` — `MotorProgram`, `MotorStep` dataclasses, `entity_state_similarity()`, param bucketing logic
- `src/maxim/embodiment/engrams.py` — `MotorEngram` dataclass, `_compute_engram_salience()`, engram config (significance thresholds, TTL)
- `src/maxim/embodiment/program_executor.py` — Step-by-step program runner with pain-gate checks, PainBus subscription for mid-sequence interrupts, engram-modulated gate tightening
- Extensions to `src/maxim/embodiment/cerebellum.py` — `observe_action_sequence()`, `find_programs_for_goal()`, `execute_program()`, `cleanup_program()`, `_form_engram()`, `_query_engrams()`
- `scenarios/embodiment/motor_program_training.yaml` — Campaign for sim-driven motor program pre-training
- `scenarios/embodiment/engram_cycle_test.yaml` — Scenario that forces context-dependent motor outcomes to test engram formation and recall (same program, different entity states, different outcomes)
- ~30 LOC in `planning/adaptive_planner.py` — program lookup before LLM decomposition
- ~20 LOC in `agents/prompt_builder.py` — motor program context injection
- `tests/unit/test_motor_programs.py` — crystallization from repeated sequences, param bucketing, sequence similarity, persistence
- `tests/unit/test_motor_engrams.py` — engram formation on significant outcomes, engram recall modulating program execution, engram decay after TTL, consolidation to LONG_TERM after reinforcement, orphan cleanup after program deletion

### Success Criteria

1. **Motor program crystallization:** after 5 reps of the same reaching sequence (same SEM structure, varied params), Cerebellum produces a MotorProgram with confidence > 0.3.
2. **Param bucketing:** sequences with params differing by <10% of sensor range are recognized as the same program.
3. **Pain-gate refinement:** after a program triggers pain at step N, next execution either modifies params or aborts before that step.
4. **Sim pre-training:** running `motor_program_training.yaml` produces >=3 crystallized programs that persist to `data/embodiment/cerebellum.json`.
5. **Persistence:** Motor programs serialize and reload correctly alongside forward models.
6. **Engram formation:** significant motor outcomes (pain > 0.3 or RPE > 0.3) produce hippocampal engrams; routine successes do not.
7. **Engram recall modulation:** when engrams indicate prior failure in similar context (scalar sensor similarity > 0.7), program execution tightens pain gates measurably (gate threshold decreases by >=10%).
8. **Engram lifecycle:** engrams at SHORT_TERM tier decay after 2 days without reinforcement; engrams accessed 5+ times appear as consolidation candidates during sleep.
9. **Cross-system cycle:** `engram_cycle_test.yaml` demonstrates the full loop: execute -> engram capture -> re-execute in same context -> engram recall modifies behavior -> different outcome than without engrams.
10. **Orphan cleanup:** after `cleanup_program("reach_forward")`, no edges to `cerebellum:program:reach_forward` remain in the hippocampal graph.

### Why Biologically

Motor programs (coordinated multi-joint sequences) are the cerebellum's core output. Mossy fiber inputs carry context; Purkinje cells output the coordinated motor plan. Hippocampal-cerebellar interactions are well-documented in neuroscience: the hippocampus provides contextual scaffolding for motor learning ("where and when did I learn this movement?"), while the cerebellum stores the procedural knowledge itself. The engram system models this division — ephemeral contextual traces in hippocampus, durable motor programs in cerebellum, linked by associative edges that either consolidate or decay.

---

## Phase 2 — Structured Composable Failures (~150 LOC)

**Goal:** embodiment failures persist and learn exactly like tool failures, using existing bridge.

### Fixed Vocabulary + Composition

Base modes (6): `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.

Compositions allow specific failures without taxonomy explosion:

```yaml
failure_modes:
  - name: tennis_elbow
    composes: [strain, fatigue]
    entity: left_arm.elbow        # SEM entity path
    trigger:
      all:
        - {field: strain, op: ">", value: 0.6}
        - {field: fatigue, op: ">", value: 0.5}
    pain_intensity: 0.5
    persistent: true
    recovery_condition: {field: fatigue, op: "<", value: 0.2}
```

### Persistence Pattern (Reuses ToolPainBridge)

Existing `ToolPainBridge` already persists tool failures through NAc/Hippocampus. Widen its input to accept embodiment-sourced failures (~20 LOC change):

- Failure fires -> `PainSignal` published with `source="embodiment"` metadata + `entity_path`
- Hippocampus captures episodic memory with `failure_mode` tag
- NAc learns `(affordance, entity_state) -> failure` causal link
- EC's associative graph links failures to triggering actions
- AdaptivePlanner consults NAc before re-proposing same action on same body state

### Success Criteria

1. After overextension failure, NAc predicts pain for same action within 3 repetitions.
2. EC associative recall returns prior failure when agent plans similar movement.
3. Failure modes serialize to `data/embodiment/failures.json` and reload correctly.

### Explicit Non-Goal

No LLM-invented failure modes at runtime. Users compose from base vocabulary in YAML; no runtime taxonomy extension.

---

## Phase 3 — Hardware Adapter via SEM (~300 LOC)

> Formerly standalone plan `embodiment_hardware_adapter_plan.md`. Now uses the SEM protocol as its interface.

**Goal:** Implement concrete `HardwareSensor` and `HardwareModulator` classes that wrap real robot SDKs. The existing `RobotController` interface stays for legacy callers; new code talks through SEM entities.

**Key change from original plan:** Phase 3 is no longer "wrap monolithic RobotController." It's "write SEM implementations backed by hardware." This means any new robot type is just a set of `HardwareSensor` and `HardwareModulator` classes — no changes to the core Embodiment or Cerebellum.

### Deliverables

- `src/maxim/embodiment/backends/hardware_sensor.py` — `HardwareSensor` implementations (joint encoder, camera, IMU, etc.)
- `src/maxim/embodiment/backends/hardware_modulator.py` — `HardwareModulator` implementations (motor, pan-tilt, lifecycle)
- `src/maxim/embodiment/backends/reachy.py` — Reachy Mini SEM factory: builds Entity tree with hardware-backed sensors/modulators from the existing `reachy-mini` SDK
- `Embodiment.sync_from_robot_state()` — pulls current pose into entity sensor state (backward compat bridge)
- `scenarios/embodiment/hardware_live_baseline.yaml`
- `tests/integration/test_embodiment_hardware.py`

### Example: Reachy Mini as SEM Entities

```python
def build_reachy_entity_tree(mini: ReachyMini) -> Entity:
    """Build SEM entity tree from a connected Reachy Mini."""

    head = Entity("head", "head_unit")

    # Head joints — each is an entity with angle sensor + motor modulator
    for joint_name in ["roll", "pitch", "yaw"]:
        joint = Entity(joint_name, "joint", parent=head)
        joint.sensors["angle"] = ReachyJointSensor(mini, f"head_{joint_name}")
        joint.modulators["motor"] = ReachyJointModulator(mini, f"head_{joint_name}")

    # Camera — entity with frame sensor + lifecycle modulator
    camera = Entity("camera", "camera", parent=head)
    camera.sensors["frame"] = ReachyCameraSensor(mini)
    camera.modulators["lifecycle"] = ReachyCameraModulator(mini)

    # Body yaw
    body = Entity("body", "base")
    body.sensors["yaw"] = ReachyBodyYawSensor(mini)
    body.modulators["motor"] = ReachyBodyYawModulator(mini)

    # Root entity
    root = Entity("reachy", "robot")
    head.parent = root
    root.children.append(head)
    body.parent = root
    root.children.append(body)

    return root
```

**Auto-generated tools from a live Reachy:**

| Tool | Backed By |
|------|-----------|
| `read_roll_angle` | `ReachyJointSensor` → `mini.head.roll.present_position` |
| `roll_rotate_angle` | `ReachyJointModulator` → `mini.head.roll.goal_position = X` |
| `read_camera_frame` | `ReachyCameraSensor` → `mini.media.get_frame()` |
| `camera_restart` | `ReachyCameraModulator` → reconnect media stream |
| `sense_head` | All head joint + camera sensors in one call |

### Adding a New Robot

To add support for a new robot (e.g., a wheeled platform, a drone, a custom arm):

1. Write `HardwareSensor` subclasses that read from the SDK
2. Write `HardwareModulator` subclasses that send commands to the SDK
3. Write a factory function that builds the Entity tree
4. Register via entry-point (`maxim.robots` group) or call `embodiment.register_entity_tree()`
5. Done. Tools auto-generate. Cerebellum learns. ATL grounds. Pain fires.

No changes to Embodiment, Cerebellum, tool_bridge, or any core module.

### Backward Compatibility

The existing `RobotController` ABC, `RobotRegistry`, `MotionTarget`, and all callers remain untouched. `HardwareModulator` implementations can internally use `RobotController` methods if convenient. The SEM layer wraps, not replaces.

### Success Criteria

1. Live Reachy Mini test: pose readings via SEM match `RobotController.get_current_pose()` within 1 tick
2. Cerebellum forward models predict head position with MAE < 5 degrees after 50 motor commands
3. Zero regression: all existing hardware tests pass unchanged
4. Adding a simulated "wheeled_base" entity type requires zero changes to core modules (validated by test)

### Risks

- Sync lag (mitigation: sync in same tick as state read)
- Double pain fire from PainDetector + Embodiment (mitigation: Embodiment defers to existing PainDetector for motor-derived pain)
- Affordance translation ambiguity (mitigation: deterministic mapping + tests per affordance type)

---

## LLM Fallback Cost Management (Spans Phases 0-2)

**Decision:** EnergySignal-based budgeting with Rescorla-Wagner-learned costs.

- Each LLM fallback emits an `EnergySignal` with tokens, latency, cost.
- **Per-context budgets** (not global): "novel action percept generation" and "failure narration" have separate learned costs.
- R-W updates expected cost per context bucket from actual observations.
- When cumulative energy on LLM fallbacks exceeds budget, cancel the current action and replan (consistent with existing resource-exhaustion handling).
- Budget is itself learned: initial guess -> updated from observations -> converges.

Reuses existing EnergySignal infrastructure. No new plumbing.

---

## Dynamic Sensor Rate-of-Change Bounds (Phase 1 polish)

**Decision:** bounds are *informative*, not *prescriptive*, and they adapt to body state.

- Each sensor has an `expected_rate_of_change` vital metric tracking the body's current state.
- Bounds widen with observed wear, injury, or malfunction — a damaged joint moves slower, its bound drops accordingly.
- Anomalies beyond bounds are still recorded (novelty signals or pain triggers) — they indicate "unexpected state change, investigate," not "reject."
- Cerebellum uses bounds to gate training weight: observations wildly outside bounds get reduced learning weight (avoid overfitting to sensor glitches).

---

## Simulation-Driven Development Process

Each phase validates through Maxim's own simulation harness:

1. Implement phase deliverable
2. Run `maxim --sim agent --goal "validate embodiment phase N" --persona researcher` with new capability
3. Measure success criteria from simulation reports (uses existing `refinement_baseline.yaml` infrastructure)
4. If criteria fail, diagnose and iterate before proceeding
5. If criteria pass, ship the phase and move to next

This plan dogfoods Maxim's testing harness for Maxim's own cognitive-architecture development.

---

## SEM Beyond Robotics: Virtual Entities in Simulation + Campaigns

The SEM protocol is hardware-agnostic by design. Nothing in `Sensor`, `Modulator`, or `Entity` assumes physical hardware — the protocols just describe "a thing that has readable state and executable actions." This means SEM can model **any interactive entity**: NPCs, items, environments, abstract systems.

### Why This Matters

The generative campaign plan (`generative_campaign_plan.md`) already creates named characters (NPCs) interacting with the AUT. Currently those interactions are free-text narration with no structured state. SEM gives them structure — an NPC has sensors (health, mood, trust), modulators (speak, attack, trade), and failure modes (death, betrayal). Items have sensors (durability, sharpness) and modulators (slash, parry, break). The same cognitive stack that learns about robot joints also learns about swords and doors.

This means:
- The agent gets **tools** to interact with world objects (same auto-generation)
- The **Cerebellum** learns forward models for world interactions ("swinging a damaged sword → it breaks")
- **NAc** learns causal links ("using the rusty sword → it shatters → pain")
- **Engrams** capture contextual episodes ("the sword broke when I hit the stone golem, not the bandit")
- **ATL** generalizes ("damaged weapons break under stress" as a concept)

### Example: A Sword as SEM Entity

```yaml
# In a campaign or generative scenario YAML
world_entities:
  - name: rusty_sword
    entity_type: weapon
    sensors:
      durability: {unit: ratio, range: [0, 1], initial: 0.3}
      sharpness: {unit: ratio, range: [0, 1], initial: 0.5}
      weight: {unit: kg, range: [0.5, 5], initial: 1.2}
    modulators:
      combat:
        affordances:
          slash: {params: {target: str, force: float}, description: "Slash at a target"}
          parry: {params: {}, description: "Parry an incoming attack"}
          throw: {params: {target: str}, description: "Throw the weapon"}
      maintenance:
        affordances:
          sharpen: {params: {}, description: "Sharpen the blade"}
          repair: {params: {material: str}, description: "Repair with materials"}
    failure_modes:
      - name: shatter
        trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}
      - name: dulled
        trigger: {field: sharpness, op: "<", value: 0.15, pain: 0.2}
```

**Auto-generated tools:**

| Tool | What it does |
|------|-------------|
| `sense_rusty_sword` | Read durability, sharpness, weight |
| `read_rusty_sword_durability` | Check how close it is to breaking |
| `rusty_sword_slash` | Attack a target (durability decreases, LLM/rules determine outcome) |
| `rusty_sword_parry` | Block attack (durability decreases more if force is high) |
| `rusty_sword_sharpen` | Increase sharpness (requires appropriate context) |

### Example: An NPC as SEM Entity

```yaml
  - name: grim_ferryman
    entity_type: npc
    sensors:
      trust: {unit: ratio, range: [0, 1], initial: 0.3}
      mood: {unit: ratio, range: [-1, 1], initial: 0.0}
      health: {unit: ratio, range: [0, 1], initial: 1.0}
    modulators:
      social:
        affordances:
          speak: {params: {message: str}, description: "Say something to the ferryman"}
          offer_payment: {params: {amount: float}, description: "Offer coins for passage"}
          threaten: {params: {}, description: "Threaten the ferryman"}
      physical:
        affordances:
          punch: {params: {force: float}, description: "Punch the ferryman"}
          shove: {params: {}, description: "Shove the ferryman aside"}
    failure_modes:
      - name: hostility
        trigger: {field: trust, op: "<", value: 0.1, pain: 0.4}
      - name: refusal
        trigger: {field: mood, op: "<", value: -0.5, pain: 0.3}
```

### Backend: `NarrativeModulator` and `NarrativeSensor`

Virtual entities need backends that resolve actions through narrative rather than physics:

```python
class NarrativeModulator:
    """Modulator for virtual entities in campaigns/simulations.

    On execute():
    1. Check Cerebellum for learned outcome prediction
    2. If no model: delegate to LLM with entity state + ATL concepts as context
    3. LLM returns updated sensor values + narrative description
    4. Cerebellum trains on the result
    """

class NarrativeSensor:
    """Sensor for virtual entities — reads from internal state model.

    State is maintained by the Embodiment runtime (vital_metrics dict on Entity).
    Updated after each modulator action via LLM or Cerebellum prediction.
    No hardware — just the simulated world model.
    """
```

This is exactly the same pattern as `LLMModulator`/`SimulatedSensor` from Phase 0 — just with narrative-flavored prompts. The Cerebellum learns "swinging the rusty sword at a stone golem reduces durability by ~0.15" the same way it learns "rotating the elbow at 90deg/s increases strain by ~0.1." Same R-W update, same forward model, same pain triggers.

### Integration with Generative Campaigns

The generative campaign plan's `_run_generative_campaign()` gains entity awareness:

1. **Arc YAML can declare `world_entities`** alongside narrative phases
2. **On campaign start:** entities are loaded into an Entity tree, `NarrativeSensor`/`NarrativeModulator` backends attached, tools auto-registered
3. **During turns:** when the narrator describes an interaction with an entity, the corresponding SEM tools fire. "The ferryman scowls" → `grim_ferryman`'s trust sensor drops. "You swing the sword" → `rusty_sword_slash` executes, Cerebellum predicts durability change
4. **Entity naming** (already in generative plan): `EntityIdentity` maps to `Entity.name` — log prefixes use the same identifiers as tool names
5. **After campaign:** Cerebellum state includes forward models for world entities. Replay the campaign with the same AUT — it remembers "the rusty sword breaks if durability < 0.1"

### What This Enables

- **Consistent world physics across campaigns:** The Cerebellum doesn't forget that rusty swords break. Run 10 campaigns — by campaign 5, the agent avoids using damaged weapons without being told
- **Cross-entity causal learning:** NAc learns "threatening the ferryman → trust drops → refusal → can't cross the river → pain." This chain is the same structure as "overextending the elbow → strain → overextension → pain." Same subsystem, same learning algorithm
- **Item/NPC memory persistence:** Cerebellum state persists across sessions. The agent remembers how swords work, how ferryman react, how doors open — all as forward models
- **Sim-to-sim transfer:** Forward models learned in one campaign type ("dungeon crawl") transfer to another ("wilderness survival") for shared entity types (weapons, NPCs). ATL provides the concept bridge
- **Benchmarking cognitive capabilities with narrative scenarios:** "Did the agent learn that the sword was unreliable?" is a testable hypothesis with engram formation + NAc prediction confidence as metrics

### Scope Impact

This is NOT a new phase — it's a natural extension of existing phases:

- **Phase 0:** `NarrativeSensor` and `NarrativeModulator` are trivial variants of `SimulatedSensor`/`LLMModulator` (~30 LOC each). The YAML loader just needs to accept `world_entities` alongside `body`
- **Phase 1a:** Cerebellum works unchanged — forward models keyed by `(entity_path, modulator, affordance, param_bucket)` don't care if the entity is a joint or a sword
- **Phase 1b:** Motor programs generalize to "action programs" — a combat sequence (draw_sword → slash → parry) crystallizes the same way a reaching sequence does. Engrams capture "this combat program failed against stone golems"
- **Generative campaign plan:** ~50 LOC to wire entity loading + tool registration into `_run_generative_campaign()`

Total additional LOC for virtual entity support: ~100-150, spread across phases. The architecture already supports it — it's just YAML + backends.

---

## What's Out of Scope (Tracked Elsewhere)

| Deferred To | Reason |
|-------------|--------|
| `agent_mesh.md` (Phase 1 bullet) | `EmbodimentCapability` in AgentIdentity |
| `generative_campaign_plan.md` | Virtual entity wiring into `_run_generative_campaign()` (~50 LOC there, SEM backends here) |
| `future_plans.md` — Phase 3 Hardware Adapter | `HardwareSensor`/`HardwareModulator` wrapping real robot SDKs. Not needed for sim/campaigns. Build when deploying to physical hardware. |
| `future_plans.md` research directions | ATL Self-Extension, federated embodiments, uncertainty-as-pain, curriculum learning, bio-multimodal sensors |

---

## Scope Summary

| Phase | Core LOC | Realistic LOC* | Sprints | Required? |
|-------|----------|----------------|---------|-----------|
| 0 (SEM + ATL-grounded MVP) | 500 | ~700 | 2 | **Yes -- gate** |
| 1a (Cerebellum forward models) | 400 | ~550 | 1 | **Yes** |
| 1b (Motor programs + engrams) | 450 | ~600 | 1 | **Yes** (after 1a stable) |
| 2 (Structured failures) | 150 | ~200 | 1 | **Yes** |
| 3 (Hardware adapter via SEM) | 300 | ~400 | 1 | After 0-2 validated |
| **Total** | **1,800** | **~2,450** | **6** | |

*\*Realistic LOC includes edge cases, error handling, YAML validation, test helpers, and the inevitable "oh wait, I also need X." Core LOC is the happy path.*

Four phases (1 split into a/b for a stable checkpoint), one hypothesis: **ATL-grounded + Cerebellum-cached LLM percepts, delivered through composable SEM triples, produce consistent-enough signals for NAc to learn stable causal links — and those links crystallize into reusable motor programs with hippocampal engrams providing context-dependent gating that transfers from simulation to hardware.**

---

## No Blockers

Everything depends on existing infrastructure:
- `Tool` base class + `ToolRegistry` + `Executor` ✓
- `PerceptSource` protocol ✓
- `PainBus` + `ToolPainBridge` ✓
- `Hippocampus` episodic memory ✓
- `NAc` causal learning ✓
- `ATL` semantic memory ✓ (new `body_part` category added in Phase 0)
- `LLMRouter` ✓
- `EnergySignal` + R-W engine ✓
- `refinement_baseline.yaml` metric expectation infrastructure ✓

Phase 0 can start today.
