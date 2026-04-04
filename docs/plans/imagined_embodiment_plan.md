# Imagined Embodiment Plan

> **Status:** Not started. Novel extension to the percept system.
> **Integrates with:** Simulation Agent (new percept source), Research Protocol (embodiment experiments), PainBus (consumes pain signals), Hippocampus+NAc (causal learning from imagined consequences)

## Vision

Give Maxim an **Embodiment layer** — a unified representation of any body (real, imagined, or hybrid) that it can inhabit. The same abstraction powers:

- A real Reachy Mini (hardware-backed components)
- An LLM-imagined robot arm, humanoid, drone, or fictional creature (LLM-backed components)
- The **user** as a body in the simulated world (LLM-backed, so Maxim reasons about the user's embodiment too — their hands, voice, presence, attention)
- Mixed real/imagined bodies (real joints + imagined fatigue, real camera + imagined olfaction)

The body is composed of modular **Components**, each with its own health, sensors, affordances, and failure modes. Components decompose recursively — the LLM can spawn sub-tasks to flesh out detail where it matters ("give the arm three joints → give each joint strain + temperature → give the shoulder a dislocation failure mode").

The agent doesn't just *know* it's embodied — it *feels* embodied because every action produces sensory consequences through the same hippocampus → NAc causal-learning loop used by real hardware.

**This is NOT a simulation-only feature.** The Embodiment layer becomes Maxim's canonical way to represent any body, and hardware integration becomes "attach hardware backends to components." Simulation and hardware are both just backends for the same abstraction.

---

## Why This Matters

1. **One representation for all bodies.** Hardware, simulation, and user embodiment share the same Component/Sensor/Affordance vocabulary. No separate code paths.
2. **Hardware-free embodiment learning.** Train motor control, spatial reasoning, and safety-awareness without a physical robot.
3. **Cross-substrate generalization.** Swap the embodiment spec to test the same agent as a robot arm vs. a humanoid vs. an insect. Does the cognitive architecture generalize?
4. **Teaches causal pain learning.** An LLM reasons "fast motion at this angle would strain the elbow" → fires pain signal → hippocampus captures episodic memory → NAc learns action→pain association → agent avoids the pattern next time.
5. **Models the user as embodied.** In simulation, the user has a body too — hands, voice, attention, presence. Maxim can reason about the user's embodied state (tired, frustrated, distracted) through the same Component layer.
6. **Composable curriculum.** Start with a one-joint body, add limbs, sensors, and failure modes progressively. Like raising a body.
7. **Research substrate.** Lets the Research Protocol spawn embodiments with specific failure modes to probe cognitive responses.
8. **Decompositional construction.** The LLM itself builds the embodiment — given a description ("4-armed alien with heat vision"), it emits a tree of components, spawns sub-tasks to detail each one, and recurses until the spec is complete.

---

## Architectural Primacy — Embodiment as a First-Class Layer

The Embodiment layer becomes a core architectural element alongside MemoryHub, PainBus, and LLMRouter. It is **not** simulation-specific. Every Maxim instance has an embodiment — the question is only which backends power it.

### Before (current architecture)

```
Maxim
├── Hardware: RobotController + RobotState (hardcoded joint dict)
├── Simulation: SimulatedController (parallel mock implementation)
├── Proprioception: MovementTracker (tracks real poses only)
└── Percepts: ad-hoc per source (vision, transcript, pain)
```

Hardware and simulation have separate code paths. Pain routing is wired to physical motor commands. There is no unified "what am I?" representation.

### After (embodiment-first architecture)

```
Maxim
├── Embodiment (canonical body representation)
│   ├── ComponentTree
│   └── Backends per component:
│       ├── HardwareBackend  (real Reachy Mini)
│       ├── LLMBackend        (imagined components)
│       ├── RuleBackend       (physics sim)
│       └── ReplayBackend     (recorded sessions)
├── MemoryHub (unchanged)
├── PainBus (unchanged, now consumes from all backends)
└── LLMRouter (unchanged)
```

`RobotController`, `RobotState`, and `SimulatedController` become **HardwareBackend** implementations. `MovementTracker` becomes a consumer of component state updates rather than a standalone tracker. Every hardware integration becomes "connect this component to this backend."

### Refactoring Plan (Existing Code)

Small, safe changes. None break existing behavior:

1. **`RobotController` → `HardwareBackend`** (or wrap it). The controller already has `get_state()`, `goto_target()`, etc. — these map cleanly to Component `sensors` and `affordances`.

2. **`RobotState.current_pose` → `Embodiment.component_states["*_joint"].value`**. The pose dict becomes component state. Existing code reading `current_pose["yaw"]` now reads `embodiment.get_sensor_value("head", "yaw")`.

3. **`MovementTracker` → embodiment observer**. Subscribes to component state changes, computes metrics, emits pain signals via `PainBus` (as it does today).

4. **`SimulatedController` → `LLMBackend` + `RuleBackend`**. The existing mock becomes two more backends sharing the same protocol.

5. **Live percepts fold into embodiment percepts**. Today's `PerceptSource` implementations (vision, transcript, etc.) become components with sensor backends. A camera becomes `Component(name="camera", sensors=[SensorSpec(modality="vision")], backend="hardware")`.

This refactor can happen incrementally: add the Embodiment layer alongside current code, migrate one subsystem at a time, delete duplicated code last.

---

## Design Principles

### 1. Everything is a Component

A body is a graph of composable components. A component owns state, declares sensors, exposes affordances, and knows its own health. Components nest arbitrarily.

```
Embodiment (root)
├── Component "left_arm"
│   ├── Component "shoulder_joint"  (proprioceptive sensor)
│   ├── Component "elbow_joint"     (proprioceptive + strain sensor)
│   ├── Component "wrist_joint"
│   └── Component "hand"
│       ├── Component "index_finger" (tactile sensor)
│       └── Component "thumb"
├── Component "head"
│   ├── Component "left_eye"   (vision sensor, cone of 60°)
│   ├── Component "right_eye"
│   ├── Component "nose"       (olfactory sensor)
│   ├── Component "left_ear"   (auditory sensor)
│   └── Component "mouth"      (taste sensor, speech affordance)
└── Component "torso"
    ├── Component "heart"      (pulse sensor, vital metric)
    ├── Component "stomach"    (hunger sensor)
    └── Component "skin"       (temperature sensor, ambient)
```

A robot arm is just a subset:

```
Embodiment (root)
├── Component "base"
├── Component "shoulder_joint"
├── Component "elbow_joint"
└── Component "gripper"
    └── Component "force_sensor"
```

A drone is also just a subset:

```
Embodiment (root)
├── Component "frame"
│   ├── Component "rotor_1"  (rpm, temperature, vibration)
│   ├── Component "rotor_2"
│   ├── Component "rotor_3"
│   └── Component "rotor_4"
├── Component "imu"          (acceleration, orientation, gyro)
├── Component "battery"      (charge, temperature, cycles)
└── Component "camera"       (vision sensor, gimbal)
```

### 2. Sensors and Affordances Are Declarative

A component declares:
- **Sensors** it provides (what percepts it can generate)
- **Affordances** it offers (what actions can be taken on/with it)
- **Vital metrics** (health values that change over time)
- **Failure modes** (thresholds where things break)

```python
Component(
    name="elbow_joint",
    sensors=[
        Sensor("angle", "proprioception", range=(0, 180)),
        Sensor("angular_velocity", "proprioception"),
        Sensor("strain", "nociception", range=(0, 1)),  # pain!
    ],
    affordances=[
        Affordance("flex", params={"target_angle": float, "duration": float}),
        Affordance("relax"),
    ],
    vital_metrics=[
        VitalMetric("wear", initial=0.0, drift_rate=0.0001),   # ages with use
        VitalMetric("temperature", initial=37.0, recovery_rate=0.1),
    ],
    failure_modes=[
        FailureMode("hyperextension", trigger="angle > 175", pain=0.9),
        FailureMode("overuse", trigger="wear > 0.8", pain=0.3, persistent=True),
    ],
)
```

The LLM is handed this spec and generates percepts consistent with it.

### 3. Body State Is the LLM's Context

Every percept-generation call gets a compact summary of the current body state + recent actions + environment + pain history. The LLM reasons over this and emits percepts. The architecture is agnostic to what the body *is* — the spec is the entire contract.

### 4. The User Is Also Embodied

In any simulation, the user has a body — however simple. By default, the user is a minimal embodiment:

```yaml
name: user_default
description: A human participant interacting with Maxim
root_component:
  name: user
  kind: abstract
  children:
    - name: voice
      kind: sensor
      sensors: [{name: speech, modality: audition}]
      affordances: [{name: speak, params: {text: str}}]
    - name: attention
      kind: abstract
      sensors: [{name: focus, modality: interoception, range: [0, 1]}]
      vital_metrics:
        - {name: engagement, initial: 0.7, drift_rate: -0.001, recovery_rate: 0.005}
        - {name: patience, initial: 0.8, drift_rate: -0.0005}
      failure_modes:
        - {name: frustrated, trigger: "patience < 0.3", persistent: true}
        - {name: distracted, trigger: "engagement < 0.4"}
    - name: presence
      kind: abstract
      sensors: [{name: proximity, modality: vision}]
```

This gives Maxim a theory-of-mind surface: it can reason about the user's patience, attention, and engagement through the same vital metrics it uses for its own body. When the user says "just answer the question already," the simulation spec can decrement user patience and mark them as frustrated — letting Maxim learn to read these signals over time.

**Custom user embodiments**: if the user specifies otherwise ("I'm on a treadmill", "I'm driving", "I'm cooking and can only occasionally glance at the screen"), the orchestrator generates a richer user spec with appropriate components (hands busy, gaze divided, ambient noise sensor).

The user embodiment is sensed, not actuated, from Maxim's perspective — Maxim doesn't apply affordances to the user, but reads their state as percepts.

---

### 5. Decompositional LLM Construction

Embodiment specs are built by the LLM itself through recursive sub-tasks. Given a high-level description, the LLM emits a skeleton tree, then spawns construction tasks for each component, and each sub-task can further decompose until the tree is complete.

```
User: "Simulate a 4-armed alien with heat vision and echolocation"
    ↓
ConstructEmbodimentTool (top-level)
    ├── Emits root spec: 4 arms, 1 head, 1 torso
    └── Spawns sub-tasks:
        ├── ConstructComponentTool("arm_1")
        │   ├── Emits: 3 joints, 1 hand, 4 fingers
        │   └── Spawns sub-tasks for each joint...
        ├── ConstructComponentTool("arm_2") → identical pattern
        ├── ConstructComponentTool("head")
        │   ├── Emits: heat-vision eyes, echolocation ears, mouth
        │   └── Spawns sub-tasks for each sensor
        └── ConstructComponentTool("torso")
```

Each sub-task is a constrained LLM call that returns a validated `ComponentSpec` fragment. The tool can refuse ambiguous requests and ask for clarification ("how many fingers per hand?"). The top-level tool assembles fragments into a complete `EmbodimentSpec`.

**Why decompose**:
- Keeps individual LLM calls focused (one component per call → less hallucination)
- Specs can be built incrementally over a conversation ("now add a tail", "give the eyes a fatigue metric")
- Different LLMs can construct different parts (cheap model for standard joints, strong model for exotic sensors)
- Invalid components can be regenerated without rebuilding the whole tree

**Construction tools** (new, add to orchestrator toolset):

| Tool | Purpose |
|------|---------|
| `construct_embodiment` | Top-level: given description, generate root component + decomposition plan |
| `construct_component` | Generate a single component with its sensors/affordances/vital metrics |
| `decompose_component` | Take an existing component and add children (for refinement) |
| `validate_embodiment` | Check spec for physical plausibility, circular refs, missing affordances |
| `visualize_embodiment` | Render the component tree as ASCII/markdown for review |

**Construction prompt pattern**:

```
You are building a single body component for an embodied simulation.

CONTEXT: You are constructing "{component_name}" as a child of "{parent_path}".
EMBODIMENT: {embodiment_name} — {embodiment_description}
PARENT DESCRIPTION: {parent_description}
DEPTH: {current_depth} / MAX DEPTH: {max_depth}

REQUIREMENTS:
- Emit a single ComponentSpec with appropriate sensors, affordances, vital metrics, failure modes
- If this component should decompose further, list child components to construct next
- Do NOT decompose atomic components (single joints, single sensors)
- Keep names lowercase_snake_case, descriptions < 100 chars

Return JSON: {component: {...}, children_to_construct: [{name, description}, ...]}
```

The orchestrator iteratively drains the `children_to_construct` queue, calling `construct_component` for each, until no more children remain.

---

### 6. The Same Substrate Supports Real, Imagined, and Hybrid Bodies

The abstraction boundary is at the **Component interface**, not at "real vs. imagined." A real robot can have real joints AND imagined skin (because it has no tactile sensors yet). A simulation can have all imagined components. A hybrid could have a real arm with imagined fatigue modeling.

Components can be backed by:
- **Hardware** (reads real sensor values via `RobotController`)
- **LLM-imagined** (LLM generates plausible readings from body state)
- **Rule-based** (deterministic physics simulation)
- **Replay** (reads from a recorded session)

This means the same cognitive architecture runs across all embodiments without change.

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │   EmbodimentSpec (declarative)   │
                    │   - component tree                │
                    │   - sensors, affordances          │
                    │   - vital metrics, failure modes  │
                    └────────────────┬─────────────────┘
                                     ↓
                    ┌─────────────────────────────────┐
                    │   Embodiment (runtime instance)   │
                    │   - current component states      │
                    │   - vital metric history          │
                    │   - action history                │
                    │   - pain events                   │
                    └────────────────┬─────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ↓                          ↓                          ↓
 ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
 │ HardwareBackend │         │  LLMBackend    │         │   RuleBackend   │
 │  (real sensors) │         │ (imagined via  │         │ (physics sim)   │
 │                 │         │  LLM prompts)  │         │                 │
 └────────┬────────┘         └────────┬───────┘         └────────┬────────┘
          └──────────────────┬────────┴──────────────────────────┘
                             ↓
               ┌─────────────────────────────┐
               │   EmbodimentPerceptSource   │
               │   (implements PerceptSource) │
               └─────────────┬────────────────┘
                             ↓
                       Agent Loop
                    (MemoryAgent, PainBus,
                     Hippocampus, NAc)
```

Action flow is the mirror:

```
Agent decides action
    ↓
Affordance check (does any component offer this action?)
    ↓
Embodiment.apply_affordance(component, affordance, params)
    ├─ Updates component state
    ├─ Updates vital metrics
    └─ Emits action history entry
    ↓
Next tick: backend generates percepts from new state
```

---

## Core Data Model

### EmbodimentSpec

```python
@dataclass(frozen=True)
class EmbodimentSpec:
    """Declarative description of a body (real or imagined)."""
    name: str                               # "reachy_mini", "human_adult", "4-armed_alien"
    description: str                        # LLM context: "humanoid with standard reach"
    root_component: ComponentSpec
    environment_hints: str = ""             # "kitchen counter, room temp 22C"
    tick_rate_hz: float = 2.0               # sensor generation cadence
```

### ComponentSpec

```python
@dataclass(frozen=True)
class ComponentSpec:
    """A body part. Recursively composable."""
    name: str
    kind: str                               # "joint", "organ", "sensor", "limb", "abstract"
    children: tuple[ComponentSpec, ...] = ()
    sensors: tuple[SensorSpec, ...] = ()
    affordances: tuple[AffordanceSpec, ...] = ()
    vital_metrics: tuple[VitalMetricSpec, ...] = ()
    failure_modes: tuple[FailureModeSpec, ...] = ()
    backend: str = "llm"                    # "llm", "hardware", "rule", "replay"
    backend_config: dict[str, Any] = field(default_factory=dict)
    description: str = ""                   # LLM context: "load-bearing elbow"
```

### SensorSpec / AffordanceSpec / VitalMetricSpec / FailureModeSpec

```python
@dataclass(frozen=True)
class SensorSpec:
    name: str                           # "angle", "strain", "temperature"
    modality: str                       # "proprioception", "vision", "olfaction",
                                        # "nociception", "tactile", "audition",
                                        # "vestibular", "interoception"
    unit: str = ""                      # "deg", "celsius", "lumens"
    range: tuple[float, float] | None = None
    percept_source: str = "proprioception"  # maps to Percept.source field

@dataclass(frozen=True)
class AffordanceSpec:
    name: str                           # "flex", "grasp", "look_at", "speak"
    params: dict[str, type]             # {"target_angle": float, "duration": float}
    preconditions: tuple[str, ...] = () # expressions checked before execution
    cost: dict[str, float] = field(default_factory=dict)  # {"energy": 0.1}

@dataclass(frozen=True)
class VitalMetricSpec:
    name: str                           # "wear", "fatigue", "charge"
    initial: float
    min_value: float = 0.0
    max_value: float = 1.0
    drift_rate: float = 0.0             # change per tick when idle
    recovery_rate: float = 0.0          # change per tick toward resting value
    resting_value: float | None = None

@dataclass(frozen=True)
class FailureModeSpec:
    name: str                           # "hyperextension", "overheating", "fracture"
    trigger: str                        # expression over component state
    pain_intensity: float               # 0-1
    percept_modality: str = "nociception"
    persistent: bool = False            # stays active until healed
    description: str = ""               # LLM context: "joint forced past limit"
```

### Embodiment (runtime)

```python
class Embodiment:
    """Runtime instance of an EmbodimentSpec."""
    spec: EmbodimentSpec
    component_states: dict[str, ComponentState]  # name → state
    vital_metrics: dict[str, dict[str, float]]   # component → metric → value
    active_failures: dict[str, list[ActiveFailure]]
    action_history: deque[AppliedAction]
    pain_events: deque[PainSignal]
    tick_count: int
    
    def apply_affordance(self, component_name, affordance_name, params) -> Outcome
    def tick(self, dt: float) -> None
    def get_active_sensors(self) -> list[tuple[ComponentSpec, SensorSpec]]
    def summarize_for_llm(self, budget_tokens: int = 500) -> str
```

---

## LLM Prompt Template

The critical piece: teaching the LLM to reason about embodied consequences.

```
You are simulating the sensors of an embodied agent.

EMBODIMENT: {spec.name}
DESCRIPTION: {spec.description}
ENVIRONMENT: {spec.environment_hints}

ACTIVE COMPONENTS (with sensors):
{component_tree_with_sensors}

CURRENT STATE:
{vital_metrics_summary}

ACTIVE FAILURE MODES:
{active_failures}

RECENT ACTIONS (last {n} ticks):
{action_history}

RECENT PAIN:
{pain_history}

TASK:
Generate the sensor percepts that would plausibly result from this state.
Reason about: physics (friction, inertia, gravity), physiology (fatigue,
strain, recovery), and environment (temperature, obstacles, affordances).

Return a JSON array. Each percept is either:
  {
    "modality": "proprioception"|"vision"|"nociception"|"olfaction"|...,
    "component": "<component_name>",
    "sensor": "<sensor_name>",
    "value": <reading>,
    "intensity": <0-1 if pain/nociception>,
    "description": "brief text description",
    "context": {...}
  }

Only emit readings that change meaningfully or that represent ongoing
conditions worth reporting. Do not generate background noise.
```

---

## Integration With Existing Systems

### PerceptSource

`EmbodimentPerceptSource` implements the `PerceptSource` protocol:
- `capabilities` comes from the union of sensor modalities across the spec
- `next_percept()` dequeues from the generated percept buffer
- `is_exhausted()` returns True when embodiment dies (all critical failures active)

### PainBus

Every `nociception` percept fires a `PainSignal` through `PainBus` before being queued. This means existing subscribers (memory, fear, movement avoidance) see pain from imagined bodies identically to real ones.

### Energy System

Affordances have `cost` fields. Executing an affordance emits `EnergySignal` instances per cost domain. A fatiguing body consumes more energy over time because the LLM reasons about inefficiency.

### Hippocampus / NAc

Already captures pain events and learns causal links. No changes needed — imagined pain IS episodic memory.

### SimulationAdapter

Extend to accept an `Embodiment` instance. When the agent emits an action, the sim adapter:
1. Checks whether any component offers a matching affordance
2. Calls `embodiment.apply_affordance()` 
3. Feeds the outcome back into the percept stream

---

## Phases

### Phase 1 — Minimal Viable Embodiment (~400 LOC)

Goal: a single-component body with LLM-generated percepts reaching hippocampus.

- `EmbodimentSpec`, `ComponentSpec`, `SensorSpec`, `AffordanceSpec` dataclasses in `src/maxim/embodiment/spec.py`
- `Embodiment` runtime class with tick/apply_affordance in `src/maxim/embodiment/body.py`
- `LLMBackend` for percept generation (reuses existing LLMRouter)
- `EmbodimentPerceptSource` implementing the `PerceptSource` protocol
- One demo spec: `robot_arm_single_joint.yaml`
- Tests: tick advances state, affordance updates component, LLM returns valid percepts

### Phase 2 — Compositionality + Vital Metrics (~300 LOC)

Goal: nested components, vital metrics, failure modes.

- Component tree traversal (`get_active_sensors`, `find_component`, `walk_tree`)
- `VitalMetric` drift/recovery logic with homeostasis
- `FailureMode` trigger evaluation using safe expression language (asteval or similar)
- Three demo specs: `robot_arm_6dof.yaml`, `humanoid_upper_body.yaml`, `drone_quadrotor.yaml`
- Parity tests: same action sequence on same spec produces stable pain patterns

### Phase 3 — User Embodiment + Decompositional Construction (~350 LOC)

Goal: LLM can build embodiments, user becomes an embodied entity in simulations.

- Default `user_default.yaml` embodiment spec (voice, attention, presence, patience, engagement)
- `ConstructEmbodimentTool` — top-level: description → root spec + decomposition queue
- `ConstructComponentTool` — single component generation with sensors/affordances/vitals
- `DecomposeComponentTool` — refine existing component by adding children
- `ValidateEmbodimentTool` — checks physical plausibility, circular refs, missing pieces
- `VisualizeEmbodimentTool` — ASCII/markdown tree for user review
- Orchestrator integration: wire these tools into the sim orchestrator toolset
- User embodiment updates: orchestrator updates user vital metrics based on interaction patterns (curt user → decrement patience)
- Tests: construct tools produce valid specs, user patience decrements correctly

### Phase 4 — Hybrid and Multi-Backend (~300 LOC)

Goal: components can be real, imagined, rule-based, or replayed in the same embodiment.

- `Backend` protocol in `src/maxim/embodiment/backends/base.py`
- `HardwareBackend` wrapping existing `RobotController` (refactor, not rewrite)
- `RuleBackend` (simple physics: mass, friction, joint limits, integration step)
- `ReplayBackend` (reads from recorded session jsonl)
- `LLMBackend` (refined from Phase 1)
- Backend dispatch in `Embodiment.tick()` — per-component routing
- Demo: real Reachy Mini joints + imagined skin + imagined fatigue on same body

### Phase 5 — Architectural Primacy (Refactor) (~400 LOC)

Goal: Embodiment layer becomes canonical; hardware/sim code paths converge.

- Refactor `RobotController` implementations to implement `HardwareBackend`
- Migrate `RobotState.current_pose` → Component state reads
- Refactor `MovementTracker` to subscribe to Embodiment state changes
- Collapse `SimulatedController` into `LLMBackend` + `RuleBackend`
- Existing `PerceptSource` implementations (vision, transcript) folded into embodiment components with sensor backends
- Update all call sites to go through `Embodiment.get_sensor_value()` / `Embodiment.apply_affordance()`
- Delete duplicated code; single canonical path for body state
- Regression tests: existing hardware integration tests pass unchanged

### Phase 6 — Bio-Inspired Embodiments (~250 LOC)

Goal: living bodies with interoception, homeostasis, multi-modal senses.

- New modalities: `olfaction`, `taste`, `audition`, `vestibular`, `interoception`
- Homeostasis: vital metrics pull toward resting values with drift
- Illness modeling: failure modes that propagate (elbow strain → shoulder compensation → shoulder strain)
- Demo spec: `human_adult.yaml` with 40+ components

### Phase 7 — Curriculum and Research (~200 LOC)

Goal: progressive embodiments for learning curricula.

- `EmbodimentCurriculum` — graduates an agent through increasingly complex bodies
- Research persona integration: orchestrator spawns embodiments with specific failure modes
- Cross-embodiment transfer analysis: does NAc learning in a drone transfer to an arm?

### Phase 8 — Agent Mesh Integration (~350 LOC)

Goal: embodiments participate in the mesh — advertised, delegated, federated.

- `EmbodimentCapability` added to `RuntimeCapabilities` broadcast
- Serialization: `EmbodimentSpec.to_dict/from_dict`, `Embodiment.snapshot()`, etc.
- `AffordanceInvocation` mesh message type + handler
- `PerceptBundle` streaming for delegated-action percepts
- Federated embodiment view (read-only aggregator across peers)
- Cross-agent NAc transfer gated by embodiment-spec similarity
- User-embodiment sharing across co-present Maxim agents
- Decompositional construction distributed across mesh peers
- Depends on: Agent Mesh Phases 1-3 (discovery, identity, capability advertisement)

---

## Example Spec — Human Upper Body

```yaml
name: human_adult_upper_body
description: Human upper torso with arms, head, and vital organs
environment_hints: Indoor, 22°C, standing at a kitchen counter
tick_rate_hz: 2.0

root_component:
  name: body
  kind: abstract
  children:
    - name: left_arm
      kind: limb
      children:
        - name: left_shoulder
          kind: joint
          sensors:
            - {name: angle, modality: proprioception, unit: deg, range: [0, 180]}
            - {name: strain, modality: nociception, range: [0, 1]}
          affordances:
            - {name: flex, params: {target_angle: float, duration: float}}
          vital_metrics:
            - {name: fatigue, initial: 0.0, drift_rate: 0.001, recovery_rate: 0.002}
          failure_modes:
            - {name: overextension, trigger: "angle > 175 or angle < 5", pain_intensity: 0.8}
            - {name: exhaustion, trigger: "fatigue > 0.85", pain_intensity: 0.4, persistent: true}
        - name: left_elbow
          kind: joint
          # ... similar structure
    - name: head
      kind: abstract
      children:
        - name: left_eye
          kind: sensor
          sensors:
            - {name: vision, modality: vision, unit: detections}
          affordances:
            - {name: saccade, params: {target_xy: tuple}}
        - name: nose
          kind: sensor
          sensors:
            - {name: odor, modality: olfaction, unit: label}
        - name: mouth
          kind: sensor
          sensors:
            - {name: taste, modality: taste}
          affordances:
            - {name: speak, params: {text: str}}
            - {name: swallow}
    - name: torso
      kind: abstract
      children:
        - name: heart
          kind: organ
          sensors:
            - {name: pulse, modality: interoception, unit: bpm, range: [40, 200]}
          vital_metrics:
            - {name: rate, initial: 72.0, resting_value: 72.0, recovery_rate: 0.5}
          failure_modes:
            - {name: tachycardia, trigger: "rate > 140", pain_intensity: 0.6}
        - name: stomach
          kind: organ
          sensors:
            - {name: hunger, modality: interoception, range: [0, 1]}
          vital_metrics:
            - {name: fullness, initial: 0.5, drift_rate: -0.0005}
```

---

## Research Questions

These are things this plan makes testable:

1. **Does imagined pain produce the same avoidance learning as real pain?**
   Measure: NAc causal link strength after N identical action→pain episodes in imagined vs. replay-backed embodiments.

2. **Does the same cognitive architecture generalize across bodies?**
   Measure: train an agent in a robot_arm_6dof, test in drone_quadrotor without retraining. Does it learn faster than naive?

3. **Can the LLM model failure-mode cascades?**
   Measure: given a spec with interdependent failure modes (elbow→shoulder compensation), does the LLM correctly propagate strain?

4. **What's the minimum embodiment for curiosity-driven exploration?**
   Measure: at what component count does an agent's exploration pattern become "body-aware" vs. purely cognitive?

5. **Is imagined embodiment good enough for pre-training before hardware transfer?**
   Measure: agent trained in imagined Reachy Mini → deployed to real Reachy Mini. Does NAc reward prediction hold up?

---

## Integration With Existing Plans

| Plan | Integration |
|------|-------------|
| **Simulation Agent** | Orchestrator can inject an embodiment spec; AUT percepts come from the embodiment; user has their own embodiment |
| **Docker Sandbox** | Embodiment runs inside container; real filesystem stays isolated |
| **Research Protocol** | Researchers spawn embodiments with specific failure modes as experimental conditions |
| **Multi-LLM Scaling** | Different LLMs can back different components (fast model for proprioception, strong model for vision) |
| **Realtime Refinement** | Embodiment specs become reproducible experimental artifacts |
| **Agent Mesh** | Embodiments are sovereign state; agents advertise their embodiment capabilities; mesh supports embodiment sharing, delegation, and federated bodies (see below) |

---

## Agent Mesh Integration

The Agent Mesh and Embodiment layer compose in powerful ways. Both share the principle of *sovereign ownership with cooperative sharing* — each agent owns its body, but can share components, delegate to peers' bodies, or even form federated embodiments spanning multiple agents.

### 1. Embodiment as Advertised Capability

`RuntimeCapabilities` already exists for capability advertisement. Extend it with embodiment metadata:

```python
@dataclass
class EmbodimentCapability:
    embodiment_name: str              # "reachy_mini", "user_default"
    modalities: set[str]              # {"vision", "audition", "proprioception"}
    affordances: set[str]             # {"speak", "grasp", "look_at"}
    component_summary: str            # compact tree for peer discovery
    is_hardware_backed: bool          # real robot vs. fully imagined
    trust_level: float                # how much peers should discount
```

Maxim-A broadcasts: "I have a real Reachy Mini with vision, audio, head pose, and gripper." Maxim-B sees this and knows it can delegate physical-world tasks to A.

### 2. Cross-Agent Delegation via Affordances

Delegation in the mesh currently means "do this goal." With embodiments, delegation becomes richer:

```
Maxim-B (laptop, no body) wants: "hand me the cup"
    ↓
Mesh query: which peer has {affordance: grasp, modality: vision}?
    ↓
Maxim-A (Reachy Mini) advertises matching affordances
    ↓
Maxim-B delegates: "invoke affordance grasp(target=cup) on your body"
    ↓
Maxim-A evaluates via local AdaptivePlanner, decides to accept
    ↓
Maxim-A executes, streams percepts back (vision, proprioception, pain if any)
    ↓
Maxim-B receives percepts as though they were its own (with transfer discount)
```

This generalizes: an agent can borrow another's body for a task. The sovereign agent always decides whether to accept, applies safety review, and can refuse.

### 3. Federated Embodiments

Multiple agents can contribute components to one logical body:

```
Federated Embodiment "kitchen_assistant_v1":
├── arm         → Maxim-A (physical Reachy arm)
├── cameras     → Maxim-B (wall-mounted cams)
├── voice       → Maxim-C (speaker in room)
└── knowledge   → Maxim-D (recipe database agent)
```

Each agent owns its components. The federated body is a view — coordination happens through the mesh. If A goes offline, the arm component becomes unavailable, but voice and vision keep working. Naturally fault-tolerant.

This unlocks: distributed robotics without a central controller, shared ambient intelligence, multi-site embodied presence.

### 4. Embodiment Sharing as Knowledge Transfer

An agent that learned motor control on one embodiment can share its learned policies with another agent inheriting a similar spec:

```
Maxim-A trained on robot_arm_6dof for 1000 episodes
    ↓
NAc has causal links: (flex_elbow, fast) → pain=0.7
    ↓
Shared with Maxim-B who just inherited robot_arm_6dof
    ↓
Maxim-B imports causal links with transfer discount (confidence *= 0.6)
    ↓
Maxim-B starts with prior knowledge: "fast elbow flex likely hurts"
```

This uses the existing `CausalLink` serialization. The embodiment spec becomes the "compatibility key" — links are only transferable when specs match (or are similar, with a similarity discount).

### 5. User Embodiment in Multi-Agent Settings

When multiple Maxim agents share a user, they share the user's embodiment spec. All agents see the same patience/engagement/frustration metrics. A user saying "stop" to Maxim-A updates the shared user embodiment; Maxim-B sees the frustration signal and adjusts its own behavior.

This is a form of theory-of-mind: agents maintain a joint model of the user's state.

### 6. Decompositional Construction Across the Mesh

The decompositional construction tools can distribute work across the mesh:

```
User asks Maxim-A: "simulate a fire department HQ with 5 responders"
    ↓
Maxim-A's construct_embodiment tool emits root spec + 5 responder stubs
    ↓
Sub-tasks delegated across mesh:
    ├── Maxim-A: construct_component("dispatch_room")
    ├── Maxim-B: construct_component("responder_1")
    ├── Maxim-C: construct_component("responder_2")
    └── ...
    ↓
Fragments returned, assembled, validated
```

Each peer uses its local LLM (possibly different models), returning a fragment. This leverages the Multi-LLM Scaling plan: parallel construction at the cost of one wall-clock LLM call per branch instead of serializing them.

### Serialization Needs (New for Mesh)

These need `to_dict()` / `from_dict()` for embodiment to ride the mesh:

- `EmbodimentSpec` and all sub-specs (ComponentSpec, SensorSpec, etc.)
- `Embodiment` runtime state (component_states, vital_metrics, active_failures)
- `EmbodimentCapability` for capability broadcast
- `AffordanceInvocation` for delegated actions
- `PerceptBundle` for streaming sensor readings back from a delegated action

All straightforward — these are already dataclasses of primitives.

### Mesh-Native Safety Considerations

- **Sovereign refusal**: an agent can always refuse to invoke an affordance on its body. No peer can force physical action.
- **FearGate applies to borrowed bodies**: when Maxim-B delegates `grasp` to Maxim-A, A's FearAgent reviews the action as if it originated locally.
- **Embodiment isolation**: each agent's embodiment state is private by default. Broadcast only declared capabilities.
- **Pain provenance**: pain signals from a delegated action are tagged with origin ("on behalf of peer-B") so A's hippocampus captures them correctly.

---

## Open Questions

- **Persistence**: should embodiment state persist across sessions (aging, accumulated wear)? Or reset per simulation?
- **Grounding**: how do we prevent the LLM from hallucinating inconsistent physics? (stable prompt + validator layer?)
- **Cost**: LLM percept generation at 2Hz is expensive on cloud APIs. Cadence control per component modality? Batch multiple sensors per call?
- **Tool budget**: affordances could be exposed as real tools to the agent, or kept as an internal motor vocabulary. Which is better for learning?
- **Validation**: how do we assert imagined percepts are physically plausible? Rule backend as ground truth?

---

## No Blockers

This plan is self-contained. Depends on:
- Existing `PerceptSource` protocol ✓
- Existing `PainBus` routing ✓
- Existing `LLMRouter` for backend calls ✓
- Existing `EnergySignal` tracking ✓

Can be implemented at any time, incrementally phase-by-phase.
