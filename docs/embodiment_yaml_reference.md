# Embodiment YAML Reference

This document describes the YAML format for defining SEM (Sensor-Entity-Modulator) body and world entity specifications.

## Overview

Embodiment YAML files define entity trees — composable descriptions of physical or virtual things that Maxim can interact with. Each entity has sensors (readable state), modulators (executable actions), and failure modes (pain triggers).

YAML files are loaded by `maxim.embodiment.spec.load_spec()` and produce live Entity trees with auto-generated agent tools.

## Top-Level Keys

```yaml
# For robotic bodies:
body:
  name: robot_arm
  entity_type: arm
  children: [...]

# For virtual entities (campaigns, simulations):
world_entities:
  - name: rusty_sword
    entity_type: weapon
    ...
  - name: ferryman
    entity_type: npc
    ...

# Optional metadata:
name: "my_scenario"             # Spec display name (defaults to body name)
description: "..."              # Human-readable description
test_sequence: [...]            # Test/validation sequence (see below)
expectations: [...]             # Metric expectations for validation
```

A file can have both `body` and `world_entities` — the body is the agent's own entity tree, and world_entities are interactive objects/NPCs in the environment.

## Entity Definition

```yaml
name: shoulder                  # Required. Unique within parent.
entity_type: joint              # Required. Semantic type hint.

# Optional:
children: [...]                 # Nested child entities
sensors: {...}                  # Sensor definitions
modulators: {...}               # Modulator definitions
failure_modes: [...]            # Failure trigger definitions
```

### Entity Types (Conventions)

These are semantic hints — the system doesn't enforce specific types.

| Type | Usage |
|------|-------|
| `arm`, `leg`, `body` | Robot body segments |
| `joint` | Articulated joint (has angle sensor + motor) |
| `camera` | Vision sensor (has frame sensor + lifecycle) |
| `wheel` | Mobile base wheel (has speed sensor + drive) |
| `gripper` | End effector (has force sensor + grip) |
| `weapon` | Campaign item (has durability, sharpness) |
| `npc` | Non-player character (has trust, mood, health) |
| `environment` | World element (has state sensors) |

## Sensor Definition

```yaml
sensors:
  angle:                        # Sensor name (unique within entity)
    unit: degrees               # Human-readable unit string
    range: [-180, 180]          # Scalar range [min, max] — implies type: float
    initial: 0                  # Optional initial value (defaults to midpoint)

  frame:
    unit: rgb_frame
    shape: [480, 640, 3]        # NDArray shape — implies type: ndarray
    dtype: uint8                # Optional dtype (default: float32)

  durability:
    unit: ratio
    range: [0, 1]
    initial: 0.3
```

### Sensor Schema Fields

| Field | Required | Description |
|-------|----------|-------------|
| `unit` | Yes | Human-readable unit (degrees, celsius, ratio, rgb_frame, kg) |
| `range` | No* | `[min, max]` for scalar sensors. Sets `type: float` |
| `shape` | No* | `[h, w, c]` for array sensors. Sets `type: ndarray` |
| `dtype` | No | Array dtype (uint8, float32). Only with `shape` |
| `initial` | No | Initial value. Defaults to midpoint of range, or 0 |
| `type` | No | Explicit type (float, int). Usually inferred from range/shape |

\* At least one of `range` or `shape` should be provided. If neither is given, the sensor defaults to `type: float`.

### Scalar vs Non-Scalar Sensors

Sensors with `range` are **scalar** — they participate in:
- Pain-proximity warnings in prompts
- Entity state similarity for engram matching (Phase 1b)
- Failure trigger evaluation
- Vital metric tracking

Sensors with `shape` are **non-scalar** (frames, audio) — they are:
- Excluded from similarity calculations
- Not used in failure triggers
- Available via `read_` tools but not in `sense_` bulk reads of scalar-only mode

## Modulator Definition

```yaml
modulators:
  motor:                        # Modulator name (unique within entity)
    affordances:
      rotate_angle:             # Affordance name
        params:                 # Parameter schema (same format as Tool.input_schema)
          degrees: float        # Required float parameter
          speed: float          # Required float parameter
        description: "Rotate the joint to target angle"
        timeout: 30.0           # Optional execution timeout (default: 30s)

      brake:
        params: {}              # No parameters
        description: "Engage the brake"
```

### Affordance Parameter Types

```yaml
# Required parameter:
degrees: float                  # Also: int, str, bool

# Optional parameter with default:
speed:
  type: float
  default: 1.0

# String parameter:
target: str
message: str
```

Supported types: `float`, `int`, `str`, `bool`.

## Failure Mode Definition

```yaml
failure_modes:
  # Simple trigger:
  - name: overextension         # Failure name
    trigger:
      field: angle              # Sensor name to check
      op: ">"                   # Comparison: >, <, >=, <=, ==
      value: 175                # Threshold value
      pain: 0.8                 # Pain intensity when triggered (0-1)

  # Compound trigger (ALL conditions must be true):
  - name: tennis_elbow
    composes: [strain, fatigue] # Base modes this composes from
    trigger:
      all:
        - {field: strain, op: ">", value: 0.6}
        - {field: fatigue, op: ">", value: 0.5}
    pain_intensity: 0.5

  # Persistent failure (stays active until recovery):
  - name: overheating
    trigger: {field: temperature, op: ">", value: 70, pain: 0.6}
    persistent: true
    recovery_condition:
      field: temperature
      op: "<"
      value: 40
```

### Failure Mode Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Failure mode name |
| `trigger` | Yes | Single trigger `{field, op, value, pain}` or compound `{all: [...]}` |
| `composes` | No | List of base modes this composes from |
| `pain_intensity` | No | Override pain level (defaults to trigger's pain) |
| `persistent` | No | If true, stays active until recovery condition met |
| `recovery_condition` | No | Trigger that clears a persistent failure |

### Base Mode Vocabulary (Fixed)

The six base failure modes:

1. **overextension** — joint/sensor beyond safe range
2. **overheating** — thermal limit exceeded
3. **strain** — sustained force near limit
4. **fatigue** — accumulated wear over time
5. **impact** — sudden force/collision
6. **exhaustion** — energy/resource depletion

Custom failure modes should compose from these base modes using the `composes` field. New base modes require a plan discussion.

### Comparison Operators

| Op | Meaning |
|----|---------|
| `>` | Greater than |
| `<` | Less than |
| `>=` | Greater than or equal |
| `<=` | Less than or equal |
| `==` | Equal to |

## Drive Spec (Optional)

Sensors can carry a `drive:` block that wires them into the homeostatic/entropic pain system. Drives cause pain automatically as sensor values drift — no explicit failure mode needed.

```yaml
sensors:
  hunger:
    unit: ratio
    range: [0, 1]
    initial: 0.0
    drive:
      drift_mode: entropic          # "entropic" or "homeostatic"
      drift_direction: up           # "up" (toward 1) or "down" (toward 0)
      drift_rate: 0.006             # per-second drift rate
      deprivation_threshold: 0.7    # pain fires beyond this
      deprivation_pain: 0.3         # pain intensity at deprivation
      satisfaction_threshold: 0.3  # positive reaction fires when crossing back

  core_temperature:
    unit: celsius_norm
    range: [-1, 1]
    initial: 0.0
    drive:
      drift_mode: homeostatic       # self-regulates toward set_point
      set_point: 0.0                # equilibrium target
      drift_rate: 0.001             # homeostatic pull rate per second
      comfort_band: 0.25            # no pain within ±band of set_point
      pain_scale: 1.5               # intensity per unit outside comfort band
      pain_model: linear            # "linear" (v1.0); future: "exponential", "asymmetric"
```

### Entropic Drive Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `drift_mode` | Yes | — | Must be `"entropic"` |
| `drift_direction` | No | `"up"` | `"up"` (toward 1.0) or `"down"` (toward 0.0) |
| `drift_rate` | No | `0.001` | Per-second drift magnitude |
| `deprivation_threshold` | No | `0.7` | PainSignal fires when value crosses this |
| `deprivation_pain` | No | `0.3` | Pain intensity at deprivation |
| `satisfaction_threshold` | No | `0.3` | Positive reaction fires when crossing back |

### Homeostatic Drive Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `drift_mode` | Yes | — | Must be `"homeostatic"` |
| `set_point` | No | `0.0` | Body's equilibrium target |
| `drift_rate` | No | `0.001` | Homeostatic pull rate per second |
| `comfort_band` | No | `0.0` | No pain within ±band of set_point |
| `pain_scale` | No | `0.5` | Pain intensity per unit outside comfort band |
| `pain_model` | No | `"linear"` | Pain formula (`"linear"` only in v1.0) |

---

## Test Sequence (Optional)

For validation scenarios:

```yaml
test_sequence:
  - action:
      entity: shoulder
      affordance: rotate_angle
      params: {degrees: 175, speed: 1.0}
    expect:
      pain: true
      min_intensity: 0.5
      description: "Should trigger overextension pain"
    repeat: 3                   # Repeat this action N times
```

## Expectations (Optional)

For benchmark/regression testing:

```yaml
expectations:
  - type: nac_convergence
    event_type: "embodiment_action"
    event_signature: "shoulder.rotate_angle.175"
    min_confidence: 0.5
    max_repetitions: 3

  - type: response_latency_ms
    p95_max: 2000
```

## Auto-Generated Tools

When a YAML spec is loaded, the following tools are automatically created:

| Pattern | Tool Type | Example |
|---------|-----------|---------|
| `sense_{entity}` | Read all sensors | `sense_shoulder` |
| `read_{entity}_{sensor}` | Read one sensor | `read_shoulder_angle` |
| `{entity}_{affordance}` | Execute affordance | `shoulder_rotate_angle` |

### Name Collision Handling

If two entities have the same name (e.g., two robots with `shoulder`), names are progressively prefixed:

1. `shoulder_rotate_angle` (first robot)
2. `robot2_shoulder_rotate_angle` (second robot — parent name prepended)
3. Full path if still colliding

## Complete Example

```yaml
name: dungeon_encounter
description: "A combat encounter with sword and NPC"

body:
  name: hero
  entity_type: character
  children:
    - name: right_hand
      entity_type: appendage
      sensors:
        grip_strength: {unit: ratio, range: [0, 1], initial: 1.0}
        fatigue: {unit: ratio, range: [0, 1], initial: 0.0}
      modulators:
        combat:
          affordances:
            punch: {params: {force: float}, description: "Throw a punch"}

world_entities:
  - name: iron_sword
    entity_type: weapon
    sensors:
      durability: {unit: ratio, range: [0, 1], initial: 0.8}
      sharpness: {unit: ratio, range: [0, 1], initial: 0.9}
    modulators:
      combat:
        affordances:
          slash: {params: {target: str}, description: "Slash at target"}
    failure_modes:
      - name: shatter
        trigger: {field: durability, op: "<", value: 0.05, pain: 0.7}
```
