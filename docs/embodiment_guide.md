# Embodiment System Guide

## What Is the SEM Protocol?

The Sensor-Entity-Modulator (SEM) protocol is Maxim's composable hardware/world abstraction. Every interactive thing — a robot joint, a camera, a sword, an NPC — is described as a triple:

- **Entity**: the thing (shoulder joint, wrist camera, rusty sword, ferryman)
- **Sensor**: reads state from the entity (angle, temperature, durability, trust)
- **Modulator**: changes state of the entity (rotate, restart, slash, threaten)

Entities compose into trees. A robot arm is an entity with child entities for each joint, each with their own sensors and modulators. Tools are auto-generated, pain triggers auto-fire, and the cognitive stack (Cerebellum, NAc, ATL) learns from the interactions.

## Quick Start

### 1. Define your body in YAML

```yaml
# scenarios/embodiment/my_robot.yaml
body:
  name: my_robot
  entity_type: robot
  children:
    - name: arm
      entity_type: joint
      sensors:
        angle: {unit: degrees, range: [0, 180], initial: 90}
      modulators:
        motor:
          affordances:
            rotate: {params: {degrees: float}, description: "Rotate arm"}
      failure_modes:
        - name: overextension
          trigger: {field: angle, op: ">", value: 170, pain: 0.7}
```

### 2. Load and run

```python
from maxim.embodiment.spec import load_spec
from maxim.embodiment.body import Embodiment
from maxim.embodiment.tool_bridge import generate_tools_for_entity
from maxim.tools.registry import ToolRegistry

# Load
spec = load_spec("scenarios/embodiment/my_robot.yaml")

# Generate tools
registry = ToolRegistry()
tools = generate_tools_for_entity(spec.root_entity, registry)
print(f"Generated {len(tools)} tools: {[t.name for t in tools]}")

# Create runtime
emb = Embodiment(spec.root_entity)

# Check body state
print(emb.format_body_state_for_prompt())

# Evaluate failures
events = emb.evaluate_failures()
print(f"Failures: {events}")
```

### 2.5 Running an agent with a SEM body (production path)

Once a component is registered in the bundled `_data/components/` tree (or under `~/.maxim/components/` for user-local components), you can give a live agent a body in one CLI flag:

```bash
# Validate the ref before spinning up the agent
maxim doctor --embodiment weapons/rusty_sword

# Run an agent that has rusty_sword as its body
maxim --llm mistral-7b --embodiment weapons/rusty_sword
```

What happens behind the scenes:

1. `cli.py` reads `--embodiment weapons/rusty_sword`.
2. `runtime/bootstrap.py::build_executor` (the canonical agent constructor — see [docs/plans/executor_bootstrap_unification.md](plans/executor_bootstrap_unification.md)) instantiates the entity via `ComponentRegistry`, wraps it in `Embodiment(pain_bus=...)`, and calls `generate_tools_for_entity` to register the affordance tools (`rusty_sword_slash`, `rusty_sword_parry`, `rusty_sword_throw`, `rusty_sword_sharpen`, `rusty_sword_repair`) into the agent's tool registry.
3. The agent's LLM sees those tool names alongside the standard tools and can invoke them by emitting `{"tool_name": "rusty_sword_slash", "params": {"target": "...", "force": 0.9}}`.
4. When the agent invokes `rusty_sword_slash` on a low-durability sword, `embodiment.evaluate_failures()` fires `shatter` → `PainBus.publish(PainSignal)` → the executor's `ToolPainBridge` calls `record_tool_embodiment_failure` → NAc forms a NEGATIVE causal link on `tool:rusty_sword_slash` → on the next turn, `nac.predict()` returns NEGATIVE for that tool, informing the agent's policy.

The full cascade is verified end-to-end in `tests/substrate/test_sem_execution_production.py::TestSEMProductionCascade` against the real bundled `weapons/rusty_sword.yaml` — no mocks in the chain.

#### Validation flow

If you're not sure whether a ref will resolve, ask `maxim doctor` first:

```bash
$ maxim doctor --embodiment weapons/nonexistent_sword
...
━━━ Embodiment ━━━
  ✗ Embodiment ref: Component ref 'weapons/nonexistent_sword' not found
    → Components in 'weapons':
    →   weapons/combat_knife
    →   weapons/enchanted_bow
    →   weapons/longbow
    →   weapons/magic_staff
    →   weapons/neural_disruptor
    →   weapons/plasma_rifle
    →   weapons/poison_dagger
    →   weapons/rusty_sword
    →   weapons/shock_baton
    →
    → Other available components:
    →   bodies/cybernetic_arm
    →   bodies/megarm_v3
    →   bodies/reachy_mini
    →   creatures/alien_xenomorph
    →   creatures/cyberdog
    →   ...and 35 more
```

The doctor output groups same-category matches first (so a typo in `weapons/X` surfaces other weapons before unrelated categories) and caps the per-category preview at 20 entries.

#### Constraints (current)

- `--embodiment` is currently **mutually exclusive with `--sim`**. Sim-mode SEM body wiring is tracked under [agent_factory_canonicalization.md](plans/agent_factory_canonicalization.md) Stage F1+ (the original Stage 2c of the now-archived [sem_execution_hook.md](plans/archive/sem_execution_hook.md) was structurally absorbed by [executor_bootstrap_unification.md](plans/executor_bootstrap_unification.md)). For DM-campaign YAMLs, set `component: <ref>` in the encounter spec instead — the DM runtime loads components per-scene via its own path.
- Only one entity can be loaded via the flag. Multi-entity bodies (e.g., a full robot arm with child entities) are loaded the old way via `Embodiment(spec.root_entity)` in code — see step 2 above.
- The bridge attaches to the unwrapped inner `Executor`. If you wrap the executor with `FearGatedExecutor` or similar, do it AFTER `build_executor` returns. This is structurally enforced by the `build_executor` signature contract — see [docs/plans/executor_bootstrap_unification.md](plans/executor_bootstrap_unification.md).

### 3. Add virtual entities for campaigns

```yaml
# scenarios/embodiment/dungeon.yaml
world_entities:
  - name: rusty_sword
    entity_type: weapon
    sensors:
      durability: {unit: ratio, range: [0, 1], initial: 0.3}
    modulators:
      combat:
        affordances:
          slash: {params: {target: str}, description: "Slash at target"}
    failure_modes:
      - name: shatter
        trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}
```

## Architecture

```
YAML Spec ──→ Entity Tree ──→ Auto-Generated Tools ──→ Agent
                  │                                       │
                  │                                       ↓
                  ├──→ Failure Triggers ──→ PainBus ──→ NAc (causal learning)
                  │                                       │
                  ├──→ ATL Body Concepts ──→ Grounded LLM predictions
                  │                                       │
                  └──→ Cerebellum (Phase 1a) ──→ Forward models
```

### Module Map

| File | Purpose |
|------|---------|
| `embodiment/sem.py` | Core protocols: Sensor, Modulator, Entity, FailureMode |
| `embodiment/spec.py` | YAML loader, SpecSensor/SpecModulator stubs |
| `embodiment/tool_bridge.py` | Auto-tool generation with collision detection |
| `embodiment/body.py` | Embodiment runtime (failure eval, vital drift, prompt state) |
| `embodiment/percepts.py` | EmbodimentPerceptSource (1Hz polling, demand mode) |
| `embodiment/llm_backend.py` | LLM/Narrative sensor and modulator backends |
| `embodiment/cerebellum.py` | Cerebellum forward models + motor program registry + engram formation/recall |
| `embodiment/motor.py` | MotorProgram, MotorStep, ProgramRegistry, entity_state_similarity |
| `embodiment/engrams.py` | MotorEngram, salience computation, formation decision logic |
| `embodiment/program_executor.py` | Step-by-step program runner with pain gates |
| `embodiment/backends/cerebellum_modulator.py` | CerebellumModulator (predict/fallback/train loop) |
| `embodiment/atl_integration.py` | Auto-register ATL body_part concepts |

### Scenario Files

| File | Purpose |
|------|---------|
| `scenarios/embodiment/robot_arm_3dof.yaml` | Demo robot arm (3 joints + camera) |
| `scenarios/embodiment/embodiment_baseline.yaml` | Regression test with bounds violations |
| `scenarios/embodiment/sword_npc_demo.yaml` | Virtual entities (sword + NPC) |

### Bundled Robot Templates

Maxim ships at least one full SEM template for a real robot — Reachy Mini, the desktop humanoid head from Pollen Robotics that started Maxim's embodiment journey:

| Template | Purpose |
|----------|---------|
| `bodies/reachy_mini` | Full SEM model of Reachy Mini: head pose, body yaw, antennas, camera/microphone health, battery, motor temperature, pose confidence, plus motion + expression + capture + lifecycle modulators. |

Load it via the registry:

```python
from maxim.embodiment.component_registry import ComponentRegistry

registry = ComponentRegistry()
reachy = registry.instantiate("bodies/reachy_mini", name="my_reachy")
```

The SEM template is **independent of the hardware connection** — you can run an agent against the SEM model in pure simulation, or wire it through to the actual `ReachyMiniController` for live hardware. Use the same shape as a starting point when modeling your own robot (Atlas, Spot, custom drone). See [Adding a New Robot](user/robot-setup.md#adding-a-new-robot) for the 3-step plugin pattern.

## Concepts

### Failure Modes

Six base modes: `overextension`, `overheating`, `strain`, `fatigue`, `impact`, `exhaustion`.

Custom failures compose from base modes:

```yaml
- name: tennis_elbow
  composes: [strain, fatigue]
  trigger:
    all:
      - {field: strain, op: ">", value: 0.6}
      - {field: fatigue, op: ">", value: 0.5}
```

### Pain-Proximity Warnings

When a sensor value approaches a failure threshold (within 20% of range), the body state prompt promotes it to IMPORTANT priority:

```
=== Body State (pain-relevant) ===
- robot_arm.shoulder.angle: 172° (WARN: overextension threshold at 175°, pain 0.8)
```

### Sensor Polling

Default: 1Hz. Pain-relevant sensors are promoted to every-tick. Configurable:

```python
# Via environment variable:
MAXIM_EMBODIMENT_POLL_HZ=5

# Via code:
source = EmbodimentPerceptSource(emb, poll_hz=5)

# Demand mode (during motor programs):
source.set_demand_mode(True, hz=30)
```

### Tool Name Collision Handling

Two entities with the same name get progressively prefixed:

1. `shoulder_rotate_angle` (first)
2. `robot2_shoulder_rotate_angle` (second — parent name prepended)
3. Full path if still colliding

### Virtual Entities (Beyond Robotics)

SEM works for any interactive entity. A sword, NPC, or door is just an Entity with sensors and modulators backed by `NarrativeModulator` instead of hardware. The cognitive stack (Cerebellum, NAc, engrams) learns from these interactions exactly as it learns from robot joints.

## Cerebellum (Phase 1a — Shipped)

The Cerebellum stores learned forward models: after observing that `rotate_angle(degrees=45)` on a shoulder consistently produces `angle=45.0`, it caches this prediction and skips the LLM entirely for future calls.

```python
from maxim.embodiment.cerebellum import Cerebellum
from maxim.embodiment.backends.cerebellum_modulator import cerebellum_modulator_factory

cb = Cerebellum()
factory = cerebellum_modulator_factory(cb, fallback_factory=llm_mod_factory)
attach_backends(root, modulator_factory=factory)
```

Key properties:
- **Rescorla-Wagner learning**: `expected += lr * (actual - expected)`
- **Confidence threshold**: below 0.3 → LLM fallback, above 0.3 → cached prediction
- **High-variance fallback**: uncertain models fall back to LLM
- **Per-key locks**: thread-safe concurrent predict/observe
- **Param bucketing**: similar params (within 10% of range) share a model
- **Persistence**: `data/embodiment/cerebellum.json`

## Motor Programs (Phase 1b — Shipped)

Motor programs are learned SEM action sequences. When the agent repeats the same sequence 3+ times for the same goal, the Cerebellum crystallizes it as a reusable program.

The `ProgramRegistry` indexes programs in three directions:
- **By goal**: "I want to reach forward" → matching programs
- **By entity**: "I'm holding a sword" → programs involving swords
- **By affordance**: "I want to slash" → programs with slash steps

```python
# Query by entity → get all programs for that entity
programs = cb.find_programs_for_entity("sword")

# Query by affordance → get all entities that can do it
programs = cb.find_programs_for_affordance("slash")

# Unified search
programs = cb.find_related_programs("attack")
```

### Motor Engrams

Engrams are ephemeral hippocampal memories linked to motor programs via the associative graph. They form on significant outcomes (pain, surprise, novelty) and decay after ~2 days unless reinforced.

- Cerebellum stores the **how** (motor program steps)
- Hippocampus stores the **when/where/what** (contextual episode)
- The engram links them so context modulates future motor execution

### Program Executor

Executes motor programs step by step with:
- **Pain gate checks** before each step (abort if sensor near threshold)
- **PainBus subscription** for mid-sequence interrupts
- **Gate tightening** after painful executions (10% per failure)

## What's Next

- **Phase 2**: Composable failure modes — persistent failures with recovery conditions
- **Phase 3**: Hardware adapter — wrap real robot SDKs as SEM backends

See [embodiment_core_plan.md](plans/archive/embodiment_core_plan.md) (archived) for the historical roadmap.

## Troubleshooting

See [troubleshooting/embodiment.md](troubleshooting/embodiment.md) for common issues.
