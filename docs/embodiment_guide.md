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
2. `runtime/bootstrap.py::build_executor` (the canonical agent constructor — see [docs/plans/archive/executor_bootstrap_unification.md](plans/archive/executor_bootstrap_unification.md)) instantiates the entity via `ComponentRegistry`, wraps it in `Embodiment(pain_bus=...)`, and calls `generate_tools_for_entity` to register the affordance tools (`rusty_sword_slash`, `rusty_sword_parry`, `rusty_sword_throw`, `rusty_sword_sharpen`, `rusty_sword_repair`) into the agent's tool registry.
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

#### Sim mode support (0.6+)

`--embodiment` works with `--sim` (all modes: generative, DM, agent, interactive). The sim orchestrator threads `entity_ref` to the AUT's `build_executor`, which instantiates the entity, creates an `Embodiment`, and registers affordance tools. The full pain cascade (SEM affordance → embodiment_failures → ToolPainBridge → NAc) operates in sim mode. For DM-campaign YAMLs, you can also set `component: <ref>` in the encounter spec — the DM runtime loads components per-scene via its own path.

#### Scene-scoped tool management (0.7+)

In long campaigns where multiple entities appear and disappear, affordance tools are managed via **scene-scoped activation**. When the agent transitions to a new scene, the previous scene's tools are deactivated (hidden from the LLM prompt) but not deleted — if the agent returns to that scene, the tools are re-activated instantly. An active tool cap (default: 20 scene tools, core tools exempt) auto-evicts the oldest scene's tools when the prompt would overflow.

- Scene tools are registered via `registry.register_scene_tools(tools, scene_id="arena")`.
- Deactivated tools remain in the registry and can be re-activated with `registry.activate_scene("arena")`.
- The executor gates on active status — deactivated tools cannot execute, even if the LLM remembers them.

See [docs/user/tools.md](user/tools.md) for the full API.

#### Constraints

- Only one entity can be loaded via the flag. Multi-entity bodies (e.g., a full robot arm with child entities) are loaded the old way via `Embodiment(spec.root_entity)` in code — see step 2 above.
- The bridge attaches to the unwrapped inner `Executor`. If you wrap the executor with `FearGatedExecutor` or similar, do it AFTER `build_executor` returns. This is structurally enforced by the `build_executor` signature contract — see [docs/plans/archive/executor_bootstrap_unification.md](plans/archive/executor_bootstrap_unification.md).

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
| `embodiment/sem.py` | Core protocols: Sensor, Modulator, Entity, FailureMode, HomeostaticDriveSpec, EntropicDriveSpec |
| `embodiment/spec.py` | YAML loader, SpecSensor/SpecModulator stubs, attach_backends, _parse_drive_spec |
| `embodiment/tool_bridge.py` | Auto-tool generation with collision detection |
| `embodiment/body.py` | Embodiment runtime (failure eval, vital drift, prompt state) |
| `embodiment/percepts.py` | EmbodimentPerceptSource (1Hz polling, demand mode) |
| `embodiment/reflex.py` | Innate reflex system (percept-pattern → pain/reaction) |
| `embodiment/cerebellum.py` | Cerebellum forward models + motor program registry + engram formation/recall |
| `embodiment/motor.py` | MotorProgram, MotorStep, ProgramRegistry, entity_state_similarity |
| `embodiment/engrams.py` | MotorEngram, salience computation, formation decision logic |
| `embodiment/backends/cerebellum_modulator.py` | CerebellumModulator (predict/fallback/train loop) |
| `embodiment/component_registry.py` | ComponentRegistry — discover, load, instantiate entity templates |
| `embodiment/component_index.py` | ComponentIndex — two-layer semantic discovery (alias + embedding) |

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

### ComponentIndex — Semantic Discovery

The `ComponentIndex` (`embodiment/component_index.py`) bridges natural language queries to the exact-ref lookup required by `ComponentRegistry`. It enables discovery like `"old iron door"` → `environments/rusty_gate` without requiring exact naming.

**Two-layer architecture:**

| Layer | Mechanism | Latency | Example |
|-------|-----------|---------|---------|
| 1 — Alias table | O(1) hash lookup from `component.synonyms` | <1µs | `"healing draught"` → `items/healing_potion` |
| 2 — Semantic embedding | Cosine similarity (sentence-transformers) | ~5ms | `"sharp blade for combat"` → `weapons/combat_knife` (0.64) |

```python
from maxim.embodiment.component_registry import ComponentRegistry
from maxim.embodiment.component_index import ComponentIndex

registry = ComponentRegistry()
index = ComponentIndex(registry)

# Alias lookup (Layer 1)
match = index.find("old sword")  # → weapons/rusty_sword, score=1.0

# Semantic lookup (Layer 2)
match = index.find("sharp blade for combat")  # → weapons/combat_knife, score=0.64

# Top-k exploration
results = index.find_similar("hostile creature", k=5)

# Near-duplicate detection (for auto-curation)
dup = index.dedup_check(candidate_spec, threshold=0.80)
```

**Adding synonyms to components:** Add a `synonyms:` field to the `component:` header:

```yaml
component:
  name: rusty_gate
  tags: [environment, obstacle]
  synonyms: [iron gate, old gate, rusty door, decrepit entrance, old iron door]
  category: environments
```

The bundled component library ships with hand-authored synonyms on every component. Foundry-generated components get synonyms automatically via the EntityDesigner prompt.

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
# Via code (no env-var override exists):
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

SEM works for any interactive entity. A sword, NPC, or door is just an Entity with sensors and modulators backed by `SpecModulator` stubs (or a `CerebellumModulator` with LLM fallback) instead of hardware. The cognitive stack (Cerebellum, NAc, engrams) learns from these interactions exactly as it learns from robot joints.

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
- **Persistence**: `<persistence_dir>/cerebellum.json` (default: `~/.maxim/memory/cerebellum.json`)

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

## SEM Learning Loop (Phase 2 -- Shipped)

When a SEM entity interaction produces a reaction (pain on failure, satisfaction on confident prediction), the signal flows through the full bio-pipeline:

1. **CerebellumModulator** executes affordance -- emits failure reaction (NEGATIVE) or success reaction (POSITIVE)
2. **ReactionBus** dispatches to subscribers:
   - `hippocampus.capture_reaction` -- episode valence annotation
   - `nac.distribute_reward` -- EC threshold adjustment
3. **Episode close** -- `apply_hebbian_on_close` annotates edges with `metadata["valence"]`
4. **Pain spike** -- `salience_spike_rule` closes the episode boundary
5. **Future retrieval** -- `spreading_activation(propagate_valence=True)` carries affective memory

### Success reactions (negativity bias)

CerebellumModulator emits `_emit_success_reaction` when confident enough to skip LLM fallback. Intensity is lower than failure (0.1-0.3 vs 0.3-0.5) -- biologically motivated negativity bias.

### NAc reward distribution

`distribute_reward` credits eligible substrate nodes proportionally to eligibility traces. Positive rewards widen EC recognition (lower threshold); negative rewards clamp to 0 (bias never narrows).

### Cerebellum activation in production

`BioStack.cerebellum` is now constructed by `build_bio_stack` and forwarded via `build_executor(cerebellum=...)` to `generate_tools_for_entity`, which creates `CerebellumModulator` instances with a wired `reaction_bus`. This means every SEM affordance tool now has a live Cerebellum backing it -- predictions, training, and reaction emission all happen automatically.

### Behavioral convergence wiring (shipped 2026-04-17)

The SEM learning loop produces valence and reward bias in the substrate, but prior to behavioral convergence wiring, the LLM never saw this information. Four stages close the gap:

1. **Valence in PromptAssembler** -- `MemorySummary` includes valence annotations from `retrieve_on_cue(include_valence=True)`, so the LLM sees "this entity is associated with negative experiences."
2. **`observe_episode_event` in agent loop** -- the production agent loop now calls `hippocampus.observe_episode_event` on each tick, keeping the episode capture pipeline fed with real-time events.
3. **Energy→Reaction bridge** -- energy depletion fires interoceptive Reactions (hunger, fatigue, satiation) that enter the same learning loop as pain and success reactions.
4. **Food/water/poison SEM specs** -- bundled consumable entity specs for testing and demonstration.

Validated by Experiment 2 (13/13 hypotheses confirmed): food +0.753, water +0.135, poison -0.495.

## What's Next

- **Phase 2**: Composable failure modes -- persistent failures with recovery conditions
- **Phase 3**: Hardware adapter -- wrap real robot SDKs as SEM backends

See [embodiment_core_plan.md](plans/archive/embodiment_core_plan.md) (archived) for the historical roadmap.

## Troubleshooting

See [troubleshooting/embodiment.md](troubleshooting/embodiment.md) for common issues.
