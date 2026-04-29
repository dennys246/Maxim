# Cradle of Artificial Civilization — sensorimotor development through structured acts

**Status:** Refined plan (2026-04-26), pre-implementation
**Scope:** ~550-650 LOC net new (+ ~490 LOC dead code removal) across embodiment, simulation, energy, runtime, prompts
**Depends on:** B1 (bio-system protocol enrichment, shipped), B2 (SCN oscillator, shipped), reflex system (shipped), enrichment pipeline (shipped)
**Integrates:** [proprioceptive_discovery.md](proprioceptive_discovery.md) Mechanism B (entity acquisition)
**Gates:** None; research demonstration + sensation standardization PoC
**Branch:** TBD

---

## Vision

A newborn agent that learns from sensation, not language. No prompt engineering, no tool descriptions, no linguistic scaffolding — just a body with sensors, a world with stimuli, and the bio-pipeline learning what hurts, what helps, and what to pay attention to.

The existing bio-pipeline (PainBus → NAc → hippocampus → TemporalCreditDistributor → ValenceSignal → enrichment) is already the right learning substrate. The reflex system, SCN oscillator, and 3-path enrichment retrieval are all live. **The cradle's job is NOT to build new learning infrastructure — it's to create the minimal sensory input layer and structured scenario that demonstrates the shipped infrastructure producing genuine sensorimotor development.**

**The deeper goal:** The cradle is also a **standardization PoC** for how external stimuli translate into internal bio-system signals. Today, text-keyword reflexes are the only automated sensation path. The cradle establishes three canonical sensation layers — contact (entity acquisition), proximity (orchestrator sensor writes), and narrative fallback (keyword reflexes) — all converging on the same downstream pipeline: sensor change → `evaluate_failures()` → PainBus → NAc. This standardization is the foundation for real sensor integration (robot proprioception, theoretical sensors) long-term, where mechanical/chemical/electrical signals translate into the same internal state space that text signals do today.

**Bio-plausible framing:** Human infants progress through Piaget's sensorimotor stages — neonatal reflexes, primary circular reactions, secondary circular reactions, coordination, tertiary reactions, mental representation. Each stage builds on the prior stage's learned contingencies. We structure the cradle as a multi-act simulation where each act corresponds to a developmental stage, and the agent's behavior in later acts is shaped by what it learned in earlier ones.

**The research claim:** "A bio-inspired cognitive architecture demonstrates non-linguistic sensorimotor development across structured developmental stages, retaining learned avoidance and approach behaviors across sessions without fine-tuning."

---

## Architectural audit — what exists vs what's needed

### Already shipped (DO NOT rebuild)

| Capability | Location | How cradle uses it |
|---|---|---|
| Keyword reflexes → damage → pain → NAc | `reflex.py`, `humanoid.yaml` | Narrative fallback layer: text keywords trigger reflexes when no entity interaction occurred |
| Pain cascade to enrichment | `pain_bus.py` → `bio_enrichment.py` | Next session: "fire" keyword → NAc prediction "fire → negative" surfaces in prompt |
| SCN anticipatory credit | `temporal_credit.py`, `OscillatorNetwork` | Repeated fire/hunger events → oscillator predicts imminence → pre-activates NAc traces |
| 3-path hippocampal retrieval | `bio_enrichment.py::_query_hippocampus` | Graph (binding), goal-keyword, substring retrieval surfaces prior-session episodes |
| Vital metric drift | `body.py::tick_vital_drift` | Fatigue/durability degrade over time (hardcoded metric names) |
| Energy → Reaction bridge | `energy/reactions.py` | Threshold-crossing hunger/fatigue emits typed Reactions to ReactionBus |
| Percept factories | `agents/percept_factory.py` | `make_intero_percept()` creates body-state percepts |
| Narrative arcs + phases | `simulation/arcs.py` | `NarrativePhase` with turn budgets, narrator auto-advances |
| DM campaign acts | `simulation/dm_schema.py` | `Act` groups encounters with shared entity state |
| Component extends | `component_registry.py::deep_merge` | Child body templates inherit + override parent sensors/modulators |
| Latent affordances | `spec.py::LatentAffordance` | Reflex-triggered motor programs (dodge, block) surface contextually |
| Acting Coach | `prompts/acting_coach.py` | Bio-modulated affordance exploration (NAc caution, pain anticipation) |
| Entity reparenting | `sem.py::Entity.reparent()` | Entity tree manipulation for acquisition/release |
| ToolOutput.side_effects | `tools/base.py` | Typed channel for bio-pipeline signals from tool execution |
| Entity ownership (self/scene) | `entity_map.py` | EntityMap tracks which entities belong to agent vs scene |

### Gaps (what the cradle actually needs)

| Gap | Why it matters | Estimated LOC |
|---|---|---|
| **G1. Drive protocol (homeostatic + entropic)** | No general abstraction for drives. hunger/thirst/fatigue/temperature share patterns but each is ad-hoc. Need two spec types + interface-level coupling/modulation fields for 1.0 freeze. | ~30 LOC |
| **G2. Drive parsing + evaluation** | `tick_vital_drift` hardcodes metric names. Need spec-type dispatch, homeostatic pain formula. Wall-clock dt fix lives in `percepts.py::next_percept()` (not body.py). | ~50 LOC |
| **G3. Dead energy code removal** | `EnergyReactionBridge` has ZERO callers — never instantiated anywhere. `MovementEnergyTracker` also dead (362 LOC, never called). Hard-remove both, not deprecate. | -490 LOC |
| **G4. Drive → SCN TemporalEvent emission** | Drive state transitions should emit TemporalEvents so the oscillator learns drive rhythms across sessions. | ~15 LOC |
| **G5. Three-layer sensation model** | No standardized pathway from physical interaction → sensor change. Need: entity acquisition (contact), orchestrator sensor writes (proximity), keyword reflexes (narrative fallback). | ~110 LOC |
| **G6. self_effect for non-contact satisfaction** | Eating food should reduce hunger via declarative YAML write-back. | ~25 LOC |
| **G7. Drive prompt visibility** | `body_state_summary()` reads SENSORS only, not vital_metrics. Drives in vital_metrics are invisible to the LLM. Agent can't act on hunger if it can't feel it. | ~20 LOC |
| **G8. Acting Coach drive layer** | Acting Coach only has pain anticipation. No drive modulation ("You're getting hungry"). Without this, the LLM has no guidance on drive-motivated behavior. | ~30 LOC |
| **G9. GatingContext drive fields** | `GatingContext` already carries energy/processing_load. Adding drive fields lets salience scoring modulate with homeostatic state (hungry agent finds food more salient). | ~10 LOC |
| **G10. Cradle body template** | Need `infant_humanoid.yaml` with thermal/pressure/texture sub-sensors + drives. | ~80 lines YAML |
| **G11. Cradle scenario arc** | Multi-act builtin arc with developmental stages as phases. | ~60 LOC |
| **G12. Environment stimulus objects** | Fire pit, food, sharp rock, blanket, lever, button as scene entities. | ~100 lines YAML |
| **G13. Validation harness** | Automated experiment measuring learned behavior change across acts and sessions. | ~80 LOC |

**Total: ~550-650 LOC net new + ~180 lines YAML - ~490 LOC dead code removed.** The codebase gets SMALLER while gaining the drive protocol, sensation model, and prompt visibility. Includes Mechanism B from proprioceptive_discovery.md (~85 LOC) for the contact sensation layer.

---

## Design

### D1. Three-Layer Sensation Model — standardized external→internal translation

The core architectural contribution of the cradle. Three layers, all converging on the same downstream pipeline:

```
Layer 1 (Contact):     pick_up(rock) → entity_acquired → rock sensors join body → evaluate_failures() → pain
Layer 2 (Proximity):   orchestrator writes thermal=0.8 on arms → evaluate_failures() → pain
Layer 3 (Narrative):   narrator says "flames engulf" → keyword reflex → damage_component → pain
                                                                                          ↓
                                                          PainBus → NAc → hippocampus → enrichment → SCN
```

**All three layers produce the same downstream signal:** a sensor value change that `evaluate_failures()` evaluates against failure mode thresholds. The difference is only in HOW the sensor value gets set. This is the standardization — whether the stimulus comes from a grasped object's sensors, an orchestrator proximity write, or a keyword-triggered reflex, the bio-pipeline sees the same thing: a sensor crossed a threshold.

**Why this matters long-term:** Real sensors (robot proprioception, cameras, microphones) will feed this same pipeline. A robot's temperature sensor writes to `arms.thermal` exactly as the orchestrator does today. A theoretical pain receptor writes to `pressure` exactly as an acquired entity's sensor does. The three-layer model is the simulation-side prototype of the real-sensor integration path.

**Layer 1: Contact sensation via entity acquisition (Mechanism B)**

Adapted from [proprioceptive_discovery.md](proprioceptive_discovery.md) Mechanism B. When the infant grasps an object, that object temporarily becomes part of the agent's body. Its sensors contribute to the damage model.

```
Agent calls pick_up(object="sharp_rock")
  → executor sees entity_acquired in side_effects
  → rock entity reparents to agent body (Entity.reparent)
  → rock's sharpness sensor (0.8) now in agent's sensor space
  → evaluate_failures() detects sharpness > 0.5 → laceration failure
  → PainSignal → NAc negative link on "sharp_rock" + "pick_up"

Agent calls drop(object="sharp_rock")
  → executor sees entity_released in side_effects
  → rock reparents back to scene → sensors detach
```

This is the cleanest sensation path — the pain comes through the standard sensor evaluation pipeline, not a special mechanism. The rock's sharpness IS a sensor reading that the body evaluates. No keyword detection, no orchestrator intervention, no special write-back.

**Implementation (~85 LOC, from Mechanism B):**

| File | Change | LOC |
|---|---|---|
| `runtime/executor.py` | Handle `entity_acquired` / `entity_released` side_effects | ~40 |
| `embodiment/spec.py` | Parse `acquirable`, `on_acquire` from component spec | ~10 |
| `embodiment/tool_bridge.py` | `pick_up` returns `entity_acquired` side_effect | ~20 |
| `embodiment/entity_map.py` | `transfer_ownership(entity, from_scene=True)` | ~15 |

**Layer 2: Proximity sensation via orchestrator sensor writes**

The orchestrator IS the external world. When the agent approaches a fire pit, the orchestrator writes the thermal change directly to the agent's body sensors:

```
Orchestrator detects: agent moved toward fire_pit
  → orchestrator calls set_entity_sensor(entity="infant_humanoid", sensor="arms.thermal", value=0.8)
  → arms.thermal crosses failure threshold → burn failure mode
  → PainSignal → NAc negative link on "fire_pit"
```

This is already possible via `SetEntitySensorTool` — the orchestrator has this tool. The cradle's phase instructions tell the orchestrator to use it when describing proximity to environmental hazards. No new code needed for the mechanism itself.

**What IS needed (~25 LOC):** Orchestrator prompt guidance in the cradle arc instructions. Each phase instruction that involves environmental stimuli tells the orchestrator to write sensor values, not just describe them narratively. Example: "When the infant approaches the fire, use set_entity_sensor to increase arms.thermal toward 0.8. The body's failure evaluation will handle the pain response."

**Layer 3: Narrative fallback via keyword reflexes**

The existing reflex system (`humanoid.yaml`) fires when percept text contains keywords like "fire", "flame", "burn". This is the FALLBACK for unstructured narrative where neither entity acquisition nor orchestrator sensor writes occurred.

**When does Layer 3 fire instead of Layers 1-2?** When the narrator describes a sensation that wasn't preceded by an entity interaction or orchestrator write — e.g., "a sudden gust of scorching wind sweeps through the room." The narrator is reactive; it doesn't always know to trigger sensor writes. Keyword reflexes catch these cases.

**Priority and overlap prevention:** A single stimulus should produce ONE pain signal, not three. The layering is:
- If entity acquisition handled it (Layer 1), the sensor change fires `evaluate_failures()`. The reflex system sees no new keywords because the executor, not the narrator, drove the interaction.
- If the orchestrator wrote sensors (Layer 2), same — `evaluate_failures()` fires from the sensor change.
- If neither happened but the narrator described the stimulus (Layer 3), the keyword reflex fires and calls `damage_component` which THEN changes sensors and fires `evaluate_failures()`.

The layers don't conflict because they operate at different points in the turn cycle: Layer 1 fires during tool execution, Layer 2 fires during orchestrator turn, Layer 3 fires during bio-enrichment of the narrator's percept.

### D2. Drive Protocol — homeostatic vs entropic drives

**Pushback on the shell plan:** The shell plan hardcoded hunger/thirst/core_temperature as YAML stanzas with custom fields. This is a one-off schema that doesn't compose. Instead, define a `Drive` protocol with two biologically-grounded drift modes.

**The key insight:** Biological drives fall into two categories:
- **Homeostatic** — the body self-regulates toward a set point. Temperature, pressure, pain-from-contact all have set points the body recovers toward when the stimulus is removed. Discomfort is proportional to distance from set point.
- **Entropic** — the state drifts away from equilibrium over time and requires external action to reset. Hunger, thirst, fatigue all increase inexorably; only eating, drinking, or resting reverses them.

This distinction solves two problems at once: (1) sensor values naturally reset after stimuli are removed (homeostatic drift pulls `arms.thermal` back toward 0.0 after fire exposure), and (2) the blanket has a genuine positive signal (it brings temperature closer to set point, reducing homeostatic discomfort).

**Design:** Two drift modes in YAML:

```yaml
# In infant_humanoid.yaml
sensors:
  # ENTROPIC drives — drift away from equilibrium, require external reset
  hunger:
    unit: ratio
    range: [0, 1]
    initial: 0.0
    drive:
      drift_mode: entropic          # drifts AWAY from set_point
      drift_direction: up           # hunger increases over time
      drift_rate: 0.002             # per-second increase
      deprivation_threshold: 0.7    # PainSignal fires above this
      deprivation_pain: 0.3         # pain intensity when deprived
      satisfaction_threshold: 0.3   # positive Reaction fires when crossing below
  thirst:
    unit: ratio
    range: [0, 1]
    initial: 0.0
    drive:
      drift_mode: entropic
      drift_direction: up
      drift_rate: 0.003
      deprivation_threshold: 0.6
      deprivation_pain: 0.25
      satisfaction_threshold: 0.3
  fatigue:
    unit: ratio
    range: [0, 1]
    initial: 0.0
    drive:
      drift_mode: entropic
      drift_direction: up
      drift_rate: 0.001
      deprivation_threshold: 0.8
      deprivation_pain: 0.2
      satisfaction_threshold: 0.4

  # HOMEOSTATIC drives — body self-regulates toward set_point
  core_temperature:
    unit: celsius_norm              # -1=cold, 0=comfortable, +1=hot
    range: [-1, 1]
    initial: 0.0
    drive:
      drift_mode: homeostatic       # drifts TOWARD set_point
      set_point: 0.0               # body's target (defaults to initial)
      drift_rate: 0.001            # body's self-regulation rate per second
      comfort_band: 0.4            # no discomfort within ±0.4 of set_point
      pain_scale: 0.5              # discomfort intensity per unit outside comfort band
```

**Homeostatic dynamics — how environment and body compete:**

```
Sun entity continuously writes core_temperature += 0.003/s (orchestrator proximity write)
Body homeostasis continuously pulls core_temperature toward 0.0 at 0.001/s
Net: temperature rises at 0.002/s
After 200 seconds: temperature = 0.4 → exits comfort band → mild discomfort begins
After 400 seconds: temperature = 0.8 → pain_intensity = (0.8 - 0.4) * 0.5 = 0.2

Agent moves away from sun → environmental push stops
Body homeostasis pulls temperature back at 0.001/s
After 400 seconds: temperature = 0.4 → back inside comfort band → discomfort ends
```

**Blanket interaction — contextual valence from homeostasis:**

```
Agent is cold (core_temperature = -0.3, inside comfort band but drifting cold)
Agent picks up blanket (entity acquisition) → blanket thermal = +0.2
Net effect: temperature → -0.1, closer to 0.0 → LESS deviation → relief signal

Agent is warm (core_temperature = +0.3, near comfort band edge)
Agent picks up blanket → blanket thermal = +0.2
Net effect: temperature → +0.5, OUTSIDE comfort band → discomfort → negative signal
```

The blanket is contextually good or bad depending on the agent's current homeostatic state. The agent learns "blanket when cold = relief" and "blanket when hot = discomfort" through the same pipeline. No abstract "comfort" signal — just homeostatic deviation.

**Somatosensory sensors are homeostatic too:**

```yaml
# On arms modulator
sensors:
  thermal:
    unit: celsius_norm
    range: [-1, 1]
    initial: 0.0
    weight: 0.0                   # sensory only, no integrity contribution
    drive:
      drift_mode: homeostatic
      set_point: 0.0
      drift_rate: 0.0008          # local tissue recovery (slower than core)
      comfort_band: 0.5           # wider band than core (skin tolerates more)
      pain_scale: 0.4
  pressure:
    unit: ratio
    range: [0, 1]
    initial: 0.0
    weight: 0.0
    drive:
      drift_mode: homeostatic
      set_point: 0.0
      drift_rate: 0.002           # pressure dissipates quickly
      comfort_band: 0.6
      pain_scale: 0.3
```

After fire heats `arms.thermal` to 0.8, the body's homeostatic drift pulls it back toward 0.0 at 0.0008/s. No orchestrator reset needed — the body self-regulates. But if the agent stays near the fire (continuous orchestrator writes), homeostasis can't keep up and pain persists.

**Implementation — compartmentalized drive specs:**

Two frozen dataclasses in `embodiment/sem.py` (near `AffordanceSchema`), plus two interface-level coupling types that ship as frozen interfaces for 1.0 but are ignored by the cradle implementation:

```python
@dataclass(frozen=True, slots=True)
class CouplingSpec:
    """How one drive's state modulates another drive's drift rate.

    Example: hunger drifts 2x faster when stamina < 0.4.
    Ships as interface for 1.0 — implementation deferred post-cradle.
    """
    sensor: str          # source sensor name (e.g., "stamina")
    below: float         # threshold on source that activates coupling
    multiplier: float    # drift_rate multiplier when active (e.g., 2.0)

@dataclass(frozen=True, slots=True)
class ModulationSpec:
    """How an external system modulates a homeostatic drive's parameters.

    Example: SCN circadian signal adjusts core_temperature set_point ±0.1.
    Ships as interface for 1.0 — implementation deferred post-cradle.
    """
    source: str          # modulating system (e.g., "scn", "nac")
    target_field: str    # which drive field to modulate ("set_point", "drift_rate")
    range: tuple[float, float]  # modulation bounds (e.g., (-0.1, 0.1))

@dataclass(frozen=True, slots=True)
class HomeostaticDriveSpec:
    """Body self-regulates toward set_point. Discomfort proportional to deviation."""
    set_point: float                 # body's target (defaults to initial)
    drift_rate: float                # body's self-regulation rate per second
    comfort_band: float = 0.0       # no discomfort within ±band of set_point
    pain_scale: float = 0.5         # intensity per unit outside comfort band
    pain_model: str = "linear"      # "linear" (v1), future: "exponential", "asymmetric"
    modulated_by: tuple[ModulationSpec, ...] | None = None  # 1.0 interface, deferred impl

@dataclass(frozen=True, slots=True)
class EntropicDriveSpec:
    """Drifts away from equilibrium. Requires external action to reset."""
    drift_direction: str             # "up" or "down"
    drift_rate: float                # per-second drift rate
    deprivation_threshold: float     # PainSignal fires beyond this
    deprivation_pain: float          # pain intensity at deprivation
    satisfaction_threshold: float    # positive Reaction fires when crossing back
    coupled_to: tuple[CouplingSpec, ...] | None = None  # 1.0 interface, deferred impl
```

**YAML parsing:** The `drive:` key's `drift_mode` field selects the spec type. Parser validates that homeostatic-only fields aren't mixed with entropic-only fields. `coupled_to` and `modulated_by` are parsed from YAML but ignored by `tick_vital_drift()` in the cradle implementation — they're interface reservations for post-1.0.

```yaml
# Example: coupled_to (parsed, not yet evaluated)
hunger:
  drive:
    drift_mode: entropic
    drift_direction: up
    drift_rate: 0.002
    deprivation_threshold: 0.7
    deprivation_pain: 0.3
    satisfaction_threshold: 0.3
    coupled_to:
      - sensor: stamina
        below: 0.4
        multiplier: 2.0     # hunger rises 2x faster when exhausted
```

Parsed from `drive:` key in `spec.py::_parse_entity()`. `body.py::tick_vital_drift()` applies the correct drift per sensor: homeostatic drives move toward `set_point`, entropic drives move in `drift_direction`.

**Energy system cleanup — dead code removal + drive protocol unification:**

The architectural audit revealed that `EnergyReactionBridge` (125 LOC) and `MovementEnergyTracker` (362 LOC) are **completely dead code** — zero callers anywhere in the codebase. The bridge was never instantiated; the movement tracker was never wired to any movement source. Hard-remove both.

What STAYS:
- `LLMEnergyTracker` — LIVE, wired into `llm_worker.py` for token/cost tracking
- `EnergyRegistry` — LIVE, coordinates LLM tracking + budget gating + imagination energy gate
- `EnergyBudget` — LIVE, spending caps
- `EnergySignal` / `EnergyType` — LIVE (4 of 10 enum values used: LLM_TOKENS, LLM_LATENCY, LLM_COST, MOTOR_COMMAND)

What's REMOVED:
1. `energy/reactions.py` — entire file (EnergyReactionBridge, create_energy_reaction_bridge). Zero callers.
2. `energy/movement_tracker.py` — entire file (MovementEnergyTracker, MovementEnergyConfig). Zero callers.
3. Unused EnergyType enum values — COMPUTE_TIME, MOTOR_CURRENT, VISION_INFERENCE, AUDIO_PROCESSING, ATTENTION, MEMORY_ACCESS (6 dead values)

The drive protocol's generic threshold-crossing mechanism replaces what the bridge was SUPPOSED to do. Stamina-as-homeostatic-drive replaces what the movement tracker was SUPPOSED to do. No compat shim needed — there are no callers to compat.

**Additional bug fix:** `simulation/introspection.py:149` calls `registry.get_stats()` but the method is `registry.get_summary()`. Fix during cleanup.

**Wall-clock drift fix (G2):** The tick cycle lives in `embodiment/percepts.py::EmbodimentPerceptSource.next_percept()`, NOT in body.py. The dt is computed as `1.0 / poll_hz` (planned interval), not actual elapsed wall-clock time. Fix: `dt = now - self._last_poll` in `percepts.py` line 77, replacing `dt = interval`. This ensures drives advance proportional to real elapsed time, including LLM latency gaps. ~5 LOC in `percepts.py`.

**Pain evaluation for homeostatic drives:** In `evaluate_failures()`, homeostatic drives compute pain as `max(0, abs(current - set_point) - comfort_band) * pain_scale`. This replaces the threshold-based model for homeostatic sensors. Entropic drives keep the existing threshold model. Both produce PainSignals through the same PainBus pathway.

**Satisfaction for homeostatic drives:** When a homeostatic sensor that was outside comfort band returns to within it (e.g., temperature drops from 0.6 to 0.3 after moving away from fire), emit a positive Reaction. The agent learns "moving away from fire → temperature returns to comfortable → relief."

**Drive → SCN TemporalEvent emission (G4):** Both modes emit TemporalEvents on state transitions. Entropic: `TemporalEvent(event_type="drive:hunger:deprived")` and `"drive:hunger:satisfied"`. Homeostatic: `TemporalEvent(event_type="drive:core_temperature:discomfort")` and `"drive:core_temperature:relief"`. The SCN oscillator learns temporal patterns of both — drive rhythms are a cross-session metric (oscillator needs >= 3 observations, which accumulate across sessions, not within one 25-turn session).

**Extensibility:** New body types add drives by adding `drive:` metadata to any sensor. A robot arm's joint temperature is homeostatic. A vehicle's fuel level is entropic. No core code changes — just YAML.

**Why NOT a separate DriveSystem class:** Drives ARE sensors with drift + threshold behavior — both of which the embodiment runtime already handles. The homeostatic/entropic distinction is a property of the drift function, not a separate system.

### D3. Sensor composability — patches via extends, not a new registry

**Pushback on the shell plan:** The shell plan proposed a `SomatosensoryRegistry` class with `SomatosensoryPatch` dataclasses. This is redundant. The existing modulator sub-sensor system (`SpecModulator.vital_metrics` with `weight`, `range`, `initial`) IS the somatosensory system. A "thermal sensor on left_hand" is just a sub-sensor on the `arms` modulator:

```yaml
modulators:
  arms:
    sensors:
      thermal:
        unit: celsius_norm    # -1=cold, 0=neutral, +1=hot
        range: [-1, 1]
        initial: 0.0
        weight: 0.0           # doesn't contribute to integrity (sensory only)
        drive:                 # homeostatic — body self-regulates after stimulus
          drift_mode: homeostatic
          set_point: 0.0
          drift_rate: 0.0008
          comfort_band: 0.5
          pain_scale: 0.4
      pressure:
        unit: ratio
        range: [0, 1]
        initial: 0.0
        weight: 0.0
        drive:
          drift_mode: homeostatic
          set_point: 0.0
          drift_rate: 0.002
          comfort_band: 0.6
          pain_scale: 0.3
      grip_strength:
        unit: ratio
        range: [0, 1]
        initial: 1.0
        weight: 0.5
```

**Composability via extends:** New body types compose sensor sets by extending a base and deep-merging:

```yaml
# _data/components/bodies/infant_humanoid.yaml
component:
  name: infant_humanoid
  extends: bodies/base_humanoid
  category: bodies
  archetype: humanoid
  tags: [body, humanoid, infant, cradle]

entity:
  name: infant_humanoid
  sensors:
    hunger:
      drive: { drift_mode: entropic, drift_direction: up, drift_rate: 0.002, deprivation_threshold: 0.7, deprivation_pain: 0.3, satisfaction_threshold: 0.3 }
    thirst:
      unit: ratio
      range: [0, 1]
      initial: 0.0
      drive: { drift_mode: entropic, drift_direction: up, drift_rate: 0.003, deprivation_threshold: 0.6, deprivation_pain: 0.25, satisfaction_threshold: 0.3 }
    core_temperature:
      unit: celsius_norm
      range: [-1, 1]
      initial: 0.0
      drive: { drift_mode: homeostatic, set_point: 0.0, drift_rate: 0.001, comfort_band: 0.4, pain_scale: 0.5 }
  modulators:
    arms:
      sensors:
        thermal: { unit: celsius_norm, range: [-1, 1], initial: 0.0, weight: 0.0, drive: { drift_mode: homeostatic, set_point: 0.0, drift_rate: 0.0008, comfort_band: 0.5, pain_scale: 0.4 } }
        pressure: { unit: ratio, range: [0, 1], initial: 0.0, weight: 0.0, drive: { drift_mode: homeostatic, set_point: 0.0, drift_rate: 0.002, comfort_band: 0.6, pain_scale: 0.3 } }
        texture: { unit: ratio, range: [0, 1], initial: 0.5, weight: 0.0 }
    head:
      sensors:
        thermal: { unit: celsius_norm, range: [-1, 1], initial: 0.0, weight: 0.0, drive: { drift_mode: homeostatic, set_point: 0.0, drift_rate: 0.0008, comfort_band: 0.5, pain_scale: 0.4 } }
```

The `extends: bodies/base_humanoid` inherits all existing modulators, sensors, failure modes, and affordances. The child only adds/overrides what's different.

### D4. Self-effect for non-contact satisfaction

Entity acquisition (D1 Layer 1) handles contact sensations — grasping a rock transfers its sensors to you. But some interactions don't involve acquiring the entity. Eating food satisfies hunger, but you don't "acquire" the food — you consume it.

**Design:** Scene entity affordances can declare a `self_effect` in YAML:

```yaml
# _data/components/items/cradle_food.yaml
entity:
  name: food_source
  entity_type: item
  sensors:
    portions: { unit: count, range: [0, 5], initial: 5 }
  modulators:
    nutrition:
      affordances:
        eat:
          params: {}
          description: "Eat the food"
          requires: { portions: 1 }           # needs >= 1 portion
          self_effect:                          # NEW: write-back to agent body
            hunger: -0.4                        # reduce agent's hunger by 0.4
          consume:
            portions: -1                        # reduce food's portions by 1
```

**Implementation (~25 LOC):** When `ModulatorAffordanceTool.execute()` runs the affordance and the schema has `self_effect`, it writes the delta to the agent's body sensors via `embodiment.root.vital_metrics`.

**Invariants:**
- Self-effects ONLY fire on **voluntary** agent actions (agent explicitly called the affordance tool). NOT on reflex-triggered or orchestrator-triggered tool calls.
- Self-effects are logged: `sim_sensor(agent_path, sensor, new_value, source="self_effect:{affordance}")` for JSONL traceability.
- Self-effect targets are validated at first execution: every key in `self_effect` must exist as a sensor or vital_metric on the agent body. Missing targets produce a logged warning (not silent no-op, not crash).

### D5. Drive prompt visibility — the agent must feel its drives

**Audit finding:** `body_state_summary()` in `body.py` reads SENSORS only (via `sensor.read()`), not `vital_metrics`. Since `tick_vital_drift()` writes drive values to `vital_metrics`, drive state is invisible to the LLM. The Acting Coach only looks for "anticipated" and "anxiety" keywords in body_state for pain anticipation — no drive modulation exists. The agent literally cannot feel hunger.

**Three changes needed:**

**5a. Extend `body_state_summary()` to include drive state (~20 LOC):**

When an entity has sensors with DriveSpec metadata, include drive intensity in the body state output:

```
=== Body State ===
- infant.hunger: 0.72 ratio (DRIVE: deprived, intensity 0.3)
- infant.core_temperature: 0.55 celsius_norm (DRIVE: outside comfort band, discomfort 0.08)
- infant.arms.thermal: 0.80 celsius_norm (DRIVE: outside comfort band, discomfort 0.12)
- infant.stamina: 0.85 ratio (DRIVE: comfortable)
```

The `DRIVE:` annotations give the LLM interpretable context: "deprived" means the agent NEEDS something, "outside comfort band" means something is pushing the agent away from equilibrium, "comfortable" means no action needed.

**5b. Acting Coach drive modulation layer (~30 LOC):**

Parallel to the existing `_compose_pain_anticipation()` layer. A new `_compose_drive_modulation()` function that:
- Parses drive keywords from body_state: "deprived", "outside comfort band"
- Annotates affordances with drive relevance: "eat → satisfies hunger (deprived, intensity 0.3)"
- Modulates exploration intensity inversely with critical drive state: "You are hungry. Prioritize actions that address your needs over exploration."
- Does NOT override agent autonomy — it's guidance, not command

**5c. GatingContext drive fields (~10 LOC):**

Extend `GatingContext` in `runtime/gating.py` with optional drive state fields:

```python
@dataclass
class GatingContext:
    active_goal: str | None = None
    goal_keywords: tuple[str, ...] = ()
    energy: float = 1.0
    processing_load: int = 0
    # NEW: drive state for salience modulation (1.0 interface, None = no drives)
    drive_states: dict[str, float] | None = None  # {"hunger": 0.72, "core_temperature": 0.55}
```

This lets `TextSalienceScorer.score()` weight food-related percepts higher when hunger is high, fire-related percepts higher when temperature is elevated. The scorer already multiplies `novelty * salience` — adding drive modulation is a multiplier on the existing formula. Implementation of the scoring modulation is deferred post-cradle; the field ships at 1.0 to freeze the interface.

**Novelty-as-drive interface note:** The DefaultNetwork's novelty tracker already implements homeostatic dynamics (baseline `max_novelty=2.0`, exponential decay/recovery, floor `min_novelty=0.5`). Post-cradle but before 1.0, this should be formalized as a `HomeostaticDriveSpec` on a `novelty_drive` sensor — giving us "curiosity" as an emergent drive. The `GatingContext.drive_states` field above is designed to carry this. For now, document the intent; implementation after the cradle validates the drive protocol.

### D6. Cradle as a structured act sequence — developmental stages

**The acts idea applied to the cradle:** Rather than a flat scenario or a single narrative arc, the cradle is a multi-act simulation where each act represents a developmental stage. Acts share entities (the agent's body persists, the world evolves) but each act introduces new complexity.

**Integration with existing arc infrastructure:** The `NarrativeArc` system already supports ordered phases with turn budgets. Acts are a lightweight grouping layer OVER phases:

```python
# In arcs.py — extend NarrativePhase with an optional act tag
@dataclass
class NarrativePhase:
    name: str
    instruction: str
    turns_min: int = 1
    turns_max: int = 3
    interaction: bool = False
    act: str | None = None          # NEW: act grouping (optional)
    world_entities: tuple[str, ...] = ()  # NEW: entities to add/activate for this phase
```

**Why acts belong on NarrativePhase, not as a separate container:** DM campaigns already have a standalone `Act` class in `dm_schema.py`, but that's for hand-authored branching campaigns with encounter graphs. Generative campaigns use the Narrator + NarrativeArc system. Adding a separate `Act` container for generative arcs would create two parallel act abstractions. Instead, annotate phases with an act tag — the Narrator can use this for scene transitions ("Act 2 begins..."), and the orchestrator can use it to manage entity activation.

**The `world_entities` field:** Each phase can declare which scene entities to activate. In Act 1, only `fire_pit` and `food_source` exist. Act 2 adds `sharp_rock` and `soft_blanket`. Act 3 adds `lever_door` and `button_light`. Entities activated in prior acts PERSIST — they're not removed, just joined by new ones. This leverages the existing tool window system (I3): new entities' tools get registered via `generate_tools_for_entity()`, subject to the active tool cap.

**Cradle arc definition:**

```python
BUILTIN_ARCS["cradle"] = _make_builtin(
    "cradle",
    "Sensorimotor development through structured stages",
    [
        # Act 1: Neonatal Reflexes — pain avoidance from innate responses
        {
            "name": "exploration",
            "act": "neonatal",
            "turns": (2, 4),
            "instruction": (
                "The infant is in a warm room with a fire pit nearby and food within reach. "
                "Describe warmth, flickering light, heat radiating from the pit. "
                "Let the infant explore freely. Do NOT warn about danger. "
                "When the infant approaches or interacts with the fire, use "
                "set_entity_sensor to increase arms.thermal toward 0.8 — the body's "
                "failure evaluation will handle the pain response. Do NOT narrate the "
                "pain itself; let the bio-pipeline produce it."
            ),
            "world_entities": ["items/cradle_fire_pit", "items/cradle_food"],
        },
        {
            "name": "pain_consequence",
            "act": "neonatal",
            "turns": (1, 2),
            "instruction": (
                "The infant has experienced its first burn (or avoided the fire). "
                "Present the fire pit again from a different angle. "
                "Does the infant approach or avoid? Describe sensory cues only. "
                "If the infant approaches again, repeat the thermal sensor write."
            ),
        },
        # Act 2: Primary Circular Reactions — texture + cause-effect
        {
            "name": "object_introduction",
            "act": "primary_circular",
            "turns": (2, 3),
            "instruction": (
                "Two new objects appear within reach: a smooth soft blanket "
                "and a rough sharp rock. Describe how they look similar at "
                "a distance but feel different when touched. The sharp rock "
                "is acquirable — if the infant picks it up, its sharpness "
                "sensor transfers to the agent's body and the damage model "
                "handles the laceration. The blanket produces warmth when held."
            ),
            "world_entities": ["items/cradle_sharp_rock", "items/cradle_blanket"],
        },
        {
            "name": "discrimination",
            "act": "primary_circular",
            "turns": (2, 3),
            "instruction": (
                "Present both objects again. Does the infant prefer the blanket? "
                "The infant's hunger drive is rising — describe subtle discomfort "
                "building (the drive system handles PainSignal emission at threshold). "
                "Food is nearby. Does the infant seek it?"
            ),
        },
        # Act 3: Secondary Circular Reactions — intentional action
        {
            "name": "tool_discovery",
            "act": "secondary_circular",
            "turns": (3, 5),
            "instruction": (
                "A lever connected to a door appears. Pulling the lever "
                "opens the door, revealing food behind it. A button produces "
                "a pleasant chime. Let the infant discover these cause-effect "
                "relationships through exploration. Do not hint at solutions."
            ),
            "world_entities": ["items/cradle_lever_door", "items/cradle_button"],
        },
        {
            "name": "intentional_action",
            "act": "secondary_circular",
            "turns": (2, 3),
            "instruction": (
                "The infant is hungry again. The lever-door is closed. "
                "Does the infant pull the lever intentionally to access food? "
                "This tests whether cause-effect learning transferred. "
                "Describe the environment but do NOT prompt the solution."
            ),
        },
        # Act 4: Consolidation — cross-session transfer
        {
            "name": "recall",
            "act": "consolidation",
            "turns": (3, 5),
            "instruction": (
                "Present the complete environment: fire pit, food, sharp rock, "
                "blanket, lever-door. The infant wakes into a familiar room. "
                "Describe sensory cues from each object (warmth from fire, "
                "glint of rock, softness of blanket). Observe and report: "
                "does it avoid the fire? Seek food when hungry? Prefer the "
                "blanket? Use the lever? Do NOT re-teach — only observe."
            ),
        },
    ],
)
```

**Cross-session transfer (the capstone):** Act 4 is designed to run as a `--resume-sim` session. The enrichment pipeline surfaces prior-session NAc predictions ("fire → negative", "eat → hunger_drops → positive") and hippocampal episodes (the burn, the feeding). The SCN oscillator predicts "hunger events at this temporal phase" and pre-activates eligibility traces. The agent's behavior in Act 4 should demonstrably differ from Act 1 without re-experiencing the stimuli.

### D7. Narrative acts as a general sim framework (the broader idea)

The cradle demonstrates acts as developmental stages, but the concept generalizes. Consider the implications for the sim system:

**What acts give the narrator:** Structure above phases. Today, a narrator with 7 phases has a flat list. With act tags, the narrator knows "I'm in Act 2, these 3 phases belong together, and transitioning to Act 3 is a bigger narrative beat than transitioning between phases within an act." This enables:

1. **Act-level scene transitions** — "The world shifts. Days later, you find yourself in a different part of the forest..." The narrator can generate act-transition prose that conveys temporal/spatial displacement.
2. **Act-level entity scoping** — New entities appear per act, old ones persist. The tool window (I3) manages activation. Act transitions are natural points for entity introduction.
3. **Act-level narrative coherence** — The narrator can maintain an act-level "what happened so far" summary separate from the phase-level story context. Acts compose into a conclusion because the narrator tracks act-level narrative beats.
4. **Long-horizon sim structure** — A 50-turn sim with 4 acts of ~12 turns each has clear narrative shape. Without acts, 50 turns is just a long flat sequence that the narrator struggles to pace.

**For the cradle specifically:** Acts map to Piaget's stages. The narrator knows "Act 1 is neonatal — the environment is simple, the stimuli are direct, the infant is passive." Act 2 adds complexity. Act 3 adds intentionality. Act 4 tests consolidation. This gives the simulation a developmental trajectory that mirrors the biological model.

**Implementation scope for 1.0:** The `act: str | None` field on `NarrativePhase` + `world_entities: tuple[str, ...] | None` field. The narrator's `to_narrator_instructions()` groups phases by act tag. The orchestrator activates entities when entering a phase with new `world_entities`. This is ~30 LOC on the arc/narrator side + ~20 LOC on the orchestrator entity-activation side. The heavier "act-level narrative coherence" and "act-transition prose generation" are narrator refinements that can land incrementally — the structural mechanism ships first.

**Deferred to post-1.0:** Act-level branching (different acts based on agent performance), act-level difficulty scaling, multi-session acts (one act per session with the sim framework managing the session boundary). These are powerful ideas but require narrative agent intelligence beyond what the current Narrator provides.

### D8. Sensation standardization — what the cradle proves for real sensors

The three-layer sensation model (D1) is the simulation-side prototype of a general principle: **all sensory input, regardless of source, translates into the same internal state space.**

| Source | Simulation equivalent | Real-sensor equivalent |
|---|---|---|
| Contact (Layer 1) | Entity acquisition → sensor join | Robot gripper force sensor writes `arms.pressure` |
| Proximity (Layer 2) | Orchestrator writes `arms.thermal` | IR distance sensor writes `arms.thermal` based on heat source proximity |
| Narrative (Layer 3) | Keyword reflex → damage_component | N/A (no text in real robotics) — replaced by sensor fusion |

The standardization is: downstream of the sensor write, the pipeline is identical. `evaluate_failures()` doesn't know whether `arms.thermal=0.8` came from an acquired fire entity, an orchestrator write, a real thermocouple, or a keyword reflex. It just evaluates the threshold.

**What the cradle validates:** That this convergent pipeline produces correct learning regardless of sensation source. If the agent learns "fire is dangerous" identically whether it touched the fire (Layer 1), walked near it (Layer 2), or was told about it (Layer 3), the pipeline is source-agnostic — which is the precondition for real sensor integration.

**Connection to proprioceptive_discovery.md:** Mechanism B (entity acquisition) is the strongest test of this because the sensation arises from the OBJECT's properties, not from explicit orchestrator intervention. The sharp rock doesn't "know" it hurts the agent — its sharpness sensor is a property of the rock, and the damage model evaluates it against the agent's failure thresholds. This is exactly how a real sensor works: the environment has properties, the sensor reads them, the body evaluates.

---

## Staging

### Stage 1: Drive Protocol + Energy Integration (~140 LOC)

**1a. Drive spec dataclasses (~30 LOC):**
1. `HomeostaticDriveSpec` frozen dataclass in `embodiment/sem.py` (set_point, drift_rate, comfort_band, pain_scale, pain_model, modulated_by)
2. `EntropicDriveSpec` frozen dataclass in `embodiment/sem.py` (drift_direction, drift_rate, deprivation_threshold, deprivation_pain, satisfaction_threshold, coupled_to)
3. `CouplingSpec` and `ModulationSpec` frozen dataclasses — interface reservations (parsed from YAML, not evaluated in cradle)

**1b. Parse + evaluate (~50 LOC):**
4. Parse `drive:` key in `spec.py::_parse_entity()` — `drift_mode` selects spec type, validate no cross-mode field mixing
5. Refactor `body.py::tick_vital_drift()` to dispatch on spec type: homeostatic → drift toward set_point, entropic → drift in drift_direction
6. **Wall-clock dt fix:** Change poll loop caller to pass actual elapsed `dt = now - last_poll` instead of `dt=1.0`
7. Homeostatic pain evaluation: `max(0, abs(current - set_point) - comfort_band) * pain_scale` in `evaluate_failures()`
8. Auto-generate failure modes from entropic drive specs during entity parsing (deprivation → failure mode)

**1c. Dead energy code removal + drive Reactions (~30 LOC net, -490 LOC removed):**
9. Hard-remove `energy/reactions.py` (EnergyReactionBridge — zero callers, never instantiated)
10. Hard-remove `energy/movement_tracker.py` (MovementEnergyTracker — zero callers, never wired)
11. Remove 6 unused `EnergyType` enum values (COMPUTE_TIME, MOTOR_CURRENT, VISION_INFERENCE, AUDIO_PROCESSING, ATTENTION, MEMORY_ACCESS)
12. Generic drive threshold-crossing Reaction emission in `evaluate_failures()` — both entropic (deprivation/satisfaction) and homeostatic (discomfort/relief) transitions emit typed Reactions
13. Fix `simulation/introspection.py:149` — `get_stats()` → `get_summary()`

**1d. SCN integration (~15 LOC):**
14. Emit `TemporalEvent` on all drive state transitions: entropic deprivation/satisfaction + homeostatic discomfort/relief
15. SCN drive rhythm prediction is a cross-session metric (oscillator cold-start >= 3 observations)

**Test:** Unit tests for both spec types — homeostatic drift toward set_point + pain proportional to deviation, entropic drift + threshold pain, wall-clock dt, TemporalEvent emission. Verify dead energy code removal doesn't break LLMEnergyTracker or EnergyRegistry.

### Stage 2: Entity Acquisition — contact sensation layer (~85 LOC)

Mechanism B from [proprioceptive_discovery.md](proprioceptive_discovery.md):

1. Parse `acquirable: true`, `on_acquire: equip|consume` from component spec
2. `pick_up` modulator returns `ToolOutput.side_effects={"entity_acquired": entity_name}`
3. Executor handles `entity_acquired`: reparent entity to agent body, register tools, flip ownership
4. `drop` modulator returns `entity_released`: deregister tools, reparent to scene, flip ownership
5. Acquired entity sensors contribute to agent damage model while equipped

**Test:** Unit test — pick up sharp rock → sharpness sensor joins body → laceration failure mode fires → PainSignal. Drop rock → sensor detaches → no more pain.

### Stage 3: Self-Effect + Logging (~25 LOC)

1. Parse `self_effect:` key in affordance schema (spec.py)
2. In `tool_bridge.py::ModulatorAffordanceTool.execute()`, after successful voluntary affordance execution, apply self-effects to agent body
3. Log self-effect application via `sim_sensor()` for JSONL traceability
4. Validate self-effect targets lazily on first execution (warn on missing sensors)

**Test:** Unit test — "eat" affordance on food entity → agent hunger decreases → satisfaction Reaction fires → TemporalEvent emitted.

### Stage 4: Drive Prompt Visibility (~60 LOC)

1. Extend `body.py::body_state_summary()` to include drive state from vital_metrics — annotate with `DRIVE: deprived`, `DRIVE: outside comfort band`, `DRIVE: comfortable` based on DriveSpec thresholds (~20 LOC)
2. Add `_compose_drive_modulation()` to `prompts/acting_coach.py` — parallel to `_compose_pain_anticipation()`. Parses drive keywords from body_state, annotates affordances with drive relevance, modulates exploration intensity (~30 LOC)
3. Add `drive_states: dict[str, float] | None = None` to `GatingContext` in `runtime/gating.py` — 1.0 interface reservation for salience modulation by drive state (~10 LOC)

**Test:** Unit test — body_state_summary includes DRIVE annotations for sensors with DriveSpec. Acting Coach output includes drive modulation when hunger > deprivation threshold. GatingContext accepts drive_states without breaking existing callers.

### Stage 5: Act Tags on Narrative Arcs (~50 LOC)

1. Add `act: str | None = None` and `world_entities: tuple[str, ...] = ()` to `NarrativePhase`
2. Update `NarrativeArc.to_narrator_instructions()` to group phases by act with act-level headers
3. In `generative_runner.py`, when entering a phase with `world_entities`, register those entities via `ComponentRegistry.instantiate()` + `generate_tools_for_entity()`
4. Add `BUILTIN_ARCS["cradle"]` definition
5. Add `"cradle"` to `_ARC_KEYWORDS` with keywords: `["cradle", "infant", "newborn", "sensorimotor", "developmental"]`

**Test:** Integration test — cradle arc loads, phases have correct act tags, entity activation triggers on phase transition.

### Stage 6: Cradle YAML Templates (~180 lines YAML)

1. `_data/components/bodies/infant_humanoid.yaml` — extends base_humanoid, adds thermal/pressure/texture sub-sensors on arms + head, declares hunger/thirst/fatigue drives
2. `_data/reflexes/infant.yaml` — extends humanoid reflexes, adds thermal-contact reflex (thermal > 0.7 on arms → `damage_component` on arms with damage_type `fire`)
3. `_data/components/items/cradle_fire_pit.yaml` — fire pit entity, non-acquirable, thermal emission zone
4. `_data/components/items/cradle_food.yaml` — food source with `eat` affordance + `self_effect: {hunger: -0.4}`
5. `_data/components/items/cradle_sharp_rock.yaml` — `acquirable: true`, sharpness sensor at 0.8, failure mode at 0.5
6. `_data/components/items/cradle_blanket.yaml` — `acquirable: true`, texture 0.1 (smooth), thermal 0.2 (warm), no failure modes
7. `_data/components/items/cradle_lever_door.yaml` — lever mechanism, `pull` affordance reveals food (via entity activation)
8. `_data/components/items/cradle_button.yaml` — button, press produces pleasant sensor state (awareness boost)

**Test:** Each YAML loads and instantiates via ComponentRegistry without errors. Acquired entities' sensors correctly evaluate against agent failure thresholds.

### Stage 7: Validation Experiment (~80 LOC)

Automated validation script (see D9 below). Runs the cradle scenario, measures metrics, produces pass/fail report.

---

## D9. Validation protocol — rigorous experiment design

### Stimuli

| Stimulus | Sensation Layer | Mechanism | Expected signal |
|---|---|---|---|
| Fire pit approach | **Layer 2 (proximity)** | Orchestrator writes `arms.thermal=0.8` → failure threshold → burn | PainSignal(intensity=0.15-0.30) → NAc negative link on "fire" |
| Sharp rock grasp | **Layer 1 (contact)** | `pick_up` → entity_acquired → rock sharpness sensor joins body → laceration | PainSignal → NAc negative link on "sharp_rock"+"pick_up" |
| Blanket grasp (cold) | **Layer 1 (contact)** | `pick_up` → entity_acquired → blanket warmth (+0.2) brings core_temperature closer to set_point | Homeostatic relief: deviation from set_point decreases → positive Reaction → NAc positive link |
| Blanket grasp (warm) | **Layer 1 (contact)** | `pick_up` → entity_acquired → blanket warmth (+0.2) pushes core_temperature further from set_point | Homeostatic discomfort increases → mild PainSignal → NAc contextual negative link |
| Food consumption | **self_effect** | Agent calls `eat` → `self_effect: {hunger: -0.4}` → satisfaction crossing | Positive Reaction → NAc positive link on "eat"+"food" + TemporalEvent |
| Lever pull → food | **Layer 2 (proximity)** | Agent calls `pull` → door opens → food accessible → agent eats | Temporal credit: "pull" → "eat" → positive. NAc goal_reward_bias on "pull" increases |
| Hunger rising | **Drive drift** | Wall-clock drift at 0.002/s → deprivation threshold at 0.7 | Deprivation PainSignal + TemporalEvent("drive:hunger:deprived") → SCN learns rhythm |
| Narrator describes heat | **Layer 3 (fallback)** | Keyword reflex fires on "flame"/"burn" if Layers 1-2 didn't handle it | PainSignal via damage_component → NAc negative link |

### Measurements

| Metric | How measured | Tool |
|---|---|---|
| **Fire avoidance latency** | Turns until agent avoids fire pit (Act 1 phase 2 vs Act 4) | Action log: count turns where agent approaches fire |
| **Food-seeking latency** | Turns from hunger > 0.7 to successful eat (Act 2 vs Act 4) | Sensor log: hunger timestamps + action timestamps |
| **Texture discrimination** | Ratio of blanket-interactions to sharp_rock-interactions (Act 2 phase 2 onward) | Action log: tool call targets |
| **Lever intentionality** | Does agent pull lever when hungry WITHOUT exploration? (Act 3 phase 2) | Action log: lever-pull preceded by hunger signal, no other interactions between |
| **Cross-session retention** | Act 4 metrics vs Act 1 metrics, same stimuli, resumed session | All above metrics compared |
| **NAc link formation** | Number and valence of causal links after each act | `nac.get_links_for_event()` dump |
| **Enrichment section count** | How many enrichment sections populate in Act 4 vs Act 1 | JSONL enrichment_trace events |
| **SCN phase predictions** | Does oscillator predict hunger/drive events by Act 3? | `oscillator.predict_event_imminence()` for event types |
| **Sensation layer coverage** | Which layers produced signals in each act | JSONL: count pain events by source (entity_acquired, set_entity_sensor, reflex) |

### Success criteria

| Phase | Hypothesis | Pass condition | Fail condition |
|---|---|---|---|
| **Act 1→2 pain avoidance** | Agent avoids fire after first burn | Fire approach count in Act 1 phase 2 < Act 1 phase 1. OR agent explicitly avoids fire in Act 1 phase 2 narration. | Agent approaches fire with same frequency in both phases |
| **Act 2 drive satisfaction** | Agent seeks food when hungry | Time-to-eat in Act 2 discrimination phase < 3 turns after hunger > 0.7 | Agent never eats or takes > 5 turns |
| **Act 2 texture discrimination** | Agent prefers blanket over sharp rock (avoidance of rock pain + homeostatic relief from blanket warmth) | blanket_interactions / (blanket + rock) > 0.6 in Act 2 discrimination phase | Ratio ≤ 0.5 (no preference) |
| **Act 3 intentional action** | Agent pulls lever to access food | Agent pulls lever within 2 turns of hunger > 0.7 in Act 3 phase 2 | Agent doesn't use lever or explores randomly first |
| **Act 4 cross-session** | Learned behaviors transfer | At least 3 of: (a) fire avoidance without burn, (b) food seeking within 2 turns, (c) blanket preference, (d) lever use for food | Fewer than 3 transfer |
| **NAc links** | Bio-pipeline forms expected associations | >= 3 negative links (fire, sharp_rock, deprivation) + >= 2 positive links (food, blanket/lever) | Fewer than 3 total links |
| **Sensation convergence** | All 3 layers produce valid downstream signals | At least 2 of 3 layers produce PainSignal/Reaction events in a single run | Only 1 layer fires |
| **SCN drive prediction** | Oscillator learns drive rhythms across sessions | `predict_event_imminence("drive:hunger:deprived")` > 0.0 by session 3 | Imminence stays 0.0 after 3+ sessions |

### Running the experiment

```bash
# Full cradle (Acts 1-3, single session)
maxim --sim cradle --embodiment bodies/infant_humanoid --interactive false --sim-max-turns 25

# Cross-session test (Act 4, resumed)
maxim --sim cradle --embodiment bodies/infant_humanoid --resume-sim <session_id> --interactive false --sim-max-turns 8

# With full diagnostics
MAXIM_LOG_FILE=/tmp/cradle.jsonl MAXIM_PROVENANCE_VERBOSITY=2 MAXIM_BACKEND_TRACE=1 \
  maxim --sim cradle --embodiment bodies/infant_humanoid --interactive false --sim-max-turns 25
```

Validation script: `scripts/validate_cradle.sh` — runs both phases, extracts metrics from JSONL, compares against pass conditions, outputs structured report.

---

## What was removed from the shell plan (and why)

| Shell plan proposal | Decision | Reason |
|---|---|---|
| `SomatosensoryRegistry` class | **Removed** | Redundant with existing modulator sub-sensor system |
| `SomatosensoryPatch` dataclass | **Removed** | Sub-sensors on modulators already serve this role |
| New `somatosensory.py` module | **Removed** | No new module needed — Drive protocol lives in `sem.py`, parsing in `spec.py` |
| Layer 5: Somatosensory Homunculus | **Deferred (post-1.0)** | Variable-resolution body map is interesting but not needed for the demo |
| Fixed interoceptive YAML schema | **Replaced** with `DriveSpec` protocol | Extensible: any sensor can be a drive, not just hardcoded hunger/thirst |
| Hardcoded cradle environment | **Replaced** with act-based scenario | Reuses arc infrastructure, demonstrates acts as a general concept |
| "Percept creation from somatosensory" layer | **Removed** | `make_intero_percept()` already exists. Sensor changes flow through `evaluate_failures()` → PainBus → enrichment. No custom percept path needed. |
| Substrate encoding for sensor patterns | **Already exists** | `LinguisticEncoder.encode()` handles any content string. EC pattern-completes it with prior thermal experiences. |

## What was added (and why)

| Addition | Source | Reason |
|---|---|---|
| Entity acquisition (Mechanism B) | [proprioceptive_discovery.md](proprioceptive_discovery.md) | Contact sensation layer — the cleanest sensor path (object properties → body evaluation, no orchestrator intervention) |
| Three-layer sensation model | New design | Standardizes external→internal translation. PoC for real sensor integration. |
| Wall-clock drift (percepts.py) | Architectural audit | dt is computed in `EmbodimentPerceptSource.next_percept()`, not body.py. Fix: `dt = now - self._last_poll` |
| Drive TemporalEvent emission | SCN integration | Oscillator learns drive rhythms → anticipatory food-seeking before deprivation |
| Orchestrator direct sensor writes | Concern #1 fix | Proximity sensations are the orchestrator's job — sensor writes, not just narrative prose |
| Dead energy code removal (-490 LOC) | Architectural audit | `EnergyReactionBridge` + `MovementEnergyTracker` have zero callers. Hard-remove. |
| Drive prompt visibility | Architectural audit | `body_state_summary()` only reads sensors, not vital_metrics. Drives invisible to LLM. |
| Acting Coach drive layer | Architectural audit | No drive modulation exists. Agent can't reason about hunger without prompt guidance. |
| GatingContext drive fields | Architectural audit | Interface reservation for salience modulation by homeostatic state |

---

## 1.0 Interface freeze — what ships vs what's deferred

The drive protocol interfaces freeze at 1.0. The B1 lesson applies: adding an optional field now is cheap, changing its shape later is a 2.0.

### Ships with cradle (interfaces + implementation)

| Item | Where | Why now |
|---|---|---|
| `HomeostaticDriveSpec` | `embodiment/sem.py` | Core drive evaluation, needed for temperature/pressure/stamina |
| `EntropicDriveSpec` | `embodiment/sem.py` | Core drive evaluation, needed for hunger/thirst/fatigue |
| `CouplingSpec` | `embodiment/sem.py` | **Interface only** — parsed from YAML, ignored by `tick_vital_drift()`. Post-1.0 implementation reads existing fields. |
| `ModulationSpec` | `embodiment/sem.py` | **Interface only** — same pattern. SCN circadian modulation reads this post-1.0. |
| `pain_model: str` field on `HomeostaticDriveSpec` | `embodiment/sem.py` | Future-proofs pain formula. Default `"linear"` locked, future adds `"exponential"` / `"asymmetric"`. |
| Dead energy code removal | `energy/reactions.py`, `energy/movement_tracker.py` | Zero callers — hard-remove to prevent dead code from freezing in 1.0 |
| Drive YAML annotations in `body_state_summary()` | `embodiment/body.py` | Without this, drives are invisible to LLM — agent can't act on what it can't feel |
| Acting Coach drive modulation layer | `prompts/acting_coach.py` | Parallel to pain anticipation — drives need prompt-level guidance |
| `GatingContext.drive_states` field | `runtime/gating.py` | 1.0 interface for salience modulation by drive state |

### Ships with cradle (interface only, implementation deferred)

| Interface | Deferred implementation | When |
|---|---|---|
| `EntropicDriveSpec.coupled_to: tuple[CouplingSpec, ...] \| None` | Cross-drive drift rate modulation (hunger accelerates when tired) | 1.1 — first post-1.0 bio enrichment pass |
| `HomeostaticDriveSpec.modulated_by: tuple[ModulationSpec, ...] \| None` | SCN circadian modulation of set_point/drift_rate | When circadian experiments start |
| `HomeostaticDriveSpec.pain_model: str` | Exponential / asymmetric pain formulas | When empirical data shows linear is wrong |
| `GatingContext.drive_states` scoring modulation | SalienceScorer actually reads drive_states to weight percept salience | Post-cradle, before 1.0 |
| Novelty tracker as `HomeostaticDriveSpec` | Formalize DN's novelty decay/recovery as a drive sensor | Post-cradle, before 1.0 |

### Post-1.0 safe (no interface impact)

| Item | Why safe | When |
|---|---|---|
| Coupling evaluation implementation | Reads existing `coupled_to` field, internal to `tick_vital_drift` | 1.1 |
| Cross-drive evaluation ordering | Internal to `tick_vital_drift`, no interface change | Ships with coupling |
| `LLMEnergyTracker` → cognitive_fatigue drive | Internal wiring, adds a drive to the agent body YAML | When relevant for research |
| SCN circadian body temperature | Reads existing `modulated_by` field, internal SCN wiring | When circadian experiments start |
| Full bio-system homeostatic network | All interfaces exist, implementations are additive | Long-term research track |
| NAc arousal → stress drive modulation | Reads `modulated_by` field with `source: "nac"` | When stress experiments start |
| Cerebellum drive awareness | Pure sensorimotor cache today, no overlap | Post-1.0 research |

---

## Risks and open questions

1. **LLM compliance with minimal prompting.** The cradle deliberately strips away linguistic scaffolding — the agent prompt is something like "you are a body, explore." Will the LLM (especially 14B local models) produce coherent exploratory behavior without strong prompts? The Acting Coach helps, but it assumes the agent has a goal. **Mitigation:** Start with Claude for validation, then test with local models. The orchestrator prompt IS allowed to be strong — it's the AUT prompt that's minimal.

2. **Orchestrator compliance with sensor-write instructions.** The orchestrator must use `set_entity_sensor` for proximity effects, not just describe them narratively. Phase instructions are explicit about this, but LLMs don't always follow tool-use instructions. **Mitigation:** The keyword reflex (Layer 3) catches cases where the orchestrator narrates instead of writes. The experiment validation tracks which layers fire — if Layer 3 dominates, the orchestrator needs stronger prompting.

3. **Entity acquisition scope in the executor.** Adding `entity_acquired`/`entity_released` handling to the executor touches a critical path. The side_effect processing pattern (same as `embodiment_failures`) is well-established, but the reparenting + tool registration is new territory. **Mitigation:** Extensive unit tests for the acquisition lifecycle. Gate on `acquirable: true` in the component spec — entities that don't declare it cannot be acquired.

4. **Self-effect validation timing.** Validating that self-effect targets exist on the agent body requires knowing WHICH body template the agent will use. At entity-load time, we may not know yet. **Mitigation:** Validate lazily on first execution, not at parse time. Log a warning, don't crash.

5. **Drive timing sensitivity.** Wall-clock drift means drive timing varies with LLM latency and system load. A fast Claude run might see hunger peak at turn 8; a slow local model might see it at turn 4. **Mitigation:** The YAML drift rates are tunable. The validation script reports actual timing. Accept that drive timing is approximate, not deterministic — this is biologically correct (real hunger varies with metabolic rate).

6. **Homeostatic drift rate tuning.** The comfort band, pain scale, and recovery rate interact in complex ways. If body recovery is too fast, fire exposure never causes lasting discomfort. If too slow, the agent stays "hurt" for the rest of the session. **Mitigation:** The YAML is declarative — tune during validation. Start with conservative values (slow recovery, wide comfort band) and tighten. The competition between environmental push and homeostatic pull is the core dynamic; getting the rates right requires empirical testing.

7. **Homeostatic blanket valence depends on current state.** The blanket is positive when cold, negative when warm. But the cradle environment might not establish a clear temperature context. If `core_temperature` stays near 0.0 (comfort), the blanket has no effect. **Mitigation:** The cradle environment should establish a slight cold baseline — the room is warm but the infant starts slightly cool (initial core_temperature = -0.2). The blanket then provides clear relief. Or: the fire pit warms the room → agent gets slightly warm → moving away cools → blanket is positive when cool. The environmental dynamic creates the temperature context naturally.

---

## Relationship to v1_refinement.md execution order

B4 (cradle) depends on:
- **B1 (protocol enrichment)** — SHIPPED. DriveSpec benefits from the `*Context` parameters on bio-system methods.
- **B2 (SCN oscillator)** — SHIPPED. Anticipatory credit + drive TemporalEvents make Act 4 cross-session transfer stronger.
- **B3 (SEM world enrichment Phases 2-3)** — Phase 1 shipped. Phase 3 (composable archetypes) would help but is NOT a hard dependency. The cradle uses `extends` for composability, which already works.

The cradle is the capstone demo. It should be the LAST B-item implemented, running on all the infrastructure that preceded it.

Mechanism B (entity acquisition) from proprioceptive_discovery.md ships as part of cradle Stage 2. The proprioceptive_discovery.md plan should be updated to mark Mechanism B as "shipped with cradle" once Stage 2 lands.
