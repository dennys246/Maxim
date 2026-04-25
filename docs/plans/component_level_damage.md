# Component-Level Damage Model

**Branch:** `feat/sem-damage-autosense` (continues from Phase 1 manifest work)
**Status:** STAGES 1-5 SHIPPED, Stage 6 deferred
**Depends on:** Phase 1 scene manifest (committed)

## Motivation

Damage today is a flat `health -= X` on the entity root. A sword hitting a dragon's wing and exhaustion from flying both look identical — no body-part localization, no cascading failure, no reason for the agent to learn directional defense. The double-damage bug (auto-damage in `SendMessageTool` + explicit `DamageEntityTool` fire on the same attack) is a symptom of this flat model.

The fix: damage targets body **components** (modulators/body-parts), not the entity root. Part damage cascades upward to entity health. The agent learns "wing damage → can't fly" not just "health low → bad."

## Existing infrastructure (audit results)

**Already supports this:**
- `Entity` has tree composition: `children`, `walk()`, `find(path)` — body parts as child entities are structurally possible without any model change
- `_parse_entity()` already recursively parses `children` from YAML
- `FailureMode.evaluate()` checks `sensor_readings` dict — adding modulator-level readings is additive
- `Body.evaluate_failures()` walks `root.walk()` — already evaluates all descendants

**Needs to change:**
- `DamageEntityTool` targets `root.vital_metrics["health"]` — must target components
- `SendMessageTool._detect_attack()` auto-damage — must be removed (single source of truth)
- Modulators have no sensors today — they're capability axes, not body parts with state
- Entity YAML format needs per-modulator sensor + integrity support
- `--deep-embodiment` runtime flag for resolution depth

## Design: LOD (Level of Detail) Embodiment

### The spec is always full fidelity

YAML carries the richest representation. The runtime collapses based on depth:

```yaml
# creatures/dragon.yaml (upgraded)
entity:
  name: dragon
  entity_type: creature
  modulators:
    head:
      sensors:                        # NEW: per-modulator sensors
        awareness: {range: [0,1], initial: 0.9}
        jaw_integrity: {range: [0,1], initial: 1.0}
      integrity: weighted_mean        # NEW: aggregation function
      affordances:
        bite: {params: {target: str}, description: "Seize in massive jaws"}
        roar: {params: {}, description: "Terrifying roar"}
      damage_affinities:              # NEW: level 3 only
        slash: {jaw_integrity: 0.6, awareness: 0.2}
        blunt: {jaw_integrity: 0.8}
    wing:
      sensors:
        membrane_integrity: {range: [0,1], initial: 1.0, weight: 0.4}
        bone_integrity: {range: [0,1], initial: 1.0, weight: 0.4}
        joint_mobility: {range: [0,1], initial: 1.0, weight: 0.2}
      integrity: weighted_mean
      affordances:
        take_flight: {params: {}, requires: {integrity: 0.3}}   # NEW: precondition
        dive_attack: {params: {target: str}, requires: {integrity: 0.5}}
        land: {params: {}}
        circle: {params: {}}
      damage_affinities:
        slash: {membrane_integrity: 0.8, bone_integrity: 0.1}
        blunt: {bone_integrity: 0.7, joint_mobility: 0.3}
        fire: {membrane_integrity: 0.9}
    torso:
      sensors:
        armor_integrity: {range: [0,1], initial: 0.95}
        organ_health: {range: [0,1], initial: 1.0}
      integrity: min                  # torso integrity = weakest sub-sensor
      affordances: {}
    combat:
      sensors: {}
      affordances:
        fire_breath: {params: {target: str}, description: "Cone of flame"}
        claw_strike: {params: {target: str}}
        tail_sweep: {params: {}}
  sensors:                            # entity-level sensors (non-body-part)
    fire_breath_charge: {range: [0,1], initial: 0.7}
    aggression: {range: [0,1], initial: 0.8}
  health: derived                     # NEW: entity health derived from parts
  health_weights:                     # which parts matter most for survival
    head: 0.3
    wing: 0.15
    torso: 0.4
    combat: 0.15
  failure_modes:
    - name: grounded
      trigger: {modulator: wing, field: integrity, op: "<", value: 0.3, pain: 0.4}
    - name: death
      trigger: {field: health, op: "<=", value: 0, pain: 1.0}
```

### Runtime resolution

**Level 2 (default, `--embodiment`):**
- Each modulator exposes ONE `integrity` value (aggregated from sub-sensors)
- `DamageComponentTool` takes `component="wing"` + `amount=0.3`
- Runtime sets `wing.integrity -= 0.3` and distributes proportionally to sub-sensors
- Entity `health` derived as weighted mean of component integrities
- Failure modes fire on `modulator.integrity` thresholds
- Pain context: `{entity: "dragon", component: "wing", failure_mode: "grounded"}`
- Affordance `requires` checked against component integrity

**Level 3 (`--deep-embodiment`):**
- Sub-sensors individually exposed: `wing.membrane_integrity`, `wing.bone_integrity`, etc.
- `DamageComponentTool` takes optional `damage_type="slash"` → routes via `damage_affinities`
- Agent sees all sub-sensors, can reason about specific damage types
- Pain context richer: `{component: "wing", sub_sensor: "membrane_integrity", damage_type: "slash"}`
- Affordance `requires` can reference sub-sensors

**Level 1 (backward compat, no per-modulator sensors):**
- Old specs without modulator sensors still work
- Single entity-level `health` sensor, damage goes to `vital_metrics["health"]`
- This is the current behavior — zero migration needed for old specs


## Implementation Stages

### Stage 1: Per-modulator integrity sensors + aggregation

Add `sensors` field to modulators in the spec parser. Each modulator can optionally have sensors with an `integrity` aggregation function. Entity health becomes derivable from component integrities.

**Files:**
- `embodiment/spec.py` — parse modulator `sensors`, `integrity`, `health: derived`, `health_weights`
- `embodiment/sem.py` — add `sensors` dict to `SpecModulator`, add `integrity` property
- `embodiment/body.py` — aggregate component integrities into entity health during `evaluate_failures()`

**Key decisions:**
- `integrity` aggregation functions: `weighted_mean` (default), `min`, `max`
- If no weights specified, sub-sensors contribute equally
- If no modulator sensors, modulator integrity defaults to 1.0 (backward compat)
- Entity `health: derived` is optional — old specs with `health` as a direct sensor still work

### Stage 2: DamageComponentTool + remove double-damage

Replace `DamageEntityTool` with `DamageComponentTool`. Remove auto-damage from `SendMessageTool`. Single source of truth for damage.

**Files:**
- `simulation/tools.py` — new `DamageComponentTool`, remove `_detect_attack` auto-damage from `SendMessageTool`
- `simulation/orchestrator.py` — register `DamageComponentTool`, update TOOL_DESCRIPTIONS, make conditional on embodiment

**Tool API:**
```
damage_component(component="wing", amount=0.3, source="dragon_claw")
```

At level 2: directly reduces `component.integrity`.
At level 3 (with `damage_type`): routes through `damage_affinities`.

**Migration:** `DamageEntityTool` stays as a deprecated thin shim that delegates to `DamageComponentTool` with `component="torso"` (default target). This way existing orchestrator prompts that call `damage_entity` still work.

### Stage 3: Affordance `requires` gating

Affordances can declare integrity preconditions. When a component's integrity drops below the threshold, the affordance tool returns a failure result instead of executing. This produces a natural tool-failure → pain → learning chain.

**Files:**
- `embodiment/spec.py` — parse `requires` on affordances
- `embodiment/sem.py` — add `requires` to `AffordanceSchema`
- `embodiment/tool_bridge.py` — check `requires` before executing affordance, return failure if unmet
- `tools/base.py` — `ToolOutput.side_effects["affordance_blocked"]` for the blocked signal

**Key decisions:**
- `requires: {integrity: 0.3}` checks the parent modulator's `integrity`
- `requires: {membrane_integrity: 0.5}` checks a specific sub-sensor (level 3 only — ignored at level 2)
- Blocked affordances still appear in the tool list (the agent can try them), but execution fails with an explanatory error: "Wing too damaged to take flight (integrity: 0.15, requires: 0.30)"

### Stage 4: `--deep-embodiment` CLI flag + resolution collapsing

Add the CLI flag and the runtime layer that collapses full-fidelity specs to level 2 or exposes them at level 3.

**Files:**
- `cli.py` — add `--deep-embodiment` flag
- `embodiment/resolution.py` — NEW: `collapse_to_level2(entity)` / `expand_to_level3(entity)` runtime transforms
- `simulation/orchestrator.py` — apply resolution after entity construction

**Key decisions:**
- Default: level 2 (collapsed). `--deep-embodiment` enables level 3
- Level detection: if spec has `damage_affinities` AND `--deep-embodiment` is set → level 3
- Collapsing: at level 2, sub-sensors are still parsed and stored but NOT exposed to the agent (not in sensor readings, not in tool descriptions, not in prompt). Integrity is the single visible signal.
- `MAXIM_DEEP_EMBODIMENT=1` env var as alternative to CLI flag

### Stage 5: Upgrade seed components

Add per-modulator sensors, integrity aggregation, `requires` on affordances, and `damage_affinities` to the 65 seed component YAMLs. This is the content migration that makes the new infrastructure useful.

**Approach:**
- Creatures (dragon, spider, wolf, etc.): body-part modulators with sensors
- Weapons (rusty_sword, etc.): single modulator `blade` with `sharpness` + `durability` sensors
- Bodies (base_humanoid, reachy_mini): limb modulators with mobility/strength sensors
- Environments: no change (environments don't take damage the same way)
- NPCs: humanoid body structure, same as base_humanoid
- Validation: parametrized pytest loads every YAML through `_parse_entity`

### Stage 6 (deferred): Damage type system

Full damage-type routing via `damage_affinities`. Only activated at level 3.

This is the part we build the infrastructure for now but don't activate until `--deep-embodiment` ships. The YAML format supports `damage_affinities` from Stage 5, and the routing logic lives in `DamageComponentTool` with a `damage_type` parameter that's ignored at level 2.


## Backward Compatibility

| Existing feature | Impact |
|-----------------|--------|
| Old entity specs (no modulator sensors) | Work unchanged — level 1 fallback |
| `DamageEntityTool` callers | Deprecated shim delegates to `DamageComponentTool(component="torso")` |
| `SendMessageTool` auto-damage | Removed — orchestrator uses `DamageComponentTool` explicitly |
| `evaluate_failures()` | Still works — reads both modulator-level and entity-level vital_metrics |
| Pain context | Enriched with `component` field — existing subscribers see extra keys, not breaking |
| NAc learning | Richer signals — learns component-level avoidance patterns |
| Affordance tools | Still work — `requires` is optional, absent = always executable |


## Implementation Order

```
Stage 1 (integrity sensors + aggregation)  ← structural foundation
  └─ Stage 2 (DamageComponentTool)         ← fixes double-damage, single source of truth
       └─ Stage 3 (affordance requires)    ← tool-failure → pain → learning chain
            └─ Stage 4 (--deep-embodiment) ← runtime LOD system
                 └─ Stage 5 (seed upgrades) ← content migration
                      └─ Stage 6 (damage types) ← deferred, level 3 only
```

Stages 1-3 ship together as one PR (they're tightly coupled).
Stage 4 ships independently.
Stage 5 can be incremental (upgrade a few components at a time).
Stage 6 is deferred.
