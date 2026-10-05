# Embodiment Troubleshooting

## YAML Loading Issues

### "Entity must have a 'name' field"

Every entity in the YAML must have a `name` key:

```yaml
# Wrong:
children:
  - entity_type: joint
    sensors: ...

# Right:
children:
  - name: shoulder
    entity_type: joint
    sensors: ...
```

### "Expected YAML dict, got ..."

The YAML file must have a top-level dict with `body` and/or `world_entities`:

```yaml
# Wrong — bare list:
- name: sword
  entity_type: weapon

# Right — under world_entities key:
world_entities:
  - name: sword
    entity_type: weapon
```

### "YAML must have a 'body' key and/or 'world_entities' key"

The file loaded successfully as YAML but doesn't have the expected top-level structure. Make sure your file has at least one of:

```yaml
body:
  name: my_robot
  entity_type: robot
  ...

# and/or

world_entities:
  - name: my_item
    entity_type: item
    ...
```

### Sensor range not working

Ranges must be a two-element list `[min, max]`:

```yaml
# Wrong:
angle: {unit: degrees, range: 360}
angle: {unit: degrees, range: {min: 0, max: 360}}

# Right:
angle: {unit: degrees, range: [0, 360]}
angle: {unit: degrees, range: [-180, 180]}
```

### Affordance params not parsed

Parameter types must be strings matching Python type names:

```yaml
# Wrong:
params: {degrees: number, speed: decimal}

# Right:
params: {degrees: float, speed: float}
# Supported: float, int, str, bool
```

For optional parameters with defaults:

```yaml
# Right:
params:
  degrees: float
  speed:
    type: float
    default: 1.0
```

### Failure trigger not firing

1. **Check the sensor name matches**: the `field` in the trigger must match a sensor name or vital metric on the entity:

```yaml
sensors:
  angle: {unit: degrees, range: [0, 180]}

failure_modes:
  # Wrong — field doesn't match sensor name:
  - name: overextension
    trigger: {field: position, op: ">", value: 175, pain: 0.8}

  # Right:
  - name: overextension
    trigger: {field: angle, op: ">", value: 175, pain: 0.8}
```

2. **Check the operator direction**: `>` means "fire when value exceeds threshold":

```yaml
# Fire when durability drops BELOW 0.1:
trigger: {field: durability, op: "<", value: 0.1, pain: 0.6}

# Fire when temperature rises ABOVE 70:
trigger: {field: temperature, op: ">", value: 70, pain: 0.6}
```

3. **Compound triggers use `all:` key**:

```yaml
# Wrong — two separate triggers (OR logic):
failure_modes:
  - name: tennis_elbow
    trigger: {field: strain, op: ">", value: 0.6}
    trigger: {field: fatigue, op: ">", value: 0.5}  # overwrites first!

# Right — compound trigger (AND logic):
failure_modes:
  - name: tennis_elbow
    trigger:
      all:
        - {field: strain, op: ">", value: 0.6}
        - {field: fatigue, op: ">", value: 0.5}
```

## Tool Generation Issues

### Duplicate tool names

If you see `ValueError: Cannot resolve unique tool name`, two entities in different trees have the same name AND the same parent chain. Solutions:

1. Give entities unique names: `left_shoulder` vs `right_shoulder`
2. Or ensure parent entity names differ: the system will auto-prefix with parent name on collision

### Tools not appearing in agent prompt

1. Check that `generate_tools_for_entity()` was called with the entity tree
2. Check that the registry passed is the same one the agent loop uses
3. Tools are listed in the prompt's CRITICAL section — they should always appear unless the prompt is severely over-budget

### Tool execution returns "Unknown affordance"

The affordance name in the tool call must match exactly what's in the YAML:

```yaml
# YAML:
affordances:
  rotate_angle:
    params: {degrees: float}

# Tool call must use "rotate_angle", not "rotate" or "rotateAngle"
```

## Pain / Failure Issues

### Pain not publishing

1. Verify `pain_bus` is passed to `Embodiment()`:
   ```python
   emb = Embodiment(root, pain_bus=pain_bus)
   ```

2. Verify `config.enable_pain` is True (default):
   ```python
   config = EmbodimentConfig(enable_pain=True)
   ```

3. Verify the sensor value actually exceeds the threshold. Use `body_state_summary()` to check:
   ```python
   for state in emb.body_state_summary():
       print(state)
   ```

### Persistent failure won't clear

Persistent failures require a `recovery_condition` to be met:

```yaml
failure_modes:
  - name: overheating
    trigger: {field: temperature, op: ">", value: 70, pain: 0.6}
    persistent: true
    recovery_condition:
      field: temperature
      op: "<"
      value: 40    # Must drop BELOW 40, not just below 70
```

If no `recovery_condition` is specified, persistent failures never clear automatically.

## Percept Source Issues

### EmbodimentPerceptSource returns None

This is normal between poll intervals. The source only produces percepts at the configured rate (default 1Hz). If you need faster updates:

```python
source = EmbodimentPerceptSource(emb, poll_hz=10)  # 10 Hz
```

Or enable demand mode for a burst of high-rate reads:

```python
source.set_demand_mode(True, hz=30)  # 30 Hz during demand
```

### Sensor readings stale in prompt

The `EmbodimentPerceptSource` reads sensors at the configured poll rate and applies vital metric drift. If sensors seem stale:

1. Check that `evaluate_failures()` is being called (it reads sensors)
2. Check that `tick_vital_drift()` is being called (it advances degradation metrics)
3. Both are called automatically by `EmbodimentPerceptSource.next_percept()`

## Virtual Entity Issues (Swords, NPCs)

### NPC trust/mood not changing

Virtual entity sensors read from `entity.vital_metrics`. If you're using `SpecModulator` stubs (no LLM backend), they return success but don't modify state. Attach a backend that applies changes. A `CerebellumModulator` backend exists, but it is **Dormant since 2026-05-26** (#909 records it) — no production path or SEM tool uses it, and nothing new should build on it:

```python
from maxim.embodiment.backends.cerebellum_modulator import cerebellum_modulator_factory
from maxim.embodiment.spec import attach_backends
from maxim.embodiment.cerebellum import Cerebellum

cb = Cerebellum()
factory = cerebellum_modulator_factory(cb, fallback_factory=None)
attach_backends(root, modulator_factory=factory)
```

Without a LLM fallback the `CerebellumModulator` returns a no-op result, which is the same behaviour as the raw `SpecModulator` stubs. You do not need it for Cerebellum learning: the live forward model trains from every SEM affordance execution via `embodiment/tool_bridge.py` (`observe_from_action`), whatever backend applied the change.

### Item durability not degrading

Same issue — without a backend, modulator execution is a no-op. Attach a backend that applies changes to sensor values (see above; the `CerebellumModulator` shown there is dormant and is not used by any production path).

## Cerebellum Issues

### Cerebellum confidence not rising

The forward model trains live (`observe_from_action` after each SEM affordance; confidence is reported in the `sim_cerebellum` telemetry). Note that `Cerebellum.predict` is **Dormant since 2026-10-04 (#909)** — nothing in production reads predictions, so "no predictions" is expected; what you can check is confidence. A model needs `confidence >= 0.3` (default) before `predict` would return it, which takes ~2 observations. Check:

```python
cb.get_confidence("arm.shoulder", "motor", "rotate_angle", {"degrees": 45})
# Returns 0.0 if no model, or the current confidence level
```

### Model not found for similar params

Param bucketing rounds values. Two calls with slightly different params may hit different buckets:

```python
from maxim.embodiment.cerebellum import bucket_params

# These might be different buckets:
bucket_params({"degrees": 45.0}, sensor_ranges={"degrees": (0, 180)})  # step=18 → "degrees=36.00"
bucket_params({"degrees": 55.0}, sensor_ranges={"degrees": (0, 180)})  # step=18 → "degrees=54.00"
```

If you need wider buckets, increase `range_fraction` in `bucket_params()` or reduce the sensor range.

### Cerebellum state not persisting

`build_bio_stack(persistence_dir=...)` binds the Cerebellum to `<persistence_dir>/cerebellum.json` — the agent home, **not** `~/.maxim/memory/` — loads it at session start (unless `load_persisted=False`) and saves it in `BioStack.on_session_end` (#908). A stack built without a persistence dir has no Cerebellum. If the file does not appear:

1. **Session never ended cleanly** — the save happens only in `on_session_end`; a killed process saves nothing.
2. **Save refused** — look for `Cerebellum not saved: ...` at ERROR. The #971 store guard never saves over a file the store did not read; if the load failed, the unreadable file was kept as `cerebellum.json.corrupt-<UTC>` and the fresh store saves in its place.
3. **Reachy embodied runtime** — `embodied_runtime/agentic_runtime.py` neither trains nor saves a Cerebellum, so it writes no file. It does build a bio stack on `user_memory()`, so it loads `~/.maxim/memory/cerebellum.json` if one exists there, and an unreadable one there is copied aside like anywhere else.

Standalone use binds the path yourself:

```python
from maxim.embodiment.cerebellum import Cerebellum, CerebellumConfig

cb = Cerebellum(CerebellumConfig(persistence_path="/path/to/home/cerebellum.json"))
cb.save()  # Explicit save
cb.load()  # Explicit load
```

## Motor Program Issues

**Motor programs are Dormant since 2026-10-04 (#909).** Program crystallization (`observe_action_sequence` / `ProgramRegistry.observe_sequence`) has no production caller, so no program ever forms and every `find_programs_*` / `find_related` read returns `[]`. There is no program executor in `src/`. The entries below describe the dormant code for anyone reviving it (see [docs/plans/engram_formation.md](../plans/engram_formation.md) E7); in a normal run they are not applicable.

### Programs not crystallizing

Expected today — nothing calls the crystallization path. When called directly, motor programs need the same SEM sequence (same entity paths, modulators, affordances in the same order) to recur 3+ times. Params can vary — they're bucketed and averaged. Check:

```python
cb.programs.stats()
# Look at "pending_observations" — if > 0, sequences are being tracked but haven't hit 3 yet
```

### Pain gate not triggering

Pain gates check the entity's `vital_metrics`, not the raw sensor reading. Ensure the vital_metrics are updated:

```python
entity.vital_metrics["angle"] = 176  # Must be set for gate to check
```

### Motor program not appearing in prompt

Expected — not a wiring bug. `MemoryAgent.build_context()` fills `StructuredContext.motor_programs` from `cerebellum.programs.find_related()`, and the AdaptivePlanner checks `cerebellum.find_related_programs()` during `propose_plans()`, but with crystallization dormant both always return `[]`, so the motor-programs prompt section is always empty.

## Engram Issues

**Motor engrams are Dormant** (`form_engram` since 2026-09-22, the rest since 2026-10-04; #909). `form_engram`, `query_engrams` and `cleanup_program` (`embodiment/engrams.py`) have no production caller: no engram forms or is recalled in a normal run. The entries below describe the dormant code for anyone reviving it.

### Engrams not forming

Expected today — nothing calls `form_engram`. When called directly, engrams only form on significant outcomes:
- Pain intensity > 0.3
- RPE magnitude > 0.3
- Novelty > 0.7
- Program confidence < 0.3 (learning phase)

Routine successes on confident programs do NOT create engrams. This is intentional — the Cerebellum's R-W update handles those.

### Engrams not affecting behavior

Expected today — `query_engrams()` has no production caller, so engrams affect nothing. When called directly, recall requires:
1. A hippocampus instance passed to the Cerebellum
2. Engram graph nodes (`cerebellum:program:{name}`) in the hippocampal graph
3. Context similarity > 0.3 between remembered and current entity states

If engrams exist but aren't being recalled, check `entity_state_similarity()` — it only compares scalar sensors, so if all your entities have only frame sensors, similarity will always be 0.0.
