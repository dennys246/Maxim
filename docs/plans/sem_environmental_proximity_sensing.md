# SEM environmental proximity sensing for homeostatic drives

**Status:** PLANNED (2026-07-15). Motivated by the Exp 44 bring-up: the
chilled-infant body's `cold` drive advances via an opaque internal drift
ticked on tool execution, which (a) *feels* architecturally wrong — the
homeostatic clock is bundled into `Body.evaluate_failures` and gated on
tool-firing (a pragmatic hack from commit `ed8b187f`) — and (b) leaves the
*environment inert until touched*: warmth/fire entities only affect the agent
when it executes `warm_self`, so a freezing agent with no other pressure has
no grounded, sensed reason to seek warmth.

**Not an Exp 44 blocker.** The actual Exp 44 blocker is scene-control (the
generative arc spawns distractor entities instead of the warmth dilemma —
tracked separately). This plan is the medium-term "do it right" for embodied
interoception; it is worth building on its own merits but must NOT gate the
experiment.

## The idea, in one sentence

Each turn, the agent senses ambient *field* sensors already declared on scene
environment entities (a fire's `heat_output`, cold air's ambient chill),
filtered by field type, and its exposure updates its own homeostatic drive
sensors — so interoception (feeling cold) is grounded in sensed exteroception
(being near cold / away from heat) rather than an opaque internal drift.

This makes the **Proximity** layer of the existing three-layer sensation model
(Contact / Proximity / Narrative) **first-class and agent-autonomous**, where
today it exists only as orchestrator-LLM-driven sensor writes.

## Front-gate: does this need a new mechanism? (mostly no)

The infra audit (2026-07-15) found the data model is DONE and inert:
- `cradle_fire_pit.yaml` already declares entity sensors `heat_output: 0.9`,
  `fuel: 0.8`; its docstring says "proximity effects are handled by
  orchestrator sensor writes." **Zero code reads `heat_output`/`ambient`/
  `radiate`** (grep-verified). The field data exists; the reader is missing.
- SEM modulators already carry sensors (`SpecModulator._sensors`,
  `spec.py:696`); warmth/fire entities merely *choose* `abstract: true`
  (capability-only). "Ambient field" = a readable scalar on an environment
  entity, which is structurally what `heat_output` already is.

So this is **a new reader riding existing data + existing writers**, not a new
data model. It does NOT repurpose `CouplingSpec` (drive→drive, intra-body) or
`ModulationSpec` (system→drive-parameter) — those are the wrong axis
(cross-entity ambient exposure is a genuinely new, small axis, closest to an
*inverted, passive* `target_effect`).

## What it RIDES (do not rebuild)

| Existing | File:line | Role in this feature |
|---|---|---|
| `_apply_sensor_deltas` | `embodiment/tool_bridge.py:39-123` | canonical clamping/logging writer for `{sensor: delta}` into agent `vital_metrics` (handles `"cold"` + `"arms.thermal"`). **A push, not a drift** — safe re: no-double-drift. Use verbatim for the exposure write. |
| `EntityMap.list_scene_entities()` | `embodiment/entity_map.py:119` | enumerate scene entities each turn |
| Entity/modulator sensor `read()` | `sem.py:370`, `spec.py:675` | read the already-declared `heat_output` field value |
| Drive-spec + pain machinery | `body.py:218-278` | once exposure moves the drive sensor, existing pain→PainBus→NAc fires unchanged — no new pain path |
| Auto-sense §1.15 | `runtime/agent_loop.py:1306-1430` | LLM-primary hook; already holds an `EntityMap` and already iterates scene entities; inject an exposure summary into `StructuredContext.auto_sense_context` (`:2936`) |
| Inert `heat_output`/`fuel` + warmth `self_effect` magnitudes | `_data/components/items/cradle_fire_pit.yaml`, `warmth_*.yaml` | the field-strength data model — the proof the model is done |

## What it must ADD

1. **Field-query helper** — "scene entities exposing an ambient field of
   type T" (filter over `list_scene_entities()` by sensor/modulator name or a
   declarative field-type tag). ~10 lines; no existing "find entities by
   sensor/modulator type" API (resolution is name/path only today).
2. **A per-turn environmental-sense pass** (a shared function): enumerate
   matching scene entities → read their ambient field sensor(s) → map
   field-strength → agent drive-sensor delta → `_apply_sensor_deltas` onto the
   agent body → let the *existing* `evaluate_failures()` react. This is a
   **push before the existing tick**, never a new drift.
3. **Dual-path wiring (load-bearing).** Call the pass from **both** §1.15
   (LLM-primary) **and** near `agent_loop.py:854` (substrate-primary), via one
   shared helper. Wiring only in §1.15 silently no-ops in substrate-primary —
   which is the Exp 42 harness path — so a §1.15-only version would be a
   subtle mode-asymmetry bug.
4. **Declarative field→drive mapping that does NOT touch frozen sem.py
   dataclasses.** Carry "this sensor is an ambient field of type T; exposure
   magnitude→delta on agent drive sensor S" via entity/modulator YAML
   `metadata` (flows into `entity.metadata`, `spec.py:389`, forward-compat) +
   a new *non-frozen* config type, or a sensor-name/tag convention. Do NOT add
   fields to `HomeostaticDriveSpec`/`EntropicDriveSpec`/`AffordanceSchema`/
   `CouplingSpec`/`ModulationSpec` (all SHAPE-FROZEN at 1.0; even an optional
   trailing field needs the CC3 gate).
5. **An exposure model** — how field strength reaches the agent. Simplest v1:
   binary in-scene = exposed (present-in-`list_scene_entities` → full field).
   A distance/radius model is a deliberate later extension, not v1 (there is
   no spatial position in the entity model today; don't invent one).

## Invariants it must NOT break

- **No-double-drift** (`.github/workflows/test.yml:552-564`, CI grep):
  never call `tick_vital_drift`. Exposure is an external push via
  `_apply_sensor_deltas`; the single existing `evaluate_failures()` per turn
  handles drift + reaction.
- **Substrate-primary tick preservation** (`agent_loop.py:851-856`;
  **Exp 42 GRADUATED on the current dynamics**): keep `evaluate_failures()`
  firing once per proposal; add the exposure push *before* it, no second
  embodiment tick. Any behavioral change to substrate-primary must be a
  deliberate, measured decision (re-run the Exp 42 guard), not a side effect.
- **Order-of-operations** (`tool_bridge.py:369-388`): sensor writes before the
  failure evaluation that reads them.
- **Frozen sem.py specs / CC3**: prefer YAML `metadata` + a non-frozen
  dataclass over mutating frozen specs.
- **C4 abstract-modulator rule** (`spec.py:759-765`): keep the field on
  *entity-level* sensors (as `fire_pit` already does with `heat_output`) so
  warmth/fire modulators can stay `abstract: true`. Adding sensors to a
  currently-abstract modulator forces dropping its `abstract` marker.
- **Delta-attribution scope** (Exp 42 / B8, `tool_bridge.py:199-255`): a
  passive ambient exposure has **no invoking tool**, so its induced breaches
  flow through the parallel `evaluate_failures → _publish_drive_pain →
  PainBus → create_pain_nac_subscriber` channel (NOT delta-attributed). Decide
  deliberately how ambient-induced pain should attribute to NAc — this is the
  subtlest design question in the plan and interacts with the pending
  `transition_based_drive_pain.md` work.

## Scope split: environmental (this) vs metabolic (separate)

This handles **field-driven** drives (thermal: cold ← cold environment /
warmth source, sensed each turn). It does NOT replace the **metabolic** clock
for purely internal drives (hunger/thirst rise with time regardless of
environment). Those still want a per-cycle loop tick (a smaller, separate
change: own the homeostatic clock in the agent loop rather than in
`evaluate_failures`-on-tool-execution). For the cradle/thermal experiments,
thermal is the whole drive, so environmental sensing is sufficient there.

## Phases

- **P1 — reader + exposure (thermal only).** Field-query helper + per-turn
  pass + dual-path wiring + binary in-scene exposure. Wire `cradle_fire_pit`'s
  `heat_output` → agent `cold` relief and `cradle_cool_air` → `cold` increase.
  Metadata-driven field→drive map. No frozen-spec edits.
- **P2 — behavioral validation.** Re-run a cradle seed (both modes): confirm
  cold now tracks *sensed proximity to warmth vs cold* (rises away from heat,
  falls near it) WITHOUT requiring an affordance call; confirm Exp 42
  substrate-primary preference is preserved (regression guard).
- **P3 — generalize + attribution.** Non-thermal fields (toxicity/radiation)
  via the same mechanism; resolve the ambient-pain NAc-attribution question
  with `transition_based_drive_pain.md`.
- **P4 (optional, later) — metabolic clock decouple + spatial exposure.**

## Regression guards

- Unit: field-query helper (finds thermal-field scene entities, ignores
  self/non-field), exposure-write clamps + routes through
  `_apply_sensor_deltas` (asserts NO `tick_vital_drift` call), dual-path
  helper invoked from both §1.15 and substrate-primary tick.
- Behavioral: cradle seed shows sensed-proximity-driven cold WITHOUT
  affordance execution; Exp 42 substrate-primary preference unchanged
  (the graduated-result guard).
- CI: the existing `tick_vital_drift(` grep already forbids a second drift
  site — this feature must stay push-only to pass it.

## Non-goals

- No spatial/coordinate model (v1 is binary in-scene exposure).
- No repurposing of `CouplingSpec`/`ModulationSpec`.
- No change to the metabolic clock in this plan (separate, smaller change).
- Does NOT unblock Exp 44 (scene-control is that blocker).
