# A body reading's provenance is only knowable where it was written

**Established:** 2026-10-24, the T8 (b) design review of `docs/plans/thalamic_relay.md` (wiring lens, D1 and D2;
environment lens, D2; reports in [../plans/reviews/grounding_gl3/](../plans/reviews/grounding_gl3/)). Code read at
`d9dffe6e`.

## The fact

Experienced writes, narrated writes and apparatus writes all land in the same body sensor slots, and nothing records
which kind each one was:
- **Experienced:** the Minecraft pump, DoA, the AUT's own affordances.
- **Narrated:** `SetEntitySensorTool`, `DamageComponentTool`, and `OrchestratorActorTool` through `_apply_sensor_deltas`, the helper the AUT's own actions use.
- **Apparatus:** the water trial's RCON rescue and heal, which arrive back through the pump as `world_set_axis(owner="minecraft_bridge")`, and the cradle mother.

`vital_metrics` is a plain dict. `world_set_axis(owner=)` is a log tag, not a stored record. So when a reading is
encoded or transduced later, the encoder cannot tell who wrote it.

On llm-primary, `_loop_live_tick` → `evaluate_failures` can transduce a narrator's write before any deferred
"narrated" path sees it. The event it mints carries no provenance, so every consumer treats it as the AUT's own experience.

## Why it bites

Two traps:
1. **The credit is not where you check it.** The body world credits through `record_outcome`'s cluster reward bias
   and the PainBus subscribers (`record_cluster_fear` and the others), not through `credit_node`. A provenance check
   on `credit_node` alone never touches the path the Exp 58–62 rows earned on.
2. **The "loop" of the survival rows is a script.** `scripts/survival_world/water_trial.py` and its siblings call
   `propose_via_substrate` and `_encode_current_clusters` directly. A mechanism carried only on `LoopRun` never
   reaches Exp 58/60/61/62, R3 or B1.

**Rule:** stamp provenance at the WRITE (the last writer of each sensor, stored), and read it wherever an event is
minted or a reading is encoded. The writer kind belongs beside `Entity.vital_metrics`, because the writers
(`_apply_sensor_deltas`, `_write_sensor`, `SpecModulator.apply_damage`, `world_set_axis`) hold only an Entity: the
narrator's ephemeral tool writes to the AUT's Entity from its own scene Embodiment. Every write goes through one
setter that takes the kind, because a raw `vital_metrics[...] =` calls no writer and silently keeps the slot's old
kind; a restore (`Entity.from_dict`) writes the persisted kinds. The reader resolves a sensor's kind through the same
resolver as its value (`sem.py::_resolve_sensor_slot`), so a modulator's sensor reads the modulator's map. A slot
with no writer yet is missing and reads `legacy`, never `experienced` (owner G29). Per-body mechanisms that every
caller (loop, script and red gate) must reach, such as the receptor registry, hang off the Embodiment through one named
binder that builds them on first use from the root entity tree's declared sensors. Plan rows TR21–TR23 and TR29 in
[../plans/thalamic_relay.md](../plans/thalamic_relay.md) (G24, G29; pre-merge review Executor B1, Wire B1 and B2;
delta round Wire D-1 to D-3). **Enforced by:** nothing yet. The checks: red gate (g), built in GL3.B0 (#1201), flips
at GL3.B1; [#1450](https://github.com/dennys246/Maxim/issues/1450) owns its green control (g′), (g)'s docstring and
the new strict red gates (g2), (g3) and (h); mechanization backlog M39 (the relay as the one encode route) and M78
(no raw `vital_metrics[...] =` outside the setter). All land before or with GL3.B1.
