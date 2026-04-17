# Cerebellum Backend Activation

**Status:** COMPLETE — absorbed into [sem_learning_loop.md](sem_learning_loop.md) Stage 1 (shipped 2026-04-17).
**Scope:** ~150-200 LOC. Wire existing Cerebellum infrastructure into production entry points.
**Parent:** None (standalone). Companion to [substrate_valence_annotation.md](substrate_valence_annotation.md) Stage 4.
**Depends on:** `build_bio_stack` (Wave 3, SHIPPED).

## Motivation

The Cerebellum forward-model system is fully implemented: `CerebellumConfig`, prediction/training loops, persistence (`save`/`load`), motor programs, and the `CerebellumModulator` backend with `reaction_bus=` wiring. The `cerebellum_modulator_factory` builds CerebellumModulator instances that predict SEM affordance outcomes and fall back to LLM when uncertain.

None of this runs in production. The factory has zero callers. The `cerebellum=` parameter on `build_executor` is accepted but never populated. The only site that constructs a Cerebellum is the simulation AUT path (`orchestrator.py:493-503`).

### What the Cerebellum does when activated

When an agent interacts with a SEM entity (e.g., swings a rusty sword), the flow today is:

1. `ModulatorAffordanceTool.execute()` calls the modulator
2. The modulator is a `SpecModulator` stub → delegates to LLM for every invocation
3. LLM returns predicted sensor changes → applied to entity

With Cerebellum activated:

1. Same tool call
2. `CerebellumModulator.execute()` checks confidence for `(entity, modulator, affordance, params)`
3. **Confident:** returns cached prediction (no LLM call, ~0ms vs ~2-5s)
4. **Not confident:** falls back to LLM, trains Cerebellum on the result
5. On failure: emits `Reaction(kind="pain", valence=NEGATIVE)` to `reaction_bus` → captured into episode → annotates Hebbian edges with valence (valence annotation Stage 1, just shipped)

The Cerebellum replaces repeated LLM calls with learned forward models. This matters most on the Reachy robot path where motor-control latency is critical (2-5s LLM roundtrip vs ~0ms cached prediction).

## What exists (audit 2026-04-17)

| Component | File | Status |
|---|---|---|
| `Cerebellum` class + config | `embodiment/cerebellum.py` | Complete |
| Persistence (save/load/export/import) | `embodiment/cerebellum.py:423-509` | Complete |
| Motor programs | `embodiment/cerebellum.py:513-708` | Complete |
| `CerebellumModulator` | `embodiment/backends/cerebellum_modulator.py` | Complete |
| `cerebellum_modulator_factory` | Same file, line 241 | Complete (accepts `reaction_bus=`) |
| `build_executor(cerebellum=)` | `runtime/bootstrap.py:300` | Accepted, forwarded to `generate_tools_for_entity()` |
| `attach_backends(modulator_factory=)` | `embodiment/spec.py:494` | Complete |
| Sim AUT construction | `simulation/orchestrator.py:493-503` | Reference implementation |
| **CLI non-sim activation** | `cli.py:1095-1186` | **GAP** — no Cerebellum |
| **Reachy activation** | `embodied_runtime/agentic_runtime.py:122-323` | **GAP** — no Cerebellum |
| **BioStack integration** | `runtime/bio_stack.py` | **GAP** — no Cerebellum field |

## Design

### Option A: Integrate into BioStack (recommended)

Add Cerebellum to `BioStack` as an optional system, constructed when `persistence_dir` is provided. This follows the Wave 3 pattern exactly — single construction site, automatic persistence path derivation, memory_hub wiring.

```python
# bio_stack.py Step 4c (after PainBus, before DefaultNetwork)
cerebellum = None
try:
    from maxim.embodiment.cerebellum import Cerebellum, CerebellumConfig
    cerebellum = Cerebellum(config=CerebellumConfig(
        persistence_path=str(p / "cerebellum.json") if p is not None else None,
    ))
    memory_hub.cerebellum = cerebellum
except Exception:
    logger.debug("Cerebellum not available")
```

Then callers pass `bio.cerebellum` to `build_executor(cerebellum=bio.cerebellum)`.

**Pros:** Single source of truth. Persistence managed automatically. All production paths get it for free.
**Cons:** Cerebellum is only useful when SEM entities are loaded (embodiment mode). Constructing it unconditionally is wasteful for headless CLI agents that never use embodiment.

### Option B: Construct only when embodiment is active

Gate Cerebellum construction on the presence of an `entity_ref` or `component_registry` at the call site. This matches the current `build_executor` pattern where embodiment-specific parameters are only passed when `--embodiment` is set.

**Pros:** No wasted construction. Matches the opt-in embodiment pattern.
**Cons:** Wiring is per-site instead of centralized. Risk of the N-sites-drift bug class that Wave 3 was designed to eliminate.

### Recommendation: Option A with lazy initialization

Add Cerebellum to BioStack but with `cerebellum: Cerebellum | None = None` (already the right default). In `build_bio_stack`, gate construction on `persistence_dir is not None` — in-memory/test paths skip it. The Cerebellum starts empty (no trained models) and only grows as the agent interacts with SEM entities.

## Stages

### Stage 1 — BioStack integration (~80 LOC)

1. Add `cerebellum: Any = None` field to `BioStack` frozen dataclass
2. Construct `Cerebellum(config=CerebellumConfig(persistence_path=...))` in `build_bio_stack` after MemoryHub, gated on `persistence_dir is not None`
3. Wire `memory_hub.cerebellum = cerebellum`
4. Pass `bio.cerebellum` to `build_executor(cerebellum=...)` at all 4 production call sites (cli.py non-sim, orchestrator.py AUT + orch NPC, agentic_runtime.py Reachy)
5. Load persisted state on startup if `cerebellum.json` exists

### Stage 2 — Modulator factory wiring (~50 LOC)

The `cerebellum=` parameter in `build_executor` is forwarded to `generate_tools_for_entity()`. Verify that `generate_tools_for_entity` calls `cerebellum_modulator_factory(cerebellum, reaction_bus=reaction_bus)` and passes the result to `attach_backends(entity, modulator_factory=factory)`.

If the wiring doesn't exist yet in `generate_tools_for_entity`, add it. This is the "last mile" that connects the BioStack-constructed Cerebellum to the actual SEM entities.

### Stage 3 — Persistence lifecycle (~30 LOC)

1. Save Cerebellum state during session end (alongside hippocampus dump)
2. Load on session start (alongside hippocampus load)
3. Verify motor programs survive dump/load round-trip

### Stage 4 — Integration test

Sim scenario: agent interacts with a SEM entity 5 times. First interaction uses LLM fallback. By interaction 3-5, Cerebellum should be confident enough to skip LLM. Measure: (a) number of LLM fallbacks decreases, (b) prediction accuracy improves, (c) state survives persistence round-trip.

## Connection to valence annotation

With this plan shipped, the full SEM → learning pipeline is:

1. Agent interacts with SEM entity via `ModulatorAffordanceTool`
2. `CerebellumModulator.execute()` predicts or falls back
3. On failure: `_emit_failure_reaction()` → `ReactionBus` (Link 1, already wired)
4. `hippocampus.capture_reaction()` → `PendingEpisodeState.reactions` (valence annotation Stage 1, just shipped)
5. Episode close: `apply_hebbian_on_close()` annotates edges with `metadata["valence"]`
6. Future retrieval: `spreading_activation(propagate_valence=True)` carries affective signal

Stage 4 (positive valence) of the valence annotation plan also depends on this — `CerebellumModulator._emit_success_reaction()` needs a production Cerebellum to emit success reactions on confident predictions.

## Risks

1. **Import weight.** Cerebellum imports NumPy/SciPy for its internal models. These are already core deps, so no new weight.
2. **Persistence path collision.** `cerebellum.json` in the same `persistence_dir` as `hippocampus.json`. No collision risk — different filenames.
3. **Headless agents.** `api.py` headless path should keep `cerebellum=None` since there are no SEM entities to interact with. The `BioStack` default handles this.
