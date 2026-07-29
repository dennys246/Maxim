# Deferred — extract a shared `build_loop_stack()`

**Status:** Deferred, drafted 2026-07-29. Surfaced by the Architecture lens during the talk-mode pre-merge review (PR #438) and confirmed by a call-site count during the session wrap.

**Revive trigger:** the **next** time someone hand-assembles a loop stack — i.e. a 9th `run_agentic_loop(` call site, or a change that must be applied to more than two existing sites at once. Also revive if a bug is traced to one site having a component another site wires.

---

## The finding

`run_agentic_loop` has **9 call sites across 8 modules**, each hand-assembling the same stack (`MaximAgent`, environment, `RuntimeState`, `build_memory`, `build_decision_engine`, `AutonomyController`, `LLMWorker`) with no shared builder:

```
src/maxim/api.py                          headless api.run
src/maxim/cli.py                          CLI non-sim agent
src/maxim/runtime/agent_factory.py
src/maxim/simulation/orchestrator.py      AUT + orchestrator (2 sites)
src/maxim/simulation/tools.py             sub-AUT
src/maxim/simulation/interactive.py
src/maxim/embodied_runtime/agentic_runtime.py   Reachy
src/maxim/console/handle.py               talk (added 2026-07-28)
```

This is the same shape as the `build_executor(pain_bus=...)` lesson in CLAUDE.md — repeated hand-wiring whose failure mode is **silent**: a forgotten component does not raise, it just makes that entry point quietly less capable than its siblings.

## Why it is not cosmetic — the talk case

Talk shipped missing pieces the sim AUT wires, and the omission was invisible until someone diffed the two call sites:

| omitted | consequence |
|---|---|
| `bio_enrichment_pipeline` | `agent_loop` gates the **entire** enrichment block on `bio_enrichment_pipeline is not None or thought_gate is not None` — no hippocampal recall, no ATL concepts, no cerebellum predictions in the prompt |
| `thought_gate` | no salience-scored "should I think" decision |
| both | multi-cycle deliberation is gated on the pipeline too → strictly single-shot per turn |
| `agent.wire_memory_hub(...)` | `MemoryAgent → hippocampus` never connected; promotion source + deletion callback unregistered |
| `imagination_trigger` | novel nouns never become SEM entities |
| `default_network` | `enabled=False`; no reactive behaviors |
| `llm_worker.acting_coach` / `.is_embodied` | an embodied handle still gets the disembodied prompt |

The first four were fixed in PR #438's review fold — **by hand, at one site**. The remaining three are still absent from talk, deliberately (headless), but nothing records that decision structurally, so the next copy inherits the ambiguity.

## Shape of the fix

```python
# runtime/bootstrap.py
def build_loop_stack(
    instance,                 # AgentInstance — supplies bio_stack/executor/hippocampus/memory_hub
    *,
    percept_source, action_sink,
    autonomy,                 # caller owns the permission posture
    llm_worker,
    environment=None,         # default: FileSystemEnv over the agent home
    with_default_network=False,
    with_imagination=False,
) -> LoopStack:               # frozen dataclass; splat into run_agentic_loop
```

Reads `instance.bio_stack` for the enrichment pipeline + thought gate and calls `wire_memory_hub` **by construction**, so forgetting them stops being possible. Optional subsystems are explicit keyword decisions (the `build_executor(pain_bus=...)` pattern) rather than silent absences.

## Why it is deferred, not done

Migrating all 9 sites touches the sim orchestrator, which every experiment in `docs/experiments/` depends on. That is its own plan with its own pre-merge review round — not a tail-end refactor. Doing it partially (builder used by one site, others left) would *add* a mechanism without removing the duplication, which the front-gate rule in CLAUDE.md explicitly warns against.

**Recommended sequencing when revived:** land the builder + migrate the two console sites first (talk, and any future headless mode) since they are the least experiment-coupled; then `api.run`/`cli.py`; leave the orchestrator's AUT for last, behind a sim-behavior equivalence check.

## Regression guard

None yet — that is the point of the deferral. When revived, the guard is a test asserting every `run_agentic_loop` call site routes through `build_loop_stack` (a CI grep in the `write_mesh_config` allow-list style), so a 10th site cannot be hand-assembled.
