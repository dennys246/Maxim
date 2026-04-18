# AgentFactory Canonicalization — one door for every agent (running doc)

**Status:** Living doc. Not scheduled. Activates on trigger.
**Type:** Multi-session architectural plan.
**Parent:** [executor_bootstrap_unification.md](executor_bootstrap_unification.md) (must ship first).
**Related:** [sem_execution_hook.md](sem_execution_hook.md) (Stage 2b deferral lives here too).

## Goal

Make `runtime/agent_factory.py::AgentFactory` the **only** door for constructing an agent in Maxim. Every entry point (CLI, sim orchestrator, sim interactive, Reachy, public Python API, future embodied robots) constructs its agent(s) by calling `AgentFactory.create_agent(config)`. The five-bootstrap-paths-with-drift problem becomes "one bootstrap path with config variants."

## Why this matters

The repeated bug pattern that drove `executor_bootstrap_unification.md` is "five entry points each hand-roll the bio pipeline; one of them forgets to wire X." The bootstrap-unification plan fixes the *local* form of the bug by making `build_executor` enforce the bridge invariant at the constructor level. AgentFactory canonicalization fixes the *global* form by making sure there is only one place where `build_executor` is ever called.

The two plans are complementary:

- **Bootstrap unification** = structural floor (one Executor cannot exist without an explicit bridge decision).
- **AgentFactory canonicalization** = structural ceiling (one constructor exists, period; drift cannot accumulate because there are no parallel constructors to drift apart).

Bootstrap unification ships first because it is small (~550 LOC, single PR) and unblocks this plan by ensuring that whatever AgentFactory ends up calling already has the bridge invariant baked in. AgentFactory canonicalization is large (multi-PR, multi-session) and replaces the bootstrap shells in CLI, orchestrator, Reachy, and the Python API one at a time.

## Audit (running — fill in as call sites are touched)

### Today's agent entry points (the doors that need to converge)

| # | Entry point | Call site | Builds | Currently uses AgentFactory? |
|---|---|---|---|---|
| 1 | `maxim --llm X` non-sim CLI | `cli.py:1131` area | one MaximAgent | No — hand-rolled |
| 2 | `maxim --sim agent` | same as #1 + `_sim_source` | one MaximAgent (cli executor) | No — hand-rolled |
| 3 | `maxim --sim interactive` | `cli.py:1397` | one MaximAgent (cli executor) | No — hand-rolled |
| 4 | `maxim --sim <DM yaml>` | `simulation/orchestrator.py:start_simulation_mode` | TWO agents: orchestrator + AUT | No — independent hand-roll |
| 5 | `maxim --sim "test X"` (generative) | same as #4 | same | No |
| 6 | Reachy embodied runtime | `embodied_runtime/agentic_runtime.py:325` | one MaximAgent + Reachy bridge | No — hand-rolled with `pain_detector` legacy |
| 7 | `maxim.create.agent(...)` public Python API | `api.py:436` | one MaximAgent (headless) | No — hand-rolled, no PainBus |
| 8 | `simulation/tools.py` sub-AUT | `tools.py:793` | sub-Executor inside a tool | No — sandboxed |
| 9 | `maxim.create.agent("name", entity_ref="...")` (multi-agent) | `api.py` factory path | N agents via AgentFactory + AgentPool | **Yes — but only this path** |

**Eight of nine entry points do not use AgentFactory.** AgentFactory exists but it is one of N builders, not the canonical one. This plan flips the ratio.

### What AgentFactory currently does (read this before designing migration)

`runtime/agent_factory.py::AgentFactory.create_agent(config)` builds an `AgentInstance` with isolated bio-systems (Hippocampus, NAc, ATL) for multi-agent NPC scenarios. It does NOT currently construct an Executor — the `AgentPool.run_turn` path constructs that per-turn from the agent's tool registry. This is the design question that blocks `sem_execution_hook.md` Stage 2b: per-turn Executor + per-instance bridge means the bridge has to live on `AgentInstance`, not on the Executor.

The bootstrap unification plan does NOT solve this — it only ensures that whatever Executor exists has an explicit bridge decision. The Executor lifetime question (per-turn vs per-instance) is still open and is **the central design question for this plan**.

## Three design options for Executor lifetime

### Option Z1 — Per-instance Executor (one Executor lives on AgentInstance)

- Build the Executor once in `AgentFactory.create_agent`.
- `AgentPool.run_turn` reuses the same Executor across turns.
- `ToolPainBridge` lives on the Executor (current shape) — one bridge per agent, isolated from other agents.
- **Pro:** matches the bootstrap unification plan exactly. `build_executor(registry, pain_bus=instance.pain_bus, ...)` once, done.
- **Con:** Executor state leaks across turns if there's any per-turn configuration (allowed_tools, supervision policy). Need to confirm none exists.

### Option Z2 — Per-turn Executor (current AgentPool shape)

- Build the Executor fresh each turn in `AgentPool.run_turn`.
- The PainBus + bridge live on `AgentInstance` (persistent across turns).
- Each per-turn `build_executor` call passes the `AgentInstance.pain_bus` so the new Executor gets a new bridge attached, but the underlying NAc and Hippocampus subscriptions are the same persistent objects.
- **Pro:** matches current AgentPool shape; minimal disruption.
- **Con:** every turn re-attaches a fresh ToolPainBridge with empty `_pending_tools` — if a tool is in flight across a turn boundary (does this happen?), the bridge loses pending state. Needs audit.

### Option Z3 — Hybrid (registry + bridge persistent on instance, Executor per-turn)

- `AgentInstance` holds: tool registry, PainBus, NAc, Hippocampus, ToolPainBridge.
- Each turn: `build_executor(registry, pain_bus=instance.pain_bus, ...)` returns a new inner Executor; the helper assigns `instance.tool_pain_bridge` (the same bridge across turns) onto it.
- **Pro:** Executor is per-turn (matches current shape), bridge is per-instance (correct semantics for `_pending_tools` continuity).
- **Con:** `build_executor` needs a way to accept a *pre-built* bridge instead of constructing one. New parameter.

**Decision needs to be made when this plan opens.** My current lean is **Z3** — it preserves the current per-turn Executor shape while making the bridge lifecycle correct. But this needs a re-read of `AgentPool.run_turn` to confirm the per-turn Executor isn't load-bearing for some other reason.

## Migration plan (multi-PR, multi-session)

This plan ships in **independent stages**, one entry point at a time, so each migration can be reviewed and tested in isolation. After every stage the codebase is in a consistent state — partial migration is allowed.

### Stage F1 — design pass + AgentInstance shape

- Re-read `agent_factory.py` + `agent_pool.py` + `agent_instance.py` (or wherever `AgentInstance` lives).
- Decide Z1/Z2/Z3 with a written rationale.
- Extend `AgentConfig` to carry the bio-config (PainBus, NAc, Hippocampus, SCN, optional entity_ref).
- Extend `AgentInstance` to hold the persistent bio-systems and (per Z3) the bridge.
- No call site migrations yet.

### Stage F2 — CLI migration (largest single migration)

- Replace the cli.py hand-rolled bootstrap with `AgentFactory.create_agent(config)`.
- The five separate-but-similar bootstrap blocks in cli.py (non-sim, sim agent, sim interactive, Reachy, public API) collapse into one factory call with config variants.
- Removes the `_cli_pain_bus` / `_sim_pain_bus` duplication entirely.

### Stage F3 — Sim orchestrator migration (the hardest one)

- `simulation/orchestrator.py` builds **two** agents (orchestrator + AUT). Migrate to two `AgentFactory.create_agent` calls with different configs.
- AUT config gets the bio-pipeline; orchestrator config opts out (`pain_bus=None`).
- Drops ~150 LOC of hand-rolled bootstrap.

### Stage F4 — Reachy migration

- `embodied_runtime/agentic_runtime.py` migrates. The `pain_detector` legacy path stays (Reachy doesn't use PainBus) — `AgentConfig` accepts either.

### Stage F5 — Public API migration

- `api.py::maxim.create.agent` headless mode migrates. The "headless agents have no PainBus" question gets a real answer (probably: opt-in via config flag, default off).

### Stage F6 — Test review (HARD ENFORCEMENT)

**This stage is non-optional.** When the bootstrap-unification plan ships, it will surface a wave of test failures because every test that calls `build_executor` without `pain_bus=` becomes a TypeError. Most of those tests will get a quick `pain_bus=None` opt-out as part of that plan. **This stage revisits every one of those tests** and asks: "should this test have been exercising the bio-pipeline cascade all along?"

The repeated bug pattern is partly a test-coverage failure: every silent-no-op fix surfaced because someone ran the system live, not because a test caught it. Tests caught zero of the three SEM execution hook bugs. That is a structural test-suite gap that this stage exists to close.

**Stage F6 explicit deliverables:**

1. **Test surface audit** — every test in `tests/unit/` and `tests/integration/` that builds an Executor or instantiates a MaximAgent. Categorize: (a) opted out of bio-learning legitimately (sandbox tests, pure-mechanism tests); (b) should be opted in but currently isn't (silent gap); (c) already opted in.
2. **Stress test fixtures** — for each agent entry point that AgentFactory now serves, write a fixture that runs the agent through a full tool-failure cascade and asserts NAc learns NEGATIVE. Five fixtures, one per entry point. This is the "make tests catch the next instance of this bug class" deliverable.
3. **Cross-agent isolation tests** — multi-agent scenarios where two agents construct independent bridges and one agent's tool failures must NOT pollute another agent's NAc. The current `_pending_tools` dict is bridge-scoped; this test enforces that contract structurally.
4. **Per-turn vs per-instance Executor lifecycle tests** — whichever Z option won, a test that exercises the chosen lifecycle (e.g., for Z3: "the same bridge persists across two turns; a tool started in turn N completes in turn N+1 and attribution still works").
5. **Hard CI gate** — a grep check (or AST check) that no `Executor(...)` constructor call exists outside `runtime/bootstrap.py::build_executor` and `tests/`. Same shape as the existing `urllib.request.urlopen` and `_MaximPeerBackend` retry-loop CI gates.
6. **Pre-merge review round on the test surface itself** — Executor-lens reviewer runs the new tests, intentionally introduces one of the three historical bugs (e.g., comments out the `pain_bus=...` arg in cli.py), and confirms a test fails. If no test fails, the test surface is incomplete.

This stage is **load-bearing** for the plan's value. Without it, AgentFactory canonicalization is just a refactor; with it, the next bug in this class becomes a TypeError or a red test, not a silent no-op found six months later in production.

## Trigger conditions for opening this plan

This is a running doc, not a scheduled plan. Open it when one of these hits:

1. A **6th agent entry point** is proposed (currently 5: cli, sim orchestrator, Reachy, public API, sim interactive shares cli's executor). A 6th means we've crossed a complexity threshold.
2. The **next bridge-wiring or pipeline-construction bug** surfaces despite the bootstrap unification fix. That means the structural floor isn't enough and we need the ceiling.
3. **`sem_execution_hook.md` Stage 2b activates** — Stage 2b is "wire `AgentFactory.create_agent` to support `entity_ref` end-to-end" and it touches the same files this plan touches. If Stage 2b ships standalone, it should at minimum complete Stage F1 (the design pass) so the rest of this plan is downhill.
4. **Multi-agent NPC scenarios** become a real workload — the current `AgentPool` is exercised in tests but not heavily in production. When a real campaign needs N concurrent agents with isolated bio-systems, this plan goes from "nice to have" to "required for correctness."
5. **Opportunistic alignment** — if a session is touching `agent_factory.py` / `agent_pool.py` for any other reason and the marginal cost of canonicalizing is small, take the win.

## Open questions (resolve at plan-open time)

1. **Z1 vs Z2 vs Z3** — the central design question. Need a re-read of `AgentPool.run_turn` to confirm whether the per-turn Executor is load-bearing.
2. **Headless API agents and PainBus — MUST be addressed in this plan, not deferred again.** The `executor_bootstrap_unification.md` audit found that `api.py::maxim.create.agent` headless mode constructs no PainBus and wires no bridge — fourth instance of the same silent-no-op bug. The bootstrap unification PR will pass `pain_bus=None` with an inline TODO pointing here. **This plan must, during Stage F5 (or earlier), construct a real PainBus for headless API agents so pain signals flow correctly.** The decision left for plan-open is whether bio-learning is on-by-default with an opt-out flag, or off-by-default with an opt-in flag — but the *infrastructure* (PainBus + subscribers + bridge) must exist so the user can flip the flag without re-architecting. The cross-session learning gate for 1.0 depends on this path working end-to-end.
3. **Orchestrator agent's own bio stack** — the bootstrap unification audit found `simulation/orchestrator.py::orch_executor` has no bridge; the chosen migration is `pain_bus=None` (passing the AUT's bus would cross-contaminate AUT learning, breaking sim mode's isolation invariant). **This plan should re-examine whether the orchestrator agent should grow its own `orch_pain_bus` + `orch_nac` so it can learn from its own failures without polluting the AUT.** Today nothing consumes orchestrator learning, so the question is "is there a future experiment that wants this?" Add as a Stage F3 design point.
4. **`maxim.load.agent(...)`** — restoring an `AgentInstance` from persisted state needs to restore the bio-systems too. Out of scope for this plan or in scope?
5. **Sim orchestrator's two-agent shape** — the AUT/orchestrator split is somewhat unique. Does AgentFactory grow a "sub-agent" concept, or do we just call `create_agent` twice with different configs?
6. **Reachy's `pain_detector` legacy** — keep it, or migrate Reachy to PainBus and drop the legacy path?
7. **Cross-cutting: does this plan supersede `sem_execution_hook.md` Stage 2b?** Probably yes — Stage 2b becomes a sub-stage of this plan. Confirm at plan-open time.

## Wave G — Game / External Host integration (folded from game_npc_integration.md, 2026-04-18)

**Trigger #4 activated:** game NPC use case = multi-agent NPC scenarios as a real workload. The execution + architecture review (2026-04-18) found that `AgentPool.run_turn()` skips the entire bio-pipeline — no Executor, no PainBus, no ToolPainBridge. An NPC that only speaks can't learn from its actions. The gap is structural: `create_npc_agent()` wires Hippocampus/NAc/ATL/MemoryHub but never creates an Executor ([agent_factory.py:83](../../src/maxim/runtime/agent_factory.py#L83) — `executor` stays `None`).

The G-wave extends the factory (after F-wave collapses the 8 hand-rolled entry points) to produce agents with full execution capability for external hosts (game engines, web apps, virtual worlds).

### Stage G1 — Wire Executor + bio-pipeline into NPC agents

`create_npc_agent()` gains `executor_enabled: bool = False`. When `True`:
- Calls `build_bio_stack()` + `build_executor(pain_bus=bio.pain_bus, nac=bio.nac)` via canonical builders.
- `run_turn()` checks for executor, parses LLM response for tool calls, executes them with ToolPainBridge attribution.
- `AgentPool` gains `start_session()` / `end_session()` for game load/save boundaries.
- `executor_enabled=False` default preserves backward compatibility for DM NPC parties.

Depends on F1 (Z1/Z2/Z3 Executor lifetime decision). ~250 LOC.

### Stage G2 — HostContext protocol

Thin protocol abstracting the environment agents run in:
```python
class HostContext(Protocol):
    percept_source: PerceptSource
    action_sink: ActionSink
```
Three implementations: `TerminalHost`, `SimulationHost`, `GameHost`. `PerceptSource` / `ActionSink` protocols already exist — this formalizes them as the host boundary. ~180 LOC.

### Stage G3 — Emotional state readout

`TurnResult` gains `emotional_state: dict[str, float] | None` — populated from NAc `_reward_bias`, PainBus intensity, FearAgent threat level, SEM sensor values. Game engines use this for facial animation, behavior trees, dialogue tone. Read-only — evolves through bio-pipeline, not set directly. ~110 LOC.

### Stage G4 — Async tool dispatch

Tools can return `ToolOutput(deferred=True)`. Game engine calls `AgentPool.resolve_deferred(agent_id, tool_call_id, result)` when animation/physics completes. ToolPainBridge links deferred result to pending NAc event by `(tool_name, invocation_id)`. Configurable timeout fires PainBus signal on unresolved deferrals. ~220 LOC.

### Stage G5 — Memory backend protocol

Extract save/load into `MemoryBackend` protocol with `FileMemoryBackend` (current) + `InMemoryBackend` (games managing their own saves). `AgentFactory` accepts optional `memory_backend`. ~180 LOC.

### Relationship to F-wave

F-wave collapses 8 hand-rolled entry points into one factory. G-wave extends that factory for external hosts. Same architectural arc, one design pass, no conflicting Executor lifetime decisions. If F1 ships first, G1 is simpler (just add `executor_enabled` to the canonical factory). G-wave adds ~940 LOC to the total scope.

### What G-wave does NOT include

- Game engine SDK/bindings (Unity/Unreal/Godot) — builds on top of this.
- Multi-agent shared memory (Mother Maxim concept, post-1.0).
- Real-time continuous perception (needs DefaultNetwork reactive path, embodied runtime).
- Interactive/display layer (orthogonal — see [interactive_experience_031.md](interactive_experience_031.md)).

## Notes

This doc is intentionally light on implementation detail because the design questions matter more than the LOC estimate at this stage. When this plan opens for real, Stage F1 produces a written design rationale that turns the open questions into decisions, and the rest of the stages get sharpened from there.

**Estimated total scope** (rough, pre-design-pass): ~2500-3500 LOC across `src/maxim/` (F-wave ~1500-2500 + G-wave ~940) + ~1000 LOC of new tests in Stage F6. Multi-PR (one per stage). Multi-session.

**Doc + memory refinement scope** (mirrors the bootstrap unification plan):
- `CLAUDE.md` invariant line: "AgentFactory.create_agent is the canonical agent constructor; no entry point hand-rolls a MaximAgent."
- New memory file `feedback_one_door_for_every_agent.md`.
- Update every plan that mentions an entry point to point at AgentFactory.
- Sweep `docs/embodiment_guide.md`, `docs/user/cli-reference.md`, `docs/api.md` for stale references.
