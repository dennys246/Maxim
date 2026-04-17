# MemoryHub Unification — required bridge wiring at construction

**Status:** **SHIPPED** (PR #136, merged 2026-04-16). `build_memory_hub(*, hippocampus, scn, nac, ec)` always calls `.connect()`. Archived 2026-04-17.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 2 of 4 (parallel-safe with `default_network_unification.md`)
**Depends on:** [pain_bus_unification.md](pain_bus_unification.md), [reaction_bus_unification.md](reaction_bus_unification.md) — MemoryHub consumes both buses, so the builders must exist first.
**Parallel-safe with:** [default_network_unification.md](default_network_unification.md) — different files, no overlap.
**Blocks:** [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Push the `MemoryHub.connect(...)` bridge wiring into the `MemoryHub` constructor itself. Today the hub is constructed with bio-system handles, then a separate `.connect(fear_agent=..., default_network=..., ...)` call wires cross-layer bridges. Forgetting `.connect()` or omitting a bridge arg silently skips that bridge.

## The repeating shape

```python
hub = MemoryHub(hippocampus=..., scn=..., nac=..., ec=...)
# easy to forget:
hub.connect(fear_agent=fear_agent, default_network=default_network, ...)
```

If `hub.connect()` is never called: the hub exists, bio-systems are present, but cross-layer signal routing (memory promotion on fear, sleep-replay triggering, etc.) is dead. **Silent failure** — no exception, no log, just missing coordination.

If `hub.connect()` is called but a kwarg is omitted: that specific bridge is dead, the rest work. Even more silent because partial coordination looks like working coordination.

## Audit (2026-04-16)

Every `MemoryHub(...)` construction site in `src/maxim/`. The audit focuses on two questions per site:

1. **Is `.connect()` called?** `.connect()` creates PlanHistoryBridge, EscalationLearningBridge, and FearCircuitBridge (always), plus SpatialMemoryBridge and SalienceMemoryBridge (when external systems are passed). If `.connect()` is never called, all five bridges are permanently `None` — **silent failure**, no exception, no log.
2. **What external systems are wired?** `.connect()` accepts `spatial`, `attention`, `salience`, `fear_agent`, `novelty_tracker`. Omitted args produce `None` bridges for that subsystem.

✅ = wired, ❌ = **silent gap** (bridge never created), ⚪ = explicit opt-out (deliberate), ➖ = N/A (not applicable to this context), 💀 = dead code.

| # | Site | File:line | `.connect()` called? | PlanHistory | Escalation | Fear | Spatial | Salience | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CLI non-sim agent | [cli.py:1110](src/maxim/cli.py#L1110) | **NO** | ❌ **GAP** | ❌ **GAP** | ❌ **GAP** | ❌ | ❌ | Hub constructed with `(hippocampus, scn, nac, ec)`. No `.connect()` call anywhere downstream. All three always-created bridges are dead. |
| 2 | Sim orchestrator AUT | [orchestrator.py:447](src/maxim/simulation/orchestrator.py#L447) | **YES** at [line 1015](src/maxim/simulation/orchestrator.py#L1015) | ✅ | ✅ | ✅ | ➖ (no sensors) | ➖ (no sensors) | `.connect(fear_agent=_fear)`. All three always-created bridges live. No spatial/salience expected in sim. Only `fear_agent` passed from sim-local FearAgent via `locals().get("fear_agent")`. **Fragile:** `locals().get` depends on variable name staying exactly `fear_agent` in the same scope — renaming silently breaks it. |
| 3 | Sim orchestrator (orch) | [orchestrator.py:698](src/maxim/simulation/orchestrator.py#L698) | 💀 **DEAD CODE** | 💀 | 💀 | 💀 | 💀 | 💀 | `MemoryHub(hippocampus=..., nac=...)` — **missing required dataclass fields `scn` and `ec`**. This is a `TypeError` at construction time, caught by `except Exception` at line 703 and logged at `debug` level. The orchestrator's memory hub is never constructed. `orch_memory_hub` is always `None`. |
| 4 | Embodied runtime (Reachy) | [agentic_runtime.py:165](src/maxim/embodied_runtime/agentic_runtime.py#L165) | **YES** at [line 692](src/maxim/embodied_runtime/agentic_runtime.py#L692) | ✅ | ✅ | ✅ | ✅ (from DN) | ✅ (from DN) | **MOST COMPLETE.** `.connect(spatial=..., attention=..., salience=..., fear_agent=..., novelty_tracker=...)` pulls systems from DefaultNetwork. Fallback at [line 705](src/maxim/embodied_runtime/agentic_runtime.py#L705) wires just `fear_agent` when DN is absent. Also wires PainBus → hippocampus externally at [line 719](src/maxim/embodied_runtime/agentic_runtime.py#L719). |
| 5 | AgentFactory (NPC agents) | [agent_factory.py:371](src/maxim/runtime/agent_factory.py#L371) | **NO** | ❌ **GAP** | ❌ **GAP** | ❌ **GAP** | ❌ | ❌ | `_create_memory_hub` returns a bare `MemoryHub(hippocampus, scn, nac, ec, atl)`. No `.connect()` anywhere in `AgentFactory`. All NPC agents created through this path have permanently dead bridges. |
| 6 | api.py headless | — | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | **No MemoryHub constructed at all.** `pain_bus=None` to `build_executor`. Explicit opt-out. User-facing API decision deferred to `agent_factory_canonicalization.md` Stage F5. |

### Test-side construction sites

`MemoryHub(...)` is constructed in 8 test files (~16 instances total). None call `.connect()` — tests exercise individual subsystems, not bridge wiring. **Migration policy:** same as `build_pain_bus` and `build_executor` — leave raw `MemoryHub()` constructor accessible for tests. Structural enforcement lives at the production door (`build_memory_hub`), not the type.

## Pre-existing silent gaps surfaced by the audit

### Gap A (Critical, in-scope) — CLI agent path has zero bridges

Site #1: `maxim --llm X` (the most common non-sim entry point) constructs `MemoryHub` but **never calls `.connect()`**. Three bridges that `.connect()` ALWAYS creates (PlanHistoryBridge, EscalationLearningBridge, FearCircuitBridge) are permanently dead:

- **PlanHistoryBridge** — plan-template lookup and plan-outcome recording. The agent has a hippocampus and NAc that could learn from plan success/failure, but the bridge that routes those signals is never created. Every call to `hub.get_plan_templates()` or `hub.record_plan_outcome()` silently returns defaults (empty list / 0.5 probability).
- **EscalationLearningBridge** — escalation threshold learning from SCN + NAc. Every `hub.should_escalate()` call returns `(False, "no_bridge")` — the agent NEVER escalates to the user because the bridge that would learn when to escalate doesn't exist.
- **FearCircuitBridge** — risk-adjustment learning from NAc. Every `hub.should_block_action()` call falls through to a static severity check — the agent never learns from false positive/negative risk assessments.

**Blast radius:** all three bridges contribute to the agent's adaptive behavior over time. Without them, the CLI agent is functionally memoryless for planning, escalation, and fear — it captures memories in hippocampus but never uses the bridges that would close the loop. The bio-systems (hippocampus, NAc, SCN) are all present and functional; only the bridge layer is missing.

**This is the L1 silent-failure-mode shape.** Two identical sites (#1 CLI, #5 AgentFactory) with the same omission. The bug never raises, never logs, never surfaces during testing.

### Gap B (Critical, in-scope) — AgentFactory NPC agents have zero bridges

Site #5: every NPC agent created by `AgentFactory._create_memory_hub` has the same gap as the CLI path — no `.connect()`, no bridges. Multi-agent sims create NPC agents through this path. All NPC cognitive bridge functions (planning, escalation, fear learning) are dead.

### Gap C (Pre-existing dead code) — Orchestrator MemoryHub always fails to construct

Site #3: `MemoryHub(hippocampus=orch_hippocampus, nac=orch_nac)` is missing required dataclass fields `scn` and `ec`. This has been dead code since it was written — the `except Exception` swallows the `TypeError` at `debug` level. The orchestrator never gets its Phase 3 cross-session memory hub.

**Fix:** add `scn` and `ec` to the constructor call so the hub actually gets constructed. This is a one-line fix but should be done in this PR since `build_memory_hub` will surface it as a `TypeError` anyway. The `.connect()` question is moot for the orchestrator hub — it's used for cross-session hippocampus/NAc state only; there are no external systems to wire bridges to.

### Gap D (Latent fragility) — `locals().get("fear_agent")` in sim orchestrator

Site #2 at [orchestrator.py:1015](src/maxim/simulation/orchestrator.py#L1015) wires `fear_agent` via `locals().get("fear_agent")`. This depends on a local variable named exactly `fear_agent` existing in the function scope at that point. Renaming the variable silently breaks the wire — `locals().get` returns `None`, `.connect(fear_agent=None)` succeeds, FearCircuitBridge still gets created (it only uses `nac`, not `fear_agent` directly) but fear-agent integration is silently dead.

**Fix:** not in-scope for this PR's structural enforcement (it's a code smell, not a construction-discipline bug), but flagged here so the pre-merge review doesn't miss it. The `build_memory_hub` migration will replace this with an explicit parameter.

## Design — `build_memory_hub(...)` (Option C, refined by audit)

The audit clarifies the decision. **Option C** (builder function) wins over Option A (fold into `__init__`) for three reasons:

1. **Staged construction is real.** Site #2 (sim orchestrator) constructs MemoryHub at line 447 but calls `.connect()` ~570 lines later at line 1015 — after FearAgent is created. Site #4 (Reachy) similarly constructs MemoryHub at line 165 and `.connect()`s at line 692. A builder that does both in one call would require all bridge deps to be available at MemoryHub construction time. The sim orchestrator builds FearAgent AFTER MemoryHub because FearAgent needs hippocampus references that come from the hub. Inverting this order is possible but is its own refactor.

2. **Matches the established pattern.** `build_executor(*, pain_bus)` and `build_pain_bus(*, hippocampus, nac)` are both builder functions that leave the raw constructor available for tests. MemoryHub should follow the same shape.

3. **The `.connect()` bridges are the silent-failure surface, not the core systems.** The core systems (`hippocampus`, `scn`, `nac`, `ec`) are already required by the dataclass — forgetting them is a `TypeError`. The silent failure is in the `.connect()` bridges. The builder must enforce that `.connect()` happens.

### Proposed signature

```python
def build_memory_hub(
    *,
    hippocampus: Hippocampus,
    scn: SCN,
    nac: NAc,
    ec: EntorhinalCortex,
    # Optional bio-systems (explicit None opt-out)
    atl: ATL | None = None,
    angular_gyrus: AngularGyrus | None = None,
    worker_pool: WorkerPool | None = None,
    cerebellum: Any | None = None,
    embodiment: Any | None = None,
    # Bridge deps (external systems for .connect())
    fear_agent: FearAgent | None = None,
    spatial: SpatialMap | None = None,
    attention: AttentionNetwork | None = None,
    salience: SalienceNetwork | None = None,
    novelty_tracker: Any | None = None,
) -> MemoryHub:
```

**L3 (gate on the learning subject):** The three always-created bridges (PlanHistory, Escalation, Fear) all depend on the CORE systems (`hippocampus`, `nac`, `ec`, `scn`) which are already required. There is no separate gating needed — if the core exists, the bridges can be created. `fear_agent` is a bridge dep (passed through to FearCircuitBridge and stored for fear-circuit queries), not a learning subject. `spatial`/`attention`/`salience` are external system references, not learning subjects.

**L4 (declared fields):** The builder constructs `MemoryHub`, calls `.connect()` internally, and returns the hub. No attribute stashes. The bridge fields (`_spatial_bridge`, `_fear_bridge`, etc.) are already declared on the dataclass.

**Key invariant:** `build_memory_hub` ALWAYS calls `.connect()` internally, so the three always-created bridges (PlanHistoryBridge, EscalationLearningBridge, FearCircuitBridge) are ALWAYS alive on a hub returned by the builder. Callers that don't need spatial/salience/attention simply omit those kwargs — the bridges for those are `None` by design, not by accident.

### Migration plan

| Site | Current | Migration |
|---|---|---|
| #1 CLI | `MemoryHub(...)` no `.connect()` | `build_memory_hub(hippocampus=..., scn=..., nac=..., ec=...)` — **fixes Gap A** |
| #2 Sim AUT | `MemoryHub(...)` + `.connect(fear_agent=_fear)` 570 lines later | `build_memory_hub(..., fear_agent=fear_agent)` — collapses the gap. Requires reordering FearAgent construction BEFORE `build_memory_hub`, or accepting `fear_agent=None` and wiring later. Audit needed on whether FearAgent depends on the hub. |
| #3 Sim orch | 💀 Dead code (TypeError) | `build_memory_hub(hippocampus=..., scn=SCN(), nac=..., ec=EntorhinalCortex())` — **fixes Gap C** by adding missing scn/ec. |
| #4 Reachy | `MemoryHub(...)` + `.connect(spatial=..., ...)` later | `build_memory_hub(..., fear_agent=..., spatial=..., attention=..., salience=..., novelty_tracker=...)` — collapses into single call. This site already has everything available before construction. |
| #5 AgentFactory | `MemoryHub(...)` no `.connect()` | `build_memory_hub(hippocampus=..., scn=..., nac=..., ec=..., atl=...)` — **fixes Gap B**. NPC agents get bridges for the first time. |
| #6 api.py | No MemoryHub | No change. Explicit opt-out stays. |

### Open question: Site #2 construction ordering

The sim orchestrator builds `MemoryHub` at line 447 and `FearAgent` much later. If `FearAgent` construction depends on `hippocampus` from the hub, we can't simply reorder. Options:

- **A)** Construct FearAgent BEFORE `build_memory_hub` if it's independent. Pass it to the builder.
- **B)** Call `build_memory_hub(..., fear_agent=None)` and add a `hub.wire_fear_agent(fear_agent)` post-hoc setter. This weakens the structural enforcement but mirrors the existing staging.
- **C)** Leave site #2 on the raw `MemoryHub()` + `.connect()` pattern for now, same as the test exemption. The site is already behaviorally correct (calls `.connect()`).

**My lean: Option A if feasible, Option C as fallback.** Need to check the FearAgent dependency chain.

## Pre-merge review round (mandatory)

**Executor lens:**
- Does the builder correctly call `.connect()` so all three always-created bridges (PlanHistory, Escalation, Fear) are alive on every returned hub?
- Are there MemoryHub construction sites the audit missed? (`grep -rn "MemoryHub(" src/maxim/` should match only: the class def, the builder, and the two exempted sites.)
- Does the migration introduce any double-wiring (bridges created at construction AND via a leftover `.connect()` call)?
- Is site #2 (sim orchestrator) correctly handled — does FearAgent depend on the hub, or can it be constructed first?
- Does Gap C (dead orchestrator hub) actually fix correctly with added `scn`/`ec`, or is there a deeper issue?

**Architecture lens:**
- Is Option C (builder + raw constructor for tests) the right trade-off, or should `.connect()` be removed entirely?
- Should `build_memory_hub` live in `integration/memory_hub.py` (co-located) or a separate `integration/memory_hub_builder.py`?
- Cross-check: does `agent_factory_canonicalization.md` Stage F1 design need this plan's builder shape before opening? (My read: no — F1 will call `build_memory_hub` from inside `AgentFactory.create_agent`, but the builder's signature is stable regardless.)
- Cross-check: does `default_network_unification.md` (parallel Wave 2) need to coordinate? (My read: no — DN's PainBus ownership inversion is independent of MemoryHub's bridge wiring. The only touch point is that DN's `spatial`/`attention`/`salience` are currently read via `getattr(default_network, ...)` in `agentic_runtime.py:692` and passed to `.connect()`. Post-migration those will be passed directly to `build_memory_hub`. DN doesn't need to change.)
- The `locals().get("fear_agent")` fragility in site #2 — should this be fixed as part of migration?

## Estimated scope

~200 LOC builder + migration + ~250 LOC tests. Single PR. ~2 days.

## Out of scope

- Memory tier progression (already enforced by `TierTransitionError`).
- The three separate `EpisodicMemory` instances on Hippocampus/NAc/ATL (load-bearing invariant, do not merge).
- Hippocampus persistence path resolution — separate concern.
- DefaultNetwork's PainBus ownership inversion — that's `default_network_unification.md`.
- `agent_factory_canonicalization.md` territory — Wave 3+.
- Modifying `build_executor` or `build_pain_bus` — those are frozen.
