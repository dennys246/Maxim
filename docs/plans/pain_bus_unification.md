# PainBus Unification — `build_pain_bus` with required learning subscribers

**Status:** SHELL ONLY. Audit + design pending. Open this when the executor bootstrap PR merges and a session has bandwidth.
**Parent index:** [biosystem_unification.md](biosystem_unification.md)
**Wave:** 1 of 4 (parallel-safe with `reaction_bus_unification.md`)
**Depends on:** none — `executor_bootstrap_unification.md` should land first to keep the migration audit clean, but technically independent.
**Parallel-safe with:** [reaction_bus_unification.md](reaction_bus_unification.md) — different files, no overlap.
**Blocks:** [memory_hub_unification.md](memory_hub_unification.md), [default_network_unification.md](default_network_unification.md), [bio_stack_unification.md](bio_stack_unification.md).

## Goal

Push the PainBus subscriber-wiring invariant DOWN into `build_pain_bus(...)` so that constructing a PainBus without explicit learning-subscriber decisions is a `TypeError`. The current scattered pattern across 5+ entry points is the next instance of the same bug class `executor_bootstrap_unification.md` closed.

## The repeating shape (silent failure)

Every entry point that builds a PainBus does this:

```python
pain_bus = PainBus()
pain_bus.subscribe(create_pain_memory_subscriber(hippocampus))   # easy to forget
pain_bus.subscribe(create_pain_nac_subscriber(nac))               # easy to forget
```

Forgetting either subscriber is a **silent no-op** — the bus accepts publishes, dispatches to zero learners, pain signals fire into the void, NAc never learns from out-of-band pain, hippocampus never captures pain memories. Fits the L1 silent-failure-mode rule perfectly.

## Audit (2026-04-14)

Every PainBus construction site in `src/maxim/`. ✅ = subscriber wired, ❌ = silent gap, ⚪ = explicit opt-out (deliberate, not a bug), ➖ = N/A.

`hippocampus subscribed?` and `nac subscribed?` mean: is `create_pain_memory_subscriber` / `create_pain_nac_subscriber` registered against the bus instance? **Direct attribution paths** (e.g., `ToolPainBridge` calling `nac.record_outcome` from inside an in-flight tool) are NOT counted in the `nac subscribed?` column — that column is strictly about the bus-fallback path used for **out-of-band** pain (e.g., autonomous SEM ticks where no tool is in flight, ambient body-sensor decay, sandbox `PainTriggerLayer` events). Both paths matter; the bus-subscription column is what this plan is fixing.

| # | Site | File:line | hippocampus | nac | Other subscribers | Notes |
|---|---|---|---|---|---|---|
| 1 | CLI non-sim agent | [cli.py:1176](src/maxim/cli.py#L1176) | ✅ | ❌ **GAP** | none | `_cli_nac` exists at this scope (cli.py:1092/1107) but is never subscribed. Out-of-band SEM pain reaches hippocampus, never reaches NAc on this entry point. |
| 2 | CLI `--sim agent` | [cli.py:1412](src/maxim/cli.py#L1412) | ✅ | ❌ **GAP** | none | Same shape as #1. `_cli_nac` available, not subscribed. |
| 3 | CLI `--sim interactive` | [cli.py:1373](src/maxim/cli.py#L1373) | ✅ | ❌ **GAP** | none | Same shape as #1+#2. |
| 4 | Sim orchestrator AUT | [simulation/orchestrator.py:69](src/maxim/simulation/orchestrator.py#L69) | ✅ | ✅ | none | The **only** site that subscribes both. Subscription happens at line 619/622, after `aut_hippocampus` and `aut_nac` are built. Reference for the correct shape. |
| 5 | DefaultNetwork (internal) | [default_network/network.py:360](src/maxim/default_network/network.py#L360) | ❌ | ⚪ (via `PainCircuitBridge`) | `PainCircuitBridge` constructor calls `pain_bus.subscribe(self._on_pain)` ([bridges/pain_bridge.py:130](src/maxim/bridges/pain_bridge.py#L130)) | Constructed inside `_init_pain_circuit`, gated on `nac is not None`. NAc receives signals through the **PainCircuitBridge**, not via `create_pain_nac_subscriber`. Hippocampus subscription happens externally in agentic_runtime.py:719 — see #6. **Latent fragility**: if `default_network.pain_bus` is consumed by a caller that doesn't also wire hippocampus, hippo silently drops out. Two-way coupling, no enforcement. |
| 6 | Embodied runtime (Reachy) consumer | [embodied_runtime/agentic_runtime.py:719](src/maxim/embodied_runtime/agentic_runtime.py#L719) | ✅ (subscribes externally) | ➖ (relies on #5's `PainCircuitBridge`) | — | Does NOT construct a PainBus — consumes `default_network.pain_bus`. Wires hippocampus subscription at construction-completion time. The NAc path is structural (via `PainCircuitBridge`). Migration concern: this site is one of two consumers of #5's `pain_bus` property. The other is whatever `default_network.pain_bus.publish(...)` calls exist internally. |
| 7 | api.py headless `maxim.create.agent` | [api.py:444](src/maxim/api.py#L444) | ⚪ | ⚪ | none | **Explicit `pain_bus=None`** to `build_executor`. NO PainBus constructed at all. Documented as a TODO pointing at `agent_factory_canonicalization.md` Stage F5 because fixing it requires a user-facing API decision (default-on vs default-off bio-learning for headless users of `pymaxim`). **In-scope question for this plan**: is the right resolution (a) ship `build_pain_bus(...)` and have api.py call it, or (b) leave api.py opted-out and let F5 own the user-facing decision? **Recommendation:** leave the explicit `pain_bus=None` for now — the user-facing API question is genuinely orthogonal to the structural-enforcement work. Update the TODO comment to point here for the structural side and at F5 for the API side. |
| 8 | sub-AUT executor (sim tools) | [simulation/tools.py:796](src/maxim/simulation/tools.py#L796) | ➖ | ➖ | — | Does NOT construct a PainBus. Explicit `pain_bus=None` to `build_executor`. Sandboxed tool-internal sub-executor; no bio-learning by design. |

### Test-side construction sites

`PainBus()` is also constructed directly in 7 test files (counted via `grep -rln "PainBus(" tests/`):

- `tests/unit/test_build_executor.py` (~9 instances) — builds raw `PainBus()` to feed `build_executor`. Doesn't subscribe learners; tests are about the executor shape, not the bus.
- `tests/unit/test_pain_bus.py` (~12 instances) — the bus's own unit tests. Subscribers are wired ad-hoc per-test.
- `tests/unit/test_percept_simulation.py` (~4 instances) — percept→pain integration tests.
- `tests/unit/test_reaction_bus.py` (~5 instances) — reaction-bus integration via the PainBus wrapper.
- Plus three more files (substrate cascade tests, etc.) — same shape.

**Migration policy for tests** (matches the executor unification's policy): leave raw `PainBus()` constructor accessible. The structural enforcement lives at the `build_pain_bus(...)` door, which is the production entry point. Tests that want a stripped bus (no learners) keep using `PainBus()` directly — that's the test author's explicit decision and matches the existing pattern. This avoids a 30+ test mass-migration that would be churn without payoff. Same precedent as `Executor()` raw construction surviving the executor unification.

## Pre-existing silent gaps surfaced by the audit

Three identical bugs, same shape, three different CLI entry points (#1, #2, #3):

**Gap A (Critical, in-scope) — CLI agent paths skip NAc bus subscription on out-of-band pain.**

All three CLI-driven agent entry points (`maxim --llm X`, `maxim --sim agent`, `maxim --sim interactive`) construct a PainBus, subscribe `create_pain_memory_subscriber`, and **never** subscribe `create_pain_nac_subscriber`. The `_cli_nac` variable is in scope at the construction site for all three. The omission is uniform — every site imports only `create_pain_memory_subscriber` from `pain_bus`, never `create_pain_nac_subscriber`. This looks like a copy-paste lineage from before `create_pain_nac_subscriber` existed (the substrate P2 Stage 2 commit that landed the NAc subscriber updated `simulation/orchestrator.py` but did not update the cli.py paths).

**Blast radius:** for any non-sim agent run, autonomous SEM body-sensor decay, sandbox `PainTriggerLayer` events, and any `body.py::_publish_pain` call that fires while no tool is in flight (e.g., a sleeping body, a delayed sensor reading) reach hippocampus but **NEVER reach NAc**. Tool-invoked pain still reaches NAc via the direct-attribution path through `ToolPainBridge.record_tool_embodiment_failure` (the SEM execution hook Stage 1 fix), so the substrate P2 cascade test still passes — the gap is invisible to existing tests because they all go through a tool. The missing learning is in the out-of-band path.

**This is exactly the L1 silent-failure-mode shape** that justifies structural enforcement. Three identical instances of "forgot the second subscriber" across three entry points; the next CLI sibling that lands (`--sim DM`, `--sim benchmark`, future `--sim XX`) will reproduce the bug a fourth time.

**In-scope to fix in this PR.** The fix is one line per call site once `build_pain_bus(*, hippocampus, nac)` exists.

**Gap B (Latent, structural — keep an eye on it) — DefaultNetwork's PainBus has split subscriber ownership.**

DefaultNetwork constructs PainBus internally, but only NAc gets wired (via `PainCircuitBridge` constructor). Hippocampus subscription happens at the **consumer** side, externally, in `embodied_runtime/agentic_runtime.py:719`. If a future consumer of `default_network.pain_bus` forgets the external hippocampus wire, hippocampus silently drops out for that path. The current consumers are exactly one (Reachy runtime), so the bug hasn't bitten yet.

**Decision:** out of scope for this PR's migration. Reason: fixing it requires deciding whether DefaultNetwork should accept `hippocampus=` at construction time, which couples DefaultNetwork to MemoryHub and changes its constructor surface. That's a Wave 2 question (`memory_hub_unification.md`) — DefaultNetwork is the natural caller for `build_pain_bus(...)` once MemoryHub is structurally enforced. **Document the gap in this plan, log a Wave 2 follow-up, do NOT band-aid by adding hippocampus= to DefaultNetwork in this PR.** Per the no-band-aid rule.

**Gap C (Known, deferred) — api.py headless has no PainBus at all.**

Already documented in api.py:436-443 with a TODO pointing at `agent_factory_canonicalization.md` Stage F5. The reason is genuinely orthogonal: the question is "should `maxim.create.agent(...)` default to bio-learning on or off?" That's a user-facing pymaxim API decision, not a structural enforcement question. **Leave the explicit `pain_bus=None` opt-out, update the TODO comment** to acknowledge this plan as the structural-side resolver while F5 still owns the user-facing decision.

## Design sketch

```python
def build_pain_bus(
    *,
    hippocampus: Hippocampus | None,
    nac: NAc | None,
    additional_subscribers: tuple[Callable[[PainSignal], None], ...] = (),
) -> PainBus:
    """Construct a PainBus with explicit learning-subscriber decisions.

    `hippocampus` and `nac` are REQUIRED keyword args. Pass real
    instances to auto-subscribe the standard learners
    (create_pain_memory_subscriber, create_pain_nac_subscriber); pass
    None to explicitly opt out (sandbox / test).
    """
```

**Open design questions:**
- Should `additional_subscribers` be a list of callables, or a list of `Subscriber` protocol instances?
- Should the function support per-pain-type filtering at the constructor level, or delegate that to subscribers?
- Should there be a separate `build_test_pain_bus()` factory for test code so test authors don't have to type `hippocampus=None, nac=None` everywhere?

## Migration call sites (rough scope)

5-6 production sites + ~? test sites. Each migrates from `PainBus(); pain_bus.subscribe(...)` to `build_pain_bus(hippocampus=..., nac=...)` with an explicit decision.

## Pre-existing bugs to fix as part of migration

Likely `api.py::maxim.create.agent` headless mode — the executor bootstrap audit already found it constructs no PainBus at all. This plan should construct one (with the user-facing API decision: bio-learning on by default or off by default). This is the **right plan to fix that gap**, and `executor_bootstrap_unification.md` already left a TODO pointing here.

## Doc + memory refinement (load-bearing scope)

- `CLAUDE.md` invariant addition: "`build_pain_bus` is the canonical PainBus construction site. `hippocampus` and `nac` are required keyword args. Forgetting subscribers is a TypeError."
- Update `executor_bootstrap_unification.md`'s "TODO: api.py headless" note to point at this plan as the resolver.
- Update `biosystem_unification.md` status row.

## Pre-merge review round (mandatory)

Two parallel reviewers (Executor + Architecture lenses). Specific questions:

**Executor lens:**
- Does the new `build_pain_bus` correctly subscribe the standard learners? Verify both subscribers fire on a published `PainSignal`.
- Are there any PainBus construction sites I missed in the audit?
- Does the migration introduce any double-subscription (caller subscribes + `build_pain_bus` also subscribes)?
- Does the test surface include a regression guard for the original silent-no-op bug shape?

**Architecture lens:**
- Is "required keyword arg with no default" the right strictness for `hippocampus`/`nac`, or is `None` acceptable as the default?
- Should `additional_subscribers` be a kwarg or a separate `.subscribe()` call after construction?
- Does this plan supersede or complement `bio_stack_unification.md`?
- Is the pre-existing `api.py` PainBus gap correctly in-scope for this plan, or should it stay deferred?

## Estimated scope

~150 LOC + ~250 LOC of new tests. Single PR. ~2-3 days of focused work.

## Out of scope

- ReactionBus subscribers — see `reaction_bus_unification.md`.
- MemoryHub bridge wiring — see `memory_hub_unification.md`.
- The AgentFactory canonicalization that uses this builder — see `agent_factory_canonicalization.md`.
