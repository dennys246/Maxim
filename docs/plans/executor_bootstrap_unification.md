# Executor Bootstrap Unification — push the bridge invariant into `build_executor`

**Status:** Draft, audit complete, awaiting code.
**Branch:** `feat/exec-bootstrap-unify`
**Scope:** ~200-300 LOC in `src/maxim/`. Tests for new code only; bulk test migration deferred.
**Target version:** 0.4 maintenance.
**Gates:** none in the release matrix; structurally enforces an invariant that bit us three times.
**Depends on:** Stage 2 of `sem_execution_hook.md` (the helper this plan subsumes).
**Blocks:** `sem_execution_hook.md` Stage 2c (rewrites itself as a one-line migration on top of this).
**Parent:** none.
**Related:** [sem_execution_hook.md](sem_execution_hook.md) (parent context), [agent_factory_canonicalization.md](agent_factory_canonicalization.md) (Option D follow-up).

## Goal

Make it structurally impossible to build an `Executor` without making an explicit `ToolPainBridge` decision. The invariant that "every Executor needs a bridge" stops being a discipline that callers must remember and becomes a property of the constructor signature.

## The repeating bug shape

Three Stages of `sem_execution_hook` have closed three instances of the exact same bug:

| Stage | Path | Symptom |
|---|---|---|
| Stage 1 (PR #107) | tool-invoked embodiment pain attribution | bridge existed, attribution path silently scored 0.0 |
| Stage 2 (PR #110) | `maxim --llm X` non-sim CLI | bridge **never constructed** on the most common entry point |
| Stage 2c (this plan's predecessor) | `--sim agent` + `--sim interactive` + `simulation/orchestrator.py` AUT | bridge **not constructed** on three sim paths |

Three identical bugs in three locations is not coincidence — it is structural. The Stage 2 `runtime/embodiment_bootstrap.py` helper consolidated the *construction*, but each call site still has to remember to *call* the helper. The bug that comes back is "we forgot to call the helper on path N+1." The next entry point will reproduce it again unless the invariant moves down a layer.

This plan is the structural fix. It also subsumes Stage 2c (which becomes a trivial migration on top of this plan).

## The audit

Every `build_executor` call site in `src/maxim/` (tests deferred):

| # | Site | Has `pain_bus` available? | Currently wires bridge? | Migration |
|---|---|---|---|---|
| 1 | `cli.py:1131` (non-sim agent) | Yes (`_cli_pain_bus`) | Yes (via Stage 2 helper) | pass `pain_bus=_cli_pain_bus` to `build_executor`; delete helper call |
| 2 | `simulation/orchestrator.py:926` (`aut_executor`) | Yes (`aut_pain_bus`) | Yes (hand-rolled `ToolPainBridge` at L940) | pass `pain_bus=aut_pain_bus`; delete L936-950 hand-roll |
| 3 | `simulation/orchestrator.py:927` (`orch_executor`) | Yes (`aut_pain_bus` is in scope) | **No** — silent fourth instance | explicit decision: bridge or `pain_bus=None`. Likely `None` (orchestrator agent isn't being learned-on) but **needs your call** |
| 4 | `embodied_runtime/agentic_runtime.py:325` (Reachy) | No (`pain_detector` legacy path) | Yes (via Stage 2 helper, with `pain_detector=`) | pass `pain_detector=...` to `build_executor`; same XOR rule |
| 5 | `api.py:436` (public `maxim.create.agent` headless path) | **No PainBus constructed at all** | **No** — silent fifth instance | requires upstream change: build a PainBus in api.py OR pass `pain_bus=None` explicitly with a plan-level note that headless API agents don't learn from tool failures |
| 6 | `simulation/tools.py:793` (`sub_executor` inside an AUT-internal tool) | No (no PainBus in scope) | **No** — silent sixth instance | likely `pain_bus=None` (sub-executor is for tool-internal sandboxing, not bio-learning) |

**Five of six call sites are currently wrong or implicit.** Only `cli.py` (post-Stage-2) and `agentic_runtime.py` (post-Stage-2) are correct, and both still rely on the discipline of calling the helper. After this plan: 6/6 call sites make an explicit decision at the constructor.

**Two pre-existing bugs surfaced by the audit:**

- **`api.py::maxim.create.agent` (headless mode) constructs no PainBus at all.** Any tool failure during a `pymaxim` headless agent run silently skips NAc learning. Same shape as the CLI gap Stage 2 closed. **Fourth instance of the same bug.**
- **`simulation/tools.py` sub-executor** has no bridge wired. Likely intentional (sub-executors are sandboxed), but currently implicit — needs to be explicit.

## Design

### New `build_executor` signature

```python
def build_executor(
    tool_registry: ToolRegistry,
    *,
    pain_bus: PainBus | None,                      # REQUIRED keyword
    nac: NAc | None = None,
    hippocampus: Hippocampus | None = None,
    scn: SCN | None = None,
    pain_detector: PainDetector | None = None,     # legacy XOR with pain_bus
    tool_index: LearnedToolIndex | None = None,
    entity_ref: str | None = None,
    component_registry: ComponentRegistry | None = None,
    cerebellum: Cerebellum | None = None,
) -> Executor:
    """Build an Executor with an explicit ToolPainBridge decision.

    `pain_bus` is REQUIRED (keyword-only, no default). Callers pass:
      - A live `PainBus` instance to enable NAc tool-outcome learning.
      - `None` to explicitly opt out (sandbox executors, headless API
        agents that don't need bio-learning, tests).
    There is no default — every caller MUST make the decision.
    """
```

**Key properties:**

1. **`pain_bus` has no default.** Forgetting to think about it is a TypeError, not a silent no-op.
2. **`pain_bus=None` is a legal, explicit opt-out.** Tests and sandbox executors pass it; the choice is documented at every call site.
3. **`pain_bus` XOR `pain_detector`.** Same invariant the Stage 2 helper enforced; raised as `ValueError` at the top of the function before any construction.
4. **`entity_ref` + `component_registry` are optional pass-throughs** — if both are provided, `build_executor` loads the embodiment and registers affordance tools. The Stage 2 helper's `bootstrap_embodiment_and_pain_bridge` becomes obsolete and gets deleted.
5. **The wrapping-order invariant moves from helper docstring to `build_executor` docstring** — the rule is now "build_executor returns an inner Executor; wrap with FearGated/PainInterceptor AFTER."
6. **Fail-fast precondition checks** run at the top, before any object construction. Same rule the Stage 2 review folded.

### What `runtime/embodiment_bootstrap.py` becomes

**Deleted.** Its entire job is folded into `build_executor`. Its existing tests in `tests/unit/test_embodiment_bootstrap.py` move to `test_build_executor.py` with light renames; the test bodies are mostly unchanged because they were already exercising the same shape.

### What changes in each call site

Each of the six call sites becomes a single `build_executor(registry, pain_bus=..., nac=..., ...)` call. The hand-rolled `ToolPainBridge(...)` block in `orchestrator.py:936-950` deletes. The Stage 2 `bootstrap_embodiment_and_pain_bridge` calls in `cli.py` and `agentic_runtime.py` collapse into the same constructor call.

## Migration plan

1. **Audit folded in (this doc, ✅).**
2. **Write the new `build_executor` signature + helper-deletion in one commit.** Tests for the new API are written first (red), then the implementation (green), per `feedback_failing_tests_need_tight_assertions.md`.
3. **Migrate the six `src/maxim/` call sites in a second commit.** Each migration is a one-line constructor call with explicit `pain_bus=...` choice. Resolve the audit's two open questions (`orch_executor` and `api.py`) by passing `pain_bus=None` with an inline comment naming the decision.
4. **Delete `runtime/embodiment_bootstrap.py` + its test file in a third commit.** Move regression guards into `tests/unit/test_build_executor.py` with renames.
5. **Doc + memory refinement in a fourth commit** — see "Doc + memory refinement" below.
6. **Pre-merge review round** (Executor + Architecture lenses, parallel, fold cross-confirmed first).
7. **PR.**

Test scope for this plan: only tests for the **new** `build_executor` behavior. The bulk test-suite migration (every test that calls `build_executor(registry)` without keyword args) is **deferred** — those tests will fail loudly with `TypeError: missing required keyword-only argument: pain_bus` and we'll fix them as we touch each test file. We expect a wave of test failures on the first full-suite run; that is the point — every test must make the decision explicit.

## Doc + memory refinement (load-bearing scope item)

When the helper-discipline approach failed three times, the lesson is that documentation alone cannot enforce an invariant — the invariant must move into a type signature. This plan must update the docs so future Claude sessions internalize the rule:

1. **`CLAUDE.md` "Architectural invariants"** — replace the existing `runtime/embodiment_bootstrap.bootstrap_embodiment_and_pain_bridge is canonical` line with:
   > **`build_executor` is the canonical bridge wiring site.** The `pain_bus` parameter is required (keyword-only, no default). Every caller makes an explicit `pain_bus=<bus>` or `pain_bus=None` decision. Bridges cannot be retrofitted onto an Executor — wrapping (`FearGatedExecutor`, `PainInterceptorExecutor`) must happen AFTER `build_executor` returns. The previous helper `runtime/embodiment_bootstrap.py` is deleted; do not re-introduce it.
2. **`CLAUDE.md` "Lessons learned"** — add a new "Three-times-is-structural" meta-lesson:
   > When the same bug shape ("forgot to call helper X on path Y") is fixed in three different locations, the helper is the wrong layer. Push the invariant down into the type/constructor signature so the bug becomes a TypeError, not a silent no-op. The `build_executor(pain_bus=...)` keyword-only requirement is the canonical example.
3. **New memory file** `feedback_structural_enforcement_over_helper_discipline.md` — when a helper is forgotten three times, push the invariant down a layer. Cross-reference to this plan + the three SEM execution hook stages.
4. **`docs/plans/sem_execution_hook.md` Stage 2c section** — rewrite to point at this plan as the prerequisite. Stage 2c collapses to "migrate three call sites to the new `build_executor` signature" once this plan ships.
5. **`docs/plans/README.md`** — add this plan to the parallel/maintenance section.

## Pre-merge review round (mandatory)

Same template as substrate P2 Stage 2 / Stage 3 / SEM execution hook Stages 1+2. Two parallel reviewers (Executor + Architecture lenses). The specific review questions for this plan:

**Executor lens:**
- Does the new `build_executor` correctly enforce the wrapping-order contract — i.e., does it reject being called with an already-wrapped executor (or is the wrapping rule still discipline)?
- Are there any `build_executor` call sites in `src/maxim/` I missed? (Run the grep yourself.)
- Is the `pain_bus=None` opt-out at `api.py` and `tools.py` correct, or is it hiding a real bug those call sites should also fix?
- Does the deleted `embodiment_bootstrap.py` leave any orphan imports?
- Does the new `build_executor` signature break the public Python API contract for `maxim.create.agent`? (It shouldn't — `api.py` is a call site, not a re-export.)

**Architecture lens:**
- Is "required keyword arg with no default" the right strictness, or should we allow `pain_bus` to default to `None` with a runtime warning? (My recommendation: required, no default. Soft warnings get ignored.)
- Should `entity_ref` resolution stay inside `build_executor`, or should it stay separate so the constructor has one job? (Trade-off: one job is cleaner, but two functions re-introduces the "forgot to call the second one" risk.)
- Does folding `embodiment_bootstrap.py` into `build_executor` violate the "many small files" preference in CLAUDE.md? (My read: no — `runtime/bootstrap.py` is the canonical bootstrap file, and `build_executor` belongs there.)
- Is the test deferral acceptable, or should this PR also migrate every test call site?
- Cross-check: does this plan unblock or constrain `agent_factory_canonicalization.md`?

## Future work — Option D (named follow-up)

See [agent_factory_canonicalization.md](agent_factory_canonicalization.md). This plan is the structural floor; that plan is the structural ceiling (one canonical agent builder, all entry points go through it). They are complementary — this plan makes Option D a downhill rewrite by ensuring every Executor already has the bridge invariant baked in.

**Trigger conditions for opening Option D:**
- A 6th agent entry point is proposed (today there are five: cli, sim orchestrator, sim interactive, Reachy, public API).
- The next bridge-wiring or pipeline-construction bug surfaces despite this plan.
- An `AgentFactory`/`AgentPool` work session is happening anyway and the author wants to extend it to subsume CLI/orchestrator/Reachy.

## Pass criteria

- Every `build_executor` call site in `src/maxim/` passes `pain_bus=` explicitly (real value or `None`).
- Grep `_tool_pain_bridge =` in `src/maxim/` returns matches only in `runtime/bootstrap.py` (or wherever `build_executor` lives) — no per-call-site assignments.
- Grep `ToolPainBridge(` in `src/maxim/` returns matches only in `runtime/bootstrap.py` and `bridges/tool_pain_bridge.py` itself.
- `runtime/embodiment_bootstrap.py` is deleted; no orphan imports.
- New regression test: `test_build_executor_requires_pain_bus_kwarg` (TypeError on missing).
- New regression test: `test_build_executor_pain_bus_none_is_explicit_opt_out` (no bridge, no error).
- New regression test: `test_build_executor_with_entity_ref_loads_embodiment_and_registers_tools` (full Stage 2 cascade through the new constructor).
- The existing 9 tests in `test_embodiment_bootstrap.py` move to `test_build_executor.py` with renames and pass unchanged in spirit.
- CLAUDE.md updates merged in the same PR.
- Pre-merge review round completed (Executor + Architecture, parallel, fold cross-confirmed first).

## Out of scope

- **Bulk test migration.** Tests that call `build_executor(registry)` without keyword args will fail loudly when the suite runs — fix them in the same PR if cheap, defer to a follow-up sweep otherwise.
- **`AgentFactory` canonicalization** — see Option D plan.
- **`api.py` PainBus construction** — the call site passes `pain_bus=None` explicitly with an inline TODO; making the headless Python API construct a real PainBus is a separate decision (involves user-facing API question about whether headless agents learn from tool failures).
- **`orch_executor` learning** — same: passes `pain_bus=None` with an inline note. Whether the orchestrator agent should learn from its own tool outcomes is a separate decision.

## Estimated scope

| Item | LOC | Notes |
|---|---|---|
| New `build_executor` signature + helper deletion | ~150 | Mostly fold-in of existing helper |
| Six call site migrations | ~50 | ~8 LOC per site avg |
| New tests | ~250 | 9 moved + 3 new |
| Doc + memory refinement | ~50 | CLAUDE.md edits + 1 new memory file |
| Pre-merge review fold | ~50 | Typical |
| **Total** | **~550** | Single PR |
