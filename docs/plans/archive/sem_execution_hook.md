# SEM Execution Hook — Production Forward Path for Affordances

**Status:** Stages 1+2+3+4 SHIPPED (2026-04-14). Stage 2c SUPERSEDED by `executor_bootstrap_unification.md`. Stage 2b deferred to `agent_factory_canonicalization.md` Stage F1+.
**Scope:** ~1,100 LOC shipped so far (Stage 1 ~490 + Stage 2 ~820). Stages 3+4 estimated ~400-600 more.
**Target version:** Ships anytime. Not on the substrate version-gate path.
**Gates:** NOTHING in the release matrix, but unblocks behavioral convergence experiments that need real SEM execution loops (H1, H2, H4 in [behavioral_convergence_practice.md](behavioral_convergence_practice.md)).
**Depends on:** substrate_recognition (✅ shipped), P2 Stage 2 pain cascade (✅ shipped).
**Blocks:** Any production experiment that needs an agent to invoke SEM affordances from its prompt output (most of behavioral_convergence_practice H1-H4, the multi-session cross-modal sanity runs in P4).
**Parent:** none — cross-cutting plan, not part of substrate_binding_persistence.
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) (discovery), [substrate_recognition.md](substrate_recognition.md) (Stage 2 pain cascade PoC), [../experiments/p2_sem_pain_cascade.md](../experiments/p2_sem_pain_cascade.md) (the PoC this plan generalizes to production).

## Shipped status (2026-04-14)

- **Stage 1 ✅ SHIPPED** — PR #107, commit `6070241`. Direct-attribution for tool-invoked embodiment pain via the new `ToolOutput.side_effects` typed channel + `ToolPainBridge.record_tool_embodiment_failure` API + `_on_embodiment_pain` guard. See [project_sem_execution_hook_stage1.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_sem_execution_hook_stage1.md).
- **Stage 2 ✅ SHIPPED** — PR #110, commit `99057a1` (merged as `efd0a38`). `runtime/embodiment_bootstrap.bootstrap_embodiment_and_pain_bridge` shared helper + `--embodiment <REF>` CLI flag + Reachy refactor. **Also closed a pre-existing production bug** where `maxim --llm X` had NEVER constructed a `ToolPainBridge` (silent no-op on the most common agent entry point). See [project_sem_execution_hook_stage2.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_sem_execution_hook_stage2.md).
- **Stage 2b 🚧 DEFERRED** — `AgentFactory` / `AgentPool` / `maxim.create.agent()` embodiment wiring. See the "Stage 2b" section below.
- **Stage 2c 🚧 DEFERRED** — sim orchestrator + `run_interactive_sim` migration to the helper. See the "Stage 2c" section below. Currently `--embodiment` + `--sim` is a hard `sys.exit(2)` error until this lands.
- **Stage 3 ⏳ PENDING** — end-to-end production test on real `weapons/rusty_sword` via the `AgentFactory` path (not the PoC harness). Needs Stage 2b to construct the executor-per-turn wiring first, OR can use the CLI path directly.
- **Stage 4 ⏳ PENDING** — `maxim doctor` check for missing `--embodiment` refs + CLI help text + `docs/embodiment_guide.md` section + smoke-run log verification.

## Goal

Make it so that when a real agent running `maxim --llm X` (no sim orchestrator) decides to invoke a SEM affordance from its prompt output, the affordance actually executes, drives sensor state change, fires failure evaluation, and produces the Percept → Reaction → Learning loop end-to-end. This turns the P2 Stage 2 PoC into a production capability.

## Hypothesis (falsifiable)

When the standard agent loop (`runtime/agent_loop.py` + `runtime/loop_controller.py` + `runtime/executor.py`) is wired to a SEM `Embodiment` instance at startup, affordance tools generated from that tree via `tool_bridge.generate_tools_for_entity` are invocable by the agent on its own timeline, the failure cascade runs correctly through `PainBus` to `NAc`, and the agent's `nac.predict` returns NEGATIVE for actions that produced pain in prior turns — without requiring the generative_runner or dm_runtime sim orchestrators to be in the loop.

## The gap (discovered 2026-04-14 during substrate_binding planning audit)

The P2 Stage 2 retrospective in the original `substrate_recognition.md` closure flagged an "RC3 follow-up: production SEM action recording hook." The note said:

> No production code path currently records NAc pending events for SEM actions. When runtime/executor.py or embodiment/motor.py dispatches an affordance there is no call to nac.record_event("action", signature, context={source, entity}).

**The note was partially wrong.** A deeper audit during the substrate_binding planning pass found:

1. **`tool_bridge.py::ModulatorAffordanceTool.execute` is the real production forward path.** It's a subclass of `Tool` that wraps `Modulator.execute(affordance, params)`, reads back sensor state, calls `embodiment.evaluate_failures()` immediately (no 1Hz poll delay), and feeds Cerebellum for forward-model training. When invoked via `executor.execute({"tool_name": "rusty_sword_slash"})`, the pain cascade runs end-to-end:
   ```
   LLM → executor.execute → ToolPainBridge.record_tool_start (nac.record_event "tool:<name>")
       → ModulatorAffordanceTool.execute
         → Modulator.execute → sensor state change
         → embodiment.evaluate_failures()
           → body.py::_publish_pain
             → PainBus.publish(PainSignal with rich context)
               → create_pain_nac_subscriber
                 → nac.record_outcome_full
                   → CausalLink tool:<name> → pain (NEGATIVE)
   ```

2. **The Percept → Reaction → Learning loop EXISTS and WORKS** for tool-invocation-style SEM affordance calls. The P2 Stage 2 PoC's shape (`agent.record_action("slash") + driving durability + calling evaluate_failures()`) was a test-harness shortcut that bypassed the tool registry, not an indication of a missing mechanism.

3. **What IS missing:** `generate_tools_for_entity` has ONLY two production call sites, both inside sim orchestrators:
   - `simulation/generative_runner.py:79`
   - `simulation/dm_runtime.py:766`
   The standard `runtime/agent_loop.py` + `runtime/loop_controller.py` path contains **zero** references to `embodiment`, `Entity`, or `generate_tools_for_entity`. When a user runs `maxim --llm X` (no sim), no SEM tree is loaded, no affordance tools are registered, and the agent has no way to invoke embodiment actions.

So the gap is narrower but still real: **there is no production-agent path to LOAD a SEM embodiment tree and AUTO-GENERATE affordance tools** outside the sim orchestrator. The cascade mechanism is wired; the startup wiring isn't.

## Minimum implementation

The Stage 1 audit (2026-04-14, [project_sem_execution_hook_stage1.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_sem_execution_hook_stage1.md)) surfaced a pre-existing root-cause bug in tool-invoked embodiment-pain attribution — the pending tool event's context (`{"params": ...}`) shared zero keys with the rich outcome context, so `_context_similarity` silently returned 0.0 and no NAc link ever formed. That bug was fixed in Stage 1 via direct-attribution through a new `ToolOutput.side_effects` typed channel + `ToolPainBridge.record_tool_embodiment_failure` API + executor wire. The audit also found a second pre-existing gap: **`maxim --llm X` in `cli.py` does NOT construct a `ToolPainBridge` at all**, so the Stage 1 fix is a silent no-op on that path. Stage 2 closes that gap as part of the startup hook work.

### Stage 1 — tool-pain bridge root-cause fix + `ToolOutput.side_effects` — **SHIPPED** (PR #107, commit `6070241`, 2026-04-14)

See [project_sem_execution_hook_stage1.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_sem_execution_hook_stage1.md) for the complete writeup including the 6 load-bearing invariants this establishes.

### Stage 2 — `runtime/embodiment_bootstrap.py` helper + CLI `--embodiment` flag + **CLI `ToolPainBridge` gap fix**

**LOUD CALL-OUT — pre-existing bug this stage closes:** `maxim --llm X` (the non-sim, non-Reachy CLI entry point in [cli.py::main](../../src/maxim/cli.py)) has never constructed a `ToolPainBridge`. The bridge is wired in `simulation/orchestrator.py:940` (sim path) and `embodied_runtime/agentic_runtime.py:356` (Reachy path) but never in the regular CLI agent path. This means any tool failures during a regular `maxim --llm X` session silently skip NAc causal learning — the infrastructure Stage 1 shipped is a no-op on this path. This is PRE-EXISTING (since before the sem_execution_hook plan was drafted), NOT a Stage 1 regression. **Stage 2 fixes this as a side effect of the new shared helper.**

#### Shape

Extract a shared helper so all three production paths converge on one bootstrap:

**New file: `src/maxim/runtime/embodiment_bootstrap.py`** (~100 LOC)

```python
def bootstrap_embodiment_and_pain_bridge(
    *,
    nac,
    hippocampus,
    scn,
    memory_hub,
    pain_bus,          # REQUIRED — caller always constructs one
    executor,          # _tool_pain_bridge gets assigned on return
    entity_ref: str | None = None,
    component_registry: ComponentRegistry | None = None,
    tool_index: Any | None = None,
    cerebellum: Any | None = None,
) -> tuple[Embodiment | None, ToolPainBridge]:
    """Bootstrap ToolPainBridge (always) + Embodiment (when ref given).

    Wires the bridge to the executor unconditionally so tool learning
    works even without SEM. When entity_ref is provided, additionally
    resolves the component, wraps in Embodiment(pain_bus=...),
    generates affordance tools into executor.registry, and returns
    the Embodiment for the caller to hold a reference to.

    Returns (embodiment, bridge).
    """
```

#### Changes

1. **New `runtime/embodiment_bootstrap.py` helper** — the function above.
2. **`cli.py` agent path**:
   - New `--embodiment <component_ref>` CLI flag.
   - Unconditionally construct `PainBus` + `create_pain_memory_subscriber` for the non-sim agent path (matching the sim block at `cli.py:1321`). This closes a second pre-existing gap where non-sim runs had no pain-memory capture.
   - After `executor = build_executor(registry)` and before the FearGatedExecutor wrap, call `bootstrap_embodiment_and_pain_bridge(...)`. This wires `ToolPainBridge` for regular `maxim --llm X` (the core gap this stage closes) and additionally wires an `Embodiment` when `--embodiment` was passed.
3. **`embodied_runtime/agentic_runtime.py:348-368`** — migrate the existing manual `ToolPainBridge` wiring to use the helper. Behavior-preserving refactor; eliminates drift risk between the two sites.

#### Pass criteria (Stage 2)

- `maxim --llm mistral-7b` (no `--embodiment`) constructs a `ToolPainBridge` and wires it to the executor. Unit test: build the helper with `entity_ref=None`, assert `executor._tool_pain_bridge is not None`.
- `maxim --llm mistral-7b --embodiment weapons/rusty_sword` additionally loads the rusty_sword component and registers `rusty_sword_slash`, `rusty_sword_parry`, etc. in the tool registry. Unit test: build the helper with `entity_ref="weapons/rusty_sword"`, assert the expected tool names appear in `executor.registry.list()`.
- `embodied_runtime/agentic_runtime.py` bootstrap path produces byte-identical bridge behavior to pre-refactor (regression-guarded via existing tests).
- `maxim --llm X` path unconditionally constructs a `PainBus` and subscribes `create_pain_memory_subscriber` — verified by a CLI-level test that imports the entry point and walks the wiring.

#### Explicitly OUT of scope for Stage 2

- **`AgentFactory.create_agent` / `maxim.create.agent()` / `maxim.load.agent()` / `AgentPool` embodiment wiring.** These use `AgentInstance.entity` without ever constructing an `Executor` in the factory — the executor lifetime is per-turn in `AgentPool`, not per-session. Wiring the helper here requires rethinking the executor lifecycle and is deferred to **Stage 2b**. The Stage 2 helper is designed so Stage 2b can call it from inside `AgentPool.run_turn` (or wherever the per-turn executor is constructed) without modification.
- **Sim-mode `ToolPainBridge` wiring.** The Stage 2 pre-merge review surfaced that both `--sim agent` (via `run_agentic_loop`) and `--sim interactive` (via `run_interactive_sim`) construct a `PainBus` but NO `ToolPainBridge`. This is a parallel pre-existing gap that matches the non-sim CLI gap this stage closes. Deferred to **Stage 2c** — migrate `simulation/orchestrator.py` + `simulation/interactive.py` to call the helper, then `--embodiment` can work under `--sim` too. Until then, `--embodiment` + `--sim` is a hard error (`sys.exit(2)`).

### Stage 2c — Sim orchestrator + interactive migration — **SUPERSEDED by `executor_bootstrap_unification.md`**

The original Stage 2c plan was to call the `runtime/embodiment_bootstrap` helper from three more sim paths. Mid-session audit found this was the THIRD instance of the same "forgot to wire the bridge" bug shape — three-times-is-structural per [feedback_structural_enforcement_over_helper_discipline.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/feedback_structural_enforcement_over_helper_discipline.md). The fix moved DOWN a layer: `build_executor` now requires an explicit `pain_bus=` keyword arg, the `embodiment_bootstrap` helper is DELETED, and forgetting the bridge is a `TypeError` instead of a silent no-op.

After [executor_bootstrap_unification.md](executor_bootstrap_unification.md) ships, the original Stage 2c work collapses to:
- `simulation/orchestrator.py` already migrated as part of the unification PR (`aut_executor` passes `pain_detector=`, `orch_executor` passes `pain_bus=None` explicitly for AUT-isolation reasons documented inline).
- `simulation/interactive.py` and the `--sim agent` path (which share cli.py's executor) currently pass `pain_bus=None` because they are sim-mode. **The remaining work for true Stage 2c is to give those paths a real PainBus** so `--embodiment` + `--sim` can work end-to-end.
- Drop the hard error in `cli.py` when `--embodiment` + `--sim` is passed.
- Regression test: end-to-end sim run with a bundled weapon component.

**Trigger for opening the new (smaller) Stage 2c:** any sim-mode campaign that needs SEM body wiring at the CLI surface, OR a behavioral experiment that needs NAc tool-outcome learning under `--sim agent`. Until then, sim mode runs without a bridge by explicit `pain_bus=None` opt-out at the constructor.

### Stage 2b — `AgentFactory` / `AgentPool` embodiment wiring — **DEFERRED**

Not scheduled. Activates when any of these hits:

- A behavioral experiment using `maxim.create.agent("name", entity_ref="...")` needs the bio-cascade (currently the entity is stored raw on `AgentInstance.entity` and never exercised for failure learning).
- `AgentPool` multi-agent scenarios need isolated per-agent embodiment + pain bridge (the current `ToolPainBridge` model is single-bridge-per-agent; concurrent NPC scenarios need one bridge per agent instance, which needs either Executor-per-agent or a bridge dispatcher keyed on agent_id).
- `maxim.load.agent(...)` needs to restore an embodiment from persisted state.

Scope sketch (subject to full design pass when it activates):
- Decide executor lifetime in `AgentPool` (currently a per-turn concern — needs a re-read of `agent_pool.py` to confirm).
- Add `embodiment` field to `AgentInstance`.
- Call `bootstrap_embodiment_and_pain_bridge` from inside `AgentPool.run_turn` (or wherever the per-turn executor is built).
- Multi-agent isolation: every agent must have its own `ToolPainBridge` instance, because the `_pending_tools` dict is bridge-scoped and shared state across agents would corrupt attribution.

**Cross-reference:** this deferral is also noted in [project_sem_execution_hook_stage1.md](../../.claude/projects/-Users-dennyschaedig-Scripts-Maxim/memory/project_sem_execution_hook_stage1.md) under "Deferred / out of scope" and should be mirrored wherever the substrate-binding plan discusses multi-agent infrastructure.

### Stage 3 — end-to-end pain cascade integration test (no sim orchestrator) — **SHIPPED** on `feat/sem-execution-hook-stages-3-4`

`tests/substrate/test_sem_execution_production.py` ships the production-path counterpart to `test_sem_pain_cascade.py`. Six tests covering layers 3-10 of the cascade: `build_executor` → `executor.execute` → `ModulatorAffordanceTool` → `ToolOutput.side_effects` routing → `record_tool_embodiment_failure` → NAc direct attribution → `nac.predict` returns NEGATIVE → repeated cascades strengthen confidence → per-entity isolation holds → policy story (NEGATIVE prediction informs decision).

Notably the test now goes through `build_executor` (the new canonical constructor from `executor_bootstrap_unification.md`), not the old `runtime/embodiment_bootstrap.py` helper. The constructor's `entity_ref=` kwarg loads the SEM body and stores the `Embodiment` on the new declared `Executor.embodiment` field (the I1 cross-confirmed review fold).

The PoC file in `test_sem_pain_cascade.py` gets a deprecation note in its module docstring pointing readers at the production file. Deletion is deferred until the production file has been load-bearing for at least one bug-find cycle.

**Pass criteria (Stage 3):** ✓ 6 tests passing in `tests/substrate/test_sem_execution_production.py`. No mocks in the cascade chain.

### Stage 4 — CLI smoke + doctor check + docs — **SHIPPED** on `feat/sem-execution-hook-stages-3-4`

- ✓ New `maxim doctor --embodiment <REF>` flag + `check_embodiment_ref` check in `doctor/checks.py`. The check validates that the ref resolves in `ComponentRegistry`; on failure the fix hint groups same-category alternatives first (so a typo in `weapons/X` surfaces the available weapons before unrelated categories) and caps the per-category preview at 20 entries to keep doctor output bounded. Section is hidden by default — only appears when `--embodiment` is passed, preserving the existing doctor UX for users who aren't using SEM bodies.
- ✓ 8 new unit tests in `tests/unit/test_doctor.py::TestCheckEmbodimentRef` and `TestDoctorCLIEmbodimentFlag`.
- ✓ `docs/user/cli-reference.md` — new row in the Core Runtime table.
- ✓ `docs/embodiment_guide.md` — new "Running an agent with a SEM body (production path)" section between the existing "Load and run" and "Add virtual entities" subsections. Includes the validation flow + current constraints.

**Pass criteria (Stage 4):** ✓ doctor check passes for `weapons/rusty_sword`, fails with actionable hint for unknown refs; doctor section hidden when flag not passed; CLI reference + embodiment guide updated; 14 new tests passing (6 production cascade + 8 doctor).

**Smoke run validation:** deferred. The end-to-end test through the production constructor covers the cascade mechanism deterministically; a smoke run with a real LLM is optional and should be captured as `docs/experiments/results/sem_execution_hook_stage4_smoke_YYYYMMDD.md` if/when one is run.

## Pre-merge review round

Same template as substrate P2 Stage 2 and Stage 3 — spawn Executor + Architecture reviewers on the branch tip before merging. The specific review questions for this plan:

**Executor lens:**
- Is `generate_tools_for_entity` safe to call during agent startup (vs sim startup), or does it assume sim-owned state?
- Does the existing `ToolPainBridge.record_tool_start` signature match what `ModulatorAffordanceTool.execute` produces? (Event type will be `"tool"`, event signature will be `"tool:rusty_sword_slash"`, context will have `tool_name` / `invocation_id` — but NOT `source=embodiment` or `entity=<path>` unless we inject them.)
- Does `create_pain_nac_subscriber` correctly attribute the `tool` pending event to the `embodiment` pain outcome? The P2 Stage 2 fix made `_context_similarity` directional with `len(event_context)` as denominator — check that the event context is a subset of the outcome context.
- Are there re-entrancy concerns with `evaluate_failures()` firing inside `tool.execute` which runs inside `executor.execute`?

**Architecture lens:**
- Is adding `embodiment` as an optional kwarg to `LoopController` the right shape, or should it be a separate bootstrap concern owned by `AgentFactory`?
- Does this shift the agent loop toward "runs in a physical body by default" — is that the intent, or should `--embodiment` stay an explicit opt-in with a clear "no body" default?
- Does this create any circular import risk between `runtime/` and `embodiment/`? The P2 Stage 2 audit found no existing edges in this direction.
- Is the CLI flag spelling `--embodiment <ref>` consistent with existing flag conventions (`--llm`, `--sim`, `--language-model`)?

## Deferred follow-ups

- **Multi-entity bodies** — a full robot arm or NPC character has a tree of entities. Stage 1 loads a single entity; multi-entity support ships when a behavioral experiment actually needs it.
- **Runtime SEM mutation** — adding / removing entities at runtime (broken tool replaced with a new one). Current plan does not support this; adding it is a separate plan.
- **Fear bridge + predictive harm integration** — `PainCircuitBridge` and `JointLimitHarmPredictor` already exist but are currently wired only via sim. After this plan ships, they should also wire through `LoopController`. Separate plan.
- **Permissions gate on SEM affordances** — `agents/permissions.py::can_access_sem` exists but may not be enforced in the auto-generated tool path. Audit as part of Stage 1.
- **The PoC harness in `test_sem_pain_cascade.py::PoCAgent` should migrate.** Once the production path exists, the PoC harness should either (a) delete and replace with the production-path test, or (b) stay as a documented "simpler mock for mechanism tests that don't need the full tool registry." Decide during Stage 2.

## Parallel-session safety

This plan is **designed to ship as a parallel session** alongside the substrate binding track (P3a → P4). File-level overlap audit:

- **Files this plan touches:** `runtime/loop_controller.py`, `runtime/agent_loop.py` (maybe), `runtime/agent_factory.py`, `cli.py`, `tests/substrate/test_sem_execution_production.py` (new), `docs/user/cli-reference.md`, `docs/embodiment_guide.md`, `docs/user/dm-campaigns.md` (maybe).
- **Files the substrate binding track touches:** `src/maxim/memory/hippocampus.py`, `src/maxim/memory/atl.py`, `src/maxim/memory/cross_layer.py`, `src/maxim/decisions/nac.py`, `src/maxim/similarity/ec.py`, `src/maxim/agents/bus.py` (for the `DependencyGraph` — read-only, no edits), `tests/substrate/p3a_metrics.py` (new), `tests/substrate/test_p3a_episode_binding.py` (new), `scenarios/substrate/synthetic_episodes.yaml` (new), `docs/plans/substrate_p3a_*.md` (new).
- **Zero file-level overlap.**

**Coordination notes for the parallel session:**

1. **Worktree discipline.** Run in a separate git worktree from the substrate session. Follow `feedback_installed_package_shadows_worktree.md` — `PYTHONPATH=src python -m pytest ...` when running tests, or `pip install -e .` inside the worktree if staying for a long session. The parent checkout's editable install will otherwise shadow your edits.
2. **Branch off `main` AFTER the latest substrate merge.** If substrate_p3a ships before you start, rebase onto the post-merge main. If you start first, substrate_p3a rebases onto your merged output.
3. **Do NOT touch any file under `src/maxim/memory/`, `src/maxim/decisions/`, `src/maxim/similarity/`, `src/maxim/reactions/`, `src/maxim/proprioception/pain_bus.py`, or `src/maxim/embodiment/body.py`.** Those are substrate-track territory. If Stage 1's audit finds you genuinely need to read them (not edit), that's fine.
4. **`src/maxim/embodiment/tool_bridge.py` and `src/maxim/embodiment/motor.py`** are in your territory. The substrate track does not touch them.
5. **Tests in `tests/substrate/` are SHARED.** New test files you add (e.g., `test_sem_execution_production.py`) won't collide, but if you edit `test_sem_pain_cascade.py` (from P2 Stage 2), that's a substrate-track file — coordinate first.
6. **Pre-merge review round** on Executor + Architecture lenses is MANDATORY before merge, same as substrate plans. Use the `feedback_review_before_ship.md` template. Two reviewers, parallel, independent; fold all critical + important findings before opening the PR.

## Scope summary

| Item | LOC | Notes |
|---|---|---|
| Stage 1: audit existing wiring + CLI flag + `LoopController` startup hook | ~100-200 | Depends heavily on the Stage 1 audit result. Could be as small as 50 LOC if existing wiring is closer than I think. |
| Stage 2: end-to-end production test | ~150 | Ports the P2 Stage 2 PoC to use the real executor + executor-path event_type |
| Stage 3: doctor check + CLI help + docs | ~100 | Including the sim-run smoke log verification |
| Pre-merge review round fold-in | ~50 | Typical volume for a 2-reviewer round on a ~300 LOC PR |
| **Total** | **~300-500** | **Sharpens after Stage 1 audit** |

## Exit criteria

All criteria are MET as of 2026-04-14 with the Stages 1-4 ship. The Stage 2c criterion is satisfied by the structural enforcement in `executor_bootstrap_unification.md` (which made the silent-no-op bug class impossible to reproduce on any agent entry point); Stage 2b's criterion is satisfied by the explicit deferral to `agent_factory_canonicalization.md` Stage F1+, where it becomes ~2 hours of folding embodiment kwarg into the AgentFactory rewrite.

- ✓ **End-to-end pain cascade verified deterministically** through `tests/substrate/test_sem_execution_production.py` against the bundled `weapons/rusty_sword`, with zero mocks in the cascade chain. The original "smoke-run with `--sandbox tmpdir`" criterion has been replaced by this deterministic test: it covers the same code path through `cli.py → build_executor → executor.execute → ModulatorAffordanceTool → record_tool_embodiment_failure → NAc` and asserts the same NEGATIVE prediction outcome, but without the LLM cost or platform variability of a real-LLM smoke run. A manual smoke run is an OPTIONAL follow-up if a regression suggests the LLM-driven tool selection layer needs verification (capture as `docs/experiments/results/sem_execution_hook_stage4_smoke_YYYYMMDD.md` if/when run).
- ✓ End-to-end production test green: 6 tests in `test_sem_execution_production.py` covering layers 3-10 of the cascade.
- ✓ `maxim doctor --embodiment <REF>` validates refs against `ComponentRegistry` and fails with a prefix-aware available-list fix hint on typos.
- ✓ `docs/embodiment_guide.md` has section 2.5 "Running an agent with a SEM body (production path)".
- ✓ `tests/substrate/test_sem_pain_cascade.py` PoC file gets a deprecation note in its module docstring pointing readers at the production file. Deletion deferred until the production test has been load-bearing for one bug-find cycle.
- ✓ Pre-merge review round completed (Executor + Architecture lenses, parallel) with cross-confirmed C1/I1-I7 + arch-only C2/A-arch-2/A-arch-3 folded. See the fold commit on `feat/sem-execution-hook-stages-3-4`.

## Not in this plan

- Multi-entity bodies (deferred)
- Runtime SEM mutation (deferred)
- Harm/fear bridge wiring into the standard agent loop (deferred — separate plan when a behavioral scenario needs it)
- Permissions enforcement on SEM tools (audited in Stage 1; fix in a separate PR if Stage 1 finds it missing)
- Anything requiring substrate phases P3/P4/P5/P6/P8 to ship first
