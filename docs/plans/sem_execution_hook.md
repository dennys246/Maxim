# SEM Execution Hook — Production Forward Path for Affordances

**Status:** Draft, awaiting approval via [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md).
**Scope:** ~200-400 LOC (estimate sharpens after Stage 1 audit).
**Target version:** Ships anytime. Not on the substrate version-gate path.
**Gates:** NOTHING in the release matrix, but unblocks behavioral convergence experiments that need real SEM execution loops (H1, H2, H4 in [behavioral_convergence_practice.md](behavioral_convergence_practice.md)).
**Depends on:** substrate_recognition (✅ shipped), P2 Stage 2 pain cascade (✅ shipped).
**Blocks:** Any production experiment that needs an agent to invoke SEM affordances from its prompt output (most of behavioral_convergence_practice H1-H4, the multi-session cross-modal sanity runs in P4).
**Parent:** none — cross-cutting plan, not part of substrate_binding_persistence.
**Related:** [substrate_binding_split_proposal.md](substrate_binding_split_proposal.md) (discovery), [substrate_recognition.md](substrate_recognition.md) (Stage 2 pain cascade PoC), [../experiments/p2_sem_pain_cascade.md](../experiments/p2_sem_pain_cascade.md) (the PoC this plan generalizes to production).

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

### Stage 1 — audit + minimal startup hook

Before writing any code, **AUDIT the existing wiring more carefully** (the 2026-04-14 planning audit found the cascade was already wired; it's entirely possible a second audit finds that the startup hook exists too and is just undocumented). Specifically:

1. **Read `runtime/loop_controller.py` LoopController.__init__`** — does it accept an `embodiment` parameter? If yes, what does it do with it?
2. **Read `runtime/agent_factory.py`** — does AgentFactory construct an Embodiment? If yes, how is the tree specified (config file, CLI flag, default bundled component)?
3. **Read `cli.py::main`** — is there a CLI flag to load a SEM component at agent startup? (`--embodiment`, `--sem-entity`, etc.)
4. **Check `MaximAgent` / `AgentPool`** — do they hold an `Embodiment` reference?
5. **Run `grep -rn "Embodiment(" src/maxim/ | grep -v test`** — every production call site for `Embodiment` construction.

If the audit finds existing startup wiring, Stage 1 becomes "wire `generate_tools_for_entity` into the existing path." If not, Stage 1 is:

- Add `embodiment: Embodiment | None = None` parameter to `LoopController.__init__` (or wherever the agent loop accepts its runtime dependencies).
- Add a CLI flag `--embodiment <component_ref>` that loads a SEM component from the `ComponentRegistry` (using the `weapons/rusty_sword` ref format from P2 Stage 2).
- In `LoopController` startup: if `embodiment is not None`, call `generate_tools_for_entity(embodiment.root, tool_registry, embodiment=embodiment, cerebellum=cerebellum)` so affordance tools register alongside the normal tool set.
- In the executor wiring: pass `embodiment` through so `evaluate_failures()` fires post-action.

**Pass criteria (Stage 1):** `maxim --llm mistral-7b --embodiment weapons/rusty_sword` starts an agent that has `rusty_sword_slash`, `rusty_sword_parry`, `rusty_sword_throw`, `rusty_sword_sharpen`, `rusty_sword_repair` in its tool list, and invoking any of them via the LLM's tool-use path fires through `ModulatorAffordanceTool.execute`. Unit test: construct a mock agent with a rusty_sword embodiment, have it call `executor.execute({"tool_name": "rusty_sword_slash", "params": {"target": "dummy", "force": 0.5}})`, assert the returned ToolOutput contains `active_failures` list.

### Stage 2 — end-to-end pain cascade integration test (no sim orchestrator)

Port the P2 Stage 2 PoC (`tests/substrate/test_sem_pain_cascade.py`) to use the production wiring path:

- Instantiate `ComponentRegistry.instantiate("weapons/rusty_sword")`
- Wrap in `Embodiment(sword, pain_bus=pain_bus)`
- Construct the full agent via `LoopController` or `AgentFactory` (whichever is the audit-discovered entry point), passing the embodiment
- Call `executor.execute({"tool_name": "rusty_sword_slash", ...})` directly (don't route through an LLM) — this simulates the LLM's tool-use choice deterministically
- Drive durability low to trigger shatter
- Assert `nac.predict("tool", "tool:rusty_sword_slash", context={source, entity})` returns `NEGATIVE`

Compare to the existing PoC: the existing test uses a custom `PoCAgent` that calls `nac.record_event("action", "slash:rusty_sword", context={source, entity})` directly. The production test uses `executor.execute(...)` which goes through `ToolPainBridge.record_tool_start` which calls `nac.record_event("tool", "tool:rusty_sword_slash", context={...})`. The `event_type` and `event_signature` differ. This is the REAL difference between the PoC and production, and the test needs to handle it.

**Pass criteria (Stage 2):** end-to-end test in `tests/substrate/test_sem_execution_production.py` that runs the full cascade through the production executor, asserts NEGATIVE prediction after one learning cycle. No mocks in the chain. Matches the shape of `test_sem_pain_cascade.py::test_agent_prefers_drop_weapon_after_learning_slash_is_painful` but with the real agent entry point instead of the PoC harness.

### Stage 3 — CLI smoke + doctor check + docs

- `maxim doctor` check that warns if `--embodiment` was specified but the component ref doesn't exist (use the error-hint pattern from existing doctor checks)
- CLI help text for `--embodiment` + one-line note in `docs/user/cli-reference.md`
- Update `docs/embodiment_guide.md` with a "running an agent with a SEM body" section pointing at the new CLI flag
- Smoke run: `maxim --llm claude-haiku --embodiment weapons/rusty_sword --goal "test the sword" --sandbox tmpdir` and eyeball the log for a `tool_invoke` event on `rusty_sword_slash` followed by a `pain_published` event

**Pass criteria (Stage 3):** smoke run produces the expected log sequence; doctor check surfaces missing-component errors cleanly.

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

- `maxim --llm X --embodiment weapons/rusty_sword --sandbox tmpdir` runs an agent that has `rusty_sword_*` tools available and produces the full pain cascade when invoking them
- End-to-end production test green without mocks in the chain
- `maxim doctor` warns on missing `--embodiment` refs
- `docs/embodiment_guide.md` has a working "running with a body" section
- TODO comment in `tests/substrate/test_sem_pain_cascade.py::PoCAgent` removed or repointed at the production path
- Pre-merge review round completed (Executor + Architecture) with all critical + important findings folded

## Not in this plan

- Multi-entity bodies (deferred)
- Runtime SEM mutation (deferred)
- Harm/fear bridge wiring into the standard agent loop (deferred — separate plan when a behavioral scenario needs it)
- Permissions enforcement on SEM tools (audited in Stage 1; fix in a separate PR if Stage 1 finds it missing)
- Anything requiring substrate phases P3/P4/P5/P6/P8 to ship first
