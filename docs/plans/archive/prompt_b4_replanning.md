# Prompt B4 — Replanning with Failure Diagnosis (1.0-GATING)

**Status:** COMPLETE (2026-04-19). All 3 stages shipped. Results: [b4_replanning_results.md](../../experiments/b4_replanning_results.md)
**Scope:** ~400 LOC
**Target version:** 0.5 (formerly 0.4)
**Gates:** **1.0** — replanning is a core capability claim
**Depends on:** B1 (SHIPPED — prompt composition), P3a (SHIPPED — episode retrieval of prior attempts)
**Blocks:** 1.0 release
**Parent:** [substrate_binding_persistence.md](substrate_binding_persistence.md)
**Related:** [behavioral_convergence_practice.md](../deferred/behavioral_convergence_practice.md)

## Goal

When an agent's plan fails, it should diagnose the failure, recall prior attempts via episode retrieval, and generate a structurally different plan that doesn't repeat the same mistakes. This is a core 1.0 capability — without it, the agent is a one-shot planner.

## Hypothesis

Given an induced failure, the agent's second plan differs structurally from its first plan, and its third plan doesn't repeat either. Prior attempt retrieval via P3a episodes provides context that prevents repetition.

## Minimum implementation (~400 LOC)

- `runtime/replanner.py`: Structured replan engine
  - Failure point identification (which step failed, what error)
  - Evidence collection (tool outputs, observations at failure)
  - Prior attempt retrieval (hippocampus episode lookup for same goal)
  - Root cause analysis prompt (LLM-driven)
  - Alternative generation with anti-repetition constraint
  - Selection with confidence scoring
- In-session replan attempt persistence (hippocampus captures each attempt as an episode)
- Integration with `runtime/agent_loop.py` plan failure handler
- `AdaptivePlanner` or extension of existing `PlanManager`

## Stages

### Stage 1 — mechanism + metric

**What's built:**
- Replan engine with failure diagnosis prompt
- Prior attempt retrieval via hippocampus episodes
- Metric: structural difference between plan N and plan N+1 (Jaccard distance on action sequences)
- Mechanism test: induced failure → replan → verify structural difference

**Pass gate:** Plan 2 differs from Plan 1 (Jaccard distance > 0.3). Plan 3 differs from both.
**Tests:** `tests/unit/test_replanner.py`

### Stage 2 — induced-failure scenario

**What's built:**
- Scenario fixture: multi-step goal with injected failure at step 3
- Full agent loop integration (plan → execute → fail → replan → execute)
- Prior attempt accumulation across replans

**Pass gate:** Agent recovers within 3 replan attempts. Each attempt structurally different. Mean across 5 seeds.

### Stage 3 — blind A/B + pre-merge review (SHIPPED 2026-04-19)

**What's built:**
- Blind A/B: replanning agent vs no-replanning agent on 5 seeded failure scenarios
- Structural quality judge rates plan diversity, failure awareness, and goal coverage
- 12 tests in `tests/substrate/test_b4_replanning_ab.py`
- Pre-merge two-lens review

**Results:** Treatment 100% (5/5) vs control 0% (0/5). Mean Jaccard 0.894, min 0.600. All plans pass quality judge.
**Pass gate:** CLEARED — replanning agent succeeds on all seeds, control on none. Plans structurally novel.
**Reviewers:** Executor + Architecture lenses

## Pass criteria (maps to 1.0 gate)

- Induced failure → plan 2 differs structurally from plan 1
- Plan 3 doesn't repeat either plan 1 or plan 2
- Prior attempt retrieval provides useful context (ablation: replanning without retrieval is worse)
- Recovery within 3 attempts on standard failure scenarios

## Deferred follow-ups

- Cross-session replanning (remember failures from prior sessions)
- Replanning budget (cap replan attempts to prevent infinite loops)
- Collaborative replanning (multi-agent failure recovery)

## Load-bearing invariants

- **`ReplanContext.prior_attempt_actions` carries prior plan action sequences** — populated from `replan_history` (PlanManager path) or `hippocampus.recall(goal=...)` (agent loop path). Default empty list for backward compatibility.
- **`_build_replan_context` in loop_state.py returns a stub Phase, not a raw string** — pre-existing type bug fixed in this session. `to_llm_prompt_section()` accesses `failed_phase.description`.
- **Jaccard distance metric is in `planning/structural_diff.py`** — operates on `list[dict]` action sequences, no imports from agents/memory/runtime. Pure utility.
- **Anti-repetition constraint is a prompt enhancement, not a structural filter** — the LLM receives the constraint and prior attempts; structural difference is MEASURED (Jaccard > 0.3) but not ENFORCED at the code level. The measurement is for validation, not gating.
- **B4 Stage 3 SHIPPED (2026-04-19)** — blind A/B validation across 5 seeds. Treatment (replanning) 100% vs control (no replanning) 0%. Mean Jaccard 0.894, all pairwise > 0.3. Quality judge passes on all plans. Results: [b4_replanning_results.md](../../experiments/b4_replanning_results.md). Tests: `tests/substrate/test_b4_replanning_ab.py` (12 tests).
