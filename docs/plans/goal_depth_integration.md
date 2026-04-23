# Goal-Depth Integration — bio-system goal awareness

**Status:** Shell plan (2026-04-22)
**Depends on:** [pfc_deliberation_cycle.md](pfc_deliberation_cycle.md) (must ship first)
**Scope:** ~200-300 LOC across existing modules (no new bio-system)

---

## Problem

Goals flow through the system but lack depth. AgenticGoalAgent, PlanManager, and the bus messages (ProposedGoal, GoalAccepted, GoalCompleted) handle immediate proposal/execution, but:

1. **WorkingMemorySet has no GOAL entry kind.** The PFC deliberation cycle enriches from THOUGHT, PERCEPT, RECALL — but the active goal and its status are invisible as first-class working memory entries. Goals live in MemoryAgent's `_active_goal` field and get injected into StructuredContext, but the cycle can't reason about them directly.

2. **NAc learns action-outcome, not goal-outcome.** "Tool X succeeded" is learned; "pursuing escape in a guarded room tends to fail" is not. The causal link surface stops at actions.

3. **Hippocampus has no goal tagging.** Episodes aren't tagged with which goal was active when they formed. The PFC cycle can't ask hippocampus "what happened last time I tried this kind of goal?"

4. **Goals don't survive session restart.** MemoryAgent's goal state is ephemeral. PlanManager phases persist, but the goal itself doesn't.

## Bio-plausible framing

Dorsolateral PFC maintains goals in working memory. Orbitofrontal PFC evaluates goal value (NAc). Hippocampus provides episodic context for goal-relevant decisions. There is no "goal organ" — the existing structures coordinate around the active goal. This plan wires that coordination.

## Design sketch

### Stage 1: GOAL entry in WorkingMemorySet

Add `GOAL` to `WorkingMemoryEntryKind` in `agents/working_memory.py`. When AgenticGoalAgent accepts a goal (GoalAccepted bus message), add a GOAL entry to the working memory set. Update on sub-goal completion. The PFC deliberation cycle already reads all WMS entries — GOAL entries become visible to the LLM automatically.

### Stage 2: Goal tagging on episodes

Thread `goal_id` through MemoryAgent → Hippocampus at episode capture time. Add `goal_id: str | None` to `Episode` (backward-compatible default `None`). When recalling, allow filtering by goal context — "memories from when I was pursuing goal X."

### Stage 3: NAc goal-outcome learning

When `GoalCompleted` fires, record `nac.observe("goal:<description>", valence)`. This creates causal links at the goal level, not just the action level. The PFC cycle's bio-enrichment already queries NAc predictions — goal-level predictions surface automatically.

### Stage 4: Goal persistence across sessions

Persist active goal + sub-goal state alongside PlanManager's phase state. Load on session start so the agent resumes goal-aware.

## Composability with PFC

The PFC deliberation cycle doesn't need modification. It reads from working memory, queries NAc/hippocampus, and enriches via BioEnrichmentPipeline. Each stage above enriches the *data* the cycle sees, not the cycle logic. This is intentional — the cycle is a general-purpose recurrence loop, and goal awareness is richer input.

## Validation

- Run a multi-turn sim and confirm GOAL entries appear in working memory alongside THOUGHT/RECALL
- Run a cross-session sim and confirm goal-level NAc predictions surface in bio-enrichment
- Verify hippocampus recall with goal_id filter returns relevant episodes
