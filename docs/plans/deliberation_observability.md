# Deliberation Observability — sim_log for the thinking pipeline

**Status:** Shell plan (2026-04-22)
**Scope:** ~100-150 LOC
**Priority:** Highest — can't iterate on what you can't see
**Depends on:** working_memory_exec_loop.md (shipped), post-ship refinement (shipped)
**Gates:** none
**Target version:** 0.8.x patch

---

## Problem

The pre-deliberation pipeline (`_run_pre_deliberation`), ThoughtGate decisions, and bio-enrichment results are invisible in sim output. `ExecAgent` logs via `log_structured` (Python stdlib logger) which doesn't reach the sim terminal or the sim JSONL log. Other bio-systems (Fear, NAc, Hippocampus, ATL, SCN) all have `sim_*` helpers in `simulation/sim_logger.py` and show on the terminal. The thinking system is the only bio-pipeline stage with zero sim visibility.

**Consequence:** During the 2026-04-22 sim evaluation, we had to infer whether pre-deliberation fired from indirect evidence (hippocampus goal strings, action diversity). Direct confirmation was impossible.

## What to add

### 1. New sim_log helpers in `simulation/sim_logger.py`

```python
def sim_pre_deliberation(
    gate_passed: bool,
    score: float,
    threshold: float,
    enrichment_sections: int,  # how many bio-system responses were non-empty
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Layer 1 pre-deliberation decision + enrichment summary."""

def sim_contemplation(
    gate_passed: bool,
    refined: bool,
    score: float,
    *,
    agent_id: str | None = None,
) -> None:
    """Log a Layer 3 post-proposal refinement decision."""
```

### 2. Call sites in `agents/exec_agent.py`

| Location | What to log |
|----------|-------------|
| `_run_pre_deliberation` after gate decision | `sim_pre_deliberation(passed, score, threshold, N)` |
| `_run_pre_deliberation` after enrichment | section count (memories, predictions, concepts, affordances, recent_context) |
| `_maybe_contemplate` after gate decision | `sim_contemplation(passed, refined, score)` |
| `_propose_goal` after `_maybe_contemplate` returns | Whether pre_deliberated + contemplated in goal_proposed log |

### 3. Display tier

Both helpers should emit at the **BIO** display tier (same as THOUGHT, FEAR, NAc). They should show on the default terminal display, not just in debug mode.

### 4. JSONL persistence

The `sim_log` function already persists to the sim JSONL when `log_path` is set. The new helpers should include structured `data` dicts so the enrichment results can be analyzed programmatically.

## Key constraint

`sim_log` checks `_sim_active` which is set by `enable_sim_logging()`. This is called during sim setup. ExecAgent runs on its own background thread (`_worker_loop`), but `_sim_active` is a module-level bool — thread-safe for reads. The `_current_agent_id` contextvar may not propagate to the worker thread; pass `agent_id=` explicitly.

## Validation

After shipping, re-run the dungeon escape sim and confirm:
- Terminal shows `[THOUGHT]` lines with gate decisions and enrichment counts
- JSONL contains structured records with `subsystem: "THOUGHT"` or `subsystem: "DELIBERATION"`
- Both pre-deliberation and post-proposal gate decisions are visible

## Files to touch

| File | Change |
|------|--------|
| `simulation/sim_logger.py` | Add `sim_pre_deliberation()` + `sim_contemplation()` |
| `agents/exec_agent.py` | Call sites in `_run_pre_deliberation` + `_maybe_contemplate` |
