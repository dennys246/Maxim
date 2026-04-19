# Agent Loop State Repair — fix sim stall and flow issues

**Status:** Plan written (2026-04-19). Not started.
**Scope:** ~200-400 LOC across 4 files
**Priority:** HIGH — sims are stalling in production
**Blocks:** Any sim-dependent work (0.6 embodiment PoC, behavioral convergence experiments)

## Problem

Simulations stall because the AUT (Agent Under Test) enters an infinite followup loop. The bridge's settle timer never fires because the AUT keeps generating actions within the 3-second window. Root cause is a state desynchronization bug in the agent loop, compounded by missing safety caps in the bridge.

## Root Cause Analysis

### The state desync (agent_loop.py)

The agent loop initializes local variables from `LoopController` at lines 449-460:

```python
pending_proposal = ctrl.pending_proposal
pending_next_actions = ctrl.pending_next_actions
pending_action_followup = ctrl.pending_action_followup
pending_plan_proposal = ctrl.pending_plan_proposal
```

These local variables are modified throughout the loop body but **never synced back to the controller**. On re-entry:
1. A tool with `followup_type` sets local `pending_action_followup` (line 1465)
2. Followup is processed, local variable cleared to `None` (line 1936)
3. `ctrl.pending_action_followup` still holds the old value
4. Line 682 re-reads from controller → stale followup re-fires
5. LLM generates another action → bridge settle timer resets → turn never ends

### The missing action cap (bridge.py)

The bridge's `send_and_wait()` settle loop (lines 137-156) resets the settle deadline every time a new action arrives. There is no maximum actions-per-turn cap, so the AUT can extend a single turn indefinitely by generating actions faster than the settle timeout.

## Stages

### Stage 1 — Fix state desynchronization (CRITICAL, unblocks sims)

**What:** Make local loop variables authoritative. Stop re-reading stale state from the controller.

**Changes in `runtime/agent_loop.py`:**
1. After clearing `pending_action_followup` (line 1936), also clear on the controller: `ctrl.pending_action_followup = None`
2. After clearing `pending_proposal` (lines 675, 1153, 1189, 1206), also clear on controller: `ctrl.pending_proposal = None`
3. After setting `last_llm_submit_time` (line 2094), also update controller: `ctrl.last_llm_submit_time = now`
4. After setting `pending_action_followup` (lines 1178, 1465, 1733), also set on controller
5. Remove the re-read at line 682 that reloads from controller mid-iteration — the local variable should be authoritative within an iteration

**Test:** Run `maxim --sim "test basic recall" --interactive false --sim-max-turns 3` and verify it completes all 3 turns without stalling in a followup loop.

### Stage 2 — Add action cap to bridge settle loop

**What:** Add `max_actions_per_turn` parameter to `SimulationBridge` to prevent unbounded action accumulation within a single turn.

**Changes in `simulation/bridge.py`:**
1. Add `max_actions_per_turn: int = 10` to `__init__` and `send_and_wait`
2. In the settle loop (lines 137-156), add a break condition: if `current_count - action_count_before >= max_actions_per_turn`, break immediately regardless of settle timer
3. Log a warning when the cap fires so it's visible in the trace

**Test:** Verify the bridge returns after 10 actions even if the AUT keeps generating more.

### Stage 3 — Prevent empty LLM proposals

**What:** Don't create `LLMProposal` objects with `action=None`. Return `None` from the worker instead.

**Changes in `agents/llm_worker.py`:**
1. Lines 901-913: When prompt is empty, return `None` instead of creating an empty proposal
2. Callers that check `get_latest_proposal()` already handle `None` (line 846 in agent_loop.py checks `if new_proposal.action:`)

**Changes in `agents/prompt_builder.py`:**
1. Lines 621-623, 648-651: Add early return with `None` before reaching the empty-string paths

### Stage 4 — Stall detector safety improvements

**What:** Prevent nudge storms and add stop_event awareness.

**Changes in `simulation/orchestrator.py`:**
1. Add a nudge cooldown: don't inject more than 1 nudge per 15 seconds (currently fires every 3s)
2. Add stop_event check in `_stdin_reader_raw()` — use `select.select()` with a 1-second timeout instead of indefinite blocking, check stop_event between polls
3. Separate turn counting: don't count pain injections as orchestrator turns in the bridge

### Stage 5 — Unify stale thresholds

**What:** Remove the hard-coded 35s stale check in agent_loop.py and use the worker's threshold consistently.

**Changes:**
1. `agent_loop.py` line 788: Read the stale threshold from the LLM worker instead of hard-coding 35s
2. Or: pass the threshold as a parameter to `run_agentic_loop`

## Pass criteria

- `maxim --sim "test basic recall" --interactive false --sim-max-turns 3` completes in <60 seconds
- `maxim --sim "dragon attack" --interactive false --sim-max-turns 5` runs 5 full orchestrator turns without stalling
- No `[EXEC] LLM submit: followup` loop visible in trace after a terminal `respond` action
- Bridge returns from `send_and_wait` within max_actions_per_turn actions
- Full test suite passes (5220+ tests)

## Load-bearing invariants

- **Local loop variables are authoritative within an iteration.** The controller is the persistence layer, not the runtime source of truth. Local modifications must sync back to the controller immediately.
- **Bridge settle has an action cap.** The settle loop MUST break after `max_actions_per_turn` actions regardless of settle timer state.
- **Empty prompts never reach the LLM.** If the prompt builder returns empty, no proposal is created, no LLM call is made.
- **Nudges have a cooldown.** The stall detector must not inject nudges faster than once per 15 seconds.

## Files touched

| File | Stage | What changes |
|------|-------|-------------|
| `runtime/agent_loop.py` | 1, 5 | State sync, stale threshold |
| `simulation/bridge.py` | 2 | Action cap in settle loop |
| `agents/llm_worker.py` | 3 | Empty proposal prevention |
| `agents/prompt_builder.py` | 3 | Early return on empty prompt |
| `simulation/orchestrator.py` | 4 | Nudge cooldown, stdin stop_event, turn counting |
| `runtime/loop_controller.py` | 1 | Verify controller fields are writable |
