# SEM Tool Discovery S1 — Hybrid Prompt Mode PoC

**Date:** 2026-04-20
**Plan:** [docs/plans/sem_tool_discovery.md](../plans/sem_tool_discovery.md)
**Stage:** S1 (Universal sense + discover_tools + hybrid prompt mode)
**Status:** In progress

---

## Hypothesis

The hybrid prompt mode reduces AUT tool count from ~36 to ~12 while preserving turn-1 affordance visibility. The agent should:
1. Use goal-relevant affordance tools (top-k) on turn 1 without needing `discover_tools`
2. Call `discover_tools` when it needs capabilities not in the top-k
3. Successfully call `sense` to read entity state

## Setup

```bash
# Run with JSONL logging for analysis
MAXIM_LOG_FILE=/tmp/sem_discovery_s1.jsonl \
  maxim --sim "test sword combat" \
    --embodiment weapons/rusty_sword \
    --interactive false \
    --sim-max-turns 5

# Compare against baseline (no discovery)
# (Run on pre-S1 commit for A/B comparison)
```

**Version:** 0.7.0 + S1 (commit TBD)
**Model:** Whatever the leader is running (check with `maxim peer version`)

## What to check in the JSONL

1. **Tool count:** Grep for tool names in the first LLM request. Should be ~12, not ~36.
   ```bash
   grep "available_tools" /tmp/sem_discovery_s1.jsonl | head -1 | python -m json.tool
   ```

2. **Turn 1 tool use:** Does the agent call an affordance tool (e.g., `rusty_sword_slash`) on turn 1?
   ```bash
   grep "tool_call" /tmp/sem_discovery_s1.jsonl | head -3
   ```

3. **discover_tools usage:** Does the agent call `discover_tools` at any point? What query?
   ```bash
   grep "discover_tools" /tmp/sem_discovery_s1.jsonl
   ```

4. **sense usage:** Does the agent call `sense` instead of per-entity sensor tools?
   ```bash
   grep '"sense"' /tmp/sem_discovery_s1.jsonl
   ```

5. **Deactivated tools not called:** Agent should NOT call deactivated affordance tools.
   ```bash
   grep "tool_not_found\|not active" /tmp/sem_discovery_s1.jsonl
   ```

## Expected Results

| Metric | Baseline (pre-S1) | S1 (hybrid) |
|--------|-------------------|-------------|
| Active tools turn 1 | ~36 | ~12 |
| Turn-1 affordance use | Yes | Yes (via top-k) |
| discover_tools calls | N/A | 0-2 per session |
| sense calls | N/A | 1-3 per session |
| Token budget (tools section) | ~1,200 | ~400-600 |

## Actual Results

_(To be filled after sim run completes)_

## Reproduction

See [protocols/sem_tool_discovery_s1_reproduction.md](protocols/sem_tool_discovery_s1_reproduction.md).
