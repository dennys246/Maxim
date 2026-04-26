# Orchestrator Stall Recovery — smarter tool-cap and idle handling

**Status:** Shell plan (2026-04-22)
**Scope:** ~200-400 LOC
**Priority:** High — the orch is the bottleneck, not the AUT
**Depends on:** context-aware stall nudges (shipped 2026-04-22)
**Gates:** none
**Target version:** 0.8.x or 1.0.x

---

## Problem

In every sim run (4 runs, 30+ turns observed), the orchestrator hits the same failure pattern:

1. Orch calls `send_message` 6 times (1 per turn)
2. "Consecutive same-tool cap hit: send_message called 6 times — breaking chain"
3. Orch goes idle for 30s (has no recovery strategy after the cap breaks its chain)
4. Stall detector fires, nudges orch back to `send_message`
5. Repeat from step 1

The AUT is behaving reasonably (purposeful goals, diverse actions). The orch can't drive a coherent multi-turn scenario because:
- The consecutive-tool cap is designed for hallucination loops, not normal orch behavior (`send_message` IS the orch's primary tool)
- After the cap fires, there's no fallback strategy — the orch just... stops
- The orch prompt doesn't encode "you've been testing X for N turns, try something different" — it has no self-awareness of its own repetition

## Root causes

### R1. `send_message` should be exempt from the consecutive-tool cap
The cap exists to break hallucination loops where the LLM calls the same tool with the same params repeatedly. But `send_message` with different text is the orch's INTENDED behavior — it's how it drives the sim. The cap should either exempt `send_message` or distinguish "same tool, same params" from "same tool, different params."

### R2. Post-cap recovery is missing
When the cap fires, the orch gets an error-like message but no guidance on what to do instead. It should get: (a) a summary of what it's tested so far, (b) a suggestion to try a different approach, and (c) the tools it COULD use instead (inspect_aut, check_completion, spawn_sub_simulation).

### R3. Orch has no self-model of its testing strategy
The orch doesn't track "I've asked 4 variations of the same question." It needs either:
- A sliding window summary of its own recent probes (analogous to the AUT's WorkingMemorySet)
- Or a simpler "diversity nudge" injected into its prompt after N same-category probes

## Proposed approach (pick one or combine)

### Option A: Exempt send_message from consecutive cap
- In the consecutive-tool-cap logic, whitelist `send_message` (and maybe `observe_actions`)
- Keep the cap for other tools (prevents real hallucination loops on `respond`, `say`, etc.)
- ~20 LOC. Low risk. Doesn't fix R3 but unblocks the immediate pattern.

### Option B: Content-aware cap (same tool + similar content)
- Track the last N `send_message` texts
- Only fire the cap when Jaccard keyword similarity between consecutive sends > 0.7
- Different probes to the same tool don't trigger
- ~80 LOC. Moderate risk. Addresses R1 without whitelisting.

### Option C: Orch working memory / diversity injection
- After N turns, inject a "testing strategy summary" into the orch prompt
- List what categories have been tested, what the AUT's response pattern was
- Suggest untested categories or deeper follow-ups
- ~200 LOC. Higher complexity but addresses R3 (the root cause).

**Recommendation:** Start with Option A (immediate unblock) + Option C (strategic improvement). Option B is a refinement if Option A produces actual hallucination loops (unlikely — orch probes naturally vary).

## Key files

| File | Change |
|------|--------|
| `simulation/orchestrator.py` | Tool-cap exemption/refinement, post-cap recovery |
| `simulation/bridge.py` | Possibly: expose recent probe summary for diversity injection |
| `runtime/executor.py` | Consecutive-tool-cap logic lives here |

## Validation

Re-run the dungeon escape sim (8 turns). Success = no 30s idle gaps caused by tool-cap → stall → nudge cycle. The orch should drive all 8 turns without stalling.

## Where the tool cap logic lives

```
runtime/executor.py — _check_consecutive_tool_cap()
```

Grep for `Consecutive same-tool cap` to find the exact site. The cap is currently a flat count (6) with no tool-name awareness.
