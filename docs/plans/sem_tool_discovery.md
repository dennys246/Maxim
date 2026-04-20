# SEM Tool Discovery Plan (Shell)

**Status:** Shell (2026-04-20)
**Scope:** 0.7 — Simulation Scalability
**Depends on:** I3 (scene-scoped tools), E2.5 (ComponentIndex), SEM protocol

---

## Problem

As sessions grow (imagination generating entities, multiple scenes, complex encounters), the agent accumulates tools. Even after I3's scene-scoped cap (20 active), the flat tool list in the prompt is a poor interface for an LLM that needs to reason about *what it can do*. The agent sees `crystal_dragon_bite` but doesn't know what it does without reading the full description. Worse, with 20+ tools the LLM defaults to safe/known tools instead of exploring.

The SEM protocol already models this — entities have modulators (categories of capability) containing affordances (specific actions) with parameters, effects, and failure modes. This structure is richer than a flat tool list.

## Proposed Architecture

```
LLM prompt (slim):
  Core tools: say, think, examine, request_interaction, sense_<entity>
  + discover_tools(query)  ← NEW: semantic tool discovery

Agent thinks: "I want to attack the dragon"
  → discover_tools(query="attack dragon combat")
  → ComponentIndex.find("dragon") → ref → entity.modulators["combat"].affordances
  → Returns: crystal_dragon_bite(target, force), crystal_dragon_claw(target), crystal_dragon_tail_sweep(area)
  → These tools are now visible in the NEXT prompt turn

Alternative: agent discovers by modulator
  → discover_tools(query="what can I do?")
  → Returns modulator categories: locomotion, perception, manipulation, communication, combat
  → discover_tools(query="combat")
  → Returns specific affordances under combat modulator
```

## Key Design Questions

1. **Discovery vs always-visible:** Should the slim prompt show ZERO affordance tools (only discover_tools), or a curated subset (top-3 most relevant by NAc valence)?

2. **Latency:** ComponentIndex.find() is ~5ms. Acceptable per-turn. But should discovery return tool DESCRIPTIONS or just NAMES (with descriptions loaded on-demand)?

3. **SEM protocol integration:** The SEM spec has `modulators → affordances` hierarchy. Should discover_tools expose this hierarchy (two-step: modulators first, then affordances) or flatten it?

4. **Interaction with I3 scene-scoped window:** Discovery finds tools across all scenes (including deactivated). Should it auto-reactivate discovered tools? Or just report them with their scene status?

5. **Acting Coach integration:** B3's exploration directives currently enumerate all affordances. With discovery, the coach should say "explore your capabilities via discover_tools" instead of listing them.

## Stages (TBD)

**S1 — discover_tools tool + slim prompt mode**
- New tool: `DiscoverToolsTool(component_index, component_registry, tool_registry)`
- Query routes through ComponentIndex.find() for entity matching, then extracts modulator/affordance info from the spec
- Slim prompt mode: only core tools + discover_tools shown; affordance tools hidden but executable

**S2 — Modulator-aware discovery**
- Two-step discovery: modulators first, then affordances within a modulator
- SEM spec's modulator hierarchy becomes the navigation structure

**S3 — NAc-informed discovery ranking**
- Discovery results ranked by NAc valence — tools with positive outcomes float to top
- Tools with negative valence get caution annotations (reuses B3 Acting Coach pattern)

**S4 — Auto-discovery on imagination**
- When ImaginationTrigger creates a new entity, auto-run discovery and surface the top-3 affordances in the next prompt turn
- Replaces the current "register all tools immediately" pattern

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| S1 | ~200 | Tool + slim prompt mode |
| S2 | ~100 | Modulator hierarchy |
| S3 | ~100 | NAc ranking |
| S4 | ~50 | Imagination integration |
| **Total** | **~450** | |

## Open Questions

- Should the orchestrator also get discover_tools? (Probably not — orch has a fixed tool set)
- Should discovery work outside sim mode? (Yes for Reachy — camera percepts → "what tools work on this object?")
- How does this interact with the prompt token budget? Fewer tools = more room for context.
