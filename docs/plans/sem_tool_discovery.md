# SEM Tool Discovery Plan

**Status:** Ready (2026-04-20)
**Scope:** 0.7 — Simulation Scalability
**Depends on:** I3 (scene-scoped tools), E2.5 (ComponentIndex), SEM protocol, B3.1 (Acting Coach)

---

## Problem

As sessions grow (imagination generating entities, multiple scenes, complex encounters), the agent accumulates tools. The flat tool list in the prompt is a poor interface for an LLM that needs to reason about *what it can do*:

**Tool count math for a sim session with `--embodiment weapons/rusty_sword`:**

| Source | Tools |
|--------|-------|
| `build_tool_registry` core | ~27 |
| Deregistered in orchestrator (robot + dev) | -17 |
| Narrative tools (say, think, examine) | +3 |
| Display tools (display_mode, request_interaction, set_scene) | +3 |
| base_humanoid entity (5 sensors × read + 1 sense + 8 affordances) | +14 |
| rusty_sword entity (3 sensors × read + 1 sense + 5 affordances) | +9 |
| Each imagined entity (avg) | +8-15 |
| **With one imagined entity** | **~45** |

Even with I3's cap at 20 *scene* tools, core + non-scene tools stay at ~16. The LLM sees 36 active tools with no structure. Each tool costs ~30-50 tokens in the prompt. 36 tools = ~1,000-1,800 tokens per turn, every turn.

The SEM protocol already models this richer — entities have modulators (combat, locomotion, perception) containing affordances (slash, parry, throw) with parameters, effects, and failure modes. `tool_bridge.py` **flattens** this hierarchy: `rusty_sword_slash`, `rusty_sword_parry`, `rusty_sword_sharpen` appear as unrelated tools. The modulator names are the natural discovery categories.

## Architecture

```
Prompt (slim, ~8 tools):
  Core tools: say, think, examine, request_interaction, set_scene, display_mode
  + sense(entity_name)   ← universal sensor tool (replaces per-entity sense/read tools)
  + discover_tools(query) ← semantic tool discovery

Agent turn: "I want to attack the dragon with my sword"
  → discover_tools(query="attack dragon sword combat")
  → 1. Extract entity refs: ComponentIndex.find("dragon") + ToolRegistry name scan
  → 2. Match modulators by query keywords against modulator names + affordance descriptions
  → 3. Return ranked affordance tools with descriptions + NAc annotations
  → 4. Activate returned tools via register_scene_tools (I3 cap enforcement applies)
  → Output: "Found 5 combat tools: rusty_sword_slash(target, force) — slash at a target..."

Next turn: LLM sees discovered tools in prompt, calls them directly.
```

### Key Design Decisions

**1. One-step discovery, not two-step modulator hierarchy.**
Most entities have 2-3 modulators with 2-4 affordances each. A two-step flow adds an extra LLM turn for ~3 items. The LLM reasons about intent ("attack") not categories ("combat modulator"). The modulator hierarchy is used internally for ranking, not exposed as a navigation structure.

**2. Universal `sense(entity_name)` tool replaces per-entity sensor tools.**
5 sensors × 3 entities = 15 read-only tools of passive awareness. A single tool that takes an entity name mirrors how a real organism works — you don't have separate "check_health" and "check_stamina" perceptions, you sense your body. Returns all sensor readings for the named entity.

**3. Discovery activates tools via existing I3 scene-scoped mechanism.**
No new "hidden" registry state. Discovery returns tool descriptions in its output text, and activates the tools via `register_scene_tools()`. They appear in the next turn's prompt naturally. I3's cap enforcement handles overflow.

**4. NAc valence ranking on discovery results.**
Discovery results annotated with NAc valence — positive-valence tools float to top, negative-valence tools get caution notes. Reuses the Acting Coach's `_compose_nac_annotations` pattern.

**5. Imagination integration is perception-driven, not hint-driven.**
When ImaginationTrigger designs an entity, it registers the entity in ComponentIndex but does NOT register affordance tools eagerly. The Acting Coach's next turn naturally reflects the new entity via body_state ("you sense something unfamiliar"). The agent's curiosity drives `discover_tools` — no system-level "use discover_tools" hints.

### Prompt Token Budget Impact

| Scenario | Tools in prompt | ~Tokens |
|----------|----------------|---------|
| Current (no discovery) | 36 | 1,000-1,800 |
| With discovery (slim) | 8 core + discovered | 240-500 base |
| After one discovery | 8 core + 3-5 discovered | 400-750 |
| **Savings per turn** | | **~500-1,000** |

## Stages

### S0 — Fix imagination tool registration bug (DONE)

**Bug:** `imagination/trigger.py:581` called `generate_tools_for_entity(entity)` without the required `registry: ToolRegistry` parameter. `TypeError` caught by `except Exception`, logged as warning. Imagined entities never got their affordance tools registered. **Fixed:** pass `self._tool_registry`.

### S1 — Universal sense tool + discover_tools + slim prompt mode (~250 LOC)

**New files:**
- `tools/discovery.py` — `DiscoverToolsTool` and `UniversalSenseTool`

**Modified files:**
- `simulation/orchestrator.py` — slim prompt mode: after entity tools are generated, deregister all `read_*` and `sense_*` tools, register `UniversalSenseTool` and `DiscoverToolsTool` as core tools. Affordance tools stay registered but are scene-deactivated (not in active list until discovered).
- `embodiment/tool_bridge.py` — new `generate_tools_for_entity_deferred()` variant that generates tools, registers them via `register_scene_tools` with `active=False` (scene deactivated by default). Alternatively, generate normally then immediately deactivate the scene.

**DiscoverToolsTool implementation:**
```python
class DiscoverToolsTool(Tool):
    """Discover available physical capabilities by intent."""
    name = "discover_tools"
    description = "Discover what physical actions you can perform. Describe what you want to do."
    input_schema = {"query": str}

    def execute(self, query: str) -> ToolOutput:
        # 1. Entity resolution: ComponentIndex.find(query) + scan tool names
        # 2. Modulator matching: keyword overlap against modulator.name + affordance.description
        # 3. NAc ranking: annotate with valence from causal_links
        # 4. Activate matched tools via register_scene_tools
        # 5. Return formatted descriptions
```

**Discovery algorithm:**
1. Tokenize query into keywords
2. For each registered entity in ComponentRegistry:
   - Score entity relevance: ComponentIndex similarity to query
   - For each modulator on matched entities:
     - Score modulator relevance: keyword overlap with `modulator.name` + affordance descriptions
   - Collect matching affordances with scores
3. Sort by: entity relevance × modulator relevance × NAc valence boost
4. Top-k results (default k=8): activate via scene tools, return descriptions
5. If no entity match: fall back to ToolRegistry.find_similar for non-entity tools

**UniversalSenseTool implementation:**
```python
class UniversalSenseTool(Tool):
    """Read all sensors on a named entity."""
    name = "sense"
    description = "Sense the state of an entity. Returns all sensor readings."
    input_schema = {"entity_name": str}

    def execute(self, entity_name: str) -> ToolOutput:
        # Resolve entity from ComponentRegistry or tool_bridge's entity map
        # Call entity.read_all_sensors()
        # Return formatted readings
```

**Slim prompt mode activation:**
In `orchestrator.py`, after entity tools are generated for the AUT:
1. Collect all `sense_*` and `read_*_*` tool names from the registry
2. Deactivate their scenes (or deregister if core-registered)
3. Register `UniversalSenseTool` (core) and `DiscoverToolsTool` (core)
4. Deactivate all affordance tool scenes — they become discoverable but not visible

The AUT prompt now shows ~8 tools instead of ~36.

### S2 — NAc-informed ranking + Acting Coach integration (~100 LOC)

**Modified files:**
- `tools/discovery.py` — add NAc valence lookup to discovery ranking
- `prompts/acting_coach.py` — update embodiment guidance: instead of generic "explore your capabilities", reference discovery naturally ("you have a body with capabilities you can explore and discover")

**NAc integration in DiscoverToolsTool:**
The tool accepts an optional `nac` reference (wired at construction). On discovery:
- Look up causal links for each matched affordance tool name
- Positive valence (> 0.3 confidence): boost ranking score by 1.2×, annotate "this has worked well before"
- Negative valence (> 0.3 confidence): annotate with caution note, do NOT suppress — the agent can still choose to use it

**Acting Coach changes:**
- Remove the `_has_entity_tools` check that currently enables embodiment guidance based on tool name scanning
- Replace with a simpler check: embodiment guidance activates when `discover_tools` is in `available_tools`
- Guidance text: "You have a physical form with capabilities to discover. Sense your body. When you want to act physically, describe your intent to discover what you can do."
- NAc annotations (`_compose_nac_annotations`) continue working unchanged — they annotate whatever tools are currently visible

### S3 — Imagination deferred discovery (~50 LOC)

**Modified files:**
- `imagination/trigger.py` — after entity design, register in ComponentIndex (already done) but DON'T call `generate_tools_for_entity` eagerly. Instead, generate tools but register them scene-deactivated.
- `embodiment/body.py` or `prompts/acting_coach.py` — when a new entity appears (imagination), the body_state includes a subtle perception note ("you notice something new nearby"). This flows naturally through the existing Acting Coach pipeline without injecting tool names.

**Flow:**
1. ImaginationTrigger designs `crystal_dragon` → registers in ComponentIndex + ComponentRegistry
2. Tools generated and scene-registered as **inactive** (deactivated scene)
3. Body state on next turn includes new-entity awareness (via entity tree change detection)
4. Acting Coach's embodiment guidance + agent's curiosity → agent calls `discover_tools("crystal dragon")`
5. Discovery activates the dragon's affordance tools → they appear in the next turn

**What NOT to do:**
- No system-level "use discover_tools" hints in percepts
- No auto-discovery — the agent drives exploration through its own curiosity
- No eager tool registration — tools exist but are invisible until discovered

## Invariants

- **`discover_tools` is a core tool, not a scene tool.** It is always available. Deregistering it breaks the entire discovery flow.
- **Discovery activates tools via I3's `register_scene_tools` / `activate_scene`.** No new registry states. I3's cap enforcement applies to discovered tools.
- **Universal `sense` tool always reflects live entity state.** It reads from `entity.vital_metrics` / `entity.read_all_sensors()` — same data path as the per-sensor tools it replaces.
- **Discovery does not suppress tools.** Negative-valence tools get caution annotations, not removal. The agent always *can* use any discovered tool.
- **Imagination entities are discoverable but not visible until discovered.** Tools are scene-registered as inactive. ComponentIndex makes them findable.

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| S0 | 1 | Bug fix (done) |
| S1 | ~250 | DiscoverToolsTool + UniversalSenseTool + slim prompt wiring |
| S2 | ~100 | NAc ranking + Acting Coach update |
| S3 | ~50 | Imagination deferred registration |
| **Total** | **~400** | |

## Open Questions (Resolved)

- ~~Should discover_tools expose modulator hierarchy?~~ **No.** One-step with internal modulator matching.
- ~~Should sense tools stay always-visible?~~ **No.** Universal `sense(entity_name)` tool replaces per-entity sensor tools.
- ~~Should discovery add a "hidden" registry state?~~ **No.** Uses I3's existing scene deactivation.
- ~~Should imagination inject "use discover_tools" hints?~~ **No.** Acting Coach + body_state + curiosity drive discovery naturally.
- Should the orchestrator also get discover_tools? **No** — orch has a fixed, small tool set.
- Should discovery work outside sim mode? **Yes** — for Reachy, camera percepts → "what tools work on this object?" Same mechanism.
- Latency budget: 20-50ms per discovery call. Agent loop at 2Hz = 500ms/tick. 10% is acceptable, and discovery only fires when the LLM explicitly calls it.

## Interaction Map

```
ComponentIndex.find(query)  ←  semantic entity resolution
         ↓
Entity.modulators           ←  SEM hierarchy (modulator → affordance)
         ↓
NAc.get_causal_links()      ←  valence ranking
         ↓
ToolRegistry.activate_scene ←  I3 scene-scoped activation
         ↓
build_tools_section()       ←  discovered tools appear in next prompt turn
         ↓
Acting Coach                ←  embodiment guidance encourages discovery
```
