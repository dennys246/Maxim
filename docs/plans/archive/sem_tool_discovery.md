# SEM Tool Discovery Plan

**Status:** SHIPPED (2026-04-20)
**Scope:** 0.7 — Simulation Scalability
**Depends on:** I3 (scene-scoped tools), E2.5 (ComponentIndex), SEM protocol, B3.1 (Acting Coach)
**Companion plan:** [concept_exploration.md](concept_exploration.md) — handles vague goals / unknown concepts (Layer C)

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

**Hybrid approach** — goal-relevant affordances visible from turn 1, `discover_tools` expands the set.

```
Prompt (~12-15 tools, down from ~36):
  Core tools: say, think, examine, request_interaction, set_scene, display_mode
  + sense(entity_name)        ← universal sensor tool (replaces per-entity sense/read tools)
  + discover_tools(query)     ← semantic tool discovery for expansion
  + top-k affordance tools    ← auto-selected by goal keyword matching at prompt build time

Turn 1 (goal: "test sword combat"):
  Visible: sense, discover_tools, rusty_sword_slash, rusty_sword_parry,
           base_humanoid_move, + 6 core tools  (= ~12 tools)
  Hidden (discoverable): rusty_sword_throw, rusty_sword_sharpen,
           rusty_sword_repair, base_humanoid_rest, base_humanoid_pick_up, ...

Turn 3: agent wants to repair the sword after it dulls
  → discover_tools(query="repair sword maintenance")
  → Activates: rusty_sword_sharpen, rusty_sword_repair
  → These appear in turn 4's prompt

Turn 8: imagination creates crystal_dragon
  → Tools generated, scene-deactivated (invisible)
  → Agent perceives new entity, calls discover_tools("crystal dragon combat")
  → Activates: crystal_dragon_bite, crystal_dragon_claw
```

### Key Design Decisions

**1. Hybrid: top-k visible + discover_tools for expansion (revised after review).**
A pure slim prompt (zero affordance tools) causes a 2-4 turn cold start regression on local 7B-14B models — LLMs respond to concrete affordances, not abstract "discover your capabilities" prompts. The hybrid keeps 3-5 goal-relevant affordance tools visible from turn 1 (no inference gap) while `discover_tools` unlocks the rest on demand. Token savings are ~60-70% instead of ~80%, but turn-1 behavior is preserved.

**2. One-step discovery, not two-step modulator hierarchy.**
Most entities have 2-3 modulators with 2-4 affordances each. A two-step flow adds an extra LLM turn for ~3 items. The LLM reasons about intent ("attack") not categories ("combat modulator"). The modulator hierarchy is used internally for ranking, not exposed as a navigation structure.

**3. Universal `sense(entity_name)` tool replaces per-entity sensor tools.**
5 sensors × 3 entities = 15 read-only tools of passive awareness. A single tool that takes an entity name mirrors how a real organism works — you don't have separate "check_health" and "check_stamina" perceptions, you sense your body. Returns all sensor readings for the named entity. Requires an entity-name-to-live-Entity resolution map (see S1 details).

**4. Discovery activates tools via existing I3 scene-scoped mechanism.**
No new "hidden" registry state. Discovery returns tool descriptions in its output text, and activates the tools via `activate_scene()`. They appear in the next turn's prompt naturally. I3's cap enforcement handles overflow.

**5. NAc valence ranking on discovery results.**
Discovery results annotated with NAc valence — positive-valence tools float to top, negative-valence tools get caution notes. Reuses the Acting Coach's `_compose_nac_annotations` pattern.

**6. Imagination integration is perception-driven, not hint-driven.**
When ImaginationTrigger designs an entity, it registers the entity in ComponentIndex and generates tools as scene-deactivated. No system-level "use discover_tools" hints. The agent perceives the new entity naturally and curiosity drives discovery.

**7. LRU eviction prevents cumulative re-bloat.**
Discovered tools that haven't been *called* in N turns (default 5) are auto-deactivated. This prevents repeated `discover_tools` calls from monotonically growing the active tool list back to 30+. Eviction is a thin layer on top of I3's scene deactivation — timestamp on last use, check at prompt build time.

### Prompt Token Budget Impact

| Scenario | Tools in prompt | ~Tokens |
|----------|----------------|---------|
| Current (no discovery) | 36 | 1,000-1,800 |
| Hybrid (turn 1) | ~12 core + top-k | 400-600 |
| After one discovery | ~12 + 3-5 discovered | 550-850 |
| After LRU eviction | back to ~12 | 400-600 |
| **Savings per turn** | | **~400-1,000** |

## Stages

### S0 — Fix imagination tool registration bug (DONE)

**Bug:** `imagination/trigger.py:581` called `generate_tools_for_entity(entity)` without the required `registry: ToolRegistry` parameter. `TypeError` caught by `except Exception`, logged as warning. Imagined entities never got their affordance tools registered. **Fixed:** pass `self._tool_registry`.

### S1 — Universal sense tool + discover_tools + hybrid prompt mode (~300 LOC)

**New files:**
- `tools/discovery.py` — `DiscoverToolsTool` and `UniversalSenseTool`
- `embodiment/entity_map.py` — `EntityMap` (standalone, decoupled from ToolRegistry)

**Modified files:**
- `simulation/orchestrator.py` — hybrid prompt mode wiring
- `embodiment/tool_bridge.py` — entity map population + deferred scene registration

#### EntityMap — entity name resolution (finding #2 fix)

There is currently no entity-name-to-live-Entity registry. `tool_bridge.py` generates tools that hold `self._entity` references, but there's no global lookup. `UniversalSenseTool` and `DiscoverToolsTool` both need to resolve entity names to live `Entity` objects.

**EntityMap is a standalone object in `embodiment/entity_map.py`**, not on ToolRegistry. Rationale: putting it on ToolRegistry couples entity awareness to the tools layer. Future consumers (prompt builder for entity context, memory for "what entities exist?", Reachy's embodied runtime) would need a ToolRegistry reference just to look up entities — a layering violation. Standalone with its own RLock keeps entity resolution available to any layer.

```python
class EntityMap:
    """Maps entity names/paths to live Entity objects.

    Standalone object — not coupled to ToolRegistry. Passed to
    whatever needs entity resolution: tools, prompt builder, memory.
    Thread-safe via RLock.

    Populated by generate_tools_for_entity and ImaginationTrigger.
    """
    _lock: threading.RLock
    _entities: dict[str, Entity]  # name → Entity (or full_path → Entity on collision)

    def register(self, entity: Entity) -> None:
        """Register an entity tree. Walks descendants."""
        with self._lock:
            for ent in entity.walk():
                if ent.name in self._entities:
                    # Collision — store both under full_path
                    existing = self._entities.pop(ent.name)
                    self._entities[existing.full_path] = existing
                    self._entities[ent.full_path] = ent
                else:
                    self._entities[ent.name] = ent

    def resolve(self, name: str) -> Entity | None:
        """Resolve by name, then full_path. Returns None if not found."""

    def list_names(self) -> list[str]:
        """Return all known entity names (for error messages)."""
```

`generate_tools_for_entity` populates the `EntityMap` as a side effect (passed as optional parameter, no signature break for existing callers). The orchestrator creates the `EntityMap` and passes it to both `generate_tools_for_entity` (population) and the tools (reads). `ImaginationTrigger` also receives it at construction for populating on entity design.

#### DiscoverToolsTool

```python
class DiscoverToolsTool(Tool):
    """Discover available physical capabilities by intent."""
    name = "discover_tools"
    description = "Discover what physical actions you can perform. Describe what you want to do — e.g., 'attack with sword', 'repair equipment', 'move quietly'."
    input_schema = {"query": str}

    def __init__(self, *, entity_map, component_index, tool_registry, nac=None): ...

    def execute(self, query: str) -> ToolOutput:
        # 1. Entity resolution: entity_map scan + ComponentIndex.find(query)
        # 2. For matched entities: score modulators by keyword overlap
        # 3. Collect matching affordances, rank by relevance × NAc valence
        # 4. Filter: skip entities with zero registered tools (imagination race guard)
        # 5. Activate matched tools via activate_scene
        # 6. Return formatted descriptions (or fallback message on zero results)
```

**Discovery algorithm:**
1. Tokenize query into keywords
2. For each entity in EntityMap:
   - Score entity relevance: name/type keyword match + ComponentIndex similarity
   - For each modulator on matched entities:
     - Score modulator relevance: keyword overlap with `modulator.name` + affordance descriptions
   - Collect matching affordances with scores
3. Sort by: entity relevance × modulator relevance × NAc valence boost (S2)
4. Top-k results (default k=8): activate via `activate_scene`, return descriptions
5. If no entity match: fall back to `ToolRegistry.find_similar` for non-entity tools
6. **If zero results:** return "No matching capabilities found. Try describing a specific physical action like 'attack', 'move', or 'repair'." (finding #5 fix)

**Imagination race guard (finding #6 fix):** If entity is in ComponentIndex but has zero tools in the ToolRegistry (imagination still designing), skip with note: "An entity matching your query is still forming — try again next turn."

#### UniversalSenseTool

```python
class UniversalSenseTool(Tool):
    """Read all sensors on a named entity."""
    name = "sense"
    description = "Sense the state of an entity — health, durability, position, etc. Name the entity you want to sense."
    input_schema = {"entity_name": str}

    def __init__(self, *, entity_map): ...

    def execute(self, entity_name: str) -> ToolOutput:
        entity = self._entity_map.resolve(entity_name)
        if entity is None:
            return ToolOutput(success=False, error=f"Unknown entity: {entity_name}. Known entities: {self._entity_map.list_names()}")
        readings = entity.read_all_sensors()
        return ToolOutput(success=True, output={name: {"value": r.value, "unit": r.unit} for name, r in readings.items()})
```

#### Hybrid prompt mode wiring

In `orchestrator.py`, after entity tools are generated for the AUT:

1. Build `EntityMap` from all registered entities
2. Deregister all `read_*_*` and `sense_*` tools (per-entity sensor tools)
3. Register `UniversalSenseTool(entity_map=entity_map)` as core tool
4. Register `DiscoverToolsTool(entity_map=entity_map, component_index=..., tool_registry=...)` as core tool
5. **Goal-based top-k selection:** tokenize the sim goal, score each affordance tool against goal keywords, keep top 3-5 active, scene-deactivate the rest
6. Entity context section (`build_entity_context_section`) continues listing all affordance names — this is now consistent because the top-k are callable, and the rest are explicitly discoverable

**Top-k goal matching algorithm:**
```python
def _select_goal_relevant_tools(goal: str, entity_map: EntityMap, registry: ToolRegistry) -> list[str]:
    """Score affordance tools against sim goal keywords, return top-k to keep active."""
    goal_keywords = set(goal.lower().split())
    scores: list[tuple[str, float]] = []
    for tool_name in registry.list():
        tool = registry.get(tool_name)
        if not isinstance(tool, ModulatorAffordanceTool):
            continue
        # Score: keyword overlap between goal and (modulator name + affordance description)
        tool_words = set(tool.description.lower().split()) | {tool._modulator.name.lower()}
        overlap = len(goal_keywords & tool_words)
        if overlap > 0:
            scores.append((tool_name, overlap))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_k = [name for name, _ in scores[:5]]

    # Vague goal fallback: if top-k returned < 3 tools, add one affordance
    # per entity (first affordance of the modulator with the most affordances).
    # Ensures physical tools are visible even for "explore freely" or "survive".
    if len(top_k) < 3:
        for entity in entity_map.list_entities():
            best_mod = max(entity.modulators.values(), key=lambda m: len(m.affordances), default=None)
            if best_mod is not None:
                first_aff = next(iter(best_mod.affordances))
                candidate = f"{entity.name}_{first_aff}"
                if candidate not in top_k and registry.is_tool_active(candidate):
                    top_k.append(candidate)
    return top_k
```

Tools not in the top-k are scene-deactivated (still registered, discoverable).

**Vague-query graceful degradation in discover_tools:**

When discover_tools receives a vague query ("what can I do", "explore") that matches no specific modulator/affordance, instead of returning zero results it returns a **modulator category summary** across all entities:

```
"Your capabilities by category:
- Combat (rusty_sword): slash, parry, throw
- Locomotion (base_humanoid): move, rest
- Manipulation (base_humanoid): pick_up, drop, use
- Maintenance (rusty_sword): sharpen, repair
Try a more specific query like 'attack with sword' to activate those tools."
```

This uses the same entity/modulator walk as the main discovery algorithm — the vague-query path just returns the top level instead of activating specific affordances. No additional machinery needed. The summary gives the agent enough orientation to compose a targeted follow-up query.

**Concept exploration beyond this scope:** Deeply vague goals ("explore freely", "understand this world") need more than keyword matching — they need concept grounding that connects abstract intent to concrete directions. This is tracked in the companion plan [concept_exploration.md](concept_exploration.md) and will be implemented after SEM tool discovery provides the baseline to measure against.

### S2 — NAc-informed ranking + Acting Coach integration + LRU eviction (~120 LOC)

**Modified files:**
- `tools/discovery.py` — NAc valence lookup in discovery ranking + LRU eviction
- `prompts/acting_coach.py` — updated embodiment guidance
- `tools/registry.py` — `last_used` timestamp on tool call (thin addition)

**NAc integration in DiscoverToolsTool:**
The tool accepts an optional `nac` reference (wired at construction). On discovery:
- Look up causal links for each matched affordance tool name
- Positive valence (> 0.3 confidence): boost ranking score by 1.2×, annotate "this has worked well before"
- Negative valence (> 0.3 confidence): annotate with caution note, do NOT suppress — the agent can still choose to use it

**Acting Coach changes:**
- Remove the `_has_entity_tools` check that currently enables embodiment guidance based on tool name scanning
- Replace with a simpler check: embodiment guidance activates when `discover_tools` is in `available_tools`
- Guidance text: "You have a physical form. Your most relevant capabilities are already available as tools. To find more — describe what you want to do to discover_tools."
- NAc annotations (`_compose_nac_annotations`) continue working unchanged — they annotate whatever tools are currently visible

**LRU eviction (finding #4 fix):**

Without eviction, repeated `discover_tools` calls across turns monotonically grow the active tool list back toward 30+. Solution: discovered affordance tools that haven't been *called* in N turns are auto-deactivated.

**Per-tool deactivation (new ToolRegistry method):**
I3's `deactivate_scene` is per-scene — it flips all tools in a scene. LRU needs per-tool granularity: if the agent discovers 3 combat tools and only uses `slash`, `parry` and `throw` should be evicted while `slash` stays. Add `deactivate_tool(name)` to ToolRegistry (~5 lines: flip `_scene_meta[name].active = False`). This is a small, clean extension of I3 — scenes remain the unit of *bulk* activation, individual deactivation is the new capability.

```python
# In ToolRegistry:
def deactivate_tool(self, name: str) -> bool:
    """Deactivate a single tool (set active=False). Returns True if found."""
    with self._lock:
        meta = self._scene_meta.get(name)
        if meta is not None and meta.active:
            meta.active = False
            return True
        return False

# In tools/discovery.py — eviction logic:
_tool_last_used: dict[str, int] = {}  # tool_name → turn number of last call
_goal_selected: set[str] = set()  # exempt from LRU eviction
DISCOVERY_LRU_TURNS = 5

def evict_stale_discoveries(current_turn: int, registry: ToolRegistry) -> list[str]:
    """Deactivate individual discovered tools not called in DISCOVERY_LRU_TURNS."""
    evicted = []
    for tool_name, last_turn in list(_tool_last_used.items()):
        if tool_name in _goal_selected:
            continue  # top-k goal-selected tools are exempt
        if current_turn - last_turn > DISCOVERY_LRU_TURNS:
            if registry.deactivate_tool(tool_name):
                evicted.append(tool_name)
                del _tool_last_used[tool_name]
    return evicted
```

The `last_used` timestamp is set in `executor.py` when a tool is actually called (one line addition). Eviction runs at prompt build time (in `agent_loop.py` or `loop_controller.py`), before the tool list is assembled. Top-k goal-selected tools are exempt from LRU eviction — they stay visible regardless.

### S3 — Imagination deferred discovery (~50 LOC)

**Modified files:**
- `imagination/trigger.py` — generate tools scene-deactivated instead of active
- `embodiment/body.py` or Acting Coach — new-entity perception note

**Flow:**
1. ImaginationTrigger designs `crystal_dragon` → registers in ComponentIndex + ComponentRegistry
2. `generate_tools_for_entity(entity, registry)` creates tools, then immediately scene-deactivate the scene
3. Body state on next turn includes new-entity awareness (entity tree change detection or explicit flag from trigger)
4. Acting Coach's embodiment guidance + agent's curiosity → agent calls `discover_tools("crystal dragon")`
5. Discovery activates the dragon's affordance tools → they appear in the next turn

**What NOT to do:**
- No system-level "use discover_tools" hints in percepts — breaks bio-plausibility
- No auto-discovery — the agent drives exploration through its own curiosity
- No eager tool activation — tools exist but are invisible until discovered

## Findings from dual-lens review (2026-04-20)

Two parallel review agents (edge-case lens + LLM behavioral lens) identified 9 findings. Two were cross-confirmed as critical:

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | **Cold start regression** — pure slim prompt = 2-4 turns to first action on local models | Critical (cross-confirmed) | Hybrid approach: top-k goal-relevant tools visible from turn 1 |
| 2 | **No entity-name-to-Entity resolution** — UniversalSenseTool has no lookup path | Critical | EntityMap with full_path disambiguation |
| 3 | **entity_context / tool-list mismatch** — prompt describes affordances not in tool list | Important | Hybrid resolves: top-k are callable, rest explicitly discoverable |
| 4 | **Cumulative discovery re-bloats** — repeated discover_tools grows tool list monotonically | Important (cross-confirmed) | LRU eviction: deactivate after 5 turns of non-use |
| 5 | **Bad query = silent zero results** — no guidance for re-query | Important | Fallback message with example queries |
| 6 | **Imagination/discovery race** — ComponentIndex has entity but tools don't exist yet | Important | Guard: skip entities with zero registered tools |
| 7 | **ComponentIndex stale ephemeral entries** — cleared entities still in index | Minor | Within-session: non-issue. Cross-session: index rebuilt from registry |
| 8 | **Discovery spam** — LLM calls discover_tools every turn | Non-issue (cross-confirmed) | Self-limiting: activated tools persist, diminishing returns |
| 9 | **Deactivated-but-callable** — executor scene gate already handles | Non-issue | Existing I3 gate returns error + suggestions |

## Invariants

- **`discover_tools` and `sense` are core tools, not scene tools.** Always available. Deregistering either breaks the discovery/sensing flow.
- **Discovery activates tools via I3's `activate_scene`.** I3's cap enforcement applies to discovered tools. LRU eviction uses the new per-tool `deactivate_tool` method (S2).
- **Universal `sense` always reflects live entity state.** Reads from `entity.vital_metrics` / `entity.read_all_sensors()` — same data path as the per-sensor tools it replaces.
- **Discovery does not suppress tools.** Negative-valence tools get caution annotations, not removal. The agent always *can* use any discovered tool.
- **Top-k goal-selected tools are exempt from LRU eviction.** They stay visible for the session regardless of use frequency.
- **EntityMap is standalone in `embodiment/entity_map.py`**, not on ToolRegistry. Single source of truth for name → live Entity. Decoupled so future consumers (prompt builder, memory, Reachy runtime) can resolve entities without a ToolRegistry reference. Thread-safe via RLock. Collision disambiguation uses `entity.full_path`.
- **Imagination entities are discoverable but not visible until discovered.** Tools are scene-registered as inactive. ComponentIndex + EntityMap make them findable.
- **Zero-result discovery returns a modulator category summary**, not empty output. Gives the agent enough orientation to compose a targeted follow-up query.
- **Per-tool deactivation is the LRU mechanism.** `ToolRegistry.deactivate_tool(name)` flips a single tool's `active` flag. Scenes remain the unit of *bulk* activation; individual deactivation is the new per-tool capability.

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| S0 | 1 | Bug fix (done) |
| S1 | ~320 | DiscoverToolsTool + UniversalSenseTool + EntityMap + hybrid wiring + vague-goal fallback |
| S2 | ~130 | NAc ranking + Acting Coach update + LRU per-tool eviction + `deactivate_tool` |
| S3 | ~50 | Imagination deferred registration |
| **Total** | **~500** | |

## Open Questions (Resolved)

- ~~Should discover_tools expose modulator hierarchy?~~ **No.** One-step with internal modulator matching.
- ~~Should sense tools stay always-visible?~~ **No.** Universal `sense(entity_name)` tool replaces per-entity sensor tools.
- ~~Should discovery add a "hidden" registry state?~~ **No.** Uses I3's existing scene deactivation.
- ~~Should imagination inject "use discover_tools" hints?~~ **No.** Acting Coach + body_state + curiosity drive discovery naturally.
- ~~Pure slim prompt or hybrid?~~ **Hybrid.** Top-k goal-relevant affordances visible from turn 1, discover_tools for expansion. Cross-confirmed cold start regression on local models killed the pure-slim approach.
- ~~Where does EntityMap live?~~ **Standalone in `embodiment/entity_map.py`**, not on ToolRegistry. Avoids coupling entity awareness to the tools layer. Future consumers (prompt builder, memory, Reachy) can use it without a ToolRegistry reference.
- ~~LRU eviction granularity?~~ **Per-tool.** New `ToolRegistry.deactivate_tool(name)` method (~5 lines). Scenes remain the unit of bulk activation; individual deactivation is the new capability for LRU.
- ~~What about vague goals?~~ **Three-layer answer.** Layer A: top-k fallback (one affordance per entity if < 3 goal matches). Layer B: discover_tools returns modulator category summary on vague queries. Layer C (future): concept exploration plan for deep conceptual grounding — see [concept_exploration.md](concept_exploration.md).
- Should the orchestrator also get discover_tools? **No** — orch has a fixed, small tool set.
- Should discovery work outside sim mode? **Yes** — for Reachy, camera percepts → "what tools work on this object?" Same mechanism.
- Latency budget: 20-50ms per discovery call. Agent loop at 2Hz = 500ms/tick. 10% is acceptable, and discovery only fires when the LLM explicitly calls it.

## Interaction Map

```
Goal keywords              ←  top-k selection at prompt build time
         ↓
Prompt: core + top-k + discover_tools   (turn 1: ~12 tools, not ~36)
         ↓
discover_tools(query)      ←  agent calls when it wants more capabilities
         ↓
EntityMap.resolve()        ←  name → live Entity object
ComponentIndex.find()      ←  semantic entity resolution (fuzzy)
         ↓
Entity.modulators          ←  SEM hierarchy (modulator → affordance)
         ↓
NAc.get_causal_links()     ←  valence ranking (S2)
         ↓
ToolRegistry.activate_scene ← I3 scene-scoped activation
         ↓
build_tools_section()      ←  discovered tools appear in next prompt turn
         ↓
LRU eviction (S2)          ←  deactivate unused discoveries after 5 turns
         ↓
Acting Coach               ←  embodiment guidance references discovery naturally
```
