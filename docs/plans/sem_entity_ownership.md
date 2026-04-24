# SEM Entity Ownership — self vs scene entity tool separation

**Status:** Shell plan (2026-04-24)
**Scope:** ~200-300 LOC across embodiment/, tools/, imagination/
**Depends on:** [deliberative_thought_stream.md](deliberative_thought_stream.md) (shipped S1+2, exposed this gap)
**Gates:** None; behavioral correctness improvement

---

## Problem

When the imagination trigger instantiates a scene entity (e.g., `creatures/dragon` from seed data), ALL of its affordance tools (fire_breath, tail_sweep, circle) become available to the agent as callable tools. The agent controls the dragon instead of fighting it.

**Root cause:** The SEM tool registration system has zero ownership semantics. `generate_tools_for_entity` produces tools for any entity, `ToolRegistry` treats all tools equally (gated only by active/deactivated for prompt size), and `EntityMap` does not distinguish the agent's own body from scene entities.

**Original design intent (pre-refactor):** Imagined entities should have their own SEM specs that describe what THEY can do. The agent learns about other entities' capabilities through percepts ("you observe a dragon breathing fire") and discovers tools available to ITSELF (its own body's affordances). The agent fights the dragon using humanoid tools (move, dodge, attack with carried weapon), not by calling the dragon's own abilities.

**Observed in sim (2026-04-24):** "A dragon is attacking a village" → agent calls `dragon_fire_breath` 5x, `dragon_tail_sweep` 1x, `dragon_circle` 1x. The agent IS the dragon instead of fighting it.

## Bio-plausible framing

An agent perceives other entities through observation (vision, proprioception) and learns their capabilities from experience — watching a dragon breathe fire teaches the agent what dragons can do. But the agent's motor system only controls its own body. You don't gain fire breath by observing a dragon; you learn to dodge it.

## Design

### Stage 1: EntityMap ownership — self vs scene

Add ownership tracking to EntityMap:

```python
class EntityMap:
    _self_entity: Entity | None = None  # The agent's body
    _scene_entities: dict[str, Entity] = {}  # Other entities in the scene

    def register_self(self, entity: Entity) -> None: ...
    def register_scene(self, entity: Entity) -> None: ...
    def is_self(self, entity_name: str) -> bool: ...
```

**Orchestrator change:** `_aut_entity_map.register_self(_aut_instance.embodiment.root)` instead of `register()`.

**Backward compat:** `register()` defaults to `register_scene()` for non-self entities. `list_entities()` returns all (self + scene). New `list_scene_entities()` returns scene-only.

### Stage 2: Tool generation mode — callable vs observed

`generate_tools_for_entity` gets a `mode` parameter:

- `mode="callable"` (default, current behavior): Generate full affordance tools the agent can call. Used for the agent's own body.
- `mode="observed"`: Generate a percept context description of what the entity can do, but NOT callable tools. The output is a structured string for StructuredContext, not Tool objects.

```python
def generate_tools_for_entity(entity, registry, *, mode="callable", ...):
    if mode == "observed":
        return generate_entity_description(entity)  # Returns str, not list[Tool]
    # ... existing tool generation
```

### Stage 3: Imagination trigger — scene entities are observed, not controlled

`_ensure_entity_live` registers imagined/seed entities as scene entities:

```python
_entity_map.register_scene(entity)  # Not register_self
description = generate_tools_for_entity(entity, mode="observed")
# Inject description into StructuredContext as percept context
```

The agent sees: "A dragon is nearby. It appears capable of: fire_breath (ranged fire attack), tail_sweep (area knockback), circle (aerial repositioning). Its health is 1.0, fire_charge is 0.8."

The agent's available tools remain: move, look, speak, pick_up, use, discover_tools, sense, sense_presence.

### Stage 4: Interaction tools — agent actions AGAINST scene entities

The agent needs tools for interacting WITH scene entities, not AS them. Options:

**Option A — Generic interaction verbs:** The base_humanoid gets generic tools like `attack(target)`, `defend`, `dodge`, `interact(entity, action)` that work against any scene entity. The SEM system resolves the outcome based on both entities' states.

**Option B — Context-generated tools:** When a scene entity is observed, generate interaction tools specific to the pairing: `dodge_dragon_fire`, `attack_dragon_weak_point`. These are generated from the humanoid's capabilities + the dragon's observed vulnerabilities.

**Option C — Existing `use` tool:** The humanoid's `use` affordance already exists. Extend it to accept a target entity: `use(target="dragon", action="attack")`. The SEM system resolves based on what the humanoid is carrying/capable of.

**Recommendation:** Start with Option A (generic interaction verbs) — it's the simplest and doesn't require per-pairing tool generation. Add `attack(target)`, `defend`, `dodge` to the base_humanoid SEM spec.

### Stage 5: discover_tools integration

`discover_tools` should:
- Search the agent's OWN tools (self entity) for callable actions
- Include scene entity observation context in the response ("The dragon nearby can breathe fire — consider dodging")
- NOT return scene entity tools as callable

### Stage 6: sense_presence shows observed capabilities

`sense_presence` already lists entities. Update to show:
- Self entity: "You (base_humanoid) — your capabilities: move, look, attack, defend, dodge"
- Scene entities: "Dragon (creature) — observed capabilities: fire_breath, tail_sweep, circle. State: health=1.0"

## What changes

| Stage | File | Change | LOC |
|-------|------|--------|-----|
| 1 | `embodiment/entity_map.py` | `register_self`, `register_scene`, `is_self` | +20 |
| 1 | `simulation/orchestrator.py` | Use `register_self` for agent body | +2, -1 |
| 2 | `embodiment/tool_bridge.py` | `mode="callable"/"observed"` parameter | +30 |
| 3 | `imagination/trigger.py` | `_ensure_entity_live` uses `register_scene` + `mode="observed"` | +15, -5 |
| 4 | `_data/components/bodies/base_humanoid.yaml` | Add attack, defend, dodge affordances | +20 |
| 5 | `tools/discovery.py` | Filter to self-entity tools, include scene context | +15 |
| 6 | `tools/discovery.py` | sense_presence distinguishes self vs scene | +10 |
| **Net** | | | **~+110** |

## Key constraints

1. **Backward compat:** `EntityMap.register()` keeps current behavior. Only new callers use `register_self/register_scene`.
2. **Scene entity state is still live:** Scene entities have real sensors that tick (dragon health depletes as it uses fire_breath). The agent can sense them via `sense(entity_name)`.
3. **No new bus or event type.** Scene entity observations flow through StructuredContext as percept context strings.
4. **Base humanoid combat affordances are generic.** `attack(target)` resolves differently based on what the humanoid is carrying (fists, sword, staff). The weapon entity modifies the attack outcome.

## Validation

1. Run dragon sim — confirm agent uses humanoid tools (move, attack, dodge) not dragon tools (fire_breath, tail_sweep)
2. Confirm agent can observe dragon state via `sense("dragon")`
3. Confirm `discover_tools("fight")` returns humanoid combat tools, not dragon tools
4. Confirm `sense_presence` shows dragon capabilities as observed, humanoid capabilities as callable
