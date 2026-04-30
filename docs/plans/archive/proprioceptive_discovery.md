# Proprioceptive Discovery — Emergent Affordances + Entity Acquisition

**Status:** MECHANISM A SHIPPED (feat/percept-reflex-system), MECHANISM B SHIPPED (feat/cradle-sensorimotor Stage 2)
**Depends on:** Component-level damage (shipped), Percept reflex system (shipped)

## Principle

The agent's body dynamically gains and loses capabilities as a **natural consequence** of the existing bio-pipeline processing percepts. Nothing is explicitly triggered. The dragon attacks → the percept embedding matches "dodge" in the same index that matches "dragon" → the agent sees "dodge" as available → the agent dodges. The learning loop closes: NAc records whether dodging worked, cerebellum builds a forward model, next time the agent knows.

## Two Mechanisms

### A. Latent Motor Programs (intra-turn, same tick)

**What:** A humanoid body can dodge, roll, block, parry — these are contextual compositions of existing body-part capabilities. They're not separate tools; they resolve to `use(object="self", action="dodge")` which the agent already has.

**How it emerges:**

1. Body spec declares `latent_affordances` per modulator:
   ```yaml
   legs:
     latent_affordances:
       dodge:
         description: "Dodge an incoming attack by moving sideways"
         context: [attack, threat, strikes, lunges, charges]
         requires: {integrity: 0.2}
       roll:
         description: "Roll away from an area of danger"
         context: [attack, explosion, fire, collapsing]
         requires: {integrity: 0.3}
   arms:
     latent_affordances:
       block:
         description: "Raise arms to block an incoming blow"
         context: [attack, strikes, swings]
         requires: {integrity: 0.15}
       parry:
         description: "Deflect an attack with a held weapon"
         context: [attack, sword, blade, strikes]
         requires: {integrity: 0.3}
   ```

2. At body registration (orchestrator startup), latent affordances are indexed in ComponentIndex with their context keywords as the semantic signature. This happens once, zero ongoing cost.

3. `BioEnrichmentPipeline._query_component_index()` already runs `find_similar(text)` on every percept. When "the dragon lunges at you with its claws" arrives, the embedding matches "dodge" (context: attack, lunges) above the similarity threshold. The affordance name flows into `EnrichmentResult.affordances`.

4. `format_thought_response()` already injects affordances into the prompt: "Available actions: dodge, roll". The LLM sees this, recognizes it can call `use(object="self", action="dodge")`.

5. The embodiment executor resolves `use(self, dodge)` against the body's latent affordance spec. The `requires` check runs against the parent modulator's integrity — can't dodge with broken legs.

6. After execution, existing pipelines close the loop:
   - NAc records dodge → outcome (avoided damage / still hit)
   - Cerebellum trains forward model for dodge
   - Hippocampus stores episodic memory
   - On future encounters, NAc's `reward_bias` annotates "dodge" as effective or dangerous via `_annotate_affordance_valence()`

**No new tools registered. No activation/deactivation. No triggers. The embedding similarity that surfaces "this scene has a dragon" also surfaces "you could dodge." Same pipeline, same tick.**

### B. Entity Acquisition (inter-turn, via pick_up/drop)

**What:** Agent finds a shield → picks it up → shield becomes part of agent's body → shield affordances (block, bash) become agent's tools. Agent drops it → tools removed.

**How it emerges:**

1. Entity specs declare acquirability:
   ```yaml
   # weapons/iron_shield.yaml
   component:
     acquirable: true    # can be picked up
     on_acquire: equip   # equip = tools registered, consume = apply effect + destroy
   ```

2. Agent calls `pick_up(object="shield")` (existing arms affordance on base_humanoid).

3. The `pick_up` modulator execution returns `ToolOutput.side_effects={"entity_acquired": "iron_shield"}`.

4. The executor sees `entity_acquired` in side_effects (same pattern as `embodiment_failures`):
   - Resolves entity in EntityMap
   - Calls `entity.reparent(self_entity)` — shield becomes a child of the agent's body
   - Calls `generate_tools_for_entity(shield, registry, scene_id=shield.name)` — shield affordances (block, bash) registered as scene-scoped tools
   - Calls `entity_map.register_self(shield)` — ownership flips from scene to self

5. `drop(object="shield")` produces `side_effects={"entity_released": "iron_shield"}`:
   - Deregisters shield tools
   - Reparents back to scene root
   - Flips ownership back to scene

6. The shield's sensors contribute to damage model while equipped — `shield.integrity` absorbs hits (the agent can interpose the shield, reducing body damage).

**This uses existing `Entity.reparent()`, existing `generate_tools_for_entity()`, existing `ToolOutput.side_effects` processing. The only new code is the executor recognizing `entity_acquired`/`entity_released` keys.**

## Why Not First-Class Tools?

Review 2 proposed registering latent affordances as inactive tools and activating them on demand. We chose the prompt-hint approach (Review 1) instead because:

1. **Zero tool management.** No registration, activation, deactivation, LRU interaction. The `use` tool already exists.
2. **Naturally bounded.** Bio-enrichment returns top-k affordances. If nothing is attacking, "dodge" never surfaces. No cleanup needed.
3. **Learning-compatible.** NAc already annotates surfaced affordances with valence. An inactive tool wouldn't get this annotation.
4. **LLM-native.** The prompt says "Available actions: dodge" — the LLM understands this. Latent tools would add to the tool list, competing with the 20-tool cap.

First-class tools are the right approach for entity acquisition (shield.block needs its own tool because it has unique parameters), but not for motor compositions of the body's existing capabilities.

## Implementation

### Files (Mechanism A — latent motor programs)

| File | Change | LOC |
|------|--------|-----|
| `_data/components/bodies/base_humanoid.yaml` | Add `latent_affordances` to legs, arms, torso | ~30 |
| `embodiment/spec.py` | Parse `latent_affordances` from modulator spec | ~15 |
| `embodiment/component_index.py` | Index latent affordances at body registration | ~20 |
| `integration/bio_enrichment.py` | Distinguish self-latent from external affordances | ~15 |
| `simulation/orchestrator.py` | Register latent affordances in ComponentIndex at AUT setup | ~10 |
| **Total** | | **~90** |

### Files (Mechanism B — entity acquisition)

| File | Change | LOC |
|------|--------|-----|
| `runtime/executor.py` | Handle `entity_acquired` / `entity_released` side_effects | ~40 |
| `embodiment/spec.py` | Parse `acquirable`, `on_acquire` from component spec | ~10 |
| `embodiment/tool_bridge.py` | `pick_up` executor returns `entity_acquired` side_effect | ~20 |
| `embodiment/entity_map.py` | `transfer_ownership(entity, from_scene=True)` | ~15 |
| **Total** | | **~85** |

### Implementation order

```
Mechanism A (latent motor programs)     ← emergent, no new infrastructure
  └─ Mechanism B (entity acquisition)   ← uses existing reparent + side_effects
```

Both are independently shippable. Mechanism A is higher impact (every sim benefits).

## Interaction with Component-Level Damage

- Latent affordances declare `requires: {integrity: 0.2}` — same gating as regular affordances
- "Can't dodge with broken legs" naturally falls out of the existing integrity check
- Acquired entities contribute sensors to the damage model while equipped
- Dropping a shield while it's being damaged produces a Reaction for NAc to learn from

## Gap Analysis: Embedding Similarity for Verb→Response

Empirical testing (2026-04-25) revealed that **embedding similarity is the wrong mechanism for surfacing defensive responses from attack percepts.** Cosine similarity between attack descriptions and defense affordances:

| Percept | → dodge | → block | → roll |
|---------|---------|---------|--------|
| "dragon breathes fire at you" | 0.355 | 0.339 | 0.300 |
| "guard swings sword at head" | 0.464 | 0.367 | 0.138 |
| "wolf leaps at throat" | 0.298 | 0.170 | 0.197 |

All below the 0.50 threshold, let alone 0.65. The model knows these are combat-related but doesn't know dodge is the *response* to an attack. That's causal knowledge, not semantic similarity.

**Resolution (implemented):** Latent affordances piggyback on reflex firings (option 1). When ANY reflex fires, ALL body modulators' latent affordances are integrity-gated and surfaced in the prompt. No embedding similarity, no independent keyword matching. The reflex system already detects threats via keyword patterns — we reuse that signal.

**Future pivot (option 2):** If scenarios emerge where affordances should surface WITHOUT a reflex firing (e.g., seeing an enemy prepare to attack but no damage yet), add independent keyword matching in `_evaluate_reflexes` as a second trigger path. The Cradle plan will provide initial experience via structured sensorimotor development.

**Pre-existing bug fixed (2026-04-25):** `record_outcome` in `tool_dispatch.py` recorded `event_signature="tool:use"` for ALL `use()` calls — dodge, open door, attack all looked identical to NAc. Fixed: `build_tool_signature(tool_name, tool_params)` now produces `"tool:use:dodge"` for `use(action="dodge")`. This is the single source of truth for tool→NAc event signature format. Also fixed 3 pre-existing bare `tool_name` query bugs in planning_bridge.py, discovery.py, and exec_agent.py that never matched because they lacked the `"tool:"` prefix.

## Learning Loop (end-to-end)

```
Dragon attacks → percept → damage_component(arms) → pain signal →
NAc records: fire_breath → damage (negative) →
Next turn: bio-enrichment surfaces "fire_breath → damage (negative)" prediction →
LLM sees warning + use(action=dodge) in tool list → tries dodge →
Dodge succeeds → NAc records: tool:use:dodge → success (positive) →
Next encounter: NAc has both fire_breath=dangerous AND dodge=effective →
Agent dodges from turn 1 of session 2
```

The critical path: NAc compound signatures (`tool:use:dodge` not `tool:use`) ensure the agent learns "dodging specifically works" not "the generic use tool works."
