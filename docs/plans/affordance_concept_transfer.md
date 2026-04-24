# Affordance Concept Transfer — two-level abstraction for cross-entity learning

**Status:** Shell plan (2026-04-24)
**Scope:** TBD — touches ATL, NAc, SEM observation, discovery
**Depends on:** [sem_entity_ownership.md](sem_entity_ownership.md) (shipped — self vs scene entity separation)
**Gates:** None; behavioral intelligence improvement

---

## Problem

When the agent encounters `dragon_fire_breath` and later encounters `mage_fire_breath`, it has zero knowledge transfer. Each entity-specific affordance is a fresh learning target. The agent must re-learn "fire breath is dangerous" from scratch for every entity that has it.

This is biologically implausible. Humans learn abstract concepts ("fire is dangerous") from specific experiences and transfer them to novel situations. The cerebellum retains motor-specific precision (swinging sword A vs sword B), but cortical concepts and reward associations generalize.

## Bio-plausible framing

The brain operates on at least two levels for action understanding:

- **Motor layer (cerebellum):** Specific to the exact entity and context. "How does this particular sword swing?" Forward models, prediction error, timing. Does NOT transfer across entities.
- **Concept layer (cortex/ATL):** Abstract categories. "Swords are sharp," "fire burns," "ranged attacks need dodging." DOES transfer — seeing one dragon breathe fire teaches you about all fire breath.

NAc (reward/punishment) bridges both: specific associations (`dragon_fire_breath → pain`) and abstract associations (`fire_breath → pain`) coexist. The abstract link fires at lower confidence but provides immediate priors for novel encounters.

## Current state (post entity-ownership)

- **ATL** forms concepts from the agent's own action names (e.g., `base_humanoid_move → action`). Scene entity affordances are NOT fed to ATL.
- **NAc** creates causal links keyed on tool name (e.g., `tool:base_humanoid_move → positive`). Entity-specific only; no abstract level.
- **Cerebellum** forward models key on `(entity_path, modulator, affordance, param_bucket)`. Correctly entity-specific — no change needed.
- **`describe_entity_capabilities()`** outputs bare affordance names without entity prefix (from entity ownership Stage 2).
- **`sense_presence`** shows scene entity affordances by name in its output.
- **`sense(entity_name)`** reads sensor state. No affordance-level information.

## Design sketch

### Layer 1: Affordance category extraction on observation

When a scene entity is instantiated (via imagination trigger or sense_presence), extract bare affordance names and feed them to ATL as observed concepts:

```
dragon registered as scene entity
  → affordances: fire_breath, tail_sweep, circle
  → ATL.observe_concepts(["fire_breath", "tail_sweep", "circle"],
                          context={"source": "scene_entity", "entity": "dragon", "entity_type": "creature"})
```

ATL forms/reinforces concept nodes: `fire_breath`, `tail_sweep`, `circle`. Cross-references build naturally as more entities share affordance names (dragon fire_breath + mage fire_breath → fire_breath concept with two entity associations).

### Layer 2: Dual-level NAc causal links

When the agent experiences a consequence tied to an entity-specific affordance, create BOTH:

1. **Specific link:** `dragon_fire_breath → burn_pain` (full confidence, standard RPE)
2. **Abstract link:** `fire_breath → burn_pain` (reduced confidence, e.g., 0.5x the specific link)

The abstract link transfers to novel entities. When the agent later encounters `mage_fire_breath`, the NAc lookup for `fire_breath` returns the abstract prior: "fire breath causes pain, confidence 0.3."

Key constraint: abstract links should decay faster or have lower max confidence than specific links. The agent should still learn entity-specific associations that can override the abstract prior (maybe mage fire_breath heals allies, contradicting the dragon-learned prior).

### Layer 3: Discovery + sense integration

**`discover_tools`** already searches self-entity affordances. Add a secondary search against ATL affordance concepts:

```
discover_tools("fire attack")
  → Self tools: base_humanoid_use (includes "attack" in description)
  → Known affordance concepts: fire_breath (observed on dragon — dangerous, causes burn)
  → Output: "You don't have a fire tool. But you've observed fire_breath before
             (from dragon) — it's a dangerous ranged attack. Consider dodging or
             finding cover."
```

**`sense_presence`** already shows scene entity affordances. Add ATL concept annotations when available:

```
[SCENE] dragon (creature)
  Observed capabilities:
    combat: fire_breath (ranged fire attack) [DANGEROUS — you've been burned by this before]
    combat: tail_sweep (area knockback)
    flight: circle (aerial repositioning)
```

The `[DANGEROUS]` annotation comes from NAc's abstract `fire_breath → pain` link.

### Layer 4: Affordance category taxonomy (deferred)

Future work: group affordances into semantic categories (fire_attacks, melee_attacks, healing, movement, perception). This enables higher-level transfer: "all fire attacks are dangerous" even if the specific affordance name differs (`flame_jet` vs `fire_breath`). This likely requires embedding similarity on affordance descriptions, not just name matching. Defer until Layers 1-3 prove the value.

## What changes (estimated)

| Layer | Files | Change | LOC est |
|-------|-------|--------|---------|
| 1 | `imagination/trigger.py`, `tools/discovery.py` | Feed bare affordance names to ATL on scene entity observation | +30 |
| 1 | `memory/atl.py` or `memory/concept_extractor.py` | `observe_affordance_concepts()` method | +20 |
| 2 | `decisions/nac.py` | Dual-level link creation: specific + abstract | +40 |
| 2 | `decisions/nac.py` | Abstract link confidence scaling + faster decay | +15 |
| 3 | `tools/discovery.py` | ATL concept lookup in discover_tools output | +25 |
| 3 | `tools/discovery.py` | NAc annotation on sense_presence affordances | +15 |
| T | tests | Unit tests for each layer | +80 |
| **Net** | | | **~+225** |

## Key constraints

1. **Cerebellum stays entity-specific.** Forward models do NOT participate in concept transfer. Motor precision is per-entity by design.
2. **Abstract NAc links have lower confidence ceiling.** They provide priors, not certainties. Specific experience overrides abstract priors.
3. **No new StructuredContext fields.** Affordance concepts flow through existing ATL concept_context and NAc causal_context paths.
4. **Entity-specific tool names unchanged.** The motor/execution layer keeps `{entity}_{affordance}` naming. The concept layer is additive.
5. **Bio-system separation preserved.** ATL handles concept formation, NAc handles reward association. No merging of concerns.

## Open questions

- **Q1:** Should abstract NAc links use a separate confidence scale, or just a multiplier on the standard scale? Separate scale is cleaner but adds complexity.
- **Q2:** When the agent has BOTH a specific and abstract link for the same affordance, how do they compose? Max? Weighted average? The specific link should dominate when confident.
- **Q3:** Should ATL affordance concepts be tagged differently from action concepts? A `fire_breath` concept formed from observation is different from one formed from execution.
- **Q4:** How does this interact with the imagination system? If the agent imagines a fire_mage, should the imagined entity's affordances also feed ATL concepts?

## Validation

1. Agent observes dragon fire_breath, gets burned. Later encounters mage with fire_breath. Verify NAc abstract link fires: agent shows caution on first encounter with mage fire_breath (no re-learning needed).
2. Agent uses sense_presence near mage — verify ATL concept annotation: "fire_breath [DANGEROUS — observed on dragon]".
3. Agent calls discover_tools("fire") — verify ATL-informed response mentions known fire affordances even though agent has no fire tools.
4. Verify Cerebellum has NO forward model for mage fire_breath (only dragon-specific). Motor prediction requires entity-specific experience.
5. Verify abstract link doesn't override contradicting specific experience (mage fire_breath heals allies → specific positive link outweighs abstract negative prior).
