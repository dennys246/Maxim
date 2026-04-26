# Orchestrator SEM Damage Wiring — close the reactive chain

**Status:** Shell plan (2026-04-24)
**Scope:** ~50-100 LOC in simulation/orchestrator.py
**Depends on:** [affordance_concept_transfer.md](affordance_concept_transfer.md) (shipped), [temporal_credit_integration.md](temporal_credit_integration.md) (shipped)
**Blocks:** Cross-session affordance transfer validation (Set 1B shows 0 reward_bias because pain never fires)
**Branch:** TBD (focused PR)

---

## Problem

When the orchestrator narrates "the dragon breathes fire at you," it's a **text event** — no SEM modulator actually executes, no sensor changes, no failure mode fires, no PainBus signal, no NAc learning.  The reactive chain is:

```
dragon.fire_breath modulator executes
  → agent.health sensor decreases
    → FailureMode triggers (health < threshold)
      → PainBus.publish(PainSignal)
        → NAc records negative outcome
          → _reward_bias goes negative on fire substrate nodes
            → sense_presence shows [DANGEROUS]
```

The chain exists in code but the first link is never pulled — the orchestrator describes combat narratively without actually invoking SEM actions.

## Design

When the orchestrator's probe mentions a scene entity attacking the agent, the sim bridge should also trigger the entity's modulator against the agent's body.

**Option A: Orchestrator-side combat bridge (~50 LOC)**

After `send_message` delivers a probe mentioning combat/attack, scan for known scene entities and their offensive affordances.  If the probe describes an attack, call the entity's modulator to produce actual sensor changes on the agent's body.

```python
# In orchestrator probe post-processing:
if _mentions_attack(probe_text, scene_entities):
    entity, affordance = _identify_attacker(probe_text, entity_map)
    if entity and affordance:
        # Execute the entity's modulator against agent body
        result = entity.modulators[mod_name].execute(affordance, params)
        # This triggers sensor changes → FailureMode → PainBus → NAc
```

**Option B: Narrative damage events (~30 LOC, simpler)**

Don't try to parse which entity attacks — just apply damage to the agent's sensors when the narrative describes harm.  The orchestrator already knows the scenario intent.

```python
# After orchestrator sends a combat probe:
if _is_combat_turn(probe_text):
    agent_body.vital_metrics["health"] -= 0.1
    agent_body.vital_metrics["stamina"] -= 0.05
    # Next body.tick() evaluates failure modes against new readings
```

**Recommended: Option B first** — simpler, validates the full chain.  Option A later when we want entity-specific damage types (fire vs ice vs physical).

## Validation

Re-run Set 1 after this change:
- Session 1: health drops during dragon combat → PainBus fires → NAc records negative on fire nodes
- Session 2: `[DANGEROUS]` appears on mage fire affordances

## Key constraint

The damage must flow through the existing SEM → PainBus → NAc pipeline, not a shortcut.  If we bypass `body.tick()` and call `nac.credit_node()` directly, we validate nothing.
