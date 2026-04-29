# Scene actor affordances — `target_effect` + OrchestratorActorTool

**Status:** DRAFT (1.1)
**Estimated scope:** ~110 LOC
**Branch:** TBD
**Diagnostic for:** [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md)

---

## Motivation

Today, scene entities (NPCs, creatures, environmental hazards) only affect the AUT body via the **AUT's own actions** — the agent picks up the rock, gets cut. There's no mechanism for the **scene** to act on the agent. A dragon described as breathing fire produces narrative text; the AUT body never feels it unless the agent first invokes a tool that triggers `evaluate_failures()`.

This is the "dragon-narration symptom": rich narrative output, zero substrate engagement, no NAc learning.

## Design (one-line)

Add a `target_effect` field to `AffordanceSchema` that mirrors `self_effect` but writes to the AUT body instead of the affordance owner. Add an `OrchestratorActorTool` that lets the orchestrator invoke a scene entity's affordance against the AUT.

```yaml
# Scene entity (e.g. dragon)
affordances:
  fire_breath:
    target_effect:
      arms.thermal: +0.6
      core_temperature: +0.3
```

When the orchestrator narrates "the dragon breathes fire at you," it also calls `OrchestratorActorTool(actor="dragon", affordance="fire_breath", target="aut_body")`. The AUT body's sensors update → `evaluate_failures()` fires → PainBus → NAc. Same downstream pipeline as Layer 1 (entity acquisition) and Layer 2 (proximity) from the cradle's three-layer sensation model.

## Why before agent-backed entities

This is a **diagnostic**. If `target_effect` closes the dragon-narration symptom, agent-backed entities (full cognition for NPCs) is over-engineering. If it doesn't — if NPCs need pathfinding, plan, memory of past encounters with the AUT — then [deferred/agent_backed_entities.md](deferred/agent_backed_entities.md) revives.

Cheap to ship (~110 LOC), fast to evaluate, decisive.

## Stages

1. `target_effect` parsing in `embodiment/spec.py` mirroring `self_effect`.
2. `OrchestratorActorTool` in `simulation/tools.py` — the orchestrator's verb for "the scene acts on the AUT."
3. Orchestrator prompt guidance to call this tool when narrating threat / interaction.
4. Validation experiment — does AUT learn dragon avoidance over a 25-turn session? Does it transfer to a fresh session via `--resume-sim`?
5. Multi-agent attribution — when a scene entity acts on a multi-agent setup, which agent gets the pain attribution? See `feedback_v1_p4_multi_agent_attribution.md` for the open work here.

## Why 1.1 not 1.0

The cradle (B4 in [v1_refinement.md](v1_refinement.md)) already validates the cross-session learning claim with a simpler scene model. This is enrichment, not gating.