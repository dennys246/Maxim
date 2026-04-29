# Agent-backed entities — 3-tier cognition + Cradle-trained cast + mesh-pressure budget

**Status:** DEFERRED (revive if [scene_actor_affordances.md](../scene_actor_affordances.md) doesn't close the dragon-narration gap, OR if Minecraft demo exposes a cognition gap)

---

## Motivation

Today, NPCs and creatures in a Maxim sim are descriptions on the page. The orchestrator narrates them; the AUT reads the narration. They have no body, no memory, no goals. This is fine for static scenery (rocks, fire pits) but produces the dragon-narration symptom for active threats — the world describes, the AUT reads, the substrate doesn't engage.

[scene_actor_affordances.md](../scene_actor_affordances.md) is the **diagnostic** for whether NPCs need genuine cognition. If `target_effect` + `OrchestratorActorTool` closes the substrate-engagement gap, full agent backing is over-engineering. If it doesn't — if NPCs need pathfinding, memory of past encounters, plan continuity across turns — this plan revives.

## Sketch

Three tiers of NPC cognition, calibrated to mesh budget pressure (the dominant cost is LLM calls):

- **Tier 1 — reactive reflex.** No LLM call. Triggered by AUT proximity / action keywords. Fires a fixed affordance. ~the existing reflex system, just attached to a non-AUT entity. Cost: zero LLM tokens.
- **Tier 2 — bounded plan.** Single LLM call per encounter to pick from a small affordance menu. State persists across the encounter. Cost: ~1 LLM call per AUT-NPC interaction.
- **Tier 3 — full agent.** Independent `MaximAgent` instance with hippocampus + NAc + ATL. Cross-session memory of past AUT encounters. Cost: full LLM agent loop, but only for high-stakes recurring antagonists.

Mesh budget allocates Tier 3 sparingly — a single Cradle-trained boss antagonist, not every village zombie.

## Cradle-trained cast

The cradle (B4 in [v1_refinement.md](../v1_refinement.md)) demonstrates non-linguistic sensorimotor learning on an AUT. The same training pipeline can produce trained Tier 3 agents — antagonists that have learned avoidance / pursuit patterns from their own embodied history. A boss creature that's been "alive" through 50 prior sims has substrate-grounded behavior the AUT can learn against.

## Mesh-pressure budget

Tier assignment is dynamic, based on token budget remaining + agent priority. Out-of-budget Tier 3 NPCs degrade to Tier 2; out-of-budget Tier 2 NPCs degrade to Tier 1. The orchestrator allocates the budget; per-tier behavior is consistent within a single AUT encounter (no thrashing mid-fight).

## Revive when

- `scene_actor_affordances` ships and the dragon-narration substrate-engagement gap **persists** — narrative-driven sensor writes aren't enough, NPCs need genuine state.
- Minecraft benchmark exposes a clear cognition gap — zombies need pathfinding the orchestrator can't synthesize, villagers need trade memory across sessions, etc.
- Multi-agent attribution work (`feedback_v1_p4_multi_agent_attribution.md`) lights up — once that exists, a second agent in the same sim is cheap-to-instantiate; the question becomes "should this NPC be one?"

## Why deferred

Three cost reasons:

1. **Diagnostic first.** `scene_actor_affordances` is ~110 LOC and answers the "do we even need this?" question. Building Tier 3 first and then realizing Tier 1 sensor writes were enough is wasted effort.
2. **Mesh budget pressure is real.** A 25-turn sim with 4 Tier 3 NPCs costs 5x what a single-AUT sim costs. The economics need to be worth the substrate gain.
3. **1.0 doesn't need this.** The cradle proves cross-session learning on a single AUT. This plan is for the next research story, not the current one.

## Open questions

- How does the AUT's hippocampus distinguish "what *I* did to the dragon" from "what the dragon did to me"? Today, `agent_id` attribution on NAc/hippocampus events is mostly single-agent. Per-agent memory isolation in a shared scene is non-trivial and is the actual gating concern for Tier 3.
- The Cradle-trained cast idea assumes saved bio-state is portable across instances. Today, hippocampus + NAc serialization is per-agent. Loading a "trained dragon" means snapshotting + loading `Hippocampus`/`NAc`/`ATL` for a non-AUT entity, which the simulation orchestrator doesn't currently support.