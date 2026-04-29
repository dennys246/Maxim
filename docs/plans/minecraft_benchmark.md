# Minecraft benchmark — live demo + harness comparison

**Status:** STUB (1.1 splash launch)
**Branch:** TBD
**Depends on:** Cradle ([v1_refinement.md](v1_refinement.md) B4 — SHIPPED), [scene_actor_affordances.md](scene_actor_affordances.md), Mineflayer adapter (TBD)

---

## Motivation

Voyager (NeurIPS 2023), GITM (2023), and SPRING (2023) put Minecraft on the map as the embodied-agent benchmark for LLM-driven harnesses. A direct comparison on the same seeds + turn budgets + embodiment is the strongest story for "Maxim's bio-pipeline produces measurably different agent behavior than prompt-engineering harnesses."

## What's needed

- **Mineflayer adapter** — a `PerceptSource` + `ActionSink` pair implementing the Maxim simulation bridge protocols against the [Mineflayer](https://github.com/PrismarineJS/mineflayer) Node.js Minecraft bot library. Likely WebSocket bridge to Python.
- **Comparison protocol** — same world seed, same turn budget, same starting inventory, same goal (e.g. "obtain diamond"). Run Voyager, GITM, SPRING, and Maxim on identical configurations.
- **Metrics** — milestones reached, action efficiency, cross-session transfer (a metric Voyager doesn't track but Maxim claims natively), substrate engagement (NAc links, hippocampal recalls).
- **Reproducibility** — public seeds, public docker images, public results.

## Why before launch

Cross-session learning is the 1.0 research claim, validated internally via the cradle. Minecraft is the externally-recognizable benchmark that makes the claim land outside the bio-cog community.

## Why 1.1 not 1.0

1. Comparison protocol design (same seeds, same turn budgets, same embodiment) is research-grade work that benefits from time, not haste.
2. 1.0 is the interface freeze; 1.1 is the showpiece. Two news cycles, less risk per release.
3. Cradle already provides the cross-session learning evidence 1.0 needs. Minecraft strengthens the story without gating it.

## Open questions

- Does Maxim's discrete tick model align with Minecraft's 20Hz tick rate? Likely needs a sub-sampling adapter.
- What's the right success criterion — "first to diamond," "diamond per LLM-token," "diamond per cross-session-resumed run"? The last one is the most Maxim-favorable but also the most novel — needs framing.
- Does [scene_actor_affordances.md](scene_actor_affordances.md) need to ship before this? Hostile mobs (zombies, skeletons) are scene actors — without `target_effect`, AUT damage from monsters has to flow through narrative reflex, which may be too noisy in a fast-twitch game environment.