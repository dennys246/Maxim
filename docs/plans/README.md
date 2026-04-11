# Maxim Plans

Current version: **0.2.1** (published on PyPI as `pymaxim`).
Target: **1.0** — cross-session learning demonstrated without LLM fine-tuning.

## Active (gating 1.0)

- [substrate_plan.md](substrate_plan.md) — bio-stack convergence (unified Percept, EC/ATL/Hebbian, convergence harnesses)
- [embodiment_voice_plan.md](embodiment_voice_plan.md) — PromptAssembler, acting coach, replanning with failure diagnosis

These two plans run in parallel. They cross-pollinate at percept-derived state → prompt composition, and at concept memory → role memory.

## Parallel (ship anytime, not gating 1.0)

- [cleanup_wave.md](cleanup_wave.md) — fix `--interactive`, delete dead flags, display defaults, agent permissions
- [tool_refinement_plan.md](tool_refinement_plan.md) — living doc for agent tool surface curation

## Deferred (post-1.0, revive on trigger)

Design work is preserved in [deferred/](deferred/). Each plan has an explicit "revive when" condition at the top.

- [deferred/pecking_order_graph_plan.md](deferred/pecking_order_graph_plan.md) — unified hierarchy DAG
- [deferred/mother_maxim_plan.md](deferred/mother_maxim_plan.md) — persistent collective memory
- [deferred/asset_foundry_plan.md](deferred/asset_foundry_plan.md) — automated SEM component generation
- [deferred/dungeon_master_extensions.md](deferred/dungeon_master_extensions.md) — DM post-MVP features

## Archive

Completed or superseded plans live in [archive/](archive/). See the archive for historical context on pre-publication refinement, repo management, and the v1 versions of plans folded into the active spine.

## Version path to 1.0

The substrate plan is organized as **proof-obligation phases** (P1, P2, P3, P3.5, P4, P5, P6). Each phase is a falsifiable behavioral claim validated by an extended convergence simulation that compares the architecture against a negative control, with explicit swap points for when the claim fails.

| Version | Phases that must pass | What it proves |
|---|---|---|
| **0.3** | P1, P2, P3, **P3.5 (persistence)**, **P4 with minimal real vision** | Architecture's central claim holds end-to-end, across a real process boundary, with real cross-modal binding |
| **0.4** | P4 re-passed with production vision + email/Slack channel adapters | Architecture generalizes beyond minimal scope |
| **0.5** | P5 (stress persistence), P6 (extinction) | System persists under load and forgets appropriately |
| **1.0** | Stress-test sim combining all phases with full channel diversity | Cross-session learning without LLM fine-tuning at realistic scale |

Channels (SMS, email, Slack, narrative speech) are **TEXT modality with context metadata**, not separate modalities. Channel rollout: SMS + narrative in 0.3, email + Slack in 0.4. See [substrate_plan.md](substrate_plan.md) for phase definitions, convergence sims, negative controls, pass criteria, swap points, and fixture requirements.

## 1.0 exit criteria

- **Substrate P1 through P4 (with P3.5 persistence):** Pass in 0.3 with minimal vision and SMS+narrative channels. Re-pass P4 in 0.4 with production vision and email+Slack. Pass P5 and P6 in 0.5. Every phase passes both unit-sim and system-sim tiers and beats its negative control at p<0.05 across ≥5 seeds. A final stress-test sim combining all phases holds through 1.0.
- **Embodiment B4:** Replanning recovers from induced failures instead of regenerating identical plans; NPCs exhibit distinct, consistent voices in blind A/B sim runs.
- **Cleanup Wave:** Shipped. `maxim --interactive` works, `--help` fits on one screen, agent permissions layer live.

## Rules for this directory

- **Active plans stay in the root.** Anything in the root is on the critical path.
- **Deferred plans must state a revive trigger.** If you can't state the trigger, it doesn't belong in deferred — it belongs in archive.
- **No ghost plans.** If a plan references a module that doesn't exist (e.g., the old `NarrativeModulator`), fix the plan or delete the reference.
- **Merge before multiplying.** If two plans overlap by more than a phase, merge them. Historical example: salience_abstraction was folded into substrate_plan because `WhereCoord` required embedding-space percepts anyway.
