# Active initiatives — shipped history archived 2026-08-13

**Archived from CLAUDE.md on 2026-08-13** (claude_md_diet Stage 1). The enforced rule
survives as a compressed stub — in the slim CLAUDE.md core or in the owning
`docs/agents/<subsystem>.md` brief (see CLAUDE.md's routing table). This file preserves
the full original narrative: incident history, dates, PR numbers, dead-end hypotheses.

---

## Active initiatives

See [docs/plans/README.md](docs/plans/README.md) for the roadmap index. Current version: v0.7.0 on PyPI as `pymaxim` ([publication guide](docs/publication_guide.md)).

**Recently shipped (2026-04-24):**
- **Affordance Concept Transfer** — Substrate-native cross-entity learning. Routes affordance names through existing bio-pipeline (LinguisticEncoder → EC → ATL → NAc reward_bias). Transfer via EC pattern completion — "flame" maps to same node as "fire" (cosine 0.785). SCN temporal coupling for eligibility traces (first SCN-substrate PoC). Pre-existing NAc per-tick decay bug fixed. `discover_tools` renamed to `sense_tools`. Entity ownership shipped (self vs scene separation).
- **Entity Ownership** — Self vs scene entity tools. Agent controls own body, observes others. Dragon bug fixed.

**Previously shipped (2026-04-20):**
- **0.7 Feature Completion** — Self-generating simulations. All tracks landed:
  - R0 Prerequisites: ComponentRegistry thread safety, sim-mode consolidation, TOOL_ALIASES lock
  - B3.1 Acting Coach: config + prompt section with bio-system modulation (NAc caution, pain anticipation, cerebellum predictions)
  - F3-F5 Agent Factory: sim orchestrator + Reachy + headless API migrated to `create_full_agent`
  - E2 Real LLM: foundry wired to real LLM with entity context injection + synonym generation
  - E2.5 ComponentIndex: two-layer semantic discovery (alias hash O(1) + embedding cosine similarity)
  - E3 Auto-Curation: `--auto-curate` CLI for pre-sim coverage gap filling via foundry
  - I1 Imagination Trigger: entity extraction → ComponentIndex lookup → DN arousal gate → design dispatch
  - I2 Real-time Design: ImaginationDesigner with quick validation + synonym generation
  - I3 Scene-scoped Tools: tool window with cap (20 scene tools), deactivation, executor gate
  - Integration wiring: ImaginationTrigger constructed in orchestrator AUT path, session-end cleanup (imagined link decay + ephemeral entity clearing)
- **Version bump to 0.7.0.** Experiment: [docs/experiments/07_imagination_wiring.md](docs/experiments/07_imagination_wiring.md).

**Previously shipped (2026-04-17):**
- Valence annotation, SEM Learning Loop, Behavioral convergence wiring, Experiments 1-4 (41/41 hypotheses confirmed). Version 0.3.0.

**Previously shipped (2026-04-11/12):**
- Foundations wave F0.1–F0.8, Reaction abstraction Phases 1–4, Cleanup wave C1–C4, Peer/leader flexibility P1–P9, Simulator upgrades S1–S4. All archived.

**Gating 1.0** (three focused substrate plans, split from the master plan):
- [substrate_p0_pilot.md](docs/plans/archive/substrate_p0_pilot.md) — **COMPLETE** (2026-04-12). Baseline pinned at 78.5%. Results: [docs/experiments/p0_baseline_sweep.md](docs/experiments/p0_baseline_sweep.md).
- [substrate_recognition.md](docs/plans/archive/substrate_recognition.md) — **COMPLETE** (2026-04-14). B1+P1 shipped 2026-04-12 at 91.7% collapse (`paraphrase-mpnet@0.40` + centroid update). P2 Stages 1+2 shipped via PR #100 (SEM pain cascade end-to-end on real `rusty_sword` + NAc `_context_similarity` directional fix + PainBus dual-layer rewrite). P2 Stage 3 shipped via PR #102 — real-embedding sweep at `paraphrase-mpnet@0.70, reward 2.0` cleared with **+56.0 ± 29.0 pp target gain / 0.0 ± 0.0 pp distractor drift / 94% monotone / 9-of-10 seeds**, after three metric pivots (node-count → raw pair-collapse → plurality-ownership self-collapse) + a fixture pivot. Results: [docs/experiments/p1_recognition_sweep.md](docs/experiments/p1_recognition_sweep.md) + [docs/experiments/p2_reward_modulation_sweep.md](docs/experiments/p2_reward_modulation_sweep.md) + [docs/experiments/p2_sem_pain_cascade.md](docs/experiments/p2_sem_pain_cascade.md). Reproduction runbook: [docs/experiments/protocols/p2_reward_modulation_reproduction.md](docs/experiments/protocols/p2_reward_modulation_reproduction.md). 0.3-minimum gate CLOSED.
- [substrate_binding_persistence.md](docs/plans/archive/substrate_binding_persistence.md) — **SPLIT COMPLETE + ARCHIVED.** Now a pure index. All four 0.3-target phases CLOSED. Per-phase plan files created for 0.5 track.

**Living practice docs (paired with substrate_plan):**
- [behavioral_convergence_practice.md](docs/plans/deferred/behavioral_convergence_practice.md) — does the agent actually get better across sessions? Living doc, not a gate.
- [memory_consolidation_practice.md](docs/plans/deferred/memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism. Kicks in when P8 ships in 0.5.

**Parallel:**
- [tool_refinement_plan.md](docs/plans/deferred/tool_refinement_plan.md) — living doc for agent tool curation.

**Deferred (post-1.0, revive on trigger):** Bio-System Plugin Discovery, Unified Event Bus, Mother NPC Stimulus, Mother Maxim, Pecking Order Graph, Asset Foundry, DM Extensions. See [docs/plans/deferred/](docs/plans/deferred/).

