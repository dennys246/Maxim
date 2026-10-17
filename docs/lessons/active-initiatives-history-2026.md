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
- [memory_consolidation_practice.md](docs/plans/archive/memory_consolidation_practice.md) — refines the P8 sleep-replay mechanism. Kicks in when P8 ships in 0.5.

**Parallel:**
- [tool_refinement_plan.md](docs/plans/deferred/tool_refinement_plan.md) — living doc for agent tool curation.

**Deferred (post-1.0, revive on trigger):** Bio-System Plugin Discovery, Unified Event Bus, Mother NPC Stimulus, Mother Maxim, Pecking Order Graph, Asset Foundry, DM Extensions. See [docs/plans/deferred/](docs/plans/deferred/).


## CLAUDE.md §Active initiatives, verbatim, before the 2026-10-16 diet (links relative to the repo root)

The live version line stays in CLAUDE.md and [docs/plans/README.md](../plans/README.md); this copy is history and is not a sync line.

> Current version: **1.3.2** (`pyproject.toml` + `src/maxim/__init__.py`; PyPI: https://pypi.org/project/pymaxim/ — `main` is ahead of PyPI by policy, §Versioning). 1.3.2 is **"Decomposition"** — `agent_loop` taken apart under characterization tests, plus the ratchets and gates that hold it; no new behavioural claim (a breaking leader security default change; T1-2 ships STALE under `release-1.3.2-T1-2`). 1.3.1 is **"Hardening"** — fixes and their guards, no new behavioural claim (security, the public format freeze, public-API correctness, release integrity; Exp 10 re-run MAINTAINED (narrow), [#935](https://github.com/dennys246/Maxim/issues/935)). 1.3.0 is **"Oasis-2"** — the survival world: reward from the game (Paper 1.20.4, no LLM in the action path); **Exp 60 EARNED** (anticipatory drowning-avoidance) + **Exp 61 EARNED** (the survival fear transfers between agents) + the **R3 survival benchmark** (instrument + frozen baseline, nothing graduated) + Exp 56 re-baselined on 1.20.4. **Exp 62 rung A EARNED 2026-09-20** (after 1.3.0; not claimed by 1.3.1 until its different-reader pass is recorded; the shipped body carries the drowning-fear across pools — 1.4 Phase 1 closed; bounded to the sealed-shell frozen-day class; its "night miss" (0.799) is the `time_of_day` wrap just before dawn, not night — [#899](https://github.com/dennys246/Maxim/issues/899)). **Since B1 (2026-10-14/15) T1-13/14/15 (Exp 60/61/62) are PARTIAL, floor matched under BF2** (a tabular floor matches the Wire-4 fear; [baseline_floors.md](docs/plans/baseline_floors.md)). 1.2.x was **"Oasis"** (Exp 56 EARNED; 1.2.1 shipped the spoken-code pairing pieces, composition unwired — corrected in 1.3.0). This line is checked by `scripts/lint_version_sync.py` — bump it in the release transaction. Active theme **1.4** (working title "Anticipation"; [roadmap_1_4.md](docs/plans/roadmap_1_4.md)): generalization + multi-step credit on the survival world; perception fabric revives when GL3's registry+provenance stage ships and a 1.4 rung needs cross-modal binding ([grounding.md](docs/plans/grounding.md); owner 2026-10-07), microduck DEFERRED until a second physical body exists. The roadmap index is [docs/plans/README.md](docs/plans/README.md); the shipped-release history through 1.3 is archived at [docs/plans/archive/roadmap_1_1_to_1_3.md](docs/plans/archive/roadmap_1_1_to_1_3.md) (its 1.1.x item-16 block is CLOSED — verified 2026-09-20); behavioral-graduation gates live in [docs/plans/behavioral_graduation_candidates.md](docs/plans/behavioral_graduation_candidates.md). Deferred plans (revive on trigger): [docs/plans/deferred/](docs/plans/deferred/). Shipped-history through 2026-04 (the old "Recently shipped" ledger): [docs/lessons/active-initiatives-history-2026.md](docs/lessons/active-initiatives-history-2026.md).
