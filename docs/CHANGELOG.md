# Changelog

## 0.3.0 (2026-04-17)

### Highlights

**Cross-session learning without fine-tuning -- demonstrated across 3 tiers.**

An agent that interacts with SEM entities learns from outcomes (pain, success),
persists the learning, and makes different decisions in later sessions. 41/41
experimental hypotheses confirmed across 4 experiments.

### New features

- **Valence annotation** -- Reactions annotate Hebbian edges with affective valence.
  `Edge.metadata["valence"]` propagates through `spreading_activation(propagate_valence=True)`.
- **Cerebellum activation** -- `BioStack.cerebellum` wired into production. Forward model
  prediction with LLM fallback. Success/failure reactions with negativity bias.
- **NAc reward distribution** -- `distribute_reward` connects reactions to EC threshold
  adjustment via eligibility traces.
- **Pain spike episode boundary** -- `salience_spike_rule` closes episodes on high-intensity
  pain, creating clean "what went wrong" boundaries.
- **Valence in prompt assembler** -- `StructuredContext.valence_context` surfaces learned
  associations to the LLM. Strength-differentiated labels.
- **Episode observation in production** -- `observe_episode_event` fires in the agent loop
  with substrate node IDs and tool concepts.
- **Energy reaction bridge** -- `EnergyReactionBridge` emits hunger/fatigue/satiation
  reactions when energy thresholds cross.
- **SEM entity specs** -- food_ration, water_flask, poison_vial, antidote_vial, plus
  masked experimental vials (purple/teal/orange).
- **Concept decomposition** -- Stage 1 shipped with spaCy noun chunker. 100% concept-level
  recall vs 36.4% baseline.

### Experiments

- **Exp 1** (Tier 1, 11/11): Cross-session affective memory transfer
- **Exp 2** (Tier 1, 13/13): Energy-driven consumable learning
- **Exp 3** (Tier 2, 12/12): LLM acts on bio-system learning (10/10 experienced vs 0/10 fresh)
- **Exp 4** (Tier 3, 5/5): Organic LLM learning (teal rate: 0% -> 25% -> 100%)

### Infrastructure (shipped alongside substrate work)

- Reactive peer mesh: router-drain coupling (C4), auto-drain on persistent failure (C4.5)
- VRAM endpoint: `GET /v1/debug/vram`
- Bio-stack Wave 3: `build_bio_stack(*, persistence_dir)` canonical builder
- Plan split: substrate monolith -> 5 per-phase files
