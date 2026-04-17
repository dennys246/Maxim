# Prompt Track — B3 Acting Coach + B5 Embodiment/Narrative Separation

**Status:** Draft — opens after B1 (SHIPPED).
**Scope:** ~450 LOC across two phases
**Target version:** 0.5 (formerly 0.4)
**Gates:** null (B4 gates 1.0, not B3/B5)
**Depends on:** B1 (SHIPPED — prompt composition + PromptAssembler)
**Blocks:** nothing directly (B4 is independent)
**Parent:** [substrate_binding_persistence.md](archive/substrate_binding_persistence.md)

## B3 — Acting Coach Layer

### Goal

Meta-prompt scaffold for NPCs: role values, speech register, failure modes, continuity contract. Optional per-character via `AgentConfig.personality`.

### Hypothesis

An NPC with Acting Coach scaffolding produces measurably more consistent behavior in a blind A/B test compared to the same NPC without it.

### Minimum implementation (~300 LOC)

- `prompts/acting_coach.py`: Acting Coach meta-prompt builder
  - Role values (what the character cares about)
  - Speech register (formal/casual/archaic/etc.)
  - Failure modes (what the character does under stress)
  - Continuity contract (what the character remembers between turns)
- Integration with `PromptAssembler` as an optional layer
- `AgentConfig.acting_coach: ActingCoachConfig | None` field
- AgentFactory wiring when `personality` is set

### Pass criteria

- Blind A/B test: acting-coach NPC measurably more consistent (rated by LLM judge on voice consistency across 10 turns)
- No regression on existing NPC behavior when acting coach is disabled

---

## B5 — Embodiment/Narrative Separation

### Goal

Formalize the boundary between SEM (embodiment layer) and DM (narrative layer). Ensure `PromptAssembler` cleanly composes both without bleed.

### Minimum implementation (~150 LOC)

- Lint-style contract test: verify no SEM-specific tokens appear in narrative prompts and vice versa
- Clear docstring contracts on `PromptAssembler.build()` sections
- Separation audit of existing prompt templates

### Pass criteria

- Contract test passes
- No SEM tokens in narrative output
- No narrative tokens in embodiment output

## Deferred follow-ups

- Per-NPC voice tuning (practice doc territory — behavioral_convergence_practice)
- Dynamic personality evolution (substrate-driven personality shifts)

## Load-bearing invariants (filled in AFTER shipping)
