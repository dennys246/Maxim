# B5 — embodiment / narrative separation

**Status:** DEFERRED (revive when prompt-bleed bug surfaces)
**Original track:** prompt_b3_b5_track.md (B3 shipped, B5 deferred)

---

## Motivation

B3 (Acting Coach) shipped — bio-modulated affordance exploration meta-prompt. B5 was the proposed follow-up: formalize the boundary between SEM affordance prompts (what the body can do) and DM narrative prompts (what the world does).

Today the AUT prompt assembler interleaves embodiment context (sensor states, available affordances, drive guidance) with narrative context (recent percepts, scene description). This works, but a clean separation would let:

1. Embodiment tools render their own prompt section deterministically without narrative wrapping.
2. DM-style narrative campaigns format-switch independently of embodiment context.
3. Agent-backed entities (when revived from [agent_backed_entities.md](agent_backed_entities.md)) get a per-tier prompt template that scales naturally — a Tier 1 villager gets embodiment + bare narrative, a Tier 3 antagonist gets full bio-modulation.

## Revive when

- A prompt-bleed bug surfaces — embodiment context contaminating narrative output, or vice versa, in a way that the current ad-hoc separation can't isolate.
- Agent-backed entities revive (currently deferred, see [agent_backed_entities.md](agent_backed_entities.md)) — that work needs per-tier prompt templates, and B5's separation makes per-tier templates natural.
- The `prompts/acting_coach.py` module accumulates more bio-modulation hooks (drive guidance, anticipatory pain, NAc caution) and the resulting prompt becomes hard to reason about as a single composition pass.

## Why deferred

B3 alone closes the immediate "bio-modulation in prompts" gap. B5 is a structural cleanup — valuable, not urgent. The cost of writing it post-1.0 is low because the underlying prompt assembler is internal (not part of the 1.0 frozen API surface).

## Sketch

A clean B5 would introduce a `PromptSection` protocol with declared fields (kind, source, priority) and let the assembler compose sections explicitly rather than via the current threaded text concatenation. Each registered tool / bio-system / DM module produces sections; the assembler resolves priority + length budget without any one author needing to know about all the others.

This pattern matches the codebase's "declared fields, not stashes" lesson (CLAUDE.md): when the prompt becomes complex enough that any one assembler can't reason about all the producers, push the contract DOWN into a typed section interface.