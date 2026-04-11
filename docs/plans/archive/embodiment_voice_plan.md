# Embodiment & Voice Plan — Prompts, Acting Coach, Replanning

**Status:** ARCHIVED — merged into [substrate_plan.md](../substrate_plan.md) as Track B (B1–B5). B2 (`NarrativeModulator` ghost removal) landed in the substrate plan's F0 foundation wave. Merge happened because B1's PromptAssembler and substrate P1's text-to-prompt migration touch the same files — running them as separate plans was a merge-conflict generator, and the original plan already noted "P1 and B1 should land together." See substrate_plan.md's Track B section for the current version.

**Original target versions:** 0.3 → 0.4 → 1.0

## Goal

Give NPCs and the planning agent a coherent voice. Today the prompt layer is rotting: personas are extracted and thrown away, prompts live in four unrelated locations, and the "acting coach" metaphor that would guide LLM role inhabitation doesn't exist. This plan is the reason the Substrate plan's convergence harnesses will have anything interesting to measure — without coherent behavior, there's nothing to converge on.

## Evidence of rot

- **NPC persona is extracted and discarded.** [dm_runtime.py:304-342](../../src/maxim/simulation/dm_runtime.py#L304-L342) pulls `persona_prompt` from entity metadata with a comment "Used by NarrativeModulator in Slice 2" — and then never uses it. NPCs are indistinguishable at the prompt layer; only dialogue hints leak personality.
- **`NarrativeModulator` is a ghost.** Referenced in [cerebellum_modulator.py:43](../../src/maxim/embodiment/cerebellum_modulator.py#L43) and in the dm_runtime comment above. Does not exist anywhere in the codebase. Slice 2 was designed and never shipped.
- **Prompts live in four unrelated places:**
  - Hardcoded strings: [narrator.py:24-80](../../src/maxim/simulation/narrator.py#L24-L80)
  - Inline persona strings: [personas.py:27-330](../../src/maxim/simulation/personas.py#L27-L330)
  - Template `.txt` files: [_data/prompts/planning/](../../src/maxim/_data/prompts/planning/)
  - YAML one-liners: [_data/components/npcs/](../../src/maxim/_data/components/npcs/)
- **No central assembler.** [prompt_profiles.py:75-94](../../src/maxim/prompts/prompt_profiles.py#L75-L94) does ad-hoc injection; [prompt_builder.py:46-342](../../src/maxim/agents/prompt_builder.py#L46-L342) has composition methods but isn't the single source of truth.
- **Replanning is skeletal.** [replanning.txt](../../src/maxim/_data/prompts/planning/replanning.txt) is 24 lines of stateless sub-goal regeneration. No root-cause field. No alternative-approach branching. No memory of prior replans.
- **Embodiment and narrative are siblings who never speak.** SEM gives you sensors + affordances. DM gives you scene + stakes. Nothing composes `identity + sensors + affordances + scene + memory` into one system message.

## Phases

### B1. PromptAssembler — single composition point (0.3)

One class that takes structured inputs and produces the final system message. Replaces the four scattered prompt locations with a composable pipeline.

```
PromptAssembler.compose(
    identity: Persona,           # who the character IS (from YAML or persona registry)
    sensors: SensorState,        # what they perceive right now (from SEM)
    affordances: list[Action],   # what they can do (from SEM + tool registry)
    scene: SceneContext,         # what's happening around them (from DM)
    memory: MemorySummary,       # relevant recalled episodes/concepts (from MemoryHub)
    coach: ActingCoach | None,   # meta-guidance layer (B3)
) -> SystemMessage
```

**Files touched:** new `prompts/assembler.py`, refactor `agents/prompt_builder.py` to delegate, deprecate the ad-hoc injection in `prompt_profiles.py`.

**Exit:** All NPC and planning-agent system messages flow through `PromptAssembler.compose`. `grep -r "system_message = f\""` returns nothing outside the assembler.

### B2. Kill the `NarrativeModulator` ghost (0.3)

Fix [dm_runtime.py:324](../../src/maxim/simulation/dm_runtime.py#L324). Route NPC `persona_prompt` through `PromptAssembler` so it actually reaches the LLM. Remove the dead reference in [cerebellum_modulator.py:43](../../src/maxim/embodiment/cerebellum_modulator.py#L43).

**Files touched:** `simulation/dm_runtime.py`, `embodiment/cerebellum_modulator.py`.

**Exit:** A campaign with two NPCs (guard + merchant) produces visibly different dialogue and behavior. Blind A/B test: reviewers correctly identify which NPC said what >80% of the time. `grep -r NarrativeModulator` returns zero hits.

### B3. Acting Coach layer (0.4)

Meta-prompt that sits on top of identity and guides role inhabitation. Not a replacement for persona — a scaffold around it. Gives the LLM explicit instructions on:

- **Role values:** what this character prioritizes (survival, loyalty, curiosity, greed)
- **Speech register:** vocabulary, cadence, what they would never say
- **Failure modes:** how they break under stress (hostile when trust is low, evasive when cornered)
- **Continuity contract:** reference their own past actions when recalled from memory

This is optional per-character — a simple guard doesn't need one, but a campaign-critical NPC does.

**Files touched:** new `prompts/acting_coach.py`, campaign YAML schema extension for `acting_coach` blocks.

**Exit:** Acting coach block in a campaign YAML produces measurably more consistent NPC behavior across a multi-turn encounter than the same NPC without one. Tests use deterministic seed + fixed scene.

### B4. Replanning with failure diagnosis (0.4 — gates 1.0)

Rewrite [replanning.txt](../../src/maxim/_data/prompts/planning/replanning.txt) with real structure:

```
FAILED_PLAN: {plan}
COMPLETED_STEPS: {completed}
FAILURE_POINT: {failed_step}
OBSERVED_EVIDENCE: {what_happened}
PRIOR_REPLAN_ATTEMPTS: {history}           # new: not stateless anymore
ROOT_CAUSE_HYPOTHESIS: {diagnosis}         # new: must reason about why
ALTERNATIVE_APPROACHES: {branches}         # new: explicit branching
SELECTED_APPROACH: {choice + rationale}
REVISED_PLAN: {plan}
```

Plus: persist replan attempts so the agent has memory of what it already tried in-session. This overlaps with Substrate convergence harness #4 (failure avoidance).

**Files touched:** `_data/prompts/planning/replanning.txt`, `_data/prompts/planning/reflection.txt`, `runtime/loop_controller.py` (replan invocation + history threading).

**Exit:** Induced failure scenario: agent's first plan fails deterministically. Agent's second plan differs structurally (not just cosmetically) from the first. Agent's third attempt does not repeat either earlier approach. Substrate convergence harness #4 passes.

### B5. Embodiment/narrative separation (0.4)

Formalize the three roles:

- **SEM → embodiment inputs** (sensors, affordances, failure modes) — data-driven, per-entity
- **DM → narrative inputs** (scene, stakes, world state) — data-driven, per-campaign
- **PromptAssembler → composition** — single place where they combine

Document the contract. Add a lint-style check that narrative modules don't reach into SEM internals and vice versa.

**Files touched:** docs/architecture notes, `embodiment/sem.py` + `simulation/dm_runtime.py` interface boundaries, new contract test in `tests/integration/`.

**Exit:** Architecture doc describes the three roles. Contract test passes. No module imports across the boundary.

## Scope

~500 LOC net. Small compared to Substrate, but high-visibility — every sim run touches it.

## Non-goals

- **No new LLM router features.** Use the existing [models/language/router.py](../../src/maxim/models/language/router.py).
- **No vision or audio prompting.** Text-only scope for 1.0. Multi-modal prompt composition is 1.1+.
- **No prompt caching / optimization.** Correctness first. Performance after convergence is proven.

## Cross-pollination with Substrate

- **B1 PromptAssembler** consumes `MemorySummary` derived from Substrate A3 ATL centroids. As ATL moves to embedding-space centroids, the memory layer of the prompt gets qualitatively better.
- **B3 Acting Coach continuity** relies on Substrate A5 reward-modulated recall to surface the character's own past rewarded/punished actions.
- **B4 Replanning** feeds directly into Substrate A6 convergence harness #4. Both plans must land before the harness passes.

## Risks

- **B3 is subjective.** "NPC feels more coherent" is hard to measure. The blind A/B test is the best we have; budget real evaluation time, don't eyeball it.
- **B4 changes a load-bearing prompt.** Replanning is exercised by every non-trivial sim. Regressions will be immediate and loud. Keep the old replanning.txt as a fallback path during A/B comparison; remove it before 1.0.
- **B1 is a refactor, not a feature.** Easy to under-scope. Every prompt call site needs to migrate; budget migration time, not just the assembler class.
