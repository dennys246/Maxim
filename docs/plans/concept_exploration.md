# Concept Exploration Plan (Shell)

**Status:** Shell (2026-04-20)
**Scope:** 0.7 — Simulation Scalability
**Depends on:** SEM Tool Discovery (S1 baseline), ComponentIndex (E2.5), Imagination (I1-I2)
**Sequencing:** After SEM Tool Discovery ships and provides a measurable baseline

---

## Problem

SEM tool discovery solves "what can I physically do?" but not "what is interesting here?" These are different questions. When the sim goal is "test sword combat", keyword matching surfaces combat tools. But when the goal is "explore freely", "survive", or "understand this world", the agent has no mechanism to expand abstract intent into concrete directions.

This is a simulation scalability problem: as scenarios grow more open-ended, the agent needs to orient itself conceptually — not just discover tools, but discover *what to pay attention to*. A human dropped into a new environment doesn't start by listing their motor capabilities; they look around, notice things, form hypotheses, and then act.

The gap: SEM tool discovery's keyword matching + modulator walk handles specific physical queries well. It handles vague queries acceptably (modulator category summary fallback). But it cannot:
- Expand "explore" into "investigate that strange sound, examine the locked door, test whether the bridge holds weight"
- Connect "survive" to "find shelter, assess threats, locate water"
- Ground novel percepts ("the crystal hums") into actionable concepts ("crystals can resonate, shatter, amplify, store energy")

## Proposed Architecture

The concept exploration system sits between perception and action — it takes abstract intent or novel percepts and grounds them into concrete directions the agent can act on.

```
Novel percept: "the crystal hums with a faint blue light"
         ↓
Concept Grounding:
  1. Extract key concepts: [crystal, hum, blue light, resonance]
  2. Semantic expansion: crystal → {fragile, resonant, refractive, valuable, magical}
  3. Cross-reference with available entities + affordances
  4. Generate exploration directions:
     - "The crystal might respond to physical force (try striking it)"
     - "The humming suggests resonance (try different frequencies)"
     - "Blue light often indicates magical properties (examine closely)"
         ↓
Agent acts on grounded directions (via existing tools or discover_tools)
```

## Key Design Questions

1. **Tool or prompt section?** Should concept exploration be a tool the agent calls (`explore_concept(query)`) or a prompt section that runs automatically on novel percepts? The tool approach gives the agent control; the prompt approach ensures it happens even when the LLM doesn't think to ask.

2. **Expansion source:** Where does conceptual knowledge come from?
   - **ComponentIndex semantic signatures** — already contains affordance descriptions, sensor types, failure modes. Can find conceptually related entities.
   - **LLM's own knowledge** — the language model already knows that crystals are fragile and resonant. A well-crafted prompt can extract this without external calls.
   - **Embedding neighborhood** — words near "crystal" in embedding space (via `similarity/encoder.py`) surface related concepts without an LLM call. ~5ms latency.
   - **Internet/thesaurus** — richest source but adds latency + external dependency. Probably overkill for sim environments where the LLM's training data covers the domain.

3. **Interaction with imagination:** ImaginationTrigger already extracts novel noun phrases and designs entities for them. Concept exploration is the *other* response to novelty — not "create an entity for this" but "understand what this means." The two should share the novelty detection pipeline but diverge at the response: imagination creates, concept exploration orients.

4. **Interaction with Acting Coach:** The Acting Coach already modulates exploration via role_values (curiosity, survival) and bio-system signals (NAc caution, pain anticipation). Concept exploration outputs should flow through the Acting Coach as additional context, not bypass it.

5. **Latency budget:** If this runs every turn (prompt section), it needs to be < 10ms. If it's a tool the agent calls, 50-100ms is acceptable. Embedding neighborhood search is ~5ms; LLM call is ~500ms-2s (too slow for automatic).

## Possible Approaches

**A. Embedding neighborhood exploration (lightweight, no LLM call):**
Use the existing `similarity/encoder.py` to find concepts semantically near the query. "Crystal" → nearest neighbors in ComponentIndex embeddings → surfaces "glass", "gem", "prism" components and their affordances. Fast (~5ms), but limited to what's in the component library.

**B. LLM-powered concept grounding (rich, one LLM call):**
A focused LLM call: "Given this percept/goal, list 3-5 concrete actions the agent could try, grounded in the available entity types: [list]." Uses the agent's own LLM lane (small tier — this is a lightweight generation task). Rich output, but costs one LLM call per invocation.

**C. Hybrid: embedding for speed, LLM for depth:**
Embedding neighborhood runs automatically on novel percepts (prompt section, ~5ms). When the agent explicitly calls `explore_concept(query)`, it gets the richer LLM-grounded response. This mirrors how biological attention works — fast pre-attentive processing surfaces salient features, deliberate attention (costly) explores them deeply.

## Stages (TBD — flesh out after SEM tool discovery baseline)

**C1 — Embedding neighborhood exploration**
- New: `explore_concept` tool or prompt section
- Uses ComponentIndex.find_similar + embedding nearest-neighbor
- Returns related entities/affordances and conceptual associations
- Measures: does the agent explore more diverse affordances vs. SEM-discovery-only baseline?

**C2 — LLM-grounded concept expansion**
- When agent explicitly explores, make a focused LLM call for deeper grounding
- Contextualizes against available entities and the current scene
- Feeds results through Acting Coach as exploration directions

**C3 — Novelty-triggered automatic grounding**
- Share ImaginationTrigger's noun-phrase extraction pipeline
- On novel percept: fast embedding check → if conceptually interesting, add orientation note to next prompt turn
- Bio-plausible: this is the "curiosity reflex" — automatic attention to novelty

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| C1 | ~150 | Embedding exploration + ComponentIndex integration |
| C2 | ~100 | LLM grounding call + scene context |
| C3 | ~100 | Novelty pipeline sharing + automatic trigger |
| **Total** | **~350** | |

## Open Questions

- Should this share infrastructure with `discover_tools` (same tool, different mode) or be a separate tool? Leaning separate — different mental model for the agent ("what can I do" vs "what does this mean").
- How does this interact with the Cerebellum's forward models? Concept exploration could seed initial predictions ("crystals are fragile → high shatter probability on impact") that the Cerebellum refines with experience.
- What's the right measure of success? "Agent explores more diverse affordances" is one signal. "Agent achieves goals faster in open-ended scenarios" is harder to measure but more valuable.
- Does this replace or complement the Acting Coach's exploration directives? Complement — the coach says "explore", concept exploration says "explore *this specific thing*."
