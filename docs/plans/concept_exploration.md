# Deliberative Thinking + Bio-Enrichment Pipeline

**Status:** L0+L1+L2 SHIPPED (2026-04-20). L3 (NoveltyScorer extraction) deferred/optional.
**Scope:** 0.7 — Simulation Scalability
**Depends on:** SEM Tool Discovery (shipped), MemoryHub (Wave 2), Gating Abstraction G0+G1 (shipped)
**Prerequisite plan:** [gating_abstraction.md](gating_abstraction.md) — G0 (extract runtime/gating.py) + G1 (TextSalienceScorer)
**Replaces:** Concept Exploration Plan (shell, superseded by this broader design)

---

## Problem

The agent has bio-systems that learn from experience (hippocampus, NAc, ATL, EC, cerebellum) but there's no mechanism to *passively surface that learning during processing*. The `think` tool is write-only — it stores thoughts as episodes but gets no system response. Percepts arrive raw without bio-system context. The agent must explicitly call introspection tools (memory_recall, predict_outcome) to access its own experience.

Biological brains don't work this way. Every input is automatically colored by prior experience via the thalamic relay: you don't consciously recall past bridge experiences when you see a bridge — your brain fires associations and they influence perception before deliberation begins.

**Three gaps this plan addresses:**

1. **Think is write-only.** Agent thinks "how do I get past this gate?" and gets nothing back. No memories, no predictions, no associations. It has to separately call memory_recall, predict_outcome, etc.

2. **Percepts arrive uncontextualized.** User says "the bridge looks unstable" — agent processes this raw. Bio-systems have relevant experience (past bridge collapses, structural failure pain) but don't contribute to perception.

3. **No iterative deliberation.** Complex reasoning requires think → response → refine → response → act. Currently each think is isolated; there's no feedback loop where bio-system responses inform the next thought.

## Architecture

### Core Abstraction: BioEnrichmentPipeline

A source-agnostic pipeline that takes text and returns bio-system associations. Any text consumer can call it.

```python
@dataclass(frozen=True)
class EnrichmentResult:
    """Bio-system associations surfaced for a text input."""
    memories: list[EpisodicSummary]      # hippocampus: similar past episodes
    predictions: list[CausalPrediction]  # NAc: predicted outcomes
    concepts: list[ConceptLink]          # ATL/EC: related concepts
    affordances: list[str]               # ComponentIndex: available actions
    valence: float                       # overall approach/avoid signal (-1 to +1)
    novel: bool                          # EC pattern separation result

class BioEnrichmentPipeline:
    """Thalamic relay — gates and enriches text via bio-systems.

    All text the agent processes can pass through this pipeline.
    Novelty gating ensures only interesting inputs get the full
    enrichment treatment (prevents wasting ~26ms on "hello").
    """
    def enrich(self, text: str, *, context: EnrichmentContext) -> EnrichmentResult | None:
        """Enrich text with bio-system associations.

        Returns None if the novelty gate rejects the input (familiar,
        low salience). Returns full EnrichmentResult otherwise.

        Latency budget: < 50ms total (no LLM call).
        """
```

### Data Flow

```
Any text source
         ↓
[Novelty Gate] — EC pattern_separate_or_complete + ThalamicGate principles
  │  familiar (similarity > threshold) → None (no enrichment)
  │  novel (new pattern) → proceed
  ↓
[Bio-System Queries] — parallel, ~26ms total
  ├── EC.find_semantic(text) → substrate node IDs (~5ms)
  ├── Hippocampus.retrieve_on_cue(nodes, include_valence=True) → episodes (~10ms)
  ├── NAc.get_links_for_event(keywords) → causal predictions (~3ms)
  ├── ATL.recall_associated(concept_ids) → related concepts (~5ms)
  └── ComponentIndex.find_similar(text) → available affordances (~5ms)
         ↓
[Format] — compose EnrichmentResult
         ↓
[Deliver] — route to appropriate consumer
```

### Consumers and Delivery

| Source | Trigger | Delivery | Gate threshold |
|--------|---------|----------|----------------|
| `think` tool | Every call | ToolOutput (LLM sees as "thought response") | None (always enrich) |
| User percept (CLI/audio) | New text percept | Injected into StructuredContext for next prompt | High (novelty > 0.6) |
| Scene description (orchestrator) | Scene change | Injected into prompt context | Medium (novelty > 0.4) |
| Internet search results | Search returns | Concepts stored in ATL for future recall | Low (always index) |
| Imagination (novel entity) | Entity designed | Concepts feed into thought enrichment on next think | Medium |

### Relationship to Existing ThalamicGate

The Default Network's `ThalamicGate` (`default_network/gate.py`) already implements novelty + salience gating for visual/sensor percepts. It decides what reaches the deliberative layer.

**Reuse, don't rebuild.** The bio-enrichment pipeline's novelty gate should use the same scoring principles:
- `ThalamicGate.evaluate()` computes novelty × salience combined score
- We need the same for text: `EC.pattern_separate_or_complete` gives novelty; salience comes from goal-keyword overlap + NAc reward history

**Refactoring opportunity:** Extract the gating logic from `ThalamicGate` into a reusable `NoveltyScorer` that both the DN gate (for percepts) and the enrichment pipeline (for text) can use. The DN gate adds DN-specific behavior (attention locks, fear factor, adaptive thresholds); the enrichment pipeline just needs the core novelty × salience score.

This avoids duplicating the gating logic and ensures both systems evolve together.

## Stages

### L0 — BioEnrichmentPipeline core (~100 LOC) — SHIPPED (2026-04-20)

**New file:** `integration/bio_enrichment.py`

```python
class BioEnrichmentPipeline:
    def __init__(
        self,
        *,
        scorer: TextSalienceScorer,          # from runtime/gating.py (G1)
        hippocampus: Hippocampus | None = None,
        nac: NAc | None = None,
        atl: ATL | None = None,
        ec: EC | None = None,
        component_index: ComponentIndex | None = None,
        novelty_threshold: float = 0.4,
    ): ...

    def enrich(self, text: str, *, context: EnrichmentContext) -> EnrichmentResult | None:
        """Core pipeline. Uses TextSalienceScorer (G1) for novelty gate.
        Returns None if below novelty threshold."""

    def _gate(self, text: str, context: EnrichmentContext) -> bool:
        """Novelty gate via TextSalienceScorer from gating_abstraction G1."""
        score = self._scorer.score(text, context.to_gating_context())
        return score.novelty >= self._novelty_threshold

    def _extract_keywords(self, text: str) -> list[str]: ...
    def _query_hippocampus(self, node_ids: list[str]) -> list[EpisodicSummary]: ...
    def _query_nac(self, keywords: list[str]) -> list[CausalPrediction]: ...
    def _query_atl(self, concept_ids: list[str]) -> list[ConceptLink]: ...
    def _query_component_index(self, text: str) -> list[str]: ...
    def _compute_valence(self, memories, predictions) -> float: ...
```

**EnrichmentContext:** Carries current goal, recent thoughts (for convergence detection), active entity names — so the pipeline can prioritize relevant associations.

**Data types:**
```python
@dataclass(frozen=True)
class EpisodicSummary:
    memory_id: str
    summary: str          # one-line description of the episode
    valence: float        # how that episode felt (-1 to +1)
    relevance: float      # activation score (0-1)

@dataclass(frozen=True)
class CausalPrediction:
    event: str            # what was tried
    outcome: str          # what happened
    confidence: float     # how reliable
    valence: str          # "positive" / "negative" / "neutral"

@dataclass(frozen=True)
class ConceptLink:
    concept: str          # related concept name
    relationship: str     # how it relates ("similar", "causes", "part_of")
    activation: float     # spreading activation strength
```

### L1 — Passive enrichment: ThinkTool + percepts (~80 LOC) — SHIPPED (2026-04-20)

**Modified files:**
- `tools/narrative.py` — ThinkTool.execute calls BioEnrichmentPipeline, returns enriched ToolOutput
- `runtime/agent_loop.py` — On novel text percepts, run enrichment and inject result into StructuredContext
- Mode definitions — Set `think` followup_type to `"process"` so enriched response triggers LLM follow-up

**ThinkTool enriched flow:**
```python
class ThinkTool(Tool):
    def execute(self, **kwargs):
        thought = kwargs.get("thought", "")
        # Query bio-systems
        result = self._pipeline.enrich(thought, context=self._context)
        if result is None:
            return ToolOutput(success=True, output={"thought": thought, "visible": False})
        # Format enriched response
        response = self._format_thought_response(thought, result)
        return ToolOutput(
            success=True,
            output={"thought": thought, "response": response, "visible": False},
        )
```

**Thought response format** (what the LLM sees as the think tool's "result"):
```
Your experience suggests:
- Memory: You once forced open a rusty lock successfully (confidence: high)
- Prediction: Force on degraded metal → success (70% confidence)
- Caution: Past contact with corroded metal caused minor pain
- Related concepts: degradation, brittleness, structural weakness
- Available: rusty_gate_force_open, rusty_gate_examine_hinge
```

**Percept enrichment** (novel text inputs get bio-context):
- In `agent_loop.py`, after a text percept is received, run `pipeline.enrich(percept_text, ...)`
- If result is not None, add to `StructuredContext.bio_associations` (new field)
- PromptBuilder renders bio_associations as a context section at GUIDANCE priority

### L2 — Active deliberation: iterative think loop (~150 LOC) — SHIPPED (2026-04-20)

**Mechanism:** Uses the existing `ActionFollowup` system. No new loop architecture.

**Flow:**
1. Agent calls `think("how do I get past this gate?")`
2. ThinkTool returns enriched response (L1)
3. Followup type `"process"` → LLM sees result, can call think again
4. Agent thinks again with refined query: `think("force seems promising, what about the hinges?")`
5. New enrichment, new response
6. After N hops or convergence → LLM acts (calls an action tool instead of think)

**Convergence detection:**
- Track last 3 thought keywords in `EnrichmentContext.recent_thoughts`
- EC pattern separation between current thought and previous: if similarity > 0.8, inject convergence signal
- Format: "You've been thinking about similar things. You have enough context to act."

**Hard cap:** After 3 consecutive think calls without an action tool call, the think response includes "Time to act — you've thought enough about this." (configurable via `max_deliberation_hops`)

**NAc gating (bio-plausible termination):**
- Each think response carries the computed `valence` from the enrichment
- Declining novelty in responses → declining NAc reward signal
- This naturally reduces the "reward" of further thinking, biasing the agent toward action
- The NAc signal doesn't hard-block thinking — it just makes action tools relatively more attractive in the next proposal

### L3 — Refactor NoveltyScorer from ThalamicGate (optional, ~60 LOC) — DEFERRED

**Extract reusable gating logic:**
```python
# default_network/novelty.py (new, extracted from gate.py)
class NoveltyScorer:
    """Compute novelty × salience score for any input.

    Reused by ThalamicGate (percepts) and BioEnrichmentPipeline (text).
    """
    def score(self, novelty: float, salience: float, goal_relevance: float) -> float: ...
    def is_above_threshold(self, score: float, threshold: float) -> bool: ...
```

- `ThalamicGate` delegates to `NoveltyScorer` for the core computation
- `BioEnrichmentPipeline` uses the same scorer with EC-derived novelty + goal-keyword salience
- Single source of truth for "is this interesting enough to process?"

**Gate refactoring is optional** — the enrichment pipeline can inline a simpler novelty check (just EC pattern separation) for L0-L2. Extract into shared NoveltyScorer in L3 if we see the gating logic diverging or duplicating.

## Interaction with Existing Systems

**Acting Coach:** L1's thought enrichment and the Acting Coach are complementary, not competing. The coach provides *directives* ("explore, vary your approach"). Enrichment provides *context* ("here's what you know about this"). The coach can reference enrichment results: "Your experience suggests force works here — try it with varying intensity."

**SEM Tool Discovery:** Enrichment surfaces affordance names (via ComponentIndex). These are informational — they tell the agent what exists. To activate them, the agent still calls `discover_tools`. The pipeline does NOT auto-activate tools (that would bypass the I3 cap).

**Imagination:** When ImaginationTrigger fires on a novel noun phrase, the enrichment pipeline is likely already returning "novel=True" for the same text. The two systems share the novelty signal but diverge at the response: imagination creates new entities, enrichment contextualizes existing experience.

**MemoryHub:** The pipeline queries through MemoryHub's cross-layer activation when available (`recall_with_knowledge`), falling back to individual system queries when MemoryHub isn't wired.

## Design Decisions (from parallel review)

1. **Think always enriched, percepts gated.** Think is an explicit deliberation request — always give a response. Percepts are high-frequency — only enrich novel ones.

2. **Tool output, not prompt section.** Think enrichment arrives as ToolOutput (agent sees it as "what the system told me"). Percept enrichment arrives as StructuredContext (background coloring, not conversational). Different sources, different delivery.

3. **26ms budget, no LLM call in the core path.** Bio-system queries only. If the agent needs deeper reasoning, it thinks again (L2 multi-hop), which triggers another ~26ms enrichment pass. The LLM call is the agent's own next turn, not a hidden side-channel.

4. **NAc-gated termination + hard cap.** Declining novelty naturally biases toward action. Hard cap at 3 hops catches pathological loops. The agent is never blocked from thinking — just nudged.

5. **Source-agnostic pipeline, per-consumer wrappers.** The core `enrich()` call is identical regardless of whether the text came from a thought, a user, or a search result. Only the delivery mechanism differs.

## LOC Estimate

| Stage | LOC | Notes |
|-------|-----|-------|
| L0 | ~100 | BioEnrichmentPipeline + data types + keyword extraction |
| L1 | ~80 | ThinkTool enrichment + percept wiring + StructuredContext field |
| L2 | ~150 | Deliberation loop via followup + convergence detection + NAc gating |
| L3 | ~60 | NoveltyScorer extraction from ThalamicGate (optional) |
| **Total** | **~390** | |

## Open Questions

- **EC.find_semantic availability:** This is a Phase 4 feature. Is it shipped? If not, fall back to keyword-based hippocampus recall + ComponentIndex embedding search. Verify before implementing L0.
- **StructuredContext field for enrichment:** Adding a new field requires touching the prompt builder. Check whether `bio_associations` or similar already exists (per feedback_structured_context_reuse.md).
- **Followup type for think:** Need to verify `think` can have followup_type `"process"` set in mode definitions without breaking non-embodiment modes where think should be lightweight.
- **Cross-session enrichment:** The pipeline queries current-session bio-systems. With persistence (hippocampus loaded from disk), prior-session memories inform current-session thoughts. This is the desired behavior — verify it works with the load path.

## Success Metrics

1. **Think response quality:** Agent's second thought (after enrichment) is more specific and actionable than its first
2. **Time to first novel affordance use:** With enrichment, agent discovers and uses non-obvious tools faster
3. **Deliberation depth:** Agent uses 2-3 think hops before acting in complex scenarios, converging on a plan
4. **Exploration diversity:** Agent explores more varied affordances per session vs. SEM-discovery-only baseline
5. **No think loops:** Agent never exceeds 3 consecutive thinks without an action (hard cap works)
