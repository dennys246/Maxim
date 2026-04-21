# Bio-System Protocol Enrichment Plan

**Status:** Shell (needs session to audit + design)
**Scope:** Pre-1.0 — future-proof the bio-system interfaces for richer backends
**Priority:** Medium-high (interface breaks are expensive post-1.0; cheap to add now)
**Depends on:** Nothing blocking. Independent of P5.

---

## Problem

The bio-system Protocol surfaces (Hippocampus, NAc, ATL, EC, SCN) have minimal call signatures designed for the v1 implementations. Post-1.0, richer backends (real TD learning, embedding-space retrieval, temporal context vectors) will need richer input — but by then the interfaces are frozen because downstream code depends on them.

**Example — Hippocampus today:**
```python
def search_by_content(self, text: str, limit: int = 5) -> list[EpisodicMemory]: ...
```

**What a biologically-refined backend would need:**
```python
def search_by_content(
    self, text: str, limit: int = 5, *,
    context: RetrievalContext | None = None,  # temporal, emotional, goal state
) -> list[EpisodicMemory]: ...
```

Without the `context` parameter on the Protocol, adding it later is a breaking change across all callers.

## Goal

Add a standardized `*Context` dataclass parameter to each bio-system's primary methods NOW (optional, defaults to None) so future implementations can consume richer input without Protocol breaks. Current implementations ignore it; future ones use it.

---

## Audit Checklist (fill out per system)

For each bio-system, identify:
1. The primary Protocol/ABC methods (the public interface callers depend on)
2. Current call signature
3. What richer backends would need as input (even if speculative)
4. Proposed `*Context` dataclass fields

### Hippocampus

**Primary methods to enrich:**
- [ ] `search_by_content(text, limit)` — the main retrieval path
- [ ] `store(episode)` — encoding path (could accept encoding hints)
- [ ] `retrieve_on_cue(...)` — if this exists as a separate method
- [ ] Others: _____

**What richer backends might need:**
- Temporal context (when did this happen relative to now?)
- Emotional state at retrieval time (mood-congruent recall)
- Goal context (task-relevant filtering at the retrieval level, not post-hoc)
- Consolidation level filter (only well-consolidated memories? or include fragile recent?)
- Encoding specificity (how literal vs. gist-based should matching be?)

**Proposed `RetrievalContext`:**
```python
@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """Optional context for hippocampal retrieval. Future backends use these
    to implement biologically-informed retrieval dynamics."""
    # Fill in after audit
```

---

### NAc (Nucleus Accumbens)

**Primary methods to enrich:**
- [ ] `get_links_for_event(event_signature)` — prediction lookup
- [ ] `record_outcome(event_id, valence, ...)` — learning
- [ ] `record_outcome_full(context_similarity path)` — attribution
- [ ] Others: _____

**What richer backends might need:**
- Current arousal/energy state (modulates learning rate)
- Temporal discount (how far in the future is the predicted outcome?)
- Context richness (multi-dimensional context for TD learning, not just keyword match)
- Reward prediction error signal (surprise = learning rate boost)
- Eligibility trace window (how far back to attribute?)

**Proposed `PredictionContext`:**
```python
@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Optional context for NAc prediction queries."""
    # Fill in after audit
```

---

### ATL (Anterior Temporal Lobe)

**Primary methods to enrich:**
- [ ] `recall(limit, name=...)` — concept lookup
- [ ] `store_concept(...)` — concept formation
- [ ] Others: _____

**What richer backends might need:**
- Activation spreading parameters (depth, decay rate)
- Category constraints (only look in certain semantic domains)
- Abstraction level (concrete entity vs. abstract category)
- Cross-modal binding hints (concept has visual + semantic + motor associations)

**Proposed `ConceptQueryContext`:**
```python
@dataclass(frozen=True, slots=True)
class ConceptQueryContext:
    """Optional context for ATL concept queries."""
    # Fill in after audit
```

---

### EC (Entorhinal Cortex)

**Primary methods to enrich:**
- [ ] `pattern_complete_or_separate(...)` — the core novelty/recognition call
- [ ] `find_semantic(text)` — embedding lookup
- [ ] Others: _____

**What richer backends might need:**
- Grid cell resolution (coarse vs. fine-grained similarity)
- Temporal phase (are we in encoding mode or retrieval mode?)
- Attention weighting (which dimensions of the embedding matter most right now?)

**Proposed `PatternContext`:**
```python
@dataclass(frozen=True, slots=True)
class PatternContext:
    """Optional context for EC pattern operations."""
    # Fill in after audit
```

---

### SCN (Suprachiasmatic Nucleus)

**Primary methods to enrich:**
- [ ] `get_energy()` / `current_phase()` — circadian state
- [ ] Others: _____

**What richer backends might need:**
- External zeitgeber signals (light exposure, activity patterns)
- Ultradian rhythm coupling (90-min attention cycles)
- Homeostatic sleep pressure accumulation

**Proposed:** Likely minimal — SCN is already simple and its interface is mostly read-only. May not need enrichment.

---

## Design Principles

1. **Optional, never required.** Every `*Context` parameter defaults to `None`. Current code doesn't break. Current implementations can ignore it entirely.

2. **Frozen dataclass, slots=True.** Immutable, lightweight, no footguns.

3. **One context type per system, not per method.** `RetrievalContext` works for all hippocampal queries. Don't proliferate types.

4. **Fields are hints, not commands.** A backend that doesn't support temporal-context retrieval just ignores the `temporal_window` field. The caller can't assume the backend used it.

5. **No new dependencies.** Context dataclasses live alongside the system they serve (e.g., `RetrievalContext` in `memory/hippocampus.py` or a shared `memory/protocols.py`).

6. **Document expected semantics per field.** Even if v1 ignores them, write the docstring explaining what a conforming backend SHOULD do with the field. This is the contract for post-1.0 work.

---

## Implementation Plan (fill out after audit)

| Stage | What | LOC estimate |
|-------|------|-------------|
| A1 | Audit all Protocol surfaces — list every public method + current callers | — |
| A2 | Design context dataclasses (one per system) | ~50 |
| B1 | Add `context=None` parameter to Hippocampus Protocol + implementation | ~20 |
| B2 | Add `context=None` parameter to NAc Protocol + implementation | ~20 |
| B3 | Add `context=None` parameter to ATL Protocol + implementation | ~20 |
| B4 | Add `context=None` parameter to EC Protocol + implementation | ~20 |
| B5 | SCN (assess if needed) | ~5 |
| C1 | Update callers that already have context (BioEnrichmentPipeline, MemoryHub) to pass it | ~30 |
| C2 | Tests: verify None default works, verify context flows when provided | ~40 |

**Estimated total:** ~200 LOC (mostly dataclass definitions + parameter additions)

---

## Open Questions (resolve during audit session)

- Where do the context dataclasses live? Options: per-system file, shared `memory/protocols.py`, or new `bio_protocols.py`
- Should there be a base `BioQueryContext` that all system-specific contexts extend? (Shared fields: `energy`, `active_goal`, `timestamp`)
- Should callers populate context from a shared source (e.g., a `BioContextProvider` that reads SCN energy + current goal + recent percepts) or assemble ad-hoc?
- Which methods genuinely benefit vs. which are too simple to warrant it? (e.g., `SCN.get_energy()` probably doesn't need context)
- Should existing internal callers (MemoryHub, BioEnrichmentPipeline) start passing context immediately, or just add the parameter and wire later?
