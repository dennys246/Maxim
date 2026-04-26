# Bio-System Protocol Enrichment Plan

**Status:** COMPLETE (2026-04-26)
**Scope:** Pre-1.0 — future-proof the bio-system interfaces for richer backends
**Priority:** Medium-high (interface breaks are expensive post-1.0; cheap to add now)
**Depends on:** Nothing blocking. Independent of P5.
**Branch:** `feat/v1-bio-protocol-enrichment`

---

## Problem

The bio-system Protocol surfaces (Hippocampus, NAc, ATL, EC, SCN) have minimal call signatures designed for the v1 implementations. Post-1.0, richer backends (real TD learning, embedding-space retrieval, temporal context vectors) will need richer input — but by then the interfaces are frozen because downstream code depends on them.

## Solution

Added standardized `*Context` dataclass parameters to each bio-system's primary methods (optional, defaults to None). Current implementations ignore the context; future ones use it. All five context types live in a single import site: `src/maxim/models/bio_context.py`.

---

## Audit Checklist (completed)

### Hippocampus — `RetrievalContext`

**Methods enriched:**
- [x] `recall(limit, *, ..., retrieval_context=None)` — via `hippocampus_retrieval.py` mixin
- [x] `search_by_content(query, limit, *, retrieval_context=None)` — hippocampus.py
- [x] `retrieve_on_cue(cue_node_id, ..., *, retrieval_context=None)` — hippocampus.py (3 overloads)

**`RetrievalContext` fields:**
- `temporal_window_s` — restrict retrieval to recent memories
- `emotional_valence` — mood-congruent recall bias
- `active_goal` — goal-directed relevance boost
- `min_consolidation` — consolidation-aware ranking
- `encoding_specificity` — gist vs. literal matching
- `energy` — metabolic constraint on search scope

**MemoryLayer ABC** (`memory/layer.py`) — `recall()` signature updated with `retrieval_context` parameter.

---

### NAc (Nucleus Accumbens) — `PredictionContext`

**Methods enriched:**
- [x] `get_links_for_event(event_signature, prediction_context=None)` — nac.py
- [x] `predict(event_type, event_signature, context, prediction_context=None)` — nac.py
- [x] `predict_all_outcomes(event_type, event_signature, context, prediction_context=None)` — nac.py
- [x] `record_outcome(event_type, event_id, outcome_valence, context, memory_id, prediction_context=None)` — nac.py
- [x] `record_outcome_full(outcome_type, ..., prediction_context=None)` — nac.py

**Note:** Named `prediction_context` (not `context`) because these methods already have `context: dict[str, Any] | None` for causal attribution context similarity.

**`PredictionContext` fields:**
- `arousal` — learning rate modulation
- `temporal_discount` — discount for distant outcomes
- `energy` — reward sensitivity modulation
- `active_goal` — goal-conditioned prediction
- `surprise` — prediction error → learning rate boost

---

### ATL (Anterior Temporal Lobe) — `SemanticContext`

**Methods enriched:**
- [x] `recall(limit, *, semantic_context=None, **filters)` — atl.py
- [x] `find_or_create(name, category, ..., semantic_context=None)` — atl.py

**`SemanticContext` fields:**
- `abstraction_level` — concrete vs. abstract category preference
- `domain_hints` — semantic domain constraints (tuple of strings)
- `spreading_depth` — max hops for spreading activation
- `spreading_decay` — activation decay per hop
- `active_goal` — goal-relevant concept boost

---

### EC (Entorhinal Cortex) — `EncodingContext`

**Methods enriched:**
- [x] `pattern_complete_or_separate(embedding, modality, ..., encoding_context=None)` — ec.py
- [x] `find_semantic(query, k, threshold, encoding_context=None)` — ec.py

**`EncodingContext` fields:**
- `resolution` — grid cell resolution (coarse vs. fine)
- `attention_weights` — per-dimension embedding weights
- `encoding_mode` — "encode" vs. "retrieve" hint
- `decomposition_hint` — concept decomposition strategy hint

---

### SCN (Suprachiasmatic Nucleus) — `TemporalContext`

**Methods enriched:**
- [x] `register(memory_id, signature, significance, temporal_context=None)` — scn.py
- [x] `get_threshold_adjustment(signature, temporal_context=None)` — scn.py

**`TemporalContext` fields:**
- `oscillator_phase` — current dominant oscillator phase
- `prediction_confidence` — temporal prediction confidence
- `zeitgeber_strength` — external zeitgeber signal strength
- `energy` — energy level for circadian anomaly detection

---

## Design Principles (applied)

1. **Optional, never required.** Every `*Context` parameter defaults to `None`. Zero existing callers break.
2. **Frozen dataclass, slots=True.** Immutable, lightweight, no footguns.
3. **One context type per system, not per method.** `RetrievalContext` for all hippocampal queries, etc.
4. **Fields are hints, not commands.** Current v1 implementations ignore all context fields.
5. **Single import site.** All five types in `src/maxim/models/bio_context.py`.
6. **Documented semantics per field.** Docstrings explain what a conforming backend SHOULD do.

---

## Files Changed

| File | Change |
|------|--------|
| `src/maxim/models/bio_context.py` | NEW — 5 frozen dataclasses |
| `src/maxim/memory/layer.py` | `recall()` ABC gets `retrieval_context` param |
| `src/maxim/memory/hippocampus.py` | `search_by_content`, `retrieve_on_cue` (3 overloads) |
| `src/maxim/memory/hippocampus_retrieval.py` | `recall()` mixin |
| `src/maxim/decisions/nac.py` | 5 methods get `prediction_context` param |
| `src/maxim/memory/atl.py` | `recall`, `find_or_create` |
| `src/maxim/similarity/ec.py` | `pattern_complete_or_separate`, `find_semantic` |
| `src/maxim/time/scn.py` | `register`, `get_threshold_adjustment` |
| `tests/unit/test_bio_context.py` | NEW — 40 tests (dataclass invariants + acceptance per system) |

## Test Results

40/40 new tests passing. Full suite (~5966 tests) clean (1 pre-existing flaky substrate test unrelated to this change).
