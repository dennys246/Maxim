# Substrate — Concept decomposition (noun-phrase extraction before EC)

**Status:** SHELL (2026-04-15). Design only — no implementation scheduled.
**Scope:** ~300–500 LOC (extractor + integration + tests). New optional dep: `spacy` (MIT license).
**Target version:** post-0.3. Ships AFTER P4 Stage 3 proves the base cross-modal claim on bare class names.
**Parent:** None (standalone). Extends `similarity/encoder.py` → `similarity/ec.py` capture path.
**Blocks:** Nothing. Improves quality of all downstream substrate phases (P5 stress, P6 multi-session, P8 sleep-replay) by generating more meaningful nodes.

## Motivation

Today, `LinguisticEncoder` encodes an entire input string — whether that's `"mug"` or `"I see a blue mug on the table next to the red plate"` — as a single substrate node. This creates three problems:

1. **Cross-modal binding fragility.** A full sentence `"I see a blue mug"` won't reliably pattern-complete against a bare `"mug"` node from a previous session. The Hebbian edge from `"mug"` to `vision-mug` becomes unreachable from the sentence node, silently breaking cross-modal retrieval for naturalistic inputs. P4's mug test sidesteps this by using bare class names, but production usage won't.

2. **Concept granularity mismatch.** The hippocampus binds nodes that co-occur in episodes. A single-node sentence means all the concepts in that sentence are fused into one opaque blob. "The cat sat on the mat" should produce two bindable concepts (`cat`, `mat`) that can independently form cross-session associations — not one sentence-level embedding that can't be decomposed later.

3. **Hebbian edge explosion for multi-word inputs is avoided, not caused.** Without decomposition, long inputs create fewer but less useful nodes. With decomposition at the noun-phrase level (typically 2–4 chunks per sentence), the edge count stays manageable while each node represents a real concept.

## Design

### What gets extracted

**Noun phrases** (the primary payload) via spaCy's noun chunker or a lightweight dependency parse. From `"I see a blue mug on the table next to the red plate"`:

- `"blue mug"` → one substrate node
- `"table"` → one substrate node  
- `"red plate"` → one substrate node

**Not extracted** (and why):

- **Pronouns** (`"he"`, `"it"`, `"they"`): near-meaningless embeddings from sentence-transformers. Would create massively connected hub nodes that Hebbian-bind to everything. Noise that drowns signal.
- **Determiners/prepositions** (`"the"`, `"on"`, `"next to"`): same problem — function words don't carry concept-level meaning.
- **Bare verbs** (`"see"`, `"sat"`): ambiguous without their arguments. `"break"` alone doesn't distinguish breaking a mug from breaking a promise. The useful unit is the verb-object phrase, which the noun chunker already captures when the object is present.

### Stage 2 extension: role-tagged edges

Rather than building parallel graphs for different syntactic roles (verbs, subjects, objects), annotate the Hebbian edges between extracted concept nodes with a lightweight relation tag:

- `"blue mug"` ↔ `"table"` with `relation="spatial"` (from "on")
- `"cat"` ↔ `"mat"` with `relation="spatial"` (from "on")

This stays in one graph, uses the same spreading activation, gets the same reward modulation. The tag is available as a future filter (like P3b's channel filter) but requires zero new retrieval infrastructure.

**This is an extension, not a requirement for Stage 1.** Stage 1 ships concept decomposition with untagged edges. Stage 2 adds relation tags if there's demonstrated value.

### Architecture

```
Input string
    │
    ▼
┌─────────────────────────────┐
│  ConceptDecomposer          │  NEW module: similarity/decomposer.py
│  (spaCy noun-chunk extract) │
│                             │
│  "I see a blue mug on the   │
│   table next to the red     │
│   plate"                    │
│       │                     │
│       ▼                     │
│  ["blue mug", "table",      │
│   "red plate"]              │
└─────────────┬───────────────┘
              │  (one call per chunk)
              ▼
┌─────────────────────────────┐
│  LinguisticEncoder          │  EXISTING: similarity/encoder.py
│  (paraphrase-mpnet-base-v2) │  No changes — still encodes strings
│                             │
│  Returns 768-dim embedding  │
│  per chunk                  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  EC pattern_complete_or_    │  EXISTING: similarity/ec.py
│  separate()                 │  No changes — each chunk is a
│                             │  separate call, may complete
│  Node ID per chunk          │  against an existing concept
└─────────────┬───────────────┘
              │  (all node IDs from one input
              │   land in the same episode)
              ▼
┌─────────────────────────────┐
│  Hippocampus episode        │  EXISTING: memory/hippocampus.py
│  binding (Hebbian)          │  No changes — nodes co-activate
│                             │  in the same episode, get edges
└─────────────────────────────┘
```

The key property: **nothing below `ConceptDecomposer` changes.** EC, Hippocampus, Hebbian binding, cross-modal retrieval, NAc reward modulation, persistence — all stay the same. The decomposer is a pre-processing step that turns one string into N concept strings before they enter the existing pipeline.

### Integration point

The decomposer slots into the agent loop's capture path, between the raw percept and the `LinguisticEncoder` call. Specifically:

- `runtime/agent_loop.py::_capture_episodic()` currently passes the full transcript to the encoder
- After this plan, it passes the transcript to `ConceptDecomposer.extract()` first, then passes each chunk to the encoder independently
- All resulting node IDs are fed as `activated_nodes` in a single `CaptureEvent`, so they land in the same pending episode and get Hebbian-bound together

### Relationship to existing `ConceptExtractor`

`memory/concept_extractor.py` does something superficially similar — it extracts concepts from episodic memories and registers them in ATL. But it operates **after** capture (as a callback), uses token-level heuristics (not NLP parsing), and targets the ATL semantic layer (not the substrate binding graph). This plan operates **before** capture and targets the substrate node layer.

The two should eventually share a common noun-phrase extraction backend, but that unification is a post-ship cleanup, not a prerequisite. The decomposer should be a standalone module that `ConceptExtractor` can adopt later.

### Dependency: spaCy

- License: MIT (confirmed)
- Size: `en_core_web_sm` model is ~12 MB (small pipeline, no word vectors — sufficient for noun chunking + dependency parse)
- Add as a new optional extra: `nlp` or fold into the existing `semantic` extra
- Lazy-load: `import spacy` only when `ConceptDecomposer` is first called, not at module import. Follows the project pattern for optional deps.
- Fallback: when spaCy is not installed, `ConceptDecomposer.extract(text)` returns `[text]` (identity — the current behavior). No degradation in the base pipeline; decomposition is purely additive.

## Stages

### Stage 1 — Noun-phrase decomposition (core)

1. `similarity/decomposer.py` — `ConceptDecomposer` class with `extract(text: str) -> list[str]` using spaCy noun chunks
2. Integration into `_capture_episodic` (or the substrate encoding path, depending on where production integration lands by then)
3. Fallback when spaCy is not installed (return `[text]`)
4. Unit tests: decomposition quality on 20+ sentence fixtures (English only for Stage 1)
5. Regression test: P4 mug test still passes with decomposer enabled (bare class names like `"lotus"` should pass through as single-chunk identity)
6. Sweep: re-run P2 reward modulation 10-seed sweep with decomposer enabled, compare target gain and distractor drift against the P2 baseline (+56 pp / 0.0 pp). The decomposer should be neutral-to-positive — if it degrades the metric, investigate before shipping.

### Stage 2 — Role-tagged edges (extension, not blocking)

1. Extract relation type from dependency parse (`spatial`, `possessive`, `temporal`, `action`) between noun chunks
2. Add optional `relation: str | None` metadata on Hebbian edges (additive field on `EdgeType.ASSOCIATES` data)
3. `retrieve_on_cue` gains an optional `edge_filter` parameter (same pattern as `node_filter`)
4. Demonstrate: "where was the mug?" retrieves `table` via `relation="spatial"` while filtering out non-spatial associations

### Stage 3 — ConceptExtractor convergence (cleanup)

1. Migrate `ConceptExtractor` to use `ConceptDecomposer` as its noun-phrase backend
2. Retire the token-level heuristics in `_is_structured_goal` (the 4-token gate becomes unnecessary when real NLP parsing is available)
3. Unify the ATL registration path so ATL concepts and substrate nodes share the same concept identities

## When to execute

**Not before P4 Stage 3 ships.** The base cross-modal claim must be proven on bare concept names first. If hippocampus can't beat OpenCLIP on `"mug" → vision-mug`, adding richer text decomposition won't save the architecture — it'll just add complexity to a failing mechanism.

**Ideal moment:** between P4 (cross-modal binding proven) and P5 (stress testing). P5's stress test fixture uses naturalistic multi-word inputs and would directly benefit from concept decomposition. If decomposition ships before P5, the stress test exercises the decomposed path from day one rather than requiring a retrofit.

**Trigger condition:** P4 Stage 3 PASSES (Arm B beats Arm C by the margin criterion + bootstrap CI). If P4 fails, this plan is deferred indefinitely — the architecture has bigger problems.

## Cross-references

- **[substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md):** concept decomposition creates more nodes per episode (2–4 per sentence instead of 1). Without enriched episode boundaries (tool execution, semantic shift, salience spike), a long conversation episode could accumulate dozens of noun-phrase nodes with O(n^2) Hebbian edges. The two plans are complementary — decomposition makes nodes finer-grained, boundary enrichment keeps episodes bounded so the edge count stays manageable.

## Risks

1. **spaCy model quality on short fragments.** Noun chunking on 2–3 word inputs may over-decompose (`"blue mug"` → `["blue", "mug"]` instead of keeping the phrase). Mitigation: test on the P4 fixture's bare class names and common agent-loop inputs; tune the minimum chunk length.
2. **EC threshold calibration.** Individual noun phrases have different cosine-similarity distributions than full sentences. The existing EC thresholds (calibrated for sentence-level mpnet embeddings) may need adjustment. Mitigation: Stage 1 includes a sweep comparing EC collapse rates with and without decomposition.
3. **Hebbian edge inflation.** A 10-word sentence that decomposes into 4 noun phrases creates 6 pairwise edges instead of 0 (single node has no within-episode pairs). At scale this could slow spreading activation. Mitigation: 4 nodes × 6 edges is well within the binding graph's capacity; monitor during P5 stress test.
4. **Non-English inputs.** spaCy `en_core_web_sm` is English-only. Multi-language support requires either a multilingual model (`xx_ent_wiki_sm`) or per-language model selection. Deferred to a future stage.
