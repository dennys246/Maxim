# Substrate — Concept decomposition (noun-phrase extraction before EC)

**Status:** Stage 1 COMPLETE (shipped `723dbee` 2026-04-16, validated 2026-04-17). Stages 2+3 SHIPPED in 0.4.
**Scope:** ~400–600 LOC (protocol + spaCy strategy + encoder integration + tests). New optional dep: `spacy` (MIT license).
**Target version:** post-0.3. Ships AFTER P4 Stage 3 proves the base cross-modal claim on bare class names. **P4 Stage 3 PASSED (2026-04-16) — trigger fired.**
**Parent:** None (standalone). Extends `similarity/encoder.py` → `similarity/ec.py` capture path.
**Blocks:** Nothing. Improves quality of all downstream substrate phases (P5 stress, P6 multi-session, P8 sleep-replay) by generating more meaningful nodes.
**Companion:** [substrate_valence_annotation.md](archive/substrate_valence_annotation.md) — SEM reactions annotate Hebbian edges with valence (pain/pleasure). Recommended after concept decomposition Stage 1.

## Motivation

Today, `LinguisticEncoder` encodes an entire input string — whether that's `"mug"` or `"I see a blue mug on the table next to the red plate"` — as a single substrate node. This creates three problems:

1. **Cross-modal binding fragility.** A full sentence `"I see a blue mug"` won't reliably pattern-complete against a bare `"mug"` node from a previous session. The Hebbian edge from `"mug"` to `vision-mug` becomes unreachable from the sentence node, silently breaking cross-modal retrieval for naturalistic inputs. P4's mug test sidesteps this by using bare class names, but production usage won't.

2. **Concept granularity mismatch.** The hippocampus binds nodes that co-occur in episodes. A single-node sentence means all the concepts in that sentence are fused into one opaque blob. "The cat sat on the mat" should produce two bindable concepts (`cat`, `mat`) that can independently form cross-session associations — not one sentence-level embedding that can't be decomposed later.

3. **Hebbian edge explosion for multi-word inputs is avoided, not caused.** Without decomposition, long inputs create fewer but less useful nodes. With decomposition at the noun-phrase level (typically 2–4 chunks per sentence), the edge count stays manageable while each node represents a real concept.

4. **Pain association granularity (discovered in review).** NAc's `_reward_bias` maps substrate node IDs to reward values, and `update_eligibility` fires when a percept completes to a node. But `distribute_reward` has zero external callers — pain signals never flow back to the substrate nodes that were active during painful episodes. When concepts are finer-grained ("rusty sword" as its own node vs a whole sentence), they become better targets for pain association — but the association mechanism itself needs wiring. See [companion shell](#companion-shell-pain-concept-bridge).

## Design

### Protocol-based decomposition (three-lens review fold)

The decomposer is defined as a **Protocol**, not a concrete class. This allows swapping spaCy for LLM-based extraction, regex-based domain parsers, or custom strategies without touching the pipeline.

```python
# similarity/decomposer.py

@dataclass(frozen=True)
class ConceptChunk:
    """A single concept extracted from an input string."""
    text: str
    span: tuple[int, int] | None = None  # character offsets in the original
    confidence: float = 1.0
    relation: str | None = None  # Stage 2: "spatial", "possessive", etc.

class DecompositionStrategy(Protocol):
    """Protocol for concept extraction backends."""
    def extract(self, text: str) -> list[ConceptChunk]: ...

class SpaCyNounChunkStrategy:
    """Default strategy using spaCy noun chunker."""
    def extract(self, text: str) -> list[ConceptChunk]: ...

class IdentityStrategy:
    """Fallback: returns the input as a single chunk."""
    def extract(self, text: str) -> list[ConceptChunk]:
        return [ConceptChunk(text=text)]

class ConceptDecomposer:
    """Coordinator that delegates to a DecompositionStrategy."""
    def __init__(self, strategy: DecompositionStrategy | None = None, enabled: bool = True):
        if not enabled:
            self._strategy = IdentityStrategy()
        elif strategy is not None:
            self._strategy = strategy
        else:
            # Auto-detect: spaCy if available, else identity
            self._strategy = _auto_detect_strategy()

    def extract(self, text: str) -> list[ConceptChunk]: ...
```

**Key design decisions from the review:**

- **`ConceptChunk` from day one, not `str`.** Stage 2 role-tagged edges need span offsets and relation types. Defining the rich return type now avoids a breaking interface change later. Downstream uses `chunk.text` — zero behavior change for Stage 1.
- **`enabled: bool` on config.** Operators who have spaCy as a transitive dep can disable decomposition without uninstalling it.
- **Singleton spaCy model with `threading.Lock`.** Under `AgentPool` concurrency, two threads could race on `spacy.load()`. Use a lock (not bare module-level assignment).

### What gets extracted

**Noun phrases** (the primary payload) via spaCy's noun chunker or a lightweight dependency parse. From `"I see a blue mug on the table next to the red plate"`:

- `"blue mug"` → one substrate node
- `"table"` → one substrate node
- `"red plate"` → one substrate node

**Not extracted** (and why):

- **Pronouns** (`"he"`, `"it"`, `"they"`): near-meaningless embeddings. Would create massively connected hub nodes that Hebbian-bind to everything. Noise that drowns signal.
- **Determiners/prepositions** (`"the"`, `"on"`, `"next to"`): function words don't carry concept-level meaning.
- **Bare verbs** (`"see"`, `"sat"`): ambiguous without their arguments. The useful unit is the verb-object phrase, which the noun chunker captures when the object is present.

### Modality gate (embodiment review fold)

Decomposition applies to **text-modality percepts only.** Visual percepts carry CLIP labels (`"mug"`), proprioceptive/interoceptive percepts carry structured readings, and SEM affordance strings have domain-specific structure — none should be noun-chunked.

The gate is enforced structurally at the encoder level:

```python
# In LinguisticEncoder.encode() or encode_decomposed():
if modality != SubstrateModality.TEXT:
    return [original_node_id]  # no decomposition
chunks = self._decomposer.extract(text)
```

This is the same layer where `MAXIM_SUBSTRATE_PATH=1` is checked — decomposition adds zero cost when disabled or when the percept is non-textual.

### Integration point (corrected per three-lens review)

The plan originally placed decomposition in `_capture_episodic` in the agent loop. **All three review lenses flagged this as wrong:**

- The actual substrate encoding path is `LinguisticEncoder.encode()` called from `memory_hub._on_new_memory()` (gated on `MAXIM_SUBSTRATE_PATH=1`)
- Simulations and substrate fixtures create `CaptureEvent`s directly, bypassing the agent loop
- Headless API, `AgentPool`, and embodied runtime each have separate integration paths

**Corrected integration:** decomposition lives inside `LinguisticEncoder` as an optional pre-processing step. A new method `encode_decomposed(text, modality) -> list[str]` returns multiple node IDs when decomposition is active. All consumers that go through the encoder get decomposition for free.

```
Input string
    │
    ▼
┌─────────────────────────────────────────┐
│  LinguisticEncoder.encode_decomposed()  │  MODIFIED: similarity/encoder.py
│                                         │
│  1. modality gate (text only)           │
│  2. ConceptDecomposer.extract(text)     │
│     → list[ConceptChunk]               │
│  3. for each chunk:                     │
│     embed(chunk.text) → embedding       │
│     EC.pattern_complete_or_separate()   │
│     → node_id                           │
│  4. return list[node_id]                │
└──────────────┬──────────────────────────┘
               │  (all node IDs from one input
               │   land in the same episode)
               ▼
┌─────────────────────────────────────────┐
│  Hippocampus episode binding (Hebbian)  │  UNCHANGED
└─────────────────────────────────────────┘
```

**Key property:** nothing below `encode_decomposed` changes. EC, Hippocampus, Hebbian binding, cross-modal retrieval, NAc reward modulation, persistence — all stay the same.

### Stage 2 extension: role-tagged edges

Rather than building parallel graphs for different syntactic roles, annotate the Hebbian edges between extracted concept nodes with a lightweight relation tag:

- `"blue mug"` ↔ `"table"` with `relation="spatial"` (from "on")
- `"cat"` ↔ `"mat"` with `relation="spatial"` (from "on")

This stays in one graph, uses the same spreading activation, gets the same reward modulation. The tag is available as a future filter (like P3b's channel filter) but requires zero new retrieval infrastructure.

**This is an extension, not a requirement for Stage 1.** Stage 1 ships concept decomposition with untagged edges. Stage 2 adds relation tags if there's demonstrated value.

### Relationship to existing `ConceptExtractor`

`memory/concept_extractor.py` does something superficially similar — it extracts concepts from episodic memories and registers them in ATL. But it operates **after** capture (as a callback), uses token-level heuristics (not NLP parsing), and targets the ATL semantic layer (not the substrate binding graph). This plan operates **before** capture and targets the substrate node layer.

Stage 3 of this plan converges the two: `ConceptExtractor` adopts `ConceptDecomposer` as its noun-phrase backend, retiring the `_is_structured_goal` 4-token heuristic.

### NAc / pain path confirmation (embodiment review fold)

Decomposition creates more substrate nodes but does **not** touch pain context dicts or `ToolPainBridge` direct attribution (`(tool_name, invocation_id)` lookup). These are parallel namespaces. The `_context_similarity` directional denominator (P2 fix) and the `_pending_tools` guard are unaffected. This was explicitly verified during the three-lens review given P2's history with that bug class.

However, the review discovered a deeper gap: **pain signals never flow back to substrate nodes at all.** The wiring from SEM reactions to substrate edge valence is the subject of the companion plan [substrate_valence_annotation.md](archive/substrate_valence_annotation.md). Concept decomposition makes that plan more precise (finer-grained concept nodes = better valence targets).

### Dependency: spaCy

- License: MIT (confirmed)
- Size: `en_core_web_sm` model is ~12 MB (small pipeline, no word vectors — sufficient for noun chunking + dependency parse)
- Add as a new optional extra: `nlp` or fold into the existing `semantic` extra
- Lazy-load with `threading.Lock`: `import spacy` only when `ConceptDecomposer` is first called, not at module import. Follows the project pattern for optional deps. Log before loading: `logger.info("loading spaCy model...")`
- Fallback: when spaCy is not installed OR `enabled=False` in config, `IdentityStrategy` returns `[ConceptChunk(text=text)]` (the current behavior). No degradation in the base pipeline; decomposition is purely additive.

## Stages

### Stage 1 — Protocol + spaCy strategy + encoder integration (core) — COMPLETE

**Shipped:** `723dbee` (2026-04-16). **Validated:** 2026-04-17.

1. ✅ `similarity/decomposer.py` — `DecompositionStrategy` Protocol, `ConceptChunk` dataclass, `SpaCyNounChunkStrategy`, `IdentityStrategy`, `ConceptDecomposer` coordinator
2. ✅ `similarity/encoder.py` — `encode_decomposed(text, modality) -> list[str]` method with modality gate. `encode()` routes to it when decomposer is wired and modality is `"text"`.
3. ✅ `memory_hub._wire_substrate_encoder()` — constructs `ConceptDecomposer` when `MAXIM_CONCEPT_DECOMPOSITION=1`, passes to `LinguisticEncoder(decomposer=...)`. Gated via env var, with autouse conftest scrub fixture.
4. ✅ Config: `DecomposerConfig(enabled, min_chunk_len, spacy_model)` + env var `MAXIM_CONCEPT_DECOMPOSITION=1`
5. ✅ Unit tests: 28 tests in `tests/unit/test_concept_decomposer.py` — strategies, encoder integration, modality gate, thread safety, bare class name identity
6. ✅ **Validation fixture** (`tests/substrate/test_concept_decomposition_validation.py`):
   - 5 naturalistic scenes, 11 concept-level queries
   - Baseline (no decomposition): **36.4% concept-level cross-modal recall** (4/11)
   - Decomposed: **100.0% recall** (11/11)
   - **Delta: +63.6 pp**
   - Both pass criteria met: strict improvement + minimum bar (0.60)
   - Results: [docs/experiments/concept_decomposition_validation.md](../../docs/experiments/concept_decomposition_validation.md)
   - Reproduction: [docs/experiments/protocols/concept_decomposition_reproduction.md](../../docs/experiments/protocols/concept_decomposition_reproduction.md)
7. ✅ P4 regression: 21/21 mechanism tests pass — bare class names pass through as single chunks
8. ✅ P2 control: 5/5 mechanism tests pass — decomposer is neutral on single-phrase inputs

### Stage 2 — Role-tagged edges (extension, not blocking)

1. Extract relation type from dependency parse (`spatial`, `possessive`, `temporal`, `action`) between noun chunks — stored in `ConceptChunk.relation`
2. Add optional `relation: str | None` metadata on Hebbian edges (additive field on `EdgeType.ASSOCIATES` data)
3. `retrieve_on_cue` gains an optional `edge_filter` parameter (same pattern as `node_filter`)
4. Demonstrate: "where was the mug?" retrieves `table` via `relation="spatial"` while filtering out non-spatial associations

### Stage 3 — ConceptExtractor convergence (cleanup)

1. Migrate `ConceptExtractor` to use `ConceptDecomposer` as its noun-phrase backend (pass `SpaCyNounChunkStrategy` or the active strategy)
2. Retire the token-level heuristics in `_is_structured_goal` (the 4-token gate becomes unnecessary when real NLP parsing is available)
3. Unify the ATL registration path so ATL concepts and substrate nodes share the same concept identities

## When to execute

**P4 Stage 3 has PASSED (2026-04-16).** Trigger condition met. Concept decomposition is unblocked.

**Ideal moment:** between P4 (cross-modal binding proven) and P5 (stress testing). P5's stress test fixture uses naturalistic multi-word inputs and would directly benefit from concept decomposition. If decomposition ships before P5, the stress test exercises the decomposed path from day one rather than requiring a retrofit.

## Cross-references

- **[substrate_episode_boundary_enrichment.md](substrate_episode_boundary_enrichment.md):** concept decomposition creates more nodes per episode (2–4 per sentence instead of 1). Without enriched episode boundaries (tool execution, semantic shift, salience spike), a long conversation episode could accumulate dozens of noun-phrase nodes with O(n^2) Hebbian edges. The two plans are complementary — decomposition makes nodes finer-grained, boundary enrichment keeps episodes bounded so the edge count stays manageable.
- **NAc `distribute_reward` / `update_eligibility`:** `src/maxim/decisions/nac.py` — the reward bias infrastructure exists but is unwired. `src/maxim/similarity/encoder.py` line 193 calls `update_eligibility`. See companion shell above.

## Risks

1. **spaCy model quality on short fragments.** Noun chunking on 2–3 word inputs may over-decompose (`"blue mug"` → `["blue", "mug"]` instead of keeping the phrase). Mitigation: test on the P4 fixture's bare class names and common agent-loop inputs; tune the minimum chunk length.
2. **EC threshold calibration.** Individual noun phrases have different cosine-similarity distributions than full sentences. The existing EC thresholds (calibrated for sentence-level mpnet embeddings) may need adjustment. Mitigation: Stage 1 includes a sweep comparing EC collapse rates with and without decomposition.
3. **Hebbian edge inflation.** A 10-word sentence that decomposes into 4 noun phrases creates 6 pairwise edges instead of 0 (single node has no within-episode pairs). At scale this could slow spreading activation. Mitigation: 4 nodes x 6 edges is well within the binding graph's capacity; monitor during P5 stress test.
4. **Non-English inputs.** spaCy `en_core_web_sm` is English-only. Multi-language support requires either a multilingual model (`xx_ent_wiki_sm`) or per-language model selection. Deferred to a future stage.
5. **SEM affordance strings.** Modality gate (text-only) prevents decomposition of SEM strings, but if future work routes SEM text through the text modality, over-decomposition of structured affordance descriptions (e.g., `"the rusty sword feels heavy"` → `["rusty sword"]` + possibly `["heavy"]`) could occur. Document minimum-chunk-length floor as a tuning knob.

## Three-lens review findings (folded 2026-04-16)

**Architecture lens (2 CRITICAL, 3 IMPORTANT, 2 MINOR):**
- C1: ConceptDecomposer must be a Protocol (DecompositionStrategy). **FOLDED** — protocol + strategy pattern designed above.
- C2: Return type must be `ConceptChunk`, not `str`, from day one. **FOLDED** — dataclass with span, confidence, relation fields.
- I1: Integration point wrong (agent loop → encoder). **FOLDED** — moved to `LinguisticEncoder.encode_decomposed()`.
- I2: No `enabled` flag. **FOLDED** — `enabled: bool` on config.
- I3: Stage 3 (ConceptExtractor convergence) should be prerequisite, not follow-up. **PARTIALLY FOLDED** — kept as Stage 3 but noted the risk of coexistence. Can revisit if the review finding strengthens.
- M1: Singleton spaCy model needs `threading.Lock`. **FOLDED**.
- M2: `SensoryTag` inheritance for decomposed chunks not specified. **FOLDED** — all chunks inherit parent percept's `scn_tag`.

**Simulation lens (2 CRITICAL, 2 IMPORTANT, 1 MINOR):**
- C1: Simulations bypass `_capture_episodic` — decomposition would be silently skipped. **FOLDED** — integration moved to encoder layer.
- C2: P4 fixture regression test is vacuous (fixture bypasses agent loop). **FOLDED** — reframed as non-interference check, added real decomposition-specific fixture.
- I1: P2 re-run is a no-op, not a validation. **FOLDED** — reframed as control run.
- I2: `build_and_bind` needs a decomposer-aware encoding helper. **FOLDED** — `encode_decomposed` on the encoder serves this role.
- M1: spaCy model load latency on first sim turn. **FOLDED** — log line before loading.

**Embodiment lens (2 CRITICAL, 2 IMPORTANT, 2 MINOR):**
- C1: Integration point wrong (same as arch C1). **FOLDED**.
- C2: Modality gate missing — vision/proprioceptive/SEM percepts should not be decomposed. **FOLDED** — `if modality != TEXT` gate.
- I1: SEM affordance strings need minimum-chunk-length floor. **FOLDED** — documented as tuning knob in Risks.
- I2: NAc/pain path safe but should be documented + gap discovered. **FOLDED** — explicit confirmation + companion shell for the gap.
- M1: Reachy sensor streams safe with modality gate. **FOLDED**.
- M2: `CaptureEvent.activated_nodes` cardinality — `encode_decomposed` returns list. **FOLDED** — new method signature documented.
