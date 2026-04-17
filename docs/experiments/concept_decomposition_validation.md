# Concept Decomposition — Cross-Modal Validation

**Date:** 2026-04-17
**Phase:** Concept Decomposition Stage 1 validation (post-implementation)
**Status:** recorded
**Code version:** `main` branch (Stage 1 shipped at `723dbee`)
**Decision:** Decomposition produces a **+63.6 pp improvement** in concept-level cross-modal recall (36.4% → 100.0%). Baseline partial credit comes from EC pattern-completion on semantically close sentence/concept pairs; the 3-concept scene (0/3 recall) demonstrates the failure mode decomposition eliminates. Stage 1 validated.

## Hypothesis

When `LinguisticEncoder` decomposes a naturalistic sentence (e.g., "I see a blue mug on the wooden table") into concept-level noun phrases ("blue mug", "wooden table") before encoding, each concept becomes its own substrate node. These fine-grained nodes can individually bind to vision nodes via Hebbian co-activation, enabling concept-level cross-modal retrieval that whole-sentence encoding cannot.

Without decomposition, a bare concept query ("blue mug") fails to retrieve the vision node because "blue mug" and "I see a blue mug on the wooden table" are different substrate nodes with no shared binding.

## Methodology

### Pipeline

```
Sentence → ConceptDecomposer.extract() → list[ConceptChunk]
         → for each chunk: LinguisticEncoder.embed(chunk.text)
         → EC.pattern_complete_or_separate(threshold=0.70)
         → ATL.activate_substrate_node
         → CaptureEvent(activated_nodes=all_chunk_node_ids, modality="text")
         → Hippocampus episode binding (Hebbian)
```

### Fixture

5 naturalistic scene descriptions, each containing 2-4 noun phrases:

| Scene | Concepts | Vision labels |
|-------|----------|---------------|
| "I see a blue mug on the wooden table" | blue mug, wooden table | mug, table |
| "The rusty sword lies next to the leather shield" | rusty sword, leather shield | sword, shield |
| "A red ball sits in the cardboard box near the glass window" | red ball, cardboard box, glass window | ball, box, window |
| "The old book rests on the marble shelf" | old book, marble shelf | book, shelf |
| "A golden key hangs from the iron chain" | golden key, iron chain | key, chain |

Total concept-level queries: 11 (across 5 scenes).

Vision embeddings use the same `paraphrase-mpnet-base-v2` model encoding the vision label text (e.g., "mug") as a proxy for real vision embeddings. This tests the cross-modal binding mechanism, not actual CLIP-style vision encoding (that's P4's domain).

### Two arms

**Baseline (no decomposition):** The full scene description is encoded as a single text node. That blob node is bound to all vision nodes in one episode. Then bare concept nodes are encoded separately and queried for cross-modal retrieval. The bare concept has no direct Hebbian binding to vision — it depends on EC pattern-completing the bare concept to the same node as the sentence.

**Decomposed:** The scene is encoded with `ConceptDecomposer` (spaCy noun chunker), producing 2-4 text nodes. All text and vision nodes co-activate in one episode. Bare concept queries pattern-complete to the decomposed concept node, which has a direct Hebbian binding to the paired vision node.

### Metric

**Concept-level cross-modal recall rate:** For each concept, query `hippocampus.retrieve_cross_modal(concept_node, target_modality="vision", limit=5)`. A hit is counted if the expected paired vision node appears in the top-5 results. Aggregate recall = total hits / total queries.

### Pass criteria

- `decomposed_recall > baseline_recall` (strict improvement)
- `decomposed_recall >= 0.60` (minimum quality bar)

## Results

| Arm | Recall | Hits | Queries | EC nodes |
|-----|--------|------|---------|----------|
| Baseline | 36.4% | 4/11 | 11 | 23 |
| Decomposed | 100.0% | 11/11 | 11 | 22 |
| **Delta** | **+63.6 pp** | | | |

### Per-scene breakdown

| Scene | Baseline | Decomposed | Notes |
|-------|----------|------------|-------|
| blue mug / wooden table | 1/2 (50%) | 2/2 (100%) | "blue mug" pattern-completes at baseline; "wooden table" doesn't |
| rusty sword / leather shield | 1/2 (50%) | 2/2 (100%) | "rusty sword" close to sentence; "leather shield" not |
| red ball / cardboard box / glass window | 0/3 (0%) | 3/3 (100%) | 3-concept sentence too diluted for any concept to pattern-complete |
| old book / marble shelf | 1/2 (50%) | 2/2 (100%) | "old book" close; "marble shelf" not |
| golden key / iron chain | 1/2 (50%) | 2/2 (100%) | "golden key" close; "iron chain" not |

### Key finding

The baseline gets partial credit (4/11) because `paraphrase-mpnet-base-v2` embeds some full sentences close enough to their most prominent concept that they pattern-complete to the same EC node. This works for the "lead concept" (the first or most salient noun phrase) but fails for secondary concepts ("wooden table", "leather shield", "iron chain"). The 3-concept scene ("red ball / cardboard box / glass window") is the critical failure: the sentence embedding is too far from any individual concept at threshold 0.70, so recall is 0/3.

Decomposition eliminates this failure mode entirely by giving each concept its own substrate node with a direct Hebbian binding to its paired vision node.

## Regression checks

- **P4 cross-modal mechanism tests:** 21/21 passed — decomposition does not affect the cross-modal binding pipeline
- **P2 reward modulation mechanism tests:** 5/5 passed — reward bias pathway unaffected
- **Decomposer unit tests:** 28/28 passed — including bare class name identity (single nouns pass through unchanged)

## Reproduction

See [protocols/concept_decomposition_reproduction.md](protocols/concept_decomposition_reproduction.md).

Raw results: [results/concept_decomposition_validation.json](results/concept_decomposition_validation.json).
