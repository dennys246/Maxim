# Concept Decomposition — Reproduction Protocol

**Status:** Active protocol
**Purpose:** Reproduce the concept decomposition validation result on fresh hardware or after any change to the decomposer, encoder, EC threshold logic, or Hippocampus binding.
**Expected runtime:** ~10s on an M3 Mac (including model cold-load); ~2 minutes including setup.

## Background

The concept decomposition validation demonstrated:
- Baseline (no decomposition): **36.4% concept-level cross-modal recall** (4/11 hits)
- Decomposed: **100.0% recall** (11/11 hits)
- Delta: **+63.6 pp**

Full methodology + result archive: [../concept_decomposition_validation.md](../concept_decomposition_validation.md). Raw numbers: [../results/concept_decomposition_validation.json](../results/concept_decomposition_validation.json).

## Prerequisites

**Code:**
- `main` at or after the concept decomposition Stage 1 merge (`723dbee`)
- `pymaxim` importable from the checkout (`PYTHONPATH=src` or editable install)

**Dependencies:**
- `sentence-transformers` (the `pymaxim[semantic]` extra)
- `spacy` with `en_core_web_sm` model
- ~500 MB disk for `paraphrase-mpnet-base-v2` model cache
- ~12 MB for spaCy `en_core_web_sm`

**Hardware:**
- Any CPU is sufficient. M3 Mac: ~7s wall clock. No GPU required.
- No network required after initial model downloads.

## Setup (one-time)

```bash
pip install 'pymaxim[semantic]'
pip install spacy
python -m spacy download en_core_web_sm
```

## Running the validation

### Full validation (mechanism + real-embedding)

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_concept_decomposition_validation.py -xvs
```

### Mechanism tests only (fast, no optional deps)

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_concept_decomposition_validation.py::TestDecompositionMechanism -xvs
```

These 6 tests use synthetic (fallback hash) embeddings and a mock decomposer. They verify:
1. Decomposed sentences create multiple nodes (not one blob)
2. Without decomposition creates a single node
3. Decomposed chunks pattern-complete to bare concepts
4. Cross-modal binding works through decomposed nodes
5. Without decomposition, bare concepts miss vision bindings (the gap)
6. All 5 fixture scenes pass cross-modal retrieval

### Real-embedding validation (slow, requires sentence-transformers + spaCy)

```bash
PYTHONPATH=src python -m pytest tests/substrate/test_concept_decomposition_validation.py::TestDecompositionValidation -xvs
```

Runs both arms (baseline vs decomposed) on 5 naturalistic scenes with `paraphrase-mpnet-base-v2` embeddings and real spaCy noun chunking. Saves results to `docs/experiments/results/concept_decomposition_validation.json`.

### Regression suite (run alongside)

```bash
# P4 cross-modal mechanism (should be unaffected)
PYTHONPATH=src python -m pytest tests/substrate/test_p4_cross_modal_mechanism.py -xvs

# P2 reward modulation (should be unaffected)
PYTHONPATH=src python -m pytest tests/substrate/test_p2_reward_modulation.py::TestP2Mechanism -xvs

# Decomposer unit tests (bare class names, modality gate, thread safety)
PYTHONPATH=src python -m pytest tests/unit/test_concept_decomposer.py -xvs
```

## Expected results

### Validation pass criteria

Both must hold:
- `decomposed_recall > baseline_recall` (strict improvement)
- `decomposed_recall >= 0.60` (minimum quality bar)

### Reference numbers (2026-04-17, M3 Mac)

| Arm | Recall | Hits/Queries |
|-----|--------|-------------|
| Baseline | 36.4% | 4/11 |
| Decomposed | 100.0% | 11/11 |

The baseline recall may vary slightly across environments (EC pattern-completion is sensitive to floating-point ordering), but the decomposed arm should consistently reach 100% because each concept has a direct Hebbian binding to its paired vision node.

## If reproduction fails

1. **Baseline recall changed significantly:** Check EC threshold (should be 0.70), check `paraphrase-mpnet-base-v2` model version, check if encoder config changed.

2. **Decomposed recall dropped below 100%:** Check spaCy noun chunking output — run `python -c "import spacy; nlp = spacy.load('en_core_web_sm'); doc = nlp('I see a blue mug on the wooden table'); print([nc.text for nc in doc.noun_chunks])"` and verify the expected chunks are extracted.

3. **P4 or P2 regression:** Decomposition should not affect these. If they fail, the issue is in shared infrastructure (EC, Hippocampus, ATL), not in the decomposer.

## Key files

| File | Purpose |
|------|---------|
| `src/maxim/similarity/decomposer.py` | Protocol + strategies + coordinator |
| `src/maxim/similarity/encoder.py` | `encode_decomposed()` integration |
| `src/maxim/integration/memory_hub.py` | `_wire_substrate_encoder()` env-var gate |
| `tests/substrate/test_concept_decomposition_validation.py` | Validation fixture |
| `tests/unit/test_concept_decomposer.py` | Unit tests |
| `docs/experiments/results/concept_decomposition_validation.json` | Raw results |
