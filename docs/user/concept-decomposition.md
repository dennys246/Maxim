# Concept Decomposition

## Overview

Concept decomposition breaks input text into meaningful concept-level chunks before encoding them into the substrate. Instead of encoding `"I see a blue mug on the table"` as one opaque substrate node, decomposition produces separate nodes for `"blue mug"`, `"table"`, etc. Each concept can independently form cross-session associations and cross-modal bindings.

## Why It Matters

Without decomposition, the hippocampus sees sentences as atomic blobs. A full sentence `"I see a blue mug"` won't reliably match against a bare `"mug"` node from a previous session -- the Hebbian edge from `"mug"` to a vision of a mug becomes unreachable from the sentence node. With decomposition, `"blue mug"` is its own substrate node that can bind to vision-mug across episodes.

This matters most for:
- **Cross-modal retrieval** -- matching text descriptions to visual memories
- **Cross-session learning** -- recognizing the same concept described differently across sessions
- **Pain/reward association** -- finer-grained concepts make better targets for valence annotation (learning that "rusty sword" is dangerous, not just that the whole sentence was bad)

## How It Works

```
Input: "I see a blue mug on the table next to the red plate"
                    |
          ConceptDecomposer
                    |
         ["blue mug", "table", "red plate"]
                    |
    LinguisticEncoder (embeds each chunk)
                    |
    EC (pattern complete or separate per chunk)
                    |
    Hippocampus (all chunks land in same episode,
                 get Hebbian-bound together)
```

The key property: **nothing below the decomposer changes.** EC, Hippocampus, Hebbian binding, cross-modal retrieval, NAc reward modulation, persistence -- all stay the same. Decomposition is purely additive pre-processing.

## Enabling Decomposition

Concept decomposition requires the `semantic` extra (for sentence-transformers) plus spaCy:

```bash
pip install pymaxim[semantic]
pip install spacy
python -m spacy download en_core_web_sm
```

Then enable via environment variables:

```bash
MAXIM_SUBSTRATE_PATH=1 MAXIM_CONCEPT_DECOMPOSITION=1 maxim --llm mistral-7b
```

Both flags are opt-in. Without `MAXIM_CONCEPT_DECOMPOSITION=1`, the substrate path works exactly as before (whole-text encoding).

## What Gets Extracted

**Noun phrases** are the primary payload, extracted via spaCy's noun chunker:

| Input | Extracted Concepts |
|---|---|
| `"I see a blue mug on the table"` | `["blue mug", "table"]` |
| `"The red plate is next to the green cup"` | `["red plate", "green cup"]` |
| `"The rusty sword feels heavy"` | `["rusty sword"]` |
| `"lotus"` (bare class name) | `["lotus"]` (identity -- no decomposition needed) |

**Not extracted:** pronouns (`"he"`, `"it"`), determiners (`"the"`, `"a"`), bare verbs (`"see"`, `"go"`). These don't carry concept-level meaning and would create noise in the substrate.

## Modality Gate

Decomposition applies to **text-modality percepts only.** Visual percepts (CLIP embeddings), proprioceptive readings, and SEM affordance labels bypass decomposition automatically. This is enforced at the encoder level -- callers don't need to check.

## Pluggable Strategies

The decomposer uses a Protocol-based design so different NLP backends can be swapped without touching the pipeline:

```python
from maxim.similarity.decomposer import (
    ConceptDecomposer,
    DecompositionStrategy,
    ConceptChunk,
)

class MyCustomStrategy:
    """Your domain-specific extraction logic."""
    def extract(self, text: str) -> list[ConceptChunk]:
        # Your logic here
        return [ConceptChunk(text="concept", span=(0, 7))]

decomposer = ConceptDecomposer(strategy=MyCustomStrategy())
```

Built-in strategies:
- **SpaCyNounChunkStrategy** -- default, uses `en_core_web_sm` noun chunker
- **IdentityStrategy** -- fallback, returns input unchanged (used when spaCy not installed)

## Configuration

| Variable | Default | Description |
|---|---|---|
| `MAXIM_CONCEPT_DECOMPOSITION` | off | Set to `1` to enable decomposition |
| `MAXIM_SUBSTRATE_PATH` | off | Must also be `1` (substrate encoding prerequisite) |

The decomposer config supports:
- `enabled: bool` -- master toggle (also controllable via env var)
- `min_chunk_len: int` -- minimum characters for a chunk (default 2, filters noise)
- `spacy_model: str` -- spaCy model name (default `en_core_web_sm`)

## For Agent Developers

If you're building a custom agent with the Python API:

```python
from maxim.similarity.decomposer import ConceptDecomposer, DecomposerConfig
from maxim.similarity.encoder import LinguisticEncoder

# Create a decomposer with custom config
decomposer = ConceptDecomposer(config=DecomposerConfig(
    enabled=True,
    min_chunk_len=3,
))

# Pass to the encoder
encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac, decomposer=decomposer)

# encode_decomposed returns multiple node IDs
node_ids = encoder.encode_decomposed(
    "I see a blue mug on the table",
    modality="text",
    agent_id="agent-1",
)
# node_ids = ["uuid-blue-mug", "uuid-table"]
```

## Limitations

- **English only** (Stage 1). spaCy `en_core_web_sm` is English-only. Multi-language support requires a multilingual model (`xx_ent_wiki_sm`) or per-language model selection.
- **Short fragments** may over-decompose. The `min_chunk_len` filter helps, but domain-specific inputs may need a custom strategy.
- **No relation tagging yet** (Stage 2). Chunks are bound with untagged Hebbian edges. Role-tagged edges (`relation="spatial"`) are planned for a future stage.
