# Memory

## Overview

Maxim remembers what it sees, does, and learns across sessions. Memory is stored locally on disk -- nothing is sent to external services.

## What Gets Remembered

- **Episodic memories** -- Records of what happened: what the agent saw, what tools it used, what the outcomes were. Stored with timestamps, goals, and context.
- **Substrate nodes** -- Concept-level patterns extracted from text percepts and encoded into the EC/ATL substrate. These enable semantic similarity, cross-session recall, and reward modulation. Requires the substrate path (see below).
- **Causal links** -- What caused what (e.g., "gripping too hard caused a pain signal"). Tracked by NAc (causal inference).
- **Temporal patterns** -- Time-of-day patterns (e.g., "people appear more often in the morning"). Tracked by SCN (temporal indexing).
- **Spatial knowledge** -- Where objects were seen, workspace boundaries, frequently visited locations.
- **Learned thresholds** -- Novelty sensitivity, escalation thresholds, movement gains. Auto-tuned over time.

## Getting Neural Memory Quality — Substrate Path

The substrate path (`MAXIM_SUBSTRATE_PATH=1`) is the current mechanism for high-quality semantic memory. It routes text percepts through the LinguisticEncoder → EntorhinalCortex → ATL pipeline, enabling:

- Paraphrase collapse: "mug" and "cup" map to the same substrate node
- Cross-session concept recall
- NAc reward-bias annotation on concept clusters

**Prerequisites:**

```bash
# Install the semantic extra (sentence-transformers + torch + spaCy)
pip install 'pymaxim[semantic]'

# Download the spaCy English model (required for concept decomposition)
python -m spacy download en_core_web_sm
```

> **Note:** The `[all]` extra does NOT include `[semantic]`. If you installed with
> `pip install pymaxim[all]`, you still need `pip install 'pymaxim[semantic]'` separately
> to get neural embeddings. Without it, the substrate path falls back to deterministic
> bag-of-words hash embeddings that do not support paraphrase recall.

**Enable the substrate path:**

```bash
# Substrate path only (whole-text encoding)
MAXIM_SUBSTRATE_PATH=1 maxim --llm mistral-7b

# Substrate path + concept decomposition (recommended for full cross-session learning)
MAXIM_SUBSTRATE_PATH=1 MAXIM_CONCEPT_DECOMPOSITION=1 maxim --llm mistral-7b
```

See [concept-decomposition.md](concept-decomposition.md) for details on how decomposition breaks text into per-concept substrate nodes.

**What the encoder uses:**

The LinguisticEncoder loads `paraphrase-mpnet-base-v2` from sentence-transformers (part of `[semantic]`). If sentence-transformers is not installed, a one-time warning is emitted and the fallback bag-of-words hash is used -- this keeps the pipeline functional but paraphrase recall will not work.

## Memory Lifecycle

Memories move through three tiers:

1. **Forming** -- Raw percepts being processed (current cycle). Protected from eviction.
2. **Short-term** -- Recent memories with outcome. Subject to consolidation.
3. **Long-term** -- Important memories promoted during sleep or via use-based pressure.

Active-reference context (recent percepts, outcomes, recalled memories) lives in the **WorkingMemorySet** -- an Exec-owned layer, not a memory tier. It provides the LLM with immediate context without polluting the Hippocampus tier hierarchy.

Promotion from short-term to long-term is **pressure-based**: diverse-query recall accumulates promotion pressure, which decays over time. When pressure exceeds the threshold, the memory is promoted. This rewards genuinely useful memories over recently-touched ones.

## Where Memory Lives

```
~/.maxim/memory/hippocampus.json         -- Episodic memories (main store)
~/.maxim/util/nac_state.json             -- Causal links and reward biases
~/.maxim/util/scn_state.json             -- Temporal patterns
~/.maxim/util/atl_state.json             -- ATL substrate nodes (when substrate path enabled)
~/.maxim/util/semantic_embeddings.npz    -- Embedding vectors
~/.maxim/util/workspace_bounds.json      -- Workspace safety bounds
~/.maxim/util/focus_learner.json         -- Motor gain learning
~/.maxim/util/adaptive_thresholds.json   -- Auto-tuned thresholds
```

## Memory and Sleep

Sleep mode is triggered when the agent calls the `sleep` tool (or via `--mode sleep` for the robot runtime). During sleep:

- Important short-term memories are promoted to long-term.
- Redundant memories are compressed.
- The substrate's Hebbian graph is consolidated.

## Managing Memory

### Viewing Memory State

In agentic mode, use the ExplainTool to query what the system remembers and why decisions were made.

### Clearing Memory

```bash
# Clear specific types
maxim --clear-memory hippo           # Episodic memories
maxim --clear-memory nac             # Causal learning + reward biases (also clears ec)
maxim --clear-memory ec              # EC substrate (also clears nac — they are a pair)
maxim --clear-memory scn             # Temporal patterns
maxim --clear-memory atl             # ATL substrate nodes
maxim --clear-memory semantic        # Embedding vectors
maxim --clear-memory focus           # Motor gain learning
maxim --clear-memory bounds          # Workspace bounds
maxim --clear-memory pain            # Pain learning
maxim --clear-memory fear,escalation # Safety thresholds

# Clear everything
maxim --clear-memory all
```

### Starting Fresh

```bash
maxim --reset  # Clears working memory on startup (keeps long-term)
```

---

## Legacy: `--enable-embeddings` (pre-0.9)

> **This section describes an older mechanism retained for backward compatibility.**
> For current neural memory quality, use the substrate path above.

Before 0.9, the `MemoryAgent` class accepted `--enable-embeddings` to load
`all-MiniLM-L6-v2` for keyword-based similarity within the `AssociationIndex`.
This path is still present in the code (`AssociationIndex`, `MemoryAgent`,
`--enable-embeddings` CLI flag) but is not used by the primary agentic runtime.
The substrate path (`MAXIM_SUBSTRATE_PATH=1`) replaced it for paraphrase recall
and cross-session concept learning.

If you are using the legacy `MemoryAgent` API directly:

```bash
pip install 'pymaxim[semantic]'
# Then in your code:
from maxim.agents import MemoryAgent
agent = MemoryAgent(enable_embeddings=True)
```

New code should use `build_bio_stack` with `MAXIM_SUBSTRATE_PATH=1` instead.
