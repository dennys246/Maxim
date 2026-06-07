# Memory

## Overview

Maxim remembers what it sees, does, and learns across sessions. Memory is stored locally on disk -- nothing is sent to external services.

## What Gets Remembered

- **Episodic memories** -- Records of what happened: what the robot saw, what tools it used, what the outcomes were. Stored with timestamps, goals, and context.
- **Associative links** -- Connections between related memories (e.g., "using the gripper" is associated with "picking up objects"). Formed automatically during capture.
- **Spatial knowledge** -- Where objects were seen, workspace boundaries, frequently visited locations.
- **Temporal patterns** -- Time-of-day patterns (e.g., "people appear more often in the morning"). Tracked by the SCN (temporal indexing system).
- **Causal links** -- What caused what (e.g., "gripping too hard caused a pain signal"). Tracked by NAc (causal inference).
- **Learned thresholds** -- Novelty sensitivity, escalation thresholds, movement gains. Auto-tuned over time.

## Memory Lifecycle

Memories move through three tiers:

1. **Forming** -- Raw percepts being processed (current cycle). Protected from eviction.
2. **Short-term** -- Recent memories with outcome. Subject to consolidation.
3. **Long-term** -- Important memories promoted during sleep or via use-based consolidation.

Active-reference context (recent percepts, outcomes, recalled memories) lives in the **WorkingMemorySet** -- an Exec-owned layer, not a memory tier. It provides the LLM with immediate context without polluting the Hippocampus tier hierarchy.

Memories are promoted based on: use-based pressure (diverse recall contexts accumulate promotion pressure, which decays over time), access frequency, emotional significance (surprise/reward), goal relevance, and associative connections.

## Where Memory Lives

```
~/.maxim/memory/hippocampus.json       -- Episodic memories (main store)
~/.maxim/util/workspace_bounds.json    -- Workspace safety bounds
~/.maxim/util/focus_learner.json       -- Motor gain learning
~/.maxim/util/adaptive_thresholds.json -- Auto-tuned thresholds
```

## Memory and Sleep

Sleep mode (`--mode sleep` or "Maxim sleep") triggers consolidation:

- Important short-term memories are promoted to long-term.
- Redundant memories are compressed.
- The associative graph is strengthened.

## Managing Memory

### Viewing Memory State

In agentic mode, use the ExplainTool to query what the system remembers and why decisions were made.

### Clearing Memory

```bash
# Clear specific types
maxim --clear-memory hippo           # Episodic memories
maxim --clear-memory focus           # Motor gain learning
maxim --clear-memory bounds          # Workspace bounds
maxim --clear-memory nac             # Causal learning
maxim --clear-memory scn             # Temporal patterns
maxim --clear-memory pain            # Pain learning
maxim --clear-memory semantic        # Semantic embeddings
maxim --clear-memory fear,escalation # Safety thresholds

# Clear everything
maxim --clear-memory all
```

### Starting Fresh

```bash
maxim --reset  # Clears working memory on startup (keeps long-term)
```

### Semantic Similarity (Optional)

Enable neural embeddings for conceptual similarity (e.g., "cup" matches "mug"):

```bash
pip install -e '.[semantic]'
maxim --mode agentic --enable-embeddings
```

Without this, memory retrieval uses keyword matching and hash indexing.
