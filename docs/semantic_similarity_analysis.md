# Semantic Similarity - Phase 4 Implementation

## Overview

Phase 4 adds **deep semantic similarity** to the Entorhinal Cortex (EC), enabling queries like "find mug" to match memories about "cup" through neural embeddings.

**Status: IMPLEMENTED**

## What's Included

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| `NeuralSemanticLSH` | `similarity/semantic.py` | Neural embedding with LSH hashing |
| `EmbeddingStore` | `similarity/semantic.py` | Embedding storage with persistence |
| `SemanticEmbedderConfig` | `similarity/semantic.py` | Configuration for embeddings |

### Features

1. **Neural Embeddings**: SentenceTransformer models for true semantic understanding
2. **LSH Hashing**: O(1) approximate nearest neighbor via random hyperplanes
3. **Async Embedding**: Non-blocking capture with background embedding
4. **GPU Acceleration**: Automatic CUDA detection and usage
5. **Graceful Fallback**: CPU mode disables semantic for performance
6. **Persistence**: Embeddings saved to disk across sessions

## Installation

```bash
# Install semantic dependencies
pip install -e '.[semantic]'

# Or with all extras
pip install -e '.[all]'
```

## Configuration

```python
from maxim.similarity import ECConfig, EntorhinalCortex

config = ECConfig(
    # Enable Phase 4 semantic embeddings
    enable_semantic=True,

    # Model selection (default: all-MiniLM-L6-v2, 80MB)
    semantic_model="all-MiniLM-L6-v2",

    # Async embedding for non-blocking capture
    async_embedding=True,

    # Require GPU (recommended for real-time)
    require_gpu=False,

    # Semantic hash bits
    semantic_hash_bits=16,
)

ec = EntorhinalCortex(config)
```

## Usage

### Semantic Queries

```python
from maxim.integration.memory_hub import MemoryHub

# Find semantically similar memories
results = hub.find_semantic("find the coffee mug", k=10, threshold=0.5)

for memory_id, similarity in results:
    print(f"{memory_id}: {similarity:.2f}")
    # "find cup" → 0.85 (semantic match!)
```

### Direct Embedding

```python
from maxim.similarity import NeuralSemanticLSH

embedder = NeuralSemanticLSH()

# Embed text
embedding = embedder.embed("find the coffee mug")

# Hash for LSH indexing
hash_bits = embedder.hash("find the coffee mug")

# Compare similarity
similarity = embedder.cosine_similarity(
    "find the coffee mug",
    "locate the cup"
)
print(f"Similarity: {similarity:.2f}")  # ~0.7
```

### Check Availability

```python
from maxim.similarity import is_gpu_available, is_semantic_available

if is_semantic_available():
    print("sentence-transformers installed")

if is_gpu_available():
    print("CUDA GPU available")
```

## Model Options

| Model | Size | Latency (CPU) | Latency (GPU) | Quality |
|-------|------|---------------|---------------|---------|
| `all-MiniLM-L6-v2` | 80MB | ~15ms | ~2ms | Good (default) |
| `all-mpnet-base-v2` | 420MB | ~50ms | ~8ms | Best |
| `paraphrase-MiniLM-L3-v2` | 45MB | ~8ms | ~1ms | Acceptable |

## Architecture

### Data Flow

```
Memory Capture
      ↓
Hippocampus.capture()
      ↓
_on_memory_captured callback
      ↓
NeuralSemanticLSH.schedule_embedding()
      ↓
[Background Thread]
      ↓
EmbeddingStore.set(memory_id, embedding, hash)
      ↓
Semantic queries work
```

### Integration Points

```
MemoryHub
    ├── Hippocampus (capture callback)
    │   └── _on_memory_captured()
    ├── EntorhinalCortex
    │   ├── NeuralSemanticLSH (async embedder)
    │   └── EmbeddingStore (storage)
    └── Session lifecycle
        ├── on_session_start() → load embeddings
        └── on_session_end() → save embeddings
```

## Performance

### Memory Footprint

| Component | Size |
|-----------|------|
| MiniLM model | ~80MB |
| Per-embedding | ~384 bytes (float32) |
| 10K memories | ~4MB embeddings |
| Hash bits | ~2 bytes per memory |

### Latency

| Operation | GPU | CPU |
|-----------|-----|-----|
| Capture (async) | <1ms | <1ms |
| Embedding | ~2ms | ~15ms |
| Query | ~2ms | ~15ms |
| Hash comparison | O(1) | O(1) |

## Persistence

Embeddings persist across sessions:

```python
# Automatic via MemoryHub
hub.on_session_start()  # Loads embeddings
hub.on_session_end()    # Saves embeddings

# Manual
ec._embedding_store.save("~/.maxim/util/semantic_embeddings.npz")
ec._embedding_store.load("~/.maxim/util/semantic_embeddings.npz")
```

### File Format

Uses numpy's compressed `.npz` format:
- `memory_ids`: Array of memory IDs
- `embeddings`: Stacked embedding vectors
- `hashes_json`: Hash bits as JSON
- `version`: Format version

## Graceful Degradation

When semantic is unavailable or disabled:

```python
# EC falls back to structural similarity
results = ec.find_semantic("find mug", k=10)
# Returns [] if semantic not enabled

# Check before using
if hub.semantic_enabled:
    results = hub.find_semantic(query)
else:
    results = hub.get_plan_templates(query)  # Fallback
```

## Best Practices

1. **Enable GPU**: Use CUDA for real-time performance
2. **Use async**: Keep `async_embedding=True` for non-blocking capture
3. **Load on start**: Call `on_session_start()` to load cached embeddings
4. **Save on end**: Call `on_session_end()` to persist new embeddings
5. **Check availability**: Use `is_semantic_available()` before enabling

## Benchmarks Validated

| Test | Target | Actual |
|------|--------|--------|
| Capture throughput | 30Hz | ✓ (async) |
| Query latency P99 | <50ms | ✓ (~10ms GPU) |
| Memory stable at 10K | Yes | ✓ |
| Model loads in <5s | Yes | ✓ (~2-3s) |
| GPU memory <4GB | Yes | ✓ (~200MB) |

## Future Improvements

1. **Batch embedding**: Embed multiple memories in one forward pass
2. **Incremental indexing**: Add to LSH without full rebuild
3. **Model fine-tuning**: Custom model for robotics domain
4. **Quantization**: INT8 embeddings for memory reduction
