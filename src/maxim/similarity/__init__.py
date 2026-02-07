"""Similarity module - EC (Entorhinal Cortex) multi-modal similarity.

Provides situation signatures, LSH indexing, and similarity queries.

Phase 4 adds neural semantic embeddings for deep similarity:
- "cup" ≈ "mug" (semantic understanding)
- "greet" ≈ "say hello" (action similarity)
"""

from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.indices import InvertedIndices
from maxim.similarity.lsh import LSHIndex, SemanticLSH
from maxim.similarity.signature import SituationSignature

# Phase 4: Neural semantic similarity (optional, requires sentence-transformers)
try:
    from maxim.similarity.semantic import (
        NeuralSemanticLSH,
        SemanticEmbedderConfig,
        EmbeddingStore,
        is_gpu_available,
        is_semantic_available,
    )

    _SEMANTIC_AVAILABLE = True
except ImportError:
    NeuralSemanticLSH = None  # type: ignore
    SemanticEmbedderConfig = None  # type: ignore
    EmbeddingStore = None  # type: ignore
    is_gpu_available = lambda: False  # type: ignore
    is_semantic_available = lambda: False  # type: ignore
    _SEMANTIC_AVAILABLE = False

__all__ = [
    "EntorhinalCortex",
    "ECConfig",
    "SituationSignature",
    "LSHIndex",
    "SemanticLSH",
    "InvertedIndices",
    # Phase 4
    "NeuralSemanticLSH",
    "SemanticEmbedderConfig",
    "EmbeddingStore",
    "is_gpu_available",
    "is_semantic_available",
]