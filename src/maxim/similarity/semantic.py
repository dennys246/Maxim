"""Neural semantic similarity using SentenceTransformer embeddings.

Phase 4 implementation enabling queries like "find mug" to match memories
about "cup" through deep semantic understanding.

Performance Notes:
- Model loading: ~2-5 seconds (one-time at startup)
- Embedding latency: ~2ms (GPU) / ~15ms (CPU) per text
- Memory footprint: ~80MB (MiniLM) to ~420MB (MPNet)
- Recommended: Use GPU and async embedding pattern for real-time systems
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import for optional sentence-transformers dependency
_sentence_transformer_model = None
_model_lock = threading.Lock()


def is_gpu_available() -> bool:
    """Check if GPU is available for semantic embedding.

    Returns:
        True if CUDA GPU is available, False otherwise.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def is_semantic_available() -> bool:
    """Check if semantic embedding dependencies are installed.

    Returns:
        True if sentence-transformers and torch are available.
    """
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_or_load_model(model_name: str) -> Any:
    """Lazy-load SentenceTransformer model (singleton).

    Thread-safe model loading with singleton pattern to avoid
    loading the model multiple times.
    """
    global _sentence_transformer_model

    with _model_lock:
        if _sentence_transformer_model is not None:
            return _sentence_transformer_model

        try:
            from sentence_transformers import SentenceTransformer

            # Check for GPU availability
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading semantic model %s on %s", model_name, device)

            _sentence_transformer_model = SentenceTransformer(model_name, device=device)
            logger.info("Semantic model loaded successfully")
            return _sentence_transformer_model

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.error("Failed to load semantic model: %s", e)
            return None


@dataclass
class SemanticEmbedderConfig:
    """Configuration for neural semantic embeddings."""

    # Model selection (smaller = faster, larger = better quality)
    model_name: str = "all-MiniLM-L6-v2"  # 80MB, good quality/speed balance

    # Embedding dimension (model-dependent)
    embedding_dim: int = 384  # MiniLM default

    # LSH settings for hash generation
    num_hash_bits: int = 16  # Bits for LSH hash

    # Require GPU for real-time performance
    require_gpu: bool = False

    # Async embedding settings
    async_embedding: bool = True
    max_queue_size: int = 1000


@dataclass
class NeuralSemanticLSH:
    """Neural semantic LSH using SentenceTransformer embeddings.

    Provides deep semantic similarity where "cup" ≈ "mug" and
    "greet" ≈ "say hello".

    Features:
    - Neural embeddings for true semantic understanding
    - LSH hashing for O(1) approximate nearest neighbor
    - Async embedding for non-blocking captures
    - GPU acceleration when available
    - Graceful CPU fallback

    Example:
        embedder = NeuralSemanticLSH()

        # Embed text to hash
        hash1 = embedder.hash("find the coffee mug")
        hash2 = embedder.hash("locate the cup")

        # Semantic similarity preserved in hash space
        similarity = embedder.estimated_similarity(hash1, hash2)
        # similarity ≈ 0.7 (semantically related)

    Performance:
        - GPU: ~2ms per embedding
        - CPU: ~15ms per embedding
        - Hash comparison: O(1)
    """

    config: SemanticEmbedderConfig = field(default_factory=SemanticEmbedderConfig)

    # Internal state
    _model: Any = field(default=None, repr=False)
    _random_planes: np.ndarray | None = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)
    _healthy: bool = field(default=True, repr=False)

    # Async embedding
    _embedding_queue: queue.Queue | None = field(default=None, repr=False)
    _embedding_thread: threading.Thread | None = field(default=None, repr=False)
    _shutdown_event: threading.Event | None = field(default=None, repr=False)

    # Embedding cache
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _max_cache_size: int = 10000

    def __post_init__(self) -> None:
        """Initialize the embedder."""
        if not self._initialized:
            self._initialize()

    def _initialize(self) -> None:
        """Lazy initialization of model and random planes."""
        self._model = _get_or_load_model(self.config.model_name)

        if self._model is None:
            logger.warning("Semantic embedder disabled (no model)")
            self._healthy = False
            self._initialized = True
            return

        # Check GPU requirement
        if self.config.require_gpu:
            import torch

            if not torch.cuda.is_available():
                logger.warning("GPU required but not available, disabling semantic")
                self._healthy = False
                self._initialized = True
                return

        # Generate random hyperplanes for LSH
        np.random.seed(42)  # Reproducible hashes
        self._random_planes = np.random.randn(
            self.config.num_hash_bits, self.config.embedding_dim
        )
        # Normalize planes
        self._random_planes = (
            self._random_planes
            / np.linalg.norm(self._random_planes, axis=1, keepdims=True)
        )

        # Initialize async embedding if enabled
        if self.config.async_embedding:
            self._embedding_queue = queue.Queue(maxsize=self.config.max_queue_size)
            self._shutdown_event = threading.Event()
            self._embedding_thread = threading.Thread(
                target=self._embedding_worker,
                daemon=True,
                name="semantic-embedder",
            )
            self._embedding_thread.start()

        self._initialized = True
        self._healthy = True
        logger.info(
            "NeuralSemanticLSH initialized: model=%s, bits=%d, async=%s",
            self.config.model_name,
            self.config.num_hash_bits,
            self.config.async_embedding,
        )

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector (embedding_dim,)
        """
        if not self._healthy or self._model is None:
            # Return zero embedding as fallback
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Generate embedding
        try:
            embedding = self._model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Cache the result
            with self._cache_lock:
                if len(self._cache) >= self._max_cache_size:
                    # Simple cache eviction: remove oldest half
                    keys = list(self._cache.keys())
                    for k in keys[: len(keys) // 2]:
                        del self._cache[k]
                self._cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return np.zeros(self.config.embedding_dim, dtype=np.float32)

    def hash(self, text: str) -> tuple[int, ...]:
        """Generate LSH hash for text.

        Uses random hyperplane hashing to convert high-dimensional
        embeddings to binary hash codes.

        Args:
            text: Text to hash

        Returns:
            Tuple of hash bits (0 or 1)
        """
        if not self._healthy or self._random_planes is None:
            # Fallback to simple word-based hashing
            return self._fallback_hash(text)

        embedding = self.embed(text)

        # Project onto random hyperplanes
        projections = np.dot(self._random_planes, embedding)

        # Convert to binary hash
        hash_bits = tuple(int(p > 0) for p in projections)

        return hash_bits

    def _fallback_hash(self, text: str) -> tuple[int, ...]:
        """Fallback hash when model unavailable.

        Uses simple word-based hashing (not semantic).
        """
        if not text:
            return (0,) * self.config.num_hash_bits

        words = text.lower().split()
        bits = []

        for i in range(self.config.num_hash_bits):
            h = sum(hash(f"{w}:{i}:42") for w in words)
            bits.append(1 if h > 0 else 0)

        return tuple(bits)

    def estimated_similarity(
        self,
        hash1: tuple[int, ...],
        hash2: tuple[int, ...],
    ) -> float:
        """Estimate cosine similarity from hash Hamming distance.

        LSH with random hyperplanes preserves cosine similarity:
        P(h(x) = h(y)) = 1 - θ/π where θ = angle between x,y

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Estimated cosine similarity (0.0 to 1.0)
        """
        if len(hash1) != len(hash2):
            return 0.0

        hamming = sum(a != b for a, b in zip(hash1, hash2))
        agreement_ratio = 1.0 - (hamming / len(hash1))
        estimated_angle = math.pi * (1 - agreement_ratio)
        return max(0.0, math.cos(estimated_angle))

    def cosine_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute exact cosine similarity between two texts.

        Slower than hash-based similarity but exact.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (-1.0 to 1.0)
        """
        if not self._healthy:
            return 0.0

        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        return float(np.dot(emb1, emb2))

    # ─────────────────────────────────────────────────────────────────────────
    # Async Embedding
    # ─────────────────────────────────────────────────────────────────────────

    def schedule_embedding(
        self,
        memory_id: str,
        text: str,
        callback: Any | None = None,
    ) -> bool:
        """Schedule text for async embedding.

        Adds to background queue for non-blocking embedding.
        Use for real-time systems where capture latency matters.

        Args:
            memory_id: Memory identifier
            text: Text to embed
            callback: Optional callback(memory_id, embedding) when done

        Returns:
            True if queued, False if queue full or disabled
        """
        if not self.config.async_embedding or self._embedding_queue is None:
            return False

        try:
            self._embedding_queue.put_nowait((memory_id, text, callback))
            return True
        except queue.Full:
            logger.warning("Embedding queue full, dropping request")
            return False

    def _embedding_worker(self) -> None:
        """Background thread for async embedding."""
        while not self._shutdown_event.is_set():
            try:
                item = self._embedding_queue.get(timeout=0.1)
                memory_id, text, callback = item

                embedding = self.embed(text)
                hash_bits = self.hash(text)

                if callback:
                    try:
                        callback(memory_id, embedding, hash_bits)
                    except Exception as e:
                        logger.warning("Embedding callback failed: %s", e)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Embedding worker error: %s", e)

    def shutdown(self) -> None:
        """Shutdown async embedding worker."""
        if self._shutdown_event:
            self._shutdown_event.set()
        if self._embedding_thread:
            self._embedding_thread.join(timeout=2.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Health and Statistics
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_healthy(self) -> bool:
        """Check if embedder is operational."""
        return self._healthy

    def stats(self) -> dict[str, Any]:
        """Return embedder statistics."""
        queue_size = 0
        if self._embedding_queue:
            queue_size = self._embedding_queue.qsize()

        return {
            "healthy": self._healthy,
            "model": self.config.model_name if self._healthy else None,
            "async_enabled": self.config.async_embedding,
            "queue_size": queue_size,
            "cache_size": len(self._cache),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Storage for Memories
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingStore:
    """Storage for memory embeddings.

    Provides persistence and retrieval of embeddings for memories.
    Supports incremental embedding of new memories.
    """

    embedder: NeuralSemanticLSH
    _embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    _hashes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def set(
        self,
        memory_id: str,
        embedding: np.ndarray,
        hash_bits: tuple[int, ...] | None = None,
    ) -> None:
        """Store embedding for a memory."""
        self._embeddings[memory_id] = embedding
        if hash_bits:
            self._hashes[memory_id] = hash_bits

    def get(self, memory_id: str) -> np.ndarray | None:
        """Get embedding for a memory."""
        return self._embeddings.get(memory_id)

    def get_hash(self, memory_id: str) -> tuple[int, ...] | None:
        """Get hash for a memory."""
        return self._hashes.get(memory_id)

    def remove(self, memory_id: str) -> None:
        """Remove embedding for a memory."""
        self._embeddings.pop(memory_id, None)
        self._hashes.pop(memory_id, None)

    def find_similar(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find memories similar to query text.

        Performs exact cosine similarity search (slower but accurate).

        Args:
            query: Query text
            k: Maximum results
            threshold: Minimum similarity

        Returns:
            List of (memory_id, similarity) tuples
        """
        if not self.embedder.is_healthy:
            return []

        query_embedding = self.embedder.embed(query)

        results = []
        for memory_id, embedding in self._embeddings.items():
            similarity = float(np.dot(query_embedding, embedding))
            if similarity >= threshold:
                results.append((memory_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def embed_missing(
        self,
        memories: dict[str, str],
        batch_size: int = 32,
    ) -> int:
        """Embed memories that don't have embeddings yet.

        Used for incremental embedding of new memories.

        Args:
            memories: Dict of memory_id -> text to embed
            batch_size: Batch size for efficient embedding

        Returns:
            Number of memories embedded
        """
        if not self.embedder.is_healthy:
            return 0

        missing = [
            (mid, text)
            for mid, text in memories.items()
            if mid not in self._embeddings
        ]

        if not missing:
            return 0

        count = 0
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]

            for memory_id, text in batch:
                embedding = self.embedder.embed(text)
                hash_bits = self.embedder.hash(text)
                self.set(memory_id, embedding, hash_bits)
                count += 1

        return count

    def __len__(self) -> int:
        """Number of stored embeddings."""
        return len(self._embeddings)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save embeddings to file.

        Uses numpy's compressed npz format for efficiency.
        Embeddings are stored as a single stacked array.

        Args:
            path: Path to save file (should end with .npz)
        """
        import json as json_module

        if not self._embeddings:
            logger.info("No embeddings to save")
            return

        # Convert to arrays
        memory_ids = list(self._embeddings.keys())
        embeddings_array = np.stack([self._embeddings[mid] for mid in memory_ids])

        # Prepare hashes as JSON (tuples aren't directly serializable in npz)
        hashes_data = {mid: list(h) for mid, h in self._hashes.items()}
        hashes_json = json_module.dumps(hashes_data)

        # Save to compressed npz
        np.savez_compressed(
            path,
            memory_ids=np.array(memory_ids, dtype=object),
            embeddings=embeddings_array,
            hashes_json=np.array([hashes_json], dtype=object),
            version=np.array([1]),
        )

        logger.info("Saved %d embeddings to %s", len(memory_ids), path)

    def load(self, path: str) -> int:
        """Load embeddings from file.

        Args:
            path: Path to load file

        Returns:
            Number of embeddings loaded
        """
        import json as json_module

        if not os.path.exists(path):
            logger.warning("Embedding file not found: %s", path)
            return 0

        try:
            data = np.load(path, allow_pickle=True)

            version = int(data.get("version", [0])[0])
            if version != 1:
                logger.warning("Unknown embedding file version: %d", version)
                return 0

            memory_ids = data["memory_ids"]
            embeddings_array = data["embeddings"]
            hashes_json = str(data["hashes_json"][0])

            # Load embeddings
            for i, mid in enumerate(memory_ids):
                self._embeddings[str(mid)] = embeddings_array[i]

            # Load hashes
            hashes_data = json_module.loads(hashes_json)
            for mid, h in hashes_data.items():
                self._hashes[mid] = tuple(h)

            logger.info("Loaded %d embeddings from %s", len(memory_ids), path)
            return len(memory_ids)

        except Exception as e:
            logger.error("Failed to load embeddings: %s", e)
            return 0

    def clear(self) -> None:
        """Clear all stored embeddings."""
        self._embeddings.clear()
        self._hashes.clear()


__all__ = [
    "NeuralSemanticLSH",
    "SemanticEmbedderConfig",
    "EmbeddingStore",
    "is_gpu_available",
    "is_semantic_available",
]