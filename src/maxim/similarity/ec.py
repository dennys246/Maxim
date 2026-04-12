"""Entorhinal Cortex (EC) - Multi-modal similarity engine.

Provides efficient similarity queries across multiple dimensions.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from maxim.similarity.indices import InvertedIndices
from maxim.similarity.lsh import LSHIndex, SemanticLSH
from maxim.similarity.signature import SituationSignature

# Phase 4: Neural semantic embeddings (optional)
try:
    from maxim.similarity.semantic import (
        NeuralSemanticLSH,
        SemanticEmbedderConfig,
        EmbeddingStore,
    )

    _NEURAL_SEMANTIC_AVAILABLE = True
except ImportError:
    NeuralSemanticLSH = None  # type: ignore
    SemanticEmbedderConfig = None  # type: ignore
    EmbeddingStore = None  # type: ignore
    _NEURAL_SEMANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors.

    Returns 0.0 for zero-norm vectors instead of NaN.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class ECConfig:
    """Configuration for Entorhinal Cortex."""

    # LSH settings
    num_lsh_tables: int = 4
    bits_per_table: int = 8

    # Query settings
    default_k: int = 10
    min_similarity: float = 0.3

    # Phase 4: Neural semantic embeddings
    enable_semantic: bool = False  # Enable neural semantic similarity
    semantic_model: str = "all-MiniLM-L6-v2"  # SentenceTransformer model
    async_embedding: bool = True  # Embed in background thread
    require_gpu: bool = False  # Require GPU for embeddings

    # Phase 4: Semantic embedding hash bits
    semantic_hash_bits: int = 16  # Number of bits for semantic LSH hash

    # Substrate (P1): pattern completion threshold for cosine similarity
    pattern_complete_threshold: float = 0.50


@dataclass
class PatternResult:
    """Result of EC pattern_complete_or_separate.

    Attributes:
        node_id: ATL node this percept mapped to (existing or new).
        similarity: Cosine similarity to the matched node (0.0 if new).
        is_new: True if a new node was created (separation).
    """

    node_id: str
    similarity: float
    is_new: bool


class EntorhinalCortex:
    """Entorhinal Cortex - Multi-modal similarity engine.

    Provides efficient multi-modal similarity queries across all memory
    components. Just as the biological EC serves as the gateway between
    the hippocampus and neocortex, this subsystem enables:

    - Fast similarity queries - O(1) approximate nearest neighbor via LSH
    - Multi-modal matching - Combine semantic, structural, temporal signals
    - Composite signatures - Compress features into hashable representations

    KNOWN LIMITATION (Phase 2):
    The semantic_hash dimension requires Phase 4's semantic embedding model.
    Until Phase 4 is implemented, all memories have identical semantic hashes,
    meaning similarity queries rely only on:
    - structural_hash (tool, outcome type)
    - temporal_hash (SCN bins)
    - context_hash (mode, detected objects)

    Impact: Queries like "find mug" won't match memories with "find cup"
    because semantic similarity isn't computed.

    Workaround: Use explicit filters (tool=X, goal=Y) instead of semantic search.

    Example:
        ec = EntorhinalCortex()

        # Register a memory
        signature = SituationSignature.from_memory(memory)
        ec.register(memory.id, signature)

        # Find similar situations
        query_sig = SituationSignature.from_memory(current_situation)
        similar = ec.find_similar(query_sig, k=5)
        for memory_id, score in similar:
            print(f"{memory_id}: {score:.2f}")

        # Query by structural features
        tool_memories = ec.query(tool="internet_search")
    """

    def __init__(self, config: ECConfig | None = None):
        self.config = config or ECConfig()

        # LSH index for approximate nearest neighbor
        self._lsh = LSHIndex(
            num_tables=self.config.num_lsh_tables,
            bits_per_table=self.config.bits_per_table,
        )

        # Inverted indices for structural queries
        self._inverted = InvertedIndices()

        # All signatures
        self._signatures: dict[str, SituationSignature] = {}

        # Optional semantic hasher (Phase 4)
        self._semantic_hasher: SemanticLSH | None = None
        self._neural_embedder: Any | None = None
        self._embedding_store: Any | None = None

        if self.config.enable_semantic:
            # Try neural semantic first (Phase 4), fallback to simple LSH
            if _NEURAL_SEMANTIC_AVAILABLE and NeuralSemanticLSH is not None:
                try:
                    embedder_config = SemanticEmbedderConfig(
                        model_name=self.config.semantic_model,
                        async_embedding=self.config.async_embedding,
                        require_gpu=self.config.require_gpu,
                    )
                    self._neural_embedder = NeuralSemanticLSH(config=embedder_config)
                    self._embedding_store = EmbeddingStore(embedder=self._neural_embedder)
                    logger.info(
                        "EC using neural semantic embeddings: %s",
                        self.config.semantic_model,
                    )
                except Exception as e:
                    logger.warning("Neural semantic init failed, using fallback: %s", e)
                    self._semantic_hasher = SemanticLSH()
            else:
                # Fallback to simple word-based LSH
                self._semantic_hasher = SemanticLSH()

        # Substrate (P1): dense embedding store for pattern completion.
        # Keyed by node_id, stores (embedding, modality) pairs.
        self._substrate_nodes: dict[str, tuple[list[float], str]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Substrate Pattern Completion (P1)
    # ─────────────────────────────────────────────────────────────────────────

    def pattern_complete_or_separate(
        self,
        embedding: list[float],
        modality: str,
        threshold: float | None = None,
        threshold_override: dict[str, float] | None = None,
    ) -> PatternResult:
        """Route an embedding to an existing ATL node or create a new one.

        Cosine similarity against all stored embeddings of the same
        modality. If the best match exceeds ``threshold``, returns that
        node (pattern completion). Otherwise creates a new node
        (pattern separation).

        Args:
            embedding: Dense vector from LinguisticEncoder.
            modality: Substrate modality ("text" or "vision").
            threshold: Override the default pattern_complete_threshold.
            threshold_override: Per-node threshold overrides keyed by
                node_id. Used by P2 reward modulation to widen the
                recognition radius for rewarded nodes.

        Returns:
            PatternResult with the node_id, similarity score, and
            whether a new node was created.
        """
        base_threshold = threshold if threshold is not None else self.config.pattern_complete_threshold
        overrides = threshold_override or {}

        best_node: str | None = None
        best_sim = -1.0

        for node_id, (stored_emb, stored_mod) in self._substrate_nodes.items():
            if stored_mod != modality:
                continue
            sim = _cosine_similarity(embedding, stored_emb)
            # Use per-node override if available, else base threshold
            node_thresh = overrides.get(node_id, base_threshold)
            if sim >= node_thresh and sim > best_sim:
                best_sim = sim
                best_node = node_id

        if best_node is not None:
            return PatternResult(node_id=best_node, similarity=best_sim, is_new=False)

        # Separation — allocate a new node ID but don't register yet.
        # The caller (LinguisticEncoder) registers via register_substrate_node
        # after ATL activation succeeds. This keeps EC stateless for the
        # separation path and allows the test harness to inspect without
        # side effects.
        new_id = str(uuid4())
        return PatternResult(node_id=new_id, similarity=0.0, is_new=True)

    def register_substrate_node(
        self,
        node_id: str,
        embedding: list[float],
        modality: str,
    ) -> None:
        """Register or update a substrate node's embedding."""
        self._substrate_nodes[node_id] = (embedding, modality)

    def remove_substrate_node(self, node_id: str) -> None:
        """Remove a substrate node."""
        self._substrate_nodes.pop(node_id, None)

    @property
    def substrate_node_count(self) -> int:
        """Number of substrate nodes registered."""
        return len(self._substrate_nodes)

    def register(
        self,
        memory_id: str,
        signature: SituationSignature | None = None,
        memory: Any = None,
    ) -> SituationSignature:
        """Register a memory with the EC.

        Args:
            memory_id: Unique memory identifier
            signature: Pre-computed signature (optional)
            memory: EpisodicMemory to create signature from (if signature not provided)

        Returns:
            The registered signature
        """
        if signature is None:
            if memory is None:
                raise ValueError("Either signature or memory must be provided")
            signature = SituationSignature.from_memory(memory, semantic_hasher=self._semantic_hasher)

        self._signatures[memory_id] = signature
        self._lsh.add(memory_id, signature)
        self._inverted.add(memory_id, signature)

        return signature

    def unregister(self, memory_id: str) -> None:
        """Remove a memory from the EC.

        Args:
            memory_id: Memory to remove
        """
        signature = self._signatures.pop(memory_id, None)
        if signature:
            self._lsh.remove(memory_id)
            self._inverted.remove(memory_id, signature)

    def remove_signature(self, memory_id: str) -> None:
        """Alias for unregister for deletion callback compatibility."""
        self.unregister(memory_id)

    def get_signature(self, memory_id: str) -> SituationSignature | None:
        """Get the signature for a memory."""
        return self._signatures.get(memory_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity Queries
    # ─────────────────────────────────────────────────────────────────────────

    def find_similar(
        self,
        signature: SituationSignature,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find k most similar memories.

        Args:
            signature: Query signature
            k: Number of results (default from config)
            min_similarity: Minimum similarity threshold (default from config)

        Returns:
            List of (memory_id, similarity_score) tuples
        """
        k = k or self.config.default_k
        min_sim = min_similarity or self.config.min_similarity

        # Query LSH index
        results = self._lsh.query(signature, k=k * 2, probe_radius=1)

        # Filter by minimum similarity
        filtered = [(mid, score) for mid, score in results if score >= min_sim]

        return filtered[:k]

    def find_similar_by_memory(
        self,
        memory_id: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find memories similar to an existing memory.

        Args:
            memory_id: Reference memory ID
            k: Number of results
            min_similarity: Minimum threshold

        Returns:
            List of (memory_id, similarity_score) tuples (excluding reference)
        """
        signature = self._signatures.get(memory_id)
        if not signature:
            return []

        results = self.find_similar(signature, k=k, min_similarity=min_similarity)

        # Exclude the reference memory
        return [(mid, score) for mid, score in results if mid != memory_id]

    def find_similar_situations_for_action(
        self,
        tool_name: str,
        context: dict[str, Any] | None = None,
        k: int = 10,
    ) -> list[tuple[str, SituationSignature]]:
        """Find similar situations where an action was taken.

        Useful for predicting outcomes of potential actions.

        Args:
            tool_name: Tool to find similar uses of
            context: Current context for filtering
            k: Maximum results

        Returns:
            List of (memory_id, signature) tuples
        """
        # Get all memories using this tool
        tool_memories = self._inverted.query_tool(tool_name)

        if not tool_memories:
            return []

        # Score by context similarity if provided
        if context:
            # Build a pseudo-signature for context matching
            mode = context.get("mode", "")
            scored = []
            for mid in tool_memories:
                sig = self._signatures.get(mid)
                if sig:
                    # Simple context match scoring
                    score = 0.5
                    if sig.mode == mode:
                        score += 0.3
                    if sig.outcome_type == "success":
                        score += 0.2
                    scored.append((mid, sig, score))

            scored.sort(key=lambda x: x[2], reverse=True)
            return [(mid, sig) for mid, sig, _ in scored[:k]]

        # No context, return most recent
        results = []
        for mid in list(tool_memories)[:k]:
            sig = self._signatures.get(mid)
            if sig:
                results.append((mid, sig))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Semantic Search (Phase 4)
    # ─────────────────────────────────────────────────────────────────────────

    def find_semantic(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find memories semantically similar to query text.

        Phase 4 feature: Uses neural embeddings for deep semantic similarity.
        "find mug" will match memories about "cup", "find greeting" matches
        "say hello".

        Falls back to structural similarity if semantic not enabled.

        Args:
            query: Natural language query
            k: Maximum results
            threshold: Minimum similarity (0-1)

        Returns:
            List of (memory_id, similarity) tuples
        """
        if self._embedding_store is not None:
            # Use neural semantic search
            return self._embedding_store.find_similar(query, k=k, threshold=threshold)

        # Fallback: use LSH-based similarity on existing signatures
        # This provides structural but not semantic similarity
        if self._semantic_hasher:
            query_hash = self._semantic_hasher.hash(query)
            results = []
            for mid, sig in self._signatures.items():
                similarity = self._semantic_hasher.estimated_similarity(query_hash, sig.semantic_hash)
                if similarity >= threshold:
                    results.append((mid, similarity))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]

        return []

    @property
    def semantic_enabled(self) -> bool:
        """Check if neural semantic similarity is available."""
        return self._neural_embedder is not None and self._neural_embedder.is_healthy

    # ─────────────────────────────────────────────────────────────────────────
    # Structural Queries
    # ─────────────────────────────────────────────────────────────────────────

    def query(
        self,
        tool: str | None = None,
        outcome: str | None = None,
        mode: str | None = None,
        hour: int | None = None,
        day: int | None = None,
    ) -> set[str]:
        """Query by structural features.

        All filters are AND-ed together.

        Args:
            tool: Tool name filter
            outcome: Outcome type filter
            mode: Mode filter
            hour: Hour bin filter (0-23)
            day: Day bin filter (0-6)

        Returns:
            Set of matching memory IDs
        """
        return self._inverted.query_intersection(tool=tool, outcome=outcome, mode=mode, hour=hour, day=day)

    def find_by_temporal(
        self,
        hour_bin: int | None = None,
        day_bin: int | None = None,
    ) -> set[str]:
        """Find memories by temporal context.

        Args:
            hour_bin: Hour of day (0-23)
            day_bin: Day of week (0-6)

        Returns:
            Set of matching memory IDs
        """
        return self._inverted.query_intersection(hour=hour_bin, day=day_bin)

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return EC statistics."""
        stats = {
            "total_signatures": len(self._signatures),
            "lsh_entries": len(self._lsh),
            "inverted_indices": self._inverted.stats(),
            "semantic_enabled": self.config.enable_semantic,
            "neural_semantic_available": self._neural_embedder is not None,
            "substrate_nodes": len(self._substrate_nodes),
        }

        # Add neural embedder stats if available
        if self._neural_embedder is not None:
            stats["neural_embedder"] = self._neural_embedder.stats()

        if self._embedding_store is not None:
            stats["embeddings_stored"] = len(self._embedding_store)

        return stats

    def __len__(self) -> int:
        """Number of registered signatures."""
        return len(self._signatures)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save EC state to JSON file."""
        data = {
            "version": "1.0",
            "config": {
                "num_lsh_tables": self.config.num_lsh_tables,
                "bits_per_table": self.config.bits_per_table,
                "default_k": self.config.default_k,
                "min_similarity": self.config.min_similarity,
                "enable_semantic": self.config.enable_semantic,
            },
            "lsh": self._lsh.serialize(),
            "inverted": self._inverted.to_dict(),
            "signatures": {k: v.to_dict() for k, v in self._signatures.items()},
            "substrate_nodes": {
                nid: {"embedding": emb, "modality": mod} for nid, (emb, mod) in self._substrate_nodes.items()
            },
        }

        from maxim.utils.atomic_io import atomic_write_json

        atomic_write_json(path, data)

        logger.info(
            "Saved EC to %s (%d signatures, %d substrate nodes)",
            path,
            len(self._signatures),
            len(self._substrate_nodes),
        )

    def load(self, path: str) -> None:
        """Load EC state from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported EC version: {version}")

        # Load config
        cfg_data = data.get("config", {})
        self.config = ECConfig(
            num_lsh_tables=cfg_data.get("num_lsh_tables", 4),
            bits_per_table=cfg_data.get("bits_per_table", 8),
            default_k=cfg_data.get("default_k", 10),
            min_similarity=cfg_data.get("min_similarity", 0.3),
            enable_semantic=cfg_data.get("enable_semantic", False),
        )

        # Load LSH
        self._lsh = LSHIndex(
            num_tables=self.config.num_lsh_tables,
            bits_per_table=self.config.bits_per_table,
        )
        self._lsh.deserialize(data.get("lsh", {}))

        # Load inverted indices
        self._inverted = InvertedIndices.from_dict(data.get("inverted", {}))

        # Load signatures
        self._signatures = {k: SituationSignature.from_dict(v) for k, v in data.get("signatures", {}).items()}

        # Load substrate nodes (P1)
        self._substrate_nodes = {}
        for nid, ndata in data.get("substrate_nodes", {}).items():
            self._substrate_nodes[nid] = (ndata["embedding"], ndata["modality"])

        logger.info(
            "Loaded EC from %s (%d signatures, %d substrate nodes)",
            path,
            len(self._signatures),
            len(self._substrate_nodes),
        )

    def get_version(self) -> str:
        """Return data format version."""
        return "1.0"


__all__ = ["EntorhinalCortex", "ECConfig", "PatternResult", "_cosine_similarity"]
