"""Locality-Sensitive Hashing for approximate nearest neighbor search.

Enables O(1) similarity queries by hashing similar items to the same bucket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from maxim.similarity.signature import SituationSignature
from maxim.utils.seeding import stable_hash_64_signed


@dataclass
class SemanticLSH:
    """Locality-Sensitive Hashing for semantic similarity.

    Uses random hyperplane hashing to map high-dimensional
    embeddings to binary hash codes. Similar embeddings will
    have similar hash codes with high probability.

    This is a fallback implementation that works without numpy.
    For production use with embeddings, use numpy-based version.

    Time Complexity:
    - hash(): O(d * num_planes) where d = embedding dimension
    - Comparison: O(num_planes) Hamming distance

    Attributes:
        num_planes: Number of hash bits
        seed: Random seed for reproducibility
    """

    num_planes: int = 8
    seed: int = 42

    def hash(self, text: str) -> tuple[int, ...]:
        """Hash text to LSH buckets.

        This is a fallback implementation using simple word hashing.
        For better results, use with actual embeddings.

        Args:
            text: Text to hash

        Returns:
            Tuple of hash bits (0 or 1)
        """
        if not text:
            return (0,) * self.num_planes

        words = text.lower().split()
        bits = []

        for i in range(self.num_planes):
            # Hash words and sum with plane-specific salt.
            # stable_hash_64_signed, NOT builtin hash(): the `seed` param
            # routed through randomized hash() anyway, so this LOOKED
            # deterministic and was not. Output lands in
            # SituationSignature.semantic_hash, persisted via EC.save().
            h = sum(stable_hash_64_signed(f"{w}:{i}:{self.seed}") for w in words)
            bits.append(1 if h > 0 else 0)

        return tuple(bits)

    def estimated_similarity(self, hash1: tuple[int, ...], hash2: tuple[int, ...]) -> float:
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
        # agreement_ratio ≈ 1 - θ/π, so cos(θ) ≈ cos(π * (1 - agreement))
        estimated_angle = math.pi * (1 - agreement_ratio)
        return max(0.0, math.cos(estimated_angle))


@dataclass
class LSHIndex:
    """Multi-probe LSH index for fast approximate nearest neighbor.

    Uses multiple hash tables with different random projections
    to increase recall while maintaining O(1) query time.

    Attributes:
        num_tables: Number of hash tables
        bits_per_table: Hash bits per table
    """

    num_tables: int = 4
    bits_per_table: int = 8

    _tables: list[dict[tuple[int, ...], list[str]]] = field(default_factory=list)
    _hashers: list[SemanticLSH] = field(default_factory=list)
    _signatures: dict[str, SituationSignature] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize hash tables."""
        if not self._tables:
            self._tables = [{} for _ in range(self.num_tables)]

        if not self._hashers:
            self._hashers = [SemanticLSH(num_planes=self.bits_per_table, seed=42 + i) for i in range(self.num_tables)]

    def add(self, memory_id: str, signature: SituationSignature) -> None:
        """Add a memory to the LSH index.

        Args:
            memory_id: Unique memory identifier
            signature: Situation signature to index
        """
        self._signatures[memory_id] = signature

        # Add to each hash table
        for i, table in enumerate(self._tables):
            # Use subset of semantic hash for this table
            bucket = signature.semantic_hash[: self.bits_per_table]
            if len(bucket) < self.bits_per_table:
                # Pad if needed
                bucket = bucket + (0,) * (self.bits_per_table - len(bucket))

            if bucket not in table:
                table[bucket] = []
            if memory_id not in table[bucket]:
                table[bucket].append(memory_id)

    def query(
        self,
        signature: SituationSignature,
        k: int = 10,
        probe_radius: int = 1,
    ) -> list[tuple[str, float]]:
        """Find k most similar memories.

        Args:
            signature: Query signature
            k: Number of results to return
            probe_radius: Hamming distance to probe (0 = exact, 1 = ±1 bit)

        Returns:
            List of (memory_id, similarity_score) tuples
        """
        candidates: set[str] = set()

        # Collect candidates from all tables
        for i, table in enumerate(self._tables):
            bucket = signature.semantic_hash[: self.bits_per_table]
            if len(bucket) < self.bits_per_table:
                bucket = bucket + (0,) * (self.bits_per_table - len(bucket))

            # Exact bucket match
            if bucket in table:
                candidates.update(table[bucket])

            # Multi-probe: check nearby buckets
            if probe_radius > 0:
                for flip_idx in range(len(bucket)):
                    probed = list(bucket)
                    probed[flip_idx] = 1 - probed[flip_idx]
                    probed_tuple = tuple(probed)
                    if probed_tuple in table:
                        candidates.update(table[probed_tuple])

        # Rank candidates by full similarity
        scored = []
        for mid in candidates:
            cand_sig = self._signatures.get(mid)
            if cand_sig:
                score = signature.similarity(cand_sig)
                scored.append((mid, score))

        # Sort by score and return top k
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index.

        Args:
            memory_id: Memory to remove
        """
        if memory_id not in self._signatures:
            return

        signature = self._signatures.pop(memory_id)

        for table in self._tables:
            bucket = signature.semantic_hash[: self.bits_per_table]
            if len(bucket) < self.bits_per_table:
                bucket = bucket + (0,) * (self.bits_per_table - len(bucket))

            if bucket in table and memory_id in table[bucket]:
                table[bucket].remove(memory_id)
                if not table[bucket]:
                    del table[bucket]

    def __len__(self) -> int:
        """Number of indexed signatures."""
        return len(self._signatures)

    def serialize(self) -> dict[str, Any]:
        """Serialize index to dictionary."""
        return {
            "num_tables": self.num_tables,
            "bits_per_table": self.bits_per_table,
            "tables": [{str(k): v for k, v in table.items()} for table in self._tables],
            "signatures": {k: v.to_dict() for k, v in self._signatures.items()},
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Deserialize index from dictionary."""
        self.num_tables = data.get("num_tables", 4)
        self.bits_per_table = data.get("bits_per_table", 8)

        self._tables = []
        for table_data in data.get("tables", []):
            table = {}
            for k, v in table_data.items():
                # Parse tuple key from string
                key = tuple(int(x) for x in k.strip("()").split(",") if x.strip())
                table[key] = v
            self._tables.append(table)

        # Ensure we have enough tables
        while len(self._tables) < self.num_tables:
            self._tables.append({})

        self._hashers = [SemanticLSH(num_planes=self.bits_per_table, seed=42 + i) for i in range(self.num_tables)]

        self._signatures = {k: SituationSignature.from_dict(v) for k, v in data.get("signatures", {}).items()}


__all__ = ["SemanticLSH", "LSHIndex"]
