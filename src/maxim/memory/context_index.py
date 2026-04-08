"""SimilarityIndex — LSH-based similarity index for O(1) memory recall.

Replaces keyword Jaccard (AssociationIndex) and O(n) linear scans
(hippocampus.recall_similar) with MinHash + Locality-Sensitive Hashing.

Two instances are used system-wide:
- context_index: language/context similarity (upgrades AssociationIndex)
- percept_index: percept similarity (upgrades recall_similar)

Phase 3e of the consolidation expansion plan.
"""

from __future__ import annotations

import json
from typing import Any


class SimilarityIndex:
    """LSH-based similarity index that upgrades existing recall backends.

    Uses MinHash + LSH (Locality-Sensitive Hashing) to find memories with
    similar textual context without full embedding comparison. O(1) lookup
    per query instead of O(n) linear scan.

    Not a standalone recall system — slots into:
    - AssociationIndex.find_similar() (replaces keyword Jaccard)
    - hippocampus.recall_similar() (replaces object intersection)
    - ConsolidationOrchestrator (recurrence detection)
    """

    def __init__(self, num_hashes: int = 64, num_bands: int = 8) -> None:
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.rows_per_band = num_hashes // num_bands
        # Each band maps a hash signature → set of memory IDs
        self.bands: list[dict[tuple[int, ...], set[str]]] = [{} for _ in range(num_bands)]
        self.signatures: dict[str, list[int]] = {}  # memory_id → minhash signature

    def _shingle(self, text: str, k: int = 2) -> set[str]:
        """Convert text to k-shingles (word n-grams).

        Uses word-level shingles instead of character-level because
        Maxim's context strings are short and structured.
        """
        words = text.lower().split()
        if len(words) < k:
            return {text.lower().strip()} if words else set()
        return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}

    def _minhash(self, shingles: set[str]) -> list[int]:
        """Compute MinHash signature for a set of shingles."""
        sig = []
        for i in range(self.num_hashes):
            min_hash = min(hash((i, s)) & 0xFFFFFFFF for s in shingles) if shingles else 0
            sig.append(min_hash)
        return sig

    def register(self, memory_id: str, text: str) -> None:
        """Index a memory's text context for fast lookup."""
        if not text or len(text) < 10:
            return
        shingles = self._shingle(text)
        sig = self._minhash(shingles)
        self.signatures[memory_id] = sig

        # Insert into LSH bands
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = tuple(sig[start:end])
            if band_hash not in self.bands[band_idx]:
                self.bands[band_idx][band_hash] = set()
            self.bands[band_idx][band_hash].add(memory_id)

    def query_similar(self, text: str, min_similarity: float = 0.3) -> list[tuple[str, float]]:
        """Find memory IDs with similar text context.

        Returns list of (memory_id, estimated_jaccard_similarity) pairs,
        sorted by similarity descending.
        """
        shingles = self._shingle(text)
        query_sig = self._minhash(shingles)

        # Collect candidates from any matching band
        candidate_ids: set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = tuple(query_sig[start:end])
            if band_hash in self.bands[band_idx]:
                candidate_ids.update(self.bands[band_idx][band_hash])

        # Estimate Jaccard similarity from MinHash signatures
        results = []
        for mid in candidate_ids:
            stored_sig = self.signatures[mid]
            matches = sum(1 for a, b in zip(query_sig, stored_sig) if a == b)
            similarity = matches / self.num_hashes
            if similarity >= min_similarity:
                results.append((mid, similarity))

        return sorted(results, key=lambda x: -x[1])

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index."""
        if memory_id not in self.signatures:
            return
        sig = self.signatures.pop(memory_id)
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = tuple(sig[start:end])
            if band_hash in self.bands[band_idx]:
                self.bands[band_idx][band_hash].discard(memory_id)

    def save(self, path: str) -> None:
        """Persist index to disk. Atomic write to prevent corruption."""
        data: dict[str, Any] = {
            "version": "1.0",
            "num_hashes": self.num_hashes,
            "num_bands": self.num_bands,
            "signatures": self.signatures,
            "bands": [{json.dumps(list(k)): list(v) for k, v in band.items()} for band in self.bands],
        }
        from maxim.utils.atomic_io import atomic_write_json
        atomic_write_json(path, data, indent=None)

    @classmethod
    def load(cls, path: str) -> SimilarityIndex:
        """Load index from disk."""
        with open(path) as f:
            data = json.load(f)
        idx = cls(data["num_hashes"], data["num_bands"])
        idx.signatures = data["signatures"]
        idx.bands = [{tuple(json.loads(k)): set(v) for k, v in band.items()} for band in data["bands"]]
        return idx

    def __len__(self) -> int:
        return len(self.signatures)


def percept_to_canonical(percept: dict[str, Any]) -> str:
    """Convert a percept dict to a canonical string for LSH indexing.

    Extracts object labels + confidence, person labels + posture,
    and speech content. Spatial fields (distance, bearing) are omitted
    — they change between sightings of the same object.

    Returns "" for empty/malformed percepts.
    """
    parts: list[str] = []
    for obj in percept.get("objects", []):
        if "label" not in obj:
            continue
        parts.append(f"{obj['label']} {obj.get('confidence', 0):.1f}")
    for person in percept.get("people", []):
        if "label" not in person:
            continue
        parts.append(f"{person['label']} {person.get('posture', '')}")
    if percept.get("speech"):
        parts.append(percept["speech"])
    return " ".join(parts)


__all__ = ["SimilarityIndex", "percept_to_canonical"]
