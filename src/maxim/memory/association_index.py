"""AssociationIndex: keyword-based memory indexing and similarity lookup.

Extracted from memory_agent.py for modularity.

Two-tier lookup:
1. Keyword overlap (fast, coarse) - always available
2. LSH similarity (slow, precise) - optional, via set_context_index()
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any


class AssociationIndex:
    """Index for fast similarity-based memory retrieval.

    After Phase 0 unification, this index stores memory IDs and keywords
    only — full records are resolved from Hippocampus on demand.

    Two-tier lookup:
    1. Keyword overlap (fast, coarse) - always available
    2. Embedding similarity (slow, precise) - optional, lazy-computed
    """

    STOPWORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "and",
            "or",
            "in",
            "on",
            "at",
            "for",
            "it",
            "this",
            "that",
            "with",
        }
    )

    def __init__(self, embedding_model: str | None = None) -> None:
        self._keyword_index: dict[str, set[str]] = defaultdict(set)
        self._memory_keywords: dict[str, set[str]] = {}  # mid → keywords
        self._embedding_model = embedding_model
        self._embedder = None  # Lazy init
        self._lock = threading.Lock()
        # Phase 3e: LSH-backed context similarity index
        self._context_index: Any = None  # SimilarityIndex, set via set_context_index()

    def set_context_index(self, index: Any) -> None:
        """Wire a SimilarityIndex for LSH-backed recall (Phase 3e)."""
        self._context_index = index

    def add_by_id(self, memory_id: str, content: Any) -> None:
        """Index a memory by keywords extracted from content."""
        with self._lock:
            keywords = self._extract_keywords(content)
            self._memory_keywords[memory_id] = keywords
            for kw in keywords:
                self._keyword_index[kw].add(memory_id)
            # Also register in LSH index if available
            if self._context_index is not None:
                text = content if isinstance(content, str) else " ".join(
                    str(v) for v in (content.values() if isinstance(content, dict) else [str(content)])
                    if v
                )
                self._context_index.register(memory_id, text)

    def remove(self, memory_id: str) -> None:
        """Remove memory from index."""
        with self._lock:
            kws = self._memory_keywords.pop(memory_id, set())
            for kw in kws:
                self._keyword_index[kw].discard(memory_id)
            if self._context_index is not None:
                self._context_index.remove(memory_id)

    def find_similar(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find similar memory IDs by similarity.

        Uses LSH (Phase 3e) when available, falls back to keyword Jaccard.
        Returns (memory_id, score) tuples sorted by score descending.
        """
        with self._lock:
            # Phase 3e: Prefer LSH similarity when index is populated
            if (
                self._context_index is not None
                and self._context_index.signatures
            ):
                query_text = str(query) if not isinstance(query, str) else query
                lsh_results = self._context_index.query_similar(
                    query_text, min_similarity=0.3
                )
                if lsh_results:
                    return lsh_results[:top_k]

            # Fallback: keyword Jaccard similarity
            query_keywords = self._extract_keywords(query)

            candidates: dict[str, float] = {}
            for kw in query_keywords:
                for mid in self._keyword_index.get(kw, set()):
                    if mid not in candidates:
                        candidates[mid] = 0.0
                    mem_kw = self._memory_keywords.get(mid, set())
                    if mem_kw:
                        intersection = len(query_keywords & mem_kw)
                        union = len(query_keywords | mem_kw)
                        if union > 0:
                            candidates[mid] = max(candidates[mid], intersection / union)

            sorted_results = sorted(
                candidates.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            return sorted_results[:top_k]

    def _extract_keywords(self, content: Any) -> set[str]:
        """Extract keywords from content for indexing."""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):
            text = " ".join(str(v) for v in content.values() if v)
        elif hasattr(content, "raw_transcript_text"):
            text = getattr(content, "raw_transcript_text", "") or ""
        else:
            text = str(content)

        words = text.lower().split()
        return {w for w in words if len(w) > 2 and w not in self.STOPWORDS}
