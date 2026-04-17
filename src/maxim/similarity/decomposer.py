"""Concept decomposition — break input text into concept-level chunks.

Provides a pluggable ``DecompositionStrategy`` Protocol so different
NLP backends (spaCy, LLM-based, regex, domain-specific) can be swapped
without touching the downstream pipeline. The default strategy uses
spaCy's noun chunker (``en_core_web_sm``).

Usage::

    decomposer = ConceptDecomposer()  # auto-detects spaCy or falls back
    chunks = decomposer.extract("I see a blue mug on the table")
    # [ConceptChunk(text="blue mug", ...), ConceptChunk(text="table", ...)]

The output ``list[ConceptChunk]`` carries metadata (span offsets,
confidence, relation type) so Stage 2 role-tagged edges can be added
without breaking the Stage 1 interface.

Integration: lives inside ``LinguisticEncoder.encode_decomposed()``
so every consumer (agent loop, sims, headless API, embodied runtime)
gets decomposition for free.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConceptChunk:
    """A single concept extracted from an input string.

    ``text`` is the concept surface form passed to the encoder.
    Optional fields support Stage 2 role-tagged edges and diagnostics.
    """

    text: str
    span: tuple[int, int] | None = None  # character offsets in the original
    confidence: float = 1.0
    relation: str | None = None  # Stage 2: "spatial", "possessive", etc.


# ─────────────────────────────────────────────────────────────────────────
# Strategy Protocol
# ─────────────────────────────────────────────────────────────────────────


@runtime_checkable
class DecompositionStrategy(Protocol):
    """Protocol for concept extraction backends.

    Implementations must be stateless or thread-safe — the encoder
    may call ``extract`` from multiple agent threads concurrently
    under ``AgentPool``.
    """

    def extract(self, text: str) -> list[ConceptChunk]: ...


# ─────────────────────────────────────────────────────────────────────────
# Built-in strategies
# ─────────────────────────────────────────────────────────────────────────


class IdentityStrategy:
    """Fallback strategy: returns the input as a single chunk.

    Used when spaCy is not installed or decomposition is disabled.
    """

    def extract(self, text: str) -> list[ConceptChunk]:
        return [ConceptChunk(text=text, span=(0, len(text)))]


class SpaCyNounChunkStrategy:
    """Default strategy using spaCy noun chunker.

    Lazy-loads ``en_core_web_sm`` on first call. Thread-safe via lock.
    """

    _nlp: Any = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str = "en_core_web_sm", min_chunk_len: int = 2) -> None:
        self._model_name = model_name
        self._min_chunk_len = min_chunk_len

    def _ensure_model(self) -> Any:
        if SpaCyNounChunkStrategy._nlp is not None:
            return SpaCyNounChunkStrategy._nlp

        with SpaCyNounChunkStrategy._lock:
            # Double-check after acquiring lock
            if SpaCyNounChunkStrategy._nlp is not None:
                return SpaCyNounChunkStrategy._nlp

            import spacy

            logger.info("loading spaCy model '%s' for concept decomposition...", self._model_name)
            SpaCyNounChunkStrategy._nlp = spacy.load(self._model_name)
            return SpaCyNounChunkStrategy._nlp

    def extract(self, text: str) -> list[ConceptChunk]:
        nlp = self._ensure_model()
        doc = nlp(text)

        chunks: list[ConceptChunk] = []
        for nc in doc.noun_chunks:
            # Strip leading determiners/pronouns ("the", "a", "this")
            # to get cleaner concept text.
            token_start = nc.start
            while token_start < nc.end and doc[token_start].pos_ in ("DET", "PRON"):
                token_start += 1

            if token_start >= nc.end:
                continue  # chunk was entirely determiners/pronouns

            clean_text = doc[token_start : nc.end].text.strip()
            if len(clean_text) < self._min_chunk_len:
                continue

            chunks.append(
                ConceptChunk(
                    text=clean_text,
                    span=(doc[token_start].idx, nc.end_char),
                    confidence=1.0,
                )
            )

        # If no noun chunks found (e.g., single word, imperative),
        # fall back to the full text.
        if not chunks:
            return [ConceptChunk(text=text.strip(), span=(0, len(text)))]

        return chunks


# ─────────────────────────────────────────────────────────────────────────
# Coordinator
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class DecomposerConfig:
    """Configuration for concept decomposition."""

    enabled: bool = True
    min_chunk_len: int = 2
    spacy_model: str = "en_core_web_sm"


class ConceptDecomposer:
    """Coordinator that delegates to a ``DecompositionStrategy``.

    Auto-detects spaCy on construction. If spaCy is not installed or
    ``enabled=False``, falls back to ``IdentityStrategy`` (returns the
    input unchanged).
    """

    def __init__(
        self,
        strategy: DecompositionStrategy | None = None,
        config: DecomposerConfig | None = None,
    ) -> None:
        self.config = config or DecomposerConfig()

        if strategy is not None:
            self._strategy = strategy
        elif not self.config.enabled:
            self._strategy = IdentityStrategy()
        else:
            self._strategy = _auto_detect_strategy(self.config)

    def extract(self, text: str) -> list[ConceptChunk]:
        """Extract concept chunks from input text.

        Returns a list of ``ConceptChunk`` objects. Always returns at
        least one chunk (the full text as fallback).
        """
        if not text or not text.strip():
            return [ConceptChunk(text=text or "", span=(0, len(text or "")))]
        return self._strategy.extract(text)

    @property
    def strategy_name(self) -> str:
        """Human-readable name of the active strategy."""
        return type(self._strategy).__name__


def _auto_detect_strategy(config: DecomposerConfig) -> DecompositionStrategy:
    """Auto-detect the best available strategy."""
    try:
        import spacy  # noqa: F401

        return SpaCyNounChunkStrategy(
            model_name=config.spacy_model,
            min_chunk_len=config.min_chunk_len,
        )
    except ImportError:
        logger.debug("spaCy not installed; concept decomposition using identity fallback")
        return IdentityStrategy()
