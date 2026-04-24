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


class AffordanceDecompositionStrategy:
    """Split underscore-joined affordance identifiers into constituent concepts.

    Designed for SEM affordance names (``fire_breath``, ``tail_sweep``),
    NOT natural language.  Does not use spaCy — the input is domain-specific
    identifiers where underscore is the only word boundary.

    Returns the compound name (underscores replaced with spaces) PLUS each
    individual word as a separate chunk.  Single-word affordances pass
    through unchanged.

    Examples::

        "fire_breath" → ["fire breath", "fire", "breath"]
        "tail_sweep"  → ["tail sweep", "tail", "sweep"]
        "circle"      → ["circle"]
        "poison_cloud_breath" → ["poison cloud breath", "poison", "cloud", "breath"]
    """

    def __init__(self, *, min_part_len: int = 2) -> None:
        self._min_part_len = min_part_len

    def extract(self, text: str) -> list[ConceptChunk]:
        text = text.strip()
        if not text:
            return [ConceptChunk(text="", span=(0, 0))]

        normalized = text.replace("_", " ")
        parts = normalized.split()

        chunks = [ConceptChunk(text=normalized, span=(0, len(normalized)))]
        if len(parts) > 1:
            for part in parts:
                if len(part) >= self._min_part_len:
                    chunks.append(ConceptChunk(text=part, span=None))
        return chunks


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

            # Stage 2: extract relation from dependency parse.
            # The root token of each noun chunk has a dependency label
            # that reveals its syntactic role in the sentence.
            relation = _classify_relation(nc.root)

            chunks.append(
                ConceptChunk(
                    text=clean_text,
                    span=(doc[token_start].idx, nc.end_char),
                    confidence=1.0,
                    relation=relation,
                )
            )

        # If no noun chunks found (e.g., single word, imperative),
        # fall back to the full text.
        if not chunks:
            return [ConceptChunk(text=text.strip(), span=(0, len(text)))]

        return chunks


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — relation classification from dependency parse
# ─────────────────────────────────────────────────────────────────────────

# Dependency labels → relation type mapping.
# spaCy dep labels: https://spacy.io/api/annotation#dependency-parsing
_DEP_TO_RELATION: dict[str, str] = {
    # Spatial: prepositional attachment indicating location
    "pobj": "spatial",  # "mug on the table" — table is pobj of "on"
    # Possessive: possessive modifier
    "poss": "possessive",  # "the cat's paw"
    # Temporal: temporal modifiers
    "npadvmod": "temporal",  # "last Tuesday"
    # Subject/object (action relation to the verb)
    "nsubj": "action",  # "the cat sat"
    "nsubjpass": "action",
    "dobj": "action",  # "I see a mug"
    # Appositive: identity / descriptive
    "appos": "descriptive",  # "Bob, the engineer"
    # Attribute: predicate nominal
    "attr": "descriptive",  # "this is a mug"
}

# Preposition text → relation type for pobj refinement.
_PREP_SPATIAL = {
    "on",
    "in",
    "at",
    "near",
    "beside",
    "above",
    "below",
    "under",
    "behind",
    "between",
    "inside",
    "outside",
    "next",
}
_PREP_TEMPORAL = {"before", "after", "during", "since", "until", "by"}
_PREP_POSSESSIVE = {"of", "with"}


def _classify_relation(root_token: Any) -> str | None:
    """Classify the syntactic relation of a noun chunk's root token.

    Returns a coarse relation label (spatial, possessive, temporal,
    action, descriptive) or None if the relation is unclassifiable.
    """
    dep = root_token.dep_

    # For prepositional objects, refine by the actual preposition.
    if dep == "pobj" and root_token.head.pos_ == "ADP":
        prep_text = root_token.head.text.lower()
        if prep_text in _PREP_SPATIAL:
            return "spatial"
        if prep_text in _PREP_TEMPORAL:
            return "temporal"
        if prep_text in _PREP_POSSESSIVE:
            return "possessive"
        return None  # unknown preposition — leave edge untagged

    return _DEP_TO_RELATION.get(dep)


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
