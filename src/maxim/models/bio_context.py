"""Standardized context dataclasses for bio-system protocol enrichment.

Each bio-system's primary query/learning methods accept an optional
``context: *Context | None = None`` parameter.  Current (v1)
implementations ignore it — the parameter exists so that post-1.0
backends (real TD learning, embedding-space retrieval, temporal context
vectors) can consume richer input WITHOUT breaking the frozen Protocol
signatures.

All dataclasses are frozen + slots for immutability and low overhead.
Fields are *hints*, not commands — a backend that doesn't support a
field simply ignores it.  Callers cannot assume the backend used it.

Import site::

    from maxim.models.bio_context import (
        RetrievalContext,
        PredictionContext,
        SemanticContext,
        EncodingContext,
        TemporalContext,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Hippocampus — retrieval context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    """Optional context for hippocampal retrieval operations.

    A biologically-refined backend would use these fields to implement
    mood-congruent recall, goal-directed filtering at the retrieval
    level, consolidation-aware ranking, and temporal windowing.
    """

    temporal_window_s: float | None = None
    """If set, restrict retrieval to memories within this many seconds
    of the current time.  None means no temporal restriction."""

    emotional_valence: float | None = None
    """Current emotional state (-1.0 negative … +1.0 positive).
    Mood-congruent recall: positive valence biases toward positive
    memories and vice versa."""

    active_goal: str | None = None
    """Current agent goal text.  A goal-aware backend uses this to
    boost relevance of memories tagged with a matching goal."""

    min_consolidation: float | None = None
    """Minimum consolidation level (0.0–1.0).  Higher values restrict
    retrieval to well-consolidated memories; lower values include
    fragile recent traces."""

    encoding_specificity: float | None = None
    """0.0 = pure gist matching, 1.0 = exact literal matching.
    Controls how strict the retrieval criterion is."""

    energy: float | None = None
    """Current SCN energy level (0.0–1.0).  Low energy may narrow
    retrieval scope (metabolic constraint on search)."""


# ---------------------------------------------------------------------------
# NAc — prediction / learning context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Optional context for NAc prediction queries and learning updates.

    A richer backend would use these to implement arousal-modulated
    learning rates, temporal discounting, and surprise-driven
    plasticity.
    """

    arousal: float | None = None
    """Current arousal level (0.0–1.0).  High arousal increases
    effective learning rate; low arousal dampens it."""

    temporal_discount: float | None = None
    """Discount factor for temporally distant outcomes (0.0–1.0).
    Lower values discount future outcomes more heavily."""

    energy: float | None = None
    """Current SCN energy level (0.0–1.0).  May modulate reward
    sensitivity (hungrier agents value food more)."""

    active_goal: str | None = None
    """Current goal text.  Goal-conditioned prediction allows the
    same event to have different expected outcomes under different
    goals."""

    surprise: float | None = None
    """Reward prediction error magnitude (0.0+).  Higher surprise
    can boost the learning rate for this update."""


# ---------------------------------------------------------------------------
# ATL — semantic concept context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SemanticContext:
    """Optional context for ATL concept queries and formation.

    A richer backend would use these to constrain spreading
    activation depth, filter by semantic domain, and control
    abstraction level.
    """

    abstraction_level: float | None = None
    """0.0 = concrete entity, 1.0 = maximally abstract category.
    Controls whether retrieval prefers specific exemplars or
    general categories."""

    domain_hints: tuple[str, ...] = field(default_factory=tuple)
    """Semantic domain constraints (e.g., ("combat", "fire")).
    If non-empty, restrict retrieval to concepts in these domains."""

    spreading_depth: int | None = None
    """Maximum hops for spreading activation (default backend-defined).
    Lower values return more directly related concepts."""

    spreading_decay: float | None = None
    """Activation decay per hop (0.0–1.0).  Higher values spread
    activation more broadly; lower values concentrate it."""

    active_goal: str | None = None
    """Current goal text.  Goal-relevant concepts may receive an
    activation boost."""


# ---------------------------------------------------------------------------
# EC — encoding / pattern context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EncodingContext:
    """Optional context for EC pattern completion/separation operations.

    A richer backend would use these to adjust grid cell resolution,
    apply attention weighting, and hint at decomposition strategy.
    """

    resolution: float | None = None
    """Grid cell resolution (0.0 = coarse, 1.0 = fine-grained).
    Coarse resolution merges more aggressively; fine resolution
    preserves distinctions."""

    attention_weights: tuple[float, ...] | None = None
    """Per-dimension attention weights for the embedding space.
    If provided, must match the embedding dimensionality.  Dimensions
    with higher weights matter more for similarity."""

    encoding_mode: str | None = None
    """Hint: ``"encode"`` vs ``"retrieve"``.  Some backends use
    different thresholds in encoding mode (favor separation) vs
    retrieval mode (favor completion)."""

    decomposition_hint: str | None = None
    """Hint for concept decomposition strategy.  E.g., ``"noun_chunk"``
    for SpaCy-based decomposition, ``"affordance"`` for underscore-split
    affordance names.  None means use the default strategy."""


# ---------------------------------------------------------------------------
# SCN — temporal context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """Optional context for SCN temporal registration and queries.

    A richer backend would use these to incorporate external zeitgeber
    signals, ultradian rhythm coupling, and oscillator phase.
    """

    oscillator_phase: float | None = None
    """Current dominant oscillator phase (0.0–2π).  A phase-aware
    backend can align temporal binning with the oscillator's
    current position."""

    prediction_confidence: float | None = None
    """Confidence in the SCN's temporal prediction for the current
    event (0.0–1.0).  Low confidence may widen temporal bins."""

    zeitgeber_strength: float | None = None
    """External zeitgeber signal strength (0.0–1.0).  Strong
    zeitgebers (regular user interaction patterns) tighten phase
    locking."""

    energy: float | None = None
    """Current energy level (0.0–1.0).  Circadian phase interacts
    with homeostatic sleep pressure — low energy in a high-energy
    phase may signal anomaly."""
