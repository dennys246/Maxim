"""ValenceSignal — abstract reward/punishment signal from any bio-system.

A transport type carrying a pre-normalized float value from producers
(NAc, PainBus, EC novelty, TemporalCreditDistributor) to consumers
(WorkingMemorySet entries for thought salience modulation).

Consumer source-blindness invariant: consumers use ``value`` only.
``source`` exists for observability (bio_telemetry JSONL, sim_log traces,
debugging).  Bio-region-specific weighting belongs at the producer side —
each producer emits an appropriately-scaled ``value`` that already encodes
its relative importance.

NOT a replacement for:
- ``Valence`` enum in ``decisions/causal_link.py`` (discrete, on CausalLink)
- ``Episode.valence`` float in ``memory/episode.py`` (continuous, on Episode)

Those are storage types; this is a transport type.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValenceSignal:
    """Abstract reward/punishment signal from any bio-system.

    Produced by NAc (outcome valence), PainBus (negative),
    EC (novelty as mild positive), TemporalCreditDistributor.
    Consumed by WMS entries to modulate thought salience over time.

    SHAPE-FROZEN at 1.0 (CC3). Every field is load-bearing for the
    transport contract; an ``extra`` dict would tempt consumers to
    branch on producer-private side data, violating the
    consumer-source-blindness invariant declared in the module
    docstring. New fields appended at the end with sensible defaults
    are non-breaking; adding a *required* field post-1.0 is a
    major-version-bump change.
    """

    value: float
    """Pre-normalized value in [-1.0, +1.0].  Consumers use this only."""

    source: str
    """Producer identity for logging/debugging.  Consumers must NOT
    branch on this field.  Known sources: "nac", "pain", "ec_novelty",
    "temporal_credit"."""

    goal_tag: str | None
    """Active goal when this signal was emitted.  None = goalless."""

    timestamp: float
    """Unix timestamp of emission."""

    @classmethod
    def from_credit(
        cls,
        value: float,
        goal_tag: str | None = None,
        source: str = "temporal_credit",
    ) -> ValenceSignal:
        """Convenience factory for credit-distribution signals."""
        return cls(
            value=max(-1.0, min(1.0, value)),
            source=source,
            goal_tag=goal_tag,
            timestamp=time.time(),
        )
