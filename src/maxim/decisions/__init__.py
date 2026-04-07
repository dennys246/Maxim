"""Decisions module - NAc (Nucleus Accumbens) causal inference.

Provides causal learning and outcome prediction for actions.
"""

from maxim.decisions.causal_link import (
    CausalLink,
    OutcomePrediction,
    TemporalDelta,
    Valence,
)
from maxim.decisions.nac import NAc, NACConfig

__all__ = [
    "NAc",
    "NACConfig",
    "CausalLink",
    "OutcomePrediction",
    "TemporalDelta",
    "Valence",
]
