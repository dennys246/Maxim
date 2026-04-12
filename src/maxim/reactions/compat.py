"""Backward-compatibility conversion between PainSignal and Reaction.

Temporary bridge for Phase 2a. Once all publishers emit Reaction
directly (Phase 2b), this module is deleted.
"""

from __future__ import annotations

from typing import Any

from maxim.decisions.causal_link import Valence
from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot


def pain_signal_to_reaction(signal: Any) -> Reaction:
    """Convert a legacy PainSignal to a typed Reaction(kind="pain")."""
    return Reaction(
        kind="pain",
        intensity=signal.intensity,
        valence=Valence.NEGATIVE,
        timestamp=signal.timestamp,
        context=ReactionContext(
            bindings={
                k: TraceSnapshot(percept_id=str(v)) for k, v in (signal.context or {}).items() if k == "entity_path"
            },
        ),
        source=f"pain_detector:{signal.pain_type.value}",
    )
