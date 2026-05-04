"""Backward-compatibility conversion between PainSignal and Reaction.

Temporary bridge for Phase 2a. Once all publishers emit Reaction
directly (Phase 2b), this module is deleted.
"""

from __future__ import annotations

from typing import Any

from maxim.decisions.causal_link import Valence
from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot


def pain_signal_to_reaction(signal: Any) -> Reaction:
    """Convert a legacy PainSignal to a typed Reaction(kind="pain").

    ``signal.context["agent_id"]`` (set by the publishing Embodiment) is
    propagated into ``ReactionContext.agent_id`` so reaction_bus
    subscribers — notably ``_distribute_reward_from_reaction`` in
    ``runtime/bio_stack.py`` — can credit the right agent's eligibility
    traces. Without this, every pain reaction reaches the subscriber
    with ``agent_id=None`` and silently early-returns, leaving
    ``_reward_bias`` empty regardless of how much pain fires.
    """
    ctx = signal.context or {}
    agent_id = ctx.get("agent_id") or None
    if agent_id == "":
        agent_id = None
    return Reaction(
        kind="pain",
        intensity=signal.intensity,
        valence=Valence.NEGATIVE,
        timestamp=signal.timestamp,
        context=ReactionContext(
            agent_id=agent_id,
            bindings={k: TraceSnapshot(percept_id=str(v)) for k, v in ctx.items() if k == "entity_path"},
        ),
        source=f"pain_detector:{signal.pain_type.value}",
    )
