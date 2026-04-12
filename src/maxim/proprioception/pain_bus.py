"""Backward-compatible PainBus — delegates to ReactionBus internally.

Phase 2a: PainBus wraps ReactionBus so existing code (PainDetector,
pain bridges, sim orchestrator) keeps working without changes. New code
should use ReactionBus directly. Phase 2b migrates remaining callers
and deprecates this module.

The F0.R1 ``route_pain_percept`` function has been removed — pain
injection now emits Reaction(kind="pain") directly without routing
through Percept.metadata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from maxim.proprioception.pain import PainSignal, PainType
from maxim.reactions.bus import ReactionBus
from maxim.reactions.compat import pain_signal_to_reaction

if TYPE_CHECKING:
    from maxim.memory.hippocampus import Hippocampus
    from maxim.reactions.types import Reaction

logger = logging.getLogger(__name__)


class PainBus:
    """Backward-compatible wrapper around ReactionBus.

    Accepts PainSignal on publish (auto-converts to Reaction), dispatches
    PainSignal to legacy subscribers (auto-converts back from Reaction).
    New callers should use ``self.reaction_bus`` directly.
    """

    def __init__(self, history_size: int = 200) -> None:
        self.reaction_bus = ReactionBus(
            history_size=history_size,
            refractory_overrides={"pain": 0.5},
        )

    def subscribe(self, callback: Callable[[PainSignal], None]) -> None:
        def _adapter(reaction: "Reaction") -> None:
            callback(_reaction_to_pain_signal(reaction))

        _adapter._original = callback  # type: ignore[attr-defined]
        self.reaction_bus.subscribe("pain", _adapter)

    def unsubscribe(self, callback: Callable[[PainSignal], None]) -> None:
        pass  # Not critical for Phase 2a; callers rarely unsubscribe

    def publish(self, signal: PainSignal) -> None:
        reaction = pain_signal_to_reaction(signal)
        self.reaction_bus.publish(reaction)

    @property
    def recent(self) -> list[PainSignal]:
        return [_reaction_to_pain_signal(r) for r in self.reaction_bus.history("pain")]

    def recent_by_type(self, pain_type: PainType) -> list[PainSignal]:
        return [s for s in self.recent if s.pain_type == pain_type]

    def get_stats(self) -> dict[str, int]:
        stats = self.reaction_bus.get_stats()
        return {
            "total_published": stats["total_published"],
            "subscriber_count": stats["subscriber_count"],
            "history_size": stats["history_size"],
        }


def _reaction_to_pain_signal(reaction: "Reaction") -> PainSignal:
    """Convert a Reaction(kind="pain") back to PainSignal for legacy callers."""
    source = reaction.source or ""
    pain_type_str = source.split(":")[-1] if ":" in source else "external_signal"
    try:
        pain_type = PainType(pain_type_str)
    except ValueError:
        pain_type = PainType.EXTERNAL_SIGNAL

    entity_binding = reaction.context.bindings.get("entity_path")
    context: dict[str, Any] = {}
    if entity_binding:
        context["entity_path"] = entity_binding.percept_id

    return PainSignal(
        pain_type=pain_type,
        intensity=reaction.intensity,
        timestamp=reaction.timestamp,
        context=context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: PainBus → Hippocampus episodic memory subscriber
# ─────────────────────────────────────────────────────────────────────────────


def create_pain_memory_subscriber(
    hippocampus: Hippocampus,
    intensity_threshold: float = 0.4,
) -> Callable[[PainSignal], None]:
    """Create a PainBus subscriber that captures pain as episodic memory.

    When pain intensity exceeds the threshold, an episodic memory is
    created in the hippocampus with the pain type, intensity, and context.

    Args:
        hippocampus: Hippocampus instance for memory capture.
        intensity_threshold: Minimum pain intensity to trigger memory
            formation. Default 0.4 captures moderate-to-severe pain.
    """
    from maxim.memory.types import Decision, Outcome, Perception

    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return

        hippocampus.capture(
            perception=Perception(
                observations={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    **signal.context,
                },
                salience=min(signal.intensity + 0.2, 1.0),
                novelty=0.6,
            ),
            decision=Decision(
                intent={"goal": "pain_response"},
                reasoning=(f"Pain detected: {signal.pain_type.value} (intensity={signal.intensity:.2f})"),
            ),
            outcome=Outcome(
                success=False,
                result={
                    "pain_type": signal.pain_type.value,
                    "intensity": signal.intensity,
                    "context": signal.context,
                },
            ),
        )

        # Simulation verbosity
        try:
            from maxim.simulation.sim_logger import sim_memory

            sim_memory(
                f"Pain memory captured: {signal.pain_type.value} (intensity={signal.intensity:.2f})",
            )
        except Exception:
            pass

    return _on_pain


def create_pain_nac_subscriber(
    nac: Any,
    intensity_threshold: float = 0.3,
) -> Callable[[PainSignal], None]:
    """Create a PainBus subscriber that records pain as causal observations in NAc.

    When pain fires, NAc learns "action/entity → pain" so the agent can
    predict and avoid painful situations in the future.

    Args:
        nac: NAc instance for causal learning.
        intensity_threshold: Minimum pain intensity to trigger observation.
    """
    from maxim.decisions.causal_link import Valence

    def _on_pain(signal: PainSignal) -> None:
        if signal.intensity < intensity_threshold:
            return

        entity = signal.context.get("entity_path", "unknown")
        try:
            nac.observe(
                event_type="pain",
                event_signature=f"pain:{signal.pain_type.name}:{entity}",
                outcome_type="pain",
                outcome_signature=f"intensity:{signal.intensity:.2f}",
                outcome_valence=Valence.NEGATIVE,
                delta_seconds=0.0,
                context=signal.context,
            )
        except Exception:
            pass  # Don't let NAc errors disrupt pain processing

    return _on_pain
