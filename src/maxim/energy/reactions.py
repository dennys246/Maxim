"""Energy depletion → interoceptive Reaction bridge.

Checks energy levels on each tick and emits hunger/fatigue/satiation
Reactions when thresholds are crossed.  Wired into the bio-pipeline
via ReactionBus so that energy states feed into NAc causal learning
and episode valence annotation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)


def create_energy_reaction_bridge(
    reaction_bus: Any,
    *,
    hunger_threshold: float = 0.3,
    fatigue_threshold: float = 0.2,
    agent_id: str = "default",
) -> EnergyReactionBridge:
    """Create a bridge that emits Reactions when energy crosses thresholds.

    Call ``bridge.check(food_level, stamina_level)`` on each tick.
    When a level drops below its threshold, a hunger/fatigue Reaction
    is emitted.  When a level is restored above the threshold after
    being below, a satiation Reaction is emitted.

    Returns the bridge instance so the caller can call check() per tick.
    """
    return EnergyReactionBridge(
        reaction_bus=reaction_bus,
        hunger_threshold=hunger_threshold,
        fatigue_threshold=fatigue_threshold,
        agent_id=agent_id,
    )


class EnergyReactionBridge:
    """Stateful bridge tracking threshold crossings."""

    __slots__ = (
        "_bus",
        "_hunger_thresh",
        "_fatigue_thresh",
        "_agent_id",
        "_was_hungry",
        "_was_fatigued",
    )

    def __init__(
        self,
        reaction_bus: Any,
        hunger_threshold: float,
        fatigue_threshold: float,
        agent_id: str,
    ) -> None:
        self._bus = reaction_bus
        self._hunger_thresh = hunger_threshold
        self._fatigue_thresh = fatigue_threshold
        self._agent_id = agent_id
        self._was_hungry = False
        self._was_fatigued = False

    def check(
        self,
        food_level: float = 1.0,
        stamina_level: float = 1.0,
    ) -> None:
        """Check energy levels and emit Reactions on threshold crossings."""
        # Hunger
        is_hungry = food_level < self._hunger_thresh
        if is_hungry and not self._was_hungry:
            self._emit("hunger", intensity=1.0 - food_level)
        elif not is_hungry and self._was_hungry:
            self._emit_positive("satiation", intensity=food_level)
        self._was_hungry = is_hungry

        # Fatigue
        is_fatigued = stamina_level < self._fatigue_thresh
        if is_fatigued and not self._was_fatigued:
            self._emit("fatigue", intensity=1.0 - stamina_level)
        elif not is_fatigued and self._was_fatigued:
            self._emit_positive("satiation", intensity=stamina_level)
        self._was_fatigued = is_fatigued

    def _emit(self, kind: str, intensity: float) -> None:
        try:
            from maxim.decisions.causal_link import Valence
            from maxim.reactions.types import Reaction, ReactionContext

            self._bus.publish(
                Reaction(
                    kind=kind,
                    intensity=min(1.0, max(0.0, intensity)),
                    valence=Valence.NEGATIVE,
                    timestamp=time.time(),
                    context=ReactionContext(agent_id=self._agent_id),
                    source=f"energy:{kind}",
                )
            )
        except Exception as e:
            log.debug("EnergyReactionBridge._emit failed: %s", e)

    def _emit_positive(self, kind: str, intensity: float) -> None:
        try:
            from maxim.decisions.causal_link import Valence
            from maxim.reactions.types import Reaction, ReactionContext

            self._bus.publish(
                Reaction(
                    kind=kind,
                    intensity=min(1.0, max(0.0, intensity * 0.5)),
                    valence=Valence.POSITIVE,
                    timestamp=time.time(),
                    context=ReactionContext(agent_id=self._agent_id),
                    source=f"energy:{kind}_restored",
                )
            )
        except Exception as e:
            log.debug("EnergyReactionBridge._emit_positive failed: %s", e)
