"""Central publish/subscribe for pain signals from any source.

Extracted from PainDetector's internal callback mechanism to allow
pain signals from motor, tool, simulation, energy, and cognitive
sources to reach all consumers through a single channel.

Also provides routing helpers:
- route_pain_percept(): converts proprioception Percepts to PainSignals
- create_pain_memory_subscriber(): captures pain events as episodic memories
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

from maxim.proprioception.pain import PainSignal, PainType

if TYPE_CHECKING:
    from maxim.agents.bus import Percept
    from maxim.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)


class PainBus:
    """Central publish/subscribe for pain signals.

    Any system can publish a PainSignal. All subscribers are notified
    synchronously (consistent with existing PainDetector behavior).

    Example:
        bus = PainBus()
        bus.subscribe(lambda sig: print(f"Pain: {sig.pain_type}"))
        bus.publish(PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
        ))
    """

    # Minimum interval between pain signals of the same type+entity (seconds)
    REFRACTORY_S: float = 0.5

    def __init__(self, history_size: int = 200) -> None:
        self._subscribers: list[Callable[[PainSignal], None]] = []
        self._lock = threading.Lock()
        self._history: deque[PainSignal] = deque(maxlen=history_size)
        self._total_published: int = 0
        self._last_published: dict[str, float] = {}  # (type:entity) → monotonic time

    def subscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Register a pain signal consumer."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[PainSignal], None]) -> None:
        """Remove a previously registered consumer."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, signal: PainSignal) -> None:
        """Publish a pain signal to all subscribers.

        Applies a refractory period per (pain_type, entity) to prevent
        spam from rapid repeated signals. Callbacks are invoked outside
        the lock to prevent deadlocks with subscribers that query the bus.
        """
        # Refractory check — skip if same type+entity fired too recently
        entity = signal.context.get("entity_path", "") if signal.context else ""
        refractory_key = f"{signal.pain_type.name}:{entity}"
        now = time.monotonic()
        with self._lock:
            last = self._last_published.get(refractory_key, 0.0)
            if now - last < self.REFRACTORY_S:
                return  # Still in refractory period
            self._last_published[refractory_key] = now
            self._history.append(signal)
            self._total_published += 1
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(signal)
            except Exception as e:
                logger.warning("PainBus subscriber error: %s", e)

    @property
    def recent(self) -> list[PainSignal]:
        """Recent pain signals (newest last)."""
        with self._lock:
            return list(self._history)

    def recent_by_type(self, pain_type: PainType) -> list[PainSignal]:
        """Recent signals filtered by type."""
        with self._lock:
            return [s for s in self._history if s.pain_type == pain_type]

    def get_stats(self) -> dict[str, int]:
        """Bus-level statistics."""
        with self._lock:
            return {
                "total_published": self._total_published,
                "subscriber_count": len(self._subscribers),
                "history_size": len(self._history),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Percept → PainBus routing
# ─────────────────────────────────────────────────────────────────────────────


def route_pain_percept(percept: Percept, pain_bus: PainBus) -> bool:
    """Convert a proprioception Percept into a PainSignal on the bus.

    Call this in the agent loop's percept processing path. Returns True
    if a pain signal was published, False if the percept was not a pain
    percept.

    Expected percept format:
        Percept(source="proprioception", content="pain_signal",
                metadata={"pain_type": "joint_strain", "intensity": 0.8, ...})
    """
    if percept.source != "proprioception" or percept.content != "pain_signal":
        return False

    meta = percept.metadata or {}
    try:
        pain_type = PainType(meta.get("pain_type", "external_signal"))
    except ValueError:
        pain_type = PainType.EXTERNAL_SIGNAL

    signal = PainSignal(
        pain_type=pain_type,
        intensity=meta.get("intensity", 0.5),
        timestamp=percept.timestamp,
        angular_velocity=meta.get("angular_velocity", 0.0),
        translation_velocity=meta.get("translation_velocity", 0.0),
        direction_reversals=meta.get("direction_reversals", 0),
        context={
            k: v
            for k, v in meta.items()
            if k
            not in {
                "pain_type",
                "intensity",
                "angular_velocity",
                "translation_velocity",
                "direction_reversals",
            }
        },
    )
    pain_bus.publish(signal)

    # Simulation verbosity
    try:
        from maxim.simulation.sim_logger import sim_pain

        sim_pain(signal.pain_type.value, signal.intensity, **signal.context)
    except Exception:
        pass

    return True


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
