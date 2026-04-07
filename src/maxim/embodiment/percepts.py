"""EmbodimentPerceptSource — bridges SEM sensor readings into the agent loop.

Implements the ``PerceptSource`` protocol, polling sensors at a
configurable rate (default 1Hz) and producing ``Percept`` objects
from the readings.

Pain-relevant sensors (within 20% of a failure threshold) are
promoted to every-tick reading.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.embodiment.body import Embodiment

log = logging.getLogger(__name__)


class EmbodimentPerceptSource:
    """PerceptSource adapter for the Embodiment subsystem.

    Parameters
    ----------
    embodiment : Embodiment
        The runtime managing the entity tree.
    poll_hz : float
        Base polling rate in Hz (default 1.0 = once per second).
    demand_hz : float or None
        Override polling rate during high-demand periods (e.g.,
        motor program execution).  Set to None to use base rate.
    """

    def __init__(
        self,
        embodiment: Embodiment,
        *,
        poll_hz: float = 1.0,
        demand_hz: float | None = None,
    ) -> None:
        self._embodiment = embodiment
        self._poll_hz = poll_hz
        self._demand_hz = demand_hz
        self._last_poll: float = 0.0
        self._exhausted = False
        self._in_demand = False

    # -- PerceptSource protocol ---------------------------------------------

    @property
    def name(self) -> str:
        return f"embodiment:{self._embodiment.root.name}"

    def next_percept(self) -> Any:
        """Return a Percept with sensor readings, or None if not time yet.

        Non-blocking.  Returns None between poll intervals.
        """
        now = time.time()
        hz = self._demand_hz if self._in_demand and self._demand_hz else self._poll_hz
        interval = 1.0 / max(hz, 0.01)

        if now - self._last_poll < interval:
            return None

        self._last_poll = now

        # Evaluate failures (may publish PainSignals)
        failures = self._embodiment.evaluate_failures()

        # Apply vital metric drift
        dt = interval
        self._embodiment.tick_vital_drift(dt)

        # Build percept from body state
        body_state = self._embodiment.format_body_state_for_prompt()
        if not body_state and not failures:
            return None

        try:
            from maxim.agents.bus import Percept
        except ImportError:
            return None

        # Compute salience from pain proximity + failures
        salience = 0.0
        novelty = 0.0
        if failures:
            salience = max(f.pain_intensity for f in failures)
            novelty = 0.5  # failures are somewhat novel

        content_parts = []
        if body_state:
            content_parts.append(body_state)
        if failures:
            failure_lines = [
                f"FAILURE: {f.failure_name} on {f.entity_path} (pain={f.pain_intensity:.2f})" for f in failures
            ]
            content_parts.append("\n".join(failure_lines))

        return Percept(
            timestamp=now,
            source="embodiment",
            content="\n".join(content_parts),
            salience=salience,
            novelty=novelty,
            metadata={
                "failure_count": len(failures),
                "entity_root": self._embodiment.root.name,
            },
        )

    def is_exhausted(self) -> bool:
        """Always False — embodiment is a live source."""
        return self._exhausted

    @property
    def capabilities(self) -> set[str]:
        return {"proprioception"}

    # -- demand mode --------------------------------------------------------

    def set_demand_mode(self, active: bool, hz: float | None = None) -> None:
        """Enable/disable high-demand polling (e.g., during motor programs).

        Parameters
        ----------
        active : bool
            Whether to use the demand polling rate.
        hz : float or None
            Override demand Hz.  If None, uses the configured demand_hz.
        """
        self._in_demand = active
        if hz is not None:
            self._demand_hz = hz
