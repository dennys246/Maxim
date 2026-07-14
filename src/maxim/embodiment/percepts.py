"""EmbodimentPerceptSource — bridges SEM sensor readings into the agent loop.

Dormant since 2026-07-14: NEVER production-wired — pickaxe-verified that
``EmbodimentPerceptSource(`` never entered ``src/`` in any commit since the
module's 2026-04-07 creation. The tick cycle this class was designed to own
was relocated into ``Body.evaluate_failures`` by ``ed8b187f`` (2026-04-26)
precisely because nothing called ``next_percept()`` and drives were frozen;
that commit also removed this module's duplicate ``tick_vital_drift`` call
to prevent double-drift. Kept (per CLAUDE.md dormancy-over-deletion) as the
CC8 ``PerceptSource`` protocol-shape template — this class is exercised by
``tests/unit/test_producer_protocols.py`` and
``tests/unit/test_naming_events.py``; the 1.1 ``RemotePerceptSource`` adapter
(docs/plans/mesh_perception_transport.md) is its local sibling and the
resurrection trigger. Wiring it into a production loop is tick-safe
(``evaluate_failures`` owns the wall-clock drift and cannot double-drift)
but requires un-dormanting: update the CLAUDE.md embodiment-tick invariant
in the same commit.

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
        agent_id: str | None = None,
    ) -> None:
        self._embodiment = embodiment
        self._poll_hz = poll_hz
        self._demand_hz = demand_hz
        self._agent_id = agent_id
        self._last_poll: float = 0.0
        self._exhausted = False
        self._in_demand = False

        # Roy-5b naming-event scaffolding (docs/plans/roy_5_encoder_alignment_disambiguator.md
        # Stage 3, src/maxim/embodiment/naming_events.py). When the body
        # declares ``naming_events:`` metadata, this source emits a short
        # utterance into body_state text the same tick the drive crosses
        # threshold — closing the co-firing gap that left Roy-4's priming
        # with empty linguistic-modality EC nodes in substrate-primary
        # mode. Empty tuple = no body opt-in, the percept shape is
        # byte-identical to pre-Stage-3.
        #
        # The dict type-check keeps MagicMock-based unit tests opt-out by
        # default; a real Embodiment.root.metadata is always a dict
        # populated by ``_parse_entity`` (spec.py).
        #
        # Thread-safety note: ``_naming_fired_keys`` is single-writer
        # (the agent-loop tick calls ``next_percept`` from one thread per
        # agent). The ``set_demand_mode`` field exists to bump poll
        # frequency, not to introduce concurrency — there is no
        # multi-thread reader/writer on this stash. Per-agent isolation
        # is guaranteed by one ``EmbodimentPerceptSource`` instance per
        # agent (the bio-stack wiring), so no ``dict[agent_id, ...]``
        # stash is needed here.
        from maxim.embodiment.naming_events import parse_naming_patterns

        metadata = getattr(self._embodiment.root, "metadata", None)
        naming_events_raw = metadata.get("naming_events") if isinstance(metadata, dict) else None
        body_name = getattr(self._embodiment.root, "name", None)
        self._naming_patterns = parse_naming_patterns(naming_events_raw, source=body_name)
        self._naming_fired_keys: set[str] = set()

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

        # Evaluate failures (may publish PainSignals).
        # Drive drift is now applied automatically inside evaluate_failures
        # via wall-clock dt, so it works in ALL code paths (generative
        # runner, tool_bridge, sim tools) — not just this percept source.
        failures = self._embodiment.evaluate_failures()

        # Build percept from body state
        body_state = self._embodiment.format_body_state_for_prompt()
        if not body_state and not failures:
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

        # Roy-5b: derive deterministic naming utterances co-temporally
        # with the same body snapshot the body_state text rendered above.
        # Appending HERE means LinguisticEncoder processes the utterance
        # in the same agent-loop tick SensorEncoder processes the drive
        # snapshot — co-firing the Hebbian binding rule needs. When the
        # body has no naming_events config, _naming_patterns is empty
        # and this branch is a no-op (byte-identical pre-Stage-3 path).
        #
        # The sensor-value collector walks the embodiment tree directly
        # (NOT body_state_summary, which intentionally skips dotted-key
        # modulator drive specs — Roy-4 review surfaced this gap, see
        # naming_events.py::collect_sensor_values docstring).
        if self._naming_patterns:
            from maxim.embodiment.naming_events import (
                collect_sensor_values,
                derive_naming_utterances,
                format_naming_section,
            )

            sensor_values = collect_sensor_values(self._embodiment)
            utterances, self._naming_fired_keys = derive_naming_utterances(
                sensor_values, self._naming_patterns, self._naming_fired_keys
            )
            section = format_naming_section(utterances)
            if section:
                content_parts.append(section)

        try:
            from maxim.agents.modality import SensoryModality, SensoryTag
            from maxim.agents.percept_factory import make_intero_percept
        except ImportError:
            return None

        sensory = SensoryTag(
            modality=SensoryModality.INTEROCEPTION,
            submodality="pain" if failures else "vital",
            intensity=salience if salience > 0 else 0.5,
            entity_source=self._embodiment.root.name,
        )
        return make_intero_percept(
            "\n".join(content_parts),
            source="embodiment",
            agent_id=self._agent_id,
            salience=salience,
            novelty=novelty,
            metadata={
                "failure_count": len(failures),
                "entity_root": self._embodiment.root.name,
            },
            sensory=sensory,
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
