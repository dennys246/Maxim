"""PerceptSource protocol — abstraction for percept origins.

Any system that produces Percepts (hardware, scenarios, replay, CLI)
implements this protocol. The agent pipeline consumes Percepts without
knowing their origin.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from maxim.agents.bus import Percept


@runtime_checkable
class PerceptSource(Protocol):
    """Produces Percepts from any origin — hardware, scenarios, replay, CLI."""

    @property
    def name(self) -> str:
        """Human-readable source identifier."""
        ...

    def next_percept(self) -> Percept | None:
        """Return the next Percept, or None if no percept is available.

        Non-blocking. Returns None when the source has no new percept
        (idle cycle) or is exhausted (scenario complete).
        """
        ...

    def is_exhausted(self) -> bool:
        """True when this source will never produce another Percept.

        Always False for live sources (hardware, CLI).
        True for scenarios/replays after last percept is emitted.
        """
        ...

    @property
    def capabilities(self) -> set[str]:
        """What kinds of Percepts this source can produce.

        Values: {"vision", "transcript", "cli", "comms", "proprioception"}
        Used by the pipeline to skip irrelevant subsystems.
        """
        ...
