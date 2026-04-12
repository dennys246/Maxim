"""CircadianContext — minimal SCN-style temporal tag attached to percepts.

Produced by the SCN (Suprachiasmatic Nucleus) and carried on
:class:`~maxim.agents.percept_context.PerceptContext` so downstream
consumers can condition on circadian phase without re-querying SCN.

Intentionally minimal for F0.4. Richer fields (phase alignment vectors,
next-transition predictions) belong to substrate P8's sleep-replay work
where they become load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CircadianContext:
    """Snapshot of circadian state at the moment a percept was produced.

    All fields are optional so a caller without SCN access (tests,
    fixtures, pre-SCN percepts) can still construct a valid context.
    """

    phase: str | None = None  # e.g. "wake", "rest", "sleep_slow", "sleep_rem"
    tick: int | None = None  # Integer tick count from the oscillator
    wall_time: float | None = None  # Unix timestamp at capture

    def to_dict(self) -> dict[str, Any]:
        return {"phase": self.phase, "tick": self.tick, "wall_time": self.wall_time}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CircadianContext":
        return cls(
            phase=data.get("phase"),
            tick=data.get("tick"),
            wall_time=data.get("wall_time"),
        )
