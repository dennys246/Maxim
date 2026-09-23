"""The experience clock (memory-strength plan, Phase 2 decision 1).

Forgetting runs on EXPERIENCE, not on the wall clock: a robot switched off for a month should not
wake with its memory wiped, and intervening experience drives most forgetting (Jenkins &
Dallenbach). This object only holds and persists that time; WHAT advances it is the agent's world
-- elapsed world time in a real-time world, a fixed quantum per turn in a turn-based one -- decided
by ``runtime/experience_time.py::ExperienceClockDriver`` once per live loop pass. The clock itself
reads no clock of any kind.

Stored as integer microseconds. One clock per agent, owned by its Hippocampus and persisted with it.
Nothing reads it yet: the strength strategy (Phase 2c) is its first consumer, and the
stalled-clock assert lands with it.
"""

from __future__ import annotations

import threading
from typing import Any

UNIT = "world_experience_us"


class ExperienceClock:
    """Monotonic experience time for one agent. Thread-safe; takes no other lock."""

    def __init__(self, us: int = 0) -> None:
        if us < 0:
            raise ValueError(f"experience clock cannot start negative, got {us!r}")
        self._us = int(us)
        self._lock = threading.Lock()

    def now_us(self) -> int:
        with self._lock:
            return self._us

    def advance(self, dt_us: int) -> int:
        """Advance by ``dt_us`` (>= 0) and return the new time. Experience never runs backwards."""
        if dt_us < 0:
            raise ValueError(f"experience clock cannot run backwards, got dt_us={dt_us!r}")
        with self._lock:
            self._us += int(dt_us)
            return self._us

    def to_dict(self) -> dict[str, Any]:
        return {"us": self.now_us(), "unit": UNIT}

    @classmethod
    def from_dict(cls, data: Any) -> ExperienceClock:
        """Strict: a record of the wrong shape, unit or type raises ``ValueError``, never guesses."""
        if not isinstance(data, dict):
            raise ValueError(f"experience clock record must be a dict, got {type(data).__name__}")
        if data.get("unit") != UNIT:
            raise ValueError(f"experience clock unit {data.get('unit')!r} is not {UNIT!r}")
        us = data.get("us")
        if not isinstance(us, int) or isinstance(us, bool):
            raise ValueError(f"experience clock 'us' must be an int, got {us!r}")
        return cls(us)

    def restore(self, other: ExperienceClock) -> None:
        """Adopt another clock's time in place, so holders of this object keep a live reference."""
        us = other.now_us()
        with self._lock:
            self._us = us


class ExperienceClockStalled(RuntimeError):
    """A session used memory on an experience-time retention model, and the clock never moved.

    Raised by ``MemoryHub`` at session end (memory-strength Phase 2c-3). The failure it names is a
    SILENT one: with ``dt`` stuck at 0 every trace scores ``R = 1``, nothing is ever forgotten, and
    the run looks exactly like a run whose memories all deserved to be kept. The usual cause is a
    path that captures or activates without going through the agent loop -- a scripted harness --
    so nothing calls ``ExperienceClockDriver.on_live_pass``.

    Explicit keyword-only ``__init__`` (CC3 spirit for typed exceptions: no ``**kwargs``), and it
    carries the session's own ``results`` so a caller that wanted them is not also robbed of them.
    """

    def __init__(self, *, captures: int, activations: int, results: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"the experience clock did not advance, but this session captured {captures} "
            f"and activated {activations} memories on a retention model that runs on experience "
            f"time: with dt = 0 every trace stays at R = 1 and nothing is ever forgotten. "
            f"Something must advance the clock (runtime/experience_time.py::ExperienceClockDriver)."
        )
        self.captures = captures
        self.activations = activations
        self.results = results


__all__ = ["ExperienceClock", "ExperienceClockStalled", "UNIT"]
