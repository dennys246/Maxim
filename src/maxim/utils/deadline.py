"""A point in monotonic time a bounded operation must finish by (#1438; docs/plans/bounded_stop.md §5).

In-process only, never persisted (it is a ``time.monotonic()`` reading, meaningless in another process). Sub-budgets
are carved from a parent and never exceed it, so a chain of waits cannot add up past the caller's bound.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Deadline:
    """``at`` is a ``time.monotonic()`` instant, or ``math.inf`` for no bound (:meth:`none`).

    SHAPE-FROZEN at 1.0 (CC3): never persisted or sent over a wire; a monotonic instant only means anything in the
    process that took it.
    """

    at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        return cls(time.monotonic() + max(0.0, seconds))

    @classmethod
    def none(cls) -> Deadline:
        return cls(math.inf)

    def remaining(self) -> float:
        """Seconds left, never negative (``math.inf`` for no bound)."""
        return max(0.0, self.at - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def sub(self, seconds: float) -> Deadline:
        """At most ``seconds`` from now, clamped to this deadline."""
        return Deadline(min(self.at, time.monotonic() + max(0.0, seconds)))

    def ending_early(self, seconds: float) -> Deadline:
        """This deadline brought forward by ``seconds`` (never before now): a reserve kept for whoever waits on the
        work this bound limits. Not ``sub``, which counts from NOW. ``none()`` stays unbounded (and an infinite
        ``seconds`` never makes it NaN)."""
        if math.isinf(self.at):
            return self
        return Deadline(max(time.monotonic(), self.at - max(0.0, seconds)))

    def share(self, fraction: float) -> Deadline:
        """``fraction`` of the remaining budget, clamped to this deadline. No bound stays no bound."""
        if math.isinf(self.at):
            return self
        return Deadline(time.monotonic() + self.remaining() * min(1.0, max(0.0, fraction)))

    def cap(self, seconds: float) -> float:
        """The wait to pass a blocking call: ``min(remaining, seconds)``."""
        return min(self.remaining(), seconds)
