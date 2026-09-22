"""What advances an agent's experience clock: its WORLD's time (memory-strength Phase 2 decision 1).

The clock itself (``memory/experience_clock.py``) never reads a clock of any kind; this driver
decides how much experience a world produced, once per live loop pass (after the pause check,
before the idle gate -- the slot ``tick_embodiment_drift`` uses, so a resting agent still lives and
an operator-paused one does not). Two kinds of world:

- **Real-time worlds** (the survival server, the robot, a CLI session -- the default): experience is
  elapsed monotonic time between live passes, **minus the time the autonomy controller spent
  paused**. Idle passes count: the world keeps happening while the agent waits. A long in-pass block
  (a multi-cycle deliberation, a robot motion) counts in full -- the world went on meanwhile. A
  suspended machine does not count: ``time.monotonic`` excludes system sleep on macOS and Linux. The
  per-pass cap is only a stall bound. This is what the capture window's constants (``tau``,
  drowning onset) are measured in -- seconds of the world.
- **Turn-based worlds** (text simulations): wall time there is mostly LLM latency, which is not
  experience, so a slow model must not make the agent forget faster. A turn-based percept source
  declares it with two duck-typed members -- ``experience_turns() -> int`` (monotonic count of the
  world's TURNS: new messages, not deliveries, and never a pain or sensor event inside a turn) and
  ``experience_us_per_turn: int`` (how much experience one turn is) -- and the clock advances by the
  new turns times that quantum. A step-based ``ScenarioSource`` does not declare it yet, so it runs
  as real-time: a known gap, recorded in the plan with its trigger.

Chosen once per loop run from the percept source, so a world cannot switch kinds mid-run.
"""

from __future__ import annotations

import time
from typing import Any, Callable

# A stall bound, not the pause mechanism (pauses are subtracted exactly): no single live pass is
# worth more than five minutes of lived time. Deliberation and motions run seconds, not minutes.
REALTIME_PASS_CAP_US = 300_000_000


class ExperienceClockDriver:
    """Advances one agent's experience clock from its world, once per live loop pass."""

    def __init__(
        self,
        clock: Any,
        *,
        percept_source: Any | None,
        paused_seconds: Callable[[], float] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._monotonic = monotonic
        self._paused_seconds = paused_seconds or (lambda: 0.0)
        turns = getattr(percept_source, "experience_turns", None)
        if callable(turns):
            # A turn-based declaration is checked HERE, loudly, not on the loop thread later.
            us_per_turn = getattr(percept_source, "experience_us_per_turn", None)
            start = turns()
            if not (isinstance(us_per_turn, int) and not isinstance(us_per_turn, bool) and us_per_turn > 0):
                raise TypeError(f"experience_us_per_turn must be a positive int, got {us_per_turn!r}")
            if not isinstance(start, int) or isinstance(start, bool):
                raise TypeError(f"experience_turns() must return an int, got {start!r}")
            self.kind = "turns"
            self._turns: Callable[[], int] | None = turns
            self._us_per_turn = us_per_turn
            self._last_turns = start
        else:
            self.kind = "realtime"
            self._turns = None
            self._us_per_turn = 0
            self._last_turns = 0
        self._last_pass: float | None = None
        self._last_paused_s = 0.0

    @property
    def clock(self) -> Any:
        return self._clock

    def on_live_pass(self) -> int:
        """Advance by the experience the world produced since the last live pass; return it (us)."""
        if self._clock is None:
            return 0
        if self._turns is not None:
            now_turns = self._turns()
            delta = now_turns - self._last_turns
            self._last_turns = now_turns
            gained = delta * self._us_per_turn if delta > 0 else 0
        else:
            now, paused = self._monotonic(), self._paused_seconds()
            last, self._last_pass = self._last_pass, now
            paused_since_last, self._last_paused_s = paused - self._last_paused_s, paused
            if last is None:
                return 0  # the first pass only starts the count
            lived = (now - last) - max(0.0, paused_since_last)
            gained = min(max(0, round(lived * 1_000_000)), REALTIME_PASS_CAP_US)
        if gained:
            self._clock.advance(gained)
        return gained


__all__ = ["ExperienceClockDriver", "REALTIME_PASS_CAP_US"]
