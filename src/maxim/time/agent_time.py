"""The agent's two time collaborators: the machinery clock and the perceived-time port (world clock A-pre, D-WC2/D-WC6).

An agent reads time for two different jobs, and an embedder (a host that steps the agent inside its own world, such as
Overdrive) needs to supply each separately:

- **``MachineryClock``**: seconds that drive the agent's own machinery (de-duplication windows, refractories). Under
  the host default it is the wall clock. An embedded agent gets the host's world seconds instead.
- **``PerceivedTime``** (the port): the calendar the agent may *perceive*, that is, the date and time that may reach a
  prompt or a tool result. Under the host default it is the local calendar, exactly as today. ``NO_CALENDAR`` declares
  that there is none: both methods return ``None``, and a render site that gets ``None`` renders nothing.

Two rules hold for every collaborator:

- **No ambient state.** Collaborators are passed explicitly (constructor arguments with the host default kept). There
  is no module global to set and no ``ContextVar``: an ambient read that falls back to the wall when unset is a silent
  fallback (``proprioception/pain_bus.py``'s docstring rejects that back channel for the same reason).
- **The host defaults read at call time, through the module attribute** (``time.time()``, never a name bound at import),
  so a test clock that patches ``time.*`` (``tests/unit/_loop_harness.py::_StepClock``) reaches them (invariant I2).
  The port's ``datetime`` default is patched by name (``maxim.time.agent_time.datetime``), as WC5 patches the prompt's.

A render site decides on what the port *declares* (``port.now() is None``), never on an identity check against
``HOST_PERCEIVED_TIME``, so any embedder's port is honoured the same way.

Production callers in this PR: none yet. Later A-pre PRs thread these through ``LLMWorker`` → ``PromptBuilder``,
``ContextPool``, the SCN-derived renders and ``Hippocampus.__init__``, each keeping the host default (plan
``docs/plans/world_clock_a_pre.md`` §The design). The embedded caller is OD-A10's entry point.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Protocol, runtime_checkable

__all__ = [
    "HOST_MACHINERY_CLOCK",
    "HOST_PERCEIVED_TIME",
    "NO_CALENDAR",
    "HostMachineryClock",
    "HostPerceivedTime",
    "MachineryClock",
    "NoCalendar",
    "PerceivedTime",
]


@runtime_checkable
class MachineryClock(Protocol):
    """Seconds for the agent's own machinery. ``now_s`` is an epoch-like float; ``monotonic_s`` never goes backwards."""

    def now_s(self) -> float: ...

    def monotonic_s(self) -> float: ...


@runtime_checkable
class PerceivedTime(Protocol):
    """The calendar an agent may perceive. ``None`` from either method means "no calendar: render nothing"."""

    def now(self) -> datetime | None: ...

    def perceive(self, stamp_s: float) -> datetime | None: ...


class HostMachineryClock:
    """The host default: the wall clock, read through the ``time`` module attribute at every call (I2)."""

    __slots__ = ()

    def now_s(self) -> float:
        return time.time()

    def monotonic_s(self) -> float:
        return time.monotonic()

    def __repr__(self) -> str:
        return "HOST_MACHINERY_CLOCK"


class HostPerceivedTime:
    """The host default: the local calendar, as every render site reads it today."""

    __slots__ = ()

    def now(self) -> datetime | None:
        return datetime.now()

    def perceive(self, stamp_s: float) -> datetime | None:
        return datetime.fromtimestamp(stamp_s)

    def __repr__(self) -> str:
        return "HOST_PERCEIVED_TIME"


class NoCalendar:
    """The embedded declaration that the agent perceives no calendar: every render emits nothing."""

    __slots__ = ()

    def now(self) -> datetime | None:
        return None

    def perceive(self, stamp_s: float) -> datetime | None:
        return None

    def __repr__(self) -> str:
        return "NO_CALENDAR"


HOST_MACHINERY_CLOCK: MachineryClock = HostMachineryClock()
HOST_PERCEIVED_TIME: PerceivedTime = HostPerceivedTime()
NO_CALENDAR: PerceivedTime = NoCalendar()
