"""PerceptSource protocol — abstraction for percept origins.

Any system that produces Percepts (hardware, scenarios, replay, CLI,
or future external adapters like Mineflayer/Minecraft) implements this
protocol. The agent pipeline consumes Percepts without knowing their
origin.

Adapter contract (CC8, 2026-04-29 audit)
---------------------------------------

The four required members below — ``name``, ``next_percept()``,
``is_exhausted()``, ``capabilities`` — are the public adapter contract.
A non-sim adapter (e.g. a Mineflayer Minecraft client) implementing
exactly these four can drive the agent loop with no orchestrator,
narrative-phase, or conversational-turn dependency.

What the protocol does NOT assume:

- **No orchestrator coupling.** The agent loop calls ``next_percept()``
  on its own schedule; the source is free-standing.
- **No narrative phase coupling.** Phase progression is sim-orchestrator
  business; sources do not know about acts, encounters, or DM hooks.
- **No conversational turn assumption.** Sources are polled — they may
  return ``None`` for arbitrarily many ticks, then a single ``Percept``,
  then ``None`` again. The driver is turn-based but a source need not
  align with conversational turns.
- **No tool-registry coupling.** Sources produce percepts; tools are
  registered separately and called by the executor.

Optional extensions (duck-typed)
--------------------------------

The agent loop and simulation adapter probe for two optional methods
via ``hasattr`` / ``getattr`` — implementing them is not required, but
some integrations benefit:

- ``has_pending() -> bool`` — fast pre-check used by the agent loop's
  idle-skip heuristic (``runtime/agent_loop.py``). Implementations that
  buffer percepts can return ``True`` to wake the loop sooner. Default
  behavior (when absent) treats the source as always potentially
  pending.
- ``advance_step() -> None`` — called once per loop tick after
  ``next_percept()``; lets scripted sources advance an internal clock.
  Live sources (hardware, CLI, Minecraft) typically do not implement it.

These extensions are optional and additive; new ones may be introduced
post-1.0 with the same duck-typed-with-default pattern. Adapters that
do not implement them get sensible defaults at the call sites.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from maxim.agents.bus import Percept


@runtime_checkable
class PerceptSource(Protocol):
    """Produces Percepts from any origin — hardware, scenarios, replay,
    CLI, or external game/world adapters (Mineflayer, etc.).

    See module docstring for the adapter contract: what the protocol
    does and does not assume, plus optional duck-typed extensions
    (``has_pending``, ``advance_step``).
    """

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

        Always False for live sources (hardware, CLI, persistent
        external adapters).  True for scenarios/replays after last
        percept is emitted.
        """
        ...

    @property
    def capabilities(self) -> set[str]:
        """What kinds of Percepts this source can produce.

        Values: {"vision", "transcript", "cli", "comms", "proprioception"}
        Used by the pipeline to skip irrelevant subsystems. New
        capability strings may be added by external adapters; consumers
        should treat unknown capabilities as opaque rather than reject.
        """
        ...
