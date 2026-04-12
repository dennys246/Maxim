"""Producer protocols for the Percept/Reaction dual-surface architecture.

Structural (Protocol-based) — any object implementing the interface is a
valid producer. SEM sensors implement ``PerceptProducer`` via
``EmbodimentPerceptSource``. SEM modulators facilitate
``ReactionProducer`` via ``CerebellumModulator``. Non-SEM producers
(CommsGateway, PainDetector, FearPredictor) implement the protocols
directly.

The method name ``next_percept`` matches the existing codebase convention
(used by SimulationAdapter, agent_loop, ScenarioSource, etc.) rather
than the plan's ``produce()`` to avoid a rename wave. The Protocol
captures the same structural contract either way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from maxim.agents.bus import Percept
    from maxim.reactions.types import Reaction, ReactionKind


@runtime_checkable
class PerceptProducer(Protocol):
    """Produces Percepts from a sensory or environmental source."""

    @property
    def name(self) -> str: ...

    def next_percept(self) -> "Percept | None": ...


@runtime_checkable
class ReactionProducer(Protocol):
    """Produces Reactions from an evaluative or motivational source."""

    @property
    def name(self) -> str: ...

    @property
    def kind(self) -> "ReactionKind": ...

    def next_reaction(self) -> "Reaction | None": ...
