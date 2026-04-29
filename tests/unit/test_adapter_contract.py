"""Tests for the PerceptSource / ActionSink adapter contract (CC8).

The contract clarifies that a non-sim adapter (e.g. a future Mineflayer
Minecraft client) can implement these protocols without orchestrator,
narrative-phase, or conversational-turn coupling. These tests pin the
minimal contract — they pass for any conforming adapter, not just the
built-in sim variants.
"""

from __future__ import annotations

from maxim.simulation.sources import PerceptSource
from maxim.simulation.sinks import ActionRecord, ActionSink


class _MinimalPerceptSource:
    """Tiny non-sim adapter — proves the protocol is implementable
    without inheriting from any Maxim base or importing orchestrator
    code.
    """

    def __init__(self) -> None:
        self._exhausted = False

    @property
    def name(self) -> str:
        return "minimal-test-adapter"

    def next_percept(self):
        return None

    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def capabilities(self) -> set[str]:
        return {"vision"}


class _MinimalActionSink:
    def __init__(self) -> None:
        self._actions: list[ActionRecord] = []

    def record(self, action: ActionRecord) -> None:
        self._actions.append(action)

    @property
    def actions(self) -> list[ActionRecord]:
        return list(self._actions)


def test_minimal_percept_source_satisfies_protocol():
    src = _MinimalPerceptSource()
    assert isinstance(src, PerceptSource)
    assert src.name == "minimal-test-adapter"
    assert src.next_percept() is None
    assert src.is_exhausted() is False
    assert "vision" in src.capabilities


def test_minimal_action_sink_satisfies_protocol():
    sink = _MinimalActionSink()
    assert isinstance(sink, ActionSink)
    rec = ActionRecord(timestamp=1.0, tool_name="noop")
    sink.record(rec)
    assert len(sink.actions) == 1
    assert sink.actions[0].tool_name == "noop"


def test_optional_extensions_are_duck_typed():
    """has_pending() and advance_step() are NOT in the Protocol — adapters
    that omit them must still satisfy isinstance(..., PerceptSource).
    """
    src = _MinimalPerceptSource()
    # No has_pending / advance_step on _MinimalPerceptSource.
    assert not hasattr(src, "has_pending")
    assert not hasattr(src, "advance_step")
    # Still satisfies the runtime-checkable Protocol.
    assert isinstance(src, PerceptSource)
