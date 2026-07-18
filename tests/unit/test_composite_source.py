"""Guards for CompositePerceptSource — the thalamic-relay first-slice multiplexer.

Pins the forwarding + termination contract that the three-lens design review
of ``thalamus_relay_design_pass.md`` found the draft got wrong:

- ordered first-non-None drain (dumb multiplexer, not priority policy);
- ``inject_cli``/``inject_pain``/``finish`` route to the single implementing
  child; a second implementer raises (addressed routing earns the coordinator);
- ``advance_step``/``has_pending`` fan out to children;
- ``is_exhausted()`` excludes ambient (perpetual-live) children, so a live
  sensor never blocks a scripted sim from terminating;
- ``isinstance(composite, PerceptSource)`` (CC8 protocol conformance).
"""

from __future__ import annotations

from collections import deque

from maxim.agents.bus import Percept
from maxim.simulation.composite_source import (
    CompositePerceptSource,
    CompositeRoutingError,
)
from maxim.simulation.sources import PerceptSource


class _ScriptedSource:
    """A ConversationalSource-shaped stub: injectable + exhaustible."""

    def __init__(self, name: str = "scripted") -> None:
        self._name = name
        self._queue: deque[Percept] = deque()
        self._finished = False
        self.advanced = 0
        self.injected: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        return {"cli", "transcript"}

    def next_percept(self) -> Percept | None:
        return self._queue.popleft() if self._queue else None

    def is_exhausted(self) -> bool:
        return self._finished and not self._queue

    def advance_step(self) -> None:
        self.advanced += 1

    def has_pending(self) -> bool:
        return bool(self._queue)

    def inject_cli(self, text: str, salience: float = 0.8, novelty: float = 0.7) -> None:
        self.injected.append(text)
        self._queue.append(Percept(timestamp=0.0, source="cli", cli_input=text))

    def inject_pain(self, pain_type: str = "external_signal", intensity: float = 0.5, **kw) -> None:
        self.injected.append(f"pain:{pain_type}")

    def finish(self) -> None:
        self._finished = True


class _ScriptedReadOnly:
    """An exhaustible scripted source with NO injection surface (e.g. a
    pre-recorded ScenarioSource). Gates termination; safe to pair two of
    them because neither implements inject_cli/inject_pain/finish."""

    def __init__(self, name: str = "readonly") -> None:
        self._name = name
        self._queue: deque[Percept] = deque()
        self._finished = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        return {"transcript"}

    def next_percept(self) -> Percept | None:
        return self._queue.popleft() if self._queue else None

    def is_exhausted(self) -> bool:
        return self._finished and not self._queue

    def has_pending(self) -> bool:
        return bool(self._queue)

    def enqueue(self, percept: Percept) -> None:
        self._queue.append(percept)

    def mark_done(self) -> None:
        self._finished = True


class _LiveSensor:
    """An AzimuthDoASource-shaped stub: perpetual-live, no injection surface."""

    def __init__(self, name: str = "sensor") -> None:
        self._name = name
        self._next: Percept | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        return {"audio"}

    def next_percept(self) -> Percept | None:
        p, self._next = self._next, None
        return p

    def is_exhausted(self) -> bool:
        return False  # perpetual-live — never exhausts

    def emit(self, percept: Percept) -> None:
        self._next = percept


# ── construction ────────────────────────────────────────────────────────────


def test_empty_children_rejected():
    try:
        CompositePerceptSource([])
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty children")


def test_ambient_must_be_a_child():
    scripted, sensor = _ScriptedSource(), _LiveSensor()
    try:
        CompositePerceptSource([scripted], ambient=[sensor])
    except ValueError:
        return
    raise AssertionError("expected ValueError when ambient source is not a child")


def test_isinstance_percept_source():
    composite = CompositePerceptSource([_ScriptedSource(), _LiveSensor()])
    assert isinstance(composite, PerceptSource)


# ── core multiplex ──────────────────────────────────────────────────────────


def test_capabilities_unioned():
    composite = CompositePerceptSource([_ScriptedSource(), _LiveSensor()])
    assert composite.capabilities == {"cli", "transcript", "audio"}


def test_next_percept_ordered_first_non_none_wins():
    scripted, sensor = _ScriptedSource(), _LiveSensor()
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    # Both children have a percept ready; child order (scripted first) wins.
    p_scripted = Percept(timestamp=1.0, source="cli", cli_input="hi")
    p_sensor = Percept(timestamp=2.0, source="sensor")
    scripted._queue.append(p_scripted)
    sensor.emit(p_sensor)
    assert composite.next_percept() is p_scripted
    # scripted now idle → the sensor's percept surfaces next.
    assert composite.next_percept() is p_sensor
    # both idle → None.
    assert composite.next_percept() is None


# ── termination ─────────────────────────────────────────────────────────────


def test_is_exhausted_excludes_ambient_live_sensor():
    scripted, sensor = _ScriptedSource(), _LiveSensor()
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    assert composite.is_exhausted() is False
    scripted.finish()
    # Scripted child done; the perpetual-live sensor must NOT keep the
    # composite alive — else the sim never self-terminates.
    assert composite.is_exhausted() is True


def test_is_exhausted_all_children_gate_by_default():
    a, b = _ScriptedReadOnly("a"), _ScriptedReadOnly("b")
    composite = CompositePerceptSource([a, b])  # no ambient → both gate
    a.mark_done()
    assert composite.is_exhausted() is False  # b still open
    b.mark_done()
    assert composite.is_exhausted() is True


def test_pure_ambient_composite_never_exhausts():
    s1, s2 = _LiveSensor("s1"), _LiveSensor("s2")
    composite = CompositePerceptSource([s1, s2], ambient=[s1, s2])
    assert composite.is_exhausted() is False


# ── fan-out ─────────────────────────────────────────────────────────────────


def test_advance_step_broadcasts_only_to_implementers():
    scripted, sensor = _ScriptedSource(), _LiveSensor()  # sensor has no advance_step
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    composite.advance_step()
    composite.advance_step()
    assert scripted.advanced == 2  # forwarded; no AttributeError from the sensor


def test_has_pending_true_if_any_child_pending():
    scripted, sensor = _ScriptedSource(), _LiveSensor()  # sensor lacks has_pending → treated True
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    # Sensor has no has_pending → conservative default True → composite True.
    assert composite.has_pending() is True


def test_has_pending_reflects_only_pending_aware_children():
    a, b = _ScriptedReadOnly("a"), _ScriptedReadOnly("b")  # both implement has_pending
    composite = CompositePerceptSource([a, b])
    assert composite.has_pending() is False
    a.enqueue(Percept(timestamp=1.0, source="transcript"))
    assert composite.has_pending() is True


# ── injection / lifecycle routing ───────────────────────────────────────────


def test_inject_cli_routes_to_single_scripted_child():
    scripted, sensor = _ScriptedSource(), _LiveSensor()
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    composite.inject_cli("try deleting my files")
    assert scripted.injected == ["try deleting my files"]
    # And the injected percept is now drainable via the composite.
    p = composite.next_percept()
    assert p is not None and p.cli_input == "try deleting my files"


def test_inject_pain_and_finish_route_to_scripted_child():
    scripted, sensor = _ScriptedSource(), _LiveSensor()
    composite = CompositePerceptSource([scripted, sensor], ambient=[sensor])
    composite.inject_pain(pain_type="thermal", intensity=0.7)
    composite.finish()
    assert "pain:thermal" in scripted.injected
    assert scripted.is_exhausted() is True


def test_two_inject_cli_implementers_rejected_at_construction():
    a, b = _ScriptedSource("a"), _ScriptedSource("b")  # both implement inject_cli
    try:
        CompositePerceptSource([a, b])
    except CompositeRoutingError:
        return
    raise AssertionError("expected CompositeRoutingError: 2 inject_cli implementers")


def test_route_with_no_implementer_raises():
    # A composite of only sensors has no inject_cli target; calling it must
    # fail loud, not silently drop user input.
    s1, s2 = _LiveSensor("s1"), _LiveSensor("s2")
    composite = CompositePerceptSource([s1, s2], ambient=[s1, s2])
    try:
        composite.inject_cli("hello")
    except CompositeRoutingError:
        return
    raise AssertionError("expected CompositeRoutingError: no inject_cli implementer")
