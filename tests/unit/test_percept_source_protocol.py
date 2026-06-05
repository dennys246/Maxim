"""Pin that the PerceptSource Protocol shape is sufficient for the
1.1 network-backed adapter (C10 prep).

The C10 perception transport ships in 1.1 alongside Hivemind. The
``RemotePerceptSource`` adapter that the 1.1 work will land must satisfy
the existing four-member Protocol contract — adding required Protocol
members post-1.0 would break every third-party ``PerceptSource``
implementation.

This test stubs the minimum implementation a 1.1 ``RemotePerceptSource``
would need and asserts ``isinstance(stub, PerceptSource)`` returns True.
If a future edit to the Protocol adds a required member, this test
fails — signal that 1.1 cannot land cleanly without breaking the
adapter contract.
"""

from __future__ import annotations

from collections import deque

from maxim.agents.bus import Percept
from maxim.simulation.sources import PerceptSource


class _StubRemotePerceptSource:
    """Minimum-viable shape of the 1.1 RemotePerceptSource adapter.

    Behaviors pinned by this stub (per the module docstring of
    ``simulation/sources.py`` "Network-backed adapter case" section):

    - ``name`` follows the ``f"remote:{node_name}"`` convention.
    - ``next_percept`` reads from a local in-process inbox; no
      network call.
    - ``is_exhausted`` False unless the peer signaled end-of-stream.
    - ``capabilities`` populated from a handshake (here: constructor).
    - ``has_pending`` optional duck-typed extension; True when inbox
      non-empty.
    """

    def __init__(
        self,
        node_name: str,
        capabilities: set[str],
    ) -> None:
        self._node_name = node_name
        self._capabilities = capabilities
        self._inbox: deque[Percept] = deque()
        self._end_of_stream = False

    # Required Protocol members ────────────────────────────────────

    @property
    def name(self) -> str:
        return f"remote:{self._node_name}"

    def next_percept(self) -> Percept | None:
        # Non-blocking: pop one queued percept, return None if empty.
        # NEVER makes a network call — transport-layer code populates
        # the inbox out-of-band.
        if self._inbox:
            return self._inbox.popleft()
        return None

    def is_exhausted(self) -> bool:
        # Transient peer unreachability is NOT exhaustion — only an
        # explicit end-of-stream signal flips this.
        return self._end_of_stream

    @property
    def capabilities(self) -> set[str]:
        return self._capabilities

    # Optional duck-typed extension ────────────────────────────────

    def has_pending(self) -> bool:
        return bool(self._inbox)

    # Test scaffolding ─────────────────────────────────────────────

    def _enqueue(self, percept: Percept) -> None:
        """Test-only helper to simulate the transport thread pushing
        a Percept into the local inbox."""
        self._inbox.append(percept)

    def _signal_end_of_stream(self) -> None:
        """Test-only helper to simulate the peer's deliberate shutdown."""
        self._end_of_stream = True


def test_stub_remote_percept_source_satisfies_protocol():
    """The four-member contract is sufficient for a network-backed adapter.

    This pins the 1.1 ship's foundation: any future edit that adds a
    required Protocol member fails this test, surfacing the breaking
    change before it lands.
    """
    stub = _StubRemotePerceptSource(
        node_name="reachy-001",
        capabilities={"vision", "audio", "proprioception"},
    )
    assert isinstance(stub, PerceptSource)


def test_stub_name_follows_remote_prefix_convention():
    stub = _StubRemotePerceptSource(node_name="reachy-001", capabilities=set())
    assert stub.name == "remote:reachy-001"


def test_stub_next_percept_non_blocking_returns_none_when_empty():
    """An empty inbox returns None — the loop treats this identically
    to a local source with no fresh percept, including the transient-
    unreachable case."""
    stub = _StubRemotePerceptSource(node_name="reachy-001", capabilities={"vision"})
    assert stub.next_percept() is None
    assert stub.next_percept() is None  # repeated calls stay safe


def test_stub_next_percept_drains_inbox_in_fifo_order():
    """The agent loop sees percepts in transport-arrival order."""
    stub = _StubRemotePerceptSource(node_name="reachy-001", capabilities={"vision"})
    p1 = Percept(timestamp=1.0, source="vision")
    p2 = Percept(timestamp=2.0, source="vision")
    stub._enqueue(p1)
    stub._enqueue(p2)
    assert stub.next_percept() is p1
    assert stub.next_percept() is p2
    assert stub.next_percept() is None


def test_stub_is_exhausted_only_after_explicit_end_of_stream():
    """Transient unreachability ≠ exhaustion. Only the peer's deliberate
    shutdown signal flips ``is_exhausted`` so the agent loop drops the
    source."""
    stub = _StubRemotePerceptSource(node_name="reachy-001", capabilities={"vision"})
    assert stub.is_exhausted() is False

    # Empty inbox alone does NOT mean exhausted (simulates the
    # transient-unreachable case).
    assert stub.next_percept() is None
    assert stub.is_exhausted() is False

    # Explicit shutdown signal flips it.
    stub._signal_end_of_stream()
    assert stub.is_exhausted() is True


def test_stub_capabilities_populated_from_handshake():
    """``capabilities`` is set at construction time, mirroring the 1.1
    transport's handshake population."""
    stub = _StubRemotePerceptSource(
        node_name="reachy-001",
        capabilities={"vision", "audio"},
    )
    assert stub.capabilities == {"vision", "audio"}


def test_stub_has_pending_reflects_inbox_state():
    """The optional ``has_pending`` extension lets the loop's idle-skip
    heuristic wake the loop when a percept arrives between ticks."""
    stub = _StubRemotePerceptSource(node_name="reachy-001", capabilities={"vision"})
    assert stub.has_pending() is False

    stub._enqueue(Percept(timestamp=1.0, source="vision"))
    assert stub.has_pending() is True

    stub.next_percept()
    assert stub.has_pending() is False
