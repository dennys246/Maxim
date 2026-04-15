"""P3.5 Stage 2 — non-empty PerceptTraceBuffer round-trip tests.

Stage 1 shipped the empty-buffer round-trip in
``tests/unit/test_bio_system_snapshot.py::TestPTBRoundTrip``. Stage 2
covers the edge cases the plan deferred:

- 100+ entries with state-equality on reload
- Three agents with mixed entries — agent isolation survives
- Ring-buffer head/tail invariant when dump-size > live max_entries
- Concurrent insertion against ``dump()`` — the snapshot is point-in-time
  consistent (no torn reads)
- Tick counter survives the round-trip
- Activation-strength values restore exactly (no implicit re-record at 1.0)
- Tuning-drift WARN fires when dumped tuning ≠ live tuning, and live
  tuning wins
"""

from __future__ import annotations

import logging
import threading

import pytest

from maxim.memory.percept_trace_buffer import PerceptTraceBuffer
from maxim.memory.snapshot import ptb_from_snapshot, ptb_to_snapshot


def _drain_to_dump(buf: PerceptTraceBuffer) -> dict:
    """Helper: dump current state to envelope payload."""
    env = ptb_to_snapshot(buf)
    return env


def _restore(env: dict, *, max_entries: int = 500, tau: float = 10.0) -> PerceptTraceBuffer:
    fresh = PerceptTraceBuffer(max_entries=max_entries, tau=tau)
    ptb_from_snapshot(env, fresh)
    return fresh


class TestNonEmptyRoundTrip:
    def test_100_entries_survive_round_trip(self):
        buf = PerceptTraceBuffer(max_entries=200)
        for i in range(120):
            buf.record("agent_a", f"percept_{i}")
        env = _drain_to_dump(buf)

        restored = _restore(env, max_entries=200)

        # Tick counter, length, ids, activation all match
        assert len(restored) == 120
        original = buf.snapshot(min_activation=0.0)
        post = restored.snapshot(min_activation=0.0)
        assert len(original) == len(post)
        # Order is by activation desc; here all are equal so compare as set
        assert {(e.percept_id, e.agent_id) for e in original} == {(e.percept_id, e.agent_id) for e in post}

    def test_three_agents_mixed_entries_survive(self):
        buf = PerceptTraceBuffer(max_entries=200)
        for agent in ("alice", "bob", "carol"):
            for i in range(30):
                buf.record(agent, f"{agent}_p{i}")
        env = _drain_to_dump(buf)

        restored = _restore(env, max_entries=200)

        for agent in ("alice", "bob", "carol"):
            pre = sorted(e.percept_id for e in buf.snapshot(agent_id=agent, min_activation=0.0))
            post = sorted(e.percept_id for e in restored.snapshot(agent_id=agent, min_activation=0.0))
            assert pre == post
            assert len(post) == 30

    def test_activation_strength_restored_exactly(self):
        """Pre-fix legacy harness re-recorded entries at activation=1.0,
        which lost the τ-decay state. Stage 1 load_state restores activation
        from the dump verbatim."""
        buf = PerceptTraceBuffer(tau=10.0)
        buf.record("a", "old")
        for _ in range(5):
            buf.tick()  # decays 'old' to ~0.6065
        buf.record("a", "fresh")  # activation = 1.0

        env = _drain_to_dump(buf)
        restored = _restore(env)

        original_by_id = {e.percept_id: e.activation_strength for e in buf.snapshot(min_activation=0.0)}
        post_by_id = {e.percept_id: e.activation_strength for e in restored.snapshot(min_activation=0.0)}
        assert original_by_id == post_by_id
        # And neither is rounded to 1.0
        assert post_by_id["old"] < 0.7
        assert post_by_id["fresh"] == pytest.approx(1.0)

    def test_tick_counter_survives(self):
        buf = PerceptTraceBuffer()
        for _ in range(42):
            buf.tick()
        env = _drain_to_dump(buf)
        restored = _restore(env)
        assert restored.current_tick == 42


class TestRingBufferHeadTailInvariant:
    def test_dump_larger_than_live_max_entries_keeps_tail(self):
        """If a v1 snapshot was dumped with max_entries=500 and is loaded
        into an instance constructed with max_entries=100, the most recent
        100 entries must survive — never more than max_entries."""
        large = PerceptTraceBuffer(max_entries=500)
        for i in range(400):
            large.record("a", f"p{i:04d}")
        env = _drain_to_dump(large)
        assert len(env["payload"]["entries"]) == 400

        # Load into a smaller buffer: ring-buffer invariant trims to 100
        small = PerceptTraceBuffer(max_entries=100)
        ptb_from_snapshot(env, small)
        assert len(small) == 100

        # The kept entries are the TAIL (most recently inserted)
        ids = sorted(e.percept_id for e in small.snapshot(min_activation=0.0))
        assert ids == [f"p{i:04d}" for i in range(300, 400)]

    def test_dump_smaller_than_live_max_entries_loads_all(self):
        """Reverse direction — small dump into large buffer fits cleanly."""
        small = PerceptTraceBuffer(max_entries=50)
        for i in range(30):
            small.record("a", f"p{i}")
        env = _drain_to_dump(small)

        large = PerceptTraceBuffer(max_entries=500)
        ptb_from_snapshot(env, large)
        assert len(large) == 30


class TestConcurrentInsertionDuringDump:
    def test_dump_under_concurrent_writers_is_consistent(self):
        """A torn read would manifest as a partial entry / KeyError. The
        dump uses ``self._lock`` so the snapshot is point-in-time consistent.
        We can't easily prove monotone-snapshot-size from a single dump,
        but we CAN prove no entry is corrupted (every TraceEntry has
        all fields populated and round-trips cleanly).

        Stage 2 review I5 fold — no fixed dump count or wall-clock budget
        is asserted because that pattern flakes on slow CI under lock
        contention. Instead the dumper runs until the writers are
        signaled to stop; the load-bearing invariant is "every dump
        is well-formed," not "we managed N dumps in M seconds."
        """
        buf = PerceptTraceBuffer(max_entries=10000)
        stop = threading.Event()
        errors: list[BaseException] = []
        dumps: list[dict] = []

        def writer(agent_id: str) -> None:
            i = 0
            try:
                while not stop.is_set():
                    buf.record(agent_id, f"{agent_id}_p{i}")
                    i += 1
            except BaseException as exc:  # pragma: no cover - debugging path
                errors.append(exc)

        def dumper() -> None:
            try:
                while not stop.is_set():
                    dumps.append(_drain_to_dump(buf))
            except BaseException as exc:  # pragma: no cover - debugging path
                errors.append(exc)

        writers = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(4)]
        d = threading.Thread(target=dumper)
        for w in writers:
            w.start()
        d.start()

        # Let the threads run briefly, then signal stop. Even on a
        # heavily-loaded CI runner the dumper should accumulate at
        # least one dump in this window.
        threading.Event().wait(timeout=0.5)
        stop.set()

        d.join(timeout=10)
        for w in writers:
            w.join(timeout=10)

        assert not errors, f"thread errors: {errors}"
        # The exact count is non-deterministic; existence is the floor.
        assert len(dumps) >= 1, "dumper produced no snapshots"
        # Every dump's entries must round-trip cleanly into a fresh buffer
        for env in dumps:
            restored = _restore(env, max_entries=10000)
            # Every restored entry's fields are well-formed
            for e in restored.snapshot(min_activation=0.0):
                assert isinstance(e.agent_id, str) and e.agent_id
                assert isinstance(e.percept_id, str) and e.percept_id
                assert isinstance(e.tick, int)
                assert isinstance(e.activation_strength, float)
                assert isinstance(e.registered_at, float)


class TestLegacyHarnessFidelity:
    """Stage 2 review M4 fold — the legacy per-component harness path
    (``tests/substrate/persistence_harness._save_component`` +
    ``persistence_child._load_component``) MUST round-trip activation
    strength faithfully. Pre-fold it lossily replayed entries via
    ``buf.record()`` which reset every entry to activation=1.0.
    """

    def test_legacy_path_preserves_activation_strength(self):
        """Simulates the legacy save → child load cycle in-process,
        without spawning a subprocess. The save_component writes via
        ``component.dump()`` and the child loads via ``ptb.load_state``."""
        from maxim.utils.atomic_io import atomic_write_json
        import json
        import tempfile
        from pathlib import Path

        # Build a PTB with mixed activation strengths via decay
        buf = PerceptTraceBuffer(tau=10.0)
        buf.record("a", "old")
        for _ in range(5):
            buf.tick()
        buf.record("a", "fresh")
        original_dump = buf.dump()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ptb.json"
            atomic_write_json(str(path), buf.dump())

            # Mirror the child's _load_component PTB branch
            with open(path) as f:
                data = json.load(f)
            restored = PerceptTraceBuffer()
            restored.load_state(data)

        # Activation values must match exactly — no implicit reset to 1.0
        original_by_id = {e.percept_id: e.activation_strength for e in buf.snapshot(min_activation=0.0)}
        post_by_id = {e.percept_id: e.activation_strength for e in restored.snapshot(min_activation=0.0)}
        assert original_by_id == post_by_id
        assert post_by_id["old"] < 0.7  # decayed
        assert post_by_id["fresh"] == pytest.approx(1.0)
        assert restored.dump() == original_dump


class TestTuningDriftWarning:
    def test_drift_logs_warn_and_live_wins(self, caplog):
        original = PerceptTraceBuffer(max_entries=100, tau=10.0, tick_rate=1.0, min_activation=0.01)
        original.record("a", "p1")
        env = _drain_to_dump(original)

        live = PerceptTraceBuffer(max_entries=200, tau=20.0, tick_rate=2.0, min_activation=0.05)
        with caplog.at_level(logging.WARNING, logger="maxim.memory.percept_trace_buffer"):
            ptb_from_snapshot(env, live)

        assert any("tuning drift" in r.message for r in caplog.records)
        # Live tuning wins — instance fields unchanged
        assert live._max_entries == 200
        assert live._tau == 20.0
        assert live._tick_rate == 2.0
        assert live._min_activation == 0.05
        # State (entries + tick) was loaded
        assert len(live) == 1

    def test_no_drift_no_warn(self, caplog):
        original = PerceptTraceBuffer(max_entries=100, tau=10.0)
        env = _drain_to_dump(original)
        live = PerceptTraceBuffer(max_entries=100, tau=10.0)
        with caplog.at_level(logging.WARNING, logger="maxim.memory.percept_trace_buffer"):
            ptb_from_snapshot(env, live)
        assert not any("tuning drift" in r.message for r in caplog.records)
