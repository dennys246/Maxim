"""Tests for runtime/llm_call_registry.py.

Pin the load-bearing invariants:
- Register start / end semantics + idempotency
- last_byte_at tracking via register_byte_received
- ContextVar-based current_call_id (no signature threading needed)
- Tier filtering for any_call_in_flight + oldest_byte_silence_s
- Stale entry auto-filter on read (SIGKILL defense)
- Thread safety under concurrent start/end/query
"""

from __future__ import annotations

import threading
import time

import pytest

from maxim.runtime import llm_call_registry as r


@pytest.fixture(autouse=True)
def _reset_registry():
    r._reset_for_tests()
    yield
    r._reset_for_tests()


# ─── Register start / end + in-flight query ──────────────────────────────


def test_register_start_marks_in_flight():
    assert not r.any_call_in_flight()
    cid = r.register_call_start(tier="large")
    assert r.any_call_in_flight()
    assert r.any_call_in_flight(tier="large")
    assert not r.any_call_in_flight(tier="medium")
    r.register_call_end(cid)
    assert not r.any_call_in_flight()


def test_register_end_unknown_call_id_silently_noops():
    r.register_call_end("not-a-real-call-id")
    # No exception, no state change
    assert not r.any_call_in_flight()


def test_register_end_idempotent():
    cid = r.register_call_start(tier="large")
    r.register_call_end(cid)
    r.register_call_end(cid)  # second call no-ops
    assert not r.any_call_in_flight()


# ─── Multiple concurrent calls track independently ───────────────────────


def test_multiple_calls_tracked_per_call_id():
    cid1 = r.register_call_start(tier="large")
    cid2 = r.register_call_start(tier="medium")
    cid3 = r.register_call_start(tier="large")
    assert r.any_call_in_flight()
    assert r.any_call_in_flight(tier="large")
    assert r.any_call_in_flight(tier="medium")
    assert not r.any_call_in_flight(tier="small")
    r.register_call_end(cid1)
    assert r.any_call_in_flight(tier="large")  # cid3 still in flight
    r.register_call_end(cid3)
    assert not r.any_call_in_flight(tier="large")
    assert r.any_call_in_flight(tier="medium")  # cid2 still in flight
    r.register_call_end(cid2)
    assert not r.any_call_in_flight()


# ─── current_call_id (ContextVar) ────────────────────────────────────────


def test_current_call_id_set_on_start_cleared_on_end():
    assert r.current_call_id() is None
    cid = r.register_call_start(tier="large")
    assert r.current_call_id() == cid
    r.register_call_end(cid)
    assert r.current_call_id() is None


def test_register_call_end_lifo_semantics():
    """ContextVar follows LIFO stack semantics — the realistic dispatch
    pattern (inner call finishes before outer continues). Ending in LIFO
    order correctly restores each parent's call_id.
    """
    outer = r.register_call_start(tier="large")
    inner = r.register_call_start(tier="medium")
    assert r.current_call_id() == inner
    # LIFO: inner ends first, restoring outer
    r.register_call_end(inner)
    assert r.current_call_id() == outer
    # Then outer ends, restoring to None
    r.register_call_end(outer)
    assert r.current_call_id() is None


def test_nested_dispatch_inner_end_restores_outer_call_id():
    """Stack-token semantics (review C2 fix): nested register_call_start
    sets contextvar to inner cid; inner register_call_end MUST restore
    the OUTER cid so the outer dispatch's byte-arrival path stays alive.

    Pre-fix: inner end cleared the contextvar; outer's register_byte_received
    became a silent no-op until the outer ended. Wedged-call detection would
    fire erroneously while the outer call was actually receiving bytes.
    """
    outer = r.register_call_start(tier="large")
    assert r.current_call_id() == outer
    inner = r.register_call_start(tier="medium")  # nested
    assert r.current_call_id() == inner
    r.register_call_end(inner)
    # CRITICAL: contextvar restored to outer, NOT cleared
    assert r.current_call_id() == outer, "nested end stranded the outer dispatch's contextvar (C2 regression)"
    # Bytes received during the outer call's continued execution land
    # against the OUTER call_id, not silently no-op.
    r.register_byte_received()
    # Outer call's last_byte_at should be near zero silence.
    silence = r.oldest_byte_silence_s(tier="large")
    assert silence is not None and silence < 0.02
    r.register_call_end(outer)
    assert r.current_call_id() is None


# ─── Byte tracking ───────────────────────────────────────────────────────


def test_register_byte_received_updates_last_byte_at():
    r.register_call_start(tier="large")
    time.sleep(0.05)  # accumulate some silence
    initial_silence = r.oldest_byte_silence_s(tier="large")
    assert initial_silence is not None and initial_silence >= 0.04
    r.register_byte_received()  # reads ContextVar
    after_silence = r.oldest_byte_silence_s(tier="large")
    assert after_silence is not None and after_silence < initial_silence


def test_register_byte_received_with_explicit_call_id():
    cid1 = r.register_call_start(tier="large")
    r.register_call_start(tier="large")  # second call shadows cid1's contextvar
    time.sleep(0.05)
    r.register_byte_received(call_id=cid1)  # update OLD call's byte clock explicitly
    # Both calls in flight; oldest_byte_silence_s returns the min (most recent)
    s = r.oldest_byte_silence_s(tier="large")
    assert s is not None and s < 0.04, f"expected near-zero silence on min; got {s}"


def test_register_byte_received_unknown_call_id_noops():
    r.register_byte_received(call_id="not-real")  # no exception
    r.register_byte_received()  # no active call; no exception
    assert not r.any_call_in_flight()


# ─── oldest_byte_silence_s edge cases ────────────────────────────────────


def test_oldest_byte_silence_s_none_when_no_calls():
    assert r.oldest_byte_silence_s() is None
    assert r.oldest_byte_silence_s(tier="large") is None


def test_oldest_byte_silence_s_tier_filtered():
    r.register_call_start(tier="large")
    time.sleep(0.05)
    r.register_call_start(tier="medium")  # younger
    # large tier: only one call, silence ~ 0.05s
    s_large = r.oldest_byte_silence_s(tier="large")
    assert s_large is not None and s_large >= 0.04
    # medium tier: just registered, silence ~ 0
    s_medium = r.oldest_byte_silence_s(tier="medium")
    assert s_medium is not None and s_medium < 0.02


# ─── Stale entry auto-filter ─────────────────────────────────────────────


def test_stale_entry_filtered_on_read(monkeypatch):
    """Entries older than _STALE_ENTRY_TTL_S auto-filter on read.

    Simulate by patching the TTL to a small value, registering, sleeping
    past the TTL, then querying.
    """
    monkeypatch.setattr(r, "_STALE_ENTRY_TTL_S", 0.1)
    r.register_call_start(tier="large")
    assert r.any_call_in_flight()
    time.sleep(0.2)  # exceed TTL
    assert not r.any_call_in_flight(), "stale entry should be filtered on read"
    assert r.oldest_byte_silence_s(tier="large") is None


# ─── Thread safety under concurrent operations ───────────────────────────


def test_concurrent_register_and_end_threadsafe():
    """Register + end across 8 threads × 50 ops each. Final state must be empty."""
    n_threads = 8
    ops_per_thread = 50

    def worker():
        for _ in range(ops_per_thread):
            cid = r.register_call_start(tier="large")
            r.register_byte_received(call_id=cid)
            r.register_call_end(cid)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not r.any_call_in_flight(), "all calls should have been deregistered"


def test_concurrent_query_during_mutation():
    """Daemon thread queries while registrations happen; no lock errors / exceptions."""
    stop = threading.Event()
    errors: list[Exception] = []

    def query_loop():
        try:
            while not stop.is_set():
                r.any_call_in_flight(tier="large")
                r.oldest_byte_silence_s(tier="large")
        except Exception as exc:  # pragma: no cover — defensive
            errors.append(exc)

    q = threading.Thread(target=query_loop, daemon=True)
    q.start()
    try:
        for _ in range(100):
            cid = r.register_call_start(tier="large")
            r.register_byte_received(call_id=cid)
            r.register_call_end(cid)
    finally:
        stop.set()
        q.join(timeout=2.0)

    assert not errors, f"query loop errored: {errors!r}"
    assert not r.any_call_in_flight()
