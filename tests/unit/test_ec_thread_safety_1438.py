"""#1438: the EC is safe to save while percepts and deletions still write it, and its lock is a LEAF.

The session end saves the EC while other threads may still write it: percepts on sensor/bus threads reach
``pattern_complete_or_separate`` / ``register_substrate_node`` / ``record_encoder_provenance`` through
``MemoryHub.on_percept_received``, and Hippocampus evictions call ``remove_signature`` from a deletion callback under
the Hippocampus WRITE lock. Before #1438 the EC had no lock: ``save``'s comprehensions iterated live dicts
("dictionary changed size during iteration") and the EC was not saved. The #1135 / #1217 forced-switching pattern
reproduces it on a quiet box.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pytest

from maxim.similarity.ec import ECConfig, EntorhinalCortex
from maxim.similarity.signature import SituationSignature


def _sig(n: int) -> SituationSignature:
    return SituationSignature(
        semantic_hash=tuple((n >> b) & 1 for b in range(8)),
        structural_hash=n * 7919,
        temporal_hash=(n % 24, n % 7, n % 4, n % 12),
        context_hash=n * 104729,
        tool_name=f"tool_{n % 5}",
        outcome_type="success" if n % 2 else "failure",
        mode="explore",
        goal_keywords=(f"kw{n}",),
    )


def _vec(rng: np.random.Generator, dim: int = 16) -> list[float]:
    return [float(x) for x in rng.standard_normal(dim)]


def _writer(ec: EntorhinalCortex, t: int, errors: list[str]) -> None:
    rng = np.random.default_rng(t)
    try:
        for i in range(120):
            mid = f"m{t}-{i}"
            ec.register(mid, _sig(i))
            res = ec.pattern_complete_or_separate(_vec(rng), "world", threshold=0.99, geometry=f"g{i % 2}")
            if res.is_new:
                ec.register_substrate_node(res.node_id, _vec(rng), "world", geometry=f"g{i % 2}")
            ec.record_encoder_provenance(f"sensor:world{t}", {"sensor_names": [f"s{i}"], "normalization": "x"})
            if i % 3 == 0:
                ec.remove_signature(f"m{t}-{i - 1}")
            if i % 5 == 0 and res.is_new:
                ec.remove_substrate_node(res.node_id)
    except Exception:
        errors.append(traceback.format_exc())


def _run_trial(ec: EntorhinalCortex, reads: list, errors: list[str]) -> None:
    stop = threading.Event()

    def reader(read) -> None:
        try:
            while not stop.is_set():
                read(ec)
        except Exception:
            errors.append(traceback.format_exc())

    readers = [threading.Thread(target=reader, args=(r,)) for r in reads]
    writers = [threading.Thread(target=_writer, args=(ec, t, errors)) for t in range(3)]
    for t in readers + writers:
        t.start()
    for t in writers:
        t.join(timeout=60)
    stop.set()
    for t in readers:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in readers + writers), "a thread is still running (deadlock?)"


@pytest.fixture
def forced_switching():
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(old)


def test_save_and_readers_race_writers_under_forced_switching(tmp_path: Path, forced_switching) -> None:
    errors: list[str] = []
    for trial in range(3):
        path = tmp_path / f"ec{trial}.json"
        ec = EntorhinalCortex(ECConfig(persistence_path=str(path)))
        rng = np.random.default_rng(99)

        reads = [
            lambda e: e.save(),
            lambda e: e.find_similar(_sig(3), k=5),
            lambda e: e.find_similar_situations_for_action("tool_1", {"mode": "explore"}),
            lambda e: e.query(tool="tool_2", hour=3),
            lambda e: (e.stats(), len(e), e.substrate_node_count, e.encoder_provenance),
            lambda e: e.pattern_complete_readonly(_vec(rng), "world", geometry="g0"),
            lambda e: e.find_semantic("hello"),
        ]
        _run_trial(ec, reads, errors)
        assert not errors, errors[0]
        ec.save()
        loaded = EntorhinalCortex(ECConfig(persistence_path=str(path)))
        loaded.load()
        assert len(loaded) == len(ec)
        assert loaded.substrate_node_count == ec.substrate_node_count
        assert json.loads(path.read_text())["_format_version"]


def test_save_snapshots_under_the_lock_and_writes_outside_it(tmp_path: Path, monkeypatch) -> None:
    """The file write must not hold the EC lock: a writer proceeds while the write is in flight."""
    import maxim.utils.atomic_io as atomic_io

    ec = EntorhinalCortex(ECConfig(persistence_path=str(tmp_path / "ec.json")))
    ec.register("m0", _sig(0))
    in_write = threading.Event()
    release = threading.Event()
    real = atomic_io.atomic_write_json

    def slow_write(*a, **kw):
        in_write.set()
        assert release.wait(10)
        return real(*a, **kw)

    monkeypatch.setattr(atomic_io, "atomic_write_json", slow_write)
    saver = threading.Thread(target=ec.save)
    saver.start()
    assert in_write.wait(10)
    done = threading.Event()
    threading.Thread(target=lambda: (ec.register("m1", _sig(1)), done.set()), daemon=True).start()
    assert done.wait(5), "register blocked behind save's file write: the write holds the EC lock"
    release.set()
    saver.join(10)
    # The snapshot was taken before m1: the file holds only m0.
    assert set(json.loads((tmp_path / "ec.json").read_text())["signatures"]) == {"m0"}


def test_no_call_out_under_the_lock(monkeypatch) -> None:
    """The activation trace (sim_log sinks) and the embedding store run with the EC lock FREE: a call-out that
    tries the lock from another thread must get it."""
    import maxim.similarity.ec as ec_mod

    ec = EntorhinalCortex(ECConfig())
    seen: list[bool] = []

    def probe(**_kw) -> None:
        got: list[bool] = []
        t = threading.Thread(target=lambda: got.append(ec._lock.acquire(timeout=2) and (ec._lock.release() or True)))
        t.start()
        t.join()
        seen.append(bool(got and got[0]))

    monkeypatch.setattr(ec_mod, "_emit_ec_activation", probe)
    ec.pattern_complete_or_separate([1.0, 0.0], "world", geometry="g")
    ec.register_substrate_node("n1", [1.0, 0.0], "world", geometry="g")
    ec.pattern_complete_or_separate([1.0, 0.0], "world", geometry="g")
    ec.pattern_complete_readonly([1.0, 0.0], "world", geometry="g")
    assert seen == [True, True, True]


def test_hippocampus_read_holder_saves_ec_while_a_writer_waits(tmp_path: Path) -> None:
    """Lock order: Hippocampus RWLock -> EC lock, and the EC lock is a leaf.

    The session end saves the EC inside the Hippocampus read lock while a capture worker waits on the write lock
    (whose deletion callbacks call ``ec.remove_signature``). The save must complete, and the writer then runs."""
    from maxim.memory.hippocampus import Hippocampus

    hippo = Hippocampus()
    ec = EntorhinalCortex(ECConfig(persistence_path=str(tmp_path / "ec.json")))
    for i in range(20):
        ec.register(f"m{i}", _sig(i))
    hippo.register_deletion_callback(ec.remove_signature)

    writer_done = threading.Event()

    def writer() -> None:
        with hippo._rwlock.write():
            ec.remove_signature("m1")
        writer_done.set()

    with hippo._rwlock.read():
        t = threading.Thread(target=writer, daemon=True)
        t.start()
        time.sleep(0.05)  # the writer is now queued on the write lock (writer-priority)
        saver = threading.Thread(target=ec.save, daemon=True)
        saver.start()
        saver.join(5)
        assert not saver.is_alive(), "EC.save deadlocked behind a queued Hippocampus writer"
    assert writer_done.wait(5)
    assert ec.get_signature("m1") is None


def test_the_snapshot_shares_no_live_list_with_the_index(tmp_path: Path, monkeypatch) -> None:
    """``LSHIndex.serialize`` returns the live bucket lists ``LSHIndex.add`` appends to; the snapshot must copy them,
    or a percept registered while the file is being written (outside the lock) lands in the half-written payload."""
    import maxim.utils.atomic_io as atomic_io

    ec = EntorhinalCortex(ECConfig(persistence_path=str(tmp_path / "ec.json")))
    ec.register("m0", _sig(0))
    ec.register_substrate_node("n0", [1.0, 0.0], "world", geometry="g")
    real = atomic_io.atomic_write_json

    def write_after_a_concurrent_register(path, data, *a, **kw):
        ec.register("late", _sig(0))  # same LSH bucket as m0; the EC lock is free during the write
        ec.register_substrate_node("n-late", [0.0, 1.0], "world", geometry="g")
        return real(path, data, *a, **kw)

    monkeypatch.setattr(atomic_io, "atomic_write_json", write_after_a_concurrent_register)
    ec.save()
    saved = json.loads((tmp_path / "ec.json").read_text())
    assert all("late" not in ids for table in saved["lsh"]["tables"] for ids in table.values())
    assert set(saved["signatures"]) == {"m0"}
    assert set(saved["substrate_nodes"]) == {"n0"}


def test_substrate_node_ids_and_modality_counts_are_snapshots() -> None:
    ec = EntorhinalCortex(ECConfig())
    ec.register_substrate_node("w1", [1.0, 0.0], "world", geometry="g")
    ec.register_substrate_node("w2", [0.0, 1.0], "world", geometry="g")
    ec.register_substrate_node("a1", [1.0, 0.0, 0.0], "audio", geometry="h")
    ec.register("m0", _sig(0))  # a signature is not a substrate node

    ids = ec.substrate_node_ids()
    counts = ec.substrate_modality_counts()
    assert ids == frozenset({"w1", "w2", "a1"})
    assert counts == {"world": 2, "audio": 1}
    ec.remove_substrate_node("w1")
    ec.register_substrate_node("w3", [1.0, 1.0], "world", geometry="g")
    assert ids == frozenset({"w1", "w2", "a1"})  # the snapshot did not follow the live store
    assert counts == {"world": 2, "audio": 1}
    assert ec.substrate_node_ids() == frozenset({"w2", "w3", "a1"})
