"""#1217 items 1-2: ATL's query counter under concurrent recalls, and AgentPool's silent observation swallow.

``ATL.recall`` incremented ``_stats["queries"]`` outside any lock while other recalls (readers share the RWLock) did
the same and ``dump``/``stats`` copied the dict: concurrent recalls lost counts, and the counter is persisted. A
1 us switch interval (the #1135 pattern) loses increments in every trial on the pre-fix code.

``AgentPool.run_turn`` wrapped ``hippocampus.store_observation`` in a bare ``except Exception: pass``, which hid
#1135's race in production as a silently dropped observation.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path


from maxim.memory.atl import ATL


def test_concurrent_recalls_count_every_query() -> None:
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _trial in range(3):
            atl = ATL()
            n_threads, per = 8, 300
            barrier = threading.Barrier(n_threads)

            def recall(atl: ATL = atl, barrier: threading.Barrier = barrier) -> None:
                barrier.wait(timeout=10)
                for _ in range(per):
                    atl.recall(limit=1)

            threads = [threading.Thread(target=recall) for _ in range(n_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=60)
            assert not any(t.is_alive() for t in threads), "a recall thread is still running (deadlock?)"
            assert atl.dump()["stats"]["queries"] == n_threads * per
    finally:
        sys.setswitchinterval(old)


def test_a_failed_npc_observation_is_reported_and_the_turn_completes(tmp_path: Path, monkeypatch) -> None:
    import maxim.runtime.agent_pool as agent_pool
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.runtime.agent_pool import AgentPool

    reports: list[dict] = []
    monkeypatch.setattr(agent_pool, "log_swallowed_exception", lambda *a, **kw: reports.append(kw), raising=False)
    instance = AgentFactory(base_data_dir=str(tmp_path)).create_agent(
        AgentConfig(agent_id="npc_1217", personality="stern", remembers=True, policy=None)
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("hippocampus down")

    monkeypatch.setattr(instance.hippocampus, "store_observation", _boom)
    pool = AgentPool()
    pool.add(instance)
    try:
        result = pool.run_turn("npc_1217", "A dragon attacks.")
    finally:
        pool.shutdown()
    assert result is not None
    assert len(reports) == 1


def test_counter_copies_race_recalls_after_a_load_without_the_key() -> None:
    """End to end, the issue's own shape: a loaded state whose stats lack ``queries`` while ``dump``/``stats`` copy
    on other threads. NOT a deletion-proof guard on its own (the crash needs the copy to overlap the one key
    insertion, too narrow to hit reliably); the two deterministic tests below pin the copy lock and the load."""
    import traceback

    errors: list[str] = []
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _trial in range(3):
            atl = ATL()
            atl.load_state({**atl.dump(), "stats": {"concepts_stored": 0}})
            stop = threading.Event()

            def copier(atl: ATL = atl, stop: threading.Event = stop) -> None:
                try:
                    while not stop.is_set():
                        atl.dump()
                        atl.stats()
                except Exception:
                    errors.append(traceback.format_exc())

            def recaller(atl: ATL = atl) -> None:
                for _ in range(200):
                    atl.recall(limit=1)

            copiers = [threading.Thread(target=copier) for _ in range(3)]
            recallers = [threading.Thread(target=recaller) for _ in range(4)]
            for t in copiers + recallers:
                t.start()
            for t in recallers:
                t.join(timeout=60)
            stop.set()
            for t in copiers:
                t.join(timeout=60)
            assert not errors, errors[0]
            assert atl.stats()["queries"] == 4 * 200
            assert atl.stats()["concepts_stored"] == 0
    finally:
        sys.setswitchinterval(old)


class _SpyLock:
    """A lock that records every acquisition."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.acquired = 0

    def __enter__(self):
        self._lock.acquire()
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        self._lock.release()
        return False


def test_every_counter_copy_takes_the_stats_lock() -> None:
    atl = ATL()
    spy = _SpyLock()
    atl._stats_lock = spy
    atl.dump()
    after_dump = spy.acquired
    atl.stats()
    assert after_dump >= 1 and spy.acquired > after_dump


def test_load_state_fills_missing_counters_and_keeps_no_caller_dict() -> None:
    atl = ATL()
    loaded = {"concepts_stored": 3}
    atl.load_state({**atl.dump(), "stats": loaded})
    loaded["queries"] = 999  # a caller mutating its own dict must not reach the live counters
    assert {k: atl.stats()[k] for k in ("concepts_stored", "queries")} == {"concepts_stored": 3, "queries": 0}
