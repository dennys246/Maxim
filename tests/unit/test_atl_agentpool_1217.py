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

import pytest

from maxim.memory.atl import ATL


@pytest.mark.xfail(strict=True, reason="#1217: ATL.recall's query counter is incremented outside any lock")
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


@pytest.mark.xfail(strict=True, reason="#1217: AgentPool.run_turn swallows a failed observation silently")
def test_a_failed_npc_observation_is_reported_and_the_turn_completes(tmp_path: Path, monkeypatch) -> None:
    import maxim.runtime.agent_pool as agent_pool
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.runtime.agent_pool import AgentPool

    reports: list[dict] = []
    monkeypatch.setattr(agent_pool, "log_swallowed_exception", lambda *a, **kw: reports.append(kw), raising=False)
    instance = AgentFactory(base_data_dir=str(tmp_path)).create_agent(
        AgentConfig(agent_id="npc_1217", personality="stern", remembers=True)
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
