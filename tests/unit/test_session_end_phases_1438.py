"""The phased session end (#1438 PR 1a, #1520; ``integration/session_end.py``, docs/plans/bounded_stop.md §5).

Each test pins one mechanism of the end, on a REAL persistent agent where the mechanism is a composition: it saves
first, in one hippocampus read section that never re-takes the (non-re-entrant, writer-priority) lock; phase a flushes
the writers and a failing flush costs no save; a full end keeps the previous session's file as ``.backup``; the change
mark tells a closer whether anything reached the hippocampus after the end saved it; consolidation is skipped by name
when the deadline is short, and the bookkeeping ends after it; the loaders' prune runs once and the reverse index is
rebuilt after ATL loads; ``end_bio_session`` saves the hippocampus only when the hub's end did not.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from maxim.utils.deadline import Deadline


def _agent(tmp_path: Path, agent_id: str = "phases_agent") -> Any:
    """A REAL persistent agent (full bio-stack) homed in ``tmp_path``."""
    from maxim.runtime.agent_factory import AgentConfig, AgentFactory
    from maxim.tools.registry import ToolRegistry

    config = AgentConfig(
        agent_id=agent_id,
        role="pc",
        persistence_dir=str(tmp_path / agent_id),
        with_bio_stack=True,
        with_executor=False,
        with_pain_bridge=False,
        with_fear_gate=False,
        policy=None,
    )
    return AgentFactory(base_data_dir=tmp_path).create_full_agent(config, tool_registry=ToolRegistry(), auto_load=True)


def _home(tmp_path: Path, agent_id: str = "phases_agent") -> Path:
    return tmp_path / agent_id


def _stored_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m["id"] for m in data["memories"]}


def _end_in_thread(end: Any, timeout_s: float = 10.0) -> tuple[bool, list[BaseException]]:
    """Run ``end`` in a DAEMON thread joined for ``timeout_s``: a deadlock fails the test instead of hanging it."""
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            end()
        except BaseException as e:  # reported to the test, never swallowed
            errors.append(e)

    thread = threading.Thread(target=_run, daemon=True, name="test-session-end")
    thread.start()
    thread.join(timeout=timeout_s)
    return not thread.is_alive(), errors


# ── 1. saves first ────────────────────────────────────────────────────────────


def test_the_stores_are_saved_before_sleep_runs(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    home = _home(tmp_path)
    record_id = inst.hippocampus.store_observation("the blue mug is on the top shelf")
    assert not (home / "ec.json").exists() and record_id not in _stored_ids(home / "hippocampus.json")  # precondition

    seen_at_sleep: list[dict[str, Any]] = []
    real_sleep = inst.hippocampus.sleep

    def _sleep(*a: Any, **k: Any) -> Any:
        seen_at_sleep.append(
            {
                "hippocampus": record_id in _stored_ids(home / "hippocampus.json"),
                "ec": (home / "ec.json").exists(),
                "nac": (home / "nac.json").exists(),
            }
        )
        return real_sleep(*a, **k)

    inst.hippocampus.sleep = _sleep  # type: ignore[method-assign]

    results = hub.on_session_end(inst._session_ticket, deadline=Deadline.none())

    assert "consolidation_skipped" not in results  # no bound: sleep ran ...
    assert seen_at_sleep == [{"hippocampus": True, "ec": True, "nac": True}]  # ... after this session's saves


# ── 2. no re-entrant hippocampus lock inside the read section ─────────────────


def test_a_writer_queued_during_the_hippocampus_save_does_not_deadlock_the_end(tmp_path: Path) -> None:
    """The RWLock is writer-priority and NOT re-entrant: a save that re-took the read lock (``save``/``dump``) while
    a writer waits would deadlock the section. ``_save_locked`` must not."""
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    rwlock = inst.hippocampus._rwlock
    record_id = inst.hippocampus.store_observation("the spare key is under the mat")
    writer_done = threading.Event()
    queued: list[bool] = []
    real_save_locked = inst.hippocampus._save_locked

    def _save_locked(*a: Any, **k: Any) -> Any:
        def _writer() -> None:
            with rwlock.write():
                pass
            writer_done.set()

        threading.Thread(target=_writer, daemon=True, name="test-queued-writer").start()
        wait_until = time.monotonic() + 2.0
        while rwlock._writers_waiting == 0 and time.monotonic() < wait_until:
            time.sleep(0.005)
        queued.append(rwlock._writers_waiting > 0)
        return real_save_locked(*a, **k)

    inst.hippocampus._save_locked = _save_locked  # type: ignore[method-assign]

    finished, errors = _end_in_thread(lambda: hub.on_session_end_lightweight(inst._session_ticket), timeout_s=5.0)

    assert queued == [True]  # precondition: the writer really was queued while the section held the read lock
    assert finished, "the session end deadlocked: something in the read section re-took the hippocampus lock"
    assert errors == []
    assert writer_done.wait(timeout=2.0)  # the section released the lock and the writer got in afterwards
    assert record_id in _stored_ids(_home(tmp_path) / "hippocampus.json")


# ── 3. phase a: the writers are flushed ──────────────────────────────────────


def test_the_end_stops_every_capture_worker_and_saves_the_queued_capture(tmp_path: Path) -> None:
    from maxim.memory.encoding import EncodingSignals

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    hippocampus = inst.hippocampus
    worker = hippocampus.start_capture_worker()
    captured: list[str] = []
    real_capture = hippocampus.capture_from_loop

    def _slow_capture(*a: Any, **k: Any) -> str:
        time.sleep(0.3)  # the save would beat an unflushed capture
        memory_id = real_capture(*a, **k)
        captured.append(memory_id)
        return memory_id

    hippocampus.capture_from_loop = _slow_capture  # type: ignore[method-assign]
    hippocampus.capture_from_loop_async(
        observation={"detected_objects": ["kettle"]},
        state=None,
        intent={"goal": "boil water"},
        decision={"action": "look"},
        action={"tool": "focus_interests", "params": {}},
        result=None,
        run_id="phase-a",
        situation=None,
        encoding=EncodingSignals.unmeasured("api"),
    )

    hub.on_session_end_lightweight(inst._session_ticket)

    assert not worker.thread.is_alive()
    assert hippocampus._capture_workers == []
    assert len(captured) == 1 and captured[0] in _stored_ids(_home(tmp_path) / "hippocampus.json")


def test_the_end_drains_the_semantic_embedder_within_its_share(tmp_path: Path, caplog: Any) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    drains: list[float] = []

    class _Embedder:
        is_healthy = False  # not semantic-enabled: phase b saves no embedding store, phase a still drains

        def drain(self, timeout: float) -> bool:
            drains.append(timeout)
            return False

    hub.ec._neural_embedder = _Embedder()
    with caplog.at_level(logging.WARNING, logger="maxim.integration.session_end"):
        hub.on_session_end_lightweight(inst._session_ticket, deadline=Deadline.after(30.0))

    assert len(drains) == 1 and 0 < drains[0] <= 5.0  # its share of the deadline, capped at the old 5 s
    assert "semantic embedder still had work queued" in caplog.text


def _bare_embedder() -> Any:
    """A ``NeuralSemanticLSH`` with no model (``_initialized=True`` skips the load) and a hand-made async queue."""
    from maxim.similarity.semantic import NeuralSemanticLSH

    lsh = NeuralSemanticLSH(_initialized=True)
    lsh._embedding_queue = queue.Queue()
    lsh._shutdown_event = threading.Event()
    return lsh


def test_drain_waits_until_every_scheduled_embedding_reached_its_callback() -> None:
    lsh = _bare_embedder()
    lsh.embed = lambda text: time.sleep(0.05) or [0.0]  # type: ignore[method-assign]
    lsh.hash = lambda text: (0,)  # type: ignore[method-assign]
    done: list[str] = []
    for i in range(3):
        assert lsh.schedule_embedding(f"m{i}", "text", lambda memory_id, emb, bits: done.append(memory_id))
    worker = threading.Thread(target=lsh._embedding_worker, daemon=True)
    worker.start()
    try:
        assert lsh.drain(5.0) is True
        assert done == ["m0", "m1", "m2"]
    finally:
        lsh._shutdown_event.set()
        worker.join(timeout=2.0)


def test_drain_counts_a_failed_embedding_as_done() -> None:
    """The worker marks each item done in a ``finally``: a raising embed must not leave drain waiting forever."""
    lsh = _bare_embedder()

    def _boom(text: str) -> Any:
        raise RuntimeError("model fell over")

    lsh.embed = _boom  # type: ignore[method-assign]
    assert lsh.schedule_embedding("m0", "text")
    worker = threading.Thread(target=lsh._embedding_worker, daemon=True)
    worker.start()
    try:
        assert lsh.drain(2.0) is True
    finally:
        lsh._shutdown_event.set()
        worker.join(timeout=2.0)


def test_drain_returns_false_when_its_timeout_runs_out() -> None:
    lsh = _bare_embedder()  # no worker: the item is never processed
    assert lsh.schedule_embedding("m0", "text")
    started = time.monotonic()
    assert lsh.drain(0.05) is False
    assert time.monotonic() - started < 1.0
    assert _bare_embedder().drain(0.0) is True  # nothing queued


# ── 4. a failing flush costs no save ─────────────────────────────────────────


def test_a_raising_capture_worker_stop_still_saves_every_store(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    home = _home(tmp_path)
    record_id = inst.hippocampus.store_observation("the bike is chained by the gate")

    def _raise(timeout: float) -> bool:
        raise RuntimeError("capture worker wedged")

    inst.hippocampus.stop_capture_workers = _raise  # type: ignore[method-assign]

    hub.on_session_end_lightweight(inst._session_ticket)

    assert record_id in _stored_ids(home / "hippocampus.json")
    for name in ("ec.json", "nac.json", "atl.json", "scn.json", "angular_gyrus.json"):
        assert (home / name).exists(), name


# ── 5. the backup is the previous session's file ─────────────────────────────


def _previous_session_file(tmp_path: Path) -> bytes:
    first = _agent(tmp_path)
    first.hippocampus.store_observation("the previous session's memory")
    first.shutdown(consolidation="lightweight")
    return (_home(tmp_path) / "hippocampus.json").read_bytes()


def test_a_full_end_backs_up_the_file_as_it_was_before_this_session(tmp_path: Path) -> None:
    before = _previous_session_file(tmp_path)
    second = _agent(tmp_path)
    assert second.hippocampus.config.auto_save_after_sleep  # precondition: the backup is sleep's to replace
    second.hippocampus.store_observation("this session's memory")
    home = _home(tmp_path)
    assert not (home / "hippocampus.json.backup").exists()  # precondition

    second.memory_hub.on_session_end(second._session_ticket, deadline=Deadline.none())

    assert (home / "hippocampus.json.backup").read_bytes() == before
    assert (home / "hippocampus.json").read_bytes() != before


def test_a_lightweight_end_makes_no_backup(tmp_path: Path) -> None:
    _previous_session_file(tmp_path)
    second = _agent(tmp_path)
    second.hippocampus.store_observation("this session's memory")

    second.memory_hub.on_session_end_lightweight(second._session_ticket)

    assert not (_home(tmp_path) / "hippocampus.json.backup").exists()


# ── 6. the change mark ───────────────────────────────────────────────────────


def test_a_capture_after_the_end_is_unsaved_and_the_shutdown_after_steps_save_it(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    hub.on_session_end_lightweight(inst._session_ticket)
    assert hub.hippocampus_unsaved() is False  # precondition: the end saved it

    late_id = inst.hippocampus.store_observation("a percept that landed after the end")

    assert hub.hippocampus_unsaved() is True
    inst.shutdown()  # the session is closed: its after-steps run here
    assert late_id in _stored_ids(_home(tmp_path) / "hippocampus.json")


def test_with_no_change_after_the_end_the_hippocampus_is_not_saved_again(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    inst.hippocampus.store_observation("saved by the end")
    hub.on_session_end_lightweight(inst._session_ticket)
    saves: list[int] = []
    real_save = inst.hippocampus.save
    inst.hippocampus.save = lambda *a, **k: saves.append(1) or real_save(*a, **k)  # type: ignore[method-assign]

    assert hub.hippocampus_unsaved() is False
    inst.shutdown()
    assert saves == []


# ── 7. results and phase order ───────────────────────────────────────────────


def test_a_short_deadline_names_every_skipped_consolidation_step(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    assert hub._promoter is not None and hub.atl is not None and hub.angular_gyrus is not None  # preconditions

    results = hub.on_session_end(inst._session_ticket, deadline=Deadline.after(0.0))

    assert results["consolidation_skipped"] == ["sleep", "promotion", "atl", "angular_gyrus"]


def test_the_plan_manager_and_provenance_end_after_consolidation(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    order: list[str] = []

    def _spy(name: str, fn: Any) -> Any:
        return lambda *a, **k: order.append(name) or fn(*a, **k)

    class _PlanManager:
        def on_session_end(self) -> dict[str, Any]:
            order.append("plan_manager")
            return {"plan_saved": "plan.json"}

    class _Collector:
        def on_session_end(self) -> None:
            order.append("provenance")

    inst.hippocampus.sleep = _spy("sleep", inst.hippocampus.sleep)  # type: ignore[method-assign]
    hub._promoter.scan_for_promotions = _spy("promotion", hub._promoter.scan_for_promotions)
    hub.atl.consolidate = _spy("atl", hub.atl.consolidate)  # type: ignore[method-assign]
    hub.angular_gyrus.consolidate = _spy("angular_gyrus", hub.angular_gyrus.consolidate)  # type: ignore[method-assign]
    hub._plan_manager = _PlanManager()
    hub._collector = _Collector()

    results = hub.on_session_end(inst._session_ticket, deadline=Deadline.none())

    assert order == ["sleep", "promotion", "atl", "angular_gyrus", "plan_manager", "provenance"]
    assert results["plan_saved"] == "plan.json"


# ── 8. the loaders' prune and the reverse index ──────────────────────────────


def test_the_session_start_prune_runs_once_per_hub(tmp_path: Path) -> None:
    first = _agent(tmp_path)
    first.hippocampus.store_observation("a memory so the hippocampus file is read back")
    first.shutdown()
    second = _agent(tmp_path)
    hub = second.memory_hub
    assert hub._refs_pruned is True  # precondition: the build's open ran the pass
    calls: list[int] = []
    real_prune = hub.nac.prune_memory_refs
    hub.nac.prune_memory_refs = lambda ids: calls.append(1) or real_prune(ids)  # type: ignore[method-assign]
    hub.on_session_end_lightweight(second._session_ticket)

    ticket = hub.on_session_start(owner=object())  # a second FRESH open on the same hub

    assert ticket.fresh and calls == []
    assert "pruned_refs" not in ticket.start_stats
    hub.on_session_end_lightweight(ticket)


def test_a_restored_concept_ref_is_cleaned_when_its_memory_is_deleted_next_session(tmp_path: Path) -> None:
    """The reverse index is first built at hub construction, before ATL loads: rebuilt after the load, or a restored
    concept keeps pointing at a memory deleted in the next session."""
    first = _agent(tmp_path)
    victim = first.hippocampus.store_observation("the red kettle sits on the left burner")
    concept_id, _ = first.memory_hub.atl.find_or_create(name="kettle", category="object")
    first.memory_hub.atl.get(concept_id).add_ref("hippocampus", victim)  # by hand: the extractor's index never saw it
    first.shutdown()

    second = _agent(tmp_path)
    atl = second.memory_hub.atl
    assert victim in atl.get(concept_id).memory_refs.get("hippocampus", {})  # precondition: restored, not pruned

    second.hippocampus.remove(victim)

    assert victim not in atl.get(concept_id).memory_refs.get("hippocampus", {})


# ── 9. NAc.prune_memory_refs ─────────────────────────────────────────────────


def _link(link_id: str, memory_ids: list[str], *, imagined: bool = False) -> Any:
    from maxim.decisions.causal_link import CausalLink, TemporalDelta
    from maxim.decisions.causal_link import Valence

    return CausalLink(
        id=link_id,
        event_type="tool",
        event_signature="tool:look",
        event_context={},
        outcome_type="tool_result",
        outcome_signature="success",
        outcome_valence=Valence.POSITIVE,
        temporal_delta=TemporalDelta(),
        memory_ids=list(memory_ids),
        imagined=imagined,
    )


def test_nac_prune_memory_refs_drops_and_counts_dangling_ids_and_leaves_imagined_links() -> None:
    from maxim.decisions.nac import NAc

    nac = NAc()
    real = _link("real", ["m_alive", "m_gone_1", "m_gone_2"])
    imagined = _link("imagined", ["m_gone_1", "m_imagined"], imagined=True)
    other = _link("other", ["m_alive"])
    nac._links["tool:look"] = [real, imagined]
    nac._links["tool:walk"] = [other]

    assert nac.prune_memory_refs(frozenset({"m_alive"})) == 2

    assert real.memory_ids == ["m_alive"]
    assert imagined.memory_ids == ["m_gone_1", "m_imagined"]
    assert other.memory_ids == ["m_alive"]
    assert nac.prune_memory_refs(frozenset({"m_alive"})) == 0  # idempotent


# ── 10. end_bio_session's fallback save ──────────────────────────────────────


def test_end_bio_session_saves_the_hippocampus_when_the_hub_end_did_not(tmp_path: Path) -> None:
    from maxim.integration.session_ownership import SessionTicket
    from maxim.runtime.bio_integration import end_bio_session

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    record_id = inst.hippocampus.store_observation("the plant needs water on fridays")
    stale = SessionTicket(generation=999, owner=object())
    saves: list[int] = []
    real_save = inst.hippocampus.save
    inst.hippocampus.save = lambda *a, **k: saves.append(1) or real_save(*a, **k)  # type: ignore[method-assign]

    end_bio_session(memory_hub=hub, ticket=stale, hippocampus=inst.hippocampus, consolidation="lightweight")

    assert saves == [1]  # the hub's end returned {} for the stale ticket: this save is the only one
    assert record_id in _stored_ids(_home(tmp_path) / "hippocampus.json")
    hub.on_session_end_lightweight(inst._session_ticket)


def test_end_bio_session_does_not_save_the_hippocampus_twice(tmp_path: Path) -> None:
    from maxim.runtime.bio_integration import end_bio_session

    inst = _agent(tmp_path)
    record_id = inst.hippocampus.store_observation("the plant needs water on fridays")
    saves: list[int] = []
    real_save = inst.hippocampus.save
    inst.hippocampus.save = lambda *a, **k: saves.append(1) or real_save(*a, **k)  # type: ignore[method-assign]

    end_bio_session(
        memory_hub=inst.memory_hub,
        ticket=inst._session_ticket,
        hippocampus=inst.hippocampus,
        consolidation="lightweight",
    )

    assert saves == []  # the hub's end saved it (in its read section)
    assert record_id in _stored_ids(_home(tmp_path) / "hippocampus.json")
