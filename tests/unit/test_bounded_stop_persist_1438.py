"""Bounded stop PR 1a (#1438, #1520; docs/plans/bounded_stop.md §5): the session end saves first, within its deadline.

Today the hub's session end consolidates BEFORE it saves (a deadline that runs out during sleep loses every store),
the lightweight end never saves the hippocampus (only ``end_bio_session`` and ``AgentInstance.shutdown`` do), the
hippocampus and the stores keyed on its ids are saved outside any shared lock (an eviction between them leaves the
files disagreeing), and a shutdown that has to wait for a running loop never saves the cerebellum (#1520).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


def _agent(tmp_path: Path, agent_id: str = "persist_agent") -> Any:
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


def _hippocampus_file(inst: Any) -> Path:
    return Path(inst.hippocampus.config.persistence_path).expanduser()


def _stored_ids(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m["id"] for m in data["memories"]}


def test_a_deadline_too_short_for_sleep_still_saves_every_store_and_names_the_skipped_step(tmp_path: Path) -> None:
    from maxim.utils.deadline import Deadline

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    ticket = hub.on_session_start(owner=inst)  # idempotent: the instance opened it at build
    record_id = inst.hippocampus.store_observation("the kettle is on the left burner")
    slept: list[int] = []
    real_sleep = inst.hippocampus.sleep
    inst.hippocampus.sleep = lambda *a, **k: slept.append(1) or real_sleep(*a, **k)  # type: ignore[method-assign]

    results = hub.on_session_end(ticket, deadline=Deadline.after(0.0))

    assert slept == []  # no time left: consolidation is skipped ...
    assert "sleep" in results["consolidation_skipped"]  # ... by name
    assert record_id in _stored_ids(_hippocampus_file(inst))  # ... and the saves still ran
    for name in ("ec.json", "nac.json"):
        assert (tmp_path / "persist_agent" / name).exists(), name


def test_the_lightweight_end_saves_the_hippocampus(tmp_path: Path) -> None:
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    ticket = hub.on_session_start(owner=inst)
    record_id = inst.hippocampus.store_observation("the window faces east")

    hub.on_session_end_lightweight(ticket)

    assert record_id in _stored_ids(_hippocampus_file(inst))


def test_no_eviction_lands_between_the_hippocampus_save_and_the_ec_save(tmp_path: Path) -> None:
    """Evictions run their deletion callbacks (EC, embeddings, cross-layer) under the hippocampus WRITE lock, so a
    writer that gets in between the two saves leaves ``ec.json`` holding a signature for a memory the saved
    hippocampus still has (or the reverse). Inside one READ section no writer can."""
    inst = _agent(tmp_path)
    hub = inst.memory_hub
    ticket = hub.on_session_start(owner=inst)
    inst.hippocampus.store_observation("a memory so the stores have something to save")

    writer_got_in: list[bool] = []
    real_ec_save = hub.ec.save

    def _ec_save(*a: Any, **k: Any) -> None:
        acquired = threading.Event()

        def _writer() -> None:
            with inst.hippocampus._rwlock.write():
                acquired.set()

        threading.Thread(target=_writer, daemon=True).start()
        writer_got_in.append(acquired.wait(timeout=0.5))
        real_ec_save(*a, **k)

    hub.ec.save = _ec_save  # type: ignore[method-assign]
    hub.on_session_end_lightweight(ticket)

    assert writer_got_in == [False]


def test_a_shutdown_that_waits_for_a_running_loop_hands_its_after_steps_to_that_loop(tmp_path: Path) -> None:
    from maxim.utils.deadline import Deadline

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    loop_ticket = hub.on_session_start(owner=object(), live=True)  # a loop takes the session over
    saves: list[int] = []
    cerebellum = inst.bio_stack.cerebellum  # BioStack is frozen: spy on the store its save_cerebellum writes
    real_save = cerebellum.save
    cerebellum.save = lambda *a, **k: saves.append(1) or real_save(*a, **k)  # type: ignore[method-assign]

    inst.shutdown(deadline=Deadline.after(0.05))  # the loop is still running: the wait runs out
    assert saves == []  # nothing may run before the loop's own end ...

    hub.on_session_end_lightweight(loop_ticket)  # ... the loop ends its session
    assert saves == [1]  # ... and the handed-over after-step ran there, once
