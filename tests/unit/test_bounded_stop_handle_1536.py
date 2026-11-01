"""Red gates for #1536: a per-handle ``MaximHandle.stop(deadline=)`` bounds the Talk loop's own session end.

``stop(deadline=)`` makes no process stop request (owner decision 2026-10-30), and the loop's ``bio_session`` exit read
only the process stop, so the loop's full consolidation ran past the embedder's kill. The design (approach note v5,
owner decisions 2026-10-31): the handle gives each Talk loop it builds an ``EndBound``; ``stop()`` sets it before
signalling; the bound cuts CONSOLIDATION only (persist, flushes and abandon follow the process stop alone); a loop
already ending (a rest, a rebuild, a straggler) is bounded too. Each gate asserts its surface first, so on the pre-fix
code it fails by assertion (``xfail(strict=True, raises=AssertionError)`` in the red-gate commit), flipped here.
"""

from __future__ import annotations

import inspect
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.utils import deadline as deadline_mod
from maxim.utils.cancellation import request_stop
from maxim.utils.deadline import Deadline


def _agent(tmp_path: Path, agent_id: str = "bound_agent") -> Any:
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
    )
    return AgentFactory(base_data_dir=tmp_path).create_full_agent(config, tool_registry=ToolRegistry(), auto_load=True)


def _has_surface() -> None:
    from maxim.runtime.bio_integration import bio_session

    assert hasattr(deadline_mod, "EndBound"), "no EndBound: nothing carries a handle's bound to its loop"
    assert hasattr(Deadline, "earliest")
    assert "end_bound" in inspect.signature(bio_session.__wrapped__).parameters


# ── G10: the bound source ───────────────────────────────────────────────────────────────────────────────────────


def test_g10_earliest_ignores_unbounded_and_the_holder_only_narrows() -> None:
    _has_surface()
    a, b = Deadline.after(5.0), Deadline.after(1.0)
    assert Deadline.earliest(a, b) == b
    assert Deadline.earliest(None, Deadline.none(), a) == a
    assert Deadline.earliest(None, Deadline.none()) is None
    holder = deadline_mod.EndBound()
    assert holder.get() is None
    holder.set(a)
    holder.set(Deadline.after(60.0))  # a later deadline never widens
    assert holder.get() == a
    holder.set(b)
    assert holder.get() == b


# ── G1: the loop's clean end under a handle bound alone ─────────────────────────────────────────────────────────


def test_g1_a_handle_bound_cuts_the_clean_ends_consolidation_and_still_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _has_surface()
    from maxim.runtime.bio_integration import bio_session

    inst = _agent(tmp_path)
    hub = inst.memory_hub
    inst.hippocampus.store_observation("the red ball is under the bed")
    inst.hippocampus.flush(timeout=5.0)
    slept: list[bool] = []
    monkeypatch.setattr(hub.hippocampus, "sleep", lambda *a, **k: (slept.append(True), time.sleep(3.0), {})[2])
    holder = deadline_mod.EndBound()
    started = time.monotonic()
    with bio_session(
        memory_hub=hub, hippocampus=inst.hippocampus, owner=object(), consolidation="full", end_bound=holder
    ):
        holder.set(Deadline.after(2.0))
    assert slept == [], "sleep ran: the handle's bound did not reach the loop's consolidation"
    assert time.monotonic() - started < 2.5
    path = Path(inst.hippocampus.config.persistence_path).expanduser()
    assert "red ball" in path.read_text(encoding="utf-8"), "the bounded end did not persist"


# ── G6/G9: with a process stop too, abandon and flush follow the process stop alone ─────────────────────────────


def test_g6_a_spent_handle_bound_never_abandons_an_abnormal_exit_the_process_stop_still_allows(
    tmp_path: Path,
) -> None:
    _has_surface()
    from maxim.runtime.bio_integration import bio_session

    inst = _agent(tmp_path)
    inst.hippocampus.store_observation("the yellow kite is on the roof")
    inst.hippocampus.flush(timeout=5.0)
    holder = deadline_mod.EndBound()
    holder.set(Deadline.after(0.0))  # spent
    request_stop("test", Deadline.after(10.0))  # the process stop still has time
    with pytest.raises(RuntimeError):
        with bio_session(memory_hub=inst.memory_hub, hippocampus=inst.hippocampus, owner=object(), end_bound=holder):
            raise RuntimeError("the loop failed")
    path = Path(inst.hippocampus.config.persistence_path).expanduser()
    assert "yellow kite" in path.read_text(encoding="utf-8"), "the spent handle bound abandoned the session"


# ── G7: consolidation only, so the flushes keep their no-bound caps ─────────────────────────────────────────────


def test_g7_a_spent_handle_bound_leaves_the_capture_flush_at_its_no_bound_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _has_surface()
    from maxim.runtime.bio_integration import bio_session

    inst = _agent(tmp_path)
    timeouts: list[float] = []
    real = inst.hippocampus.stop_capture_worker
    monkeypatch.setattr(
        inst.hippocampus,
        "stop_capture_worker",
        lambda w, timeout=None: (timeouts.append(timeout), real(w, timeout=timeout))[1],
    )
    holder = deadline_mod.EndBound()
    with bio_session(
        memory_hub=inst.memory_hub, hippocampus=inst.hippocampus, owner=object(), consolidation="full", end_bound=holder
    ):
        holder.set(Deadline.after(0.0))
    assert timeouts and timeouts[0] == 5.0, f"the capture flush was cut by the handle bound: {timeouts}"


# ── G8/G14/G15: the bound is read before each step, with its reserve computed once ─────────────────────────────


def test_g8_a_bound_set_after_consolidation_began_cuts_the_steps_still_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    _has_surface()
    from maxim.integration import session_end

    assert "consolidation_bound" in inspect.signature(session_end._consolidate).parameters
    holder: dict[str, Deadline | None] = {"d": None}
    hub = MagicMock()
    hub.hippocampus.__len__.return_value = 0
    hub.hippocampus.sleep.side_effect = lambda **k: holder.update(d=Deadline.after(0.0)) or {}
    results: dict[str, Any] = {}
    ran = session_end._consolidate(hub, Deadline.none(), results, consolidation_bound=lambda: holder["d"])
    assert ran == ["sleep"], ran
    assert results["consolidation_skipped"] == ["promotion", "atl", "angular_gyrus"]


def test_g15_the_reserve_on_a_handle_bound_is_sized_once_per_bound() -> None:
    _has_surface()
    from maxim.runtime import bio_integration

    assert hasattr(bio_integration, "_reserved_consolidation_bound")
    holder = deadline_mod.EndBound()
    holder.set(Deadline.after(20.0))
    read = bio_integration._reserved_consolidation_bound(holder)
    first = read()
    time.sleep(0.05)
    second = read()
    assert first is not None and second is not None
    assert second.at == first.at, "the reserve was re-derived from a falling remaining time"
    assert 19.0 - 5.0 - 0.5 < first.remaining() < 20.0 - 5.0 + 0.1  # ~25% reserve


def test_g15b_the_reserve_is_sized_when_the_stop_sets_the_bound_not_at_the_first_read() -> None:
    """Review SF-A (all three lenses): the flushes and persist run before the first consolidation read, so a reserve
    sized at that read shrank by whatever they spent. It is sized from the time left when the bound was set."""
    from maxim.runtime import bio_integration

    holder = deadline_mod.EndBound()
    raw = Deadline.after(20.0)  # above the 1 s floor: the 25% share is what moves
    holder.set(raw)
    read = bio_integration._reserved_consolidation_bound(holder)
    assert read is not None
    time.sleep(1.0)  # the end's work before consolidation
    first = read()
    assert first is not None
    assert first.at == pytest.approx(raw.at - 5.0, abs=0.05), "the reserve was sized at the read, not at the set"


def test_a_talk_racing_stop_cannot_launch_a_loop_stop_never_sees(tmp_path: Path) -> None:
    """Review SF-2: stop() sets _stopped before its _talk_lock section, so the launch re-checks it under the lock."""
    h = _handle()
    h._stopped = True
    h._talk_generation = None
    h._workspace = tmp_path / "ws"
    with pytest.raises(RuntimeError, match="stopped"):
        h._ensure_talk_loop()
    assert h._talk_thread is None


# ── G1h/G11/G16: the handle sets the bound before signalling, on the live loop and on loops still ending ────────


def _handle() -> Any:
    from maxim.console.handle import MaximHandle

    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "t"
    h._stopped = False
    h._stop_result = None
    h._campaign_lock = threading.Lock()
    h._talk_lock = threading.Lock()
    h._talk_bridge = h._talk_thread = h._talk_stop = h._talk_worker = None
    h._talk_bound = None
    h._ending_talk_loops = []
    h.instance = MagicMock()
    h.instance.shutdown.return_value = True
    return h


def test_g1h_stop_sets_the_live_talk_loops_bound_before_it_signals_the_loop() -> None:
    _has_surface()
    h = _handle()
    holder = deadline_mod.EndBound()
    seen: list[Deadline | None] = []
    bridge = MagicMock()
    bridge.finish.side_effect = lambda: seen.append(holder.get())
    h._talk_bridge, h._talk_thread, h._talk_stop, h._talk_bound = bridge, None, threading.Event(), holder
    d = Deadline.after(2.0)
    h.stop(deadline=d)
    assert seen == [d], "the loop was signalled before its bound was set"


def test_g16_a_loop_already_ending_from_a_rest_is_bounded_by_a_later_stop() -> None:
    _has_surface()
    h = _handle()
    holder = deadline_mod.EndBound()
    release = threading.Event()
    straggler = threading.Thread(target=release.wait, args=(5.0,), daemon=True)
    straggler.start()
    try:
        h._talk_bridge, h._talk_thread, h._talk_stop, h._talk_bound = MagicMock(), straggler, threading.Event(), holder
        h._stop_talk_loop(join_s=0.05)  # rest()'s path: no bound, the loop outlives the join
        d = Deadline.after(2.0)
        h.stop(deadline=d)
        assert holder.get() == d, "a loop still ending from a rest was left unbounded by stop()"
    finally:
        release.set()
        straggler.join(1.0)


def test_g11_a_republished_straggler_keeps_its_bound() -> None:
    _has_surface()
    h = _handle()
    holder = deadline_mod.EndBound()
    release = threading.Event()
    straggler = threading.Thread(target=release.wait, args=(5.0,), daemon=True)
    straggler.start()
    try:
        h._talk_bridge, h._talk_thread, h._talk_stop, h._talk_bound = MagicMock(), straggler, threading.Event(), holder
        with pytest.raises(RuntimeError):
            h._stop_talk_loop(join_s=0.05, required=True)  # a campaign start that times out re-publishes it
        assert h._talk_thread is straggler and h._talk_bound is holder, "the re-published straggler lost its bound"
    finally:
        release.set()
        straggler.join(1.0)


def test_g5_a_stop_with_no_talk_loop_touches_no_bound() -> None:
    """Guard (green on main): a campaign's session is never bounded by a per-handle stop (v2's DNB-1)."""
    from maxim.console.handle import MaximHandle

    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "t"
    h._stopped = False
    h._stop_result = None
    h._campaign_lock = threading.Lock()
    h._talk_lock = threading.Lock()
    h._talk_bridge = h._talk_thread = h._talk_stop = h._talk_worker = None
    if hasattr(MaximHandle, "_stop_talk_loop") and "bound" in inspect.signature(MaximHandle._stop_talk_loop).parameters:
        h._talk_bound = None
        h._ending_talk_loops = []
    h.instance = MagicMock()
    h.instance.shutdown.return_value = True
    assert h.stop(deadline=Deadline.after(1.0)) is True
    kwargs = h.instance.shutdown.call_args.kwargs
    assert kwargs["deadline"] is not None  # shutdown's own end keeps its contract, untouched


# ── The composition: the handle's bound reaches the loop, and the loop's reaches its session ─────────────────────


def _calls_in(func: Any, callee: str) -> list[Any]:
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == callee)
            or (isinstance(n.func, ast.Attribute) and n.func.attr == callee)
        )
    ]


def test_the_wrapper_hands_its_end_bound_to_the_runs_bio_session() -> None:
    from maxim.runtime import agent_loop

    [call] = _calls_in(agent_loop.run_agentic_loop, "bio_session")
    kw = {k.arg: getattr(k.value, "id", None) for k in call.keywords}
    assert kw.get("end_bound") == "end_bound", "run_agentic_loop drops the handle's bound before bio_session"


def test_the_talk_loop_is_built_with_its_own_end_bound_and_publishes_it() -> None:
    from maxim.console.handle import MaximHandle

    [call] = _calls_in(MaximHandle._launch_talk_loop, "run_agentic_loop")
    kw = {k.arg: getattr(k.value, "id", None) for k in call.keywords}
    assert kw.get("end_bound") == "end_bound", "the talk loop is built without its EndBound"
    [stop_call] = _calls_in(MaximHandle.stop, "_stop_talk_loop")
    assert any(k.arg == "bound" for k in stop_call.keywords), "stop() does not hand its deadline to the talk loop"


def test_the_shipped_path_handle_stop_bounds_a_live_talk_loops_consolidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review SF-3: the real launch and the real ``stop(deadline=)`` on a real agent; only the loop body is a stub that
    holds the session open in a real ``bio_session`` until the handle stops it."""
    from maxim.runtime import agent_loop
    from maxim.runtime.bio_integration import bio_session

    inst = _agent(tmp_path)
    inst.hippocampus.store_observation("the blue cup is on the shelf")
    inst.hippocampus.flush(timeout=5.0)
    slept: list[bool] = []
    monkeypatch.setattr(
        inst.memory_hub.hippocampus, "sleep", lambda *a, **k: (slept.append(True), time.sleep(3.0), {})[2]
    )
    opened = threading.Event()

    def loop(
        *_a: Any,
        memory_hub: Any,
        hippocampus: Any,
        stop_event: threading.Event,
        consolidation: Any,
        end_bound: Any,
        **_k: Any,
    ) -> None:
        with bio_session(
            memory_hub=memory_hub,
            hippocampus=hippocampus,
            owner=object(),
            consolidation=consolidation,
            end_bound=end_bound,
        ):
            opened.set()
            stop_event.wait(10.0)

    monkeypatch.setattr(agent_loop, "run_agentic_loop", loop)
    h = _handle()
    h.instance = inst
    with h._talk_lock:
        h._launch_talk_loop(MagicMock(), tmp_path / "ws")
    assert opened.wait(5.0), "the talk loop never opened its session"
    started = time.monotonic()
    assert h.stop(deadline=Deadline.after(3.0)) is True
    assert slept == [], "sleep ran: the handle's stop did not bound the live loop's consolidation"
    assert time.monotonic() - started < 3.5
    path = Path(inst.hippocampus.config.persistence_path).expanduser()
    assert "blue cup" in path.read_text(encoding="utf-8"), "the bounded end did not persist"
