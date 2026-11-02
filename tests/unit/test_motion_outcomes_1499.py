"""The motor queue reports what each move did (#1499 PR 2a: the queue, the two-stage ticket, the tools, the turn).

THE GAP PR 1 LEFT (by design, docs/plans/motion_path_model.md §13): every controller dispatch is now checked and
returns a ``MotionOutcome``, but a move that goes through Selfy's motor queue is fire-and-forget. ``_enqueue_motor``
returns ``None``, a replaced or cleared command vanishes without a word, and the model hears "queued" as success, so a
refused move (too fast, an unreadable pose, a sweeping path) is reported to it as done. The queue also sleeps the
requested time in ``act`` while the controller may have stretched the move, and an explicit duration from the model is
treated as a default, so it is stretched rather than refused.

PR 2a: a ``MotorQueue`` that resolves every ticket it removes, a two-stage ``MotionTicket`` (``accepted`` when the
checks pass and the first SDK call starts, ``completed`` with the outcome), the tools waiting on ``accepted`` and
reporting the reason, the explicit-duration policies, ``turn_around`` returning an outcome from a fresh body read,
``move_antenna`` through the controller, and the simulated controller's outcomes. The DN inbox and the pain bridge's
event ids are PR 2b. Gates: §9 items 10, 11, 16 (queue half), 21, 26, and 5.4, 5.5, 5.8.

Written as strict xfails before the build (the red-gate rule); each drives the real controller over
tests/unit/_reachy_fakes.py where a controller is involved.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time

import pytest

from tests.unit._reachy_fakes import FakeDaemon, controller, head_pose, install

D = math.radians


@pytest.fixture
def daemon(monkeypatch):
    d = FakeDaemon()
    install(monkeypatch, d)
    return d


def _place(d: FakeDaemon, *, body_deg: float, rel_deg: float = 0.0) -> None:
    d.body = D(body_deg)
    d.head = head_pose(yaw=D(body_deg + rel_deg))


def _outcome_kind(ticket) -> str | None:
    return ticket.completed.result(timeout=5.0).kind


def _selfy(ctl, daemon):
    """A Selfy-shaped object with the real movement and media-loop mixins over a real controller."""
    from maxim.embodied_runtime.media_loop import MediaLoopMixin
    from maxim.embodied_runtime.movement import MovementMixin, SyncResult

    class _Selfy(MovementMixin, MediaLoopMixin):
        def __init__(self) -> None:
            self.log = logging.getLogger("test.motion_outcomes_1499")
            self._default_network = None
            self._workspace_limit_override = None
            self._head_max_step = None
            self.mini = daemon
            self._robot = ctl
            self.duration = 0.5
            self.x = self.y = self.z = self.roll = self.pitch = self.yaw = 0.0
            self.body_yaw = 0.0
            self._last_turn_around_time = 0.0
            self.actions: dict = {}
            self._motor_queue = None
            self.audio = False
            self.mode = "active"
            self.sleeping = False
            self.verbosity = 0

        def sync_head_position(self):
            return SyncResult.OK

        def _note_connection_failure(self, subsystem, error):
            pass

    return _Selfy()


def _start_worker(selfy):
    """Give *selfy* a MotorQueue and a running motor worker; returns the stop event."""
    from maxim.embodied_runtime.workers import MotorQueue, motor_worker

    selfy._motor_queue = MotorQueue(maxsize=4)
    stop = threading.Event()
    threading.Thread(target=motor_worker, args=(selfy._motor_queue, stop, selfy), daemon=True).start()
    return stop


def _blocker(release: threading.Event, started: threading.Event):
    def _block():
        started.set()
        release.wait(5.0)
        from maxim.hardware.controller import MotionOutcome

        return MotionOutcome(ok=True, chunks_planned=1, chunks_done=1)

    return _block


# ─────────────────────────────────────────────────────────────────────────────
# Gate 10: the queue resolves every ticket it removes
# ─────────────────────────────────────────────────────────────────────────────


def test_a_replaced_command_resolves_superseded():
    from maxim.embodied_runtime.workers import MotorQueue

    q = MotorQueue(maxsize=4)
    first = q.put(lambda: None)
    second = q.put(lambda: None)
    assert _outcome_kind(first) == "superseded"
    assert not second.completed.done()


def test_a_full_queue_resolves_the_new_command_superseded(monkeypatch):
    from maxim.embodied_runtime.workers import MotorQueue

    q = MotorQueue(maxsize=4)

    def _full(_item):
        raise queue.Full

    monkeypatch.setattr(q._q, "put_nowait", _full)
    ticket = q.put(lambda: None)
    assert _outcome_kind(ticket) == "superseded"


def test_clear_and_the_reconnect_drain_resolve_superseded():
    from maxim.embodied_runtime.workers import MotorQueue

    q = MotorQueue(maxsize=4)
    cleared = q.put(lambda: None)
    assert q.clear() == 1
    assert _outcome_kind(cleared) == "superseded"
    drained = q.put(lambda: None)
    assert q.drain() == 1
    out = drained.completed.result(timeout=1.0)
    assert out.kind == "superseded" and "reconnect" in out.reason


def test_close_cancels_what_is_queued_and_everything_after():
    from maxim.embodied_runtime.workers import MotorQueue

    q = MotorQueue(maxsize=4)
    queued = q.put(lambda: None)
    q.close()
    assert _outcome_kind(queued) == "cancelled"
    late = q.put(lambda: None)
    assert _outcome_kind(late) == "cancelled"
    with pytest.raises(queue.Empty):
        q.get(timeout=0.05)


def test_selfy_enqueue_returns_a_ticket_and_clear_resolves_it(daemon):
    from maxim.embodied_runtime.workers import MotorQueue
    from maxim.hardware.controller import MotionTicket

    selfy = _selfy(controller(daemon), daemon)
    selfy._motor_queue = MotorQueue(maxsize=4)
    ticket = selfy._enqueue_motor(lambda: None)
    assert isinstance(ticket, MotionTicket)
    assert selfy._clear_motor_queue() == 1
    assert _outcome_kind(ticket) == "superseded"


def test_the_reconnect_and_media_loop_no_longer_touch_the_raw_queue():
    """Raw queue access leaves connection.py and media_loop.py (5.6): only MotorQueue removes a command."""
    import inspect

    from maxim.embodied_runtime import connection, media_loop

    for module in (connection, media_loop):
        src = inspect.getsource(module)
        assert "motor_queue.get_nowait" not in src and "q.get_nowait" not in src, module.__name__
        assert "queue.Queue(maxsize=4)" not in src, module.__name__


# ─────────────────────────────────────────────────────────────────────────────
# Gate 16, the queue half: nothing queued runs behind a stop
# ─────────────────────────────────────────────────────────────────────────────


def test_a_command_queued_behind_the_sleep_is_never_dispatched(daemon, monkeypatch):
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        release, started = threading.Event(), threading.Event()
        selfy._enqueue_motor(_blocker(release, started))
        assert started.wait(2.0)
        ran = []
        behind = selfy._enqueue_motor(lambda: ran.append(1))
        monkeypatch.setattr(selfy, "live", lambda **kw: None)
        threading.Timer(0.2, release.set).start()
        selfy.sleep(home_dir="/tmp", parallel=True, run_id="t")
        assert _outcome_kind(behind) == "cancelled"
        time.sleep(0.3)
        assert ran == []
    finally:
        stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Gate 21: the ticket seam (the controller accepts; a plugin's result is resolved by the worker)
# ─────────────────────────────────────────────────────────────────────────────


def test_with_a_ticket_bound_the_controller_accepts_before_the_sdk_call(daemon):
    from maxim.hardware.controller import MotionTarget, MotionTicket, current_motion_ticket

    ctl = controller(daemon)
    ticket = MotionTicket()
    seen = []
    real_goto = daemon.goto_target

    def _goto(**kw):
        seen.append(ticket.accepted.done())
        return real_goto(**kw)

    daemon.goto_target = _goto  # type: ignore[method-assign]
    token = current_motion_ticket.set(ticket)
    try:
        assert ctl.goto_target(MotionTarget(head_yaw=D(20), duration=1.0))
    finally:
        current_motion_ticket.reset(token)
    assert seen and all(seen)
    assert ticket.accepted.result(timeout=0) is None  # started, not refused


def test_a_refusal_resolves_accepted_with_the_reason(daemon):
    selfy = _selfy(controller(daemon), daemon)
    daemon.rest_mode = "timeout"  # every fresh readback fails: reference
    stop = _start_worker(selfy)
    try:
        ticket = selfy.move(yaw=20.0)
        refusal = ticket.accepted.result(timeout=5.0)
        assert refusal is not None and refusal.kind == "reference"
        assert ticket.completed.result(timeout=5.0).kind == "reference"
    finally:
        stop.set()


def test_a_plugin_result_resolves_both_stages_through_the_worker(daemon):
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        ticket = selfy._enqueue_motor(lambda: True)  # a bool-only plugin's controller call
        assert ticket.accepted.result(timeout=5.0) is None
        assert ticket.completed.result(timeout=5.0).ok
    finally:
        stop.set()


def test_a_ticket_cancelled_before_it_runs_never_moves(daemon):
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        release, started = threading.Event(), threading.Event()
        selfy._enqueue_motor(_blocker(release, started))
        assert started.wait(2.0)
        ran = []
        behind = selfy._enqueue_motor(lambda: ran.append(1))
        assert behind.cancel()
        release.set()
        time.sleep(0.3)
        assert ran == []
        assert _outcome_kind(behind) == "cancelled"
    finally:
        stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# Gate 11: the tools learn the real outcome
# ─────────────────────────────────────────────────────────────────────────────


def test_a_refused_queued_move_reaches_maxim_command_with_its_reason(daemon):
    from maxim.tools.reachy import MaximCommandTool

    _place(daemon, body_deg=0)
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        # 40° in an explicit 0.1 s is far over the 180°/s cap: "exact" refuses with rate.
        result = MaximCommandTool(selfy).execute(command="move", params={"yaw": 40.0, "duration": 0.1})
        assert not result.success
        assert "rate" in (result.error or "") or "too fast" in (result.error or "")
        assert daemon.gotos() == []
    finally:
        stop.set()


def test_a_busy_motor_past_the_wait_is_not_started_and_never_moves(daemon, monkeypatch):
    import maxim.hardware.controller as hc
    from maxim.tools.reachy import MaximCommandTool

    monkeypatch.setattr(hc, "MOTION_ACCEPT_WAIT_S", 0.3)
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        release, started = threading.Event(), threading.Event()
        selfy._enqueue_motor(_blocker(release, started))
        assert started.wait(2.0)
        result = MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0})
        assert not result.success and "not started" in (result.error or "")
        release.set()
        time.sleep(0.4)
        assert daemon.gotos() == []
    finally:
        stop.set()


def test_maxim_command_reports_success_only_once_the_move_started(daemon):
    from maxim.tools.reachy import MaximCommandTool

    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        result = MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0})
        assert result.success
        # ``accepted`` resolves just before the worker's SDK call (§4), so the goto follows within a moment.
        deadline = time.monotonic() + 2.0
        while not daemon.gotos() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert daemon.gotos(), "success was reported for a move that was never dispatched"
    finally:
        stop.set()


def test_move_tool_returns_the_controllers_reason(daemon):
    from maxim.tools.reachy import MoveTool

    ctl = controller(daemon)
    result = MoveTool(ctl).execute(yaw=40.0, duration=0.1)
    assert not result.success
    assert "Motion command failed" not in (result.error or "")
    assert "rate" in (result.error or "") or "too fast" in (result.error or "")


# ─────────────────────────────────────────────────────────────────────────────
# 5.5: an explicit duration is "exact"; an absent one "at_least"
# ─────────────────────────────────────────────────────────────────────────────


class _RecordingController:
    """A controller double that records every target and accepts it."""

    robot_id = "rec"
    last_clamped_axes = ()

    def __init__(self) -> None:
        self.targets: list = []

    def goto_target(self, target):
        self.targets.append(target)
        from maxim.hardware.controller import MotionOutcome

        return MotionOutcome(ok=True, chunks_planned=1, chunks_done=1)

    def look_at_pixel(self, target):
        return self.goto_target(target)

    def get_current_pose(self):
        return {"yaw": 0.0, "body_yaw": 0.0}


def _bind_controller(monkeypatch, ctl):
    import maxim.tools.reachy as tools_reachy

    monkeypatch.setattr(tools_reachy, "_is_robot_controller", lambda m: m is ctl)
    monkeypatch.setattr(tools_reachy, "_get_robot_from_registry", lambda robot_id, maxim=None: ctl)


def test_move_tool_marks_a_model_duration_exact_and_an_absent_one_at_least(monkeypatch):
    from maxim.tools.reachy import MoveTool

    ctl = _RecordingController()
    _bind_controller(monkeypatch, ctl)
    tool = MoveTool(ctl)
    assert tool.execute(yaw=10.0, duration=1.0).success
    assert tool.execute(yaw=12.0).success
    assert [t.duration_policy for t in ctl.targets] == ["exact", "at_least"]


def test_maxim_command_marks_a_model_duration_exact(daemon, monkeypatch):
    from maxim.tools.reachy import MaximCommandTool

    selfy = _selfy(controller(daemon), daemon)
    seen = []
    real_move = selfy.move

    def _move(*a, **kw):
        seen.append(kw.get("duration_policy"))
        return real_move(*a, **kw)

    monkeypatch.setattr(selfy, "move", _move)
    MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0, "duration": 1.0})
    MaximCommandTool(selfy).execute(command="move", params={"yaw": 12.0})
    assert seen == ["exact", "at_least"]


def test_a_pose_table_duration_is_at_least(daemon, monkeypatch):
    """goto_pose's authored duration is not the model's choice (5.5): it stretches, never refuses."""
    selfy = _selfy(controller(daemon), daemon)
    selfy.poses = {"far": [0, 0, 0, 0, 0, 50, 0.1]}  # 50° yaw authored at 0.1 s
    ticket = selfy.goto_pose("far")
    assert ticket.completed.result(timeout=5.0).ok
    assert daemon.gotos() and float(daemon.gotos()[-1]["duration"]) >= 50 / 180 - 1e-6


def test_the_track_tools_advertise_no_duration_default():
    """A schema default is copied into calls by models, which would make every track 'exact' (5.5)."""
    from maxim.tools import reachy, reachy_stubs

    checked = []
    for module in (reachy, reachy_stubs):
        for name, obj in vars(module).items():
            schema = getattr(obj, "input_schema", None)
            if isinstance(schema, dict) and "duration_s" in schema:
                checked.append(name)
                assert schema["duration_s"][1] is None, (module.__name__, name)
    assert len(checked) >= 3  # TrackTargetTool, NoveltyTrackTool and the stubs


# ─────────────────────────────────────────────────────────────────────────────
# Selfy's queued-motion methods return the ticket; act waits for completion
# ─────────────────────────────────────────────────────────────────────────────


def test_selfy_move_returns_a_ticket_and_a_no_op_is_already_ok(daemon):
    from maxim.hardware.controller import MotionTicket

    selfy = _selfy(controller(daemon), daemon)
    moved = selfy.move(yaw=10.0)
    assert isinstance(moved, MotionTicket) and moved.completed.result(timeout=5.0).ok
    same = selfy.move(yaw=10.0)
    out = same.completed.result(timeout=0)
    assert out.ok and out.chunks_planned == 0


def test_an_unreachable_look_at_returns_a_limit_ticket(daemon, monkeypatch):
    selfy = _selfy(controller(daemon), daemon)
    monkeypatch.setattr(selfy, "_is_pixel_reachable", lambda position, min_reachability: False)
    ticket = selfy.look_at_image(320, 240)
    assert _outcome_kind(ticket) == "limit"


def test_act_waits_for_each_move_to_complete_not_the_requested_time(daemon, monkeypatch):
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        selfy.actions = {"nod": {"movements": [(0, 0, 0, 0, 0, 40, 0.1)]}}  # 40° asked in 0.1 s: stretched
        slept = []
        import maxim.embodied_runtime.media_loop as ml

        monkeypatch.setattr(ml.time, "sleep", lambda s: slept.append(s))
        ticket = selfy.act("nod")
        assert ticket.completed.done(), "act returned before its move completed"
        assert 0.1 not in slept
    finally:
        stop.set()


# ─────────────────────────────────────────────────────────────────────────────
# 5.8 and gate 26: turn_around reads the body fresh, refuses what cannot fit, and reports an outcome
# ─────────────────────────────────────────────────────────────────────────────


def test_turn_around_returns_an_outcome(daemon):
    from maxim.hardware.controller import MotionOutcome

    selfy = _selfy(controller(daemon), daemon)
    out = selfy.turn_around(30.0, duration=3.0)
    assert isinstance(out, MotionOutcome) and out.ok


def test_turn_around_reads_the_body_from_the_fresh_readback_not_the_mirror(daemon):
    """The WS read fails and the mirror says 0°; the daemon says 100°: a +90° turn cannot fit (190° > 160°)."""
    _place(daemon, body_deg=100)
    ctl = controller(daemon)
    ctl.get_current_pose = lambda: (_ for _ in ()).throw(RuntimeError("ws read failed"))  # type: ignore[method-assign]
    selfy = _selfy(ctl, daemon)
    before = len(daemon.gotos())
    out = selfy.turn_around(90.0, duration=5.0)
    assert not out and out.kind == "limit"
    assert len(daemon.gotos()) == before


def test_a_turn_with_no_body_read_is_refused_not_guessed(daemon):
    ctl = controller(daemon)
    daemon.rest_mode = "timeout"
    selfy = _selfy(ctl, daemon)
    out = selfy.turn_around(30.0, duration=3.0)
    assert not out and out.kind == "reference"
    assert daemon.gotos() == []


def test_a_refused_turn_reaches_the_model_as_an_error(daemon):
    from maxim.tools.reachy import MaximCommandTool

    _place(daemon, body_deg=150)
    selfy = _selfy(controller(daemon), daemon)
    result = MaximCommandTool(selfy).execute(command="turn_around", params={"angle": 90.0})
    assert not result.success
    assert "limit" in (result.error or "") or "cannot" in (result.error or "")


def test_look_at_books_a_refused_automatic_turn_as_a_failure(daemon):
    """_maybe_turn_around reads the outcome's truth (D3): a refused turn is not a success to the pain bridge."""
    from maxim.hardware.controller import MotionOutcome

    selfy = _selfy(controller(daemon), daemon)
    completes = []

    class _Pain:
        def get_pain_risk(self, sig):
            return 1.0

        def record_action_start(self, *a, **kw):
            return "e1"

        def record_action_complete(self, success=True, **kw):
            completes.append(success)

    selfy.turn_around = lambda *a, **kw: MotionOutcome.refused("limit", "cannot fit")  # type: ignore[method-assign]
    selfy._maybe_turn_around(du=-400.0, looking_left=True, looking_right=False, h_usage=0.99, pain_bridge=_Pain())
    assert completes == [False]


# ─────────────────────────────────────────────────────────────────────────────
# 5.4: move_antenna goes through the controller (lock, rate cap, outcome)
# ─────────────────────────────────────────────────────────────────────────────


def test_move_antenna_is_dispatched_by_the_controller_under_the_rate_cap(daemon):
    selfy = _selfy(controller(daemon), daemon)
    ticket = selfy.move_antenna(right=80.0, left=80.0)  # degrees; no duration: a default, so stretched
    out = ticket.completed.result(timeout=5.0)
    assert out.ok
    antenna_gotos = [g for g in daemon.gotos() if g.get("antennas") is not None]
    assert antenna_gotos and float(antenna_gotos[-1]["duration"]) >= 80 / 180 - 1e-6


def test_an_explicit_too_fast_antenna_move_is_refused(daemon):
    selfy = _selfy(controller(daemon), daemon)
    ticket = selfy.move_antenna(right=80.0, left=80.0, duration=0.1, duration_policy="exact")
    assert _outcome_kind(ticket) == "rate"
    assert daemon.gotos() == []


# ─────────────────────────────────────────────────────────────────────────────
# The simulated controller returns outcomes (O7 consumers read fields)
# ─────────────────────────────────────────────────────────────────────────────


def test_the_simulated_controller_returns_motion_outcomes():
    import inspect

    from maxim.hardware.simulation.controller import SimulatedController

    for name in ("goto_target", "look_at_pixel", "wake_up", "goto_sleep"):
        annotation = inspect.signature(getattr(SimulatedController, name)).return_annotation
        assert "MotionOutcome" in str(annotation), name


# ─────────────────────────────────────────────────────────────────────────────
# Guards added after the deletion probes (each mechanism below survived deletion under the gates above)
# ─────────────────────────────────────────────────────────────────────────────


def test_the_worker_binds_the_ticket_so_accepted_resolves_while_the_move_runs(daemon):
    """Without the worker's binding, ``accepted`` would only resolve at ``completed``: a tool would wait out the move."""
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    in_goto, release = threading.Event(), threading.Event()
    real_goto = daemon.goto_target

    def _slow_goto(**kw):
        in_goto.set()
        release.wait(5.0)
        return real_goto(**kw)

    daemon.goto_target = _slow_goto  # type: ignore[method-assign]
    try:
        ticket = selfy.move(yaw=10.0)
        assert in_goto.wait(5.0)
        assert ticket.accepted.done() and ticket.accepted.result(timeout=0) is None
        assert not ticket.completed.done()
    finally:
        release.set()
        stop.set()


def test_a_turn_whose_body_read_fails_dispatches_nothing():
    """5.8: the turn refuses at planning, even on a controller whose dispatch would succeed without a read."""
    from maxim.exceptions import MotionRefused

    ctl = _RecordingController()

    def _no_read():
        raise MotionRefused("reference", "no body read")

    ctl.read_body_yaw = _no_read  # type: ignore[attr-defined]
    selfy = _selfy(ctl, FakeDaemon())
    out = selfy.turn_around(30.0, duration=3.0)
    assert not out and out.kind == "reference"
    assert ctl.targets == []


# ─────────────────────────────────────────────────────────────────────────────
# Review round folds (2026-11-01): each guards a fold, and each is deletion-probed
# ─────────────────────────────────────────────────────────────────────────────


def _hold_until_in_hand(selfy, ctl):
    """Hold the controller's lock, queue a move, and wait until the worker holds it (claimed, not accepted)."""
    ctl._motion_lock.acquire()
    ticket = selfy.move(yaw=10.0)
    deadline = time.monotonic() + 2.0
    while selfy._motor_queue._in_hand is not ticket and time.monotonic() < deadline:
        time.sleep(0.01)
    assert selfy._motor_queue._in_hand is ticket and not ticket.accepted.done()
    return ticket


def test_a_cancel_while_the_worker_waits_for_the_lock_holds_and_nothing_moves(daemon):
    """Architecture S1: ``accepted`` is the point of no return, not the worker's claim."""
    ctl = controller(daemon)
    selfy = _selfy(ctl, daemon)
    stop = _start_worker(selfy)
    try:
        ticket = _hold_until_in_hand(selfy, ctl)
        assert ticket.cancel("not started: the motor is busy")
        ctl._motion_lock.release()
        time.sleep(0.3)
        assert ticket.completed.result(timeout=0).kind == "cancelled"
        assert daemon.gotos() == []
    finally:
        if ctl._motion_lock._is_owned():
            ctl._motion_lock.release()
        stop.set()


@pytest.mark.parametrize("how", ["clear", "close"])
def test_clear_and_close_withdraw_the_command_the_worker_holds(daemon, how):
    """Executor 3: a command taken off the queue but not yet accepted is withdrawn by a clear or a stop."""
    ctl = controller(daemon)
    selfy = _selfy(ctl, daemon)
    stop = _start_worker(selfy)
    try:
        q = selfy._motor_queue
        ticket = _hold_until_in_hand(selfy, ctl)
        assert getattr(q, how)() == 1
        ctl._motion_lock.release()
        time.sleep(0.3)
        assert ticket.completed.result(timeout=0).kind == ("superseded" if how == "clear" else "cancelled")
        assert daemon.gotos() == []
    finally:
        if ctl._motion_lock._is_owned():
            ctl._motion_lock.release()
        stop.set()


def test_move_tool_on_selfy_reports_the_queued_outcome_and_the_exact_policy(daemon):
    """Executor 1 / Wire 2: MoveTool's Selfy fallback waits on the ticket and marks a model duration exact."""
    from maxim.tools.reachy import MoveTool

    _place(daemon, body_deg=0)
    selfy = _selfy(controller(daemon), daemon)
    stop = _start_worker(selfy)
    try:
        result = MoveTool(selfy).execute(yaw=40.0, duration=0.1)
        assert not result.success and "rate" in (result.error or "")
        assert daemon.gotos() == []
    finally:
        stop.set()


def test_a_move_that_never_started_does_not_leave_the_mirror_at_its_target(daemon, monkeypatch):
    """Executor 2 (owner: stale flag + re-read): an identical retry after a 'not started' really moves."""
    import maxim.hardware.controller as hc
    from maxim.tools.reachy import MaximCommandTool

    monkeypatch.setattr(hc, "MOTION_ACCEPT_WAIT_S", 0.3)
    selfy = _selfy(controller(daemon), daemon)
    synced = []
    selfy.sync_head_position = lambda: (synced.append(1), setattr(selfy, "yaw", 0.0))[-1]  # the real pose: yaw 0
    stop = _start_worker(selfy)
    try:
        release, started = threading.Event(), threading.Event()
        selfy._enqueue_motor(_blocker(release, started))
        assert started.wait(2.0)
        first = MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0})
        assert not first.success and "not started" in (first.error or "")
        release.set()
        time.sleep(0.2)
        retry = MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0})
        assert retry.success and synced
        deadline = time.monotonic() + 2.0
        while not daemon.gotos() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert daemon.gotos(), "the retry was a no-op against a mirror at a pose the robot never reached"
    finally:
        release.set()
        stop.set()


def test_an_automatic_turn_that_cannot_fit_books_nothing_and_looks_clamped(daemon):
    """Owner 2026-11-01 (§5.8 over §5.6): a turn refused before it moves records no pain start and no completion."""
    _place(daemon, body_deg=150)  # +45° would end at 195°, past ±160°
    selfy = _selfy(controller(daemon), daemon)
    starts, completes = [], []

    class _Pain:
        def get_pain_risk(self, sig):
            return 1.0

        def record_action_start(self, *a, **kw):
            starts.append(a)

        def record_action_complete(self, success=True, **kw):
            completes.append(success)

    before = len(daemon.gotos())
    out = selfy._maybe_turn_around(du=-400.0, looking_left=True, looking_right=False, h_usage=0.99, pain_bridge=_Pain())
    assert out is None  # look_at proceeds with the head clamped
    assert starts == [] and completes == []
    assert len(daemon.gotos()) == before


def test_dn_books_a_refused_turn_as_not_executed():
    """Architecture S4 (owner: fold into 2a): DN's own turn reads the outcome, so a 'limit' refusal feeds no learner."""
    from maxim.default_network.messages import ActionProposal
    from maxim.default_network.network import DefaultNetwork
    from maxim.hardware.controller import MotionOutcome

    class _Maxim:
        yaw = 0.0

        def __init__(self, outcome):
            self.outcome = outcome

        def turn_around(self, angle, *, duration):
            return self.outcome

    dn = DefaultNetwork.__new__(DefaultNetwork)
    turn = ActionProposal(behavior_name="b", action_type="turn_around", target=None, priority=1.0, confidence=1.0)
    dn._maxim = _Maxim(MotionOutcome.refused("limit", "cannot fit"))
    assert dn._dispatch_action_to_motor(turn, 5.0) == (False, False)
    dn._maxim = _Maxim(MotionOutcome(ok=True))
    assert dn._dispatch_action_to_motor(turn, 5.0) == (True, False)


def test_headless_selfy_queues_nothing():
    """Wire 3: with no robot the motor worker never starts, so the motion path stays synchronous."""
    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    class _Headless(MediaLoopMixin):
        _robot = None

    host = _Headless()
    host._open_motor_queue(active=True)
    assert host._motor_queue is None


def test_a_split_antenna_move_holds_the_antennas_until_the_final_chunk(daemon):
    """Executor 6: intermediate chunks carry the start antennas, so the final chunk's antenna delta is rate-capped."""
    import numpy as np

    from maxim.hardware.reachy import motion_path as mp

    start = mp.Pose(mp.compose_head(0.0, 0.0, 0.0, 0.0), 0.0, (0.0, 0.0))
    target = mp.Pose(mp.compose_head(0.0, 0.0, 0.0, D(155)), D(155), (D(60), D(60)))
    chunks = mp.plan_chunks(start, target)
    assert len(chunks) > 1
    assert all(c.antennas == (0.0, 0.0) for c in chunks[:-1])
    assert np.allclose(chunks[-1].antennas, (D(60), D(60)))
    assert mp.move_delta_deg(chunks[-2], chunks[-1]) >= 60 - 1e-6


def test_move_antenna_with_no_side_is_refused_before_the_queue(daemon):
    """Architecture NIT: a bad call is the caller's error, never a counted motor failure."""
    selfy = _selfy(controller(daemon), daemon)
    failures = []
    selfy._note_connection_failure = lambda *a: failures.append(a)
    with pytest.raises(ValueError):
        selfy.move_antenna()
    assert failures == []


def test_a_command_queued_after_a_failed_move_does_not_hide_the_stale_mirror(daemon, monkeypatch):
    """Delta round (Architecture SF1): the re-read keys on the move that wrote the mirror, not the last enqueue."""
    import maxim.hardware.controller as hc
    from maxim.tools.reachy import MaximCommandTool

    monkeypatch.setattr(hc, "MOTION_ACCEPT_WAIT_S", 0.3)
    selfy = _selfy(controller(daemon), daemon)
    synced = []
    selfy.sync_head_position = lambda: (synced.append(1), setattr(selfy, "yaw", 0.0))[-1]
    stop = _start_worker(selfy)
    try:
        release, started = threading.Event(), threading.Event()
        selfy._enqueue_motor(_blocker(release, started))
        assert started.wait(2.0)
        assert not MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0}).success
        release.set()
        other = selfy._enqueue_motor(lambda: True)  # an antenna gesture or a look-at that ran ok
        assert other.completed.result(timeout=2.0).ok
        before = len(daemon.gotos())
        retry = MaximCommandTool(selfy).execute(command="move", params={"yaw": 10.0})
        assert retry.success and synced
        deadline = time.monotonic() + 2.0
        while len(daemon.gotos()) == before and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(daemon.gotos()) > before
    finally:
        release.set()
        stop.set()


def test_a_cancelled_no_op_is_refused_not_reported_done(daemon):
    """Delta round (Architecture NIT 4): a command that asks for nothing is accepted like any dispatch."""
    from maxim.hardware.controller import MotionTarget, MotionTicket, current_motion_ticket

    ctl = controller(daemon)
    ticket = MotionTicket()
    assert ticket.claim() and ticket.cancel("not started: the motor is busy")
    token = current_motion_ticket.set(ticket)
    try:
        out = ctl.goto_target(MotionTarget(duration=1.0))  # no head axis, no body: nothing to move
    finally:
        current_motion_ticket.reset(token)
    assert not out and out.kind == "cancelled"
