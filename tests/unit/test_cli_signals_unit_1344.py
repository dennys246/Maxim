"""``maxim.cli_signals`` in-process (#1344): what the real-signal gates in ``test_bounded_stop_cli_1344.py`` prove in a
child, measured here. Each test calls the piece directly, with the process-ending step stubbed."""

from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

import pytest

from maxim import cli_signals
from maxim.integration import session_ownership
from maxim.utils import cancellation
from maxim.utils.deadline import Deadline


@pytest.fixture
def died(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    monkeypatch.setattr(cli_signals, "_die", lambda code, reason: calls.append((code, reason)))
    return calls


@pytest.fixture
def written(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    bytes_out: list[int] = []
    monkeypatch.setattr(cli_signals, "_write", bytes_out.append)
    return bytes_out


def test_a_pre_session_signal_records_writes_and_raises(written: list[int]) -> None:
    with pytest.raises(KeyboardInterrupt):
        cli_signals._handler(signal.SIGTERM, None)
    assert written == [signal.SIGTERM]
    assert cli_signals._signalled == signal.SIGTERM and not cli_signals._cooperative
    assert cancellation.stop_noted()


def test_a_signal_with_a_session_open_returns_without_raising(
    monkeypatch: pytest.MonkeyPatch, written: list[int]
) -> None:
    monkeypatch.setattr(session_ownership, "_ever_claimed", True)
    monkeypatch.setattr(cli_signals, "session_ever_claimed", lambda: True)
    cli_signals._handler(signal.SIGTERM, None)
    assert cli_signals._cooperative


def test_a_signal_in_an_interruptible_region_raises_though_a_session_is_open(
    monkeypatch: pytest.MonkeyPatch, written: list[int]
) -> None:
    monkeypatch.setattr(cli_signals, "session_ever_claimed", lambda: True)
    monkeypatch.setattr(cli_signals, "interruptible_depth", lambda: 1)
    with pytest.raises(KeyboardInterrupt):
        cli_signals._handler(signal.SIGINT, None)


def test_a_repeat_is_marked_as_one_and_never_raises(monkeypatch: pytest.MonkeyPatch, written: list[int]) -> None:
    monkeypatch.setattr(cli_signals, "_signalled", signal.SIGTERM)
    cli_signals._handler(signal.SIGINT, None)
    assert written == [signal.SIGINT | cli_signals._REPEAT]


def test_write_survives_a_failed_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _full(*_a: Any) -> int:
        raise BlockingIOError

    monkeypatch.setattr(cli_signals, "_pipe_w", 99)
    monkeypatch.setattr(cli_signals.os, "write", _full)
    cli_signals._write(1)


def test_write_with_no_pipe_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_signals, "_pipe_w", -1)
    cli_signals._write(1)


def _signalled_install(monkeypatch: pytest.MonkeyPatch, signum: int = signal.SIGTERM) -> None:
    monkeypatch.setattr(cli_signals, "_depth", 1)
    monkeypatch.setattr(cli_signals, "_signalled", signum)
    monkeypatch.setattr(cli_signals, "_signal_time", time.monotonic())


def test_the_first_byte_stores_the_kill_time_then_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    _signalled_install(monkeypatch)
    order: list[str] = []

    def _start(signum: int, library_at: float) -> None:
        assert cli_signals._kill_at is not None, "the request started before the kill time was stored"
        order.append("request")

    monkeypatch.setattr(cli_signals, "_start_request", _start)
    cli_signals._on_signal_byte(signal.SIGTERM)
    cli_signals._on_signal_byte(signal.SIGTERM)  # a second first-byte (a nested handler) does not re-request
    assert order == ["request"]


def test_a_byte_with_nothing_installed_is_ignored(monkeypatch: pytest.MonkeyPatch, died: list) -> None:
    monkeypatch.setattr(cli_signals, "_depth", 0)
    cli_signals._on_signal_byte(signal.SIGINT | cli_signals._REPEAT)
    assert died == [] and cli_signals._kill_at is None


def test_a_repeated_sigint_forces_and_a_repeated_sigterm_does_not(
    monkeypatch: pytest.MonkeyPatch, died: list[tuple[int, str]]
) -> None:
    _signalled_install(monkeypatch)
    cli_signals._on_signal_byte(signal.SIGTERM | cli_signals._REPEAT)
    assert died == []
    cli_signals._on_signal_byte(signal.SIGINT | cli_signals._REPEAT)
    assert died[0][0] == cli_signals.EXIT_FORCED


def test_a_fault_before_the_request_leaves_it_owed_for_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _signalled_install(monkeypatch)

    def _boom(_t: float) -> Any:
        raise RuntimeError("fault")

    monkeypatch.setattr(cli_signals, "_library_deadline", _boom)
    with pytest.raises(RuntimeError):
        cli_signals._on_signal_byte(signal.SIGTERM)
    assert not cli_signals._handled


def test_the_fallback_requests_sleeps_and_dies(monkeypatch: pytest.MonkeyPatch, died: list) -> None:
    _signalled_install(monkeypatch)
    requested: list[int] = []
    monkeypatch.setattr(cli_signals, "_start_request", lambda signum, at: requested.append(signum))
    monkeypatch.setattr(cli_signals.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cli_signals.os, "write", lambda *_a: 0)
    cli_signals._fallback()
    assert requested == [signal.SIGTERM]
    assert died[0][0] == cli_signals.EXIT_WATCHDOG


def test_the_fallback_survives_a_closed_stderr(monkeypatch: pytest.MonkeyPatch, died: list) -> None:
    def _closed(*_a: Any) -> int:
        raise OSError("EBADF")

    _signalled_install(monkeypatch)
    monkeypatch.setattr(cli_signals, "_handled", True)
    monkeypatch.setattr(cli_signals.time, "sleep", lambda _s: None)
    monkeypatch.setattr(cli_signals.os, "write", _closed)
    cli_signals._fallback()
    assert died[0][0] == cli_signals.EXIT_WATCHDOG


def test_the_stop_thread_turn_reads_bytes_and_fires_when_due(
    monkeypatch: pytest.MonkeyPatch, died: list[tuple[int, str]]
) -> None:
    read_end, write_end = os.pipe()
    try:
        monkeypatch.setattr(cli_signals, "_pipe_r", read_end)
        seen: list[int] = []
        monkeypatch.setattr(cli_signals, "_on_signal_byte", seen.append)
        os.write(write_end, bytes([cli_signals._REREAD, signal.SIGTERM]))
        monkeypatch.setattr(cli_signals, "_next_timeout", lambda: 0.0)
        cli_signals._stop_thread_turn()
        assert seen == [signal.SIGTERM]  # the re-read byte is not a signal
        assert died[0][0] == cli_signals.EXIT_WATCHDOG
    finally:
        os.close(read_end)
        os.close(write_end)


def test_the_stop_thread_keeps_running_after_an_idle_fault_and_falls_back_on_a_pending_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stop(BaseException):
        pass

    turns = {"n": 0}

    def _turn() -> None:
        turns["n"] += 1
        if turns["n"] == 1:
            raise RuntimeError("idle fault")
        if turns["n"] == 2:
            monkeypatch.setattr(cli_signals, "_signalled", signal.SIGTERM)
            monkeypatch.setattr(cli_signals, "_depth", 1)
            raise RuntimeError("pending fault")

    def _fell_back() -> None:
        raise _Stop

    monkeypatch.setattr(cli_signals, "_stop_thread_turn", _turn)
    monkeypatch.setattr(cli_signals, "_fallback", _fell_back)
    monkeypatch.setattr(cli_signals.time, "sleep", lambda _s: None)
    with pytest.raises(_Stop):
        cli_signals._stop_thread()
    assert turns["n"] == 2


def test_next_timeout_is_the_earliest_armed_time_never_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli_signals._next_timeout() is None
    monkeypatch.setattr(cli_signals, "_kill_at", time.monotonic() - 5)
    assert cli_signals._next_timeout() == 0.0
    monkeypatch.setattr(cli_signals, "_kill_at", None)
    monkeypatch.setattr(cli_signals, "_arm_kill_at", time.monotonic() + 5)
    assert cli_signals._next_timeout() is None  # an arm counts only while installed
    monkeypatch.setattr(cli_signals, "_depth", 1)
    assert 0 < cli_signals._next_timeout() <= 5


def test_arm_past_and_future_deadlines_and_disarm(monkeypatch: pytest.MonkeyPatch, written: list[int]) -> None:
    cli_signals.arm(Deadline.after(5))
    assert cli_signals._arm_kill_at is None  # not installed: a no-op
    monkeypatch.setattr(cli_signals, "_depth", 1)
    cli_signals.arm(Deadline.after(5))
    assert cli_signals._arm_kill_at == pytest.approx(time.monotonic() + 5 + cli_signals.FIRE_AFTER_LIBRARY_S, abs=0.5)
    cli_signals.arm(Deadline.after(0))
    assert cli_signals._arm_kill_at == pytest.approx(time.monotonic() + cli_signals.ARMED_PAST_DEADLINE_S, abs=0.5)
    assert written == [cli_signals._REREAD, cli_signals._REREAD]
    cli_signals.disarm()
    assert cli_signals._arm_kill_at is None


def test_disarm_never_cancels_a_watchdog_a_signal_armed(monkeypatch: pytest.MonkeyPatch, written: list[int]) -> None:
    monkeypatch.setattr(cli_signals, "_depth", 1)
    monkeypatch.setattr(cli_signals, "_kill_at", 123.0)
    cli_signals.disarm()
    assert cli_signals._kill_at == 123.0


def test_the_stop_hook_arms_with_the_requests_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    armed: list[Any] = []
    monkeypatch.setattr(cli_signals, "arm", armed.append)
    cli_signals._on_stop_requested()
    assert armed == []
    request = cancellation.request_stop("test", Deadline.after(5))
    cli_signals._on_stop_requested()
    assert armed[-1] is request.deadline


def test_installed_restores_nests_and_stays_out_of_other_threads() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with cli_signals.installed():
        assert signal.getsignal(signal.SIGTERM) is cli_signals._handler
        with cli_signals.installed():
            assert cli_signals._depth == 2
        assert cli_signals._depth == 1
        seen: dict[str, Any] = {}

        def _other() -> None:
            with cli_signals.installed():
                seen["depth"] = cli_signals._depth

        thread = threading.Thread(target=_other)
        thread.start()
        thread.join()
        assert seen["depth"] == 1  # off the main thread: nothing was installed or nested
    assert signal.getsignal(signal.SIGTERM) is before
    assert cli_signals._depth == 0


def test_installed_forgets_an_earlier_noted_signal_and_armed_watchdog_before_the_thread_can_see_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(cli_signals, "_ensure_stop_thread", lambda: seen.append(cli_signals._kill_at))
    cancellation.note_signal()
    monkeypatch.setattr(cli_signals, "_kill_at", 1.0)
    with cli_signals.installed():
        assert not cancellation.stop_noted()
    assert seen == [None], "the stop thread could start with an earlier install's kill time due"


def test_native_windows_keeps_its_old_ctrl_c_handler_and_nothing_else(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_signals.sys, "platform", "win32")
    before_int, before_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    with cli_signals.installed():
        assert signal.getsignal(signal.SIGTERM) is before_term
        assert signal.getsignal(signal.SIGINT) is not before_int
        assert cli_signals._depth == 0
    assert signal.getsignal(signal.SIGINT) is before_int


def test_windows_first_ctrl_c_cancels_and_raises_the_second_forces(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.models.language import cancellation as llm_cancellation
    from maxim.runtime import lane_backends

    calls: list[str] = []
    exits: list[int] = []
    monkeypatch.setattr(llm_cancellation, "request_shutdown", lambda: calls.append("cancel"))
    monkeypatch.setattr(lane_backends, "stop_active_spawner", lambda: calls.append("kill"))
    monkeypatch.setattr(cli_signals.os, "_exit", exits.append)
    pressed = [False]
    with pytest.raises(KeyboardInterrupt):
        cli_signals._windows_interrupt(pressed, signal.SIGINT, None)
    assert calls == ["cancel"] and exits == []
    with pytest.raises(KeyboardInterrupt):  # the stubbed exit returns, so it falls through to the raise
        cli_signals._windows_interrupt(pressed, signal.SIGINT, None)
    assert calls == ["cancel", "cancel", "kill"] and exits == [1]


def test_windows_ctrl_c_survives_a_failed_cancel_and_a_failed_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.models.language import cancellation as llm_cancellation
    from maxim.runtime import lane_backends

    def _boom() -> None:
        raise RuntimeError("no")

    exits: list[int] = []
    monkeypatch.setattr(llm_cancellation, "request_shutdown", _boom)
    monkeypatch.setattr(lane_backends, "stop_active_spawner", _boom)
    monkeypatch.setattr(cli_signals.os, "_exit", exits.append)
    with pytest.raises(KeyboardInterrupt):
        cli_signals._windows_interrupt([True], signal.SIGINT, None)
    assert exits == [1]


def test_an_unbounded_deadline_arms_nothing(monkeypatch: pytest.MonkeyPatch, written: list[int]) -> None:
    monkeypatch.setattr(cli_signals, "_depth", 1)
    cli_signals.arm(Deadline.none())
    assert cli_signals._arm_kill_at is None


def test_a_handler_installed_outside_python_restores_as_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, Any]] = []
    monkeypatch.setattr(cli_signals.signal, "signal", lambda sig, handler: calls.append((sig, handler)))
    monkeypatch.setattr(cli_signals, "_saved", (None, None))
    cli_signals._restore_handlers()
    assert calls == [(signal.SIGINT, signal.SIG_DFL), (signal.SIGTERM, signal.SIG_DFL)]


def test_a_forked_child_forgets_the_parents_thread_pipe_and_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    read_end, write_end = os.pipe()
    monkeypatch.setattr(cli_signals, "_pipe_r", read_end)
    monkeypatch.setattr(cli_signals, "_pipe_w", write_end)
    monkeypatch.setattr(cli_signals, "_thread", threading.current_thread())
    monkeypatch.setattr(cli_signals, "_kill_at", 5.0)
    monkeypatch.setattr(cli_signals, "_depth", 1)
    monkeypatch.setattr(cli_signals, "_saved", (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)))
    cancellation.note_signal()
    cli_signals._after_fork_in_child()
    assert cli_signals._pipe_r == -1 and cli_signals._thread is None and cli_signals._kill_at is None
    assert cli_signals._depth == 0
    assert not cancellation.stop_noted()
    with pytest.raises(OSError):
        os.fstat(read_end)


def test_die_runs_its_steps_on_a_helper_and_always_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    exits: list[int] = []
    release = threading.Event()
    monkeypatch.setattr(cli_signals.os, "_exit", exits.append)
    monkeypatch.setattr(cli_signals, "LAST_WORDS_S", 0.05)
    monkeypatch.setattr(cli_signals, "_kill_spawned_server", lambda: 1 / 0)
    monkeypatch.setattr(cli_signals, "_last_words", lambda _reason: release.wait(5))
    cli_signals._die(75, "reason")  # a kill that raises and a last line that blocks
    release.set()
    assert exits == [75]


def test_last_words_restore_the_terminal_then_write(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(cli_signals, "_restore_terminal", lambda: order.append("restore"))
    monkeypatch.setattr(cli_signals.os, "write", lambda fd, data: order.append(f"write {fd}"))
    cli_signals._last_words("reason")
    assert order == ["restore", "write 2"]


class _Server:
    def __init__(self, pid: int | None) -> None:
        self.pid = pid

    def spawned_server_pid(self) -> int | None:
        return self.pid


def test_the_kill_targets_the_servers_group_never_our_own(monkeypatch: pytest.MonkeyPatch) -> None:
    killed: list[int] = []
    monkeypatch.setattr(cli_signals.os, "killpg", lambda pid, sig: killed.append(pid))
    monkeypatch.setitem(cli_signals.sys.modules, "maxim.runtime.llm_server", _Server(None))
    cli_signals._kill_spawned_server()
    monkeypatch.setitem(cli_signals.sys.modules, "maxim.runtime.llm_server", _Server(os.getpgrp()))
    cli_signals._kill_spawned_server()
    assert killed == []
    monkeypatch.setitem(cli_signals.sys.modules, "maxim.runtime.llm_server", _Server(4242))
    cli_signals._kill_spawned_server()
    assert killed == [4242]


def test_the_kill_tolerates_a_server_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gone(*_a: Any) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(cli_signals.os, "killpg", _gone)
    monkeypatch.setitem(cli_signals.sys.modules, "maxim.runtime.llm_server", _Server(4242))
    cli_signals._kill_spawned_server()


def test_spawned_server_pid_reads_a_live_process_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime import llm_server

    monkeypatch.setattr(llm_server, "_active_spawner", None)
    assert llm_server.spawned_server_pid() is None
    process = type("P", (), {"pid": 7, "returncode": None})()
    monkeypatch.setattr(llm_server, "_active_spawner", type("S", (), {"_process": process})())
    assert llm_server.spawned_server_pid() == 7
    process.returncode = -9
    assert llm_server.spawned_server_pid() is None


def test_make_and_start_request(monkeypatch: pytest.MonkeyPatch) -> None:
    done = threading.Event()
    monkeypatch.setattr(cli_signals, "_make_request", lambda name, at: done.set())
    cli_signals._start_request(signal.SIGTERM, time.monotonic() + 5)
    assert done.wait(5)


def test_make_request_logs_a_failure_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fails(*_a: Any) -> None:
        raise RuntimeError("no")

    monkeypatch.setattr(cli_signals, "request_stop", _fails)
    cli_signals._make_request("SIGTERM", time.monotonic())


def test_mode_markers() -> None:
    cli_signals.entered_sim_mode()
    cli_signals.entered_non_sim_mode()
    assert cli_signals._mode == "sim"  # only left_sim_mode undoes a sim
    cli_signals.left_sim_mode()
    assert cli_signals._mode == "non_sim"
