"""A child process for the #1344 CLI-signal gates: runs the REAL ``maxim.cli.main`` with only the mode's body stubbed.

``python _cli_signal_driver.py <scenario>``. Each scenario replaces ``cli._main_impl`` with a body that does what a run
mode does (claims a real memory session, waits cooperatively, blocks in a human prompt, ignores the stop), prints
``READY`` when the parent may signal it, and returns or exits. The parent (``test_bounded_stop_cli_1344.py``) sends
real signals and reads the exit code, so the handler, the gate, the watchdog and the exit mapping are the shipped
ones. No LLM, no sim, no robot.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Any


def _ready() -> None:
    sys.stdout.write("READY\n")
    sys.stdout.flush()


def _claim_session() -> Any:
    """A real claim on a real session ledger, as ``MemoryHub.on_session_start`` makes."""
    from maxim.integration.session_ownership import SessionOwnership

    return SessionOwnership().claim(owner=object(), live=True)


def _wait_for_stop(limit_s: float = 60.0) -> bool:
    from maxim.utils.cancellation import yield_requested

    end = time.monotonic() + limit_s
    while time.monotonic() < end:
        if yield_requested():
            return True
        time.sleep(0.05)
    return False


def _mode(kind: str) -> None:
    from maxim import cli_signals

    if kind == "sim":
        cli_signals.entered_sim_mode()
    elif kind == "non_sim":
        cli_signals.entered_non_sim_mode()


def _body(scenario: str) -> int:
    from maxim import cli, cli_signals
    from maxim.utils import cancellation
    from maxim.utils.deadline import Deadline

    if scenario == "session_cooperative":  # a non-sim mode with a session that stops when asked
        _mode("non_sim")
        _claim_session()
        _ready()
        stopped = _wait_for_stop()
        print("SAVED", flush=True)  # the work after the stop ran: the handler did not raise through it
        return 0 if stopped else 3
    if scenario == "prompt_while_the_request_cannot_land":  # the signal is noted, the request is still blocked
        _mode("non_sim")
        _claim_session()
        with cancellation._hook_lock:
            _ready()
            while not cli_signals._signalled:  # the signal lands here; its request waits on this lock
                time.sleep(0.01)
            try:
                with cancellation.interruptible():
                    input()
            except KeyboardInterrupt:
                print("PROMPT_REFUSED", flush=True)
        return 0
    if scenario == "sim_cooperative":  # a sim with a session that yields; the body returns 0 (as a passing scenario
        _mode("sim")  # sim does today), so the yield code must come from the mapping
        _claim_session()
        _ready()
        return 0 if _wait_for_stop() else 3
    if scenario == "sim_sys_exit":  # a branch that ends in its own sys.exit (--benchmark)
        _mode("sim")
        _claim_session()
        _ready()
        _wait_for_stop()
        sys.exit(1)
    if scenario in ("pre_session_non_sim", "pre_session_sim", "pre_session_unknown"):
        _mode({"pre_session_non_sim": "non_sim", "pre_session_sim": "sim"}.get(scenario, "unknown"))
        _ready()
        time.sleep(60)
        return 3
    if scenario == "pre_session_handles_interrupt":  # a command that handles Ctrl+C itself (server mode, tunnel)
        _mode("non_sim")
        _ready()
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            return 0
        return 3
    if scenario == "session_ignores_stop":  # a save that never finishes: only the watchdog ends it
        _mode("non_sim")
        _claim_session()
        _ready()
        _wait_for_stop()
        print("STOP_SEEN", flush=True)
        while True:
            time.sleep(0.1)
    if scenario == "stderr_full":  # a full stderr pipe nobody drains: the watchdog's last line must not hold the exit
        _mode("non_sim")
        _claim_session()
        read_end, write_end = os.pipe()
        os.set_blocking(write_end, False)
        try:
            while True:
                os.write(write_end, b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(write_end, True)
        os.dup2(write_end, 2)
        _ready()
        while True:
            time.sleep(0.1)
    if scenario == "kills_the_spawned_server":  # a real server-shaped child, in its own session, as the spawner makes
        import subprocess

        _mode("non_sim")
        _claim_session()
        child = cancellation.spawn_registered(  # as LocalServerSpawner.start registers it (#1344 PR 3a)
            lambda: subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True),
            kind="llm_server",
            stop=lambda: None,
        )
        print(f"SERVER_PID {child.pid}", flush=True)
        _ready()
        while True:
            time.sleep(0.1)
    if scenario == "hook_lock_held":  # the main thread holds the stop module's hook lock when the signal lands
        _mode("non_sim")
        _claim_session()
        with cancellation._hook_lock:
            _ready()
            while not cancellation.stop_noted():  # the signal lands here, with the lock held
                time.sleep(0.01)
            time.sleep(0.2)
        return 0 if _wait_for_stop(10) else 3
    if scenario == "hook_lock_wedged":  # a lock holder that never lets go: the request never lands, the kill must
        _mode("non_sim")
        _claim_session()
        cancellation._hook_lock.acquire()
        _ready()
        while True:
            time.sleep(0.1)
    if scenario == "prompt_after_session":  # a human-input read after a session, the menu's shape
        _mode("non_sim")
        _claim_session()
        _ready()
        import readline  # noqa: F401  (input() goes through it when stdin AND stdout are ttys, as the menu's does)

        from maxim.utils.cancellation import interruptible

        try:
            with interruptible():
                input()
        except KeyboardInterrupt:
            pass
        return 0 if cancellation.stop_noted() else 3
    if scenario == "late_claim":  # a signal before any session, then a loop claims on another thread
        _mode("non_sim")
        _ready()
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            ev = threading.Event()
            cancellation.register_stop_event(ev)
            print("BORN_STOPPED" if ev.wait(5) else "NOT_STOPPED", flush=True)
            raise
        return 3
    if scenario == "sim_refused_under_own_stop":  # the sim after a stop refuses: a yield, never a traceback or 2
        _mode("sim")
        _claim_session()
        _ready()
        _wait_for_stop()
        from maxim.exceptions import StopStillPending

        raise StopStillPending("a simulation refused to start: a stop is pending")
    if scenario == "armed_then_hangs":  # serve's path: a stop made outside the handler arms the idle stop thread
        _mode("non_sim")
        cli_signals.arm(Deadline.after(1.0))
        _ready()
        while True:
            time.sleep(0.1)
    if scenario == "armed_then_disarmed":
        _mode("non_sim")
        cli_signals.arm(Deadline.after(1.0))
        cli_signals.disarm()
        _ready()
        time.sleep(3.0)
        return 0
    if scenario == "stale_byte_then_signal":  # a byte from an ended install must not read as this run's signal
        _mode("non_sim")
        _claim_session()
        cli_signals._write(int(signal.SIGINT))
        time.sleep(0.3)
        print("EARLY_STOP" if cancellation.yield_requested() else "NO_STOP", flush=True)
        _ready()
        return 0 if _wait_for_stop() else 3
    if scenario == "idle_fault_then_signal":  # the stop thread faults while idle, then a signal must still work
        _mode("non_sim")
        _claim_session()
        real = cli_signals._next_timeout
        calls = {"n": 0}

        def _once(*a: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("injected idle fault")
            return real(*a)

        cli_signals._next_timeout = _once  # type: ignore[assignment]
        cli_signals._write(0)
        time.sleep(0.3)
        _ready()
        return 0 if _wait_for_stop() else 3
    if scenario == "fork_child_signal":  # a child forked inside the parent's install must install for itself
        pid = os.fork()
        if pid == 0:  # the child's cli.main call must create its own pipe and stop thread
            cli._main_impl = lambda argv=None: _body("session_cooperative")  # type: ignore[assignment]
            os._exit(cli.main([]))
        _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status)
    raise SystemExit(f"unknown scenario {scenario!r}")


def main() -> None:
    scenario = sys.argv[1]
    from maxim import cli

    if not scenario.startswith("real_"):
        cli._main_impl = lambda argv=None: _body(scenario)  # type: ignore[assignment]
    if scenario == "real_sim_dispatch":  # the REAL _main_impl: only the sim itself is stubbed
        from maxim.simulation import orchestrator

        def _fake_sim(*_a: Any, **_k: Any) -> Any:
            _claim_session()
            _ready()
            _wait_for_stop()
            return type("R", (), {"finish_reason": "completed", "result": None})()

        orchestrator.start_simulation_mode = _fake_sim  # type: ignore[assignment]
        code = cli.main(["--sim", "a narrow goal", "--interactive", "false"])
        sys.stdout.flush()
        os._exit(code)
    if scenario == "real_non_sim_dispatch":  # the REAL parser and dispatch; a pre-session wait in a non-sim branch

        def _waits() -> int:
            _ready()
            time.sleep(60)
            return 3

        cli._handle_list_models = _waits  # type: ignore[assignment]
        code = cli.main(["--list-models"])
        sys.stdout.flush()
        os._exit(code)
    if scenario == "real_subcommand_dispatch":  # a positional subcommand is a non-sim mode
        import maxim.doctor as doctor

        def _doctor_waits(argv: Any) -> int:
            _ready()
            time.sleep(60)
            return 3

        doctor.run_doctor_subcommand = _doctor_waits  # type: ignore[assignment]
        code = cli.main(["doctor"])
        sys.stdout.flush()
        os._exit(code)
    if scenario == "real_roy_run":  # `roy run` is positional but runs sims in-process
        import maxim.roy as roy

        def _roy_waits(argv: Any) -> int:
            _ready()
            time.sleep(60)
            return 3

        roy.run_roy_subcommand = _roy_waits  # type: ignore[assignment]
        code = cli.main(["roy", "run"])
        sys.stdout.flush()
        os._exit(code)
    if scenario == "real_serve":  # the REAL `maxim serve`, on a free port
        code = cli.main(["serve", "--port", sys.argv[2]])
        sys.stdout.flush()
        os._exit(code)
    if scenario == "stop_thread_raises":  # inject a fault into the stop thread's deadline step (A1/B1)
        from maxim import cli_signals

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("injected stop-thread fault")

        cli_signals._library_deadline = _boom  # type: ignore[attr-defined]
        cli._main_impl = lambda argv=None: _body("session_ignores_stop")  # type: ignore[assignment]
    if scenario == "stop_thread_raises_cooperative":  # the same fault: the fallback's request must still land
        from maxim import cli_signals

        def _boom_too(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("injected stop-thread fault")

        cli_signals._library_deadline = _boom_too  # type: ignore[attr-defined]
        cli._main_impl = lambda argv=None: _body("session_cooperative")  # type: ignore[assignment]
    code = cli.main([])
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
