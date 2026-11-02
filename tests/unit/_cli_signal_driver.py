"""A child process for the #1344 CLI-signal gates: runs the REAL ``maxim.cli.main`` with only the mode's body stubbed.

``python _cli_signal_driver.py <scenario>``. Each scenario replaces ``cli._main_impl`` with a body that does what a run
mode does (claims a real memory session, waits cooperatively, blocks in a human prompt, ignores the stop), prints
``READY`` when the parent may signal it, and returns or exits. The parent (``test_bounded_stop_cli_1344.py``) sends
real signals and reads the exit code, so the handler, the gate, the watchdog and the exit mapping are the shipped
ones. No LLM, no sim, no robot.
"""

from __future__ import annotations

import os
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
    from maxim import cli_signals
    from maxim.utils import cancellation

    if scenario == "session_cooperative":  # a non-sim mode with a session that stops when asked
        _mode("non_sim")
        _claim_session()
        _ready()
        return 0 if _wait_for_stop() else 3
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
    if scenario == "session_ignores_stop":  # a save that never finishes: only the watchdog ends it
        _mode("non_sim")
        _claim_session()
        _ready()
        while True:
            time.sleep(0.1)
    if scenario == "hook_lock_held":  # the main thread holds the stop module's hook lock when the signal lands
        _mode("non_sim")
        _claim_session()
        with cancellation._hook_lock:
            _ready()
            time.sleep(1.0)  # the signal arrives here
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
        try:
            with cli_signals.interruptible():
                sys.stdin.readline()
        except KeyboardInterrupt:
            pass
        return 0 if cli_signals.stop_pending() else 3
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
    raise SystemExit(f"unknown scenario {scenario!r}")


def main() -> None:
    scenario = sys.argv[1]
    from maxim import cli

    cli._main_impl = lambda argv=None: _body(scenario)  # type: ignore[assignment]
    if scenario == "stop_thread_raises":  # inject a fault into the stop thread's deadline step (A1/B1)
        from maxim import cli_signals

        def _boom(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("injected stop-thread fault")

        cli_signals._library_deadline = _boom  # type: ignore[attr-defined]
        cli._main_impl = lambda argv=None: _body("session_ignores_stop")  # type: ignore[assignment]
    code = cli.main([])
    sys.stdout.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
