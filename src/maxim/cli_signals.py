"""The ``maxim`` CLI's own SIGINT/SIGTERM handling (#1344; docs/plans/bounded_stop.md §6).

Only a process Maxim owns handles signals and exits (§2): ``cli.main`` enters :func:`installed` around its whole body,
so every subcommand is covered (owner decision 2026-11-02). **A signal during ``cli.main`` ends the process**: the
caller's handlers are restored, but a watchdog a signal armed stays armed so it bounds the exit (atexit, non-daemon
joins). The library never imports this module; it marks its human-input reads with ``utils.cancellation.interruptible``
and polls ``yield_requested()``.

**The handler takes no lock and prints nothing.** It runs on the main thread between bytecodes, where the main thread
may itself hold the stop module's locks, so everything that takes a lock runs on the **stop thread**: the handler
records the signal in plain globals, writes one byte to a self-pipe, and raises ``KeyboardInterrupt`` only:

- before any memory session was ever claimed and before any stop was requested (the pre-session gate), so a process
  with nothing to save unwinds at once;
- inside an ``interruptible()`` region (a human-input read that only a raise can wake, PEP 475).

Otherwise it returns and the modes stop cooperatively on the request the stop thread makes.

**The stop thread is the watchdog.** One daemon thread and one pipe per process, created on the first install and
never closed (no writer can race a close). On the first signal it stores the kill time FIRST, then a short-lived helper
makes ``request_stop`` (the only call that takes the stop module's lock), so a wedged lock delays the request but never
the kill. A repeat SIGTERM is ignored; a repeat SIGINT forces exit 130. At the kill time it SIGKILLs the spawned LLM
server's process group, gives the terminal restore and a last line one second on a helper (a full stderr pipe or a
flow-stopped tty must not outlive the watchdog), and ``os._exit(75)``.

A stop made some other way (``maxim serve``'s uvicorn path, the console's drain) arms the same watchdog through a stop
hook, and clearing that request disarms it. Both are no-ops unless installed, so an embedder of the console is
untouched. Serve's budget stays ``console.stop_budget_s``: it does not read K.

Native Windows has none of this (see :func:`_windows_ctrl_c`): it keeps the CLI's old Ctrl+C handling.

The budget, from the signal time t: Y = :data:`YIELD_TIMEOUT_S` (30 s, owner decision 2026-10-25; a constant, not a
config key: ``MaximConfig``'s root is SHAPE-FROZEN, and a supervisor tunes the window through K). K =
``MAXIM_SUPERVISOR_KILL_S``, the seconds after the signal at which a supervisor sends SIGKILL. The library gets
t + min(Y, K - 3); the watchdog fires 1 s later, so the process is gone by t + K - 1.

Exit codes (:func:`exit_code`): a sim mode, or a mode not yet known (the strict default), exits 4 on a stop. A non-sim
mode keeps a real failure (a body's 1 or 2) and otherwise exits 0. A pre-session SIGINT keeps the command's own code:
130 when it reached ``main`` uncaught, server mode's 0 when the command handles it. The watchdog exits 75.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import select
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Literal

from maxim.integration.session_ownership import session_ever_claimed
from maxim.utils.cancellation import (
    current_stop,
    forget_noted_signal,
    interruptible_depth,
    note_signal,
    register_stop_hook,
    request_stop,
    yield_requested,
)
from maxim.utils.deadline import Deadline

logger = logging.getLogger(__name__)

EXIT_WATCHDOG = 75
EXIT_SIM_ABORT = 4  # sim_types.SIMULATION_ABORT_EXIT_CODE; pinned equal by test
EXIT_FORCED = 130
EXIT_INTERRUPTED = 130
YIELD_TIMEOUT_S = 30.0
LIBRARY_RESERVE_S = 3.0  # the library's deadline ends this long before K
FIRE_AFTER_LIBRARY_S = 1.0  # the watchdog fires this long after the library's deadline
ARMED_PAST_DEADLINE_S = 2.0  # an arm() whose deadline already passed fires this long from now
LAST_WORDS_S = 1.0  # what the terminal restore and the last line may take before the exit
IDLE_FAULT_BACKOFF_S = 0.1
# The kill time when the stop thread faulted before it could compute one: no env read on that path.
FALLBACK_KILL_AFTER_S = YIELD_TIMEOUT_S + FIRE_AFTER_LIBRARY_S

_REREAD = 0  # the pipe byte for "re-read the armed state" (arm/disarm)
_REPEAT = 0x40  # OR'd onto a signal number the handler saw as a repeat

Mode = Literal["unknown", "sim", "non_sim"]

# ── Per process: the pipe, the stop thread and the stop hook (never closed; a forked child forgets the first two) ──
_pipe_r = -1
_pipe_w = -1
_thread: threading.Thread | None = None
_hook_registered = False

# ── Per install ───────────────────────────────────────────────────────────────────────────────────────────────────
_depth = 0
_saved: tuple[Any, Any] | None = None
_mode: Mode = "unknown"
_supervisor_kill_s: float | None = None
# Written by the handler (main thread, lock-free): the first signal, when, and whether the gate said cooperative.
_signalled: int | None = None
_signal_time: float | None = None
_cooperative = False
# Written by the stop thread and by arm().
_handled = False
_kill_at: float | None = None  # armed by a signal; kept past uninstall so it bounds the exit
_arm_kill_at: float | None = None
# Resolved at install on the main thread, so the kill path imports nothing.
_restore_terminal: Callable[[], None] | None = None


def _write(byte: int) -> None:
    if _pipe_w < 0:
        return
    try:
        os.write(_pipe_w, bytes([byte]))
    except OSError:  # full (the stop thread is wedged) or closed: the watchdog's own timer still runs
        pass


def _handler(signum: int, frame: Any) -> None:
    global _signalled, _signal_time, _cooperative
    if _signalled is not None:
        _write(signum | _REPEAT)  # the stop thread ignores a repeated SIGTERM and forces on a repeated SIGINT
        return
    cooperative = session_ever_claimed() or yield_requested()
    _signal_time = time.monotonic()
    _cooperative = cooperative
    _signalled = signum
    note_signal()
    _write(signum)
    if not cooperative or interruptible_depth() > 0:
        raise KeyboardInterrupt()


# ── The budget ────────────────────────────────────────────────────────────────────────────────────────────────────


def _validated_kill_s(raw: str | None) -> float | None:
    """K: a finite number of seconds above the library's reserve, else None with one WARNING."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= LIBRARY_RESERVE_S:
        logger.warning(
            "MAXIM_SUPERVISOR_KILL_S=%r is not a finite number of seconds above %s; ignored", raw, LIBRARY_RESERVE_S
        )
        return None
    return value


def _library_deadline(signal_time: float) -> tuple[float, float]:
    """(the library's deadline, the watchdog's kill time), both monotonic."""
    budget = YIELD_TIMEOUT_S
    if _supervisor_kill_s is not None:
        budget = min(budget, _supervisor_kill_s - LIBRARY_RESERVE_S)
    return signal_time + budget, signal_time + budget + FIRE_AFTER_LIBRARY_S


# ── The stop thread ───────────────────────────────────────────────────────────────────────────────────────────────


def _kill_spawned_server() -> None:
    """SIGKILL the spawned LLM server's process group, lock-free (PR 3 covers a server not yet registered)."""
    server_mod = sys.modules.get("maxim.runtime.llm_server")
    pid = server_mod.spawned_server_pid() if server_mod is not None else None
    if pid is None or pid == os.getpgrp():  # spawned with start_new_session, so its group is its pid; never ours
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:  # already gone
        pass


def _last_words(reason: str) -> None:
    if _restore_terminal is not None:
        _restore_terminal()
    os.write(2, f"\n  {reason}\n".encode())


def _die(code: int, reason: str) -> None:
    """End the process. No step may stop the exit: on a helper, the steps that can block."""
    try:
        _kill_spawned_server()
    except BaseException:  # noqa: BLE001 -- the exit below must happen whatever this did
        pass
    try:
        helper = threading.Thread(target=_last_words, args=(reason,), name="maxim.last-words", daemon=True)
        helper.start()
        helper.join(LAST_WORDS_S)
    except BaseException:  # noqa: BLE001 -- as above
        pass
    finally:
        os._exit(code)


def _make_request(name: str, library_at: float) -> None:
    try:
        request_stop(f"signal {name}", Deadline(at=library_at))
    except Exception:
        logger.exception("The stop request for %s failed", name)


def _start_request(signum: int, library_at: float) -> None:
    name = signal.Signals(signum).name
    threading.Thread(target=_make_request, args=(name, library_at), name="maxim.stop-request", daemon=True).start()


def _on_signal_byte(byte: int) -> None:
    global _handled, _kill_at
    if _depth == 0 or _signalled is None:
        return  # nothing is installed or signalled: a byte from an install that already ended
    if byte & _REPEAT:
        if byte & ~_REPEAT == signal.SIGINT:
            _die(EXIT_FORCED, "Forced shutdown (second Ctrl+C).")
        return  # a repeated SIGTERM is ignored
    if _handled:
        return
    assert _signal_time is not None
    library_at, kill_at = _library_deadline(_signal_time)
    _kill_at = kill_at
    _start_request(byte, library_at)
    _handled = True


def _next_timeout() -> float | None:
    times = [t for t in (_kill_at, _arm_kill_at if _depth > 0 else None) if t is not None]
    return max(0.0, min(times) - time.monotonic()) if times else None


def _fallback() -> None:
    """A fault while a signal is pending: still make the request and end the process, from a constant, reading nothing."""
    start = _signal_time if _signal_time is not None else time.monotonic()
    try:
        os.write(2, b"\n  The stop thread failed; the stop continues on its fallback.\n")
    except OSError:  # stderr closed by a supervisor
        pass
    try:
        if not _handled and _signalled is not None:
            _start_request(_signalled, start + YIELD_TIMEOUT_S)
        time.sleep(max(0.0, (_kill_at or start + FALLBACK_KILL_AFTER_S) - time.monotonic()))
    finally:
        _die(EXIT_WATCHDOG, "Stop watchdog: the run did not stop in time.")


def _stop_thread_turn() -> None:
    ready, _, _ = select.select([_pipe_r], [], [], _next_timeout())
    if ready:
        for byte in os.read(_pipe_r, 64):
            if byte != _REREAD:
                _on_signal_byte(byte)
    timeout = _next_timeout()
    if timeout is not None and timeout <= 0:
        _die(EXIT_WATCHDOG, "Stop watchdog: the run did not stop in time.")


def _stop_thread() -> None:
    while True:
        try:
            _stop_thread_turn()
        except Exception:
            if _signalled is not None and _depth > 0:
                _fallback()
            logger.exception("The stop thread's turn failed while idle; it keeps running")
            time.sleep(IDLE_FAULT_BACKOFF_S)


def _ensure_stop_thread() -> None:
    global _pipe_r, _pipe_w, _thread
    if _thread is not None:
        return
    _pipe_r, _pipe_w = os.pipe()
    os.set_blocking(_pipe_w, False)
    _thread = threading.Thread(target=_stop_thread, name="maxim.stop", daemon=True)
    _thread.start()


def _after_fork_in_child() -> None:
    """A forked child (a DataLoader worker) has no stop thread: forget the parent's, and its handlers and watchdog."""
    global _pipe_r, _pipe_w, _thread, _depth, _kill_at
    for fd in (_pipe_r, _pipe_w):
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:  # already closed by the child's own cleanup
                pass
    _pipe_r = _pipe_w = -1
    _thread = None
    _kill_at = None
    forget_noted_signal()
    if _depth > 0 and _saved is not None:
        _restore_handlers()
        _depth = 0
        _reset_run_state()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


# ── Install, the mode marker, the armed watchdog ──────────────────────────────────────────────────────────────────


def _reset_run_state() -> None:
    global _signalled, _signal_time, _cooperative, _handled, _mode, _arm_kill_at
    _signalled = None
    _signal_time = None
    _cooperative = False
    _handled = False
    _mode = "unknown"
    _arm_kill_at = None


def _restore_handlers() -> None:
    assert _saved is not None
    # A handler installed outside Python reads back as None and cannot be re-installed: the default stands in.
    signal.signal(signal.SIGINT, _saved[0] if _saved[0] is not None else signal.SIG_DFL)
    signal.signal(signal.SIGTERM, _saved[1] if _saved[1] is not None else signal.SIG_DFL)


def _uninstall() -> None:
    """Restore the caller's handlers with both signals held on this thread (a signal aimed at the process may still
    land on another thread; ``cli.main`` then returns 130)."""
    global _depth
    both = {signal.SIGINT, signal.SIGTERM}
    held = signal.pthread_sigmask(signal.SIG_BLOCK, both)
    try:
        _restore_handlers()
        _depth = 0
        _reset_run_state()
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, held)


def _on_stop_requested() -> None:
    request = current_stop()
    if request is not None:
        arm(request.deadline)


def _windows_interrupt(pressed: list[bool], signum: int, frame: Any) -> None:
    """Native Windows' Ctrl+C, as the CLI always had it: the first press wakes any backend retry loop, so it stops
    spending, and raises; the second force-kills the spawned server and exits."""
    try:
        from maxim.models.language.cancellation import request_shutdown

        request_shutdown()
    except Exception:
        logger.warning("Could not signal the LLM cancellation at Ctrl+C", exc_info=True)
    if pressed[0]:
        try:
            from maxim.runtime.lane_backends import stop_active_spawner

            stop_active_spawner()
        except Exception:
            logger.warning("Could not stop the spawned server at a forced Ctrl+C", exc_info=True)
        print("\n  Forced shutdown.", file=sys.stderr)
        os._exit(1)
    pressed[0] = True
    raise KeyboardInterrupt


@contextmanager
def _windows_ctrl_c() -> Iterator[None]:
    """Everything below is POSIX (a pipe in ``select``, ``pthread_sigmask``), so native Windows keeps its old handler,
    for SIGINT only, with no watchdog and no yield."""
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, functools.partial(_windows_interrupt, [False]))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous if previous is not None else signal.SIG_DFL)


@contextmanager
def installed() -> Iterator[None]:
    """Handle SIGINT and SIGTERM for the block (``cli.main``'s body); restore the caller's handlers after."""
    global _depth, _saved, _restore_terminal, _supervisor_kill_s, _kill_at, _hook_registered
    if threading.current_thread() is not threading.main_thread():
        logger.info("Not on the main thread: the CLI's signal handling is not installed")
        yield
        return
    if sys.platform == "win32":
        with _windows_ctrl_c():
            yield
        return
    if _depth > 0:
        _depth += 1
        try:
            yield
        finally:
            _depth -= 1
        return
    from maxim.simulation import tty_stdin

    _restore_terminal = tty_stdin._restore_terminal
    _supervisor_kill_s = _validated_kill_s(os.environ.get("MAXIM_SUPERVISOR_KILL_S"))
    _kill_at = None  # before the thread starts: it must not find an earlier install's kill time due
    _reset_run_state()
    forget_noted_signal()
    _ensure_stop_thread()
    if not _hook_registered:
        register_stop_hook(_on_stop_requested, disarm)
        _hook_registered = True
    _saved = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    _depth = 1
    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        yield
    finally:
        _uninstall()


def entered_sim_mode() -> None:
    """This run is a simulation: a stop exits 4 (a typed abort, never evidence)."""
    global _mode
    _mode = "sim"


def entered_non_sim_mode() -> None:
    """This run is not a simulation. It never undoes :func:`entered_sim_mode`: only :func:`left_sim_mode` does."""
    global _mode
    if _mode != "sim":
        _mode = "non_sim"


def left_sim_mode() -> None:
    """A sim started from the menu finished with no stop: the menu is a non-sim mode again."""
    global _mode
    _mode = "non_sim"


def arm(deadline: Deadline) -> None:
    """Arm the watchdog for a stop request (the stop hook; the signal path stores its own kill time too)."""
    global _arm_kill_at
    if _depth == 0 or not math.isfinite(deadline.at):
        return
    now = time.monotonic()
    _arm_kill_at = deadline.at + FIRE_AFTER_LIBRARY_S if deadline.at > now else now + ARMED_PAST_DEADLINE_S
    _write(_REREAD)


def disarm() -> None:
    """Cancel an :func:`arm` (its request was cleared: the drain finished); never a watchdog a signal armed."""
    global _arm_kill_at
    if _depth == 0:
        return
    _arm_kill_at = None
    _write(_REREAD)


def exit_code(code: int) -> int:
    """The process's exit code once its body returned ``code`` (mapped inside :func:`installed`, before its state
    resets). With no signal, ``code``. A pre-session SIGINT keeps ``code``. Otherwise a sim, or a mode not yet known,
    exits 4; a non-sim mode keeps a real failure and exits 0 for a clean or interrupted end."""
    if _signalled is None or (not _cooperative and _signalled == signal.SIGINT):
        return code
    if _mode != "non_sim":
        return EXIT_SIM_ABORT
    return 0 if code in (0, EXIT_INTERRUPTED) else code


def _reset_for_tests() -> None:
    """Forget a signal-armed watchdog and the per-run state (the conftest guard; production never calls it)."""
    global _kill_at
    _kill_at = None
    _reset_run_state()
