"""Bounded stop PR 2 (#1344): the CLI's own signal handling (docs/plans/bounded_stop.md §6; approach note v6).

The process gates run ``tests/unit/_cli_signal_driver.py`` in a child: the REAL ``maxim.cli.main`` (handler, gate,
watchdog, exit mapping) with only the mode's body stubbed, and send it real signals. The unit gates cover the pieces a
child cannot reach cheaply: the robot run's stop event, the post-sim prompt, ``spawn_evidence``'s exit-0 allowlist.

Exit codes (owner decisions 2026-10-25/26 and 2026-11-02): a yield exits 4 in a sim mode and 0 otherwise; a signal
before any session exits 0 (SIGTERM, non-sim), 4 (SIGTERM, sim or unknown mode) or 130 (SIGINT); the watchdog 75.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import maxim

RED = pytest.mark.xfail(strict=True, raises=AssertionError, reason="#1344 red gate: flips when PR 2 lands")

_DRIVER = Path(__file__).with_name("_cli_signal_driver.py")
_SRC = str(Path(maxim.__file__).resolve().parents[1])

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


class _Child:
    def __init__(self, scenario: str, tmp_path: Path, env: dict[str, str] | None = None) -> None:
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
        child_env.update({"PYTHONPATH": _SRC, "HOME": str(tmp_path), "PYTHONUNBUFFERED": "1"})
        child_env.update(env or {})
        self.proc = subprocess.Popen(
            [sys.executable, str(_DRIVER), scenario],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            text=True,
        )
        self.lines: list[str] = []

    def wait_ready(self, timeout_s: float = 30.0) -> None:
        end = time.monotonic() + timeout_s
        assert self.proc.stdout is not None
        while time.monotonic() < end:
            line = self.proc.stdout.readline()
            if not line:
                break
            self.lines.append(line.strip())
            if line.strip() == "READY":
                return
        self.proc.kill()
        _, err = self.proc.communicate()
        raise AssertionError(f"the child never got READY (exit {self.proc.returncode}): {err[-2000:]}")

    def signal(self, sig: int) -> float:
        self.proc.send_signal(sig)
        return time.monotonic()

    def exit_code(self, timeout_s: float) -> int:
        try:
            out, err = self.proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, err = self.proc.communicate()
            raise AssertionError(f"the child did not exit within {timeout_s}s: {err[-2000:]}") from None
        self.lines.extend(out.splitlines())
        self.stderr = err
        return self.proc.returncode


def _run(scenario: str, tmp_path: Path, sig: int, *, timeout_s: float = 20.0, env: dict | None = None) -> tuple:
    child = _Child(scenario, tmp_path, env)
    child.wait_ready()
    sent = child.signal(sig)
    code = child.exit_code(timeout_s)
    return code, time.monotonic() - sent, child


# ── The yield: a session open, the mode stops cooperatively ───────────────────────────────────────────────────────


@RED
def test_g2_sigterm_to_a_non_sim_mode_with_a_session_yields_and_exits_0(tmp_path: Path) -> None:
    code, took, _ = _run("session_cooperative", tmp_path, signal.SIGTERM)
    assert code == 0
    assert took < 10


@RED
def test_g2b_sigint_to_a_non_sim_mode_with_a_session_also_yields_and_exits_0(tmp_path: Path) -> None:
    code, _, _ = _run("session_cooperative", tmp_path, signal.SIGINT)
    assert code == 0


@RED
def test_g1_g11_a_yielded_sim_exits_4_even_when_its_body_returned_0(tmp_path: Path) -> None:
    code, _, _ = _run("sim_cooperative", tmp_path, signal.SIGTERM)
    assert code == 4


@RED
def test_g24_a_yielded_sim_branch_that_ends_in_its_own_sys_exit_still_exits_4(tmp_path: Path) -> None:
    code, _, _ = _run("sim_sys_exit", tmp_path, signal.SIGTERM)
    assert code == 4


# ── Before any session ────────────────────────────────────────────────────────────────────────────────────────────


@RED
def test_g3_sigterm_before_any_session_exits_0_in_a_non_sim_mode(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_non_sim", tmp_path, signal.SIGTERM)
    assert code == 0


@RED
def test_g3_sigterm_before_any_session_exits_4_in_a_sim_mode(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_sim", tmp_path, signal.SIGTERM)
    assert code == 4


@RED
def test_g3_sigterm_before_the_mode_is_known_exits_4_the_strict_default(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_unknown", tmp_path, signal.SIGTERM)
    assert code == 4


@RED
def test_g3_sigint_before_any_session_exits_130(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_non_sim", tmp_path, signal.SIGINT)
    assert code == 130


@RED
def test_g10_a_loop_that_claims_after_a_pre_session_signal_is_born_stopped(tmp_path: Path) -> None:
    code, _, child = _run("late_claim", tmp_path, signal.SIGTERM)
    assert "BORN_STOPPED" in child.lines
    assert code == 0


# ── Repeats, the watchdog, the lock the handler must never take ───────────────────────────────────────────────────


@RED
def test_g4_a_stop_that_never_finishes_is_ended_by_the_watchdog_with_75_before_the_supervisor_kill(
    tmp_path: Path,
) -> None:
    code, took, child = _run(
        "session_ignores_stop", tmp_path, signal.SIGTERM, timeout_s=15, env={"MAXIM_SUPERVISOR_KILL_S": "6"}
    )
    assert code == 75
    assert took < 6.0
    assert "watchdog" in child.stderr.lower()


@RED
def test_g8_a_second_sigint_forces_130(tmp_path: Path) -> None:
    child = _Child("session_ignores_stop", tmp_path)
    child.wait_ready()
    child.signal(signal.SIGINT)
    time.sleep(0.5)
    child.signal(signal.SIGINT)
    assert child.exit_code(5) == 130


@RED
def test_g8_a_second_sigterm_is_ignored(tmp_path: Path) -> None:
    child = _Child("session_ignores_stop", tmp_path, {"MAXIM_SUPERVISOR_KILL_S": "6"})
    child.wait_ready()
    child.signal(signal.SIGTERM)
    time.sleep(0.5)
    child.signal(signal.SIGTERM)
    assert child.exit_code(15) == 75  # still alive after the repeat: only the watchdog ended it


@RED
def test_g9_a_signal_while_the_main_thread_holds_the_hook_lock_does_not_deadlock(tmp_path: Path) -> None:
    code, took, _ = _run("hook_lock_held", tmp_path, signal.SIGTERM)
    assert code == 0
    assert took < 10


@RED
def test_g16_the_watchdog_fires_while_the_request_is_blocked_on_a_wedged_lock(tmp_path: Path) -> None:
    code, took, _ = _run(
        "hook_lock_wedged", tmp_path, signal.SIGTERM, timeout_s=15, env={"MAXIM_SUPERVISOR_KILL_S": "6"}
    )
    assert code == 75
    assert took < 6.0


@RED
def test_g23_a_fault_in_the_stop_thread_still_ends_the_run_with_75(tmp_path: Path) -> None:
    code, took, _ = _run("stop_thread_raises", tmp_path, signal.SIGTERM, timeout_s=45)
    assert code == 75
    assert took < 40


# ── A human-input read after a session ────────────────────────────────────────────────────────────────────────────


@RED
def test_g7_g18_sigint_at_a_prompt_after_a_session_interrupts_the_read_and_exits_with_the_yield_code(
    tmp_path: Path,
) -> None:
    code, took, _ = _run("prompt_after_session", tmp_path, signal.SIGINT)
    assert code == 0  # not 75: the read woke and the catch saw the stop pending
    assert took < 10


# ── In-process: install/uninstall leaves the caller's process as it found it ─────────────────────────────────────


@RED
def test_g14_cli_main_in_process_restores_the_callers_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli

    before_int, before_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    seen: dict[str, Any] = {}

    def _body(argv: Any = None) -> int:
        seen["int"] = signal.getsignal(signal.SIGINT)
        return 0

    monkeypatch.setattr(cli, "_main_impl", _body)
    assert cli.main([]) == 0
    assert seen["int"] is not before_int, "main ran with the caller's handler, not its own"
    assert signal.getsignal(signal.SIGINT) is before_int
    assert signal.getsignal(signal.SIGTERM) is before_term


# ── The robot run and the post-sim prompt wake on a stop request ──────────────────────────────────────────────────


@RED
def test_g5_the_robot_runs_stop_event_is_set_by_a_process_stop_request() -> None:
    from maxim.embodied_runtime import media_loop
    from maxim.utils.cancellation import clear_stop_request, request_stop
    from maxim.utils.deadline import Deadline

    opener = getattr(media_loop.MediaLoopMixin, "_open_live_stop_event", None)
    assert opener is not None, "the robot run's stop event is not registered for the process stop"
    runtime = type("R", (), {})()
    event = opener(runtime)
    try:
        request = request_stop("test", Deadline.after(5))
        assert event.is_set()
    finally:
        media_loop.MediaLoopMixin._close_live_stop_event(runtime)
        clear_stop_request(request)


@RED
def test_g6_the_post_sim_prompt_returns_when_a_stop_is_requested() -> None:
    import pty

    from maxim.simulation.tty_stdin import read_post_sim_goal_raw
    from maxim.utils.cancellation import clear_stop_request, request_stop
    from maxim.utils.deadline import Deadline

    master, slave = pty.openpty()
    stdin = os.fdopen(slave, "r")
    result: dict[str, Any] = {}
    reader = threading.Thread(
        target=lambda: result.setdefault("goal", read_post_sim_goal_raw(_stdin=stdin, display=None))
    )
    reader.daemon = True
    reader.start()
    request = request_stop("test", Deadline.after(5))
    try:
        reader.join(3)
        assert not reader.is_alive(), "the post-sim prompt kept reading after a stop request"
        assert result["goal"] == ""
    finally:
        clear_stop_request(request)
        os.write(master, b"\n")
        reader.join(2)
        stdin.close()
        os.close(master)


# ── spawn_evidence: exit 0 is evidence only for a known verdict ───────────────────────────────────────────────────


def _report_home(tmp_path: Path, finish_reason: str | None) -> tuple[Path, str]:
    import json

    run_id = "run-1344"
    session = tmp_path / "sim_reports" / "s1"
    session.mkdir(parents=True)
    report: dict[str, Any] = {"provenance": {"harness_run_id": run_id}}
    if finish_reason is not None:
        report["finish_reason"] = finish_reason
    (session / "report.json").write_text(json.dumps(report))
    return tmp_path, run_id


def _spawn_evidence() -> Any:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        import _provenance
    finally:
        sys.path.pop(0)
    return _provenance


@RED
@pytest.mark.parametrize("finish_reason", ["yielded", "cancel", "aborted", "stuck", "error", None, "something_new"])
def test_g12_spawn_evidence_refuses_exit_0_without_a_known_verdict(tmp_path: Path, finish_reason: str | None) -> None:
    prov = _spawn_evidence()
    home, run_id = _report_home(tmp_path, finish_reason)
    try:
        prov.spawn_evidence(home, run_id, set(), returncode=0)
    except prov.SimRunFailed:
        return
    raise AssertionError(f"exit 0 with finish_reason={finish_reason!r} was accepted as evidence")


@pytest.mark.parametrize("finish_reason", ["completed", "max_turns", "failed", "inconclusive", "blocked"])
def test_g19_spawn_evidence_keeps_accepting_verdict_data_at_exit_0(tmp_path: Path, finish_reason: str) -> None:
    prov = _spawn_evidence()
    home, run_id = _report_home(tmp_path, finish_reason)
    _, report = prov.spawn_evidence(home, run_id, set(), returncode=0)
    assert report["finish_reason"] == finish_reason
