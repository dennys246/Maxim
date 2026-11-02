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

_DRIVER = Path(__file__).with_name("_cli_signal_driver.py")
_SRC = str(Path(maxim.__file__).resolve().parents[1])

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


class _Child:
    def __init__(self, scenario: str, tmp_path: Path, env: dict[str, str] | None = None, *, stdin: Any = None) -> None:
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
        child_env.update({"PYTHONPATH": _SRC, "HOME": str(tmp_path), "PYTHONUNBUFFERED": "1"})
        child_env.update(env or {})
        self.proc = subprocess.Popen(
            [sys.executable, str(_DRIVER), scenario],
            stdin=subprocess.PIPE if stdin is None else stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_env,
            text=True,
        )
        self.lines: list[str] = []

    def wait_for(self, marker: str = "READY") -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.append(line.strip())
            if line.strip() == marker:
                return
        self.proc.kill()
        _, err = self.proc.communicate()
        raise AssertionError(f"the child never printed {marker} (exit {self.proc.returncode}): {err[-2000:]}")

    def wait_ready(self) -> None:
        self.wait_for("READY")

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


def test_g2_sigterm_to_a_non_sim_mode_with_a_session_yields_and_exits_0(tmp_path: Path) -> None:
    code, took, child = _run("session_cooperative", tmp_path, signal.SIGTERM)
    assert code == 0
    assert took < 10
    assert "SAVED" in child.lines, "the handler raised through the mode's stop work"


def test_g2b_sigint_to_a_non_sim_mode_with_a_session_also_yields_and_exits_0(tmp_path: Path) -> None:
    code, _, _ = _run("session_cooperative", tmp_path, signal.SIGINT)
    assert code == 0


def test_g1_g11_a_yielded_sim_exits_4_even_when_its_body_returned_0(tmp_path: Path) -> None:
    code, _, _ = _run("sim_cooperative", tmp_path, signal.SIGTERM)
    assert code == 4


def test_g24_a_yielded_sim_branch_that_ends_in_its_own_sys_exit_still_exits_4(tmp_path: Path) -> None:
    code, _, _ = _run("sim_sys_exit", tmp_path, signal.SIGTERM)
    assert code == 4


# ── Before any session ────────────────────────────────────────────────────────────────────────────────────────────


def test_g3_sigterm_before_any_session_exits_0_in_a_non_sim_mode(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_non_sim", tmp_path, signal.SIGTERM)
    assert code == 0


def test_g3_sigterm_before_any_session_exits_4_in_a_sim_mode(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_sim", tmp_path, signal.SIGTERM)
    assert code == 4


def test_g3_sigterm_before_the_mode_is_known_exits_4_the_strict_default(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_unknown", tmp_path, signal.SIGTERM)
    assert code == 4


def test_g3_sigint_before_any_session_exits_130(tmp_path: Path) -> None:
    code, _, _ = _run("pre_session_non_sim", tmp_path, signal.SIGINT)
    assert code == 130


def test_g10_a_loop_that_claims_after_a_pre_session_signal_is_born_stopped(tmp_path: Path) -> None:
    code, _, child = _run("late_claim", tmp_path, signal.SIGTERM)
    assert "BORN_STOPPED" in child.lines
    assert code == 0


# ── Repeats, the watchdog, the lock the handler must never take ───────────────────────────────────────────────────


def test_g4_a_stop_that_never_finishes_is_ended_by_the_watchdog_with_75_before_the_supervisor_kill(
    tmp_path: Path,
) -> None:
    code, took, child = _run(
        "session_ignores_stop", tmp_path, signal.SIGTERM, timeout_s=20, env={"MAXIM_SUPERVISOR_KILL_S": "6"}
    )
    assert code == 75
    assert took < 15  # far under the 31 s default: the arithmetic itself is pinned by g17
    assert "watchdog" in child.stderr.lower()


def test_g8_a_second_sigint_forces_130(tmp_path: Path) -> None:
    child = _Child("session_ignores_stop", tmp_path)
    child.wait_ready()
    child.signal(signal.SIGINT)
    child.wait_for("STOP_SEEN")
    child.signal(signal.SIGINT)
    assert child.exit_code(10) == 130


def test_g8_a_second_sigterm_is_ignored(tmp_path: Path) -> None:
    child = _Child("session_ignores_stop", tmp_path, {"MAXIM_SUPERVISOR_KILL_S": "6"})
    child.wait_ready()
    child.signal(signal.SIGTERM)
    child.wait_for("STOP_SEEN")
    child.signal(signal.SIGTERM)
    assert child.exit_code(20) == 75  # still alive after the repeat: only the watchdog ended it


def test_g9_a_signal_while_the_main_thread_holds_the_hook_lock_does_not_deadlock(tmp_path: Path) -> None:
    code, took, _ = _run("hook_lock_held", tmp_path, signal.SIGTERM)
    assert code == 0
    assert took < 10


def test_g16_the_watchdog_fires_while_the_request_is_blocked_on_a_wedged_lock(tmp_path: Path) -> None:
    code, took, _ = _run(
        "hook_lock_wedged", tmp_path, signal.SIGTERM, timeout_s=20, env={"MAXIM_SUPERVISOR_KILL_S": "6"}
    )
    assert code == 75
    assert took < 15


def test_g23_a_fault_in_the_stop_thread_still_ends_the_run_with_75(tmp_path: Path) -> None:
    code, took, _ = _run("stop_thread_raises", tmp_path, signal.SIGTERM, timeout_s=45)
    assert code == 75
    assert took < 40


# ── A human-input read after a session ────────────────────────────────────────────────────────────────────────────


def test_g7_g18_sigint_at_a_prompt_after_a_session_interrupts_the_read_and_exits_with_the_yield_code(
    tmp_path: Path,
) -> None:
    code, _, _ = _run("prompt_after_session", tmp_path, signal.SIGINT)
    assert code == 0  # not 75: the read woke and the catch saw the stop noted


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_g7_on_a_real_tty_with_readline_a_signal_wakes_input(tmp_path: Path, sig: int) -> None:
    """stdin AND stdout on a pty, so ``input()`` really goes through readline; its own session, because libedit can
    re-send a signal to its process group."""
    import pty
    import select

    master, slave = pty.openpty()
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
    child_env.update({"PYTHONPATH": _SRC, "HOME": str(tmp_path), "PYTHONUNBUFFERED": "1"})
    proc = subprocess.Popen(
        [sys.executable, str(_DRIVER), "prompt_after_session"],
        stdin=slave,
        stdout=slave,
        stderr=subprocess.PIPE,
        env=child_env,
        start_new_session=True,
    )
    os.close(slave)
    try:
        seen = b""
        end = time.monotonic() + 30
        while b"READY" not in seen and time.monotonic() < end:
            if select.select([master], [], [], 1.0)[0]:
                seen += os.read(master, 1024)
        assert b"READY" in seen, proc.stderr.read()[-2000:] if proc.poll() is not None else "no READY"
        time.sleep(0.5)  # into the blocking read
        proc.send_signal(sig)
        assert proc.wait(10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        proc.stderr.close()
        os.close(master)


# ── In-process: install/uninstall leaves the caller's process as it found it ─────────────────────────────────────


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


def test_g5_live_itself_uses_the_registered_event_and_releases_it() -> None:
    import ast

    from maxim.embodied_runtime import media_loop

    tree = ast.parse(Path(media_loop.__file__).read_text())
    live = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "live")
    calls = {n.func.attr for n in ast.walk(live) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert {"_open_live_stop_event", "_release_live_run_handles"} <= calls


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


def test_the_spawn_success_set_equals_its_sources() -> None:
    """The literal in ``_provenance`` is exactly its sources: verdict statuses that are not failures, the turn cap,
    and the evidence records' accepted outcomes (a drift in any of them fails here, not in a harness)."""
    from maxim.simulation.sim_types import SIMULATION_FAILURE_FINISH_REASONS
    from maxim.simulation.tools import FinishSimulationTool

    prov = _spawn_evidence()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    try:
        import _evidence_records
    finally:
        sys.path.pop(0)
    derived = (
        (frozenset(FinishSimulationTool.VALID_STATUSES) - SIMULATION_FAILURE_FINISH_REASONS)
        | {"max_turns"}
        | _evidence_records.FINISH_OK
    )
    assert prov.SPAWN_SUCCESS_FINISH_REASONS == derived
    assert prov.SPAWN_SUCCESS_FINISH_PREFIX == _evidence_records.FINISH_OK_PREFIX
    assert not prov.SPAWN_SUCCESS_FINISH_REASONS & SIMULATION_FAILURE_FINISH_REASONS


# ── The stop thread's seams ───────────────────────────────────────────────────────────────────────────────────────


def test_g21_arm_wakes_the_idle_stop_thread_and_fires(tmp_path: Path) -> None:
    child = _Child("armed_then_hangs", tmp_path)
    child.wait_ready()
    assert child.exit_code(10) == 75


def test_disarm_cancels_an_arm(tmp_path: Path) -> None:
    child = _Child("armed_then_disarmed", tmp_path)
    child.wait_ready()
    assert child.exit_code(10) == 0


def test_g27_a_byte_with_no_signal_behind_it_is_not_a_signal(tmp_path: Path) -> None:
    code, _, child = _run("stale_byte_then_signal", tmp_path, signal.SIGTERM)
    assert "NO_STOP" in child.lines  # the stale byte did not stop the run
    assert code == 0  # nor did it make the real signal read as a repeat


def test_g25_an_idle_fault_in_the_stop_thread_leaves_a_later_signal_working(tmp_path: Path) -> None:
    code, _, _ = _run("idle_fault_then_signal", tmp_path, signal.SIGTERM)
    assert code == 0


def test_g17_the_supervisor_budget_is_validated_and_the_arithmetic_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli_signals

    for bad in ("nan", "inf", "-1", "3", "abc"):
        assert cli_signals._validated_kill_s(bad) is None, bad
    assert cli_signals._validated_kill_s(None) is None
    monkeypatch.setattr(cli_signals, "_supervisor_kill_s", None)
    assert cli_signals._library_deadline(100.0) == (130.0, 131.0)
    monkeypatch.setattr(cli_signals, "_supervisor_kill_s", cli_signals._validated_kill_s("10"))
    assert cli_signals._library_deadline(100.0) == (107.0, 108.0)


def test_the_cli_exit_codes_match_the_sim_contract() -> None:
    from maxim import cli_signals
    from maxim.simulation.sim_types import SIMULATION_ABORT_EXIT_CODE

    assert cli_signals.EXIT_SIM_ABORT == SIMULATION_ABORT_EXIT_CODE


def test_g26_a_forked_child_gets_its_own_pipe_and_stop_thread(tmp_path: Path) -> None:
    child = _Child("fork_child_signal", tmp_path)
    child.wait_ready()  # READY comes from the child's body
    time.sleep(0.3)
    pids = subprocess.run(["pgrep", "-P", str(child.proc.pid)], capture_output=True, text=True).stdout.split()
    assert pids, "the forked child is gone"
    os.kill(int(pids[0]), signal.SIGTERM)
    assert child.exit_code(10) == 0


# ── The shipped paths: the real dispatch, the real `maxim serve` ─────────────────────────────────────────────────


def test_g1_the_real_sim_dispatch_declares_the_mode_and_a_sigterm_exits_4(tmp_path: Path) -> None:
    code, _, _ = _run("real_sim_dispatch", tmp_path, signal.SIGTERM, timeout_s=30)
    assert code == 4


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_g13_a_sigterm_to_the_real_maxim_serve_exits_0_not_143(tmp_path: Path) -> None:
    import urllib.error
    import urllib.request

    pytest.importorskip("uvicorn")

    port = _free_port()
    child_env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
    child_env.update({"PYTHONPATH": _SRC, "HOME": str(tmp_path), "PYTHONUNBUFFERED": "1"})
    proc = subprocess.Popen(
        [sys.executable, str(_DRIVER), "real_serve", str(port)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=child_env,
        text=True,
    )
    try:
        end = time.monotonic() + 60
        while time.monotonic() < end:
            assert proc.poll() is None, f"serve exited early: {proc.communicate()[0][-2000:]}"
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1)
                break
            except urllib.error.HTTPError:
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise AssertionError("serve never answered")
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    assert proc.returncode == 0, out[-2000:]


def test_g3_the_real_dispatch_declares_a_flag_run_non_sim(tmp_path: Path) -> None:
    code, _, _ = _run("real_non_sim_dispatch", tmp_path, signal.SIGTERM, timeout_s=30)
    assert code == 0  # 4 would mean the mode stayed unknown


def test_g3_the_real_dispatch_declares_a_subcommand_non_sim(tmp_path: Path) -> None:
    code, _, _ = _run("real_subcommand_dispatch", tmp_path, signal.SIGTERM, timeout_s=30)
    assert code == 0


def test_a_sim_that_refuses_under_this_processs_own_stop_exits_with_the_yield_code(tmp_path: Path) -> None:
    code, _, child = _run("sim_refused_under_own_stop", tmp_path, signal.SIGTERM)
    assert code == 4, child.stderr[-2000:]


# ── The seams a child does not reach ──────────────────────────────────────────────────────────────────────────────


def test_a_prompt_entered_after_a_signal_was_noted_raises_at_once() -> None:
    """The request lands a moment after the signal, on a helper: a prompt entered in that window must not block."""
    from maxim.utils import cancellation

    cancellation.note_signal()
    with pytest.raises(KeyboardInterrupt):
        with cancellation.interruptible():
            raise AssertionError("the read ran")


def test_serve_arms_the_watchdog_at_shutdown_and_a_cleared_request_disarms_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    uvicorn = pytest.importorskip("uvicorn")

    from maxim import cli_signals
    from maxim.console import server
    from maxim.utils.cancellation import clear_stop_request

    async def _done(self: Any, sockets: Any = None) -> None:
        return None

    monkeypatch.setattr(uvicorn.Server, "shutdown", _done)
    instance = server._MaximServer.__new__(server._MaximServer)
    instance._stop_signal = None
    instance._stop_budget_s = 5.0
    with cli_signals.installed():
        asyncio.run(instance.shutdown())
        armed = cli_signals._arm_kill_at
        clear_stop_request()
        disarmed = cli_signals._arm_kill_at
    assert armed is not None and disarmed is None


def test_the_menu_ends_after_a_stop_instead_of_looping_into_a_refused_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli
    from maxim.utils import cancellation

    runs: list[str] = []
    renders: list[int] = []

    def _yielded_sim(action: str) -> None:
        runs.append(action)
        cancellation.note_signal()

    monkeypatch.setattr(cli, "_run_menu_sim", _yielded_sim)
    monkeypatch.setattr(cli, "_llm_backend_available", lambda: True)
    monkeypatch.setattr(cli, "_render_plain_menu", lambda *_a: renders.append(1))
    monkeypatch.setattr("builtins.input", lambda *_a: "1")
    monkeypatch.setitem(sys.modules, "rich.console", None)  # the plain menu: `input` above answers it
    assert cli._bare_maxim_menu() == 0
    assert len(runs) == 1
    assert len(renders) == 1, "the menu was drawn again after the stop"


def test_a_robot_run_that_yielded_never_re_execs_into_a_requested_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import cli
    from maxim.utils import cancellation

    reexecs: list[str] = []
    monkeypatch.setattr(cli, "_runtime_mode_switch_allowed", lambda *_a, **_k: True)
    monkeypatch.setattr(cli, "_reexec_with_mode", lambda _args, mode: reexecs.append(mode))
    monkeypatch.setenv("MAXIM_MODE_SWITCH_DELAY_S", "0")
    runtime = type("R", (), {"requested_mode": "reflection"})()
    args = type("A", (), {"operational_mode": None})()
    cli._after_run_mode_request(args, runtime, "active")
    assert reexecs == ["reflection"], "the positive control: with no stop, an allowed switch re-execs"
    reexecs.clear()
    cancellation.note_signal()
    assert cli._after_run_mode_request(args, runtime, "active") is None
    assert reexecs == []


def test_a_prompt_entered_while_the_request_cannot_land_yet_is_refused_by_the_noted_signal(tmp_path: Path) -> None:
    child = _Child("prompt_while_the_request_cannot_land", tmp_path)
    child.wait_ready()
    child.signal(signal.SIGTERM)
    assert child.exit_code(10) == 0
    assert "PROMPT_REFUSED" in child.lines


def test_a_command_that_handles_its_own_ctrl_c_keeps_its_exit_code(tmp_path: Path) -> None:
    """Server mode's documented "0 stopped by Ctrl+C", tunnel and `peer logs`: nothing to save, the command decides."""
    code, _, _ = _run("pre_session_handles_interrupt", tmp_path, signal.SIGINT)
    assert code == 0


@pytest.mark.parametrize("ask", ["_prompt_confirm", "_prompt_text"])
def test_a_rich_prompt_entered_after_a_stop_returns_its_default_without_reading(
    ask: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rich.prompt

    from maxim.interactive import prompts
    from maxim.utils import cancellation

    def _reads(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the prompt read after a stop")

    monkeypatch.setattr(rich.prompt.Confirm, "ask", _reads)
    monkeypatch.setattr(rich.prompt.Prompt, "ask", _reads)
    cancellation.note_signal()
    handler = prompts.RichPromptHandler.__new__(prompts.RichPromptHandler)
    request = prompts.PromptRequest.__new__(prompts.PromptRequest)
    object.__setattr__(request, "question", "go on?")
    object.__setattr__(request, "default", "no")
    response = getattr(handler, ask)(request, 0.0)
    assert response.value == "no"


def test_a_fault_in_the_stop_thread_still_makes_the_request_so_a_cooperative_run_stops_at_once(tmp_path: Path) -> None:
    code, took, child = _run("stop_thread_raises_cooperative", tmp_path, signal.SIGTERM, timeout_s=15)
    assert code == 0
    assert took < 10, "the run waited out the watchdog: the fallback never made the request"
    assert "SAVED" in child.lines


def test_a_full_stderr_pipe_cannot_hold_the_watchdogs_exit(tmp_path: Path) -> None:
    code, _, _ = _run("stderr_full", tmp_path, signal.SIGTERM, timeout_s=20, env={"MAXIM_SUPERVISOR_KILL_S": "6"})
    assert code == 75


def test_the_watchdog_kills_the_spawned_servers_process_group(tmp_path: Path) -> None:
    child = _Child("kills_the_spawned_server", tmp_path, {"MAXIM_SUPERVISOR_KILL_S": "6"})
    child.wait_ready()
    pid = next(int(line.split()[1]) for line in child.lines if line.startswith("SERVER_PID"))
    child.signal(signal.SIGTERM)
    assert child.exit_code(20) == 75
    for _ in range(50):
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.killpg(pid, signal.SIGKILL)
    raise AssertionError("the spawned server outlived the watchdog")


def test_g3_roy_run_is_a_sim_mode_though_it_is_positional(tmp_path: Path) -> None:
    code, _, _ = _run("real_roy_run", tmp_path, signal.SIGTERM, timeout_s=30)
    assert code == 4


def test_a_real_failure_of_a_non_sim_mode_survives_a_signal() -> None:
    from maxim import cli_signals

    cli_signals._signalled = signal.SIGTERM
    cli_signals._cooperative = True
    cli_signals._mode = "non_sim"
    assert cli_signals.exit_code(3) == 3  # a failed serve startup stays a failure
    assert cli_signals.exit_code(2) == 2
    assert cli_signals.exit_code(130) == 0
    assert cli_signals.exit_code(0) == 0
    cli_signals._mode = "sim"
    assert cli_signals.exit_code(1) == cli_signals.EXIT_SIM_ABORT


def test_foundry_is_a_sim_mode() -> None:
    from maxim import cli, cli_signals

    cli._declare_run_mode(type("A", (), {"foundry": "theme"})())
    assert cli_signals._mode == "sim"


def test_a_subcommand_nobody_declared_stays_unknown_and_strict() -> None:
    from maxim import cli, cli_signals

    cli._declare_subcommand_mode(["frobnicate"])
    assert cli_signals._mode == "unknown"
    cli._declare_subcommand_mode(["doctor"])
    assert cli_signals._mode == "non_sim"


def test_a_menu_sim_that_finished_leaves_sim_mode_so_a_later_ctrl_c_is_not_an_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from maxim import cli, cli_signals

    answers = iter(["1", ""])
    modes: list[str] = []

    def _finished_sim(action: str) -> None:
        cli_signals.entered_sim_mode()

    monkeypatch.setattr(cli, "_run_menu_sim", _finished_sim)
    monkeypatch.setattr(cli, "_llm_backend_available", lambda: True)
    monkeypatch.setattr(cli, "_render_plain_menu", lambda *_a: modes.append(cli_signals._mode))
    monkeypatch.setattr("builtins.input", lambda *_a: next(answers))
    monkeypatch.setitem(sys.modules, "rich.console", None)
    assert cli._bare_maxim_menu() == 0
    assert modes[-1] == "non_sim"


def test_the_plain_prompt_returns_its_default_after_a_stop_without_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.interactive import prompts
    from maxim.utils import cancellation

    def _reads(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the prompt read after a stop")

    monkeypatch.setattr("select.select", _reads)
    cancellation.note_signal()
    handler = prompts.PlainPromptHandler()
    response = handler.prompt(prompts.PromptRequest(prompts.PromptType.FREEFORM, "go on?", default="no", timeout_sec=5))
    assert response.value == "no"


def test_conftest_does_not_assume_python_default_sigint(tmp_path: Path) -> None:
    """A background job inherits SIGINT ignored; the guard must compare with what the session started with."""
    probe = tmp_path / "test_probe.py"
    probe.write_text("def test_nothing() -> None:\n    pass\n")
    script = (
        f"import signal, subprocess, sys; signal.signal(signal.SIGINT, signal.SIG_IGN); "
        f"sys.exit(subprocess.call([sys.executable, '-m', 'pytest', {str(Path(__file__).with_name('test_tty_stdin.py'))!r}, "
        f"'-q', '-p', 'no:cacheprovider', '--no-cov', '-k', 'test_the_module_holds_only_the_tty_loops']))"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env["PYTHONPATH"] = _SRC
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:]
