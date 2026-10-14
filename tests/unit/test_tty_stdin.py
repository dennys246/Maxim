"""``simulation/tty_stdin.py``: the TTY read loops moved out of ``start_simulation_mode`` (orchestrator slice 0, SF-4).

The module is on the coverage exclusion list: its loops need a real TTY, and no test here drives one. What
these tests pin instead is the SEAM, structurally:

  - ``orchestrator.py`` no longer defines or imports any raw-mode reader of its own (no
    ``_stdin_reader_raw``, no ``termios``/``tty``), so the excluded code cannot quietly grow back in the
    measured file;
  - the module holds ONLY the TTY loops and their restore state (the exclusion's stated scope, NIT-5);
  - every call site in ``orchestrator.py`` passes exactly the keyword-only parameters the moved function
    declares, and the interactive branch is the one that selects the raw reader.

The one behaviour a non-TTY test CAN run is the raw reader's fallback head (stdin not a TTY -> the caller's
line reader), which is pinned below.

The behavioural tests patch ``orchestrator.stdin_reader_raw``/``read_post_sim_goal_raw`` at the IMPORT site,
unlike agent_loop's precedent of reading the moved function through its module: ``orchestrator.py`` binds
both names at import time (``from maxim.simulation.tty_stdin import ...``), so the orchestrator's own name
is what its call sites resolve and the only patch that intercepts them; patching ``tty_stdin`` would let
the real termios loops run.
"""

from __future__ import annotations

import ast
import inspect
import io
import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from maxim.simulation import tty_stdin

SIMULATION = Path(__file__).resolve().parents[2] / "src" / "maxim" / "simulation"


def _tree(name: str) -> ast.Module:
    return ast.parse((SIMULATION / name).read_text())


def _start_simulation_mode() -> ast.FunctionDef:
    (fn,) = [
        n
        for n in ast.walk(_tree("orchestrator.py"))
        if isinstance(n, ast.FunctionDef) and n.name == "start_simulation_mode"
    ]
    return fn


def _keyword_only(fn) -> set[str]:
    params = inspect.signature(fn).parameters.values()
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params), "rule (d): keyword-only parameters"
    return {p.name for p in params}


def test_orchestrator_no_longer_defines_a_raw_reader() -> None:
    tree = _tree("orchestrator.py")
    defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert not defs & {"_stdin_reader_raw", "stdin_reader_raw", "read_post_sim_goal_raw", "_restore_terminal"}
    imported = {
        alias.name.split(".")[0]
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for alias in (n.names if isinstance(n, ast.Import) else [ast.alias(name=n.module or "")])
    }
    assert not imported & {"termios", "tty"}, "raw-mode terminal code belongs in tty_stdin.py"
    # The non-TTY fallback and the command parser stay measured in orchestrator.py.
    assert {"_stdin_reader_line", "_process_line"} <= defs


def test_the_module_holds_only_the_tty_loops() -> None:
    tree = _tree("tty_stdin.py")
    # Every top-level statement is an import, an assignment, a def, the docstring or the one atexit hook:
    # nothing (an ``if``/``try`` at module level) can hide another def from the name checks below.
    for i, stmt in enumerate(tree.body):
        docstring = i == 0 and isinstance(stmt, ast.Expr) and isinstance(getattr(stmt, "value", None), ast.Constant)
        allowed = isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef))
        hook = ast.unparse(stmt) == "atexit.register(_restore_terminal)"
        assert docstring or allowed or hook, f"line {stmt.lineno}: {ast.unparse(stmt)[:80]}"
        # Every assignment target is a plain name (no ``a, b = ...``/``*a = ...``/``x.y = ...`` hiding a binding
        # from ``assigned`` below).
        targets = (
            stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        )
        assert all(isinstance(t, ast.Name) for t in targets), f"line {stmt.lineno}: {ast.unparse(stmt)[:80]}"
    defs = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    assert defs == {"_restore_terminal", "_sigterm_handler", "stdin_reader_raw", "read_post_sim_goal_raw"}
    assigned = {
        t.id
        for n in tree.body
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
        if isinstance(t, ast.Name)
    }
    assert assigned == {"_termios_restore", "_KEYMAP"}


def test_the_interactive_reader_call_matches_the_signature() -> None:
    (assign,) = [
        n
        for n in ast.walk(_start_simulation_mode())
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["_stdin_target"]
    ]
    choice = assign.value
    assert isinstance(choice, ast.IfExp) and ast.unparse(choice.test) == "_is_interactive"
    assert ast.unparse(choice.orelse) == "_stdin_reader_line"
    call = choice.body
    assert isinstance(call, ast.Call) and ast.unparse(call.func) == "functools.partial"
    assert [ast.unparse(a) for a in call.args] == ["stdin_reader_raw"]
    passed = {k.arg for k in call.keywords}
    assert passed == _keyword_only(tty_stdin.stdin_reader_raw)
    # Each closure variable is passed under its own name (the body reads it unchanged).
    assert all(isinstance(k.value, ast.Name) and k.value.id == k.arg for k in call.keywords)


def test_the_post_sim_call_matches_the_signature() -> None:
    fn = _start_simulation_mode()
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "read_post_sim_goal_raw"]
    assert len(calls) == 1
    (call,) = calls
    assert not call.args
    assert {k.arg for k in call.keywords} == _keyword_only(tty_stdin.read_post_sim_goal_raw)
    assert all(isinstance(k.value, ast.Name) and k.value.id == k.arg for k in call.keywords)
    # Called only on a TTY, and its return value is the goal the run reads.
    guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "_can_raw")
    assert any(c is call for c in ast.walk(guard))
    (assign,) = [n for n in guard.body if isinstance(n, ast.Assign)]
    assert ast.unparse(assign) == "_new_goal = read_post_sim_goal_raw(_stdin=_stdin, display=display)"


def test_a_non_tty_stdin_falls_back_to_the_line_reader(monkeypatch) -> None:
    """The raw reader's fallback head, the one branch reachable without a TTY."""
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    fell_back: list[bool] = []

    def _fail(*_a, **_k):
        raise AssertionError("the raw loop must not run on a non-TTY stdin")

    tty_stdin.stdin_reader_raw(
        stop_event=threading.Event(),
        operator_stop=threading.Event(),
        _sim_prompt_handler=None,
        _process_line=_fail,
        _stdin_reader_line=lambda: fell_back.append(True),
        _stop_by_operator=_fail,
        get_active_display=_fail,
    )
    assert fell_back == [True]
    assert tty_stdin._termios_restore is None


# ── behavioural: the orchestrator's real TTY decisions, without a TTY ─────────────────────────────
# The real ``start_simulation_mode`` runs once (a one-turn pre-campaign arm built from the slice-0
# harness's pieces) in interactive mode with a stdin stub whose ``isatty()`` is True. Both moved TTY
# functions are replaced at the orchestrator's import site by recording fakes, so no termios call is
# ever made; what runs for real is the orchestrator's DECISION to call them, and with what.
# NOT run here: the moved loops themselves (coverage-excluded) and the ``/new`` recursion (a returned
# goal re-enters ``start_simulation_mode`` without the pre-campaign turns, i.e. a full narrator run).


class _TtyStdin(io.TextIOBase):
    """``sys.stdin`` as a TTY: ``isatty()`` True and a real (pipe) ``fileno``. Nothing reads it, because
    both readers that would are fakes."""

    def __init__(self, fd: int) -> None:
        super().__init__()
        self._fd = fd

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def readable(self) -> bool:
        return True


class _Display:
    """An active-display stub: every method is a no-op; the post-sim block's numbers are real ints."""

    log_count = 0
    page_height = 10

    def __getattr__(self, name: str):
        return lambda *a, **k: None


_ONE_TURN = [{"text": "You wake in a quiet room.", "phase": "intro"}]


def _snapshot() -> dict:
    import maxim.runtime.lane_backends as LB
    from maxim.simulation import orchestrator as orch
    from maxim.simulation import sim_logger as SL

    return {
        "interactive_mode": SL.get_interactive_mode(),
        "active_display": SL.get_active_display(),
        "root_handlers": list(logging.getLogger().handlers),
        "stdin": sys.stdin,
        "cwd": os.getcwd(),
        "maxim_env": {k: v for k, v in os.environ.items() if k.startswith("MAXIM_")},
        "orch_names": (orch.stdin_reader_raw, orch.read_post_sim_goal_raw, orch._select_aut_prompt_handler),
        "router": LB.build_primary_router,
    }


def _run_interactive(work: Path) -> dict:
    """One interactive run with both TTY entry points faked. Every global it touches is restored in the
    ``finally`` before this returns (``test_the_interactive_run_restores_its_globals``)."""
    from tests.unit._orch_harness import FakeRouter, _GlobalState, assert_this_checkout

    assert_this_checkout()
    import maxim.runtime.lane_backends as LB
    from maxim.models.language.cancellation import reset_shutdown
    from maxim.simulation import orchestrator as orch
    from maxim.simulation import sim_logger as SL
    from maxim.utils.paths import _reset_caches

    read_fd, write_fd = os.pipe()
    stdin = _TtyStdin(read_fd)
    display = _Display()
    rec: dict = {"reader": [], "post_sim": [], "before": _snapshot()}

    def fake_reader(**kwargs):
        rec["reader"].append({"kwargs": kwargs, "thread": threading.current_thread().name})
        rec["reader_returned"] = kwargs["stop_event"].wait(timeout=120)

    def fake_post_sim(**kwargs):
        rec["post_sim"].append(kwargs)
        return ""

    orig_select = orch._select_aut_prompt_handler

    def spy_select(prompt_handler, stop_event):
        out = orig_select(prompt_handler, stop_event)
        rec["select"] = {"stop_event": stop_event, "sim_handler": out[1]}
        return out

    patches = [
        (orch, "stdin_reader_raw", fake_reader),
        (orch, "read_post_sim_goal_raw", fake_post_sim),
        (orch, "_select_aut_prompt_handler", spy_select),
        (LB, "build_primary_router", lambda *a, **k: (FakeRouter("precampaign"), None)),
    ]
    saved = [(obj, name, getattr(obj, name)) for obj, name, _ in patches]
    state = _GlobalState()
    prev_display = SL.get_active_display()
    try:
        for key in [k for k in os.environ if k.startswith("MAXIM_")]:
            del os.environ[key]
        os.environ.update(MAXIM_SIM_HARD_ABORT="0", MAXIM_DATA_HOME=str(work / "data"))
        _reset_caches()
        os.chdir(work)
        sys.stdin = stdin
        SL.reset_sim_display_state()
        SL.set_interactive_mode(SL.InteractiveMode.ON)
        SL.set_active_display(display)
        reset_shutdown()
        for obj, name, replacement in patches:
            setattr(obj, name, replacement)
        t0 = time.monotonic()
        rec["result"] = orch.start_simulation_mode(
            goal="check one short turn", max_turns=3, sandbox_backend="tmpdir", pre_campaign_turns=_ONE_TURN
        )
        rec["elapsed_s"] = time.monotonic() - t0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(t.name == "sim.stdin" for t in threading.enumerate()):
            time.sleep(0.05)
        rec["stdin_thread_alive"] = any(t.name == "sim.stdin" for t in threading.enumerate())
        rec.update(stdin=stdin, display=display)
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)
        SL.set_active_display(prev_display)
        state.restore()
        os.close(read_fd)
        os.close(write_fd)
    rec["after"] = _snapshot()
    return rec


@pytest.fixture(scope="module")
def interactive_run(tmp_path_factory) -> dict:
    return _run_interactive(tmp_path_factory.mktemp("orch_tty"))


@pytest.mark.timeout(120)
def test_interactive_post_sim_calls_the_raw_goal_reader_on_a_tty(interactive_run) -> None:
    rec = interactive_run
    assert len(rec["post_sim"]) == 1
    (kwargs,) = rec["post_sim"]
    assert set(kwargs) == {"_stdin", "display"}
    assert kwargs["_stdin"] is rec["stdin"]
    assert kwargs["display"] is rec["display"]
    # An empty goal finishes the run normally: one SimulationResult, no /new recursion.
    assert rec["result"] is not None and rec["result"].session_id
    assert rec["result"].finish_reason != "error"


@pytest.mark.timeout(120)
def test_interactive_startup_runs_the_raw_reader_with_the_runs_own_events(interactive_run) -> None:
    from maxim.simulation import orchestrator as orch
    from maxim.simulation import sim_logger as SL

    rec = interactive_run
    assert len(rec["reader"]) == 1
    (call,) = rec["reader"]
    kwargs = call["kwargs"]
    assert call["thread"] == "sim.stdin"
    assert set(kwargs) == _keyword_only(tty_stdin.stdin_reader_raw)
    run_stop = rec["select"]["stop_event"]  # the run's stop_event, as the run hands it to its own builder
    assert kwargs["stop_event"] is run_stop
    assert isinstance(kwargs["operator_stop"], threading.Event) and kwargs["operator_stop"] is not run_stop
    assert kwargs["_sim_prompt_handler"] is not None  # interactive mode: the SimPromptHandler
    assert kwargs["_sim_prompt_handler"] is rec["select"]["sim_handler"]
    assert kwargs["_stop_by_operator"] is orch._stop_by_operator
    assert kwargs["get_active_display"] is SL.get_active_display
    assert kwargs["_process_line"].__name__ == "_process_line"
    assert kwargs["_stdin_reader_line"].__name__ == "_stdin_reader_line"
    # The run set its stop_event at shutdown, the reader returned, and the thread is gone.
    assert rec.get("reader_returned") is True
    assert rec["stdin_thread_alive"] is False


def test_the_interactive_run_restores_its_globals(interactive_run) -> None:
    """Interactive mode, the active display (and its root-logger handler), sys.stdin, the CWD, the MAXIM_*
    env and every patched name are back to what they were before the run."""
    before, after = interactive_run["before"], interactive_run["after"]
    for key in before:
        assert after[key] == before[key], key
