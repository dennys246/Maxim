"""#1002 -- a finished run reports ``completed``; only a human stop reports ``cancel``.

``stop_event`` is set when a runner finishes and again at shutdown, so resolving the finish reason from it
read every run without an explicit ``finish_context`` as ``cancel`` -- 27 of the 37 committed ``report.json``
files under ``docs/experiments/data/`` (the other 10 carry ``max_turns`` or ``planning_failed``), complete
cradle runs included. The operator's stop is now its own event, and the resolution is a
module-level function that does not take ``stop_event`` at all.
"""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path

import pytest

import maxim
from maxim.simulation.orchestrator import _resolve_finish_reason, _stop_by_operator

ORCHESTRATOR = Path(maxim.__file__).resolve().parent / "simulation" / "orchestrator.py"


def _event(is_set: bool) -> threading.Event:
    event = threading.Event()
    if is_set:
        event.set()
    return event


@pytest.mark.parametrize(
    "orch_error, finish_context, operator, expected",
    [
        (RuntimeError("boom"), {"status": "completed"}, True, "error"),
        (None, {"status": "planning_failed"}, True, "planning_failed"),
        (None, {"status": "max_turns"}, False, "max_turns"),
        (None, None, True, "cancel"),
        (None, {"status": ""}, True, "cancel"),
        (None, None, False, "completed"),  # a runner finished: this used to read "cancel"
        (None, {}, False, "completed"),
    ],
)
def test_the_finish_reason_follows_its_priority(orch_error, finish_context, operator, expected) -> None:
    assert _resolve_finish_reason(orch_error, finish_context, _event(operator)) == expected


def test_the_resolution_cannot_see_the_shared_stop_event() -> None:
    """``stop_event`` is set on every normal ending; it must not be an input."""
    assert "stop_event" not in inspect.signature(_resolve_finish_reason).parameters


def test_a_human_stop_sets_both_events() -> None:
    stop, operator = threading.Event(), threading.Event()
    _stop_by_operator(stop, operator)
    assert stop.is_set() and operator.is_set()


# ── the orchestrator's wiring (no test runs a sim to its end) ────────────


def _start_simulation_mode() -> ast.FunctionDef:
    tree = ast.parse(ORCHESTRATOR.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "start_simulation_mode")


def _is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def test_the_run_resolves_its_finish_reason_from_the_operator_stop() -> None:
    (call,) = [n for n in ast.walk(_start_simulation_mode()) if _is_call(n, "_resolve_finish_reason")]
    assert [ast.unparse(a) for a in call.args] == ["orch_error", "llm_finish", "operator_stop"]


def _calls(fn: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(fn) if _is_call(n, name)]


def test_only_a_human_stop_marks_the_operator_stop() -> None:
    """The operator stop is set only through the helper, and the helper is called only from the reader's
    /cancel, Ctrl+C and interrupt paths and the top-level KeyboardInterrupt handler -- never where a
    runner finishes."""
    fn = _start_simulation_mode()
    direct = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and ast.unparse(n.func.value) == "operator_stop"
    ]
    assert direct == []
    helper_calls = _calls(fn, "_stop_by_operator")
    handlers = [
        h
        for h in ast.walk(fn)
        if isinstance(h, ast.ExceptHandler) and h.type is not None and "KeyboardInterrupt" in ast.unparse(h.type)
    ]
    in_interrupt_handlers = [c for c in helper_calls if any(c in list(ast.walk(h)) for h in handlers)]
    cancel_branch = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and "/cancel" in ast.unparse(n.test))
    in_cancel = [c for c in helper_calls if c in list(ast.walk(cancel_branch))]
    ctrl_c = next(n for n in ast.walk(fn) if isinstance(n, ast.If) and ast.unparse(n.test) == "ch == '\\x03'")
    in_ctrl_c = [c for c in helper_calls if c in list(ast.walk(ctrl_c))]
    assert len(in_cancel) == 1 and len(in_ctrl_c) == 1
    assert len(helper_calls) == len(in_interrupt_handlers) + 2
    nested = [n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn]
    top_level = [h for h in handlers if not any(h in list(ast.walk(d)) for d in nested)]
    assert top_level, "the run's own KeyboardInterrupt handler is gone: re-pin"
    for handler in top_level:  # the run's own Ctrl+C, not a reader thread's
        assert any(c in list(ast.walk(handler)) for c in helper_calls)


# ── each runner's own ending, read from the REAL producers ───────────────
# (Round 2 of review: a version fed hand-built dicts passed while every real DM ending read "completed"
# and every real fixture run read "error". These inputs come from the producers themselves.)


def _generative(finish_reason: str):
    from maxim.simulation.generative_runner import GenerativeCampaignResult

    return GenerativeCampaignResult(goal="g", arc_name="a", total_turns=1, finish_reason=finish_reason)


def _dm_rollup(finish_reason: str) -> dict:
    from types import SimpleNamespace

    from maxim.simulation.dm_runtime import CampaignState, DMRuntime

    runtime = SimpleNamespace(
        _campaign=SimpleNamespace(name="c", goal="g", seed=1),
        _state=CampaignState(finish_reason=finish_reason),
        _scene=None,
    )
    return DMRuntime.get_rollup(runtime)


def _fixture_report() -> dict:
    from maxim.simulation.fixture_orchestrator import FixtureDrivenOrchestrator, FixtureResult

    return FixtureDrivenOrchestrator.to_report_dict(None, FixtureResult())


def _precampaign(tmp_path, monkeypatch, *, fail: bool) -> dict:
    from types import SimpleNamespace

    from maxim.simulation import campaign_runner

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(campaign_runner.time, "sleep", lambda s: None)

    def send(text, **kwargs):
        if fail:
            raise ConnectionError("the AUT is gone")
        return {"actions": [], "blocked": [], "response": "ok"}

    return campaign_runner.run_precampaign_turns(
        turns=[{"text": "hello"}], bridge=SimpleNamespace(send_and_wait=send), introspector=None
    )


@pytest.mark.parametrize(
    "runner, make, expected",
    [
        ("generative_runner", lambda: None, "error"),  # it raised and returned None
        ("generative_runner", lambda: _generative("completed"), None),
        ("generative_runner", lambda: _generative("max_turns"), "max_turns"),
        ("generative_runner", lambda: _generative("error"), "error"),  # the narrator failed
        ("dm_runner", lambda: _dm_rollup("all_encounters_complete"), None),
        ("dm_runner", lambda: _dm_rollup("campaign_end:vault:escape"), None),
        ("dm_runner", lambda: _dm_rollup("cancel"), "cancel"),  # Ctrl+C inside dm.run()
        ("dm_runner", lambda: _dm_rollup("unknown encounter: vault"), "error"),
        ("dm_runner", lambda: _dm_rollup("encounter_not_in_order:vault"), "error"),
        ("dm_runner", lambda: _dm_rollup(""), "error"),  # fail closed
        ("dm_runner", lambda: {"error": "boom"}, "error"),  # run_dm_campaign caught a failure
        ("fixture_runner", _fixture_report, None),  # a successful fixture run finishes "complete"
        ("fixture_runner", lambda: {"error": "boom"}, "error"),
        ("mystery_runner", lambda: {"turns": []}, "error"),  # an unknown runner fails closed
    ],
)
def test_a_runners_ending_maps_to_a_finish_status(runner, make, expected) -> None:
    from maxim.simulation.orchestrator import _runner_outcome

    ended = _runner_outcome(runner, make())
    assert (ended[0] if ended else None) == expected


@pytest.mark.parametrize("fail, expected", [(False, None), (True, "error")])
def test_a_precampaign_run_with_a_failed_turn_is_an_error(tmp_path, monkeypatch, fail, expected) -> None:
    from maxim.simulation.orchestrator import _runner_outcome

    ended = _runner_outcome("precampaign_runner", _precampaign(tmp_path, monkeypatch, fail=fail))
    assert (ended[0] if ended else None) == expected


def test_a_runner_failure_is_recorded_and_a_guard_is_not_overwritten() -> None:
    from types import SimpleNamespace

    from maxim.simulation.orchestrator import _finish_runner

    bridge, stop = SimpleNamespace(finish_context={}), threading.Event()
    _finish_runner(bridge, stop, "fixture_runner", {"error": "boom"})
    assert stop.is_set()
    assert (bridge.finish_context["status"], bridge.finish_context["initiated_by"]) == ("error", "fixture_runner")
    assert _resolve_finish_reason(None, bridge.finish_context, threading.Event()) == "error"

    finished = SimpleNamespace(finish_context={})
    _finish_runner(finished, threading.Event(), "fixture_runner", _fixture_report())
    assert _resolve_finish_reason(None, finished.finish_context, threading.Event()) == "completed"

    guarded = SimpleNamespace(finish_context={"status": "max_turns", "initiated_by": "max_turns_guard"})
    _finish_runner(guarded, threading.Event(), "generative_runner", None)
    assert guarded.finish_context["status"] == "max_turns"  # the guard's verdict stands

    worker = SimpleNamespace(finish_context={})
    _finish_runner(
        worker, threading.Event(), "dm_runner", _dm_rollup("all_encounters_complete"), errors=[RuntimeError("x")]
    )
    assert worker.finish_context["status"] == "error"  # the worker thread's caught exception wins


def test_every_runner_call_records_its_ending() -> None:
    """Each runner's return value reaches _finish_runner under its own name; none ends with a bare stop."""
    fn = _start_simulation_mode()
    pairs = sorted((ast.literal_eval(c.args[2]), ast.unparse(c.args[3])) for c in _calls(fn, "_finish_runner"))
    assert pairs == [
        ("dm_runner", "dm_rollup"),
        ("dm_runner", "dm_rollup"),
        ("fixture_runner", "fixture_result"),
        ("generative_runner", "gen_result"),
        ("precampaign_runner", "campaign_analysis"),
    ]
    for runner, var in [("_run_gen", "gen_result"), ("_run_pre", "campaign_analysis"), ("_run_fix", "fixture_result")]:
        (assign,) = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and _is_call(n.value, runner)]
        assert ast.unparse(assign.targets[0]) == var
