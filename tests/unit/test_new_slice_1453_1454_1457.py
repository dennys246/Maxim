"""The post-sim ``/new`` slice: #1453 (run settings carry), #1454 (the learning carries), #1457 (``--report-json``
describes the returned run). Owner decisions 2026-10-26, on each issue; one slice, Session A (1.3.5).

Driven through the real ``start_simulation_mode`` by the orchestrator harness's ``new_goal`` arm, except the settings
classification, which reads the signature and the re-entry call: every parameter says whether it carries, so a new one
cannot be dropped by forgetting it.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from maxim.simulation import orchestrator
from tests.unit._orch_harness import GOAL, NEW_GOAL, assert_this_checkout, canonical_json, run_arm
from tests.unit._orchestrator_source import orchestrator_source

_RED = pytest.mark.xfail(strict=True, reason="the /new slice has not landed")


def _new_goal_trace(workdir: Path) -> dict[str, Any]:
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    return json.loads(canonical_json(run_arm("new_goal", workdir)))


def _re_entry_keywords() -> set[str]:
    """The keywords of the one call that re-enters ``start_simulation_mode``, wherever the slice leaves it."""
    calls = [
        node
        for node in ast.walk(ast.parse(orchestrator_source()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "start_simulation_mode"
    ]
    assert len(calls) == 1
    return {kw.arg for kw in calls[0].keywords if kw.arg is not None}


@_RED
def test_every_parameter_is_classified_and_the_re_entry_carries_what_is_a_run_setting_1453() -> None:
    params = set(inspect.signature(orchestrator.start_simulation_mode).parameters) - {"goal"}
    carried, dropped = orchestrator.NEW_RUN_CARRIES, orchestrator.NEW_RUN_DROPS
    assert carried | dropped == params  # a new parameter fails here until it says which it is
    assert not carried & dropped
    assert _re_entry_keywords() - {"goal"} - {"carry_aut_from"} == carried - {"carry_aut_from"}


@_RED
def test_campaign_scoped_inputs_do_not_carry_1453() -> None:
    assert {"pre_campaign_turns", "fixture_path", "dm_campaign", "resume_session", "experiment_log"} <= (
        orchestrator.NEW_RUN_DROPS
    )
    assert {"sandbox_backend", "sandbox_image", "sandbox_network", "aut_model", "aut_mode"} <= (
        orchestrator.NEW_RUN_CARRIES
    )


@_RED
@pytest.mark.timeout(240)
def test_the_re_entered_aut_starts_from_the_first_runs_learning_1454(tmp_path: Path) -> None:
    trace = _new_goal_trace(tmp_path)
    # The saved NAc of the returned run holds the FIRST run's goal: only the carried state can have put it there.
    assert set(trace["aut_nac"]["goal_reward_bias"]) == {GOAL, NEW_GOAL}
    carried = trace["report"]["provenance"]["resume"]
    assert carried["via"] == "/new" and carried["resume_loaded"] is True


@_RED
@pytest.mark.timeout(240)
def test_report_json_describes_the_returned_run_1457(tmp_path: Path) -> None:
    trace = _new_goal_trace(tmp_path)
    assert trace["result"]["goal"] == NEW_GOAL
    assert json.loads((tmp_path / "report_json.json").read_text())["goal"] == NEW_GOAL


def _set_prompt_args(source: str) -> list[ast.expr]:
    return [
        node.args[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "set_prompt"
    ]


@_RED
def test_the_post_sim_prompt_is_one_constant_and_it_promises_the_memory_1454() -> None:
    from maxim.simulation import tty_stdin

    assert "memory carries over" in tty_stdin.POST_SIM_PROMPT
    assert "memory carries over" not in orchestrator_source().replace("POST_SIM_PROMPT", "")
    # Neither reader spells the words: the panel shows the constant, the raw reader the constant plus what is typed.
    assert "Tell me a new simulation" not in "".join(
        ast.dump(a) for a in _set_prompt_args(Path(tty_stdin.__file__).read_text())
    )
