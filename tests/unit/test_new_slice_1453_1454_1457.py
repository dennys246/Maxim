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
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maxim.simulation import orchestrator, sim_teardown, tty_stdin
from maxim.simulation.report import RESUME_STORES, resume_stamp
from tests.unit._orch_harness import GOAL, NEW_GOAL, assert_this_checkout, canonical_json, run_arm
from tests.unit._orchestrator_source import orchestrator_source


def _new_goal_trace(workdir: Path) -> dict[str, Any]:
    assert_this_checkout()  # a worktree without PYTHONPATH would otherwise test main's code
    return json.loads(canonical_json(run_arm("new_goal", workdir)))


def test_every_parameter_is_classified_1453() -> None:
    params = set(inspect.signature(orchestrator.start_simulation_mode).parameters) - {"goal"}
    carried, dropped = orchestrator.NEW_RUN_CARRIES, orchestrator.NEW_RUN_DROPS
    assert carried | dropped == params  # a new parameter fails here until it says which it is
    assert not carried & dropped


def test_the_re_entry_takes_its_settings_from_the_classification_1453() -> None:
    """The one re-entry passes the carried settings as ``**``, so the set alone decides what the new run gets."""
    calls = [
        node
        for node in ast.walk(ast.parse(orchestrator_source()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "start_simulation_mode"
    ]
    assert len(calls) == 1
    kept = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert kept.pop(None) == "_new_run_settings"
    assert set(kept) == {"goal", "generative", "carry_aut_from"}
    every = dict.fromkeys(inspect.signature(orchestrator.start_simulation_mode).parameters, "value")
    assert set(orchestrator._carried_settings(every)) == orchestrator.NEW_RUN_CARRIES


def test_no_carried_setting_is_reassigned_after_it_is_taken_1453() -> None:
    fn = next(
        node
        for node in ast.walk(ast.parse(Path(orchestrator.__file__).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "start_simulation_mode"
    )
    taken = next(
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_new_run_settings"
    )
    stores = {
        node.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.lineno > taken
    }
    assert not stores & orchestrator.NEW_RUN_CARRIES


def test_campaign_scoped_inputs_do_not_carry_and_run_settings_do_1453() -> None:
    assert {"pre_campaign_turns", "fixture_path", "dm_campaign", "resume_session", "experiment_log"} <= (
        orchestrator.NEW_RUN_DROPS
    )
    assert {
        "sandbox_backend",
        "sandbox_image",
        "sandbox_network",
        "aut_model",
        "aut_mode",
        "persistent_agent",  # dropping it would rebuild a throwaway AUT for the follow-up goal
        "prompt_handler",  # the caller's handler outlives /new too (D40)
    } <= orchestrator.NEW_RUN_CARRIES


@pytest.mark.parametrize(
    ("goal", "generative", "arc_yaml", "expected"),
    [
        ("write me a poem", True, None, False),  # the flag came from the first goal: this goal matches no arc
        ("cradle", False, None, True),  # and the reverse: this goal does
        ("write me a poem", True, "arc.yaml", True),  # a named arc is a run setting
        ("cradle", False, "arc.yaml", False),
    ],
)
def test_generative_is_decided_for_the_new_goal_unless_an_arc_was_named(
    goal: str, generative: bool, arc_yaml: str | None, expected: bool
) -> None:
    assert sim_teardown.generative_for_new_goal(goal, generative=generative, arc_yaml=arc_yaml) is expected


@pytest.mark.timeout(240)
def test_the_re_entered_aut_starts_from_the_first_runs_learning_1454(tmp_path: Path) -> None:
    trace = _new_goal_trace(tmp_path)
    # The saved NAc of the returned run holds the FIRST run's goal: only the carried state can have put it there.
    assert set(trace["aut_nac"]["goal_reward_bias"]) == {GOAL, NEW_GOAL}
    carried = trace["report"]["provenance"]["resume"]
    assert carried["via"] == "/new" and carried["resume_loaded"] is True
    assert carried["stores"] == dict.fromkeys(("hippocampus", "nac", "ec", "atl"), "loaded")
    assert carried["context_loaded"] is False  # a carry, not a resume: the narrator got the kickoff prompt


@pytest.mark.timeout(240)
def test_report_json_describes_the_returned_run_1457(tmp_path: Path) -> None:
    trace = _new_goal_trace(tmp_path)
    assert trace["result"]["goal"] == NEW_GOAL
    assert json.loads((tmp_path / "report_json.json").read_text())["goal"] == NEW_GOAL


def _tail(*, persistent_agent: Any = None) -> SimpleNamespace:
    return SimpleNamespace(aut_memory_hub=None, aut_hippocampus=None, aut_nac=None, persistent_agent=persistent_agent)


def test_new_waits_for_the_session_end_then_saves_1454(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(orchestrator, "_await_aut_session_end_before_new", lambda _hub: events.append("wait"))
    monkeypatch.setattr(orchestrator, "save_session_aut_state", lambda *_a: events.append("save"))
    assert orchestrator._save_for_new(_tail(), SimpleNamespace(session_id="s1"), "reports") == "s1"
    assert events == ["wait", "save"]


def test_an_adopted_agent_has_nothing_to_carry_and_no_snapshot_1454(monkeypatch: pytest.MonkeyPatch) -> None:
    saves: list[Any] = []
    monkeypatch.setattr(orchestrator, "_await_aut_session_end_before_new", lambda _hub: None)
    monkeypatch.setattr(orchestrator, "save_session_aut_state", lambda *a: saves.append(a))
    tail = _tail(persistent_agent=object())
    assert orchestrator._save_for_new(tail, SimpleNamespace(session_id="s1"), "reports") is None
    assert saves == []


def _stamp(**stores: str) -> dict[str, Any]:
    return resume_stamp({"requested": "s1", "state_dir": ".", "via": "/new", "stores": stores})


def test_a_carry_loads_only_when_a_store_loaded_1454() -> None:
    assert _stamp(hippocampus="loaded", nac="absent", ec="absent", atl="absent")["resume_loaded"] is True
    nothing = _stamp(hippocampus="absent", nac="absent", ec="absent", atl="absent")
    assert nothing["resume_loaded"] is False and "resumed_from_session" not in nothing


def test_a_store_this_run_has_and_the_session_lacks_is_missing_and_warns_1454(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("maxim.utils.paths.sim_reports", lambda: tmp_path)
    (tmp_path / "s1").mkdir()
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.orchestrator"):
        record = orchestrator._restore_aut_from_session(
            None,
            persistent_agent=None,
            aut_hippocampus=SimpleNamespace(),
            aut_nac=None,
            aut_memory_hub=None,
            carry_from="s1",
        )
    assert record["stores"]["hippocampus"] == "missing" and record["stores"]["nac"] == "absent"
    assert resume_stamp(record)["resume_loaded"] is False
    assert "did not carry" in caplog.text


def test_a_resume_cannot_also_carry_1454() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        orchestrator._restore_aut_from_session(
            "s1", persistent_agent=None, aut_hippocampus=None, aut_nac=None, aut_memory_hub=None, carry_from="s2"
        )


def test_a_resume_keeps_absent_and_its_context_rule_1454() -> None:
    stamp = resume_stamp({"requested": "s1", "state_dir": ".", "stores": dict.fromkeys(RESUME_STORES, "loaded")})
    assert stamp["resume_loaded"] is False  # no via: a resume still needs its context
    assert "via" not in stamp


class _Display:
    log_count = 0

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls: list[str] = []

    def scroll(self, _n: int) -> None:
        pass

    def set_prompt(self, prompt: str) -> None:
        self.prompts.append(prompt)

    def set_urgent(self, urgent: bool) -> None:
        self.calls.append(f"urgent={urgent}")

    def stop(self) -> None:
        self.calls.append("stop")


def test_the_panel_shows_the_one_prompt_and_the_block_closes_it_1454(monkeypatch: pytest.MonkeyPatch) -> None:
    display = _Display()
    monkeypatch.setattr("maxim.simulation.sim_logger.get_active_display", lambda: display)
    monkeypatch.setattr("maxim.simulation.report.print_report", lambda *_a, **_k: None)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(orchestrator, "read_post_sim_goal_raw", lambda **_kw: "/CANCEL")
    goal = orchestrator._read_new_goal(SimpleNamespace(session_id="s1"), "reports")
    assert goal == ""  # /cancel finishes, in any case
    assert display.prompts == [orchestrator.POST_SIM_PROMPT]
    assert display.calls == ["urgent=True", "urgent=False", "stop"]


def test_the_raw_reader_shows_the_prompt_it_is_given_1454() -> None:
    """The raw reader is uncovered (a TTY loop), so its one prompt line is pinned by its text."""
    tree = ast.parse(Path(tty_stdin.__file__).read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "read_post_sim_goal_raw")
    args = [
        ast.unparse(node.args[0])
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "set_prompt"
    ]
    assert args == ["f'{prompt}{typed}'"]


def test_report_json_is_written_once_and_its_request_is_consumed_1457(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    written: list[str] = []
    monkeypatch.setattr("maxim.simulation.report.emit_report_json", lambda _r, path: written.append(path))
    monkeypatch.setenv("MAXIM_REPORT_JSON", str(tmp_path / "r.json"))
    sim_teardown.emit_requested_report(object())
    sim_teardown.emit_requested_report(object())  # a back-to-back sim: its report must not land on the first's file
    assert written == [str(tmp_path / "r.json")]
    assert "MAXIM_REPORT_JSON" not in os.environ
