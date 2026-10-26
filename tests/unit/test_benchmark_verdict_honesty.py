"""Benchmark verdict honesty (#1391, #1392, #1394, #1395): checks that passed or failed without checking anything.

- #1391: ``pipeline_continued`` passed on ANY action once its tag was emitted, never "an action AFTER the tag".
- #1392: the CLI scenario path validated the bio expectations against an empty snapshot, so they always failed.
  Both callers now go through ONE seam, ``validation.py::validate_finished_run`` (D43).
- #1394: scenarios named a tool by its class (``tool: RespondTool``) where ``action_taken`` matches the registered name
  (``respond``), so those expectations could never pass.
- #1395: an unknown threshold operator (``pas:``) counted as applied and checked nothing.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from maxim.simulation.scenario_source import Expectation, load_scenario
from maxim.simulation.sinks import ActionRecord, RecordingSink

_REPO = Path(__file__).resolve().parents[2]


def _sink(*times: float) -> RecordingSink:
    sink = RecordingSink()
    for t in times:
        sink.record(ActionRecord(timestamp=t, tool_name="respond", result_success=True))
    return sink


def _pipeline(tag: str = "interference") -> Expectation:
    return Expectation(type="pipeline_continued", after_tag=tag)


# ── #1391: an action AFTER the tag ─────────────────────────────────────────────


def test_actions_only_before_the_tag_do_not_continue_the_pipeline():
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(10.0, 11.0), tag_times={"interference": 20.0})
    assert result.passed is False


def test_an_action_after_the_tag_continues_the_pipeline():
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(10.0, 25.0), tag_times={"interference": 20.0})
    assert result.passed is True


def test_no_delivery_times_fails_closed():
    """Weak evidence never gates: a run that reported no delivery times cannot show an action came after the tag."""
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(30.0), tag_times=None)
    assert result.passed is False
    assert "order" in result.detail.lower()


def test_the_scenario_source_records_when_each_tag_reached_the_agent(tmp_path):
    from maxim.simulation.scenario_source import ScenarioSource

    path = tmp_path / "s.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: s
            timing: step_based
            percepts:
              - {at: 0, source: user, content: hi, metadata: {scenario_tag: first}}
            """
        )
    )
    source = ScenarioSource(path)
    for _ in range(3):
        source.next_percept()
    assert set(source.tag_times) == {"first"}
    assert source.tag_times["first"] > 0


def test_campaign_delivery_records_carry_a_sent_time():
    from maxim.simulation.experiment import delivered_scenario_tag_times

    turns = [{"metadata": {"scenario_tag": "a"}}, {"metadata": {"scenario_tag": "b"}}]
    records = {"turns": [{"turn": 1, "sent_at": 5.0}, {"turn": 2, "sent_at": 9.0, "error": "x"}]}
    assert delivered_scenario_tag_times(turns, records) == {"a": 5.0}


# ── #1392: one seam builds the inputs from a finished run ──────────────────────


def test_the_seam_builds_the_snapshot_and_tool_stats_from_the_live_run(monkeypatch):
    """#1392: given the live run (not a precomputed snapshot), the seam takes ``Observer.benchmark_snapshot()`` and
    ``executor.tool_usage_stats()``, so no bio expectation is judged against ``{}``."""
    from maxim.memory.hippocampus import Hippocampus
    from maxim.simulation import validation

    class _Executor:
        def tool_usage_stats(self):
            return {"hallucination_rate": 0.0}

    seen = {}
    real = validation.validate_expectations

    def _spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(validation, "validate_expectations", _spy)
    exp = Expectation(type="hallucination_rate_below", params={"max": 0.5})
    (result,) = validation.validate_finished_run(
        [exp], sink=_sink(), tag_times={}, hippocampus=Hippocampus(), executor=_Executor()
    )
    assert result.passed is True, result.detail
    assert "hippocampus_stats" in seen["subsystem_snapshot"]
    assert seen["tool_stats"] == {"hallucination_rate": 0.0}


def test_every_caller_validates_through_the_one_seam():
    """D43: no caller outside validation.py calls ``validate_expectations`` directly (the CLI scenario path did, with
    no snapshot and no tool stats, so every bio expectation failed)."""
    offenders = []
    for path in (_REPO / "src" / "maxim").rglob("*.py"):
        if path.name == "validation.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name == "validate_expectations":
                    offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert offenders == []


# ── #1394: tool names ──────────────────────────────────────────────────────────


def test_a_class_named_tool_is_refused_at_load(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("name: s\nexpectations:\n  - {type: action_taken, tool: RespondTool}\n")
    with pytest.raises(ValueError, match="RespondTool"):
        load_scenario(path)


def test_an_entity_tool_needs_the_explicit_opt_in(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("name: s\nexpectations:\n  - {type: action_taken, tool: sense_food_source, entity_tool: true}\n")
    (exp,) = load_scenario(path).expectations
    assert exp.tool == "sense_food_source"


def test_every_shipped_scenario_names_registered_tools():
    """Every scenario YAML (``scenarios/campaigns/`` holds campaign files, a different format with its own loader)."""
    paths = [p for p in sorted((_REPO / "scenarios").rglob("*.yaml")) if "campaigns" not in p.parts]
    assert paths
    named = []
    for path in paths:
        named += [e.tool for e in load_scenario(path).expectations if e.tool]  # raises on an unregistered tool
    assert "respond" in named and "RespondTool" not in named


# ── #1395: threshold operators ─────────────────────────────────────────────────


@pytest.mark.parametrize("threshold", [{"pas": 1.0}, {}])
def test_an_unknown_or_empty_threshold_is_refused_at_load(tmp_path, threshold):
    import yaml

    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"name": "s", "benchmark": {"scoring": {"recall_x": threshold}}}))
    with pytest.raises(ValueError, match="recall_x"):
        load_scenario(path)


def test_a_synthetic_record_after_the_tag_is_not_an_action():
    """The ``_llm_unavailable`` stall marker (and ``_deliberation_noop``) are not actions: a run whose LLM died after
    the tag did not continue the pipeline."""
    from maxim.simulation.validation import validate_finished_run

    sink = RecordingSink()
    sink.record(ActionRecord(timestamp=30.0, tool_name="_llm_unavailable", result_success=False))
    (result,) = validate_finished_run([_pipeline()], sink=sink, tag_times={"interference": 20.0})
    assert result.passed is False


def test_the_campaign_runner_stamps_each_turn_before_sending_it(monkeypatch):
    """The producer half of #1391: ``sent_at`` is taken before the bridge delivers the turn, so the turn's own
    actions come after it."""
    from maxim.simulation import campaign_runner

    monkeypatch.setattr(campaign_runner.time, "sleep", lambda s: None)

    class _Bridge:
        def __init__(self):
            self.acted_at = []

        def send_and_wait(self, text, **kwargs):
            import time as _t

            self.acted_at.append(_t.time())
            return {"actions": [], "response": "ok"}

    bridge = _Bridge()
    analysis = campaign_runner.run_precampaign_turns(
        turns=[{"text": "a", "metadata": {"scenario_tag": "a"}}, {"text": "b"}], bridge=bridge, introspector=None
    )
    records = analysis["turns"]
    assert [r["turn"] for r in records] == [1, 2]
    assert all(r["sent_at"] <= acted for r, acted in zip(records, bridge.acted_at))


def test_the_cli_scenario_report_judges_the_live_run(tmp_path, monkeypatch):
    """#1392 on the CLI path: ``_report_scenario_validation`` hands the live hub/executor to the seam (so the bio and
    tool checks see the run), and writes the ``--sim-report`` JSON."""
    import json
    from types import SimpleNamespace

    from maxim import cli
    from maxim.memory.hippocampus import Hippocampus

    class _Executor:
        def tool_usage_stats(self):
            return {"hallucination_rate": 0.0}

    class _Source:
        expectations = [Expectation(type="hallucination_rate_below", params={"max": 0.5})]
        definition = SimpleNamespace(name="s")
        tag_times = {}

    report = tmp_path / "r.json"
    args = SimpleNamespace(_sim_hippo=Hippocampus(), sim_report=str(report))
    passed = cli._report_scenario_validation(args, _Source(), _sink(), nac=None, memory_hub=None, executor=_Executor())
    assert passed is True
    assert json.loads(report.read_text())["passed"] is True


@pytest.mark.parametrize(
    ("tool", "entity_tool", "ok"),
    [
        ("respond", False, True),
        ("RespondTool", False, False),
        ("sense_food", True, True),
        ("respond", True, False),
        ("Bad-Name", True, False),
    ],
)
def test_the_shared_tool_check(tool, entity_tool, ok):
    from maxim.simulation.scenario_source import expectation_tool_problem

    assert (expectation_tool_problem(tool, entity_tool=entity_tool) is None) is ok


def test_the_generator_drops_an_unregistered_tool_instead_of_the_scenario():
    from maxim.simulation.simulation_generator import _clean_expectations

    kept = _clean_expectations(
        [{"type": "action_taken", "tool": "RespondTool"}, {"type": "action_taken", "tool": "respond"}]
    )
    assert [e["tool"] for e in kept] == ["respond"]


@pytest.mark.parametrize("op", ["pass", "pass_above", "pass_below"])
def test_every_accepted_operator_is_applied(op):
    """One table: every key ``load_scenario`` accepts is a comparison ``_check_thresholds`` runs (#1395)."""
    from maxim.simulation.benchmark import BenchmarkRunner
    from maxim.simulation.scenario_source import THRESHOLD_OPERATORS

    assert set(THRESHOLD_OPERATORS) == {"pass", "pass_above", "pass_below"}
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._scoring = {"m": {op: 0.5}}
    failing = 0.0 if op != "pass_below" else 1.0
    assert runner._check_thresholds({"m": failing}) is False
