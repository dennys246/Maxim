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
_STRICT = pytest.mark.xfail(strict=True, reason="red gate: benchmark verdict honesty")


def _sink(*times: float) -> RecordingSink:
    sink = RecordingSink()
    for t in times:
        sink.record(ActionRecord(timestamp=t, tool_name="respond", result_success=True))
    return sink


def _pipeline(tag: str = "interference") -> Expectation:
    return Expectation(type="pipeline_continued", after_tag=tag)


# ── #1391: an action AFTER the tag ─────────────────────────────────────────────


@_STRICT
def test_actions_only_before_the_tag_do_not_continue_the_pipeline():
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(10.0, 11.0), tag_times={"interference": 20.0})
    assert result.passed is False


@_STRICT
def test_an_action_after_the_tag_continues_the_pipeline():
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(10.0, 25.0), tag_times={"interference": 20.0})
    assert result.passed is True


@_STRICT
def test_no_delivery_times_fails_closed():
    """Weak evidence never gates: a run that reported no delivery times cannot show an action came after the tag."""
    from maxim.simulation.validation import validate_finished_run

    (result,) = validate_finished_run([_pipeline()], sink=_sink(30.0), tag_times=None)
    assert result.passed is False
    assert "order" in result.detail.lower()


@_STRICT
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


@_STRICT
def test_campaign_delivery_records_carry_a_sent_time():
    from maxim.simulation.experiment import delivered_scenario_tag_times

    turns = [{"metadata": {"scenario_tag": "a"}}, {"metadata": {"scenario_tag": "b"}}]
    records = {"turns": [{"turn": 1, "sent_at": 5.0}, {"turn": 2, "sent_at": 9.0, "error": "x"}]}
    assert delivered_scenario_tag_times(turns, records) == {"a": 5.0}


# ── #1392: one seam builds the inputs from a finished run ──────────────────────


@_STRICT
def test_the_seam_builds_the_snapshot_and_tool_stats_from_the_live_run():
    from maxim.simulation.validation import validate_finished_run

    class _Executor:
        def tool_usage_stats(self):
            return {"hallucination_rate": 0.0}

    class _Hippo:
        def stats(self):
            return {"total_memories": 3}

        def search_by_content(self, *a, **k):
            return []

        def recall(self, *a, **k):
            return []

    exp = Expectation(type="hallucination_rate_below", params={"max": 0.5})
    (result,) = validate_finished_run([exp], sink=_sink(), tag_times={}, hippocampus=_Hippo(), executor=_Executor())
    assert result.passed is True, result.detail


@_STRICT
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


@_STRICT
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


@_STRICT
def test_every_shipped_scenario_names_registered_tools():
    """Every scenario YAML (``scenarios/campaigns/`` holds campaign files, a different format with its own loader)."""
    paths = [p for p in sorted((_REPO / "scenarios").rglob("*.yaml")) if "campaigns" not in p.parts]
    assert paths
    named = []
    for path in paths:
        named += [e.tool for e in load_scenario(path).expectations if e.tool]  # raises on an unregistered tool
    assert "respond" in named and "RespondTool" not in named


# ── #1395: threshold operators ─────────────────────────────────────────────────


@_STRICT
@pytest.mark.parametrize("threshold", [{"pas": 1.0}, {}])
def test_an_unknown_or_empty_threshold_is_refused_at_load(tmp_path, threshold):
    import yaml

    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"name": "s", "benchmark": {"scoring": {"recall_x": threshold}}}))
    with pytest.raises(ValueError, match="recall_x"):
        load_scenario(path)
