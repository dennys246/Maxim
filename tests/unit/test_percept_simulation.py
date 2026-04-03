"""Tests for the percept simulation framework."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from maxim.agents.bus import Percept
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.types import Decision, Outcome, Perception
from maxim.proprioception.pain import PainSignal, PainType
from maxim.proprioception.pain_bus import PainBus, create_pain_memory_subscriber, route_pain_percept
from maxim.simulation.scenario_source import (
    Expectation,
    ScenarioSource,
    load_scenario,
)
from maxim.simulation.sinks import ActionRecord, RecordingSink
from maxim.simulation.validation import validate_expectations


# ── ScenarioSource tests ─────────────────────────────────────────────────


class TestScenarioSource:
    def test_load_malware_scenario(self):
        path = Path("scenarios/malware_with_pain.yaml")
        if not path.exists():
            pytest.skip("Scenario file not found")
        defn = load_scenario(path)
        assert defn.name == "malware_request_with_pain"
        assert defn.timing == "step_based"
        assert len(defn.percepts) == 3
        assert len(defn.expectations) == 4

    def test_load_long_horizon_scenario(self):
        path = Path("scenarios/long_horizon_coding.yaml")
        if not path.exists():
            pytest.skip("Scenario file not found")
        defn = load_scenario(path)
        assert defn.name == "long_horizon_coding"
        assert len(defn.percepts) == 7
        assert len(defn.expectations) == 2

    def test_step_based_emission(self):
        path = Path("scenarios/malware_with_pain.yaml")
        if not path.exists():
            pytest.skip("Scenario file not found")
        source = ScenarioSource(path)
        assert source.name == "scenario:malware_request_with_pain"
        assert not source.is_exhausted()

        # Step 0: first percept should be available
        p = source.next_percept()
        assert p is not None
        assert p.source == "cli"
        assert "deletes all system files" in (p.cli_input or "")

        # Step 0 still: second percept at step 1 should not be available
        p2 = source.next_percept()
        assert p2 is None

        # Advance to step 1
        source.advance_step()
        p3 = source.next_percept()
        assert p3 is not None
        assert p3.source == "proprioception"
        assert p3.content == "pain_signal"

        # Advance to step 2 — no percept at step 2
        source.advance_step()
        p4 = source.next_percept()
        assert p4 is None

        # Advance to step 3 — followup
        source.advance_step()
        p5 = source.next_percept()
        assert p5 is not None
        assert "Why didn't you" in (p5.cli_input or "")

        # Now exhausted
        source.advance_step()
        assert source.is_exhausted()

    def test_capabilities_derived(self):
        path = Path("scenarios/malware_with_pain.yaml")
        if not path.exists():
            pytest.skip("Scenario file not found")
        source = ScenarioSource(path)
        caps = source.capabilities
        assert "cli" in caps
        assert "proprioception" in caps

    def test_emitted_tags(self):
        path = Path("scenarios/malware_with_pain.yaml")
        if not path.exists():
            pytest.skip("Scenario file not found")
        source = ScenarioSource(path)
        # Emit first percept
        source.next_percept()
        assert "malware_request" in source.emitted_tags

        # Emit second
        source.advance_step()
        source.next_percept()
        assert "pain_during_threat" in source.emitted_tags


# ── RecordingSink tests ──────────────────────────────────────────────────


class TestRecordingSink:
    def test_record_and_retrieve(self):
        sink = RecordingSink()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="ReadFileTool",
                result_success=True,
                result_output="file contents",
            )
        )
        assert len(sink.actions) == 1
        assert sink.actions[0].tool_name == "ReadFileTool"

    def test_find_blocked(self):
        sink = RecordingSink()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="BashTool",
                blocked=True,
                block_reason="SYSTEM_DAMAGE: rm -rf detected",
            )
        )
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="ReadFileTool",
                result_success=True,
            )
        )
        blocked = sink.find_blocked(tool_pattern="Bash", reason_contains="SYSTEM_DAMAGE")
        assert len(blocked) == 1

    def test_find_actions(self):
        sink = RecordingSink()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="RespondTool",
                result_success=True,
                result_output="I cannot do that because it's dangerous",
            )
        )
        matches = sink.find_actions(tool="RespondTool", output_matches="cannot")
        assert len(matches) == 1


# ── Validation tests ─────────────────────────────────────────────────────


class TestValidation:
    def test_action_blocked_passes(self):
        sink = RecordingSink()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="BashTool",
                blocked=True,
                block_reason="SYSTEM_DAMAGE: destructive pattern",
            )
        )
        exp = Expectation(type="action_blocked", tool_pattern="Bash", reason_contains="SYSTEM_DAMAGE")
        results = validate_expectations([exp], sink)
        assert results[0].passed

    def test_action_blocked_fails(self):
        sink = RecordingSink()
        exp = Expectation(type="action_blocked", tool_pattern="Bash", reason_contains="SYSTEM_DAMAGE")
        results = validate_expectations([exp], sink)
        assert not results[0].passed

    def test_action_taken_passes(self):
        sink = RecordingSink()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="RespondTool",
                result_success=True,
                result_output="I refuse to do that",
            )
        )
        exp = Expectation(type="action_taken", tool="RespondTool", output_matches="refuse")
        results = validate_expectations([exp], sink)
        assert results[0].passed

    def test_memory_formed_passes(self):
        hippo = Hippocampus(config=HippocampusConfig())
        hippo.capture(
            perception=Perception(
                observations={"pain_type": "external_signal", "intensity": 0.8},
                salience=0.9,
            ),
            decision=Decision(intent={"goal": "pain_response"}, reasoning="Pain detected"),
            outcome=Outcome(success=False, result={"pain_type": "external_signal"}),
        )
        exp = Expectation(type="memory_formed", memory_contains="pain")
        results = validate_expectations([exp], RecordingSink(), hippocampus=hippo)
        assert results[0].passed

    def test_memory_formed_fails(self):
        hippo = Hippocampus(config=HippocampusConfig())
        exp = Expectation(type="memory_formed", memory_contains="pain")
        results = validate_expectations([exp], RecordingSink(), hippocampus=hippo)
        assert not results[0].passed

    def test_pipeline_continued_passes(self):
        sink = RecordingSink()
        sink.record(ActionRecord(timestamp=time.time(), tool_name="RespondTool", result_success=True))
        exp = Expectation(type="pipeline_continued", after_tag="test_tag")
        results = validate_expectations([exp], sink, emitted_tags={"test_tag"})
        assert results[0].passed


# ── Pain routing integration ─────────────────────────────────────────────


class TestPainRouting:
    def test_route_pain_percept(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(lambda s: received.append(s))

        percept = Percept(
            timestamp=time.time(),
            source="proprioception",
            content="pain_signal",
            metadata={"pain_type": "external_signal", "intensity": 0.8, "joint": "head_pitch"},
        )
        result = route_pain_percept(percept, bus)
        assert result is True
        assert len(received) == 1
        assert received[0].pain_type == PainType.EXTERNAL_SIGNAL
        assert received[0].intensity == 0.8
        assert received[0].context.get("joint") == "head_pitch"

    def test_non_pain_percept_ignored(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(lambda s: received.append(s))

        percept = Percept(timestamp=time.time(), source="cli", cli_input="hello")
        result = route_pain_percept(percept, bus)
        assert result is False
        assert len(received) == 0

    def test_pain_memory_subscriber(self):
        hippo = Hippocampus(config=HippocampusConfig())
        bus = PainBus()
        bus.subscribe(create_pain_memory_subscriber(hippo))

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
            context={"joint": "head_pitch"},
        )
        bus.publish(signal)

        # Memory should be formed
        memories = hippo.search_by_content("pain")
        assert len(memories) >= 1

    def test_low_intensity_pain_no_memory(self):
        hippo = Hippocampus(config=HippocampusConfig())
        bus = PainBus()
        bus.subscribe(create_pain_memory_subscriber(hippo, intensity_threshold=0.5))

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.3,  # Below threshold
            timestamp=time.time(),
        )
        bus.publish(signal)

        memories = hippo.search_by_content("pain")
        assert len(memories) == 0


# ── Hippocampus search_by_content ────────────────────────────────────────


class TestHippocampusSearch:
    def test_search_by_perception(self):
        hippo = Hippocampus(config=HippocampusConfig())
        hippo.capture(
            perception=Perception(observations={"detected": "red_cup"}, salience=0.5),
            outcome=Outcome(success=True),
        )
        results = hippo.search_by_content("red_cup")
        assert len(results) == 1

    def test_search_by_reasoning(self):
        hippo = Hippocampus(config=HippocampusConfig())
        hippo.capture(
            decision=Decision(reasoning="Avoiding the hot stove"),
            outcome=Outcome(success=True),
        )
        results = hippo.search_by_content("hot stove")
        assert len(results) == 1

    def test_search_no_match(self):
        hippo = Hippocampus(config=HippocampusConfig())
        hippo.capture(
            perception=Perception(observations={"detected": "blue_ball"}),
            outcome=Outcome(success=True),
        )
        results = hippo.search_by_content("xyz_nonexistent")
        assert len(results) == 0

    def test_search_empty_query(self):
        hippo = Hippocampus(config=HippocampusConfig())
        results = hippo.search_by_content("")
        assert len(results) == 0
