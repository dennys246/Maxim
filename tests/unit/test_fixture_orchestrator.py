"""Tests for FixtureDrivenOrchestrator (S1 — simulator_upgrades_plan)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from maxim.simulation.bridge import SimulationBridge
from maxim.simulation.fixture_orchestrator import (
    FixtureDrivenOrchestrator,
    FixtureResult,
)
from maxim.simulation.report import SimulationReport
from maxim.simulation.sinks import ActionRecord


# ── Helpers ─────────────────────────────────────────────���────────────────────


def _write_fixture(tmp_path: Path, data: dict) -> Path:
    """Write a YAML fixture file and return its path."""
    path = tmp_path / "test_fixture.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path


def _fake_aut_responder(bridge: SimulationBridge, count: int = 10) -> threading.Thread:
    """Spawn a thread that auto-responds to bridge percepts with a respond action."""

    def _worker():
        for _ in range(count):
            # Wait for a percept to appear
            deadline = time.time() + 5.0
            while time.time() < deadline:
                p = bridge.percept_source.next_percept()
                if p is not None:
                    # Simulate AUT processing delay
                    time.sleep(0.05)
                    bridge.action_sink.record(
                        ActionRecord(
                            timestamp=time.time(),
                            tool_name="respond",
                            result_success=True,
                            result_output=f"Response to: {(p.cli_input or p.content or 'signal')[:40]}",
                        )
                    )
                    break
                time.sleep(0.02)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# ── Fixture loading ─────────────────────────────────────────────────────────


class TestFixtureLoading:
    def test_load_minimal_fixture(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "minimal",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [],
            },
        )
        orch = FixtureDrivenOrchestrator(fixture)
        assert orch.name == "minimal"

    def test_load_fixture_with_expectations(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "with_expectations",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [
                    {"type": "action_taken", "tool": "respond"},
                ],
            },
        )
        orch = FixtureDrivenOrchestrator(fixture)
        assert orch._definition.expectations[0].type == "action_taken"

    def test_load_p0_fixture(self):
        """The P0 pilot fixture loads without error."""
        p0 = Path("scenarios/substrate/P0_paraphrase_collapse.yaml")
        if not p0.exists():
            pytest.skip("P0 fixture not found")
        orch = FixtureDrivenOrchestrator(p0)
        assert orch.name == "P0_paraphrase_collapse"
        assert len(orch._definition.percepts) == 4
        assert len(orch._definition.expectations) == 2


# ── Run with fake AUT ───────────────────────────────────────────────────────


class TestFixtureRun:
    def test_single_percept_run(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "single",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hi there"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=3.0, settle_s=0.3)
        aut = _fake_aut_responder(bridge, count=1)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=3.0, settle_s=0.3)
        result = orch.run(bridge)
        aut.join(timeout=5)

        assert isinstance(result, FixtureResult)
        assert result.turns_delivered == 1
        assert result.fixture_name == "single"
        assert result.finish_reason == "complete"
        assert len(result.turn_records) == 1
        assert result.turn_records[0]["source"] == "cli"
        assert "hi there" in result.turn_records[0]["text"]

    def test_multi_percept_run(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "multi",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "first"},
                    {"at": 1, "source": "cli", "cli_input": "second"},
                    {"at": 2, "source": "cli", "cli_input": "third"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=3.0, settle_s=0.3)
        aut = _fake_aut_responder(bridge, count=3)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=3.0, settle_s=0.3)
        result = orch.run(bridge)
        aut.join(timeout=10)

        assert result.turns_delivered == 3
        assert len(result.turn_records) == 3
        assert result.turn_records[0]["text"] == "first"
        assert result.turn_records[2]["text"] == "third"

    def test_pain_percept_injection(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "pain_test",
                "percepts": [
                    {
                        "at": 0,
                        "source": "proprioception",
                        "content": "pain_signal",
                        "salience": 0.8,
                        "metadata": {"pain_type": "test_pain"},
                    },
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=2.0, settle_s=0.3)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.3)
        result = orch.run(bridge)

        assert result.turns_delivered == 1
        assert result.turn_records[0]["source"] == "proprioception"
        assert result.turn_records[0]["pain_type"] == "test_pain"

    def test_timeout_on_no_response(self, tmp_path):
        """If AUT doesn't respond, turn records show timed_out."""
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "timeout_test",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello?"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        # No AUT responder — should time out

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge)

        assert result.turns_delivered == 1
        assert result.turn_records[0]["timed_out"] is True


# ── Substrate state collection ───────────────────────────────────────────────


class TestSubstrateCollection:
    def test_hippocampus_snapshot(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "hippo_test",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "remember this"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        mock_hippo = MagicMock()
        mock_hippo.__len__ = MagicMock(return_value=5)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge, hippocampus=mock_hippo)

        assert "hippocampus" in result.substrate_metrics
        assert result.substrate_metrics["hippocampus"]["episode_count"] == 5

    def test_percept_trace_buffer_snapshot(self, tmp_path):
        from maxim.memory.percept_trace_buffer import PerceptTraceBuffer

        fixture = _write_fixture(
            tmp_path,
            {
                "name": "trace_test",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "input"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        ptb = PerceptTraceBuffer()
        ptb.record("agent_0", "percept_1", activation=0.9)
        ptb.record("agent_0", "percept_2", activation=0.5)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge, percept_trace_buffer=ptb)

        assert "percept_trace" in result.substrate_metrics
        assert result.substrate_metrics["percept_trace"]["active_entries"] == 2
        assert result.substrate_metrics["percept_trace"]["current_tick"] == 0

    def test_nac_snapshot(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "nac_test",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "test"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        mock_nac = MagicMock()
        mock_nac._links = {
            "sig1": [MagicMock(event_signature="ev", outcome_signature="out", confidence=0.8, observation_count=3)]
        }

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge, nac=mock_nac)

        assert "nac" in result.substrate_metrics
        assert result.substrate_metrics["nac"]["total_links"] == 1

    def test_no_bio_systems_graceful(self, tmp_path):
        """Runs fine with no bio-system references (all None)."""
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "bare",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge)

        assert result.substrate_metrics == {}


# ── Expectations ───────────────────────���───────────────────��──────────────────


class TestExpectations:
    def test_action_taken_passes(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "exp_action_taken",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [
                    {"type": "action_taken", "tool": "respond"},
                ],
            },
        )
        bridge = SimulationBridge(response_timeout=3.0, settle_s=0.3)
        aut = _fake_aut_responder(bridge, count=1)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=3.0, settle_s=0.3)
        result = orch.run(bridge)
        aut.join(timeout=5)

        assert result.expectations_total == 1
        assert result.expectations_passed == 1
        assert result.expectation_results[0]["pass"] is True

    def test_action_taken_fails(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "exp_action_taken_fail",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [
                    {"type": "action_taken", "tool": "nonexistent_tool"},
                ],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge)

        assert result.expectations_total == 1
        assert result.expectations_passed == 0
        assert result.expectation_results[0]["pass"] is False

    def test_action_blocked_expectation(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "exp_blocked",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "delete files"},
                ],
                "expectations": [
                    {"type": "action_blocked", "tool_pattern": "bash"},
                ],
            },
        )
        bridge = SimulationBridge(response_timeout=2.0, settle_s=0.3)

        # Simulate a blocked action from the AUT
        def _blocked_aut():
            time.sleep(0.1)
            bridge.action_sink.record(
                ActionRecord(
                    timestamp=time.time(),
                    tool_name="bash",
                    blocked=True,
                    block_reason="Fear agent blocked",
                )
            )

        t = threading.Thread(target=_blocked_aut, daemon=True)
        t.start()

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.3)
        result = orch.run(bridge)
        t.join(timeout=5)

        assert result.expectations_passed == 1
        assert result.expectation_results[0]["pass"] is True


# ── Report integration ──────────────────────────────────────────────────────


class TestReportIntegration:
    def test_substrate_metrics_field_exists(self):
        report = SimulationReport()
        assert hasattr(report, "substrate_metrics")
        assert report.substrate_metrics == {}

    def test_to_report_dict(self, tmp_path):
        fixture = _write_fixture(
            tmp_path,
            {
                "name": "report_test",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [],
            },
        )
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)

        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=0.5, settle_s=0.2)
        result = orch.run(bridge)
        report_dict = orch.to_report_dict(result)

        assert isinstance(report_dict, dict)
        assert report_dict["fixture_name"] == "report_test"
        assert "substrate_metrics" in report_dict
        assert "turn_records" in report_dict


# ── campaign_runner dispatch ─────────────────────────────────────────────────


class TestCampaignRunnerDispatch:
    def test_run_fixture_campaign_import(self):
        """run_fixture_campaign is importable from campaign_runner."""
        from maxim.simulation.campaign_runner import run_fixture_campaign

        assert callable(run_fixture_campaign)
