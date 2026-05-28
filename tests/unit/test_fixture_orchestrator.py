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


# ── Fix B: substrate-aware scene-load pre-trigger ───────────────────────────


class TestFixBSubstratePretrigger:
    """Fix B (W2 to fixture scene-load) — exp 32 Bug B fold.

    Pins the substrate-aware pre-trigger contract on FixtureDrivenOrchestrator.run():
    fires when ALL FOUR (nac + imagination_trigger + llm_router + goal) are non-None.
    See docs/plans/imagination_substrate_signals.md + 33_wire_a_post_fix_a.md.

    Pre-merge review folds (2026-05-28):
    - BLOCK 2 (observability): every gate emits a ``sim_log("SEM_TRACE", ...)``
      event so Roy's JSONL post-hoc analyzers can distinguish "fired with N
      entities" from "skipped (kill-switch)" from "skipped (empty biases)" etc.
      ``FixtureResult.pretrigger_entities`` surfaces the materialized entity
      names for cross-arm scene-divergence auditing.
    - Exec NIT-1 (description-suffix coverage): the fires-with-biases test
      asserts the fixture description lands in the augmented manifest goal.
    - Exec NIT-2 (monkeypatch): tests use ``monkeypatch.setattr`` instead of
      hand-rolled try/finally module patching.
    """

    def _basic_fixture(self, tmp_path):
        return _write_fixture(
            tmp_path,
            {
                "name": "fix_b_probe",
                "description": "Fix B pre-trigger smoke",
                "percepts": [
                    {"at": 0, "source": "cli", "cli_input": "hello"},
                ],
                "expectations": [],
            },
        )

    def _ready_bridge(self):
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        # Auto-responder so .run() doesn't time out on the percept loop.
        _fake_aut_responder(bridge, count=1)
        return bridge

    def _patch_narrator(self, monkeypatch, return_value: str = "Generated manifest text") -> MagicMock:
        """Patch ``generate_scene_manifest`` for the duration of a test."""
        import maxim.simulation.narrator as narrator_mod

        mock = MagicMock(return_value=return_value)
        monkeypatch.setattr(narrator_mod, "generate_scene_manifest", mock)
        return mock

    def _patch_sim_log(self, monkeypatch) -> MagicMock:
        """Patch ``sim_log`` so emission assertions don't depend on the
        sim_logger global state. The pre-trigger lazy-imports sim_log from
        ``maxim.simulation.sim_logger`` inside the method body, so patching
        the module attribute is sufficient."""
        import maxim.simulation.sim_logger as sim_logger_mod

        mock = MagicMock()
        monkeypatch.setattr(sim_logger_mod, "sim_log", mock)
        return mock

    def test_pretrigger_no_op_when_all_fix_b_params_none(self, tmp_path):
        """S1 contract preserved: no pre-trigger fires when Fix B params absent."""
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        result = orch.run(bridge)  # No Fix B kwargs.
        assert result.turns_delivered >= 0  # Run completes normally.
        # FixtureResult surfaces an empty entity tuple when the pre-trigger
        # never fires — Roy analyzers read this to distinguish "Fix B didn't
        # run" from "Fix B ran with zero entities".
        assert result.pretrigger_entities == ()

    def test_pretrigger_fires_with_all_fix_b_params_and_nac_biases(self, tmp_path, monkeypatch):
        """All four params + non-empty NAc biases → manifest LLM call + process_manifest.

        Also verifies:
        - description-suffix branch of the goal-augmentation logic (Exec NIT-1)
        - sim_log SEM_TRACE emission with entity count (BLOCK 2)
        - FixtureResult.pretrigger_entities surfaces the materialized list (BLOCK 3)
        """
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        nac = MagicMock()
        nac.get_agent_tool_biases.return_value = [("tool:sense_food_source", 0.85)]
        memory_hub = MagicMock()
        memory_hub.agent_id = "sim_aut"
        imagination_trigger = MagicMock()
        imagination_trigger.process_manifest.return_value = ["food_source", "berry_patch"]
        llm_router = MagicMock()

        narrator_mock = self._patch_narrator(monkeypatch, return_value="Generated manifest text")
        sim_log_mock = self._patch_sim_log(monkeypatch)

        result = orch.run(
            bridge,
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal="roy:test:arm_a",
        )

        # NAc bias lookup uses canonical agent_id from memory_hub.
        nac.get_agent_tool_biases.assert_called_once()
        assert nac.get_agent_tool_biases.call_args.kwargs["agent_id"] == "sim_aut"

        # Manifest LLM called with the augmented goal — includes BOTH the
        # fixture name and the fixture description (Exec NIT-1 coverage).
        narrator_mock.assert_called_once()
        call_goal = narrator_mock.call_args.args[1]
        assert "fix_b_probe" in call_goal, call_goal
        assert "roy:test:arm_a" in call_goal, call_goal
        assert "Fix B pre-trigger smoke" in call_goal, call_goal

        # process_manifest invoked with the manifest text.
        imagination_trigger.process_manifest.assert_called_once()
        assert imagination_trigger.process_manifest.call_args.args[0] == "Generated manifest text"
        assert imagination_trigger.process_manifest.call_args.kwargs["scene_id"] == "fixture_pretrigger"

        # Materialized entities land on FixtureResult for Roy auditing (BLOCK 3).
        assert result.pretrigger_entities == ("food_source", "berry_patch")

        # SEM_TRACE emissions surface to JSONL analyzers (BLOCK 2).
        sem_trace_msgs = [
            call.args[1] for call in sim_log_mock.call_args_list if call.args and call.args[0] == "SEM_TRACE"
        ]
        # Two events: "generating manifest" + "N entities resolved".
        assert any("generating manifest" in m for m in sem_trace_msgs), sem_trace_msgs
        assert any("2 entities resolved" in m for m in sem_trace_msgs), sem_trace_msgs

    def test_pretrigger_skipped_when_nac_biases_empty(self, tmp_path, monkeypatch):
        """Empty NAc biases → no manifest LLM call + SEM_TRACE 'skipped' event."""
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        nac = MagicMock()
        nac.get_agent_tool_biases.return_value = []
        memory_hub = MagicMock()
        memory_hub.agent_id = "sim_aut"
        imagination_trigger = MagicMock()
        llm_router = MagicMock()

        narrator_mock = self._patch_narrator(monkeypatch, return_value="should-not-fire")
        sim_log_mock = self._patch_sim_log(monkeypatch)

        result = orch.run(
            bridge,
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal="roy:test:arm_a",
        )
        narrator_mock.assert_not_called()
        imagination_trigger.process_manifest.assert_not_called()
        assert result.pretrigger_entities == ()

        # SEM_TRACE "skipped (NAc has no biases)" event surfaces.
        sem_trace_msgs = [
            call.args[1] for call in sim_log_mock.call_args_list if call.args and call.args[0] == "SEM_TRACE"
        ]
        assert any("no biases" in m for m in sem_trace_msgs), sem_trace_msgs

    def test_pretrigger_skipped_when_kill_switch_set(self, tmp_path, monkeypatch):
        """MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL=1 → short-circuit + kill-switch SEM_TRACE."""
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)
        monkeypatch.setenv("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL", "1")

        nac = MagicMock()
        nac.get_agent_tool_biases.return_value = [("tool:sense_food_source", 0.85)]
        memory_hub = MagicMock()
        memory_hub.agent_id = "sim_aut"
        imagination_trigger = MagicMock()
        llm_router = MagicMock()

        narrator_mock = self._patch_narrator(monkeypatch, return_value="should-not-fire")
        sim_log_mock = self._patch_sim_log(monkeypatch)

        result = orch.run(
            bridge,
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal="roy:test:arm_a",
        )
        nac.get_agent_tool_biases.assert_not_called()
        narrator_mock.assert_not_called()
        imagination_trigger.process_manifest.assert_not_called()
        assert result.pretrigger_entities == ()

        # Kill-switch SEM_TRACE event distinguishes "disabled by operator"
        # from "skipped on empty biases" in Roy's ablation arms.
        sem_trace_msgs = [
            call.args[1] for call in sim_log_mock.call_args_list if call.args and call.args[0] == "SEM_TRACE"
        ]
        assert any("MAXIM_DISABLE_IMAGINATION_SUBSTRATE_SIGNAL" in m for m in sem_trace_msgs), sem_trace_msgs

    def test_pretrigger_skipped_when_goal_none(self, tmp_path, monkeypatch):
        """Missing goal → pre-trigger is a no-op (per the four-param contract)."""
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        nac = MagicMock()
        nac.get_agent_tool_biases.return_value = [("tool:sense_food_source", 0.85)]
        imagination_trigger = MagicMock()
        llm_router = MagicMock()

        narrator_mock = self._patch_narrator(monkeypatch, return_value="should-not-fire")

        result = orch.run(
            bridge,
            nac=nac,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal=None,
        )
        nac.get_agent_tool_biases.assert_not_called()
        narrator_mock.assert_not_called()
        assert result.pretrigger_entities == ()

    def test_pretrigger_fail_soft_does_not_abort_fixture_run(self, tmp_path, monkeypatch):
        """Pre-trigger exception is logged + swallowed; the percept loop still runs."""
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        nac = MagicMock()
        nac.get_agent_tool_biases.side_effect = RuntimeError("boom")
        memory_hub = MagicMock()
        memory_hub.agent_id = "sim_aut"
        imagination_trigger = MagicMock()
        llm_router = MagicMock()

        sim_log_mock = self._patch_sim_log(monkeypatch)

        # Should NOT raise — pre-trigger errors are fail-soft.
        result = orch.run(
            bridge,
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal="roy:test:arm_a",
        )
        assert isinstance(result, FixtureResult)
        # Even on failure, FixtureResult surfaces an empty entity tuple.
        assert result.pretrigger_entities == ()
        # SEM_TRACE 'failed' event surfaces the error class + message.
        sem_trace_msgs = [
            call.args[1] for call in sim_log_mock.call_args_list if call.args and call.args[0] == "SEM_TRACE"
        ]
        assert any("failed" in m and "RuntimeError" in m for m in sem_trace_msgs), sem_trace_msgs

    def test_pretrigger_skipped_when_invalid_agent_id(self, tmp_path, monkeypatch):
        """ValueError from get_agent_tool_biases → logged + skipped (defensive, should
        not fire post-Fix-A since AgentConfig.agent_id is required non-empty).
        """
        fixture = self._basic_fixture(tmp_path)
        bridge = self._ready_bridge()
        orch = FixtureDrivenOrchestrator(fixture, turn_timeout=2.0, settle_s=0.2)

        nac = MagicMock()
        nac.get_agent_tool_biases.side_effect = ValueError("non-empty agent_id required")
        memory_hub = MagicMock()
        memory_hub.agent_id = ""  # empty, would trigger ValueError on real NAc
        imagination_trigger = MagicMock()
        llm_router = MagicMock()

        narrator_mock = self._patch_narrator(monkeypatch, return_value="should-not-fire")
        sim_log_mock = self._patch_sim_log(monkeypatch)

        result = orch.run(
            bridge,
            nac=nac,
            memory_hub=memory_hub,
            imagination_trigger=imagination_trigger,
            llm_router=llm_router,
            goal="roy:test:arm_a",
        )
        narrator_mock.assert_not_called()
        assert result.pretrigger_entities == ()
        # SEM_TRACE 'invalid agent_id' event surfaces.
        sem_trace_msgs = [
            call.args[1] for call in sim_log_mock.call_args_list if call.args and call.args[0] == "SEM_TRACE"
        ]
        assert any("invalid agent_id" in m for m in sem_trace_msgs), sem_trace_msgs
