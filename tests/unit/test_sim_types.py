"""Tests for simulation/sim_types.py and simulation/campaign_runner.py.

Extracted from orchestrator.py — tests backward-compat imports and
the extracted functions.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_import_from_sim_types(self):
        from maxim.simulation.sim_types import SimulationResult

        r = SimulationResult(
            goal="test",
            mode="generative",
            turns=5,
            total_actions=10,
            blocked_actions=2,
            duration_s=30.0,
        )
        assert r.goal == "test"
        assert r.finish_reason == "unknown"

    def test_import_from_orchestrator_backward_compat(self):
        from maxim.simulation.orchestrator import SimulationResult

        r = SimulationResult(
            goal="compat",
            mode="generative",
            turns=3,
            total_actions=5,
            blocked_actions=1,
            duration_s=15.0,
        )
        assert r.goal == "compat"

    def test_defaults(self):
        from maxim.simulation.sim_types import SimulationResult

        r = SimulationResult(
            goal="g",
            mode="m",
            turns=0,
            total_actions=0,
            blocked_actions=0,
            duration_s=0.0,
        )
        assert r.session_id == ""
        assert r.campaign_analysis == {}
        assert r.tool_stats == {}
        assert r.actions == []
        assert r.subsystem_snapshot == {}
        assert r.router_stats == {}


class TestSimulationExitContract:
    """Process and harness consumers must agree on run integrity."""

    def test_runtime_aborts_use_the_hard_abort_exit_code(self):
        from maxim.simulation.sim_types import simulation_exit_code

        for finish_reason in (
            "aut_died",
            "llm_wedged",
            "planning_failed",
            "worker_unavailable",
            "aborted",
            "cancel",
            "stuck",
        ):
            assert simulation_exit_code(finish_reason) == 4

    def test_generic_error_uses_generic_failure_exit_code(self):
        from maxim.simulation.sim_types import simulation_exit_code

        assert simulation_exit_code("error") == 1

    def test_semantic_outcomes_remain_valid_experiment_data(self):
        from maxim.simulation.sim_types import is_simulation_run_failure, simulation_exit_code

        for finish_reason in ("completed", "failed", "blocked", "inconclusive", "max_turns"):
            assert not is_simulation_run_failure(finish_reason)
            assert simulation_exit_code(finish_reason) == 0

    def test_failure_classification_is_normalized_and_fail_closed_for_known_failures(self):
        from maxim.simulation.sim_types import is_simulation_run_failure

        assert is_simulation_run_failure("  PLANNING_FAILED  ")
        assert is_simulation_run_failure("error")
        assert not is_simulation_run_failure("unknown")

    def test_cli_maps_structured_simulation_abort_to_process_failure(self):
        from maxim.cli import _simulation_result_exit_code

        assert _simulation_result_exit_code(SimpleNamespace(finish_reason="worker_unavailable")) == 4
        assert _simulation_result_exit_code(SimpleNamespace(finish_reason="completed")) == 0

    def test_research_cli_does_not_hide_underlying_simulation_abort(self):
        from maxim.cli import _research_result_exit_code

        assert (
            _research_result_exit_code(SimpleNamespace(finish_reason="planning_failed", review_verdict="not_reviewed"))
            == 4
        )
        assert _research_result_exit_code(SimpleNamespace(finish_reason="completed", review_verdict="reject")) == 1


class TestBuildResumePrompt:
    """Test resume prompt generation."""

    def test_basic_prompt(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test safety",
            "mode": "generative",
            "turns": 10,
            "total_actions": 25,
            "blocked_actions": 5,
        }
        prompt = build_resume_prompt(report, "continue testing", "generative")
        assert "RESUMING" in prompt
        assert "continue testing" in prompt
        assert "10" in prompt
        assert "25" in prompt

    def test_legacy_persona_key_read_as_mode(self):
        """Pre-1.1 report.json persisted the mode under "persona" — the
        resume prompt must still surface it."""
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test safety",
            "persona": "adversarial",
            "turns": 2,
            "total_actions": 3,
            "blocked_actions": 0,
        }
        prompt = build_resume_prompt(report, "continue", "generative")
        assert "Mode: adversarial" in prompt

    def test_includes_issues(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test",
            "mode": "m",
            "turns": 1,
            "total_actions": 1,
            "blocked_actions": 0,
            "llm_issues_found": ["issue1", "issue2"],
        }
        prompt = build_resume_prompt(report, "g", "m")
        assert "issue1" in prompt

    def test_includes_tool_usage(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test",
            "mode": "m",
            "turns": 1,
            "total_actions": 1,
            "blocked_actions": 0,
            "tool_usage": {"look": 5, "move": 3},
        }
        prompt = build_resume_prompt(report, "g", "m")
        assert "look: 5" in prompt


class TestBuildBasicAnalysis:
    """Test basic analysis helper."""

    def test_none_introspector(self):
        from maxim.simulation.sim_types import build_basic_analysis

        assert build_basic_analysis(None) == {}

    def test_working_introspector(self):
        from maxim.simulation.sim_types import build_basic_analysis

        intr = MagicMock()
        intr.full_analysis.return_value = {"hippocampus": {"count": 5}}
        result = build_basic_analysis(intr)
        assert result == {"hippocampus": {"count": 5}}

    def test_exception_returns_empty(self):
        from maxim.simulation.sim_types import build_basic_analysis

        intr = MagicMock()
        intr.full_analysis.side_effect = RuntimeError("broken")
        result = build_basic_analysis(intr)
        assert result == {}


class TestCampaignRunner:
    """Test campaign runner functions."""

    def test_run_precampaign_returns_analysis(self):
        from maxim.simulation.campaign_runner import run_precampaign_turns

        bridge = MagicMock()
        bridge.send_and_wait.return_value = {
            "actions": [],
            "blocked": [],
            "response": "hello",
            "duration_ms": 100,
        }
        result = run_precampaign_turns(
            turns=[{"text": "test turn", "phase": "intro"}],
            bridge=bridge,
            introspector=None,
        )
        assert "turns" in result
        assert len(result["turns"]) == 1
        assert result["turns"][0]["turn"] == 1

    def test_run_precampaign_handles_failure(self):
        from maxim.simulation.campaign_runner import run_precampaign_turns

        bridge = MagicMock()
        bridge.send_and_wait.side_effect = RuntimeError("bridge error")
        result = run_precampaign_turns(
            turns=[{"text": "fail", "phase": "test"}],
            bridge=bridge,
            introspector=None,
        )
        assert "error" in result["turns"][0]

    def test_run_generative_campaign_display_reads_total_turns(self, monkeypatch):
        """The wrapper's display line accesses result.total_turns (not the
        non-existent .turns_completed). Because the wrapper's broad except
        swallows AttributeError silently, a field-name mismatch would
        return None instead of the result. This test pins the field name
        against future drift.
        """
        from maxim.simulation import campaign_runner
        from maxim.simulation import generative_runner as gr_module
        from maxim.simulation import sim_logger
        from maxim.simulation.generative_runner import GenerativeCampaignResult

        fake_result = GenerativeCampaignResult(
            goal="test",
            arc_name="memory_recall",
            total_turns=7,
        )

        def _fake_run(**kwargs):
            return fake_result

        captured: list[list[str]] = []

        def _capture_summary(lines):
            captured.append(list(lines))

        monkeypatch.setattr(gr_module, "run_generative_campaign", _fake_run)
        monkeypatch.setattr(sim_logger, "display_summary", _capture_summary)
        monkeypatch.setattr(sim_logger, "display_status", lambda *a, **kw: None)

        result = campaign_runner.run_generative_campaign(
            goal="test",
            bridge=MagicMock(),
            llm_router=MagicMock(),
            arc_yaml=None,
            max_turns=3,
            tool_registry=MagicMock(),
            session_dir_base="/tmp/test_session",
        )

        # If the field name drifts, the broad except swallows AttributeError
        # and returns None — assert we got the actual result back.
        assert result is fake_result
        assert captured, "display_summary should have been called"
        assert any("7 turns" in line for line in captured[-1])


class TestImportPaths:
    """Verify all original import paths still work."""

    def test_start_simulation_mode_import(self):
        from maxim.simulation.orchestrator import start_simulation_mode

        assert callable(start_simulation_mode)

    def test_simulation_result_from_orchestrator(self):
        from maxim.simulation.orchestrator import SimulationResult

        assert SimulationResult is not None

    def test_private_helpers_from_orchestrator(self):
        from maxim.simulation.orchestrator import _load_resume_context

        assert callable(_load_resume_context)

    def test_private_helpers_build_resume(self):
        from maxim.simulation.orchestrator import _build_resume_prompt

        assert callable(_build_resume_prompt)
