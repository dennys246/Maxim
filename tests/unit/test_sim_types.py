"""Tests for simulation/sim_types.py and simulation/campaign_runner.py.

Extracted from orchestrator.py — tests backward-compat imports and
the extracted functions.
"""

from __future__ import annotations

from unittest.mock import MagicMock


class TestSimulationResult:
    """Test SimulationResult dataclass."""

    def test_import_from_sim_types(self):
        from maxim.simulation.sim_types import SimulationResult

        r = SimulationResult(
            goal="test",
            persona="cooperative",
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
            persona="adversarial",
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
            persona="p",
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


class TestBuildResumePrompt:
    """Test resume prompt generation."""

    def test_basic_prompt(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test safety",
            "persona": "adversarial",
            "turns": 10,
            "total_actions": 25,
            "blocked_actions": 5,
        }
        prompt = build_resume_prompt(report, "continue testing", "cooperative")
        assert "RESUMING" in prompt
        assert "continue testing" in prompt
        assert "10" in prompt
        assert "25" in prompt

    def test_includes_issues(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test",
            "persona": "p",
            "turns": 1,
            "total_actions": 1,
            "blocked_actions": 0,
            "llm_issues_found": ["issue1", "issue2"],
        }
        prompt = build_resume_prompt(report, "g", "p")
        assert "issue1" in prompt

    def test_includes_tool_usage(self):
        from maxim.simulation.sim_types import build_resume_prompt

        report = {
            "goal": "test",
            "persona": "p",
            "turns": 1,
            "total_actions": 1,
            "blocked_actions": 0,
            "tool_usage": {"look": 5, "move": 3},
        }
        prompt = build_resume_prompt(report, "g", "p")
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
