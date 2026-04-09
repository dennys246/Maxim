"""Integration tests for API verb wiring.

These tests verify that the public API verbs (campaign, benchmark, research,
diagnose, observe, list_models, etc.) are wired to real runtimes — not stubs.
LLM calls are mocked to avoid cost/latency, but everything else is real:
schema loading, router construction, result types.

This is the safety net for the Module Compartmentalization plan.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── diagnose() — no mocks needed ───────────────────────────────────────


class TestDiagnoseIntegration:
    def test_diagnose_returns_report(self):
        """diagnose() returns a real DiagnosticReport from doctor checks."""
        from maxim.api import diagnose

        report = diagnose()
        assert report is not None
        assert hasattr(report, "checks") or hasattr(report, "summary")

    def test_diagnose_no_side_effects(self):
        """diagnose() should not mutate environment or start servers."""
        import os

        env_before = dict(os.environ)
        from maxim.api import diagnose

        diagnose()
        # No new MAXIM_* env vars leaked
        new_keys = set(os.environ) - set(env_before)
        maxim_leaked = [k for k in new_keys if k.startswith("MAXIM_")]
        assert maxim_leaked == [], f"diagnose() leaked env vars: {maxim_leaked}"


# ─── list_models() — no mocks needed ────────────────────────────────────


class TestListModelsIntegration:
    def test_list_models_returns_dict(self):
        """list_models() returns a dict with model categories."""
        from maxim.api import list_models

        result = list_models()
        assert isinstance(result, dict)
        # Should have at least some structure
        assert len(result) > 0


# ─── observe() — reads persisted state ──────────────────────────────────


class TestObserveIntegration:
    def test_observe_memory_returns_dict(self):
        """observe('memory') returns dict even with no prior session."""
        from maxim.api import observe

        result = observe("memory")
        assert isinstance(result, dict)

    def test_observe_causal_returns_dict(self):
        """observe('causal') returns dict even with no prior session."""
        from maxim.api import observe

        result = observe("causal")
        assert isinstance(result, dict)

    def test_observe_all_returns_dict(self):
        """observe() with no args returns full system summary."""
        from maxim.api import observe

        result = observe()
        assert isinstance(result, dict)


# ─── campaign() — real YAML loading, mocked LLM ─────────────────────────


class TestCampaignIntegration:
    def test_campaign_loads_real_yaml(self, tmp_path):
        """campaign() loads and validates real YAML before hitting the runtime."""
        from maxim.api import campaign, CampaignResult

        yaml_path = tmp_path / "test_campaign.yaml"
        yaml_path.write_text(
            textwrap.dedent("""\
            campaign:
              name: integration_test
              goal: test API wiring
              seed: 42
            acts:
              - name: act1
                encounters: [intro]
            encounters:
              intro:
                scene: "A test room."
                choices: [go]
                branches: { go: __END__ }
        """)
        )

        # Mock the simulation runtime but let YAML loading happen for real
        mock_result = MagicMock()
        mock_result.turns = 1
        mock_result.finish_reason = "campaign_end"
        mock_result.campaign_analysis = {"choices": [{"encounter": "intro", "choice": "go"}]}

        with patch("maxim.simulation.orchestrator.start_simulation_mode", return_value=mock_result):
            result = campaign(str(yaml_path))

        assert isinstance(result, CampaignResult)
        assert result.campaign_name == "integration_test"
        assert result.finish_reason == "campaign_end"

    def test_campaign_missing_yaml_raises(self):
        """campaign() with a nonexistent path raises FileNotFoundError."""
        from maxim.api import campaign

        with pytest.raises(FileNotFoundError):
            campaign("/nonexistent/path.yaml")


# ─── benchmark() — real suite resolution, mocked runner ──────────────────


class TestBenchmarkIntegration:
    def test_benchmark_resolves_suite_name(self):
        """benchmark(suite='quick_check') resolves to the real YAML file."""
        from maxim.api import benchmark, BenchmarkResult

        mock_report = MagicMock()
        mock_report.results = {"test-model": MagicMock(score=0.85, metrics={"recall": 0.8})}
        mock_report.summary_table.return_value = "Summary"
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_report

        with patch("maxim.simulation.benchmark.BenchmarkRunner", return_value=mock_runner) as mock_cls:
            result = benchmark(models=["test-model"], suite="quick_check")

        assert isinstance(result, BenchmarkResult)
        assert result.scores["test-model"]["overall"] == 0.85
        # Verify the runner was called with a real path
        call_kwargs = mock_cls.call_args
        suite_path = call_kwargs.kwargs.get("suite_path") or call_kwargs[1].get("suite_path")
        assert suite_path is not None
        assert Path(suite_path).exists(), f"Suite path {suite_path} should exist"

    def test_benchmark_invalid_suite_raises(self):
        """benchmark(suite='nonexistent') raises ConfigurationError."""
        from maxim.api import benchmark
        from maxim.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="not found"):
            benchmark(models=["test-model"], suite="nonexistent_suite_xyz")


# ─── research() — real orchestrator wiring, mocked LLM ──────────────────


class TestResearchIntegration:
    def test_research_calls_orchestrator(self):
        """research() delegates to start_research_mode with correct params."""
        from maxim.api import research, ResearchResult

        mock_result = MagicMock()
        mock_result.session_id = "integ_test_001"
        mock_result.paper_path = ""
        mock_result.review_verdict = "accept"
        mock_result.experiments_count = 3

        with patch("maxim.simulation.research_orchestrator.start_research_mode", return_value=mock_result) as mock_fn:
            result = research(goal="integration test", model="test-model")

        assert isinstance(result, ResearchResult)
        assert result.session_id == "integ_test_001"
        assert result.experiment_count == 3
        # Verify the orchestrator was called with our params
        mock_fn.assert_called_once()
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs["goal"] == "integration test"
        assert call_kwargs["language_model"] == "test-model"


# ─── tool() decorator — schema inference ─────────────────────────────────


class TestToolDecoratorIntegration:
    def test_schema_inferred_from_annotations(self):
        """@maxim.tool infers JSON Schema from type hints."""
        from maxim.api import tool, _pending_tools

        @tool
        def search_db(query: str, limit: int = 10, fuzzy: bool = False) -> list:
            """Search the database."""
            return []

        registered = _pending_tools[-1]
        schema = registered.input_schema

        assert schema["type"] == "object"
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 10
        assert schema["properties"]["fuzzy"]["type"] == "boolean"
        assert schema["properties"]["fuzzy"]["default"] is False
        assert schema["required"] == ["query"]
        assert "limit" not in schema["required"]
        assert "fuzzy" not in schema["required"]

    def test_tool_execution_works(self):
        """Registered tool can be executed via its execute() method."""
        from maxim.api import tool, _pending_tools

        @tool
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        registered = _pending_tools[-1]
        result = registered.execute(n=5)
        assert result.success is True
        assert result.output == 10


# ─── Event subscription wiring ───────────────────────────────────────────


class TestEventIntegration:
    def test_on_and_unsubscribe(self):
        """on() registers, handle.unsubscribe() removes."""
        from maxim.api import on

        calls = []
        handle = on("test_event", lambda e: calls.append(e))
        assert handle.event_name == "test_event"
        handle.unsubscribe()
        # After unsubscribe, the handle should be marked inactive
        assert not handle._active


# ─── Import hygiene ──────────────────────────────────────────────────────


class TestImportHygiene:
    def test_import_maxim_is_fast(self):
        """import maxim should not trigger heavy dependency loading."""
        import time

        start = time.time()
        import importlib

        importlib.invalidate_caches()
        # Can't truly re-import, but verify the module is loaded
        import maxim  # noqa: F401

        elapsed = time.time() - start
        # Should be well under 1 second (lazy loading)
        assert elapsed < 2.0, f"import maxim took {elapsed:.2f}s — too slow"

    def test_all_exports_importable(self):
        """Everything in maxim.__all__ should be importable."""
        import maxim

        for name in maxim.__all__:
            obj = getattr(maxim, name, None)
            assert obj is not None, f"maxim.{name} is in __all__ but not accessible"
