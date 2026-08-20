"""Tests for Phase 8 — API surface expansion (new verbs + events + tools)."""

from __future__ import annotations

import textwrap


# ---------------------------------------------------------------------------
# Import verbs from maxim package
# ---------------------------------------------------------------------------


class TestVerbImports:
    """Verify all new verbs are importable from the top-level package."""

    def test_campaign_importable(self):
        from maxim import campaign

        assert callable(campaign)

    def test_benchmark_importable(self):
        from maxim import benchmark

        assert callable(benchmark)

    def test_research_importable(self):
        from maxim import research

        assert callable(research)

    def test_on_importable(self):
        from maxim import on

        assert callable(on)

    def test_register_tool_importable(self):
        from maxim import register_tool

        assert callable(register_tool)

    def test_register_persona_importable(self):
        from maxim import register_persona

        assert callable(register_persona)

    def test_tool_decorator_importable(self):
        from maxim import tool

        assert callable(tool)

    def test_result_types_importable(self):
        from maxim import CampaignResult, BenchmarkResult, ResearchResult, EventHandle

        assert CampaignResult is not None
        assert BenchmarkResult is not None
        assert ResearchResult is not None
        assert EventHandle is not None


# ---------------------------------------------------------------------------
# campaign()
# ---------------------------------------------------------------------------


class TestCampaignVerb:
    def test_campaign_loads_and_runs(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from maxim.api import campaign

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            textwrap.dedent("""\
            campaign:
              name: api_test
              goal: test the API
              seed: 42
            acts:
              - name: act1
                encounters: [intro]
            encounters:
              intro:
                scene: "Welcome."
                choices: [go]
                branches: { go: __END__ }
        """)
        )

        # Mock start_simulation_mode to avoid real LLM calls
        mock_result = MagicMock()
        mock_result.turns = 3
        mock_result.finish_reason = "complete"
        mock_result.campaign_analysis = {"choices": [{"encounter": "intro", "choice": "go"}], "flags": ["flag_a"]}
        monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", lambda **kw: mock_result)

        result = campaign(str(yaml_path))
        assert result.campaign_name == "api_test"
        assert result.turns == 3
        assert result.finish_reason == "complete"
        assert len(result.choices_made) == 1

    def test_campaign_party_mode_override(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock
        from maxim.api import campaign

        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(
            textwrap.dedent("""\
            campaign:
              name: party_test
              goal: test
              seed: 1
            acts:
              - name: act1
                encounters: [enc1]
            encounters:
              enc1:
                scene: "Test."
                choices: [go]
                branches: { go: __END__ }
        """)
        )

        mock_result = MagicMock()
        mock_result.turns = 1
        mock_result.finish_reason = "complete"
        mock_result.campaign_analysis = {}
        monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", lambda **kw: mock_result)

        result = campaign(str(yaml_path), party_mode=True)
        assert result.party_mode is True

    def test_campaign_result_defaults(self):
        from maxim.api import CampaignResult

        r = CampaignResult()
        assert r.choices_made == []
        assert r.flags == []
        assert r.npc_memories == {}


# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------


class TestBenchmarkVerb:
    def test_benchmark_returns_result(self, monkeypatch):
        from unittest.mock import MagicMock
        from maxim.api import benchmark

        # Mock BenchmarkRunner to avoid real LLM calls
        mock_report = MagicMock()
        mock_report.results = {
            "mistral-7b": MagicMock(score=0.8, metrics={"recall": 0.75}),
            "qwen2.5-14b": MagicMock(score=0.9, metrics={"recall": 0.85}),
        }
        mock_report.summary_table.return_value = "Summary table"

        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_report
        monkeypatch.setattr("maxim.simulation.benchmark.BenchmarkRunner", lambda **kw: mock_runner)

        result = benchmark(models=["mistral-7b", "qwen2.5-14b"], suite="cognitive", runs=2)
        assert result.models == ["mistral-7b", "qwen2.5-14b"]
        assert result.suite == "cognitive"
        assert result.runs_per_model == 2
        assert "mistral-7b" in result.scores
        assert result.scores["mistral-7b"]["overall"] == 0.8

    def test_benchmark_result_defaults(self):
        from maxim.api import BenchmarkResult

        r = BenchmarkResult()
        assert r.models == []
        assert r.scores == {}


# ---------------------------------------------------------------------------
# research()
# ---------------------------------------------------------------------------


class TestResearchVerb:
    def test_research_returns_result(self, monkeypatch):
        from unittest.mock import MagicMock
        from maxim.api import research

        # Mock start_research_mode to avoid real LLM calls
        mock_result = MagicMock()
        mock_result.session_id = "test_123"
        mock_result.paper_path = ""
        mock_result.review_verdict = "accept"
        mock_result.experiments_count = 2
        monkeypatch.setattr("maxim.simulation.research_orchestrator.start_research_mode", lambda **kw: mock_result)

        result = research(goal="test memory retention")
        assert result.goal == "test memory retention"
        assert result.session_id == "test_123"
        assert result.experiment_count == 2

    def test_research_result_defaults(self):
        from maxim.api import ResearchResult

        r = ResearchResult()
        assert r.paper_draft == ""
        assert r.experiment_count == 0


# ---------------------------------------------------------------------------
# on() — event subscription
# ---------------------------------------------------------------------------


class TestEventSubscription:
    def test_on_returns_handle(self):
        from maxim.api import on

        handle = on("tool_call", lambda e: None)
        assert handle.event_name == "tool_call"
        assert handle.active is True

    def test_unsubscribe(self):
        from maxim.api import on

        handle = on("pain_signal", lambda e: None)
        assert handle.active is True
        handle.unsubscribe()
        assert handle.active is False

    def test_multiple_subscriptions(self):
        from maxim.api import on

        h1 = on("tool_call", lambda e: None)
        h2 = on("pain_signal", lambda e: None)
        assert h1.event_name == "tool_call"
        assert h2.event_name == "pain_signal"
        assert h1._handle_id != h2._handle_id

        h1.unsubscribe()
        h2.unsubscribe()

    def test_unbridged_names_removed_from_supported_set(self, caplog):
        # EVENT-seam cleanup: "memory_capture"/"prompt" were declared but never
        # bridged — subscribing silently received nothing. They now warn as
        # unknown (a dead name is worse than a smaller list). Re-adding one
        # REQUIRES the matching _bridge_event_subscriptions branch.
        import logging

        from maxim.api import _EVENT_TYPES, on

        assert set(_EVENT_TYPES) == {"tool_call", "pain_signal"}
        with caplog.at_level(logging.WARNING):
            handle = on("memory_capture", lambda e: None)
        handle.unsubscribe()
        assert any("Unknown event name" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# register_tool()
# ---------------------------------------------------------------------------


class TestToolRegistration:
    def test_register_tool_class(self):
        from maxim.api import register_tool, _pending_tools
        from maxim.tools.base import Tool, ToolOutput

        initial = len(_pending_tools)

        class TestTool(Tool):
            name = "api_test_tool"
            description = "Test tool"
            input_schema = {}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output="ok")

        register_tool(TestTool())
        assert len(_pending_tools) > initial

    def test_tool_decorator(self):
        from maxim.api import tool, _pending_tools

        initial = len(_pending_tools)

        @tool
        def my_analysis(data: str, depth: int = 3) -> str:
            """Analyze data."""
            return f"analyzed: {data}"

        assert len(_pending_tools) > initial
        # Function still works normally
        assert my_analysis("hello") == "analyzed: hello"

        # Schema inferred from type annotations
        registered = _pending_tools[-1]
        schema = registered.input_schema
        assert schema["type"] == "object"
        assert "data" in schema["properties"]
        assert schema["properties"]["data"]["type"] == "string"
        assert schema["properties"]["depth"]["type"] == "integer"
        assert schema["properties"]["depth"]["default"] == 3
        assert "data" in schema["required"]
        assert "depth" not in schema["required"]


# ---------------------------------------------------------------------------
# register_persona()
# ---------------------------------------------------------------------------


class TestRegisterPersonaCompatShim:
    """`register_persona` is a 1.0.x compatibility shim, version-gated to raise
    from 1.1.

    It was hard-deleted in #482 (`feat(1.1)!`), but the version bumps carried
    that removal into the 1.0.7-1.0.9 PATCH line while PyPI's previous release
    (1.0.0) has a working call — so raising there would break a public contract
    in a patch, which docs/pypi_maintenance.md forbids. The gate keeps the 0.9
    deprecation's literal "raises in 1.1" promise without depending on anyone
    remembering the date, which is the part a comment cannot enforce.
    """

    def test_accepted_with_a_deprecation_warning_on_the_1_0_line(self, monkeypatch):
        import warnings

        import maxim
        import maxim.api as api_mod

        monkeypatch.setattr(maxim, "__version__", "1.0.9")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            api_mod.register_persona("analyst", description="x")

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, "a silently-ignored call is worse than a warned one"
        assert "1.1" in str(dep[0].message), "the warning must name when it starts raising"

    def test_raises_from_1_1_onward(self, monkeypatch):
        import pytest as _pytest

        import maxim
        import maxim.api as api_mod

        for version in ("1.1.0", "1.1.0rc1", "1.2.0", "2.0.0"):
            monkeypatch.setattr(maxim, "__version__", version)
            with _pytest.raises(RuntimeError, match="removed in 1.1"):
                api_mod.register_persona("analyst")

    def test_unparseable_version_prefers_compatibility(self, monkeypatch):
        """An unrecognised version string must not break a caller."""
        import warnings

        import maxim
        import maxim.api as api_mod

        monkeypatch.setattr(maxim, "__version__", "not-a-version")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api_mod.register_persona("analyst")  # must not raise

    def test_v1_0_0_callers_are_not_broken(self, monkeypatch):
        """The concrete upgrade path this shim exists for: code written against
        PyPI's 1.0.0 keeps working on 1.0.x."""
        import warnings

        import maxim
        import maxim.api as api_mod

        monkeypatch.setattr(maxim, "__version__", "1.0.9")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            api_mod.register_persona(
                "researcher",
                description="d",
                focus="f",
                context_prompt="p",
                max_initiative=0.7,
            )


class TestImaginePersonaAlias:
    def test_persona_kwarg_warns_and_maps_to_mode(self, monkeypatch):
        """imagine(persona=...) is a 1.1 deprecation alias: it must warn AND
        map the value onto mode= (warn-and-ignore would silently relabel
        old callers' sessions "generative"). Dropped in 1.2."""
        import warnings

        import maxim.api as api_mod

        captured: dict = {}

        def fake_start(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before running a real sim")

        import maxim.simulation.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "start_simulation_mode", fake_start)
        monkeypatch.setattr(api_mod, "_validate_model", lambda m: None)
        monkeypatch.setattr(api_mod, "configure", lambda **kw: None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            try:
                api_mod.imagine(goal="g", persona="adversarial")
            except RuntimeError:
                pass

        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("persona" in str(w.message) for w in dep), "alias must warn"
        assert captured.get("mode") == "adversarial", "alias value must map onto mode="
        assert "persona" not in captured, "removed kwarg must not reach the orchestrator"


class TestPersonaRegistration:
    def test_register_persona_raises_removed_error(self, monkeypatch):
        """1.1 keeps the deprecation promise: register_persona() raises with a
        pointer (symbol survives one cycle so old code fails loudly, not
        with an AttributeError).

        Now version-GATED rather than unconditional: the removal landed in
        #482 (`feat(1.1)!`) but the version bumps carried it into the 1.0.7-
        1.0.9 patch line, where breaking a call that works in PyPI's 1.0.0
        would violate this project's own patch policy. The promise is kept
        from 1.1 onward — which is what this test now pins. See
        TestRegisterPersonaCompatShim for the 1.0.x half.
        """
        import pytest as _pytest

        import maxim
        from maxim.api import register_persona

        monkeypatch.setattr(maxim, "__version__", "1.1.0")
        with _pytest.raises(RuntimeError, match="removed in 1.1"):
            register_persona(name="test_api_persona")

    def test_persona_module_is_gone(self):
        """personas.py was hard-deleted (Option A) — the import must fail."""
        import pytest as _pytest

        with _pytest.raises(ModuleNotFoundError):
            import maxim.simulation.personas  # noqa: F401


class TestObserve:
    def test_observe_returns_dict(self):
        from maxim.api import observe

        result = observe()
        assert isinstance(result, dict)

    def test_observe_unknown_subsystem(self):
        from maxim.api import observe

        result = observe("nonexistent_subsystem")
        assert "error" in result

    def test_introspect_alias(self):
        from maxim.api import observe, introspect

        assert introspect is observe
