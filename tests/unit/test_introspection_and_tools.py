"""Tests for AUTIntrospector, ExamineTool, and tool usage tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from maxim.simulation.introspection import AUTIntrospector
from maxim.tools.narrative import ExamineTool, SayTool, ThinkTool
from maxim.tools.registry import ToolRegistry
from maxim.runtime.executor import Executor, TOOL_ALIASES


# ── Lightweight fakes for subsystems ─────────────────────────────────


@dataclass
class FakeMemory:
    id: str = "m_001"
    timestamp: float = 1000.0

    @dataclass
    class _Context:
        goal: str = "test goal"

    @dataclass
    class _Action:
        tool_used: str = "say"

    @dataclass
    class _Outcome:
        success: bool = True

    context: _Context = field(default_factory=_Context)
    action: _Action = field(default_factory=_Action)
    outcome: _Outcome = field(default_factory=_Outcome)


class FakeHippocampus:
    def __init__(self, memories: list | None = None):
        self._memories = memories or [FakeMemory()]

    def __len__(self):
        return len(self._memories)

    def recall(self, limit=5, goal=None, tool=None):
        return self._memories[:limit]

    def search_by_content(self, keyword, limit=5):
        return [m for m in self._memories if keyword.lower() in "test pain verath"][:limit]


@dataclass
class FakeLink:
    event_signature: str = "tool:say"
    outcome_signature: str = "success"
    valence: str = "positive"
    confidence: float = 0.85
    observation_count: int = 3


class FakeNAc:
    def __init__(self):
        self._links: dict[str, list] = {"tool:say": [FakeLink()]}

    def get_links_for_event(self, sig):
        return self._links.get(sig, [])

    def predict(self, event_type, event_sig):
        if event_sig == "say":
            return MagicMock(
                predicted_value=0.9,
                confidence=0.85,
                valence="positive",
                observation_count=3,
            )
        return None


class FakeConcept:
    def __init__(self, name="Verath", category="name", confidence=0.9):
        self.name = name
        self.category = category
        self.confidence = confidence


class FakeATL:
    def __len__(self):
        return 2

    def recall(self, limit=5, name=None, category=None):
        return [FakeConcept()]


class FakeEC:
    def __len__(self):
        return 5


class FakeSCN:
    def current_phase(self):
        return {"phase": "active", "cycle_position": 0.5}


class FakeMemoryHub:
    def __init__(self):
        self.atl = FakeATL()
        self.ec = FakeEC()
        self.scn = FakeSCN()


class FakeEnergyRegistry:
    def get_stats(self):
        return {"tokens_used": 1000, "cost_usd": 0.05}


# ── AUTIntrospector tests ───────────────────────────────────────────


class TestAUTIntrospector:
    def _make_introspector(self, **overrides):
        defaults = {
            "hippocampus": FakeHippocampus(),
            "nac": FakeNAc(),
            "memory_hub": FakeMemoryHub(),
            "energy_registry": FakeEnergyRegistry(),
        }
        defaults.update(overrides)
        return AUTIntrospector(**defaults)

    def test_memory_recall_keyword(self):
        intro = self._make_introspector()
        result = intro.memory_recall(keyword="verath")
        assert result["available"] is True
        assert result["count"] >= 0
        assert "total_stored" in result

    def test_memory_recall_no_hippocampus(self):
        intro = self._make_introspector(hippocampus=None)
        result = intro.memory_recall(keyword="test")
        assert result["available"] is False
        assert "hippocampus" in result["reason"]

    def test_causal_links_all(self):
        intro = self._make_introspector()
        result = intro.causal_links()
        assert result["available"] is True
        assert result["link_count"] >= 1
        assert result["links"][0]["event"] == "tool:say"

    def test_causal_links_filtered(self):
        intro = self._make_introspector()
        result = intro.causal_links(event_signature="tool:say")
        assert result["available"] is True
        assert result["link_count"] == 1

    def test_causal_links_no_nac(self):
        intro = self._make_introspector(nac=None)
        result = intro.causal_links()
        assert result["available"] is False

    def test_predict_outcome(self):
        intro = self._make_introspector()
        result = intro.predict_outcome(event_signature="say")
        assert result["available"] is True
        assert result["prediction"]["confidence"] == 0.85

    def test_predict_outcome_unknown(self):
        intro = self._make_introspector()
        result = intro.predict_outcome(event_signature="unknown_tool")
        assert result["prediction"] is None

    def test_predict_outcome_empty_sig(self):
        intro = self._make_introspector()
        result = intro.predict_outcome(event_signature="")
        assert "error" in result

    def test_pain_history(self):
        intro = self._make_introspector()
        result = intro.pain_history()
        assert result["available"] is True
        assert "pain_memory_count" in result

    def test_energy_status(self):
        intro = self._make_introspector()
        result = intro.energy_status()
        assert result["available"] is True
        assert result["tokens_used"] == 1000

    def test_energy_status_not_wired(self):
        intro = self._make_introspector(energy_registry=None)
        result = intro.energy_status()
        assert result["available"] is False

    def test_system_stats(self):
        intro = self._make_introspector()
        result = intro.system_stats()
        assert result["available"] is True
        assert "hippocampus_memories" in result
        assert "nac_causal_links" in result
        assert "atl_concepts" in result

    def test_concept_query(self):
        intro = self._make_introspector()
        result = intro.concept_query(name="Verath")
        assert result["available"] is True
        assert result["count"] >= 1
        assert result["concepts"][0]["name"] == "Verath"

    def test_concept_query_no_atl(self):
        hub = FakeMemoryHub()
        hub.atl = None
        intro = self._make_introspector(memory_hub=hub)
        result = intro.concept_query()
        assert result["available"] is False

    def test_temporal_patterns(self):
        intro = self._make_introspector()
        result = intro.temporal_patterns()
        assert result["available"] is True
        assert "current_phase" in result

    def test_dispatch_routes_correctly(self):
        intro = self._make_introspector()
        for query in AUTIntrospector.ALLOWED_QUERIES:
            result = intro.dispatch(query)
            assert isinstance(result, dict)
            # All should return something (available True or False)

    def test_dispatch_unknown_query(self):
        intro = self._make_introspector()
        result = intro.dispatch("nonexistent_query")
        assert "error" in result

    def test_full_analysis(self):
        intro = self._make_introspector()
        analysis = intro.full_analysis(seed_keywords=["Verath"])
        assert "memory_recall" in analysis
        assert "Verath" in analysis["memory_recall"]
        assert "system_stats" in analysis
        assert "causal_links" in analysis

    def test_full_analysis_no_keywords(self):
        intro = self._make_introspector()
        analysis = intro.full_analysis()
        assert "memory_recall" not in analysis
        assert "system_stats" in analysis

    def test_summarize(self):
        intro = self._make_introspector()
        analysis = intro.full_analysis(seed_keywords=["Verath"])
        summary = intro.summarize(analysis)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "memories" in summary or "causal" in summary

    def test_summarize_empty(self):
        intro = self._make_introspector(hippocampus=None, nac=None, memory_hub=None, energy_registry=None)
        analysis = intro.full_analysis()
        summary = intro.summarize(analysis)
        assert isinstance(summary, str)


# ── InspectAUTTool delegation tests ──────────────────────────────────


class TestInspectAUTToolDelegation:
    def test_delegates_to_introspector(self):
        from maxim.simulation.tools import InspectAUTTool

        intro = AUTIntrospector(hippocampus=FakeHippocampus(), nac=FakeNAc())
        tool = InspectAUTTool(introspector=intro)

        result = tool.execute(query="system_stats")
        assert result.success is True
        assert "hippocampus_memories" in result.output

    def test_backward_compat_constructor(self):
        """InspectAUTTool still works when passed subsystems directly."""
        from maxim.simulation.tools import InspectAUTTool

        tool = InspectAUTTool(hippocampus=FakeHippocampus(), nac=FakeNAc())
        result = tool.execute(query="system_stats")
        assert result.success is True

    def test_unknown_query_rejected(self):
        from maxim.simulation.tools import InspectAUTTool

        tool = InspectAUTTool(hippocampus=FakeHippocampus())
        result = tool.execute(query="nonexistent")
        assert result.success is False
        assert "Unknown query" in result.error


# ── ExamineTool tests ────────────────────────────────────────────────


class FakeBridge:
    """Minimal bridge stub with transcript percepts."""

    def __init__(self, percepts: list[dict] | None = None):
        self.percept_source = MagicMock()
        self.percept_source._transcript_percepts = percepts or []


class TestExamineTool:
    def test_examine_finds_target_in_percept(self):
        bridge = FakeBridge(
            [
                {"cli_input": "A dusty library with bookshelves. An ornate mirror hangs on the wall."},
            ]
        )
        tool = ExamineTool(bridge=bridge)
        result = tool.execute(target="mirror")
        assert result.success is True
        assert "mirror" in result.output["observation"].lower()

    def test_examine_no_target(self):
        tool = ExamineTool()
        result = tool.execute()
        assert result.success is False
        assert "No target" in result.error

    def test_examine_target_not_found(self):
        bridge = FakeBridge(
            [
                {"cli_input": "An empty room with stone walls."},
            ]
        )
        tool = ExamineTool(bridge=bridge)
        result = tool.execute(target="dragon")
        assert result.success is True
        assert "don't see anything notable" in result.output["observation"]

    def test_examine_enriches_from_hippocampus(self):
        bridge = FakeBridge(
            [
                {"cli_input": "A stone door with carvings."},
            ]
        )
        hippo = FakeHippocampus()
        tool = ExamineTool(bridge=bridge, hippocampus=hippo)
        # "pain" is in our fake hippocampus search
        result = tool.execute(target="pain")
        assert result.success is True

    def test_examine_no_bridge(self):
        """Examine with no bridge returns graceful fallback."""
        tool = ExamineTool()
        result = tool.execute(target="something")
        assert result.success is True
        assert "don't see anything notable" in result.output["observation"]

    def test_examine_accepts_aliases(self):
        """LLM param aliases (object, text) work."""
        bridge = FakeBridge(
            [
                {"cli_input": "A silver chalice on a pedestal."},
            ]
        )
        tool = ExamineTool(bridge=bridge)
        result = tool.execute(object="chalice")
        assert result.success is True
        assert "chalice" in result.output["observation"].lower()

    def test_examine_deduplicates(self):
        bridge = FakeBridge(
            [
                {"cli_input": "A mirror. A mirror."},  # same sentence repeated
            ]
        )
        tool = ExamineTool(bridge=bridge)
        result = tool.execute(target="mirror")
        # Should deduplicate
        obs = result.output["observation"]
        assert obs.count("mirror") <= 3  # deduplication prevents excessive repetition

    def test_examine_scans_recent_percepts(self):
        """Examine checks last 3 percepts, not just the most recent."""
        bridge = FakeBridge(
            [
                {"cli_input": "You see a dragon far away."},
                {"cli_input": "The corridor is empty."},
                {"cli_input": "You reach a dead end."},
            ]
        )
        tool = ExamineTool(bridge=bridge)
        result = tool.execute(target="dragon")
        assert result.success is True
        assert "dragon" in result.output["observation"].lower()


# ── Tool Usage Tracking tests ────────────────────────────────────────


class TestToolUsageTracking:
    def _make_executor(self) -> Executor:
        registry = ToolRegistry()
        registry.register(SayTool())
        registry.register(ThinkTool())
        return Executor(tool_registry=registry)

    def test_tracks_attempted(self):
        exe = self._make_executor()
        exe.execute({"tool_name": "say", "params": {"text": "hello"}})
        stats = exe.tool_usage_stats()
        assert stats["total_attempts"] == 1
        assert "say" in stats["tools_attempted"]

    def test_tracks_succeeded(self):
        exe = self._make_executor()
        exe.execute({"tool_name": "say", "params": {"text": "hello"}})
        stats = exe.tool_usage_stats()
        assert stats["total_successes"] == 1
        assert "say" in stats["tools_succeeded"]

    def test_tracks_hallucinated(self):
        exe = self._make_executor()
        exe.execute({"tool_name": "NonExistentTool", "params": {}})
        stats = exe.tool_usage_stats()
        assert stats["total_hallucinated"] == 1
        assert "NonExistentTool" in stats["tools_hallucinated"]

    def test_hallucination_rate(self):
        exe = self._make_executor()
        exe.execute({"tool_name": "say", "params": {"text": "hi"}})
        exe.execute({"tool_name": "FakeTool", "params": {}})
        stats = exe.tool_usage_stats()
        assert stats["hallucination_rate"] == pytest.approx(0.5)

    def test_alias_redirect_tracked(self):
        exe = self._make_executor()
        exe.execute({"tool_name": "remember", "params": {}})
        # "remember" aliases to "memory_recall" but memory_recall isn't registered
        # so it should be a hallucination (alias target not in registry)
        # Let's test with a registered alias target
        registry = ToolRegistry()
        registry.register(SayTool())
        exe2 = Executor(tool_registry=registry)
        exe2.execute({"tool_name": "speech", "params": {"text": "hello"}})
        assert len(exe2.alias_redirects) == 1
        assert exe2.alias_redirects[0] == ("speech", "say")

    def test_proactive_tool_list_after_repeated_failures(self):
        """After 2+ consecutive hallucinations, error includes full tool list."""
        exe = self._make_executor()
        # First failure: standard message
        result1 = exe.execute({"tool_name": "fake1", "params": {}})
        assert "Available tools:" not in (result1.error or "")

        # Second consecutive failure: should include full tool list
        result2 = exe.execute({"tool_name": "fake2", "params": {}})
        assert "Available tools:" in (result2.error or "")
        assert "say" in (result2.error or "")
        assert "think" in (result2.error or "")

    def test_consecutive_failures_reset_on_success(self):
        exe = self._make_executor()
        # Fail once
        exe.execute({"tool_name": "fake1", "params": {}})
        assert exe._consecutive_failures == 1

        # Succeed — resets counter
        exe.execute({"tool_name": "say", "params": {"text": "hi"}})
        assert exe._consecutive_failures == 0

        # Fail once again — should NOT trigger proactive list
        result = exe.execute({"tool_name": "fake3", "params": {}})
        assert "Available tools:" not in (result.error or "")

    def test_empty_stats(self):
        exe = self._make_executor()
        stats = exe.tool_usage_stats()
        assert stats["total_attempts"] == 0
        assert stats["hallucination_rate"] == 0.0


# ── Tool Alias Map tests ────────────────────────────────────────────


class TestToolAliases:
    def test_examine_aliases_present(self):
        """Verify examine-related aliases are in the map."""
        for alias in ["inspect", "look", "observe", "look_at", "investigate"]:
            assert TOOL_ALIASES.get(alias) == "examine", f"Missing alias: {alias}"

    def test_existing_aliases_unchanged(self):
        """Verify existing aliases weren't broken."""
        assert TOOL_ALIASES["remember"] == "memory_recall"
        assert TOOL_ALIASES["speech"] == "say"
        assert TOOL_ALIASES["reflection"] == "think"
        assert TOOL_ALIASES["internet_search"] == "memory_recall"
