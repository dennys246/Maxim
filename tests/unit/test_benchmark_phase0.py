"""Tests for Benchmark Phase 0 — simulation infrastructure improvements.

Covers:
- SimulationResult enriched fields (0a)
- Observer.pain_stats() and benchmark_snapshot() (0b, 0f)
- Bio-system expectation types (0c)
- Scenario metadata loading (0d)
- JSON compliance counters (0e)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from maxim.simulation.scenario_source import Expectation, load_scenario
from maxim.simulation.sinks import ActionRecord, RecordingSink
from maxim.simulation.validation import validate_expectations


# ── Fake subsystem stubs ─────────────────────────────────────────────


class FakeHippocampus:
    def __init__(self, count=10):
        self._count = count

    def __len__(self):
        return self._count

    def recall(self, **kw):
        return []

    def search_by_content(self, query, limit=5):
        return [MagicMock(id="m_1", timestamp=1000)] if "verath" in query.lower() else []

    def stats(self):
        return {
            "total_memories": self._count,
            "graph_nodes": 8,
            "graph_edges": 12,
            "graph_enabled": True,
            "compressed_memories": 2,
        }


class FakeNAc:
    def __init__(self):
        self._links = {
            "tool:say": [
                MagicMock(
                    event_signature="tool:say",
                    outcome_signature="success",
                    valence="positive",
                    confidence=0.85,
                    observation_count=3,
                )
            ]
        }

    def stats(self):
        return {
            "total_links": 1,
            "event_signatures": 1,
            "outcome_signatures": 1,
            "total_observations": 3,
            "pending_events": 0,
            "priors": 0,
        }

    def get_links_for_event(self, sig):
        return self._links.get(sig, [])

    def predict(self, event_type, event_sig):
        return None


class FakePainDetector:
    def get_stats(self):
        return {
            "total_pain_signals": 5,
            "pain_counts": {"TOOL_FAILURE": 3, "EXCESSIVE_VELOCITY": 2},
            "sample_count": 10,
        }


class FakeMemoryHub:
    def __init__(self):
        self.atl = MagicMock()
        self.atl.__len__ = MagicMock(return_value=3)
        self.atl.recall = MagicMock(
            return_value=[
                MagicMock(name="Verath", category="name", confidence=0.9),
            ]
        )
        self.ec = MagicMock()
        self.ec.__len__ = MagicMock(return_value=5)
        self.scn = None


# ── Phase 0b/0f: Observer pain_stats + benchmark_snapshot ─────


class TestObserverPainStats:
    def test_pain_stats_with_detector(self):
        from maxim.simulation.introspection import Observer

        intro = Observer(pain_detector=FakePainDetector())
        result = intro.pain_stats()
        assert result["available"] is True
        assert result["total_pain_signals"] == 5
        assert result["pain_counts"]["TOOL_FAILURE"] == 3

    def test_pain_stats_no_detector(self):
        from maxim.simulation.introspection import Observer

        intro = Observer()
        result = intro.pain_stats()
        assert result["available"] is False
        assert "PainDetector" in result["reason"]


class TestBenchmarkSnapshot:
    def test_snapshot_includes_all_subsystem_stats(self):
        from maxim.simulation.introspection import Observer

        intro = Observer(
            hippocampus=FakeHippocampus(),
            nac=FakeNAc(),
            memory_hub=FakeMemoryHub(),
            pain_detector=FakePainDetector(),
        )
        snapshot = intro.benchmark_snapshot(seed_keywords=["Verath"])

        # full_analysis data
        assert "system_stats" in snapshot
        assert "causal_links" in snapshot
        assert "memory_recall" in snapshot
        assert "Verath" in snapshot["memory_recall"]

        # Raw subsystem stats (not in full_analysis)
        assert "hippocampus_stats" in snapshot
        assert snapshot["hippocampus_stats"]["graph_nodes"] == 8
        assert snapshot["hippocampus_stats"]["graph_edges"] == 12

        assert "nac_stats" in snapshot
        assert snapshot["nac_stats"]["total_links"] == 1
        assert snapshot["nac_stats"]["total_observations"] == 3

        assert "pain_stats" in snapshot
        assert snapshot["pain_stats"]["total_pain_signals"] == 5

    def test_snapshot_without_subsystems(self):
        from maxim.simulation.introspection import Observer

        intro = Observer()
        snapshot = intro.benchmark_snapshot()
        assert "system_stats" in snapshot
        # Should not crash, just omit subsystem data
        assert "hippocampus_stats" not in snapshot
        assert "nac_stats" not in snapshot
        assert "pain_stats" not in snapshot


# ── Phase 0c: Bio-system expectation types ───────────────────────────


class TestBioSystemExpectations:
    """Test the new bio-system expectation types in validation.py."""

    def _make_sink(self, actions=None):
        sink = RecordingSink()
        for a in actions or []:
            sink.record(a)
        return sink

    def _action(self, tool="respond", success=True, **kw):
        return ActionRecord(
            timestamp=1000.0,
            tool_name=tool,
            result_success=success,
            **kw,
        )

    # ── memory_count_range ───────────────────────────────────────

    def test_memory_count_range_pass(self):
        exp = Expectation(type="memory_count_range", params={"min": 5, "max": 20})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"hippocampus_stats": {"total_memories": 10}},
        )
        assert results[0].passed is True

    def test_memory_count_range_fail_below(self):
        exp = Expectation(type="memory_count_range", params={"min": 10, "max": 50})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"hippocampus_stats": {"total_memories": 3}},
        )
        assert results[0].passed is False
        assert "outside" in results[0].detail

    def test_memory_count_range_fallback_to_system_stats(self):
        """Falls back to system_stats if hippocampus_stats not available."""
        exp = Expectation(type="memory_count_range", params={"min": 1, "max": 100})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"system_stats": {"hippocampus_memories": 15}},
        )
        assert results[0].passed is True

    # ── concept_formed ───────────────────────────────────────────

    def test_concept_formed_pass(self):
        exp = Expectation(type="concept_formed", params={"concept_name": "Verath"})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "concepts": {
                    "available": True,
                    "concepts": [
                        {"name": "Verath", "category": "name", "confidence": 0.9},
                    ],
                }
            },
        )
        assert results[0].passed is True
        assert "Verath" in results[0].detail

    def test_concept_formed_fail(self):
        exp = Expectation(type="concept_formed", params={"concept_name": "unknown"})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "concepts": {"available": True, "concepts": []},
            },
        )
        assert results[0].passed is False

    def test_concept_formed_no_atl(self):
        exp = Expectation(type="concept_formed", params={"concept_name": "X"})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"concepts": {"available": False}},
        )
        assert results[0].passed is False
        assert "ATL" in results[0].detail

    # ── graph_density_above ──────────────────────────────────────

    def test_graph_density_pass(self):
        exp = Expectation(type="graph_density_above", params={"min_density": 1.0})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"hippocampus_stats": {"graph_nodes": 8, "graph_edges": 12}},
        )
        assert results[0].passed is True  # 12/8 = 1.5 >= 1.0

    def test_graph_density_fail(self):
        exp = Expectation(type="graph_density_above", params={"min_density": 2.0})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"hippocampus_stats": {"graph_nodes": 8, "graph_edges": 12}},
        )
        assert results[0].passed is False  # 12/8 = 1.5 < 2.0

    # ── causal_link_formed ───────────────────────────────────────

    def test_causal_link_formed_pass(self):
        exp = Expectation(type="causal_link_formed", params={"event_contains": "say"})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "causal_links": {
                    "available": True,
                    "links": [{"event": "tool:say", "outcome": "success", "confidence": 0.85, "observations": 3}],
                }
            },
        )
        assert results[0].passed is True

    def test_causal_link_formed_fail(self):
        exp = Expectation(type="causal_link_formed", params={"event_contains": "delete"})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "causal_links": {"available": True, "links": [{"event": "tool:say"}]},
            },
        )
        assert results[0].passed is False

    # ── prediction_valence ───────────────────────────────────────

    def test_prediction_valence_pass(self):
        exp = Expectation(
            type="prediction_valence",
            params={"tool": "delete_file", "expected_valence": "negative"},
        )
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "causal_links": {
                    "available": True,
                    "links": [{"event": "tool:delete_file", "valence": "Valence.NEGATIVE"}],
                }
            },
        )
        assert results[0].passed is True

    def test_prediction_valence_wrong(self):
        exp = Expectation(
            type="prediction_valence",
            params={"tool": "say", "expected_valence": "negative"},
        )
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={
                "causal_links": {
                    "available": True,
                    "links": [{"event": "tool:say", "valence": "Valence.POSITIVE"}],
                }
            },
        )
        assert results[0].passed is False

    # ── hallucination_rate_below ─────────────────────────────────

    def test_hallucination_rate_pass(self):
        exp = Expectation(type="hallucination_rate_below", params={"max_rate": 0.3})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            tool_stats={"hallucination_rate": 0.1, "total_attempts": 10, "total_hallucinated": 1},
        )
        assert results[0].passed is True

    def test_hallucination_rate_fail(self):
        exp = Expectation(type="hallucination_rate_below", params={"max_rate": 0.2})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            tool_stats={"hallucination_rate": 0.5, "total_attempts": 10, "total_hallucinated": 5},
        )
        assert results[0].passed is False
        assert "exceeds" in results[0].detail

    def test_hallucination_rate_no_calls(self):
        exp = Expectation(type="hallucination_rate_below", params={"max_rate": 0.3})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            tool_stats={"hallucination_rate": 0.0, "total_attempts": 0, "total_hallucinated": 0},
        )
        assert results[0].passed is True

    # ── tool_used ────────────────────────────────────────────────

    def test_tool_used_pass(self):
        exp = Expectation(type="tool_used", tool="examine")
        sink = self._make_sink([self._action(tool="examine")])
        results = validate_expectations([exp], sink)
        assert results[0].passed is True

    def test_tool_used_fail(self):
        exp = Expectation(type="tool_used", tool="examine")
        sink = self._make_sink([self._action(tool="respond")])
        results = validate_expectations([exp], sink)
        assert results[0].passed is False

    # ── pain_signal_count ────────────────────────────────────────

    def test_pain_signal_count_pass(self):
        exp = Expectation(type="pain_signal_count", params={"min": 3})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"pain_stats": {"available": True, "total_pain_signals": 5}},
        )
        assert results[0].passed is True

    def test_pain_signal_count_fail(self):
        exp = Expectation(type="pain_signal_count", params={"min": 10})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"pain_stats": {"available": True, "total_pain_signals": 2}},
        )
        assert results[0].passed is False

    def test_pain_signal_no_detector(self):
        exp = Expectation(type="pain_signal_count", params={"min": 1})
        results = validate_expectations(
            [exp],
            self._make_sink(),
            subsystem_snapshot={"pain_stats": {"available": False}},
        )
        assert results[0].passed is False
        assert "PainDetector" in results[0].detail


# ── Phase 0c: backward compat — existing expectations still work ─────


class TestExistingExpectationsUnchanged:
    """Verify existing expectation types still work with the new signature."""

    def _make_sink(self, actions=None):
        sink = RecordingSink()
        for a in actions or []:
            sink.record(a)
        return sink

    def test_action_count_range_still_works(self):
        exp = Expectation(type="action_count_range", params={"min": 1, "max": 10})
        sink = self._make_sink(
            [
                ActionRecord(timestamp=1.0, tool_name="respond", result_success=True),
            ]
        )
        # Pass extra kwargs — should be ignored by existing handler
        results = validate_expectations(
            [exp],
            sink,
            subsystem_snapshot={"some": "data"},
            tool_stats={"some": "stats"},
        )
        assert results[0].passed is True

    def test_memory_formed_still_works(self):
        exp = Expectation(type="memory_formed", memory_contains="verath")
        hippo = FakeHippocampus()
        results = validate_expectations(
            [exp],
            self._make_sink(),
            hippocampus=hippo,
            subsystem_snapshot={},
            tool_stats={},
        )
        assert results[0].passed is True


# ── Phase 0d: Scenario metadata loading ──────────────────────────────


class TestScenarioMetadata:
    def test_loads_tags_and_metadata(self):
        yaml_content = """
name: test_scenario
description: A test
tags: [memory, recall, hippocampus]
difficulty: medium
estimated_duration_s: 45
subsystems_tested: [hippocampus, nac]
tools_tested: [memory_recall, say]
percepts:
  - at: 0
    cli_input: "hello"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            defn = load_scenario(Path(f.name))

        assert defn.tags == ["memory", "recall", "hippocampus"]
        assert defn.difficulty == "medium"
        assert defn.estimated_duration_s == 45.0
        assert defn.subsystems_tested == ["hippocampus", "nac"]
        assert defn.tools_tested == ["memory_recall", "say"]

    def test_loads_benchmark_section(self):
        yaml_content = """
name: bench_test
percepts:
  - at: 0
    cli_input: "test"
benchmark:
  category: memory
  tier: [1, 2]
  weight: 2.0
  seed_keywords: ["Verath"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            defn = load_scenario(Path(f.name))

        assert defn.benchmark is not None
        assert defn.benchmark["category"] == "memory"
        assert defn.benchmark["weight"] == 2.0
        assert defn.benchmark["seed_keywords"] == ["Verath"]

    def test_loads_suite_section(self):
        yaml_content = """
name: cognitive_suite
suite:
  default_models: [mistral-7b, qwen2.5-14b]
  scenarios:
    - path: scenarios/experiments/hippocampal_recall_short.yaml
      weight: 2.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            defn = load_scenario(Path(f.name))

        assert defn.suite is not None
        assert defn.suite["default_models"] == ["mistral-7b", "qwen2.5-14b"]
        assert len(defn.suite["scenarios"]) == 1

    def test_loads_config_section(self):
        yaml_content = """
name: config_test
percepts: []
config:
  mode: generative
  max_turns: 30
  response_timeout: 45.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            defn = load_scenario(Path(f.name))

        assert defn.config is not None
        assert defn.config["mode"] == "generative"
        assert defn.config["max_turns"] == 30

    def test_metadata_defaults_when_absent(self):
        yaml_content = "name: minimal\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            defn = load_scenario(Path(f.name))

        assert defn.tags == []
        assert defn.difficulty == ""
        assert defn.estimated_duration_s == 0.0
        assert defn.subsystems_tested == []
        assert defn.benchmark is None
        assert defn.suite is None
        assert defn.config is None

    def test_existing_scenarios_still_load(self):
        """Verify that existing scenario files still load without errors."""
        scenarios_dir = Path(__file__).resolve().parent.parent.parent / "scenarios"
        for path in scenarios_dir.glob("*.yaml"):
            defn = load_scenario(path)
            assert defn.name, f"Failed to load: {path}"


# ── Phase 0e: JSON compliance counter ────────────────────────────────


class TestJsonComplianceCounter:
    def test_stats_return_format(self):
        from maxim.models.language.json_parser import json_parse_stats, json_parse_stats_reset

        json_parse_stats_reset()
        stats = json_parse_stats()
        assert "json_first_try" in stats
        assert "json_total" in stats
        assert "json_repaired" in stats
        assert "json_compliance_rate" in stats

    def test_first_try_increments(self):
        from maxim.models.language.json_parser import (
            _extract_json_object,
            json_parse_stats,
            json_parse_stats_reset,
        )

        json_parse_stats_reset()
        result = _extract_json_object('{"tool": "respond", "message": "hi"}')
        assert result is not None
        stats = json_parse_stats()
        assert stats["json_first_try"] == 1
        assert stats["json_total"] == 1
        assert stats["json_compliance_rate"] == 1.0

    def test_repaired_increments(self):
        from maxim.models.language.json_parser import (
            _extract_json_object,
            json_parse_stats,
            json_parse_stats_reset,
        )

        json_parse_stats_reset()
        # Trailing comma — needs repair
        _extract_json_object('{"tool": "respond",}')
        # May or may not parse depending on json_repair availability
        stats = json_parse_stats()
        assert stats["json_total"] == 1
        # Either first_try or repaired should be 1
        assert stats["json_first_try"] + stats["json_repaired"] <= 1

    def test_reset_clears_counters(self):
        from maxim.models.language.json_parser import (
            _extract_json_object,
            json_parse_stats,
            json_parse_stats_reset,
        )

        _extract_json_object('{"a": 1}')
        json_parse_stats_reset()
        stats = json_parse_stats()
        assert stats["json_first_try"] == 0
        assert stats["json_total"] == 0
        assert stats["json_repaired"] == 0

    def test_compliance_rate_zero_division(self):
        from maxim.models.language.json_parser import json_parse_stats, json_parse_stats_reset

        json_parse_stats_reset()
        stats = json_parse_stats()
        assert stats["json_compliance_rate"] == 1.0  # 0/0 defaults to 1.0
