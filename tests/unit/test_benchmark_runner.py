"""Tests for BenchmarkRunner — suite loading, metrics, aggregation, scoring."""

from __future__ import annotations

import tempfile

import pytest

from maxim.simulation.benchmark import (
    BenchmarkReport,
    BenchmarkRunner,
    ModelResult,
    RunResult,
)


# ── Suite loading tests ──────────────────────────────────────────────


class TestSuiteLoading:
    def test_load_single_scenario(self):
        yaml_content = """
name: test_scenario
percepts:
  - at: 0
    cli_input: "hello"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test-model"], suite_path=f.name)

        assert len(runner._scenarios) == 1
        assert runner._scenarios[0]["path"] == f.name
        assert runner._scenarios[0]["weight"] == 1.0

    def test_load_suite_with_scenarios(self):
        yaml_content = """
name: test_suite
suite:
  default_models: [mistral-7b, qwen2.5-14b]
  scenarios:
    - path: scenarios/experiments/hippocampal_recall_short.yaml
      weight: 2.0
    - path: scenarios/benchmarks/tool_discovery.yaml
      weight: 1.0
  scoring:
    memory_recall_success: { pass: 1.0 }
    hallucination_rate: { pass_below: 0.3 }
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test-model"], suite_path=f.name)

        assert len(runner._scenarios) == 2
        assert runner._scenarios[0]["weight"] == 2.0
        assert runner._scenarios[1]["weight"] == 1.0
        assert runner._scoring["memory_recall_success"] == {"pass": 1.0}

    def test_load_scenario_with_benchmark_section(self):
        yaml_content = """
name: recall_test
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
            runner = BenchmarkRunner(models=["test-model"], suite_path=f.name)

        assert len(runner._scenarios) == 1
        assert runner._scenarios[0]["seed_keywords"] == ["Verath"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            BenchmarkRunner(models=["test"], suite_path="/nonexistent.yaml")

    def test_empty_suite_raises(self):
        yaml_content = """
name: empty_suite
suite:
  scenarios: []
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            with pytest.raises(ValueError, match="no scenarios"):
                BenchmarkRunner(models=["test"], suite_path=f.name)


# ── Metric computation tests ─────────────────────────────────────────


class _FakeExperimentResult:
    """Minimal stub matching ExperimentResult interface."""

    def __init__(self, analysis=None, **kw):
        self.analysis = analysis or {}
        self.sim_turns = kw.get("sim_turns", 7)
        self.sim_duration_s = kw.get("sim_duration_s", 5.0)
        self.finish_reason = kw.get("finish_reason", "complete")
        self.total_actions = kw.get("total_actions", 10)
        self.campaign_path = "test.yaml"
        self.aut_model = "test-model"


class _FakeScenario:
    """Minimal stub for scenario definition."""

    def __init__(self):
        self.benchmark = None
        self.expectations = []


class TestMetricComputation:
    def _make_runner(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return BenchmarkRunner(models=["test"], suite_path=f.name)

    def test_memory_formation_rate(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(
            analysis={
                "system_stats": {"hippocampus_memories": 14},
                "hippocampus_stats": {"total_memories": 14, "graph_nodes": 8, "graph_edges": 12},
                "nac_stats": {"total_links": 0, "event_signatures": 0, "total_observations": 0},
            },
            sim_turns=7,
        )
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["memory_formation_rate"] == pytest.approx(14 / 7)

    def test_graph_density(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(
            analysis={
                "system_stats": {},
                "hippocampus_stats": {"total_memories": 10, "graph_nodes": 8, "graph_edges": 12},
                "nac_stats": {"total_links": 0},
            },
        )
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["associative_graph_density"] == pytest.approx(12 / 8)

    def test_causal_learning_metrics(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(
            analysis={
                "system_stats": {"nac_causal_links": 5},
                "hippocampus_stats": {},
                "nac_stats": {
                    "total_links": 5,
                    "event_signatures": 3,
                    "total_observations": 15,
                },
            },
        )
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["causal_link_count"] == 5.0
        assert metrics["causal_diversity"] == pytest.approx(3 / 5)
        assert metrics["observation_density"] == pytest.approx(15 / 5)

    def test_keyword_recall(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(
            analysis={
                "system_stats": {},
                "hippocampus_stats": {},
                "nac_stats": {},
                "memory_recall": {
                    "Verath": {"count": 2, "total_stored": 14},
                    "missing": {"count": 0, "total_stored": 14},
                },
            },
        )
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["recall_Verath"] == 1.0
        assert metrics["recall_missing"] == 0.0

    def test_pain_metrics(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(
            analysis={
                "system_stats": {},
                "hippocampus_stats": {},
                "nac_stats": {},
                "pain_stats": {"total_pain_signals": 3, "available": True},
            },
        )
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["pain_signal_count"] == 3.0

    def test_empty_analysis(self):
        runner = self._make_runner()
        result = _FakeExperimentResult(analysis={})
        metrics = runner._compute_metrics(result, _FakeScenario())
        assert metrics["memory_formation_rate"] == 0.0
        assert metrics["causal_link_count"] == 0.0


# ── Aggregation tests ────────────────────────────────────────────────


class TestAggregation:
    def _make_runner(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return BenchmarkRunner(models=["test"], suite_path=f.name)

    def test_aggregate_single_run(self):
        runner = self._make_runner()
        runs = [
            RunResult(
                scenario="test",
                model="mistral-7b",
                run_index=0,
                metrics={"memory_formation_rate": 2.0, "causal_link_count": 5.0},
                expectations_passed=3,
                expectations_total=4,
            ),
        ]
        result = runner._aggregate_runs("mistral-7b", runs)
        assert result.metrics["memory_formation_rate"] == 2.0
        assert result.metrics["causal_link_count"] == 5.0
        assert result.expectations_passed == 3
        assert result.expectations_total == 4

    def test_aggregate_multiple_runs(self):
        runner = self._make_runner()
        runs = [
            RunResult(
                scenario="test",
                model="mistral-7b",
                run_index=0,
                metrics={"memory_formation_rate": 2.0},
            ),
            RunResult(
                scenario="test",
                model="mistral-7b",
                run_index=1,
                metrics={"memory_formation_rate": 4.0},
            ),
        ]
        result = runner._aggregate_runs("mistral-7b", runs)
        assert result.metrics["memory_formation_rate"] == pytest.approx(3.0)
        assert result.metrics_stddev["memory_formation_rate"] > 0

    def test_aggregate_with_errors(self):
        runner = self._make_runner()
        runs = [
            RunResult(scenario="test", model="m", run_index=0, error="failed"),
            RunResult(
                scenario="test",
                model="m",
                run_index=1,
                metrics={"memory_formation_rate": 2.0},
            ),
        ]
        result = runner._aggregate_runs("m", runs)
        # Only successful run contributes
        assert result.metrics["memory_formation_rate"] == 2.0

    def test_aggregate_all_errors(self):
        runner = self._make_runner()
        runs = [
            RunResult(scenario="test", model="m", run_index=0, error="failed"),
        ]
        result = runner._aggregate_runs("m", runs)
        assert result.metrics == {}


# ── Scoring tests ────────────────────────────────────────────────────


class TestScoring:
    def _make_runner(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return BenchmarkRunner(models=["test"], suite_path=f.name)

    def test_threshold_pass(self):
        runner = self._make_runner()
        runner._scoring = {
            "memory_formation_rate": {"pass_above": 1.0},
            "hallucination_rate": {"pass_below": 0.3},
        }
        assert runner._check_thresholds({"memory_formation_rate": 2.0, "hallucination_rate": 0.1})

    def test_threshold_fail(self):
        runner = self._make_runner()
        runner._scoring = {"memory_formation_rate": {"pass_above": 5.0}}
        assert not runner._check_thresholds({"memory_formation_rate": 2.0})

    def test_threshold_binary_pass(self):
        runner = self._make_runner()
        runner._scoring = {"recall_Verath": {"pass": 1.0}}
        assert runner._check_thresholds({"recall_Verath": 1.0})

    def test_threshold_binary_fail(self):
        runner = self._make_runner()
        runner._scoring = {"recall_Verath": {"pass": 1.0}}
        assert not runner._check_thresholds({"recall_Verath": 0.0})

    def test_no_thresholds_auto_passes(self):
        runner = self._make_runner()
        runner._scoring = {}
        assert runner._check_thresholds({"anything": 0.0})

    def test_composite_score_nonzero(self):
        runner = self._make_runner()
        score = runner._compute_composite_score({"memory_formation_rate": 0.8, "recall_Verath": 1.0})
        assert score > 0.0

    def test_composite_score_empty(self):
        runner = self._make_runner()
        assert runner._compute_composite_score({}) == 0.0

    def test_composite_score_inverts_hallucination(self):
        runner = self._make_runner()
        low_halluc = runner._compute_composite_score({"hallucination_rate": 0.1})
        high_halluc = runner._compute_composite_score({"hallucination_rate": 0.9})
        assert low_halluc > high_halluc  # Lower hallucination = higher score


# ── Report tests ─────────────────────────────────────────────────────


class TestBenchmarkReport:
    def test_summary_table_renders(self):
        report = BenchmarkReport(
            timestamp="20260406_210000",
            suite="cognitive_suite",
            models=["mistral-7b", "qwen-14b"],
            runs_per_model=1,
            results={
                "mistral-7b": ModelResult(
                    model="mistral-7b",
                    metrics={"memory_formation_rate": 2.0, "hallucination_rate": 0.3},
                    score=0.7,
                    passed=True,
                    expectations_passed=5,
                    expectations_total=6,
                ),
                "qwen-14b": ModelResult(
                    model="qwen-14b",
                    metrics={"memory_formation_rate": 1.5, "hallucination_rate": 0.5},
                    score=0.5,
                    passed=False,
                    expectations_passed=3,
                    expectations_total=6,
                ),
            },
            overall_ranking=["mistral-7b", "qwen-14b"],
            duration_s=30.0,
        )
        table = report.summary_table()
        assert "BENCHMARK REPORT" in table
        assert "cognitive_suite" in table
        assert "mistral-7b" in table
        assert "qwen-14b" in table
        assert "TIER 1" in table
        assert "TIER 2" in table

    def test_summary_table_empty_results(self):
        report = BenchmarkReport(
            timestamp="test",
            suite="empty",
            models=[],
            runs_per_model=0,
        )
        table = report.summary_table()
        assert "No results" in table

    def test_save_report(self):
        report = BenchmarkReport(
            timestamp="20260406_test",
            suite="test_suite",
            models=["test-model"],
            runs_per_model=1,
            results={
                "test-model": ModelResult(
                    model="test-model",
                    metrics={"memory_formation_rate": 2.0},
                    score=0.6,
                    passed=True,
                ),
            },
            overall_ranking=["test-model"],
            duration_s=5.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = "name: test\npercepts: []\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                runner = BenchmarkRunner(
                    models=["test-model"],
                    suite_path=f.name,
                    output_dir=tmpdir,
                )
            report_dir = runner.save_report(report)
            assert (report_dir / "benchmark_report.json").exists()
            assert (report_dir / "summary.md").exists()


# ── Helper method tests ──────────────────────────────────────────────


class TestHelperMethods:
    def _make_runner(self):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return BenchmarkRunner(models=["test"], suite_path=f.name)

    def test_count_think_chains(self):
        runner = self._make_runner()
        actions = [
            {"tool_name": "think"},
            {"tool_name": "say"},
            {"tool_name": "respond"},
            {"tool_name": "think"},
            {"tool_name": "memory_recall"},
        ]
        assert runner._count_think_chains(actions) == 2

    def test_count_think_chains_no_chains(self):
        runner = self._make_runner()
        actions = [{"tool_name": "respond"}, {"tool_name": "say"}]
        assert runner._count_think_chains(actions) == 0

    def test_compute_ttr(self):
        runner = self._make_runner()
        actions = [
            {"result_output": "I see a door"},
            {"result_output": "I see a door"},  # repeated
            {"result_output": "The elm is silver"},
        ]
        ttr = runner._compute_ttr(actions)
        assert 0.0 < ttr < 1.0  # Some repetition, but not all unique

    def test_compute_ttr_empty(self):
        runner = self._make_runner()
        assert runner._compute_ttr([]) == 0.0

    def test_rankings(self):
        runner = self._make_runner()
        results = {
            "model_a": ModelResult(model="model_a", metrics={"memory_formation_rate": 2.0}),
            "model_b": ModelResult(model="model_b", metrics={"memory_formation_rate": 3.0}),
        }
        rankings = runner._compute_rankings(results)
        assert rankings["memory_formation_rate"][0] == "model_b"  # Higher is better


# ── Phase 3: Unified loader tests ────────────────────────────────────


class TestUnifiedLoader:
    def test_suite_with_config(self):
        """Suite entries can carry per-scenario config."""
        yaml_content = """
name: config_suite
suite:
  scenarios:
    - path: test.yaml
      weight: 2.0
      config:
        persona: adversarial
        max_turns: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        assert runner._scenarios[0]["config"]["persona"] == "adversarial"
        assert runner._scenarios[0]["config"]["max_turns"] == 10

    def test_single_scenario_with_metadata(self):
        """Single scenario loads tags from unified format."""
        yaml_content = """
name: tagged_test
tags: [memory, recall]
difficulty: medium
estimated_duration_s: 45
percepts:
  - at: 0
    cli_input: "hello"
benchmark:
  category: memory
  weight: 2.5
  seed_keywords: ["test"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            runner = BenchmarkRunner(models=["test"], suite_path=f.name)

        assert runner._scenarios[0]["weight"] == 2.5
        assert runner._scenarios[0]["seed_keywords"] == ["test"]
        assert runner._scenarios[0]["tags"] == ["memory", "recall"]

    def test_loads_real_cognitive_suite(self):
        """The cognitive_suite.yaml we just created loads correctly."""
        import os

        suite_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scenarios", "benchmarks", "cognitive_suite.yaml"
        )
        if not os.path.exists(suite_path):
            pytest.skip("cognitive_suite.yaml not found")

        runner = BenchmarkRunner(models=["test"], suite_path=suite_path)
        assert len(runner._scenarios) == 6
        assert runner._scoring  # Should have scoring thresholds

    def test_loads_real_quick_check(self):
        """The quick_check.yaml scenario loads correctly."""
        import os

        path = os.path.join(os.path.dirname(__file__), "..", "..", "scenarios", "benchmarks", "quick_check.yaml")
        if not os.path.exists(path):
            pytest.skip("quick_check.yaml not found")

        runner = BenchmarkRunner(models=["test"], suite_path=path)
        assert len(runner._scenarios) == 1
        assert runner._scenarios[0]["seed_keywords"] == ["seven-three-nine", "Kael"]


# ── Phase 4: Output format tests ─────────────────────────────────────


class TestOutputFormat:
    def test_summary_with_stddev(self):
        """Report shows stddev when runs > 1."""
        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["model_a"],
            runs_per_model=3,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0},
                    metrics_stddev={"memory_formation_rate": 0.5},
                    score=0.7,
                    passed=True,
                ),
            },
            overall_ranking=["model_a"],
        )
        table = report.summary_table()
        assert "+-" in table  # stddev shown as +-

    def test_summary_with_per_scenario(self):
        """Report shows per-scenario breakdown."""
        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0},
                    score=0.7,
                    passed=True,
                    per_scenario={
                        "quick_check": {"memory_formation_rate": 2.0},
                        "causal_learning": {"causal_link_count": 5.0},
                    },
                ),
            },
            overall_ranking=["model_a"],
        )
        table = report.summary_table()
        assert "PER-SCENARIO" in table
        assert "quick_check" in table
        assert "causal_learning" in table

    def test_summary_shows_pass_fail(self):
        """Expectations section shows PASS/FAIL status."""
        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["pass_model", "fail_model"],
            runs_per_model=1,
            results={
                "pass_model": ModelResult(
                    model="pass_model",
                    metrics={},
                    score=0.8,
                    passed=True,
                    expectations_passed=5,
                    expectations_total=5,
                ),
                "fail_model": ModelResult(
                    model="fail_model",
                    metrics={},
                    score=0.3,
                    passed=False,
                    expectations_passed=2,
                    expectations_total=5,
                ),
            },
            overall_ranking=["pass_model", "fail_model"],
        )
        table = report.summary_table()
        assert "[PASS]" in table
        assert "[FAIL]" in table


# ── Phase 5: Scenario YAML validation ────────────────────────────────


class TestBenchmarkScenarios:
    """Validate that all benchmark scenario YAML files load correctly."""

    _BENCHMARKS_DIR = None

    @classmethod
    def _get_dir(cls):
        import os

        d = os.path.join(os.path.dirname(__file__), "..", "..", "scenarios", "benchmarks")
        return d if os.path.isdir(d) else None

    def test_all_scenarios_loadable(self):
        """Every YAML in scenarios/benchmarks/ loads without errors."""
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        for path in sorted(Path(d).glob("*.yaml")):
            defn = load_scenario(path)
            assert defn.name, f"Failed to load: {path}"

    def test_quick_check_structure(self):
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        defn = load_scenario(Path(d) / "quick_check.yaml")
        assert defn.name == "quick_check"
        assert len(defn.percepts) == 3
        assert len(defn.expectations) >= 3
        assert defn.tags == ["memory", "recall", "quick"]
        assert defn.benchmark is not None
        assert defn.benchmark["seed_keywords"] == ["seven-three-nine", "Kael"]

    def test_cognitive_suite_structure(self):
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        defn = load_scenario(Path(d) / "cognitive_suite.yaml")
        assert defn.name == "cognitive_suite_v1"
        assert defn.suite is not None
        assert len(defn.suite["scenarios"]) == 6
        assert "recall_Verath" in defn.suite.get("scoring", {})

    def test_causal_learning_has_pain(self):
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        defn = load_scenario(Path(d) / "causal_learning.yaml")
        pain_percepts = [p for p in defn.percepts if p.get("source") == "proprioception"]
        assert len(pain_percepts) >= 2, "Should have at least 2 pain signals"
        assert "nac" in defn.subsystems_tested

    def test_aversion_learning_phases(self):
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        defn = load_scenario(Path(d) / "aversion_learning.yaml")
        phases = set()
        for p in defn.percepts:
            meta = p.get("metadata", {})
            if "phase" in meta:
                phases.add(meta["phase"])
        assert "present" in phases
        assert "retest" in phases

    def test_concept_formation_entities(self):
        from pathlib import Path
        from maxim.simulation.scenario_source import load_scenario

        d = self._get_dir()
        if not d:
            pytest.skip("scenarios/benchmarks/ not found")

        defn = load_scenario(Path(d) / "concept_formation.yaml")
        assert defn.benchmark is not None
        assert "Elara" in defn.benchmark["seed_keywords"]
        assert "Kael" in defn.benchmark["seed_keywords"]


# ── Phase 6: Baseline comparison tests ──────────────────────────────


class TestBaselineComparison:
    def _make_runner(self, baseline_path=None):
        yaml_content = "name: test\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            return BenchmarkRunner(
                models=["test"], suite_path=f.name, baseline_path=baseline_path
            )

    def _make_baseline_json(self, results_data):
        """Write a baseline JSON file and return its path."""
        import json

        data = {
            "timestamp": "20260401_000000",
            "suite": "test_suite",
            "models": list(results_data.keys()),
            "runs_per_model": 1,
            "duration_s": 10.0,
            "overall_ranking": list(results_data.keys()),
            "results": results_data,
            "rankings": {},
        }
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f, indent=2)
        f.flush()
        f.close()
        return f.name

    def test_load_baseline_valid(self):
        baseline_path = self._make_baseline_json(
            {
                "model_a": {
                    "model": "model_a",
                    "metrics": {"memory_formation_rate": 2.0},
                    "metrics_stddev": {},
                    "score": 0.6,
                    "passed": True,
                    "expectations_passed": 3,
                    "expectations_total": 4,
                    "per_scenario": {},
                }
            }
        )
        runner = self._make_runner(baseline_path=baseline_path)
        report = runner._load_baseline(baseline_path)
        assert report is not None
        assert "model_a" in report.results
        assert report.results["model_a"].metrics["memory_formation_rate"] == 2.0

    def test_load_baseline_missing_file(self):
        runner = self._make_runner()
        result = runner._load_baseline("/nonexistent/baseline.json")
        assert result is None

    def test_load_baseline_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json!!!")
            f.flush()
            runner = self._make_runner()
            result = runner._load_baseline(f.name)
        assert result is None

    def test_compare_improvement_higher_is_better(self):
        """Positive delta on memory_formation_rate = improvement."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 3.0, "concept_formation_rate": 0.5},
                ),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0, "concept_formation_rate": 0.5},
                ),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        assert "model_a" in comparison
        mem_delta = comparison["model_a"]["memory_formation_rate"]
        assert mem_delta["delta"] == pytest.approx(1.0)
        assert mem_delta["direction"] == "improved"
        # concept_formation_rate unchanged (delta ~0)
        concept_delta = comparison["model_a"]["concept_formation_rate"]
        assert concept_delta["direction"] == "unchanged"

    def test_compare_regression_higher_is_better(self):
        """Negative delta on memory_formation_rate = regression."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 1.0},
                ),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0},
                ),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        delta = comparison["model_a"]["memory_formation_rate"]
        assert delta["delta"] == pytest.approx(-1.0)
        assert delta["direction"] == "regressed"

    def test_compare_lower_is_better_improvement(self):
        """Negative delta on hallucination_rate = improvement."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"hallucination_rate": 0.1},
                ),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"hallucination_rate": 0.3},
                ),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        delta = comparison["model_a"]["hallucination_rate"]
        assert delta["delta"] == pytest.approx(-0.2)
        assert delta["direction"] == "improved"

    def test_compare_lower_is_better_regression(self):
        """Positive delta on hallucination_rate = regression."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"hallucination_rate": 0.5},
                ),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"hallucination_rate": 0.2},
                ),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        delta = comparison["model_a"]["hallucination_rate"]
        assert delta["delta"] == pytest.approx(0.3)
        assert delta["direction"] == "regressed"

    def test_compare_skips_missing_model(self):
        """Model in current but not in baseline is skipped gracefully."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a", "model_b"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(model="model_a", metrics={"mem": 1.0}),
                "model_b": ModelResult(model="model_b", metrics={"mem": 2.0}),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(model="model_a", metrics={"mem": 0.5}),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        assert "model_a" in comparison
        assert "model_b" not in comparison

    def test_compare_skips_missing_metrics(self):
        """Metrics present in only one side are skipped."""
        runner = self._make_runner()
        current = BenchmarkReport(
            timestamp="now",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0, "new_metric": 1.0},
                ),
            },
        )
        baseline = BenchmarkReport(
            timestamp="before",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 1.0, "old_metric": 0.5},
                ),
            },
        )
        comparison = runner._compare_to_baseline(current, baseline)
        assert "memory_formation_rate" in comparison["model_a"]
        assert "new_metric" not in comparison["model_a"]
        assert "old_metric" not in comparison["model_a"]

    def test_summary_table_shows_deltas(self):
        """Summary table includes delta indicators when baseline_comparison is set."""
        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 3.0, "hallucination_rate": 0.1},
                    score=0.7,
                    passed=True,
                ),
            },
            overall_ranking=["model_a"],
            baseline_comparison={
                "model_a": {
                    "memory_formation_rate": {"delta": 1.0, "direction": "improved"},
                    "hallucination_rate": {"delta": -0.2, "direction": "improved"},
                },
            },
        )
        table = report.summary_table()
        assert "+1.00" in table
        assert "-0.20" in table

    def test_summary_table_no_baseline(self):
        """Summary table works normally when baseline_comparison is None."""
        report = BenchmarkReport(
            timestamp="test",
            suite="test",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0},
                    score=0.6,
                    passed=True,
                ),
            },
            overall_ranking=["model_a"],
        )
        table = report.summary_table()
        assert "memory_formation_rate" in table
        # No delta indicators
        assert "(" not in table or "+-" in table or "(" not in table.split("memory_formation_rate")[1].split("\n")[0]

    def test_save_report_includes_baseline(self):
        """Saved JSON includes baseline_comparison when present."""
        import json

        report = BenchmarkReport(
            timestamp="20260406_baseline_test",
            suite="test_suite",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 3.0},
                    score=0.7,
                    passed=True,
                ),
            },
            overall_ranking=["model_a"],
            baseline_comparison={
                "model_a": {
                    "memory_formation_rate": {"delta": 1.0, "direction": "improved"},
                },
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = "name: test\npercepts: []\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                runner = BenchmarkRunner(models=["model_a"], suite_path=f.name, output_dir=tmpdir)
            report_dir = runner.save_report(report)
            with open(report_dir / "benchmark_report.json") as fp:
                data = json.load(fp)
            assert "baseline_comparison" in data
            assert data["baseline_comparison"]["model_a"]["memory_formation_rate"]["delta"] == 1.0

    def test_save_report_omits_baseline_when_none(self):
        """Saved JSON does not include baseline_comparison when None."""
        import json

        report = BenchmarkReport(
            timestamp="20260406_no_baseline",
            suite="test_suite",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.0},
                    score=0.6,
                    passed=True,
                ),
            },
            overall_ranking=["model_a"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = "name: test\npercepts: []\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                runner = BenchmarkRunner(models=["model_a"], suite_path=f.name, output_dir=tmpdir)
            report_dir = runner.save_report(report)
            with open(report_dir / "benchmark_report.json") as fp:
                data = json.load(fp)
            assert "baseline_comparison" not in data

    def test_load_baseline_roundtrip(self):
        """A saved report can be loaded back as a baseline."""
        report = BenchmarkReport(
            timestamp="20260406_roundtrip",
            suite="test_suite",
            models=["model_a"],
            runs_per_model=1,
            results={
                "model_a": ModelResult(
                    model="model_a",
                    metrics={"memory_formation_rate": 2.5, "causal_link_count": 4.0},
                    metrics_stddev={"memory_formation_rate": 0.3},
                    score=0.65,
                    passed=True,
                    expectations_passed=4,
                    expectations_total=5,
                ),
            },
            overall_ranking=["model_a"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_content = "name: test\npercepts: []\n"
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                runner = BenchmarkRunner(models=["model_a"], suite_path=f.name, output_dir=tmpdir)
            report_dir = runner.save_report(report)
            loaded = runner._load_baseline(str(report_dir / "benchmark_report.json"))

        assert loaded is not None
        assert loaded.suite == "test_suite"
        mr = loaded.results["model_a"]
        assert mr.metrics["memory_formation_rate"] == pytest.approx(2.5)
        assert mr.score == pytest.approx(0.65)
        assert mr.expectations_passed == 4
