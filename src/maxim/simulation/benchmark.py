"""BenchmarkRunner — multi-model comparative testing.

Automates running campaign scenarios across multiple LLM models,
computing tiered metrics (LLM behavior, cognitive architecture,
embodiment), and producing comparative reports.

Wraps ``run_campaign()`` from ``experiment.py`` for each model/scenario
combination, aggregates results across runs, and scores against
configurable thresholds.

CLI entry: ``maxim --sim benchmark --models X,Y --campaign Z``

Example::

    runner = BenchmarkRunner(
        models=["mistral-7b", "qwen2.5-14b"],
        suite_path="scenarios/benchmarks/cognitive_suite.yaml",
        runs=3,
    )
    report = runner.run()
    print(report.summary_table())
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class RunResult:
    """Result from a single scenario run on a single model."""

    scenario: str
    model: str
    run_index: int
    metrics: dict[str, float] = field(default_factory=dict)
    expectations_passed: int = 0
    expectations_total: int = 0
    duration_s: float = 0.0
    turns: int = 0
    error: str | None = None


@dataclass
class ModelResult:
    """Aggregated results for one model across all runs."""

    model: str
    runs: list[RunResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_stddev: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    passed: bool = False
    expectations_passed: int = 0
    expectations_total: int = 0
    per_scenario: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report across all models."""

    timestamp: str
    suite: str
    models: list[str]
    runs_per_model: int
    results: dict[str, ModelResult] = field(default_factory=dict)
    rankings: dict[str, list[str]] = field(default_factory=dict)
    overall_ranking: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    baseline_comparison: dict[str, dict[str, dict[str, float]]] | None = None

    def summary_table(self) -> str:
        """Generate a human-readable summary table with tiered metrics."""
        w = max(60, 20 + 16 * len(self.models))
        lines = [
            "=" * w,
            f"  BENCHMARK REPORT — {self.suite}",
            f"  Models: {', '.join(self.models)}",
            f"  Runs per model: {self.runs_per_model} | Duration: {self.duration_s:.1f}s",
            "=" * w,
            "",
        ]

        if not self.results:
            lines.append("  No results.")
            return "\n".join(lines)

        # Collect all metric names across all models
        all_metrics: set[str] = set()
        for mr in self.results.values():
            all_metrics.update(mr.metrics.keys())

        # Group metrics by tier
        tier1_keys = [k for k in sorted(all_metrics) if k in _TIER1_METRICS]
        tier2_keys = [k for k in sorted(all_metrics) if k in _TIER2_METRICS or k.startswith("recall_")]
        other_keys = [k for k in sorted(all_metrics) if k not in tier1_keys and k not in tier2_keys]

        def _fmt_val(model: str, key: str) -> str:
            """Format a metric value with optional stddev and baseline delta."""
            mr = self.results.get(model)
            if not mr:
                return "—"
            val = mr.metrics.get(key, 0)
            sd = mr.metrics_stddev.get(key, 0)
            if sd > 0.001:
                base = f"{val:.2f}+-{sd:.2f}"
            else:
                base = f"{val:.2f}"
            # Append baseline delta if available
            if self.baseline_comparison and model in self.baseline_comparison:
                delta_info = self.baseline_comparison[model].get(key)
                if delta_info is not None:
                    delta = delta_info["delta"]
                    direction = delta_info["direction"]
                    sign = "+" if delta >= 0 else ""
                    arrow = "↑" if direction == "improved" else "↓" if direction == "regressed" else "="
                    base += f" ({sign}{delta:.2f} {arrow})"
            return base

        for section, keys in [
            ("TIER 1 — LLM BEHAVIOR", tier1_keys),
            ("TIER 2 — COGNITIVE ARCHITECTURE", tier2_keys),
        ]:
            if keys:
                lines.append(f"  {section}")
                for key in keys:
                    vals = "  ".join(_fmt_val(m, key) for m in self.models)
                    lines.append(f"    {key:<30s}{vals}")
                lines.append("")

        if other_keys:
            lines.append("  OTHER METRICS")
            for key in other_keys:
                vals = "  ".join(_fmt_val(m, key) for m in self.models)
                lines.append(f"    {key:<30s}{vals}")
            lines.append("")

        # Per-scenario breakdown (if multiple scenarios)
        has_per_scenario = any(mr.per_scenario for mr in self.results.values())
        if has_per_scenario:
            lines.append("  PER-SCENARIO BREAKDOWN")
            # Collect all scenario names
            all_scenarios: set[str] = set()
            for mr in self.results.values():
                all_scenarios.update(mr.per_scenario.keys())
            for scenario in sorted(all_scenarios):
                lines.append(f"    {scenario}")
                # Show key metrics per model for this scenario
                for m in self.models:
                    mr = self.results.get(m)
                    if mr and scenario in mr.per_scenario:
                        sm = mr.per_scenario[scenario]
                        key_vals = ", ".join(f"{k}={v:.2f}" for k, v in sorted(sm.items())[:5])
                        lines.append(f"      {m}: {key_vals}")
            lines.append("")

        # Expectations summary
        lines.append("  EXPECTATIONS")
        for m in self.models:
            if m in self.results:
                mr = self.results[m]
                status = "PASS" if mr.passed else "FAIL"
                lines.append(f"    {m}: {mr.expectations_passed}/{mr.expectations_total} passed [{status}]")
        lines.append("")

        # Overall ranking
        if self.overall_ranking:
            lines.append("  OVERALL RANKING")
            for i, m in enumerate(self.overall_ranking, 1):
                if m in self.results:
                    lines.append(f"    {i}. {m:<20s} score: {self.results[m].score:.2f}")
            lines.append("")

        lines.append("=" * w)
        return "\n".join(lines)


# ── Known metric names for tier classification ───────────────────────

_TIER1_METRICS = {
    "hallucination_rate",
    "alias_redirect_rate",
    "correct_tool_usage_rate",
    "json_compliance_rate",
    "think_before_act_rate",
    "reasoning_depth",
    "action_latency_p50_ms",
    "action_latency_p95_ms",
    "token_efficiency",
    "cost_per_turn",
}

_TIER2_METRICS = {
    "memory_formation_rate",
    "associative_graph_density",
    "concept_formation_rate",
    "concept_avg_confidence",
    "causal_link_count",
    "learning_efficiency",
    "causal_diversity",
    "observation_density",
    "pain_signal_count",
    "type_token_ratio",
    "engagement_depth",
}


# ── BenchmarkRunner ──────────────────────────────────────────────────


class BenchmarkRunner:
    """Run benchmark scenarios across multiple models and aggregate results.

    Usage::

        runner = BenchmarkRunner(
            models=["mistral-7b", "qwen2.5-14b"],
            suite_path="scenarios/benchmarks/cognitive_suite.yaml",
        )
        report = runner.run()
    """

    def __init__(
        self,
        models: list[str],
        suite_path: str,
        *,
        runs: int = 1,
        output_dir: str | None = None,
        baseline_path: str | None = None,
        seed_keywords: list[str] | None = None,
        mode: str = "generative",
        max_turns: int = 50,
        response_timeout: float = 60.0,
        debug: bool = False,
    ) -> None:
        self.models = models
        self.suite_path = suite_path
        self.runs = max(runs, 1)
        if output_dir is None:
            from maxim.utils.paths import benchmarks_dir

            self.output_dir = benchmarks_dir()
        else:
            self.output_dir = Path(output_dir)
        self.baseline_path = baseline_path
        self.seed_keywords = seed_keywords or []
        self.mode = mode
        self.max_turns = max_turns
        self.response_timeout = response_timeout
        self.debug = debug

        # Load suite or single scenario
        self._scoring: dict[str, dict] = {}
        self._scenarios = self._load_suite(suite_path)

    def _load_suite(self, path: str) -> list[dict[str, Any]]:
        """Load benchmark suite or single scenario via unified YAML loader.

        Uses ``load_scenario()`` from ``scenario_source.py`` which parses
        the unified schema (percepts, expectations, metadata, benchmark,
        suite, config sections).

        Returns a list of scenario descriptors:
        [{"path": str, "weight": float, "seed_keywords": list, "config": dict|None}, ...]
        """
        from maxim.simulation.scenario_source import load_scenario

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Suite/scenario not found: {path}")

        defn = load_scenario(p)

        # Suite file — references child scenarios
        if defn.suite:
            suite = defn.suite
            self._scoring = suite.get("scoring", {})
            scenarios = []
            for entry in suite.get("scenarios", []):
                scenarios.append(
                    {
                        "path": entry["path"],
                        "weight": entry.get("weight", 1.0),
                        "seed_keywords": entry.get("seed_keywords", []),
                        "config": entry.get("config"),
                    }
                )
            if not scenarios:
                raise ValueError(f"Suite {path} has no scenarios listed")
            logger.info(
                "Loaded benchmark suite: %s (%d scenarios, %d scoring thresholds)",
                defn.name,
                len(scenarios),
                len(self._scoring),
            )
            return scenarios

        # Single scenario with optional benchmark section
        seed_kw = []
        weight = 1.0
        if defn.benchmark:
            seed_kw = defn.benchmark.get("seed_keywords", [])
            weight = defn.benchmark.get("weight", 1.0)
        self._scoring = {}

        return [
            {
                "path": path,
                "weight": weight,
                "seed_keywords": seed_kw,
                "config": defn.config,
                "tags": defn.tags,
            }
        ]

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark suite across all models.

        For each model:
            For each run (1..N):
                For each scenario:
                    run_campaign() → compute metrics → validate expectations
            Aggregate across runs (mean, stddev)
        Score against thresholds, rank models.
        """
        from maxim.simulation.experiment import run_campaign
        from maxim.simulation.scenario_source import load_scenario
        from maxim.simulation.validation import validate_expectations
        from maxim.simulation.sinks import RecordingSink

        start = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        all_model_results: dict[str, ModelResult] = {}
        total_models = len(self.models)

        for model_idx, model in enumerate(self.models):
            from maxim.simulation.sim_logger import display_status, display_summary

            display_status(f"Model: {model} ({model_idx + 1}/{total_models})")
            model_runs: list[RunResult] = []

            for run_idx in range(self.runs):
                for scenario_desc in self._scenarios:
                    scenario_path = scenario_desc["path"]
                    scenario_name = Path(scenario_path).stem
                    scenario_seed_kw = scenario_desc.get("seed_keywords", self.seed_keywords)

                    display_status(f"Scenario: {scenario_name} (run {run_idx + 1}/{self.runs})")

                    try:
                        # Load scenario definition for expectations
                        defn = load_scenario(Path(scenario_path))

                        # Run the campaign
                        exp_result = run_campaign(
                            campaign=scenario_path,
                            aut_model=model,
                            seed_keywords=scenario_seed_kw,
                            goal=f"benchmark: {scenario_name}",
                            mode=self.mode,
                            max_turns=self.max_turns,
                            response_timeout=self.response_timeout,
                            debug=self.debug,
                        )

                        # Compute metrics from the enriched SimulationResult
                        # (Phase 0a populated tool_stats, actions, subsystem_snapshot)
                        metrics = self._compute_metrics(exp_result, defn)

                        # Validate expectations if defined
                        exp_passed = 0
                        exp_total = 0
                        if defn.expectations:
                            # Build a minimal RecordingSink from serialized actions
                            sink = RecordingSink()
                            # We can't reconstruct full ActionRecords from dicts,
                            # but expectations that need the sink use find_blocked/
                            # find_actions which need real ActionRecord objects.
                            # For bio-system expectations, we pass snapshot + tool_stats.
                            exp_results = validate_expectations(
                                defn.expectations,
                                sink,
                                subsystem_snapshot=exp_result.analysis,
                                tool_stats={},  # TODO: carry tool_stats through ExperimentResult
                            )
                            exp_passed = sum(1 for r in exp_results if r.passed)
                            exp_total = len(exp_results)

                        run_result = RunResult(
                            scenario=scenario_name,
                            model=model,
                            run_index=run_idx,
                            metrics=metrics,
                            expectations_passed=exp_passed,
                            expectations_total=exp_total,
                            duration_s=exp_result.sim_duration_s,
                            turns=exp_result.sim_turns,
                        )
                        model_runs.append(run_result)

                        duration = run_result.duration_s
                        turns = run_result.turns
                        passed = run_result.expectations_passed
                        total = run_result.expectations_total
                        display_status(f"  OK {duration:.1f}s — {turns} turns — {passed}/{total} expectations")

                    except Exception as e:
                        logger.warning("Scenario %s failed on %s: %s", scenario_name, model, e)
                        error_msg = str(e)
                        model_runs.append(
                            RunResult(
                                scenario=scenario_name,
                                model=model,
                                run_index=run_idx,
                                error=error_msg,
                            )
                        )
                        display_status(f"  FAIL: {error_msg}")

            # Aggregate across runs
            model_result = self._aggregate_runs(model, model_runs)
            all_model_results[model] = model_result
            display_summary([f"Score: {model_result.score:.3f}"])

        total_duration = time.time() - start
        display_summary([f"Benchmark complete: {total_duration:.1f}s across {total_models} model(s)"])

        # Build report
        report = BenchmarkReport(
            timestamp=timestamp,
            suite=Path(self.suite_path).stem,
            models=self.models,
            runs_per_model=self.runs,
            results=all_model_results,
            duration_s=total_duration,
        )

        # Rank models
        report.overall_ranking = sorted(
            self.models,
            key=lambda m: all_model_results.get(m, ModelResult(model=m)).score,
            reverse=True,
        )
        report.rankings = self._compute_rankings(all_model_results)

        # Baseline comparison (if a baseline path was provided)
        if self.baseline_path:
            baseline = self._load_baseline(self.baseline_path)
            if baseline is not None:
                report.baseline_comparison = self._compare_to_baseline(report, baseline)
                logger.info(
                    "Baseline comparison computed against %s (%d models compared)",
                    self.baseline_path,
                    len(report.baseline_comparison or {}),
                )

        return report

    # ── Metric computation ───────────────────────────────────────────

    def _compute_metrics(self, result: Any, scenario: Any) -> dict[str, float]:
        """Compute tiered metrics from an ExperimentResult.

        Uses data populated by Phase 0a (SimulationResult enrichment):
        - result.analysis (from Observer.full_analysis/benchmark_snapshot)
        - Plus ExperimentResult convenience properties
        """
        metrics: dict[str, float] = {}
        analysis = result.analysis or {}
        stats = analysis.get("system_stats", {})
        hippo = analysis.get("hippocampus_stats", {})
        nac = analysis.get("nac_stats", {})
        pain = analysis.get("pain_stats", {})
        turns = max(result.sim_turns, 1)

        # ── Tier 1: LLM Behavior ────────────────────────────────
        # (tool_stats not yet on ExperimentResult — uses analysis data)

        # ── Tier 2: Cognitive Architecture ───────────────────────
        # Memory system
        mem_count = hippo.get("total_memories", 0) or stats.get("hippocampus_memories", 0)
        metrics["memory_formation_rate"] = mem_count / turns

        graph_nodes = hippo.get("graph_nodes", 0)
        graph_edges = hippo.get("graph_edges", 0)
        if graph_nodes > 0:
            metrics["associative_graph_density"] = graph_edges / graph_nodes

        atl_concepts = stats.get("atl_concepts", 0)
        if atl_concepts:
            metrics["concept_formation_rate"] = atl_concepts / turns

        # Causal learning
        total_links = nac.get("total_links", 0) or stats.get("nac_causal_links", 0)
        metrics["causal_link_count"] = float(total_links)
        if total_links > 0:
            metrics["learning_efficiency"] = total_links / max(turns, 1)
            event_sigs = nac.get("event_signatures", 0)
            if event_sigs:
                metrics["causal_diversity"] = event_sigs / total_links
            total_obs = nac.get("total_observations", 0)
            if total_obs:
                metrics["observation_density"] = total_obs / total_links

        # Pain
        pain_count = pain.get("total_pain_signals", 0)
        if pain_count:
            metrics["pain_signal_count"] = float(pain_count)

        # Keyword recall
        recall_data = analysis.get("memory_recall", {})
        for kw, data in recall_data.items():
            metrics[f"recall_{kw}"] = 1.0 if data.get("count", 0) > 0 else 0.0

        # ── Tier 3: auto-detected ────────────────────────────────
        # (filled when Embodiment ships — introspector.embodiment_stats())

        return metrics

    def _count_think_chains(self, actions: list[dict]) -> int:
        """Count turns where 'think' preceded another action."""
        chains = 0
        for i in range(len(actions) - 1):
            if actions[i].get("tool_name") == "think" and actions[i + 1].get("tool_name") != "think":
                chains += 1
        return chains

    def _compute_ttr(self, actions: list[dict]) -> float:
        """Compute type-token ratio across action outputs."""
        all_tokens: list[str] = []
        for a in actions:
            text = str(a.get("result_output", ""))
            all_tokens.extend(text.lower().split())
        if not all_tokens:
            return 0.0
        return len(set(all_tokens)) / len(all_tokens)

    # ── Aggregation ──────────────────────────────────────────────

    def _aggregate_runs(self, model: str, runs: list[RunResult]) -> ModelResult:
        """Aggregate metrics across multiple runs of the same model."""
        successful = [r for r in runs if r.error is None]
        if not successful:
            return ModelResult(model=model, runs=runs)

        # Collect all metric keys
        all_keys: set[str] = set()
        for r in successful:
            all_keys.update(r.metrics.keys())

        # Compute mean and stddev for each metric
        aggregated: dict[str, float] = {}
        stddevs: dict[str, float] = {}
        for key in all_keys:
            values = [r.metrics.get(key, 0.0) for r in successful]
            aggregated[key] = statistics.mean(values) if values else 0.0
            stddevs[key] = statistics.stdev(values) if len(values) > 1 else 0.0

        # Per-scenario breakdown
        per_scenario: dict[str, dict[str, float]] = {}
        for r in successful:
            if r.scenario not in per_scenario:
                per_scenario[r.scenario] = {}
            for k, v in r.metrics.items():
                # Average across runs for same scenario
                if k in per_scenario[r.scenario]:
                    per_scenario[r.scenario][k] = (per_scenario[r.scenario][k] + v) / 2
                else:
                    per_scenario[r.scenario][k] = v

        # Expectations
        total_passed = sum(r.expectations_passed for r in successful)
        total_expected = sum(r.expectations_total for r in successful)

        # Composite score (equal weight T1:T2, normalized 0-1)
        score = self._compute_composite_score(aggregated)

        # Check pass/fail against suite scoring thresholds
        passed = self._check_thresholds(aggregated)

        return ModelResult(
            model=model,
            runs=runs,
            metrics=aggregated,
            metrics_stddev=stddevs,
            score=score,
            passed=passed,
            expectations_passed=total_passed,
            expectations_total=total_expected,
            per_scenario=per_scenario,
        )

    def _compute_composite_score(self, metrics: dict[str, float]) -> float:
        """Compute a weighted composite score from metrics.

        Currently uses simple normalization: each metric contributes
        its value (capped at 1.0) divided by total metrics.
        """
        if not metrics:
            return 0.0

        # Metrics where lower is better
        invert = {"hallucination_rate", "alias_redirect_rate", "cost_per_turn"}

        score_sum = 0.0
        count = 0
        for key, value in metrics.items():
            if key.startswith("recall_"):
                # Binary: 0 or 1
                score_sum += value
            elif key in invert:
                score_sum += max(0.0, 1.0 - value)
            else:
                # Cap at 1.0 for ratio metrics, allow higher for counts
                score_sum += min(value, 1.0) if value <= 10 else min(value / 100, 1.0)
            count += 1

        return score_sum / max(count, 1)

    def _check_thresholds(self, metrics: dict[str, float]) -> bool:
        """Check if metrics meet suite-level scoring thresholds."""
        if not self._scoring:
            return True  # No thresholds defined = auto-pass

        for metric_name, threshold in self._scoring.items():
            value = metrics.get(metric_name)
            if value is None:
                continue
            if "pass" in threshold and value < threshold["pass"]:
                return False
            if "pass_above" in threshold and value < threshold["pass_above"]:
                return False
            if "pass_below" in threshold and value > threshold["pass_below"]:
                return False
        return True

    def _compute_rankings(self, results: dict[str, ModelResult]) -> dict[str, list[str]]:
        """Rank models per metric."""
        if not results:
            return {}

        all_keys: set[str] = set()
        for mr in results.values():
            all_keys.update(mr.metrics.keys())

        invert = {"hallucination_rate", "alias_redirect_rate", "cost_per_turn"}
        rankings: dict[str, list[str]] = {}
        for key in all_keys:
            ranked = sorted(
                results.keys(),
                key=lambda m: results[m].metrics.get(key, 0),
                reverse=(key not in invert),
            )
            rankings[key] = ranked

        return rankings

    # ── Baseline comparison ────────────────────────────────────────

    # Metrics where a *lower* value means better performance.
    _LOWER_IS_BETTER: set[str] = {
        "hallucination_rate",
        "alias_redirect_rate",
        "cost_per_turn",
        "action_latency_p50_ms",
        "action_latency_p95_ms",
    }

    def _load_baseline(self, path: str) -> BenchmarkReport | None:
        """Load a previously saved benchmark_report.json as a BenchmarkReport.

        Returns ``None`` (with a warning) if the file is missing or corrupt.
        """
        import json as _json

        p = Path(path)
        if not p.exists():
            logger.warning("Baseline file not found: %s — skipping comparison", path)
            return None

        try:
            with open(p) as f:
                data = _json.load(f)
        except (OSError, _json.JSONDecodeError) as exc:
            logger.warning("Failed to read baseline %s: %s — skipping comparison", path, exc)
            return None

        # Reconstruct ModelResult objects from serialized data
        results: dict[str, ModelResult] = {}
        for model, mr_data in data.get("results", {}).items():
            results[model] = ModelResult(
                model=mr_data.get("model", model),
                metrics=mr_data.get("metrics", {}),
                metrics_stddev=mr_data.get("metrics_stddev", {}),
                score=mr_data.get("score", 0.0),
                passed=mr_data.get("passed", False),
                expectations_passed=mr_data.get("expectations_passed", 0),
                expectations_total=mr_data.get("expectations_total", 0),
                per_scenario=mr_data.get("per_scenario", {}),
            )

        return BenchmarkReport(
            timestamp=data.get("timestamp", ""),
            suite=data.get("suite", ""),
            models=data.get("models", []),
            runs_per_model=data.get("runs_per_model", 0),
            results=results,
            rankings=data.get("rankings", {}),
            overall_ranking=data.get("overall_ranking", []),
            duration_s=data.get("duration_s", 0.0),
            baseline_comparison=data.get("baseline_comparison"),
        )

    def _compare_to_baseline(
        self, report: BenchmarkReport, baseline: BenchmarkReport
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compute per-model, per-metric deltas between *report* and *baseline*.

        Returns::

            {
                "model_name": {
                    "metric_name": {"delta": float, "direction": "improved"|"regressed"|"unchanged"},
                    ...
                },
                ...
            }

        Metrics present in only one side are silently skipped.
        """
        comparison: dict[str, dict[str, dict[str, float]]] = {}

        for model, current_mr in report.results.items():
            baseline_mr = baseline.results.get(model)
            if baseline_mr is None:
                logger.info("Model %s not in baseline — skipping baseline comparison for it", model)
                continue

            model_deltas: dict[str, dict[str, float]] = {}
            # Only compare metrics present in both current and baseline
            shared_keys = set(current_mr.metrics.keys()) & set(baseline_mr.metrics.keys())
            for key in shared_keys:
                delta = current_mr.metrics[key] - baseline_mr.metrics[key]
                if abs(delta) < 1e-9:
                    direction = "unchanged"
                elif key in self._LOWER_IS_BETTER:
                    direction = "improved" if delta < 0 else "regressed"
                else:
                    direction = "improved" if delta > 0 else "regressed"
                model_deltas[key] = {"delta": delta, "direction": direction}

            if model_deltas:
                comparison[model] = model_deltas

        return comparison

    # ── Persistence ──────────────────────────────────────────────

    def save_report(self, report: BenchmarkReport) -> Path:
        """Save report to output directory. Returns the report path."""

        report_dir = self.output_dir / report.timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        report_path = report_dir / "benchmark_report.json"
        report_data = {
            "timestamp": report.timestamp,
            "suite": report.suite,
            "models": report.models,
            "runs_per_model": report.runs_per_model,
            "duration_s": report.duration_s,
            "overall_ranking": report.overall_ranking,
            "results": {
                model: {
                    "model": mr.model,
                    "metrics": mr.metrics,
                    "metrics_stddev": mr.metrics_stddev,
                    "score": mr.score,
                    "passed": mr.passed,
                    "expectations_passed": mr.expectations_passed,
                    "expectations_total": mr.expectations_total,
                    "per_scenario": mr.per_scenario,
                }
                for model, mr in report.results.items()
            },
            "rankings": report.rankings,
        }
        if report.baseline_comparison is not None:
            report_data["baseline_comparison"] = report.baseline_comparison
        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(str(report_path), with_format_version(report_data))

        # Markdown summary
        summary_path = report_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# Benchmark Report — {report.suite}\n\n")
            f.write(f"**Generated:** {report.timestamp}\n")
            f.write(f"**Models:** {', '.join(report.models)}\n")
            f.write(f"**Runs per model:** {report.runs_per_model}\n")
            f.write(f"**Duration:** {report.duration_s:.1f}s\n\n")
            f.write("```\n")
            f.write(report.summary_table())
            f.write("\n```\n")

        logger.info("Benchmark report saved: %s", report_dir)
        return report_dir

    # ── Phase 7: Research protocol integration ───────────────────

    def write_paper(self, report: BenchmarkReport, output_dir: Path) -> Path | None:
        """Generate a comparative research paper from benchmark results.

        Uses the existing WriterAgent pipeline from the research protocol.
        Converts benchmark metrics into experiment log entries.

        Returns the paper path, or None if generation fails.
        """
        try:
            from maxim.mesh.bus import LocalMessageBus
            from maxim.simulation.research_agents import WriterAgent
            from maxim.simulation.research_tools import ExperimentLog

            exp_log = ExperimentLog(session_dir=output_dir, nickname="benchmark")

            for model, mr in report.results.items():
                metrics_summary = ", ".join(f"{k}={v:.2f}" for k, v in sorted(mr.metrics.items())[:10])
                exp_log.record(
                    hypothesis=f"Model {model} performs well on {report.suite}",
                    method=f"Benchmark suite: {report.suite}, {report.runs_per_model} run(s)",
                    result=metrics_summary,
                    conclusion=f"Score: {mr.score:.2f}, passed: {mr.passed}",
                    tags=["benchmark", model],
                    metrics=mr.metrics,
                )

            from maxim.models.language.router import LLMRouter

            try:
                router = LLMRouter()
            except Exception:
                logger.warning("No LLM available for paper generation")
                return None

            bus = LocalMessageBus()
            writer = WriterAgent(
                llm=router,
                experiment_log=exp_log,
                bus=bus,
                session_dir=output_dir,
            )

            goal = (
                f"Write a comparative analysis of {len(report.models)} LLM models "
                f"benchmarked on the {report.suite} suite. "
                f"Models tested: {', '.join(report.models)}. "
                f"Include methodology, per-model results, and ranking analysis."
            )

            draft = writer.run(goal)
            paper_path = output_dir / "paper.md"
            draft.save(paper_path)
            logger.info("Benchmark paper generated: %s", paper_path)
            return paper_path

        except Exception as e:
            logger.warning("Paper generation failed: %s", e)
            return None

    # ── Phase 9: Tier 3 embodiment hooks ─────────────────────────

    def _collect_tier3_metrics(self, result: Any) -> dict[str, float]:
        """Collect Tier 3 (Embodiment) metrics if available.

        Auto-detected: when Embodiment Core ships and adds
        ``embodiment_stats()`` to the Observer, this method
        picks up those metrics without any benchmark code changes.

        Returns an empty dict pre-embodiment.
        """
        introspector = getattr(result, "introspector", None)
        if introspector is None:
            return {}

        if not hasattr(introspector, "embodiment_stats"):
            return {}

        try:
            return introspector.embodiment_stats()
        except Exception:
            return {}
