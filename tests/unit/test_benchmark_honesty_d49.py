"""D49: the benchmark honours the suite format it ships (owner decision 2026-10-07).

The suite format promised a per-scenario ``weight`` and a per-scenario ``benchmark.metrics`` selection; the
runner parsed ``weight`` and never read it, scored every collected metric, and folded repeat runs with a
running half-mean ``(old + new) / 2`` (the true mean only for two runs). These tests load real suite YAMLs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maxim.simulation.benchmark import BenchmarkRunner, RunResult


def _scenario(tmp_path: Path, name: str, metrics: list[str] | None = None) -> Path:
    body = f"name: {name}\n"
    if metrics is not None:
        body += f"benchmark:\n  metrics: [{', '.join(metrics)}]\n"
    path = tmp_path / f"{name}.yaml"
    path.write_text(body)
    return path


def _suite(tmp_path: Path, entries: list[tuple[Path, float]]) -> Path:
    lines = ["name: suite", "suite:", "  scenarios:"]
    for path, weight in entries:
        lines += [f"    - path: {path}", f"      weight: {weight}"]
    suite = tmp_path / "suite.yaml"
    suite.write_text("\n".join(lines) + "\n")
    return suite


def _run(scenario: str, index: int, **metrics: float) -> RunResult:
    return RunResult(scenario=scenario, model="m", run_index=index, metrics=dict(metrics))


def test_repeat_runs_of_a_scenario_average_to_their_true_mean(tmp_path):
    a = _scenario(tmp_path, "a")
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=0.0), _run("a", 1, m1=0.0), _run("a", 2, m1=0.9)])
    assert result.per_scenario["a"]["m1"] == pytest.approx(0.3)


def test_the_composite_is_the_weighted_mean_of_scenario_scores(tmp_path):
    a, b = _scenario(tmp_path, "a"), _scenario(tmp_path, "b")
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 2.0), (b, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0), _run("b", 0, m2=0.0)])
    assert result.score == pytest.approx(2.0 / 3.0)


def test_only_a_scenarios_selected_metrics_score(tmp_path):
    a = _scenario(tmp_path, "a", metrics=["m1"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0, noise=0.0)])
    assert result.score == pytest.approx(1.0)


def test_a_selected_metric_the_run_did_not_emit_scores_zero(tmp_path):
    """Collectors emit some metrics only when non-zero (e.g. ``pain_signal_count``)."""
    a = _scenario(tmp_path, "a", metrics=["m1", "pain_signal_count"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0)])
    assert result.score == pytest.approx(0.5)


def test_a_single_scenario_without_a_selection_scores_every_metric(tmp_path):
    """Pinned before and after: no ``benchmark.metrics`` means every collected metric counts."""
    a = _scenario(tmp_path, "a")
    runner = BenchmarkRunner(models=["m"], suite_path=str(a))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0, m2=0.0)])
    assert result.score == pytest.approx(0.5)


def test_a_missing_suite_names_the_suites_that_exist(tmp_path):
    (tmp_path / "cognitive_suite.yaml").write_text("name: c\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        BenchmarkRunner(models=["m"], suite_path=str(tmp_path / "biosystem_suite.yaml"))
    message = str(excinfo.value)
    assert "biosystem_suite.yaml" in message and "cognitive_suite.yaml" in message


# -- review folds (three lenses, 2026-10-07) ------------------------------------------------------


def test_a_run_that_did_not_emit_a_metric_counts_zero_in_its_scenarios_mean(tmp_path):
    """Collectors emit some metrics only when non-zero: one run of three emitting 1.0 is a mean of 1/3."""
    a = _scenario(tmp_path, "a", metrics=["pain_signal_count"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    runs = [_run("a", 0, pain_signal_count=1.0), _run("a", 1, other=0.0), _run("a", 2, other=0.0)]
    result = runner._aggregate_runs("m", runs)
    assert result.per_scenario["a"]["pain_signal_count"] == pytest.approx(1 / 3)
    assert result.score == pytest.approx(1 / 3)


def test_a_scenario_with_no_successful_run_scores_zero_at_its_weight(tmp_path):
    a, b = _scenario(tmp_path, "a"), _scenario(tmp_path, "b")
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0), (b, 1.0)])))
    crashed = RunResult(scenario="b", model="m", run_index=0, error="boom")
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0), crashed])
    assert result.score == pytest.approx(0.5)


def test_an_unemitted_lower_is_better_metric_is_not_a_perfect_score(tmp_path):
    a = _scenario(tmp_path, "a", metrics=["hallucination_rate"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    assert runner._aggregate_runs("m", [_run("a", 0, other=1.0)]).score == pytest.approx(0.0)


def test_latency_is_reported_but_not_scored(tmp_path):
    a = _scenario(tmp_path, "a")
    runner = BenchmarkRunner(models=["m"], suite_path=str(a))
    slow = runner._aggregate_runs("m", [_run("a", 0, m1=0.5, action_latency_p50_ms=9000.0)])
    fast = runner._aggregate_runs("m", [_run("a", 0, m1=0.5, action_latency_p50_ms=5.0)])
    assert slow.score == fast.score == pytest.approx(0.5)
    assert slow.metrics["action_latency_p50_ms"] == pytest.approx(9000.0)


def test_a_single_scenarios_own_selection_is_honoured(tmp_path):
    a = _scenario(tmp_path, "a", metrics=["m1"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(a))
    assert runner._aggregate_runs("m", [_run("a", 0, m1=1.0, noise=0.0)]).score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("weights", "match"),
    [((1.0, -1.0), "negative weight"), ((0.0, 0.0), "no weight")],
)
def test_a_suite_with_invalid_weights_is_refused(tmp_path, weights, match):
    a, b = _scenario(tmp_path, "a"), _scenario(tmp_path, "b")
    with pytest.raises(ValueError, match=match):
        BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, list(zip((a, b), weights)))))


def test_two_suite_scenarios_with_one_name_are_refused(tmp_path):
    a = _scenario(tmp_path, "a")
    other = tmp_path / "sub"
    other.mkdir()
    twin = _scenario(other, "a")
    with pytest.raises(ValueError, match="two scenarios named 'a'"):
        BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0), (twin, 1.0)])))


@pytest.mark.parametrize("body", [None, "name: [unclosed\n"], ids=["missing", "unparseable"])
def test_a_suite_whose_child_cannot_be_loaded_is_refused_at_construction(tmp_path, body):
    child = tmp_path / "child.yaml"
    if body is not None:
        child.write_text(body)
    with pytest.raises(ValueError, match="child.yaml cannot be loaded"):
        BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(child, 1.0)])))


def test_the_shipped_cognitive_suite_registers_each_scenarios_weight_and_selection():
    root = Path(__file__).resolve().parents[2]
    import os

    cwd = os.getcwd()
    os.chdir(root)  # the suite's child paths are repo-relative
    try:
        runner = BenchmarkRunner(models=["m"], suite_path="scenarios/benchmarks/cognitive_suite.yaml")
    finally:
        os.chdir(cwd)
    assert runner._scenario_weight["hippocampal_recall_short"] == pytest.approx(2.0)
    assert runner._scenario_weight["causal_learning"] == pytest.approx(1.5)
    assert "causal_link_count" in (runner._scenario_metrics["causal_learning"] or [])


def test_a_report_is_stamped_with_its_score_scheme_and_an_unstamped_baseline_warns(tmp_path, caplog):
    import json
    import logging

    from maxim.simulation.benchmark import SCORE_SCHEME, BenchmarkReport

    a = _scenario(tmp_path, "a")
    runner = BenchmarkRunner(models=["m"], suite_path=str(a), output_dir=str(tmp_path / "out"))
    report = BenchmarkReport(timestamp="t", suite="s", models=["m"], runs_per_model=1, duration_s=0.0)
    report_dir = Path(runner.save_report(report))
    data = json.loads((report_dir / "benchmark_report.json").read_text())
    assert data["score_scheme"] == SCORE_SCHEME
    data.pop("score_scheme")
    old = tmp_path / "old.json"
    old.write_text(json.dumps(data))
    with caplog.at_level(logging.WARNING, logger="maxim.simulation.benchmark"):
        runner._load_baseline(str(old))
    assert any("not comparable" in r.getMessage() for r in caplog.records)


def test_a_single_scenario_with_zero_weight_is_refused(tmp_path):
    a = tmp_path / "a.yaml"
    a.write_text("name: a\nbenchmark:\n  weight: 0\n")
    with pytest.raises(ValueError, match="no weight"):
        BenchmarkRunner(models=["m"], suite_path=str(a))


def test_rankings_sort_a_lower_is_better_metric_ascending(tmp_path):
    from maxim.simulation.benchmark import ModelResult

    runner = BenchmarkRunner(models=["m"], suite_path=str(_scenario(tmp_path, "a")))
    results = {
        "slow": ModelResult(model="slow", metrics={"action_latency_p50_ms": 900.0, "m1": 0.9}),
        "fast": ModelResult(model="fast", metrics={"action_latency_p50_ms": 10.0, "m1": 0.1}),
    }
    rankings = runner._compute_rankings(results)
    assert rankings["action_latency_p50_ms"] == ["fast", "slow"]
    assert rankings["m1"] == ["slow", "fast"]
