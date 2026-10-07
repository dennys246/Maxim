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


@pytest.mark.xfail(strict=True, reason="D49(b): repeat runs fold with a running half-mean")
def test_repeat_runs_of_a_scenario_average_to_their_true_mean(tmp_path):
    a = _scenario(tmp_path, "a")
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=0.0), _run("a", 1, m1=0.0), _run("a", 2, m1=0.9)])
    assert result.per_scenario["a"]["m1"] == pytest.approx(0.3)


@pytest.mark.xfail(strict=True, reason="D49(a): a scenario's weight is parsed and never read")
def test_the_composite_is_the_weighted_mean_of_scenario_scores(tmp_path):
    a, b = _scenario(tmp_path, "a"), _scenario(tmp_path, "b")
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 2.0), (b, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0), _run("b", 0, m2=0.0)])
    assert result.score == pytest.approx(2.0 / 3.0)


@pytest.mark.xfail(strict=True, reason="D49(c): every collected metric scores, not the scenario's selection")
def test_only_a_scenarios_selected_metrics_score(tmp_path):
    a = _scenario(tmp_path, "a", metrics=["m1"])
    runner = BenchmarkRunner(models=["m"], suite_path=str(_suite(tmp_path, [(a, 1.0)])))
    result = runner._aggregate_runs("m", [_run("a", 0, m1=1.0, noise=0.0)])
    assert result.score == pytest.approx(1.0)


@pytest.mark.xfail(strict=True, reason="D49(c): a selected metric a run never emitted is not counted")
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


@pytest.mark.xfail(strict=True, reason="D49(d): a missing suite says only that it is missing")
def test_a_missing_suite_names_the_suites_that_exist(tmp_path):
    (tmp_path / "cognitive_suite.yaml").write_text("name: c\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        BenchmarkRunner(models=["m"], suite_path=str(tmp_path / "biosystem_suite.yaml"))
    message = str(excinfo.value)
    assert "biosystem_suite.yaml" in message and "cognitive_suite.yaml" in message
