"""#1310: the benchmark judges a scenario's expectations against what the run did.

Before: ``BenchmarkRunner.run`` validated every expectation against an EMPTY ``RecordingSink()``, with
``hippocampus=None``, no emitted tags and ``tool_stats={}``, because ``experiment.py::run_campaign`` dropped
the ``SimulationResult``'s actions, snapshot and introspector. So ``memory_formed``, ``action_count_range``
and ``pipeline_continued`` could never pass, for any model: the 1.3.2 checklist's "mistral-7b 0/3 on
quick_check" (2026-10-16) was this, not the model (its own report shows ``recall_seven-three-nine = 1.0``).

Owner decision 2026-10-17 (strict): ``passed`` requires every expectation met AND at least one threshold
applied, so the fix to what the expectations SEE has to land first.

The gate drives the shipped ``quick_check`` suite through the real ``BenchmarkRunner.run`` and
``run_campaign``; only ``start_simulation_mode`` (the LLM-driven sim) is replaced, by a run that did the work.
"""

from __future__ import annotations

from typing import Any

import pytest

QUICK_CHECK = "scenarios/benchmarks/quick_check.yaml"


def _hippocampus_holding(text: str) -> Any:
    from maxim.memory.encoding import EncodingSignals
    from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
    from maxim.memory.types import Perception

    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    h.capture(perception=Perception(observations={"text": text}), encoding=EncodingSignals.unmeasured("api"))
    return h


def _a_run_that_did_the_work(**kwargs: Any) -> Any:
    """What a healthy quick_check run returns: three turns delivered, an answer each, the code remembered."""
    from maxim.simulation.introspection import Observer
    from maxim.simulation.sim_types import SimulationResult

    turns = kwargs["pre_campaign_turns"]
    actions = [
        {"timestamp": float(i), "tool_name": "respond", "tool_args": {"message": "ok"}, "result_success": True}
        for i in range(len(turns))
    ]
    return SimulationResult(
        goal=kwargs["goal"],
        mode=kwargs["mode"],
        turns=len(turns),
        total_actions=len(actions),
        blocked_actions=0,
        duration_s=1.0,
        finish_reason="completed",
        # Each turn is sent just before its answer, as run_precampaign_turns records it (``sent_at``, #1391).
        campaign_analysis={
            "turns": [{"turn": i, "phase": t.get("phase", ""), "sent_at": i - 1.5} for i, t in enumerate(turns, 1)]
        },
        introspector=Observer(hippocampus=_hippocampus_holding("Kael: the vault combination is seven-three-nine")),
        actions=actions,
    )


@pytest.fixture
def healthy_sim(monkeypatch):
    import maxim.simulation.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "start_simulation_mode", _a_run_that_did_the_work)


def test_a_run_that_did_the_work_meets_quick_checks_expectations(healthy_sim) -> None:
    from maxim.simulation.benchmark import BenchmarkRunner

    result = BenchmarkRunner(models=["m"], suite_path=QUICK_CHECK).run().results["m"]

    assert (result.expectations_passed, result.expectations_total) == (3, 3)


# ── Strict ``passed`` (owner decision 2026-10-17) ───────────────────────────────────────────────


def _suite_with_threshold(tmp_path, threshold: dict | None):
    from pathlib import Path

    scenario = Path(tmp_path) / "a.yaml"
    scenario.write_text("name: a\n")
    lines = ["name: suite", "suite:", "  scenarios:", f"    - path: {scenario}", "      weight: 1.0"]
    if threshold is not None:
        lines += ["  scoring:", f"    recall_x: {threshold}"]
    suite = Path(tmp_path) / "suite.yaml"
    suite.write_text("\n".join(lines) + "\n")
    return str(suite)


def _aggregate(tmp_path, threshold, *runs):
    from maxim.simulation.benchmark import BenchmarkRunner

    return BenchmarkRunner(models=["m"], suite_path=_suite_with_threshold(tmp_path, threshold))._aggregate_runs(
        "m", list(runs)
    )


def _run(met: int, total: int, recall: float = 1.0, error: str | None = None):
    from maxim.simulation.benchmark import RunResult

    if error is not None:
        return RunResult(scenario="a", model="m", run_index=1, error=error)
    return RunResult(
        scenario="a",
        model="m",
        run_index=0,
        metrics={"recall_x": recall},
        expectations_passed=met,
        expectations_total=total,
    )


def test_every_expectation_met_and_a_threshold_met_passes(tmp_path) -> None:
    result = _aggregate(tmp_path, "{ pass: 1.0 }", _run(3, 3))
    assert result.passed is True and result.thresholds_checked == 1


def test_a_failed_threshold_fails(tmp_path) -> None:
    assert _aggregate(tmp_path, "{ pass: 1.0 }", _run(3, 3, recall=0.0)).passed is False


def test_a_missed_expectation_fails_even_when_the_thresholds_pass(tmp_path) -> None:
    assert _aggregate(tmp_path, "{ pass: 1.0 }", _run(2, 3)).passed is False


def test_no_threshold_applied_is_not_a_pass(tmp_path) -> None:
    result = _aggregate(tmp_path, None, _run(3, 3))
    assert result.thresholds_checked == 0 and result.passed is False


def test_a_run_that_errored_fails_the_model(tmp_path) -> None:
    assert _aggregate(tmp_path, "{ pass: 1.0 }", _run(3, 3), _run(0, 0, error="boom")).passed is False


def test_the_summary_line_gives_the_verdict_beside_both_results(tmp_path) -> None:
    from maxim.simulation.benchmark import BenchmarkReport

    result = _aggregate(tmp_path, "{ pass: 1.0 }", _run(2, 3))
    table = BenchmarkReport(
        timestamp="t", suite="s", models=["m"], runs_per_model=1, results={"m": result}
    ).summary_table()
    line = next(ln for ln in table.splitlines() if ln.strip().startswith("m:") and "expectations" in ln)
    assert line.strip() == "m: FAIL — expectations 2/3 met · thresholds PASS", line


def test_thresholds_passed_survives_save_and_load(tmp_path) -> None:
    from maxim.simulation.benchmark import BenchmarkReport, BenchmarkRunner

    runner = BenchmarkRunner(
        models=["m"], suite_path=_suite_with_threshold(tmp_path, "{ pass: 1.0 }"), output_dir=str(tmp_path / "out")
    )
    saved = _aggregate(tmp_path, "{ pass: 1.0 }", _run(2, 3))
    report_dir = runner.save_report(
        BenchmarkReport(timestamp="t", suite="s", models=["m"], runs_per_model=1, results={"m": saved})
    )
    loaded = runner._load_baseline(str(report_dir / "benchmark_report.json")).results["m"]
    assert (loaded.passed, loaded.thresholds_passed) == (False, True)


# ── The seam's pieces ──────────────────────────────────────────────────────────────────────────


def test_a_serialized_action_history_rebuilds_into_a_sink() -> None:
    from maxim.simulation.sinks import RecordingSink

    sink = RecordingSink.from_serialized(
        [{"timestamp": 1, "tool_name": "respond", "result_success": True, "utterance": "hi", "unknown_key": 1}]
    )
    (a,) = sink.actions
    assert (a.tool_name, a.result_success, a.utterance, a.timestamp) == ("respond", True, "hi", 1.0)


def test_a_serialized_action_without_its_identity_is_an_error_not_a_skip() -> None:
    from maxim.simulation.sinks import RecordingSink

    with pytest.raises(KeyError):
        RecordingSink.from_serialized([{"timestamp": 1.0}])


def test_delivered_tags_are_the_turns_delivered_without_error() -> None:
    from maxim.simulation.experiment import delivered_scenario_tags

    turns = [{"metadata": {"scenario_tag": t}} for t in ("seed", "interference", "recall")]
    records = {"turns": [{"turn": 1}, {"turn": 2, "error": "timeout"}, {"turn": 3}, {"turn": 9}]}
    assert delivered_scenario_tags(turns, records) == {"seed", "recall"}
    assert delivered_scenario_tags(turns, {}) is None  # no delivery reported is not "nothing delivered"


def test_an_undelivered_tag_fails_pipeline_continued(healthy_sim, monkeypatch) -> None:
    """The tags the expectation reads are the delivered ones: a run whose interference turn errored fails it."""
    import maxim.simulation.orchestrator as orchestrator
    from maxim.simulation.benchmark import BenchmarkRunner

    def _interference_lost(**kwargs):
        result = _a_run_that_did_the_work(**kwargs)
        result.campaign_analysis["turns"][1]["error"] = "timeout"
        return result

    monkeypatch.setattr(orchestrator, "start_simulation_mode", _interference_lost)
    result = BenchmarkRunner(models=["m"], suite_path=QUICK_CHECK).run().results["m"]
    assert (result.expectations_passed, result.expectations_total) == (2, 3)


def test_the_bio_checks_get_the_runs_snapshot(monkeypatch) -> None:
    """``Observer.benchmark_snapshot()`` already extends ``full_analysis()``, so the checks get it whole (#1310 review)."""
    import maxim.simulation.orchestrator as orchestrator
    from maxim.simulation.benchmark import BenchmarkRunner

    seen: dict = {}

    def _spy(
        expectations,
        sink,
        hippocampus=None,
        emitted_tags=None,
        *,
        subsystem_snapshot=None,
        tool_stats=None,
        tag_times=None,
    ):
        seen.update(subsystem_snapshot or {})
        return []

    def _with_snapshot(**kwargs):
        result = _a_run_that_did_the_work(**kwargs)
        result.subsystem_snapshot = {"hippocampus_stats": {"total_memories": 4}, "concepts": {"vault": 1}}
        return result

    monkeypatch.setattr("maxim.simulation.validation.validate_expectations", _spy)
    monkeypatch.setattr(orchestrator, "start_simulation_mode", _with_snapshot)
    BenchmarkRunner(models=["m"], suite_path=QUICK_CHECK).run()
    assert seen == {"hippocampus_stats": {"total_memories": 4}, "concepts": {"vault": 1}}


def test_quick_check_run_alone_passes_on_a_run_that_did_the_work(healthy_sim) -> None:
    """The scenario carries its own threshold (owner decision 2026-10-19), so the release check can pass."""
    from maxim.simulation.benchmark import BenchmarkRunner

    result = BenchmarkRunner(models=["m"], suite_path=QUICK_CHECK).run().results["m"]
    assert (result.thresholds_checked, result.thresholds_passed, result.passed) == (1, True, True)


def test_a_scenario_without_its_own_threshold_still_cannot_pass(tmp_path) -> None:
    from maxim.simulation.benchmark import BenchmarkRunner, RunResult

    (tmp_path / "s.yaml").write_text("name: s\n")
    runner = BenchmarkRunner(models=["m"], suite_path=str(tmp_path / "s.yaml"))
    run = RunResult(
        scenario="s", model="m", run_index=0, metrics={"x": 1.0}, expectations_passed=1, expectations_total=1
    )
    assert runner._aggregate_runs("m", [run]).passed is False


def test_an_empty_tag_set_means_the_tag_was_never_emitted() -> None:
    from maxim.simulation.sinks import RecordingSink
    from maxim.simulation.scenario_source import Expectation
    from maxim.simulation.validation import validate_expectations

    sink = RecordingSink.from_serialized([{"timestamp": 1.0, "tool_name": "respond", "result_success": True}])
    exp = Expectation(type="pipeline_continued", after_tag="interference")
    (empty,) = validate_expectations([exp], sink, emitted_tags=set())
    (unknown,) = validate_expectations([exp], sink, emitted_tags=None)
    # #1391: with no delivery time the order is unknown, so ``None`` (tags not reported) fails closed too.
    assert (empty.passed, unknown.passed) == (False, False)
    assert "never emitted" in empty.detail and "order unknown" in unknown.detail


def test_the_orchestrators_own_action_dicts_rebuild_with_their_blocks() -> None:
    """The exact keys ``orchestrator.py`` serializes into ``SimulationResult.actions``."""
    from maxim.simulation.sinks import RecordingSink

    shape = dict(tool_args={}, result_output=None, block_reason=None, utterance=None)
    sink = RecordingSink.from_serialized(
        [
            {"timestamp": 1.0, "tool_name": "respond", "result_success": True, "blocked": False, **shape},
            {**shape, "timestamp": 2.0, "tool_name": "bash", "result_success": False, "blocked": True,
             "block_reason": "forbidden"},
        ]
    )  # fmt: skip
    assert [a.tool_name for a in sink.find_blocked()] == ["bash"]
    assert sink.find_blocked()[0].block_reason == "forbidden"


def test_a_delivered_turn_without_a_tag_adds_none() -> None:
    from maxim.simulation.experiment import delivered_scenario_tags

    assert delivered_scenario_tags([{"metadata": {}}, {}], {"turns": [{"turn": 1}, {"turn": 2}]}) == set()


def test_a_run_without_an_introspector_has_no_hippocampus(monkeypatch) -> None:
    import maxim.simulation.orchestrator as orchestrator
    from maxim.simulation.experiment import run_campaign

    def _bare(**kwargs):
        result = _a_run_that_did_the_work(**kwargs)
        result.introspector = None
        return result

    monkeypatch.setattr(orchestrator, "start_simulation_mode", _bare)
    result = run_campaign(QUICK_CHECK)
    assert (
        result.hippocampus is None
        and len(result.actions) == 3
        and result.delivered_tags
        == {
            "seed",
            "interference",
            "recall",
        }
    )


from maxim.simulation.benchmark import RunResult  # noqa: E402  (the parametrize below builds one)


@pytest.mark.parametrize(
    ("result_kwargs", "expected"),
    [
        (
            {"passed": True, "thresholds_checked": 2},
            "m: n/a (pre-#1310 report) — expectations 0/0 met · thresholds PASS",
        ),
        (
            {"passed": False, "runs": [RunResult(scenario="a", model="m", run_index=0, error="boom")]},
            "m: FAIL — expectations 0/0 met · thresholds n/a (every run errored)",
        ),
    ],
)
def test_a_legacy_or_all_errored_result_does_not_claim_a_verdict(result_kwargs, expected) -> None:
    from maxim.simulation.benchmark import BenchmarkReport, ModelResult

    report = BenchmarkReport(
        timestamp="t", suite="s", models=["m"], runs_per_model=1, results={"m": ModelResult(model="m", **result_kwargs)}
    )
    line = next(
        ln for ln in report.summary_table().splitlines() if ln.strip().startswith("m:") and "expectations" in ln
    )
    assert line.strip() == expected
