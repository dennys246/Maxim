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
        campaign_analysis={"turns": [{"turn": i, "phase": t.get("phase", "")} for i, t in enumerate(turns, 1)]},
        introspector=Observer(hippocampus=_hippocampus_holding("Kael: the vault combination is seven-three-nine")),
        actions=actions,
    )


@pytest.fixture
def healthy_sim(monkeypatch):
    import maxim.simulation.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "start_simulation_mode", _a_run_that_did_the_work)


@pytest.mark.xfail(strict=True, reason="#1310: expectations are validated against an empty sink and no hippocampus")
def test_a_run_that_did_the_work_meets_quick_checks_expectations(healthy_sim) -> None:
    from maxim.simulation.benchmark import BenchmarkRunner

    result = BenchmarkRunner(models=["m"], suite_path=QUICK_CHECK).run().results["m"]

    assert (result.expectations_passed, result.expectations_total) == (3, 3)
