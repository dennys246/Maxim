"""#1200 Stage 2: a tool failure that RAN is booked once, by the tool-pain bridge, with a real surprise.

Before Stage 2 the executor's failure branches never called the bridge: a failed tool booked only
``tool_dispatch.record_outcome``'s NEGATIVE ``tool_result`` link, and ``ToolOutput.rpe`` was ``None`` for every
failure, so a failed action entered memory with no surprise. Stage 2 (``docs/plans/tool_failure_credit.md``,
owner decisions TF1-TF9) books ONE attributed NEGATIVE through the bridge, stamps the booked signature on the
output, and dispatch skips its own NEGATIVE for exactly that signature. Speculative prefetch calls book
nothing (TF9); a human-driven, bridge-less, aliased or compound-signature failure books as before.

Driven through the REAL executor (``build_executor`` with a real NAc, so the bridge is the production one),
the real ``tool_dispatch.execute_and_learn`` (stand-ins only for the parts the failure path does not learn
through) and the real ``SpeculativePrefetcher``.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime import tool_dispatch
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

TOOL = "flaky_1200"


class _Flaky(Tool):
    """Returns a failure while ``fail`` is set, a success otherwise."""

    name = TOOL
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def execute(self, **kwargs: Any) -> Any:
        if self.fail:
            return ToolOutput(success=False, error="boom")
        return ToolOutput(success=True, output="ok")


class _Raiser(Tool):
    name = "raiser_1200"
    description = "stub"
    input_schema: dict = {}

    def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError("kaput")


class _Reader(Tool):
    """Stands in for ``read_file``, the tool prefetch calls speculatively."""

    name = "read_file"
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.ok = True

    def execute(self, **kwargs: Any) -> Any:
        if self.ok:
            return ToolOutput(success=True, output="contents")
        return ToolOutput(success=False, error="no such file")


def _executor(*tools: Tool, nac: NAc | None = None) -> tuple[Any, NAc]:
    nac = nac if nac is not None else NAc(NACConfig())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return build_executor(registry, pain_bus=None, permissions=None, nac=nac), nac


def _links(nac: NAc, signature: str, valence: Valence) -> list[Any]:
    return [link for link in nac._links.get(signature, []) if link.outcome_valence is valence]


def _execute_and_learn(executor: Any, nac: NAc, action: dict[str, Any]) -> Any:
    """The real dispatch, with the loop's own ``rec_outcome`` (a partial of the real ``record_outcome``)."""
    sim = MagicMock()
    sim.is_sim_mode = False
    return tool_dispatch.execute_and_learn(
        agent=MagicMock(_execution_tracker=None),
        agent_name="aut",
        agent_id="aut_1200",
        executor=executor,
        sim=sim,
        state=MagicMock(data={}),
        environment=MagicMock(),
        memory=MagicMock(),
        hippocampus=None,
        memory_hub=None,
        result_cache=MagicMock(),
        autonomy_controller=MagicMock(),
        rec_outcome=functools.partial(tool_dispatch.record_outcome, drive_relief_only=False),
        recent_outcomes=[],
        max_recent=5,
        llm_worker=None,
        context_pool=MagicMock(),
        nac=nac,
        run_id="run_1200",
        action=action,
        confidence=0.9,
        proposal=SimpleNamespace(reasoning="probe", citations=[], cluster_id=None, clusters=None, triggering_input=""),
        observation={},
        human_involved=False,
    )


# ── red gates: a failure that ran ─────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1200: the executor's failure branches never call the tool-pain bridge")
@pytest.mark.parametrize("tool", ["returned", "raised"])
def test_a_failure_that_ran_carries_its_surprise(tool: str) -> None:
    executor, _ = _executor(_Flaky(), _Raiser())
    name = TOOL if tool == "returned" else "raiser_1200"
    out = executor.execute({"tool_name": name, "params": {}})
    assert out.success is False
    assert out.rpe is not None and out.rpe > 0.0


@pytest.mark.xfail(strict=True, reason="#1200: a failure books only dispatch's tool_result link")
def test_a_dispatched_failure_books_exactly_one_negative_on_the_bridge_link() -> None:
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [(link.outcome_type, link.outcome_signature, link.observation_count) for link in negatives] == [
        ("result", f"tool:{TOOL}:negative", 1)
    ]


@pytest.mark.xfail(
    strict=True, reason="#1200: a model-chosen action can never opt out of credit, but books no bridge link"
)
def test_a_model_written_speculative_key_books_as_normal() -> None:
    """TF9 is carried by the executor's context, never by the action dict: that is the model's own JSON."""
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}, "_speculative": True})
    assert [link.outcome_signature for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == [
        f"tool:{TOOL}:negative"
    ]


def _prefetch(ok: bool) -> NAc:
    from maxim.runtime.prefetch import SpeculativePrefetcher

    reader = _Reader()
    reader.ok = ok
    executor, nac = _executor(reader)
    SpeculativePrefetcher(executor=executor)._execute_cached("read_file", {"path": "x.txt"})
    return nac


def _booked_read_file(nac: NAc) -> tuple[list[Any], list[Any]]:
    return nac._links.get("tool:read_file", []), [e for e in nac._pending_events if e["signature"] == "tool:read_file"]


@pytest.mark.xfail(
    strict=True, reason="#1200 TF9: a speculative prefetch hit is booked POSITIVE as if the agent chose it"
)
def test_a_speculative_prefetch_hit_books_nothing() -> None:
    assert _booked_read_file(_prefetch(ok=True)) == ([], [])


# ── guards: what Stage 2 must not change ──────────────────────────────────────────────────────────────


def test_a_success_books_as_before() -> None:
    """Guard (green before and after): one bridge POSITIVE plus dispatch's own success link."""
    flaky = _Flaky()
    flaky.fail = False
    executor, nac = _executor(flaky)
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    positives = _links(nac, f"tool:{TOOL}", Valence.POSITIVE)
    assert sorted(link.outcome_type for link in positives) == ["result", "tool_result"]
    assert _links(nac, f"tool:{TOOL}", Valence.NEGATIVE) == []


def test_a_human_driven_failure_books_through_dispatch_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard (TF3): the interactive gate suppresses the bridge, so dispatch keeps its NEGATIVE."""
    import maxim.simulation.sim_logger as sim_logger

    monkeypatch.setattr(sim_logger, "get_interactive_mode", lambda: sim_logger.InteractiveMode.ON)
    executor, nac = _executor(_Flaky())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    negatives = _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)
    assert [link.outcome_type for link in negatives] == ["tool_result"]


def test_a_failure_with_no_bridge_books_through_dispatch_as_before() -> None:
    """Guard: an executor without an NAc has no bridge (``api.py``'s learning=False, the sandbox); dispatch
    still books into the loop's NAc."""
    registry = ToolRegistry()
    registry.register(_Flaky())
    executor = build_executor(registry, pain_bus=None, permissions=None)
    assert executor._tool_pain_bridge is None
    nac = NAc(NACConfig())
    _execute_and_learn(executor, nac, {"tool_name": TOOL, "params": {}})
    assert [link.outcome_type for link in _links(nac, f"tool:{TOOL}", Valence.NEGATIVE)] == ["tool_result"]


def test_a_speculative_prefetch_miss_books_nothing() -> None:
    """Guard (green today only because no failure reaches the bridge yet): once Stage 2 books failures, a
    speculative miss must still book nothing (TF9)."""
    assert _booked_read_file(_prefetch(ok=False)) == ([], [])


def test_an_aliased_failure_keeps_dispatchs_alias_link() -> None:
    """Guard (TF7): the bridge books the canonical name, dispatch the alias the model wrote; they differ, so
    dispatch's link under the alias is not cut."""

    class _Recall(_Flaky):
        name = "memory_recall"

    executor, nac = _executor(_Recall())
    _execute_and_learn(executor, nac, {"tool_name": "recall", "params": {}})
    assert [link.outcome_type for link in _links(nac, "tool:recall", Valence.NEGATIVE)] == ["tool_result"]
