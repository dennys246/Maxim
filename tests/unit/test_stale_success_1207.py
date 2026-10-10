"""#1207: a failed tool call is never booked POSITIVE by a later success (tool-failure credit, Stage 1).

``ToolPainBridge.record_tool_start`` queues an NAc pending event ``tool:X`` for every invocation. Before
#1207 the bridge attributed its outcomes by SIGNATURE (``NAc.record_outcome(event_id=<signature>)``), so a
success matched EVERY pending ``tool:X`` inside the 300 s window, and nothing retired the pending event of a
failed or never-run invocation (``finish_invocation`` popped only the bridge's own entry). The next success
therefore booked the stale ones POSITIVE as well. ``docs/plans/tool_failure_credit.md`` Stage 1.

Driven through the REAL executor (``build_executor`` with an NAc, so the bridge is the production one).
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry

RED = pytest.mark.xfail(strict=True, reason="#1207: the bridge attributes by signature and nothing retires the event")


class _Flaky(Tool):
    """Fails while ``fail`` is set, succeeds otherwise."""

    name = "flaky_1207"
    description = "stub"
    input_schema: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self.fail = True

    def execute(self, **kwargs: Any) -> Any:
        return ToolOutput(
            success=not self.fail, output="ok" if not self.fail else None, error="boom" if self.fail else None
        )


class _Scene(Tool):
    name = "scene_1207"
    description = "stub"
    input_schema: dict = {}

    def execute(self, **kwargs: Any) -> Any:
        return ToolOutput(success=True, output="ok")


def _executor(*tools: Tool) -> tuple[Any, NAc, ToolRegistry]:
    nac = NAc(NACConfig())
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    executor = build_executor(registry, pain_bus=None, permissions=None, nac=nac)
    assert executor._tool_pain_bridge is not None  # the production bridge, built because an NAc exists
    return executor, nac, registry


def _positive_observations(nac: NAc, signature: str) -> int:
    return sum(
        link.observation_count for link in nac._links.get(signature, []) if link.outcome_valence is Valence.POSITIVE
    )


def _pending(nac: NAc, signature: str) -> list[dict[str, Any]]:
    return [e for e in nac._pending_events if e["signature"] == signature]


@RED
def test_a_failure_then_a_success_books_exactly_one_positive() -> None:
    flaky = _Flaky()
    executor, nac, _ = _executor(flaky)
    assert executor.execute({"tool_name": "flaky_1207", "params": {}}).success is False
    flaky.fail = False
    assert executor.execute({"tool_name": "flaky_1207", "params": {}}).success is True
    assert _positive_observations(nac, "tool:flaky_1207") == 1


@RED
def test_a_never_run_call_queues_nothing_and_its_later_success_books_one_positive() -> None:
    """An inactive scene tool's call never reaches ``tool.run``: no NAc pending event and no bridge booking
    for it (``tool_dispatch`` books its own NEGATIVE, as today). Once the scene activates, one success books
    exactly one positive."""
    executor, nac, registry = _executor()
    registry.register_scene_tools([_Scene()], scene_id="scene_1207")
    registry.deactivate_scene("scene_1207")
    assert executor.execute({"tool_name": "scene_1207", "params": {}}).success is False
    assert _pending(nac, "tool:scene_1207") == []
    registry.activate_scene("scene_1207")
    assert executor.execute({"tool_name": "scene_1207", "params": {}}).success is True
    assert _positive_observations(nac, "tool:scene_1207") == 1


def test_confidence_still_accrues_on_one_link_across_invocations() -> None:
    """Guard (green before and after): attributing by event id must not mint a link per invocation."""
    flaky = _Flaky()
    flaky.fail = False
    executor, nac, _ = _executor(flaky)
    for _ in range(3):
        assert executor.execute({"tool_name": "flaky_1207", "params": {}}).success is True
    positives = [link for link in nac._links.get("tool:flaky_1207", []) if link.outcome_valence is Valence.POSITIVE]
    assert len(positives) == 1 and positives[0].observation_count == 3


@RED
def test_a_finished_invocation_leaves_no_pending_event() -> None:
    """The retire half: after any invocation ends, its NAc pending event is gone."""
    flaky = _Flaky()
    executor, nac, _ = _executor(flaky)
    executor.execute({"tool_name": "flaky_1207", "params": {}})  # a failure: nothing books it here
    assert _pending(nac, "tool:flaky_1207") == []
