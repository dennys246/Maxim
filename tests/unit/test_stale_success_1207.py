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

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime.bootstrap import build_executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


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


def test_a_failure_then_a_success_books_exactly_one_positive() -> None:
    flaky = _Flaky()
    executor, nac, _ = _executor(flaky)
    assert executor.execute({"tool_name": "flaky_1207", "params": {}}).success is False
    flaky.fail = False
    assert executor.execute({"tool_name": "flaky_1207", "params": {}}).success is True
    assert _positive_observations(nac, "tool:flaky_1207") == 1


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


def test_a_finished_invocation_leaves_no_pending_event() -> None:
    """The retire half: after any invocation ends, its NAc pending event is gone."""
    flaky = _Flaky()
    executor, nac, _ = _executor(flaky)
    executor.execute({"tool_name": "flaky_1207", "params": {}})  # a failure: nothing books it here
    assert _pending(nac, "tool:flaky_1207") == []


def test_a_never_run_call_never_starts_an_invocation() -> None:
    """Pins the executor move on its own (the retire alone would also leave nothing pending): an inactive or
    unregistered call never reaches ``NAc.record_event`` at all (owner decision TF2)."""
    from unittest.mock import patch

    executor, nac, registry = _executor()
    registry.register_scene_tools([_Scene()], scene_id="scene_1207")
    registry.deactivate_scene("scene_1207")
    with patch.object(nac, "record_event", wraps=nac.record_event) as spy:
        executor.execute({"tool_name": "scene_1207", "params": {}})
        executor.execute({"tool_name": "not_registered_1207", "params": {}})
    assert spy.call_count == 0


def test_with_a_pain_detector_a_never_run_call_books_no_failure() -> None:
    """With tool-failure pain wired (a detector), a never-run call used to book ``tool:X:negative`` and an RPE
    through ``_on_pain``; TF2 says only tools that ran earn failure credit."""
    from maxim.proprioception.pain import PainDetector

    nac = NAc(NACConfig())
    registry = ToolRegistry()
    executor = build_executor(registry, pain_bus=None, permissions=None, nac=nac, pain_detector=PainDetector())
    registry.register_scene_tools([_Scene()], scene_id="scene_1207")
    registry.deactivate_scene("scene_1207")
    out = executor.execute({"tool_name": "scene_1207", "params": {}})
    assert out.success is False
    assert nac._links.get("tool:scene_1207", []) == []
    assert out.rpe is None


def test_event_ids_are_unique_within_one_clock_tick() -> None:
    """Booking and retiring are by id (#1207), so two same-signature events must never share one."""
    nac = NAc(NACConfig())
    ids = [nac.record_event("tool", "tool:same", context={}) for _ in range(5000)]
    assert len(set(ids)) == len(ids)


def test_an_nac_error_while_booking_cannot_strand_the_event() -> None:
    """The bridge reads the invocation's event id rather than popping it, so ``finish_invocation`` still retires
    the event when NAc raises mid-booking (the executor catches the bridge error)."""
    from unittest.mock import patch

    flaky = _Flaky()
    flaky.fail = False
    executor, nac, _ = _executor(flaky)
    with patch.object(nac, "record_outcome_full", side_effect=RuntimeError("probe")):
        executor.execute({"tool_name": "flaky_1207", "params": {}})
    assert _pending(nac, "tool:flaky_1207") == []


def test_a_missing_event_id_books_nothing_and_never_by_signature() -> None:
    """A broken lifecycle (no id for the invocation) is reported and booked nowhere: never a fallback to the
    signature attribution that #1207 removed."""
    from unittest.mock import patch

    executor, nac, _ = _executor()
    bridge = executor._tool_pain_bridge
    with patch.object(nac, "record_outcome") as by_signature, patch.object(nac, "record_outcome_full") as by_id:
        links = bridge._book_invocation("ghost_1207", "inv-x", "tool:ghost_1207", Valence.POSITIVE)
    assert links == [] and not by_signature.called and not by_id.called
