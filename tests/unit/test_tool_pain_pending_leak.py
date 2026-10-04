"""#851: a failure whose pain never reaches the bridge must not disable embodiment-pain attribution.

``ToolPainBridge._pending_tools`` was popped only when a completion, an embodiment failure or the
failure's own pain signal arrived. A failure whose pain was dropped (the PainDetector cooldown, the
PainBus refractory gate, or no detector at all) left its entry behind for the rest of the session, and
``_on_embodiment_pain`` skips attribution whenever ANY entry is pending -- so world-driven body pain
silently stopped reaching NAc. The executor owns the invocation's lifecycle, so it retires the entry
when ``execute()`` returns.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.nac import NAc
from maxim.proprioception.pain import PainConfig, PainDetector, PainSignal, PainType
from maxim.proprioception.pain_bus import PainBus
from maxim.runtime.executor import Executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


class _Fails(Tool):
    name = "grab"
    description = "Always fails"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolOutput:
        return ToolOutput(success=False, error="collision")


class _Raises(Tool):
    """Raises out of ``run()`` itself: ``Tool.run`` would turn an ``execute()`` exception into a failed output,
    and only an escaping exception reaches the executor's own ``except`` branch."""

    name = "lift"
    description = "Always raises"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolOutput:
        raise NotImplementedError

    def run(self, **kwargs: Any) -> ToolOutput:
        raise RuntimeError("motor stalled")


_RED_851 = pytest.mark.xfail(strict=True, reason="#851: a dropped failure pain leaves its pending entry behind")


def _world_pain() -> PainSignal:
    """Out-of-band body pain: a joint strains while no tool is running."""
    return PainSignal(
        pain_type=PainType.TOOL_FAILURE,
        intensity=0.9,
        timestamp=time.time(),
        context={"source": "embodiment", "entity": "arm", "failure_mode": "strain"},
    )


def _world_pain_attributions(spy: MagicMock) -> int:
    """How many times NAc booked world-driven body pain (``record_outcome`` also routes through the spy)."""
    return sum(1 for c in spy.call_args_list if c.kwargs.get("outcome_type") == "embodiment_failure")


def _rig(
    *, detector: PainDetector | None, pain_bus: PainBus | None = None
) -> tuple[Executor, ToolPainBridge, MagicMock]:
    nac = NAc()
    spy = MagicMock(wraps=nac.record_outcome_full)
    nac.record_outcome_full = spy  # type: ignore[method-assign]
    bridge = ToolPainBridge(nac=nac, pain_detector=detector, pain_bus=pain_bus)
    registry = ToolRegistry()
    registry.register(_Fails())
    registry.register(_Raises())
    executor = Executor(tool_registry=registry, pain_detector=detector, tool_pain_bridge=bridge)
    return executor, bridge, spy


@_RED_851
def test_a_cooldown_dropped_failure_does_not_disable_world_pain_attribution():
    """The issue's red test: the second failure lands inside the detector's cooldown, so its pain never
    reaches the bridge. World pain afterwards must still be attributed through NAc."""
    executor, bridge, spy = _rig(detector=PainDetector())
    for _ in range(2):
        assert executor.execute({"tool_name": "grab", "params": {}}).success is False
    bridge._on_pain(_world_pain())
    assert _world_pain_attributions(spy) == 1


@_RED_851
def test_a_failure_with_no_pain_detector_leaves_nothing_pending():
    executor, bridge, spy = _rig(detector=None)
    executor.execute({"tool_name": "grab", "params": {}})
    assert bridge._pending_tools == {}
    assert bridge._pending_contexts == {}
    bridge._on_pain(_world_pain())
    assert _world_pain_attributions(spy) == 1


@_RED_851
def test_a_refractory_dropped_failure_leaves_nothing_pending():
    """The PainBus refractory gate drops the second failure's signal before any subscriber sees it."""
    bus = PainBus(pain_refractory_s=60.0, _allow_raw=True)
    detector = PainDetector(config=PainConfig(pain_cooldown_seconds=0.0))  # the BUS gate must be the dropper
    detector.add_pain_callback(bus.publish)
    executor, bridge, spy = _rig(detector=detector, pain_bus=bus)
    for _ in range(2):
        executor.execute({"tool_name": "grab", "params": {}})
    assert bridge._pending_tools == {}
    bridge._on_pain(_world_pain())
    assert _world_pain_attributions(spy) == 1


@_RED_851
def test_a_tool_that_raises_leaves_nothing_pending():
    executor, bridge, _ = _rig(detector=None)
    out = executor.execute({"tool_name": "lift", "params": {}})
    assert out.success is False and "execution failed: motor stalled" in (out.error or "")
    assert bridge._pending_tools == {}


@_RED_851
def test_an_unregistered_tool_leaves_nothing_pending():
    executor, bridge, _ = _rig(detector=None)
    assert executor.execute({"tool_name": "fly", "params": {}}).success is False
    assert bridge._pending_tools == {}


def test_the_failure_whose_pain_arrives_is_still_attributed_to_its_tool():
    """Retiring at return must not pre-empt the failure's own attribution: pain dispatch is synchronous,
    so ``_on_pain`` has already popped the entry and booked NEGATIVE before the executor retires it."""
    executor, _, _ = _rig(detector=PainDetector())
    nac = executor._tool_pain_bridge._nac  # type: ignore[union-attr]
    record = MagicMock(wraps=nac.record_outcome)
    nac.record_outcome = record  # type: ignore[method-assign]
    executor.execute({"tool_name": "grab", "params": {}})
    record.assert_called_once()
    assert record.call_args.kwargs["event_id"] == "tool:grab"


@_RED_851
def test_an_inactive_scene_tool_leaves_nothing_pending():
    executor, bridge, _ = _rig(detector=None)

    class _SceneTool(_Fails):
        name = "open_door"

    executor.registry.register_scene_tools([_SceneTool()], "cellar")
    executor.registry.deactivate_scene("cellar")
    out = executor.execute({"tool_name": "open_door", "params": {}})
    assert out.success is False and "not active" in (out.error or "")
    assert bridge._pending_tools == {}


@_RED_851
def test_an_exception_escaping_execute_still_retires_the_invocation(monkeypatch):
    """The retire is a ``finally``, not a step after the call: an exception that escapes ``execute()`` (here a
    failure report that raises) must not leave the entry pending."""
    executor, bridge, _ = _rig(detector=None)

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("report failed")

    monkeypatch.setattr(executor, "_report_failure", _boom)
    with pytest.raises(RuntimeError, match="report failed"):
        executor.execute({"tool_name": "grab", "params": {}})
    assert bridge._pending_tools == {}
