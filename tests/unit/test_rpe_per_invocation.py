"""#847: a capture reads the surprise of the action it captures -- never an earlier tool's.

The bridge used to keep one ``_last_rpe`` slot that nothing reset; ``capture_episodic_memory`` read
it through ``executor.get_last_rpe()`` on every capture. Any invocation that produced no surprise of
its own -- here, a repeat failure inside the pain detector's cooldown, which never reaches the
bridge -- inherited the previous invocation's surprise into its salience.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.nac import NAc
from maxim.proprioception.pain import PainDetector
from maxim.runtime.bio_integration import capture_episodic_memory
from maxim.runtime.executor import Executor
from maxim.tools.base import Tool, ToolOutput
from maxim.tools.registry import ToolRegistry


class _Fails(Tool):
    name = "grab"
    description = "Always fails"
    input_schema: dict[str, Any] = {}

    def execute(self, **kwargs: Any) -> ToolOutput:
        return ToolOutput(success=False, error="collision")


def _executor() -> Executor:
    nac = NAc()
    detector = PainDetector()  # default cooldown: a repeat failure right away does not fire pain
    bridge = ToolPainBridge(nac=nac, pain_detector=detector)
    registry = ToolRegistry()
    registry.register(_Fails())
    return Executor(tool_registry=registry, pain_detector=detector, tool_pain_bridge=bridge)


def _capture_salience(executor: Executor, result: Any) -> float:
    obs = {"source": "test", "salience": 0.5}
    capture_episodic_memory(
        hippocampus=MagicMock(),
        executor=executor,
        observation=obs,
        state=MagicMock(),
        intent={"goal": "grab the cup"},
        action={"tool_name": "grab", "params": {}},
        result=result,
        run_id="run-1",
        novelty=None,
        situation=None,
    )
    return obs["salience"]


def test_back_to_back_failures_capture_their_own_salience():
    """Red on main for the RIGHT reason: it never touches ``.rpe`` -- only what the capture stores.

    On untouched main both captures store 0.75 (the second inherits the first's surprise)."""
    executor = _executor()
    saliences = [_capture_salience(executor, executor.execute({"tool_name": "grab", "params": {}})) for _ in range(2)]
    assert saliences == [0.75, 0.5]


def test_world_driven_embodiment_pain_does_not_boost_the_next_capture():
    import time

    from maxim.proprioception.pain import PainSignal, PainType

    executor = _executor()
    bridge = executor._tool_pain_bridge
    first = executor.execute({"tool_name": "grab", "params": {}})
    assert first.rpe is not None
    # Out-of-band body pain between actions: it belongs to no invocation.
    bridge._on_pain(
        PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.9,
            timestamp=time.time(),
            context={"source": "embodiment", "entity": "arm", "failure_mode": "strain"},
        )
    )
    assert bridge._rpe_by_invocation == {}


def test_a_tool_cannot_set_its_own_surprise():
    class Liar(Tool):
        name = "liar"
        description = "claims a surprise"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            return ToolOutput(success=True, rpe=1.0)

    registry = ToolRegistry()
    registry.register(Liar())
    assert Executor(tool_registry=registry).execute({"tool_name": "liar", "params": {}}).rpe is None


def test_a_surprise_free_invocation_does_not_inherit_the_previous_one():
    executor = _executor()
    first = executor.execute({"tool_name": "grab", "params": {}})
    second = executor.execute({"tool_name": "grab", "params": {}})  # inside the cooldown

    assert first.rpe is not None and first.rpe > 0.0  # first contact is surprising
    assert second.rpe is None  # its failure never reached the bridge: no surprise of its own
    assert _capture_salience(executor, first) > 0.5
    assert _capture_salience(executor, second) == 0.5  # pre-#847: 0.5 + first.rpe * 0.5


def test_each_invocation_rpe_is_read_once():
    executor = _executor()
    first = executor.execute({"tool_name": "grab", "params": {}})
    assert first.rpe is not None
    # The bridge's record for that invocation was collected by the stamp, not left behind.
    assert executor._tool_pain_bridge._rpe_by_invocation == {}
