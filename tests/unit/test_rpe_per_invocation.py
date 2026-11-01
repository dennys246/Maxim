"""#847: a capture reads the surprise of the action it captures -- never an earlier tool's.

The bridge used to keep one ``_last_rpe`` slot that nothing reset; ``capture_episodic_memory`` read
it through ``executor.get_last_rpe()`` on every capture. Any invocation that produced no surprise of
its own (originally a repeat failure inside the pain detector's cooldown; since #1200 Stage 2, which books
every failure that ran, a speculative call) inherited the previous invocation's surprise into its salience.
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
from tests.policy_helpers import TEST_POLICY


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
    return Executor(tool_registry=registry, policy=TEST_POLICY, pain_detector=detector, tool_pain_bridge=bridge)


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

    On pre-#847 code both captures store 0.75 (the second inherits the first's surprise); the known-answer
    surprise-free construction is now the speculative call in the test below. Since #1200 Stage 2
    the second failure reaches the bridge too and carries its OWN, smaller surprise (its link already exists),
    so the two saliences differ: 0.5 + 0.5 * 0.5 and 0.5 + 0.5 * ~0.43."""
    import pytest

    executor = _executor()
    saliences = [_capture_salience(executor, executor.execute({"tool_name": "grab", "params": {}})) for _ in range(2)]
    assert saliences[0] == 0.75
    assert saliences[1] == pytest.approx(0.715, abs=0.01)


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
    assert Executor(tool_registry=registry, policy=TEST_POLICY).execute({"tool_name": "liar", "params": {}}).rpe is None


def test_a_surprise_free_invocation_does_not_inherit_the_previous_one():
    """Since #1200 Stage 2 every failure that ran reaches the bridge, so the surprise-free invocation is a
    speculative one (no invocation is started for it, TF9) rather than a repeat failure inside the cooldown."""
    from maxim.runtime.executor import speculative_calls

    executor = _executor()
    first = executor.execute({"tool_name": "grab", "params": {}})
    with speculative_calls():
        second = executor.execute({"tool_name": "grab", "params": {}})

    assert first.rpe is not None and first.rpe > 0.0  # first contact is surprising
    assert second.rpe is None  # it never reached the bridge: no surprise of its own
    assert _capture_salience(executor, first) > 0.5
    assert _capture_salience(executor, second) == 0.5  # pre-#847: 0.5 + first.rpe * 0.5


def test_each_invocation_rpe_is_read_once():
    executor = _executor()
    first = executor.execute({"tool_name": "grab", "params": {}})
    assert first.rpe is not None
    # The bridge's record for that invocation was collected by the stamp, not left behind.
    assert executor._tool_pain_bridge._rpe_by_invocation == {}
