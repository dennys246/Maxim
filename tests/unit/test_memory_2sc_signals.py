"""Memory-strength Phase 2S-c (#848): a survival capture records its pain and its novelty.

- **Pain** is measured per invocation by the tool-pain bridge (which already hears every PainBus
  signal and knows which invocation is running) and stamped by the executor onto ``ToolOutput.pain``:
  the pain the action CAUSED (its delta-attributed embodiment failures), else the peak it FELT while
  it ran; 0.0 when a pain source was watched and nothing fired; None when none was watched.
- **Novelty** is ``1 - margin`` of the EC match margins the situation's clusters were encoded with,
  carried on the proposal because the encoder's stash is overwritten by the next tick.
- **Salience stays unmeasured**: the tag already scores pain and drive pressure on their own.

The real-loop composition (an executed escape records a measured pain and novelty) is pinned in
``test_water_trial_smoke.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.bridges.tool_pain_bridge import ToolPainBridge
from maxim.decisions.nac import NAc
from maxim.proprioception.pain import PainSignal, PainType
from maxim.runtime.agent_loop import _attach_live_situation, _situation_margins, situation_novelty


def _signal(intensity: float) -> PainSignal:
    return PainSignal(pain_type=PainType.EXTERNAL_SIGNAL, intensity=intensity, timestamp=0.0)


def _bridge(*, watched: bool = True) -> ToolPainBridge:
    return ToolPainBridge(nac=NAc(), pain_bus=MagicMock() if watched else None)


# ── pain, per invocation ─────────────────────────────────────────────────────


def test_pain_felt_while_the_action_ran_is_its_pain():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b._on_pain(_signal(0.3))
    b._on_pain(_signal(0.6))
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.6)


def test_a_watched_action_with_no_pain_measures_zero_not_none():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    assert b.pop_invocation_pain("inv-1") == 0.0


def test_no_pain_source_is_not_measured():
    b = _bridge(watched=False)
    b.record_tool_start("swim", "inv-1")
    assert b.pop_invocation_pain("inv-1") is None


def test_pain_the_action_caused_takes_precedence_over_pain_felt():
    """The owner's rule: the action's own delta-attributed failures first, felt pain as fallback."""
    b = _bridge()
    b.record_tool_start("dig", "inv-1")
    b._on_pain(_signal(0.9))
    # through the entry the executor calls with the tool's delta-attributed failures
    b.record_tool_embodiment_failure("dig", "inv-1", [{"name": "f", "entity": "body", "pain": 0.4}])
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.4)


# Only what a real FailureEvent can carry: spec.py parses ``pain_intensity`` with ``float()``, so a
# body spec can yield an out-of-range or non-finite number, never a string or None.
@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
def test_a_malformed_failure_pain_is_skipped_and_felt_pain_stands(bad):
    b = _bridge()
    b.record_tool_start("dig", "inv-1")
    b._on_pain(_signal(0.2))
    b.record_tool_embodiment_failure("dig", "inv-1", [{"name": "f", "entity": "body", "pain": bad}])
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.2)


def test_pain_after_the_action_completed_is_not_its_pain():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b.record_tool_complete("swim", "inv-1", success=True)
    b._on_pain(_signal(0.8))  # no longer pending
    assert b.pop_invocation_pain("inv-1") == 0.0


def test_each_invocation_pain_is_read_once():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b._on_pain(_signal(0.5))
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.5)
    assert b.pop_invocation_pain("inv-1") is None


def test_the_executor_stamps_the_invocations_pain_onto_its_output():
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    bridge = _bridge()

    class _Hurts(Tool):
        name = "swim"
        description = "Hurts while it runs"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            bridge._on_pain(_signal(0.7))  # the body publishes pain mid-action
            return ToolOutput(success=True)

    registry = ToolRegistry()
    registry.register(_Hurts())
    executor = Executor(tool_registry=registry, tool_pain_bridge=bridge)
    out = executor.execute({"tool_name": "swim", "params": {}})
    assert out.pain == pytest.approx(0.7)


def test_a_tool_cannot_set_its_own_pain():
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    class _Lies(Tool):
        name = "lie"
        description = "Claims pain"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            return ToolOutput(success=True, pain=0.99)

    registry = ToolRegistry()
    registry.register(_Lies())
    executor = Executor(tool_registry=registry, tool_pain_bridge=_bridge())
    assert executor.execute({"tool_name": "lie", "params": {}}).pain == 0.0


# ── only NOCICEPTION is pain (the owner rule, on the type) ──────────────────────
# Both review lenses reproduced the first draft recording these as pain: air hunger 0.7, fear 0.5,
# a failed tool 0.3+, and a caused drive discomfort masking real harm.


def _drive(intensity: float, drive: str = "oxygen") -> PainSignal:
    return PainSignal(
        pain_type=PainType.EXTERNAL_SIGNAL, intensity=intensity, timestamp=0.0, context={"source": f"drive:{drive}"}
    )


@pytest.mark.parametrize(
    "signal",
    [
        _drive(0.7),  # air hunger: a homeostatic breach, published as EXTERNAL_SIGNAL
        PainSignal(pain_type=PainType.ANTICIPATED, intensity=0.5, timestamp=0.0),  # fear
        PainSignal(pain_type=PainType.TOOL_FAILURE, intensity=0.3, timestamp=0.0),  # frustration
        PainSignal(pain_type=PainType.RESOURCE_EXHAUSTION, intensity=0.4, timestamp=0.0),
    ],
)
def test_signals_that_are_not_nociception_are_not_felt_as_pain(signal):
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b._on_pain(signal)
    assert b.pop_invocation_pain("inv-1") == 0.0


def test_health_is_the_one_drive_that_is_pain():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b._on_pain(_drive(0.6, "health"))
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.6)


def test_a_failed_tool_through_the_real_executor_records_no_pain():
    """The executor reports a tool failure to the pain detector, which publishes TOOL_FAILURE
    while the invocation is still pending -- frustration, not pain."""
    from maxim.proprioception.pain import PainDetector
    from maxim.runtime.executor import Executor
    from maxim.tools.base import Tool, ToolOutput
    from maxim.tools.registry import ToolRegistry

    class _Fails(Tool):
        name = "grab"
        description = "Always fails"
        input_schema: dict[str, Any] = {}

        def execute(self, **kwargs: Any) -> ToolOutput:
            return ToolOutput(success=False, error="collision")

    detector = PainDetector()
    bridge = ToolPainBridge(nac=NAc(), pain_detector=detector)
    registry = ToolRegistry()
    registry.register(_Fails())
    out = Executor(tool_registry=registry, pain_detector=detector, tool_pain_bridge=bridge).execute(
        {"tool_name": "grab", "params": {}}
    )
    assert out.pain == 0.0


def test_a_caused_drive_discomfort_is_not_pain_and_cannot_mask_felt_harm():
    b = _bridge()
    b.record_tool_start("swim", "inv-1")
    b._on_pain(_signal(0.3))  # real harm felt
    b.record_tool_embodiment_failure("swim", "inv-1", [{"name": "drive:oxygen:discomfort", "entity": "b", "pain": 0.6}])
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.3)


def test_a_caused_health_breach_is_pain():
    b = _bridge()
    b.record_tool_start("dig", "inv-1")
    b.record_tool_embodiment_failure("dig", "inv-1", [{"name": "drive:health:deprived", "entity": "b", "pain": 0.5}])
    assert b.pop_invocation_pain("inv-1") == pytest.approx(0.5)


def test_every_pain_type_has_a_kind():
    """A new PainType must be classified -- classify_pain raises rather than defaulting."""
    from maxim.proprioception.pain import classify_pain

    for pain_type in PainType:
        classify_pain(pain_type, "")


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("drive:oxygen:discomfort", "drive"),
        ("drive:arms.thermal:deprived", "drive"),
        ("drive:health:discomfort", "nociceptive"),
        ("torso_broken", "nociceptive"),
        # malformed (no band): the delta filter parses these as standard failures, so they are harm
        ("drive:oxygen", "nociceptive"),
        ("drive:health", "nociceptive"),
        ("drive:", "nociceptive"),
    ],
)
def test_failure_pain_kind(name, kind):
    from maxim.proprioception.pain import failure_pain_kind

    assert failure_pain_kind(name).value == kind


# ── novelty ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("margins", "expected"),
    [
        (None, None),
        ({}, None),
        ({"world": 0.95}, 0.05),
        ({"world": 0.95, "interoception": 0.40}, 0.60),  # the MOST novel modality
        ({"world": -1.0}, None),  # nothing comparable: not measured (the reference set is empty)
        ({"world": -1.0, "interoception": 0.9}, 0.1),
        ({"world": 1.2}, 0.0),  # clamped
    ],
)
def test_situation_novelty(margins, expected):
    got = situation_novelty(margins)
    assert got == (None if expected is None else pytest.approx(expected))


class _Encoder:
    def __init__(self, margins: dict[str, float | None]) -> None:
        self._margins = margins

    def last_encode_margin(self, *, agent_id: str, modality: str) -> float | None:
        return self._margins.get(modality)


def test_margins_are_read_for_the_situations_modalities_only():
    enc = _Encoder({"world": 0.9, "audio": 0.5, "interoception": None})
    got = _situation_margins(enc, "a", {"world": "w", "interoception": "i"})
    assert got == {"world": 0.9}  # no scan ran for interoception; audio is not in the situation


def test_the_llm_primary_encode_records_the_margins_with_the_clusters(monkeypatch):
    import dataclasses

    import maxim.runtime.agent_loop as al
    from maxim.agents.llm_types import LLMProposal

    monkeypatch.setattr(al, "_encode_current_clusters", lambda enc, agent_id, ex: {"interoception": "c-i"})
    proposal = LLMProposal(
        request_id="r",
        action={"tool_name": "t"},
        reasoning="",
        strategy_used=None,
        confidence=1.0,
        mode_goal_achieved=False,
    )
    executor = MagicMock()
    out = _attach_live_situation(
        proposal,
        aut_mode="llm-primary",
        sensor_encoder=_Encoder({"interoception": 0.7}),
        agent_id="a",
        executor=executor,
    )
    assert out.clusters == {"interoception": "c-i"} and out.cluster_margins == {"interoception": 0.7}
    # substrate-primary already captured its clusters: untouched
    sp = _attach_live_situation(
        proposal, aut_mode="substrate-primary", sensor_encoder=_Encoder({}), agent_id="a", executor=executor
    )
    assert sp is proposal and dataclasses.asdict(sp)["clusters"] is None
