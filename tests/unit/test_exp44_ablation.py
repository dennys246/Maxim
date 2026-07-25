"""Guard test for the Exp 44 counterfactual harness ablation.

This is the load-bearing safety net for ``scripts/exp44/capture_paired_prompts.py``:
if a substrate carrier is added upstream and ``_ablate`` misses it, arm B would
silently keep substrate content and every counterfactual result would be
invalid. This test pins the carrier list so that rot fails CI, not the run.

Mirrors the fixture in ``tests/integration/test_prompt_builder_audit.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import PromptBuilder
from maxim.prompts.acting_coach import ActingCoachConfig

# Load the harness module by path (scripts/ is not an importable package).
_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "exp44" / "capture_paired_prompts.py"
_spec = importlib.util.spec_from_file_location("exp44_capture", _MOD_PATH)
assert _spec and _spec.loader
_capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_capture)
_ablate = _capture._ablate
assert_fully_ablated = _capture.assert_fully_ablated
CLUSTER_BIAS_MARKER = _capture.CLUSTER_BIAS_MARKER


class _CharTokenCounter:
    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class _LLMStub:
    pass


class _CarryoverStub:
    @staticmethod
    def get_prompt_text() -> str:
        return ""


_BODY_STATE = "=== Body State ===\nstamina: 0.40\nwarmth: 0.70\nhunger: 0.30"


def _make_builder() -> PromptBuilder:
    return PromptBuilder(
        llm=_LLMStub(),
        reasoning_carryover=_CarryoverStub(),
        n_ctx=32000,
        token_counter=_CharTokenCounter(),
        tool_index=None,
    )


def _make_request() -> LLMRequest:
    ctx = StructuredContext(timestamp=1_700_000_000.0)
    ctx.body_state = _BODY_STATE
    ctx.cluster_bias_annotations = [("infant_humanoid_touch", 0.42)]
    return LLMRequest(
        request_id="t0",
        context=ctx,
        mode=ModeInfo(
            name="autonomous",
            goal="explore",
            context_prompt="",
            max_response_tokens=512,
            context_window_tokens=4096,
        ),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        use_tool_prompting=True,
        available_tools={"infant_humanoid_touch", "sense_tools"},
        tool_descriptions={
            "infant_humanoid_touch": "touch: a body affordance.",
            "sense_tools": "sense_tools: list affordances.",
        },
        triggering_input="explore",
        acting_coach=ActingCoachConfig(body_state_layers=True),
    )


def test_full_prompt_contains_substrate_markers():
    builder = _make_builder()
    full = builder.build_prompt(_make_request())
    assert CLUSTER_BIAS_MARKER in full  # cluster-bias section present
    assert _BODY_STATE in full  # body_state section present


def test_ablated_prompt_strips_all_substrate_carriers():
    builder = _make_builder()
    req = _make_request()
    ablated = builder.build_prompt(_ablate(req))
    # The guard must pass...
    assert_fully_ablated(ablated, arm_a_body_state=_BODY_STATE)
    # ...and the markers must actually be gone.
    assert CLUSTER_BIAS_MARKER not in ablated
    assert _BODY_STATE not in ablated


def test_ablate_does_not_mutate_arm_a():
    """Arm A's request/context must be untouched — otherwise the live run that
    executes arm A would run on a silently-ablated prompt."""
    req = _make_request()
    _ablate(req)
    assert req.context.cluster_bias_annotations == [("infant_humanoid_touch", 0.42)]
    assert req.context.body_state == _BODY_STATE
    assert req.acting_coach.body_state_layers is True


def test_guard_fires_on_leak():
    """If a marker survives, assert_fully_ablated must raise (not silently pass)."""
    import pytest

    leaked = f"prompt with {CLUSTER_BIAS_MARKER} still in it"
    with pytest.raises(AssertionError):
        assert_fully_ablated(leaked, arm_a_body_state=None)
