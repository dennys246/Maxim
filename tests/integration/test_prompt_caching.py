"""End-to-end prompt-caching regression guard (Phase 3).

Regression guard for the CLAUDE.md invariant "System prompt content MUST be
byte-stable across turns within a session". Unlike test_prompt_builder_audit.py
(which checks the PromptBuilder segment boundary), this drives the FULL path the
backend sees: PromptBuilder.build_prompt -> router._generate_tool_response split
-> the actual ``system`` string. If the system block drifts across turns, the
cloud backends' cache_control breakpoint misses on every turn and ITPM is
re-spent — so this test fails loudly.

See docs/plans/archive/prompt_caching_for_cloud_backends.md.
"""

from __future__ import annotations

import dataclasses
import hashlib
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import PromptBuilder
from maxim.models.language.config import LLMConfig
from maxim.models.language.router import LLMRouter


class _CharTokenCounter:
    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class _LLMStub:
    """Lacks get_cost_tracker so build_budget_context short-circuits."""


class _CarryoverStub:
    @staticmethod
    def get_prompt_text() -> str:
        return ""


_TOOLS = {"infant_humanoid_sense", "infant_humanoid_touch", "infant_humanoid_grasp"}
_TOOL_DESCRIPTIONS = {n: f"{n}: a body affordance." for n in _TOOLS}


def _make_builder() -> PromptBuilder:
    return PromptBuilder(
        llm=_LLMStub(),
        reasoning_carryover=_CarryoverStub(),
        n_ctx=32000,
        token_counter=_CharTokenCounter(),
        tool_index=None,
    )


def _make_router() -> LLMRouter:
    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={
            "fake": {
                "type": "maxim_peer",
                "base_url": "http://127.0.0.1:9999/v1",
                "api_key_env": "FAKE_KEY",
                "model": "fake",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        },
    )
    return LLMRouter(cfg)


def _turn_request(turn: int) -> LLMRequest:
    ctx = StructuredContext(timestamp=1_700_000_000.0 + turn)
    # Per-turn dynamic state — must NOT perturb the system block.
    ctx.body_state = f"=== Body State ===\nstamina: {max(0.0, 1.0 - turn * 0.08):.2f}"
    ctx.current_percept = SimpleNamespace(transcript_chunk="", detections=[{"label": f"obj_{turn}"}], cli_input="")
    ctx.relevant_memories = [
        {"content": {"action": f"touch obj_{i}", "success": i % 2 == 0}, "source": "episodic", "salience": 0.5}
        for i in range(turn + 1)
    ]
    ctx.cluster_bias_annotations = [("infant_humanoid_touch", round(0.1 * turn, 3))]
    return LLMRequest(
        request_id=f"t{turn}",
        context=ctx,
        mode=ModeInfo(name="autonomous", goal="explore the cradle", context_prompt="", max_response_tokens=512),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        use_tool_prompting=True,
        available_tools=set(_TOOLS),
        tool_descriptions=dict(_TOOL_DESCRIPTIONS),
        is_embodied=True,
        triggering_input="explore the cradle",
    )


def _capture_system_strings(monkeypatch, n_turns: int = 12) -> list[str]:
    """Drive build_prompt -> router split for each turn; capture the system arg
    the backend would receive."""
    builder = _make_builder()
    router = _make_router()
    systems: list[str] = []
    base = datetime(2026, 6, 8, 10, 0, 0)

    for turn in range(n_turns):
        fake_now = base.replace(minute=turn % 60)

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return fake_now

        monkeypatch.setattr("maxim.agents.prompt_builder.datetime", _FixedDateTime)
        prompt = builder.build_prompt(_turn_request(turn))
        assert prompt.startswith("TOOL_PROMPT|")
        tool_prompt = prompt[len("TOOL_PROMPT|") :].strip()

        captured: dict[str, str] = {}

        def _fake_complete(system, user, **kw):  # noqa: ANN001
            captured["system"] = system
            return '{"action":{"tool_name":"infant_humanoid_sense","params":{}},"confidence":0.9,"reasoning":"x"}', None

        with patch.object(router, "_complete_text", side_effect=_fake_complete):
            router._generate_tool_response(MagicMock(), tool_prompt, 0.0, 64)
        systems.append(captured["system"])

    return systems


def test_system_prompt_byte_stable_across_turns(monkeypatch):
    """The ``system`` string sent to the backend must be byte-identical across
    all turns of a session (the prompt-caching prerequisite)."""
    systems = _capture_system_strings(monkeypatch, n_turns=12)
    hashes = {hashlib.sha256(s.encode()).hexdigest() for s in systems}
    if len(hashes) != 1:
        # Build a helpful diff message naming the first divergent turn.
        first = systems[0]
        for i, s in enumerate(systems):
            if s != first:
                raise AssertionError(
                    f"system prompt drifted at turn {i}: {len(hashes)} distinct of {len(systems)}. "
                    f"A per-turn-dynamic section leaked into the cacheable system prefix — "
                    f"see docs/plans/archive/prompt_caching_for_cloud_backends.md."
                )
    assert len(hashes) == 1


def test_system_prompt_contains_stable_sections_not_dynamic(monkeypatch):
    """Sanity: the system block carries the stable sections and excludes the
    per-turn dynamic ones."""
    system = _capture_system_strings(monkeypatch, n_turns=1)[0]
    assert "OPERATIONAL STATE" in system  # identity (stable)
    assert "Body State" not in system  # drives (dynamic)
    assert "Relevant Memories" not in system  # memories (dynamic)
    assert "Current time" not in system  # datetime (dynamic)
