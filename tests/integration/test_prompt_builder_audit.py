"""Phase 0 audit harness for prompt-caching (docs/plans/prompt_caching_for_cloud_backends.md).

These tests document the BASELINE state of the prompt architecture with respect
to prompt caching, BEFORE any Phase 1 refactor. They are the regression anchor
the Phase 1/3 work flips green.

Decisive architectural finding (2026-06-08 audit):

    The entire ~120K-token cradle context is assembled by ``PromptBuilder`` into
    ONE string returned as ``TOOL_PROMPT|<everything>``. The router
    (``LLMRouter._generate_tool_response``) routes that whole string into the
    **user** message and sets ``system`` to the tiny static constant
    ``SYSTEM_TOOL_RESPONSE`` (~100 tokens). The ``cache_control`` marker wired in
    ``_AnthropicBackend._build_system_blocks`` is applied to ``system`` — i.e. to
    the ~100-token static constant, NOT the 120K payload. Flipping
    ``prompt_cache: true`` therefore caches ~0% of the per-turn tokens.

    Worse, the user payload has no stable cacheable prefix: dynamic content
    (the user request / percept, datetime to minute precision, drive states in
    ``body_state``, observations, memories, bio annotations) is interleaved from
    near the top of the assembled prompt onward.

See the Phase 0 results table in the plan doc for the full per-section audit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from types import SimpleNamespace

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import PromptBuilder
from maxim.models.language.cloud_dispatch import SYSTEM_TOOL_RESPONSE


class _CharTokenCounter:
    """Deterministic char-estimate token counter (no model load)."""

    @staticmethod
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class _LLMStub:
    """Minimal LLM stub. Deliberately lacks ``get_cost_tracker`` so
    ``PromptBuilder.build_budget_context`` short-circuits to ''."""


class _CarryoverStub:
    @staticmethod
    def get_prompt_text() -> str:
        return ""


def _make_builder(n_ctx: int = 32000) -> PromptBuilder:
    return PromptBuilder(
        llm=_LLMStub(),
        reasoning_carryover=_CarryoverStub(),
        n_ctx=n_ctx,
        token_counter=_CharTokenCounter(),
        tool_index=None,
    )


def _cradle_mode() -> ModeInfo:
    return ModeInfo(
        name="autonomous",
        goal="explore the cradle and learn from your body",
        context_prompt="",
        max_response_tokens=512,
        context_window_tokens=4096,
    )


# Stable-across-the-sub-sim scene roster (cradle infant body affordances).
_CRADLE_TOOLS = {
    "infant_humanoid_sense",
    "infant_humanoid_touch",
    "infant_humanoid_grasp",
    "infant_humanoid_respond",
    "sense_tools",
}
_TOOL_DESCRIPTIONS = {name: f"{name}: a body affordance for interacting with the world." for name in _CRADLE_TOOLS}


def _make_turn_request(turn: int) -> LLMRequest:
    """Construct an embodied cradle-turn request whose DYNAMIC state mutates
    per turn the way a real sub-sim's does: drive drift in body_state, a new
    observation, accumulating memories, and per-turn substrate annotations.

    The stable session-scoped fields (identity, goal, tool roster, tool
    guidance) are held constant — exactly what a cacheable prefix would want.
    """
    ctx = StructuredContext(timestamp=1_700_000_000.0 + turn)

    # --- DYNAMIC per-turn state (the cache invalidators) ---
    # Drive states drift every tick (homeostatic/entropic). This is rendered
    # verbatim into the CRITICAL "body_state" section.
    stamina = max(0.0, 1.0 - turn * 0.07)
    warmth = 0.5 + (turn % 3) * 0.1
    ctx.body_state = (
        f"=== Body State ===\nstamina: {stamina:.2f}\nwarmth: {warmth:.2f}\nhunger: {min(1.0, turn * 0.06):.2f}"
    )
    # New observation each turn.
    ctx.current_percept = SimpleNamespace(
        transcript_chunk="",
        detections=[{"label": f"object_{turn}"}],
        cli_input="",
    )
    # Memories accumulate across turns.
    ctx.relevant_memories = [
        {"content": {"action": f"touch object_{i}", "success": i % 2 == 0}, "source": "episodic", "salience": 0.5}
        for i in range(turn + 1)
    ]
    # Wire-A substrate annotation: NAc bias values shift as learning proceeds.
    ctx.cluster_bias_annotations = [("infant_humanoid_touch", round(0.1 * turn, 3))]
    ctx.recent_outcomes = [{"tool": "infant_humanoid_touch", "success": turn % 2 == 0, "result": f"r{turn}"}]

    req = LLMRequest(
        request_id=f"turn-{turn}",
        context=ctx,
        mode=_cradle_mode(),
        autonomy_level=AutonomyLevel.AUTONOMOUS,
        internet_access=False,
        internet_policy_summary="",
        use_tool_prompting=True,
        available_tools=set(_CRADLE_TOOLS),
        tool_descriptions=dict(_TOOL_DESCRIPTIONS),
        is_embodied=True,
        # Stable session-scoped trigger (the root goal), constant across turns.
        triggering_input="explore the cradle and learn from your body",
    )
    return req


def _collect_payloads(monkeypatch, n_turns: int = 12) -> list[str]:
    """Return the assembled user-message payload (build_prompt output) for each
    turn, with datetime advanced per turn the way wall-clock does."""
    builder = _make_builder()
    payloads: list[str] = []

    base = datetime(2026, 6, 8, 10, 0, 0)

    for turn in range(n_turns):
        # Advance wall clock by a minute per turn — build_datetime_section
        # renders time to minute precision, a classic silent invalidator.
        fake_now = base.replace(minute=turn % 60)

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return fake_now

        monkeypatch.setattr("maxim.agents.prompt_builder.datetime", _FixedDateTime)
        prompt = builder.build_prompt(_make_turn_request(turn))
        assert prompt.startswith("TOOL_PROMPT|"), f"turn {turn}: unexpected prompt path: {prompt[:40]!r}"
        payloads.append(prompt[len("TOOL_PROMPT|") :])

    return payloads


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Baseline characterization (PASS now — documents the broken state)
# ─────────────────────────────────────────────────────────────────────────────


def test_system_block_carries_negligible_tokens_vs_user_payload(monkeypatch):
    """The literal ``system`` block (where cache_control lives) is the static
    SYSTEM_TOOL_RESPONSE constant; the per-turn payload is ~100x larger and
    lives in the user message. Caching the system block saves ~nothing."""
    payloads = _collect_payloads(monkeypatch, n_turns=3)
    system_tokens = _CharTokenCounter.count_tokens(SYSTEM_TOOL_RESPONSE)
    user_tokens = _CharTokenCounter.count_tokens(payloads[0])
    # The cacheable system block is a small minority of total tokens even on
    # this tiny synthetic fixture (~12%); in a real cradle sub-sim the ratio is
    # ~800x (≈145 system tokens vs ~120K user tokens).
    system_fraction = system_tokens / (system_tokens + user_tokens)
    assert system_fraction < 0.25, (
        f"cacheable system block should be a small minority of tokens; "
        f"system={system_tokens} user={user_tokens} fraction={system_fraction:.2f}"
    )


def test_baseline_user_payload_is_not_byte_stable_across_turns(monkeypatch):
    """BASELINE: the per-turn user payload changes every turn — 12 unique
    hashes. This is the documented pre-fix state; the cache would miss on
    every turn even if the breakpoint were moved onto the user message."""
    payloads = _collect_payloads(monkeypatch, n_turns=12)
    hashes = [_sha(p) for p in payloads]
    # Documented baseline: every turn differs.
    assert len(set(hashes)) == 12, (
        "baseline expected 12 distinct payloads; if this changed, re-audit "
        "(see docs/plans/prompt_caching_for_cloud_backends.md Phase 0)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The target invariant (currently FAILS — flipped green by Phase 1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Phase 0 baseline: no byte-stable cacheable prefix exists today — the "
        "whole payload is in the user message with dynamic content interleaved "
        "from the top. Phase 1 (prompt_caching_for_cloud_backends.md) introduces "
        "a stable system-prompt prefix; flipping this green is the Phase 1/3 "
        "acceptance gate. xfail(strict) makes this XPASS-fail the moment a stable "
        "prefix lands, forcing the marker removal."
    ),
)
def test_cacheable_prefix_is_byte_stable_across_turns(monkeypatch):
    """TARGET: across 12 turns the cacheable prefix must be byte-identical.

    Today there is no separated prefix, so we measure the whole payload and it
    is not stable. Phase 1 should: (a) move the stable session-scoped sections
    into the ``system`` prompt (where cache_control lives), and (b) keep them
    byte-identical across turns. When that lands, replace this body with an
    assertion over the actual system prompt and remove the xfail marker (this
    becomes ``test_system_prompt_byte_stable_across_turns`` per Phase 3)."""
    payloads = _collect_payloads(monkeypatch, n_turns=12)
    hashes = [_sha(p) for p in payloads]
    assert len(set(hashes)) == 1, (
        f"cacheable prefix not byte-stable: {len(set(hashes))} distinct of {len(hashes)} turns"
    )
