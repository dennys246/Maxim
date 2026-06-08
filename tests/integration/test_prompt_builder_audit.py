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

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_builder import PROMPT_SEGMENT_DELIMITER, PromptBuilder
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


def _collect_segments(monkeypatch, n_turns: int = 12) -> list[tuple[str, str]]:
    """Return (stable_prefix, dynamic_remainder) for each turn's build_prompt
    output, advancing wall-clock per turn the way a real sub-sim does.

    Post-Phase-1, build_prompt emits ``TOOL_PROMPT|<stable>\x1e<dynamic>``."""
    builder = _make_builder()
    segments: list[tuple[str, str]] = []

    base = datetime(2026, 6, 8, 10, 0, 0)

    for turn in range(n_turns):
        # Advance wall clock by a minute per turn — build_datetime_section
        # renders time to minute precision, a classic silent invalidator that
        # MUST land in the dynamic segment, never the cacheable prefix.
        fake_now = base.replace(minute=turn % 60)

        class _FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return fake_now

        monkeypatch.setattr("maxim.agents.prompt_builder.datetime", _FixedDateTime)
        prompt = builder.build_prompt(_make_turn_request(turn))
        assert prompt.startswith("TOOL_PROMPT|"), f"turn {turn}: unexpected prompt path: {prompt[:40]!r}"
        payload = prompt[len("TOOL_PROMPT|") :]
        assert PROMPT_SEGMENT_DELIMITER in payload, f"turn {turn}: missing segment delimiter"
        stable, dynamic = payload.split(PROMPT_SEGMENT_DELIMITER, 1)
        segments.append((stable, dynamic))

    return segments


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Post-Phase-1 invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_cacheable_prefix_is_byte_stable_across_turns(monkeypatch):
    """The cacheable (stable) segment must be byte-identical across all turns.

    This is the Phase 1 acceptance gate (was the Phase 0 xfail baseline). The
    stable prefix carries only class-(a)/(c) sections (identity, instructions,
    tool guidance, foundational, entity_context, tools); the byte-stability is
    what lets a single cache_control breakpoint at the end of the system block
    cache the whole prefix across turns 2..N."""
    segments = _collect_segments(monkeypatch, n_turns=12)
    stable_hashes = [_sha(s) for s, _d in segments]
    assert len(set(stable_hashes)) == 1, (
        f"cacheable prefix not byte-stable: {len(set(stable_hashes))} distinct of {len(segments)} turns"
    )


def test_stable_prefix_excludes_dynamic_content(monkeypatch):
    """The stable prefix must NOT contain per-turn dynamic markers — those are
    the silent invalidators Phase 0 identified. They belong in the dynamic
    segment."""
    stable, dynamic = _collect_segments(monkeypatch, n_turns=2)[1]
    # Stable carries the static sections...
    assert "OPERATIONAL STATE" in stable  # identity section
    # ...and excludes per-turn dynamic content.
    assert "Body State" not in stable, "drive states leaked into cacheable prefix"
    assert "Current time" not in stable, "datetime leaked into cacheable prefix"
    assert "Relevant Memories" not in stable, "memories leaked into cacheable prefix"
    # The dynamic segment carries them.
    assert "Body State" in dynamic
    assert "Relevant Memories" in dynamic


def test_dynamic_segment_varies_across_turns(monkeypatch):
    """The dynamic segment correctly changes every turn (drives, observations,
    memories, datetime)."""
    segments = _collect_segments(monkeypatch, n_turns=12)
    dyn_hashes = [_sha(d) for _s, d in segments]
    assert len(set(dyn_hashes)) == 12, f"expected 12 distinct dynamic segments; got {len(set(dyn_hashes))}"


def test_cacheable_prefix_is_non_trivial(monkeypatch):
    """The stable prefix must carry real content (else caching it is pointless).
    On a real cradle run the dominant stable chunk is the foundational
    Constitution/AGENTS block; here we just assert the prefix is non-empty and a
    meaningful fraction of the total."""
    stable, dynamic = _collect_segments(monkeypatch, n_turns=1)[0]
    stable_tokens = _CharTokenCounter.count_tokens(stable)
    dynamic_tokens = _CharTokenCounter.count_tokens(dynamic)
    assert stable_tokens > 0, "stable prefix is empty — nothing to cache"
    # System block actually sent = SYSTEM_TOOL_RESPONSE + stable prefix.
    system_tokens = _CharTokenCounter.count_tokens(SYSTEM_TOOL_RESPONSE) + stable_tokens
    total = system_tokens + dynamic_tokens
    assert system_tokens / total > 0.05, (
        f"cacheable system fraction implausibly small; system={system_tokens} dynamic={dynamic_tokens}"
    )
