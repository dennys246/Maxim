"""Phase 0 audit harness for prompt-caching (docs/plans/archive/prompt_caching_for_cloud_backends.md).

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


# ─────────────────────────────────────────────────────────────────────────────
# P21 (sandbox plan, 2026-09-05): phase-scoped roster last; file guidance gated
# ─────────────────────────────────────────────────────────────────────────────


def _stable_for(monkeypatch, *, tool_descriptions: dict, available_tools: set[str], embodied: bool = True) -> str:
    builder = _make_builder()
    req = _make_turn_request(0)
    req.available_tools = set(available_tools)
    req.tool_descriptions = dict(tool_descriptions)
    req.is_embodied = embodied
    prompt = builder.build_prompt(req)
    payload = prompt[len("TOOL_PROMPT|") :]
    return payload.split(PROMPT_SEGMENT_DELIMITER, 1)


def test_tools_section_is_the_last_stable_section(monkeypatch):
    """The scene roster is stable only within a narrative phase; every other
    cacheable section is stable for the whole session. Emitting the roster
    LAST is what lets an encounter boundary keep the head of the prefix cached
    (Phase 0 of the sandbox plan: 559 of 3,450 cacheable tokens survived when
    the roster sat second)."""
    stable, _dynamic = _stable_for(monkeypatch, tool_descriptions=_TOOL_DESCRIPTIONS, available_tools=_CRADLE_TOOLS)
    tools_at = stable.index("=== Available Tools ===")
    for marker in ("OPERATIONAL STATE", "CORE PRINCIPLES", "Body Tool Discipline", "=== Instructions ==="):
        assert stable.index(marker) < tools_at, f"{marker!r} must precede the phase-scoped roster"
    assert "=== Available Tools ===" not in stable[tools_at + 1 :]


def test_encounter_change_keeps_the_session_stable_head(monkeypatch):
    """A DM campaign's ``choose`` description carries the current options, so
    it differs per encounter. Everything before the roster must still be a
    byte-identical common prefix across the two encounters."""
    base = dict(_TOOL_DESCRIPTIONS)
    tools = set(_CRADLE_TOOLS) | {"choose"}
    one, _ = _stable_for(
        monkeypatch,
        tool_descriptions={**base, "choose": "Pick one of the available choices: attack, defend"},
        available_tools=tools,
    )
    two, _ = _stable_for(
        monkeypatch,
        tool_descriptions={**base, "choose": "Pick one of the available choices: flee, bargain, fight"},
        available_tools=tools,
    )
    assert one != two
    head = one[: one.index("=== Available Tools ===")]
    assert two.startswith(head)
    # The head is the bulk of the prefix — the whole point of the ordering.
    assert len(head) > 0.5 * len(one)


def test_file_guidance_is_absent_when_the_roster_has_no_file_tool(monkeypatch):
    """An embodied AUT with body affordances only must not be told how to
    glob or where to write files: the arena narration "study his patterns"
    used to buy ~600 tokens of glob guidance per call (P21 measurement)."""
    req_text = "study his patterns and find files if you can"
    builder = _make_builder()
    req = _make_turn_request(0)
    req.triggering_input = req_text
    prompt = builder.build_prompt(req)
    assert "GLOB PATTERN GUIDE" not in prompt
    assert "FILE WORKSPACE REMINDER" not in prompt
    assert "CWD Context" not in prompt and "EXISTING WORKSPACE" not in prompt


def test_relevance_filtered_roster_cannot_advertise_a_tool_the_loop_did_not_offer(monkeypatch):
    """The learned tool index answers from everything it has ever seen; a
    denied tool it learned must not reach the passive-mode prompt (D82)."""

    class _Index:
        def get_relevant_tools(self, _q):
            return ["infant_humanoid_sense", "write_file"], ["infant_humanoid_touch", "bash"]

    builder = PromptBuilder(
        llm=_LLMStub(),
        reasoning_carryover=_CarryoverStub(),
        n_ctx=32000,
        token_counter=_CharTokenCounter(),
        tool_index=_Index(),
    )
    req = _make_turn_request(0)
    req.mode.uses_tool_relevance_filter = True
    prompt = builder.build_prompt(req)
    assert "infant_humanoid_sense" in prompt and "infant_humanoid_touch" in prompt
    assert "write_file" not in prompt and "- bash" not in prompt
    assert req.surfaced_tools == ["infant_humanoid_sense"]


def test_file_guidance_returns_with_a_file_tool_on_the_roster(monkeypatch, tmp_path):
    from maxim.agents.prompt_builder import has_file_tools

    builder = _make_builder()
    req = _make_turn_request(0)
    req.triggering_input = "study his patterns and find files if you can"
    req.available_tools = set(_CRADLE_TOOLS) | {"write_file"}
    req.tool_descriptions = {**_TOOL_DESCRIPTIONS, "write_file": "write a file"}
    assert has_file_tools(req)
    prompt = builder.build_prompt(req)
    assert "GLOB PATTERN GUIDE" in prompt
    assert "FILE WORKSPACE REMINDER" in prompt
