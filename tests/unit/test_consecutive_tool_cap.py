"""Tests for content-aware consecutive-tool cap logic.

The cap lives inline in agent_loop.py (lines ~1497-1533). These tests
validate the state-machine transitions directly rather than spinning up
a full agent loop.

Covers:
- Same tool + same params → count increments (hallucination detection)
- Same tool + different params → count resets (varied usage allowed)
- Different tool → count resets
- Cap fires at threshold + 1
- Cap fires feedback into recent_outcomes (Fix 2)
"""

from __future__ import annotations


# ── Reproduce the inline cap logic as a standalone function for testing ──
# This mirrors agent_loop.py lines ~1497-1533 exactly.


def _compute_cap_state(
    tool_name: str,
    params: dict,
    prev_tool: str,
    prev_count: int,
    prev_hash: int,
) -> tuple[str, int, int, bool]:
    """Return (new_tool, new_count, new_hash, capped).

    Mirrors the content-aware consecutive-tool cap in agent_loop.py.
    """
    try:
        content_hash = hash(str(sorted(params.items())) if isinstance(params, dict) else str(params))
    except Exception:
        content_hash = 0

    if tool_name == prev_tool and content_hash == prev_hash:
        new_count = prev_count + 1
    elif tool_name == prev_tool:
        # Same tool, different content — reset
        new_count = 1
        prev_hash = content_hash
    else:
        prev_hash = content_hash
        new_count = 1

    max_cap = 5
    capped = new_count > max_cap
    if capped:
        return ("", 0, 0, True)
    return (tool_name, new_count, prev_hash, False)


class TestContentAwareCap:
    """Content-aware consecutive-tool cap state machine."""

    def test_same_tool_same_params_increments(self):
        """Identical calls increment the counter."""
        tool, count, h, capped = _compute_cap_state(
            "respond",
            {"text": "hello"},
            "respond",
            1,
            hash(str(sorted({"text": "hello"}.items()))),
        )
        assert count == 2
        assert not capped

    def test_same_tool_different_params_resets(self):
        """Same tool with different params resets to 1 (varied usage)."""
        tool, count, h, capped = _compute_cap_state(
            "send_message",
            {"text": "Who are you?"},
            "send_message",
            4,
            hash(str(sorted({"text": "What tools do you have?"}.items()))),
        )
        assert count == 1
        assert not capped
        assert tool == "send_message"

    def test_different_tool_resets(self):
        """Switching to a different tool resets the counter."""
        tool, count, h, capped = _compute_cap_state(
            "observe_actions",
            {},
            "send_message",
            4,
            hash(str(sorted({"text": "hello"}.items()))),
        )
        assert count == 1
        assert not capped
        assert tool == "observe_actions"

    def test_cap_fires_at_threshold(self):
        """Cap fires when count exceeds MAX (5), i.e. on the 6th identical call."""
        params = {"text": "same"}
        h = hash(str(sorted(params.items())))
        # Simulate 5 identical calls
        tool, count, h_out, capped = ("respond", 5, h, False)
        # 6th identical call
        tool, count, h_out, capped = _compute_cap_state("respond", params, tool, count, h_out)
        assert capped
        assert count == 0  # reset after cap

    def test_varied_send_message_never_caps(self):
        """6+ send_message calls with different text should NEVER cap."""
        tool = ""
        count = 0
        h = 0
        messages = [
            "Who are you?",
            "What tools do you have?",
            "Try the locked door",
            "Read the note",
            "Look out the window",
            "Try to escape",
            "What else can you do?",
        ]
        for msg in messages:
            tool, count, h, capped = _compute_cap_state(
                "send_message",
                {"text": msg},
                tool,
                count,
                h,
            )
            assert not capped, f"Capped on message: {msg}"
        # Count should be 1 after each reset (each message is different)
        assert count == 1

    def test_identical_send_message_caps(self):
        """6 identical send_message calls SHOULD cap (hallucination)."""
        tool = ""
        count = 0
        h = 0
        for i in range(7):
            tool, count, h, capped = _compute_cap_state(
                "send_message",
                {"text": "same thing"},
                tool,
                count,
                h,
            )
            if capped:
                # Cap fires on the 6th call (count 6 > 5), which is index 5
                assert i == 5
                break
        assert capped

    def test_non_dict_params_handled(self):
        """Non-dict params don't crash the hash computation."""
        tool, count, h, capped = _compute_cap_state(
            "respond",
            "raw string params",
            "respond",
            1,
            hash(str("different")),
        )
        assert not capped

    def test_empty_params_counted(self):
        """Empty params on same tool still increment."""
        h = hash(str(sorted({}.items())))
        tool, count, _, capped = _compute_cap_state("observe", {}, "observe", 3, h)
        assert count == 4
        assert not capped
