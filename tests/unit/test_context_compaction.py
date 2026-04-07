"""Tests for conversation compaction (Phase 3 #10 + PERF-4)."""

from __future__ import annotations


from maxim.agents.prompt_budgeter import _compact_conversation


def _build_conversation(n_turns: int) -> str:
    """Build a fake conversation with n_turns alternating User/Maxim lines."""
    lines = []
    for i in range(n_turns):
        if i % 2 == 0:
            lines.append(f"User: message {i}")
        else:
            lines.append(f"Maxim: reply {i}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. No compaction needed
# ---------------------------------------------------------------------------


class TestNoCompaction:
    """When turns <= max_turns, content is unchanged."""

    def test_five_turns_max_twelve(self) -> None:
        content = _build_conversation(5)
        result = _compact_conversation(content, max_turns=12)
        assert result == content

    def test_equal_to_max(self) -> None:
        content = _build_conversation(12)
        result = _compact_conversation(content, max_turns=12)
        assert result == content


# ---------------------------------------------------------------------------
# 2. Compaction keeps first + last N
# ---------------------------------------------------------------------------


class TestCompactionKeepsFirstAndLastN:
    """With 20 turns and max_turns=5, first turn + last 4 should be kept."""

    def test_first_plus_last_four(self) -> None:
        content = _build_conversation(20)
        result = _compact_conversation(content, max_turns=5)

        result_lines = result.split("\n")

        # First line should be the very first turn
        assert result_lines[0] == "User: message 0"

        # Last 4 turns should be present (turns 16, 17, 18, 19)
        assert "User: message 16" in result
        assert "Maxim: reply 17" in result
        assert "User: message 18" in result
        assert "Maxim: reply 19" in result

        # Middle turns should be dropped
        assert "User: message 2" not in result
        assert "Maxim: reply 3" not in result
        assert "User: message 10" not in result


# ---------------------------------------------------------------------------
# 3. Compaction with single turn
# ---------------------------------------------------------------------------


class TestSingleTurnCompaction:
    """A single turn should remain unchanged regardless of max_turns."""

    def test_single_turn_unchanged(self) -> None:
        content = "User: hello"
        result = _compact_conversation(content, max_turns=12)
        assert result == content

    def test_single_turn_max_one(self) -> None:
        content = "User: hello"
        result = _compact_conversation(content, max_turns=1)
        assert result == content


# ---------------------------------------------------------------------------
# 4. LongHorizonConfig has compact_after_turns
# ---------------------------------------------------------------------------


class TestLongHorizonConfigField:
    """LongHorizonConfig should have compact_after_turns with default 12."""

    def test_field_exists_with_default(self) -> None:
        from maxim.planning.plan_document import LongHorizonConfig

        config = LongHorizonConfig()
        assert hasattr(config, "compact_after_turns")
        assert config.compact_after_turns == 12
