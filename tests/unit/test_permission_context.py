"""Unit tests for PermissionDenial and SupervisionPolicy permission context (Phase 2, #3)."""

from __future__ import annotations

import dataclasses

import pytest

from maxim.agents.autonomy import PermissionDenial, SupervisionPolicy


class TestPermissionDenial:
    """Tests for the frozen PermissionDenial dataclass."""

    def test_creation(self) -> None:
        """Frozen instance stores all fields correctly."""
        denial = PermissionDenial(
            tool_name="reachy_move",
            reason="blocked by prefix",
            rule="forbidden_prefix",
        )

        assert denial.tool_name == "reachy_move"
        assert denial.reason == "blocked by prefix"
        assert denial.rule == "forbidden_prefix"

    def test_immutable(self) -> None:
        """Mutating a frozen dataclass raises FrozenInstanceError."""
        denial = PermissionDenial(
            tool_name="x", reason="r", rule="forbidden_tool",
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            denial.tool_name = "y"  # type: ignore[misc]


class TestSupervisionPolicyPrefixes:
    """Tests for forbidden_prefixes on SupervisionPolicy."""

    def test_forbidden_prefix_blocks(self) -> None:
        """A tool whose name starts with a forbidden prefix is blocked."""
        policy = SupervisionPolicy(forbidden_prefixes=("reachy_",))
        action = {"tool_name": "reachy_move", "params": {}}

        allowed, reason = policy.can_execute(action)

        assert allowed is False
        assert "prefix" in reason.lower()

    def test_forbidden_prefix_allows_non_matching(self) -> None:
        """A tool that does not match any forbidden prefix is not blocked by that rule."""
        policy = SupervisionPolicy(forbidden_prefixes=("reachy_",))
        action = {"tool_name": "read_file", "params": {}}

        allowed, _reason = policy.can_execute(action)

        # read_file is in ALWAYS_ALLOWED_TOOLS at the controller level,
        # but SupervisionPolicy.can_execute itself should not block it.
        assert allowed is True


class TestSupervisionPolicyCategories:
    """Tests for forbidden_categories on SupervisionPolicy."""

    def test_forbidden_category_blocks(self) -> None:
        """An action whose category is forbidden is blocked."""
        policy = SupervisionPolicy(
            forbidden_categories=frozenset({"initiative"}),
        )
        action = {
            "tool_name": "suggest_action",
            "category": "initiative",
            "params": {},
        }

        allowed, reason = policy.can_execute(action)

        assert allowed is False
        assert "category" in reason.lower()


class TestSupervisionPolicyBackwardCompat:
    """Backward compatibility: default SupervisionPolicy works as before."""

    def test_default_policy_allows_unknown_tool(self) -> None:
        """Default policy (no allowed_tools set, no forbidden rules) allows a generic tool."""
        policy = SupervisionPolicy()
        action = {"tool_name": "some_tool", "params": {}}

        allowed, _reason = policy.can_execute(action)

        assert allowed is True

    def test_default_fields(self) -> None:
        """New fields have sensible defaults that do not alter legacy behavior."""
        policy = SupervisionPolicy()

        assert policy.forbidden_prefixes == ()
        assert policy.forbidden_categories == frozenset()
