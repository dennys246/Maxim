"""Tests for maxim.reactions.types — Phase 1 core Reaction types."""

from __future__ import annotations

import pytest

from maxim.decisions.causal_link import Valence
from maxim.reactions.types import (
    Reaction,
    ReactionContext,
    TraceSnapshot,
)
from maxim.time.circadian import CircadianContext


# ---------------------------------------------------------------------------
# TraceSnapshot
# ---------------------------------------------------------------------------


class TestTraceSnapshot:
    def test_defaults(self):
        snap = TraceSnapshot(percept_id="p-001")
        assert snap.percept_id == "p-001"
        assert snap.activation_strength == 1.0
        assert snap.content_hash is None
        assert snap.decay_factor == 1.0

    def test_round_trip(self):
        snap = TraceSnapshot(
            percept_id="p-002",
            activation_strength=0.75,
            content_hash="abc123",
            decay_factor=0.9,
        )
        restored = TraceSnapshot.from_dict(snap.to_dict())
        assert restored == snap

    def test_round_trip_defaults_only(self):
        snap = TraceSnapshot(percept_id="p-003")
        restored = TraceSnapshot.from_dict(snap.to_dict())
        assert restored == snap


# ---------------------------------------------------------------------------
# ReactionContext
# ---------------------------------------------------------------------------


class TestReactionContext:
    def test_defaults(self):
        ctx = ReactionContext()
        assert ctx.agent_id is None
        assert ctx.timestamp is None
        assert ctx.scn_tag is None
        assert ctx.bindings == {}

    def test_full_construction(self):
        scn = CircadianContext(phase="wake", tick=42, wall_time=1000.0)
        trigger = TraceSnapshot(percept_id="p-010", activation_strength=0.8)
        action = TraceSnapshot(percept_id="p-011", activation_strength=0.6)
        ctx = ReactionContext(
            agent_id="baby-001",
            timestamp=1234.5,
            scn_tag=scn,
            bindings={"trigger": trigger, "action": action},
        )
        assert ctx.agent_id == "baby-001"
        assert ctx.timestamp == 1234.5
        assert ctx.scn_tag == scn
        assert len(ctx.bindings) == 2
        assert ctx.bindings["trigger"] == trigger
        assert ctx.bindings["action"] == action

    def test_round_trip_with_nested(self):
        scn = CircadianContext(phase="rest", tick=99, wall_time=2000.0)
        ctx = ReactionContext(
            agent_id="agent-x",
            timestamp=555.5,
            scn_tag=scn,
            bindings={
                "trigger": TraceSnapshot(percept_id="p-a", activation_strength=0.9, content_hash="h1"),
                "action": TraceSnapshot(percept_id="p-b", decay_factor=0.5),
                "expectation": TraceSnapshot(percept_id="p-c"),
            },
        )
        restored = ReactionContext.from_dict(ctx.to_dict())
        assert restored == ctx

    def test_round_trip_empty(self):
        ctx = ReactionContext()
        restored = ReactionContext.from_dict(ctx.to_dict())
        assert restored == ctx

    def test_frozen(self):
        ctx = ReactionContext(agent_id="test")
        with pytest.raises((AttributeError, TypeError)):
            ctx.agent_id = "other"  # type: ignore[misc]

    def test_multiple_binding_roles(self):
        """Ensure different reaction kinds can use different binding roles."""
        ctx = ReactionContext(
            bindings={
                "trigger": TraceSnapshot(percept_id="stove-visual"),
                "action": TraceSnapshot(percept_id="reach-motor"),
                "expectation": TraceSnapshot(percept_id="safe-grasp"),
            },
        )
        assert set(ctx.bindings.keys()) == {"trigger", "action", "expectation"}


# ---------------------------------------------------------------------------
# Reaction
# ---------------------------------------------------------------------------


class TestReaction:
    def _make_reaction(self, **overrides):
        defaults = dict(
            kind="pain",
            intensity=0.8,
            valence=Valence.NEGATIVE,
            timestamp=1000.0,
            context=ReactionContext(
                agent_id="baby",
                bindings={
                    "trigger": TraceSnapshot(percept_id="hot-stove"),
                },
            ),
            source="pain_detector:velocity",
        )
        defaults.update(overrides)
        return Reaction(**defaults)

    def test_full_construction(self):
        r = self._make_reaction()
        assert r.kind == "pain"
        assert r.intensity == 0.8
        assert r.valence == Valence.NEGATIVE
        assert r.timestamp == 1000.0
        assert r.source == "pain_detector:velocity"
        assert r.context.agent_id == "baby"

    def test_round_trip(self):
        r = self._make_reaction(
            kind="surprise",
            intensity=0.5,
            valence=Valence.POSITIVE,
            source="cerebellum:prediction_error",
            context=ReactionContext(
                agent_id="agent-1",
                timestamp=42.0,
                scn_tag=CircadianContext(phase="wake", tick=10),
                bindings={
                    "trigger": TraceSnapshot(
                        percept_id="p-100",
                        activation_strength=0.95,
                        content_hash="xyz",
                    ),
                    "expectation": TraceSnapshot(percept_id="p-101"),
                },
            ),
        )
        restored = Reaction.from_dict(r.to_dict())
        assert restored == r

    def test_frozen(self):
        r = self._make_reaction()
        with pytest.raises((AttributeError, TypeError)):
            r.intensity = 0.5  # type: ignore[misc]

    def test_valence_serializes_as_string(self):
        r = self._make_reaction(valence=Valence.NEUTRAL)
        d = r.to_dict()
        assert d["valence"] == "neutral"
        restored = Reaction.from_dict(d)
        assert restored.valence == Valence.NEUTRAL


# ---------------------------------------------------------------------------
# Isolation-hygiene docstring assertion
# ---------------------------------------------------------------------------


class TestIsolationHygiene:
    def test_isolation_hygiene_rule_documented(self):
        """The module docstring must spell out the banned-field rule.

        This is a review-discipline rule with no code enforcement --
        the docstring is the contract. If someone trims the docstring
        without replacing the rule, the test points them back here.
        """
        from maxim.reactions import types

        doc = types.__doc__ or ""
        assert "isolation hygiene" in doc.lower()
        assert "cross-agent intent" in doc.lower()
        # Concrete banned examples must stay in the docstring.
        for example in (
            "mother_lesson",
            "sender_reward_history",
            "expected_reaction_kind",
            "optimal_action_from_nac",
        ):
            assert example in doc, f"isolation rule must reference {example}"
