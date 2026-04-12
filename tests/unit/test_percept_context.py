"""Unit tests for F0.4 — PerceptContext schema + Percept round-trip."""

from __future__ import annotations

from maxim.agents.bus import Percept
from maxim.agents.percept_context import PerceptContext
from maxim.time.circadian import CircadianContext


class TestCircadianContext:
    def test_defaults_all_none(self):
        c = CircadianContext()
        assert c.phase is None and c.tick is None and c.wall_time is None

    def test_round_trip(self):
        c = CircadianContext(phase="wake", tick=42, wall_time=123.4)
        r = CircadianContext.from_dict(c.to_dict())
        assert r == c


class TestPerceptContext:
    def test_defaults_all_none(self):
        ctx = PerceptContext()
        assert ctx.channel is None
        assert ctx.sender is None
        assert ctx.thread_id is None
        assert ctx.subject is None
        assert ctx.timestamp is None
        assert ctx.latency_class is None
        assert ctx.scn_tag is None
        assert ctx.agent_id is None

    def test_full_construction(self):
        scn = CircadianContext(phase="wake", tick=7, wall_time=1.0)
        ctx = PerceptContext(
            channel="sms",
            sender="alice",
            thread_id="conv-42",
            subject="hello",
            timestamp=1.5,
            latency_class="seconds",
            scn_tag=scn,
            agent_id="agent-1",
        )
        assert ctx.channel == "sms"
        assert ctx.scn_tag.phase == "wake"

    def test_round_trip_preserves_scn_tag(self):
        ctx = PerceptContext(
            channel="email",
            sender="bob",
            scn_tag=CircadianContext(phase="sleep_rem", tick=9),
        )
        restored = PerceptContext.from_dict(ctx.to_dict())
        assert restored == ctx
        assert isinstance(restored.scn_tag, CircadianContext)

    def test_is_frozen(self):
        """PerceptContext is frozen so producers cannot mutate it in-flight."""
        ctx = PerceptContext(channel="sms")
        try:
            ctx.channel = "email"  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("PerceptContext must be frozen")

    def test_isolation_hygiene_rule_documented(self):
        """The module docstring must spell out the banned-field rule.

        This is a review-discipline rule with no code enforcement —
        the docstring is the contract. If someone trims the docstring
        without replacing the rule, the test points them back here.
        """
        from maxim.agents import percept_context

        doc = percept_context.__doc__ or ""
        assert "isolation hygiene" in doc.lower()
        assert "cross-agent intent" in doc.lower()
        # Concrete banned examples must stay in the docstring.
        for example in ("mother_intent", "peer_reward_hint", "scenario_answer"):
            assert example in doc, f"isolation rule must reference {example}"


class TestPerceptRoundTrip:
    def test_percept_without_context_round_trips(self):
        p = Percept(timestamp=1.0, source="vision", salience=0.5)
        r = Percept.from_dict(p.to_dict())
        assert r.timestamp == 1.0
        assert r.source == "vision"
        assert r.context is None
        assert r.modality is None

    def test_percept_with_typed_context_round_trips(self):
        ctx = PerceptContext(
            channel="sms",
            sender="alice",
            scn_tag=CircadianContext(phase="wake", tick=3),
        )
        p = Percept(
            timestamp=2.0,
            source="comms:sms",
            content="hi",
            context=ctx,
            modality="text",
        )
        r = Percept.from_dict(p.to_dict())
        assert r.modality == "text"
        assert isinstance(r.context, PerceptContext)
        assert r.context.channel == "sms"
        assert r.context.sender == "alice"
        assert isinstance(r.context.scn_tag, CircadianContext)
        assert r.context.scn_tag.phase == "wake"

    def test_percept_preserves_metadata_escape_hatch(self):
        """Non-messaging metadata (pain signals, YAML extras) stays in the dict."""
        p = Percept(
            timestamp=3.0,
            source="proprioception",
            content="pain_signal",
            metadata={"pain_type": "sharp", "intensity": 0.8, "scenario_tag": "replay_1"},
            modality="intero",
        )
        r = Percept.from_dict(p.to_dict())
        assert r.metadata == {"pain_type": "sharp", "intensity": 0.8, "scenario_tag": "replay_1"}
        assert r.modality == "intero"
        assert r.context is None
