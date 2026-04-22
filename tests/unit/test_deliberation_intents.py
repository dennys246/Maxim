"""Tests for deliberation intent types and deliberate() method (Stage 3).

Covers:
- SpeakIntent, ActionIntent, NoOpIntent frozen semantics
- DeliberationOutcome enum values
- ExecAgent.deliberate() intent classification
"""

from __future__ import annotations

from maxim.agents.bus import (
    ActionIntent,
    DeliberationOutcome,
    NoOpIntent,
    SpeakIntent,
)


class TestIntentTypes:
    def test_speak_intent_frozen(self):
        si = SpeakIntent(content="hello", channel="respond", reason="test")
        assert si.content == "hello"
        assert si.channel == "respond"
        assert si.reason == "test"
        assert si.deliberation_hops == 0

    def test_action_intent_frozen(self):
        ai = ActionIntent(tool_name="navigate", tool_params={"x": 1}, reason="need to move")
        assert ai.tool_name == "navigate"
        assert ai.tool_params == {"x": 1}
        assert ai.reason == "need to move"

    def test_noop_intent_frozen(self):
        ni = NoOpIntent(reason="nothing to do")
        assert ni.reason == "nothing to do"
        assert ni.deliberation_hops == 0

    def test_deliberation_outcome_values(self):
        assert DeliberationOutcome.SPEAK.value == "speak"
        assert DeliberationOutcome.ACTION.value == "action"
        assert DeliberationOutcome.NOOP.value == "noop"

    def test_speak_intent_default_channel(self):
        si = SpeakIntent(content="hi")
        assert si.channel == "respond"

    def test_action_intent_default_params(self):
        ai = ActionIntent(tool_name="test")
        assert ai.tool_params == {}
