"""Unit tests for Phase 2a — ReactionBus + PainBus backward compat."""

from __future__ import annotations

import threading
import time

from maxim.decisions.causal_link import Valence
from maxim.proprioception.pain import PainSignal, PainType
from maxim.proprioception.pain_bus import PainBus
from maxim.reactions.bus import ReactionBus, build_reaction_bus
from maxim.reactions.types import Reaction, ReactionContext


def _make_pain_reaction(source: str = "test:pain", intensity: float = 0.5) -> Reaction:
    return Reaction(
        kind="pain",
        intensity=intensity,
        valence=Valence.NEGATIVE,
        timestamp=time.time(),
        source=source,
        context=ReactionContext(),
    )


class TestReactionBus:
    def test_subscribe_and_publish(self):
        bus = ReactionBus()
        received: list[Reaction] = []
        bus.subscribe("pain", received.append)
        bus.publish(_make_pain_reaction())
        assert len(received) == 1
        assert received[0].kind == "pain"

    def test_subscribe_all(self):
        bus = ReactionBus()
        received: list[Reaction] = []
        bus.subscribe_all(received.append)
        bus.publish(_make_pain_reaction())
        assert len(received) == 1

    def test_kind_filtering(self):
        bus = ReactionBus()
        pain_received: list[Reaction] = []
        fear_received: list[Reaction] = []
        bus.subscribe("pain", pain_received.append)
        bus.subscribe("fear", fear_received.append)

        bus.publish(_make_pain_reaction())
        assert len(pain_received) == 1
        assert len(fear_received) == 0

    def test_refractory_period(self):
        bus = ReactionBus(refractory_overrides={"pain": 0.5})
        received: list[Reaction] = []
        bus.subscribe("pain", received.append)

        r = _make_pain_reaction(source="test:same_source")
        bus.publish(r)
        bus.publish(r)  # Should be blocked by refractory
        assert len(received) == 1

    def test_different_sources_bypass_refractory(self):
        bus = ReactionBus(refractory_overrides={"pain": 0.5})
        received: list[Reaction] = []
        bus.subscribe("pain", received.append)

        bus.publish(_make_pain_reaction(source="test:a"))
        bus.publish(_make_pain_reaction(source="test:b"))
        assert len(received) == 2

    def test_history(self):
        bus = ReactionBus()
        bus.publish(_make_pain_reaction(source="test:1"))
        bus.publish(_make_pain_reaction(source="test:2"))
        assert len(bus.history()) == 2
        assert len(bus.history("pain")) == 2
        assert len(bus.history("fear")) == 0

    def test_get_stats(self):
        bus = ReactionBus()
        bus.subscribe("pain", lambda r: None)
        bus.subscribe_all(lambda r: None)
        bus.publish(_make_pain_reaction())
        stats = bus.get_stats()
        assert stats["total_published"] == 1
        assert stats["subscriber_count"] == 2
        assert stats["history_size"] == 1

    def test_thread_safety(self):
        bus = ReactionBus()
        received: list[Reaction] = []
        bus.subscribe("pain", received.append)

        def publish_many():
            for i in range(20):
                bus.publish(_make_pain_reaction(source=f"thread:{threading.current_thread().name}:{i}"))

        threads = [threading.Thread(target=publish_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 80

    def test_unsubscribe(self):
        bus = ReactionBus()
        received: list[Reaction] = []
        bus.subscribe("pain", received.append)
        bus.unsubscribe("pain", received.append)
        bus.publish(_make_pain_reaction())
        assert len(received) == 0


class TestBuildReactionBus:
    """Regression guards for the canonical ReactionBus construction door.

    See ``docs/plans/reaction_bus_unification.md`` for the audit.
    The builder exists for downstream Wave 3 sequencing
    (``build_bio_stack`` constructs ReactionBus BEFORE PainBus).
    """

    def test_returns_reaction_bus_instance(self):
        bus = build_reaction_bus()
        assert isinstance(bus, ReactionBus)

    def test_minimal_construction_no_subscribers(self):
        """Empty construction is the test / standalone path."""
        bus = build_reaction_bus()
        stats = bus.get_stats()
        assert stats["subscriber_count"] == 0
        assert stats["total_published"] == 0

    def test_per_kind_subscribers_registered(self):
        received: list[Reaction] = []
        bus = build_reaction_bus(
            per_kind_subscribers={"pain": (received.append,)},
        )
        bus.publish(_make_pain_reaction())
        assert len(received) == 1
        assert received[0].kind == "pain"

    def test_per_kind_multiple_kinds(self):
        pain_received: list[Reaction] = []
        fear_received: list[Reaction] = []
        bus = build_reaction_bus(
            per_kind_subscribers={
                "pain": (pain_received.append,),
                "fear": (fear_received.append,),
            },
        )
        bus.publish(_make_pain_reaction())
        assert len(pain_received) == 1
        assert len(fear_received) == 0

    def test_all_subscribers_registered(self):
        received: list[Reaction] = []
        bus = build_reaction_bus(
            all_subscribers=(received.append,),
        )
        bus.publish(_make_pain_reaction())
        assert len(received) == 1

    def test_per_kind_and_all_both_fire(self):
        """Both per-kind and subscribe_all callbacks fire on the same publish."""
        per_kind: list[Reaction] = []
        all_sub: list[Reaction] = []
        bus = build_reaction_bus(
            per_kind_subscribers={"pain": (per_kind.append,)},
            all_subscribers=(all_sub.append,),
        )
        bus.publish(_make_pain_reaction())
        assert len(per_kind) == 1
        assert len(all_sub) == 1

    def test_history_size_forwarded(self):
        bus = build_reaction_bus(history_size=42)
        assert bus._history.maxlen == 42

    def test_refractory_overrides_forwarded(self):
        received: list[Reaction] = []
        bus = build_reaction_bus(
            per_kind_subscribers={"pain": (received.append,)},
            refractory_overrides={"pain": 0.5},
        )
        r = _make_pain_reaction(source="same:source")
        bus.publish(r)
        bus.publish(r)  # blocked by refractory
        assert len(received) == 1

    def test_multiple_callbacks_per_kind(self):
        """Multiple callbacks for the same kind all register."""
        a: list[Reaction] = []
        b: list[Reaction] = []
        bus = build_reaction_bus(
            per_kind_subscribers={"pain": (a.append, b.append)},
        )
        bus.publish(_make_pain_reaction())
        assert len(a) == 1
        assert len(b) == 1


class TestPainBusBackwardCompat:
    def test_publish_pain_signal_reaches_subscriber(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.7,
            timestamp=time.time(),
        )
        bus.publish(signal)
        assert len(received) == 1
        assert received[0].pain_type == PainType.EXTERNAL_SIGNAL

    def test_pain_bus_exposes_reaction_bus(self):
        bus = PainBus()
        assert isinstance(bus.reaction_bus, ReactionBus)

    def test_reaction_bus_receives_converted_pain(self):
        bus = PainBus()
        reaction_received: list[Reaction] = []
        bus.reaction_bus.subscribe("pain", reaction_received.append)

        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.9,
            timestamp=time.time(),
        )
        bus.publish(signal)
        assert len(reaction_received) == 1
        assert reaction_received[0].kind == "pain"
        assert reaction_received[0].intensity == 0.9

    def test_recent_returns_pain_signals(self):
        bus = PainBus()
        signal = PainSignal(
            pain_type=PainType.MOVEMENT_FAILURE,
            intensity=0.6,
            timestamp=time.time(),
        )
        bus.publish(signal)
        recent = bus.recent
        assert len(recent) == 1
        assert isinstance(recent[0], PainSignal)

    def test_stats(self):
        bus = PainBus()
        bus.subscribe(lambda s: None)
        signal = PainSignal(pain_type=PainType.EXTERNAL_SIGNAL, intensity=0.5, timestamp=time.time())
        bus.publish(signal)
        stats = bus.get_stats()
        assert stats["total_published"] == 1
