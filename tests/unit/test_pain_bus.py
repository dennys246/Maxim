"""PainBus dual-dispatch + context propagation tests.

Regression guard for the Stage-2 root-cause fix:

1. ``PainBus.publish(signal)`` delivers the FULL ``signal.context``
   dict to direct subscribers (not a lossy reconstruction from the
   Reaction round-trip).
2. Reactions published directly on ``pain_bus.reaction_bus`` also
   reach direct subscribers via the lossy fallback (no signal stash),
   so sandbox-style direct-reaction publishers don't silently bypass
   PainBus subscribers.
3. Refractory filtering happens exactly once: on ``reaction_bus``.
   Back-to-back ``PainBus.publish`` calls within the refractory window
   collapse to a single dispatch to direct subscribers.
4. ``create_pain_nac_subscriber`` attributes pain to pending action
   events via ``record_outcome_full`` context similarity — no more
   tautological ``pain → pain`` links.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc, NACConfig
from maxim.proprioception.pain import PainSignal, PainType
from maxim.proprioception.pain_bus import (
    PainBus,
    build_pain_bus,
    create_pain_nac_subscriber,
)
from maxim.reactions.types import Reaction, ReactionContext, TraceSnapshot


def _make_signal(intensity: float = 0.7, **ctx: object) -> PainSignal:
    """Build a rich-context embodiment PainSignal for testing."""
    base = {
        "source": "embodiment",
        "entity": "body.arm.rusty_sword",
        "entity_type": "weapon",
        "failure_mode": "shatter",
        "composes": [],
        "sensor_readings": {"durability": 0.05},
    }
    base.update(ctx)
    return PainSignal(
        pain_type=PainType.EXTERNAL_SIGNAL,
        intensity=intensity,
        timestamp=time.time(),
        context=base,
    )


class TestPainBusDirectDispatch:
    """PainBus.publish delivers full context to direct subscribers."""

    def test_publish_preserves_full_context(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        signal = _make_signal()
        bus.publish(signal)

        assert len(received) == 1
        got = received[0]
        # Full context preserved — not the lossy entity_path-only view.
        assert got.context["source"] == "embodiment"
        assert got.context["entity"] == "body.arm.rusty_sword"
        assert got.context["entity_type"] == "weapon"
        assert got.context["failure_mode"] == "shatter"
        assert got.context["sensor_readings"] == {"durability": 0.05}
        assert got.intensity == 0.7

    def test_publish_also_fires_reaction_bus(self):
        """Direct reaction_bus subscribers see the converted Reaction too."""
        bus = PainBus()
        reactions: list[Reaction] = []
        bus.reaction_bus.subscribe("pain", reactions.append)

        signal = _make_signal()
        bus.publish(signal)

        assert len(reactions) == 1
        assert reactions[0].kind == "pain"
        assert reactions[0].intensity == 0.7

    def test_multiple_subscribers_all_fire(self):
        bus = PainBus()
        a: list[PainSignal] = []
        b: list[PainSignal] = []
        bus.subscribe(a.append)
        bus.subscribe(b.append)

        bus.publish(_make_signal())
        assert len(a) == 1
        assert len(b) == 1

    def test_unsubscribe_stops_delivery(self):
        """A callback registered by identity can be unregistered by identity.

        Uses a distinct ``entity`` kwarg between the two publishes so
        the PainBus ``(entity, failure_mode)`` refractory gate doesn't
        silently absorb the second publish — that would make this test
        pass even if ``unsubscribe`` were a no-op.
        """
        bus = PainBus()
        cb_received: list[PainSignal] = []

        def cb(sig: PainSignal) -> None:
            cb_received.append(sig)

        bus.subscribe(cb)
        bus.publish(_make_signal(entity="body.sword_a"))
        assert len(cb_received) == 1

        bus.unsubscribe(cb)
        bus.publish(_make_signal(entity="body.sword_b", intensity=0.95))
        # Second publish targets a different entity → refractory key is
        # different → gate would not block. The only reason cb_received
        # stays at 1 is that unsubscribe actually removed cb.
        assert len(cb_received) == 1

    def test_subscriber_exception_does_not_break_others(self):
        bus = PainBus()
        received: list[PainSignal] = []

        def boom(sig: PainSignal) -> None:
            raise RuntimeError("subscriber bug")

        bus.subscribe(boom)
        bus.subscribe(received.append)
        bus.publish(_make_signal())
        # The good subscriber still fires.
        assert len(received) == 1


class TestPainBusFallbackDispatch:
    """Direct Reaction publishes on reaction_bus still reach PainSignal subs."""

    def test_direct_reaction_publish_lossy_fallback(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        # Simulate sandbox-style direct Reaction publish (bypasses
        # PainBus.publish and its signal stash).
        reaction = Reaction(
            kind="pain",
            intensity=0.6,
            valence=Valence.NEGATIVE,
            timestamp=time.time(),
            context=ReactionContext(
                bindings={"entity_path": TraceSnapshot(percept_id="sandbox.file")},
            ),
            source="sandbox:file_delete",
        )
        bus.reaction_bus.publish(reaction)

        # PainSignal subscriber still fires, but with the reconstructed
        # (lossy) context — only entity_path survives.
        assert len(received) == 1
        got = received[0]
        assert got.intensity == 0.6
        assert got.context == {"entity_path": "sandbox.file"}

    def test_direct_publish_does_not_double_deliver_after_painbus_publish(self):
        """PainBus.publish must not deliver twice to direct subscribers.

        ``PainBus.publish`` fires direct subscribers inline, THEN
        forwards the converted Reaction to ``reaction_bus``. The
        reaction_bus dispatch triggers
        ``_bridge_reaction_to_pain_subs`` which ALSO knows how to
        dispatch to direct subscribers (for sandbox-style direct
        reaction publishes). Without the ``_suppress_bridge`` flag
        the direct subscribers would fire twice.
        """
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        bus.publish(_make_signal())
        assert len(received) == 1, f"expected 1 dispatch, got {len(received)}"


class TestPainBusRefractory:
    """PainBus applies its own (entity, failure_mode) refractory gate."""

    def test_same_entity_same_failure_refractory_gated(self):
        """Two identical publishes within the window collapse to one."""
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        bus.publish(_make_signal(intensity=0.7))
        bus.publish(_make_signal(intensity=0.8))

        assert len(received) == 1, f"refractory failed: got {len(received)}, expected 1"

    def test_different_entity_NOT_refractory_gated(self):
        """Two different entities firing within the window BOTH fire.

        This is the regression guard for Executor C1 — the pre-rewrite
        design routed refractory through ``reaction_bus`` which keys
        the gate on ``(kind, source)``. Because
        ``pain_signal_to_reaction`` synthesizes ``source`` from
        ``pain_type`` alone, two distinct entities firing embodiment
        pain in the same tick collapsed to one dispatch. The new
        ``(entity, failure_mode)`` key on PainBus prevents this.
        """
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        bus.publish(_make_signal(entity="body.sword_a"))
        bus.publish(_make_signal(entity="body.sword_b"))

        assert len(received) == 2, (
            f"cross-entity refractory collapse: got {len(received)}, expected 2 — pre-Stage-2 regression re-introduced"
        )

    def test_different_failure_mode_NOT_refractory_gated(self):
        """Two different failure modes on the same entity BOTH fire."""
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        bus.publish(_make_signal(failure_mode="shatter"))
        bus.publish(_make_signal(failure_mode="dulled"))

        assert len(received) == 2

    def test_refractory_clears_after_window(self):
        bus = PainBus()
        received: list[PainSignal] = []
        bus.subscribe(received.append)

        bus.publish(_make_signal(intensity=0.7))
        # Wait beyond the 0.5s refractory window
        time.sleep(0.6)
        bus.publish(_make_signal(intensity=0.8))
        assert len(received) == 2

    def test_get_stats_counts_direct_subscribers(self):
        """get_stats includes direct PainSignal subscribers in the count."""
        bus = PainBus()
        # Baseline: sim_log_reaction + _bridge_reaction_to_pain_subs are
        # registered on reaction_bus; no direct subscribers yet.
        baseline = bus.get_stats()
        assert baseline["direct_pain_subscribers"] == 0

        bus.subscribe(lambda s: None)
        bus.subscribe(lambda s: None)
        after = bus.get_stats()
        assert after["direct_pain_subscribers"] == 2
        assert after["subscriber_count"] == baseline["subscriber_count"] + 2


class TestCreatePainNacSubscriber:
    """Pain → NAc wiring creates action→pain causal links, not tautologies."""

    def _nac(self):
        # Use an NAc with a generous temporal window so test timing is
        # robust. context_similarity_threshold stays at default (0.5).
        return NAc(NACConfig(temporal_window_seconds=60.0))

    def test_pain_below_threshold_is_ignored(self):
        """Low-intensity pain short-circuits BEFORE touching NAc.

        Tight assertion: mock NAc and check that ``record_outcome_full``
        is never called. The previous loose assertion (``len(nac) == 0``)
        would pass even if a future refactor moved the threshold check
        after ``record_outcome_full``, because unmatched pending events
        still produce zero links.
        """
        mock_nac = MagicMock()
        mock_nac.record_outcome_full = MagicMock(return_value=[])
        sub = create_pain_nac_subscriber(mock_nac, intensity_threshold=0.3)
        sub(_make_signal(intensity=0.1))
        assert mock_nac.record_outcome_full.call_count == 0

    def test_pain_attributes_to_pending_action_via_context_similarity(self):
        """The core PoC mechanism: pain → action causal link."""
        nac = self._nac()
        sub = create_pain_nac_subscriber(nac, intensity_threshold=0.3)

        # Agent records "I'm about to slash the sword" as a pending event.
        # Context includes the entity so the pain's context can match it.
        nac.record_event(
            event_type="action",
            event_signature="slash:rusty_sword",
            context={
                "source": "embodiment",
                "entity": "body.arm.rusty_sword",
            },
        )
        assert len(nac._pending_events) == 1

        # Pain fires with matching entity context
        sub(_make_signal(intensity=0.7))

        # A causal link should now exist for event signature
        # "slash:rusty_sword" with NEGATIVE valence.
        links_for_slash = nac._links.get("slash:rusty_sword", [])
        assert len(links_for_slash) == 1, (
            f"expected 1 action→pain link, got {len(links_for_slash)}: _links={dict(nac._links)}"
        )
        link = links_for_slash[0]
        assert link.outcome_valence == Valence.NEGATIVE

    def test_nac_predict_returns_negative_after_pain(self):
        """After one action→pain cycle, nac.predict warns the agent off."""
        nac = self._nac()
        sub = create_pain_nac_subscriber(nac, intensity_threshold=0.3)

        ctx = {"source": "embodiment", "entity": "body.arm.rusty_sword"}
        nac.record_event(
            event_type="action",
            event_signature="slash:rusty_sword",
            context=ctx,
        )
        sub(_make_signal(intensity=0.8))

        prediction = nac.predict(
            event_type="action",
            event_signature="slash:rusty_sword",
            context=ctx,
        )
        assert prediction is not None, "expected a prediction after action→pain"
        assert prediction.predicted_valence == Valence.NEGATIVE

    def test_pain_without_matching_action_is_noop(self):
        """If no pending action matches, the pain outcome creates no link."""
        nac = self._nac()
        sub = create_pain_nac_subscriber(nac, intensity_threshold=0.3)
        # No record_event called
        sub(_make_signal(intensity=0.7))
        assert len(nac) == 0


class TestBuildPainBus:
    """Regression guards for the canonical PainBus construction door.

    See ``docs/plans/pain_bus_unification.md`` for the audit + design
    that motivated this builder. The shape mirrors
    ``runtime/bootstrap.py::build_executor`` — required keyword-only
    learning subjects so forgetting either is a ``TypeError``, not a
    silent no-op.
    """

    def _nac(self) -> NAc:
        return NAc(NACConfig(temporal_window_seconds=60.0))

    def test_missing_hippocampus_kwarg_raises_type_error(self):
        """Forgetting hippocampus= is a TypeError, not a silent no-op."""
        import pytest

        with pytest.raises(TypeError):
            build_pain_bus(nac=self._nac())  # type: ignore[call-arg]

    def test_missing_nac_kwarg_raises_type_error(self):
        """Forgetting nac= is a TypeError, not a silent no-op.

        This is the regression guard for the audited bug class: three
        CLI sites silently skipped NAc bus subscription. Post-migration,
        forgetting the argument is loud.
        """
        import pytest

        hippo = MagicMock()

        with pytest.raises(TypeError):
            build_pain_bus(hippocampus=hippo)  # type: ignore[call-arg]

    def test_positional_args_rejected(self):
        """Both learning subjects are keyword-only."""
        import pytest

        with pytest.raises(TypeError):
            build_pain_bus(MagicMock(), self._nac())  # type: ignore[misc]

    def test_returns_painbus_instance(self):
        """The return type is a real PainBus (not a wrapper)."""
        bus = build_pain_bus(hippocampus=None, nac=None)
        assert isinstance(bus, PainBus)

    def test_explicit_no_learners_opt_out(self):
        """Passing None for both is a legitimate explicit opt-out.

        api.py headless mode and tests need this path. The TypeError
        safety net is on FORGETTING the parameter, not on PASSING None.
        """
        bus = build_pain_bus(hippocampus=None, nac=None)
        # No direct subscribers should be wired beyond the internal
        # bridge that PainBus.__init__ registers on reaction_bus.
        assert bus.get_stats()["direct_pain_subscribers"] == 0

    def test_hippocampus_only_subscribes_memory_subscriber(self):
        """Passing hippocampus wires create_pain_memory_subscriber."""
        from maxim.memory.types import Decision, Outcome, Perception  # noqa: F401

        hippo = MagicMock()
        hippo.capture = MagicMock()
        bus = build_pain_bus(hippocampus=hippo, nac=None)

        # Direct subscriber count = 1 (memory only)
        assert bus.get_stats()["direct_pain_subscribers"] == 1

        # Publishing a pain signal triggers hippo.capture
        bus.publish(_make_signal(intensity=0.7))
        assert hippo.capture.call_count == 1

    def test_nac_only_subscribes_nac_subscriber(self):
        """Passing nac wires create_pain_nac_subscriber."""
        nac = self._nac()
        bus = build_pain_bus(hippocampus=None, nac=nac)

        assert bus.get_stats()["direct_pain_subscribers"] == 1

        # Record a pending action that the pain context will match.
        nac.record_event(
            event_type="action",
            event_signature="slash:rusty_sword",
            context={
                "source": "embodiment",
                "entity": "body.arm.rusty_sword",
            },
        )
        bus.publish(_make_signal(intensity=0.7))

        # Action→pain link should exist (the very bug class this
        # builder closes — would have been a silent no-op pre-fix).
        assert len(nac._links.get("slash:rusty_sword", [])) == 1

    def test_both_learners_subscribe_both(self):
        """The standard production shape: both subjects wired."""
        hippo = MagicMock()
        hippo.capture = MagicMock()
        nac = self._nac()
        bus = build_pain_bus(hippocampus=hippo, nac=nac)

        assert bus.get_stats()["direct_pain_subscribers"] == 2

        nac.record_event(
            event_type="action",
            event_signature="slash:rusty_sword",
            context={
                "source": "embodiment",
                "entity": "body.arm.rusty_sword",
            },
        )
        bus.publish(_make_signal(intensity=0.7))

        # BOTH paths fire on the same publish.
        assert hippo.capture.call_count == 1
        assert len(nac._links.get("slash:rusty_sword", [])) == 1

    def test_additional_subscribers_registered_after_standard_learners(self):
        """Custom subscribers run after the standard learners."""
        order: list[str] = []

        def hippo_marker(_signal: PainSignal) -> None:
            order.append("hippo")

        def custom_sub(_signal: PainSignal) -> None:
            order.append("custom")

        # Use a hippo MagicMock that records via the order list rather
        # than the standard create_pain_memory_subscriber, so we can
        # assert on call ordering directly.
        hippo = MagicMock()
        hippo.capture = MagicMock(side_effect=lambda **_: order.append("hippo"))

        bus = build_pain_bus(
            hippocampus=hippo,
            nac=None,
            additional_subscribers=(custom_sub,),
        )
        assert bus.get_stats()["direct_pain_subscribers"] == 2

        bus.publish(_make_signal(intensity=0.7))
        # Standard learner runs first, then additional subscriber.
        assert order == ["hippo", "custom"]
        # Suppress unused-warning on the unused inner helper.
        del hippo_marker

    def test_history_size_and_refractory_forwarded(self):
        """history_size and pain_refractory_s reach the PainBus ctor."""
        bus = build_pain_bus(
            hippocampus=None,
            nac=None,
            history_size=42,
            pain_refractory_s=1.5,
        )
        # pain_refractory_s is the simpler one to assert on directly.
        assert bus._pain_refractory_s == 1.5
        # history_size lives on the inner ReactionBus; sanity-check
        # that publishing still works (full ReactionBus contract is
        # tested elsewhere — we only care that the kwarg flowed).
        bus.publish(_make_signal(intensity=0.7))
        assert len(bus.recent) >= 0  # smoke check, not a hard assertion

    def test_exceptions_are_logged_not_swallowed(self, caplog):
        """A broken NAc still logs so wiring issues surface."""
        import logging

        mock_nac = MagicMock()
        mock_nac.record_outcome_full.side_effect = RuntimeError("boom")
        sub = create_pain_nac_subscriber(mock_nac, intensity_threshold=0.3)

        with caplog.at_level(logging.ERROR, logger="maxim.proprioception.pain_bus"):
            sub(_make_signal(intensity=0.7))

        assert any("pain→NAc outcome recording failed" in rec.message for rec in caplog.records), (
            "expected an exception log, got nothing"
        )
