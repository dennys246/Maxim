"""P3b Stage 1 — channel-aware boundary rules + filtered retrieval.

Synthetic mechanism tests on hand-crafted episode geometry. No fixture
YAML, no metadata-grep baseline, no 10-seed sweep — those are Stage 2.

Test classes:
- TestP3bRuleFactories — channel_gap_rule, sender_change_rule,
  channel_specific_rule (incl. the cold-start guard regression for
  Round 1 Exec critical #1 and the event.channel gating regression
  for Round 1 Exec important #2).
- TestP3bRuleComposition — additive composition with P3a defaults.
- TestP3bMembershipFilter — EpisodeStore.episode_membership_filter
  in both ANY and EXCLUSIVE modes, with channel + sender criteria
  (incl. the empty-criteria, no-episodes, and exclusive-mode
  parameter-shape regressions).
- TestP3bChannelFilteredRetrieval — Hippocampus.retrieve_on_cue
  with a channel filter (incl. the path-strict mixed-hop semantic
  regression for Round 1 Exec critical #3).
- TestP3bPersistenceContract — boundary-rule re-registration
  contract regression for Round 1 Exec important #3.
- TestP3bConcurrency — capture-thread + filter deadlock guard
  inherited from the P3a Stage 1 pattern.
- TestP3bWireDiscipline — AST grep for truthy bio-system checks.
"""

from __future__ import annotations

import inspect
import re
import threading
import time

import pytest

from maxim.memory.episode import (
    CaptureEvent,
    Episode,
    EpisodeBoundaryDetector,
    EpisodeStore,
    PendingEpisodeState,
    channel_gap_rule,
    channel_specific_rule,
    sender_change_rule,
)
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_episode(
    *,
    id: str,
    channel: str,
    activated_nodes: tuple[str, ...],
    sender_ids: tuple[str, ...] = (),
    thread_id: str | None = None,
    scn_tag: str | None = None,
    start_tick: int = 0,
    end_tick: int = 0,
) -> Episode:
    return Episode(
        id=id,
        start_tick=start_tick,
        end_tick=end_tick,
        channel=channel,
        sender_ids=sender_ids,
        thread_id=thread_id,
        activated_nodes=activated_nodes,
        reward_events=(),
        scn_tag=scn_tag,
    )


def _empty_pending(channel: str = "sms") -> PendingEpisodeState:
    return PendingEpisodeState(id="ep_test", start_tick=0, last_tick=0, channel=channel)


def _populated_pending(channel: str, senders: set[str]) -> PendingEpisodeState:
    p = _empty_pending(channel=channel)
    p.sender_ids = senders
    return p


# ─────────────────────────────────────────────────────────────────────────
# Rule factories
# ─────────────────────────────────────────────────────────────────────────


class TestP3bRuleFactories:
    """channel_gap_rule / sender_change_rule + channel_specific_rule wrapper.

    Round 2 Arch important #2 fold: the earlier ``sms_*`` factories
    were renamed to channel-agnostic equivalents to avoid the rule-zoo
    pattern (per-channel factories N times for N channels).
    """

    def test_channel_gap_rule_does_not_fire_within_gap(self):
        rule = channel_gap_rule("sms", max_gap_ticks=500)
        pending = _populated_pending("sms", {"alice"})
        pending.last_tick = 100
        event = CaptureEvent(tick=400, channel="sms", sender_id="alice")  # gap=300 < 500
        assert rule(pending, event) is False

    def test_channel_gap_rule_fires_when_gap_exceeded(self):
        rule = channel_gap_rule("sms", max_gap_ticks=500)
        pending = _populated_pending("sms", {"alice"})
        pending.last_tick = 100
        event = CaptureEvent(tick=700, channel="sms", sender_id="alice")  # gap=600 > 500
        assert rule(pending, event) is True

    def test_channel_gap_rule_does_not_fire_on_other_channel(self):
        """channel_gap_rule wraps tick_gap_rule with
        channel_specific_rule; an event on a different channel must
        NOT trigger the channel-tuned gap."""
        rule = channel_gap_rule("sms", max_gap_ticks=500)
        pending = _populated_pending("narrative", set())
        pending.last_tick = 100
        event = CaptureEvent(tick=700, channel="narrative", sender_id=None)
        assert rule(pending, event) is False

    def test_channel_gap_rule_works_for_arbitrary_channel(self):
        """Round 2 Arch important #2: the renamed factory takes the
        channel name as a parameter, not hard-coded. P5/P6 can use it
        for email, slack, voice without spawning per-channel
        factories."""
        rule = channel_gap_rule("email", max_gap_ticks=10000)
        pending = _populated_pending("email", set())
        pending.last_tick = 100
        # Event on a non-email channel must not trigger
        assert rule(pending, CaptureEvent(tick=15000, channel="slack")) is False
        # Event on email past the gap must trigger
        assert rule(pending, CaptureEvent(tick=15000, channel="email")) is True

    def test_sender_change_rule_fires_on_new_contact(self):
        rule = sender_change_rule()
        pending = _populated_pending("sms", {"alice"})
        event = CaptureEvent(tick=10, channel="sms", sender_id="bob")
        assert rule(pending, event) is True

    def test_sender_change_rule_no_op_on_known_contact(self):
        rule = sender_change_rule()
        pending = _populated_pending("sms", {"alice", "bob"})
        event = CaptureEvent(tick=10, channel="sms", sender_id="alice")
        assert rule(pending, event) is False

    def test_sender_change_rule_no_op_when_event_sender_is_none(self):
        rule = sender_change_rule()
        pending = _populated_pending("sms", {"alice"})
        event = CaptureEvent(tick=10, channel="sms", sender_id=None)
        assert rule(pending, event) is False

    def test_sender_change_rule_no_op_on_cold_start(self):
        """Round 1 Exec critical #1 regression: empty pending.sender_ids
        + non-None event sender must NOT trivially fire the rule.

        Without the cold-start guard, ``"bob" not in empty_set`` is True
        and the rule would fire on the first non-None sender after a
        sender_id=None initial event.
        """
        rule = sender_change_rule()
        empty_pending = _empty_pending("sms")  # sender_ids = set() default
        assert empty_pending.sender_ids == set()
        event = CaptureEvent(tick=10, channel="sms", sender_id="bob")
        assert rule(empty_pending, event) is False

    def test_sender_change_rule_is_channel_agnostic_when_unwrapped(self):
        """sender_change_rule is channel-agnostic by default. Callers
        wrap with channel_specific_rule if they want gating. This test
        verifies the rule fires on ANY channel when not wrapped.
        """
        rule = sender_change_rule()
        pending = _populated_pending("narrative", {"narrator_a"})
        event = CaptureEvent(tick=10, channel="narrative", sender_id="narrator_b")
        assert rule(pending, event) is True

    def test_sender_change_rule_composes_with_channel_specific_rule(self):
        """The intended composition pattern: wrap sender_change_rule
        with channel_specific_rule for channel-gated behavior."""
        wrapped = channel_specific_rule("sms", sender_change_rule())
        pending = _populated_pending("sms", {"alice"})
        # Non-SMS event must not fire even with new sender
        narr = CaptureEvent(tick=10, channel="narrative", sender_id="bob")
        assert wrapped(pending, narr) is False
        # SMS event with new sender must fire
        sms = CaptureEvent(tick=10, channel="sms", sender_id="bob")
        assert wrapped(pending, sms) is True


class TestChannelSpecificRule:
    """Round 1 Exec important #2 regression: the wrapper gates on
    ``event.channel``, NOT ``pending.channel``. Pending-channel gating
    only happens to work because the default ``channel_change_rule`` is
    always installed; removing it (a reasonable cross-channel-threading
    refactor) would silently disable every wrapped rule under the wrong
    gating choice.
    """

    def test_wrapper_gates_on_event_channel(self):
        def always_fire(_p, _e):
            return True

        wrapped = channel_specific_rule("sms", always_fire)
        pending = _empty_pending("sms")
        sms_event = CaptureEvent(tick=1, channel="sms", sender_id=None)
        narr_event = CaptureEvent(tick=1, channel="narrative", sender_id=None)
        assert wrapped(pending, sms_event) is True
        assert wrapped(pending, narr_event) is False

    def test_wrapper_does_not_gate_on_pending_channel(self):
        """Construct a pending episode whose channel does NOT match the
        wrapper's target, but where the incoming event DOES match. The
        wrapper should fire (gating on event.channel, not pending.channel).
        """

        def always_fire(_p, _e):
            return True

        wrapped = channel_specific_rule("sms", always_fire)
        # pending says "narrative" but the incoming event is sms
        pending = _empty_pending("narrative")
        sms_event = CaptureEvent(tick=1, channel="sms", sender_id="alice")
        # Gating on event.channel → fires
        assert wrapped(pending, sms_event) is True

    def test_wrapper_passes_through_inner_rule_result(self):
        def never_fire(_p, _e):
            return False

        wrapped = channel_specific_rule("sms", never_fire)
        pending = _empty_pending("sms")
        sms_event = CaptureEvent(tick=1, channel="sms", sender_id=None)
        # Channel matches but inner says no → no fire
        assert wrapped(pending, sms_event) is False

    def test_wrapper_correct_in_isolation_without_default_rules(self):
        """Round 2 Arch minor #3: the wrapper must behave correctly
        even when ``channel_change_rule`` is NOT installed alongside it.

        Build a detector with ONLY ``channel_specific_rule("sms",
        always_fire)`` — no defaults. Pending episode's channel is
        "narrative" but the incoming event is "sms". The wrapper must
        STILL fire (event-channel gating), not silently no-op (which
        would be the symptom if the wrapper were gating on
        pending.channel).
        """

        def always_fire(_p, _e):
            return True

        detector = EpisodeBoundaryDetector([channel_specific_rule("sms", always_fire)])
        pending = _empty_pending("narrative")
        sms_event = CaptureEvent(tick=1, channel="sms", sender_id="alice")
        # In isolation, with no other rules in the detector, the wrapper
        # must STILL fire on the cross-pending sms event.
        assert detector.should_close(pending, sms_event) is True


# ─────────────────────────────────────────────────────────────────────────
# Composition with P3a defaults
# ─────────────────────────────────────────────────────────────────────────


class TestRuleComposition:
    """Round 1 Arch minor #1 regression: P3b commits to ADDITIVE
    composition — P3b rules run alongside the P3a defaults
    (tick_gap_rule, channel_change_rule, scn_tag_change_rule), they
    do NOT replace them. All rules compose via
    ``EpisodeBoundaryDetector.should_close = any(rule(...))``.
    """

    def test_p3b_rules_compose_additively_with_defaults(self):
        """A Hippocampus that adds a channel-gated sender-change rule
        via add_boundary_rule must still honor the default
        channel-change rule on a non-matching event."""
        h = Hippocampus(HippocampusConfig())
        h.add_boundary_rule(channel_specific_rule("sms", sender_change_rule()))

        # Open a narrative episode
        h.observe_episode_event(CaptureEvent(tick=0, channel="narrative", sender_id=None, activated_nodes=("n1",)))
        # Switch to SMS — the default channel_change_rule must close
        # the narrative episode (the SMS rule wouldn't fire on its own
        # because the narrative pending has no senders).
        h.observe_episode_event(CaptureEvent(tick=10, channel="sms", sender_id="alice", activated_nodes=("s1",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        # Two episodes: one narrative, one sms
        assert len(episodes) == 2
        channels = {ep.channel for ep in episodes}
        assert channels == {"narrative", "sms"}


# ─────────────────────────────────────────────────────────────────────────
# EpisodeStore.episode_membership_filter
# ─────────────────────────────────────────────────────────────────────────


class TestMembershipFilter:
    """The general filter on EpisodeStore. Lives here (Round 1 Arch
    important #3) — ``Hippocampus.episode_filter`` is a thin
    convenience forwarder over this method.
    """

    def _store_with_mixed_episodes(self) -> EpisodeStore:
        store = EpisodeStore()
        store.add(
            _make_episode(
                id="sms_alice_1",
                channel="sms",
                sender_ids=("alice",),
                activated_nodes=("a", "b"),
            )
        )
        store.add(
            _make_episode(
                id="sms_bob_1",
                channel="sms",
                sender_ids=("bob",),
                activated_nodes=("a", "d"),
            )
        )
        store.add(
            _make_episode(
                id="narr_1",
                channel="narrative",
                sender_ids=(),
                activated_nodes=("a", "c"),
            )
        )
        return store

    def test_channel_filter_returns_only_channel_members_any_mode(self):
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms")
        assert f("a") is True  # in two sms episodes + one narrative
        assert f("b") is True  # sms only (alice)
        assert f("d") is True  # sms only (bob)
        assert f("c") is False  # narrative only

    def test_channel_filter_empty_episodes_returns_false(self):
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms")
        assert f("nonexistent_node") is False

    def test_sender_criterion_matches_via_collection_membership(self):
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms", sender_ids="alice")
        assert f("a") is True  # in sms_alice_1
        assert f("b") is True  # in sms_alice_1
        assert f("d") is False  # in sms_bob_1, not sms_alice_1

    def test_sender_criterion_no_match_returns_false(self):
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms", sender_ids="carol")
        assert f("a") is False
        assert f("b") is False

    def test_exclusive_mode_drops_cross_channel_nodes(self):
        """Round 1 Arch critical #1: exclusive mode is the parameter
        shape committed for P4. Stage 1 must implement it."""
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms", membership_mode="exclusive")
        # 'a' appears in both sms AND narrative → not exclusive to sms
        assert f("a") is False
        # 'b' and 'd' are sms-only
        assert f("b") is True
        assert f("d") is True

    def test_membership_mode_validation(self):
        store = self._store_with_mixed_episodes()
        with pytest.raises(ValueError, match="membership_mode must be"):
            store.episode_membership_filter(channel="sms", membership_mode="bogus")

    def test_filter_introspects_arbitrary_episode_fields(self):
        """Round 1 Arch important #4: filter axes are introspected from
        Episode dataclass fields, not hard-coded. Filtering on thread_id
        works without any new code.
        """
        store = EpisodeStore()
        store.add(_make_episode(id="t1", channel="sms", thread_id="thread_x", activated_nodes=("a",)))
        store.add(_make_episode(id="t2", channel="sms", thread_id="thread_y", activated_nodes=("b",)))

        f = store.episode_membership_filter(thread_id="thread_x")
        assert f("a") is True
        assert f("b") is False

    def test_unknown_episode_field_raises_at_build_time(self):
        """Round 2 cross-confirmed critical (Exec I1 + Arch C1):
        a typo in a criterion field name must raise at filter build
        time, not silently return zero matches.

        Pre-fold behavior: passing ``channnel="sms"`` (typo) silently
        returned an empty filter because ``getattr(episode, key, None)``
        + the None short-circuit conflated typos with missing fields.
        P4 cross-modal mug test would pass at 0% collapse with no
        diagnostic.
        """
        store = self._store_with_mixed_episodes()
        with pytest.raises(ValueError, match="unknown Episode field"):
            store.episode_membership_filter(channnel="sms")  # noqa
        with pytest.raises(ValueError, match="unknown Episode field"):
            store.episode_membership_filter(modality="vision")  # P4 will add this; today it's unknown
        with pytest.raises(ValueError, match="unknown Episode field"):
            store.episode_membership_filter(some_unknown_thing="x")

    def test_collection_criterion_value_raises_typeerror(self):
        """Round 2 Exec important #3: passing a tuple/list/set as a
        criterion VALUE silently returned empty results because
        ``tuple in tuple`` checks element membership, not subset. Raise
        explicitly. Subset / set-intersection semantics deferred.
        """
        store = self._store_with_mixed_episodes()
        with pytest.raises(TypeError, match="collection"):
            store.episode_membership_filter(sender_ids=("alice", "bob"))
        with pytest.raises(TypeError, match="collection"):
            store.episode_membership_filter(channel=["sms", "narrative"])

    def test_filter_matches_legitimate_none_scalar_field(self):
        """Round 2 Arch critical #2: a caller passing ``thread_id=None``
        (intent: episodes with no thread) must match episodes whose
        thread_id IS None. The pre-fold version had
        ``if field_value is None: return False`` which conflated
        "field is missing" with "field value is legitimately None"
        and broke the null filter case.
        """
        store = EpisodeStore()
        store.add(_make_episode(id="t1", channel="sms", thread_id=None, activated_nodes=("a",)))
        store.add(_make_episode(id="t2", channel="sms", thread_id="thread_x", activated_nodes=("b",)))

        # thread_id=None should match the episode where thread_id IS None
        f = store.episode_membership_filter(thread_id=None)
        assert f("a") is True
        assert f("b") is False

    def test_exclusive_mode_zero_episodes_returns_false(self):
        """Round 2 Exec important #2: pin the ``exclusive`` semantic for
        nodes with zero containing episodes.

        Under ``"exclusive"`` mode, a node with zero containing episodes
        could be argued either way — vacuous-truth says "every (zero)
        episode matches the criteria." But the practical semantic is
        "this node has no evidence either way, exclude it." Stage 1 picks
        the practical reading; this test pins it so a future refactor
        doesn't silently flip the answer.
        """
        store = self._store_with_mixed_episodes()
        f = store.episode_membership_filter(channel="sms", membership_mode="exclusive")
        # 'never_seen' is not in any episode at all
        assert f("never_seen") is False

    def test_exclusive_mode_with_two_criteria_cross_mug_shape(self):
        """Round 2 Arch important #4: exercise the ``exclusive`` mode
        with a 2-criteria filter that approximates the P4 cross-modal
        mug-test shape. Uses ``scn_tag`` as a stand-in for ``modality``
        (the P4 field that doesn't exist yet) to validate the
        cross-criteria pattern.
        """
        store = EpisodeStore()
        # Episode 1: sms + scene_a, contains x
        store.add(_make_episode(id="e1", channel="sms", scn_tag="scene_a", activated_nodes=("x",)))
        # Episode 2: sms + scene_a, contains x and y
        store.add(_make_episode(id="e2", channel="sms", scn_tag="scene_a", activated_nodes=("x", "y")))
        # Episode 3: sms + scene_b, contains z (not x)
        store.add(_make_episode(id="e3", channel="sms", scn_tag="scene_b", activated_nodes=("z",)))

        # Exclusive mode: nodes whose containing episodes are ALL in
        # {channel=sms, scn_tag=scene_a}
        f_excl = store.episode_membership_filter(channel="sms", scn_tag="scene_a", membership_mode="exclusive")
        assert f_excl("x") is True  # in e1 + e2, both sms+scene_a
        assert f_excl("y") is True  # in e2 only, sms+scene_a
        assert f_excl("z") is False  # in e3, scene_b — fails exclusive

        # Any mode: nodes that appear in at least one matching episode
        f_any = store.episode_membership_filter(channel="sms", scn_tag="scene_a", membership_mode="any")
        assert f_any("x") is True
        assert f_any("y") is True
        assert f_any("z") is False  # z is in scene_b only, no scene_a episode contains it


# ─────────────────────────────────────────────────────────────────────────
# Channel-filtered retrieval through Hippocampus.retrieve_on_cue
# ─────────────────────────────────────────────────────────────────────────


class TestChannelFilteredRetrieval:
    """End-to-end: filter built via the Hippocampus convenience alias
    + passed to retrieve_on_cue's node_filter kwarg."""

    def _hippocampus_with_mixed_episodes(self) -> Hippocampus:
        h = Hippocampus(HippocampusConfig())
        # SMS episode: alice texts about cooking (a, b)
        h.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a", "b")))
        h.finalize_pending_episode()
        # Narrative episode: a appears with c (different context)
        h.observe_episode_event(
            CaptureEvent(tick=1000, channel="narrative", sender_id=None, activated_nodes=("a", "c"))
        )
        h.finalize_pending_episode()
        return h

    def test_channel_filter_drops_cross_channel_neighbors(self):
        h = self._hippocampus_with_mixed_episodes()
        f = h.episode_filter(channel="sms")
        results = dict(h.retrieve_on_cue("a", limit=10, node_filter=f))
        # 'b' is sms-only → kept; 'c' is narrative-only → dropped
        assert "b" in results
        assert "c" not in results

    def test_sender_filter_drops_other_sender_neighbors(self):
        h = self._hippocampus_with_mixed_episodes()
        # Add another SMS episode with bob as sender
        h.observe_episode_event(CaptureEvent(tick=2000, channel="sms", sender_id="bob", activated_nodes=("a", "d")))
        h.finalize_pending_episode()

        f_alice = h.episode_filter(channel="sms", sender_ids="alice")
        results = dict(h.retrieve_on_cue("a", limit=10, node_filter=f_alice))
        # alice's episode with 'a' has 'b' → kept
        assert "b" in results
        # bob's episode with 'a' has 'd' → dropped (alice never co-occurred with d)
        assert "d" not in results

    def test_no_filter_returns_all_neighbors(self):
        """Sanity: without a filter, 'a' should reach both b (sms) and
        c (narrative) via direct co-occurrence."""
        h = self._hippocampus_with_mixed_episodes()
        results = dict(h.retrieve_on_cue("a", limit=10))
        assert "b" in results
        assert "c" in results

    def test_channel_filter_mixed_hop_breaks_transitive_path(self):
        """Round 1 Exec critical #3 regression: multi-hop traversal is
        path-strict, NOT bridge-transparent. If an intermediate node is
        filtered out, the BFS stops there even if downstream nodes
        would pass the filter.

        Scenario: cue (sms) → intermediate (narrative-only) → target
        (sms). Under channel=sms filter, the BFS rejects the
        intermediate at hop 1 → target unreachable.
        """
        h = Hippocampus(HippocampusConfig())
        # SMS episode: cue co-occurs with intermediate
        h.observe_episode_event(
            CaptureEvent(
                tick=0,
                channel="narrative",  # intermediate is narrative-only
                sender_id=None,
                activated_nodes=("cue", "intermediate"),
            )
        )
        h.finalize_pending_episode()
        # Narrative episode: intermediate co-occurs with target
        h.observe_episode_event(
            CaptureEvent(
                tick=1000,
                channel="narrative",
                sender_id=None,
                activated_nodes=("intermediate", "target"),
            )
        )
        h.finalize_pending_episode()
        # SMS episode: cue and target both appear (gives them direct
        # SMS membership so the filter retains them as endpoints)
        h.observe_episode_event(
            CaptureEvent(
                tick=2000,
                channel="sms",
                sender_id="alice",
                activated_nodes=("cue", "another_sms_node"),
            )
        )
        h.finalize_pending_episode()
        h.observe_episode_event(
            CaptureEvent(
                tick=3000,
                channel="sms",
                sender_id="alice",
                activated_nodes=("target", "another_sms_node"),
            )
        )
        h.finalize_pending_episode()

        # Without filter: cue → intermediate → target should reach target
        unfiltered = dict(h.retrieve_on_cue("cue", limit=20))
        assert "intermediate" in unfiltered, (
            f"Unfiltered baseline: cue should reach intermediate via narrative episode. Got: {unfiltered}"
        )

        # With SMS filter: intermediate is rejected (only in narrative
        # episodes), so the BFS stops at hop 1 from cue → target via the
        # narrative chain is unreachable.
        f_sms = h.episode_filter(channel="sms")
        filtered = dict(h.retrieve_on_cue("cue", limit=20, node_filter=f_sms))
        # intermediate must be dropped (narrative-only)
        assert "intermediate" not in filtered, f"intermediate should be filtered out (narrative-only). Got: {filtered}"
        # The narrative chain to target is broken; target is only
        # reachable via SMS-direct co-occurrence (another_sms_node bridge).
        # This is the load-bearing semantic: channel-filtered multi-hop
        # is path-strict, not bridge-transparent.


# ─────────────────────────────────────────────────────────────────────────
# Persistence round-trip + rule re-registration contract
# ─────────────────────────────────────────────────────────────────────────


class TestPersistenceContract:
    """Round 1 Exec important #3: boundary rules are NOT persisted.
    After load_state, the reloaded Hippocampus's _episode_detector
    contains only the defaults the new instance's __init__ registered.
    Callers that added P3b rules must re-add them post-load.
    """

    def test_episodes_round_trip_preserves_channel_metadata(self):
        h1 = Hippocampus(HippocampusConfig())
        h1.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a", "b")))
        h1.finalize_pending_episode()
        h1.observe_episode_event(
            CaptureEvent(tick=1000, channel="narrative", sender_id=None, activated_nodes=("a", "c"))
        )
        h1.finalize_pending_episode()

        dumped = h1.dump()
        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)

        # Channel filter on the reloaded instance must produce identical
        # results (filter closure re-resolves against fresh _episode_store).
        f1_sms = h1.episode_filter(channel="sms")
        f2_sms = h2.episode_filter(channel="sms")
        for node in ("a", "b", "c"):
            assert f1_sms(node) == f2_sms(node), f"channel filter drift on {node!r}"

    def test_boundary_rules_NOT_persisted_post_load(self):
        """Pre-load: add a custom always-fire rule. Dump. Load into a
        fresh Hippocampus (which has only the default rules). The
        reloaded instance must NOT have the custom rule installed.

        Round 2 Arch minor #1: snapshot the default rule count from a
        baseline Hippocampus rather than hard-coding a number. If
        P3.5 or a future wave changes the default rule count, this
        test stays correct without coupling to the specific value.
        """
        # Snapshot the default rule count from a fresh instance — this
        # is the count any Hippocampus has immediately after __init__.
        baseline = len(Hippocampus(HippocampusConfig())._episode_detector._rules)

        h1 = Hippocampus(HippocampusConfig())

        def always_close(_p, _e):
            return True

        h1.add_boundary_rule(always_close)

        # Verify pre-dump: the always_close rule is installed
        assert always_close in h1._episode_detector._rules
        assert len(h1._episode_detector._rules) == baseline + 1

        dumped = h1.dump()
        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)

        # Post-load: the custom rule is NOT in the reloaded detector
        assert always_close not in h2._episode_detector._rules
        # h2 has only the defaults — same count as a fresh instance
        assert len(h2._episode_detector._rules) == baseline

    def test_post_load_caller_can_re_register_rules(self):
        """Contract: callers re-add their P3b rules at construction time
        after every load_state. Verify the rule actually fires after
        re-registration."""
        h1 = Hippocampus(HippocampusConfig())
        dumped = h1.dump()

        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)
        h2.add_boundary_rule(channel_specific_rule("sms", sender_change_rule()))

        # Verify the re-registered rule fires
        h2.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a",)))
        h2.observe_episode_event(CaptureEvent(tick=1, channel="sms", sender_id="bob", activated_nodes=("b",)))
        h2.finalize_pending_episode()

        episodes = h2._episode_store.all_episodes()
        assert len(episodes) == 2  # alice's episode closed by the wrapped sender_change_rule


# ─────────────────────────────────────────────────────────────────────────
# Concurrency — capture-thread + filter deadlock guard
# ─────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    """The channel filter callback runs inside ``spreading_activation``'s
    binding-graph lock. ``Hippocampus._close_pending_episode_locked``
    acquires ``EpisodeStore._lock`` then ``binding_graph._lock`` (via
    ``apply_hebbian_on_close``). If the filter callback re-entered
    ``EpisodeStore._lock`` lazily, that's an AB-BA inversion against
    the worker.

    **Snapshot fix (Round 2 self-found critical):** the filter is
    built by snapshotting matching nodes once under
    ``EpisodeStore._lock`` and returning a lock-free closure. The
    tests below regression-guard the snapshot semantics so a future
    refactor cannot quietly re-introduce a lazy filter.
    """

    def test_filter_closure_holds_no_lock_after_construction(self):
        """The filter callable must be a frozen-set lookup — not a
        live query against EpisodeStore. Hold EpisodeStore._lock from
        a side thread and verify the filter returns immediately.

        Without the snapshot, the filter would block on store_lock
        and this test would time out at the join.
        """
        h = Hippocampus(HippocampusConfig())
        h.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a", "b")))
        h.finalize_pending_episode()

        # Build the filter NOW (this acquires + releases store lock once)
        f = h.episode_filter(channel="sms")

        # Spawn a side thread that grabs the store lock and holds it
        store_lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_store_lock():
            with h._episode_store._lock:
                store_lock_held.set()
                release_lock.wait(timeout=5.0)

        holder = threading.Thread(target=hold_store_lock, daemon=True)
        holder.start()
        assert store_lock_held.wait(timeout=2.0), "side thread failed to grab store lock"

        # Now call the filter on the main thread. With snapshot
        # semantics this is instantaneous (no lock acquisition). With a
        # lazy filter, this would block on store_lock until release.
        start = time.monotonic()
        result = f("a")
        elapsed = time.monotonic() - start

        # Release the side thread before any assertions can fail
        release_lock.set()
        holder.join(timeout=2.0)

        assert elapsed < 0.5, (
            f"filter call took {elapsed:.3f}s with side thread holding store lock — "
            f"filter is lazy, not snapshot-based. This is the load-bearing "
            f"deadlock-prevention invariant: the filter MUST be lock-free."
        )
        assert result is True

    def test_filter_snapshot_does_not_see_episodes_added_after_construction(self):
        """Snapshot semantics: episodes added AFTER filter construction
        are NOT reflected. This is the documented contract — and the
        thread-safety mechanism that prevents the lock inversion.
        """
        h = Hippocampus(HippocampusConfig())
        h.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a",)))
        h.finalize_pending_episode()

        f = h.episode_filter(channel="sms")
        assert f("a") is True
        assert f("b") is False  # not in any episode yet

        # Add a new episode AFTER filter construction
        h.observe_episode_event(CaptureEvent(tick=1000, channel="sms", sender_id="bob", activated_nodes=("b",)))
        h.finalize_pending_episode()

        # Filter still reflects the snapshot from before the addition
        assert f("b") is False, (
            "filter must reflect the snapshot at construction time, not the "
            "live store state — this is the documented snapshot contract"
        )

        # A fresh filter sees the new state
        f2 = h.episode_filter(channel="sms")
        assert f2("b") is True

    def test_channel_filter_no_deadlock_under_concurrent_capture(self):
        h = Hippocampus(HippocampusConfig())
        # Seed a few episodes so filters have something to query
        for i in range(5):
            h.observe_episode_event(
                CaptureEvent(
                    tick=i * 1000,
                    channel="sms",
                    sender_id="alice",
                    activated_nodes=(f"n{i}", "common"),
                )
            )
            h.finalize_pending_episode()

        stop = threading.Event()

        def worker():
            tick = 100_000
            while not stop.is_set():
                h.observe_episode_event(
                    CaptureEvent(
                        tick=tick,
                        channel="sms",
                        sender_id="bob",
                        activated_nodes=("common", f"x{tick}"),
                    )
                )
                h.finalize_pending_episode()
                tick += 1000

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        deadline = time.monotonic() + 2.0
        iterations = 0
        f = h.episode_filter(channel="sms")
        while time.monotonic() < deadline:
            h.retrieve_on_cue("common", limit=10, node_filter=f)
            iterations += 1

        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "worker thread deadlocked"
        assert iterations > 1, (
            f"channel-filtered retrieval did not run at least twice ({iterations} iters) — possible deadlock"
        )


# ─────────────────────────────────────────────────────────────────────────
# Wire discipline — `is not None` over truthy
# ─────────────────────────────────────────────────────────────────────────


class TestP3bWireDiscipline:
    """Inherits the P3a Stage 1 AST-grep regression pattern."""

    def test_p3b_source_has_no_truthy_biosystem_checks(self):
        import maxim.memory.episode as episode_module

        forbidden = re.compile(r"\bif self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)\b")

        episode_src = inspect.getsource(episode_module)
        matches = forbidden.findall(episode_src)
        assert not matches, f"Forbidden truthy bio-system check in memory/episode.py: {matches}"

        # The new Hippocampus method (episode_filter)
        from maxim.memory.hippocampus import Hippocampus

        method = getattr(Hippocampus, "episode_filter", None)
        assert method is not None, "episode_filter must exist on Hippocampus"
        src = inspect.getsource(method)
        assert not forbidden.findall(src), "Forbidden truthy bio-system check in Hippocampus.episode_filter"
