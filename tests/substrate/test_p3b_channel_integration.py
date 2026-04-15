"""P3b Stage 1 — channel-aware boundary rules + filtered retrieval.

Synthetic mechanism tests on hand-crafted episode geometry. No fixture
YAML, no metadata-grep baseline, no 10-seed sweep — those are Stage 2.

Test classes:
- TestP3bRuleFactories — sms_gap_rule, sms_sender_change_rule,
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
    EpisodeStore,
    PendingEpisodeState,
    channel_specific_rule,
    sms_gap_rule,
    sms_sender_change_rule,
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
    """SMS / channel rule factories + the channel_specific_rule wrapper."""

    def test_sms_gap_rule_does_not_fire_within_gap(self):
        rule = sms_gap_rule(max_gap_ticks=500)
        pending = _populated_pending("sms", {"alice"})
        pending.last_tick = 100
        event = CaptureEvent(tick=400, channel="sms", sender_id="alice")  # gap=300 < 500
        assert rule(pending, event) is False

    def test_sms_gap_rule_fires_when_gap_exceeded(self):
        rule = sms_gap_rule(max_gap_ticks=500)
        pending = _populated_pending("sms", {"alice"})
        pending.last_tick = 100
        event = CaptureEvent(tick=700, channel="sms", sender_id="alice")  # gap=600 > 500
        assert rule(pending, event) is True

    def test_sms_gap_rule_does_not_fire_on_non_sms_channel(self):
        """sms_gap_rule wraps tick_gap_rule with channel_specific_rule;
        a narrative event must NOT trigger the SMS-tuned gap."""
        rule = sms_gap_rule(max_gap_ticks=500)
        pending = _populated_pending("narrative", set())
        pending.last_tick = 100
        event = CaptureEvent(tick=700, channel="narrative", sender_id=None)
        assert rule(pending, event) is False

    def test_sms_sender_change_rule_fires_on_new_contact(self):
        rule = sms_sender_change_rule()
        pending = _populated_pending("sms", {"alice"})
        event = CaptureEvent(tick=10, channel="sms", sender_id="bob")
        assert rule(pending, event) is True

    def test_sms_sender_change_rule_no_op_on_known_contact(self):
        rule = sms_sender_change_rule()
        pending = _populated_pending("sms", {"alice", "bob"})
        event = CaptureEvent(tick=10, channel="sms", sender_id="alice")
        assert rule(pending, event) is False

    def test_sms_sender_change_rule_no_op_when_event_sender_is_none(self):
        rule = sms_sender_change_rule()
        pending = _populated_pending("sms", {"alice"})
        event = CaptureEvent(tick=10, channel="sms", sender_id=None)
        assert rule(pending, event) is False

    def test_sms_sender_change_rule_no_op_on_cold_start(self):
        """Round 1 Exec critical #1 regression: empty pending.sender_ids
        + non-None event sender must NOT trivially fire the rule.

        Without the cold-start guard, ``"bob" not in empty_set`` is True
        and the rule would fire on the first non-None sender after a
        sender_id=None initial event.
        """
        rule = sms_sender_change_rule()
        empty_pending = _empty_pending("sms")  # sender_ids = set() default
        assert empty_pending.sender_ids == set()
        event = CaptureEvent(tick=10, channel="sms", sender_id="bob")
        assert rule(empty_pending, event) is False

    def test_sms_sender_change_rule_no_op_on_non_sms_channel(self):
        """Wrapped channel gate: SMS rule must not fire on narrative."""
        rule = sms_sender_change_rule()
        pending = _populated_pending("sms", {"alice"})
        event = CaptureEvent(tick=10, channel="narrative", sender_id="bob")
        assert rule(pending, event) is False


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
        """A Hippocampus that adds an SMS-specific rule via
        add_boundary_rule must still honor the default channel-change
        rule on a non-SMS event."""
        h = Hippocampus(HippocampusConfig())
        h.add_boundary_rule(sms_sender_change_rule())

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
    important #3) — Hippocampus.channel_membership_filter is a thin
    convenience alias forwarding to it.
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
        f = h.channel_membership_filter("sms")
        results = dict(h.retrieve_on_cue("a", limit=10, node_filter=f))
        # 'b' is sms-only → kept; 'c' is narrative-only → dropped
        assert "b" in results
        assert "c" not in results

    def test_sender_filter_drops_other_sender_neighbors(self):
        h = self._hippocampus_with_mixed_episodes()
        # Add another SMS episode with bob as sender
        h.observe_episode_event(CaptureEvent(tick=2000, channel="sms", sender_id="bob", activated_nodes=("a", "d")))
        h.finalize_pending_episode()

        f_alice = h.channel_membership_filter("sms", sender="alice")
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
        f_sms = h.channel_membership_filter("sms")
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
        f1_sms = h1.channel_membership_filter("sms")
        f2_sms = h2.channel_membership_filter("sms")
        for node in ("a", "b", "c"):
            assert f1_sms(node) == f2_sms(node), f"channel filter drift on {node!r}"

    def test_boundary_rules_NOT_persisted_post_load(self):
        """Pre-load: add a custom always-fire rule. Dump. Load into a
        fresh Hippocampus (which has only the default rules). The
        reloaded instance must NOT have the custom rule installed.
        """
        h1 = Hippocampus(HippocampusConfig())

        def always_close(_p, _e):
            return True

        h1.add_boundary_rule(always_close)

        # Verify pre-dump: the always_close rule is installed
        assert always_close in h1._episode_detector._rules

        dumped = h1.dump()
        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)

        # Post-load: the custom rule is NOT in the reloaded detector
        assert always_close not in h2._episode_detector._rules
        # h2 has only the defaults (3 rules from EpisodeConfig)
        assert len(h2._episode_detector._rules) == 3

    def test_post_load_caller_can_re_register_rules(self):
        """Contract: callers re-add their P3b rules at construction time
        after every load_state. Verify the rule actually fires after
        re-registration."""
        h1 = Hippocampus(HippocampusConfig())
        dumped = h1.dump()

        h2 = Hippocampus(HippocampusConfig())
        h2.load_state(dumped)
        h2.add_boundary_rule(sms_sender_change_rule())

        # Verify the re-registered rule fires
        h2.observe_episode_event(CaptureEvent(tick=0, channel="sms", sender_id="alice", activated_nodes=("a",)))
        h2.observe_episode_event(CaptureEvent(tick=1, channel="sms", sender_id="bob", activated_nodes=("b",)))
        h2.finalize_pending_episode()

        episodes = h2._episode_store.all_episodes()
        assert len(episodes) == 2  # alice's episode closed by sms_sender_change_rule


# ─────────────────────────────────────────────────────────────────────────
# Concurrency — capture-thread + filter deadlock guard
# ─────────────────────────────────────────────────────────────────────────


class TestConcurrency:
    """Inherits the P3a Stage 1 deadlock-test pattern. The channel
    filter callback runs inside spreading_activation's graph lock and
    queries episodes_containing inside EpisodeStore._lock — verifies
    the acquire order doesn't reverse.
    """

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
        f = h.channel_membership_filter("sms")
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

        # The new Hippocampus method (channel_membership_filter)
        from maxim.memory.hippocampus import Hippocampus

        method = getattr(Hippocampus, "channel_membership_filter", None)
        assert method is not None, "channel_membership_filter must exist on Hippocampus"
        src = inspect.getsource(method)
        assert not forbidden.findall(src), "Forbidden truthy bio-system check in Hippocampus.channel_membership_filter"
