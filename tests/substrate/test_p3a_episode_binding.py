"""P3a Stage 1 — synthetic mechanism tests on hand-crafted node geometry.

Covers:
- Episode close creates Hebbian edges at hebbian_init.
- Episode close on an existing pair strengthens by exactly hebbian_delta
  (not 2× — pair enumeration is unordered). Regression guard for Round 1
  Exec critical #1.
- Repeated closes of the same pair set do NOT duplicate edges in
  binding_graph._outgoing. Regression guard for Round 1 Exec critical
  #2 (add_edge has no dedupe).
- Hebbian weight saturates at hebbian_max.
- Partial-cue retrieval returns co-activated nodes ranked by weight.
- Partial cue on a non-member returns nothing.
- Multiple episodes sharing a cue merge into ranked retrieval.
- Boundary detector: tick gap, channel change, scn_tag change each close
  a pending episode. Custom rule appends cleanly.
- Hebbian updates fire whether ATL is wired, None, or an empty ATL
  (regression guard for the is-not-none vs truthy bug class).
- Zero `if self._(nac|atl|...)` truthy checks in the P3a diff.
- Lock ordering under an adversarial capture thread: no deadlock within
  a 2-second budget.
- Persistence round-trip via Hippocampus._to_dict / load_state.
"""

from __future__ import annotations

import inspect
import threading
import time

import numpy as np
import pytest

from maxim.agents.bus import EdgeType
from maxim.memory.episode import (
    CaptureEvent,
    Episode,
    EpisodeBoundaryDetector,
    PendingEpisodeState,
    tick_gap_rule,
)
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_hippocampus(
    *,
    hebbian_init: float = 0.3,
    hebbian_delta: float = 0.1,
    hebbian_max: float = 1.0,
    boundary_tick_gap: int = 50,
) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            episode=EpisodeConfig(
                boundary_tick_gap=boundary_tick_gap,
                hebbian=HebbianConfig(
                    init=hebbian_init,
                    delta=hebbian_delta,
                    max_weight=hebbian_max,
                ),
            )
        )
    )


def _close_episode_with_nodes(h: Hippocampus, *nodes: str, tick: int = 0) -> Episode:
    """Open a fresh episode seeded with the given nodes, then force-close it.

    Each test scenario is one episode; we use finalize_pending_episode()
    so we don't depend on the boundary detector to fire.
    """
    h.observe_episode_event(CaptureEvent(tick=tick, channel="text", activated_nodes=tuple(nodes)))
    episode = h.finalize_pending_episode()
    assert episode is not None
    return episode


# ─────────────────────────────────────────────────────────────────────────
# Mechanism tests
# ─────────────────────────────────────────────────────────────────────────


class TestP3aMechanism:
    def test_episode_close_creates_hebbian_edges(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c")

        for src, tgt in [("a", "b"), ("a", "c"), ("b", "c")]:
            edge = h._binding_graph.find_edge(src, tgt, EdgeType.ASSOCIATES)
            assert edge is not None, f"edge {src}→{tgt} missing"
            assert edge.weight == pytest.approx(0.3)
            # Reverse direction must also exist (add_bidirectional)
            rev = h._binding_graph.find_edge(tgt, src, EdgeType.ASSOCIATES)
            assert rev is not None and rev.weight == pytest.approx(0.3)

    def test_episode_close_strengthens_existing_edges_by_exactly_delta(self):
        """Round 1 Exec critical #1 regression guard.

        Ordered-pair enumeration would visit each pair twice under
        add_bidirectional, double-applying HEBBIAN_DELTA. itertools
        .combinations (unordered) visits each pair exactly once.
        """
        h = _fresh_hippocampus(hebbian_init=0.3, hebbian_delta=0.1)
        _close_episode_with_nodes(h, "a", "b")  # creates @ 0.3
        _close_episode_with_nodes(h, "a", "b")  # strengthens to 0.4, not 0.5

        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.weight == pytest.approx(0.4), (
            f"second episode must apply exactly one delta; got weight={edge.weight}"
        )
        rev = h._binding_graph.find_edge("b", "a", EdgeType.ASSOCIATES)
        assert rev is not None
        assert rev.weight == pytest.approx(0.4)

    def test_repeated_closes_no_edge_duplication(self):
        """Round 1 Exec critical #2 regression guard.

        DependencyGraph.add_edge has no dedupe; a second add_edge call
        with the same (src, tgt, type) would append a second parallel
        edge. apply_hebbian_on_close guards against this by calling
        find_edge first; N closes of the same pair must produce exactly
        ONE edge per direction, not N edges.
        """
        h = _fresh_hippocampus()
        for _ in range(5):
            _close_episode_with_nodes(h, "a", "b")

        outgoing = h._binding_graph._outgoing.get("a", [])
        a_to_b = [e for e in outgoing if e.target == "b" and e.edge_type == EdgeType.ASSOCIATES]
        assert len(a_to_b) == 1, f"expected exactly one a→b edge, got {len(a_to_b)}"

        outgoing_b = h._binding_graph._outgoing.get("b", [])
        b_to_a = [e for e in outgoing_b if e.target == "a" and e.edge_type == EdgeType.ASSOCIATES]
        assert len(b_to_a) == 1, f"expected exactly one b→a edge, got {len(b_to_a)}"

    def test_strengthen_saturates_at_max(self):
        h = _fresh_hippocampus(hebbian_init=0.3, hebbian_delta=0.5, hebbian_max=0.9)
        # 0.3 → 0.8 → 0.9 (clamped) → 0.9 (clamped)
        _close_episode_with_nodes(h, "a", "b")
        _close_episode_with_nodes(h, "a", "b")
        _close_episode_with_nodes(h, "a", "b")
        _close_episode_with_nodes(h, "a", "b")

        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.weight == pytest.approx(0.9)


class TestP3aRetrieval:
    """Stage 1 retrieval tests exercise ONE-HOP semantics explicitly.

    Stage 2 flipped the ``retrieve_on_cue`` default to ``multi_hop=True``
    per the pre-merge Arch-lens review (the "backward compat" hedge
    was protecting exactly these tests while silently degrading every
    future production caller). These Stage 1 tests now opt into
    ``multi_hop=False`` explicitly to preserve their one-hop weight
    assertions.
    """

    def test_partial_cue_retrieves_co_activated_nodes(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c", "d")

        results = h.retrieve_on_cue("a", multi_hop=False)
        returned_ids = {node for node, _ in results}
        assert returned_ids == {"b", "c", "d"}
        for _, weight in results:
            assert weight == pytest.approx(0.3)

    def test_partial_cue_non_member_returns_nothing(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c")
        assert h.retrieve_on_cue("z", multi_hop=False) == []

    def test_multiple_episodes_shared_cue_merge(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b")
        _close_episode_with_nodes(h, "a", "c")

        results = dict(h.retrieve_on_cue("a", multi_hop=False))
        assert set(results.keys()) == {"b", "c"}
        # a↔b reinforced once (one episode), so stays at hebbian_init
        assert results["b"] == pytest.approx(0.3)
        assert results["c"] == pytest.approx(0.3)

    def test_retrieve_respects_limit(self):
        h = _fresh_hippocampus()
        nodes = [f"n{i}" for i in range(10)]
        _close_episode_with_nodes(h, *nodes)

        results = h.retrieve_on_cue("n0", limit=3, multi_hop=False)
        assert len(results) == 3


class TestP3aBoundaryDetector:
    def test_tick_gap_closes_pending(self):
        h = _fresh_hippocampus(boundary_tick_gap=50)
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=10, channel="text", activated_nodes=("b",)))
        h.observe_episode_event(CaptureEvent(tick=20, channel="text", activated_nodes=("c",)))
        # Gap of 80 > 50 → closes episode 1, opens episode 2
        h.observe_episode_event(CaptureEvent(tick=100, channel="text", activated_nodes=("d",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        assert set(episodes[0].activated_nodes) == {"a", "b", "c"}
        assert set(episodes[1].activated_nodes) == {"d"}

    def test_channel_change_closes_pending(self):
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="vision", activated_nodes=("b",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        channels = {ep.channel for ep in episodes}
        assert channels == {"text", "vision"}

    def test_scn_tag_change_closes_pending(self):
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", scn_tag="scene1", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", scn_tag="scene2", activated_nodes=("b",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2

    def test_scn_tag_none_does_not_close(self):
        """None tag on either side must not spuriously close the pending episode."""
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", scn_tag=None, activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", scn_tag=None, activated_nodes=("b",)))
        h.finalize_pending_episode()
        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 1
        assert set(episodes[0].activated_nodes) == {"a", "b"}

    def test_custom_rule_extension(self):
        """P3b extension seam: appending a rule works without touching Stage 1 code."""

        def even_tick_rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
            return event.tick % 2 == 0 and pending.last_tick != event.tick

        detector = EpisodeBoundaryDetector(rules=[tick_gap_rule(100)])
        detector.add_rule(even_tick_rule)

        pending = PendingEpisodeState(id="ep_1", start_tick=1, last_tick=1, channel="text")
        # Even tick → fires
        assert detector.should_close(pending, CaptureEvent(tick=2, channel="text")) is True
        # Odd tick within gap → does not fire
        assert detector.should_close(pending, CaptureEvent(tick=3, channel="text")) is False


# ─────────────────────────────────────────────────────────────────────────
# Wire discipline — `is not None` over truthy
# ─────────────────────────────────────────────────────────────────────────


class TestP3aWireDiscipline:
    """Round 2 fold: `test_hebbian_fires_when_atl_wired_and_empty` was
    deleted as a no-op — Hippocampus has no ``_atl`` field in Stage 1,
    so wiring an empty one was exercising no real code path. The
    general wire-discipline regression is covered by the AST-based
    source grep below."""

    def test_hebbian_fires_without_atl_dependency(self):
        """Binding graph is fully Hippocampus-owned; Hebbian updates
        fire regardless of whether an ATL reference exists on
        Hippocampus. This is the architectural invariant the Round 1
        pivot established."""
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b")
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None

    def test_p3a_source_has_no_truthy_biosystem_checks(self):
        """Round 2 Exec important #2 regression — replace the git-diff
        sentinel (which silently skipped on shallow-clone CI) with an
        in-tree source grep over the P3a-owned modules.

        Reads the source of ``memory/episode.py`` and the P3a-added
        block of ``memory/hippocampus.py`` (matched by the
        ``observe_episode_event`` / ``_apply_hebbian_on_close`` names)
        and asserts no truthy bio-system check regex hits.
        """
        import re

        import maxim.memory.episode as episode_module
        from maxim.memory.hippocampus import Hippocampus

        forbidden = re.compile(r"\bif self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)\b")

        # Full episode.py module source
        episode_src = inspect.getsource(episode_module)
        assert not forbidden.findall(episode_src), "Forbidden truthy bio-system check in memory/episode.py"

        # The P3a-added methods on Hippocampus
        for method_name in (
            "observe_episode_event",
            "finalize_pending_episode",
            "retrieve_on_cue",
            "add_boundary_rule",
            "_start_episode",
            "_close_pending_episode_locked",
        ):
            method = getattr(Hippocampus, method_name, None)
            if method is None:
                continue
            src = inspect.getsource(method)
            matches = forbidden.findall(src)
            assert not matches, f"Forbidden truthy bio-system check in Hippocampus.{method_name}: {matches}"


# ─────────────────────────────────────────────────────────────────────────
# Lock ordering — capture thread + finalize does not deadlock
# ─────────────────────────────────────────────────────────────────────────


class TestP3aLockOrdering:
    def test_finalize_pending_episode_no_deadlock_under_capture_thread(self):
        """Acquire order: episode_lock → episode_store._lock → binding_graph._lock.

        Spawn a background thread that repeatedly opens + closes episodes
        via observe_episode_event, while the main thread repeatedly calls
        retrieve_on_cue (which acquires binding_graph lock). Under the
        correct acquire order this terminates within a 2-second budget.
        """
        h = _fresh_hippocampus(boundary_tick_gap=1)
        stop = threading.Event()

        def worker():
            tick = 0
            while not stop.is_set():
                h.observe_episode_event(CaptureEvent(tick=tick, channel="text", activated_nodes=("a", "b")))
                h.observe_episode_event(CaptureEvent(tick=tick + 10, channel="text", activated_nodes=("c",)))
                h.finalize_pending_episode()
                tick += 100

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        deadline = time.monotonic() + 2.0
        iterations = 0
        while time.monotonic() < deadline:
            h.retrieve_on_cue("a", multi_hop=False)
            iterations += 1

        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "worker thread deadlocked"
        # Round 2 Exec minor #1: loosened from `> 10` to `> 1` to avoid
        # flaking on slow CI runners under lock contention. The
        # "didn't deadlock" assertion above is the load-bearing one;
        # the iter count is a soft sanity check.
        assert iterations > 1, f"retrieve_on_cue did not run at least twice ({iterations} iters) — possible deadlock"


# ─────────────────────────────────────────────────────────────────────────
# Persistence round-trip — depends on P3.5 Stage 1
# ─────────────────────────────────────────────────────────────────────────


class TestP3aPersistence:
    def test_episode_round_trips_via_hippocampus_dump(self):
        """Hippocampus._to_dict() reserved 'episodes' key must carry
        EpisodeStore contents through a dump → load_state cycle.

        Depends on P3.5 Stage 1 Hippocampus.dump / load_state.
        """
        h1 = _fresh_hippocampus()
        _close_episode_with_nodes(h1, "a", "b", "c")
        _close_episode_with_nodes(h1, "x", "y")

        dumped = h1.dump()
        assert len(dumped["episodes"]) == 2

        h2 = _fresh_hippocampus()
        h2.load_state(dumped)

        assert len(h2._episode_store) == 2

        episodes = h2._episode_store.all_episodes()
        all_nodes = set()
        for ep in episodes:
            all_nodes.update(ep.activated_nodes)
        assert all_nodes == {"a", "b", "c", "x", "y"}

    def test_binding_graph_rebuilt_on_load_state(self):
        """Round 2 Arch important #4 regression — the binding graph is
        rebuilt from loaded episodes so retrieve_on_cue is consistent
        post-load. Without the rebuild, retrieve_on_cue returns [] until
        new episodes land, and then rebuilds inconsistent with pre-load.
        """
        h1 = _fresh_hippocampus()
        _close_episode_with_nodes(h1, "a", "b", "c")

        dumped = h1.dump()
        h2 = _fresh_hippocampus()
        h2.load_state(dumped)

        # retrieve_on_cue on the fresh instance must return the same
        # co-activated nodes as the original. Use multi_hop=False to
        # assert exact direct-edge weights; multi-hop scores would
        # include decay and wouldn't equal the raw 0.3 hebbian_init.
        hits = dict(h2.retrieve_on_cue("a", multi_hop=False))
        assert set(hits.keys()) == {"b", "c"}
        assert hits["b"] == pytest.approx(0.3)
        assert hits["c"] == pytest.approx(0.3)

    def test_next_episode_ordinal_restored_on_load_state(self):
        """Round 2 Exec critical #1 regression — dump+reload+observe
        must not re-issue a duplicate episode id. Pre-fix this crashed
        with 'duplicate episode id: ep_1' on the next close."""
        h1 = _fresh_hippocampus()
        _close_episode_with_nodes(h1, "a", "b")
        _close_episode_with_nodes(h1, "c", "d")
        assert h1._next_episode_ordinal == 2

        dumped = h1.dump()
        assert dumped["next_episode_ordinal"] == 2

        h2 = _fresh_hippocampus()
        h2.load_state(dumped)
        assert h2._next_episode_ordinal == 2

        # Creating a new episode post-load must issue id=ep_3, not ep_1
        new_ep = _close_episode_with_nodes(h2, "e", "f")
        assert new_ep.id == "ep_3"

    def test_ordinal_derived_from_max_id_when_dump_missing_ordinal(self):
        """Recovery path: if the dumped state is missing
        next_episode_ordinal (e.g., old dump format or corrupt file),
        derive it from max(episode.id). Keeps load resilient to
        partial states."""
        h = _fresh_hippocampus()
        # Synthetic legacy-shaped state: no ordinal, two episodes
        state = {
            "memories": [],
            "context_index": {},
            "stats": {},
            "compressed_count": 0,
            "associative_graph": {"nodes": [], "edges": []},
            "episodes": [
                {
                    "id": "ep_7",
                    "start_tick": 0,
                    "end_tick": 0,
                    "channel": "text",
                    "sender_ids": [],
                    "thread_id": None,
                    "activated_nodes": ["a", "b"],
                    "reward_events": [],
                    "scn_tag": None,
                },
                {
                    "id": "ep_12",
                    "start_tick": 10,
                    "end_tick": 10,
                    "channel": "text",
                    "sender_ids": [],
                    "thread_id": None,
                    "activated_nodes": ["c", "d"],
                    "reward_events": [],
                    "scn_tag": None,
                },
            ],
        }
        h.load_state(state)
        assert h._next_episode_ordinal == 12

        # Next close must issue ep_13, not a collision
        new_ep = _close_episode_with_nodes(h, "x", "y")
        assert new_ep.id == "ep_13"


class TestP3aBindingGraphInvariants:
    """Round 2 Exec important #3 regression — add_node is called
    before edge creation so the binding graph has a real node list
    (required for Stage 2 persistence)."""

    def test_binding_graph_nodes_populated_after_hebbian(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c")
        for node_id in ("a", "b", "c"):
            assert h._binding_graph.get_node(node_id) is not None, (
                f"binding graph must have node {node_id} registered, not just edges"
            )

    def test_binding_graph_to_dict_includes_nodes(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c")
        graph_dict = h._binding_graph.to_dict()
        node_ids = set(graph_dict.get("nodes", []))
        assert {"a", "b", "c"}.issubset(node_ids), f"binding graph to_dict nodes = {node_ids}, expected a, b, c"


class TestP3aEpisodeStoreLoadDuplicates:
    """Round 2 Exec important #4 regression — load_from_dict raises
    on duplicate episode ids rather than silently overwriting."""

    def test_duplicate_episode_id_in_loaded_state_raises(self):
        from maxim.memory.episode import EpisodeStore

        store = EpisodeStore()
        state = {
            "episodes": [
                {
                    "id": "ep_1",
                    "start_tick": 0,
                    "end_tick": 0,
                    "channel": "text",
                    "sender_ids": [],
                    "thread_id": None,
                    "activated_nodes": ["a"],
                    "reward_events": [],
                    "scn_tag": None,
                },
                {
                    "id": "ep_1",  # duplicate
                    "start_tick": 10,
                    "end_tick": 10,
                    "channel": "text",
                    "sender_ids": [],
                    "thread_id": None,
                    "activated_nodes": ["b"],
                    "reward_events": [],
                    "scn_tag": None,
                },
            ]
        }
        with pytest.raises(ValueError, match="duplicate episode id"):
            store.load_from_dict(state)


class TestP3aBoundaryRuleSeam:
    """Round 2 Arch important #1 — Hippocampus.add_boundary_rule is the
    public seam P3b uses to register per-channel rules."""

    def test_add_boundary_rule_is_public_and_consulted(self):
        h = _fresh_hippocampus(boundary_tick_gap=1000)

        closed_ids = []

        def force_close_rule(pending: PendingEpisodeState, event: CaptureEvent) -> bool:
            # Close on any event after the first (hard forcing)
            return pending.last_tick != event.tick

        h.add_boundary_rule(force_close_rule)

        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        closed_ids = [ep.id for ep in episodes]
        # Two episodes: the first was closed by the custom rule firing
        # when event2 arrived; the second was closed by finalize.
        assert len(episodes) == 2, f"got {closed_ids}"


# ─────────────────────────────────────────────────────────────────────────
# Episode boundary enrichment Stage 1 — tool execution boundary
# ─────────────────────────────────────────────────────────────────────────


class TestToolExecutionBoundary:
    """Episode boundary enrichment Stage 1: tool execution creates a clean
    episodic break between pre-tool and post-tool contexts."""

    def test_tool_execution_closes_pending_episode(self):
        """[text, text, TOOL, text, text] → 2 episodes."""
        h = _fresh_hippocampus(boundary_tick_gap=1000)
        # Pre-tool events
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        # Post-tool event (after_tool_execution=True closes the pending episode)
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",), after_tool_execution=True))
        h.observe_episode_event(CaptureEvent(tick=3, channel="text", activated_nodes=("d",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        assert set(episodes[0].activated_nodes) == {"a", "b"}
        assert set(episodes[1].activated_nodes) == {"c", "d"}

    def test_no_tool_execution_stays_single_episode(self):
        """Without after_tool_execution, events stay in one episode."""
        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 1
        assert set(episodes[0].activated_nodes) == {"a", "b", "c"}

    def test_tool_execution_on_first_event_no_empty_episode(self):
        """Tool execution on the very first event should not create an
        empty pre-tool episode — there's nothing to close."""
        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",), after_tool_execution=True))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 1
        assert set(episodes[0].activated_nodes) == {"a", "b"}

    def test_consecutive_tool_executions(self):
        """Two consecutive tool executions create separate episodes."""
        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("pre",)))
        h.observe_episode_event(
            CaptureEvent(tick=1, channel="text", activated_nodes=("tool1",), after_tool_execution=True)
        )
        h.observe_episode_event(
            CaptureEvent(tick=2, channel="text", activated_nodes=("tool2",), after_tool_execution=True)
        )
        h.observe_episode_event(CaptureEvent(tick=3, channel="text", activated_nodes=("post",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 3
        assert set(episodes[0].activated_nodes) == {"pre"}
        assert set(episodes[1].activated_nodes) == {"tool1"}
        assert set(episodes[2].activated_nodes) == {"tool2", "post"}

    def test_tool_execution_with_hebbian(self):
        """Hebbian edges should form within each episode, not across
        the tool execution boundary."""
        h = _fresh_hippocampus(boundary_tick_gap=1000)
        # Episode 1: a + b co-activate
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a", "b")))
        # Episode 2: c + d co-activate (tool boundary separates them)
        h.observe_episode_event(
            CaptureEvent(tick=1, channel="text", activated_nodes=("c", "d"), after_tool_execution=True)
        )
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2

        # Retrieve: "a" should find "b" but NOT "c" or "d"
        results = h.retrieve_on_cue("a")
        result_ids = {nid for nid, _ in results}
        assert "b" in result_ids
        assert "c" not in result_ids
        assert "d" not in result_ids


# ─────────────────────────────────────────────────────────────────────────
# Concept decomposition Stage 2 — role-tagged Hebbian edges
# ─────────────────────────────────────────────────────────────────────────


class TestRoleTaggedEdges:
    """Concept decomposition Stage 2: relation metadata from the
    dependency parse annotates Hebbian edges between co-activated
    concept nodes."""

    def test_relation_metadata_on_edge(self):
        """When node_relations is provided, edges get a 'relation' metadata field."""
        from maxim.agents.bus import EdgeType

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(
            CaptureEvent(
                tick=0,
                channel="text",
                activated_nodes=("mug", "table"),
                node_relations={frozenset({"mug", "table"}): "spatial"},
            )
        )
        h.finalize_pending_episode()

        edge = h._binding_graph.find_edge("mug", "table", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("relation") == "spatial"

    def test_no_relation_leaves_edge_clean(self):
        """Without node_relations, edges have no 'relation' metadata."""
        from maxim.agents.bus import EdgeType

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a", "b")))
        h.finalize_pending_episode()

        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert "relation" not in edge.metadata

    def test_relation_first_write_wins(self):
        """If two episodes provide different relations for the same pair,
        the first one is kept."""
        from maxim.agents.bus import EdgeType

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        # Episode 1: spatial
        h.observe_episode_event(
            CaptureEvent(
                tick=0,
                channel="text",
                activated_nodes=("mug", "table"),
                node_relations={frozenset({"mug", "table"}): "spatial"},
            )
        )
        h.finalize_pending_episode()
        # Episode 2: possessive (should not overwrite)
        h.observe_episode_event(
            CaptureEvent(
                tick=100,
                channel="text",
                activated_nodes=("mug", "table"),
                node_relations={frozenset({"mug", "table"}): "possessive"},
            )
        )
        h.finalize_pending_episode()

        edge = h._binding_graph.find_edge("mug", "table", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("relation") == "spatial"

    def test_mixed_relations_in_one_episode(self):
        """Multiple node pairs in one episode can each have their own relation."""
        from maxim.agents.bus import EdgeType

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h.observe_episode_event(
            CaptureEvent(
                tick=0,
                channel="text",
                activated_nodes=("cat", "mat", "box"),
                node_relations={
                    frozenset({"cat", "mat"}): "spatial",
                    frozenset({"cat", "box"}): "possessive",
                },
            )
        )
        h.finalize_pending_episode()

        edge_cm = h._binding_graph.find_edge("cat", "mat", EdgeType.ASSOCIATES)
        edge_cb = h._binding_graph.find_edge("cat", "box", EdgeType.ASSOCIATES)
        edge_mb = h._binding_graph.find_edge("mat", "box", EdgeType.ASSOCIATES)

        assert edge_cm.metadata.get("relation") == "spatial"
        assert edge_cb.metadata.get("relation") == "possessive"
        # mat-box pair has no explicit relation
        assert "relation" not in edge_mb.metadata


# ─────────────────────────────────────────────────────────────────────────
# v1_refinement P2 — semantic shift episode boundary
# ─────────────────────────────────────────────────────────────────────────


class TestSemanticShiftRule:
    """v1_refinement P2 — embedding-based episode boundary detection.

    These tests use synthetic orthogonal numpy vectors instead of a real
    sentence-transformer model so they run in the fast suite without a
    heavy dependency. The realistic threshold-tuning evidence lives in
    ``tests/calibration/p2_episode_boundary_calibration.py`` (slow).
    """

    def _vec(self, *vals: float) -> np.ndarray:
        return np.asarray(vals, dtype=np.float32)

    def test_no_centroid_no_close(self):
        """Cold start: pending state with no centroid never fires the rule."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.5)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        ev = CaptureEvent(tick=1, channel="text", embedding=self._vec(1.0, 0.0))
        assert rule(pending, ev) is False

    def test_no_event_embedding_no_close(self):
        """Backwards-compat: legacy callers that don't pass an embedding
        never trigger the rule even when the pending centroid exists."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.5)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(self._vec(1.0, 0.0))
        ev = CaptureEvent(tick=1, channel="text")  # no embedding
        assert rule(pending, ev) is False

    def test_aligned_embedding_no_close(self):
        """Identical-direction event embedding has cos=1, 1-cos=0 < threshold."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.4)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(self._vec(1.0, 0.0))
        ev = CaptureEvent(tick=1, channel="text", embedding=self._vec(0.5, 0.0))
        assert rule(pending, ev) is False

    def test_orthogonal_embedding_closes(self):
        """Orthogonal vectors have cos=0, 1-cos=1.0 > threshold → close."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.4)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(self._vec(1.0, 0.0))
        ev = CaptureEvent(tick=1, channel="text", embedding=self._vec(0.0, 1.0))
        assert rule(pending, ev) is True

    def test_opposite_embedding_closes(self):
        """Antipodal vectors have cos=-1, 1-cos=2.0 > threshold → close."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.5)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(self._vec(1.0, 0.0))
        ev = CaptureEvent(tick=1, channel="text", embedding=self._vec(-1.0, 0.0))
        assert rule(pending, ev) is True

    def test_zero_norm_centroid_closes(self):
        """Degenerate zero-vector centroid: rule returns True (treated as
        a maximal shift) rather than NaN-comparing."""
        from maxim.memory.episode import semantic_shift_rule

        rule = semantic_shift_rule(threshold=0.5)
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.centroid_embedding = self._vec(0.0, 0.0)
        pending.centroid_count = 1
        ev = CaptureEvent(tick=1, channel="text", embedding=self._vec(1.0, 0.0))
        assert rule(pending, ev) is True

    def test_threshold_validation(self):
        """Out-of-range thresholds raise at construction, not at first call."""
        from maxim.memory.episode import semantic_shift_rule

        with pytest.raises(ValueError, match=r"threshold must be in"):
            semantic_shift_rule(threshold=-0.1)
        with pytest.raises(ValueError, match=r"threshold must be in"):
            semantic_shift_rule(threshold=1.1)
        # Boundary values OK
        semantic_shift_rule(threshold=0.0)
        semantic_shift_rule(threshold=1.0)

    def test_published_default_in_range(self):
        """Smoke test: the published default lives inside the valid range
        and constructs without error."""
        from maxim.memory.episode import semantic_shift_rule

        default = inspect.signature(semantic_shift_rule).parameters["threshold"].default
        assert isinstance(default, float)
        assert 0.0 <= default <= 1.0
        # No raise:
        semantic_shift_rule()


class TestPendingCentroidUpdate:
    """v1_refinement P2 — running-mean centroid arithmetic."""

    def test_first_update_initialises_centroid(self):
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        emb = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        pending.update_centroid(emb)
        assert pending.centroid_count == 1
        np.testing.assert_array_equal(pending.centroid_embedding, emb)

    def test_running_mean_two_events(self):
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(np.asarray([1.0, 0.0], dtype=np.float32))
        pending.update_centroid(np.asarray([0.0, 1.0], dtype=np.float32))
        assert pending.centroid_count == 2
        # Mean of [(1, 0), (0, 1)] is (0.5, 0.5).
        np.testing.assert_allclose(pending.centroid_embedding, [0.5, 0.5], rtol=1e-6)

    def test_running_mean_three_events(self):
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        for vec in [(3.0, 0.0), (0.0, 3.0), (3.0, 3.0)]:
            pending.update_centroid(np.asarray(vec, dtype=np.float32))
        assert pending.centroid_count == 3
        # Mean = (2.0, 2.0).
        np.testing.assert_allclose(pending.centroid_embedding, [2.0, 2.0], rtol=1e-6)

    def test_none_embedding_no_op(self):
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(np.asarray([1.0, 0.0], dtype=np.float32))
        pending.update_centroid(None)
        # Count and centroid unchanged.
        assert pending.centroid_count == 1
        np.testing.assert_array_equal(pending.centroid_embedding, [1.0, 0.0])

    def test_zero_norm_embedding_skipped(self):
        """Pre-merge review fold (Executor E1): a zero-norm embedding
        from the encoder fallback path (``semantic.py`` returns
        ``np.zeros(...)`` when the model is unhealthy) MUST NOT seed the
        centroid. If it did, every subsequent ``semantic_shift_rule``
        call would hit ``denom == 0.0`` and return True, slicing the
        episode at every event."""
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        pending.update_centroid(np.zeros(3, dtype=np.float32))
        assert pending.centroid_count == 0
        assert pending.centroid_embedding is None
        # A real embedding then succeeds.
        pending.update_centroid(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        assert pending.centroid_count == 1
        np.testing.assert_array_equal(pending.centroid_embedding, [1.0, 0.0, 0.0])

    def test_centroid_does_not_mutate_caller_array(self):
        """``update_centroid`` MUST own its centroid copy; the caller's
        embedding array (which the encoder layer may cache) must not be
        mutated by subsequent updates."""
        pending = PendingEpisodeState(id="ep_1", start_tick=0, last_tick=0, channel="text")
        original = np.asarray([1.0, 0.0], dtype=np.float32)
        snapshot = original.copy()
        pending.update_centroid(original)
        # Subsequent update with a different vector must not write back
        # into ``original``.
        pending.update_centroid(np.asarray([0.0, 1.0], dtype=np.float32))
        np.testing.assert_array_equal(original, snapshot)


class TestSemanticShiftHippocampusIntegration:
    """End-to-end: install ``semantic_shift_rule`` on a Hippocampus,
    feed events with embeddings, observe episode close behavior."""

    def test_shift_closes_pending_episode(self):
        """Two same-topic events then one orthogonal event closes the
        pending episode and opens a new one."""
        from maxim.memory.episode import semantic_shift_rule

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h._episode_detector.add_rule(semantic_shift_rule(threshold=0.5))

        v1 = np.asarray([1.0, 0.0], dtype=np.float32)
        v2 = np.asarray([0.9, 0.1], dtype=np.float32)  # similar to v1
        v3 = np.asarray([0.0, 1.0], dtype=np.float32)  # orthogonal to centroid

        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",), embedding=v1))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",), embedding=v2))
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",), embedding=v3))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        assert set(episodes[0].activated_nodes) == {"a", "b"}
        assert set(episodes[1].activated_nodes) == {"c"}

    def test_no_embeddings_legacy_callers_unchanged(self):
        """Backwards-compat: with the rule installed but NO event
        carrying an embedding, behavior matches a Hippocampus without
        the rule."""
        from maxim.memory.episode import semantic_shift_rule

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h._episode_detector.add_rule(semantic_shift_rule(threshold=0.5))

        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",)))
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",)))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 1
        assert set(episodes[0].activated_nodes) == {"a", "b", "c"}

    def test_partial_embeddings_only_count_toward_centroid(self):
        """Mixed populated/None embeddings: the centroid should reflect
        only events that supplied an embedding."""
        from maxim.memory.episode import semantic_shift_rule

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h._episode_detector.add_rule(semantic_shift_rule(threshold=0.5))

        v1 = np.asarray([1.0, 0.0], dtype=np.float32)
        v_orthogonal = np.asarray([0.0, 1.0], dtype=np.float32)

        # ev0: embedding v1 → centroid = v1, count=1
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",), embedding=v1))
        # ev1: no embedding → centroid unchanged
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",)))
        # ev2: orthogonal embedding → 1-cos(v_orth, v1) = 1 > 0.5 → close
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",), embedding=v_orthogonal))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        assert set(episodes[0].activated_nodes) == {"a", "b"}
        assert set(episodes[1].activated_nodes) == {"c"}

    def test_centroid_resets_on_episode_close(self):
        """After close, the new pending episode starts with a fresh
        centroid seeded by the event that caused the close."""
        from maxim.memory.episode import semantic_shift_rule

        h = _fresh_hippocampus(boundary_tick_gap=1000)
        h._episode_detector.add_rule(semantic_shift_rule(threshold=0.5))

        v1 = np.asarray([1.0, 0.0], dtype=np.float32)
        v2 = np.asarray([0.0, 1.0], dtype=np.float32)
        v3 = np.asarray([0.0, 0.95], dtype=np.float32)  # similar to v2 → no shift

        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a",), embedding=v1))
        # v2 closes ep1, opens ep2 with centroid seeded at v2.
        h.observe_episode_event(CaptureEvent(tick=1, channel="text", activated_nodes=("b",), embedding=v2))
        # v3 is similar to v2 → no shift; stays in ep2.
        h.observe_episode_event(CaptureEvent(tick=2, channel="text", activated_nodes=("c",), embedding=v3))
        h.finalize_pending_episode()

        episodes = h._episode_store.all_episodes()
        assert len(episodes) == 2
        assert set(episodes[0].activated_nodes) == {"a"}
        assert set(episodes[1].activated_nodes) == {"b", "c"}
