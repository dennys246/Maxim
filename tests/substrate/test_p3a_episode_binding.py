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

import subprocess
import threading
import time

import pytest

from maxim.agents.bus import EdgeType
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.episode import (
    CaptureEvent,
    Episode,
    EpisodeBoundaryDetector,
    PendingEpisodeState,
    tick_gap_rule,
)
from maxim.memory.hippocampus import EpisodeConfig, Hippocampus, HippocampusConfig


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
                hebbian_init=hebbian_init,
                hebbian_delta=hebbian_delta,
                hebbian_max=hebbian_max,
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
    def test_partial_cue_retrieves_co_activated_nodes(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c", "d")

        results = h.retrieve_on_cue("a")
        returned_ids = {node for node, _ in results}
        assert returned_ids == {"b", "c", "d"}
        for _, weight in results:
            assert weight == pytest.approx(0.3)

    def test_partial_cue_non_member_returns_nothing(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b", "c")
        assert h.retrieve_on_cue("z") == []

    def test_multiple_episodes_shared_cue_merge(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes(h, "a", "b")
        _close_episode_with_nodes(h, "a", "c")

        results = dict(h.retrieve_on_cue("a"))
        assert set(results.keys()) == {"b", "c"}
        # a↔b reinforced once (one episode), so stays at hebbian_init
        assert results["b"] == pytest.approx(0.3)
        assert results["c"] == pytest.approx(0.3)

    def test_retrieve_respects_limit(self):
        h = _fresh_hippocampus()
        nodes = [f"n{i}" for i in range(10)]
        _close_episode_with_nodes(h, *nodes)

        results = h.retrieve_on_cue("n0", limit=3)
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
    def test_hebbian_fires_when_atl_is_none(self):
        """Hippocampus has no _atl field in Stage 1; this verifies the
        binding graph is fully Hippocampus-owned and doesn't depend on
        ATL being wired."""
        h = _fresh_hippocampus()
        assert not hasattr(h, "_atl") or getattr(h, "_atl", None) is None
        _close_episode_with_nodes(h, "a", "b")
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None

    def test_hebbian_fires_when_atl_wired_and_empty(self):
        """Even with a wired but empty ATL (`len(atl) == 0` is falsy),
        Hebbian updates must still fire. Regression guard for the
        `if self._atl` truthy-trap bug class that bit NAc twice in P2."""
        h = _fresh_hippocampus()
        # Wire a freshly-constructed (empty) ATL — not load-bearing for
        # Stage 1 Hebbian, but exercises the wire-check discipline.
        empty_atl = ATL(ATLConfig())
        assert len(empty_atl) == 0  # falsy under truthy check
        h._atl = empty_atl  # type: ignore[attr-defined]

        _close_episode_with_nodes(h, "a", "b", "c")
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None, "Hebbian must fire even with empty ATL wired"

    def test_p3a_diff_has_no_truthy_biosystem_checks(self):
        """Sentinel grep — the P3a diff must never contain truthy checks on
        bio-systems that have __len__."""
        result = subprocess.run(
            ["git", "diff", "main", "--", "src/maxim/memory/episode.py", "src/maxim/memory/hippocampus.py"],
            capture_output=True,
            text=True,
            check=False,
            cwd="/Users/dennyschaedig/Scripts/Maxim/.worktrees/p3",
        )
        if result.returncode != 0:
            pytest.skip(f"git diff unavailable: {result.stderr}")

        # Only check ADDED lines (prefixed with "+")
        added = [line for line in result.stdout.splitlines() if line.startswith("+") and not line.startswith("+++")]
        added_src = "\n".join(added)

        import re

        forbidden = re.compile(r"\bif self\._(atl|nac|hippocampus|scn|ec|angular_gyrus)\b")
        matches = forbidden.findall(added_src)
        assert not matches, (
            f"Forbidden truthy bio-system check in P3a diff: {matches}. Use `if self._X is not None` instead."
        )


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
            h.retrieve_on_cue("a")
            iterations += 1

        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "worker thread deadlocked"
        assert iterations > 10, f"retrieve_on_cue too slow ({iterations} iters in 2s) — possible lock contention"


# ─────────────────────────────────────────────────────────────────────────
# Persistence round-trip — depends on P3.5 Stage 1
# ─────────────────────────────────────────────────────────────────────────


class TestP3aPersistence:
    def test_episode_round_trips_via_hippocampus_dump(self):
        """Hippocampus._to_dict() reserved 'episodes' key must carry
        EpisodeStore contents through a dump → load_state cycle.

        Depends on P3.5 Stage 1 Hippocampus._to_dict() / load_state().
        """
        h1 = _fresh_hippocampus()
        _close_episode_with_nodes(h1, "a", "b", "c")
        _close_episode_with_nodes(h1, "x", "y")

        dumped = h1.dump()
        assert len(dumped["episodes"]) == 2

        h2 = _fresh_hippocampus()
        h2.load_state(dumped)

        assert len(h2._episode_store) == 2

        # The binding graph is NOT persisted in Stage 1 (it's derivable
        # from episodes, so Stage 2 can either persist it or rebuild it
        # from episode replay). This test asserts episodes round-trip,
        # not binding edges.
        episodes = h2._episode_store.all_episodes()
        all_nodes = set()
        for ep in episodes:
            all_nodes.update(ep.activated_nodes)
        assert all_nodes == {"a", "b", "c", "x", "y"}
