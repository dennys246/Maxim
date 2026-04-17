"""Valence annotation — Stage 1 + Stage 2 mechanism tests.

Covers:
- Reaction captured into pending episode via capture_reaction.
- Episode valence computed from reaction list (positive, negative, mixed).
- apply_hebbian_on_close annotates edges with metadata["valence"].
- Valence accumulates across multiple episodes with decay.
- Episode valence field persists through to_dict / from_dict round-trip.
- Edge metadata["valence"] persists through DependencyGraph to_dict / from_dict.
- spreading_activation with propagate_valence returns valence alongside activation.
- Valence decays through multi-hop paths.
- retrieve_on_cue with include_valence passes through.
"""

from __future__ import annotations

import time

import pytest

from maxim.agents.bus import DependencyGraph, Edge, EdgeType
from maxim.decisions.causal_link import Valence
from maxim.memory.episode import (
    CaptureEvent,
    Episode,
    PendingEpisodeState,
    apply_hebbian_on_close,
)
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
)
from maxim.reactions.types import Reaction, ReactionContext


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


def _make_reaction(
    kind: str = "pain",
    intensity: float = 0.8,
    valence: Valence = Valence.NEGATIVE,
    source: str = "test",
) -> Reaction:
    return Reaction(
        kind=kind,
        intensity=intensity,
        valence=valence,
        timestamp=time.time(),
        context=ReactionContext(),
        source=source,
    )


def _close_episode_with_nodes_and_reactions(
    h: Hippocampus,
    nodes: tuple[str, ...],
    reactions: list[Reaction] | None = None,
    tick: int = 0,
) -> Episode:
    """Open episode, inject reactions, then force-close."""
    h.observe_episode_event(CaptureEvent(tick=tick, channel="text", activated_nodes=nodes))
    if reactions:
        for r in reactions:
            h.capture_reaction(r)
    episode = h.finalize_pending_episode()
    assert episode is not None
    return episode


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — Reaction capture + edge valence annotation
# ─────────────────────────────────────────────────────────────────────────


class TestReactionCapture:
    def test_reaction_appended_to_pending_episode(self):
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a", "b")))
        r = _make_reaction()
        h.capture_reaction(r)
        assert len(h._pending_episode.reactions) == 1
        assert h._pending_episode.reactions[0] is r

    def test_capture_noop_when_no_pending_episode(self):
        h = _fresh_hippocampus()
        # No episode started — should not raise
        h.capture_reaction(_make_reaction())

    def test_multiple_reactions_accumulate(self):
        h = _fresh_hippocampus()
        h.observe_episode_event(CaptureEvent(tick=0, channel="text", activated_nodes=("a", "b")))
        h.capture_reaction(_make_reaction(intensity=0.3))
        h.capture_reaction(_make_reaction(intensity=0.5, valence=Valence.POSITIVE))
        assert len(h._pending_episode.reactions) == 2


class TestEpisodeValenceComputation:
    def test_negative_reaction_produces_negative_valence(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        pending.reactions.append(_make_reaction(intensity=0.8, valence=Valence.NEGATIVE))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(-0.8)

    def test_positive_reaction_produces_positive_valence(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        pending.reactions.append(_make_reaction(intensity=0.6, valence=Valence.POSITIVE))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(0.6)

    def test_neutral_reaction_produces_zero_valence(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        pending.reactions.append(_make_reaction(intensity=0.5, valence=Valence.NEUTRAL))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(0.0)

    def test_mixed_reactions_sum_and_clamp(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        # -0.8 + 0.3 = -0.5
        pending.reactions.append(_make_reaction(intensity=0.8, valence=Valence.NEGATIVE))
        pending.reactions.append(_make_reaction(intensity=0.3, valence=Valence.POSITIVE))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(-0.5)

    def test_valence_clamped_at_negative_one(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        pending.reactions.append(_make_reaction(intensity=0.8, valence=Valence.NEGATIVE))
        pending.reactions.append(_make_reaction(intensity=0.9, valence=Valence.NEGATIVE))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(-1.0)

    def test_valence_clamped_at_positive_one(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        pending.reactions.append(_make_reaction(intensity=0.8, valence=Valence.POSITIVE))
        pending.reactions.append(_make_reaction(intensity=0.9, valence=Valence.POSITIVE))
        episode = pending.finalize()
        assert episode.valence == pytest.approx(1.0)

    def test_no_reactions_produces_zero_valence(self):
        pending = PendingEpisodeState(id="ep1", start_tick=0, last_tick=10, channel="text")
        episode = pending.finalize()
        assert episode.valence == pytest.approx(0.0)


class TestEdgeValenceAnnotation:
    def test_negative_episode_annotates_edges(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b", "c"),
            reactions=[_make_reaction(intensity=0.8, valence=Valence.NEGATIVE)],
        )
        # Check edge a→b
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("valence") == pytest.approx(-0.8)
        # Check reverse
        rev = h._binding_graph.find_edge("b", "a", EdgeType.ASSOCIATES)
        assert rev is not None
        assert rev.metadata.get("valence") == pytest.approx(-0.8)

    def test_positive_episode_annotates_edges(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(
            h,
            ("x", "y"),
            reactions=[_make_reaction(intensity=0.5, valence=Valence.POSITIVE)],
        )
        edge = h._binding_graph.find_edge("x", "y", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("valence") == pytest.approx(0.5)

    def test_zero_valence_episode_does_not_annotate(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(h, ("a", "b"))
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert "valence" not in edge.metadata

    def test_valence_accumulates_with_decay(self):
        h = _fresh_hippocampus()
        # First episode: valence = -0.8
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b"),
            reactions=[_make_reaction(intensity=0.8, valence=Valence.NEGATIVE)],
            tick=0,
        )
        # Second episode: same nodes, valence = -0.5
        # The boundary detector will close the first episode (tick gap)
        # then open a new one. But we already finalized above, so start fresh.
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b"),
            reactions=[_make_reaction(intensity=0.5, valence=Valence.NEGATIVE)],
            tick=100,
        )
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        # Expected: (-0.8 * 0.95) + (-0.5) = -0.76 - 0.5 = -1.26, clamped to -1.0
        assert edge.metadata["valence"] == pytest.approx(-1.0)

    def test_valence_decay_prevents_saturation_over_many_episodes(self):
        """Many small negative episodes should converge, not stack to -1.0 immediately."""
        h = _fresh_hippocampus()
        for i in range(5):
            _close_episode_with_nodes_and_reactions(
                h,
                ("a", "b"),
                reactions=[_make_reaction(intensity=0.1, valence=Valence.NEGATIVE)],
                tick=i * 100,
            )
        edge = h._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        # With decay=0.95, 5 episodes of -0.1 each:
        # v0 = -0.1
        # v1 = -0.1*0.95 + -0.1 = -0.195
        # v2 = -0.195*0.95 + -0.1 = -0.28525
        # v3 = -0.28525*0.95 + -0.1 = -0.3709875
        # v4 = -0.3709875*0.95 + -0.1 = -0.452438125
        assert -0.5 < edge.metadata["valence"] < -0.4


class TestEpisodePersistence:
    def test_episode_valence_survives_round_trip(self):
        ep = Episode(
            id="ep1",
            start_tick=0,
            end_tick=10,
            channel="text",
            sender_ids=("alice",),
            thread_id=None,
            activated_nodes=("a", "b"),
            reward_events=(),
            scn_tag=None,
            valence=-0.75,
        )
        data = ep.to_dict()
        restored = Episode.from_dict(data)
        assert restored.valence == pytest.approx(-0.75)

    def test_episode_default_valence_on_old_data(self):
        """Old episode dicts without valence field should default to 0.0."""
        data = {
            "id": "ep1",
            "start_tick": 0,
            "end_tick": 10,
            "channel": "text",
            "sender_ids": [],
            "activated_nodes": ["a", "b"],
            "reward_events": [],
        }
        ep = Episode.from_dict(data)
        assert ep.valence == pytest.approx(0.0)

    def test_edge_valence_survives_graph_round_trip(self):
        """Edge.metadata["valence"] persists through DependencyGraph.to_dict/from_dict."""
        graph: DependencyGraph = DependencyGraph()
        graph.add_node("a", "a")
        graph.add_node("b", "b")
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=0.5, metadata={"valence": -0.7})

        data = graph.to_dict()
        restored = DependencyGraph.from_dict(data)
        edge = restored.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("valence") == pytest.approx(-0.7)

    def test_hippocampus_binding_graph_valence_survives_dump_load(self):
        """Full Hippocampus persistence round-trip preserves edge valence."""
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b"),
            reactions=[_make_reaction(intensity=0.6, valence=Valence.NEGATIVE)],
        )

        # Dump and reload
        state = h.dump()
        h2 = _fresh_hippocampus()
        h2.load_state(state)

        edge = h2._binding_graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge is not None
        assert edge.metadata.get("valence") == pytest.approx(-0.6)


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — Spreading activation valence propagation
# ─────────────────────────────────────────────────────────────────────────


class TestSpreadingActivationValence:
    def test_propagate_valence_off_returns_plain_dict(self):
        """Default behavior unchanged — returns dict[str, float]."""
        graph: DependencyGraph = DependencyGraph()
        graph.add_node("a", "a")
        graph.add_node("b", "b")
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=0.8, metadata={"valence": -0.5})
        graph.add_edge("b", "a", EdgeType.ASSOCIATES, weight=0.8, metadata={"valence": -0.5})

        result = graph.spreading_activation(["a"], decay=0.5)
        assert isinstance(result, dict)
        # Should be plain floats (activation only)
        assert isinstance(result.get("b", 0.0), float)

    def test_propagate_valence_on_returns_valence(self):
        graph: DependencyGraph = DependencyGraph()
        graph.add_node("a", "a")
        graph.add_node("b", "b")
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=0.8, metadata={"valence": -0.7})
        graph.add_edge("b", "a", EdgeType.ASSOCIATES, weight=0.8, metadata={"valence": -0.7})

        result = graph.spreading_activation(["a"], decay=0.5, propagate_valence=True)
        assert "b" in result
        activation, valence = result["b"]
        assert activation > 0.0
        assert valence < 0.0  # negative valence propagated

    def test_valence_decays_through_multi_hop(self):
        graph: DependencyGraph = DependencyGraph()
        for n in ("a", "b", "c"):
            graph.add_node(n, n)
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": -0.8})
        graph.add_edge("b", "a", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": -0.8})
        graph.add_edge("b", "c", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": -0.3})
        graph.add_edge("c", "b", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": -0.3})

        result = graph.spreading_activation(
            ["a"],
            decay=0.5,
            propagate_valence=True,
            max_depth=3,
        )
        assert "b" in result and "c" in result
        _, val_b = result["b"]
        _, val_c = result["c"]
        # b is 1-hop: valence = -0.8 (edge valence)
        # c is 2-hop: valence from b decays + edge valence
        assert val_b < 0.0
        assert val_c < 0.0
        # c's valence should be weaker than b's (more hops)
        assert abs(val_c) < abs(val_b)

    def test_positive_and_negative_valence_cancel(self):
        graph: DependencyGraph = DependencyGraph()
        for n in ("a", "b", "c"):
            graph.add_node(n, n)
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": -0.5})
        graph.add_edge("b", "a", EdgeType.ASSOCIATES, weight=1.0)
        graph.add_edge("a", "c", EdgeType.ASSOCIATES, weight=1.0, metadata={"valence": 0.5})
        graph.add_edge("c", "a", EdgeType.ASSOCIATES, weight=1.0)
        # "a" has both negative (→b) and positive (→c) edges
        result = graph.spreading_activation(["a"], decay=0.5, propagate_valence=True)
        _, val_b = result["b"]
        _, val_c = result["c"]
        assert val_b < 0.0
        assert val_c > 0.0

    def test_no_valence_metadata_treated_as_zero(self):
        graph: DependencyGraph = DependencyGraph()
        graph.add_node("a", "a")
        graph.add_node("b", "b")
        graph.add_edge("a", "b", EdgeType.ASSOCIATES, weight=0.8)  # no valence in metadata
        graph.add_edge("b", "a", EdgeType.ASSOCIATES, weight=0.8)

        result = graph.spreading_activation(["a"], decay=0.5, propagate_valence=True)
        _, val = result["b"]
        assert val == pytest.approx(0.0)


class TestRetrieveOnCueValence:
    def test_include_valence_returns_tuples_with_three_elements(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b", "c"),
            reactions=[_make_reaction(intensity=0.6, valence=Valence.NEGATIVE)],
        )
        results = h.retrieve_on_cue("a", include_valence=True)
        assert len(results) > 0
        for item in results:
            assert len(item) == 3, f"Expected (node, activation, valence), got {item}"
            node, activation, valence = item
            assert isinstance(node, str)
            assert activation > 0.0
            assert valence < 0.0  # negative from pain reaction

    def test_include_valence_false_returns_standard_pairs(self):
        h = _fresh_hippocampus()
        _close_episode_with_nodes_and_reactions(
            h,
            ("a", "b"),
            reactions=[_make_reaction(intensity=0.5, valence=Valence.NEGATIVE)],
        )
        results = h.retrieve_on_cue("a")
        assert len(results) > 0
        for item in results:
            assert len(item) == 2, f"Expected (node, activation), got {item}"
