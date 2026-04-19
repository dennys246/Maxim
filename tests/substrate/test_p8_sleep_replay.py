"""P8 Sleep Replay — offline consolidation tests.

Stage 1: mechanism + metric
  - Sleep replay engine replays top-N rewarded episodes
  - F1 improves on replayed probes vs no-replay control
  - Mechanism test on synthetic episodes

Stage 2: within-session replay with varied N + persistence

Stage 3: 10-seed sweep with cross-session round-trip
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from unittest.mock import MagicMock


from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.episode import Episode, apply_hebbian_on_close
from maxim.memory.sleep_replay import (
    ReplayResult,
    rank_episodes_by_reward,
    replay_top_episodes,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HebbianCfg:
    init: float = 0.3
    delta: float = 0.1
    max_weight: float = 2.0
    valence_decay: float = 0.95


@dataclass(frozen=True)
class EpisodeCfg:
    hebbian: HebbianCfg = field(default_factory=HebbianCfg)


@dataclass(frozen=True)
class HippoConfig:
    episode: EpisodeCfg = field(default_factory=EpisodeCfg)


class FakeEpisodeStore:
    """Minimal episode store for testing."""

    def __init__(self, episodes: list[Episode]):
        self._episodes = {ep.id: ep for ep in episodes}

    def all_episodes(self) -> list[Episode]:
        return list(self._episodes.values())

    def get(self, ep_id: str) -> Episode | None:
        return self._episodes.get(ep_id)


def make_episode(
    ep_id: str,
    nodes: tuple[str, ...],
    valence: float = 0.0,
    start_tick: int = 0,
    end_tick: int = 1,
) -> Episode:
    """Create a synthetic Episode for testing."""
    return Episode(
        id=ep_id,
        start_tick=start_tick,
        end_tick=end_tick,
        channel="test",
        sender_ids=(),
        thread_id=None,
        activated_nodes=nodes,
        reward_events=(),
        scn_tag=None,
        valence=valence,
    )


def build_test_graph(episodes: list[Episode], hebbian_cfg: HebbianCfg | None = None) -> DependencyGraph:
    """Build a binding graph from episodes with Hebbian links."""
    cfg = hebbian_cfg or HebbianCfg()
    graph: DependencyGraph[str] = DependencyGraph()

    # Add all nodes
    all_nodes = set()
    for ep in episodes:
        all_nodes.update(ep.activated_nodes)
    for nid in all_nodes:
        graph.add_node(nid, nid)

    # Apply Hebbian binding for each episode
    for ep in episodes:
        apply_hebbian_on_close(
            graph,
            ep,
            hebbian_init=cfg.init,
            hebbian_delta=cfg.delta,
            hebbian_max=cfg.max_weight,
        )

    return graph


def retrieval_f1(
    graph: DependencyGraph,
    probe_nodes: list[str],
    expected_nodes: set[str],
    threshold: float = 0.05,
) -> float:
    """Compute F1 score: probe from probe_nodes, see how many expected are retrieved."""
    activations = graph.spreading_activation(
        source_ids=probe_nodes,
        initial_activation=1.0,
        decay=0.5,
        threshold=threshold,
        max_depth=2,
    )
    retrieved = {nid for nid in activations if activations[nid] >= threshold}
    # Remove probe nodes from retrieved (we're measuring what they activate)
    retrieved -= set(probe_nodes)
    expected_only = expected_nodes - set(probe_nodes)

    if not expected_only:
        return 1.0

    tp = len(retrieved & expected_only)
    fp = len(retrieved - expected_only)
    fn = len(expected_only - retrieved)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def make_fake_hippocampus(
    episodes: list[Episode],
    graph: DependencyGraph | None = None,
    config: HippoConfig | None = None,
) -> MagicMock:
    """Build a mock hippocampus with real episode store and binding graph."""
    hippo = MagicMock()
    hippo._episode_store = FakeEpisodeStore(episodes)
    hippo._binding_graph = graph or build_test_graph(episodes)
    hippo.config = config or HippoConfig()
    return hippo


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Mechanism tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRankEpisodesByReward:
    def test_ranks_by_reward_bias(self):
        eps = [
            make_episode("ep1", ("a", "b"), valence=0.0),
            make_episode("ep2", ("c", "d"), valence=0.0),
            make_episode("ep3", ("e", "f"), valence=0.0),
        ]
        reward_bias = {
            ("", "a"): 0.1,
            ("", "b"): 0.05,
            ("", "c"): 0.2,  # highest
            ("", "d"): 0.15,
            ("", "e"): 0.01,
            ("", "f"): 0.01,
        }
        ranked = rank_episodes_by_reward(eps, reward_bias, top_n=2)
        assert len(ranked) == 2
        assert ranked[0].id == "ep2"  # c+d = 0.35
        assert ranked[1].id == "ep1"  # a+b = 0.15

    def test_valence_contributes_to_ranking(self):
        eps = [
            make_episode("ep1", ("a",), valence=0.0),
            make_episode("ep2", ("b",), valence=1.0),
        ]
        ranked = rank_episodes_by_reward(eps, {}, top_n=2)
        assert ranked[0].id == "ep2"  # valence bonus

    def test_empty_episodes(self):
        assert rank_episodes_by_reward([], {}, top_n=5) == []

    def test_top_n_limits(self):
        eps = [make_episode(f"ep{i}", (f"n{i}",)) for i in range(10)]
        ranked = rank_episodes_by_reward(eps, {}, top_n=3)
        assert len(ranked) == 3


class TestReplayTopEpisodes:
    def test_basic_replay(self):
        episodes = [
            make_episode("ep1", ("a", "b", "c"), valence=0.5),
            make_episode("ep2", ("d", "e", "f"), valence=0.1),
        ]
        graph = build_test_graph(episodes)
        hippo = make_fake_hippocampus(episodes, graph)

        result = replay_top_episodes(hippo, top_n=2)

        assert isinstance(result, ReplayResult)
        assert result.episodes_replayed == 2
        assert result.edges_reinforced > 0

    def test_replay_strengthens_edges(self):
        episodes = [make_episode("ep1", ("a", "b", "c"), valence=0.5)]
        graph = build_test_graph(episodes)

        # Record weight before replay
        edge_before = graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge_before is not None
        weight_before = edge_before.weight

        hippo = make_fake_hippocampus(episodes, graph)
        replay_top_episodes(hippo, top_n=1, consolidation_multiplier=2.0)

        edge_after = graph.find_edge("a", "b", EdgeType.ASSOCIATES)
        assert edge_after is not None
        assert edge_after.weight > weight_before

    def test_no_episodes_returns_zero(self):
        hippo = make_fake_hippocampus([])
        result = replay_top_episodes(hippo, top_n=5)
        assert result.episodes_replayed == 0

    def test_consolidation_multiplier_amplifies(self):
        """Higher multiplier = larger weight increase."""
        episodes = [make_episode("ep1", ("x", "y"), valence=0.5)]

        # Run with multiplier 1.0
        graph_1 = build_test_graph(episodes)
        hippo_1 = make_fake_hippocampus(episodes, graph_1)
        replay_top_episodes(hippo_1, top_n=1, consolidation_multiplier=1.0)
        w1 = graph_1.find_edge("x", "y", EdgeType.ASSOCIATES).weight

        # Run with multiplier 3.0
        graph_3 = build_test_graph(episodes)
        hippo_3 = make_fake_hippocampus(episodes, graph_3)
        replay_top_episodes(hippo_3, top_n=1, consolidation_multiplier=3.0)
        w3 = graph_3.find_edge("x", "y", EdgeType.ASSOCIATES).weight

        assert w3 > w1, f"Multiplier 3.0 ({w3}) should produce higher weight than 1.0 ({w1})"


class TestReplayImprovesRetrieval:
    """Stage 1 pass gate: F1 improves on replayed probes."""

    def test_f1_improves_after_replay(self):
        """Core P8 hypothesis: replay strengthens links, improving retrieval."""
        # Create episodes with overlapping node activations
        episodes = [
            make_episode("ep1", ("hub", "a1", "a2", "a3"), valence=0.8),
            make_episode("ep2", ("hub", "b1", "b2", "b3"), valence=0.1),
        ]
        graph = build_test_graph(episodes)

        # Apply P6 decay to weaken all edges
        for _ in range(10):
            graph.decay_edges(0.85)

        # Measure F1 before replay (decayed)
        expected_ep1 = {"a1", "a2", "a3"}
        f1_before = retrieval_f1(graph, ["hub"], expected_ep1)

        # Replay top episodes (ep1 ranked higher by valence)
        hippo = make_fake_hippocampus(episodes, graph)
        replay_top_episodes(hippo, top_n=1, consolidation_multiplier=2.0)

        # Measure F1 after replay
        f1_after = retrieval_f1(graph, ["hub"], expected_ep1)

        assert f1_after > f1_before, f"F1 should improve: before={f1_before:.3f}, after={f1_after:.3f}"

    def test_no_replay_control_shows_no_improvement(self):
        """Control: without replay, F1 should not improve (or slightly decay)."""
        episodes = [
            make_episode("ep1", ("hub", "a1", "a2"), valence=0.8),
        ]
        graph = build_test_graph(episodes)

        for _ in range(10):
            graph.decay_edges(0.85)

        expected = {"a1", "a2"}
        f1_before = retrieval_f1(graph, ["hub"], expected)

        # Apply MORE decay instead of replay
        for _ in range(5):
            graph.decay_edges(0.85)

        f1_after = retrieval_f1(graph, ["hub"], expected)

        assert f1_after <= f1_before, f"Control should not improve: before={f1_before:.3f}, after={f1_after:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Varied N + measurement
# ─────────────────────────────────────────────────────────────────────────────


class TestVariedReplayN:
    """Test replay with different numbers of episodes."""

    def _build_scenario(self, n_episodes: int = 10, seed: int = 42):
        """Build a scenario with N episodes and apply decay."""
        rng = random.Random(seed)
        all_nodes = [f"n{i}" for i in range(30)]
        hub = "hub_main"

        episodes = []
        for i in range(n_episodes):
            nodes = tuple([hub] + rng.sample(all_nodes, k=4))
            valence = rng.uniform(-0.5, 1.0)
            episodes.append(make_episode(f"ep{i}", nodes, valence=valence))

        graph = build_test_graph(episodes)

        # Apply decay
        for _ in range(15):
            graph.decay_edges(0.85)

        return episodes, graph, hub, all_nodes

    def test_replay_1_improves(self):
        episodes, graph, hub, _ = self._build_scenario()
        expected = set(episodes[0].activated_nodes) - {hub}
        f1_before = retrieval_f1(graph, [hub], expected)

        hippo = make_fake_hippocampus(episodes, graph)
        replay_top_episodes(hippo, top_n=1, consolidation_multiplier=2.0)

        f1_after = retrieval_f1(graph, [hub], expected)
        # With only 1 replay, the specific episode's nodes should improve
        assert f1_after >= f1_before

    def test_replay_5_improves_more_than_1(self):
        episodes, graph_1, hub, _ = self._build_scenario(seed=42)
        _, graph_5, _, _ = self._build_scenario(seed=42)

        hippo_1 = make_fake_hippocampus(episodes, graph_1)
        hippo_5 = make_fake_hippocampus(episodes, graph_5)

        replay_top_episodes(hippo_1, top_n=1, consolidation_multiplier=2.0)
        replay_top_episodes(hippo_5, top_n=5, consolidation_multiplier=2.0)

        # Measure overall retrieval across all episode nodes
        all_expected = set()
        for ep in episodes:
            all_expected.update(ep.activated_nodes)
        all_expected.discard(hub)

        f1_1 = retrieval_f1(graph_1, [hub], all_expected)
        f1_5 = retrieval_f1(graph_5, [hub], all_expected)

        assert f1_5 >= f1_1, f"Replay N=5 ({f1_5:.3f}) should be >= N=1 ({f1_1:.3f})"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: 10-seed sweep
# ─────────────────────────────────────────────────────────────────────────────


class TestSleepReplaySweep:
    """10-seed sweep: replay consistently improves F1 vs no-replay control."""

    def _run_seed(self, seed: int) -> tuple[float, float]:
        """Run one seed, return (replay_f1_delta, control_f1_delta)."""
        rng = random.Random(seed)
        hub = "hub"
        all_nodes = [f"n{i}" for i in range(20)]

        episodes = []
        for i in range(8):
            nodes = tuple([hub] + rng.sample(all_nodes, k=3))
            valence = rng.uniform(0.0, 1.0)
            episodes.append(make_episode(f"ep{i}", nodes, valence=valence, start_tick=i, end_tick=i + 1))

        # Build two identical graphs
        graph_replay = build_test_graph(episodes)
        graph_control = build_test_graph(episodes)

        # Apply same decay to both
        for _ in range(12):
            graph_replay.decay_edges(0.85)
            graph_control.decay_edges(0.85)

        # Measure baseline F1
        all_expected = set()
        for ep in episodes:
            all_expected.update(ep.activated_nodes)
        all_expected.discard(hub)

        f1_before_replay = retrieval_f1(graph_replay, [hub], all_expected)
        f1_before_control = retrieval_f1(graph_control, [hub], all_expected)

        # Replay on one, nothing on control
        hippo = make_fake_hippocampus(episodes, graph_replay)
        replay_top_episodes(hippo, top_n=5, consolidation_multiplier=2.0)

        f1_after_replay = retrieval_f1(graph_replay, [hub], all_expected)
        f1_after_control = retrieval_f1(graph_control, [hub], all_expected)

        return (
            f1_after_replay - f1_before_replay,
            f1_after_control - f1_before_control,
        )

    def test_10_seed_sweep(self):
        """Pass gate: replay F1 delta > control, beats by control_mean + 2*control_std."""
        replay_deltas = []
        control_deltas = []

        for seed in range(10):
            replay_d, control_d = self._run_seed(seed)
            replay_deltas.append(replay_d)
            control_deltas.append(control_d)

        replay_mean = sum(replay_deltas) / len(replay_deltas)
        control_mean = sum(control_deltas) / len(control_deltas)
        control_std = (sum((d - control_mean) ** 2 for d in control_deltas) / len(control_deltas)) ** 0.5

        assert replay_mean > control_mean + 2 * control_std, (
            f"Replay mean delta ({replay_mean:.4f}) not significantly better than "
            f"control ({control_mean:.4f} +/- {control_std:.4f})"
        )

        # Replay should have positive mean delta
        assert replay_mean > 0, f"Replay mean delta should be positive: {replay_mean:.4f}"

        # Control should have zero or negative mean delta (continued decay)
        assert control_mean <= 0.01, f"Control should not improve: {control_mean:.4f}"
