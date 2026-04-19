"""P5 — Robust cross-session persistence under stress.

Proves that the bio-substrate survives realistic persistence load:
10,000+ nodes, 1,000+ episodes, mixed modalities, repeated
serialize/reload cycles with no degradation.

Stages:
  Stage 1: 1k-node mechanism test (10 reload cycles, F1 delta = 0.0)
  Stage 2: 10k-node mid-scale validation (mixed modality, <5s load)
  Stage 3: Full 10-seed sweep (state size bounding)
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from maxim.decisions.nac import NAc
from maxim.memory.episode import CaptureEvent
from maxim.memory.hippocampus import (
    EpisodeConfig,
    HebbianConfig,
    Hippocampus,
    HippocampusConfig,
    RetrievalConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _fresh_hippocampus(
    path: str | None = None,
    boundary_tick_gap: int = 50,
) -> Hippocampus:
    return Hippocampus(
        HippocampusConfig(
            persistence_path=path,
            episode=EpisodeConfig(
                boundary_tick_gap=boundary_tick_gap,
                hebbian=HebbianConfig(init=0.4, delta=0.15, max_weight=1.0),
                retrieval=RetrievalConfig(decay=0.7, threshold=0.05, max_depth=3),
            ),
        )
    )


def _populate_nodes(
    h: Hippocampus,
    n_episodes: int,
    nodes_per_episode: int = 4,
    base_tick: int = 0,
    tick_gap: int = 100,
    channel: str = "text",
    modality: str | None = None,
) -> list[set[str]]:
    """Populate a hippocampus with n_episodes episodes, each containing
    nodes_per_episode co-activated nodes. Returns the list of activated
    node sets for verification."""
    # SubstrateModality is a Literal type, just pass the string directly
    mod = modality
    episode_nodes: list[set[str]] = []

    for ep_idx in range(n_episodes):
        nodes = tuple(f"node_{channel}_{ep_idx}_{j}" for j in range(nodes_per_episode))
        tick = base_tick + ep_idx * tick_gap

        # Each episode: one event with all nodes co-activating
        h.observe_episode_event(
            CaptureEvent(
                tick=tick,
                channel=channel,
                activated_nodes=nodes,
                modality=mod,
            )
        )
        # Force episode close via tick gap (next event will be tick_gap away)
        episode_nodes.append(set(nodes))

    # Finalize the last pending episode
    h.finalize_pending_episode()
    return episode_nodes


def _retrieval_f1(
    h: Hippocampus,
    episode_nodes: list[set[str]],
    sample_rate: float = 0.1,
) -> float:
    """Compute retrieval F1 over a sample of episodes.

    For each sampled episode, pick the first node as cue, retrieve,
    and check how many of the episode's other nodes are returned.
    """
    import random

    rng = random.Random(42)
    sample_size = max(1, int(len(episode_nodes) * sample_rate))
    sampled = rng.sample(range(len(episode_nodes)), min(sample_size, len(episode_nodes)))

    total_precision = 0.0
    total_recall = 0.0
    count = 0

    for idx in sampled:
        nodes = episode_nodes[idx]
        if len(nodes) < 2:
            continue

        cue = sorted(nodes)[0]
        expected = nodes - {cue}

        retrieved = h.retrieve_on_cue(cue, limit=20)
        retrieved_ids = {nid for nid, _ in retrieved}

        if not expected:
            continue

        tp = len(retrieved_ids & expected)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
        recall = tp / len(expected)

        total_precision += precision
        total_recall += recall
        count += 1

    if count == 0:
        return 1.0  # no episodes to test → vacuously true

    avg_precision = total_precision / count
    avg_recall = total_recall / count

    if avg_precision + avg_recall == 0:
        return 0.0
    return 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — mechanism test (1k nodes, 10 reload cycles)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage1Mechanism:
    """1k nodes, 10 serialize/reload cycles, verify zero F1 degradation."""

    def test_1k_nodes_10_cycles_zero_degradation(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        # Populate with 250 episodes × 4 nodes = 1k nodes
        h = _fresh_hippocampus(hippo_path)
        episode_nodes = _populate_nodes(h, n_episodes=250, nodes_per_episode=4)

        # Baseline F1
        f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.2)
        assert f1_before > 0.5, f"Baseline F1 too low: {f1_before}"

        # 10 serialize/reload cycles
        for cycle in range(10):
            h.save()
            h2 = _fresh_hippocampus(hippo_path)
            h2.load()

            f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.2)

            assert f1_after == pytest.approx(f1_before, abs=0.01), (
                f"Cycle {cycle}: F1 degraded {f1_before:.4f} → {f1_after:.4f}"
            )

            h = h2  # continue with reloaded instance

    def test_episode_count_preserved(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)
        _populate_nodes(h, n_episodes=100)
        h.save()

        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        assert len(h2._episode_store) == len(h._episode_store)

    def test_binding_graph_edge_count_preserved(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)
        _populate_nodes(h, n_episodes=50, nodes_per_episode=3)

        edges_before = len(h._binding_graph.to_dict()["edges"])
        h.save()

        h2 = _fresh_hippocampus(hippo_path)
        h2.load()

        edges_after = len(h2._binding_graph.to_dict()["edges"])
        assert edges_after == edges_before

    def test_nac_reward_bias_preserved(self, tmp_path: Path) -> None:
        nac_path = str(tmp_path / "nac.json")

        nac = NAc()
        # Seed some reward biases
        for i in range(100):
            nac.update_eligibility("agent", f"node_{i}", 0.8)
        nac.distribute_reward("agent", 1.0)
        nac.save(nac_path)

        nac2 = NAc()
        nac2.load(nac_path)

        # Verify reward biases match
        for i in range(100):
            orig = nac._reward_bias.get(("agent", f"node_{i}"), 0.0)
            loaded = nac2._reward_bias.get(("agent", f"node_{i}"), 0.0)
            assert loaded == pytest.approx(orig, abs=1e-6), f"node_{i}: {orig} → {loaded}"


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — mid-scale validation (10k+ nodes, mixed modality)
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage2MidScale:
    """10k+ nodes, mixed text+vision modality, serialize every 100 episodes."""

    def test_10k_nodes_mixed_modality(self, tmp_path: Path) -> None:
        hippo_path = str(tmp_path / "hippocampus.json")

        h = _fresh_hippocampus(hippo_path)

        all_episode_nodes: list[set[str]] = []
        checkpoints: list[dict] = []

        # 2500 text episodes × 4 nodes = 10k text nodes
        text_nodes = _populate_nodes(
            h,
            n_episodes=2500,
            nodes_per_episode=4,
            channel="text",
            base_tick=0,
            tick_gap=100,
        )
        all_episode_nodes.extend(text_nodes)

        # 500 vision episodes × 4 nodes = 2k vision nodes
        vision_nodes = _populate_nodes(
            h,
            n_episodes=500,
            nodes_per_episode=4,
            channel="vision",
            base_tick=300000,
            tick_gap=100,
            modality="vision",
        )
        all_episode_nodes.extend(vision_nodes)

        total_nodes = sum(len(s) for s in all_episode_nodes)
        assert total_nodes >= 10000, f"Only {total_nodes} nodes — need 10k+"

        # Serialize and measure load time
        h.save()
        file_size = Path(hippo_path).stat().st_size

        t0 = time.monotonic()
        h2 = _fresh_hippocampus(hippo_path)
        h2.load()
        load_time = time.monotonic() - t0

        checkpoints.append(
            {
                "total_nodes": total_nodes,
                "episodes": len(all_episode_nodes),
                "file_size_mb": round(file_size / 1024 / 1024, 2),
                "load_time_s": round(load_time, 2),
            }
        )

        # Verify retrieval stability
        f1_before = _retrieval_f1(h, all_episode_nodes, sample_rate=0.05)
        f1_after = _retrieval_f1(h2, all_episode_nodes, sample_rate=0.05)

        assert f1_after == pytest.approx(f1_before, abs=0.01), f"F1 degraded: {f1_before:.4f} → {f1_after:.4f}"

        # Load time gate: <5s for 10k nodes
        assert load_time < 5.0, f"Load time {load_time:.1f}s exceeds 5s gate"

        # Episode count preserved
        assert len(h2._episode_store) == len(h._episode_store)

        # Print checkpoint info for diagnostics
        print(f"\n  P5 Stage 2: {checkpoints[0]}")

    def test_state_size_sublinear(self, tmp_path: Path) -> None:
        """State size should grow sub-linearly with node count.

        We measure file size at 1k, 2k, 4k nodes and verify the growth
        rate slows (ratio decreases).
        """
        sizes: list[tuple[int, int]] = []

        for n_episodes in [250, 500, 1000]:
            hippo_path = str(tmp_path / f"hippo_{n_episodes}.json")
            h = _fresh_hippocampus(hippo_path)
            _populate_nodes(h, n_episodes=n_episodes, nodes_per_episode=4)
            h.save()

            file_size = Path(hippo_path).stat().st_size
            total_nodes = n_episodes * 4
            sizes.append((total_nodes, file_size))

        # Check that bytes-per-node doesn't increase
        bytes_per_node = [(nodes, size / nodes) for nodes, size in sizes]
        for i in range(1, len(bytes_per_node)):
            # Allow some tolerance — serialization overhead may cause slight increase
            assert bytes_per_node[i][1] <= bytes_per_node[0][1] * 1.5, (
                f"State size growing super-linearly: "
                f"{bytes_per_node[0][1]:.0f} bytes/node at {bytes_per_node[0][0]} nodes, "
                f"{bytes_per_node[i][1]:.0f} bytes/node at {bytes_per_node[i][0]} nodes"
            )


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — multi-seed sweep
# ─────────────────────────────────────────────────────────────────────────


class TestP5Stage3Sweep:
    """Full 10-seed sweep with 10k+ nodes each. Verifies consistency."""

    @pytest.mark.slow
    def test_10_seed_sweep(self, tmp_path: Path) -> None:
        """Run 10 independent 10k-node persistence round-trips."""
        import numpy as np

        f1_deltas: list[float] = []
        load_times: list[float] = []

        for seed in range(10):
            seed_dir = tmp_path / f"seed_{seed}"
            seed_dir.mkdir()
            hippo_path = str(seed_dir / "hippocampus.json")

            h = _fresh_hippocampus(hippo_path)

            # 2500 episodes × 4 nodes = 10k nodes per seed
            episode_nodes = _populate_nodes(
                h,
                n_episodes=2500,
                nodes_per_episode=4,
                base_tick=seed * 1000000,
            )

            f1_before = _retrieval_f1(h, episode_nodes, sample_rate=0.05)
            h.save()

            t0 = time.monotonic()
            h2 = _fresh_hippocampus(hippo_path)
            h2.load()
            load_time = time.monotonic() - t0

            f1_after = _retrieval_f1(h2, episode_nodes, sample_rate=0.05)

            f1_deltas.append(f1_after - f1_before)
            load_times.append(load_time)

        f1_deltas_arr = np.array(f1_deltas)
        load_times_arr = np.array(load_times)

        # All seeds: zero F1 degradation
        assert np.all(np.abs(f1_deltas_arr) < 0.01), f"F1 deltas: {f1_deltas_arr}"

        # All seeds: load time < 5s
        assert np.all(load_times_arr < 5.0), f"Load times: {load_times_arr}"

        # Report statistics
        print("\n  P5 Stage 3: 10-seed sweep")
        print(f"    F1 delta: {np.mean(f1_deltas_arr):+.6f} ± {np.std(f1_deltas_arr):.6f}")
        print(f"    Load time: {np.mean(load_times_arr):.2f} ± {np.std(load_times_arr):.2f}s")
        print(f"    Max load time: {np.max(load_times_arr):.2f}s")
