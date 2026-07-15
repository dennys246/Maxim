"""P8 Sleep Replay — offline consolidation during sleep phases.

Dormant since 2026-07-14: the P8 experiment gate PASSED
(docs/experiments/p8_sleep_replay_results.md) but the production consumer was
never wired — no src/ code calls ``replay_top_episodes()`` on the SLEEP
transition (only tests/substrate/test_p8_sleep_replay.py exercises it).
The production session-end path is ``hippocampus_consolidation.py::sleep()``
(promote/compress/remove — a different mechanism, NOT Hebbian replay).
Resurrection trigger: docs/plans/deferred/memory_consolidation_practice.md (ACTIVE)
deciding to wire the SLEEP-transition consumer. Until then no new features
build on this module.

During an explicit sleep phase, replays the top-N rewarded episodes
with Hebbian link weight updates. Retrieval F1 improves on replayed
probes without any new input being presented.

The replay engine is invoked when ProcessingState transitions to SLEEP.
It is NOT called automatically — the agent loop or session manager
must call ``replay_top_episodes()`` explicitly.

Depends on:
- P3a (episode binding / Hebbian links)
- P6 (extinction — baseline decay so replay improvement is measurable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayResult:
    """Summary of a sleep replay cycle."""

    episodes_replayed: int
    edges_reinforced: int
    consolidation_multiplier: float


def rank_episodes_by_reward(
    episodes: list,
    reward_bias: dict,
    *,
    agent_id: str = "",
    top_n: int = 5,
) -> list:
    """Rank episodes by cumulative reward from NAc reward_bias.

    For each episode, sums the reward bias of its activated nodes.
    Returns the top-N episodes sorted by descending total reward.

    Args:
        episodes: List of Episode objects.
        reward_bias: NAc ``_reward_bias`` dict keyed by (agent_id, node_id).
        agent_id: Agent ID for reward bias lookup.
        top_n: Number of episodes to return.
    """
    scored: list[tuple[float, object]] = []
    for ep in episodes:
        nodes = getattr(ep, "activated_nodes", ())
        total_reward = sum(reward_bias.get((agent_id, nid), 0.0) for nid in nodes)
        # Also factor in episode valence if positive
        valence = getattr(ep, "valence", 0.0)
        if valence > 0:
            total_reward += valence * 0.5
        scored.append((total_reward, ep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ep for _, ep in scored[:top_n]]


def replay_top_episodes(
    hippocampus,
    nac=None,
    *,
    agent_id: str = "",
    top_n: int = 5,
    consolidation_multiplier: float = 1.5,
) -> ReplayResult:
    """Replay top-N rewarded episodes with Hebbian weight updates.

    This is the core P8 mechanism: during sleep, re-fire Hebbian
    updates on the binding graph for the most rewarded episodes.
    The consolidation_multiplier amplifies the Hebbian delta so
    replay produces stronger links than the original episode close.

    Args:
        hippocampus: Hippocampus instance (needs ``_episode_store``,
                     ``_binding_graph``, ``config``).
        nac: NAc instance (needs ``_reward_bias``). If None, falls
             back to episode valence for ranking.
        agent_id: Agent ID for NAc reward bias lookup.
        top_n: Number of top episodes to replay.
        consolidation_multiplier: Amplification factor for Hebbian
            delta during replay (> 1.0 means replay strengthens
            more than original binding).

    Returns:
        ReplayResult with counts.
    """
    from maxim.memory.episode import apply_hebbian_on_close

    # Get all closed episodes
    episode_store = getattr(hippocampus, "_episode_store", None)
    if episode_store is None:
        return ReplayResult(0, 0, consolidation_multiplier)

    all_episodes = episode_store.all_episodes()
    if not all_episodes:
        return ReplayResult(0, 0, consolidation_multiplier)

    # Rank by reward
    reward_bias = getattr(nac, "_reward_bias", {}) if nac is not None else {}
    top_episodes = rank_episodes_by_reward(
        all_episodes,
        reward_bias,
        agent_id=agent_id,
        top_n=top_n,
    )

    if not top_episodes:
        return ReplayResult(0, 0, consolidation_multiplier)

    # Get Hebbian config — use getattr for safety against missing attrs
    binding_graph = getattr(hippocampus, "_binding_graph", None)
    if binding_graph is None:
        return ReplayResult(0, 0, consolidation_multiplier)

    config = getattr(hippocampus, "config", None)
    if config is None or not hasattr(config, "episode") or not hasattr(config.episode, "hebbian"):
        return ReplayResult(0, 0, consolidation_multiplier)
    hebbian_cfg = config.episode.hebbian

    # Replay each episode with amplified Hebbian delta
    total_edges = 0
    for episode in top_episodes:
        # Count edges before replay to measure reinforcement
        nodes = episode.activated_nodes
        edges_before = _count_existing_edges(binding_graph, nodes)

        apply_hebbian_on_close(
            binding_graph,
            episode,
            hebbian_init=hebbian_cfg.init,
            hebbian_delta=hebbian_cfg.delta * consolidation_multiplier,
            hebbian_max=hebbian_cfg.max_weight,
            valence_decay=hebbian_cfg.valence_decay,
        )

        edges_after = _count_existing_edges(binding_graph, nodes)
        # New edges = edges_after - edges_before; reinforced = edges_before
        total_edges += max(edges_after, edges_before)

    logger.info(
        "Sleep replay: %d episodes, %d edges reinforced (multiplier=%.1f)",
        len(top_episodes),
        total_edges,
        consolidation_multiplier,
    )

    return ReplayResult(
        episodes_replayed=len(top_episodes),
        edges_reinforced=total_edges,
        consolidation_multiplier=consolidation_multiplier,
    )


def _count_existing_edges(binding_graph, nodes: tuple) -> int:
    """Count existing ASSOCIATES edges between node pairs."""
    from itertools import combinations

    from maxim.agents.bus import EdgeType

    count = 0
    for a, b in combinations(nodes, 2):
        if binding_graph.find_edge(a, b, EdgeType.ASSOCIATES) is not None:
            count += 1
    return count
