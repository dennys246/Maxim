"""P3a Stage 2 — TF-IDF bag-of-nodes retrieval baseline.

This is the head-to-head baseline the Hebbian binding mechanism must
beat by ``baseline_mean + 2 × baseline_std`` on mean F1 to clear
Stage 2's pass gate. The baseline reads the fixture's episode list
directly (no Hebbian edges, no episode boundary detector, no
Hippocampus surface) and exposes a ``retrieve`` method with the same
callable shape as ``Hippocampus.retrieve_on_cue``.

Design
======

Each episode is a "document" whose terms are its ``activated_nodes``.

- **Term frequency (TF)** for a node in an episode is 1 if the node
  is in the episode's activated set, 0 otherwise. (We do not count
  duplicates — each episode lists unique nodes.)
- **Inverse document frequency (IDF)** for a node is
  ``log(N / df(node))`` where ``N`` is the total number of episodes
  and ``df(node)`` is the number of episodes containing the node.
- For a cue, ranking a candidate node by TF-IDF-weighted overlap
  reduces to: for each episode containing the cue, add the candidate's
  IDF weight if the candidate is also in that episode. Sum across
  episodes.

Intuitively, candidates that co-occur with the cue in MANY episodes
score higher (more summands), but each occurrence is weighted by
INVERSE document frequency — common candidates get small weight,
rare candidates get large weight. This is the classic information-
retrieval baseline.

The fixture is deliberately designed to be adversarial for TF-IDF:

- Core nodes appear in 10 episodes per topic, so their IDF is small.
- Peripheral nodes appear in exactly 1 episode, so their IDF is
  large.
- When the cue is a core node, peripherals that co-occurred with it
  in its single episode get a big IDF push even though they're
  topic-specific noise.

A Hebbian mechanism that saturates core↔core edge weights via
repetition should cleanly beat this ranking. If it does NOT, the
mechanism isn't doing anything TF-IDF couldn't, and the plan says
that's a plan-ending finding for P3a (the margin is the whole point).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TfidfBaseline:
    """Read-only TF-IDF retriever over a fixture's episode list.

    Built once from ``episodes`` (the fixture's episode dicts). Call
    ``retrieve(cue, limit)`` with the same shape as
    ``Hippocampus.retrieve_on_cue``.
    """

    _episodes_by_node: dict[str, list[frozenset[str]]]
    _df: dict[str, int]
    _n_episodes: int

    @classmethod
    def from_episodes(cls, episodes: list[dict[str, Any]]) -> TfidfBaseline:
        """Build a baseline from fixture episode dicts.

        Each episode must have an ``activated_nodes`` key carrying a
        list of node-id strings.
        """
        episodes_by_node: dict[str, list[frozenset[str]]] = {}
        df: dict[str, int] = {}
        n_episodes = len(episodes)

        for ep in episodes:
            node_set = frozenset(ep["activated_nodes"])
            for node_id in node_set:
                episodes_by_node.setdefault(node_id, []).append(node_set)
                df[node_id] = df.get(node_id, 0) + 1

        return cls(
            _episodes_by_node=episodes_by_node,
            _df=df,
            _n_episodes=n_episodes,
        )

    def _idf(self, node_id: str) -> float:
        """``log(N / df)``, with df=0 producing 0.0 (unseen terms)."""
        doc_freq = self._df.get(node_id, 0)
        if doc_freq == 0 or self._n_episodes == 0:
            return 0.0
        return math.log(self._n_episodes / doc_freq)

    def retrieve(self, cue: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return candidates ranked by IDF-weighted episode co-overlap.

        Mirrors the ``Retriever`` protocol in ``p3a_metrics.py``.
        Returns an empty list if the cue was never seen in the
        fixture.
        """
        containing = self._episodes_by_node.get(cue, [])
        if not containing:
            return []

        scores: dict[str, float] = {}
        for episode_nodes in containing:
            for candidate in episode_nodes:
                if candidate == cue:
                    continue
                scores[candidate] = scores.get(candidate, 0.0) + self._idf(candidate)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit]
