"""Scale stress test for per-agent attribution under Roy-1-class loads.

The CC4 per-agent stash dicts and the P4 ``agent_id``-keyed
``record_outcome`` were tested at small N in
[test_multi_agent_attribution.py](test_multi_agent_attribution.py)
(5 tools × 3 outcomes per agent, 50 stash iterations per agent).
Roy-1a produces ~50 priming turns and ~30 test turns per arm — but
forward iterations (Roy-1: Adversarial in particular, 1,000+
priming turns) will push N two orders of magnitude higher.

This module locks down the per-agent attribution invariants at the
scale forward Roys will hit:

* **N=1,000 cluster updates per agent** — exercises
  ``NAc.update_cluster_reward`` under concurrent writes from many
  agents sharing a single NAc instance. Each agent's cluster bias
  triples must be partitionable by ``agent_id`` without cross-
  contamination.
* **N=1,000 ``record_outcome`` calls per agent** — exercises the
  P4 ``agent_id``-keyed causal-link attribution chain (``record_outcome``
  → ``nac.observe`` → persisted ``event_context["agent_id"]``).
* **N=500 stash R/W cycles per agent across 4 agents** — exercises
  the bio_integration per-agent dicts at higher producer/consumer
  contention than the original tests.

The test mocks the LLM (no real sims invoked — per CLAUDE.md "Don't
run sims from test suites"). Wall time on the dev box is well under
60s for the whole module; the tests are NOT marked ``slow``.

Reference: persona_convergence_crucible.md §"Open questions" item 3.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.runtime.bio_integration import (
    consume_pain_intensity,
    consume_substrate_nodes,
    observe_episode,
    record_pain_intensity,
    record_substrate_nodes,
    reset_agent_stash,
)
from maxim.runtime.tool_dispatch import record_outcome


# Number of distinct cluster updates per agent. Forward Roys (Roy-1:
# Adversarial, 1000+ priming turns) will produce updates of this
# order; this is the tripwire that catches per-agent attribution
# regressions before a 4-hour priming burn exposes them.
_N_PER_AGENT = 1000

# Four agents — enough to surface partitioning bugs that two-agent
# tests can miss (e.g. modulo-2 hash collisions). Roy-1 multi-agent
# crucible scenarios are likely to run 3-6 agents in parallel.
_AGENT_IDS = ("agent_alpha", "agent_bravo", "agent_charlie", "agent_delta")


def _make_pool() -> Any:
    pool = MagicMock()
    pool.add_outcome = MagicMock()
    return pool


@pytest.fixture
def stash_reset() -> Any:
    """Reset the bio_integration per-agent stash after every test."""
    yield
    for aid in _AGENT_IDS:
        reset_agent_stash(aid)


class TestClusterRewardScaleAttribution:
    """``NAc.update_cluster_reward`` at N=1000 per agent under shared-NAc concurrency.

    Each agent writes to a private ``(agent_id, cluster, tool)``
    triple space. After the writes, ``_cluster_reward_bias`` must
    be partitionable by ``agent_id`` with zero cross-agent
    contamination — every key's ``agent_id`` component must match
    the agent that wrote it, and every agent's keyspace must have
    exactly ``_N_PER_AGENT`` distinct entries.
    """

    def test_shared_nac_n1000_cluster_writes_partition_cleanly(self, tmp_path: Path) -> None:
        nac = NAc(config=NACConfig(persistence_path=str(tmp_path / "shared_nac.json")))

        def writer(agent_id: str) -> None:
            # Distinct cluster ids per agent so cross-agent contamination
            # shows up as either (a) the wrong agent_id on the key or
            # (b) the wrong cluster_id under a known agent_id.
            for i in range(_N_PER_AGENT):
                nac.update_cluster_reward(
                    agent_id=agent_id,
                    cluster_id=f"{agent_id}_cluster_{i}",
                    tool_signature=f"tool:{agent_id}_action",
                    reward=1.0,
                )

        threads = [threading.Thread(target=writer, args=(aid,)) for aid in _AGENT_IDS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Round-trip persistence — the realistic failure surface is
        # the dump/load path, not the in-memory dict.
        nac_path = tmp_path / "shared_nac.json"
        nac.save(str(nac_path))
        state = json.loads(nac_path.read_text())

        cluster_bias = state.get("cluster_reward_bias", {})
        assert cluster_bias, "shared NAc persisted no cluster_reward_bias"

        # Bucket every persisted key by its agent_id (first component
        # of the `\x1f`-joined triple). Each agent must have exactly
        # _N_PER_AGENT entries, all with cluster ids matching their
        # own namespace.
        per_agent_keys: dict[str, set[str]] = {aid: set() for aid in _AGENT_IDS}
        for key in cluster_bias:
            parts = key.split("\x1f")
            assert len(parts) == 3, f"malformed cluster_reward_bias key: {key!r}"
            aid, cluster_id, tool_sig = parts
            assert aid in per_agent_keys, f"unknown agent_id in key: {aid!r}"
            assert cluster_id.startswith(f"{aid}_cluster_"), (
                f"cross-agent contamination — cluster_id {cluster_id!r} under agent_id {aid!r}"
            )
            assert tool_sig == f"tool:{aid}_action", f"cross-agent tool contamination — {tool_sig!r} under {aid!r}"
            per_agent_keys[aid].add(key)

        for aid, keys in per_agent_keys.items():
            assert len(keys) == _N_PER_AGENT, f"agent {aid}: expected {_N_PER_AGENT} keys, got {len(keys)}"

        # Bias values are clamped to [-max_cluster_reward_bias, +cap]
        # and each agent's keys received reward=1.0 once, so every
        # value should be positive (>= reward_bias_alpha * 1.0 - epsilon).
        for key, value in cluster_bias.items():
            assert isinstance(value, (int, float)), f"non-numeric bias: {value!r}"
            assert value > 0.0, f"non-positive bias on {key!r}: {value}"


class TestRecordOutcomeScaleAttribution:
    """``record_outcome`` at N=1000 per agent under shared-NAc concurrency.

    Each persisted ``CausalLink.event_context['agent_id']`` must
    match the agent that produced the outcome. Forward Roys filter
    learning per-agent via this field; cross-agent leakage at scale
    would silently corrupt every downstream NAc query.
    """

    def test_shared_nac_n1000_record_outcomes_tag_agent_id(self, tmp_path: Path) -> None:
        nac = NAc(config=NACConfig(persistence_path=str(tmp_path / "shared_nac.json")))

        def writer(agent_id: str) -> None:
            pool = _make_pool()
            for i in range(_N_PER_AGENT):
                record_outcome(
                    agent_id=agent_id,
                    tool_name=f"{agent_id}_tool_{i % 50}",
                    success=(i % 3 != 0),
                    result_summary=f"{agent_id}-result-{i}",
                    error=None if i % 3 != 0 else "boom",
                    reasoning=f"{agent_id} doing thing {i}",
                    recent_outcomes=[],
                    max_recent=10,
                    llm_worker=None,
                    context_pool=pool,
                    nac=nac,
                    elapsed_s=0.01,
                    active_goal=f"{agent_id}_goal",
                )

        threads = [threading.Thread(target=writer, args=(aid,)) for aid in _AGENT_IDS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        nac_path = tmp_path / "shared_nac.json"
        nac.save(str(nac_path))
        state = json.loads(nac_path.read_text())
        links = state["links"]

        # Bucket every persisted link by event_context.agent_id and
        # assert tool prefixes match the originating agent.
        per_agent_tool_prefixes: dict[str, set[str]] = {aid: set() for aid in _AGENT_IDS}
        untagged = 0
        for sig, link_dicts in links.items():
            for link in link_dicts:
                aid = link["event_context"].get("agent_id")
                if aid is None:
                    untagged += 1
                    continue
                assert aid in per_agent_tool_prefixes, f"unknown agent_id in link: {aid!r}"
                per_agent_tool_prefixes[aid].add(sig)

        assert untagged == 0, f"{untagged} links missing agent_id tag — P4 attribution gap regression"

        # Each agent's tool signatures must all start with that
        # agent's prefix (no cross-agent leakage into the same NAc
        # link bucket).
        for aid, sigs in per_agent_tool_prefixes.items():
            assert sigs, f"agent {aid} produced no persisted tool links"
            for sig in sigs:
                assert f"{aid}_tool_" in sig, f"agent {aid} link bucket leaked sig {sig!r} from a different agent"

        # Pairwise disjoint tool-signature sets across agents.
        seen_total: dict[str, str] = {}
        for aid, sigs in per_agent_tool_prefixes.items():
            for sig in sigs:
                prior = seen_total.get(sig)
                assert prior is None or prior == aid, f"signature {sig!r} appeared under both {prior!r} and {aid!r}"
                seen_total[sig] = aid


class TestStashScaleIsolation:
    """``bio_integration`` per-agent stash dicts under 4-way concurrency.

    Producer/consumer interleaving at N=500 per agent surfaces lock
    granularity bugs (e.g. the pain-intensity RMW that needs an
    explicit ``threading.Lock`` per the CLAUDE.md "Per-agent stash
    dicts" lesson). Original test runs N=50 per two agents; this one
    quadruples both axes.
    """

    _STASH_ITERS = 500

    def test_substrate_stash_round_trip_isolation(self, stash_reset: Any) -> None:
        del stash_reset

        recorded: dict[str, list[tuple[str, ...]]] = {aid: [] for aid in _AGENT_IDS}

        def producer(agent_id: str, label: str) -> None:
            for i in range(self._STASH_ITERS):
                record_substrate_nodes((f"{label}-{i}",), agent_id=agent_id)
                got = consume_substrate_nodes(agent_id=agent_id)
                recorded[agent_id].append(got)

        threads = [threading.Thread(target=producer, args=(aid, aid.split("_")[1])) for aid in _AGENT_IDS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every consume returned the producing agent's nodes — never
        # another agent's. With 4 agents writing concurrently for 500
        # iterations each, a single shared-stash slot would interleave
        # and produce thousands of cross-contamination events.
        for aid in _AGENT_IDS:
            expected_prefix = aid.split("_")[1]
            for got in recorded[aid]:
                assert len(got) == 1
                assert got[0].startswith(f"{expected_prefix}-"), f"agent {aid} got cross-contaminated node: {got!r}"

    def test_pain_intensity_lock_holds_under_n500_producers(self, stash_reset: Any) -> None:
        del stash_reset

        # Each agent records intensities in its own non-overlapping
        # band so cross-agent contamination shows up as an out-of-band
        # max-merged value.
        bands = {
            "agent_alpha": (0.1, 0.2),
            "agent_bravo": (0.3, 0.4),
            "agent_charlie": (0.5, 0.6),
            "agent_delta": (0.7, 0.8),
        }

        def producer(agent_id: str) -> None:
            lo, hi = bands[agent_id]
            for i in range(self._STASH_ITERS):
                # Cycle through five values inside this agent's band.
                v = lo + (i % 5) * (hi - lo) / 5
                record_pain_intensity(v, agent_id=agent_id)

        threads = [threading.Thread(target=producer, args=(aid,)) for aid in _AGENT_IDS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for aid in _AGENT_IDS:
            lo, hi = bands[aid]
            got = consume_pain_intensity(agent_id=aid)
            assert got is not None, f"agent {aid}: stash was empty after producer"
            assert lo <= got <= hi, (
                f"agent {aid}: pain {got} outside band [{lo}, {hi}] — "
                f"another agent's higher value max-merged into this slot"
            )

    def test_observe_episode_ticks_monotonic_under_4way_concurrency(self, stash_reset: Any) -> None:
        del stash_reset

        hippos = {aid: MagicMock() for aid in _AGENT_IDS}

        def caller(agent_id: str) -> None:
            for _ in range(self._STASH_ITERS):
                observe_episode(hippocampus=hippos[agent_id], agent_id=agent_id)

        threads = [threading.Thread(target=caller, args=(aid,)) for aid in _AGENT_IDS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each agent's tick sequence is the contiguous range
        # [1, _STASH_ITERS]. Pre-fix a shared global counter would
        # interleave the four agents and produce a sparse sequence
        # for each.
        for aid, hippo in hippos.items():
            ticks = [c.args[0].tick for c in hippo.observe_episode_event.call_args_list]
            assert sorted(ticks) == list(range(1, self._STASH_ITERS + 1)), (
                f"agent {aid} tick sequence is not contiguous "
                f"1..{self._STASH_ITERS}: head={sorted(ticks)[:5]} "
                f"tail={sorted(ticks)[-5:]}"
            )
