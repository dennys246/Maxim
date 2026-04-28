"""P4 multi-agent learning attribution regression tests.

These tests lock down two correctness gaps surfaced during the
agent-backed entities audit (see ``docs/plans/v1_refinement.md`` Section 4
P4):

1. ``runtime/bio_integration.py`` previously kept module-level globals
   for ``_episode_tick``, ``_latest_substrate_nodes`` and
   ``_latest_pain_intensity``.  In a multi-agent scenario every agent
   shared a single tick counter and a single substrate-nodes slot, so
   substrate nodes encoded for agent A could land on agent B's next
   episode event.  P4 replaces those with per-agent dicts keyed at
   the ``Hippocampus.observe_episode_event()`` call site.

2. ``runtime/tool_dispatch.record_outcome()`` previously took no
   ``agent_id`` parameter, so the NAc ``event_context`` produced by
   each tool outcome had no per-agent attribution.  P4 makes
   ``agent_id`` a required keyword-only argument and tags every NAc
   observation's context dict with it.

The motivating bug class is the "silent no-op invariant" rule from
CLAUDE.md: forgetting to scope substrate nodes per agent or forgetting
to tag a NAc observation with an agent_id never raised — it just
produced quietly-wrong learning.  Pushing both into required types
makes the next instance of this bug a ``TypeError`` instead of silent
cross-agent attribution.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.runtime.agent_factory import AgentConfig, AgentFactory
from maxim.runtime.bio_integration import (
    consume_pain_intensity,
    consume_substrate_nodes,
    observe_episode,
    record_pain_intensity,
    record_substrate_nodes,
    reset_agent_stash,
)
from maxim.runtime.tool_dispatch import record_outcome


@pytest.fixture
def agent_pair(tmp_path: Path) -> tuple[Any, Any]:
    """Two fully-isolated AgentFactory agents sharing a tmp persistence root."""
    factory = AgentFactory(base_data_dir=tmp_path)
    agent_a = factory.create_agent(
        AgentConfig(
            agent_id="agent_alpha",
            role="npc",
            persistence_dir=str(tmp_path / "agent_alpha"),
            remembers=True,
            learns=True,
        )
    )
    agent_b = factory.create_agent(
        AgentConfig(
            agent_id="agent_bravo",
            role="npc",
            persistence_dir=str(tmp_path / "agent_bravo"),
            remembers=True,
            learns=True,
        )
    )
    yield agent_a, agent_b
    # Reset module-level per-agent stash so a later test can re-use these
    # agent_ids without seeing stale ticks / nodes / pain intensity.
    for aid in ("agent_alpha", "agent_bravo"):
        reset_agent_stash(aid)


class TestMultiAgentNacAttribution:
    """Two AgentFactory agents call record_outcome concurrently — their
    persisted nac.json files must contain NON-OVERLAPPING causal links,
    each correctly tagged with the originating agent_id."""

    @staticmethod
    def _make_pool() -> Any:
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def test_concurrent_record_outcome_isolates_per_agent_nac(
        self, agent_pair: tuple[Any, Any], tmp_path: Path
    ) -> None:
        agent_a, agent_b = agent_pair
        assert agent_a.nac is not None and agent_b.nac is not None

        pool_a = self._make_pool()
        pool_b = self._make_pool()

        # Each agent uses its OWN tool name set so we can reason about
        # link ownership purely from the persisted event_signatures.
        a_tools = ("a_tool_1", "a_tool_2", "a_tool_3")
        b_tools = ("b_tool_1", "b_tool_2", "b_tool_3")

        def run_agent_loop(agent_id: str, nac: Any, pool: Any, tools: tuple[str, ...]) -> None:
            for tool in tools:
                for success in (True, False, True):
                    record_outcome(
                        agent_id=agent_id,
                        tool_name=tool,
                        success=success,
                        result_summary=f"{tool}:{success}",
                        error=None if success else "boom",
                        reasoning=f"{agent_id} doing {tool}",
                        recent_outcomes=[],
                        max_recent=10,
                        llm_worker=None,
                        context_pool=pool,
                        nac=nac,
                        elapsed_s=0.5,
                        active_goal=f"{agent_id}_goal",
                    )

        # Run both "loops" concurrently so any shared mutable state would
        # interleave its writes.
        t_a = threading.Thread(
            target=run_agent_loop,
            args=("agent_alpha", agent_a.nac, pool_a, a_tools),
        )
        t_b = threading.Thread(
            target=run_agent_loop,
            args=("agent_bravo", agent_b.nac, pool_b, b_tools),
        )
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # Persist each agent's NAc to its own json file
        nac_a_path = tmp_path / "agent_alpha" / "nac.json"
        nac_b_path = tmp_path / "agent_bravo" / "nac.json"
        nac_a_path.parent.mkdir(parents=True, exist_ok=True)
        nac_b_path.parent.mkdir(parents=True, exist_ok=True)
        agent_a.nac.save(str(nac_a_path))
        agent_b.nac.save(str(nac_b_path))

        # Each file exists and parses
        a_state = json.loads(nac_a_path.read_text())
        b_state = json.loads(nac_b_path.read_text())

        a_links = a_state["links"]
        b_links = b_state["links"]

        # Both agents produced learning
        assert any(sig.startswith("tool:a_tool_") for sig in a_links), a_links.keys()
        assert any(sig.startswith("tool:b_tool_") for sig in b_links), b_links.keys()

        # NON-OVERLAPPING: agent_alpha's nac.json must NOT carry any of
        # agent_bravo's tool signatures, and vice versa.
        a_tool_sigs = {sig for sig in a_links if sig.startswith("tool:")}
        b_tool_sigs = {sig for sig in b_links if sig.startswith("tool:")}
        assert a_tool_sigs.isdisjoint(b_tool_sigs), (
            "Cross-agent NAc contamination",
            a_tool_sigs,
            b_tool_sigs,
        )
        # Stronger: every signature in agent_alpha's links is from a_tools,
        # and every signature in agent_bravo's is from b_tools.
        assert all("a_tool_" in sig for sig in a_tool_sigs), a_tool_sigs
        assert all("b_tool_" in sig for sig in b_tool_sigs), b_tool_sigs

        # Every persisted link's event_context must be tagged with the
        # owning agent_id — that's the second half of P4.
        for sig, link_dicts in a_links.items():
            for link in link_dicts:
                assert link["event_context"].get("agent_id") == "agent_alpha", (
                    sig,
                    link["event_context"],
                )
        for sig, link_dicts in b_links.items():
            for link in link_dicts:
                assert link["event_context"].get("agent_id") == "agent_bravo", (
                    sig,
                    link["event_context"],
                )


class TestBioIntegrationStashIsolation:
    """The bio_integration substrate-node + pain-intensity stash must
    survive concurrent producers/consumers across two agents without
    cross-contamination."""

    def test_concurrent_substrate_stash_isolation(self, agent_pair: tuple[Any, Any]) -> None:
        del agent_pair  # only used as a fixture-driven stash reset

        # 50 producer iterations per agent — interleaving guarantees
        # the GIL hands control between threads many times mid-test.
        ITERS = 50
        recorded: dict[str, list[tuple[str, ...]]] = {
            "agent_alpha": [],
            "agent_bravo": [],
        }

        def producer(agent_id: str, label: str) -> None:
            for i in range(ITERS):
                nodes = (f"{label}-{i}",)
                record_substrate_nodes(nodes, agent_id=agent_id)
                # Immediately consume what we just produced.  In the
                # pre-fix code another thread could have overwritten
                # the global stash before we got here.
                got = consume_substrate_nodes(agent_id=agent_id)
                recorded[agent_id].append(got)

        ta = threading.Thread(target=producer, args=("agent_alpha", "alpha"))
        tb = threading.Thread(target=producer, args=("agent_bravo", "bravo"))
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        # Every consume returned alpha's own nodes for agent_alpha and
        # bravo's own nodes for agent_bravo — never the other agent's
        # value.
        for got in recorded["agent_alpha"]:
            assert len(got) == 1 and got[0].startswith("alpha-"), got
        for got in recorded["agent_bravo"]:
            assert len(got) == 1 and got[0].startswith("bravo-"), got

    def test_concurrent_pain_stash_isolation(self, agent_pair: tuple[Any, Any]) -> None:
        del agent_pair
        ITERS = 50

        def alpha_producer() -> None:
            for i in range(ITERS):
                # Alpha records intensities in [0.1, 0.5]
                record_pain_intensity(0.1 + (i % 5) * 0.1, agent_id="agent_alpha")

        def bravo_producer() -> None:
            for i in range(ITERS):
                # Bravo records intensities in [0.6, 0.99]
                record_pain_intensity(0.6 + (i % 4) * 0.1, agent_id="agent_bravo")

        ta = threading.Thread(target=alpha_producer)
        tb = threading.Thread(target=bravo_producer)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        # After the producers finish, each agent's max-merged pain is in
        # its own band — never crosses the other agent's range.
        a_pain = consume_pain_intensity(agent_id="agent_alpha")
        b_pain = consume_pain_intensity(agent_id="agent_bravo")
        assert a_pain is not None and 0.1 <= a_pain <= 0.5, a_pain
        assert b_pain is not None and 0.6 <= b_pain <= 0.99, b_pain
        # Second consume returns None — the stash was popped.
        assert consume_pain_intensity(agent_id="agent_alpha") is None
        assert consume_pain_intensity(agent_id="agent_bravo") is None

    def test_concurrent_observe_episode_per_agent_ticks(self, agent_pair: tuple[Any, Any]) -> None:
        """Two agents calling observe_episode in parallel get
        independent monotonically-increasing tick counters — they do
        NOT share a single global counter."""
        del agent_pair

        hippo_a = MagicMock()
        hippo_b = MagicMock()
        ITERS = 30

        def caller(hippo: Any, agent_id: str) -> None:
            for _ in range(ITERS):
                observe_episode(hippocampus=hippo, agent_id=agent_id)

        ta = threading.Thread(target=caller, args=(hippo_a, "agent_alpha"))
        tb = threading.Thread(target=caller, args=(hippo_b, "agent_bravo"))
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        a_ticks = [c.args[0].tick for c in hippo_a.observe_episode_event.call_args_list]
        b_ticks = [c.args[0].tick for c in hippo_b.observe_episode_event.call_args_list]

        # Each agent's ticks form a strictly-increasing sequence 1..N,
        # independent of the other.  Pre-fix the shared global tick
        # counter would interleave: agent_alpha would see ticks like
        # [1, 3, 4, 7, 9, ...] and agent_bravo [2, 5, 6, 8, ...].
        assert sorted(a_ticks) == list(range(1, ITERS + 1)), a_ticks
        assert sorted(b_ticks) == list(range(1, ITERS + 1)), b_ticks
