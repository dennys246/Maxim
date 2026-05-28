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


@pytest.mark.multi_agent_modes
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


@pytest.mark.multi_agent_modes
class TestSharedNacIsolation:
    """Pre-merge review N1: when two agents share ONE NAc instance,
    causal links must be partitionable by ``event_context["agent_id"]``.

    The headline P4 test (``test_concurrent_record_outcome_isolates_per_agent_nac``)
    runs each agent against its own NAc, so non-overlapping signatures
    are structurally guaranteed by separate NAc instances — not by
    ``agent_id`` itself.  This variant shares one NAc so the only
    thing distinguishing per-agent attribution is the ``agent_id``
    tag in the persisted ``event_context``."""

    @staticmethod
    def _make_pool() -> Any:
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def test_shared_nac_partitions_by_agent_id(self, tmp_path: Path) -> None:
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(config=NACConfig(persistence_path=str(tmp_path / "shared_nac.json")))

        def runner(agent_id: str, tool_prefix: str) -> None:
            for i in range(5):
                record_outcome(
                    agent_id=agent_id,
                    tool_name=f"{tool_prefix}_{i}",
                    success=True,
                    result_summary=f"{agent_id}-ok",
                    error=None,
                    reasoning=f"{agent_id} reasoning",
                    recent_outcomes=[],
                    max_recent=10,
                    llm_worker=None,
                    context_pool=self._make_pool(),
                    nac=nac,
                )

        ta = threading.Thread(target=runner, args=("agent_alpha", "alpha"))
        tb = threading.Thread(target=runner, args=("agent_bravo", "bravo"))
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        nac.save(str(tmp_path / "shared_nac.json"))
        state = json.loads((tmp_path / "shared_nac.json").read_text())

        # Bucket every link by event_context.agent_id and assert
        # tool prefixes match the originating agent.
        alpha_sigs: set[str] = set()
        bravo_sigs: set[str] = set()
        for sig, links in state["links"].items():
            for link in links:
                aid = link["event_context"].get("agent_id")
                if aid == "agent_alpha":
                    alpha_sigs.add(sig)
                elif aid == "agent_bravo":
                    bravo_sigs.add(sig)

        assert all("alpha_" in s for s in alpha_sigs), alpha_sigs
        assert all("bravo_" in s for s in bravo_sigs), bravo_sigs
        assert alpha_sigs.isdisjoint(bravo_sigs)


@pytest.mark.multi_agent_modes
class TestEmptyAgentIdRejected:
    """Pre-merge review C2: empty-string agent_id is the same band-aid
    pattern P4 was supposed to eliminate.  Reject it loudly."""

    @staticmethod
    def _make_pool() -> Any:
        pool = MagicMock()
        pool.add_outcome = MagicMock()
        return pool

    def test_record_outcome_rejects_empty_agent_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            record_outcome(
                agent_id="",
                tool_name="t",
                success=True,
                result_summary="ok",
                error=None,
                reasoning="",
                recent_outcomes=[],
                max_recent=10,
                llm_worker=None,
                context_pool=self._make_pool(),
            )

    def test_bio_integration_rejects_empty_agent_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            record_substrate_nodes(("n1",), agent_id="")
        with pytest.raises(ValueError, match="non-empty"):
            consume_substrate_nodes(agent_id="")
        with pytest.raises(ValueError, match="non-empty"):
            record_pain_intensity(0.5, agent_id="")
        with pytest.raises(ValueError, match="non-empty"):
            consume_pain_intensity(agent_id="")
        with pytest.raises(ValueError, match="non-empty"):
            observe_episode(hippocampus=MagicMock(), agent_id="")


@pytest.mark.multi_agent_modes
class TestBackwardCompatibilityNacJson:
    """Pre-merge review I4: pre-P4 nac.json has no ``agent_id`` in
    event_context.  Loading old data must not crash; missing values
    deserialize as untagged links."""

    def test_pre_p4_nac_json_loads_without_crash(self, tmp_path: Path) -> None:
        from maxim.decisions.causal_link import CausalLink

        # Hand-craft a v0 link dict (no agent_id in event_context)
        legacy_dict = {
            "id": "legacy_link_1",
            "event_type": "tool",
            "event_signature": "tool:legacy",
            "event_context": {"goal": "do the thing"},  # no agent_id
            "outcome_type": "tool_result",
            "outcome_signature": "success:done",
            "outcome_valence": "positive",
            "temporal_delta": {"mean": 0.5, "variance": 0.1, "count": 1, "samples": [0.5]},
            "predicted_value": 0.5,
            "prediction_history": [],
            "observation_count": 1,
            "confidence": 0.5,
            "last_observed": 1700000000.0,
            "memory_ids": [],
            "context_factors": {},
            "last_rpe": None,
            "percept_refs": [],
            "imagined": False,
        }
        link = CausalLink.from_dict(legacy_dict)
        assert link.event_context == {"goal": "do the thing"}
        assert link.event_context.get("agent_id") is None  # gracefully None


@pytest.mark.multi_agent_modes
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


@pytest.mark.multi_agent_modes
class TestCreateFullAgentBioStackAgentIdPropagation:
    """Pin ``config.agent_id`` propagation through
    ``AgentFactory.create_full_agent`` into the bio-stack's MemoryHub.

    Regression for the silent ``default_agent`` fallback surfaced by
    experiment 32 (Roy-3a post-W1+W2 integration test, Bug A).
    Pre-fix, ``create_full_agent`` called ``build_bio_stack(...)``
    without threading ``config.agent_id``, so the MemoryHub built
    inside the bio-stack carried the builder default
    ``agent_id="default_agent"`` regardless of what the AgentConfig
    asked for.  Wire-A's read path
    (``agent_loop.py::_loop_agent_id = memory_hub.agent_id``) then
    diverged from every other agent_id surface (sim_agent_context,
    request_context, executor) — silently rendering empty
    cluster_bias annotations.

    See docs/experiments/32_wire_a_post_w1_w2.md.
    """

    def test_memory_hub_agent_id_matches_config_agent_id(self, tmp_path: Path) -> None:
        factory = AgentFactory(base_data_dir=tmp_path)
        instance = factory.create_full_agent(
            AgentConfig(
                agent_id="probe_agent_xyz",
                role="pc",
                persistence_dir=str(tmp_path / "probe"),
                with_bio_stack=True,
                with_executor=False,
                with_pain_bridge=False,
                with_fear_gate=False,
            )
        )
        assert instance.memory_hub is not None
        assert instance.memory_hub.agent_id == "probe_agent_xyz"

    def test_two_full_agents_have_disjoint_memory_hub_agent_ids(self, tmp_path: Path) -> None:
        factory = AgentFactory(base_data_dir=tmp_path)
        a = factory.create_full_agent(
            AgentConfig(
                agent_id="alpha",
                role="pc",
                persistence_dir=str(tmp_path / "alpha"),
                with_bio_stack=True,
                with_executor=False,
                with_pain_bridge=False,
                with_fear_gate=False,
            )
        )
        b = factory.create_full_agent(
            AgentConfig(
                agent_id="bravo",
                role="pc",
                persistence_dir=str(tmp_path / "bravo"),
                with_bio_stack=True,
                with_executor=False,
                with_pain_bridge=False,
                with_fear_gate=False,
            )
        )
        assert a.memory_hub is not None and b.memory_hub is not None
        assert a.memory_hub.agent_id == "alpha"
        assert b.memory_hub.agent_id == "bravo"
        # Two factory-built agents with distinct configs must not share
        # the MemoryHub identity (the pre-fix bug collapsed both to
        # "default_agent").
        assert a.memory_hub.agent_id != b.memory_hub.agent_id

    def test_nac_cluster_reward_bias_keys_use_config_agent_id(self, tmp_path: Path) -> None:
        """The downstream symptom of Bug A: ``NAc._cluster_reward_bias``
        keys carry the loop-time ``agent_id``.  After Fix A, writing
        through ``update_cluster_reward_bias`` with the config's
        agent_id must land on a key prefixed by that same agent_id —
        not by ``default_agent``.
        """
        factory = AgentFactory(base_data_dir=tmp_path)
        instance = factory.create_full_agent(
            AgentConfig(
                agent_id="downstream_probe",
                role="pc",
                persistence_dir=str(tmp_path / "ds"),
                with_bio_stack=True,
                with_executor=False,
                with_pain_bridge=False,
                with_fear_gate=False,
            )
        )
        assert instance.nac is not None
        assert instance.memory_hub is not None
        # Resolve agent_id the way agent_loop.py:1074 does.
        loop_agent_id = instance.memory_hub.agent_id
        assert loop_agent_id == "downstream_probe"
        # Write a small bias using the loop's agent_id.
        instance.nac.update_cluster_reward(
            agent_id=loop_agent_id,
            cluster_id="test_cluster",
            tool_signature="tool:probe",
            reward=0.5,
        )
        # Read back: agent_id="downstream_probe" must find the key;
        # "default_agent" (the pre-fix accidental key) must not.
        # Expected magnitude is the closed-form Welford update:
        #   reward_bias_alpha (NACConfig default 0.15) × reward (0.5) = 0.075.
        # Reading from the same private dict the bias was written to would
        # be a self-tautology — the pre-merge executor-lens review caught
        # this and asked for the closed-form expected value instead.
        biases = instance.nac.get_agent_tool_biases(agent_id="downstream_probe", top_n=5)
        assert biases == [("tool:probe", pytest.approx(0.15 * 0.5))]
        # Pre-fix this read would have succeeded (because the loop
        # wrote under default_agent).  Post-fix, default_agent has no
        # entry for downstream_probe's bias.
        assert instance.nac.get_agent_tool_biases(agent_id="default_agent", top_n=5) == []
