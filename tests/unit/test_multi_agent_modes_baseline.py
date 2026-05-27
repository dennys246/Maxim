"""Baseline tests for the ``multi_agent_modes`` fixture.

Demonstrates the marker + fixture pattern works correctly across all
three modes (single_agent / shared_instance / isolated_instances).
Serves as the canonical example for new tests using the fixture.

See ``tests/multi_agent_fixtures.py`` for the load-bearing TRIPWIRE
warning on ``two_agents_shared_instance`` and the rationale for the
three-mode design.

The fixture is itself **structurally** the way to enforce CLAUDE.md L43
on new code: tests written against ``multi_agent_modes`` cannot
accidentally regress to single-agent assumptions because the fixture
ALWAYS hands them at least one agent_id and may hand them two on the
same underlying instance.
"""

from __future__ import annotations

import pytest

from maxim.decisions.causal_link import Valence


@pytest.mark.multi_agent_modes
class TestMultiAgentModesBaseline:
    """Validate the fixture supplies the expected shape per mode."""

    def test_each_mode_supplies_at_least_one_agent_id(self, multi_agent_modes) -> None:
        assert len(multi_agent_modes.agent_ids) >= 1

    def test_two_agent_modes_supply_distinct_ids(self, multi_agent_modes) -> None:
        if multi_agent_modes.mode == "single_agent":
            pytest.skip("single_agent mode by definition has only one id")
        ids = multi_agent_modes.agent_ids
        assert len(ids) == 2
        assert ids[0] != ids[1]

    def test_shared_mode_returns_same_nac_instance_for_both_agents(self, multi_agent_modes) -> None:
        if not multi_agent_modes.is_shared_instance_tripwire:
            pytest.skip("only the shared-instance tripwire shares NAc objects")
        a, b = multi_agent_modes.agent_ids
        assert multi_agent_modes.nac_for(a) is multi_agent_modes.nac_for(b), (
            "shared-instance tripwire mode must back both agents with the "
            "same NAc object — otherwise the silent-merge surface is invisible"
        )

    def test_isolated_mode_returns_distinct_nac_instances(self, multi_agent_modes) -> None:
        if multi_agent_modes.mode != "two_agents_isolated_instances":
            pytest.skip("only isolated-instances mode tests separate-object isolation")
        a, b = multi_agent_modes.agent_ids
        assert multi_agent_modes.nac_for(a) is not multi_agent_modes.nac_for(b), (
            "isolated-instances mode must back each agent with its own NAc"
        )


@pytest.mark.multi_agent_modes
class TestMultiAgentAttributionAcrossModes:
    """The actual P4 invariant — per-agent state stays separated under
    all three modes.

    Drives some reward credit through each agent and asserts the
    ``(agent_id, ...)`` keys never overlap across agents. This is the
    structural invariant the fixture is designed to enforce.
    """

    def test_credit_does_not_leak_across_agents(self, multi_agent_modes) -> None:
        for agent_id in multi_agent_modes.agent_ids:
            nac = multi_agent_modes.nac_for(agent_id)
            # Credit ONE node per agent — distinct node ids per agent so
            # we can spot cross-agent leakage clearly.
            nac.credit_node(agent_id, f"node_{agent_id}", reward=0.5)

        # Each agent should ONLY appear in the (agent_id, node_id) keys
        # for its own node. In shared-instance mode, both agents write
        # to the same NAc — but the agent_id key partitions them.
        for agent_id in multi_agent_modes.agent_ids:
            nac = multi_agent_modes.nac_for(agent_id)
            keys = {k for k in nac._reward_bias if k[0] == agent_id}
            assert keys == {(agent_id, f"node_{agent_id}")}, (
                f"[{multi_agent_modes.mode}] agent {agent_id} keys leaked: {keys}"
            )

    def test_record_outcome_partitions_welford_by_agent_id(self, multi_agent_modes) -> None:
        for agent_id in multi_agent_modes.agent_ids:
            nac = multi_agent_modes.nac_for(agent_id)
            nac.record_event("tool", "tool:t", context={"agent_id": agent_id})
            nac.record_outcome("tool", "tool:t", Valence.POSITIVE, context={"agent_id": agent_id})

        for agent_id in multi_agent_modes.agent_ids:
            nac = multi_agent_modes.nac_for(agent_id)
            # The Welford state for this agent must be present and
            # contain exactly one observation (agent_id partitions).
            state = nac._event_outcome_welford[(agent_id, "tool:t")]
            assert state["n"] == 1.0, (
                f"[{multi_agent_modes.mode}] agent {agent_id} welford observation count = {state['n']}, expected 1"
            )
