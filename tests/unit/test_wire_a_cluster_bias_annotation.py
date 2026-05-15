"""Tests for Wire-A (release_0_9_1.md Stage 2 — cluster-bias annotation).

Three layers:

1. ``NAc.get_agent_tool_biases`` — agent-wide aggregation across all
   clusters (not active-cluster-restricted, per the Roy-2c finding fix).
2. ``compose_cluster_bias_annotation_section`` — bias-band mapping +
   prompt block rendering.
3. End-to-end (light) — PromptBuilder helper picks up
   ``StructuredContext.cluster_bias_annotations`` and produces the
   section.

The Roy-2c regression guard ("a sim where the test percept's active EC
clusters are DISJOINT from the priming clusters MUST still produce the
annotation block") is encoded in the per-NAc test that varies cluster
IDs across entries — the aggregation MUST surface bias regardless of
which clusters are seen at read time.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from maxim.decisions.nac import NAc, NACConfig
from maxim.prompts.cluster_bias_annotation import (
    bias_to_band,
    compose_cluster_bias_annotation_section,
)


# ─────────────────────────────────────────────────────────────────────
# Layer 1: NAc.get_agent_tool_biases
# ─────────────────────────────────────────────────────────────────────


class TestGetAgentToolBiases:
    """Agent-wide aggregation across ALL clusters (no active-cluster filter)."""

    def _fresh_nac(self) -> NAc:
        return NAc(config=NACConfig())

    def test_aggregates_across_clusters_per_agent(self) -> None:
        """The Roy-2pc shape: 6 distinct cluster_ids all carrying bias
        for the same tool. Aggregation must yield ONE entry per tool,
        with the max-|bias| value across the clusters."""
        nac = self._fresh_nac()
        # Simulate 6 cluster_ids accumulating sense_food_source bias.
        for i in range(6):
            nac.update_cluster_reward(
                agent_id="sim_aut",
                cluster_id=f"cluster-{i}",
                tool_signature="tool:sense_food_source",
                reward=10.0,  # cap to +1.0 after alpha
            )
        # One distinct tool: pick_up with a mild bias on a different cluster.
        nac.update_cluster_reward(
            agent_id="sim_aut",
            cluster_id="cluster-x",
            tool_signature="tool:infant_humanoid_pick_up",
            reward=2.0,  # mild positive after alpha
        )
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=5)
        # Exactly 2 distinct tools surface — aggregation collapsed the 6
        # clusters for sense_food_source into a single entry.
        assert len(biases) == 2
        tools = {t for t, _ in biases}
        assert tools == {"tool:sense_food_source", "tool:infant_humanoid_pick_up"}
        food_bias = dict(biases)["tool:sense_food_source"]
        # Sense_food_source hit the cap (the alpha+clamping produces +1.0).
        assert food_bias == pytest.approx(1.0, abs=1e-6)

    def test_disjoint_cluster_regression_guard(self) -> None:
        """The Roy-2c regression: priming cluster IDs are structurally
        disjoint from test-fixture cluster IDs. Wire-A must STILL
        produce the annotation block — that's the whole reason for
        agent-wide aggregation over active-cluster restriction."""
        nac = self._fresh_nac()
        # "Priming" clusters: 3 distinct IDs.
        for cid in ["priming-A", "priming-B", "priming-C"]:
            nac.update_cluster_reward(
                agent_id="sim_aut",
                cluster_id=cid,
                tool_signature="tool:sense_food_source",
                reward=10.0,
            )
        # If we now query without naming any cluster (i.e. the test-
        # fixture's active clusters would be completely different), the
        # bias MUST still surface.
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=5)
        assert len(biases) == 1
        assert biases[0][0] == "tool:sense_food_source"
        assert biases[0][1] > 0.0

    def test_max_abs_bias_wins_per_tool(self) -> None:
        """Per the plan: ``aggregate per-tool by max(|bias|) across all
        clusters``. A strong negative on one cluster should win over a
        mild positive on another cluster for the SAME tool — the sign
        is preserved in the returned value."""
        nac = self._fresh_nac()
        nac.update_cluster_reward(
            agent_id="sim_aut",
            cluster_id="cluster-positive",
            tool_signature="tool:risky_action",
            reward=2.0,  # mild positive after alpha
        )
        nac.update_cluster_reward(
            agent_id="sim_aut",
            cluster_id="cluster-negative",
            tool_signature="tool:risky_action",
            reward=-10.0,  # capped to -1.0
        )
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=5)
        assert len(biases) == 1
        assert biases[0][0] == "tool:risky_action"
        # The -1.0 (cap) wins over +0.2 by |bias|; sign is preserved.
        assert biases[0][1] == pytest.approx(-1.0, abs=1e-6)

    def test_sorted_by_abs_bias_descending(self) -> None:
        """top_n entries returned in order of decreasing |bias|. Tied
        bias values break by tool_signature ascending (stable output
        across runs / test reruns)."""
        nac = self._fresh_nac()
        # Three tools with distinct biases.
        nac.update_cluster_reward("sim_aut", "c1", "tool:weak", reward=1.0)  # +0.1
        nac.update_cluster_reward("sim_aut", "c1", "tool:strong", reward=10.0)  # +1.0
        nac.update_cluster_reward("sim_aut", "c1", "tool:medium", reward=5.0)  # +0.5
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=5)
        # Sorted by |bias| descending.
        names_in_order = [t for t, _ in biases]
        assert names_in_order == ["tool:strong", "tool:medium", "tool:weak"]

    def test_top_n_respected(self) -> None:
        nac = self._fresh_nac()
        for i in range(7):
            nac.update_cluster_reward("sim_aut", f"c{i}", f"tool:tool{i}", reward=float(i + 1))
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=3)
        assert len(biases) == 3

    def test_empty_returns_empty_list(self) -> None:
        """Cold-start agent: no entries in _cluster_reward_bias yet.
        Must return [] without raising."""
        nac = self._fresh_nac()
        biases = nac.get_agent_tool_biases(agent_id="sim_aut", top_n=5)
        assert biases == []

    def test_per_agent_isolation(self) -> None:
        """Two agents on the same NAc instance must not cross-contaminate."""
        nac = self._fresh_nac()
        nac.update_cluster_reward("agent_a", "c1", "tool:foo", reward=10.0)
        nac.update_cluster_reward("agent_b", "c1", "tool:bar", reward=10.0)
        a_biases = nac.get_agent_tool_biases(agent_id="agent_a", top_n=5)
        b_biases = nac.get_agent_tool_biases(agent_id="agent_b", top_n=5)
        a_tools = {t for t, _ in a_biases}
        b_tools = {t for t, _ in b_biases}
        assert a_tools == {"tool:foo"}
        assert b_tools == {"tool:bar"}

    def test_empty_agent_id_rejected(self) -> None:
        """Per CLAUDE.md per-agent stash invariant: empty agent_id is
        a ValueError, not a silent merge."""
        nac = self._fresh_nac()
        with pytest.raises(ValueError, match="non-empty agent_id"):
            nac.get_agent_tool_biases(agent_id="", top_n=5)


# ─────────────────────────────────────────────────────────────────────
# Layer 2: bias-band mapping + section rendering
# ─────────────────────────────────────────────────────────────────────


class TestBiasToBand:
    """Five bands per the plan. Boundary semantics matter — Roy-3's
    annotation strength depends on the band-string the LLM reads."""

    def test_strong_rewarding_at_0_5_inclusive(self) -> None:
        """0.5 is the lower bound of "strongly rewarding" (inclusive)."""
        assert bias_to_band(0.5) == "strongly rewarding"
        assert bias_to_band(0.7) == "strongly rewarding"
        assert bias_to_band(1.0) == "strongly rewarding"

    def test_mild_rewarding_band(self) -> None:
        assert bias_to_band(0.1) == "mildly rewarding"
        assert bias_to_band(0.3) == "mildly rewarding"
        # 0.5 is the boundary — belongs to "strongly rewarding".
        assert bias_to_band(0.49) == "mildly rewarding"

    def test_neutral_band(self) -> None:
        """-0.1 < bias < 0.1 → neutral / mixed."""
        assert bias_to_band(0.0) == "neutral / mixed"
        assert bias_to_band(0.05) == "neutral / mixed"
        assert bias_to_band(-0.05) == "neutral / mixed"
        # 0.1 exactly belongs to "mildly rewarding".
        assert bias_to_band(0.1) == "mildly rewarding"
        # -0.1 exactly belongs to "mildly aversive".
        assert bias_to_band(-0.1) == "mildly aversive"

    def test_mild_aversive_band(self) -> None:
        assert bias_to_band(-0.1) == "mildly aversive"
        assert bias_to_band(-0.3) == "mildly aversive"
        # -0.5 is the boundary — belongs to "strongly aversive".
        assert bias_to_band(-0.49) == "mildly aversive"

    def test_strong_aversive_at_neg_0_5_inclusive(self) -> None:
        """-0.5 is the upper bound of "strongly aversive" (inclusive)."""
        assert bias_to_band(-0.5) == "strongly aversive"
        assert bias_to_band(-0.7) == "strongly aversive"
        assert bias_to_band(-1.0) == "strongly aversive"


class TestComposeSection:
    """End-to-end section rendering on synthetic bias lists."""

    def test_renders_canonical_block(self) -> None:
        biases = [
            ("tool:sense_food_source", 0.9),
            ("tool:infant_humanoid_pick_up", 0.05),
        ]
        text = compose_cluster_bias_annotation_section(biases)
        assert "Substrate associations from prior experience" in text
        assert "sense_food_source" in text
        assert "strongly rewarding from prior experience" in text
        assert "infant_humanoid_pick_up" in text
        # Neutral entries get the band-only suffix (no "from prior experience").
        assert "[neutral / mixed]" in text
        # The `tool:` prefix is stripped in rendered output.
        assert "tool:sense_food_source" not in text

    def test_strips_use_action_prefix_correctly(self) -> None:
        """``tool:use:dodge`` renders as ``use:dodge`` — strip only the
        outer ``tool:`` prefix, NOT the inner ``use:`` action prefix."""
        biases = [("tool:use:dodge", 0.8)]
        text = compose_cluster_bias_annotation_section(biases)
        assert "use:dodge" in text
        # Make sure the `tool:` prefix didn't sneak through.
        assert "tool:use:dodge" not in text

    def test_empty_returns_empty_string(self) -> None:
        assert compose_cluster_bias_annotation_section([]) == ""
        assert compose_cluster_bias_annotation_section(None) == ""

    def test_all_neutral_skipped(self) -> None:
        """All-neutral block adds tokens without signal — skip entirely."""
        biases = [
            ("tool:a", 0.05),
            ("tool:b", -0.02),
            ("tool:c", 0.0),
        ]
        assert compose_cluster_bias_annotation_section(biases) == ""

    def test_mixed_neutral_preserves_block(self) -> None:
        """Neutral entries are retained when at least one non-neutral
        entry is present — the ordering itself is informative."""
        biases = [
            ("tool:strong", 0.9),
            ("tool:neutral", 0.05),
        ]
        text = compose_cluster_bias_annotation_section(biases)
        assert "strong" in text
        assert "neutral" in text


# ─────────────────────────────────────────────────────────────────────
# Layer 3: PromptBuilder section helper
# ─────────────────────────────────────────────────────────────────────


class TestPromptBuilderSectionHelper:
    """The helper reads StructuredContext.cluster_bias_annotations and
    adds an IMPORTANT-priority section to the budgeter. The env-var
    gate is enforced at the AGENT LOOP (producer site), not here — the
    helper just renders whatever the context says."""

    def _make_request_with_biases(self, biases: list[tuple[str, float]] | None):
        request = MagicMock()
        request.context = MagicMock()
        request.context.cluster_bias_annotations = biases
        return request

    def test_helper_skips_when_none(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder

        budgeter = MagicMock()
        request = self._make_request_with_biases(None)
        PromptBuilder._add_cluster_bias_annotation_section(budgeter, request)
        budgeter.add.assert_not_called()

    def test_helper_skips_when_empty_list(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder

        budgeter = MagicMock()
        request = self._make_request_with_biases([])
        PromptBuilder._add_cluster_bias_annotation_section(budgeter, request)
        budgeter.add.assert_not_called()

    def test_helper_adds_section_when_biases_present(self) -> None:
        from maxim.agents.prompt_builder import PromptBuilder, SectionPriority

        budgeter = MagicMock()
        request = self._make_request_with_biases([("tool:sense_food_source", 0.9)])
        PromptBuilder._add_cluster_bias_annotation_section(budgeter, request)
        budgeter.add.assert_called_once()
        args, _ = budgeter.add.call_args
        name, text, priority = args[:3]
        assert name == "cluster_bias_annotations"
        assert "Substrate associations" in text
        assert priority == SectionPriority.IMPORTANT


# ─────────────────────────────────────────────────────────────────────
# Layer 4: env-var gate (producer-side semantics)
# ─────────────────────────────────────────────────────────────────────


class TestEnvVarGateSemantics:
    """The env-var gate at the agent-loop producer reads
    ``MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION``. Truthy values
    (``1``, ``true``, ``yes``, ``on``) disable the read. The conftest
    autouse scrub guarantees this var is unset at test entry; these
    tests verify the truthy-value parsing matches the same semantics
    as the other 0.9.1 env-var gates."""

    def test_default_unset_is_enabled(self) -> None:
        assert "MAXIM_DISABLE_CLUSTER_BIAS_ANNOTATION" not in os.environ

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "y", "t"])
    def test_truthy_values_match(self, value: str) -> None:
        """The producer reads via ``.strip().lower() in ("1", "true", ...)``
        — these values must all disable the read."""
        disabled_set = {"1", "true", "t", "yes", "y", "on"}
        assert value.strip().lower() in disabled_set

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
    def test_falsy_values_do_not_match(self, value: str) -> None:
        disabled_set = {"1", "true", "t", "yes", "y", "on"}
        assert value.strip().lower() not in disabled_set
