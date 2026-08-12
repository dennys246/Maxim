"""Guards for S1 credit provenance on cluster reward bias.

The gap being closed (annotation_context_and_provenance.md, from the Exp 44b
pilot): the substrate KNOWS why a bias exists — ``tool_dispatch`` branches on
``drive_potential_diff`` to distinguish drive relief from the bare tool-success
floor — and then discards it, so the prompt says "strongly rewarding" whether
the agent's cold went away or a call merely returned True. These pin the
recording layer; rendering is the next commit.
"""

from __future__ import annotations

import pytest

from maxim.decisions.nac import NAc


@pytest.fixture
def nac():
    return NAc()


class TestRecording:
    def test_source_recorded_alongside_bias(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:warm_self", 1.0, source="drive_relief")
        assert nac.get_cluster_reward_sources(agent_id="a1") == {"tool:warm_self": "drive_relief"}
        assert nac.cluster_reward_bias("a1", "c1", "tool:warm_self") > 0.0

    def test_omitting_source_is_backward_compatible(self, nac):
        """Every pre-S1 caller passes no source — bias must still move, and no
        provenance is invented for it."""
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0)
        assert nac.cluster_reward_bias("a1", "c1", "tool:x") > 0.0
        assert nac.get_cluster_reward_sources(agent_id="a1") == {}

    def test_repeat_same_source_stays_put(self, nac):
        for _ in range(5):
            nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="tool_success")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:x"] == "tool_success"

    def test_two_sources_promote_to_mixed_and_stay(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="tool_success")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:x"] == "mixed"
        # promotion is ONE-WAY: a triple credited two ways cannot honestly be
        # narrated as either one again
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:x"] == "mixed"

    def test_unknown_source_is_dropped_not_stored(self, nac):
        """A typo degrades to 'no provenance', never to a new category."""
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive-relief")  # hyphen
        assert nac.get_cluster_reward_sources(agent_id="a1") == {}

    def test_mixed_cannot_be_asserted_directly(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="mixed")
        assert nac.get_cluster_reward_sources(agent_id="a1") == {}

    def test_agent_scoping(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        nac.update_cluster_reward("a2", "c1", "tool:x", 1.0, source="tool_success")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:x"] == "drive_relief"
        assert nac.get_cluster_reward_sources(agent_id="a2")["tool:x"] == "tool_success"

    def test_cross_cluster_disagreement_reports_mixed(self, nac):
        """Agent-wide aggregation mirrors get_agent_tool_biases: a tool credited
        differently in different clusters IS mixed."""
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        nac.update_cluster_reward("a1", "c2", "tool:x", 1.0, source="orient_relief")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:x"] == "mixed"

    def test_empty_agent_id_rejected(self, nac):
        with pytest.raises(ValueError):
            nac.get_cluster_reward_sources(agent_id="")


class TestPersistence:
    def test_round_trip(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:warm", 1.0, source="drive_relief")
        nac.update_cluster_reward("a1", "c1", "tool:turn", 1.0, source="orient_relief")
        restored = NAc()
        restored.load_state(nac.dump())
        assert restored.get_cluster_reward_sources(agent_id="a1") == {
            "tool:warm": "drive_relief",
            "tool:turn": "orient_relief",
        }

    def test_pre_s1_file_loads_without_provenance(self, nac):
        """A nac.json written before S1 has no cluster_reward_source key: it must
        load cleanly and simply carry no 'why' (never fabricate one)."""
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        state = nac.dump()
        state.pop("cluster_reward_source", None)  # simulate the older file
        restored = NAc()
        restored.load_state(state)
        assert restored.cluster_reward_bias("a1", "c1", "tool:x") > 0.0
        assert restored.get_cluster_reward_sources(agent_id="a1") == {}

    def test_corrupt_entries_skipped(self, nac):
        nac.update_cluster_reward("a1", "c1", "tool:x", 1.0, source="drive_relief")
        state = nac.dump()
        state["cluster_reward_source"]["malformed-no-separators"] = "drive_relief"
        state["cluster_reward_source"]["a\x1fb\x1fc"] = "not_a_known_source"
        state["cluster_reward_source"]["a\x1fb\x1fd"] = 12345  # not a str
        restored = NAc()
        restored.load_state(state)
        assert restored.get_cluster_reward_sources(agent_id="a1") == {"tool:x": "drive_relief"}

    def test_tool_signature_with_colons_round_trips(self, nac):
        """build_tool_signature emits colons (tool:use:dodge); the \\x1f encoding
        must survive them — same invariant as cluster_reward_bias."""
        nac.update_cluster_reward("a1", "c1", "tool:use:dodge", 1.0, source="tool_success")
        restored = NAc()
        restored.load_state(nac.dump())
        assert restored.get_cluster_reward_sources(agent_id="a1") == {"tool:use:dodge": "tool_success"}


class TestProducerWiring:
    """The producer must pass the source it ALREADY computes — the branch in
    tool_dispatch distinguishing drive relief from the tool-success floor is the
    only place this information exists.

    SCOPE (honest): this records WHICH MECHANISM deposited a bias. It does NOT
    verify that mechanism attributed correctly — Exp 48's contested finding is
    that operant credit lands on the last action, a coin flip under alternation,
    which would tag cleanly as "operant" here while being uncorrelated with the
    taught behaviour. The read-side counterpart is
    docs/plans/decision_provenance.md (explore_decisive / score components).
    """

    def test_operant_credit_is_tagged(self, nac):
        nac.set_pending_operant_action(agent_id="a1", cluster_id="c1", tool_signature="tool:turn_left")
        assert nac.credit_operant_reward(agent_id="a1", reward=1.0) is not None
        assert nac.get_cluster_reward_sources(agent_id="a1") == {"tool:turn_left": "operant"}

    def test_operant_then_drive_relief_is_mixed(self, nac):
        nac.set_pending_operant_action(agent_id="a1", cluster_id="c1", tool_signature="tool:turn_left")
        nac.credit_operant_reward(agent_id="a1", reward=1.0)
        nac.update_cluster_reward("a1", "c1", "tool:turn_left", 1.0, source="drive_relief")
        assert nac.get_cluster_reward_sources(agent_id="a1")["tool:turn_left"] == "mixed"

    def test_dispatch_branch_labels_match_the_vocabulary(self):
        """Pins the producer's literals against CREDIT_SOURCES: a rename on
        either side fails here rather than silently storing nothing."""
        import re
        from pathlib import Path as _P

        src = _P(__file__).resolve().parents[2] / "src/maxim/runtime/tool_dispatch.py"
        labels = set(re.findall(r'credit_source = "([a-z_]+)"', src.read_text()))
        assert labels, "producer no longer assigns credit_source"
        assert labels <= NAc.CREDIT_SOURCES
        assert "mixed" not in labels  # mixed is derived, never asserted
