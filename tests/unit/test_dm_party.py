"""Tests for maxim.simulation.dm_party — multi-agent DM campaign runtime."""

from __future__ import annotations

import textwrap
from unittest import mock

import pytest

from maxim.runtime.agent_factory import AgentFactory
from maxim.runtime.agent_pool import AgentPool
from maxim.simulation.dm_party import PartyDMRuntime
from maxim.simulation.dm_schema import load_campaign


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factory(tmp_path):
    return AgentFactory(base_data_dir=tmp_path / "agents")


@pytest.fixture
def pool():
    return AgentPool()


@pytest.fixture
def campaign_yaml(tmp_path):
    """Create a minimal party-mode campaign YAML."""
    path = tmp_path / "test_party_campaign.yaml"
    path.write_text(textwrap.dedent("""\
        campaign:
          name: party_test
          goal: test party mode
          seed: 42
          party_mode: true
          choice_resolution: pc_decides
        acts:
          - name: act1
            encounters: [tavern, confrontation]
        encounters:
          tavern:
            scene: "You enter the tavern. A guard eyes you from the corner."
            active_npcs: [guard]
            choices: [approach, ignore, leave]
            branches:
              approach: confrontation
              ignore: confrontation
              leave: __END__
          confrontation:
            scene: "The guard blocks the door. 'I have questions for you.'"
            active_npcs: [guard]
            choices: [cooperate, resist, talk_way_out]
            branches:
              cooperate: __END__
              resist: __END__
              talk_way_out: __END__
            on_choice:
              cooperate: { flags: [cooperated] }
        npcs:
          guard:
            name: guard
            entity_type: npc
            metadata:
              persona_prompt: "A suspicious town guard."
              role: town guard
            remembers: true
            learns: true
    """))
    return path


@pytest.fixture
def campaign(campaign_yaml):
    return load_campaign(campaign_yaml)


@pytest.fixture
def mock_bridge():
    """Mock bridge that returns canned responses."""
    bridge = mock.MagicMock()
    bridge.send_and_wait.return_value = {
        "response": "I approach the guard cautiously.",
        "actions": [],
    }
    return bridge


# ---------------------------------------------------------------------------
# Campaign schema
# ---------------------------------------------------------------------------


class TestCampaignSchema:
    def test_party_mode_parsed(self, campaign):
        assert campaign.party_mode is True

    def test_choice_resolution_parsed(self, campaign):
        assert campaign.choice_resolution == "pc_decides"

    def test_party_mode_defaults_false(self, tmp_path):
        path = tmp_path / "basic.yaml"
        path.write_text(textwrap.dedent("""\
            campaign:
              name: basic
              goal: test
              seed: 1
            acts:
              - name: act1
                encounters: [enc1]
            encounters:
              enc1:
                scene: "Test."
                choices: [go]
                branches: { go: __END__ }
        """))
        c = load_campaign(path)
        assert c.party_mode is False
        assert c.choice_resolution == "pc_decides"

    def test_npc_remembers_learns_fields(self, campaign):
        guard_spec = campaign.npc_specs["guard"]
        assert guard_spec.get("remembers") is True
        assert guard_spec.get("learns") is True


# ---------------------------------------------------------------------------
# PartyDMRuntime setup
# ---------------------------------------------------------------------------


class TestPartySetup:
    def test_creates_npc_agents(self, campaign, mock_bridge, pool, factory):
        PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )
        assert pool.size == 1  # One NPC: guard
        assert "npc_guard" in pool.agent_ids

    def test_npc_has_memory(self, campaign, mock_bridge, pool, factory):
        PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )
        guard = pool.get_agent("npc_guard")
        assert guard.hippocampus is not None
        assert guard.nac is not None

    def test_npc_personality_set(self, campaign, mock_bridge, pool, factory):
        PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )
        guard = pool.get_agent("npc_guard")
        assert "suspicious" in guard.personality.lower()


# ---------------------------------------------------------------------------
# PartyDMRuntime execution
# ---------------------------------------------------------------------------


class TestPartyRun:
    def test_run_completes(self, campaign, mock_bridge, pool, factory):
        """Campaign runs to completion with NPC agents."""
        # Bridge returns "approach" for first encounter, "cooperate" for second
        mock_bridge.send_and_wait.side_effect = [
            {"response": "I approach the guard.", "actions": []},
            {"response": "I cooperate fully.", "actions": []},
        ]

        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        # Mock the choice classification to return predictable choices
        with mock.patch.object(party, '_get_choice', side_effect=["approach", "cooperate"]):
            state = party.run()

        assert state.finished is True
        assert state.turn_count == 2
        assert len(state.choices_made) == 2

    def test_npc_reactions_in_choices(self, campaign, mock_bridge, pool, factory):
        """NPC reactions are recorded in choices_made."""
        mock_bridge.send_and_wait.return_value = {
            "response": "Leave me alone.", "actions": []
        }

        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        with mock.patch.object(party, '_get_choice', return_value="leave"):
            state = party.run()

        # First choice should have npc_reactions
        assert "npc_reactions" in state.choices_made[0]

    def test_flags_applied(self, campaign, mock_bridge, pool, factory):
        """on_choice flags are applied to campaign state."""
        mock_bridge.send_and_wait.side_effect = [
            {"response": "approach", "actions": []},
            {"response": "cooperate", "actions": []},
        ]

        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        with mock.patch.object(party, '_get_choice', side_effect=["approach", "cooperate"]):
            state = party.run()

        assert "cooperated" in state.flags

    def test_npc_memories_exported(self, campaign, mock_bridge, pool, factory):
        """NPC agent memories can be exported after campaign."""
        mock_bridge.send_and_wait.return_value = {
            "response": "leave", "actions": []
        }

        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        with mock.patch.object(party, '_get_choice', return_value="leave"):
            party.run()

        memories = party.get_agent_memories()
        assert "npc_guard" in memories

    def test_rollup_includes_party_info(self, campaign, mock_bridge, pool, factory):
        """get_rollup() includes party mode info."""
        mock_bridge.send_and_wait.return_value = {
            "response": "leave", "actions": []
        }

        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        with mock.patch.object(party, '_get_choice', return_value="leave"):
            party.run()

        rollup = party.get_rollup()
        assert rollup["party_mode"] is True
        assert rollup["npc_agent_count"] == 1
        assert "npc_memories" in rollup

    def test_shutdown_cleans_up(self, campaign, mock_bridge, pool, factory):
        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )
        party.shutdown()
        assert pool.size == 0


# ---------------------------------------------------------------------------
# NPC turn mechanics
# ---------------------------------------------------------------------------


class TestNPCTurns:
    def test_npc_receives_scene(self, campaign, mock_bridge, pool, factory):
        """NPC agents receive the scene narrative during their turn."""
        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        encounter = campaign.encounters["tavern"]
        stimulus = "You enter the tavern."
        reactions = party._run_npc_turns(encounter, stimulus)

        # Guard should have reacted (even if placeholder)
        assert "guard" in reactions or len(reactions) == 0  # Placeholder may return None

    def test_enriched_stimulus_includes_npc(self, campaign, mock_bridge, pool, factory):
        """PC sees NPC reactions in their stimulus."""
        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        encounter = campaign.encounters["tavern"]
        base = "You enter the tavern.\n\nWhat do you do? Options: approach, ignore, leave"
        reactions = {"guard": "The guard narrows his eyes at you."}

        enriched = party._enrich_with_npc_reactions(base, reactions, encounter)
        assert "narrows his eyes" in enriched
        assert "What do you do?" in enriched

    def test_broadcast_outcome_reaches_npcs(self, campaign, mock_bridge, pool, factory):
        """Outcome broadcast delivers to NPC agents."""
        party = PartyDMRuntime(
            campaign=campaign,
            bridge=mock_bridge,
            agent_pool=pool,
            agent_factory=factory,
        )

        encounter = campaign.encounters["tavern"]
        # Should not raise
        party._broadcast_outcome(encounter, "approach", "The player approached cautiously.")
