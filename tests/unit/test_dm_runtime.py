"""Unit tests for DMRuntime stimulus composition."""

from __future__ import annotations

from maxim.simulation.dm_runtime import DMRuntime
from maxim.simulation.dm_schema import Act, CampaignDef, EncounterDef


def _make_campaign(npc_specs: dict) -> CampaignDef:
    encounter = EncounterDef(
        name="start",
        scene="You enter the tavern.",
        active_npcs=list(npc_specs.keys()),
        dialogue_hints={"default": "Welcome, traveler."},
    )
    return CampaignDef(
        name="persona_test",
        goal="verify persona injection",
        seed=0,
        acts=[Act(name="act1", encounters=["start"])],
        encounters={"start": encounter},
        pc_spec={},
        npc_specs=npc_specs,
    )


class TestPersonaInjection:
    """Regression tests for F0.3 — NarrativeModulator ghost removal."""

    def test_persona_appears_in_stimulus(self):
        """When an NPC has a persona_prompt, it must reach the composed stimulus."""
        campaign = _make_campaign(
            {
                "guard": {"metadata": {"role": "Guard", "persona_prompt": "stern and loyal"}},
            }
        )
        runtime = DMRuntime(campaign=campaign, bridge=None)

        stimulus = runtime._compose_stimulus(campaign.encounters["start"])

        assert "stern and loyal" in stimulus

    def test_two_npcs_produce_distinct_stimulus_framing(self):
        """Two NPCs with different personas must yield visibly different stimulus text."""
        campaign_a = _make_campaign(
            {
                "guard": {"metadata": {"role": "Guard", "persona_prompt": "stern town guard"}},
            }
        )
        campaign_b = _make_campaign(
            {
                "thief": {"metadata": {"role": "Thief", "persona_prompt": "cunning shadow dweller"}},
            }
        )
        runtime_a = DMRuntime(campaign=campaign_a, bridge=None)
        runtime_b = DMRuntime(campaign=campaign_b, bridge=None)

        stim_a = runtime_a._compose_stimulus(campaign_a.encounters["start"])
        stim_b = runtime_b._compose_stimulus(campaign_b.encounters["start"])

        assert "stern town guard" in stim_a
        assert "cunning shadow dweller" in stim_b
        assert stim_a != stim_b

    def test_missing_persona_is_silent(self):
        """NPCs without a persona_prompt must not add a persona framing line."""
        campaign = _make_campaign(
            {
                "merchant": {"metadata": {"role": "Merchant"}},
            }
        )
        runtime = DMRuntime(campaign=campaign, bridge=None)

        stimulus = runtime._compose_stimulus(campaign.encounters["start"])

        assert "—" not in stimulus.split("merchant (Merchant)")[0]
        assert 'merchant (Merchant): "Welcome, traveler."' in stimulus
