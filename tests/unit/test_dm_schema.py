"""Tests for DM campaign schema, YAML loading, and validation."""

from __future__ import annotations

import random
import textwrap
from pathlib import Path

import pytest

from maxim.simulation.dm_schema import (
    Act,
    CampaignDef,
    EncounterDef,
    load_campaign,
    roll_dice,
    validate_campaign,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _minimal_campaign(**overrides) -> CampaignDef:
    """Create a minimal valid campaign for testing."""
    encounters = {
        "start": EncounterDef(
            name="start",
            scene="You arrive.",
            choices=["go", "stay"],
            branches={"go": "end_enc", "stay": "__END__"},
        ),
        "end_enc": EncounterDef(
            name="end_enc",
            scene="The end.",
            choices=["leave"],
            branches={"leave": "__END__"},
        ),
    }
    defaults = dict(
        name="test",
        goal="test goal",
        seed=42,
        acts=[Act(name="act1", encounters=["start", "end_enc"])],
        encounters=encounters,
        pc_spec={"name": "hero", "entity_type": "character"},
        npc_specs={},
        object_specs={},
    )
    defaults.update(overrides)
    return CampaignDef(**defaults)


# ── CampaignDef tests ─────────────────────────────────────────────────────────


class TestCampaignDef:
    def test_encounter_order(self) -> None:
        c = _minimal_campaign()
        assert c.encounter_order == ["start", "end_enc"]

    def test_first_encounter(self) -> None:
        c = _minimal_campaign()
        assert c.first_encounter == "start"


# ── YAML Loading tests ────────────────────────────────────────────────────────


class TestYAMLLoading:
    def test_load_heist_campaign(self) -> None:
        path = Path("scenarios/campaigns/heist_v1.yaml")
        if not path.exists():
            pytest.skip("heist_v1.yaml not found")
        campaign = load_campaign(path)
        assert campaign.name == "the_heist"
        assert campaign.seed == 42
        assert len(campaign.acts) == 3
        assert len(campaign.encounters) == 3
        assert "marta" in campaign.npc_specs
        assert "guard_captain" in campaign.npc_specs

    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            campaign:
              name: test
              goal: test
              seed: 1
            acts:
              - name: act1
                encounters: [room]
            encounters:
              room:
                scene: "A room."
                choices: [leave]
                branches:
                  leave: __END__
            player_character:
              name: hero
              entity_type: character
            npcs: {}
        """)
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        campaign = load_campaign(p)
        assert campaign.name == "test"
        assert campaign.seed == 1
        assert len(campaign.encounters) == 1

    def test_case_normalization(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            campaign: {name: test, goal: test, seed: 1}
            acts:
              - name: a
                encounters: [Room]
            encounters:
              Room:
                scene: "A room."
                choices: [Leave]
                branches:
                  Leave: __END__
                dialogue_hints:
                  Haggler: "Nice."
            player_character: {name: h, entity_type: c}
            npcs: {}
        """)
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        campaign = load_campaign(p)
        # Encounter names lowered
        assert "room" in campaign.encounters
        # Branch keys lowered
        assert "leave" in campaign.encounters["room"].branches
        # Dialogue hint keys lowered
        assert "haggler" in campaign.encounters["room"].dialogue_hints


# ── Validation tests ──────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_campaign_passes(self) -> None:
        c = _minimal_campaign()
        errors = validate_campaign(c)
        assert errors == []

    def test_dangling_branch_detected(self) -> None:
        c = _minimal_campaign()
        c.encounters["start"].branches["go"] = "nonexistent"
        errors = validate_campaign(c)
        assert any("nonexistent" in e for e in errors)

    def test_unreachable_encounter_detected(self) -> None:
        c = _minimal_campaign()
        c.encounters["orphan"] = EncounterDef(
            name="orphan",
            scene="Nobody comes here.",
            choices=["wait"],
            branches={"wait": "__END__"},
        )
        errors = validate_campaign(c)
        assert any("orphan" in e and "unreachable" in e for e in errors)

    def test_undefined_npc_detected(self) -> None:
        c = _minimal_campaign()
        c.encounters["start"].active_npcs = ["ghost_npc"]
        errors = validate_campaign(c)
        assert any("ghost_npc" in e for e in errors)

    def test_undefined_object_detected(self) -> None:
        c = _minimal_campaign()
        c.encounters["start"].world_objects = ["magic_door"]
        errors = validate_campaign(c)
        assert any("magic_door" in e for e in errors)

    def test_unknown_on_choice_key_detected(self) -> None:
        c = _minimal_campaign()
        c.encounters["start"].on_choice = {"nonexistent_choice": {"flags": ["x"]}}
        errors = validate_campaign(c)
        assert any("nonexistent_choice" in e for e in errors)

    def test_no_termination_path_detected(self) -> None:
        """Encounter that can only loop without reaching __END__."""
        encounters = {
            "a": EncounterDef(name="a", scene=".", choices=["go"], branches={"go": "b"}),
            "b": EncounterDef(name="b", scene=".", choices=["back"], branches={"back": "a"}),
        }
        c = _minimal_campaign(encounters=encounters, acts=[Act(name="a", encounters=["a", "b"])])
        errors = validate_campaign(c)
        # Both a and b should be flagged as unable to reach __END__
        assert any("cannot reach __END__" in e for e in errors)


# ── Dice tests ────────────────────────────────────────────────────────────────


class TestDice:
    def test_1d20(self) -> None:
        rng = random.Random(42)
        result = roll_dice("1d20", rng)
        assert 1 <= result <= 20

    def test_2d6_plus_3(self) -> None:
        rng = random.Random(42)
        result = roll_dice("2d6+3", rng)
        assert 5 <= result <= 15  # 2-12 + 3

    def test_seeded_determinism(self) -> None:
        r1 = roll_dice("1d20", random.Random(99))
        r2 = roll_dice("1d20", random.Random(99))
        assert r1 == r2

    def test_1d6_minus_1(self) -> None:
        rng = random.Random(42)
        result = roll_dice("1d6-1", rng)
        assert 0 <= result <= 5
