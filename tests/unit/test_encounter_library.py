"""Tests for maxim.simulation.encounter_library — DM encounter template catalog."""

from __future__ import annotations

import textwrap

import pytest

from maxim.simulation.encounter_library import (
    EncounterLibrary,
    _read_encounter_header,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def encounters_dir(tmp_path):
    """Create a temporary encounter directory with test YAML files."""
    combat = tmp_path / "combat"
    combat.mkdir()
    social = tmp_path / "social"
    social.mkdir()

    (combat / "ambush.yaml").write_text(textwrap.dedent("""\
        encounter:
          name: ambush
          tags: [combat, outdoor, surprise]
          difficulty_range: [2, 5]
          narrative_role: rising_action
          suggested_npcs: 2

          scene: "Bandits leap from the bushes!"
          choices: [fight, negotiate, flee]
          dice:
            fight: { roll: "1d20", dc: 12 }
    """))

    (combat / "duel.yaml").write_text(textwrap.dedent("""\
        encounter:
          name: duel
          tags: [combat, indoor, formal]
          difficulty_range: [3, 6]
          narrative_role: climax
          suggested_npcs: 1

          scene: "Your opponent draws their blade."
          choices: [attack, feint, defend]
    """))

    (social / "interrogation.yaml").write_text(textwrap.dedent("""\
        encounter:
          name: interrogation
          tags: [social, authority, tense]
          difficulty_range: [2, 4]
          narrative_role: rising_action
          suggested_npcs: 1

          scene: "The guard blocks your path. 'Where were you last night?'"
          choices: [tell_truth, lie, deflect, refuse]
          dice:
            lie: { roll: "1d20", dc: 14 }
    """))

    return tmp_path


@pytest.fixture
def library(encounters_dir):
    """Create a library using ONLY the test encounter directory."""
    return EncounterLibrary(search_paths=[encounters_dir], include_defaults=False)


# ---------------------------------------------------------------------------
# _read_encounter_header
# ---------------------------------------------------------------------------

class TestReadEncounterHeader:
    def test_reads_header(self, tmp_path):
        p = tmp_path / "test.yaml"
        p.write_text("encounter:\n  name: test\n  scene: foo\n  tags: [a]\n")
        header = _read_encounter_header(p)
        assert header is not None
        assert header["name"] == "test"

    def test_returns_none_for_non_encounter(self, tmp_path):
        p = tmp_path / "other.yaml"
        p.write_text("component:\n  name: sword\n")
        assert _read_encounter_header(p) is None

    def test_returns_none_for_empty_encounter(self, tmp_path):
        p = tmp_path / "empty.yaml"
        p.write_text("encounter:\n  tags: [a]\n")
        assert _read_encounter_header(p) is None

    def test_returns_none_for_bad_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(":::broken:::\n")
        assert _read_encounter_header(p) is None


# ---------------------------------------------------------------------------
# Discovery & indexing
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_discovers_encounters(self, library):
        assert library.has("combat/ambush")
        assert library.has("combat/duel")
        assert library.has("social/interrogation")

    def test_list_categories(self, library):
        cats = library.list_categories()
        assert "combat" in cats
        assert "social" in cats

    def test_list_refs(self, library):
        refs = library.list_refs()
        assert "combat/ambush" in refs
        assert "social/interrogation" in refs

    def test_list_refs_filtered(self, library):
        combat = library.list_refs(category="combat")
        assert "combat/ambush" in combat
        assert "social/interrogation" not in combat

    def test_has_false_for_missing(self, library):
        assert not library.has("combat/nonexistent")


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_full_spec(self, library):
        spec = library.get("combat/ambush")
        enc = spec["encounter"]
        assert enc["scene"] == "Bandits leap from the bushes!"
        assert enc["choices"] == ["fight", "negotiate", "flee"]
        assert "fight" in enc["dice"]

    def test_get_returns_deep_copy(self, library):
        s1 = library.get("combat/ambush")
        s2 = library.get("combat/ambush")
        s1["encounter"]["scene"] = "MODIFIED"
        assert s2["encounter"]["scene"] == "Bandits leap from the bushes!"

    def test_get_missing_raises(self, library):
        with pytest.raises(KeyError, match="not found"):
            library.get("combat/nonexistent")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_query_by_category(self, library):
        results = library.query(category="combat")
        assert len(results) == 2
        refs = [r.ref for r in results]
        assert "combat/ambush" in refs
        assert "combat/duel" in refs

    def test_query_by_tags(self, library):
        results = library.query(tags=["surprise"])
        assert len(results) == 1
        assert results[0].ref == "combat/ambush"

    def test_query_by_difficulty(self, library):
        # Difficulty 3 matches ambush [2,5], duel [3,6], interrogation [2,4]
        results = library.query(difficulty=3)
        assert len(results) == 3

        # Difficulty 6 matches only duel [3,6]
        results = library.query(difficulty=6)
        assert len(results) == 1
        assert results[0].ref == "combat/duel"

    def test_query_by_narrative_role(self, library):
        results = library.query(narrative_role="climax")
        assert len(results) == 1
        assert results[0].ref == "combat/duel"

    def test_query_combined_filters(self, library):
        results = library.query(category="combat", tags=["outdoor"])
        assert len(results) == 1
        assert results[0].ref == "combat/ambush"

    def test_query_no_matches(self, library):
        assert library.query(tags=["nonexistent"]) == []

    def test_encounter_info_fields(self, library):
        results = library.query(category="combat", tags=["surprise"])
        info = results[0]
        assert info.name == "ambush"
        assert info.category == "combat"
        assert info.difficulty_range == (2, 5)
        assert info.narrative_role == "rising_action"
        assert info.suggested_npcs == 2
        assert "surprise" in info.tags


# ---------------------------------------------------------------------------
# Template merging in dm_schema.load_campaign
# ---------------------------------------------------------------------------

class TestTemplateMerging:
    def test_template_merges_into_encounter(self, encounters_dir, tmp_path):
        """Campaign encounter with template: key loads template fields."""
        from maxim.simulation.dm_schema import load_campaign

        campaign_yaml = tmp_path / "test_campaign.yaml"
        campaign_yaml.write_text(textwrap.dedent("""\
            campaign:
              name: test
              goal: test encounter templates
              seed: 42
            acts:
              - name: act1
                encounters: [ambush_encounter]
            encounters:
              ambush_encounter:
                template: "combat/ambush"
                active_npcs: [bandit_chief]
                branches:
                  fight: __END__
                  negotiate: __END__
                  flee: __END__
                on_choice:
                  negotiate: { flags: [talked_it_out] }
            npcs:
              bandit_chief:
                name: korrath
                entity_type: npc
        """))

        lib = EncounterLibrary(search_paths=[encounters_dir], include_defaults=False)
        campaign = load_campaign(campaign_yaml, encounter_library=lib)

        enc = campaign.encounters["ambush_encounter"]
        # Template fields should be present
        assert "Bandits leap from the bushes!" in enc.scene
        assert enc.choices == ["fight", "negotiate", "flee"]
        assert "fight" in enc.dice
        # Campaign-specific fields should override/add
        assert enc.active_npcs == ["bandit_chief"]
        assert enc.branches["fight"] == "__END__"
        assert "talked_it_out" in enc.on_choice.get("negotiate", {}).get("flags", [])

    def test_campaign_scene_overrides_template(self, encounters_dir, tmp_path):
        """Campaign can replace template scene text."""
        from maxim.simulation.dm_schema import load_campaign

        campaign_yaml = tmp_path / "override_campaign.yaml"
        campaign_yaml.write_text(textwrap.dedent("""\
            campaign:
              name: test
              goal: test
              seed: 42
            acts:
              - name: act1
                encounters: [custom_ambush]
            encounters:
              custom_ambush:
                template: "combat/ambush"
                scene: "A completely different scene."
                branches:
                  fight: __END__
                  negotiate: __END__
                  flee: __END__
        """))

        lib = EncounterLibrary(search_paths=[encounters_dir], include_defaults=False)
        campaign = load_campaign(campaign_yaml, encounter_library=lib)

        enc = campaign.encounters["custom_ambush"]
        assert enc.scene == "A completely different scene."
        # Template choices should still be present
        assert enc.choices == ["fight", "negotiate", "flee"]

    def test_inline_encounter_unchanged(self, encounters_dir, tmp_path):
        """Encounters without template: key work as before."""
        from maxim.simulation.dm_schema import load_campaign

        campaign_yaml = tmp_path / "inline_campaign.yaml"
        campaign_yaml.write_text(textwrap.dedent("""\
            campaign:
              name: test
              goal: test
              seed: 42
            acts:
              - name: act1
                encounters: [inline_enc]
            encounters:
              inline_enc:
                scene: "Inline scene."
                choices: [go, stay]
                branches:
                  go: __END__
                  stay: __END__
        """))

        campaign = load_campaign(campaign_yaml)
        enc = campaign.encounters["inline_enc"]
        assert enc.scene == "Inline scene."
        assert enc.choices == ["go", "stay"]


# ---------------------------------------------------------------------------
# Bundled encounters
# ---------------------------------------------------------------------------

class TestBundledEncounters:
    def test_bundled_encounters_exist(self):
        """Verify seed encounters shipped in _data/ are discoverable."""
        from maxim.utils.paths import bundled_data
        enc_dir = bundled_data() / "encounters"
        if not enc_dir.is_dir():
            pytest.skip("Bundled encounters not found")

        lib = EncounterLibrary(search_paths=[enc_dir], include_defaults=False)
        assert lib.has("combat/forest_ambush")
        assert lib.has("social/merchant_negotiation")
        assert lib.has("puzzle/riddle_gate")
        assert lib.has("exploration/trapped_corridor")

    def test_bundled_encounter_categories(self):
        from maxim.utils.paths import bundled_data
        enc_dir = bundled_data() / "encounters"
        if not enc_dir.is_dir():
            pytest.skip("Bundled encounters not found")

        lib = EncounterLibrary(search_paths=[enc_dir], include_defaults=False)
        cats = lib.list_categories()
        assert "combat" in cats
        assert "social" in cats
        assert "exploration" in cats
        assert "puzzle" in cats
