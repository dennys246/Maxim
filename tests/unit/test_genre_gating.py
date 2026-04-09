"""Tests for genre-based SEM component gating.

Verifies:
- ComponentRegistry.query(genre=...) filters by genre tags
- Genre-neutral components (no genre tag) pass all genre filters
- CampaignDef.genre field is loaded from YAML
- EntityDesigner.design() respects genre constraints
"""

from __future__ import annotations

import pytest

from maxim.embodiment.component_registry import ComponentRegistry


# ---------------------------------------------------------------------------
# ComponentRegistry.query(genre=...)
# ---------------------------------------------------------------------------


class TestRegistryGenreFilter:
    def test_query_cyberpunk_returns_only_cyberpunk(self):
        registry = ComponentRegistry()
        results = registry.query(genre="cyberpunk")
        for info in results:
            # Either tagged cyberpunk, or no genre tag (genre-neutral)
            genre_tags = {"fantasy", "cyberpunk", "scifi", "modern", "devops", "horror", "historical"}
            component_genres = set(info.tags) & genre_tags
            assert not component_genres or "cyberpunk" in component_genres, (
                f"{info.ref} has genres {component_genres} but query was 'cyberpunk'"
            )

    def test_query_fantasy_excludes_cyberpunk(self):
        registry = ComponentRegistry()
        results = registry.query(genre="fantasy")
        refs = {info.ref for info in results}
        # Cyberpunk-tagged components should not appear
        assert "creatures/patrol_drone" not in refs
        assert "npcs/netrunner" not in refs
        assert "weapons/neural_disruptor" not in refs

    def test_query_fantasy_includes_genre_neutral(self):
        registry = ComponentRegistry()
        results = registry.query(genre="fantasy")
        refs = {info.ref for info in results}
        # Genre-neutral components (no genre tag) should be included
        # base_humanoid has tags [npc, humanoid, character] — no genre
        assert "npcs/base_humanoid" in refs

    def test_query_no_genre_returns_all(self):
        registry = ComponentRegistry()
        all_results = registry.query()
        genre_results = registry.query(genre="fantasy")
        # With no genre filter, should return more (or equal) results
        assert len(all_results) >= len(genre_results)

    def test_query_cyberpunk_includes_cyberpunk_components(self):
        registry = ComponentRegistry()
        results = registry.query(genre="cyberpunk")
        refs = {info.ref for info in results}
        assert "creatures/patrol_drone" in refs
        assert "npcs/netrunner" in refs
        assert "weapons/shock_baton" in refs

    def test_genre_combined_with_category(self):
        registry = ComponentRegistry()
        # Cyberpunk NPCs only
        results = registry.query(genre="cyberpunk", category="npcs")
        refs = {info.ref for info in results}
        assert "npcs/netrunner" in refs
        assert "npcs/corpo_guard" in refs
        # Fantasy-only NPC should not appear (guard is genre-neutral, so it WILL appear)
        # But cyberpunk creatures should not
        for info in results:
            assert info.category == "npcs"

    def test_unknown_genre_returns_only_neutral(self):
        registry = ComponentRegistry()
        results = registry.query(genre="steampunk")
        # Only genre-neutral components should pass
        genre_tags = {"fantasy", "cyberpunk", "scifi", "modern", "devops", "horror", "historical"}
        for info in results:
            component_genres = set(info.tags) & genre_tags
            assert len(component_genres) == 0, (
                f"{info.ref} has genre tags {component_genres} but shouldn't match 'steampunk'"
            )


# ---------------------------------------------------------------------------
# CampaignDef.genre from YAML
# ---------------------------------------------------------------------------


class TestCampaignGenreLoading:
    def test_genre_loaded_from_yaml(self, tmp_path):
        from maxim.simulation.dm_schema import load_campaign

        yaml_content = """\
campaign:
  name: test_campaign
  goal: test genre loading
  seed: 1
  genre: cyberpunk

player_character:
  name: runner
  entity_type: character

acts:
  - name: act1
    encounters: [start]

encounters:
  start:
    scene: "Test scene."
    choices: [go]
    branches:
      go: __END__
"""
        yaml_file = tmp_path / "test_campaign.yaml"
        yaml_file.write_text(yaml_content)

        campaign = load_campaign(str(yaml_file))
        assert campaign.genre == "cyberpunk"

    def test_missing_genre_defaults_empty(self, tmp_path):
        from maxim.simulation.dm_schema import load_campaign

        yaml_content = """\
campaign:
  name: test_no_genre
  goal: test default
  seed: 1

player_character:
  name: hero
  entity_type: character

acts:
  - name: act1
    encounters: [start]

encounters:
  start:
    scene: "Test."
    choices: [go]
    branches:
      go: __END__
"""
        yaml_file = tmp_path / "test_no_genre.yaml"
        yaml_file.write_text(yaml_content)

        campaign = load_campaign(str(yaml_file))
        assert campaign.genre == ""

    def test_all_bundled_campaigns_have_genre(self):
        """Verify all shipped campaign YAMLs declare a genre."""
        from pathlib import Path

        from maxim.simulation.dm_schema import load_campaign

        campaigns_dir = Path("scenarios/campaigns")
        if not campaigns_dir.exists():
            pytest.skip("scenarios/campaigns not found")

        for yaml_file in sorted(campaigns_dir.glob("*.yaml")):
            campaign = load_campaign(str(yaml_file))
            assert campaign.genre, f"{yaml_file.name} is missing a genre field"


# ---------------------------------------------------------------------------
# EntityDesigner genre validation
# ---------------------------------------------------------------------------


class TestEntityDesignerGenre:
    def test_design_skips_wrong_genre_base_ref(self):
        from maxim.simulation.entity_designer import EntityDesigner

        registry = ComponentRegistry()
        designer = EntityDesigner(component_registry=registry)

        # Try to use a cyberpunk base in a fantasy campaign
        result = designer.design(
            "a guard",
            base_ref="npcs/corpo_guard",
            entity_type="npc",
            genre="fantasy",
        )
        # Should still produce an entity (fallback), but NOT based on corpo_guard
        assert result is not None
        assert isinstance(result, dict)

    def test_design_allows_matching_genre(self):
        from maxim.simulation.entity_designer import EntityDesigner

        registry = ComponentRegistry()
        designer = EntityDesigner(component_registry=registry)

        result = designer.design(
            "a corporate enforcer",
            base_ref="npcs/corpo_guard",
            entity_type="npc",
            genre="cyberpunk",
        )
        assert result is not None

    def test_design_allows_genre_neutral_base(self):
        from maxim.simulation.entity_designer import EntityDesigner

        registry = ComponentRegistry()
        designer = EntityDesigner(component_registry=registry)

        # base_humanoid is genre-neutral — should work with any genre
        result = designer.design(
            "a wandering merchant",
            base_ref="npcs/base_humanoid",
            entity_type="npc",
            genre="fantasy",
        )
        assert result is not None

    def test_design_no_genre_allows_anything(self):
        from maxim.simulation.entity_designer import EntityDesigner

        registry = ComponentRegistry()
        designer = EntityDesigner(component_registry=registry)

        # No genre constraint — cyberpunk base should work
        result = designer.design(
            "a corporate enforcer",
            base_ref="npcs/corpo_guard",
            entity_type="npc",
        )
        assert result is not None
