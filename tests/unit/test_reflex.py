"""Tests for the Percept Reflex System.

Covers:
- ReflexSpec / ReflexResponse / ReflexFiring dataclasses
- ReflexRegistry.evaluate() — keyword matching, cooldown, habituation,
  sensitization, pre-emption suppression, tool dispatch
- YAML loading (humanoid + quadruped archetypes)
- build_reflex_registry() canonical builder
- Integration: BioEnrichmentPipeline.enrich() → reflexes_fired field
- Regression: auto-damage migration (same percepts still produce damage)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maxim.embodiment.reflex import (
    ReflexRegistry,
    ReflexResponse,
    ReflexSpec,
    build_reflex_registry,
    load_archetype_reflexes,
    load_reflex_specs,
)
from maxim.integration.bio_enrichment import (
    BioEnrichmentPipeline,
    CausalPrediction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _attack_reflex() -> ReflexSpec:
    return ReflexSpec(
        name="attack_flinch",
        detect_keywords=("attack", "strikes", "slashes"),
        response=ReflexResponse(tool="damage_component", params={"component": "torso", "source": "reflex_attack"}),
        base_intensity=0.15,
        intensity_scale={"devastating": 0.30, "light": 0.05},
        cooldown_s=1.0,
        suppressible=True,
    )


def _fire_reflex() -> ReflexSpec:
    return ReflexSpec(
        name="fire_burn",
        detect_keywords=("fire", "flame", "burn"),
        response=ReflexResponse(
            tool="damage_component",
            params={"component": "torso", "source": "reflex_fire", "damage_type": "fire"},
        ),
        base_intensity=0.15,
        cooldown_s=1.0,
        suppressible=True,
    )


def _startle_reflex() -> ReflexSpec:
    return ReflexSpec(
        name="startle",
        detect_keywords=("explosion", "roar", "deafening"),
        response=ReflexResponse(
            tool="set_entity_sensor",
            params={"sensor": "awareness", "value": -0.1, "source": "reflex_startle"},
        ),
        base_intensity=0.10,
        cooldown_s=5.0,
        suppressible=True,
    )


def _cold_reflex() -> ReflexSpec:
    return ReflexSpec(
        name="environment_cold",
        detect_keywords=("freezing", "blizzard"),
        response=ReflexResponse(
            tool="set_entity_sensor",
            params={"sensor": "stamina", "value": -0.05, "source": "reflex_cold"},
        ),
        base_intensity=0.05,
        cooldown_s=10.0,
        suppressible=False,  # can't learn to not feel cold
    )


class _Clock:
    """Deterministic clock for test isolation."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


# ---------------------------------------------------------------------------
# ReflexRegistry.evaluate — keyword detection
# ---------------------------------------------------------------------------


class TestKeywordDetection:
    def test_exact_keyword_match(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("The dragon attacks you with its claws")
        assert len(firings) == 1
        assert firings[0].reflex_name == "attack_flinch"

    def test_no_match_returns_empty(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("The dragon looks at you menacingly")
        assert firings == ()

    def test_empty_text_returns_empty(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        assert reg.evaluate("") == ()

    def test_case_insensitive_matching(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("The DRAGON ATTACKS you FIERCELY")
        assert len(firings) == 1

    def test_multiple_reflexes_can_fire(self):
        """When text matches keywords from different reflexes, both fire."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(), _fire_reflex()), clock=clock)
        firings = reg.evaluate("The dragon attacks and breathes fire at you")
        assert len(firings) == 2
        names = {f.reflex_name for f in firings}
        assert names == {"attack_flinch", "fire_burn"}


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    def test_cooldown_prevents_rapid_refiring(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        firings1 = reg.evaluate("dragon attacks")
        assert len(firings1) == 1

        # Same time — should be blocked
        firings2 = reg.evaluate("dragon attacks again")
        assert firings2 == ()

    def test_cooldown_expires_allows_refiring(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        reg.evaluate("dragon attacks")
        clock.advance(1.5)  # cooldown_s=1.0

        firings = reg.evaluate("dragon attacks again")
        assert len(firings) == 1

    def test_different_reflexes_have_independent_cooldowns(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(), _fire_reflex()), clock=clock)

        reg.evaluate("dragon attacks with fire")  # both fire
        clock.advance(0.5)  # within cooldown for both

        # Neither should fire yet
        firings = reg.evaluate("dragon attacks with fire")
        assert firings == ()


# ---------------------------------------------------------------------------
# Intensity scaling
# ---------------------------------------------------------------------------


class TestIntensityScaling:
    def test_base_intensity_when_no_keyword_match(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("the guard attacks you")
        assert firings[0].raw_intensity == 0.15

    def test_devastating_keyword_scales_up(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("a devastating attack strikes you")
        assert firings[0].raw_intensity == 0.30

    def test_light_keyword_scales_down(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("a light attack grazes you")
        assert firings[0].raw_intensity == 0.05

    def test_zero_base_intensity_no_crash(self):
        """A reflex with base_intensity=0 should not crash with ZeroDivisionError."""
        clock = _Clock()
        spec = ReflexSpec(
            name="zero_base",
            detect_keywords=("test",),
            response=ReflexResponse(
                tool="set_entity_sensor",
                params={"sensor": "stamina", "value": -0.1},
            ),
            base_intensity=0.0,
        )
        reg = ReflexRegistry((spec,), clock=clock)
        # Should not raise — the value scaling is guarded
        firings = reg.evaluate("this is a test")
        assert len(firings) == 0  # effective intensity = 0.0 * ... < 0.01 threshold


# ---------------------------------------------------------------------------
# Habituation
# ---------------------------------------------------------------------------


class TestHabituation:
    def test_repeated_exposure_reduces_intensity(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        # First firing — full intensity
        f1 = reg.evaluate("dragon attacks")
        clock.advance(2.0)

        # Second firing — habituation kicks in
        f2 = reg.evaluate("dragon attacks again")
        clock.advance(2.0)

        # Third firing — even more habituation
        f3 = reg.evaluate("dragon attacks once more")

        assert f1[0].effective_intensity > f2[0].effective_intensity > f3[0].effective_intensity

    def test_habituation_factor_decreases(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        f1 = reg.evaluate("dragon attacks")
        assert f1[0].habituation_factor == 1.0  # exposure_count=0

        clock.advance(2.0)
        f2 = reg.evaluate("dragon attacks")
        assert f2[0].habituation_factor < 1.0  # exposure_count=1

    def test_different_context_resets_habituation(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        # Build up habituation in context "cave"
        reg.evaluate("dragon attacks", context_key="cave")
        clock.advance(2.0)
        f_cave = reg.evaluate("dragon attacks", context_key="cave")

        clock.advance(2.0)

        # New context "forest" — fresh habituation
        f_forest = reg.evaluate("dragon attacks", context_key="forest")
        assert f_forest[0].habituation_factor > f_cave[0].habituation_factor

    def test_reset_state_clears_habituation(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        reg.evaluate("dragon attacks")
        clock.advance(2.0)
        f_before = reg.evaluate("dragon attacks")

        reg.reset_state()
        clock.advance(2.0)
        f_after = reg.evaluate("dragon attacks")

        # After reset, habituation factor should be back to 1.0
        assert f_after[0].habituation_factor == 1.0
        assert f_after[0].habituation_factor > f_before[0].habituation_factor


# ---------------------------------------------------------------------------
# Sensitization
# ---------------------------------------------------------------------------


class TestSensitization:
    def test_damaged_component_amplifies_intensity(self):
        clock = _Clock()
        # Component at 50% integrity → sensitization factor > 1
        reg = ReflexRegistry(
            (_attack_reflex(),),
            clock=clock,
            get_component_integrity=lambda _name: 0.5,
        )
        firings = reg.evaluate("dragon attacks")
        assert firings[0].sensitization_factor > 1.0
        assert firings[0].effective_intensity > firings[0].raw_intensity

    def test_full_integrity_no_sensitization(self):
        clock = _Clock()
        reg = ReflexRegistry(
            (_attack_reflex(),),
            clock=clock,
            get_component_integrity=lambda _name: 1.0,
        )
        firings = reg.evaluate("dragon attacks")
        assert firings[0].sensitization_factor == 1.0

    def test_zero_integrity_maximum_sensitization(self):
        clock = _Clock()
        reg = ReflexRegistry(
            (_attack_reflex(),),
            clock=clock,
            get_component_integrity=lambda _name: 0.0,
        )
        firings = reg.evaluate("dragon attacks")
        # sensitization = 1 + 0.5 * (1 - 0) = 1.5
        assert firings[0].sensitization_factor == 1.5


# ---------------------------------------------------------------------------
# Pre-emption suppression
# ---------------------------------------------------------------------------


class TestPreemption:
    def test_negative_prediction_suppresses_reflex(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        predictions = (CausalPrediction(event="attack", outcome="damage", confidence=0.8, valence="negative"),)
        firings = reg.evaluate("dragon attacks", predictions=predictions)
        assert len(firings) == 1
        assert firings[0].preemption_factor == 0.8
        assert firings[0].effective_intensity < firings[0].raw_intensity

    def test_positive_prediction_does_not_suppress(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        predictions = (CausalPrediction(event="attack", outcome="victory", confidence=0.9, valence="positive"),)
        firings = reg.evaluate("dragon attacks", predictions=predictions)
        assert firings[0].preemption_factor == 0.0

    def test_non_suppressible_reflex_ignores_preemption(self):
        clock = _Clock()
        reg = ReflexRegistry((_cold_reflex(),), clock=clock)

        preds = (CausalPrediction(event="freezing", outcome="stamina_loss", confidence=0.9, valence="negative"),)
        firings = reg.evaluate("a freezing blizzard hits", predictions=preds)
        assert firings[0].preemption_factor == 0.0

    def test_full_suppression_prevents_firing(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        predictions = (CausalPrediction(event="attack", outcome="damage", confidence=1.0, valence="negative"),)
        firings = reg.evaluate("dragon attacks", predictions=predictions)
        # effective = 0.15 * 1.0 * 1.0 * (1 - 1.0) = 0.0 → below threshold
        assert firings == ()


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestToolDispatch:
    def test_damage_component_tool_called(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        tool_calls: list[tuple] = []

        def capture(tool_name: str, **params):
            tool_calls.append((tool_name, params))

        reg.evaluate("dragon attacks", execute_tool=capture)
        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "damage_component"
        assert tool_calls[0][1]["component"] == "torso"
        assert "amount" in tool_calls[0][1]

    def test_sensor_tool_called_for_startle(self):
        clock = _Clock()
        reg = ReflexRegistry((_startle_reflex(),), clock=clock)

        tool_calls: list[tuple] = []

        def capture(tool_name: str, **params):
            tool_calls.append((tool_name, params))

        reg.evaluate("a deafening explosion rocks the cave", execute_tool=capture)
        assert len(tool_calls) == 1
        assert tool_calls[0][0] == "set_entity_sensor"
        assert tool_calls[0][1]["sensor"] == "awareness"

    def test_no_execution_without_dispatcher(self):
        """Dry run — evaluate without execute_tool."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        firings = reg.evaluate("dragon attacks")
        assert len(firings) == 1  # firing recorded
        # No tool called (no dispatcher)

    def test_tool_failure_does_not_crash(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        def failing_tool(tool_name: str, **params):
            raise RuntimeError("Tool failed")

        firings = reg.evaluate("dragon attacks", execute_tool=failing_tool)
        assert len(firings) == 1  # failure logged but reflex still recorded

    def test_failed_dispatch_does_not_consume_cooldown(self):
        """When tool dispatch fails, cooldown should NOT be consumed
        so the reflex can retry on the next tick."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        call_count = 0

        def failing_then_ok(tool_name: str, **params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First call fails")

        # First call: dispatch fails
        f1 = reg.evaluate("dragon attacks", execute_tool=failing_then_ok)
        assert len(f1) == 1

        # Second call at same time: should NOT be blocked by cooldown
        # because the first dispatch failed
        f2 = reg.evaluate("dragon attacks again", execute_tool=failing_then_ok)
        assert len(f2) == 1
        assert call_count == 2


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


class TestYAMLLoading:
    def test_load_humanoid_reflexes(self):
        specs = load_archetype_reflexes("humanoid")
        assert len(specs) > 0
        names = {s.name for s in specs}
        assert "attack_flinch" in names
        assert "fire_burn" in names

    def test_load_quadruped_reflexes(self):
        specs = load_archetype_reflexes("quadruped")
        assert len(specs) > 0
        names = {s.name for s in specs}
        assert "wing_fold" in names

    def test_unknown_archetype_returns_empty(self):
        specs = load_archetype_reflexes("nonexistent_archetype")
        assert specs == ()

    def test_malformed_yaml_raises(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_reflexes: {}")
        with pytest.raises(ValueError, match="reflexes"):
            load_reflex_specs(bad)

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_reflex_specs(tmp_path / "ghost.yaml")

    def test_humanoid_attack_keywords_match_old_auto_damage(self):
        """Regression: the humanoid attack_flinch reflex covers the same
        keywords that _detect_attack used to detect.  If this test fails,
        auto-damage migration is incomplete — percepts that used to trigger
        damage no longer do."""
        specs = load_archetype_reflexes("humanoid")
        attack = [s for s in specs if s.name == "attack_flinch"][0]

        # These keywords were in the old _ATTACK_KEYWORDS frozenset
        old_keywords = {
            "attack",
            "attacks",
            "strikes",
            "hits",
            "slashes",
            "bites",
            "claws",
            "stabs",
            "smashes",
            "crushes",
            "wounds",
            "injures",
            "lunges",
            "swipes",
            "charges at",
            "deals damage",
        }
        for kw in old_keywords:
            assert kw in attack.detect_keywords, f"Old keyword '{kw}' missing from attack_flinch reflex"


# ---------------------------------------------------------------------------
# build_reflex_registry
# ---------------------------------------------------------------------------


class TestBuilder:
    def test_builds_from_entity_spec_with_archetype(self):
        spec = MagicMock()
        spec.archetype = None
        spec.component = MagicMock()
        spec.component.archetype = "humanoid"

        reg = build_reflex_registry(entity_spec=spec)
        assert reg is not None
        assert len(reg.reflexes) > 0

    def test_returns_none_for_no_archetype(self):
        spec = MagicMock()
        spec.archetype = None
        spec.component = MagicMock()
        spec.component.archetype = None

        reg = build_reflex_registry(entity_spec=spec)
        assert reg is None


# ---------------------------------------------------------------------------
# BioEnrichmentPipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_reflexes_fired_appears_in_enrichment_result(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        pipeline = BioEnrichmentPipeline(reflex_registry=reg)
        result = pipeline.enrich("The dragon attacks you violently", bypass_gate=True)

        assert result is not None
        assert "attack_flinch" in result.reflexes_fired

    def test_no_reflexes_when_registry_not_set(self):
        pipeline = BioEnrichmentPipeline()
        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)

        assert result is not None
        assert result.reflexes_fired == ()

    def test_reflexes_fired_is_empty_when_no_match(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        pipeline = BioEnrichmentPipeline(reflex_registry=reg)
        result = pipeline.enrich("The dragon looks at you", bypass_gate=True)

        assert result is not None
        assert result.reflexes_fired == ()


# ---------------------------------------------------------------------------
# Regression: auto-damage migration
# ---------------------------------------------------------------------------


class TestAutoDamageRegression:
    """Ensure the same percept text that used to trigger auto-damage in
    SendMessageTool still triggers damage through the reflex system."""

    OLD_AUTO_DAMAGE_TEXTS = [
        "The dragon attacks you with its claws",
        "The guard strikes you across the face",
        "The wolf bites your arm",
        "A fireball engulfs you in flame",
        "The giant smashes you into the wall",
        "An arrow wounds your shoulder",
        "The assassin stabs you in the back",
    ]

    def test_all_old_auto_damage_texts_trigger_reflex(self):
        """Every text that triggered _detect_attack should fire at least one reflex."""
        clock = _Clock()
        specs = load_archetype_reflexes("humanoid")
        assert specs, "Humanoid reflexes must exist"

        reg = ReflexRegistry(specs, clock=clock)

        for i, text in enumerate(self.OLD_AUTO_DAMAGE_TEXTS):
            clock.advance(20.0)  # clear all cooldowns
            firings = reg.evaluate(text)
            assert len(firings) > 0, f"Text '{text}' did not trigger any reflex"

    def test_intensity_keywords_still_scale(self):
        """Old auto-damage had 'devastating' → 0.3 and 'light' → 0.05."""
        clock = _Clock()
        specs = load_archetype_reflexes("humanoid")
        reg = ReflexRegistry(specs, clock=clock)

        # Devastating
        f_dev = reg.evaluate("a devastating attack strikes you")
        assert f_dev[0].raw_intensity >= 0.25  # was 0.30 in old code

        clock.advance(2.0)

        # Light
        f_light = reg.evaluate("a light attack grazes you")
        assert f_light[0].raw_intensity <= 0.10  # was 0.05 in old code
