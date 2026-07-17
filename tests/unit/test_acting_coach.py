"""Tests for prompts/acting_coach.py — Acting Coach meta-prompting (B3.1)."""

from __future__ import annotations

from maxim.prompts.acting_coach import (
    ActingCoachConfig,
    compose_acting_coach_section,
    _has_entity_tools,
    _compose_nac_annotations,
    _compose_pain_anticipation,
    _compose_cerebellum_predictions,
    _compose_drive_modulation,
)


class TestComposeDriveModulation:
    """Drive-state guidance is per-sensor and action-neutral (review SF-4 /
    Layer 3b). Regression: a homeostatic breach on an EXTEROCEPTIVE drive (a
    sound-bearing azimuth/centeredness drive) must NOT emit thermal
    'seek warmth/shelter' guidance — the category error the pre-fix hardcoded
    branch produced. Guidance names the specific sensor and leaves the action
    to the LLM."""

    # The exact line shape format_body_state_for_prompt emits.
    _AZIMUTH_BREACH = (
        "=== Body State (pain-relevant) ===\n- body.azimuth: 0.5 (DRIVE: outside comfort band, discomfort 0.40)"
    )
    _TEMP_BREACH = (
        "=== Body State (pain-relevant) ===\n- body.temperature: 0.9C (DRIVE: outside comfort band, discomfort 0.30)"
    )
    _HUNGER = "=== Body State ===\n- body.hunger: 0.8 (DRIVE: deprived, intensity 0.30)"
    _RISING = "=== Body State ===\n- body.fatigue: 0.5 (DRIVE: rising)"
    _COMFORTABLE = "=== Body State ===\n- body.azimuth: 0.02 (DRIVE: comfortable)"

    def test_empty_body_state(self):
        assert _compose_drive_modulation("") == ""

    def test_no_drive_annotations(self):
        assert _compose_drive_modulation("=== Body State ===\n- body.hp: 100/100") == ""

    def test_comfortable_and_satisfied_produce_no_note(self):
        assert _compose_drive_modulation(self._COMFORTABLE) == ""

    def test_azimuth_breach_does_not_prescribe_warmth(self):
        """THE regression guard: exteroceptive breach must not say warmth/shelter."""
        result = _compose_drive_modulation(self._AZIMUTH_BREACH).lower()
        assert result  # something is reported
        assert "warmth" not in result
        assert "shelter" not in result
        assert "temperature" not in result
        # names the specific sensor instead
        assert "body.azimuth" in result

    def test_homeostatic_breach_is_reported_generically(self):
        result = _compose_drive_modulation(self._TEMP_BREACH)
        assert "Body Needs" in result
        assert "body.temperature" in result
        assert "comfortable range" in result

    def test_deprivation_reported_with_sensor(self):
        result = _compose_drive_modulation(self._HUNGER)
        assert "Unmet need" in result
        assert "body.hunger" in result

    def test_rising_reported_with_sensor(self):
        result = _compose_drive_modulation(self._RISING)
        assert "Building" in result
        assert "body.fatigue" in result

    def test_multiple_signals_all_named(self):
        body = self._TEMP_BREACH + "\n- body.hunger: 0.8 (DRIVE: deprived, intensity 0.30)"
        result = _compose_drive_modulation(body)
        assert "body.temperature" in result
        assert "body.hunger" in result

    def test_no_modality_prescription_for_any_sensor(self):
        """Behavioral guard against re-introducing a hardcoded per-drive branch:
        NO breach/deprivation, on ANY sensor name, may emit modality-specific
        action words. The guidance stays action-neutral and sensor-named."""
        banned = ("warmth", "shelter", "food", "water", "rest as", "move away", "temperature or pressure")
        for sensor in ("azimuth", "temperature", "pressure", "hunger", "proximity", "luminance"):
            for descriptor in ("outside comfort band, discomfort 0.40", "deprived, intensity 0.30", "rising"):
                body = f"=== Body State ===\n- body.{sensor}: 0.5 (DRIVE: {descriptor})"
                out = _compose_drive_modulation(body).lower()
                assert f"body.{sensor}" in out, f"{sensor}/{descriptor} not named"
                for word in banned:
                    assert word not in out, f"{sensor}/{descriptor} leaked prescription {word!r}"


class TestActingCoachConfig:
    """ActingCoachConfig defaults and immutability."""

    def test_defaults(self):
        config = ActingCoachConfig()
        assert config.role_values == ("curiosity", "survival")
        assert config.speech_register == "casual"
        assert config.failure_mode == "creative"
        assert config.embodiment_guidance is True
        assert config.exploration_intensity == 0.7

    def test_frozen(self):
        config = ActingCoachConfig()
        try:
            config.speech_register = "formal"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_custom_values(self):
        config = ActingCoachConfig(
            role_values=("duty", "honor"),
            speech_register="formal",
            failure_mode="defensive",
            exploration_intensity=0.3,
        )
        assert config.role_values == ("duty", "honor")
        assert config.exploration_intensity == 0.3


class TestHasEntityTools:
    """Detection of SEM entity affordance tools."""

    def test_empty_tools(self):
        assert _has_entity_tools(None) is False
        assert _has_entity_tools(set()) is False

    def test_only_builtin_tools(self):
        assert _has_entity_tools({"say", "think", "examine", "respond"}) is False

    def test_entity_affordance_tool(self):
        tools = {"say", "think", "rusty_sword_slash"}
        assert _has_entity_tools(tools) is True

    def test_sensor_read_tool(self):
        tools = {"say", "read_rusty_sword_hp"}
        assert _has_entity_tools(tools) is True

    def test_builtin_with_underscore_not_matched(self):
        """Built-in tools with underscores are excluded."""
        assert _has_entity_tools({"memory_recall"}) is False
        assert _has_entity_tools({"internet_search"}) is False
        assert _has_entity_tools({"request_interaction"}) is False


class TestComposeNacAnnotations:
    """NAc valence modulation of coach directives."""

    def test_no_context(self):
        assert _compose_nac_annotations(None, {"tool"}) == []
        assert _compose_nac_annotations([], {"tool"}) == []

    def test_negative_valence_annotation(self):
        causal = [
            {"event": "sword_slash", "outcome": "took damage", "valence": "negative", "confidence": 0.8},
        ]
        tools = {"sword_slash", "say"}
        result = _compose_nac_annotations(causal, tools)
        assert len(result) == 1
        assert "problems before" in result[0]
        assert "high confidence" in result[0]

    def test_positive_valence_annotation(self):
        causal = [
            {"event": "pick_lock", "outcome": "opened door", "valence": "positive", "confidence": 0.9},
        ]
        tools = {"pick_lock"}
        result = _compose_nac_annotations(causal, tools)
        assert len(result) == 1
        assert "reliably succeeded" in result[0]

    def test_low_confidence_filtered(self):
        causal = [
            {"event": "sword_slash", "outcome": "damage", "valence": "negative", "confidence": 0.2},
        ]
        result = _compose_nac_annotations(causal, {"sword_slash"})
        assert result == []

    def test_event_not_in_tools_filtered(self):
        causal = [
            {"event": "unknown_tool", "outcome": "damage", "valence": "negative", "confidence": 0.9},
        ]
        result = _compose_nac_annotations(causal, {"sword_slash"})
        assert result == []


class TestComposePainAnticipation:
    """Pain anticipation extraction from body_state."""

    def test_empty_body_state(self):
        assert _compose_pain_anticipation("") == ""

    def test_no_anticipation(self):
        assert _compose_pain_anticipation("energy: 80%, hp: 100/100") == ""

    def test_anticipated_pain_detected(self):
        result = _compose_pain_anticipation("energy: 50%, anticipated pain in current context (intensity: 0.7)")
        assert "anticipated discomfort" in result

    def test_anxiety_detected(self):
        result = _compose_pain_anticipation("energy: 50%, rising anxiety (0.6)")
        assert "anticipated discomfort" in result


class TestComposeCerebellumPredictions:
    """Cerebellum forward model prediction annotations."""

    def test_no_programs(self):
        assert _compose_cerebellum_predictions(None) == []
        assert _compose_cerebellum_predictions([]) == []

    def test_low_success_with_risks(self):
        programs = [
            {"name": "force_gate", "confidence": 0.8, "success_rate": 0.2, "risks": ["jammed", "broken handle"]},
        ]
        result = _compose_cerebellum_predictions(programs)
        assert len(result) == 1
        assert "low success" in result[0]
        assert "jammed" in result[0]

    def test_high_success(self):
        programs = [
            {"name": "pick_lock", "confidence": 0.9, "success_rate": 0.85, "goal": "open the door"},
        ]
        result = _compose_cerebellum_predictions(programs)
        assert len(result) == 1
        assert "predicted success" in result[0]

    def test_low_confidence_filtered(self):
        programs = [
            {"name": "test", "confidence": 0.1, "success_rate": 0.9},
        ]
        assert _compose_cerebellum_predictions(programs) == []


class TestComposeActingCoachSection:
    """Full section composition with bio-system modulation."""

    def test_basic_section_without_entity_tools(self):
        config = ActingCoachConfig()
        result = compose_acting_coach_section(config)
        assert "Acting Coach" in result
        assert "curiosity, survival" in result
        assert "casual" in result
        # No embodiment guidance without entity tools
        assert "Physical Exploration" not in result

    def test_embodiment_guidance_with_entity_tools(self):
        config = ActingCoachConfig()
        result = compose_acting_coach_section(
            config,
            available_tools={"say", "rusty_sword_slash", "read_rusty_sword_hp"},
        )
        assert "Physical Exploration" in result
        assert "Explore them" in result

    def test_low_exploration_intensity(self):
        config = ActingCoachConfig(exploration_intensity=0.3)
        result = compose_acting_coach_section(
            config,
            available_tools={"rusty_sword_slash"},
        )
        assert "thoughtfully" in result
        assert "Explore them" not in result

    def test_bio_system_modulation_composition_order(self):
        """Bio-system layers add information; none removes the base directive."""
        config = ActingCoachConfig()
        causal = [
            {"event": "sword_slash", "outcome": "took damage", "valence": "negative", "confidence": 0.8},
        ]
        programs = [
            {"name": "pick_lock", "confidence": 0.9, "success_rate": 0.85, "goal": "open door"},
        ]
        result = compose_acting_coach_section(
            config,
            available_tools={"sword_slash", "pick_lock"},
            causal_context=causal,
            body_state="anticipated pain (intensity: 0.7)",
            motor_programs=programs,
        )
        # All layers present
        assert "Physical Exploration" in result
        assert "Learned Experience" in result
        assert "anticipated discomfort" in result
        assert "Predicted Outcomes" in result
        # Base directive NOT removed
        assert "Explore them" in result

    def test_embodiment_guidance_disabled(self):
        config = ActingCoachConfig(embodiment_guidance=False)
        result = compose_acting_coach_section(
            config,
            available_tools={"rusty_sword_slash"},
        )
        assert "Physical Exploration" not in result


class TestBodyStateLayersToggle:
    """Exp 44 arm-B toggle (acting_coach_body_state_ablation.md): Layers 2+4
    gate on ``body_state_layers`` while Layers 1+3 + exploration directives
    stay up — ``embodiment_guidance`` must NOT be the arm-B lever (it would
    remove Layers 1+3 too and break the additive factorial)."""

    _BODY = "energy: 50%, anticipated pain in current context (intensity: 0.7), hunger: 0.9 (deprived)"
    _TOOLS = ["dragon_fire_breath", "sense_tools"]

    def test_default_on_renders_body_layers(self):
        result = compose_acting_coach_section(
            ActingCoachConfig(),
            available_tools=self._TOOLS,
            body_state=self._BODY,
        )
        assert "anticipated discomfort" in result  # Layer 2

    def test_layers_off_removes_2_and_4_keeps_exploration(self):
        result = compose_acting_coach_section(
            ActingCoachConfig(body_state_layers=False),
            available_tools=self._TOOLS,
            body_state=self._BODY,
        )
        assert "anticipated discomfort" not in result  # Layer 2 gone
        assert "Physical Exploration" in result  # directives intact (Layer 1/3 tier)

    def test_env_factory_default(self):
        from maxim.prompts.acting_coach import acting_coach_config_from_env

        cfg = acting_coach_config_from_env()
        assert cfg.body_state_layers is True

    def test_env_factory_disable(self, monkeypatch):
        from maxim.prompts.acting_coach import acting_coach_config_from_env

        for val in ("1", "true", "t", " YES ", "y", "on"):
            monkeypatch.setenv("MAXIM_DISABLE_COACH_BODY_LAYERS", val)
            assert acting_coach_config_from_env().body_state_layers is False
        monkeypatch.setenv("MAXIM_DISABLE_COACH_BODY_LAYERS", "0")
        assert acting_coach_config_from_env().body_state_layers is True
