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
    ReflexFiring,
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
            params={"sensor": "awareness", "delta": -0.1, "source": "reflex_startle"},
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
            params={"sensor": "stamina", "delta": -0.05, "source": "reflex_cold"},
        ),
        base_intensity=0.05,
        cooldown_s=10.0,
        suppressible=False,  # can't learn to not feel cold
    )


def _ok(tool_name: str = "", **params):
    """A dispatcher that succeeds (the contract: return a ToolOutput)."""
    from maxim.tools.base import ToolOutput

    return ToolOutput(success=True, output={})


class _Clock:
    """Deterministic clock for test isolation."""

    def __init__(self, start: float = 0.0):
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


class _OkTool:
    """A wired reflex tool that succeeds, so a pipeline's reflexes ACT."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, **params):
        from maxim.tools.base import ToolOutput

        self.calls.append(params)
        return ToolOutput(success=True, output={})


def _wired(pipeline: BioEnrichmentPipeline) -> BioEnrichmentPipeline:
    """Wire both reflex tools, as the orchestrator does in production."""
    pipeline._reflex_damage_tool = _OkTool()
    pipeline._reflex_sensor_tool = _OkTool()
    return pipeline


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
                params={"sensor": "stamina", "delta": -0.1},
            ),
            base_intensity=0.0,
        )
        reg = ReflexRegistry((spec,), clock=clock)
        # Should not raise — the value scaling is guarded
        firings = reg.evaluate("this is a test")
        # effective intensity = 0.0 * ... < 0.01 threshold: recorded as suppressed
        assert [f.outcome for f in firings] == ["suppressed"]


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
        calls: list[str] = []
        firings = reg.evaluate("dragon attacks", predictions=predictions, execute_tool=lambda t, **_: calls.append(t))
        # effective = 0.15 * 1.0 * 1.0 * (1 - 1.0) = 0.0 → below threshold.
        # The fully ANTICIPATED case is recorded, not dropped, and nothing runs.
        assert [(f.outcome, f.preemption_factor) for f in firings] == [("suppressed", 1.0)]
        assert calls == []


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
            return _ok()

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
            return _ok()

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
        assert [f.outcome for f in firings] == ["failed"]  # recorded, as failed

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
            return _ok()

        # First call: dispatch fails
        f1 = reg.evaluate("dragon attacks", execute_tool=failing_then_ok)
        assert len(f1) == 1

        # Second call at same time: should NOT be blocked by cooldown
        # because the first dispatch failed
        f2 = reg.evaluate("dragon attacks again", execute_tool=failing_then_ok)
        assert len(f2) == 1
        assert call_count == 2


class TestDispatchOutcome:
    """A firing records whether the body actually responded.

    ``evaluate`` used to read only whether the dispatcher RAISED: a tool that
    RETURNED ``success=False`` (or a pipeline whose reflex tools were never
    wired) counted as the body having responded — it consumed cooldown and
    habituation, logged a ``sim_reflex``, and surfaced dodge/block/brace to
    the agent for a response that never happened.
    """

    def _failing_output(self, tool_name: str, **params):
        from maxim.tools.base import ToolOutput

        return ToolOutput(success=False, error="No embodiment configured")

    def test_a_returned_failure_is_failed_not_acted(self, caplog):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        with caplog.at_level("WARNING"):
            [f] = reg.evaluate("dragon attacks", execute_tool=self._failing_output)
        assert (f.outcome, f.acted, f.error) == ("failed", False, "No embodiment configured")
        assert any("did not respond" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")

    def test_a_returned_failure_consumes_neither_cooldown_nor_habituation(self):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        reg.evaluate("dragon attacks", execute_tool=self._failing_output)
        # Same instant: a consumed cooldown would block this; unconsumed
        # habituation leaves the factor at 1.0.
        [again] = reg.evaluate("dragon attacks", execute_tool=_ok)
        assert again.outcome == "acted"
        assert again.habituation_factor == 1.0

    def test_a_returned_failure_warns_once_then_debug(self, caplog):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        with caplog.at_level("DEBUG"):
            for _ in range(3):
                reg.evaluate("dragon attacks", execute_tool=self._failing_output)
        warnings = [r for r in caplog.records if "did not respond" in r.getMessage() and r.levelname == "WARNING"]
        assert len(warnings) == 1

    def test_a_raised_failure_is_failed_and_reported(self, caplog):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())

        def _boom(tool_name, **params):
            raise RuntimeError("Tool failed")

        with caplog.at_level("WARNING"):
            [f] = reg.evaluate("dragon attacks", execute_tool=_boom)
        assert f.outcome == "failed" and "RuntimeError" in f.error
        assert any(r.levelname == "WARNING" and "swallowed" in r.getMessage().lower() for r in caplog.records)

    def test_success_is_acted_and_consumes_cooldown(self):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        [f] = reg.evaluate("dragon attacks", execute_tool=_ok)
        assert f.outcome == "acted" and f.acted
        assert reg.evaluate("dragon attacks", execute_tool=_ok) == ()  # in cooldown

    def test_outcome_is_required_on_construction(self):
        with pytest.raises(TypeError):
            ReflexFiring(
                reflex_name="x",
                tool="damage_component",
                params={},
                effective_intensity=0.1,
                raw_intensity=0.1,
                habituation_factor=1.0,
                sensitization_factor=1.0,
                preemption_factor=0.0,
            )

    def test_an_unwired_pipeline_reports_its_reflex_as_failed_not_fired(self, caplog):
        """A pipeline whose reflex tools were never wired used to report the
        reflex as fired and surface latent motor programs."""
        pipeline = BioEnrichmentPipeline(reflex_registry=ReflexRegistry((_attack_reflex(),), clock=_Clock()))
        # A body WITH latent programs, so "none surfaced" is a real assertion.
        pipeline._entity_root = TestLatentAffordances()._make_entity_with_latent()
        latent: list[str] = []
        with caplog.at_level("WARNING"):
            names = pipeline._evaluate_reflexes("The dragon attacks you", (), latent_out=latent)
        assert names == ()
        assert latent == []
        assert any(r.levelname == "WARNING" and "swallowed" in r.getMessage().lower() for r in caplog.records)

    def test_a_wired_pipeline_reports_the_reflex_that_acted(self):
        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=ReflexRegistry((_attack_reflex(),), clock=_Clock())))
        pipeline._entity_root = TestLatentAffordances()._make_entity_with_latent()
        latent: list[str] = []
        assert pipeline._evaluate_reflexes("The dragon attacks you", (), latent_out=latent) == ("attack_flinch",)
        assert len(pipeline._reflex_damage_tool.calls) == 1
        assert latent  # the control: a body that responded DOES surface them


class TestDispatchContract:
    """The dispatcher contract is typed: only a successful ToolOutput is a response."""

    @pytest.mark.parametrize("returned", [None, {"success": True}, "ok"])
    def test_anything_but_a_tooloutput_is_failed(self, returned):
        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        [f] = reg.evaluate("dragon attacks", execute_tool=lambda t, **_: returned)
        assert f.outcome == "failed" and "not a ToolOutput" in f.error

    def test_a_real_tool_failure_through_the_pipeline_is_failed(self):
        """The production failure: a real DamageComponentTool with no body."""
        from maxim.simulation.tools import DamageComponentTool

        pipeline = BioEnrichmentPipeline(reflex_registry=ReflexRegistry((_attack_reflex(),), clock=_Clock()))
        pipeline._reflex_damage_tool = DamageComponentTool(embodiment=None, entity_map=None)
        pipeline._entity_root = TestLatentAffordances()._make_entity_with_latent()
        latent: list[str] = []
        assert pipeline._evaluate_reflexes("The dragon attacks you", (), latent_out=latent) == ()
        assert latent == []
        [f] = pipeline._reflex_registry.evaluate("dragon attacks", execute_tool=pipeline._dispatch_reflex_tool)
        assert (f.outcome, f.error) == ("failed", "No embodiment configured")

    def test_sim_reflex_is_emitted_only_when_the_body_responded(self, monkeypatch):
        import maxim.simulation.sim_logger as sim_logger
        from maxim.integration.bio_enrichment import CausalPrediction

        emitted: list[str] = []
        monkeypatch.setattr(sim_logger, "sim_reflex", lambda name, *a, **kw: emitted.append(name))

        def _fail(t, **_):
            raise RuntimeError("x")

        ReflexRegistry((_attack_reflex(),), clock=_Clock()).evaluate("dragon attacks", execute_tool=_fail)
        pre = (CausalPrediction(event="attack", outcome="damage", confidence=1.0, valence="negative"),)
        ReflexRegistry((_attack_reflex(),), clock=_Clock()).evaluate(
            "dragon attacks", predictions=pre, execute_tool=_ok
        )
        assert emitted == []  # failed and suppressed emit nothing
        ReflexRegistry((_attack_reflex(),), clock=_Clock()).evaluate("dragon attacks", execute_tool=_ok)
        assert emitted == ["attack_flinch"]

    def test_reset_state_rearms_the_returned_failure_warning(self, caplog):
        from maxim.tools.base import ToolOutput

        reg = ReflexRegistry((_attack_reflex(),), clock=_Clock())
        fail = lambda t, **_: ToolOutput(success=False, error="down")  # noqa: E731
        reg.evaluate("dragon attacks", execute_tool=fail)
        reg.reset_state()
        caplog.clear()  # only the post-reset call may satisfy the assertion
        with caplog.at_level("WARNING"):
            reg.evaluate("dragon attacks", execute_tool=fail)
        assert any("did not respond" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


class TestSensorReflexDelta:
    """#871: sensor reflexes declared deltas that set_entity_sensor SET, zeroing the sensor."""

    def test_a_sensor_reflex_must_declare_delta_not_value(self):
        with pytest.raises(ValueError, match="delta"):
            ReflexSpec(
                name="bad",
                detect_keywords=("x",),
                response=ReflexResponse(tool="set_entity_sensor", params={"sensor": "awareness", "value": -0.1}),
            )

    @pytest.mark.parametrize("archetype", ["humanoid", "infant", "quadruped"])
    def test_shipped_reflex_files_load_in_full(self, archetype):
        """Read each file DIRECTLY: ``load_archetype_reflexes`` swallows a bad
        spec and returns (), which made the first version of this test vacuous."""
        from maxim.utils.paths import bundled_data

        specs = load_reflex_specs(bundled_data() / "reflexes" / f"{archetype}.yaml")
        assert specs
        for spec in specs:
            if spec.response.tool == "set_entity_sensor":
                assert "delta" in spec.response.params, spec.name

    @pytest.mark.parametrize(
        ("archetype", "body_ref"),
        [
            ("humanoid", "bodies/base_humanoid"),  # the Exp 09 body
            ("humanoid", "bodies/infant_humanoid"),
            ("infant", "bodies/infant_humanoid"),
        ],
    )
    def test_every_shipped_reflex_acts_on_a_real_body(self, archetype, body_ref):
        """Fire every reflex through the REAL tools against the REAL body.

        A synthetic body let ``startle`` target a bare ``awareness`` that no
        shipped body has (it is the ``head`` modulator's sub-sensor): the old
        path wrote an orphan root key, and the first #871 fix would have made
        every startle fail. Only a real body can catch that.
        """
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.simulation.tools import DamageComponentTool, SetEntitySensorTool

        root = ComponentRegistry().instantiate(body_ref)
        emb = _SensorEmbodiment(root)
        tools = {
            "damage_component": DamageComponentTool(embodiment=emb, entity_map=None),
            "set_entity_sensor": SetEntitySensorTool(embodiment=emb, entity_map=None),
        }
        specs = load_archetype_reflexes(archetype)
        assert specs
        from maxim.simulation.tools import _sensor_slot

        for spec in specs:
            reg = ReflexRegistry((spec,), clock=_Clock())
            text = f"it {spec.detect_keywords[0]} here"
            sensor = spec.response.params.get("sensor")
            if sensor is not None:
                metrics, key = _sensor_slot(root, sensor)
                before = metrics[key]
            outputs: list = []

            def _dispatch(t, **p):
                outputs.append(tools[t].execute(**p))
                return outputs[-1]

            [f] = reg.evaluate(text, execute_tool=_dispatch)
            assert f.outcome == "acted", (spec.name, f.error)
            if spec.response.tool == "damage_component":
                # DamageComponentTool falls back to root ``health`` (and still
                # succeeds) when the part is missing — "acted" alone would pass
                # for a reflex aimed at a part the body does not have.
                assert outputs[-1].output["fallback_to_entity"] is False, spec.name
            if sensor is not None:
                # Moved by exactly the DELTA — not set to an absolute value. The
                # shipped bodies start these sensors healthy and the deltas are
                # small, so no range clamp applies here.
                assert metrics[key] - before == pytest.approx(f.params["delta"]), (spec.name, before, metrics[key])

    def test_a_startle_lowers_head_awareness_by_its_delta_on_the_real_body(self):
        from maxim.embodiment.component_registry import ComponentRegistry
        from maxim.simulation.tools import SetEntitySensorTool

        root = ComponentRegistry().instantiate("bodies/base_humanoid")
        before = root.modulators["head"].vital_metrics["awareness"]
        [startle] = [s for s in load_archetype_reflexes("humanoid") if s.name == "startle"]
        pipeline = BioEnrichmentPipeline(reflex_registry=ReflexRegistry((startle,), clock=_Clock()))
        pipeline._reflex_sensor_tool = SetEntitySensorTool(embodiment=_SensorEmbodiment(root), entity_map=None)
        assert pipeline._evaluate_reflexes("a deafening explosion", ()) == ("startle",)
        after = root.modulators["head"].vital_metrics["awareness"]
        assert 0.0 < after < before  # lowered, not zeroed
        assert "awareness" not in root.vital_metrics  # no orphan root key

    def test_intensity_now_scales_the_sensor_change(self):
        from maxim.simulation.tools import SetEntitySensorTool

        body = _SensorBody({"awareness": 0.8})
        tool = SetEntitySensorTool(embodiment=_SensorEmbodiment(body), entity_map=None)
        reg = ReflexRegistry((_startle_reflex(),), clock=_Clock())
        [f] = reg.evaluate("a deafening explosion", execute_tool=lambda t, **p: tool.execute(**p))
        # effective == base here, so the delta is exactly the declared -0.1
        assert f.params["delta"] == pytest.approx(-0.1)


class _SensorBody:
    name = "body"
    full_path = "body"

    def __init__(self, metrics: dict) -> None:
        self.vital_metrics = dict(metrics)
        self.sensors: dict = {}
        self.modulators: dict = {}


class _SensorEmbodiment:
    def __init__(self, root) -> None:
        self.root = root

    def evaluate_failures(self):
        return []


class TestSetEntitySensorDelta:
    def _tool(self, body):
        from maxim.simulation.tools import SetEntitySensorTool

        return SetEntitySensorTool(embodiment=_SensorEmbodiment(body), entity_map=None)

    def test_delta_adjusts_a_root_sensor(self):
        body = _SensorBody({"stamina": 0.5})
        out = self._tool(body).execute(sensor="stamina", delta=-0.2)
        assert out.success and body.vital_metrics["stamina"] == pytest.approx(0.3)

    def test_delta_reaches_a_qualified_sub_sensor_within_its_declared_range(self):
        """``arms.thermal`` used to become an orphan key on the ROOT."""

        class _Arms:
            vital_metrics = {"thermal": 0.5}
            _sensors = {"thermal": {"range": [-1.0, 1.0]}}

        body = _SensorBody({})
        body.modulators["arms"] = _Arms()
        out = self._tool(body).execute(sensor="arms.thermal", delta=-0.8)
        assert out.success
        assert body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(-0.3)  # range, not [0, 1]
        assert "arms.thermal" not in body.vital_metrics

    def test_a_missing_sensor_is_a_failed_call_not_a_silent_no_op(self):
        out = self._tool(_SensorBody({})).execute(sensor="awareness", delta=-0.1)
        assert out.success is False and "not found" in out.error

    @pytest.mark.parametrize("bad", [float("nan"), "lots"])
    def test_a_bad_delta_is_rejected(self, bad):
        out = self._tool(_SensorBody({"stamina": 0.5})).execute(sensor="stamina", delta=bad)
        assert out.success is False

    def test_value_and_delta_together_are_rejected(self):
        out = self._tool(_SensorBody({"stamina": 0.5})).execute(sensor="stamina", value=0.3, delta=-0.1)
        assert out.success is False

    def test_value_still_sets(self):
        body = _SensorBody({"health": 0.2})
        assert self._tool(body).execute(sensor="health", value=0.9).success
        assert body.vital_metrics["health"] == pytest.approx(0.9)


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

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        result = pipeline.enrich("The dragon attacks you violently", bypass_gate=True)

        assert result is not None
        assert "attack_flinch" in result.reflexes_fired

    def test_no_reflexes_when_registry_not_set(self):
        pipeline = BioEnrichmentPipeline()
        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)

        assert result is not None
        assert result.reflexes_fired == ()

    def test_a_failure_after_evaluate_cannot_erase_fired_reflexes(self, monkeypatch, caplog):
        """evaluate() RUNS body tools, so once it returns the firings are fact.

        The whole block sat under one ``except Exception: return ()``, so a
        failure after evaluate() reported "no reflexes fired" for reflexes that
        had (#863 review round). Propagating instead is no better: every
        enrich() caller swallows quietly, which would drop the whole enrichment
        and still hide the firing. The names must come back, and the telemetry
        failure must be reported.
        """
        from maxim.embodiment.reflex import ReflexFiring

        firing = ReflexFiring(
            reflex_name="attack_flinch",
            tool="damage_component",
            params={},
            effective_intensity=None,  # the telemetry's ``:.2f`` raises on this
            raw_intensity=0.5,
            habituation_factor=1.0,
            sensitization_factor=1.0,
            preemption_factor=1.0,
            outcome="acted",
        )
        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=ReflexRegistry((_attack_reflex(),), clock=_Clock())))
        monkeypatch.setattr(pipeline._reflex_registry, "evaluate", lambda *_a, **_kw: [firing])
        with caplog.at_level("WARNING"):
            assert pipeline._evaluate_reflexes("The dragon attacks you", (), latent_out=[]) == ("attack_flinch",)
        assert any(
            "swallowed" in r.getMessage().lower() or "_evaluate_reflexes" in r.getMessage() for r in caplog.records
        )

    def test_an_evaluate_failure_is_contained_and_reported(self, monkeypatch, caplog):
        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=ReflexRegistry((_attack_reflex(),), clock=_Clock())))

        def _boom(*_a, **_kw):
            raise RuntimeError("evaluate failed")

        monkeypatch.setattr(pipeline._reflex_registry, "evaluate", _boom)
        with caplog.at_level("WARNING"):
            assert pipeline._evaluate_reflexes("The dragon attacks you", ()) == ()
        assert any("bio_enrichment" in r.getMessage() or "_evaluate_reflexes" in r.getMessage() for r in caplog.records)

    def test_reflexes_fired_is_empty_when_no_match(self):
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
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


# ---------------------------------------------------------------------------
# Latent affordance surfacing (proprioceptive discovery)
# ---------------------------------------------------------------------------


class TestLatentAffordances:
    """Latent motor programs surface when reflexes fire, piggybacking on
    reflex detection to reveal body responses the agent could take."""

    def _make_entity_with_latent(self, *, legs_integrity: float = 1.0, arms_integrity: float = 1.0):
        """Build a minimal Entity with latent affordances on legs and arms."""
        from maxim.embodiment.sem import Entity
        from maxim.embodiment.spec import LatentAffordance, SpecModulator

        entity = Entity(name="test_body", entity_type="body")

        legs = SpecModulator(
            _name="legs",
            _entity_name="test_body",
            _affordances={},
            _sensors={"leg_mobility": {"unit": "ratio", "range": [0, 1], "weight": 1.0}},
            _latent_affordances=(
                LatentAffordance(name="dodge", description="Dodge sideways", requires={"integrity": 0.2}),
                LatentAffordance(name="roll", description="Roll away", requires={"integrity": 0.3}),
            ),
        )
        legs.vital_metrics["leg_mobility"] = legs_integrity
        entity.modulators["legs"] = legs

        arms = SpecModulator(
            _name="arms",
            _entity_name="test_body",
            _affordances={},
            _sensors={"arm_mobility": {"unit": "ratio", "range": [0, 1], "weight": 1.0}},
            _latent_affordances=(
                LatentAffordance(name="block", description="Block a blow", requires={"integrity": 0.15}),
                LatentAffordance(name="parry", description="Deflect with weapon", requires={"integrity": 0.3}),
            ),
        )
        arms.vital_metrics["arm_mobility"] = arms_integrity
        entity.modulators["arms"] = arms

        return entity

    def test_latent_affordances_surface_when_reflex_fires(self):
        """When a reflex fires, latent affordances from ALL body modulators appear."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        entity = self._make_entity_with_latent()

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)
        assert result is not None

        # Should see latent affordances in the affordances tuple
        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert "dodge" in aff_names
        assert "block" in aff_names
        assert "roll" in aff_names
        assert "parry" in aff_names

    def test_no_latent_affordances_without_reflex(self):
        """When no reflex fires, no latent affordances surface."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        entity = self._make_entity_with_latent()

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The weather is pleasant today", bypass_gate=True)
        assert result is not None
        # No reflexes fired → no latent affordances
        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert "dodge" not in aff_names
        assert "block" not in aff_names

    def test_integrity_gating_excludes_damaged_affordances(self):
        """Broken legs → dodge and roll excluded."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        # Legs at 0.1 integrity — below dodge (0.2) and roll (0.3) thresholds
        entity = self._make_entity_with_latent(legs_integrity=0.1)

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)
        assert result is not None

        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert "dodge" not in aff_names  # requires 0.2, legs at 0.1
        assert "roll" not in aff_names  # requires 0.3, legs at 0.1
        assert "block" in aff_names  # arms are fine

    def test_partial_integrity_gating(self):
        """Legs at 0.25 → dodge available (0.2) but roll excluded (0.3)."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        entity = self._make_entity_with_latent(legs_integrity=0.25)

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)
        assert result is not None

        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert "dodge" in aff_names  # 0.25 >= 0.2 threshold
        assert "roll" not in aff_names  # 0.25 < 0.3 threshold

    def test_no_entity_root_no_crash(self):
        """Pipeline without entity_root still works — no latent affordances."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        # No _entity_root set

        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)
        assert result is not None
        assert "attack_flinch" in result.reflexes_fired
        # No crash, no latent affordances
        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert "dodge" not in aff_names

    def test_latent_affordances_include_descriptions(self):
        """Latent affordance strings include the description."""
        clock = _Clock()
        reg = ReflexRegistry((_attack_reflex(),), clock=clock)
        entity = self._make_entity_with_latent()

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The dragon attacks you", bypass_gate=True)
        assert result is not None

        # Find the dodge entry — should have description
        dodge_entries = [a for a in result.affordances if a.startswith("dodge")]
        assert len(dodge_entries) == 1
        assert "Dodge sideways" in dodge_entries[0]

    def test_deduplication(self):
        """Same latent affordance from multiple reflexes is deduplicated."""
        clock = _Clock()
        # Two reflexes that can both fire
        reg = ReflexRegistry((_attack_reflex(), _fire_reflex()), clock=clock)
        entity = self._make_entity_with_latent()

        pipeline = _wired(BioEnrichmentPipeline(reflex_registry=reg))
        pipeline._entity_root = entity

        result = pipeline.enrich("The dragon attacks with fire", bypass_gate=True)
        assert result is not None

        # Even though 2 reflexes fired, dodge should appear only once
        aff_names = [a.split(" — ")[0] for a in result.affordances]
        assert aff_names.count("dodge") == 1
