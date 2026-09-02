"""Unit tests for self-effect on affordances.

Self-effects allow scene entity affordances to write back to the agent's
body sensors. Example: eating food reduces hunger.

**TestSelfEffectFailureCascade** (added 2026-06-01 via Exp 37 root-cause
investigation) pins the order-of-operations invariant: when an affordance's
self_effect writes a body sensor that breaches a homeostatic drive's
comfort_band, the resulting drive pain MUST appear in
``ToolOutput.side_effects["embodiment_failures"]`` so the executor can
route it to ``ToolPainBridge.record_tool_embodiment_failure`` for direct
NAc attribution. Per CLAUDE.md cradle Stage 3 invariant + the
"Tool-invoked embodiment pain attributes directly, not via context
similarity" architectural rule, this is the path that produces the
negative-valence NAc link the 1.0 cross-session graduation depends on.

The bug this test pins: ``ModulatorAffordanceTool.execute`` called
``evaluate_failures`` BEFORE applying self_effect/target_effect, so the
failure check saw the body in its pre-effect state (no breach) and
returned empty active_failures. The fix swaps the order.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.sem import AffordanceSchema


class TestSelfEffectSchema:
    def test_self_effect_parsed_from_yaml(self):
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "abstract": True,
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat the food",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        entity = _parse_entity(data)
        schema = entity.modulators["nutrition"].affordances["eat"]
        assert schema.self_effect == {"hunger": -0.4}

    def test_no_self_effect_default(self):
        schema = AffordanceSchema(params={}, description="test")
        assert schema.self_effect == {}


class TestSelfEffectExecution:
    def test_eat_reduces_hunger(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        # Agent body with hunger sensor
        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
        }
        # Food entity with self_effect
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "abstract": True,
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        # Execute eat on food entity, with embodiment pointing to agent body
        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(
            food,
            nutrition_mod,
            "eat",
            schema,
            "food_eat",
            embodiment=emb,
        )
        result = tool.execute()
        assert result.success

        # hunger should drop from 0.8 to 0.4
        assert body.vital_metrics["hunger"] == pytest.approx(0.4, abs=0.01)

    def test_self_effect_clamps_at_zero(self):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.2}},
        }
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "abstract": True,
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(food, nutrition_mod, "eat", schema, "food_eat", embodiment=emb)
        tool.execute()

        # 0.2 - 0.4 = -0.2 → clamped to 0.0
        assert body.vital_metrics["hunger"] == 0.0

    def test_self_effect_missing_sensor_logs_warning(self, caplog):
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "agent",
            "entity_type": "body",
            "sensors": {"hp": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
        }
        food_data = {
            "name": "food",
            "entity_type": "item",
            "modulators": {
                "nutrition": {
                    "abstract": True,
                    "affordances": {
                        "eat": {
                            "params": {},
                            "description": "Eat",
                            "self_effect": {"hunger": -0.4},  # hunger doesn't exist on body
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        food = _parse_entity(food_data)
        emb = Embodiment(body)

        nutrition_mod = food.modulators["nutrition"]
        schema = nutrition_mod.affordances["eat"]
        tool = ModulatorAffordanceTool(food, nutrition_mod, "eat", schema, "food_eat", embodiment=emb)

        import logging

        with caplog.at_level(logging.WARNING):
            tool.execute()

        # Should log warning about missing sensor, not crash
        assert any("self_effect target" in r.message for r in caplog.records)

    def test_touch_fire_writes_modulator_sub_sensor(self):
        """self_effect with qualified mod.sensor name writes to modulator vital_metrics."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "agent",
            "entity_type": "body",
            "modulators": {
                "arms": {
                    "sensors": {
                        "thermal": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
                    },
                },
            },
            "sensors": {
                "core_temperature": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
            },
        }
        fire_data = {
            "name": "fire_pit",
            "entity_type": "hazard",
            "modulators": {
                "flame": {
                    "abstract": True,
                    "affordances": {
                        "touch": {
                            "params": {},
                            "description": "Touch the fire",
                            "self_effect": {"arms.thermal": 0.6, "core_temperature": 0.15},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        fire = _parse_entity(fire_data)
        emb = Embodiment(body)

        flame_mod = fire.modulators["flame"]
        schema = flame_mod.affordances["touch"]
        tool = ModulatorAffordanceTool(fire, flame_mod, "touch", schema, "fire_pit_touch", embodiment=emb)
        result = tool.execute()
        assert result.success

        # arms.thermal should be 0.6 (modulator sub-sensor)
        assert body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.6, abs=0.01)
        # core_temperature should be 0.15 (entity-level sensor)
        assert body.vital_metrics["core_temperature"] == pytest.approx(0.15, abs=0.01)


class TestSelfEffectFailureCascade:
    """The bug-pin tests (Exp 37 root-cause investigation, 2026-06-01).

    Pre-fix: ``ModulatorAffordanceTool.execute`` called ``evaluate_failures``
    BEFORE applying self_effect, so the failure check saw the body in its
    pre-effect state and missed the breach. ``side_effects[
    "embodiment_failures"]`` was never populated and ``ToolPainBridge``
    couldn't attribute the pain to the tool.

    Post-fix: self_effect / target_effect apply BEFORE evaluate_failures
    so the failure check sees the post-effect state.
    """

    def _build_fire_pit_scenario(self):
        """Build the minimal infant_humanoid + fire_pit replica from
        production YAML (cradle_fire_pit + infant_humanoid arms config)."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity

        body_data = {
            "name": "infant_humanoid",
            "entity_type": "body",
            "modulators": {
                "arms": {
                    "sensors": {
                        "thermal": {
                            "unit": "celsius_norm",
                            "range": [-1, 1],
                            "initial": 0.0,
                            "weight": 0.0,
                            "drive": {
                                "drift_mode": "homeostatic",
                                "set_point": 0.0,
                                "drift_rate": 0.0008,
                                "comfort_band": 0.5,
                                "pain_scale": 0.4,
                            },
                        },
                    },
                },
            },
        }
        fire_data = {
            "name": "fire_pit",
            "entity_type": "hazard",
            "sensors": {
                "heat_output": {"unit": "celsius_norm", "range": [0, 1], "initial": 0.9},
            },
            "modulators": {
                "flame": {
                    "abstract": True,
                    "affordances": {
                        "touch": {
                            "params": {},
                            "description": "Touch the fire",
                            "self_effect": {"arms.thermal": 0.6},
                        },
                    },
                },
            },
        }
        body = _parse_entity(body_data)
        fire = _parse_entity(fire_data)
        emb = Embodiment(body)
        return body, fire, emb

    def test_post_self_effect_breach_populates_side_effects(self):
        """The load-bearing assertion: fire_pit_touch's self_effect writes
        arms.thermal=0.6 (above comfort_band 0.5), so evaluate_failures
        MUST detect the breach AND ToolOutput.side_effects MUST carry the
        resulting embodiment_failures entry. Without this, ToolPainBridge
        can't attribute the pain to the tool and the bio-substrate's
        cross-session learning pipeline is dead for self_effect-based
        failure modes (the Exp 37 root cause)."""
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body, fire, emb = self._build_fire_pit_scenario()
        flame_mod = fire.modulators["flame"]
        schema = flame_mod.affordances["touch"]
        tool = ModulatorAffordanceTool(
            fire,
            flame_mod,
            "touch",
            schema,
            "fire_pit_touch",
            embodiment=emb,
        )

        result = tool.execute()

        assert result.success, "touch tool itself succeeded"
        assert body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.6, abs=0.01), (
            "self_effect should have written arms.thermal=0.6"
        )
        assert result.side_effects is not None, (
            "side_effects MUST be populated when self_effect causes a body-sensor breach"
        )
        failures = result.side_effects.get("embodiment_failures")
        assert failures, (
            "side_effects['embodiment_failures'] MUST contain the drive-pain "
            "event from arms.thermal breaching comfort_band. Empty failures means "
            "evaluate_failures ran BEFORE self_effect (the Exp 37 root cause bug)."
        )
        # The drive pain event names the drive that breached.
        assert any("thermal" in f.get("name", "") for f in failures), (
            f"expected a drive:thermal:discomfort failure; got {failures!r}"
        )

    def test_post_self_effect_active_failures_in_output_dict(self):
        """The agent-visible ``output["active_failures"]`` field MUST also
        reflect the post-self_effect breach so the LLM can reason about
        pain in subsequent turns."""
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body, fire, emb = self._build_fire_pit_scenario()
        flame_mod = fire.modulators["flame"]
        schema = flame_mod.affordances["touch"]
        tool = ModulatorAffordanceTool(
            fire,
            flame_mod,
            "touch",
            schema,
            "fire_pit_touch",
            embodiment=emb,
        )

        result = tool.execute()

        assert result.success
        active = result.output.get("active_failures", [])
        assert active, (
            "output['active_failures'] must reflect the post-self_effect breach "
            "so the LLM sees the pain signal in the same response"
        )

    def test_below_comfort_band_self_effect_does_not_populate_failures(self):
        """Negative control: self_effect that stays WITHIN comfort_band
        must NOT populate embodiment_failures (we're testing the order
        invariant, not a 'every self_effect fires failures' shortcut)."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body, _, _ = self._build_fire_pit_scenario()
        # Build a benign affordance: writes arms.thermal=0.3 (within ±0.5 band)
        benign_data = {
            "name": "warm_pad",
            "entity_type": "item",
            "modulators": {
                "surface": {
                    "abstract": True,
                    "affordances": {
                        "touch": {
                            "params": {},
                            "description": "Touch the warm pad",
                            "self_effect": {"arms.thermal": 0.3},
                        },
                    },
                },
            },
        }
        warm_pad = _parse_entity(benign_data)
        emb = Embodiment(body)
        surface_mod = warm_pad.modulators["surface"]
        schema = surface_mod.affordances["touch"]
        tool = ModulatorAffordanceTool(
            warm_pad,
            surface_mod,
            "touch",
            schema,
            "warm_pad_touch",
            embodiment=emb,
        )

        result = tool.execute()

        assert result.success
        assert body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.3, abs=0.01)
        # Within comfort_band — no drive pain, no failure event.
        failures = (result.side_effects or {}).get("embodiment_failures") or []
        thermal_failures = [f for f in failures if "thermal" in f.get("name", "")]
        assert not thermal_failures, (
            f"warm_pad touch (arms.thermal=0.3, within band 0.5) should not fire drive pain; got {thermal_failures!r}"
        )


class TestRequiresSilentNoOp:
    """A `requires` key that can never gate must say so at parse time.

    `SpecModulator.check_affordance_requires` resolves only the literal
    "integrity" and the modulator's OWN sub-sensors (`self.vital_metrics`).
    Any other key — an entity-level sensor being the common case — matches
    neither branch and falls through to `return True, ""`, so the
    precondition silently never gates. `SpecModulator` holds only
    `_entity_name` (a string), so it cannot reach entity sensors at all.

    Found 2026-09-01 during the Minecraft embodiment dive, where
    "mine requires a pickaxe" / "eat requires food in inventory" are exactly
    the shape that does not work. `items/cradle_food.yaml`'s
    `requires: {portions: 1}` is the only such key in the bundled library and
    is a live no-op — eating is unlimited.

    Deliberately surfaced, NOT fixed: making entity-level requires actually
    gate would make feeding finite in the cradle apparatus, and Exp 52 was
    mid-run. The warning removes the silence; the semantics decision is filed.
    """

    def test_unresolvable_requires_key_warns(self, caplog):
        from maxim.embodiment.spec import _parse_entity

        with caplog.at_level("WARNING"):
            _parse_entity(
                {
                    "name": "food",
                    "entity_type": "item",
                    "sensors": {"portions": {"unit": "count", "range": [0, 5], "initial": 5.0}},
                    "modulators": {
                        "nutrition": {
                            "abstract": True,
                            "affordances": {"eat": {"params": {}, "requires": {"portions": 1}}},
                        }
                    },
                }
            )
        msgs = " ".join(r.getMessage() for r in caplog.records)
        assert "NEVER gate" in msgs
        assert "portions" in msgs

    def test_integrity_requires_is_silent(self, caplog):
        """The one key that DOES resolve must not warn."""
        from maxim.embodiment.spec import _parse_entity

        with caplog.at_level("WARNING"):
            _parse_entity(
                {
                    "name": "wing",
                    "entity_type": "item",
                    "modulators": {
                        "flight": {
                            "abstract": True,
                            "affordances": {"fly": {"params": {}, "requires": {"integrity": 0.3}}},
                        }
                    },
                }
            )
        assert "NEVER gate" not in " ".join(r.getMessage() for r in caplog.records)

    def test_own_sub_sensor_requires_is_silent(self, caplog):
        """A modulator's own sub-sensor resolves, so it must not warn."""
        from maxim.embodiment.spec import _parse_entity

        with caplog.at_level("WARNING"):
            _parse_entity(
                {
                    "name": "torch",
                    "entity_type": "item",
                    "modulators": {
                        "burn": {
                            "sensors": {"fuel": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
                            "affordances": {"light": {"params": {}, "requires": {"fuel": 0.2}}},
                        }
                    },
                }
            )
        assert "NEVER gate" not in " ".join(r.getMessage() for r in caplog.records)
