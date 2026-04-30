"""Unit tests for target-effect on affordances.

``target_effect`` is the parallel of ``self_effect`` for affordances that
write to a *resolved target body* — e.g. a dragon's ``breathe_fire``
affordance writes thermal deltas onto the AUT.  Stage 1 of
``docs/plans/scene_actor_affordances.md``.
"""

from __future__ import annotations

import pytest

from maxim.embodiment.sem import AffordanceSchema


class TestTargetEffectSchema:
    def test_target_effect_default_empty(self) -> None:
        schema = AffordanceSchema(params={}, description="test")
        assert schema.target_effect == {}

    def test_target_effect_parsed_from_yaml(self) -> None:
        from maxim.embodiment.spec import _parse_entity

        data = {
            "name": "dragon",
            "entity_type": "creature",
            "modulators": {
                "fire_breath": {
                    "affordances": {
                        "breathe_fire": {
                            "params": {"target": "str"},
                            "description": "Exhale fire on target",
                            "self_effect": {"stamina": -0.2},
                            "target_effect": {"arms.thermal": 0.6, "torso.thermal": 0.4},
                        },
                    },
                },
            },
            "vital_metrics": {"stamina": 1.0},
        }
        entity = _parse_entity(data)
        schema = entity.modulators["fire_breath"].affordances["breathe_fire"]
        assert schema.self_effect == {"stamina": -0.2}
        assert schema.target_effect == {"arms.thermal": 0.6, "torso.thermal": 0.4}


class TestTargetEffectExecution:
    def _make_target_body(self) -> "object":
        from maxim.embodiment.spec import _parse_entity

        body_data = {
            "name": "player",
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
        return _parse_entity(body_data)

    def _make_dragon(self) -> "object":
        from maxim.embodiment.spec import _parse_entity

        return _parse_entity(
            {
                "name": "dragon",
                "entity_type": "creature",
                "modulators": {
                    "fire_breath": {
                        "affordances": {
                            "breathe_fire": {
                                "params": {"target": "str"},
                                "description": "Exhale fire on target",
                                "target_effect": {"arms.thermal": 0.6, "core_temperature": 0.15},
                            },
                        },
                    },
                },
            }
        )

    def test_target_effect_writes_to_resolved_entity(self) -> None:
        """target_effect resolves a non-self target via entity_map and writes deltas."""
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        target_body = self._make_target_body()
        dragon = self._make_dragon()
        emap = EntityMap()
        emap.register_scene(target_body)
        emap.register_scene(dragon)

        mod = dragon.modulators["fire_breath"]
        schema = mod.affordances["breathe_fire"]
        tool = ModulatorAffordanceTool(
            dragon,
            mod,
            "breathe_fire",
            schema,
            "dragon_breathe_fire",
            embodiment=None,  # the dragon is scene-side; no executor body
            entity_map=emap,
        )

        result = tool.execute(target="player")
        assert result.success

        # arms.thermal (modulator sub-sensor) should be 0.6
        assert target_body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.6, abs=0.01)
        # core_temperature (entity-level sensor) should be 0.15
        assert target_body.vital_metrics["core_temperature"] == pytest.approx(0.15, abs=0.01)

    def test_target_effect_silent_when_no_target(self) -> None:
        """No target → target_effect is silently skipped (backward compat)."""
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        target_body = self._make_target_body()
        dragon = self._make_dragon()
        emap = EntityMap()
        emap.register_scene(target_body)
        emap.register_scene(dragon)

        mod = dragon.modulators["fire_breath"]
        schema = mod.affordances["breathe_fire"]
        tool = ModulatorAffordanceTool(
            dragon,
            mod,
            "breathe_fire",
            schema,
            "dragon_breathe_fire",
            embodiment=None,
            entity_map=emap,
        )

        result = tool.execute()  # no target= kwarg
        assert result.success
        # Target body must be unchanged
        assert target_body.modulators["arms"].vital_metrics["thermal"] == 0.0
        assert target_body.vital_metrics["core_temperature"] == 0.0

    def test_target_effect_clamps_to_sensor_range(self) -> None:
        """target_effect clamps to the target sensor's declared range."""
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body_data = {
            "name": "player",
            "entity_type": "body",
            "sensors": {
                "core_temperature": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.7},
            },
        }
        target_body = _parse_entity(body_data)
        dragon = _parse_entity(
            {
                "name": "dragon",
                "entity_type": "creature",
                "modulators": {
                    "fire_breath": {
                        "affordances": {
                            "breathe_fire": {
                                "params": {"target": "str"},
                                "target_effect": {"core_temperature": 0.9},  # 0.7+0.9=1.6 → clamps to 1.0
                            },
                        },
                    },
                },
            }
        )
        emap = EntityMap()
        emap.register_scene(target_body)
        emap.register_scene(dragon)

        mod = dragon.modulators["fire_breath"]
        schema = mod.affordances["breathe_fire"]
        tool = ModulatorAffordanceTool(
            dragon,
            mod,
            "breathe_fire",
            schema,
            "dragon_breathe_fire",
            embodiment=None,
            entity_map=emap,
        )
        tool.execute(target="player")
        # 0.7 + 0.9 = 1.6 → clamped to 1.0
        assert target_body.vital_metrics["core_temperature"] == pytest.approx(1.0, abs=0.01)

    def test_target_aliases_resolve_to_self_body(self) -> None:
        """'self', 'aut', 'player' map to the executor's own body when embodiment is wired."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        # Simulate self-targeted invocation where the executor is the
        # target — e.g. the orchestrator has the dragon "lash itself".
        # In production the OrchestratorActorTool runs the affordance
        # against an arbitrary target; here we exercise the alias path.
        own_body = _parse_entity(
            {
                "name": "agent",
                "entity_type": "body",
                "sensors": {
                    "core_temperature": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
                },
            }
        )
        emb = Embodiment(own_body)
        sword = _parse_entity(
            {
                "name": "sword",
                "entity_type": "weapon",
                "modulators": {
                    "blade": {
                        "affordances": {
                            "swing": {
                                "params": {"target": "str"},
                                "target_effect": {"core_temperature": 0.2},
                            },
                        },
                    },
                },
            }
        )
        emap = EntityMap()
        emap.register_self(own_body)
        emap.register_scene(sword)

        mod = sword.modulators["blade"]
        schema = mod.affordances["swing"]
        tool = ModulatorAffordanceTool(
            sword,
            mod,
            "swing",
            schema,
            "sword_swing",
            embodiment=emb,
            entity_map=emap,
        )
        tool.execute(target="self")
        assert own_body.vital_metrics["core_temperature"] == pytest.approx(0.2, abs=0.01)

    def test_target_effect_unresolvable_target_is_silent(self) -> None:
        """Unknown target name → no-op, no exception, no sensor mutation."""
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        target_body = self._make_target_body()
        dragon = self._make_dragon()
        emap = EntityMap()
        emap.register_scene(target_body)
        emap.register_scene(dragon)

        mod = dragon.modulators["fire_breath"]
        schema = mod.affordances["breathe_fire"]
        tool = ModulatorAffordanceTool(
            dragon,
            mod,
            "breathe_fire",
            schema,
            "dragon_breathe_fire",
            embodiment=None,
            entity_map=emap,
        )
        result = tool.execute(target="nonexistent_entity")
        assert result.success
        assert target_body.modulators["arms"].vital_metrics["thermal"] == 0.0
        assert target_body.vital_metrics["core_temperature"] == 0.0

    def test_target_effect_missing_sensor_logs_warning(self, caplog: "pytest.LogCaptureFixture") -> None:
        """Missing target sensor logs a warning, doesn't crash."""
        import logging

        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        target_body = _parse_entity(
            {
                "name": "player",
                "entity_type": "body",
                "sensors": {"hp": {"unit": "ratio", "range": [0, 1], "initial": 1.0}},
            }
        )
        dragon = _parse_entity(
            {
                "name": "dragon",
                "entity_type": "creature",
                "modulators": {
                    "fire_breath": {
                        "affordances": {
                            "breathe_fire": {
                                "params": {"target": "str"},
                                "target_effect": {"thermal": 0.4},  # not on body
                            },
                        },
                    },
                },
            }
        )
        emap = EntityMap()
        emap.register_scene(target_body)
        emap.register_scene(dragon)

        mod = dragon.modulators["fire_breath"]
        schema = mod.affordances["breathe_fire"]
        tool = ModulatorAffordanceTool(
            dragon,
            mod,
            "breathe_fire",
            schema,
            "dragon_breathe_fire",
            embodiment=None,
            entity_map=emap,
        )
        with caplog.at_level(logging.WARNING):
            tool.execute(target="player")
        assert any("target_effect target" in r.message for r in caplog.records)


class TestSelfEffectStillWorksAfterRefactor:
    """Regression check: extracting `_apply_sensor_deltas` must not break self_effect."""

    def test_self_effect_unchanged_behavior(self) -> None:
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.spec import _parse_entity
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        body = _parse_entity(
            {
                "name": "agent",
                "entity_type": "body",
                "sensors": {"hunger": {"unit": "ratio", "range": [0, 1], "initial": 0.8}},
            }
        )
        food = _parse_entity(
            {
                "name": "food",
                "entity_type": "item",
                "modulators": {
                    "nutrition": {
                        "affordances": {
                            "eat": {
                                "params": {},
                                "self_effect": {"hunger": -0.4},
                            },
                        },
                    },
                },
            }
        )
        emb = Embodiment(body)
        mod = food.modulators["nutrition"]
        schema = mod.affordances["eat"]
        tool = ModulatorAffordanceTool(food, mod, "eat", schema, "food_eat", embodiment=emb)
        tool.execute()
        assert body.vital_metrics["hunger"] == pytest.approx(0.4, abs=0.01)
