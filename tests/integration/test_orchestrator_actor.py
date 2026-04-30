"""Integration test for OrchestratorActorTool.

Verifies the end-to-end path described in
``docs/plans/scene_actor_affordances.md`` Stage 2:

    orchestrator narrates "the dragon breathes fire on you" AND
    actor(entity='dragon', affordance='breathe_fire', target='player')
        ↓
    scene entity affordance dispatched against AUT body
        ↓
    target_effect deltas applied to AUT sensors
        ↓
    AUT.evaluate_failures() fires
        ↓
    PainSignal published on AUT PainBus
        ↓
    NAc records causal link with dragon provenance
"""

from __future__ import annotations

import pytest


def _make_aut_body() -> "object":
    from maxim.embodiment.spec import _parse_entity

    return _parse_entity(
        {
            "name": "player",
            "entity_type": "humanoid",
            "modulators": {
                "arms": {
                    "sensors": {
                        "thermal": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
                    },
                },
                "torso": {
                    "sensors": {
                        "thermal": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
                    },
                },
            },
            "sensors": {
                "core_temperature": {"unit": "celsius_norm", "range": [-1, 1], "initial": 0.0},
            },
            "failure_modes": [
                {
                    "name": "burned",
                    "trigger": {"field": "core_temperature", "op": ">", "value": 0.5, "pain": 0.7},
                    "pain_intensity": 0.7,
                },
            ],
        }
    )


def _make_dragon() -> "object":
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
                            "self_effect": {"stamina": -0.2},
                            "target_effect": {
                                "arms.thermal": 0.6,
                                "torso.thermal": 0.4,
                                "core_temperature": 0.7,
                            },
                        },
                    },
                },
            },
            "vital_metrics": {"stamina": 1.0},
        }
    )


@pytest.fixture
def actor_world():
    """Build an orchestrator-actor world: AUT body + dragon + bus + tool."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.entity_map import EntityMap
    from maxim.proprioception.pain_bus import PainBus
    from maxim.simulation.tools import OrchestratorActorTool

    aut_body = _make_aut_body()
    dragon = _make_dragon()
    pain_bus = PainBus()
    aut_emb = Embodiment(aut_body, pain_bus=pain_bus)

    emap = EntityMap()
    emap.register_self(aut_body)
    emap.register_scene(dragon)

    tool = OrchestratorActorTool(embodiment=aut_emb, entity_map=emap)

    received_signals: list = []
    pain_bus.subscribe(received_signals.append)

    return {
        "aut_body": aut_body,
        "dragon": dragon,
        "pain_bus": pain_bus,
        "aut_embodiment": aut_emb,
        "entity_map": emap,
        "tool": tool,
        "signals": received_signals,
    }


class TestEndToEnd:
    def test_dragon_breathes_fire_writes_aut_sensors(self, actor_world):
        """target_effect lands on AUT body sensors after the actor call."""
        tool = actor_world["tool"]
        aut_body = actor_world["aut_body"]

        result = tool.execute(entity="dragon", affordance="breathe_fire", target="player")

        assert result.success, f"actor failed: {result.error}"
        assert aut_body.modulators["arms"].vital_metrics["thermal"] == pytest.approx(0.6, abs=0.01)
        assert aut_body.modulators["torso"].vital_metrics["thermal"] == pytest.approx(0.4, abs=0.01)
        assert aut_body.vital_metrics["core_temperature"] == pytest.approx(0.7, abs=0.01)

    def test_aut_failure_mode_fires_on_target_effect(self, actor_world):
        """target_effect crosses an AUT failure threshold → PainSignal."""
        tool = actor_world["tool"]
        signals = actor_world["signals"]

        result = tool.execute(entity="dragon", affordance="breathe_fire", target="player")

        assert result.success
        assert result.output["aut_failures"], "expected the burned failure to fire"
        burned = [f for f in result.output["aut_failures"] if f["name"] == "burned"]
        assert burned, f"expected 'burned' failure mode in {result.output['aut_failures']}"
        # PainSignal must have landed on the AUT's PainBus.
        assert signals, "expected at least one PainSignal published on AUT PainBus"

    def test_provenance_reaches_side_effects(self, actor_world):
        """side_effects carries dragon→breathe_fire provenance."""
        tool = actor_world["tool"]

        result = tool.execute(entity="dragon", affordance="breathe_fire", target="player")

        assert result.success
        assert result.side_effects is not None
        assert result.side_effects.get("actor_invocation") == {
            "source_entity": "dragon",
            "source_affordance": "breathe_fire",
            "target": "player",
        }
        # Output also tagged for the orchestrator's narration.
        assert result.output["source_entity"] == "dragon"
        assert result.output["source_affordance"] == "breathe_fire"
        assert result.output["target"] == "player"


class TestInvariants:
    def test_self_side_entity_is_rejected(self, actor_world):
        """Invariant 1: AUT body cannot be operated as a scene actor."""
        tool = actor_world["tool"]

        result = tool.execute(entity="player", affordance="breathe_fire", target="player")

        assert not result.success
        assert "AUT's own body" in (result.error or "")

    def test_unknown_entity_errors(self, actor_world):
        tool = actor_world["tool"]

        result = tool.execute(entity="unicorn", affordance="charge", target="player")

        assert not result.success
        assert "unicorn" in (result.error or "")

    def test_unknown_affordance_errors_with_available_list(self, actor_world):
        tool = actor_world["tool"]

        result = tool.execute(entity="dragon", affordance="teleport", target="player")

        assert not result.success
        assert "teleport" in (result.error or "")
        assert "breathe_fire" in (result.error or ""), "should list available affordances"

    def test_unresolvable_target_errors(self, actor_world):
        """Invariant 5: target resolution is explicit, not magic."""
        tool = actor_world["tool"]

        result = tool.execute(entity="dragon", affordance="breathe_fire", target="bystander_npc")

        assert not result.success
        assert "bystander_npc" in (result.error or "")

    def test_target_aliases_all_resolve_to_aut(self, actor_world):
        """'self', 'aut', 'player' are interchangeable AUT references."""
        tool = actor_world["tool"]
        aut_body = actor_world["aut_body"]

        # Reset between calls; only the FIRST will fire the burned trigger
        # because failure_modes track active state across invocations.
        for alias in ("self", "aut", "player"):
            aut_body.vital_metrics["core_temperature"] = 0.0
            aut_body.modulators["arms"].vital_metrics["thermal"] = 0.0
            aut_body.modulators["torso"].vital_metrics["thermal"] = 0.0
            result = tool.execute(entity="dragon", affordance="breathe_fire", target=alias)
            assert result.success, f"{alias} → expected success, got {result.error}"
            assert aut_body.vital_metrics["core_temperature"] == pytest.approx(0.7, abs=0.01)

    def test_missing_entity_arg_errors(self, actor_world):
        tool = actor_world["tool"]

        result = tool.execute(entity="", affordance="breathe_fire", target="player")
        assert not result.success
        assert "entity is required" in (result.error or "")

    def test_missing_affordance_arg_errors(self, actor_world):
        tool = actor_world["tool"]

        result = tool.execute(entity="dragon", affordance="", target="player")
        assert not result.success
        assert "affordance is required" in (result.error or "")

    def test_target_inside_params_does_not_collide(self, actor_world):
        """LLM mistakenly nests `target` inside `params` → no TypeError."""
        tool = actor_world["tool"]
        aut_body = actor_world["aut_body"]

        # The LLM passes target= correctly AND also inside params — a real
        # 14B failure mode.  The tool must strip the duplicate before
        # forwarding to the ephemeral affordance call.
        result = tool.execute(
            entity="dragon",
            affordance="breathe_fire",
            target="player",
            params={"target": "player_alias_inside_params"},
        )
        assert result.success, f"actor failed: {result.error}"
        assert aut_body.vital_metrics["core_temperature"] == pytest.approx(0.7, abs=0.01)


class TestRequiresGating:
    def test_requires_gating_blocks_when_modulator_too_damaged(self):
        """Affordance ``requires`` gating fires naturally — invariant 4."""
        from maxim.embodiment.body import Embodiment
        from maxim.embodiment.entity_map import EntityMap
        from maxim.embodiment.spec import _parse_entity
        from maxim.proprioception.pain_bus import PainBus
        from maxim.simulation.tools import OrchestratorActorTool

        aut_body = _make_aut_body()
        # Dragon with a fragile fire_breath gated on integrity > 0.3.
        dragon = _parse_entity(
            {
                "name": "dragon",
                "entity_type": "creature",
                "modulators": {
                    "fire_breath": {
                        "sensors": {
                            "stamina": {"range": [0, 1], "initial": 0.1},  # damaged
                        },
                        "affordances": {
                            "breathe_fire": {
                                "params": {"target": "str"},
                                "requires": {"integrity": 0.3},
                                "target_effect": {"core_temperature": 0.5},
                            },
                        },
                    },
                },
            }
        )
        pain_bus = PainBus()
        aut_emb = Embodiment(aut_body, pain_bus=pain_bus)
        emap = EntityMap()
        emap.register_self(aut_body)
        emap.register_scene(dragon)
        tool = OrchestratorActorTool(embodiment=aut_emb, entity_map=emap)

        result = tool.execute(entity="dragon", affordance="breathe_fire", target="player")

        # Affordance is BLOCKED (success=False) but the actor tool itself
        # still surfaces the block reason for the orchestrator to narrate.
        assert not result.success
        # Target body must be unchanged when the affordance is blocked.
        assert aut_body.vital_metrics["core_temperature"] == 0.0
