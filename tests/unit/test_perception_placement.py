"""Tests for per-stage perception placement value type (commit 1).

Pins the lane-placement idioms this type borrows: str-enum serialization,
frozen value semantics, the CC3 path-(a) ``extra`` escape hatch +
collision guard, and coherence-validated-at-the-producer-boundary.

Regression guard for the ``perception_pipeline_placement.md`` plan.
"""

from __future__ import annotations

import dataclasses

import pytest

from maxim.runtime.perception_placement import (
    CANONICAL_PERCEPTION_PIPELINE,
    PerceptionStage,
    PerceptionStagePlacement,
    StageOrigin,
    resolve_pipeline_placements,
    validate_perception_placement_coherence,
)


class TestStageOrigin:
    def test_str_enum_value_equality(self):
        # str subclass: compares equal to its own value string (JSON-friendly).
        assert StageOrigin.NODE == "node"
        assert StageOrigin.SELF == "self"
        assert StageOrigin.SENSOR == "sensor"
        assert StageOrigin.SUBSTRATE_OWNER == "substrate_owner"

    def test_str_returns_bare_value(self):
        # __str__ override returns the bare value, not "StageOrigin.NODE"
        # (closes the f-string trap mirrored from worker_pool.Origin).
        assert str(StageOrigin.NODE) == "node"
        assert f"{StageOrigin.SENSOR}" == "sensor"

    def test_four_members(self):
        assert {o.value for o in StageOrigin} == {
            "self",
            "sensor",
            "substrate_owner",
            "node",
        }


class TestPerceptionStagePlacementShape:
    def test_frozen(self):
        p = PerceptionStagePlacement(stage="capture", origin=StageOrigin.SENSOR)
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.stage = "dsp"  # type: ignore[misc]

    def test_extra_hatch_accepts_additive_metadata(self):
        p = PerceptionStagePlacement(
            stage="segmentation",
            origin=StageOrigin.NODE,
            node="gpu-box",
            extra={"routing_weight": 0.5},
        )
        assert p.extra["routing_weight"] == 0.5

    def test_extra_collision_with_declared_field_raises(self):
        # CC3 path-(a): an extra key colliding with a declared field name
        # must raise, not silently shadow the declared slot.
        with pytest.raises(ValueError, match="collide with declared fields"):
            PerceptionStagePlacement(
                stage="capture",
                origin=StageOrigin.SENSOR,
                extra={"stage": "evil"},
            )

    def test_extra_excluded_from_eq_and_hash(self):
        # hash=False, compare=False on extra: two placements that differ only
        # in extra are equal and both hashable.
        a = PerceptionStagePlacement(stage="capture", origin=StageOrigin.SENSOR, extra={"a": 1})
        b = PerceptionStagePlacement(stage="capture", origin=StageOrigin.SENSOR, extra={"b": 2})
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1


class TestCoherenceValidator:
    def test_node_origin_requires_node(self):
        p = PerceptionStagePlacement(stage="segmentation", origin=StageOrigin.NODE)
        with pytest.raises(ValueError, match="requires a non-empty 'node'"):
            validate_perception_placement_coherence(p)

    def test_node_origin_empty_string_node_rejected(self):
        p = PerceptionStagePlacement(stage="segmentation", origin=StageOrigin.NODE, node="")
        with pytest.raises(ValueError, match="requires a non-empty 'node'"):
            validate_perception_placement_coherence(p)

    def test_node_origin_with_node_passes(self):
        p = PerceptionStagePlacement(stage="segmentation", origin=StageOrigin.NODE, node="gpu-box")
        validate_perception_placement_coherence(p)  # no raise

    @pytest.mark.parametrize(
        "origin",
        [StageOrigin.SELF, StageOrigin.SENSOR, StageOrigin.SUBSTRATE_OWNER],
    )
    def test_symbolic_origin_without_node_passes(self, origin):
        p = PerceptionStagePlacement(stage="capture", origin=origin)
        validate_perception_placement_coherence(p)  # no raise

    @pytest.mark.parametrize(
        "origin",
        [StageOrigin.SELF, StageOrigin.SENSOR, StageOrigin.SUBSTRATE_OWNER],
    )
    def test_symbolic_origin_with_node_rejected(self, origin):
        # Naming a node for a symbolic role is incoherent — it would shadow
        # the role's construction-time resolution.
        p = PerceptionStagePlacement(stage="capture", origin=origin, node="some-node")
        with pytest.raises(ValueError, match="must not name a 'node'"):
            validate_perception_placement_coherence(p)

    def test_where_label_appears_in_message(self):
        p = PerceptionStagePlacement(stage="dsp", origin=StageOrigin.NODE)
        with pytest.raises(ValueError, match="config.json: perception.audio"):
            validate_perception_placement_coherence(p, where="config.json: perception.audio.stages[1]")


class TestPerceptionStage:
    def test_default_origin_node_rejected(self):
        # A default cannot hardcode a concrete node (no node field on a stage).
        with pytest.raises(ValueError, match="must be a symbolic role"):
            PerceptionStage("segmentation", pinned=False, default_origin=StageOrigin.NODE)

    @pytest.mark.parametrize(
        "origin",
        [StageOrigin.SELF, StageOrigin.SENSOR, StageOrigin.SUBSTRATE_OWNER],
    )
    def test_symbolic_default_origin_accepted(self, origin):
        s = PerceptionStage("x", pinned=True, default_origin=origin)
        assert s.default_origin is origin


class TestCanonicalPipeline:
    def test_expected_stage_order(self):
        assert tuple(s.name for s in CANONICAL_PERCEPTION_PIPELINE) == (
            "capture",
            "dsp",
            "segmentation",
            "sensor_encode",
            "substrate",
            "cognition",
        )

    def test_pinned_placeable_structure(self):
        pinned = {s.name for s in CANONICAL_PERCEPTION_PIPELINE if s.pinned}
        placeable = {s.name for s in CANONICAL_PERCEPTION_PIPELINE if not s.pinned}
        # raw stream + sub-ms DSP at the sensor; substrate + cognition at the
        # single substrate owner — all pinned by physics. The GPU-heavy
        # middle is the movable part.
        assert pinned == {"capture", "dsp", "substrate", "cognition"}
        assert placeable == {"segmentation", "sensor_encode"}


class TestPipelineResolution:
    def test_unconfigured_resolves_to_defaults(self):
        # No overrides => every stage at its default_origin (all-symbolic,
        # all-local single-node => byte-identical to today's no-distribution).
        resolved = resolve_pipeline_placements()
        assert [(p.stage, p.origin, p.node) for p in resolved] == [
            ("capture", StageOrigin.SENSOR, None),
            ("dsp", StageOrigin.SENSOR, None),
            ("segmentation", StageOrigin.SELF, None),
            ("sensor_encode", StageOrigin.SELF, None),
            ("substrate", StageOrigin.SUBSTRATE_OWNER, None),
            ("cognition", StageOrigin.SUBSTRATE_OWNER, None),
        ]

    def test_none_overrides_equals_empty_overrides(self):
        assert resolve_pipeline_placements() == resolve_pipeline_placements(overrides={})

    def test_override_placeable_stage(self):
        ov = {"segmentation": PerceptionStagePlacement(stage="segmentation", origin=StageOrigin.NODE, node="gpu-box")}
        resolved = {p.stage: p for p in resolve_pipeline_placements(overrides=ov)}
        assert resolved["segmentation"].origin is StageOrigin.NODE
        assert resolved["segmentation"].node == "gpu-box"
        # untouched stages keep their defaults
        assert resolved["capture"].origin is StageOrigin.SENSOR

    def test_override_pinned_stage_fails_loud(self):
        ov = {"capture": PerceptionStagePlacement(stage="capture", origin=StageOrigin.NODE, node="elsewhere")}
        with pytest.raises(ValueError, match="pinned by physics and cannot"):
            resolve_pipeline_placements(overrides=ov)

    def test_override_unknown_stage_fails_loud(self):
        ov = {"nope": PerceptionStagePlacement(stage="nope", origin=StageOrigin.SELF)}
        with pytest.raises(ValueError, match="unknown stage"):
            resolve_pipeline_placements(overrides=ov)

    def test_override_key_value_mismatch_fails_loud(self):
        ov = {"segmentation": PerceptionStagePlacement(stage="sensor_encode", origin=StageOrigin.SELF)}
        with pytest.raises(ValueError, match="mismatched stage field"):
            resolve_pipeline_placements(overrides=ov)

    def test_override_incoherent_placement_fails_loud(self):
        # NODE origin without a node, on a placeable stage => coherence raises.
        ov = {"segmentation": PerceptionStagePlacement(stage="segmentation", origin=StageOrigin.NODE)}
        with pytest.raises(ValueError, match="requires a non-empty 'node'"):
            resolve_pipeline_placements(overrides=ov)
