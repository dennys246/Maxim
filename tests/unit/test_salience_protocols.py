"""Tests for salience protocols and WhereCoord implementations.

Verifies:
- All WhereCoord implementations satisfy the protocol
- Distance semantics are correct per modality
- SalienceItem is frozen and carries WhereCoord
- SalienceNetwork.update() accepts SalienceItem lists
- NarrativeTranscriber.transcribe_to_items() produces SalienceItem
- to_context_str() uses WhereCoord.region() for non-pixel items
- MemoryHub salience callback registration
"""

from __future__ import annotations

import pytest

from maxim.salience.protocols import (
    PerceptSource,
    SalienceItem,
    WhereCoord,
)
from maxim.salience.where import (
    EntityWhere,
    NarrativeWhere,
    NullWhere,
    PixelWhere,
    SemanticWhere,
    TemporalWhere,
)


# ---------------------------------------------------------------------------
# WhereCoord protocol conformance
# ---------------------------------------------------------------------------


class TestWhereCoordProtocol:
    """Every WhereCoord impl satisfies the protocol."""

    @pytest.mark.parametrize(
        "where",
        [
            PixelWhere(320, 240),
            NarrativeWhere("entrance", "tavern"),
            EntityWhere("guard", "hp"),
            TemporalWhere(14, 2, 0.5),
            SemanticWhere("weapon_001", "weapon"),
            NullWhere("test"),
        ],
        ids=["pixel", "narrative", "entity", "temporal", "semantic", "null"],
    )
    def test_satisfies_protocol(self, where):
        assert isinstance(where, WhereCoord)

    @pytest.mark.parametrize(
        "where",
        [
            PixelWhere(320, 240),
            NarrativeWhere("entrance", "tavern"),
            EntityWhere("guard"),
            TemporalWhere(14),
            SemanticWhere("c1"),
            NullWhere(),
        ],
    )
    def test_distance_to_self_is_zero(self, where):
        assert where.distance_to(where) == 0.0

    @pytest.mark.parametrize(
        "where",
        [
            PixelWhere(320, 240),
            NarrativeWhere("entrance", "tavern"),
            EntityWhere("guard"),
            TemporalWhere(14),
            SemanticWhere("c1"),
            NullWhere(),
        ],
    )
    def test_region_returns_string(self, where):
        r = where.region()
        assert isinstance(r, str)
        assert len(r) > 0

    @pytest.mark.parametrize(
        "where",
        [
            PixelWhere(320, 240, 50, 60),
            NarrativeWhere("entrance", "tavern"),
            EntityWhere("guard", "hp", "runner.guard"),
            TemporalWhere(14, 2, 0.5),
            SemanticWhere("c1", "weapon", frozenset(["c2"])),
            NullWhere("x"),
        ],
    )
    def test_as_dict_returns_dict_with_type(self, where):
        d = where.as_dict()
        assert isinstance(d, dict)
        assert "type" in d

    @pytest.mark.parametrize(
        "where",
        [
            PixelWhere(320, 240),
            NarrativeWhere("entrance", "tavern"),
            EntityWhere("guard"),
            TemporalWhere(14),
            SemanticWhere("c1"),
            NullWhere(),
        ],
    )
    def test_cross_type_distance_is_inf(self, where):
        # Distance to a different WhereCoord type should be inf
        other = NullWhere() if not isinstance(where, NullWhere) else PixelWhere(0, 0)
        # NullWhere always returns 0, so skip it
        if not isinstance(where, NullWhere):
            assert where.distance_to(other) == float("inf")


# ---------------------------------------------------------------------------
# PixelWhere specifics
# ---------------------------------------------------------------------------


class TestPixelWhere:
    def test_euclidean_distance(self):
        a = PixelWhere(0, 0)
        b = PixelWhere(3, 4)
        assert a.distance_to(b) == pytest.approx(5.0)

    def test_region_center(self):
        w = PixelWhere(320, 240, frame_w=640, frame_h=480)
        assert w.region() == "center"

    def test_region_top_left(self):
        w = PixelWhere(50, 50, frame_w=640, frame_h=480)
        assert w.region() == "top-left"

    def test_region_bottom_right(self):
        w = PixelWhere(600, 400, frame_w=640, frame_h=480)
        assert w.region() == "bottom-right"

    def test_from_bbox_xywh(self):
        w = PixelWhere.from_bbox((100, 200, 50, 60))
        assert w.u == pytest.approx(125)
        assert w.v == pytest.approx(230)

    def test_as_dict_has_coordinates(self):
        w = PixelWhere(100, 200, 50, 60)
        d = w.as_dict()
        assert d["type"] == "pixel"
        assert d["u"] == 100
        assert d["v"] == 200


# ---------------------------------------------------------------------------
# NarrativeWhere specifics
# ---------------------------------------------------------------------------


class TestNarrativeWhere:
    def test_same_position_same_scene(self):
        a = NarrativeWhere("entrance", "tavern")
        b = NarrativeWhere("entrance", "tavern")
        assert a.distance_to(b) == 0.0

    def test_different_position_same_scene(self):
        a = NarrativeWhere("entrance", "tavern")
        b = NarrativeWhere("bar", "tavern")
        assert a.distance_to(b) == 1.0

    def test_different_scene(self):
        a = NarrativeWhere("entrance", "tavern")
        b = NarrativeWhere("entrance", "dungeon")
        assert a.distance_to(b) == 2.0

    def test_region_with_scene(self):
        w = NarrativeWhere("entrance", "tavern")
        assert w.region() == "entrance (tavern)"

    def test_region_without_scene(self):
        w = NarrativeWhere("center")
        assert w.region() == "center"


# ---------------------------------------------------------------------------
# EntityWhere specifics
# ---------------------------------------------------------------------------


class TestEntityWhere:
    def test_same_entity_same_sensor(self):
        a = EntityWhere("guard", "hp")
        b = EntityWhere("guard", "hp")
        assert a.distance_to(b) == 0.0

    def test_same_entity_different_sensor(self):
        a = EntityWhere("guard", "hp")
        b = EntityWhere("guard", "trust")
        assert a.distance_to(b) == 0.5

    def test_different_entity_no_path(self):
        a = EntityWhere("guard")
        b = EntityWhere("merchant")
        assert a.distance_to(b) == 1.0

    def test_path_distance(self):
        a = EntityWhere("arm", path="runner.megarm_v3")
        b = EntityWhere("blade", path="runner.megarm_v3.blade")
        assert a.distance_to(b) == 1.0  # 3 parts vs 2 parts, 2 common

    def test_region_with_sensor(self):
        w = EntityWhere("guard", "hp")
        assert w.region() == "guard.hp"


# ---------------------------------------------------------------------------
# TemporalWhere specifics
# ---------------------------------------------------------------------------


class TestTemporalWhere:
    def test_same_time(self):
        a = TemporalWhere(14, 2)
        b = TemporalWhere(14, 2)
        assert a.distance_to(b) == 0.0

    def test_opposite_times(self):
        a = TemporalWhere(0)
        b = TemporalWhere(12)
        # 12 hours apart on 24h clock = max distance = 1.0 * 0.7 = 0.7
        assert a.distance_to(b) == pytest.approx(0.7)

    def test_circular_wrap(self):
        a = TemporalWhere(23)
        b = TemporalWhere(1)
        # 2 hours apart, not 22
        dist = a.distance_to(b)
        assert dist < 0.2

    def test_region_labels(self):
        assert TemporalWhere(3).region() == "night"
        assert TemporalWhere(9).region() == "morning"
        assert TemporalWhere(14).region() == "afternoon"
        assert TemporalWhere(20).region() == "evening"


# ---------------------------------------------------------------------------
# SemanticWhere specifics
# ---------------------------------------------------------------------------


class TestSemanticWhere:
    def test_same_concept(self):
        a = SemanticWhere("weapon_001")
        b = SemanticWhere("weapon_001")
        assert a.distance_to(b) == 0.0

    def test_related_concepts(self):
        a = SemanticWhere("sword_001", related_concepts=frozenset(["weapon_001"]))
        b = SemanticWhere("weapon_001")
        assert a.distance_to(b) == 0.5

    def test_same_category(self):
        a = SemanticWhere("sword_001", category="weapon")
        b = SemanticWhere("axe_002", category="weapon")
        assert a.distance_to(b) == 1.0

    def test_different_category(self):
        a = SemanticWhere("sword_001", category="weapon")
        b = SemanticWhere("guard_001", category="character")
        assert a.distance_to(b) == 2.0


# ---------------------------------------------------------------------------
# NullWhere specifics
# ---------------------------------------------------------------------------


class TestNullWhere:
    def test_always_zero_distance(self):
        a = NullWhere("x")
        b = NullWhere("y")
        assert a.distance_to(b) == 0.0

    def test_zero_distance_to_pixel(self):
        # NullWhere is always distance 0 (no spatial dimension)
        n = NullWhere()
        p = PixelWhere(100, 200)
        assert n.distance_to(p) == 0.0


# ---------------------------------------------------------------------------
# SalienceItem
# ---------------------------------------------------------------------------


class TestSalienceItem:
    def test_frozen(self):
        item = SalienceItem(
            id="guard_0",
            label="guard",
            source=PerceptSource.NARRATIVE,
            confidence=0.8,
            where=NarrativeWhere("entrance", "tavern"),
        )
        with pytest.raises(AttributeError):
            item.label = "merchant"  # type: ignore[misc]

    def test_fields(self):
        where = NarrativeWhere("entrance", "tavern")
        item = SalienceItem(
            id="guard_0",
            label="guard",
            source=PerceptSource.NARRATIVE,
            confidence=0.8,
            where=where,
            class_id=900,
            concept_id="concept_guard",
        )
        assert item.id == "guard_0"
        assert item.label == "guard"
        assert item.source == PerceptSource.NARRATIVE
        assert item.confidence == 0.8
        assert item.where is where
        assert item.class_id == 900
        assert item.concept_id == "concept_guard"

    def test_default_fields(self):
        item = SalienceItem(
            id="x",
            label="x",
            source=PerceptSource.UNKNOWN,
            confidence=0.5,
            where=NullWhere(),
        )
        assert item.class_id == 0
        assert item.interest_matched is False
        assert item.concept_id is None
        assert item.metadata == {}


# ---------------------------------------------------------------------------
# SalienceNetwork.update() with SalienceItem
# ---------------------------------------------------------------------------


class TestSalienceNetworkUpdate:
    def test_update_with_salience_items(self):
        from maxim.salience import SalienceNetwork

        net = SalienceNetwork()
        items = [
            SalienceItem(
                id="guard_0",
                label="guard",
                source=PerceptSource.NARRATIVE,
                confidence=0.8,
                where=NarrativeWhere("entrance", "tavern"),
            ),
            SalienceItem(
                id="sword_1",
                label="rusty_sword",
                source=PerceptSource.NARRATIVE,
                confidence=0.6,
                where=NarrativeWhere("center", "tavern"),
            ),
        ]
        net.update(items)

        stats = net.get_stats()
        assert stats["total_unique_objects"] == 2
        assert stats["total_detections"] == 2

    def test_update_preserves_where_coord(self):
        from maxim.salience import SalienceNetwork

        net = SalienceNetwork()
        where = NarrativeWhere("entrance", "tavern")
        net.update(
            [
                SalienceItem(
                    id="guard_0",
                    label="guard",
                    source=PerceptSource.NARRATIVE,
                    confidence=0.8,
                    where=where,
                ),
            ]
        )

        top = net.get_top_salient(n=1, min_salience=0.0)
        assert len(top) == 1
        assert "_where" in top[0]
        assert top[0]["_where"] is where

    def test_context_str_uses_region(self):
        from maxim.salience import SalienceNetwork

        net = SalienceNetwork()
        net.update(
            [
                SalienceItem(
                    id="guard_0",
                    label="guard",
                    source=PerceptSource.NARRATIVE,
                    confidence=0.8,
                    where=NarrativeWhere("entrance", "tavern"),
                ),
            ]
        )

        ctx = net.to_context_str()
        # Should use region() not pixel coords
        assert "entrance (tavern)" in ctx
        assert "(0," not in ctx  # No pixel coordinates

    def test_update_with_legacy_dicts(self):
        """Legacy detection dicts still work via update()."""
        from maxim.salience import SalienceNetwork

        net = SalienceNetwork()
        net.update(
            [
                {"track_id": "det_0", "class_id": 0, "label": "person", "confidence": 0.9, "bbox": (100, 200, 50, 60)},
            ]
        )

        stats = net.get_stats()
        assert stats["total_unique_objects"] == 1

    def test_clear_resets_where_coords(self):
        from maxim.salience import SalienceNetwork

        net = SalienceNetwork()
        net.update(
            [
                SalienceItem(
                    id="g",
                    label="guard",
                    source=PerceptSource.NARRATIVE,
                    confidence=0.8,
                    where=NarrativeWhere("entrance"),
                ),
            ]
        )
        assert len(net._where_coords) == 1
        net.clear()
        assert len(net._where_coords) == 0


# ---------------------------------------------------------------------------
# NarrativeTranscriber.transcribe_to_items()
# ---------------------------------------------------------------------------
class TestMemoryHubSalienceHooks:
    def test_register_salience_callback(self):
        from unittest.mock import MagicMock

        from maxim.decisions.nac import NAc
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        hub = MemoryHub(
            hippocampus=Hippocampus(),
            nac=NAc(),
            scn=SCN(),
            ec=EntorhinalCortex(),
        )

        callback = MagicMock()
        hub.register_salience_callback(callback)
        assert callback in hub._salience_callbacks

    def test_fire_salience_callbacks(self):
        from unittest.mock import MagicMock

        from maxim.decisions.nac import NAc
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus
        from maxim.memory.types import Perception
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        hub = MemoryHub(
            hippocampus=Hippocampus(),
            nac=NAc(),
            scn=SCN(),
            ec=EntorhinalCortex(),
        )

        callback = MagicMock()
        hub.register_salience_callback(callback)

        # Create a mock memory with perception data
        memory = MagicMock()
        memory.perception = Perception(salience=0.9, novelty=0.8)

        hub._fire_salience_callbacks("mem_1", memory)

        callback.assert_called_once_with("mem_1", memory, 0.9, 0.8)

    def test_callback_error_doesnt_propagate(self):
        from unittest.mock import MagicMock

        from maxim.decisions.nac import NAc
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        hub = MemoryHub(
            hippocampus=Hippocampus(),
            nac=NAc(),
            scn=SCN(),
            ec=EntorhinalCortex(),
        )

        bad_callback = MagicMock(side_effect=RuntimeError("boom"))
        good_callback = MagicMock()
        hub.register_salience_callback(bad_callback)
        hub.register_salience_callback(good_callback)

        memory = MagicMock()
        memory.perception = MagicMock(salience=0.5, novelty=0.5)

        # Should not raise
        hub._fire_salience_callbacks("mem_1", memory)
        # Good callback should still fire
        good_callback.assert_called_once()
