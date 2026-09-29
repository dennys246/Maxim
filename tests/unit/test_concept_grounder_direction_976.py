"""#976 -- the concept grounder updates each typed relation once, in the direction it is stored.

Both grounding paths (the sync ``_modulate_relationships`` and the pooled ``_compute_relationship_updates``
→ ``_apply_updates``) iterated ``find_by_relationship(direction="both")``, which returns incoming edges
too, and always updated ``(concept.id, other_id, type)``:
- a SYMMETRIC relation came back twice (outgoing and incoming) and ``Semantics.update_edge`` already
  keeps both directions in sync, so each pass moved its confidence by TWO deltas;
- an INCOMING non-symmetric relation resolved to a forward edge that does not exist, so it was silently
  never grounded.
"""

from __future__ import annotations

import pytest

from maxim.math.angular_gyrus import AngularGyrus, AngularGyrusConfig
from maxim.math.ips import IPS
from maxim.memory.atl import ATL, ATLConfig
from maxim.memory.concept_grounder import ConceptGrounder
from maxim.memory.cross_layer import CrossLayerGraph
from maxim.memory.semantic_types import Concept

STRENGTHEN = 0.05
WEAKEN = -0.1


@pytest.fixture
def atl():
    return ATL(ATLConfig(persistence_path=None))


@pytest.fixture
def grounder(atl):
    ag = AngularGyrus(AngularGyrusConfig(persistence_path=None))
    return ConceptGrounder(
        atl=atl, angular_gyrus=ag, ips=IPS(), cross_layer=CrossLayerGraph(layers={"atl": atl, "angular_gyrus": ag})
    )


def _concept(atl, name: str, category: str = "object") -> Concept:
    cid, _ = atl.find_or_create(name=name, category=category)
    concept = atl.get(cid)
    assert isinstance(concept, Concept)
    return concept


def _co_occur(a: Concept, b: Concept, n: int = 5) -> None:
    """High Jaccard with enough shared evidence: a strengthen."""
    for i in range(n):
        a.add_ref("hippocampus", f"ep-{i}")
        b.add_ref("hippocampus", f"ep-{i}")


def _apart(a: Concept, b: Concept) -> None:
    """Low Jaccard over many observations: a weaken."""
    for i in range(12):
        a.add_ref("hippocampus", f"ep-a-{i}")
        b.add_ref("hippocampus", f"ep-b-{i}")


def _confidence(atl, source: str, target: str, rel_type: str) -> float:
    for other, rel in atl.find_by_relationship(source, rel_type=rel_type, direction="outgoing"):
        if other == target:
            return rel.confidence
    raise AssertionError(f"no {rel_type} edge {source} -> {target}")


def _ground_sync(grounder, concept) -> None:
    grounder._modulate_relationships(concept)


def _ground_pooled(grounder, concept) -> None:
    grounder._apply_updates(concept.id, grounder._compute_relationship_updates(concept), [])


PATHS = pytest.mark.parametrize("ground", [_ground_sync, _ground_pooled], ids=["sync", "pooled"])


@PATHS
@pytest.mark.parametrize("delta, setup", [(STRENGTHEN, _co_occur), (WEAKEN, _apart)], ids=["strengthen", "weaken"])
def test_a_symmetric_relation_moves_by_exactly_one_delta(grounder, atl, ground, delta, setup) -> None:
    a, b = _concept(atl, "mug"), _concept(atl, "kitchen", "location")
    setup(a, b)
    assert atl.semantics.registry.is_symmetric("RELATED_TO")
    atl.define_relationship(a.id, b.id, "RELATED_TO", weight=0.3, confidence=0.5)
    ground(grounder, a)
    assert _confidence(atl, a.id, b.id, "RELATED_TO") == pytest.approx(0.5 + delta)
    assert _confidence(atl, b.id, a.id, "RELATED_TO") == pytest.approx(0.5 + delta)  # kept in sync


@PATHS
@pytest.mark.parametrize("delta, setup", [(STRENGTHEN, _co_occur), (WEAKEN, _apart)], ids=["strengthen", "weaken"])
def test_an_incoming_non_symmetric_relation_is_grounded(grounder, atl, ground, delta, setup) -> None:
    """``b HAS_PART a``, grounded from ``a``: the edge is stored b -> a, and that is the one updated."""
    a, b = _concept(atl, "handle"), _concept(atl, "mug")
    setup(a, b)
    assert not atl.semantics.registry.is_symmetric("HAS_PART")
    atl.define_relationship(b.id, a.id, "HAS_PART", weight=0.3, confidence=0.5)
    ground(grounder, a)
    assert _confidence(atl, b.id, a.id, "HAS_PART") == pytest.approx(0.5 + delta)


@PATHS
def test_an_outgoing_non_symmetric_relation_is_grounded_once(grounder, atl, ground) -> None:
    a, b = _concept(atl, "mug"), _concept(atl, "handle")
    _co_occur(a, b)
    atl.define_relationship(a.id, b.id, "HAS_PART", weight=0.3, confidence=0.5)
    ground(grounder, a)
    assert _confidence(atl, a.id, b.id, "HAS_PART") == pytest.approx(0.5 + STRENGTHEN)


def test_the_pooled_updates_name_each_relation_once_in_its_stored_direction(grounder, atl) -> None:
    a, b, c = _concept(atl, "mug"), _concept(atl, "kitchen", "location"), _concept(atl, "handle")
    _co_occur(a, b)
    _co_occur(a, c)
    atl.define_relationship(a.id, b.id, "RELATED_TO", weight=0.3, confidence=0.5)
    atl.define_relationship(c.id, a.id, "HAS_PART", weight=0.3, confidence=0.5)
    updates = grounder._compute_relationship_updates(a)
    keys = [(s, t, r) for s, t, r, _w, _d in updates]
    assert len(keys) == len(set(keys)) == 2
    assert (c.id, a.id, "HAS_PART") in keys
    assert sum(1 for s, t, r in keys if r == "RELATED_TO" and {s, t} == {a.id, b.id}) == 1


@PATHS
def test_opposite_direction_relations_of_one_type_are_each_grounded_once(grounder, atl, ground) -> None:
    """``a HAS_PART b`` and ``b HAS_PART a``: two stored edges, each updated once."""
    a, b = _concept(atl, "left"), _concept(atl, "right")
    _co_occur(a, b)
    atl.define_relationship(a.id, b.id, "HAS_PART", weight=0.3, confidence=0.5)
    atl.define_relationship(b.id, a.id, "HAS_PART", weight=0.3, confidence=0.5)
    ground(grounder, a)
    assert _confidence(atl, a.id, b.id, "HAS_PART") == pytest.approx(0.5 + STRENGTHEN)
    assert _confidence(atl, b.id, a.id, "HAS_PART") == pytest.approx(0.5 + STRENGTHEN)


@PATHS
def test_two_types_on_one_pair_are_each_grounded_once(grounder, atl, ground) -> None:
    a, b = _concept(atl, "mug"), _concept(atl, "handle")
    _co_occur(a, b)
    atl.define_relationship(a.id, b.id, "RELATED_TO", weight=0.3, confidence=0.5)
    atl.define_relationship(a.id, b.id, "HAS_PART", weight=0.3, confidence=0.5)
    ground(grounder, a)
    assert _confidence(atl, a.id, b.id, "RELATED_TO") == pytest.approx(0.5 + STRENGTHEN)
    assert _confidence(atl, a.id, b.id, "HAS_PART") == pytest.approx(0.5 + STRENGTHEN)


@PATHS
def test_a_newly_grounded_incoming_edge_gets_its_weight_too(grounder, atl, ground) -> None:
    """Outgoing edges always had their weight set from Jaccard; an incoming one now does as well."""
    a, b = _concept(atl, "handle"), _concept(atl, "mug")
    _co_occur(a, b)  # identical refs: Jaccard 1.0 -> weight min(1, 1.0 * scale) = 1.0
    atl.define_relationship(b.id, a.id, "HAS_PART", weight=0.3, confidence=0.5)
    ground(grounder, a)
    weight = next(rel.weight for other, rel in atl.find_by_relationship(b.id, "HAS_PART", "outgoing") if other == a.id)
    assert weight == pytest.approx(min(1.0, 1.0 * grounder.JACCARD_WEIGHT_SCALE))
