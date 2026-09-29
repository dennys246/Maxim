"""#812 -- reinforcing one typed ATL relation never touches another on the same concept pair.

`Semantics.define` maps every relation type except CAUSES onto `EdgeType.ASSOCIATES`, with the real type in
edge metadata. `Semantics.update_edge` FOUND the right edge by that metadata, then asked
`DependencyGraph.update_edge` to apply the update, which wrote to the FIRST `(source, target, edge_type)`
match and ignored metadata. So on a pair holding both IS_A and PROPERTY_OF, reinforcing PROPERTY_OF could
bump IS_A instead; the reverse update of a symmetric relation had the same flaw.
"""

from __future__ import annotations

from maxim.agents.bus import DependencyGraph, EdgeType
from maxim.memory.semantic_types import RelationshipRegistry
from maxim.memory.semantics import Semantics


def _semantics() -> Semantics:
    return Semantics(DependencyGraph(), RelationshipRegistry())


def _edge(sem: Semantics, a: str, b: str, rel: str):
    return sem._graph.find_edge(a, b, EdgeType.ASSOCIATES, metadata_match={"relationship_type": rel})


def test_reinforcing_one_relation_leaves_the_other_on_the_pair_alone() -> None:
    sem = _semantics()
    assert sem.define("wolf", "animal", "IS_A", weight=1.0, confidence=0.5)  # defined FIRST: the first match
    assert sem.define("wolf", "animal", "PROPERTY_OF", weight=1.0, confidence=0.5)

    assert sem.update_edge("wolf", "animal", "PROPERTY_OF", weight=3.0, confidence_delta=0.2)

    prop, is_a = _edge(sem, "wolf", "animal", "PROPERTY_OF"), _edge(sem, "wolf", "animal", "IS_A")
    assert prop.weight == 3.0 and abs(prop.metadata["confidence"] - 0.7) < 1e-9
    assert is_a.weight == 1.0 and is_a.metadata["confidence"] == 0.5  # untouched


def test_a_symmetric_relations_reverse_update_lands_on_its_own_edge() -> None:
    """The reverse (target -> source) update of a symmetric type had the same first-match flaw."""
    sem = _semantics()
    assert sem.define("b", "a", "IS_A", weight=1.0)  # a REVERSE-direction edge of another type, first
    assert sem.define("a", "b", "RELATED_TO", weight=1.0)  # symmetric: a->b and b->a

    assert sem.update_edge("a", "b", "RELATED_TO", weight=5.0)

    assert _edge(sem, "b", "a", "RELATED_TO").weight == 5.0
    assert _edge(sem, "b", "a", "IS_A").weight == 1.0  # untouched


def test_the_graph_update_can_be_scoped_by_metadata() -> None:
    graph = DependencyGraph()
    graph.add_node("x", "x")
    graph.add_node("y", "y")
    graph.add_edge("x", "y", EdgeType.ASSOCIATES, 1.0, {"kind": "first"})
    graph.add_edge("x", "y", EdgeType.ASSOCIATES, 1.0, {"kind": "second"})

    assert graph.update_edge("x", "y", EdgeType.ASSOCIATES, weight=2.0, metadata_match={"kind": "second"})
    assert graph.find_edge("x", "y", metadata_match={"kind": "second"}).weight == 2.0
    assert graph.find_edge("x", "y", metadata_match={"kind": "first"}).weight == 1.0
    assert not graph.update_edge("x", "y", EdgeType.ASSOCIATES, weight=9.0, metadata_match={"kind": "absent"})


def test_caller_metadata_cannot_overwrite_the_relation_type() -> None:
    """The type key is the edge's identity for updates; a metadata `relationship_type` must not replace it."""
    sem = _semantics()
    assert sem.define("a", "b", "IS_A", metadata={"relationship_type": "PROPERTY_OF"})
    assert _edge(sem, "a", "b", "IS_A") is not None
    assert _edge(sem, "a", "b", "PROPERTY_OF") is None
