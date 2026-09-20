"""The EC's match MARGIN — the evidence behind a pattern decision, not the decision.

`pattern_complete_or_separate` returns a node id and, on a completion, the similarity that
earned it. On a SEPARATION it reported `similarity=0.0`, and the scan itself returned -1.0 for
anything under threshold — so a percept that separated at 0.849 against a 0.85 threshold and one
that separated at 0.05 were indistinguishable, downstream and in every record. The decision was
kept and its evidence discarded (issue #786).

`best_similarity` carries that evidence on BOTH branches. It is read-only instrumentation: no
consumer decides on it, and a graded read AT the boundary is a mechanism that needs its own
experiment (Exp 62 Rung B).
"""

from __future__ import annotations

import math

import pytest

from maxim.similarity.ec import EntorhinalCortex


def _unit(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in vec))
    return [v / n for v in vec]


def _rotated(base: list[float], cos_target: float) -> list[float]:
    """A unit vector at exactly `cos_target` from `base` (base is e0-aligned)."""
    return _unit([cos_target, math.sqrt(max(0.0, 1.0 - cos_target**2))] + [0.0] * (len(base) - 2))


DIM = 8
BASE = _unit([1.0] + [0.0] * (DIM - 1))


def _ec() -> EntorhinalCortex:
    ec = EntorhinalCortex()
    r = ec.pattern_complete_or_separate(BASE, modality="world", threshold=0.85, geometry="g1")
    ec.register_substrate_node(r.node_id, BASE, "world")
    return ec


@pytest.mark.parametrize("cos_target", [0.84, 0.60, 0.05])
def test_a_separation_reports_the_score_it_separated_BY(cos_target: float) -> None:
    """The regression gate: these all reported 0.0 before, and were indistinguishable."""
    ec = _ec()
    probe = _rotated(BASE, cos_target)
    res = ec.pattern_complete_or_separate(probe, modality="world", threshold=0.85, geometry="g1")
    assert res.is_new is True, "must separate at this threshold"
    assert res.similarity == 0.0, "the existing field is unchanged — no consumer moves"
    assert res.best_similarity == pytest.approx(cos_target, abs=1e-6)


def test_a_near_miss_and_a_far_miss_are_now_distinguishable() -> None:
    """The defect, stated as a test: both separate, both report similarity 0.0."""
    ec_near, ec_far = _ec(), _ec()
    near = ec_near.pattern_complete_or_separate(_rotated(BASE, 0.849), modality="world", threshold=0.85, geometry="g1")
    far = ec_far.pattern_complete_or_separate(_rotated(BASE, 0.05), modality="world", threshold=0.85, geometry="g1")
    assert near.is_new and far.is_new
    assert near.similarity == far.similarity == 0.0, "indistinguishable on the old field"
    assert near.best_similarity != far.best_similarity
    assert near.best_similarity > 0.84 > far.best_similarity


def test_on_a_completion_the_margin_equals_the_similarity() -> None:
    ec = _ec()
    res = ec.pattern_complete_or_separate(_rotated(BASE, 0.95), modality="world", threshold=0.85, geometry="g1")
    assert res.is_new is False
    assert res.best_similarity == pytest.approx(res.similarity, abs=1e-9)


def test_an_empty_store_reports_minus_one_not_zero() -> None:
    """ "nothing to score against" must not read as "scored 0.0"."""
    res = EntorhinalCortex().pattern_complete_or_separate(BASE, modality="world", threshold=0.85, geometry="g1")
    assert res.is_new is True and res.best_similarity == -1.0


def test_an_incomparable_geometry_is_not_a_near_miss() -> None:
    """A near neighbour in ANOTHER encoding space is not a near miss — it is incomparable.

    The margin respects the geometry mask for the same reason the match does (gate 1 / D1). The
    node must be STAMPED first: an unstamped node is permissive by design and adopts the first
    live tag that touches it ("stamp on first touch"), so a freshly registered one is comparable
    to everything — which this test would otherwise mistake for a working mask.
    """
    ec = EntorhinalCortex()
    r = ec.pattern_complete_or_separate(BASE, modality="world", threshold=0.85, geometry="OTHER")
    ec.register_substrate_node(r.node_id, BASE, "world")
    stamped = ec.pattern_complete_or_separate(BASE, modality="world", threshold=0.85, geometry="OTHER")
    assert stamped.is_new is False, "the completion that stamps the node must land"
    res = ec.pattern_complete_or_separate(_rotated(BASE, 0.84), modality="world", threshold=0.85, geometry="g1")
    assert res.is_new is True
    assert res.best_similarity == -1.0, "an incomparable node must not be scored as a near miss"


def test_an_UNSTAMPED_node_is_comparable_and_does_score_as_a_near_miss() -> None:
    """The other half, pinned so the mask above is not mistaken for something it is not:
    an unstamped node matches any geometry, so it is a legitimate near miss."""
    ec = EntorhinalCortex()
    r = ec.pattern_complete_or_separate(BASE, modality="world", threshold=0.85, geometry="OTHER")
    ec.register_substrate_node(r.node_id, BASE, "world")
    res = ec.pattern_complete_or_separate(_rotated(BASE, 0.84), modality="world", threshold=0.85, geometry="g1")
    assert res.is_new is True and res.best_similarity == pytest.approx(0.84, abs=1e-6)


def test_the_margin_is_read_only_the_decision_is_unchanged() -> None:
    """Anti-regression: adding the field must not move which node anything resolves to."""
    ec = _ec()
    for cos_target, expect_new in ((0.95, False), (0.84, True)):
        res = ec.pattern_complete_or_separate(
            _rotated(BASE, cos_target), modality="world", threshold=0.85, geometry="g1"
        )
        assert res.is_new is expect_new


# ───────────────── the encoder's per-(agent, modality) accessor ─────────────────


def _encoder():
    from maxim.similarity.encoder import SensorEncoder

    return SensorEncoder(ec=EntorhinalCortex())


RANGES = {"x": (0.0, 1.0), "y": (0.0, 1.0)}


def test_the_margin_reaches_the_encoder_accessor_per_agent_and_modality() -> None:
    enc = _encoder()
    enc.encode_sensors(agent_id="t", sensors={"x": 1.0, "y": 0.0}, modality="world", ranges=RANGES)
    # nothing comparable in an empty store — NOT the same as "scored 0.0"
    assert enc.last_encode_margin(agent_id="t", modality="world") == -1.0
    enc.encode_sensors(agent_id="t", sensors={"x": 0.0, "y": 1.0}, modality="world", ranges=RANGES)
    m = enc.last_encode_margin(agent_id="t", modality="world")
    assert m is not None and 0.0 < m < 0.85, m
    # an untouched pair has no margin at all
    assert enc.last_encode_margin(agent_id="someone_else", modality="world") is None
    assert enc.last_encode_margin(agent_id="t", modality="interoception") is None


def test_a_min_delta_bypass_reports_NOT_MEASURED_rather_than_the_previous_scans_number() -> None:
    """The gate returns the cached node without scanning. Handing back the previous margin as if
    it described this call is the defect this whole issue is about, one layer up."""
    enc = _encoder()
    enc.encode_sensors(agent_id="t", sensors={"x": 0.0, "y": 1.0}, modality="world", ranges=RANGES)
    enc.encode_sensors(agent_id="t", sensors={"x": 1.0, "y": 0.0}, modality="world", ranges=RANGES)
    measured = enc.last_encode_margin(agent_id="t", modality="world")
    assert measured is not None
    node = enc.encode_sensors(agent_id="t", sensors={"x": 1.0, "y": 0.0}, modality="world", ranges=RANGES)
    assert node is not None, "the cached node is still returned"
    assert enc.last_encode_margin(agent_id="t", modality="world") is None, "no scan ran → no margin"
