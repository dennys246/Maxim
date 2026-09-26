"""#914: one fold for cluster-keyed NAc rows at both seams (merge.fold_cluster_rows).

When several donor clusters align onto ONE receiver cluster (ingest's re-key), or two tool signatures
scrub to one (the export scrub), their rows land on one key. They FOLD with the merge layer's own
semantics -- bias mean, fear min, source common-or-"mixed" -- and the inherent (safety-floor) marker
survives only when every folded bias row was inherent. Before: ingest kept whichever row came LAST
and kept the marker if ANY source was inherent, so a learned bias could overwrite an inherent one and
stay decay-exempt."""

from __future__ import annotations

from pathlib import Path

import pytest

S = "\x1f"


def _state(**over):
    state = {
        "cluster_reward_bias": {f"a{S}d1{S}tool:flee": 0.8, f"a{S}d2{S}tool:flee": -0.4},
        "cluster_fear": {f"a{S}d1{S}drive:oxygen": -0.9, f"a{S}d2{S}drive:oxygen": -0.3},  # min is NOT last
        "cluster_reward_source": {f"a{S}d1{S}tool:flee": "operant", f"a{S}d2{S}tool:flee": "relief"},
        "inherent_bias_keys": [f"a{S}d1{S}tool:flee"],
    }
    state.update(over)
    return state


def test_rekey_folds_colliding_donor_clusters_with_the_merge_semantics():
    from maxim.hivemind.merge import rekey_nac_state

    out = rekey_nac_state(_state(), {"d1": "r1", "d2": "r1"}, to_agent_id="me")
    key, fear_key = f"me{S}r1{S}tool:flee", f"me{S}r1{S}drive:oxygen"
    assert out["cluster_reward_bias"] == {key: pytest.approx(0.2)}  # mean, not the last row (-0.4)
    assert out["cluster_fear"] == {fear_key: -0.9}  # the most aversive fear wins
    assert out["cluster_reward_source"] == {key: "mixed"}
    assert out["inherent_bias_keys"] == []  # a learned row folded in: the marker does not survive


def test_the_marker_survives_when_every_folded_row_was_inherent():
    from maxim.hivemind.merge import rekey_nac_state

    state = _state(inherent_bias_keys=[f"a{S}d1{S}tool:flee", f"a{S}d2{S}tool:flee"])
    out = rekey_nac_state(state, {"d1": "r1", "d2": "r1"}, to_agent_id="me")
    assert out["inherent_bias_keys"] == [f"me{S}r1{S}tool:flee"]


def test_distinct_clusters_are_untouched():
    from maxim.hivemind.merge import rekey_nac_state

    out = rekey_nac_state(_state(), {"d1": "r1", "d2": "r2"}, to_agent_id="me")
    assert out["cluster_reward_bias"] == {f"me{S}r1{S}tool:flee": 0.8, f"me{S}r2{S}tool:flee": -0.4}
    assert out["inherent_bias_keys"] == [f"me{S}r1{S}tool:flee"]


def test_ingest_folds_two_donor_situations_that_align_onto_one(tmp_path: Path):
    """The real seam: two donor EC nodes with the receiver's embedding both align onto its node."""
    from maxim.hivemind.ingest import IngestionJournal
    from tests.unit.test_hivemind_ingest import DONOR, _ingest, _nac_state, _node, _write_bundle

    donor_nac = _nac_state(
        cluster_reward_bias={f"donor{S}d1{S}tool:flee": 0.8, f"donor{S}d2{S}tool:flee": -0.4},
        inherent_bias_keys=[f"donor{S}d1{S}tool:flee"],
    )
    bundle = _write_bundle(tmp_path / "b.zip", nac_state=donor_nac, ec_nodes={"d1": _node(), "d2": _node()})
    report = _ingest(
        bundle,
        IngestionJournal(tmp_path / "j.json"),
        receiver_ec={"r1": _node()},
        inherent_trusted=frozenset({DONOR}),
        receiver_agent_id="me",
    )
    folded = {k: v for k, v in report.nac["cluster_reward_bias"].items() if k.startswith(f"me{S}")}
    assert len(folded) == 1, folded
    (value,) = folded.values()
    assert value != pytest.approx(-0.4) and value != pytest.approx(0.8)  # folded, not last- or first-write
    assert not [k for k in report.nac.get("inherent_bias_keys", []) if k in folded]


# ── review round: the third seam (substrate_merge's markers), dangling markers, a scrub that must not fail open ──


K = f"me{S}L-a{S}tool:flee"
D = f"donor{S}R-a{S}tool:flee"


def _ec(prefix: str) -> dict:
    return {f"{prefix}-a": {"embedding": [1.0, 0.0], "modality": "interoception", "count": 1, "source": prefix}}


@pytest.mark.parametrize(
    ("receiver", "donor", "marked"),
    [
        # receiver inherent + donor learned: the safety floor is the RECEIVER's -- no donor removes it
        ({"cluster_reward_bias": {K: -0.6}, "inherent_bias_keys": [K]}, {"cluster_reward_bias": {D: 0.4}}, True),
        # receiver learned + donor inherent: a Queen prior never makes the receiver's learned value exempt
        ({"cluster_reward_bias": {K: 0.6}}, {"cluster_reward_bias": {D: -0.2}, "inherent_bias_keys": [D]}, False),
        # both inherent: stays inherent
        (
            {"cluster_reward_bias": {K: 0.6}, "inherent_bias_keys": [K]},
            {"cluster_reward_bias": {D: -0.2}, "inherent_bias_keys": [D]},
            True,
        ),
        # receiver holds no row: the donor's inherent prior arrives marked
        ({}, {"cluster_reward_bias": {D: -0.2}, "inherent_bias_keys": [D]}, True),
        # a donor marker with no bias row of its own (dangling) never marks a receiver's learned bias
        ({"cluster_reward_bias": {K: 0.6}}, {"cluster_reward_bias": {}, "inherent_bias_keys": [D]}, False),
        # neither side holds a row: a marker on either side marks nothing
        ({"inherent_bias_keys": [K]}, {"inherent_bias_keys": [D]}, False),
    ],
)
def test_substrate_merge_markers_are_receiver_first(receiver, donor, marked):
    from maxim.hivemind.merge import substrate_merge

    result = substrate_merge(
        receiver_nac=receiver,
        receiver_ec=_ec("L"),
        donor_nac=donor,
        donor_ec=_ec("R"),
        receiver_source="recv",
        donor_source="donor",
        receiver_agent_id="me",
    )
    assert (K in result.nac["inherent_bias_keys"]) is marked
    assert set(result.nac["inherent_bias_keys"]) <= set(result.nac.get("cluster_reward_bias") or {})


def test_nac_merge_stays_commutative_on_markers():
    """The receiver-first rule lives in substrate_merge; the bare fold is a symmetric union."""
    from maxim.hivemind.merge import nac_merge

    a = {"cluster_reward_bias": {K: 0.6}, "inherent_bias_keys": [K]}
    b = {"cluster_reward_bias": {K: -0.2}}
    ab = nac_merge(left=a, left_source="x", right=b, right_source="y")
    ba = nac_merge(left=b, left_source="y", right=a, right_source="x")
    assert ab["inherent_bias_keys"] == ba["inherent_bias_keys"] == [K]


def test_a_dangling_marker_is_dropped_by_the_fold():
    from maxim.hivemind.merge import fold_cluster_rows

    state = {"cluster_reward_bias": {f"a{S}d1{S}tool:flee": 0.3}, "inherent_bias_keys": [f"a{S}d9{S}tool:ghost"]}
    out = fold_cluster_rows(state, lambda k: k, fields=("cluster_reward_bias",))
    assert out["inherent_bias_keys"] == []


def test_the_scrub_replaces_its_fields_and_never_fails_open():
    from maxim.hivemind.bundle import scrub_nac_state_for_bundle

    assert scrub_nac_state_for_bundle({"links": {}})["cluster_reward_bias"] == {}  # always present
    with pytest.raises(ValueError, match="inherent_bias_keys is not a list"):
        scrub_nac_state_for_bundle(
            {
                "links": {},
                "cluster_reward_bias": {f"a{S}c{S}tool:use:x y": 0.1},
                "inherent_bias_keys": (f"a{S}c{S}tool:use:x y",),
            }
        )
    with pytest.raises(ValueError, match="cluster_reward_source is not an object"):
        scrub_nac_state_for_bundle({"links": {}, "cluster_reward_source": [1]})
    # an absent (None) field is not a copy of the raw one: the scrubbed key never leaks through
    raw = f"a{S}c{S}tool:use:ps aux"
    out = scrub_nac_state_for_bundle(
        {"links": {}, "cluster_reward_bias": {raw: 0.1}, "cluster_reward_source": None, "inherent_bias_keys": [raw]}
    )
    assert raw not in out["cluster_reward_bias"] and raw not in out.get("inherent_bias_keys", [])
    # a None bias field must not survive as a raw None (the pop, not setdefault, is what replaces it)
    assert scrub_nac_state_for_bundle({"links": {}, "cluster_reward_bias": None})["cluster_reward_bias"] == {}


def test_admission_ignores_a_dangling_donor_marker_directly():
    """Via substrate_merge the re-key fold already drops a dangling donor marker; the admission rule
    must not depend on that upstream step."""
    from maxim.hivemind.merge import _admit_inherent_markers

    assert _admit_inherent_markers({}, {"cluster_reward_bias": {}, "inherent_bias_keys": [K]}) == []
