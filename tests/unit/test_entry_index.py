"""The v2 entry index (src/maxim/hivemind/entry_index.py; docs/plans/oasis_entry_index_v2.md). Pure."""

from __future__ import annotations

import copy

import pytest

from maxim.utils.optional_deps import optional_dependency_available

pytestmark = pytest.mark.skipif(
    not optional_dependency_available("rfc8785"), reason="entry digests use RFC 8785 (the [sign] extra)"
)

S = "\x1f"


def _state(agent="donor_1"):
    nac = {
        "cluster_fear": {f"{agent}{S}n1{S}drive:oxygen": -0.5},
        "cluster_reward_bias": {f"{agent}{S}n1{S}tool:flee": 0.4, f"{agent}{S}orient{S}tool:turn": 0.2},
        "cluster_reward_source": {f"{agent}{S}n1{S}tool:flee": "relief"},
        "inherent_bias_keys": [f"{agent}{S}n1{S}tool:flee"],
        "percept_valences": {f"{agent}{S}zombie{S}drive:health": -0.3},
        "event_outcome_welford": {f"{agent}{S}tool:flee": {"n": 2}},
        "reward_bias": {f"{agent}:n1": 0.1},
        "links": {"tool:flee": []},
    }
    ec = {
        "n1": {
            "modality": "world",
            "embedding": [0.1, 0.2],
            "geometry": "g",
            "count": 3,
            "domain": None,
            "source": "donor",
            "contributors": ["donor"],
        }
    }
    return nac, ec


def _normalized():
    from maxim.hivemind.entry_index import normalize_agent_segment

    nac, ec = _state()
    normalized, dropped = normalize_agent_segment(nac)
    assert dropped == 0
    return normalized, ec


def test_normalization_rewrites_every_agent_segment():
    from maxim.hivemind.entry_index import AGENT_TOKEN

    nac, _ = _normalized()
    for field in (
        "cluster_fear",
        "cluster_reward_bias",
        "cluster_reward_source",
        "percept_valences",
        "event_outcome_welford",
    ):
        assert all(k.startswith(AGENT_TOKEN + S) for k in nac[field])
    assert nac["inherent_bias_keys"][0].startswith(AGENT_TOKEN + S)
    assert list(nac["reward_bias"]) == [f"{AGENT_TOKEN}:n1"]


def test_several_real_agents_need_an_explicit_own_agent_and_the_others_are_dropped():
    from maxim.hivemind.entry_index import EntryIndexError, normalize_agent_segment

    two, _ = _state()
    two["cluster_fear"][f"other{S}n2{S}drive:oxygen"] = -0.2
    two["event_outcome_welford"][f"other{S}tool:flee"] = {"n": 900}
    with pytest.raises(EntryIndexError, match="name which one is yours"):
        normalize_agent_segment(two)
    out, dropped = normalize_agent_segment(two, own_agent_id="donor_1")
    assert dropped == 2
    assert f"_agent{S}n2{S}drive:oxygen" not in out["cluster_fear"]
    assert out["event_outcome_welford"] == {f"_agent{S}tool:flee": {"n": 2}}  # own n=2, not the other's 900
    with pytest.raises(EntryIndexError, match="holds no rows"):
        normalize_agent_segment(two, own_agent_id="nobody")


def test_rows_an_ingested_release_left_under_the_token_are_dropped_never_merged():
    """The review probe: own -0.9 and a leftover token +0.5 on one key -- relabelling kept +0.5 (a
    sign flip inside a signed release). The token is never the exporter's own agent."""
    from maxim.hivemind.entry_index import normalize_agent_segment

    nac = {"percept_valences": {f"queen{S}wolf{S}bite": -0.9, f"_agent{S}wolf{S}bite": 0.5}}
    out, dropped = normalize_agent_segment(nac)
    assert dropped == 1 and out["percept_valences"] == {f"_agent{S}wolf{S}bite": -0.9}
    only_token, dropped = normalize_agent_segment({"percept_valences": {f"_agent{S}wolf{S}bite": 0.5}})
    assert dropped == 1 and only_token["percept_valences"] == {}


def test_entries_group_a_node_with_its_rows_and_allow_node_less_entries():
    from maxim.hivemind.entry_index import entries

    nac, ec = _normalized()
    got = entries(nac, ec)
    assert set(got) == {"n1", "orient"}
    assert got["orient"]["ec_node"] is None  # node-less (a NAc-only lineage)
    assert got["n1"]["nac"]["cluster_fear"] == {f"n1{S}drive:oxygen": -0.5}
    assert got["n1"]["inherent"] == [f"n1{S}tool:flee"]
    assert "source" not in got["n1"]["ec_node"]


def test_a_digest_ignores_provenance_but_not_content():
    from maxim.hivemind.entry_index import digest, entries

    nac, ec = _normalized()
    base = digest(entries(nac, ec)["n1"])
    ec2 = copy.deepcopy(ec)
    ec2["n1"]["source"], ec2["n1"]["contributors"] = "other-exporter", ["other-exporter"]
    assert digest(entries(nac, ec2)["n1"]) == base  # same entry, another exporter
    nac2 = copy.deepcopy(nac)
    nac2["cluster_fear"][f"_agent{S}n1{S}drive:oxygen"] = -0.6
    assert digest(entries(nac2, ec)["n1"]) != base


def test_the_index_round_trips_and_every_tamper_is_refused():
    from maxim.hivemind.entry_index import EntryIndexError, build_index, verify_index

    nac, ec = _normalized()
    index = build_index(nac, ec)
    assert [e["id"] for e in index["entries"]] == ["n1", "orient"]
    assert set(verify_index(index, nac, ec, max_entries=100)) == {"n1", "orient"}

    def refused(idx, n=nac, e=ec, cap=100):
        with pytest.raises(EntryIndexError):
            verify_index(idx, n, e, max_entries=cap)

    refused({**index, "version": 2})
    refused({**index, "entries": list(reversed(index["entries"]))})  # unsorted
    refused({**index, "entries": index["entries"] + index["entries"][-1:]})  # duplicate
    refused({**index, "entries": index["entries"][:1]})  # an unindexed entry
    refused({**index, "entries": [*index["entries"], {"id": "zz", "modality": None, "digest": "sha256:" + "0" * 64}]})
    refused({**index, "entries": [{**index["entries"][0], "digest": "sha256:" + "0" * 64}, index["entries"][1]]})
    refused({**index, "entries": [{**index["entries"][0], "modality": "audio"}, index["entries"][1]]})
    refused({**index, "entries": [{**index["entries"][0], "id": "bad id!"}, index["entries"][1]]})
    refused({**index, "entries": [{**index["entries"][0], "digest": "md5:x"}, index["entries"][1]]})
    refused(index, cap=1)
    raw_agent, _ = _state()  # an agent segment other than the token
    refused(build_index(nac, ec), n=raw_agent)


def test_malformed_situation_state_cannot_be_indexed():
    from maxim.hivemind.entry_index import EntryIndexError, build_index

    nac, ec = _normalized()
    bad = copy.deepcopy(nac)
    bad["cluster_reward_source"]["_agent-only-two"] = "relief"  # not a triple
    with pytest.raises(EntryIndexError, match="not agent"):
        build_index(bad, ec)
    dangling = copy.deepcopy(nac)
    dangling["inherent_bias_keys"] = [f"_agent{S}n1{S}tool:ghost"]
    with pytest.raises(EntryIndexError, match="dangling"):
        build_index(dangling, ec)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "\ud800"])
def test_jcs_refuses_what_it_cannot_represent_instead_of_crashing(value):
    from maxim.hivemind.entry_index import EntryIndexError, jcs

    with pytest.raises(EntryIndexError):
        jcs({"x": value})


def test_jcs_uses_ecmascript_numbers():
    from maxim.hivemind.entry_index import jcs

    assert jcs({"b": 1.0, "a": [1e21, 0.1]}) == b'{"a":[1e+21,0.1],"b":1}'


@pytest.mark.parametrize(
    ("nac", "ec"),
    [
        ({"cluster_fear": [1]}, {}),  # a field of the wrong type
        ({"inherent_bias_keys": 5}, {}),  # iterating it would raise TypeError
        ({}, {"n1": ["not", "a", "node"]}),  # a node of the wrong type
        ({}, ["n1"]),
    ],
)
def test_entries_refuse_a_hostile_shape_as_an_index_error_not_a_crash(nac, ec):
    from maxim.hivemind.entry_index import EntryIndexError, entries

    with pytest.raises(EntryIndexError):
        entries(nac, ec)
