"""#913: links pair on NAc's own identity, and an ingested link is readable by the receiver.

Before: ``merge._merge_link_lists`` paired links by ``outcome_signature`` alone, so a receiver's links
that differ only in ``event_context`` (two links to NAc) overwrote each other on EVERY merge -- a real
state went 607 -> 443 links against an EMPTY donor. And ``rekey_nac_state`` re-keyed a link's
``event_context.agent_id`` only from the release token, so a link from an unsigned bundle kept the
donor's id and ``NAc.predict`` (which matches link context against the reader's own) never used it.
"""

from __future__ import annotations

import copy

from tests.unit.test_hivemind_ingest import _link

S = "\x1f"


def _ctx_link(link_id: str, context: dict, **over):
    link = _link(**over)
    link["id"] = link_id
    link["event_context"] = context
    return link


def _merge(left_links, right_links):
    from maxim.hivemind.merge import nac_merge

    out = nac_merge(
        left={"links": {"tool:probe": left_links}},
        left_source="local",
        right={"links": {"tool:probe": right_links} if right_links else {}},
        right_source="peer",
    )
    return out["links"]["tool:probe"]


def test_an_empty_donor_merge_is_the_identity_on_the_receivers_links():
    """The issue's reproduction, plus a within-receiver duplicate identity: nothing is dropped."""
    cold = _ctx_link("l1", {"agent_id": "me", "goal": "cold"})
    hungry = _ctx_link("l2", {"agent_id": "me", "goal": "hungry"})
    negative = _ctx_link("l3", {"agent_id": "me", "goal": "cold"}, valence="negative")
    duplicate = _ctx_link("l4", {"agent_id": "me", "goal": "cold"})  # same identity as l1
    left = [cold, hungry, negative, duplicate]
    before = copy.deepcopy(left)

    assert _merge(left, []) == before
    assert left == before  # the input is not mutated


def test_a_donor_link_pairs_only_with_the_receiver_link_of_its_own_context():
    cold = _ctx_link("l1", {"agent_id": "me", "goal": "cold"})
    hungry = _ctx_link("l2", {"agent_id": "me", "goal": "hungry"})
    donor = _ctx_link("d1", {"goal": "hungry", "agent_id": "me"}, observation_count=5)  # key order differs

    merged = {link["id"]: link for link in _merge([cold, hungry], [donor])}

    assert set(merged) == {"l1", "l2"}
    assert merged["l2"]["observation_count"] == 3 + 5  # paired
    assert merged["l1"] == cold  # untouched


def test_a_donor_link_in_a_new_context_is_added_and_donor_duplicates_fold():
    cold = _ctx_link("l1", {"agent_id": "me", "goal": "cold"})
    first = _ctx_link("d1", {"agent_id": "me"}, observation_count=2)
    second = _ctx_link("d2", {"agent_id": "me"}, observation_count=4)
    for donor in (first, second):
        del donor["source"]  # so the contributor label comes from the side the link was merged in from

    merged = _merge([cold], [first, second])

    assert [link["id"] for link in merged] == ["l1", _nac_id(first)]
    assert merged[1]["observation_count"] == 6  # the second donor link folded, not overwrote
    assert merged[1]["contributors"] == ["peer"]  # folded right-into-right: both halves are the peer's


def _nac_id(link):
    from maxim.decisions.nac import NAc

    nac = NAc()
    return nac._generate_link_id(
        link["event_signature"], link["outcome_signature"], nac._hash_context(link["event_context"])
    )


def test_appended_donor_links_take_the_id_nac_gives_their_identity():
    """A bundle's link id carries no context, so two donors' links in different contexts arrive with ONE
    id; NAc treats ids as unique (EC registration, prune, lookup). Appended, each takes NAc's own id."""
    a = _ctx_link("B1", {"agent_id": "me"})
    b = _ctx_link("B1", {})  # same bundle id, different context
    merged = _merge([], [a, b])

    ids = [link["id"] for link in merged]
    assert len(set(ids)) == 2
    assert ids == [_nac_id(a), _nac_id(b)]


def test_a_receiver_duplicate_takes_the_fold_on_its_first_copy_only():
    first = _ctx_link("l1", {"agent_id": "me"})
    twin = _ctx_link("l2", {"agent_id": "me"})
    donor = _ctx_link("d1", {"agent_id": "me"}, observation_count=5)

    merged = {link["id"]: link for link in _merge([first, twin], [donor])}

    assert merged["l1"]["observation_count"] == 8 and merged["l2"] == twin


def test_a_link_without_an_agent_id_is_not_re_keyed():
    from maxim.hivemind.merge import rekey_nac_state

    link = _ctx_link("l1", {"goal": "cold"})
    out = rekey_nac_state({"links": {"tool:probe": [link]}}, {}, to_agent_id="me")
    assert out["links"]["tool:probe"] == [link]


def test_an_unsigned_bundles_link_is_readable_by_the_receiver_after_ingest():
    """Through substrate_merge (the ingest composition) into a real NAc: the donor's own agent id is
    re-keyed to the receiver's, so predict() uses the link."""
    from maxim.decisions.nac import NAc
    from maxim.hivemind.merge import substrate_merge

    node = {"embedding": [1.0, 0.0], "modality": "world", "count": 1, "source": "local"}
    donor_link = _ctx_link("d1", {"agent_id": "donor"}, confidence=0.9, observation_count=10)
    result = substrate_merge(
        receiver_nac={},
        receiver_ec={"L-a": dict(node)},
        donor_nac={"links": {"tool:probe": [donor_link]}},
        donor_ec={"R-a": dict(node)},
        receiver_source="recv",
        donor_source="donor",
        receiver_agent_id="me",
    )
    (link,) = result.nac["links"]["tool:probe"]
    assert link["event_context"] == {"agent_id": "me"}
    assert link["id"] == _nac_id(link)  # the id of its RE-KEYED identity, not the donor's stale one

    nac = NAc()
    nac.load_state(result.nac)
    prediction = nac.predict("tool_execution", "tool:probe", context={"agent_id": "me"})
    assert prediction is not None, "an ingested link the receiver cannot read is lost learning"


def test_the_merged_outcome_index_names_exactly_the_merged_links():
    """Rebuilt from the merged links: a renamed donor link is findable by outcome, and no stale id stays."""
    from maxim.decisions.nac import NAc
    from maxim.hivemind.merge import nac_merge

    receiver = _ctx_link("l1", {"agent_id": "me", "goal": "cold"})
    donor = _ctx_link("d1", {"agent_id": "me"})
    out = nac_merge(
        left={"links": {"tool:probe": [receiver]}, "outcome_index": {"tool_result:positive": ["l1"]}},
        left_source="local",
        right={"links": {"tool:probe": [donor]}, "outcome_index": {"tool_result:positive": ["d1"]}},
        right_source="peer",
    )
    link_ids = sorted(link["id"] for link in out["links"]["tool:probe"])
    assert out["outcome_index"] == {"tool_result:positive": link_ids}
    assert "d1" not in link_ids

    nac = NAc()
    nac.load_state(out)
    assert len(nac.get_links_for_outcome("tool_result:positive")) == 2
