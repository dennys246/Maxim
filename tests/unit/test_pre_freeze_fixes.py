"""The pre-freeze fixes (public_oasis Phase 0 item 2): what a public format must not promise.

Found by the format-freeze pass's code extraction, fixed BEFORE the format is frozen in public, each
pinned here: a real agent id and free text inside releases, two manifest readers disagreeing, readers
that crashed or silently coerced, the stored-vs-migrated schema in the listing, and identity strings
with no grammar."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from maxim.utils.optional_deps import optional_dependency_available

S = "\x1f"
DONOR = "donor-1"
BODY = "test_body"
_needs_sign = pytest.mark.skipif(
    not (optional_dependency_available("cryptography") and optional_dependency_available("rfc8785")),
    reason="releases need the [sign] extra",
)


def _nac(**over):
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    link = _link()
    link["event_context"] = {"agent_id": "aut-local-7"}
    link["context_factors"] = {"the llm said something private": 0.4}
    link["event_type"] = "a free text sentence"
    link["outcome_type"] = "tool_result"
    state = _nac_state(links={"tool:probe": [link]}, cluster_fear={f"aut-local-7{S}n1{S}drive:oxygen": -0.5})
    state.update(over)
    return state


def _compose(tmp_path: Path, *, signed: bool, nac=None, contributor=DONOR, name="b.zip") -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit.test_hivemind_ingest import _node

    release = None
    if signed:
        from maxim.hivemind.signing import UNCOUNTED, BundleSigner, SignedRelease

        release = SignedRelease(
            signer=BundleSigner.generate(signer_identity="queen-a"),
            release_sequence=1,
            license="CDLA-Permissive-2.0",
            counter=UNCOUNTED,
        )
    out = tmp_path / name
    compose_bundle(
        nac_state=_nac() if nac is None else nac,
        ec_substrate_nodes={"n1": _node()},
        output_path=out,
        contributor_id=contributor,
        body_ref=BODY,
        apply_identity_filter=False,
        release=release,
    )
    return out


def _link_of(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("nac.json"))["links"]["tool:probe"][0]


@_needs_sign
def test_a_release_ships_no_local_agent_id_even_inside_a_link(tmp_path):
    release = _compose(tmp_path, signed=True)
    with zipfile.ZipFile(release) as zf:
        assert "aut-local-7" not in "".join(zf.read(n).decode() for n in zf.namelist())
    assert _link_of(release)["event_context"] == {"agent_id": "_agent"}


def test_free_text_never_ships_in_a_link(tmp_path):
    link = _link_of(_compose(tmp_path, signed=False))
    assert link["context_factors"] == {}
    assert link["event_type"] == "redacted" and link["outcome_type"] == "tool_result"
    assert link["outcome_signature"] == "tool_result:positive"


def test_a_bool_or_fractional_schema_version_is_refused(tmp_path):
    from maxim.hivemind.bundle import read_bundle_manifest_bytes
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    for bad in (True, 3.0):
        manifest = json.loads(members["manifest.json"])
        manifest["schema_version"] = bad
        members["manifest.json"] = json.dumps(manifest).encode()
        raw = write_members(tmp_path / "bad.zip", members).read_bytes()
        with pytest.raises(ValueError, match="schema_version"):
            read_bundle_manifest_bytes(raw)


@_needs_sign
def test_a_release_claiming_schema_3_as_a_float_does_not_verify(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip
    from maxim.hivemind.signing import BundleSigner, SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = BundleSigner.generate(signer_identity="queen-a")
    members = read_members(_compose(tmp_path, signed=True))
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = 3.0
    members["manifest.json"] = json.dumps(manifest).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    with zipfile.ZipFile(write_members(tmp_path / "f.zip", members)) as zf:
        result = verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=False)
    assert not result.ok and "schema" in result.reason


def _ingest(path: Path, tmp_path: Path, **kw):
    from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

    return ingest_bundle(
        path,
        journal=IngestionJournal(tmp_path / f"j-{path.stem}.json"),
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        **kw,
    )


@pytest.mark.parametrize("value", ["-0.5", True])
def test_a_number_field_holding_a_string_or_bool_is_refused(tmp_path, value):
    """Injected into the shipped slice (compose would coerce it): the receiver refuses, never coerces."""
    from maxim.hivemind.ingest import IngestRefused
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    nac = json.loads(members["nac.json"])
    nac["cluster_reward_bias"] = {f"a{S}n1{S}tool:flee": value}
    members["nac.json"] = json.dumps(nac).encode()
    with pytest.raises(IngestRefused, match="not a JSON number"):
        _ingest(write_members(tmp_path / "n.zip", members), tmp_path)


@pytest.mark.parametrize("sources", [{"short-key": "relief"}, {f"a{S}n1{S}tool:flee": ["relief"]}])
def test_a_malformed_cluster_reward_source_is_refused_not_crashed_on(tmp_path, sources):
    """Before the fix: never validated -- a malformed key crashed the receiver scrub (bare ValueError) and a
    list value escaped as a TypeError."""
    from maxim.hivemind.ingest import IngestRefused
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    nac = json.loads(members["nac.json"])
    nac["cluster_reward_source"] = sources
    members["nac.json"] = json.dumps(nac).encode()
    with pytest.raises(IngestRefused):
        _ingest(write_members(tmp_path / "s.zip", members), tmp_path)


def test_an_unknown_credit_source_is_dropped_with_a_note_not_refused(tmp_path):
    """An open vocabulary: a newer producer's source is dropped as NAc drops it, not a format break."""
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    nac = json.loads(members["nac.json"])
    nac["cluster_reward_bias"] = {f"a{S}n1{S}tool:flee": 0.3}
    nac["cluster_reward_source"] = {f"a{S}n1{S}tool:flee": "a-future-source"}
    members["nac.json"] = json.dumps(nac).encode()
    report = _ingest(write_members(tmp_path / "s.zip", members), tmp_path)
    assert any("credit source this build does not know" in n for n in report.notes)
    assert not (report.nac.get("cluster_reward_source") or {})


def test_a_known_credit_source_still_ingests(tmp_path):
    from maxim.decisions.nac import NAc

    source = sorted(NAc.CREDIT_SOURCES)[0]
    nac = _nac(
        cluster_reward_bias={f"a{S}n1{S}tool:flee": 0.3},
        cluster_reward_source={f"a{S}n1{S}tool:flee": source},
    )
    _ingest(_compose(tmp_path, signed=False, nac=nac), tmp_path)


def test_the_listing_reports_the_stored_schema_not_the_migrated_one(tmp_path):
    import hashlib

    from maxim.hivemind.bundle import content_payload_digest
    from maxim.hivemind.store import OasisStore

    store = OasisStore(tmp_path / "oasis")
    store.releases_dir.mkdir(parents=True)
    legacy = _compose(tmp_path, signed=False, name="legacy.zip")  # schema 2 on the wire
    with zipfile.ZipFile(legacy) as zf:
        identity = content_payload_digest(zf) or hashlib.sha256(legacy.read_bytes()).hexdigest()
    (store.releases_dir / f"{identity}.zip").write_bytes(legacy.read_bytes())
    assert [r["schema_version"] for r in store.list_releases()] == [2]


@_needs_sign
def test_a_contributed_v2_release_records_its_algorithm(tmp_path):
    from maxim.hivemind.store import OasisStore

    store = OasisStore(tmp_path / "oasis")
    store.accept_contribution(_compose(tmp_path, signed=True).read_bytes(), source="10.0.0.1")
    assert store.list_contributions()[0]["signature_algorithm"] == "ed25519"


@pytest.mark.parametrize("contributor", ["has space", "tab\tid", "x" * 129, "emoji-☃"])
def test_a_contributor_id_outside_the_public_grammar_cannot_be_composed(tmp_path, contributor):
    with pytest.raises(ValueError, match="public identity"):
        _compose(tmp_path, signed=False, contributor=contributor)


@_needs_sign
def test_a_release_whose_identity_breaks_the_grammar_does_not_verify(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip
    from maxim.hivemind.signing import BundleSigner, SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = BundleSigner.generate(signer_identity="queen-a")
    members = read_members(_compose(tmp_path, signed=True))
    manifest = json.loads(members["manifest.json"])
    manifest["contributor_id"] = "a contributor with spaces"
    members["manifest.json"] = json.dumps(manifest).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    with zipfile.ZipFile(write_members(tmp_path / "g.zip", members)) as zf:
        result = verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=False)
    assert not result.ok and "public identity" in result.reason


# ── review round (both lenses) ─────────────────────────────────────────────────────────────────


def test_a_link_id_is_derived_from_scrubbed_fields_not_the_private_context(tmp_path):
    """The shipped id hashed the PRE-scrub context (LLM goal, tool params, agent id): a guess could be
    confirmed against it. It is now derived from the scrubbed signatures only."""
    from maxim.hivemind.bundle import _bundle_link_id

    nac = _nac()
    nac["links"]["tool:probe"][0]["id"] = "73e060994475e688"  # NAc's context-hash id
    for signed in (False, True):
        link = _link_of(_compose(tmp_path, signed=signed, nac=nac, name=f"i{signed}.zip"))
        assert link["id"] == _bundle_link_id(link["event_signature"], link["outcome_signature"])


def test_only_allowlisted_fields_ship(tmp_path):
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit.test_hivemind_ingest import _node

    nac = _nac()
    nac["saved_at"] = "2026-09-26T03:14:15"
    nac["a_future_field"] = {"secret": 1}
    link = nac["links"]["tool:probe"][0]
    link["an_unknown_link_key"] = "free text"
    link["domain"] = "a free text domain"
    link["imagined"] = "yes please"
    out = tmp_path / "a.zip"
    compose_bundle(
        nac_state=nac,
        ec_substrate_nodes={"n1": {**_node(), "label": "the kitchen near the window"}},
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        encoder_provenance={"world": {"model_name": "org/model-1", "note": "host dennys-mbp user dennys"}},
    )
    with zipfile.ZipFile(out) as zf:
        shipped_nac = json.loads(zf.read("nac.json"))
        shipped_ec = json.loads(zf.read("ec.json"))
        manifest = json.loads(zf.read("manifest.json"))
    assert "saved_at" not in shipped_nac and "a_future_field" not in shipped_nac
    shipped_link = shipped_nac["links"]["tool:probe"][0]
    assert "an_unknown_link_key" not in shipped_link
    assert shipped_link["domain"] is None and shipped_link["imagined"] is False
    assert "label" not in shipped_ec["substrate_nodes"]["n1"]
    recorded = manifest["encoder_provenance"]["recorded"]["world"]
    assert recorded == {"model_name": "org/model-1", "note": "[REDACTED]"}


@pytest.mark.parametrize(
    ("types", "expected"), [(("free text one", "free text two"), 1), (("free one", "free two"), 2)]
)
def test_free_text_types_fold_by_valence(tmp_path, types, expected):
    """Two free-text outcome types of the same valence fold into one redacted link (observations summed);
    different valences stay separate -- the collision fold nac_merge's pairing relies on."""
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    a, b = _link(), _link()
    a["outcome_type"], b["outcome_type"] = types
    a["id"], b["id"] = "l1", "l2"
    if expected == 2:
        b["outcome_valence"] = "negative"
    nac = _nac_state(links={"tool:probe": [a, b]})
    with zipfile.ZipFile(_compose(tmp_path, signed=False, nac=nac)) as zf:
        links = json.loads(zf.read("nac.json"))["links"]["tool:probe"]
    assert len(links) == expected
    assert all(link["outcome_type"] == "redacted" for link in links)
    if expected == 1:
        assert links[0]["observation_count"] == a["observation_count"] + b["observation_count"]


@_needs_sign
def test_a_received_release_link_predicts_for_the_receiver(tmp_path):
    """The re-read's DO-NOT-MERGE: NAc.predict matches a stored link's event context, so a token left in
    ``event_context.agent_id`` made every released link dead for prediction. Ingest re-keys it."""
    from maxim.decisions.nac import NAc
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity="queen-a")
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.signing import UNCOUNTED, SignedRelease
    from tests.unit.test_hivemind_ingest import _node

    out = tmp_path / "r.zip"
    compose_bundle(
        nac_state=_nac(),
        ec_substrate_nodes={"n1": _node()},
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0", counter=UNCOUNTED),
    )
    report = _ingest(
        out,
        tmp_path,
        require_signed=True,
        trusted_keys={"queen-a": signer.public_key_b64},
        receiver_agent_id="receiver",
    )
    link = report.nac["links"]["tool:probe"][0]
    assert link["event_context"] == {"agent_id": "receiver"}
    nac = NAc()
    nac.load_state(report.nac)
    assert nac.predict("tool_execution", "tool:probe", context={"agent_id": "receiver"}) is not None


@_needs_sign
def test_a_release_whose_links_name_a_local_agent_is_refused_on_receipt(tmp_path):
    from maxim.hivemind.ingest import IngestRefused
    from maxim.hivemind.signing import BundleSigner, SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = BundleSigner.generate(signer_identity="queen-a")
    members = read_members(_compose(tmp_path, signed=True))
    nac = json.loads(members["nac.json"])
    nac["links"]["tool:probe"][0]["event_context"] = {"agent_id": "someones-local-agent"}
    members["nac.json"] = json.dumps(nac, indent=2, sort_keys=True).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    with pytest.raises(IngestRefused, match="name local agent id"):
        _ingest(write_members(tmp_path / "leak.zip", members), tmp_path, receiver_agent_id="receiver")


def test_an_overflowing_float_is_refused_by_the_strict_reader():
    from maxim.hivemind.bundle import _strict_json

    with pytest.raises(ValueError, match="non-finite"):
        _strict_json(b'{"a": 1e999}', "manifest.json")


@pytest.mark.parametrize("identity", ["has space", "_consensus", "x" * 129])
def test_a_signer_outside_the_public_grammar_cannot_be_constructed(identity):
    from maxim.hivemind.signing import BundleSigner

    pytest.importorskip("cryptography")
    with pytest.raises(ValueError, match="public identity"):
        BundleSigner.generate(signer_identity=identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [("contributor_id", "evil\n\x1b[31mFAKE"), ("domain", "a" * 500), ("body_ref", "line\nbreak")],
)
def test_the_contribute_door_refuses_unsafe_strings(tmp_path, field, value):
    from maxim.hivemind.store import OasisStore, OasisStoreError
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    manifest = json.loads(members["manifest.json"])
    manifest[field] = value
    members["manifest.json"] = json.dumps(manifest).encode()
    raw = write_members(tmp_path / "c.zip", members).read_bytes()
    with pytest.raises(OasisStoreError):
        OasisStore(tmp_path / "oasis").accept_contribution(raw, source="10.0.0.1")


@_needs_sign
def test_the_reserved_prefix_is_refused_in_a_release(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip
    from maxim.hivemind.signing import BundleSigner, SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = BundleSigner.generate(signer_identity="queen-a")
    members = read_members(_compose(tmp_path, signed=True))
    manifest = json.loads(members["manifest.json"])
    manifest["contributor_id"] = "_consensus"
    members["manifest.json"] = json.dumps(manifest).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    with zipfile.ZipFile(write_members(tmp_path / "r.zip", members)) as zf:
        result = verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=False)
    assert not result.ok and "public identity" in result.reason


def test_the_content_identity_frames_only_an_integer_schema_3_as_v2(tmp_path):
    import hashlib

    from maxim.hivemind.bundle import content_payload_digest
    from maxim.hivemind.signing import bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = 3.0
    members["manifest.json"] = json.dumps(manifest).encode()
    path = write_members(tmp_path / "f.zip", members)
    v2_framed = hashlib.sha256(
        bundle_signing_payload_v2({n: d for n, d in members.items() if n in ("manifest.json", "nac.json", "ec.json")})
    ).hexdigest()
    with zipfile.ZipFile(path) as zf:
        assert content_payload_digest(zf) != v2_framed


# ── re-read round (links-only releases, signatures, the last unguarded paths) ──────────────────


def _links_only_nac(agent="aut-local-7", other=None):
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    mine = _link()
    mine["event_context"] = {"agent_id": agent}
    links = [mine]
    if other is not None:
        theirs = _link()
        theirs["id"], theirs["outcome_valence"] = "l2", "negative"
        theirs["event_context"] = {"agent_id": other}
        links.append(theirs)
    return _nac_state(links={"tool:probe": links})


@_needs_sign
def test_a_links_only_release_is_re_keyed_and_its_links_predict(tmp_path):
    """The re-read's DO-NOT-MERGE: the common real shape (links, no keyed rows) slipped both refusals and
    shipped dead links. Links now count as agent-scoped: no receiver id -> refused; with one -> they predict."""
    from maxim.decisions.nac import NAc
    from maxim.hivemind.ingest import IngestRefused
    from maxim.hivemind.signing import BundleSigner, UNCOUNTED, SignedRelease
    from maxim.hivemind.bundle import compose_bundle

    signer = BundleSigner.generate(signer_identity="queen-a")
    out = tmp_path / "links.zip"
    compose_bundle(
        nac_state=_links_only_nac(),
        ec_substrate_nodes=None,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0", counter=UNCOUNTED),
    )
    signed = dict(require_signed=True, trusted_keys={"queen-a": signer.public_key_b64})
    with pytest.raises(IngestRefused, match="agent token"):
        _ingest(out, tmp_path, **signed)
    report = _ingest(out, tmp_path / "second", receiver_agent_id="receiver", **signed)
    nac = NAc()
    nac.load_state(report.nac)
    assert nac.predict("tool_execution", "tool:probe", context={"agent_id": "receiver"}) is not None


@_needs_sign
def test_a_release_ships_only_its_own_agents_links(tmp_path):
    """Own rows only covers links: another agent's link (or one an earlier ingest left) is dropped, never
    relabelled into the exporter's -- at the receiver it would have become live."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.signing import BundleSigner, UNCOUNTED, SignedRelease

    out = tmp_path / "own.zip"
    compose_bundle(
        nac_state=_links_only_nac(other="someone-else"),
        ec_substrate_nodes=None,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=SignedRelease(
            signer=BundleSigner.generate(signer_identity="queen-a"),
            release_sequence=1,
            license="CDLA-Permissive-2.0",
            counter=UNCOUNTED,
        ),
        agent_id="aut-local-7",
    )
    with zipfile.ZipFile(out) as zf:
        links = json.loads(zf.read("nac.json"))["links"]["tool:probe"]
    assert len(links) == 1 and links[0]["outcome_valence"] == "positive"


def test_a_free_text_signature_segment_ships_redacted(tmp_path):
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    link = _link()
    link["event_signature"] = "tool:ps aux"
    with zipfile.ZipFile(_compose(tmp_path, signed=False, nac=_nac_state(links={"tool:ps aux": [link]}))) as zf:
        links = json.loads(zf.read("nac.json"))["links"]
    assert list(links) == ["tool:redacted"] and links["tool:redacted"][0]["event_signature"] == "tool:redacted"


def test_provenance_keys_and_event_context_values_are_token_shaped(tmp_path):
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit.test_hivemind_ingest import _node

    nac = _nac()
    nac["links"]["tool:probe"][0]["event_context"] = {"agent_id": "a sentence, not an id"}
    out = tmp_path / "p.zip"
    compose_bundle(
        nac_state=nac,
        ec_substrate_nodes={"n1": _node()},
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        encoder_provenance={"world": {"a free text key": "x", "model_name": "org/m"}},
    )
    with zipfile.ZipFile(out) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        link = json.loads(zf.read("nac.json"))["links"]["tool:probe"][0]
    assert manifest["encoder_provenance"]["recorded"]["world"] == {"model_name": "org/m"}
    assert link["event_context"] == {}


def test_the_scheme_probe_parses_strictly(tmp_path):
    from maxim.hivemind.bundle import bundle_signature_scheme
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    text = members["manifest.json"].decode()
    members["manifest.json"] = (
        text.replace('"kind"', '"signature": "x", "kind"', 1)
        .replace('"kind"', '"kind": "substrate_bundle", "kind"', 1)
        .encode()
    )
    assert bundle_signature_scheme(write_members(tmp_path / "dup.zip", members)) is None


@_needs_sign
def test_a_contribution_records_the_algorithm_it_claims_not_a_constant(tmp_path):
    from maxim.hivemind.signing import SIGNATURE_MEMBER
    from maxim.hivemind.store import OasisStore
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=True))
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature_algorithm"] = "pkcs7"
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    store = OasisStore(tmp_path / "oasis")
    store.accept_contribution(write_members(tmp_path / "c.zip", members).read_bytes(), source="10.0.0.1")
    assert store.list_contributions()[0]["signature_algorithm"] == "pkcs7"


# ── final re-read ──────────────────────────────────────────────────────────────────────────────


def test_motor_parameter_segments_are_vocabulary_not_text(tmp_path):
    """look_at:dy=<n>:dp=<n> (Reachy's real links) must not collapse into one redacted link."""
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    a, b = _link(), _link()
    a["event_signature"], b["event_signature"] = "look_at:dy=10:dp=-5", "look_at:dy=20:dp=-0"  # :.0f, as emitted
    b["id"] = "l2"
    nac = _nac_state(links={"look_at:dy=10:dp=-5": [a], "look_at:dy=20:dp=-0": [b]})
    with zipfile.ZipFile(_compose(tmp_path, signed=False, nac=nac)) as zf:
        links = json.loads(zf.read("nac.json"))["links"]
    assert sorted(links) == ["look_at:dy=10:dp=-5", "look_at:dy=20:dp=-0"]


@_needs_sign
def test_a_resigned_release_whose_links_name_another_agent_does_not_verify(tmp_path):
    """verify_index's token check covers links (no keyed rows needed): a links-only release re-signed with a
    real agent id in a link is refused at verification, not admitted."""
    from maxim.hivemind.bundle import compose_bundle, verify_bundle_zip
    from maxim.hivemind.signing import UNCOUNTED, BundleSigner, SignedRelease
    from tests.unit._signed_bundle_helpers import read_members, resign_v2, write_members

    signer = BundleSigner.generate(signer_identity="queen-a")
    out = tmp_path / "l.zip"
    compose_bundle(
        nac_state=_links_only_nac(),
        ec_substrate_nodes=None,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0", counter=UNCOUNTED),
    )
    members = read_members(out)
    nac = json.loads(members["nac.json"])
    nac["links"]["tool:probe"][0]["event_context"] = {"agent_id": "bob-local"}
    members["nac.json"] = json.dumps(nac, indent=2, sort_keys=True).encode()
    with zipfile.ZipFile(write_members(tmp_path / "bob.zip", resign_v2(members, signer))) as zf:
        result = verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=False)
    assert not result.ok and "bob-local" in result.reason


def test_a_learned_bias_folded_into_an_inherent_one_loses_the_marker(tmp_path):
    """Two tsigs that scrub to one key: the result stays inherent only if EVERY source was inherent."""
    from maxim.hivemind.bundle import scrub_nac_state_for_bundle

    inherent_key = f"a{S}n1{S}tool:use:a free text one"
    learned_key = f"a{S}n1{S}tool:use:another free text"
    state = {
        "links": {},
        "cluster_reward_bias": {inherent_key: 0.8, learned_key: -0.4},
        "inherent_bias_keys": [inherent_key],
    }
    assert scrub_nac_state_for_bundle(state)["inherent_bias_keys"] == []
    state["inherent_bias_keys"] = [inherent_key, learned_key]
    assert scrub_nac_state_for_bundle(state)["inherent_bias_keys"] == [f"a{S}n1{S}tool:use"]


@_needs_sign
def test_a_link_naming_no_agent_is_left_unscoped_in_a_release(tmp_path):
    """An empty agent_id names no agent (NAc reads it as none): it is neither counted as an agent nor
    tokenised -- a link naming no agent must not become scoped to the receiver after re-keying."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.signing import UNCOUNTED, BundleSigner, SignedRelease
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    link = _link()
    link["event_context"] = {"agent_id": ""}
    out = tmp_path / "e.zip"
    compose_bundle(
        nac_state=_nac_state(links={"tool:probe": [link]}, cluster_fear={f"aut{S}n1{S}drive:oxygen": -0.5}),
        ec_substrate_nodes=None,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=SignedRelease(
            signer=BundleSigner.generate(signer_identity="queen-a"),
            release_sequence=1,
            license="CDLA-Permissive-2.0",
            counter=UNCOUNTED,
        ),
    )
    with zipfile.ZipFile(out) as zf:
        shipped = json.loads(zf.read("nac.json"))["links"]["tool:probe"][0]
    assert shipped["event_context"].get("agent_id") in (None, "")
