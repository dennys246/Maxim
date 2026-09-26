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
    manifest["signer_identity"] = "queen-a"
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


@pytest.mark.parametrize(
    ("sources", "match"),
    [({f"a{S}n1{S}tool:flee": "made-up-category"}, "is not a credit source"), ({"short-key": "relief"}, "short-key")],
)
def test_cluster_reward_source_is_validated_not_crashed_on(tmp_path, sources, match):
    """Before the fix: never validated, and a malformed key crashed the receiver scrub with a bare ValueError."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.ingest import IngestRefused
    from tests.unit._signed_bundle_helpers import read_members, write_members

    members = read_members(_compose(tmp_path, signed=False))
    nac = json.loads(members["nac.json"])
    nac["cluster_reward_source"] = sources
    members["nac.json"] = json.dumps(nac).encode()
    with pytest.raises(IngestRefused, match=match):
        _ingest(write_members(tmp_path / "s.zip", members), tmp_path)
    assert compose_bundle  # (import kept: the valid vocabulary still composes below)


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
