"""Release format v2 end to end (docs/plans/oasis_entry_index_v2.md): the real compose -> the real verifier
and the real ingest. Each rule here is proven by deleting its mechanism."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from maxim.utils.optional_deps import optional_dependency_available

pytestmark = pytest.mark.skipif(
    not (optional_dependency_available("cryptography") and optional_dependency_available("rfc8785")),
    reason="v2 releases need the [sign] extra",
)

S = "\x1f"
DONOR = "donor-1"
BODY = "test_body"


def _nac(agent="donor_agent"):
    from tests.unit.test_hivemind_ingest import _link, _nac_state

    return _nac_state(
        links={"tool:probe": [_link()]},
        cluster_fear={f"{agent}{S}n1{S}drive:oxygen": -0.5},
        cluster_reward_bias={f"{agent}{S}orient{S}tool:turn": 0.4},
    )


def _ec():
    from tests.unit.test_hivemind_ingest import _node

    return {"n1": _node()}


def _signer():
    from maxim.hivemind.signing import BundleSigner

    return BundleSigner.generate(signer_identity="queen-a")


def _compose(tmp_path: Path, signer, *, nac=None, ec=None, name="r.zip") -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import release

    out = tmp_path / name
    compose_bundle(
        nac_state=_nac() if nac is None else nac,
        ec_substrate_nodes=_ec() if ec is None else ec,
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=release(signer),
    )
    return out


def _verify(path: Path, signer):
    from maxim.hivemind.bundle import verify_bundle_zip

    with zipfile.ZipFile(path) as zf:
        return verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64})


def test_a_release_verifies_carries_its_fields_and_ships_no_local_agent_id(tmp_path):
    signer = _signer()
    path = _compose(tmp_path, signer)
    result = _verify(path, signer)
    assert result.ok, result.reason
    assert (result.scheme, result.release_sequence, result.license) == (2, 1, "CDLA-Permissive-2.0")
    assert set(result.entry_digests) == {"n1", "orient"}  # "orient" is node-less (NAc-only)
    with zipfile.ZipFile(path) as zf:
        raw = "".join(zf.read(n).decode() for n in zf.namelist())
    assert "donor_agent" not in raw  # the agent segment is normalized before signing


def test_a_release_of_two_agents_is_refused_at_compose(tmp_path):
    from maxim.hivemind.entry_index import EntryIndexError

    nac = _nac()
    nac["cluster_fear"][f"other_agent{S}n2{S}drive:oxygen"] = -0.2
    with pytest.raises(EntryIndexError, match="agent ids"):
        _compose(tmp_path, _signer(), nac=nac)


def test_a_slice_re_signed_with_a_stale_index_is_refused(tmp_path):
    """A signer who edits a slice and re-signs without rebuilding the index is caught by the index."""
    from maxim.hivemind.signing import SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    path = _compose(tmp_path, signer)
    members = read_members(path)
    nac = json.loads(members["nac.json"])
    key = next(iter(nac["cluster_fear"]))
    nac["cluster_fear"][key] = -0.9
    members["nac.json"] = json.dumps(nac, indent=2, sort_keys=True).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    result = _verify(write_members(tmp_path / "stale.zip", members), signer)
    assert not result.ok and "entry index refused" in result.reason and "digest" in result.reason


@pytest.mark.parametrize(
    ("mutate", "why"),
    [
        (lambda m: m.update({"../evil.json": b"{}"}), "malformed member name"),
        (lambda m: m.update({"Nac.JSON ": b"{}"}), "malformed member name"),
    ],
)
def test_member_names_are_constrained(tmp_path, mutate, why):
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    mutate(members)
    result = _verify(write_members(tmp_path / "bad.zip", members), signer)
    assert not result.ok and why in result.reason


def test_duplicate_member_names_are_refused(tmp_path):
    import warnings

    signer = _signer()
    path = _compose(tmp_path, signer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on the duplicate it is asked to write
        with zipfile.ZipFile(path, "a") as zf:
            zf.writestr("nac.json", b"{}")
    result = _verify(path, signer)
    assert not result.ok and "duplicate" in result.reason


def test_the_signature_member_may_not_be_declared_as_a_slice(tmp_path):
    from maxim.hivemind.signing import SIGNATURE_MEMBER, bundle_signing_payload_v2
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    manifest = json.loads(members["manifest.json"])
    manifest["contents"]["sig"] = {"file": SIGNATURE_MEMBER}
    members["manifest.json"] = json.dumps(manifest).encode()
    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    members[SIGNATURE_MEMBER] = json.dumps(sig).encode()
    result = _verify(write_members(tmp_path / "d.zip", members), signer)
    assert not result.ok and "declares signature.json" in result.reason


def test_a_duplicate_key_manifest_is_refused(tmp_path):
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    text = members["manifest.json"].decode()
    members["manifest.json"] = text.replace('"kind"', '"kind": "x", "kind"', 1).encode()
    result = _verify(write_members(tmp_path / "dup.zip", members), signer)
    assert not result.ok and "duplicate key" in result.reason


def test_a_release_ingests_with_a_receiver_agent_id_and_is_refused_without_one(tmp_path):
    from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle

    signer = _signer()
    path = _compose(tmp_path, signer)
    common = dict(
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        require_signed=True,
        trusted_keys={"queen-a": signer.public_key_b64},
    )
    with pytest.raises(IngestRefused, match="agent token"):
        ingest_bundle(path, journal=IngestionJournal(tmp_path / "j1.json"), **common)
    report = ingest_bundle(path, journal=IngestionJournal(tmp_path / "j2.json"), receiver_agent_id="receiver", **common)
    fear_keys = list((report.nac or {}).get("cluster_fear", {}))
    assert fear_keys and all(k.startswith(f"receiver{S}") for k in fear_keys)


def test_the_published_entries_match_the_gated_evidence_via_inspect(tmp_path, capsys):
    """`substrate inspect --entries` projects ANY bundle (unsigned gated evidence included), so a
    release's entries can be checked against the zip it was built from."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.cli import run_substrate_subcommand

    unsigned = tmp_path / "evidence.zip"
    compose_bundle(
        nac_state=_nac(agent="_agent"),
        ec_substrate_nodes=_ec(),
        output_path=unsigned,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    assert run_substrate_subcommand(["inspect", str(unsigned), "--entries"]) == 0
    evidence = capsys.readouterr().out
    signer = _signer()
    assert run_substrate_subcommand(["inspect", str(_compose(tmp_path, signer)), "--entries"]) == 0
    release_out = capsys.readouterr().out
    pick = lambda text: sorted(line for line in text.splitlines() if line.strip().startswith("entry "))  # noqa: E731
    assert pick(evidence) and pick(evidence) == pick(release_out)
