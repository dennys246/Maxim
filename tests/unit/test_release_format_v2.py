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
        return verify_bundle_zip(zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=True)


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


# ── review-round fold (security + architecture lenses) ─────────────────────────────────────────


def _patch_member_header(path: Path, name: str, *, crc: int | None = None, usize: int | None = None) -> None:
    """Rewrite one member's CRC and/or declared uncompressed size in BOTH its local and central headers
    (the lying-header shape V6's bounded reader exists for)."""
    import struct

    data = bytearray(path.read_bytes())
    eocd = data.rfind(b"PK\x05\x06")
    count, _size, cd_offset = struct.unpack_from("<HII", data, eocd + 10)
    pos = cd_offset
    for _ in range(count):
        n_len, x_len, c_len = struct.unpack_from("<HHH", data, pos + 28)
        member = data[pos + 46 : pos + 46 + n_len].decode()
        if member == name:
            local = struct.unpack_from("<I", data, pos + 42)[0]
            if crc is not None:
                struct.pack_into("<I", data, pos + 16, crc)
                struct.pack_into("<I", data, local + 14, crc)
            if usize is not None:
                struct.pack_into("<I", data, pos + 24, usize)
                struct.pack_into("<I", data, local + 22, usize)
        pos += 46 + n_len + x_len + c_len
    path.write_bytes(bytes(data))


def _ingest_signed(path: Path, signer, tmp_path: Path, **kw):
    from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

    return ingest_bundle(
        path,
        journal=IngestionJournal(tmp_path / f"j-{path.stem}.json"),
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        require_signed=True,
        trusted_keys={"queen-a": signer.public_key_b64},
        receiver_agent_id="receiver",
        **kw,
    )


def test_a_lying_signature_header_is_refused_without_inflating_it(tmp_path):
    """signature.json is read before any trust check -- so it is read BOUNDED (a 4 MB stream that
    declares 100 bytes is refused as a lying header, never inflated, never a BadZipFile traceback)."""
    from maxim.hivemind.ingest import IngestRefused
    from maxim.hivemind.signing import SIGNATURE_MEMBER
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    members[SIGNATURE_MEMBER] = b"{" + b" " * (4 * 1024 * 1024) + b"}"
    path = write_members(tmp_path / "bomb.zip", members)
    _patch_member_header(path, SIGNATURE_MEMBER, usize=100)
    result = _verify(path, signer)
    assert not result.ok and "corrupt or lies about its size" in result.reason
    with pytest.raises(IngestRefused, match="corrupt or lies about its size"):
        _ingest_signed(path, signer, tmp_path)


def test_a_corrupt_member_is_a_refusal_not_a_crash(tmp_path):
    from maxim.hivemind.ingest import IngestRefused

    signer = _signer()
    path = _compose(tmp_path, signer)
    _patch_member_header(path, "nac.json", crc=0)
    assert not _verify(path, signer).ok
    with pytest.raises(IngestRefused):
        _ingest_signed(path, signer, tmp_path)


def test_an_undeclared_member_is_refused_before_it_is_decompressed(tmp_path):
    """The member is corrupt: had the verifier read it, the reason would say so. It says the member
    set differs -- the release is refused on its shape, before any undeclared byte is inflated (V7)."""
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    members["extra.json"] = b"{}"
    path = write_members(tmp_path / "extra.zip", members)
    _patch_member_header(path, "extra.json", crc=0)
    result = _verify(path, signer)
    assert not result.ok and "members differ" in result.reason and "extra.json" in result.reason


def test_a_v2_signature_on_a_schema_2_manifest_is_refused(tmp_path):
    from tests.unit._signed_bundle_helpers import resign_manifest_v2

    signer = _signer()
    path = resign_manifest_v2(
        _compose(tmp_path, signer), signer, lambda m: m.__setitem__("schema_version", 2), out=tmp_path / "s2.zip"
    )
    result = _verify(path, signer)
    assert not result.ok and "schema 2 manifest" in result.reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("release_sequence", 0),
        ("release_sequence", True),
        ("release_sequence", "1"),
        ("release_sequence", 2**53),
        ("license", "<script>"),
        ("license", None),
    ],
)
def test_a_signed_but_invalid_release_field_is_refused(tmp_path, field, value):
    """Validly signed -- the SIGNER wrote the bad value -- and still refused."""
    from tests.unit._signed_bundle_helpers import resign_manifest_v2

    signer = _signer()
    path = resign_manifest_v2(
        _compose(tmp_path, signer), signer, lambda m: m.__setitem__(field, value), out=tmp_path / "f.zip"
    )
    result = _verify(path, signer)
    assert not result.ok and "signed manifest field invalid" in result.reason


def test_a_resigned_malformed_member_name_is_still_refused(tmp_path):
    from tests.unit._signed_bundle_helpers import read_members, resign_v2, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    members["../evil.json"] = b"{}"
    result = _verify(write_members(tmp_path / "n.zip", resign_v2(members, signer)), signer)
    assert not result.ok and "malformed member name" in result.reason


def test_a_resigned_duplicate_key_manifest_is_still_refused(tmp_path):
    """Two parsers must never disagree: a validly signed manifest with a duplicate key is refused."""
    from tests.unit._signed_bundle_helpers import read_members, resign_v2, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    text = members["manifest.json"].decode()
    members["manifest.json"] = text.replace('"kind"', '"kind": "x", "kind"', 1).encode()
    result = _verify(write_members(tmp_path / "dup.zip", resign_v2(members, signer)), signer)
    assert not result.ok and "duplicate key" in result.reason


@pytest.mark.parametrize(
    ("member", "edit"),
    [
        ("ec.json", lambda d: d.__setitem__("substrate_nodes", ["n1"])),
        ("nac.json", lambda d: d.__setitem__("cluster_fear", [1])),
        ("nac.json", lambda d: d.__setitem__("inherent_bias_keys", {"a": 1})),
    ],
)
def test_a_signed_slice_of_the_wrong_shape_is_refused_not_a_crash(tmp_path, member, edit):
    from tests.unit._signed_bundle_helpers import read_members, resign_v2, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    doc = json.loads(members[member])
    edit(doc)
    members[member] = json.dumps(doc).encode()
    result = _verify(write_members(tmp_path / "shape.zip", resign_v2(members, signer)), signer)
    assert not result.ok and "entry index refused" in result.reason


def test_a_hostile_unsigned_nac_is_refused_before_the_token_check(tmp_path):
    """No receiver id, unsigned: the token check reads the agent segments -- of a well-formed NAc only."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle
    from tests.unit._signed_bundle_helpers import read_members, write_members

    path = tmp_path / "u.zip"
    compose_bundle(
        nac_state=_nac(),
        ec_substrate_nodes=_ec(),
        output_path=path,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    members = read_members(path)
    nac = json.loads(members["nac.json"])
    nac["cluster_fear"] = [1]
    members["nac.json"] = json.dumps(nac).encode()
    write_members(path, members)
    with pytest.raises(IngestRefused, match="not an object"):
        ingest_bundle(
            path,
            journal=IngestionJournal(tmp_path / "j.json"),
            receiver_nac=None,
            receiver_ec_nodes=None,
            trusted_sources=frozenset({DONOR}),
            receiver_body=BODY,
        )


def test_a_v1_bundle_ingests_signed_and_is_refused_when_v1_is_not_accepted(tmp_path):
    """The headline v1 rule at the INGEST site: verified as stored, before the envelope migration."""
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.ingest import IngestRefused
    from tests.unit._signed_bundle_helpers import write_v1_bundle

    signer = _signer()
    unsigned = tmp_path / "unsigned.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_ec(),
        output_path=unsigned,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    v1 = write_v1_bundle(unsigned, signer, out=tmp_path / "v1.zip")
    report = _ingest_signed(v1, signer, tmp_path, accept_v1=True)
    assert report.verification is not None and report.verification.scheme == 1
    with pytest.raises(IngestRefused, match="v2 releases only"):
        _ingest_signed(v1, signer, tmp_path, accept_v1=False)


def test_the_ingest_report_carries_the_verification(tmp_path):
    signer = _signer()
    report = _ingest_signed(_compose(tmp_path, signer), signer, tmp_path)
    v = report.verification
    assert v is not None and v.ok and v.scheme == 2 and v.release_sequence == 1 and v.payload_digest
    assert set(v.entry_digests) == {"n1", "orient"}


def test_verification_runs_before_the_trust_duties(tmp_path):
    """A forged release from an UNTRUSTED contributor is refused as unsigned, not on V1."""
    from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle

    signer, impostor = _signer(), _signer()
    path = _compose(tmp_path, impostor)
    with pytest.raises(IngestRefused) as exc:
        ingest_bundle(
            path,
            journal=IngestionJournal(tmp_path / "j.json"),
            receiver_nac=None,
            receiver_ec_nodes=None,
            trusted_sources=frozenset(),  # V1 would refuse too -- the signature must refuse FIRST
            receiver_body=BODY,
            require_signed=True,
            trusted_keys={"queen-a": signer.public_key_b64},
            accept_v1=True,
        )
    assert exc.value.duty == "signature"


def test_a_missing_rfc8785_is_a_refusal_with_a_fix_hint(tmp_path, monkeypatch):
    import maxim.hivemind.entry_index as entry_index
    from maxim.utils.optional_deps import OptionalDependencyError

    signer = _signer()
    path = _compose(tmp_path, signer)

    def missing(_value):
        raise OptionalDependencyError("rfc8785", extra="sign")

    monkeypatch.setattr(entry_index, "jcs", missing)
    result = _verify(path, signer)
    assert not result.ok and "cannot check the entry index" in result.reason and "sign" in result.reason


def test_the_total_decompressed_bytes_are_capped(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip

    signer = _signer()
    path = _compose(tmp_path, signer)
    with zipfile.ZipFile(path) as zf:
        biggest = max(i.file_size for i in zf.infolist())
        result = verify_bundle_zip(
            zf, trusted_keys={"queen-a": signer.public_key_b64}, accept_v1=True, max_total_bytes=biggest + 1
        )
    assert not result.ok and "in total" in result.reason


def test_only_a_signed_release_is_schema_3(tmp_path):
    """A 1.3.x reader refuses schema > 2: unsigned contributions stay readable by it, and a signed v2
    release -- which it cannot verify -- is the only bundle it refuses."""
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import read_members

    unsigned = tmp_path / "u.zip"
    compose_bundle(
        nac_state=_nac(),
        ec_substrate_nodes=_ec(),
        output_path=unsigned,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    u = json.loads(read_members(unsigned)["manifest.json"])
    r = json.loads(read_members(_compose(tmp_path, _signer()))["manifest.json"])
    assert (u["schema_version"], r["schema_version"]) == (2, 3)
    assert "signature" not in u and "entry_index" not in u


def test_ingest_drops_non_situation_rows_it_can_never_read(tmp_path):
    """Situation rows re-key to the receiver; percept valences / outcome stats / node-keyed reward bias
    cannot, so rows under the token are dropped (counted) instead of stored as unreadable clutter."""
    from tests.unit.test_hivemind_ingest import _nac_state

    nac = _nac_state(
        cluster_fear={f"donor_agent{S}n1{S}drive:oxygen": -0.5},
        percept_valences={f"donor_agent{S}zombie{S}drive:health": -0.4},
    )
    signer = _signer()
    report = _ingest_signed(_compose(tmp_path, signer, nac=nac), signer, tmp_path)
    assert report.foreign_rows_dropped == 1
    assert not (report.nac.get("percept_valences") or {})
    assert all(k.startswith(f"receiver{S}") for k in report.nac["cluster_fear"])


# ── second review round (re-read of the fold) ──────────────────────────────────────────────────


def _patch_member_field(path: Path, name: str, *, flag: int | None = None, method: int | None = None) -> None:
    """Set one member's general-purpose flag / compression method in both headers."""
    import struct

    data = bytearray(path.read_bytes())
    eocd = data.rfind(b"PK\x05\x06")
    count, _size, cd_offset = struct.unpack_from("<HII", data, eocd + 10)
    pos = cd_offset
    for _ in range(count):
        n_len, x_len, c_len = struct.unpack_from("<HHH", data, pos + 28)
        if data[pos + 46 : pos + 46 + n_len].decode() == name:
            local = struct.unpack_from("<I", data, pos + 42)[0]
            if flag is not None:
                struct.pack_into("<H", data, pos + 8, flag)
                struct.pack_into("<H", data, local + 6, flag)
            if method is not None:
                struct.pack_into("<H", data, pos + 10, method)
                struct.pack_into("<H", data, local + 8, method)
        pos += 46 + n_len + x_len + c_len
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(("flag", "method"), [(0x1, None), (None, 99)], ids=["encrypted", "unknown-compression"])
def test_an_unreadable_member_is_a_refusal_not_a_crash(tmp_path, flag, method):
    """zipfile raises RuntimeError (encrypted) / NotImplementedError (unknown method) -- both refusals."""
    from maxim.hivemind.bundle import MemberReadError, bounded_member_read
    from maxim.hivemind.signing import SIGNATURE_MEMBER

    signer = _signer()
    path = _compose(tmp_path, signer)
    _patch_member_field(path, SIGNATURE_MEMBER, flag=flag, method=method)
    result = _verify(path, signer)
    assert not result.ok and "unreadable member" in result.reason
    with zipfile.ZipFile(path) as zf, pytest.raises(MemberReadError):
        bounded_member_read(zf, SIGNATURE_MEMBER, max_bytes=1 << 20)


def test_the_verifier_caps_the_member_count_itself(tmp_path):
    """Direct callers (verify_bundle_signature, a future Oasis gate) do not get ingest's V6 pre-check."""
    from maxim.hivemind.bundle import MAX_BUNDLE_ENTRIES, verify_bundle_signature
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    for i in range(MAX_BUNDLE_ENTRIES):
        members[f"pad{i}.json"] = b"{}"
    ok, reason = verify_bundle_signature(
        write_members(tmp_path / "many.zip", members),
        trusted_keys={"queen-a": signer.public_key_b64},
        accept_v1=True,
    )
    assert not ok and f"(cap {MAX_BUNDLE_ENTRIES})" in reason


def _deep(n: int = 20_000) -> bytes:
    return b"[" * n + b"]" * n


def test_a_deeply_nested_manifest_is_a_refusal_everywhere(tmp_path):
    from maxim.hivemind.bundle import read_bundle_manifest_bytes
    from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer = _signer()
    members = read_members(_compose(tmp_path, signer))
    members["manifest.json"] = _deep()
    path = write_members(tmp_path / "deep.zip", members)
    result = _verify(path, signer)
    assert not result.ok and "nests too deeply" in result.reason
    with pytest.raises(ValueError, match="nests too deeply"):
        read_bundle_manifest_bytes(path.read_bytes())
    with pytest.raises((IngestRefused, ValueError)):
        ingest_bundle(
            path,
            journal=IngestionJournal(tmp_path / "j.json"),
            receiver_nac=None,
            receiver_ec_nodes=None,
            trusted_sources=frozenset({DONOR}),
            receiver_body=BODY,
        )


def test_a_deeply_nested_slice_is_refused_at_ingest(tmp_path):
    from maxim.hivemind.ingest import _loads_strict, IngestRefused

    with pytest.raises(IngestRefused, match="nests too deeply"):
        _loads_strict(_deep(), slice_name="nac.json")


def test_the_manifest_reader_and_extract_are_bounded(tmp_path, monkeypatch):
    """The manifest is read before any trust check on /contribute, hive pull and inspect; extract reads
    every member. Both go through the capped reader (a lying header is a ValueError, not an inflate)."""
    import maxim.hivemind.bundle as bundle_mod
    from maxim.hivemind.bundle import compose_bundle, extract_bundle, read_bundle_manifest_bytes

    path = tmp_path / "u.zip"
    compose_bundle(
        nac_state=_nac(),
        ec_substrate_nodes=_ec(),
        output_path=path,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    _patch_member_header(path, "manifest.json", usize=10)
    with pytest.raises(ValueError, match="lies about its size"):
        read_bundle_manifest_bytes(path.read_bytes())
    fresh = tmp_path / "f.zip"
    compose_bundle(
        nac_state=_nac(),
        ec_substrate_nodes=_ec(),
        output_path=fresh,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    monkeypatch.setattr(bundle_mod, "MAX_TOTAL_UNCOMPRESSED_BYTES", 64)
    with pytest.raises(ValueError, match="in total"):
        extract_bundle(fresh, tmp_path / "out")


def test_ingest_reads_the_file_once(tmp_path, monkeypatch):
    """The digest, the manifest and every verified member come from ONE snapshot: no ZipFile is ever
    opened on the path (a file swapped mid-ingest cannot pair two bundles)."""
    import maxim.hivemind.ingest as ingest_mod

    signer = _signer()
    path = _compose(tmp_path, signer)
    real_zipfile = zipfile.ZipFile

    def only_bytes(source, *a, **kw):
        assert not isinstance(source, (str, Path)), f"ingest opened the path {source!r} again"
        return real_zipfile(source, *a, **kw)

    monkeypatch.setattr(ingest_mod.zipfile, "ZipFile", only_bytes)
    assert _ingest_signed(path, signer, tmp_path).verification is not None


@pytest.mark.parametrize("bad", ["a:b", f"a{S}b", ""])
def test_an_agent_id_holding_a_key_separator_is_refused(tmp_path, bad):
    from maxim.hivemind.entry_index import EntryIndexError, normalize_agent_segment

    with pytest.raises(EntryIndexError):
        normalize_agent_segment({}, own_agent_id=bad)
    with pytest.raises(ValueError, match="not an agent id"):
        from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

        ingest_bundle(
            _compose(tmp_path, _signer()),
            journal=IngestionJournal(tmp_path / "j.json"),
            receiver_nac=None,
            receiver_ec_nodes=None,
            trusted_sources=frozenset({DONOR}),
            receiver_body=BODY,
            receiver_agent_id=bad,
        )


def _unsigned_with_valences(tmp_path: Path) -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit.test_hivemind_ingest import _nac_state

    path = tmp_path / "exp-shape.zip"
    compose_bundle(
        nac_state=_nac_state(
            cluster_fear={f"donor_agent{S}n1{S}drive:oxygen": -0.5},
            percept_valences={f"donor_agent{S}zombie{S}drive:health": -0.4},
        ),
        ec_substrate_nodes=_ec(),
        output_path=path,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    return path


@pytest.mark.parametrize(
    ("receiver_agent_id", "dropped"),
    [("recv_taught_42", 1), (None, 0), ("donor_agent", 0)],
    ids=["other-agent-dropped", "no-receiver-id-kept", "same-agent-kept"],
)
def test_the_unsigned_drop_rule_the_experiment_harnesses_take(tmp_path, receiver_agent_id, dropped):
    """The Exp 56/61/R3 shape: an UNSIGNED bundle whose rows carry the donor's real agent id. With a
    different receiver id they are unreadable to it and dropped (counted); with no receiver id, or the
    same id, they are exactly the rows a read could reach before, and are kept. The V4 valence report
    lists the donor's claim either way."""
    from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

    report = ingest_bundle(
        _unsigned_with_valences(tmp_path),
        journal=IngestionJournal(tmp_path / "j.json"),
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        receiver_agent_id=receiver_agent_id,
    )
    assert report.foreign_rows_dropped == dropped
    assert bool(report.nac.get("percept_valences")) is (dropped == 0)
    assert report.valence_entries == {"zombie (drive:health)": -0.4}
    assert report.journal_entry["foreign_rows_dropped"] == dropped
