"""Receiver state for release format v2 (docs/plans/oasis_entry_index_v2.md §Receiver state, decisions
(b)/(c); item 7 PR B): the journal records each verified release, and ingest derives from it payload-
digest dedup, equivocation and downgrade refusals. Through the REAL compose and the REAL ingest."""

from __future__ import annotations

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


def _signer(identity="queen-a"):
    from maxim.hivemind.signing import BundleSigner

    return BundleSigner.generate(signer_identity=identity)


def _release(tmp_path: Path, signer, *, sequence=1, fear=-0.5, name=None) -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import release
    from tests.unit.test_hivemind_ingest import _link, _nac_state, _node

    out = tmp_path / (name or f"r{sequence}_{abs(fear)}.zip")
    compose_bundle(
        nac_state=_nac_state(links={"tool:probe": [_link()]}, cluster_fear={f"a{S}n1{S}drive:oxygen": fear}),
        ec_substrate_nodes={"n1": _node()},
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=release(signer, sequence=sequence),
    )
    return out


def _v1(tmp_path: Path, signer) -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import write_v1_bundle
    from tests.unit.test_hivemind_ingest import _node

    unsigned = tmp_path / "v1-source.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={"n9": _node()},
        output_path=unsigned,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    return write_v1_bundle(unsigned, signer, out=tmp_path / "v1.zip")


def _admit(path: Path, journal, signer, **kw):
    """Ingest and APPLY the journal entry (what `substrate ingest --apply` does)."""
    from maxim.hivemind.ingest import ingest_bundle

    report = ingest_bundle(
        path,
        journal=journal,
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        require_signed=True,
        trusted_keys={signer.signer_identity: signer.public_key_b64},
        receiver_agent_id="receiver",
        **kw,
    )
    journal.record(report.journal_entry)
    journal.save()
    return report


def _journal(tmp_path: Path):
    from maxim.hivemind.ingest import IngestionJournal

    return IngestionJournal(tmp_path / "journal.json")


def test_the_journal_records_each_verified_release(tmp_path):
    signer = _signer()
    report = _admit(_release(tmp_path, signer), _journal(tmp_path), signer)
    entry = _journal(tmp_path).entries[0]  # reloaded from disk
    v = report.verification
    assert entry["signature_scheme"] == 2 and entry["release_sequence"] == 1
    assert entry["signer_identity"] == "queen-a" and entry["license"] == "CDLA-Permissive-2.0"
    assert entry["signer_key"] == v.signer_key and len(entry["signer_key"]) == 64  # 32 bytes, hex
    assert entry["payload_digest"] == v.payload_digest and entry["payload_digest"] != entry["digest"]


def test_a_re_zipped_release_is_the_same_release(tmp_path):
    """Recompressed and reordered: different ZIP bytes, one signed payload -> V8 dedup."""
    from maxim.hivemind.ingest import IngestRefused
    from tests.unit._signed_bundle_helpers import read_members

    signer, journal = _signer(), _journal(tmp_path)
    path = _release(tmp_path, signer)
    _admit(path, journal, signer)
    copy = tmp_path / "rezipped.zip"
    with zipfile.ZipFile(copy, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in reversed(list(read_members(path).items())):
            zf.writestr(name, data)
    assert copy.read_bytes() != path.read_bytes()
    with pytest.raises(IngestRefused) as exc:
        _admit(copy, journal, signer)
    assert exc.value.duty == "V8" and "same signed release" in str(exc.value)


def test_a_second_payload_for_one_sequence_is_equivocation_and_force_does_not_waive_it(tmp_path):
    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    _admit(_release(tmp_path, signer, sequence=1, fear=-0.5), journal, signer)
    other = _release(tmp_path, signer, sequence=1, fear=-0.9)
    for force in (False, True):
        with pytest.raises(IngestRefused) as exc:
            _admit(other, journal, signer, force_digest=force)
        assert exc.value.duty == "equivocation"
    # The next sequence is a new release, admitted.
    assert _admit(_release(tmp_path, signer, sequence=2, fear=-0.9), journal, signer).verification.ok


def test_equivocation_is_keyed_by_the_signing_key_not_the_identity(tmp_path):
    """A different key under the same identity is not the same signer's history (a rotated key starts clean)."""
    first, rotated = _signer(), _signer()
    journal = _journal(tmp_path)
    _admit(_release(tmp_path, first, sequence=1, fear=-0.5, name="a.zip"), journal, first)
    assert _admit(_release(tmp_path, rotated, sequence=1, fear=-0.9, name="b.zip"), journal, rotated).verification.ok


def test_a_v1_bundle_from_a_key_that_released_v2_is_a_downgrade(tmp_path):
    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    v1 = _v1(tmp_path, signer)
    assert _admit(v1, _fresh(tmp_path), signer).verification.scheme == 1  # no v2 history: a v1 is admitted
    _admit(_release(tmp_path, signer), journal, signer)
    with pytest.raises(IngestRefused) as exc:
        _admit(v1, journal, signer)
    assert exc.value.duty == "downgrade"


def _fresh(tmp_path: Path):
    from maxim.hivemind.ingest import IngestionJournal

    return IngestionJournal(tmp_path / "fresh-journal.json")


def test_a_journal_written_before_the_change_still_dedups_on_the_zip_bytes(tmp_path):
    """Pre-change entries carry only the ZIP sha256; has_digest checks both keys."""
    import hashlib

    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    path = _release(tmp_path, signer)
    journal.record({"digest": hashlib.sha256(path.read_bytes()).hexdigest(), "contributor_id": DONOR})
    with pytest.raises(IngestRefused, match="already ingested"):
        _admit(path, journal, signer)


def test_substrate_ingest_refuse_v1_refuses_a_v1_bundle(tmp_path, capsys):
    import json

    from maxim.hivemind.cli import run_substrate_subcommand

    signer = _signer()
    v1 = _v1(tmp_path, signer)
    session = tmp_path / "session"
    session.mkdir()
    (session / "aut_nac.json").write_text(json.dumps({}))
    (session / "aut_ec.json").write_text(json.dumps({"substrate_nodes": {}}))
    base = ["ingest", str(v1), "--session", str(session), "--receiver-body", BODY, "--trust", DONOR]
    base += ["--require-signed", "--trust-key", f"queen-a={signer.public_key_b64}"]
    assert run_substrate_subcommand([*base, "--refuse-v1"]) == 2
    assert "v2 releases only" in capsys.readouterr().err
    assert run_substrate_subcommand(base) == 0


# ── review round (PR B) ────────────────────────────────────────────────────────────────────────


def _admit_unverified(path: Path, journal):
    """The unverified path (`allow_unsigned`, or a plain ingest without --require-signed)."""
    from maxim.hivemind.ingest import ingest_bundle

    report = ingest_bundle(
        path,
        journal=journal,
        receiver_nac=None,
        receiver_ec_nodes=None,
        trusted_sources=frozenset({DONOR}),
        receiver_body=BODY,
        receiver_agent_id="receiver",
    )
    journal.record(report.journal_entry)
    journal.save()
    return report


def _rezip(path: Path, out: Path) -> Path:
    from tests.unit._signed_bundle_helpers import read_members

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in reversed(list(read_members(path).items())):
            zf.writestr(name, data)
    assert out.read_bytes() != path.read_bytes()
    return out


def test_a_release_first_admitted_unverified_is_not_merged_again_when_its_signed_copy_arrives(tmp_path):
    """The security review's probe: a re-zipped copy ingested unverified, then the signed original --
    the content-implied payload digest makes them one release."""
    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    original = _release(tmp_path, signer)
    report = _admit_unverified(_rezip(original, tmp_path / "copy.zip"), journal)
    entry = report.journal_entry
    assert entry["payload_verified"] is False and "signer_key" not in entry  # seeds no ordering rule
    with pytest.raises(IngestRefused) as exc:
        _admit(original, journal, signer)
    assert exc.value.duty == "V8"


def test_a_release_admitted_verified_is_not_merged_again_through_the_unverified_path(tmp_path):
    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    original = _release(tmp_path, signer)
    assert _admit(original, journal, signer).journal_entry["payload_verified"] is True
    with pytest.raises(IngestRefused) as exc:
        _admit_unverified(_rezip(original, tmp_path / "copy.zip"), journal)
    assert exc.value.duty == "V8"


def test_a_re_zipped_v1_bundle_is_the_same_bundle(tmp_path):
    from maxim.hivemind.ingest import IngestRefused

    signer, journal = _signer(), _journal(tmp_path)
    v1 = _v1(tmp_path, signer)
    _admit(v1, journal, signer)
    with pytest.raises(IngestRefused) as exc:
        _admit(_rezip(v1, tmp_path / "v1-copy.zip"), journal, signer)
    assert exc.value.duty == "V8"


def test_the_content_digest_equals_the_verified_one(tmp_path):
    from maxim.hivemind.bundle import content_payload_digest

    signer = _signer()
    for path in (_release(tmp_path, signer), _v1(tmp_path, signer)):
        report = _admit(path, _journal(tmp_path / path.stem), signer)
        with zipfile.ZipFile(path) as zf:
            assert content_payload_digest(zf) == report.verification.payload_digest


@pytest.mark.parametrize(
    ("field", "value"),
    [("release_sequence", True), ("release_sequence", "1"), ("signature_scheme", 2.0), ("signer_key", 7)],
)
def test_a_journal_with_a_mistyped_release_field_fails_loud(tmp_path, field, value):
    import json

    from maxim.hivemind.ingest import IngestionJournal

    path = tmp_path / "journal.json"
    path.write_text(json.dumps({"_format_version": "1.0", "entries": [{"digest": "x", field: value}]}))
    with pytest.raises(ValueError, match=field):
        IngestionJournal(path)


def test_refuse_v1_without_require_signed_is_an_error_not_a_no_op(tmp_path, capsys):
    import json

    from maxim.hivemind.cli import run_substrate_subcommand

    session = tmp_path / "session"
    session.mkdir()
    (session / "aut_nac.json").write_text(json.dumps({}))
    (session / "aut_ec.json").write_text(json.dumps({"substrate_nodes": {}}))
    argv = ["ingest", str(_v1(tmp_path, _signer())), "--session", str(session), "--receiver-body", BODY]
    assert run_substrate_subcommand([*argv, "--trust", DONOR, "--refuse-v1"]) == 2
    assert "needs --require-signed" in capsys.readouterr().err
