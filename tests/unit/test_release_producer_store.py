"""Producer + store for release format v2 (docs/plans/oasis_entry_index_v2.md §Producer, §Oasis store;
item 7 PR C): the per-key release counter, named key files, verify-on-publish with payload-digest ids,
store equivocation, and the one-time id migration. Through the REAL CLI, compose and store."""

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


def _signer(identity="queen-a"):
    from maxim.hivemind.signing import BundleSigner

    return BundleSigner.generate(signer_identity=identity)


# ── the release counter ────────────────────────────────────────────────────────────────────────


def test_a_fresh_key_counts_from_one_and_each_commit_moves_it_on(tmp_path):
    from maxim.hivemind.signing import commit_release_sequence, next_release_sequence, register_fresh_key

    counter, signer = tmp_path / "seq.json", _signer()
    register_fresh_key(signer, path=counter)  # what open_signer does when it mints a key
    for expected in (1, 2, 3):
        seq = next_release_sequence(signer, requested=None, path=counter)
        assert seq == expected
        commit_release_sequence(signer, seq, path=counter)
    data = json.loads(counter.read_text())
    assert data["_format_version"] == "1.0" and list(data["signers"].values())[0]["last"] == 3


def test_an_existing_key_the_counter_never_saw_must_name_its_sequence(tmp_path):
    """A restored key must not restart at 1 -- that re-uses a sequence it may already have released."""
    from maxim.hivemind.signing import ReleaseSequenceError, commit_release_sequence, next_release_sequence

    counter, signer = tmp_path / "seq.json", _signer()
    with pytest.raises(ReleaseSequenceError, match="no release counter"):
        next_release_sequence(signer, requested=None, path=counter)
    assert next_release_sequence(signer, requested=7, path=counter) == 7
    commit_release_sequence(signer, 7, path=counter)
    assert next_release_sequence(signer, requested=None, path=counter) == 8


@pytest.mark.parametrize("requested", [3, 2])
def test_an_explicit_sequence_may_only_move_forward(tmp_path, requested):
    from maxim.hivemind.signing import ReleaseSequenceError, commit_release_sequence, next_release_sequence

    counter, signer = tmp_path / "seq.json", _signer()
    commit_release_sequence(signer, 3, path=counter)
    with pytest.raises(ReleaseSequenceError, match="does not move forward"):
        next_release_sequence(signer, requested=requested, path=counter)


def test_a_sequence_taken_meanwhile_is_refused_at_commit(tmp_path):
    from maxim.hivemind.signing import ReleaseSequenceError, commit_release_sequence

    counter, signer = tmp_path / "seq.json", _signer()
    commit_release_sequence(signer, 1, path=counter)
    with pytest.raises(ReleaseSequenceError, match="taken meanwhile"):
        commit_release_sequence(signer, 1, path=counter)


def test_the_counter_is_per_key_not_per_identity(tmp_path):
    """The Queen key and a development key under one identity never share (or advance) a counter."""
    from maxim.hivemind.signing import commit_release_sequence, next_release_sequence

    from maxim.hivemind.signing import register_fresh_key

    counter = tmp_path / "seq.json"
    queen, dev = _signer("maxim-queen"), _signer("maxim-queen")
    commit_release_sequence(dev, 40, path=counter)
    register_fresh_key(queen, path=counter)
    assert next_release_sequence(queen, requested=None, path=counter) == 1


# ── the export CLI ─────────────────────────────────────────────────────────────────────────────


def _session(tmp_path: Path, *, journal_entries=None) -> Path:
    from tests.unit.test_hivemind_ingest import _link, _nac_state, _node

    session = tmp_path / "session"
    session.mkdir(exist_ok=True)
    nac = _nac_state(links={"tool:probe": [_link()]}, cluster_fear={f"aut{S}n1{S}drive:oxygen": -0.5})
    (session / "aut_nac.json").write_text(json.dumps(nac))
    (session / "aut_ec.json").write_text(json.dumps({"substrate_nodes": {"n1": _node()}}))
    if journal_entries is not None:
        (session / "substrate_ingest_journal.json").write_text(
            json.dumps({"_format_version": "1.0", "entries": journal_entries, "tombstones": []})
        )
    return session


def _export(session: Path, out: Path, *extra: str) -> int:
    from maxim.hivemind.cli import run_substrate_subcommand

    argv = ["export", "--session", str(session), "--contributor-id", DONOR, "--no-identity-filter"]
    argv += ["--body-ref", BODY, "--sign", "--license", "CDLA-Permissive-2.0", *extra, str(out)]
    return run_substrate_subcommand(argv)


def _manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


def test_export_sign_takes_its_sequence_from_the_named_key_s_counter(tmp_path):
    session, key = _session(tmp_path), tmp_path / "queen.key"
    assert _export(session, tmp_path / "a.zip", "--key-file", str(key)) == 0
    assert _export(session, tmp_path / "b.zip", "--key-file", str(key)) == 0
    assert [_manifest(tmp_path / n)["release_sequence"] for n in ("a.zip", "b.zip")] == [1, 2]
    assert key.is_file() and (tmp_path / "queen.key.pub").is_file()
    # the release verifies against the named key's public key
    from maxim.hivemind.bundle import verify_bundle_zip

    pub = (tmp_path / "queen.key.pub").read_text().strip()
    with zipfile.ZipFile(tmp_path / "b.zip") as zf:
        assert verify_bundle_zip(zf, trusted_keys={DONOR: pub}, accept_v1=False).ok


def test_a_failed_commit_leaves_no_signed_release_behind(tmp_path, monkeypatch, capsys):
    import maxim.hivemind.bundle as bundle_mod
    import maxim.hivemind.signing as signing

    def taken(*_a, **_k):
        raise signing.ReleaseSequenceError("release sequence 1 was taken meanwhile")

    monkeypatch.setattr(bundle_mod, "commit_release_sequence", taken)
    out = tmp_path / "a.zip"
    assert _export(_session(tmp_path), out, "--key-file", str(tmp_path / "k")) == 2
    assert not out.exists() and "taken meanwhile" in capsys.readouterr().err


def test_key_file_and_release_sequence_need_sign(tmp_path, capsys):
    from maxim.hivemind.cli import run_substrate_subcommand

    argv = ["export", "--session", str(_session(tmp_path)), "--contributor-id", DONOR, str(tmp_path / "u.zip")]
    assert run_substrate_subcommand([*argv, "--key-file", str(tmp_path / "k")]) == 2
    assert "need --sign" in capsys.readouterr().err
    assert run_substrate_subcommand([*argv, "--release-sequence", "3"]) == 2


def test_keygen_writes_a_named_key_file(tmp_path, capsys):
    from maxim.hivemind.cli import run_substrate_subcommand

    key = tmp_path / "queen.key"
    assert run_substrate_subcommand(["keygen", "--signer-id", "maxim-queen", "--key-file", str(key)]) == 0
    out = capsys.readouterr().out
    assert key.is_file() and str(key) in out and (tmp_path / "queen.key.pub").read_text().strip() in out


def test_a_release_warns_about_non_permissive_input_licenses(tmp_path, capsys):
    entries = [
        {
            "digest": "a" * 64,
            "signer_key": "ab",
            "signer_identity": "q",
            "release_sequence": 1,
            "license": "CC-BY-NC-4.0",
        },
        {"digest": "b" * 64, "signer_key": "ab", "signer_identity": "q", "release_sequence": 2, "license": "CC0-1.0"},
    ]
    session = _session(tmp_path, journal_entries=entries)
    assert _export(session, tmp_path / "r.zip", "--release", "--key-file", str(tmp_path / "k")) == 0
    err = capsys.readouterr().err
    assert "non-permissive" in err and "CC-BY-NC-4.0" in err and "CC0-1.0" not in err


# ── the store ──────────────────────────────────────────────────────────────────────────────────


def _release(tmp_path: Path, signer, *, sequence=1, fear=-0.5, name=None) -> Path:
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit._signed_bundle_helpers import release
    from tests.unit.test_hivemind_ingest import _nac_state, _node

    out = tmp_path / (name or f"r{sequence}_{abs(fear)}.zip")
    compose_bundle(
        nac_state=_nac_state(cluster_fear={f"a{S}n1{S}drive:oxygen": fear}),
        ec_substrate_nodes={"n1": _node()},
        output_path=out,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
        release=release(signer, sequence=sequence),
    )
    return out


def _store(tmp_path):
    from maxim.hivemind.store import OasisStore

    return OasisStore(tmp_path / "oasis")


def test_publish_verifies_and_ids_a_release_by_its_signed_payload(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip

    signer, store = _signer(), _store(tmp_path)
    path = _release(tmp_path, signer)
    keys = {"queen-a": signer.public_key_b64}
    release_id = store.publish_release(path, queen_keys=keys)
    with zipfile.ZipFile(path) as zf:
        assert release_id == verify_bundle_zip(zf, trusted_keys=keys, accept_v1=False).payload_digest
    # a re-zipped copy is the same release: idempotent, one file
    from tests.unit._signed_bundle_helpers import read_members

    copy = tmp_path / "copy.zip"
    with zipfile.ZipFile(copy, "w", compression=zipfile.ZIP_STORED) as zf:
        for member, data in reversed(list(read_members(path).items())):
            zf.writestr(member, data)
    assert store.publish_release(copy, queen_keys=keys) == release_id
    assert len(list(store.releases_dir.glob("*.zip"))) == 1


@pytest.mark.parametrize("case", ["no-keys", "wrong-key", "v1", "unsigned"])
def test_publish_refuses_what_is_not_a_verified_v2_release(tmp_path, case):
    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.store import OasisStoreError
    from tests.unit._signed_bundle_helpers import write_v1_bundle
    from tests.unit.test_hivemind_ingest import _node

    signer, store = _signer(), _store(tmp_path)
    keys = {"queen-a": signer.public_key_b64}
    path = _release(tmp_path, signer)
    if case == "no-keys":
        keys = {}
    elif case == "wrong-key":
        keys = {"queen-a": _signer().public_key_b64}
    elif case in ("v1", "unsigned"):
        unsigned = tmp_path / "u.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes={"n9": _node()},
            output_path=unsigned,
            contributor_id=DONOR,
            body_ref=BODY,
            apply_identity_filter=False,
        )
        path = write_v1_bundle(unsigned, signer, out=tmp_path / "v1.zip") if case == "v1" else unsigned
    with pytest.raises(OasisStoreError) as exc:
        store.publish_release(path, queen_keys=keys)
    if case == "no-keys":  # said plainly, not as an untrusted-signer refusal
        assert "Queen key(s) this Oasis publishes under" in str(exc.value)
    assert store.list_releases() == []


def test_publish_refuses_a_second_payload_under_a_held_sequence(tmp_path):
    from maxim.hivemind.store import OasisStoreError

    signer, store = _signer(), _store(tmp_path)
    keys = {"queen-a": signer.public_key_b64}
    store.publish_release(_release(tmp_path, signer, sequence=1, fear=-0.5), queen_keys=keys)
    with pytest.raises(OasisStoreError, match="equivocation"):
        store.publish_release(_release(tmp_path, signer, sequence=1, fear=-0.9), queen_keys=keys)
    store.publish_release(_release(tmp_path, signer, sequence=2, fear=-0.9), queen_keys=keys)
    assert len(store.list_releases()) == 2


def test_releases_stored_under_their_zip_sha_migrate_to_their_payload_identity_once(tmp_path):
    import hashlib

    from maxim.hivemind.bundle import content_payload_digest
    from tests.unit._signed_bundle_helpers import write_v1_bundle

    signer, store = _signer(), _store(tmp_path)
    store.releases_dir.mkdir(parents=True)
    v2 = _release(tmp_path, signer)
    from maxim.hivemind.bundle import compose_bundle
    from tests.unit.test_hivemind_ingest import _node

    unsigned = tmp_path / "u.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={"n9": _node()},
        output_path=unsigned,
        contributor_id=DONOR,
        body_ref=BODY,
        apply_identity_filter=False,
    )
    v1 = write_v1_bundle(unsigned, signer, out=tmp_path / "v1.zip")
    expected = set()
    for path in (v2, v1):
        raw = path.read_bytes()
        (store.releases_dir / f"{hashlib.sha256(raw).hexdigest()}.zip").write_bytes(raw)
        with zipfile.ZipFile(path) as zf:
            expected.add(content_payload_digest(zf))
    assert store.migrate_release_ids() == 2
    assert {p.stem for p in store.releases_dir.glob("*.zip")} == expected
    assert store.migrate_release_ids() == 0  # idempotent


def test_two_concurrent_commits_of_one_sequence_cannot_both_succeed(tmp_path, monkeypatch):
    """The ~/.maxim/util read-modify-write rule: the counter commit runs under a FileLock. A slowed read
    widens the race window -- without the lock both threads read "no entry" and both commit."""
    import threading
    import time

    import maxim.hivemind.signing as signing

    real_read = signing._read_counter

    def slow_read(target):
        data = real_read(target)
        time.sleep(0.2)
        return data

    monkeypatch.setattr(signing, "_read_counter", slow_read)
    counter, signer = tmp_path / "seq.json", _signer()
    outcomes: list[str] = []

    def commit():
        try:
            signing.commit_release_sequence(signer, 1, path=counter)
            outcomes.append("ok")
        except signing.ReleaseSequenceError:
            outcomes.append("refused")

    threads = [threading.Thread(target=commit) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok", "refused"]


# ── review round (PR C) ────────────────────────────────────────────────────────────────────────


def test_an_interrupted_commit_leaves_no_signed_release_and_burns_nothing(tmp_path, monkeypatch):
    """The commit runs inside compose_bundle, between the signed .tmp and the output path: an interrupt
    there leaves neither the output nor the .tmp, and the counter never recorded the number."""
    import maxim.hivemind.bundle as bundle_mod

    session, key = _session(tmp_path), tmp_path / "k"
    real_commit = bundle_mod.commit_release_sequence

    def interrupted(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(bundle_mod, "commit_release_sequence", interrupted)
    out = tmp_path / "a.zip"
    with pytest.raises(KeyboardInterrupt):
        _export(session, out, "--key-file", str(key))
    assert not out.exists() and not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(bundle_mod, "commit_release_sequence", real_commit)
    assert _export(session, tmp_path / "b.zip", "--key-file", str(key), "--release-sequence", "1") == 0
    assert _manifest(tmp_path / "b.zip")["release_sequence"] == 1


def test_keygen_then_export_starts_at_one(tmp_path):
    from maxim.hivemind.cli import run_substrate_subcommand

    key = tmp_path / "queen.key"
    assert run_substrate_subcommand(["keygen", "--signer-id", DONOR, "--key-file", str(key)]) == 0
    assert _export(_session(tmp_path), tmp_path / "a.zip", "--key-file", str(key)) == 0
    assert _manifest(tmp_path / "a.zip")["release_sequence"] == 1


def test_a_malformed_counter_entry_fails_loud(tmp_path):
    from maxim.hivemind.signing import ReleaseSequenceError, _public_key_hex, next_release_sequence

    signer, counter = _signer(), tmp_path / "seq.json"
    counter.write_text(json.dumps({"_format_version": "1.0", "signers": {_public_key_hex(signer): 7}}))
    with pytest.raises(ReleaseSequenceError, match="malformed entry"):
        next_release_sequence(signer, requested=1, path=counter)


def test_a_missing_or_stale_pub_file_is_rewritten_from_the_private_key(tmp_path):
    from maxim.hivemind.signing import load_or_create_signer

    key = tmp_path / "k"
    signer = load_or_create_signer(signer_identity="q", key_file=key)
    pub = tmp_path / "k.pub"
    pub.unlink()
    load_or_create_signer(signer_identity="q", key_file=key)
    assert pub.read_text().strip() == signer.public_key_b64
    pub.write_text("someone-else\n")
    load_or_create_signer(signer_identity="q", key_file=key)
    assert pub.read_text().strip() == signer.public_key_b64


def test_relabelling_a_key_does_not_hide_its_published_sequences(tmp_path):
    """The review's probe: key K published sequence 1 as "queen-a"; the same K registered as "queen-b"
    must not publish a DIFFERENT sequence 1 -- equivocation compares key bytes, never labels."""
    from maxim.hivemind.signing import BundleSigner
    from maxim.hivemind.store import OasisStoreError

    a = _signer("queen-a")
    b = BundleSigner.from_private_pem(a.private_pem(), signer_identity="queen-b")  # the same key
    store = _store(tmp_path)
    store.publish_release(_release(tmp_path, a, sequence=1, fear=-0.5), queen_keys={"queen-a": a.public_key_b64})
    with pytest.raises(OasisStoreError, match="equivocation"):
        store.publish_release(_release(tmp_path, b, sequence=1, fear=-0.9), queen_keys={"queen-b": b.public_key_b64})


def test_a_held_file_that_verifies_under_no_queen_key_binds_no_sequence(tmp_path):
    other, queen, store = _signer("stranger"), _signer(), _store(tmp_path)
    store.releases_dir.mkdir(parents=True)
    planted = _release(tmp_path, other, sequence=1, fear=-0.9, name="planted.zip")
    (store.releases_dir / ("f" * 64 + ".zip")).write_bytes(planted.read_bytes())
    assert store.publish_release(_release(tmp_path, queen, sequence=1), queen_keys={"queen-a": queen.public_key_b64})


def test_a_non_verifying_file_at_a_release_id_is_replaced_on_publish(tmp_path):
    from maxim.hivemind.bundle import verify_bundle_zip
    from maxim.hivemind.signing import SIGNATURE_MEMBER
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer, store = _signer(), _store(tmp_path)
    keys = {"queen-a": signer.public_key_b64}
    genuine = _release(tmp_path, signer)
    members = read_members(genuine)
    members[SIGNATURE_MEMBER] = b'{"signature_scheme": 2, "signature_algorithm": "ed25519", "signature": "AAAA"}'
    forged = write_members(tmp_path / "forged.zip", members)
    with zipfile.ZipFile(genuine) as zf:
        release_id = verify_bundle_zip(zf, trusted_keys=keys, accept_v1=False).payload_digest
    store.releases_dir.mkdir(parents=True)
    (store.releases_dir / f"{release_id}.zip").write_bytes(forged.read_bytes())
    assert store.publish_release(genuine, queen_keys=keys) == release_id
    with zipfile.ZipFile(store.releases_dir / f"{release_id}.zip") as zf:
        assert verify_bundle_zip(zf, trusted_keys=keys, accept_v1=False).ok


def test_migration_never_deletes_a_different_file_sharing_an_identity(tmp_path):
    """A genuine release and a copy with a forged signature member share a payload identity (the identity
    ignores signature.json). The migration must not pick one and delete the other."""
    import hashlib

    from maxim.hivemind.signing import SIGNATURE_MEMBER
    from tests.unit._signed_bundle_helpers import read_members, write_members

    signer, store = _signer(), _store(tmp_path)
    genuine = _release(tmp_path, signer)
    members = read_members(genuine)
    members[SIGNATURE_MEMBER] = b'{"signature_scheme": 2, "signature_algorithm": "ed25519", "signature": "AAAA"}'
    forged = write_members(tmp_path / "forged.zip", members)
    store.releases_dir.mkdir(parents=True)
    for path in (genuine, forged):
        raw = path.read_bytes()
        (store.releases_dir / f"{hashlib.sha256(raw).hexdigest()}.zip").write_bytes(raw)
    store.migrate_release_ids()
    held = {p.read_bytes() for p in store.releases_dir.glob("*.zip")}
    assert genuine.read_bytes() in held and forged.read_bytes() in held


def test_writing_oasis_verbs_migrate_release_ids_and_status_only_reports(tmp_path, capsys):
    import hashlib

    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    signer, store = _signer(), _store(tmp_path)
    store.releases_dir.mkdir(parents=True)
    raw = _release(tmp_path, signer).read_bytes()
    legacy = store.releases_dir / f"{hashlib.sha256(raw).hexdigest()}.zip"
    legacy.write_bytes(raw)
    assert run_oasis_subcommand(["status", "--root", str(store.root)]) == 0
    assert "1 release id(s) pending migration" in capsys.readouterr().out and legacy.is_file()  # read-only
    new = _release(tmp_path, signer, sequence=2, fear=-0.7)
    key = f"queen-a={signer.public_key_b64}"
    assert run_oasis_subcommand(["publish", str(new), "--root", str(store.root), "--queen-key", key]) == 0
    assert "migrated 1 release id" in capsys.readouterr().out and not legacy.exists()


def test_a_copied_key_the_counter_never_saw_must_name_its_sequence_through_the_cli(tmp_path, capsys):
    """The headline counter rule at the real caller: a key file this host did not mint (restored/copied)."""
    signer = _signer(DONOR)
    key = tmp_path / "copied.key"
    key.write_bytes(signer.private_pem())
    session = _session(tmp_path)
    assert _export(session, tmp_path / "a.zip", "--key-file", str(key)) == 2
    assert "--release-sequence 1 if it never signed one" in capsys.readouterr().err
    assert _export(session, tmp_path / "b.zip", "--key-file", str(key), "--release-sequence", "5") == 0
    assert _export(session, tmp_path / "c.zip", "--key-file", str(key)) == 0
    assert [_manifest(tmp_path / n)["release_sequence"] for n in ("b.zip", "c.zip")] == [5, 6]


def test_a_release_warns_about_unlicensed_inputs(tmp_path, capsys):
    entries = [{"digest": "a" * 64, "contributor_id": "peer"}]  # an unsigned contribution: no license
    session = _session(tmp_path, journal_entries=entries)
    assert _export(session, tmp_path / "r.zip", "--release", "--key-file", str(tmp_path / "k")) == 0
    assert "1 ingested input(s) with no recorded license" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "pass the Queen key(s)"),
        (["--queen-key", "queen-a=x"], "not base64"),
        (["--queen-key", "queen-a=AAAA"], "3 bytes"),
    ],
)
def test_oasis_publish_names_what_is_wrong_with_its_keys(tmp_path, capsys, args, message):
    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    out = _release(tmp_path, _signer())
    assert run_oasis_subcommand(["publish", str(out), "--root", str(tmp_path / "s"), *args]) == 2
    assert message in capsys.readouterr().err


def test_two_concurrent_publishes_of_one_sequence_cannot_both_land(tmp_path, monkeypatch):
    """publish_release checks equivocation and writes under one lock. A slowed held-release scan widens the
    window -- without the lock both threads see no clash and both land."""
    import threading
    import time

    from maxim.hivemind.store import OasisStore, OasisStoreError

    real_scan = OasisStore._held_release_records

    def slow_scan(self, keys):
        records = real_scan(self, keys)
        time.sleep(0.3)
        return records

    monkeypatch.setattr(OasisStore, "_held_release_records", slow_scan)
    signer, store = _signer(), _store(tmp_path)
    keys = {"queen-a": signer.public_key_b64}
    bundles = [_release(tmp_path, signer, sequence=1, fear=f) for f in (-0.5, -0.9)]
    store.releases_dir.mkdir(parents=True)
    outcomes: list[str] = []

    def publish(path):
        try:
            store.publish_release(path, queen_keys=keys)
            outcomes.append("ok")
        except OasisStoreError:
            outcomes.append("refused")

    threads = [threading.Thread(target=publish, args=(b,)) for b in bundles]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["ok", "refused"]


# ── re-read round (PR C) ───────────────────────────────────────────────────────────────────────


def test_a_signed_release_must_say_which_counter_owns_its_sequence():
    """No default: forgetting the counter is a TypeError, never a silently uncounted release."""
    from maxim.hivemind.signing import SignedRelease

    with pytest.raises(TypeError):
        SignedRelease(signer=_signer(), release_sequence=1, license="CDLA-Permissive-2.0")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="counter"):
        SignedRelease(signer=_signer(), release_sequence=1, license="CDLA-Permissive-2.0", counter="seq.json")


def test_a_replaced_sequence_is_recorded_as_the_one_signed(tmp_path):
    """The re-read's probe: dataclasses.replace(release, release_sequence=9) must commit 9 -- the commit
    reads the value's own fields, so what is signed and what is recorded cannot diverge."""
    import dataclasses

    from maxim.hivemind.bundle import compose_bundle
    from maxim.hivemind.signing import counted_release, next_release_sequence, register_fresh_key
    from tests.unit.test_hivemind_ingest import _node

    counter, signer = tmp_path / "seq.json", _signer()
    register_fresh_key(signer, path=counter)
    release = dataclasses.replace(
        counted_release(signer, license="CDLA-Permissive-2.0", requested=None, path=counter), release_sequence=9
    )
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={"n1": _node()},
        output_path=tmp_path / "r.zip",
        contributor_id=DONOR,
        body_ref=BODY,
        release=release,
    )
    assert _manifest(tmp_path / "r.zip")["release_sequence"] == 9
    assert next_release_sequence(signer, requested=None, path=counter) == 10


def test_open_signer_registers_a_minted_key_in_the_counter_it_is_given(tmp_path):
    from maxim.hivemind.signing import next_release_sequence, open_signer

    counter = tmp_path / "seq.json"
    signer, created = open_signer(signer_identity="q", key_file=tmp_path / "k", counter_path=counter)
    assert created and next_release_sequence(signer, requested=None, path=counter) == 1


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX permission bits")
def test_a_group_readable_key_warns(tmp_path, caplog):
    import logging

    from maxim.hivemind.signing import load_or_create_signer

    key = tmp_path / "k"
    load_or_create_signer(signer_identity="q", key_file=key)
    key.chmod(0o644)
    with caplog.at_level(logging.WARNING, logger="maxim.hivemind.signing"):
        load_or_create_signer(signer_identity="q", key_file=key)
    assert "readable by group/others" in caplog.text


def test_a_corrupt_counter_file_is_named_in_the_error(tmp_path):
    from maxim.hivemind.signing import ReleaseSequenceError, next_release_sequence

    counter = tmp_path / "seq.json"
    counter.write_text("{not json")
    with pytest.raises(ReleaseSequenceError, match="seq.json is not valid JSON"):
        next_release_sequence(_signer(), requested=1, path=counter)


def test_attribution_licenses_warn_on_release(tmp_path, capsys):
    """The permissive list is attribution-free: a release strips per-row provenance, which CC-BY keeps."""
    entries = [
        {"digest": "a" * 64, "signer_key": "ab", "signer_identity": "q", "release_sequence": 1, "license": "CC-BY-4.0"}
    ]
    session = _session(tmp_path, journal_entries=entries)
    assert _export(session, tmp_path / "r.zip", "--release", "--key-file", str(tmp_path / "k")) == 0
    assert "CC-BY-4.0" in capsys.readouterr().err


@pytest.mark.skipif(__import__("os").name == "nt" or __import__("os").geteuid() == 0, reason="POSIX, non-root")
def test_a_store_that_cannot_be_written_warns_instead_of_crashing_the_verb(tmp_path, capsys):
    import hashlib

    from maxim.hivemind.oasis_cli import run_oasis_subcommand

    signer, store = _signer(), _store(tmp_path)
    store.releases_dir.mkdir(parents=True)
    raw = _release(tmp_path, signer).read_bytes()
    (store.releases_dir / f"{hashlib.sha256(raw).hexdigest()}.zip").write_bytes(raw)
    store.releases_dir.chmod(0o555)
    try:
        other = _release(tmp_path, signer, sequence=2, fear=-0.7)
        key = f"queen-a={signer.public_key_b64}"
        run_oasis_subcommand(["publish", str(other), "--root", str(store.root), "--queen-key", key])
        assert "could not migrate release ids" in capsys.readouterr().err
    finally:
        store.releases_dir.chmod(0o755)


def test_pending_migrations_ignore_non_release_files(tmp_path):
    store = _store(tmp_path)
    store.releases_dir.mkdir(parents=True)
    (store.releases_dir / "notes.zip").write_bytes(_release(tmp_path, _signer()).read_bytes())
    assert store.pending_release_migrations() == 0
