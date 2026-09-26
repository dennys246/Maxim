"""End-to-end maxim hive pull / contribute against a real served Oasis (1.2 P2P Slice C).

Proves the consumer composition: a real leader-proxy Oasis with a signed release,
a registry entry naming its Queen key, and `hive pull` fetching → verifying →
delegating to `substrate ingest`. Signed-release paths need the [sign] extra;
the contribute path (unsigned push to the experimental tier) runs unconditionally.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from maxim.hivemind.signing import SignedRelease
from maxim.hivemind import substrate_client as sc  # noqa: F401  (ensures module import path)
from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.hive_cli import run_hive_subcommand
from maxim.hivemind.registry import HiveRegistry
from maxim.hivemind.store import OasisStore
from maxim.runtime.leader_proxy import start_leader_proxy
from maxim.utils.optional_deps import optional_dependency_available

_HAS_CRYPTO = optional_dependency_available("cryptography") and optional_dependency_available("rfc8785")
_needs_crypto = pytest.mark.skipif(
    not _HAS_CRYPTO, reason="signed bundles need the [sign] extra (cryptography + rfc8785)"
)

_KEY = "hive-e2e-bearer"
# Realistic released nodes carry a first-touch geometry stamp, so the default
# (no --allow-unstamped-geometry) pull path is what these exercise.
_EC_NODES = {"node-1": {"modality": "world", "embedding": [0.1, 0.2, 0.3], "domain": None, "geometry": "g1"}}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(port: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.02)
    raise RuntimeError(f"proxy on {port} never came up")


def _start(store):
    port = _free_port()
    server = start_leader_proxy(proxy_port=port, api_key=_KEY, bind_host="127.0.0.1", oasis_store=store)
    assert server is not None
    _wait_ready(port)
    return server, f"http://127.0.0.1:{port}"


def _stop(server) -> None:
    server.shutdown()
    server.server_close()


def _receiver_session(tmp_path):
    sess = tmp_path / "receiver"
    sess.mkdir()
    (sess / "nac.json").write_text(json.dumps({}), encoding="utf-8")
    (sess / "ec.json").write_text(json.dumps({"substrate_nodes": {}}), encoding="utf-8")
    return sess


@_needs_crypto
def test_hive_pull_dry_run_and_apply(tmp_path):
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity="queen-a")
    bundle = tmp_path / "rel.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=bundle,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0"),
    )
    store = OasisStore(tmp_path / "oasis")
    store.publish_release(bundle)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": signer.public_key_b64})
        sess = _receiver_session(tmp_path)

        # dry run: fetch + verify + delegate to `substrate ingest` (no --apply)
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(sess),
                "--receiver-body",
                "minecraft_bench",
                "--api-key",
                _KEY,
            ]
        )
        assert rc == 0
        # dry run wrote nothing durable
        assert not (sess / "substrate_ingest_journal.json").is_file()

        # apply: now the ingest writes state + journal
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(sess),
                "--receiver-body",
                "minecraft_bench",
                "--api-key",
                _KEY,
                "--apply",
            ]
        )
        assert rc == 0
        assert (sess / "substrate_ingest_journal.json").is_file()
    finally:
        _stop(server)


@_needs_crypto
def test_hive_pull_of_a_release_with_nac_rows_re_keys_them_to_the_receiver_agent(tmp_path):
    """A v2 release's NAc rows arrive under the agent token: pull refuses them without a receiver id
    and, with one, the applied state holds them under the receiver's agent -- where NAc reads look."""
    from maxim.hivemind.signing import BundleSigner
    from tests.unit.test_hivemind_ingest import _nac_state

    signer = BundleSigner.generate(signer_identity="queen-a")
    bundle = tmp_path / "rel.zip"
    compose_bundle(
        nac_state=_nac_state(cluster_fear={"donor_agent\x1fnode-1\x1fdrive:oxygen": -0.5}),
        ec_substrate_nodes=_EC_NODES,
        output_path=bundle,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        apply_identity_filter=False,
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0"),
    )
    store = OasisStore(tmp_path / "oasis")
    store.publish_release(bundle)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": signer.public_key_b64})
        sess = _receiver_session(tmp_path)
        pull = ["--registry", reg, "pull", "--from", "alpha", "--session", str(sess)]
        pull += ["--receiver-body", "minecraft_bench", "--api-key", _KEY, "--apply"]
        assert run_hive_subcommand(pull) == 2  # refused: the rows would stay under the token
        assert run_hive_subcommand([*pull, "--receiver-agent-id", "aut"]) == 0
        fear = json.loads((sess / "nac.json").read_text(encoding="utf-8")).get("cluster_fear") or {}
        assert fear and all(key.startswith("aut\x1f") for key in fear)
    finally:
        _stop(server)


@_needs_crypto
def test_hive_pull_ingests_releases_in_ascending_sequence(tmp_path, monkeypatch):
    """The Oasis lists newest first; pull ingests ascending by release_sequence (decision (b)), so the
    receiver's journal sees one signer's releases in order."""
    import maxim.hivemind.bundle as bundle_mod
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity="queen-a")
    store = OasisStore(tmp_path / "oasis")
    for seq, stamp in ((1, "2026-01-01T00:00:00+00:00"), (2, "2026-01-02T00:00:00+00:00")):
        monkeypatch.setattr(bundle_mod, "_utc_now_iso", lambda stamp=stamp: stamp)
        bundle = tmp_path / f"rel{seq}.zip"
        node = {**_EC_NODES["node-1"], "embedding": [0.1 * seq, 0.2, 0.3]}
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes={f"node-{seq}": node},
            output_path=bundle,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
            release=SignedRelease(signer=signer, release_sequence=seq, license="CDLA-Permissive-2.0"),
        )
        store.publish_release(bundle)
    assert [r["release_sequence"] for r in store.list_releases()] == [2, 1]  # the listing is newest first
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": signer.public_key_b64})
        sess = _receiver_session(tmp_path)
        pull = ["--registry", reg, "pull", "--from", "alpha", "--session", str(sess)]
        pull += ["--receiver-body", "minecraft_bench", "--api-key", _KEY, "--apply"]
        assert run_hive_subcommand(pull) == 0
        journal = json.loads((sess / "substrate_ingest_journal.json").read_text(encoding="utf-8"))
        assert [e["release_sequence"] for e in journal["entries"]] == [1, 2]
    finally:
        _stop(server)


@_needs_crypto
def test_a_newly_added_oasis_refuses_its_v1_releases_with_the_fix_and_still_takes_v2(tmp_path, capsys):
    """A fresh `hive add` writes accept_v1: false. A legacy v1 release is skipped with the command that
    takes it; the v2 release in the same listing is still admitted (one refusal never aborts the pull)."""
    from maxim.hivemind.signing import BundleSigner
    from tests.unit._signed_bundle_helpers import write_v1_bundle

    signer = BundleSigner.generate(signer_identity="queen-a")
    store = OasisStore(tmp_path / "oasis")
    unsigned = tmp_path / "legacy-src.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes={"node-9": _EC_NODES["node-1"]},
        output_path=unsigned,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
    )
    store.publish_release(write_v1_bundle(unsigned, signer, out=tmp_path / "legacy.zip"))
    v2 = tmp_path / "rel.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=v2,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0"),
    )
    store.publish_release(v2)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": signer.public_key_b64})
        sess = _receiver_session(tmp_path)
        pull = ["--registry", reg, "pull", "--from", "alpha", "--session", str(sess)]
        pull += ["--receiver-body", "minecraft_bench", "--api-key", _KEY, "--apply"]
        capsys.readouterr()
        assert run_hive_subcommand(pull) == 2
        assert "maxim hive trust alpha --accept-v1" in capsys.readouterr().err
        journal = json.loads((sess / "substrate_ingest_journal.json").read_text(encoding="utf-8"))
        assert [e.get("signature_scheme") for e in journal["entries"]] == [2]
    finally:
        _stop(server)


@_needs_crypto
def test_hive_pull_from_a_loopback_oasis_uses_the_leader_key_implicitly(tmp_path, monkeypatch):
    """public_oasis Phase 0 item 5, the side that must keep working: with no --api-key, a pull from an
    Oasis on THIS machine still authenticates with the local leader key, against a real server."""
    from maxim.hivemind.signing import BundleSigner

    monkeypatch.setattr("maxim.tunnel.keys.read_key", lambda *a, **k: _KEY)
    signer = BundleSigner.generate(signer_identity="queen-a")
    bundle = tmp_path / "rel.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=bundle,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0"),
    )
    store = OasisStore(tmp_path / "oasis")
    store.publish_release(bundle)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": signer.public_key_b64})
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(_receiver_session(tmp_path)),
                "--receiver-body",
                "minecraft_bench",
            ]
        )
        assert rc == 0  # authenticated without --api-key: the server requires _KEY
    finally:
        _stop(server)


@_needs_crypto
def test_hive_pull_untrusted_signer_refused(tmp_path):
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity="queen-a")
    impostor = BundleSigner.generate(signer_identity="queen-a")
    bundle = tmp_path / "rel.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=bundle,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        release=SignedRelease(signer=signer, release_sequence=1, license="CDLA-Permissive-2.0"),
    )
    store = OasisStore(tmp_path / "oasis")
    store.publish_release(bundle)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        # registry holds a DIFFERENT key for queen-a → signature must not verify
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": impostor.public_key_b64})
        sess = _receiver_session(tmp_path)
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(sess),
                "--receiver-body",
                "minecraft_bench",
                "--api-key",
                _KEY,
                "--apply",
            ]
        )
        assert rc == 2
        assert not (sess / "substrate_ingest_journal.json").is_file()
    finally:
        _stop(server)


def _serve_unsigned_release(store, tmp_path):
    """Place an UNSIGNED bundle in the release dir, as a lenient/compromised Oasis would.

    `oasis publish` refuses unsigned, but `list_releases` serves whatever sits in the
    directory — so this is the reachable path the consumer's --allow-unsigned opt-in defends.
    """
    import hashlib

    b = tmp_path / "unsigned_rel.zip"
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=b,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
    )
    raw = b.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    store.releases_dir.mkdir(parents=True, exist_ok=True)
    (store.releases_dir / f"{digest}.zip").write_bytes(raw)
    return digest


def test_unsigned_release_refused_by_default_admitted_under_opt_in(tmp_path):
    """Default trust is Queen-only; disabling verification is an explicit opt-in (Slice D)."""
    store = OasisStore(tmp_path / "oasis")
    _serve_unsigned_release(store, tmp_path)
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        # a registered Queen key exists, but the offered release is unsigned
        HiveRegistry(reg).add("alpha", base, queen_keys={"queen-a": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8="})
        sess = _receiver_session(tmp_path)
        pull = [
            "--registry",
            reg,
            "pull",
            "--from",
            "alpha",
            "--session",
            str(sess),
            "--receiver-body",
            "minecraft_bench",
            "--api-key",
            _KEY,
            "--apply",
        ]
        # default: refused, nothing written
        assert run_hive_subcommand(pull) == 2
        assert not (sess / "substrate_ingest_journal.json").is_file()

        # explicit per-Oasis opt-in: the same unsigned release is now admitted
        run_hive_subcommand(["--registry", reg, "trust", "alpha", "--allow-unsigned"])
        assert run_hive_subcommand(pull) == 0
        assert (sess / "substrate_ingest_journal.json").is_file()
    finally:
        _stop(server)


def test_trusted_source_allow_list_refuses_other_contributors(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    _serve_unsigned_release(store, tmp_path)  # contributor_id = "oasis-alpha"
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base)
        # allow unsigned so the signature gate isn't what refuses, then
        # restrict the operator allow-list to a DIFFERENT contributor
        run_hive_subcommand(["--registry", reg, "trust", "alpha", "--allow-unsigned", "--trust-source", "someone-else"])
        sess = _receiver_session(tmp_path)
        rc = run_hive_subcommand(
            [
                "--registry",
                reg,
                "pull",
                "--from",
                "alpha",
                "--session",
                str(sess),
                "--receiver-body",
                "minecraft_bench",
                "--api-key",
                _KEY,
                "--apply",
            ]
        )
        assert rc == 2
        assert not (sess / "substrate_ingest_journal.json").is_file()
    finally:
        _stop(server)


def test_hive_contribute_over_http(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    server, base = _start(store)
    try:
        reg = str(tmp_path / "hive.json")
        HiveRegistry(reg).add("alpha", base)
        bundle = tmp_path / "contrib.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=bundle,
            contributor_id="peer-9",
            body_ref="minecraft_bench",
        )
        rc = run_hive_subcommand(["--registry", reg, "contribute", str(bundle), "--to", "alpha", "--api-key", _KEY])
        assert rc == 0
        assert len(store.list_contributions()) == 1
    finally:
        _stop(server)
