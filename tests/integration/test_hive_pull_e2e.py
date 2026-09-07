"""End-to-end maxim hive pull / contribute against a real served Oasis (1.2 P2P Slice C).

Proves the consumer composition: a real leader-proxy Oasis with a signed release,
a registry entry naming its Queen key, and `hive pull` fetching → verifying →
delegating to `substrate ingest`. Signed-release paths need the [sign] extra;
the contribute path (unsigned experimental) runs unconditionally.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

from maxim.hivemind import substrate_client as sc  # noqa: F401  (ensures module import path)
from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.hive_cli import run_hive_subcommand
from maxim.hivemind.registry import HiveRegistry
from maxim.hivemind.store import OasisStore
from maxim.runtime.leader_proxy import start_leader_proxy
from maxim.utils.optional_deps import optional_dependency_available

_HAS_CRYPTO = optional_dependency_available("cryptography")
_needs_crypto = pytest.mark.skipif(not _HAS_CRYPTO, reason="signed bundles need the [sign] extra (cryptography)")

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
        signer=signer,
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
        signer=signer,
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
