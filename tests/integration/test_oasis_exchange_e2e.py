"""End-to-end substrate exchange over real HTTP (1.2 P2P Slice B).

Proves the composition, not the pieces: a real ``leader_proxy`` on a loopback
port with an :class:`OasisStore`, driven by the production
:mod:`maxim.hivemind.substrate_client` over httpx. The three endpoints
(releases / bundle / contribute), the Bearer auth reuse, and the
store-gates-the-surface 404 all exercise the same path Slice C's CLI will.

Signed-release paths skip without the ``[sign]`` extra; the contribution and
auth/404 paths run unconditionally.
"""

from __future__ import annotations

import socket
import time

import pytest

from maxim.hivemind import substrate_client as sc
from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.store import OasisStore
from maxim.runtime.leader_proxy import start_leader_proxy
from maxim.utils import http
from maxim.utils.optional_deps import optional_dependency_available

_HAS_CRYPTO = optional_dependency_available("cryptography")
_needs_crypto = pytest.mark.skipif(not _HAS_CRYPTO, reason="signed bundles need the [sign] extra (cryptography)")

_KEY = "oasis-test-bearer-key"
_EC_NODES = {"node-1": {"modality": "world", "embedding": [0.1, 0.2, 0.3], "domain": None}}


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


def _start(store, *, api_key=_KEY):
    port = _free_port()
    server = start_leader_proxy(proxy_port=port, api_key=api_key, bind_host="127.0.0.1", oasis_store=store)
    assert server is not None
    _wait_ready(port)
    return server, f"http://127.0.0.1:{port}"


def _stop(server) -> None:
    server.shutdown()
    server.server_close()
    # No module-global reset needed: start_leader_proxy only stashes the
    # singleton for the DEFAULT port, and these servers bind custom ports.


def _unsigned_bundle(path, *, contributor_id="oasis-alpha"):
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=path,
        contributor_id=contributor_id,
        body_ref="minecraft_bench",
    )
    return path


def _signed_bundle(path, *, signer_identity="queen-alpha"):
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity=signer_identity)
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=path,
        contributor_id="oasis-alpha",
        body_ref="minecraft_bench",
        signer=signer,
    )
    return path


@_needs_crypto
def test_publish_then_pull_over_http(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    release_id = store.publish_release(_signed_bundle(tmp_path / "rel.zip"))
    server, base = _start(store)
    try:
        releases = sc.list_releases(base, api_key=_KEY)
        assert [r["id"] for r in releases] == [release_id]

        dest = tmp_path / "pulled.zip"
        sc.fetch_bundle(base, release_id, dest, api_key=_KEY)
        assert dest.read_bytes() == (tmp_path / "rel.zip").read_bytes()
    finally:
        _stop(server)


def test_contribute_over_http_lands_in_experimental(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    server, base = _start(store)
    try:
        bundle = _unsigned_bundle(tmp_path / "contrib.zip", contributor_id="peer-7")
        receipt = sc.contribute(base, bundle, api_key=_KEY)
        assert receipt["status"] == "accepted"
        assert receipt["tier"] == "experimental"

        # server-side: it landed in experimental with the peer's IP as source,
        # and the release tier is untouched.
        records = store.list_contributions()
        assert len(records) == 1
        assert records[0]["contributor_id"] == "peer-7"
        assert records[0]["source"] == "127.0.0.1"
        assert store.list_releases() == []
    finally:
        _stop(server)


def test_fetch_unknown_release_raises(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    server, base = _start(store)
    try:
        with pytest.raises(sc.SubstrateExchangeError):
            sc.fetch_bundle(base, "0" * 64, tmp_path / "x.zip", api_key=_KEY)
    finally:
        _stop(server)


def test_wrong_bearer_key_is_rejected(tmp_path):
    store = OasisStore(tmp_path / "oasis")
    server, base = _start(store)
    try:
        with pytest.raises(http.HTTPAuthError):
            sc.list_releases(base, api_key="wrong-key")
    finally:
        _stop(server)


def test_substrate_surface_absent_without_a_store(tmp_path):
    # A leader with no OasisStore exposes no substrate surface — releases 404.
    server, base = _start(None)
    try:
        with pytest.raises(http.HTTPClientError):
            sc.list_releases(base, api_key=_KEY)
    finally:
        _stop(server)
