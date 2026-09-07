"""Guard tests for the Oasis substrate-bundle store + endpoint handlers (1.2 P2P Slice B).

The store's engineering invariants:
- a release bundle MUST be signed (Queen tier); an unsigned bundle is refused
  at :meth:`OasisStore.publish_release` and can only enter the experimental tier;
- a contribution lands in the experimental tier tagged with provenance and is
  NEVER promoted to the release tier as a side effect of receipt;
- a release id that is not a bare sha256 digest never reaches the filesystem.

Signed-bundle tests skip when the optional ``[sign]`` dependency is absent; the
unsigned/experimental paths run unconditionally.
"""

from __future__ import annotations

import json

import pytest

from maxim.hivemind import oasis_endpoints as ep
from maxim.hivemind.bundle import compose_bundle
from maxim.hivemind.store import OasisStore, OasisStoreError
from maxim.utils.optional_deps import optional_dependency_available

_HAS_CRYPTO = optional_dependency_available("cryptography")
_needs_crypto = pytest.mark.skipif(not _HAS_CRYPTO, reason="signed bundles need the [sign] extra (cryptography)")

_EC_NODES = {
    "node-1": {"modality": "world", "embedding": [0.1, 0.2, 0.3], "domain": None},
}


def _unsigned_bundle(path, *, contributor_id="oasis-alpha"):
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=path,
        contributor_id=contributor_id,
        body_ref="minecraft_bench",
    )
    return path


def _signed_bundle(path, *, signer_identity="queen-alpha", contributor_id="oasis-alpha"):
    from maxim.hivemind.signing import BundleSigner

    signer = BundleSigner.generate(signer_identity=signer_identity)
    compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=path,
        contributor_id=contributor_id,
        body_ref="minecraft_bench",
        signer=signer,
    )
    return path


class TestReleaseTier:
    def test_unsigned_release_is_refused(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        bundle = _unsigned_bundle(tmp_path / "b.zip")
        with pytest.raises(OasisStoreError, match="signed"):
            store.publish_release(bundle)
        # nothing landed in the release tier
        assert store.list_releases() == []

    @_needs_crypto
    def test_signed_release_publishes_lists_and_downloads(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        bundle = _signed_bundle(tmp_path / "b.zip")
        release_id = store.publish_release(bundle)
        # listing surfaces a summary keyed by the content digest
        releases = store.list_releases()
        assert len(releases) == 1
        summary = releases[0]
        assert summary["id"] == release_id
        assert summary["contributor_id"] == "oasis-alpha"
        assert summary["signer_identity"] == "queen-alpha"
        assert summary["signature_algorithm"] == "ed25519"
        # the payload comes back byte-identical
        assert store.open_release(release_id) == bundle.read_bytes()

    def test_open_release_rejects_non_digest_id(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        for bad in ("../etc/passwd", "not-hex", "a" * 63, "A" * 64):
            with pytest.raises(OasisStoreError, match="invalid release id"):
                store.open_release(bad)

    def test_open_unknown_release_returns_none(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        assert store.open_release("0" * 64) is None


class TestExperimentalTier:
    def test_contribution_accepted_then_duplicate(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        raw = _unsigned_bundle(tmp_path / "c.zip").read_bytes()

        first = store.accept_contribution(raw, source="10.0.0.5")
        assert first["status"] == "accepted"
        assert first["tier"] == "experimental"
        digest = first["digest"]

        second = store.accept_contribution(raw, source="10.0.0.5")
        assert second == {"digest": digest, "tier": "experimental", "status": "duplicate"}

    def test_contribution_records_provenance(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        raw = _unsigned_bundle(tmp_path / "c.zip", contributor_id="contributor-x").read_bytes()
        receipt = store.accept_contribution(raw, source="192.168.1.9")

        records = store.list_contributions()
        assert len(records) == 1
        rec = records[0]
        assert rec["digest"] == receipt["digest"]
        assert rec["contributor_id"] == "contributor-x"  # self-declared author
        assert rec["source"] == "192.168.1.9"  # transport origin — a different fact
        assert rec["size_bytes"] == len(raw)
        # the bytes are retrievable for a later (gated) promotion
        assert store.open_contribution(receipt["digest"]) == raw

    def test_contribution_does_not_touch_release_tier(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        raw = _unsigned_bundle(tmp_path / "c.zip").read_bytes()
        store.accept_contribution(raw, source="10.0.0.5")
        # receipt never promotes: the release tier stays empty
        assert store.list_releases() == []

    def test_malformed_contribution_is_refused_and_not_stored(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        with pytest.raises(OasisStoreError, match="valid substrate bundle"):
            store.accept_contribution(b"this is not a zip", source="10.0.0.5")
        assert store.list_contributions() == []
        # no orphan blob left behind
        assert not list((tmp_path / "oasis" / "experimental").glob("*.zip"))


class TestConcurrencyAndCorruption:
    def test_concurrent_contributions_all_recorded(self, tmp_path):
        """The provenance log must not lose a record under the threaded proxy.

        Regression guard for the cross-confirmed HIGH finding: a read-modify-
        write on contributions.json without a lock silently drops records when
        two contributions land at once. Fire N distinct bundles from N threads
        and assert every one appears in the audit trail.
        """
        import threading

        store = OasisStore(tmp_path / "oasis")
        n = 12
        raws = []
        for i in range(n):
            raws.append(_unsigned_bundle(tmp_path / f"c{i}.zip", contributor_id=f"peer-{i}").read_bytes())

        start = threading.Barrier(n)
        errors: list[Exception] = []

        def _submit(raw: bytes) -> None:
            start.wait()  # maximize overlap on the read-modify-write window
            try:
                store.accept_contribution(raw, source="10.0.0.1")
            except Exception as exc:  # pragma: no cover — surfaced via assert below
                errors.append(exc)

        threads = [threading.Thread(target=_submit, args=(raw,)) for raw in raws]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, errors
        digests = {r["digest"] for r in store.list_contributions()}
        assert len(digests) == n  # not one lost

    def test_corrupt_log_is_refused_not_silently_reset(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        # accept one so the log exists, then corrupt it on disk
        store.accept_contribution(_unsigned_bundle(tmp_path / "c.zip").read_bytes(), source="10.0.0.1")
        store._contrib_log.write_text("{ this is not valid json", encoding="utf-8")

        # a present-but-corrupt log must fail loud, never be reset-and-clobbered
        with pytest.raises(OasisStoreError, match="present but unreadable"):
            store.list_contributions()
        # a genuinely new contribution (distinct digest, so not a dup short-circuit)
        # must hit the corrupt log and refuse rather than overwrite it
        fresh = _unsigned_bundle(tmp_path / "c2.zip", contributor_id="a-different-peer").read_bytes()
        with pytest.raises(OasisStoreError, match="present but unreadable"):
            store.accept_contribution(fresh, source="10.0.0.2")


class TestEndpointHandlers:
    """The transport-agnostic handlers (no HTTP server)."""

    def test_list_releases_shape(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        resp = ep.handle_list_releases(store)
        assert resp.status == 200
        assert resp.content_type == "application/json"
        assert json.loads(resp.body) == {"releases": []}

    def test_get_unknown_bundle_is_404(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        resp = ep.handle_get_bundle(store, "0" * 64)
        assert resp.status == 404

    def test_get_bundle_bad_id_is_400(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        resp = ep.handle_get_bundle(store, "../secrets")
        assert resp.status == 400

    def test_contribute_empty_body_is_400(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        resp = ep.handle_contribute(store, b"", source="10.0.0.5")
        assert resp.status == 400

    def test_contribute_valid_bundle_is_200(self, tmp_path):
        store = OasisStore(tmp_path / "oasis")
        raw = _unsigned_bundle(tmp_path / "c.zip").read_bytes()
        resp = ep.handle_contribute(store, raw, source="10.0.0.5")
        assert resp.status == 200
        assert json.loads(resp.body)["status"] == "accepted"
