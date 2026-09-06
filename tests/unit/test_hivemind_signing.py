"""Guard tests for Hivemind bundle signing + verification (1.2 P2P Slice A).

Covers the sign→verify round trip, tamper/wrong-key/unknown-algorithm
rejection, and the ingest ``require_signed`` refusal-vs-admit behaviour.
Skipped whole when the optional ``[sign]`` dependency (``cryptography``)
is absent — the non-signing paths are unaffected and tested elsewhere.
"""

from __future__ import annotations

import pytest

from maxim.utils.optional_deps import optional_dependency_available

pytestmark = pytest.mark.skipif(
    not optional_dependency_available("cryptography"),
    reason="bundle signing needs the [sign] extra (cryptography)",
)

from maxim.hivemind.bundle import (  # noqa: E402
    compose_bundle,
    read_bundle_manifest,
    verify_bundle_signature,
    verify_bundle_signature_parts,
)
from maxim.hivemind.signing import (  # noqa: E402
    SIGNATURE_ALGORITHM,
    BundleSigner,
    bundle_signing_payload,
    verify_payload,
)

_EC_NODES = {
    "node-1": {"modality": "world", "embedding": [0.1, 0.2, 0.3], "domain": None},
}


def _compose_signed(tmp_path, signer, *, contributor_id="oasis-alpha"):
    out = tmp_path / "bundle.zip"
    manifest = compose_bundle(
        nac_state=None,
        ec_substrate_nodes=_EC_NODES,
        output_path=out,
        contributor_id=contributor_id,
        body_ref="minecraft_bench",
        signer=signer,
    )
    return out, manifest


def _slices_from_bundle(bundle_path):
    import zipfile

    with zipfile.ZipFile(bundle_path, "r") as zf:
        return {name: zf.read(name).decode("utf-8") for name in zf.namelist() if name != "manifest.json"}


class TestSignVerifyRoundTrip:
    def test_signed_bundle_verifies_with_the_signer_key(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, manifest = _compose_signed(tmp_path, signer)
        assert manifest["signature_algorithm"] == SIGNATURE_ALGORITHM
        assert manifest["signer_identity"] == "queen-alpha"
        ok, reason = verify_bundle_signature(bundle, trusted_keys={"queen-alpha": signer.public_key_b64})
        assert ok, reason

    def test_manifest_signature_survives_disk_round_trip(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        # Re-read the manifest from disk (parses + would migrate) and verify.
        manifest = read_bundle_manifest(bundle)
        slices = _slices_from_bundle(bundle)
        ok, reason = verify_bundle_signature_parts(
            manifest, slices, trusted_keys={"queen-alpha": signer.public_key_b64}
        )
        assert ok, reason


class TestRejection:
    def test_tampered_slice_bytes_fail(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, manifest = _compose_signed(tmp_path, signer)
        slices = _slices_from_bundle(bundle)
        tampered = dict(slices)
        # flip one byte of the ec slice
        (name, content) = next(iter(slices.items()))
        tampered[name] = content + " "
        ok, reason = verify_bundle_signature_parts(
            manifest, tampered, trusted_keys={"queen-alpha": signer.public_key_b64}
        )
        assert not ok
        assert "does not verify" in reason

    def test_wrong_key_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        impostor = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        ok, _ = verify_bundle_signature(bundle, trusted_keys={"queen-alpha": impostor.public_key_b64})
        assert not ok

    def test_untrusted_signer_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        ok, reason = verify_bundle_signature(bundle, trusted_keys={"queen-beta": signer.public_key_b64})
        assert not ok
        assert "not among" in reason

    def test_unknown_algorithm_refused(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="q")
        _, manifest = _compose_signed(tmp_path, signer)
        manifest = dict(manifest)
        manifest["signature_algorithm"] = "rsa-9000"
        ok, reason = verify_bundle_signature_parts(manifest, {}, trusted_keys={"q": signer.public_key_b64})
        assert not ok
        assert "unsupported" in reason

    def test_unsigned_bundle_reports_no_signature(self, tmp_path):
        out = tmp_path / "unsigned.zip"
        manifest = compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=out,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
        )
        assert manifest["signature"] is None
        ok, reason = verify_bundle_signature(out, trusted_keys={"anyone": "x"})
        assert not ok
        assert "no signature" in reason

    def test_signer_and_explicit_signature_conflict(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="q")
        with pytest.raises(ValueError, match="EITHER signer"):
            compose_bundle(
                nac_state=None,
                ec_substrate_nodes=_EC_NODES,
                output_path=tmp_path / "b.zip",
                contributor_id="oasis-alpha",
                body_ref="minecraft_bench",
                signer=signer,
                signature="deadbeef",
            )


class TestPayloadPrimitive:
    def test_payload_drops_signature_fields(self):
        base = {"contributor_id": "a", "signature": "X", "signature_algorithm": "ed25519", "signer_identity": "q"}
        p1 = bundle_signing_payload(base, {})
        p2 = bundle_signing_payload({"contributor_id": "a", "signature": None}, {})
        assert p1 == p2  # the three sig fields never enter the signed bytes

    def test_verify_payload_roundtrip_and_reject(self):
        signer = BundleSigner.generate(signer_identity="q")
        payload = b"hello substrate"
        sig = signer.sign_payload(payload)
        assert verify_payload(payload, sig, signer.public_key_b64)
        assert not verify_payload(b"hello substrat3", sig, signer.public_key_b64)
        assert not verify_payload(payload, sig, "not-base64!!")


class TestDiskKey:
    def test_mint_then_load_same_key_and_0600(self, tmp_path, monkeypatch):
        import platform as _platform

        from maxim.hivemind import signing as sg

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        first = sg.load_or_create_signer(signer_identity="queen-alpha")
        assert sg.signing_key_path().is_file()
        # second call loads the SAME persisted key (identical public key)
        second = sg.load_or_create_signer(signer_identity="queen-alpha")
        assert first.public_key_b64 == second.public_key_b64
        # public key file holds the base64 pubkey
        assert sg.public_key_path().read_text().strip() == first.public_key_b64
        # private key is 0600 on POSIX (never umask-wide)
        if _platform.system() != "Windows":
            assert (sg.signing_key_path().stat().st_mode & 0o777) == 0o600

    def test_persisted_key_signs_a_verifiable_bundle(self, tmp_path, monkeypatch):
        from maxim.hivemind import signing as sg

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        signer = sg.load_or_create_signer(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        ok, reason = verify_bundle_signature(bundle, trusted_keys={"queen-alpha": signer.public_key_b64})
        assert ok, reason


class TestIngestRequireSigned:
    def _receiver(self):
        return {}, {}

    def test_require_signed_refuses_unsigned(self, tmp_path):
        from maxim.hivemind.ingest import IngestionJournal, IngestRefused, ingest_bundle

        out = tmp_path / "unsigned.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=out,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
        )
        journal = IngestionJournal(tmp_path / "journal.json")
        with pytest.raises(IngestRefused) as exc:
            ingest_bundle(
                out,
                receiver_nac={},
                receiver_ec_nodes={},
                receiver_body="minecraft_bench",
                trusted_sources=frozenset({"oasis-alpha"}),
                journal=journal,
                allow_unstamped_geometry=True,
                require_signed=True,
                trusted_keys={},
            )
        assert exc.value.duty == "signature"

    def test_require_signed_admits_trusted_signature(self, tmp_path):
        from maxim.hivemind.ingest import IngestionJournal, ingest_bundle

        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        journal = IngestionJournal(tmp_path / "journal.json")
        report = ingest_bundle(
            bundle,
            receiver_nac={},
            receiver_ec_nodes={},
            receiver_body="minecraft_bench",
            trusted_sources=frozenset({"oasis-alpha"}),
            journal=journal,
            allow_unstamped_geometry=True,
            require_signed=True,
            trusted_keys={"queen-alpha": signer.public_key_b64},
        )
        assert any("signature verified" in n for n in report.notes)
