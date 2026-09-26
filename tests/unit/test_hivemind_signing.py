"""Guard tests for Hivemind bundle signing + verification (1.2 P2P Slice A).

Covers the sign→verify round trip, tamper/wrong-key/unknown-algorithm
rejection, and the ingest ``require_signed`` refusal-vs-admit behaviour.
Skipped whole when the optional ``[sign]`` dependency (``cryptography``)
is absent — the non-signing paths are unaffected and tested elsewhere.
"""

from __future__ import annotations

import json
import zipfile

import pytest

from maxim.utils.optional_deps import optional_dependency_available

pytestmark = pytest.mark.skipif(
    not (optional_dependency_available("cryptography") and optional_dependency_available("rfc8785")),
    reason="bundle signing needs the [sign] extra (cryptography)",
)

from maxim.hivemind.bundle import (  # noqa: E402
    compose_bundle,
    read_bundle_manifest,
    verify_bundle_signature,
    verify_bundle_zip,
)
from maxim.hivemind.signing import (  # noqa: E402
    SIGNATURE_MEMBER,
    BundleSigner,
    bundle_signing_payload,
    verify_payload,
)
from tests.unit._signed_bundle_helpers import (  # noqa: E402
    read_members,
    release,
    rewrite_manifest,
    rewrite_member,
    write_members,
    write_v1_bundle,
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
        release=release(signer),
    )
    return out, manifest


class TestSignVerifyRoundTrip:
    def test_a_v2_release_verifies_with_the_signer_key(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, manifest = _compose_signed(tmp_path, signer)
        assert manifest["signer_identity"] == "queen-alpha" and manifest["release_sequence"] == 1
        assert "signature" not in manifest  # v3: the signature is the detached signature.json member
        assert SIGNATURE_MEMBER in read_members(bundle)
        ok, reason = verify_bundle_signature(
            bundle, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True
        )
        assert ok, reason

    def test_verification_survives_a_re_zip(self, tmp_path):
        """The payload digest is over UNCOMPRESSED member bytes: a re-zip is the same release."""
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        rezip = write_members(tmp_path / "rezip.zip", dict(reversed(list(read_members(bundle).items()))))
        keys = {"queen-alpha": signer.public_key_b64}
        with zipfile.ZipFile(bundle) as a, zipfile.ZipFile(rezip) as b:
            va, vb = (
                verify_bundle_zip(a, trusted_keys=keys, accept_v1=True),
                verify_bundle_zip(b, trusted_keys=keys, accept_v1=True),
            )
        assert va.ok and vb.ok and va.payload_digest == vb.payload_digest


class TestRejection:
    def test_tampered_slice_bytes_fail(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        rewrite_member(bundle, "ec.json", read_members(bundle)["ec.json"] + b" ")
        ok, reason = verify_bundle_signature(
            bundle, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True
        )
        assert not ok and "does not verify" in reason

    def test_an_appended_member_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        rewrite_member(bundle, "extra.json", b"{}")
        ok, reason = verify_bundle_signature(
            bundle, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True
        )
        assert not ok and "members differ" in reason and "extra.json" in reason

    def test_an_appended_member_the_manifest_declares_still_fails_the_signature(self, tmp_path):
        """The member-set rule passes (manifest and archive agree) -- the signature over every member does not."""
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        rewrite_member(bundle, "extra.json", b"{}")
        rewrite_manifest(bundle, lambda m: m["contents"].update({"extra": {"file": "extra.json"}}))
        ok, reason = verify_bundle_signature(
            bundle, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True
        )
        assert not ok and "does not verify" in reason

    def test_wrong_key_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        impostor = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        ok, _ = verify_bundle_signature(bundle, trusted_keys={"queen-alpha": impostor.public_key_b64}, accept_v1=True)
        assert not ok

    def test_untrusted_signer_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        bundle, _ = _compose_signed(tmp_path, signer)
        ok, reason = verify_bundle_signature(bundle, trusted_keys={"queen-beta": signer.public_key_b64}, accept_v1=True)
        assert not ok
        assert "not among" in reason

    @pytest.mark.parametrize(
        ("doc", "why"),
        [
            ({"signature_scheme": 2, "signature_algorithm": "rsa-9000", "signature": "x"}, "unsupported"),
            ({"signature_scheme": 3, "signature_algorithm": "ed25519", "signature": "x"}, "unknown signature_scheme"),
            (
                {"signature_scheme": True, "signature_algorithm": "ed25519", "signature": "x"},
                "unknown signature_scheme",
            ),
        ],
    )
    def test_an_unknown_scheme_or_algorithm_is_refused_never_read_as_v1(self, tmp_path, doc, why):
        signer = BundleSigner.generate(signer_identity="q")
        bundle, _ = _compose_signed(tmp_path, signer)
        rewrite_member(bundle, SIGNATURE_MEMBER, json.dumps(doc).encode())
        ok, reason = verify_bundle_signature(bundle, trusted_keys={"q": signer.public_key_b64}, accept_v1=True)
        assert not ok and why in reason

    def test_unsigned_bundle_reports_no_signature(self, tmp_path):
        out = tmp_path / "unsigned.zip"
        manifest = compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=out,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
        )
        assert "signature" not in manifest and manifest["license"] is None
        ok, reason = verify_bundle_signature(out, trusted_keys={"anyone": "x"}, accept_v1=True)
        assert not ok
        assert "no signature" in reason

    def test_a_release_carries_its_own_license(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="q")
        with pytest.raises(ValueError, match="carries its license"):
            compose_bundle(
                nac_state=None,
                ec_substrate_nodes=_EC_NODES,
                output_path=tmp_path / "b.zip",
                contributor_id="oasis-alpha",
                body_ref="minecraft_bench",
                release=release(signer),
                license="MIT",
            )


class TestLegacyV1:
    """A schema-2 bundle signed under v1 (the pre-v2 shape) still verifies -- read BEFORE migration."""

    def _v1(self, tmp_path, signer, *, schema_version=2):
        unsigned = tmp_path / "unsigned.zip"
        compose_bundle(
            nac_state=None,
            ec_substrate_nodes=_EC_NODES,
            output_path=unsigned,
            contributor_id="oasis-alpha",
            body_ref="minecraft_bench",
        )
        return write_v1_bundle(unsigned, signer, out=tmp_path / "v1.zip", schema_version=schema_version)

    def test_a_v1_bundle_verifies_after_the_schema_bump(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        v1 = self._v1(tmp_path, signer)
        ok, reason = verify_bundle_signature(v1, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True)
        assert ok, reason
        assert read_bundle_manifest(v1)["schema_version"] == 3  # the migrated view still reads

    def test_a_v1_bundle_is_refused_where_v1_is_not_accepted(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        with zipfile.ZipFile(self._v1(tmp_path, signer)) as zf:
            result = verify_bundle_zip(zf, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=False)
        assert not result.ok and "v2 releases only" in result.reason

    def test_a_v1_signature_on_a_schema_3_manifest_is_a_downgrade(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        v1 = self._v1(tmp_path, signer, schema_version=3)  # a v1 signature that VERIFIES over schema 3
        ok, reason = verify_bundle_signature(v1, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True)
        assert not ok and "downgrade" in reason

    def test_a_v1_tampered_slice_fails(self, tmp_path):
        signer = BundleSigner.generate(signer_identity="queen-alpha")
        v1 = self._v1(tmp_path, signer)
        rewrite_member(v1, "ec.json", read_members(v1)["ec.json"] + b" ")
        ok, reason = verify_bundle_signature(v1, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True)
        assert not ok and "does not verify" in reason


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
        ok, reason = verify_bundle_signature(
            bundle, trusted_keys={"queen-alpha": signer.public_key_b64}, accept_v1=True
        )
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
