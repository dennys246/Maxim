"""Bundle signing + verification (Hivemind 1.2 P2P — Slice A).

Activates the reserved manifest ``signature`` / ``signature_algorithm`` /
``signer_identity`` slots that :mod:`maxim.hivemind.bundle` has carried
as ``None`` since 1.0. The trust model is ASYMMETRIC: a Queen-tier Oasis
signs a release with a private key; any consumer verifies with the
corresponding public key (``--trust-key <identity>=<pubkey>``). Bearer
tokens prove "you may talk to this server"; a bundle signature proves
"this substrate is the signer's, unmodified" — a different property, so
this rides no existing key surface (front-gate: needs-own, per
``docs/plans/hivemind_p2p_scope.md`` Slice A).

Algorithm: **ed25519** (``cryptography``, optional ``[sign]`` extra). The
``signature_algorithm`` string vocabulary is the one already published in
``docs/user/hivemind_bundle_format.md``; this module implements exactly
``"ed25519"`` and refuses any other declared algorithm at verify time
(an unknown algorithm is unverifiable, never trusted). The design record
and the front-gate "needs-own" justification for signing live in
``docs/plans/maxim_hivemind.md`` (§"Trust topology" + decision point 2).

Signed payload (canonical, identical at sign and verify)
--------------------------------------------------------

The signature covers the manifest MINUS its three signature fields, plus
the raw bytes of every payload slice (``nac.json`` / ``ec.json``) keyed
by filename. See :func:`bundle_signing_payload`. Signing over the
sig-excluded manifest is what lets the three fields be populated AFTER
the signature is computed without invalidating it; covering the raw
slice bytes is what makes tampering with ``nac.json`` fail verification.
"""

from __future__ import annotations

import base64
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from maxim.tunnel.keys import key_file_path
from maxim.utils.atomic_io import atomic_write_secret
from maxim.utils.optional_deps import require_optional_dependency

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

#: The one algorithm this module implements. A bundle declaring anything
#: else is refused at verify time (unverifiable ≠ trusted).
SIGNATURE_ALGORITHM = "ed25519"

#: Manifest fields excluded from the signed payload (they carry the
#: signature itself, so they cannot be part of what is signed).
_SIGNATURE_FIELDS = ("signature", "signature_algorithm", "signer_identity")

#: Default key file names under ``~/.config/maxim/`` (the tunnel-keys dir).
_PRIVATE_KEY_NAME = "hive_signing_key"
_PUBLIC_KEY_NAME = "hive_signing_key.pub"


def bundle_signing_payload(manifest: Mapping[str, object], slices: Mapping[str, str]) -> bytes:
    """Return the canonical bytes a bundle signature covers.

    ``manifest`` MINUS its three signature fields, canonicalized (sorted
    keys, compact separators), followed by each slice's filename + raw
    content string. Deterministic and identical at sign and verify time.

    ``slices`` is keyed by the on-disk filename (``"nac.json"`` /
    ``"ec.json"``) mapping to the exact serialized content string that is
    (or was) written into the ZIP — NOT a re-parsed/re-serialized value,
    so a byte-level tamper is caught.
    """
    signed_manifest = {k: v for k, v in manifest.items() if k not in _SIGNATURE_FIELDS}
    parts: list[bytes] = [
        b"manifest",
        json.dumps(signed_manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"),
    ]
    for name in sorted(slices):
        parts.append(name.encode("utf-8"))
        parts.append(slices[name].encode("utf-8"))
    # Length-prefixed framing (8-byte big-endian per part): unambiguous
    # regardless of part contents, so no two distinct (manifest, slices)
    # can ever collide onto the same signed bytes (defense-in-depth over a
    # separator, which would rely on the separator byte never appearing).
    return b"".join(struct.pack(">Q", len(p)) + p for p in parts)


def _load_ed25519():
    """Import the ed25519 primitives through the canonical optional-dep surface."""
    require_optional_dependency("cryptography", feature="Hivemind bundle signing")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    return Ed25519PrivateKey, Ed25519PublicKey


class BundleSigner:
    """An ed25519 keypair bound to a ``signer_identity`` string.

    The identity is the "who claims to have signed this" label written to
    ``manifest["signer_identity"]``; a consumer trusts an (identity,
    public-key) pair, so the identity travels with the signature.
    """

    def __init__(self, private_key: Ed25519PrivateKey, *, signer_identity: str) -> None:
        if not isinstance(signer_identity, str) or not signer_identity:
            raise ValueError("signer_identity must be a non-empty string")
        if signer_identity.startswith("_"):
            raise ValueError("signer_identity must not start with the reserved '_' prefix")
        self._private_key = private_key
        self.signer_identity = signer_identity

    @classmethod
    def generate(cls, *, signer_identity: str) -> BundleSigner:
        """Mint a fresh keypair for ``signer_identity``."""
        Ed25519PrivateKey, _ = _load_ed25519()
        return cls(Ed25519PrivateKey.generate(), signer_identity=signer_identity)

    @classmethod
    def from_private_pem(cls, pem: bytes, *, signer_identity: str) -> BundleSigner:
        """Load a signer from a PKCS8 PEM private key."""
        require_optional_dependency("cryptography", feature="Hivemind bundle signing")
        from cryptography.hazmat.primitives.serialization import load_pem_private_key  # noqa: PLC0415

        key = load_pem_private_key(pem, password=None)
        return cls(key, signer_identity=signer_identity)  # type: ignore[arg-type]

    def sign_payload(self, payload: bytes) -> str:
        """Return the base64 ed25519 signature over ``payload``."""
        return base64.b64encode(self._private_key.sign(payload)).decode("ascii")

    @property
    def public_key_b64(self) -> str:
        """The base64 of the 32 raw public-key bytes — the shareable trust anchor."""
        from cryptography.hazmat.primitives.serialization import (  # noqa: PLC0415
            Encoding,
            PublicFormat,
        )

        raw = self._private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        return base64.b64encode(raw).decode("ascii")

    def private_pem(self) -> bytes:
        """Serialize the private key as unencrypted PKCS8 PEM (for persistence)."""
        from cryptography.hazmat.primitives.serialization import (  # noqa: PLC0415
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        return self._private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())


def verify_payload(payload: bytes, signature_b64: str, public_key_b64: str) -> bool:
    """Return True iff ``signature_b64`` is a valid ed25519 signature over ``payload``.

    Never raises on a bad signature or malformed key/signature material —
    returns False. Raises only if the optional dependency is missing.
    """
    _, Ed25519PublicKey = _load_ed25519()
    from cryptography.exceptions import InvalidSignature  # noqa: PLC0415

    try:
        pub_raw = base64.b64decode(public_key_b64, validate=True)
        sig = base64.b64decode(signature_b64, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(pub_raw)
    except (ValueError, TypeError):
        return False
    try:
        public_key.verify(sig, payload)
        return True
    except InvalidSignature:
        return False


# --- Key file management (rides the tunnel-keys ~/.config/maxim/ convention) ---


def signing_key_path() -> Path:
    """Path to the persisted private signing key (PKCS8 PEM)."""
    return key_file_path(_PRIVATE_KEY_NAME)


def public_key_path() -> Path:
    """Path to the persisted public key (base64)."""
    return key_file_path(_PUBLIC_KEY_NAME)


def load_or_create_signer(*, signer_identity: str) -> BundleSigner:
    """Load the persisted signer, or mint + persist one on first use.

    The private key is written through ``atomic_write_secret`` — 0600 from
    fd creation (the key never sits umask-wide, not even in the tmp
    window) — and the public key alongside it, world-readable, is the
    string a consumer trusts.

    Note: the keypair is the trust anchor; ``signer_identity`` is only the
    LABEL bound to this call. Re-invoking with a different ``signer_identity``
    re-labels the SAME persisted key, so the identity a receiver trusts can
    drift from the key. Keep one identity per key (or delete the key file to
    rotate) — the identity is not persisted beside the key.
    """
    path = signing_key_path()
    if path.is_file():
        return BundleSigner.from_private_pem(path.read_bytes(), signer_identity=signer_identity)
    signer = BundleSigner.generate(signer_identity=signer_identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_secret(str(path), signer.private_pem().decode("ascii"))
    public_key_path().write_text(signer.public_key_b64 + "\n")
    return signer
