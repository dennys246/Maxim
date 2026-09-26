"""Bundle signing + verification (Hivemind 1.2 P2P — Slice A).

Activates the reserved manifest ``signature`` / ``signature_algorithm`` /
``signer_identity`` slots that :mod:`maxim.hivemind.bundle` has carried
as ``None`` since 1.0. The trust model is ASYMMETRIC: a Queen-tier Oasis
signs a release with a private key; any consumer verifies with the
corresponding public key (``--trust-key <identity>=<pubkey>``). Bearer
tokens prove "you may talk to this server"; a bundle signature proves
"this substrate is the signer's, unmodified" — a different property, so
this rides no existing key surface (front-gate: needs-own, per
``docs/plans/archive/hivemind_p2p_scope.md`` Slice A).

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
import logging
import os
import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from maxim.tunnel.keys import key_file_path
from maxim.utils.atomic_io import atomic_write_secret
from maxim.utils.optional_deps import require_optional_dependency

logger = logging.getLogger(__name__)

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


# ─── Scheme v2 (docs/plans/oasis_entry_index_v2.md) ─────────────────────────────────────────────
#
# A DETACHED signature over the raw, uncompressed bytes of every ZIP member except the signature member
# itself -- no canonical JSON anywhere in what is signed (the v1 hazards: `default=str` stringifying
# numpy floats, NaN, duplicate keys read differently by two parsers). The domain tag differs from v1's
# ``b"manifest"``, so neither scheme's signature can be replayed as the other's.

#: The v2 domain-separation tag (first framed part of the v2 payload).
BUNDLE_V2_TAG = b"maxim-bundle-v2"

#: The detached-signature member of a v2 bundle.
SIGNATURE_MEMBER = "signature.json"

#: The one v2 scheme number this build implements.
SIGNATURE_SCHEME_V2 = 2


def bundle_signing_payload_v2(members: Mapping[str, bytes]) -> bytes:
    """Return the bytes a scheme-v2 signature covers.

    ``members`` maps each ZIP member name to its UNCOMPRESSED bytes; the signature member is excluded
    here, so the caller may pass the whole archive. Members are ordered by their UTF-8 name bytes and
    each part is framed with an 8-byte big-endian length prefix, exactly as v1 frames its parts.
    Hashing uncompressed bytes keeps the payload -- and so the release's identity -- stable across
    re-zips.
    """
    parts: list[bytes] = [BUNDLE_V2_TAG]
    for name in sorted((n for n in members if n != SIGNATURE_MEMBER), key=lambda n: n.encode("utf-8")):
        parts.append(name.encode("utf-8"))
        parts.append(members[name])
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
        from maxim.hivemind.merge import is_public_identity  # noqa: PLC0415 -- merge is heavy; one owner

        # The public identity grammar, in the TYPE: a signer that cannot publish is never constructed (no
        # key minted and counter-registered under an identity compose would then refuse).
        if not is_public_identity(signer_identity):
            raise ValueError(
                f"signer_identity {signer_identity!r} is not a public identity "
                "([A-Za-z0-9_.@:-], 1-128 chars, no reserved '_' prefix)"
            )
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


#: The largest release_sequence a verifier accepts (a JSON number every parser holds exactly).
MAX_RELEASE_SEQUENCE = 2**53 - 1

#: An SPDX license id (or expression) as the manifest carries it -- displayed, so charset-capped.
_LICENSE = re.compile(r"^[A-Za-z0-9.+\-() ]{1,64}$")


def validate_release_sequence(value: object) -> int:
    """An int (never a bool) in ``1..MAX_RELEASE_SEQUENCE``; ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_RELEASE_SEQUENCE:
        raise ValueError(f"release_sequence must be an int in 1..{MAX_RELEASE_SEQUENCE}, got {value!r}")
    return value


def validate_license(value: object) -> str:
    """An SPDX-shaped license string; ``ValueError`` otherwise."""
    if not isinstance(value, str) or not _LICENSE.match(value):
        raise ValueError(f"license must be an SPDX id (charset-capped, <= 64 chars), got {value!r}")
    return value


class _Uncounted:
    """The type of :data:`UNCOUNTED`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNCOUNTED"


#: A :class:`SignedRelease` whose sequence no release counter owns -- stated explicitly, never defaulted.
UNCOUNTED = _Uncounted()


@dataclass(frozen=True)
class SignedRelease:
    """Everything a SIGNED bundle needs, as one value (docs/plans/oasis_entry_index_v2.md).

    A signed bundle is a release artifact: it carries its signer, its place in that signer's sequence
    and its license, all inside the signature. Passing them as one frozen value means none can be
    forgotten -- a missing field is a ``TypeError`` at construction, not a runtime default. Runtime-
    ephemeral (passed into ``compose_bundle``, never persisted), so outside the CC3 roster.
    """

    signer: BundleSigner
    release_sequence: int
    license: str
    #: The release counter this sequence is recorded in -- REQUIRED, so a producer cannot forget it: the
    #: counter file's path (what :func:`counted_release` sets), or :data:`UNCOUNTED`, said out loud, for a
    #: release no counter owns (tests, hand-composed evidence). ``compose_bundle`` commits
    #: ``(signer, release_sequence)`` to it after writing the signed bundle to its temporary path and
    #: BEFORE moving it onto the output path -- from THIS value's own fields, so a ``dataclasses.replace``
    #: of the sequence cannot desynchronise what is signed from what is recorded. A signed release at its
    #: output path therefore always has its counter record, whatever crashes when.
    counter: Path | _Uncounted

    def __post_init__(self) -> None:
        if not isinstance(self.signer, BundleSigner):
            raise TypeError(f"signer must be a BundleSigner, got {type(self.signer).__name__}")
        if not isinstance(self.counter, (Path, _Uncounted)):
            raise TypeError(f"counter must be a Path or UNCOUNTED, got {self.counter!r}")
        validate_release_sequence(self.release_sequence)
        validate_license(self.license)


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


def signing_key_path(key_file: str | Path | None = None) -> Path:
    """Path to a private signing key (PKCS8 PEM): ``key_file`` when named (``--key-file`` -- e.g. the
    Queen's key, kept apart from this host's development key), else the host default."""
    return Path(key_file).expanduser() if key_file is not None else key_file_path(_PRIVATE_KEY_NAME)


def public_key_path(key_file: str | Path | None = None) -> Path:
    """Path to the public key (base64) beside :func:`signing_key_path` (``<key_file>.pub`` for a named key)."""
    if key_file is None:
        return key_file_path(_PUBLIC_KEY_NAME)
    private = signing_key_path(key_file)
    return private.with_name(private.name + ".pub")


def open_signer(
    *, signer_identity: str, key_file: str | Path | None = None, counter_path: str | Path | None = None
) -> tuple[BundleSigner, bool]:
    """Load the signer at ``key_file`` (default: the host key), or mint + persist one on first use.

    Returns ``(signer, created)``. A minted key is registered in the release counter at once
    (:func:`register_fresh_key`), so "has this key released before?" is counter STATE, never a flag.

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
    from maxim.utils.atomic_io import atomic_write_text

    path = signing_key_path(key_file)
    pub_path = public_key_path(key_file)
    if path.is_file():
        if os.name != "nt" and path.stat().st_mode & 0o077:  # POSIX mode bits mean nothing on Windows
            logger.warning("signing key %s is readable by group/others; chmod 600 it", path)
        signer = BundleSigner.from_private_pem(path.read_bytes(), signer_identity=signer_identity)
        # The .pub is derived state: rewrite it when missing or not this key's (a crash between the two
        # first-use writes, or two concurrent mints), so what an operator shares always matches.
        # Best-effort: a key on read-only media still signs.
        current = pub_path.read_text().strip() if pub_path.is_file() else None
        if current != signer.public_key_b64:
            try:
                atomic_write_text(str(pub_path), signer.public_key_b64 + "\n")
            except OSError as exc:
                logger.warning("cannot rewrite %s from the private key: %s", pub_path, exc)
        return signer, False
    signer = BundleSigner.generate(signer_identity=signer_identity)
    # Registered in the release counter BEFORE the key is written (a stray entry for a key that never got
    # written is harmless; a written key the counter never saw would be refused as restored), with nothing
    # released: its first release is 1 however it is first used -- keygen, or an export that then fails.
    register_fresh_key(signer, path=counter_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_secret(str(path), signer.private_pem().decode("ascii"))
    atomic_write_text(str(pub_path), signer.public_key_b64 + "\n")
    return signer, True


def load_or_create_signer(*, signer_identity: str, key_file: str | Path | None = None) -> BundleSigner:
    """:func:`open_signer` without the ``created`` flag."""
    return open_signer(signer_identity=signer_identity, key_file=key_file)[0]


# --- The producer's release counter (docs/plans/oasis_entry_index_v2.md §Producer) ---

_SEQUENCE_FILE_TYPE = "hive_release_sequence"


class ReleaseSequenceError(ValueError):
    """A release sequence that would repeat or move backwards for its key, or cannot be derived."""


def release_sequence_path() -> Path:
    """``~/.maxim/util/hive_release_sequence.json`` -- the last sequence each signing key released."""
    from maxim.utils.paths import data_home

    return data_home() / "util" / "hive_release_sequence.json"


def _public_key_hex(signer: BundleSigner) -> str:
    return base64.b64decode(signer.public_key_b64).hex()


def _read_counter(target: Path) -> dict[str, dict[str, object]]:
    from maxim.utils.format_version import check_format_version

    if not target.is_file():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseSequenceError(f"release counter {target} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("signers", {}), dict):
        raise ReleaseSequenceError(f"release counter {target} is malformed")
    check_format_version(data, _SEQUENCE_FILE_TYPE)
    return dict(data.get("signers", {}))


def _last_released(signers: Mapping[str, object], key: str, target: Path) -> int | None:
    entry = signers.get(key)
    if entry is not None and not isinstance(entry, dict):
        raise ReleaseSequenceError(f"release counter {target}: key {key[:12]}… has a malformed entry {entry!r}")
    last = entry.get("last") if isinstance(entry, dict) else None
    if last is not None and (isinstance(last, bool) or not isinstance(last, int)):
        raise ReleaseSequenceError(f"release counter {target}: key {key[:12]}… has a non-integer last {last!r}")
    return last


def next_release_sequence(
    signer: BundleSigner,
    *,
    requested: int | None,
    path: str | Path | None = None,
) -> int:
    """The ``release_sequence`` this signing key's next release carries. Reads the counter; writes nothing
    (:func:`commit_release_sequence` records it once the release is composed).

    Keyed by the signing key's public key (hex) -- the same key a receiver's journal keys its ordering
    rules on -- so the Queen key and a development key never share a counter, and experiment bundles can
    never advance the Queen's sequence. ``requested`` (``--release-sequence N``) may only move the counter
    FORWARD. With none: the next number after the last one released (``1`` for a key minted on this host,
    which :func:`open_signer` registers at ``0``); REFUSED for a key this counter has never seen -- one
    minted elsewhere, restored or copied -- which must not restart at 1 and re-use a sequence it already
    released (exactly what a receiver refuses as equivocation).
    """
    target = Path(path) if path is not None else release_sequence_path()
    key = _public_key_hex(signer)
    last = _last_released(_read_counter(target), key, target)
    if requested is not None:
        sequence = validate_release_sequence(requested)
        if last is not None and sequence <= last:
            raise ReleaseSequenceError(
                f"--release-sequence {sequence} does not move forward: this key already released {last} "
                "(a receiver refuses a second release under one sequence)"
            )
        return sequence
    if last is not None:
        return validate_release_sequence(last + 1)
    raise ReleaseSequenceError(
        f"this signing key ({signer.signer_identity!r}, {key[:12]}…) has no release counter on this host: it was "
        "minted before the counter existed, elsewhere, or restored/copied, and may have released before. Pass "
        "--release-sequence N above the last v2 release it published (--release-sequence 1 if it never signed "
        "one; v1 bundles carry no sequence); the counter then continues from N."
    )


def commit_release_sequence(signer: BundleSigner, sequence: int, *, path: str | Path | None = None) -> None:
    """Record that this key released ``sequence`` -- right after the release is composed, BEFORE it can be
    published. Re-checks it still moves forward (another export may have committed meanwhile) and raises
    otherwise; the caller then discards the composed release, so a signed release never exists without its
    counter record and a failed compose never burns a number.

    The read-check-write runs under a ``filelock.FileLock`` (the ``~/.maxim/util/`` rule), so two
    concurrent exports with ONE key can both pass :func:`next_release_sequence` but only one commits;
    the other refuses.
    """
    from filelock import FileLock

    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    target = Path(path) if path is not None else release_sequence_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=10):
        signers = _read_counter(target)
        key = _public_key_hex(signer)
        last = _last_released(signers, key, target)
        if last is not None and sequence <= last:
            raise ReleaseSequenceError(
                f"release sequence {sequence} was taken meanwhile (this key's counter is at {last}); re-run the export"
            )
        signers[key] = {"last": sequence, "signer_identity": signer.signer_identity}
        atomic_write_json(str(target), with_format_version({"signers": signers}))


def counted_release(
    signer: BundleSigner,
    *,
    license: str,
    requested: int | None,
    path: str | Path | None = None,
) -> SignedRelease:
    """The :class:`SignedRelease` for this key's next release, its counter commit bound in: the one way a
    producer (``substrate export --sign``, the orient merge script) gets a sequence. ``compose_bundle``
    commits it between writing the signed bundle and moving it onto its output path."""
    sequence = next_release_sequence(signer, requested=requested, path=path)
    return SignedRelease(
        signer=signer,
        release_sequence=sequence,
        license=license,
        counter=Path(path) if path is not None else release_sequence_path(),
    )


def register_fresh_key(signer: BundleSigner, *, path: str | Path | None = None) -> None:
    """Record a key minted just now (:func:`open_signer` calls it) with nothing released, so its first
    release is 1 -- not refused as an unseen, possibly restored key. Does nothing if the counter knows it."""
    from filelock import FileLock

    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    target = Path(path) if path is not None else release_sequence_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock", timeout=10):
        signers = _read_counter(target)
        key = _public_key_hex(signer)
        if key not in signers:
            signers[key] = {"last": 0, "signer_identity": signer.signer_identity}
            atomic_write_json(str(target), with_format_version({"signers": signers}))
