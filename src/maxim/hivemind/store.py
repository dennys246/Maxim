"""Oasis substrate-bundle store — the two-tier on-disk home for shared bundles.

1.2 P2P Slice B. An :class:`OasisStore` is the durable side of the substrate
exchange surface an Oasis serves (the HTTP endpoints live in
:mod:`maxim.hivemind.oasis_endpoints`; the client that reaches them lives in
:mod:`maxim.hivemind.substrate_client`). Two tiers, mirroring the trust model
of ``sharing_threat_model.md`` §5:

- **releases/** — Queen-tier published bundles. VERIFIED at the door:
  :meth:`publish_release` refuses anything that is not a v2 release verifying
  against the Queen keys it is given, and a release equivocating against one it
  holds (release format v2).
- **experimental/** — received contributions. :meth:`accept_contribution`
  lands a foreign bundle here tagged with provenance and NOTHING more — it is
  never merged, never promoted to the release tier as a side effect of receipt.
  Promotion (running the V1–V10 ``ingest_bundle`` gauntlet + re-signing) is a
  separate, gated Slice D operation. This is the engineering invariant the slice
  establishes: *a contribution's arrival changes no trusted state.*

Bundles are content-addressed: a Queen release by its signed-payload digest (release format v2 -- a
re-zipped copy is the same release), a contribution by the sha256 of its raw ZIP bytes. Both are
``^[0-9a-f]{64}$``, so the on-disk path is never built from unvalidated caller input (traversal-proof
by construction; :func:`_validate_release_id` re-checks the shape anyway).

All writes go through :func:`maxim.utils.atomic_io.atomic_write_bytes` /
``atomic_write_json`` (the canonical writers); the store never hand-rolls a
rename.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from maxim.hivemind.bundle import (
    BundleVerification,
    bounded_member_read,
    bundle_signature_scheme,
    content_payload_digest,
    find_equivocation,
    read_bundle_manifest,
    read_bundle_manifest_bytes,
    stored_schema_version,
    verify_bundle_zip,
)
from maxim.hivemind.merge import is_public_identity
from maxim.hivemind.signing import SIGNATURE_ALGORITHM, SIGNATURE_MEMBER, SIGNATURE_SCHEME_V2
from maxim.utils.atomic_io import atomic_write_bytes, atomic_write_json
from maxim.utils.format_version import check_format_version, with_format_version

logger = logging.getLogger(__name__)

_CONTRIB_FILE_TYPE = "oasis_contribution_log"
_RELEASE_ID_RE = re.compile(r"^[0-9a-f]{64}$")

# Everything ``read_bundle_manifest`` raises on a bundle that is not a
# well-formed, kind-correct substrate ZIP: a bad archive, a missing/duplicate
# manifest entry, a schema/kind/format-version mismatch (ValueError), or an I/O
# failure. Narrow by intent — never a bare ``except``.
_MALFORMED_BUNDLE = (zipfile.BadZipFile, KeyError, ValueError, OSError)

# Manifest keys surfaced in a release listing (a summary — never the payload).
_SUMMARY_KEYS = (
    "contributor_id",
    "domain",
    "body_ref",
    "created_at",
    "schema_version",
    "signer_identity",
    "affordance_namespace",
    "release_sequence",
    "license",
)


class OasisStoreError(Exception):
    """A store operation was refused (unsigned release, malformed bundle)."""


def _verify(raw: bytes, queen_keys: Mapping[str, str]) -> BundleVerification:
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            return verify_bundle_zip(zf, trusted_keys=dict(queen_keys), accept_v1=True)
    except zipfile.BadZipFile as exc:
        return BundleVerification(False, f"not a ZIP archive: {exc}")


_LOG_SAFE = re.compile(r"^[\x20-\x7e]{0,128}$")


def _check_contribution_strings(manifest: dict[str, Any]) -> None:
    if not is_public_identity(manifest.get("contributor_id")):
        raise OasisStoreError(f"contributor_id {manifest.get('contributor_id')!r} is not a public identity")
    signer = manifest.get("signer_identity")
    if signer is not None and not is_public_identity(signer):
        raise OasisStoreError(f"signer_identity {signer!r} is not a public identity")
    for field in ("domain", "body_ref"):
        value = manifest.get(field)
        if value is not None and not (isinstance(value, str) and _LOG_SAFE.match(value)):
            raise OasisStoreError(f"{field} {value!r} is not a short printable string")


def _claimed_algorithm(raw: bytes) -> str | None:
    """The algorithm a v2 release's signature member CLAIMS (unverified -- the store does not check a
    contribution's signature)."""
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            if SIGNATURE_MEMBER not in zf.namelist():
                return None
            doc = json.loads(bounded_member_read(zf, SIGNATURE_MEMBER, max_bytes=64 * 1024))
    except (zipfile.BadZipFile, ValueError):
        return None
    algorithm = doc.get("signature_algorithm") if isinstance(doc, dict) else None
    return algorithm if isinstance(algorithm, str) and _LOG_SAFE.match(algorithm) else None


def _has_signature_member(raw: bytes) -> bool:
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            return SIGNATURE_MEMBER in zf.namelist()
    except zipfile.BadZipFile:
        return False


def _claimed_signer(raw: bytes) -> str | None:
    try:
        manifest = read_bundle_manifest_bytes(raw)
    except _MALFORMED_BUNDLE:
        return None
    signer = manifest.get("signer_identity")
    return signer if isinstance(signer, str) and signer else None


def _verify_by_key(raw: bytes, keys: list[str]) -> BundleVerification | None:
    """The verification of ``raw`` under whichever of ``keys`` signed it -- the release's own claimed
    identity bound to each key in turn, so the result depends on key bytes, not on a registry label."""
    signer = _claimed_signer(raw)
    if signer is None:
        return None
    for key in keys:
        result = _verify(raw, {signer: key})
        if result.ok:  # a v1 result carries no release_sequence, so it binds no sequence
            return result
    return None


def _verifies_as(raw: bytes, release_id: str, queen_keys: Mapping[str, str]) -> bool:
    held = _verify_by_key(raw, sorted(set(queen_keys.values())))
    return held is not None and held.payload_digest == release_id


def is_valid_release_id(release_id: str) -> bool:
    """True iff ``release_id`` is a bare sha256 hex digest (no separators, no ``..``).

    The canonical shape check for a release id — reused by the client transport
    and the ``hive pull`` loop so an id from an untrusted Oasis can never reach a
    filesystem path (traversal / absolute-path escape) before it is validated.
    """
    return bool(_RELEASE_ID_RE.match(release_id))


def _validate_release_id(release_id: str) -> str:
    """Return ``release_id`` iff it is a bare sha256 hex digest, else raise.

    The id becomes a filename; anything but ``[0-9a-f]{64}`` (no separators,
    no ``..``) is refused before it can reach the filesystem.
    """
    if not is_valid_release_id(release_id):
        raise OasisStoreError(f"invalid release id {release_id!r} (expected a sha256 hex digest)")
    return release_id


class OasisStore:
    """Two-tier bundle store rooted at ``root`` (``releases/`` + ``experimental/``)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.releases_dir = self.root / "releases"
        self.experimental_dir = self.root / "experimental"
        self._contrib_log = self.experimental_dir / "contributions.json"
        # One store instance is shared across the proxy's handler threads
        # (ThreadingMixIn). This serializes the accept path's dup-check → write
        # → provenance-append so concurrent contributions cannot lose an audit
        # record via a read-modify-write race on the log.
        self._accept_lock = threading.Lock()

    # ── release tier (Queen) ─────────────────────────────────────────────

    def publish_release(self, bundle_path: str | Path, *, queen_keys: Mapping[str, str]) -> str:
        """VERIFY a v2 release and publish it into the release tier; return its release id.

        ``queen_keys`` (identity -> base64 public key; required, so forgetting it is a TypeError, not an
        unverified publish) are the Queen keys this Oasis publishes under. The store refuses
        (``OasisStoreError``) anything that is not a v2 release verifying against one of them -- the
        signature, the signed entry index and the signed license (release format v2, decision (b)/(d))
        -- and a release that EQUIVOCATES against one it already holds (same signing KEY and sequence, a
        different payload; ``bundle.find_equivocation``, the predicate receivers apply too). The check
        compares key bytes, never labels: a held release counts under whichever of ``queen_keys`` signed
        it, whatever identity it names.

        The release id is the release's signed-payload digest: a re-zipped copy of a held release is the
        same release (idempotent). A file already at that id that does NOT verify (planted, or left by a
        pre-v2 store) is replaced by the verified bytes. Serialized with the id migration by a lock, so
        two concurrent publishes cannot both pass the equivocation check. The experimental tier keeps
        ZIP-sha ids -- one id shape, two meanings by tier.
        """
        if not queen_keys:
            raise OasisStoreError("publishing needs the Queen key(s) this Oasis publishes under (--queen-key)")
        raw = Path(bundle_path).read_bytes()
        try:
            read_bundle_manifest_bytes(raw)  # validates kind/schema/format-version
        except _MALFORMED_BUNDLE as exc:
            raise OasisStoreError(f"not a valid substrate bundle: {exc}") from exc
        verification = _verify(raw, queen_keys)
        if not verification.ok or verification.scheme != SIGNATURE_SCHEME_V2:
            reason = verification.reason if not verification.ok else "a legacy v1 signature"
            raise OasisStoreError(
                f"release bundles must be v2 releases signed by a registered Queen key: {reason} "
                "(unsigned bundles may only enter the experimental tier)"
            )
        release_id = str(verification.payload_digest)
        path = self.releases_dir / f"{release_id}.zip"
        with self._release_lock():
            clash = find_equivocation(
                self._held_release_records(queen_keys),
                signer_key=str(verification.signer_key),
                release_sequence=int(verification.release_sequence or 0),
                payload_digest=release_id,
            )
            if clash is not None:
                raise OasisStoreError(
                    f"signer key {str(verification.signer_key)[:12]}… already published sequence "
                    f"{verification.release_sequence} as release {str(clash.get('payload_digest'))[:12]}… -- "
                    "one sequence binds to one release (equivocation)"
                )
            if path.is_file():
                held = path.read_bytes()
                if _verifies_as(held, release_id, queen_keys):
                    logger.info("oasis: release %s is already published (same signed payload)", release_id[:12])
                    return release_id
                logger.warning(
                    "oasis: the file held as release %s does not verify; replacing it with the verified release",
                    release_id[:12],
                )
            atomic_write_bytes(str(path), raw)
        logger.info(
            "oasis: published release %s (signer=%s, sequence %s, %s)",
            release_id[:12],
            verification.signer_identity,
            verification.release_sequence,
            verification.license,
        )
        return release_id

    def _release_lock(self) -> Any:
        from filelock import FileLock

        self.releases_dir.mkdir(parents=True, exist_ok=True)
        return FileLock(str(self.releases_dir / ".release.lock"), timeout=30)

    def _held_release_records(self, queen_keys: Mapping[str, str]) -> list[dict[str, Any]]:
        """``{signer_key, release_sequence, payload_digest}`` of each held release that VERIFIES as v2 under
        one of ``queen_keys``' KEYS -- tried against every key, under the identity the release itself names,
        so relabelling a key (``--queen-key other-name=SAME_KEY``) cannot hide its history. A held file
        that verifies under none binds no sequence. O(held releases x keys) verifications per publish."""
        records: list[dict[str, Any]] = []
        if not self.releases_dir.is_dir():
            return records
        keys = sorted(set(queen_keys.values()))
        for path in sorted(self.releases_dir.glob("*.zip")):
            try:
                raw = path.read_bytes()
            except OSError as exc:
                logger.warning("oasis: cannot read held release %s: %s", path.name, exc)
                continue
            held = _verify_by_key(raw, keys)
            if held is not None:
                records.append(
                    {
                        "signer_key": held.signer_key,
                        "release_sequence": held.release_sequence,
                        "payload_digest": held.payload_digest,
                    }
                )
        return records

    def release_migration_status(self) -> tuple[int, list[tuple[str, str]]]:
        """Read-only: ``(renames pending, identity collisions)`` in the release tier. A collision is two
        DIFFERENT files sharing one payload identity (e.g. a genuine release and a copy with a forged
        signature member) -- :meth:`migrate_release_ids` leaves both in place, so it stays reported here
        until an operator removes the one that does not verify."""
        pending = 0
        collisions: list[tuple[str, str]] = []
        if not self.releases_dir.is_dir():
            return pending, collisions
        for path in sorted(self.releases_dir.glob("*.zip")):
            if not _RELEASE_ID_RE.match(path.stem):
                continue
            try:
                raw = path.read_bytes()
                with zipfile.ZipFile(path) as zf:
                    identity = content_payload_digest(zf)
                if identity is None or identity == path.stem:
                    continue
                target = self.releases_dir / f"{identity}.zip"
                if target.is_file() and target.read_bytes() != raw:
                    collisions.append((path.name, target.name))
                    continue
            except (zipfile.BadZipFile, OSError):
                continue
            pending += 1
        return pending, collisions

    def migrate_release_ids(self) -> int:
        """Rename releases stored under their ZIP sha256 to their payload identity; return how many moved.

        Release format v2 made a Queen release's id its signed-payload digest. A release published before
        that sits under its ZIP sha; ``content_payload_digest`` computes its payload identity with no key
        (equal to the verified digest whenever it verifies), so the rename needs no trust decision -- and
        it makes NONE: when the target id already holds DIFFERENT bytes (two files, one identity -- e.g.
        a genuine release and a copy with a forged signature member), both are left in place and the
        collision is warned about; only a byte-identical duplicate is removed. A file whose identity
        cannot be computed keeps its name. Idempotent; the oasis verbs run it once each, under the same
        lock as publishing. Written through ``atomic_write_bytes`` then the old name removed -- never a
        rename.
        """
        moved = 0
        if not self.releases_dir.is_dir():
            return moved
        with self._release_lock():
            for path in sorted(self.releases_dir.glob("*.zip")):
                if not _RELEASE_ID_RE.match(path.stem):
                    continue
                try:
                    raw = path.read_bytes()
                    with zipfile.ZipFile(path) as zf:
                        identity = content_payload_digest(zf)
                except (zipfile.BadZipFile, OSError) as exc:
                    logger.warning("oasis: cannot migrate release %s: %s", path.name, exc)
                    continue
                if identity is None:
                    logger.warning("oasis: release %s has no computable payload identity; left as is", path.name)
                    continue
                if identity == path.stem:
                    continue
                target = self.releases_dir / f"{identity}.zip"
                try:
                    if target.is_file():
                        if target.read_bytes() != raw:
                            logger.warning(
                                "oasis: releases %s and %s share payload identity %s but differ; both left in "
                                "place (check which verifies)",
                                path.name,
                                target.name,
                                identity[:12],
                            )
                            continue
                    else:
                        atomic_write_bytes(str(target), raw)
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning("oasis: cannot migrate release %s: %s", path.name, exc)
                    continue
                moved += 1
                logger.info("oasis: migrated release id %s -> %s (payload identity)", path.stem[:12], identity[:12])
        return moved

    def list_releases(self) -> list[dict[str, Any]]:
        """Summaries (never payloads) of every published release, newest first.

        Derived by scanning ``releases/*.zip`` and reading each manifest — no
        separate index file to drift out of sync with the directory. A file
        whose manifest no longer parses is skipped with a warning rather than
        failing the whole listing. This is O(n) manifest parses per call;
        acceptable at 1.2 scale (a Queen publishes few releases), and a cached
        summary index is the growth path if a large release count ever bites.
        """
        out: list[dict[str, Any]] = []
        if not self.releases_dir.is_dir():
            return out
        for path in sorted(self.releases_dir.glob("*.zip")):
            release_id = path.stem
            if not _RELEASE_ID_RE.match(release_id):
                continue
            try:
                manifest = read_bundle_manifest(path)
                stored_schema = stored_schema_version(path)
            except _MALFORMED_BUNDLE as exc:
                logger.warning("oasis: skipping unreadable release %s: %s", release_id, exc)
                continue
            summary = {"id": release_id, **{k: manifest.get(k) for k in _SUMMARY_KEYS}}
            # The schema the bundle carries ON THE WIRE (2 unsigned, 3 release), not the migrated view.
            summary["schema_version"] = stored_schema
            # Ordering / display only, never trust: the scheme a bundle CLAIMS and its algorithm (a v2
            # release keeps both in its detached signature member, a v1 bundle in the manifest).
            scheme = bundle_signature_scheme(path)
            summary["signature_scheme"] = scheme
            summary["signature_algorithm"] = (
                SIGNATURE_ALGORITHM if scheme == SIGNATURE_SCHEME_V2 else manifest.get("signature_algorithm")
            )
            out.append(summary)
        out.sort(key=lambda s: s.get("created_at") or "", reverse=True)
        return out

    def open_release(self, release_id: str) -> bytes | None:
        """Return the raw bundle bytes for ``release_id``, or ``None`` if absent.

        Reads the whole bundle into memory (bounded: bundles are small, and the
        proxy's concurrency semaphore caps how many load at once). True server-
        side streaming is a deliberate later optimization, not needed at 1.2
        bundle sizes. ``open_*``/``list_*`` are named for a content-addressed
        byte-blob store returning raw bytes — deliberately NOT the MemoryLayer
        §4b ``get``/``recall`` vocabulary, which is for hydrated memory records.
        """
        _validate_release_id(release_id)
        path = self.releases_dir / f"{release_id}.zip"
        if not path.is_file():
            return None
        return path.read_bytes()

    # ── experimental tier (received contributions) ───────────────────────

    def accept_contribution(self, raw: bytes, *, source: str) -> dict[str, Any]:
        """Land a foreign bundle in the experimental tier tagged with provenance.

        Validates only that ``raw`` is a well-formed bundle manifest (a real
        contribution, not garbage) and records it. It does NOT merge, ingest,
        or promote — the V1–V10 receiver gauntlet runs at promotion (Slice D),
        never on receipt. Idempotent by digest: a re-sent contribution returns
        ``status="duplicate"`` without rewriting.

        ``source`` is the transport-level origin (peer IP) recorded for the
        slow-poison audit trail; the manifest's ``contributor_id`` is the
        self-declared author. Both are retained — they are different facts.
        """
        digest = hashlib.sha256(raw).hexdigest()

        # Validate from the in-memory bytes BEFORE anything touches disk — a
        # malformed contribution never leaves a blob behind, and there is no
        # window in which an un-validated blob is visible to open_contribution.
        try:
            manifest = read_bundle_manifest_bytes(raw)
        except _MALFORMED_BUNDLE as exc:
            raise OasisStoreError(f"contribution is not a valid substrate bundle: {exc}") from exc
        # The /contribute door is where strangers write: identities follow the public grammar and every
        # string that lands in the provenance log is bounded and printable (no newline / escape / 5 KB id
        # in a listing or a log). Lenience is for READING old bundles, not for admitting new ones.
        _check_contribution_strings(manifest)

        record = {
            "digest": digest,
            "contributor_id": manifest.get("contributor_id"),
            "domain": manifest.get("domain"),
            "body_ref": manifest.get("body_ref"),
            # CLAIMED, not verified (the store never checks a contribution's signature): a v2 release keeps
            # its algorithm in signature.json, a v1 bundle in the manifest.
            "signature_algorithm": (
                _claimed_algorithm(raw) if _has_signature_member(raw) else manifest.get("signature_algorithm")
            ),
            "signer_identity": manifest.get("signer_identity"),
            "source": source,
            "received_at": time.time(),
            "size_bytes": len(raw),
        }

        # Serialize the dup-check → load-log → write-blob → write-log. Without
        # the lock, concurrent contributions race the read-modify-write on the
        # log and one audit record is silently dropped (both review lenses,
        # cross-confirmed). The log is loaded FIRST inside the lock so a corrupt
        # log fails loud BEFORE any blob is written (no orphan blob that a later
        # retry's dup-check would leave permanently unrecorded).
        blob_path = self.experimental_dir / f"{digest}.zip"
        with self._accept_lock:
            if blob_path.is_file():
                return {"digest": digest, "tier": "experimental", "status": "duplicate"}
            records = self._load_contribution_log()
            atomic_write_bytes(str(blob_path), raw)
            records.append(record)
            atomic_write_json(str(self._contrib_log), with_format_version({"records": records}))
        logger.info("oasis: accepted experimental contribution %s from %s", digest[:12], source)
        return {"digest": digest, "tier": "experimental", "status": "accepted"}

    def list_contributions(self) -> list[dict[str, Any]]:
        """Every experimental contribution's provenance record (the audit surface)."""
        return list(self._load_contribution_log())

    def open_contribution(self, digest: str) -> bytes | None:
        """Return the raw bytes of an experimental contribution (Slice D promotion input)."""
        _validate_release_id(digest)  # same sha256-hex shape
        path = self.experimental_dir / f"{digest}.zip"
        if not path.is_file():
            return None
        return path.read_bytes()

    # ── provenance log persistence ───────────────────────────────────────

    def _load_contribution_log(self) -> list[dict[str, Any]]:
        if not self._contrib_log.is_file():
            return []
        # A PRESENT-but-unreadable log is NOT treated as empty: returning []
        # here would let the next append overwrite the whole audit trail with a
        # one-record file. Fail loud instead — a corrupt provenance log is an
        # operator problem, not something to silently destroy.
        try:
            data = json.loads(self._contrib_log.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise OasisStoreError(
                f"contribution log {self._contrib_log} is present but unreadable ({exc}); "
                "refusing to overwrite it — inspect/repair the file by hand"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("records", []), list):
            raise OasisStoreError(
                f"contribution log {self._contrib_log} is malformed (expected an object with a 'records' list); "
                "refusing to overwrite it — inspect/repair the file by hand"
            )
        check_format_version(data, _CONTRIB_FILE_TYPE, log=logger)
        return [r for r in data.get("records", []) if isinstance(r, dict)]
