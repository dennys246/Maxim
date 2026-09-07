"""Oasis substrate-bundle store — the two-tier on-disk home for shared bundles.

1.2 P2P Slice B. An :class:`OasisStore` is the durable side of the substrate
exchange surface an Oasis serves (the HTTP endpoints live in
:mod:`maxim.hivemind.oasis_endpoints`; the client that reaches them lives in
:mod:`maxim.hivemind.substrate_client`). Two tiers, mirroring the trust model
of ``sharing_threat_model.md`` §5:

- **releases/** — Queen-tier published bundles. Signed by definition:
  :meth:`publish_release` REFUSES an unsigned manifest, so the "Queen releases
  are signed" decision (P2P scope decision 2) is enforced at the store door,
  not merely by convention downstream.
- **experimental/** — received contributions. :meth:`accept_contribution`
  lands a foreign bundle here tagged with provenance and NOTHING more — it is
  never merged, never promoted to the release tier as a side effect of receipt.
  Promotion (running the V1–V10 ``ingest_bundle`` gauntlet + re-signing) is a
  separate, gated Slice D operation. This is the engineering invariant the slice
  establishes: *a contribution's arrival changes no trusted state.*

Bundles are content-addressed by the sha256 of their raw ZIP bytes — the same
digest ``ingest_bundle`` computes — so the id is collision-free and the on-disk
path is never built from unvalidated caller input (traversal-proof by
construction; :func:`_validate_release_id` re-checks the shape anyway).

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
from pathlib import Path
from typing import Any

from maxim.hivemind.bundle import read_bundle_manifest, read_bundle_manifest_bytes
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
    "signature_algorithm",
    "signer_identity",
    "affordance_namespace",
)


class OasisStoreError(Exception):
    """A store operation was refused (unsigned release, malformed bundle)."""


def _validate_release_id(release_id: str) -> str:
    """Return ``release_id`` iff it is a bare sha256 hex digest, else raise.

    The id becomes a filename; anything but ``[0-9a-f]{64}`` (no separators,
    no ``..``) is refused before it can reach the filesystem.
    """
    if not _RELEASE_ID_RE.match(release_id):
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

    def publish_release(self, bundle_path: str | Path) -> str:
        """Publish a SIGNED bundle into the release tier; return its release id.

        Refuses (``OasisStoreError``) a bundle whose manifest carries no
        ed25519 signature — a release is Queen-tier by definition, and decision
        2 of the P2P scope requires promoted-domain releases to be signed.
        Verifying the signature against a trusted key is the consumer's job at
        pull time (``ingest_bundle(require_signed=True)``); the store only
        enforces that an unsigned artifact never occupies the release tier.
        """
        raw = Path(bundle_path).read_bytes()
        manifest = read_bundle_manifest(bundle_path)  # validates kind/schema/format-version
        if not manifest.get("signature") or not manifest.get("signature_algorithm"):
            raise OasisStoreError(
                "release bundles must be signed (compose with --sign); "
                "unsigned bundles may only enter the experimental tier"
            )
        digest = hashlib.sha256(raw).hexdigest()
        atomic_write_bytes(str(self.releases_dir / f"{digest}.zip"), raw)
        logger.info("oasis: published release %s (signer=%s)", digest[:12], manifest.get("signer_identity"))
        return digest

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
            except _MALFORMED_BUNDLE as exc:
                logger.warning("oasis: skipping unreadable release %s: %s", release_id, exc)
                continue
            summary = {"id": release_id, **{k: manifest.get(k) for k in _SUMMARY_KEYS}}
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

        record = {
            "digest": digest,
            "contributor_id": manifest.get("contributor_id"),
            "domain": manifest.get("domain"),
            "body_ref": manifest.get("body_ref"),
            "signature_algorithm": manifest.get("signature_algorithm"),
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
