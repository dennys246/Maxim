"""Substrate snapshot bundle format (Hivemind shareability, PR D).

v1_refinement.md §B5 PR D. Composes a portable, versioned, optionally
signed archive of one Maxim's NAc + EC substrate, suitable for exchange
between Oases and substrate-primary Maxims. The 1.1 Oasis software will
build on this format; the 1.2 P2P Hivemind protocol will exchange these
bundles between peers.

The format is a ZIP containing JSON files at the bundle root:

    maxim-substrate.zip
    ├── manifest.json   # _format_version, contributor_id, domain, signature
    ├── nac.json        # NAc.dump() output (optionally identity-filtered)
    └── ec.json         # EC substrate_nodes slice (optionally domain-filtered)

Hippocampus episodes are NEVER included by construction — the
"hippocampus-episodes-stay-local" rule from ``maxim_hivemind.md`` is the
load-bearing privacy invariant. ATL, reflexes, and cerebellum payloads
are reserved for 1.1 (Phase B5 spec mentions them; this 1.0 ship
includes only NAc + EC because that's what PRs A/B/C give us merge
math for).

Manifest signature slot
-----------------------

The ``signature``, ``signature_algorithm``, and ``signer_identity``
fields are reserved (per the 2026-05-30 design decision: "Reserve
signature field in manifest, no verification yet" + the CC13 auth
format-freeze, which added ``signer_identity``). At 1.0 they are always
``None`` — the slots exist so 1.1+ verification can land WITHOUT bumping
the bundle's ``_format_version`` and breaking 1.0 bundles. Callers that
want signing build their own ZIP with a populated ``signature`` field
and a custom verifier; this module does NOT validate.

The recognized ``signature_algorithm`` vocabulary (``ed25519``,
``ed25519-pgp``, ``webauthn``, ``pkcs7``, reserved ``hsm:*`` / ``kms:*``
/ ``vendor:*`` prefixes, ...) is published in
``docs/user/hivemind_bundle_format.md`` so the 1.2 P2P protocol's
heterogeneous producers and consumers share a string vocabulary. The
registry is documentation-only at 1.0 (no validator), consistent with
the no-verification-yet decision. ``signer_identity`` is the reserved
"who claims to have signed this" string, parallel to ``contributor_id``,
so 1.1+ can bind a verified identity to the claimed contributor without
retrofitting the manifest shape.

Format version contract
-----------------------

The ``manifest.json`` root carries ``_format_version`` per the CC1 1.0
freeze invariant. Older bundles will fail to load via
:func:`extract_bundle` — there is no 0.x bundle in the wild because
this is a 1.0 feature.

Identity filter
---------------

When ``apply_identity_filter=True`` (the default), :func:`compose_bundle`
routes the NAc links through
:func:`maxim.hivemind.identity.filter_identity_bearing_links` and drops
EC nodes whose ``domain`` is the
:data:`maxim.hivemind.identity.IDENTITY_DOMAIN_MARKER`. The identity
threshold defaults to ``2`` per the PR C review fold (game-substrate
contexts over-flag at threshold=1 because every "Dragon"/"Goblin" trips
the proper-noun signal).
"""

from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import json
import logging
import os
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from maxim.hivemind.identity import (
    IDENTITY_DOMAIN_MARKER,
    filter_identity_bearing_links,
)
from maxim.hivemind.merge import _validate_source
from maxim.utils.atomic_io import atomic_write_text
from maxim.utils.format_version import FORMAT_VERSION, check_format_version

logger = logging.getLogger(__name__)

# Schema version for the bundle envelope itself. Separate from the
# bio-system payload ``_format_version`` — bumping this would require
# a migration registered alongside the bump.
BUNDLE_SCHEMA_VERSION: int = 1

# Bundle-level kind marker for the manifest.
BUNDLE_KIND: str = "substrate_bundle"

# Default identity-filter threshold for the bundle composer. PR C
# Architecture-lens review flagged game-substrate over-flagging at
# threshold=1; bundles default to the stricter 2 so generic creature
# tokens like "Dragon" ride along.
_DEFAULT_BUNDLE_IDENTITY_THRESHOLD: int = 2


# ─────────────────────────────────────────────────────────────────────────
# Migration registry (Architecture review IMPORTANT fold)
#
# Pre-emptive seam matching the ``memory/snapshot.py`` envelope-migration
# pattern. Empty at 1.0 — bumping ``BUNDLE_SCHEMA_VERSION`` to 2 in 1.1
# registers a single ``v1 → v2`` function via the decorator below and
# the extract path upgrades 1.0-shaped bundles transparently. Reserving
# the seam now avoids a painful retrofit when 1.1 first needs it.
# ─────────────────────────────────────────────────────────────────────────


MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

_BUNDLE_MIGRATIONS: dict[int, MigrationFn] = {}


def register_bundle_migration(from_version: int) -> Callable[[MigrationFn], MigrationFn]:
    """Decorator: register a bundle-manifest migration ``from_version → from_version+1``.

    Bumping :data:`BUNDLE_SCHEMA_VERSION` requires registering a
    matching migration so older bundles upgrade transparently on
    extract. The registry follows the same shape as the
    ``memory/snapshot.py`` envelope-migration registry.
    """

    def _decorator(fn: MigrationFn) -> MigrationFn:
        if from_version in _BUNDLE_MIGRATIONS:
            raise ValueError(f"bundle migration from version {from_version} already registered")
        _BUNDLE_MIGRATIONS[from_version] = fn
        return fn

    return _decorator


def migrate_bundle_envelope(manifest: dict[str, Any], *, target_version: int | None = None) -> dict[str, Any]:
    """Upgrade a bundle manifest through the migration chain.

    At v1 with an empty registry this is a no-op deep-copy (callers may
    mutate the returned dict without aliasing the input). Bumping
    ``BUNDLE_SCHEMA_VERSION`` to 2 + registering a ``v1`` migration is
    the only change 1.1 needs to make for older bundles to load.
    """
    if target_version is None:
        target_version = BUNDLE_SCHEMA_VERSION
    if not isinstance(manifest, dict):
        raise ValueError(f"bundle manifest must be dict, got {type(manifest).__name__}")
    version = manifest.get("schema_version")
    if not isinstance(version, int):
        # Defer the validation error to extract_bundle's existing branch;
        # passing through unchanged keeps a single point of error reporting.
        return copy.deepcopy(manifest)
    if version > target_version:
        return copy.deepcopy(manifest)

    current = copy.deepcopy(manifest)
    while True:
        cur_version = current["schema_version"]
        if cur_version == target_version:
            return current
        migration = _BUNDLE_MIGRATIONS.get(cur_version)
        if migration is None:
            raise ValueError(
                f"bundle envelope schema_version {cur_version} has no migration to {cur_version + 1} "
                f"(registry keys: {sorted(_BUNDLE_MIGRATIONS.keys())})"
            )
        current = migration(current)
        if not isinstance(current, dict) or current.get("schema_version") != cur_version + 1:
            raise ValueError(f"bundle migration {cur_version}→{cur_version + 1} returned an invalid envelope")


@contextlib.contextmanager
def isolated_bundle_migrations() -> Iterator[None]:
    """Context manager: snapshot + clear + restore the bundle-migration registry.

    Test-only helper matching ``memory/snapshot.py::isolated_migrations``.
    Tests that need to register synthetic migrations wrap setup in this
    context so they start from a clean registry and the pre-test
    registry is restored on exit.
    """
    saved = dict(_BUNDLE_MIGRATIONS)
    _BUNDLE_MIGRATIONS.clear()
    try:
        yield
    finally:
        _BUNDLE_MIGRATIONS.clear()
        _BUNDLE_MIGRATIONS.update(saved)


def _utc_now_iso() -> str:
    """ISO 8601 timestamp in UTC with second-level resolution."""
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _filter_ec_nodes_by_domain(
    nodes: dict[str, dict[str, Any]],
    *,
    domain: str | None,
) -> dict[str, dict[str, Any]]:
    """Filter EC substrate_nodes by domain.

    Drops nodes with ``domain == IDENTITY_DOMAIN_MARKER`` unconditionally.
    When ``domain`` is non-None, additionally drops nodes whose ``domain``
    field doesn't match (passes through undomained ``None`` nodes — they
    are generic).
    """
    out: dict[str, dict[str, Any]] = {}
    for nid, nd in nodes.items():
        nd_domain = nd.get("domain")
        # Always drop reserved-identity nodes.
        if nd_domain == IDENTITY_DOMAIN_MARKER:
            continue
        # Per-domain filter: when caller scopes to a specific domain,
        # admit only that domain plus undomained generic nodes.
        if domain is not None and nd_domain is not None and nd_domain != domain:
            continue
        out[nid] = nd
    return out


def compose_bundle(
    *,
    nac_state: dict[str, Any] | None,
    ec_substrate_nodes: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    contributor_id: str,
    domain: str | None = None,
    apply_identity_filter: bool = True,
    identity_threshold: int = _DEFAULT_BUNDLE_IDENTITY_THRESHOLD,
    signature: str | None = None,
    signature_algorithm: str | None = None,
    signer_identity: str | None = None,
    encoder_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a substrate snapshot bundle.

    Pure function: inputs are not mutated. Writes a zip file at
    ``output_path``. Returns the manifest dict.

    Parameters
    ----------
    nac_state
        Output of ``NAc.dump()`` — the shareable NAc state. ``None``
        skips the ``nac.json`` slice.
    ec_substrate_nodes
        The ``substrate_nodes`` slice from ``EC.save()``'s payload
        (``json.loads(ec.json)["substrate_nodes"]``). ``None`` skips
        the ``ec.json`` slice.
    output_path
        Where to write the zip file. Parent directory is created if
        absent.
    contributor_id
        Opaque ID identifying which Maxim composed the bundle. Goes
        into the manifest. Per PR B's reserved-namespace rule, this
        MUST NOT start with the reserved ``_`` prefix.
    domain
        Optional substrate-domain scope (``"combat"``, ``"cooking"``,
        ...). When set, EC nodes whose ``domain`` field disagrees are
        dropped (undomained nodes still ride along).
    apply_identity_filter
        When True (default), drops identity-bearing NAc event
        signatures + reserved-identity-domain EC nodes from the
        bundle. False disables the filter for trusted-internal
        backups.
    identity_threshold
        Threshold passed to
        :func:`maxim.hivemind.identity.is_identity_bearing` when
        filtering. Default 2 — bundle-stricter than the heuristic's
        default of 1, per PR C's game-substrate fold.
    signature, signature_algorithm
        Reserved slots — populate at the caller's discretion. This
        module does NOT compute signatures and does NOT validate
        them at extract time. Default ``None`` / ``None``. Recognized
        ``signature_algorithm`` values are published in
        ``docs/user/hivemind_bundle_format.md`` (the 1.2 P2P verifier
        vocabulary); the registry is documentation-only at 1.0.
    signer_identity
        Reserved slot (CC13 auth format-freeze) — the "who claims to
        have signed this" string, parallel to ``contributor_id``.
        Always ``None`` at 1.0; reserved so 1.1+ bundle verification
        can bind a verified identity to ``contributor_id`` without
        retrofitting the manifest shape. NOT validated here.
    encoder_provenance
        Encode-time encoder stamps from the source EC payload
        (``ec.json``'s ``encoder_provenance`` key — recorded by the
        encoders via ``EC.record_encoder_provenance``, never authored
        post-hoc). Carried into ``manifest["encoder_provenance"]
        ["recorded"]`` verbatim; ``None`` (pre-stamping payloads) is
        carried as ``None`` — an honest "unknown", not a fabricated
        default. Independent of this parameter, the manifest ALWAYS
        carries ``observed_embedding_dims`` derived from the ACTUAL
        arrays in the EC slice at write time (checked truth, per the
        fabric plan's "stamp the realized state, not its name" rule).

        MERGE SEMANTICS (pinned for 1.2 — do not build on the naive
        reading): ``recorded`` describes the COMPOSING substrate's own
        encoders only. A substrate that previously imported foreign
        nodes via ``ec_merge`` ships arrays encoded elsewhere that its
        local stamps do not describe — the 1.2 P2P merge must union
        provenance per-contributor rather than trusting a merged
        substrate's local stamps. ``observed_embedding_dims`` is the
        measured backstop either way, but dims alone cannot distinguish
        a 384-dim fallback from a real 384-dim model.
    """
    # Fold (Executor IMPORTANT): route through the same validator
    # PR B's merge functions use, instead of duplicating the
    # reserved-prefix check inline. This catches the divergence the
    # reviewer flagged: PR B's _validate_source also rejects empty
    # strings and non-string types, which the inline check missed.
    _validate_source(contributor_id, label="contributor_id")

    # Snapshot the input pieces (NAc state + filtered EC nodes).
    bundle_contents: dict[str, str] = {}  # filename -> serialized JSON

    if nac_state is not None:
        filtered_nac = dict(nac_state)
        if apply_identity_filter:
            filtered_links = filter_identity_bearing_links(
                filtered_nac.get("links", {}) or {},
                threshold=identity_threshold,
            )
            filtered_nac["links"] = filtered_links
        bundle_contents["nac.json"] = json.dumps(filtered_nac, indent=2, sort_keys=True, default=str)

    observed_embedding_dims: dict[str, list[int]] = {}
    if ec_substrate_nodes is not None:
        ec_nodes_filtered = _filter_ec_nodes_by_domain(ec_substrate_nodes, domain=domain)
        bundle_contents["ec.json"] = json.dumps(
            {"substrate_nodes": ec_nodes_filtered},
            indent=2,
            sort_keys=True,
            default=str,
        )
        # Artifact stamping (1.1 item 7): dims measured on the ACTUAL
        # arrays being shipped — a per-modality dim SET, so a mixed-space
        # slice (the #467 corruption class) is visible in the manifest
        # rather than discovered at merge time.
        dims_by_modality: dict[str, set[int]] = {}
        for node in ec_nodes_filtered.values():
            modality = str(node.get("modality") or "unknown")
            emb = node.get("embedding") or []
            dims_by_modality.setdefault(modality, set()).add(len(emb))
        observed_embedding_dims = {m: sorted(d) for m, d in sorted(dims_by_modality.items())}

    manifest: dict[str, Any] = {
        "_format_version": FORMAT_VERSION,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "contributor_id": contributor_id,
        "domain": domain,
        "created_at": _utc_now_iso(),
        "identity_filter_applied": bool(apply_identity_filter),
        "identity_threshold": int(identity_threshold) if apply_identity_filter else None,
        "contents": {
            slice_name.removesuffix(".json"): {"file": slice_name} for slice_name in sorted(bundle_contents.keys())
        },
        "encoder_provenance": {
            "observed_embedding_dims": observed_embedding_dims,
            "recorded": encoder_provenance,
        },
        "signature": signature,
        "signature_algorithm": signature_algorithm,
        "signer_identity": signer_identity,
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True, default=str)

    # Atomic write via tmp + os.replace. Zip writing is single-shot;
    # if any step raises we tear down the tmp file.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", manifest_json)
            for filename, content in bundle_contents.items():
                zf.writestr(filename, content)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as cleanup_err:
                logger.warning("Failed to clean up %s: %s", tmp_path, cleanup_err)
        raise

    logger.info("Composed substrate bundle at %s (%d slices)", output_path, len(bundle_contents))
    return manifest


def _safe_join(output_dir: Path, name: str) -> Path:
    """Resolve ``output_dir / name`` and reject ZIP-slip escape attempts.

    Pre-merge review CRITICAL (Executor lens): a malicious bundle entry
    named ``../../../etc/passwd`` (or an absolute path, or any name
    containing ``..``) resolves outside ``output_dir`` and would let a
    crafted bundle clobber arbitrary files on extract. The 1.2 P2P
    protocol will exchange bundles between peers, so this is a real
    threat surface even before the import verb is widely used.

    Rejects: empty names, absolute paths, and any resolved target that
    falls outside ``output_dir`` (catches ``..`` traversal + symlink
    escape). Returns the safe resolved Path on success; raises
    ``ValueError`` otherwise.
    """
    if not name:
        raise ValueError("bundle contains an empty path entry; refusing to extract")
    name_path = Path(name)
    if name_path.is_absolute():
        raise ValueError(f"bundle entry {name!r} is an absolute path; refusing to extract")
    candidate = (output_dir / name_path).resolve()
    base = output_dir.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"bundle entry {name!r} resolves outside output_dir; refusing to extract (ZIP slip)") from exc
    return candidate


def extract_bundle(
    bundle_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Extract a substrate bundle to a directory.

    Validates the manifest's ``kind`` + ``schema_version`` +
    ``_format_version`` (via :func:`check_format_version`). Writes
    ``manifest.json``, ``nac.json``, and ``ec.json`` (whichever are
    present) to ``output_dir``. Returns the parsed manifest dict.

    Does NOT auto-merge into a live NAc / EC — that's the caller's
    decision. The 1.1 Oasis software will wrap this in a pipeline that
    calls ``nac_merge`` / ``ec_merge`` against the extracted dicts; the
    1.0 CLI verb just round-trips the data.

    Every ZIP entry is routed through :func:`_safe_join` before being
    written — absolute paths, ``..`` traversal, and symlink escape are
    all rejected (ZIP-slip CVE class). A malicious bundle with one
    safe slice and one escape slice writes nothing — the safety check
    runs in a pre-validation pass before any disk writes.

    Raises ``ValueError`` on a manifest with wrong kind or unrecognized
    schema_version, or on any ZIP entry that fails the path-safety
    check. Raises ``zipfile.BadZipFile`` on a malformed archive.
    """
    bundle_path = Path(bundle_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(f"bundle {bundle_path} missing manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"manifest.json must be a JSON object, got {type(manifest).__name__}")

        # Bundle envelope migration (post-fold scaffolding for 1.1+):
        # routes the manifest through the migration registry before the
        # kind / schema_version validation runs. At 1.0 the registry is
        # empty so this is a no-op; 1.1's first bundle migration just
        # registers a ``v1 → v2`` function and existing 1.0 bundles
        # upgrade transparently.
        manifest = migrate_bundle_envelope(manifest)

        check_format_version(manifest, "substrate_bundle", log=logger)

        if manifest.get("kind") != BUNDLE_KIND:
            raise ValueError(f"manifest kind {manifest.get('kind')!r} != {BUNDLE_KIND!r}")
        schema_v = manifest.get("schema_version")
        if not isinstance(schema_v, int) or schema_v > BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version {schema_v!r} unsupported (this build supports up to {BUNDLE_SCHEMA_VERSION})"
            )

        # Pre-validate EVERY entry path before writing anything — a
        # bundle with one good slice and one ZIP-slip slice writes
        # nothing.
        safe_targets: list[tuple[str, Path]] = []
        for name in sorted(zf.namelist()):
            safe_targets.append((name, _safe_join(output_dir, name)))

        for name, target in safe_targets:
            if name == "manifest.json":
                # Re-serialize the (possibly migrated) manifest for the
                # extracted copy so external readers see the upgraded
                # shape instead of the on-disk legacy one.
                atomic_write_text(str(target), json.dumps(manifest, indent=2, sort_keys=True))
                continue
            content = zf.read(name).decode("utf-8")
            atomic_write_text(str(target), content)

    logger.info("Extracted substrate bundle from %s to %s", bundle_path, output_dir)
    return manifest


def read_bundle_manifest(bundle_path: str | Path) -> dict[str, Any]:
    """Read the manifest from a bundle without extracting it.

    Convenience for CLI ``maxim substrate inspect`` and 1.1 Oasis
    discovery. Validates ``kind`` + ``schema_version`` like
    :func:`extract_bundle`.
    """
    bundle_path = Path(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError(f"bundle {bundle_path} missing manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    if not isinstance(manifest, dict):
        raise ValueError(f"manifest.json must be a JSON object, got {type(manifest).__name__}")
    manifest = migrate_bundle_envelope(manifest)
    check_format_version(manifest, "substrate_bundle", log=logger)
    if manifest.get("kind") != BUNDLE_KIND:
        raise ValueError(f"manifest kind {manifest.get('kind')!r} != {BUNDLE_KIND!r}")
    schema_v = manifest.get("schema_version")
    if not isinstance(schema_v, int) or schema_v > BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {schema_v!r} unsupported (this build supports up to {BUNDLE_SCHEMA_VERSION})"
        )
    return manifest


__all__ = [
    "BUNDLE_KIND",
    "BUNDLE_SCHEMA_VERSION",
    "compose_bundle",
    "extract_bundle",
    "isolated_bundle_migrations",
    "migrate_bundle_envelope",
    "read_bundle_manifest",
    "register_bundle_migration",
]
