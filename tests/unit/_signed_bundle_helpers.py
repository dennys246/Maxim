"""Test helpers for signed bundles: v2 releases, LEGACY v1 bundles, and member rewrites.

``compose_bundle`` writes only schema-3 bundles, signed as v2 releases. A legacy v1 bundle -- schema 2,
the signature in the manifest over the manifest minus its signature fields plus the raw slices -- is
the shape every signed bundle had before release format v2 (the Exp 56/61 era), and the verifier must
keep accepting it. ``write_v1_bundle`` builds exactly that shape from an unsigned compose.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

LICENSE = "CDLA-Permissive-2.0"


def release(signer: Any, *, sequence: int = 1, license: str = LICENSE) -> Any:
    from maxim.hivemind.signing import SignedRelease

    return SignedRelease(signer=signer, release_sequence=sequence, license=license)


def read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def write_members(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def rewrite_member(path: Path, name: str, data: bytes, *, out: Path | None = None) -> Path:
    members = read_members(path)
    members[name] = data
    return write_members(out or path, members)


def rewrite_manifest(path: Path, edit: Any, *, out: Path | None = None) -> Path:
    members = read_members(path)
    manifest = json.loads(members["manifest.json"])
    edit(manifest)
    members["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    return write_members(out or path, members)


def resign_v2(members: dict[str, bytes], signer: Any) -> dict[str, bytes]:
    """Re-sign edited members as a v2 release -- so a test reaches the check it targets instead of
    being refused earlier because the signature no longer verifies."""
    from maxim.hivemind.signing import SIGNATURE_MEMBER, bundle_signing_payload_v2

    sig = json.loads(members[SIGNATURE_MEMBER])
    sig["signature"] = signer.sign_payload(bundle_signing_payload_v2(members))
    return {**members, SIGNATURE_MEMBER: json.dumps(sig).encode()}


def resign_manifest_v2(path: Path, signer: Any, edit: Any, *, out: Path) -> Path:
    members = read_members(path)
    manifest = json.loads(members["manifest.json"])
    edit(manifest)
    members["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    return write_members(out, resign_v2(members, signer))


def write_v1_bundle(unsigned_path: Path, signer: Any, *, out: Path, schema_version: int = 2) -> Path:
    """Re-sign an UNSIGNED compose output as a legacy v1 bundle (schema 2, manifest signature).
    ``schema_version`` other than 2 builds the DOWNGRADE shape -- a v1 signature validly covering a
    schema-3 manifest."""
    from maxim.hivemind.signing import SIGNATURE_ALGORITHM, bundle_signing_payload

    members = read_members(unsigned_path)
    manifest = json.loads(members["manifest.json"])
    manifest["schema_version"] = schema_version
    manifest.pop("license", None)
    manifest["signature"] = manifest["signature_algorithm"] = manifest["signer_identity"] = None
    slices = {n: d.decode() for n, d in members.items() if n != "manifest.json"}
    manifest["signature"] = signer.sign_payload(bundle_signing_payload(manifest, slices))
    manifest["signature_algorithm"] = SIGNATURE_ALGORITHM
    manifest["signer_identity"] = signer.signer_identity
    members["manifest.json"] = json.dumps(manifest, indent=2, sort_keys=True).encode()
    return write_members(out, members)
