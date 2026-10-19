"""Synthetic pulse Console bundles and the pins that match them (R4, #1333).

Shared by the vendor-step and release-audit tests. A bundle is a dict {relpath: bytes}; it can be written as a
directory, packed as pulse packs its release asset (members prefixed `./`, directory entries included), or laid into
a wheel/sdist under their prefixes. `pin_for` derives the pin the real `--update-pin` would write for it.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

STAMP = {
    "target": "console",
    "app_version": "0.4.1",
    "contract_version": None,  # filled with the backend contract
    "commit": "da20a72",
    "commit_date": "2026-10-15T00:25:33Z",
    "describe": "v0.4.1",
    "dirty": False,
}
COMMIT = "da20a721f569c109e2d649531d5d8c73671ae90a"
OLD_COMMIT = "b39ca6300000000000000000000000000000000a"


def bundle(cp, *, drop: tuple[str, ...] = (), **stamp_overrides) -> dict[str, bytes]:
    stamp = {**STAMP, "contract_version": cp.backend_contract(), **stamp_overrides}
    for key in drop:
        stamp.pop(key, None)
    return {
        "index.html": b"<html></html>",
        "assets/index-abc.js": b"//",
        "assets/index-abc.css": b"*{}",
        cp.MANIFEST_NAME: json.dumps(stamp).encode(),
    }


def write_dir(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def tar_bytes(files: dict[str, bytes], *, extra: list[tarfile.TarInfo] | None = None, mtime: int = 0) -> bytes:
    """Deterministic: the gzip header's mtime is fixed, so two calls give one sha256 (`mtime` varies the bytes
    without varying the files)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=mtime) as gz, tarfile.open(fileobj=gz, mode="w") as tf:
        for d in ("./", "./assets/"):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            tf.addfile(info)
        for rel, data in sorted(files.items()):
            info = tarfile.TarInfo(f"./{rel}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for info in extra or []:
            tf.addfile(info, io.BytesIO(b"x" * info.size) if info.isfile() else None)
    return buf.getvalue()


def pin_for(cp, files: dict[str, bytes], **overrides):
    data = tar_bytes(files)
    fields = {
        "repo": cp.PULSE_REPO,
        "tag": "v0.4.1",
        "commit": COMMIT,
        "contract_version": cp.backend_contract(),
        "asset": cp.PULSE_ASSET,
        "asset_sha256": hashlib.sha256(data).hexdigest(),
        "tree_sha256": cp.tree_digest(list(files.items())),
    }
    fields.update(overrides)
    return cp.Pin(**fields)
