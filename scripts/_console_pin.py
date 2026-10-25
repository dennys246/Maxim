"""The vendored-console pin and its canonical digest: R4 (#1333), shared by the vendor step and the release audit.

pymaxim 1.3.2 vendored pulse v0.4.0 two days after v0.4.1 had been tagged: the vendor step took any directory, and
nothing read the vendored stamp at release. The fix is a committed pin (``scripts/console_ui_pin.json``) naming the
pulse tag, the commit it peels to, its contract, the release asset's sha256 and a digest over the bundle's files.
``vendor_console_ui.py`` refuses a bundle that does not match it and proves it is the newest tag for the contract
(``--verify-pin``); ``audit_release_build.py`` re-checks the built wheel and sdist against it.

Everything here is offline and pure except `releasing()`, so both scripts and their tests share one definition. The design and its
adversarial pass: docs/plans/archive/r4_console_vendor_gate.md.

Two definitions live here and nowhere else:

* **The tree digest** (fold 5): sha256 over the sorted lines ``"<sha256 of the bytes>  <relpath>\\n"`` of every
  regular file, fed straight from tar members, zip entries or a directory walk, never from an extracted copy whose
  names a case-insensitive filesystem may have merged. Directory entries are skipped and modes ignored. A member
  that is not a regular file, a name with ``..``, a leading ``/``, a backslash or a newline, two names equal up to
  case, and any dotfile are REFUSED: the ``console/ui_dist/**/*`` package-data glob drops dotfiles, so a bundle
  carrying one could never match itself once it is inside a wheel.
* **The backend contract** (fold 11): read as TEXT from a ``ui_bundle.py``, never imported, so an editable install
  of another tree cannot answer for this one; the audit reads the artifact's own copy the same way.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # once, at import: `check_nightlies` for `releasing()`
REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_PATH = REPO_ROOT / "scripts" / "console_ui_pin.json"
UI_BUNDLE_SOURCE = REPO_ROOT / "src" / "maxim" / "console" / "ui_bundle.py"
MANIFEST_NAME = "maxim-ui.json"

# Constants, never read from the pin (fold 2): a pin that named a fork could otherwise carry a self-consistent
# tag/commit/digest set, and a check that derived from `pin.repo` would happily re-derive it against the fork.
PULSE_REPO = "dennys246/Maxim-pulse"
PULSE_ASSET = "console-dist.tar.gz"
PIN_FORMAT_VERSION = "1.0"

# "Newest" (fold 4): stable semver only, ordered as integer tuples; tag dates are never consulted.
STABLE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
_CONTRACT = re.compile(r'^CONSOLE_CONTRACT_VERSION\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
_HEX = re.compile(r"^[0-9a-f]+$")


class PinError(Exception):
    """The pin, or an input it is checked against, cannot be read: exit 2, never a pass."""


class DigestError(ValueError):
    """A bundle member the canonical digest refuses (see the module docstring)."""


@dataclass(frozen=True)
class Pin:
    """The committed pin. SHAPE-FROZEN at 1.0 (CC3): read only by these scripts from one committed file whose root
    `_format_version` is checked on load; a new field is a reviewed edit of both, never a silent extra."""

    repo: str
    tag: str
    commit: str
    contract_version: str
    asset: str
    asset_sha256: str
    tree_sha256: str


def tag_key(tag: str) -> tuple[int, int, int] | None:
    m = STABLE_TAG.match(tag)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def load_pin(path: Path = PIN_PATH) -> Pin:
    """The committed pin, or PinError. Hand-checks `_format_version` on purpose: `maxim.utils.format_version` is lenient
    by contract (it must not raise on a missing stamp), a gate input must fail closed, and `release-build` does not
    install `maxim`."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PinError(f"cannot read the console pin {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("_format_version") != PIN_FORMAT_VERSION:
        raise PinError(f"{path}: `_format_version` must be {PIN_FORMAT_VERSION!r}")
    fields = ("repo", "tag", "commit", "contract_version", "asset", "asset_sha256", "tree_sha256")
    unknown = sorted(set(raw) - set(fields) - {"_format_version"})
    if unknown:
        raise PinError(f"{path}: unknown field(s) {unknown}; a new field is a reviewed edit of this loader too")
    missing = [f for f in fields if not isinstance(raw.get(f), str) or not raw.get(f)]
    if missing:
        raise PinError(f"{path}: missing or non-string field(s) {missing}")
    pin = Pin(**{f: raw[f] for f in fields})
    if tag_key(pin.tag) is None:
        raise PinError(f"{path}: tag {pin.tag!r} is not a stable vX.Y.Z tag")
    for name, value, length in (("commit", pin.commit, 40), ("asset_sha256", pin.asset_sha256, 64)):
        if len(value) != length or not _HEX.match(value):
            raise PinError(f"{path}: {name} must be {length} lowercase hex characters")
    if len(pin.tree_sha256) != 64 or not _HEX.match(pin.tree_sha256):
        raise PinError(f"{path}: tree_sha256 must be 64 lowercase hex characters")
    return pin


def pin_payload(pin: Pin) -> dict[str, str]:
    return {"_format_version": PIN_FORMAT_VERSION, **pin.__dict__}


def pin_identity_problems(pin: Pin) -> list[str]:
    """The pin may echo the repo and asset, but they are compared to the constants, never used (fold 2)."""
    problems = []
    if pin.repo != PULSE_REPO:
        problems.append(f"the pin names repo {pin.repo!r}; the console bundle comes only from {PULSE_REPO}")
    if pin.asset != PULSE_ASSET:
        problems.append(f"the pin names asset {pin.asset!r}; the console bundle is {PULSE_ASSET}")
    return problems


def contract_from_source(text: str) -> str | None:
    m = _CONTRACT.search(text)
    return m.group(1) if m else None


def backend_contract(source: Path = UI_BUNDLE_SOURCE) -> str:
    try:
        found = contract_from_source(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PinError(f"cannot read {source}: {exc}") from exc
    if not found:
        raise PinError(f"no CONSOLE_CONTRACT_VERSION assignment in {source}")
    return found


# ── the canonical digest ─────────────────────────────────────────────────────────────────────────────────────────


def normalize_member(name: str) -> str | None:
    """A member's relative path, or None for a directory entry. Raises DigestError for a refused name."""
    if "\\" in name or "\n" in name or "\r" in name:
        raise DigestError(f"refused member name {name!r} (backslash or newline)")
    rel = name[2:] if name.startswith("./") else name
    if rel in ("", ".") or rel.endswith("/"):
        return None
    if rel.startswith("/"):
        raise DigestError(f"refused absolute member {name!r}")
    parts = rel.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise DigestError(f"refused member {name!r} (empty, `.` or `..` component)")
    if any(p.startswith(".") for p in parts):
        raise DigestError(
            f"refused dotfile {name!r}: the wheel's `console/ui_dist/**/*` package-data glob drops dotfiles, "
            f"so the shipped bundle could never match the pin"
        )
    return rel


def tree_digest(entries: Iterable[tuple[str, bytes]]) -> str:
    """The digest of (relpath, bytes) pairs already normalized by `normalize_member`."""
    lines: dict[str, str] = {}
    folded: dict[str, str] = {}
    for rel, data in entries:
        if rel in lines:
            raise DigestError(f"duplicate member {rel!r}")
        clash = folded.get(rel.lower())
        if clash is not None:
            raise DigestError(f"members {clash!r} and {rel!r} differ only by case")
        folded[rel.lower()] = rel
        lines[rel] = f"{hashlib.sha256(data).hexdigest()}  {rel}\n"
    if not lines:
        raise DigestError("the bundle has no files")
    return hashlib.sha256("".join(lines[k] for k in sorted(lines)).encode("utf-8")).hexdigest()


def tar_entries(tf) -> list[tuple[str, bytes]]:  # tf: tarfile.TarFile
    out = []
    for member in tf.getmembers():
        rel = normalize_member(member.name)
        if member.isdir():
            continue
        if rel is None or not member.isfile():
            raise DigestError(f"refused tar member {member.name!r}: only regular files may ship")
        extracted = tf.extractfile(member)
        if extracted is None:
            raise DigestError(f"cannot read tar member {member.name!r}")
        out.append((rel, extracted.read()))
    return out


# A local Vite build writes `.vite/manifest.json` (pulse's `build.manifest: true`); pulse's `pack-dist.mjs` excludes
# that one top-level directory from the release asset, so a directory bundle skips it the same way and nothing else.
BUILD_ONLY_DIRS = (".vite",)


def dir_entries(root: Path) -> list[tuple[str, bytes]]:
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if rel.split("/", 1)[0] in BUILD_ONLY_DIRS:
            continue
        if path.is_symlink():
            raise DigestError(f"refused symlink {rel!r}: only regular files may ship")
        if path.is_dir():
            continue
        if not path.is_file():
            raise DigestError(f"refused {rel!r}: not a regular file")
        normalized = normalize_member(rel)
        if normalized is None:
            continue
        out.append((normalized, path.read_bytes()))
    return out


def prefixed_entries(names_and_reader: Iterable[tuple[str, bytes | None]], prefix: str) -> list[tuple[str, bytes]]:
    """Entries under `prefix` (a wheel's `maxim/console/ui_dist/` or an sdist's `pymaxim-<v>/src/...`)."""
    out = []
    for name, data in names_and_reader:
        if not name.startswith(prefix):
            continue
        rel = normalize_member(name[len(prefix) :])
        if rel is None:
            continue
        if data is None:
            raise DigestError(f"refused {name!r}: not a regular file")
        out.append((rel, data))
    return out


# ── the stamp, against the pin ───────────────────────────────────────────────────────────────────────────────────


def stamp_problems(stamp: dict, pin: Pin, contract: str, *, releasing: bool) -> tuple[list[str], list[str]]:
    """(problems, notes) for a bundle's `maxim-ui.json` read against the pin and a backend contract.

    The digest is the gate; these exist so a mismatch reads as "you vendored v0.4.0" rather than "hash differs".
    Contract equality is enforced only when a release can ship (fold 1): on any other tree a drift is a loud note,
    so a PR that bumps the contract can merge before pulse has tagged a bundle for it.
    """
    problems: list[str] = []
    notes: list[str] = []
    if stamp.get("target") != "console":
        problems.append(f"bundle target is {stamp.get('target')!r}, expected 'console' (the reachy build?)")
    if stamp.get("dirty") is not False:
        problems.append(
            f"bundle `dirty` is {stamp.get('dirty')!r}, not false: the commit it names cannot be shown to "
            f"describe its contents"
        )
    commit = str(stamp.get("commit") or "")
    if len(commit) < 7 or not _HEX.match(commit) or not pin.commit.startswith(commit):
        problems.append(
            f"bundle was built at commit {commit or '(none)'!r} ({stamp.get('describe')}), not the pinned "
            f"{pin.tag} ({pin.commit[:7]}): vendor the pinned tag (`--from-pin`)"
        )
    if stamp.get("describe") != pin.tag:
        notes.append(f"bundle describe is {stamp.get('describe')!r}; the pin names {pin.tag}")
    drift = []
    if str(stamp.get("contract_version") or "") != contract:
        drift.append(
            f"the bundle was built for contract {stamp.get('contract_version')!r}, the backend is {contract!r}"
        )
    if pin.contract_version != contract:
        drift.append(f"the pin is for contract {pin.contract_version!r}, the backend is {contract!r}")
    for d in drift:
        if releasing:
            problems.append(f"{d}: move the pin to a {contract}-contract pulse tag (`--update-pin <tag>`)")
        else:
            notes.append(f"CONTRACT DRIFT: {d}. A release is blocked until the pin moves to a {contract}-contract tag")
            if os.environ.get("GITHUB_ACTIONS") == "true":  # surfaces in the PR's checks summary, not only the log
                print(f"::warning title=Console contract drift (R4)::{d}; main cannot be released until the pin moves")
    return problems, notes


def releasing() -> tuple[bool, str]:
    """Is this tree cutting a release? `check_nightlies.releasing`, the one definition (it fails closed). This is the
    only function here that touches the network (`git ls-remote`)."""
    from check_nightlies import releasing as _releasing

    return _releasing()
