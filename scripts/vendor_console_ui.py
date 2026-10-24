#!/usr/bin/env python3
"""Vendor the pinned Console UI bundle into the package for release (R4, #1333).

The Console UI is built in **maxim-pulse**; the wheel ships a copy so that a
plain ``pip install pymaxim[console] && maxim serve`` serves a working Console
with no flag and no config. This script is the copy step, and since R4 it only
copies the bundle the committed pin names (``scripts/console_ui_pin.json``):

  * the release path, and CI's: download the pinned tag's release asset, check
    its sha256 and its file digest against the pin, then vendor it::

        python scripts/vendor_console_ui.py --from-pin

  * prove the pin is the NEWEST pulse tag for this backend's contract (the
    release PR, and release.yml's publish workflow from #1372; network, fails closed)::

        python scripts/vendor_console_ui.py --verify-pin [--only-when-releasing]

  * move the pin (the release PR, after pulse tags a new bundle)::

        python scripts/vendor_console_ui.py --update-pin v0.4.2

  * a local directory still vendors, but only if it is byte-identical to the
    pinned asset (a dev build is not; ``--force`` is the dev escape hatch, and
    the release audit has none)::

        python scripts/vendor_console_ui.py ~/Scripts/Maxim-pulse/apps/console/dist

  * ``--check`` validates what is vendored now; ``--clean`` removes it.

WHY. pymaxim 1.3.2 vendored pulse v0.4.0 two days after v0.4.1 (the stall-line
fix for that very release) was tagged: this script took any directory and
refused only a contract mismatch or a dirty build. Staleness used to be "a
judgment call this script cannot make"; the pin now makes "current" a
committed, reviewed fact, and ``--verify-pin`` proves it against pulse's tags.
Design and adversarial pass: docs/plans/archive/r4_console_vendor_gate.md.

The destination (``src/maxim/console/ui_dist/``) is .gitignore'd: vendoring is
a RELEASE step, not a commit.

Exits: 0 ok; 1 refused (a mismatch, a stale pin); 2 could not check (network,
an unreadable pin). Never a pass on an unreadable input.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _console_pin as cp  # noqa: E402

REPO_ROOT = cp.REPO_ROOT
DEST = REPO_ROOT / "src" / "maxim" / "console" / "ui_dist"
MANIFEST_NAME = cp.MANIFEST_NAME

ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"
RELEASE_API = "https://api.github.com/repos/{repo}/releases/tags/{tag}"
OPENAPI_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/packages/kit/openapi.json"
PULSE_GIT = "https://github.com/{repo}.git"

Fetch = Callable[[str], bytes]
LsRemote = Callable[[str], str]


class Unverified(Exception):
    """An input could not be read: exit 2."""


# ── network seams (injected in tests; the defaults are the only code that touches the network) ─────────────────────


def http_fetch(url: str, *, attempts: int = 3) -> bytes:
    """GET with a bounded retry (fold 7). Uses GH_TOKEN for api.github.com when set."""
    import os

    headers = {"User-Agent": "pymaxim-vendor-console-ui"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    raise Unverified(f"GET {url} failed after {attempts} attempts: {last}")


def git_ls_remote(url: str) -> str:
    proc = subprocess.run(("git", "ls-remote", "--tags", url), capture_output=True, text=True, check=False, timeout=60)
    if proc.returncode != 0:
        raise Unverified(f"git ls-remote --tags {url}: {proc.stderr.strip() or proc.returncode}")
    return proc.stdout


def _shown(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ── the bundle ─────────────────────────────────────────────────────────────────────────────────────────────────────


def validate(source: Path, *, pin: cp.Pin, releasing: bool = True) -> tuple[bool, list[str]]:
    """Is ``source`` the pinned Console bundle? Returns (ok, problems); notes are printed.

    ``pin`` is required (fold 6): no caller may silently validate against no pin. ``releasing=False`` downgrades a
    contract drift to a loud note (fold 1); everything else refuses either way.
    """
    if not source.is_dir():
        return False, [f"{source} is not a directory"]
    problems: list[str] = list(cp.pin_identity_problems(pin))
    if not (source / "index.html").is_file():
        problems.append("no index.html — is this really a built bundle?")
    if not (source / "assets").is_dir():
        problems.append("no assets/ directory")
    manifest_path = source / MANIFEST_NAME
    if not manifest_path.is_file():
        problems.append(f"no {MANIFEST_NAME} — cannot tell which pulse build this is")
        return False, problems
    try:
        stamp = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return False, [*problems, f"{MANIFEST_NAME} is unreadable: {e}"]
    if not isinstance(stamp, dict):
        return False, [*problems, f"{MANIFEST_NAME} is not a JSON object"]
    try:
        contract = cp.backend_contract()
    except cp.PinError as e:
        return False, [*problems, str(e)]
    stamp_problems, notes = cp.stamp_problems(stamp, pin, contract, releasing=releasing)
    problems.extend(stamp_problems)
    try:
        digest = cp.tree_digest(cp.dir_entries(source))
    except cp.DigestError as e:
        problems.append(str(e))
    else:
        if digest != pin.tree_sha256:
            problems.append(
                f"the bundle's files differ from the pinned {pin.tag} asset (tree digest {digest[:12]} != "
                f"{pin.tree_sha256[:12]}): vendor the pinned release asset with `--from-pin`"
            )
    for key in ("app_version", "describe", "commit_date"):
        if stamp.get(key):
            notes.append(f"{key} {stamp[key]}")
    for note in notes:
        print(f"  bundle: {note}")
    return (not problems), problems


def vendor(source: Path, *, pin: cp.Pin, force: bool = False, releasing: bool = True) -> int:
    ok, problems = validate(source, pin=pin, releasing=releasing)
    if not ok and not force:
        print(f"REFUSING to vendor {source}:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("\nPass --force to vendor anyway (a dev build only: the release audit refuses it).", file=sys.stderr)
        return 1
    for p in problems:  # forced: still say what's wrong
        print(f"WARNING: {p}", file=sys.stderr)

    if source.resolve() == DEST.resolve():
        print(f"REFUSED: {source} IS the vendored bundle; vendor from a pulse bundle or use --check", file=sys.stderr)
        return 1
    if DEST.exists():
        shutil.rmtree(DEST)
    shutil.copytree(
        source, DEST, ignore=lambda d, names: [n for n in names if Path(d) == source and n in cp.BUILD_ONLY_DIRS]
    )
    if not force and cp.tree_digest(cp.dir_entries(DEST)) != pin.tree_sha256:
        print(f"REFUSED: {_shown(DEST)} does not re-digest to the pin after the copy", file=sys.stderr)
        shutil.rmtree(DEST)
        return 1
    files = sum(1 for _ in DEST.rglob("*") if _.is_file())
    print(f"vendored {pin.tag} ({files} files) → {_shown(DEST)}")
    print("This is a RELEASE step; ui_dist/ is .gitignore'd. Build the wheel now (python -I -m build).")
    return 0


# ── the pinned release asset ───────────────────────────────────────────────────────────────────────────────────────


def fetch_asset(pin: cp.Pin, *, fetch: Fetch = http_fetch, cache_dir: Path | None = None) -> bytes:
    """The pinned asset's bytes, sha256-checked before anything uses them (no --force reaches this)."""
    cached = cache_dir / f"{pin.asset_sha256}.tar.gz" if cache_dir else None
    if cached is not None and cached.is_file():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == pin.asset_sha256:
            return data
        print(f"cached asset {cached} has the wrong sha256; downloading again", file=sys.stderr)
    data = fetch(ASSET_URL.format(repo=cp.PULSE_REPO, tag=pin.tag, asset=cp.PULSE_ASSET))
    got = hashlib.sha256(data).hexdigest()
    if got != pin.asset_sha256:
        raise ValueError(f"{cp.PULSE_ASSET} for {pin.tag} has sha256 {got}, the pin says {pin.asset_sha256}")
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        except OSError as exc:  # the cache is an optimisation; the verified bytes are already in hand
            print(f"WARNING: could not keep the asset in {cache_dir}: {exc}", file=sys.stderr)
    return data


def asset_tree_digest(data: bytes) -> str:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        return cp.tree_digest(cp.tar_entries(tf))


def extract_asset(data: bytes, into: Path) -> None:
    if not hasattr(tarfile, "data_filter"):  # fold 12
        raise RuntimeError("this Python's tarfile has no `data` filter (needs 3.10.12+/3.11.4+); refusing to extract")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        cp.tar_entries(tf)  # refuses links, devices, dotfiles, traversal before a byte is written
        tf.extractall(into, filter="data")


def from_pin(pin: cp.Pin, *, fetch: Fetch = http_fetch, cache_dir: Path | None = None, releasing: bool = True) -> int:
    try:
        data = fetch_asset(pin, fetch=fetch, cache_dir=cache_dir)
    except Unverified as e:
        print(f"UNVERIFIED: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        try:
            extract_asset(data, Path(tmp))
        except (cp.DigestError, RuntimeError, tarfile.TarError) as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 1
        return vendor(Path(tmp), pin=pin, releasing=releasing)


# ── is the pin the newest tag for the contract? (network) ──────────────────────────────────────────────────────────


def parse_ls_remote(text: str) -> dict[str, str]:
    """{tag: commit}, peeling annotated tags (`^{}` lines win) and keeping lightweight ones (fold 9)."""
    direct: dict[str, str] = {}
    peeled: dict[str, str] = {}
    for line in text.splitlines():
        sha, _, ref = line.partition("\t")
        if not ref.startswith("refs/tags/"):
            continue
        name = ref[len("refs/tags/") :]
        if name.endswith("^{}"):
            peeled[name[:-3]] = sha
        else:
            direct[name] = sha
    return {name: peeled.get(name, sha) for name, sha in direct.items()}


def tag_contract(commit: str, *, fetch: Fetch) -> str:
    try:
        version = json.loads(fetch(OPENAPI_RAW.format(repo=cp.PULSE_REPO, commit=commit)))["info"]["version"]
    except (ValueError, KeyError, TypeError) as e:
        raise Unverified(f"packages/kit/openapi.json at {commit[:12]} has no readable info.version ({e})") from e
    return str(version)


def release_asset_digest(tag: str, *, fetch: Fetch) -> str:
    try:
        release = json.loads(fetch(RELEASE_API.format(repo=cp.PULSE_REPO, tag=tag)))
        asset = next(a for a in release["assets"] if a.get("name") == cp.PULSE_ASSET)
    except (ValueError, KeyError, TypeError, AttributeError, StopIteration) as e:
        raise Unverified(f"release {tag} has no readable {cp.PULSE_ASSET} asset ({e!r})") from e
    digest = str(asset.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise Unverified(f"release {tag}'s {cp.PULSE_ASSET} carries no sha256 digest ({digest!r})")
    return digest[len("sha256:") :]


def resolve_tag(tag: str, contract: str, *, ls_remote: LsRemote, fetch: Fetch) -> tuple[cp.Pin, list[str]]:
    """Derive a full pin for ``tag`` from live data, plus the newer same-contract tags (empty when it is newest)."""
    tags = parse_ls_remote(ls_remote(PULSE_GIT.format(repo=cp.PULSE_REPO)))
    if tag not in tags:
        raise LookupError(f"{cp.PULSE_REPO} has no tag {tag}")
    key = cp.tag_key(tag)
    if key is None:
        raise LookupError(f"{tag} is not a stable vX.Y.Z tag")
    newer_same = []
    for other, commit in sorted(tags.items(), key=lambda kv: cp.tag_key(kv[0]) or (-1, -1, -1)):
        other_key = cp.tag_key(other)
        if other_key is None or other_key <= key:
            continue  # pre-releases and older tags never decide "newest" (fold 4, fold 9)
        if tag_contract(commit, fetch=fetch) == contract:  # unreadable -> Unverified, never skipped
            newer_same.append(other)
    commit = tags[tag]
    tag_contract_version = tag_contract(commit, fetch=fetch)
    if tag_contract_version != contract:
        raise LookupError(f"{tag} was built for contract {tag_contract_version}, the backend is {contract}")
    asset_sha = release_asset_digest(tag, fetch=fetch)
    data = fetch(ASSET_URL.format(repo=cp.PULSE_REPO, tag=tag, asset=cp.PULSE_ASSET))
    if hashlib.sha256(data).hexdigest() != asset_sha:
        raise LookupError(f"the downloaded {cp.PULSE_ASSET} for {tag} does not match its release digest")
    try:
        tree_sha256 = asset_tree_digest(data)
    except tarfile.TarError as e:
        raise Unverified(f"{cp.PULSE_ASSET} for {tag} is not a readable gzip tar ({e})") from e
    pin = cp.Pin(
        repo=cp.PULSE_REPO,
        tag=tag,
        commit=commit,
        contract_version=contract,
        asset=cp.PULSE_ASSET,
        asset_sha256=asset_sha,
        tree_sha256=tree_sha256,
    )
    return pin, newer_same


def verify_pin(pin: cp.Pin, *, ls_remote: LsRemote = git_ls_remote, fetch: Fetch = http_fetch) -> int:
    problems = cp.pin_identity_problems(pin)
    try:
        contract = cp.backend_contract()
        live, newer = resolve_tag(pin.tag, contract, ls_remote=ls_remote, fetch=fetch)
    except (Unverified, cp.PinError) as e:
        print(f"UNVERIFIED: {e}", file=sys.stderr)
        return 2
    except (LookupError, cp.DigestError) as e:
        problems.append(str(e))
        live, newer = None, []
    if newer:
        problems.append(
            f"the pin is {pin.tag}, but {newer[-1]} is the newest pulse tag for contract {contract}: "
            f"`python scripts/vendor_console_ui.py --update-pin {newer[-1]}`"
        )
    if live is not None:
        for field in ("commit", "contract_version", "asset_sha256", "tree_sha256"):
            if getattr(live, field) != getattr(pin, field):
                problems.append(
                    f"pin {field} {getattr(pin, field)!r} != live {getattr(live, field)!r}"
                    + (" (was the tag moved?)" if field == "commit" else "")
                )
    if problems:
        print("REFUSED: the console pin is not the newest pulse bundle for this contract:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"console pin current: {pin.tag} ({pin.commit[:7]}, contract {pin.contract_version})")
    return 0


def update_pin(
    tag: str, *, path: Path = cp.PIN_PATH, ls_remote: LsRemote = git_ls_remote, fetch: Fetch = http_fetch
) -> int:
    try:
        pin, newer = resolve_tag(tag, cp.backend_contract(), ls_remote=ls_remote, fetch=fetch)
    except (Unverified, cp.PinError) as e:
        print(f"UNVERIFIED: {e}", file=sys.stderr)
        return 2
    except (LookupError, cp.DigestError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    if newer:
        print(f"WARNING: {newer[-1]} is newer for the same contract; --verify-pin will refuse {tag}", file=sys.stderr)
    path.write_text(json.dumps(cp.pin_payload(pin), indent=2) + "\n", encoding="utf-8")
    print(f"pinned {tag} ({pin.commit[:7]}, contract {pin.contract_version}) in {_shown(path)}")
    return 0


def check(*, pin: cp.Pin, releasing: bool = True) -> int:
    """Is the pinned bundle vendored right now?"""
    if not DEST.is_dir() or not any(DEST.rglob("*")):
        print(f"no vendored bundle at {_shown(DEST)} (source checkout state)")
        return 1
    ok, problems = validate(DEST, pin=pin, releasing=releasing)
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print(f"vendored bundle at {_shown(DEST)}: {'the pinned ' + pin.tag if ok else 'NOT the pin'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", help="A built Console bundle directory (must match the pin byte for byte)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--from-pin", action="store_true", help="Download, check and vendor the pinned release asset")
    mode.add_argument("--verify-pin", action="store_true", help="Prove the pin is the newest tag for the contract")
    mode.add_argument("--update-pin", metavar="TAG", help="Rewrite the pin from pulse's live TAG")
    mode.add_argument("--check", action="store_true", help="Validate the vendored bundle against the pin")
    mode.add_argument("--clean", action="store_true", help="Remove the vendored bundle and exit")
    ap.add_argument("--force", action="store_true", help="Vendor a directory that does not match the pin (dev only)")
    ap.add_argument("--asset-cache", type=Path, default=None, help="--from-pin: reuse/keep the asset here by sha256")
    ap.add_argument(
        "--only-when-releasing", action="store_true", help="--verify-pin: a no-op unless this tree cuts a release"
    )
    ap.add_argument(
        "--contract-drift-ok-unless-releasing",
        action="store_true",
        help="CI only: on a tree that is not cutting a release, a contract drift is a loud note, not a refusal "
        "(a PR that bumps the contract can merge before pulse tags a bundle for it). Releasing trees refuse.",
    )
    args = ap.parse_args(argv)
    if args.only_when_releasing and not args.verify_pin:
        ap.error("--only-when-releasing applies to --verify-pin only")
    if args.from_pin and (args.force or args.source):
        ap.error("--from-pin vendors the pinned asset only: no source path, and no --force")
    if args.asset_cache and not args.from_pin:
        ap.error("--asset-cache applies to --from-pin only")

    if args.clean:
        if DEST.exists():
            shutil.rmtree(DEST)
            print(f"removed {_shown(DEST)}")
        else:
            print("nothing to clean")
        return 0
    if args.update_pin:
        return update_pin(args.update_pin)
    try:
        pin = cp.load_pin()
    except cp.PinError as e:
        print(f"UNVERIFIED: {e}", file=sys.stderr)
        return 2
    try:
        cp.backend_contract()  # unreadable is exit 2 here, not a "refused" from inside validate
    except cp.PinError as e:
        print(f"UNVERIFIED: {e}", file=sys.stderr)
        return 2

    releasing = True
    if args.contract_drift_ok_unless_releasing or args.only_when_releasing:
        releasing, why = cp.releasing()
        print(why)
    if args.verify_pin:
        if args.only_when_releasing and not releasing:
            return 0
        return verify_pin(pin)
    if args.from_pin:
        return from_pin(pin, cache_dir=args.asset_cache, releasing=releasing)
    if args.check:
        return check(pin=pin, releasing=releasing)
    if not args.source:
        ap.error("a source path is required (or use --from-pin / --verify-pin / --update-pin / --check / --clean)")
    return vendor(Path(args.source).expanduser().resolve(), pin=pin, force=args.force, releasing=releasing)


if __name__ == "__main__":
    raise SystemExit(main())
