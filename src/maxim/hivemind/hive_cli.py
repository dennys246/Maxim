"""``maxim hive`` — the user-facing registry, pull, and contribute verbs (1.2 P2P Slice C).

The consumer half of the substrate exchange:

- ``hive add`` / ``hive remove`` / ``hive list`` manage the ``hive.json`` Oasis registry.
- ``hive pull`` fetches signed releases from a registered Oasis and ingests them —
  by DELEGATING to the already-tested ``maxim substrate ingest`` verb (verification,
  the V1–V10 gauntlet, journal, and backups are reused verbatim, not reimplemented).
- ``hive contribute`` pushes a bundle to an Oasis's experimental tier via the Slice-B
  client.

rc 0 on success, rc 2 on any operator/input error (message to stderr, no traceback).
The default consumer trust *policy* (auto-Queen-require) is Slice D; ``hive pull``
here requires an explicit or registry-sourced set of Queen keys.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from maxim.hivemind.registry import HiveRegistry, HiveRegistryError


def _parse_key_specs(specs: list[str] | None, label: str) -> dict[str, str]:
    """Parse repeated ``IDENTITY=PUBKEY_B64`` args into a dict (raises on a bad spec)."""
    out: dict[str, str] = {}
    for spec in specs or []:
        identity, sep, pubkey = spec.partition("=")
        if not sep or not identity or not pubkey:
            raise HiveRegistryError(f"{label} must be IDENTITY=PUBKEY_B64, got {spec!r}")
        out[identity] = pubkey
    return out


def _run_add(args: argparse.Namespace) -> int:
    registry = HiveRegistry(args.registry)
    try:
        queen_keys = _parse_key_specs(args.queen_key, "--queen-key")
        entry = registry.add(args.name, args.url, queen_keys=queen_keys, domains=tuple(args.domain or ()))
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"registered oasis {entry['name']} → {entry['url']}\n"
        f"  queen keys: {len(entry['queen_keys'])}   domains: {', '.join(entry['domains']) or '(all)'}"
    )
    return 0


def _run_remove(args: argparse.Namespace) -> int:
    registry = HiveRegistry(args.registry)
    try:
        removed = registry.remove(args.name)
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not removed:
        print(f"error: no registered oasis named {args.name!r}", file=sys.stderr)
        return 2
    print(f"removed oasis {args.name}")
    return 0


def _run_list(args: argparse.Namespace) -> int:
    registry = HiveRegistry(args.registry)
    try:
        oases = registry.list_oases()
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not oases:
        print("no registered Oases (add one with `maxim hive add <name> <url>`)")
        return 0
    for o in oases:
        domains = ", ".join(o.get("domains") or []) or "(all)"
        print(f"{o.get('name')}\t{o.get('url')}\tqueen_keys={len(o.get('queen_keys') or {})}\tdomains={domains}")
    return 0


def _resolve_oasis(registry: HiveRegistry, name: str) -> dict:
    entry = registry.get(name)
    if entry is None:
        raise HiveRegistryError(f"no registered oasis named {name!r} (see `maxim hive list`)")
    return entry


def _run_pull(args: argparse.Namespace) -> int:
    import zipfile

    from maxim.hivemind import substrate_client as sc
    from maxim.hivemind.bundle import read_bundle_manifest
    from maxim.hivemind.cli import run_substrate_subcommand
    from maxim.hivemind.store import is_valid_release_id
    from maxim.tunnel.keys import read_key
    from maxim.utils import http

    _malformed = (zipfile.BadZipFile, KeyError, ValueError, OSError)

    registry = HiveRegistry(args.registry)
    try:
        oasis = _resolve_oasis(registry, args.from_oasis)
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    url = oasis["url"]
    queen_keys: dict[str, str] = dict(oasis.get("queen_keys") or {})
    if not queen_keys:
        print(
            f"error: oasis {args.from_oasis!r} has no Queen keys registered — a signed pull needs at least one "
            "(`maxim hive add … --queen-key IDENTITY=PUBKEY_B64`).",
            file=sys.stderr,
        )
        return 2
    api_key = args.api_key or read_key()

    try:
        releases = sc.list_releases(url, api_key=api_key)
    except (sc.SubstrateExchangeError, http.HTTPError) as exc:
        print(f"error: could not list releases from {url}: {exc}", file=sys.stderr)
        return 2

    # Filter by domain / explicit release id.
    if args.release:
        releases = [r for r in releases if r.get("id") == args.release]
    if args.domain:
        releases = [r for r in releases if r.get("domain") == args.domain]
    if not releases:
        print("no matching releases to pull.")
        return 0

    rc_final = 0
    with tempfile.TemporaryDirectory(prefix="maxim-hive-pull-") as tmp:
        for rel in releases:
            release_id = rel.get("id")
            if not isinstance(release_id, str):
                continue
            # Never build a filesystem path from an untrusted id: a crafted
            # absolute or "../" id would otherwise escape the temp dir and write
            # attacker bytes (before the signature is even checked).
            if not is_valid_release_id(release_id):
                print(f"skipping malformed release id {release_id!r} from {url}", file=sys.stderr)
                rc_final = 2
                continue
            dest = Path(tmp) / f"{release_id}.zip"
            try:
                sc.fetch_bundle(url, release_id, dest, api_key=api_key)
            except (sc.SubstrateExchangeError, http.HTTPError) as exc:
                print(f"error: fetch {release_id[:12]}… failed: {exc}", file=sys.stderr)
                rc_final = 2
                continue

            # The signature is verified by `substrate ingest --require-signed`
            # against these Queen keys — that is the real trust gate. We read the
            # contributor_id only to satisfy ingest's V1 front door; V1 adds no
            # protection HERE (contributor_id is inside the signed payload, so a
            # valid Queen signature already vouches for it), and operator-level
            # source allow-listing is deliberately Slice D policy, not baked in.
            # signer_identity is the early-fail check that the release claims a
            # Queen we trust (the cryptographic check still runs regardless).
            try:
                manifest = read_bundle_manifest(dest)
            except _malformed as exc:
                print(f"error: {release_id[:12]}… is not a readable bundle: {exc}", file=sys.stderr)
                rc_final = 2
                continue
            signer = manifest.get("signer_identity")
            if signer not in queen_keys:
                print(
                    f"skipping {release_id[:12]}…: signed by {signer!r}, not a Queen key registered for "
                    f"{args.from_oasis!r}.",
                    file=sys.stderr,
                )
                rc_final = 2
                continue

            argv = [
                "ingest",
                str(dest),
                "--session",
                args.session,
                "--receiver-body",
                args.receiver_body,
                "--require-signed",
                "--trust",
                str(manifest.get("contributor_id")),
            ]
            for identity, pubkey in queen_keys.items():
                argv += ["--trust-key", f"{identity}={pubkey}"]
            if args.allow_unstamped_geometry:
                argv.append("--allow-unstamped-geometry")
            if args.apply:
                argv.append("--apply")
            rc = run_substrate_subcommand(argv)
            rc_final = rc_final or rc
    return rc_final


def _run_contribute(args: argparse.Namespace) -> int:
    from maxim.hivemind import substrate_client as sc
    from maxim.tunnel.keys import read_key
    from maxim.utils import http

    registry = HiveRegistry(args.registry)
    try:
        oasis = _resolve_oasis(registry, args.to_oasis)
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    bundle_path = Path(args.bundle).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2
    api_key = args.api_key or read_key()
    try:
        receipt = sc.contribute(oasis["url"], bundle_path, api_key=api_key)
    except (sc.SubstrateExchangeError, http.HTTPError) as exc:
        print(f"error: contribution to {args.to_oasis!r} failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"contributed to {args.to_oasis} (experimental tier)\n"
        f"  digest: {receipt.get('digest', '')[:12]}…   status: {receipt.get('status')}"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim hive", description="Pull from and contribute to registered Oases.")
    parser.add_argument("--registry", type=Path, default=None, help=argparse.SUPPRESS)  # test seam
    sub = parser.add_subparsers(dest="action", required=True)

    p_add = sub.add_parser("add", help="register an Oasis")
    p_add.add_argument("name")
    p_add.add_argument("url")
    p_add.add_argument(
        "--queen-key", action="append", metavar="IDENTITY=PUBKEY_B64", help="a Queen public key (repeatable)"
    )
    p_add.add_argument("--domain", action="append", help="subscribe to a substrate domain (repeatable)")
    p_add.set_defaults(func=_run_add)

    p_rm = sub.add_parser("remove", help="unregister an Oasis")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=_run_remove)

    p_list = sub.add_parser("list", help="list registered Oases")
    p_list.set_defaults(func=_run_list)

    p_pull = sub.add_parser("pull", help="fetch + ingest signed releases from a registered Oasis")
    p_pull.add_argument("--from", dest="from_oasis", required=True, help="registered Oasis name")
    p_pull.add_argument("--domain", default=None, help="only pull releases tagged with this domain")
    p_pull.add_argument("--release", default=None, help="pull only this release id")
    p_pull.add_argument("--session", required=True, help="receiver session dir or id (a maxim.create.agent() home)")
    p_pull.add_argument("--receiver-body", required=True, help="the receiver's body_ref (gate-7 body check)")
    p_pull.add_argument("--api-key", default=None, help="bearer token for the Oasis (default: local api_key)")
    p_pull.add_argument(
        "--allow-unstamped-geometry",
        action="store_true",
        help="admit EC nodes without a geometry stamp (legacy archives; passed through to ingest)",
    )
    p_pull.add_argument("--apply", action="store_true", help="write the merge (default: dry run)")
    p_pull.set_defaults(func=_run_pull)

    p_con = sub.add_parser("contribute", help="push a bundle to an Oasis's experimental tier")
    p_con.add_argument("bundle")
    p_con.add_argument("--to", dest="to_oasis", required=True, help="registered Oasis name")
    p_con.add_argument("--api-key", default=None, help="bearer token for the Oasis (default: local api_key)")
    p_con.set_defaults(func=_run_contribute)

    return parser


def run_hive_subcommand(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
