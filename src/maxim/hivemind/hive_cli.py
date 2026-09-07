"""``maxim hive`` — the user-facing registry, trust, pull, and contribute verbs.

The consumer half of the substrate exchange (1.2 P2P Slices C + D):

- ``hive add`` / ``hive remove`` / ``hive list`` manage the ``hive.json`` Oasis registry.
- ``hive trust`` sets the per-Oasis CONSUMER TRUST POLICY (Slice D): the
  ``--allow-unsigned`` escape hatch, the inherent (safety-floor) opt-in, and the
  operator contributor allow-list. NOTE ``--allow-unsigned`` is not a subscription
  to an Oasis's server-side ``experimental/`` tier — nothing in 1.2 fetches that
  tier; it disables signature verification for the Oasis's *release* stream.
- ``hive pull`` fetches releases from a registered Oasis and ingests them —
  by DELEGATING to the already-tested ``maxim substrate ingest`` verb (verification,
  the V1–V10 gauntlet, journal, and backups are reused verbatim, not reimplemented).
- ``hive contribute`` pushes a bundle to an Oasis's experimental tier via the Slice-B
  client.

**Default trust is Queen-only**: a release that is not signed by a Queen key registered
for that Oasis is REFUSED, and the decay-exempt inherent class is refused, unless the
operator opts in per Oasis. Policy is assembled entirely over ``ingest_bundle``'s shipped
hooks (``trusted_sources`` / ``inherent_trusted_sources`` / ``require_signed`` /
``trusted_keys``) — no new merge code.

rc 0 on success, rc 2 on any operator/input error (message to stderr, no traceback).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from maxim.hivemind.registry import POLICY_FIELDS, HiveRegistry, HiveRegistryError, trust_policy


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
        try:
            policy = trust_policy(o)
        except HiveRegistryError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        verification = "unsigned-allowed" if policy["allow_unsigned"] else "queen-only"
        allow = ", ".join(policy["trusted_sources"]) or "(any Queen-signed)"
        # Reduce the key map to a COUNT before it can reach output. Nothing here
        # prints key material (and Queen keys are public verification anchors,
        # not secrets), but the house rule from `leader_proxy._check_auth` is that
        # key-shaped values never flow into a log or print at all.
        n_anchors: int = len(o.get("queen_keys") or {})
        print(
            f"{o.get('name')}\t{o.get('url')}\tqueen_keys={n_anchors}\tdomains={domains}\n"
            f"    trust: {verification}   inherent: {'yes' if policy['inherent_trust'] else 'no'}   sources: {allow}"
        )
    return 0


def _run_trust(args: argparse.Namespace) -> int:
    """Set the consumer trust policy for a registered Oasis (1.2 Slice D).

    The opposing flags are mutually exclusive at the parser, so a contradictory
    command is an operator error (rc 2) rather than silently granting the looser
    setting — the wrong default for a security-policy verb.
    """
    registry = HiveRegistry(args.registry)
    allow_unsigned = True if args.allow_unsigned else (False if args.require_signed else None)
    inherent = True if args.inherent else (False if args.no_inherent else None)
    sources = list(args.trust_source) if args.trust_source else ([] if args.clear_trust_sources else None)
    if allow_unsigned is None and inherent is None and sources is None:
        print(
            "error: nothing to set — pass --allow-unsigned/--require-signed, --inherent/--no-inherent, "
            "--trust-source ID (repeatable), or --clear-trust-sources.",
            file=sys.stderr,
        )
        return 2
    try:
        entry = registry.set_trust(
            args.name, allow_unsigned=allow_unsigned, inherent_trust=inherent, trusted_sources=sources
        )
        # Derive the printed policy from the POLICY FIELDS ONLY — never from the
        # whole entry, which also carries the Oasis's key map. Same house rule as
        # above: key-shaped values do not flow toward output, even when (as here)
        # they are public anchors and nothing would print them.
        policy = trust_policy({field: entry[field] for field in POLICY_FIELDS if field in entry})
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    verification = "DISABLED (unsigned admitted)" if policy["allow_unsigned"] else "required (Queen-only, default)"
    print(
        f"trust policy for {args.name}:\n"
        f"  signature: {verification}\n"
        f"  inherent:  {'admitted from Queen-verified releases' if policy['inherent_trust'] else 'refused'}\n"
        f"  sources:   {', '.join(policy['trusted_sources']) or '(any contributor the Queen signed)'}"
    )
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
    try:
        policy = trust_policy(oasis)
    except HiveRegistryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not queen_keys and not policy["allow_unsigned"]:
        print(
            f"error: oasis {args.from_oasis!r} has no Queen keys registered — a signed pull needs at least one "
            "(`maxim hive add … --queen-key IDENTITY=PUBKEY_B64`), or disable verification for it with "
            f"`maxim hive trust {args.from_oasis} --allow-unsigned`.",
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

            try:
                manifest = read_bundle_manifest(dest)
            except _malformed as exc:
                print(f"error: {release_id[:12]}… is not a readable bundle: {exc}", file=sys.stderr)
                rc_final = 2
                continue

            signer = manifest.get("signer_identity")
            contributor = manifest.get("contributor_id")
            if not isinstance(contributor, str) or not contributor:
                print(
                    f"skipping {release_id[:12]}…: manifest declares no usable contributor_id ({contributor!r}).",
                    file=sys.stderr,
                )
                rc_final = 2
                continue
            # `signer` is attacker bytes; an unhashable value would raise on the
            # `in` test, so type-check before using it as a key.
            queen_verified = bool(manifest.get("signature")) and isinstance(signer, str) and signer in queen_keys
            if not queen_verified:
                if not policy["allow_unsigned"]:
                    print(
                        f"skipping {release_id[:12]}…: not signed by a Queen key registered for "
                        f"{args.from_oasis!r} (signer={signer!r}). Default trust is Queen-only — opt in with "
                        f"`maxim hive trust {args.from_oasis} --allow-unsigned`.",
                        file=sys.stderr,
                    )
                    rc_final = 2
                    continue
                print(
                    f"warning: admitting UNVERIFIED {release_id[:12]}… from {args.from_oasis!r} — signature "
                    "verification is DISABLED for this Oasis (--allow-unsigned).",
                    file=sys.stderr,
                )

            # Early, friendlier refusal. The AUTHORITATIVE enforcement is the
            # trusted_sources set handed to ingest below (V1), not this check —
            # on the unverified path `contributor` is attacker-chosen, so a CLI-only
            # filter would be trivially bypassable.
            if policy["trusted_sources"] and contributor not in policy["trusted_sources"]:
                print(
                    f"skipping {release_id[:12]}…: contributor {contributor!r} is not in the trusted-source "
                    f"allow-list for {args.from_oasis!r}.",
                    file=sys.stderr,
                )
                rc_final = 2
                continue

            argv = _build_ingest_argv(
                dest,
                session=args.session,
                receiver_body=args.receiver_body,
                contributor=contributor,
                queen_verified=queen_verified,
                queen_keys=queen_keys,
                policy=policy,
                allow_unstamped_geometry=args.allow_unstamped_geometry,
                apply=args.apply,
            )
            rc = run_substrate_subcommand(argv)
            rc_final = rc_final or rc
    return rc_final


def _build_ingest_argv(
    bundle_path: Path,
    *,
    session: str,
    receiver_body: str,
    contributor: str,
    queen_verified: bool,
    queen_keys: dict[str, str],
    policy: dict,
    allow_unstamped_geometry: bool = False,
    apply: bool = False,
) -> list[str]:
    """Compose the ``substrate ingest`` argv that enforces this pull's trust policy.

    Extracted so the security-critical decisions are directly testable:

    - **V1 ``trusted_sources``** is the operator allow-list when one is configured;
      it falls back to the bundle's own ``contributor_id`` ONLY when no allow-list
      exists (the "accept whoever the Queen signed" default). Passing the manifest's
      self-declared id while an allow-list is set would make V1 a tautology and leave
      the allow-list enforceable only in this CLI — trivially bypassable on the
      unverified path, where the id is attacker-chosen.
    - **``--require-signed``** (plus every registered Queen key) rides only on the
      Queen-verified path; omitting it is exactly what ``allow_unsigned`` buys.
    - **``--inherent-trust``** (the decay-exempt safety floor) rides only on the
      Queen-verified path AND is scoped to the same id set as V1, so an allow-list
      also bounds who may reach the safety floor.
    """
    trusted = list(policy["trusted_sources"]) or [contributor]
    argv = ["ingest", str(bundle_path), "--session", session, "--receiver-body", receiver_body]
    for source in trusted:
        argv += ["--trust", source]
    if queen_verified:
        argv.append("--require-signed")
        for identity, pubkey in queen_keys.items():
            argv += ["--trust-key", f"{identity}={pubkey}"]
        if policy["inherent_trust"]:
            for source in trusted:
                argv += ["--inherent-trust", source]
    if allow_unstamped_geometry:
        argv.append("--allow-unstamped-geometry")
    if apply:
        argv.append("--apply")
    return argv


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

    p_trust = sub.add_parser("trust", help="set the consumer trust policy for a registered Oasis")
    p_trust.add_argument("name")
    # Opposing flags are mutually exclusive: for a security-policy verb a
    # contradictory command must be an error, never a silent grant of the looser
    # setting.
    g_sig = p_trust.add_mutually_exclusive_group()
    g_sig.add_argument(
        "--allow-unsigned",
        action="store_true",
        help="DISABLE signature verification for this Oasis's releases (default: refuse unsigned)",
    )
    g_sig.add_argument("--require-signed", action="store_true", help="require Queen-signed releases (the default)")
    g_inh = p_trust.add_mutually_exclusive_group()
    g_inh.add_argument(
        "--inherent",
        action="store_true",
        help="admit the decay-exempt inherent (safety-floor) bias class from Queen-verified releases",
    )
    g_inh.add_argument("--no-inherent", action="store_true", help="refuse the inherent bias class (the default)")
    g_src = p_trust.add_mutually_exclusive_group()
    g_src.add_argument(
        "--trust-source",
        action="append",
        metavar="CONTRIBUTOR_ID",
        help="operator allow-list of contributor ids (repeatable; replaces the list)",
    )
    g_src.add_argument("--clear-trust-sources", action="store_true", help="accept any contributor the Queen signed")
    p_trust.set_defaults(func=_run_trust)

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
