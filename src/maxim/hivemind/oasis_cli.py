"""``maxim oasis`` — serve, publish, and inspect an Oasis's substrate tiers (1.2 P2P Slice C).

The formal, server-side half of the exchange: ``oasis serve`` starts the Slice-B
substrate endpoints (by injecting an :class:`OasisStore` into the leader proxy),
``oasis publish`` adds a signed bundle to the release tier, and ``oasis status``
reports the tier counts. The user-facing pull/contribute/registry half is ``maxim
hive`` (:mod:`maxim.hivemind.hive_cli`).

Verb functions follow the ``hivemind/cli.py`` contract: rc 0 on success, rc 2 on any
operator/input error (message to stderr, never a traceback).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from maxim.hivemind.store import OasisStore, OasisStoreError


def _default_root() -> Path:
    from maxim.utils.paths import data_home

    return data_home() / "oasis"


_LOOPBACK = ("127.0.0.1", "::1", "localhost")


def _run_serve(args: argparse.Namespace) -> int:
    import threading

    from maxim.runtime.leader_proxy import DEFAULT_PROXY_PORT, start_leader_proxy
    from maxim.tunnel.keys import read_key

    store = OasisStore(args.root or _default_root())
    api_key = read_key()
    # Fail closed: an unauthenticated `/v1/substrate/contribute` (a WRITE) on a
    # non-loopback interface is an open door to the LAN/internet. Refuse unless
    # the operator explicitly opts into it with --insecure.
    if api_key is None and args.bind_host not in _LOOPBACK and not args.insecure:
        print(
            f"error: refusing to serve on {args.bind_host} without auth. Set up a bearer key "
            "(`maxim tunnel setup`), or bind loopback (`--bind-host 127.0.0.1`), or pass --insecure "
            "to deliberately serve auth-off on this interface.",
            file=sys.stderr,
        )
        return 2
    if api_key is None:
        print(
            "warning: serving substrate endpoints WITHOUT auth (no api_key). "
            "Anyone who can reach this address may read releases and push contributions.",
            file=sys.stderr,
        )
    port = args.port or DEFAULT_PROXY_PORT
    server = start_leader_proxy(
        proxy_port=port,
        api_key=api_key,
        bind_host=args.bind_host,
        oasis_store=store,
    )
    if server is None:
        print(
            f"error: could not bind the Oasis server on {args.bind_host}:{port} (port in use?).",
            file=sys.stderr,
        )
        return 2
    # Guard against the leader-proxy singleton: on the default port a proxy
    # already running IN THIS PROCESS is returned store-less, and the substrate
    # routes would silently 404. Detect it rather than serve a dead surface.
    if getattr(server.RequestHandlerClass, "oasis_store", None) is not store:
        print(
            f"error: a leader proxy is already running on port {port} without an Oasis store. "
            "Run `maxim oasis serve` as its own process, or pass a distinct --port.",
            file=sys.stderr,
        )
        return 2
    try:
        n_releases = len(store.list_releases())
        n_contrib = len(store.list_contributions())
    except OasisStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        server.shutdown()
        server.server_close()
        return 2
    print(
        f"Oasis serving on {args.bind_host}:{port} (store: {store.root}, auth: {'on' if api_key else 'OFF'}).\n"
        f"  releases: {n_releases}   experimental: {n_contrib}\n"
        "Ctrl+C to stop."
    )
    try:
        threading.Event().wait()  # block until interrupted; the proxy runs in a daemon thread
    except KeyboardInterrupt:
        print("\nstopping Oasis…")
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _run_publish(args: argparse.Namespace) -> int:
    store = OasisStore(args.root or _default_root())
    bundle_path = Path(args.bundle).expanduser().resolve()
    if not bundle_path.is_file():
        print(f"error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2
    try:
        release_id = store.publish_release(bundle_path)
    except OasisStoreError as exc:
        # The commonest case: an unsigned bundle. Point at the signing verb.
        print(
            f"error: {exc}\n  hint: sign it first with `maxim substrate export --sign` (or `keygen` to mint a key).",
            file=sys.stderr,
        )
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"published release {release_id}\n  store: {store.releases_dir}")
    return 0


def _run_status(args: argparse.Namespace) -> int:
    store = OasisStore(args.root or _default_root())
    try:
        releases = store.list_releases()
        contributions = store.list_contributions()
    except OasisStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Oasis store: {store.root}\n"
        f"  releases (Queen tier):        {len(releases)}\n"
        f"  contributions (experimental): {len(contributions)}"
    )
    for r in releases:
        print(f"    release {r['id'][:12]}…  contributor={r.get('contributor_id')}  domain={r.get('domain')}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim oasis", description="Serve and publish an Oasis's shared substrate.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_serve = sub.add_parser("serve", help="start the substrate exchange endpoints")
    p_serve.add_argument("--root", type=Path, default=None, help="store root (default: ~/.maxim/oasis)")
    p_serve.add_argument("--port", type=int, default=None, help="listen port (default: leader proxy port 8099)")
    p_serve.add_argument("--bind-host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p_serve.add_argument(
        "--insecure",
        action="store_true",
        help="allow serving auth-off on a non-loopback interface (default: refuse without a bearer key)",
    )
    p_serve.set_defaults(func=_run_serve)

    p_pub = sub.add_parser("publish", help="add a SIGNED bundle to the release tier")
    p_pub.add_argument("bundle", help="path to a signed bundle (maxim substrate export --sign)")
    p_pub.add_argument("--root", type=Path, default=None, help="store root (default: ~/.maxim/oasis)")
    p_pub.set_defaults(func=_run_publish)

    p_status = sub.add_parser("status", help="show release/experimental tier counts")
    p_status.add_argument("--root", type=Path, default=None, help="store root (default: ~/.maxim/oasis)")
    p_status.set_defaults(func=_run_status)

    return parser


def run_oasis_subcommand(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
