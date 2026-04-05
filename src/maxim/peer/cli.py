"""`maxim peer connect/show/forget` — peer-side config management.

Symmetric to `maxim tunnel setup` on the leader side. One-time setup on
the peer machine, then `maxim` auto-reads the stored URL + key on every
subsequent launch.
"""
from __future__ import annotations

import getpass
import sys
from collections.abc import Sequence

from maxim.peer.config import (
    PeerConfig,
    delete_peer_config,
    peer_config_path,
    read_peer_config,
    truncate_key,
    write_peer_config,
)


CONNECT_USAGE = """\
Usage: maxim peer connect <url> [options]

One-time setup: stores a leader URL + API key so subsequent `maxim` runs
route inference through it automatically. Mirrors `maxim tunnel setup` on
the leader side.

Options:
  --key KEY        Bearer token (prompts if omitted)
  --model MODEL    Model name to send (optional; server picks default)
  --skip-test      Don't verify connectivity before saving
"""


def run_peer_connect_subcommand(argv: Sequence[str]) -> int:
    """Dispatch peer config subcommands: connect / show / forget."""
    if not argv or argv[0] in ("-h", "--help"):
        _print_peer_usage()
        return 0 if argv else 2
    action = argv[0]
    if action == "connect":
        return _cmd_connect(list(argv[1:]))
    if action == "show":
        return _cmd_show()
    if action == "forget":
        return _cmd_forget()
    # Fall through to maxim.doctor.cli for `peer test` (kept in doctor/ because
    # test is a diagnostic, not a configuration subcommand)
    if action == "test":
        from maxim.doctor.cli import run_peer_subcommand
        return run_peer_subcommand(argv)
    print(f"Unknown peer action: {action}", file=sys.stderr)
    _print_peer_usage()
    return 2


def _print_peer_usage() -> None:
    print("Usage: maxim peer <action> [options]")
    print()
    print("Actions:")
    print("  connect <url>    Configure this peer to route inference to a leader")
    print("  show             Show current peer configuration")
    print("  forget           Remove stored peer config")
    print("  test <url>       Verify a leader URL is reachable + authenticated")


# ─── connect ──────────────────────────────────────────────────────────────

def _cmd_connect(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(CONNECT_USAGE)
        return 0 if argv else 2

    url = argv[0].rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"

    # Parse options
    key: str | None = None
    model: str | None = None
    skip_test = False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--key" and i + 1 < len(argv):
            key = argv[i + 1]
            i += 2
        elif a == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif a == "--skip-test":
            skip_test = True
            i += 1
        else:
            print(f"Unknown option: {a}", file=sys.stderr)
            return 2

    if key is None:
        try:
            key = getpass.getpass("Paste the leader's API key (hidden input): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1
    if not key:
        print("✗ API key required.", file=sys.stderr)
        return 1

    is_cloud = _is_public_url(url)

    print()
    print("─" * 62)
    print(f"  Peer connect — {url}")
    print("─" * 62)

    # Verify connectivity (unless skipped)
    if not skip_test:
        print("Verifying connection...")
        rc = _run_peer_test(url, key)
        if rc != 0:
            print()
            print("✗ Connection test failed. Config NOT saved.")
            print("  Use --skip-test to save anyway, or `maxim peer test <url>` to")
            print("  debug independently.")
            return 1

    # Save config
    cfg = PeerConfig(url=url, api_key=key, model=model, is_cloud=is_cloud)
    path = write_peer_config(cfg)
    print()
    print(f"✓ Saved peer config to {path}")
    print(f"  url:      {url}")
    print(f"  api_key:  {truncate_key(key)}")
    if model:
        print(f"  model:    {model}")
    if is_cloud:
        print("  is_cloud: true (cloud-lane gate will be auto-enabled)")
    print()
    print("You can now run `maxim` — it will auto-route to this leader.")
    return 0


def _cmd_show() -> int:
    cfg = read_peer_config()
    if cfg is None:
        print(f"No peer config at {peer_config_path()}")
        print("Run: maxim peer connect <url>")
        return 1
    print(f"Peer config: {peer_config_path()}")
    print(f"  url:      {cfg.url}")
    print(f"  api_key:  {truncate_key(cfg.api_key)}")
    if cfg.model:
        print(f"  model:    {cfg.model}")
    if cfg.is_cloud:
        print("  is_cloud: true")
    return 0


def _cmd_forget() -> int:
    path = peer_config_path()
    if delete_peer_config():
        print(f"✓ Removed peer config: {path}")
        return 0
    print(f"No peer config to remove at {path}")
    return 1


# ─── helpers ──────────────────────────────────────────────────────────────

def _is_public_url(url: str) -> bool:
    """Mirror lane_backends._is_cloud_url but avoid importing runtime here."""
    from urllib.parse import urlparse
    import socket
    try:
        parsed = urlparse(url)
    except Exception:
        return True
    host = parsed.hostname or ""
    if not host:
        return False
    try:
        import ipaddress
        addr = ipaddress.ip_address(host)
        return not (
            addr.is_private or addr.is_loopback
            or addr.is_link_local or addr.is_reserved
        )
    except ValueError:
        pass
    if host.lower() in ("localhost", "local.home", "local"):
        return False
    try:
        addrinfos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return True  # fail-safe: treat as public if we can't resolve
    for info in addrinfos:
        ip = info[4][0]
        try:
            import ipaddress
            addr = ipaddress.ip_address(ip)
            if not (
                addr.is_private or addr.is_loopback
                or addr.is_link_local or addr.is_reserved
            ):
                return True
        except ValueError:
            return True
    return False


def _run_peer_test(url: str, key: str) -> int:
    """Delegate to the existing `maxim peer test` implementation."""
    import os
    os.environ["MAXIM_LANE_INFER_REMOTE_API_KEY"] = key
    from maxim.doctor.cli import run_peer_subcommand
    return run_peer_subcommand(["test", url])


__all__ = ["run_peer_connect_subcommand"]
