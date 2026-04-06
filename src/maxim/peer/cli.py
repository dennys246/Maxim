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
    if action == "update":
        return _cmd_update(list(argv[1:]))
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
    print("  update [url]     Pull + install on leader (--dry-run to preview)")


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
    # API keys are used as HTTP Bearer tokens — headers must be latin-1 encodable.
    # Catches accidental paste of decorative output (e.g. "✓ key: sk-..." lines).
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        print(
            "✗ API key contains non-ASCII characters — likely pasted from\n"
            "  decorative output (e.g. a line starting with ✓). Paste only the\n"
            "  key string itself, or re-run `maxim tunnel key export` on the\n"
            "  leader and copy the value after `=`.",
            file=sys.stderr,
        )
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


def _cmd_update(argv: list[str]) -> int:
    """Trigger git pull + pip install on the leader via /v1/admin/update."""
    import json
    import urllib.error
    import urllib.request

    # Determine URL: from arg, or from peer config
    url: str | None = None
    key: str | None = None
    branch = "main"
    dry_run = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--branch":
            i += 1
            branch = argv[i] if i < len(argv) else "main"
        elif a in ("--dry-run", "--preview"):
            dry_run = True
        elif a.startswith("http"):
            url = a
        i += 1

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL: maxim peer update <url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    # Strip /v1 suffix, build admin endpoint
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/admin/update"

    # Build request
    body = json.dumps({"branch": branch, "dry_run": dry_run}).encode()
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            # Cloudflare Bot Fight Mode blocks Python's default User-Agent.
            # Use a neutral UA to avoid error 1010.
            "User-Agent": "maxim-peer/1.0",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    print(f"{'Previewing' if dry_run else 'Updating'} leader ({base})...")
    print(f"  branch: {branch}")
    print()

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            data = {"error": str(e)}
        if e.code == 403:
            print("Remote update is disabled on the leader.", file=sys.stderr)
            print("  Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.", file=sys.stderr)
            return 1
        if e.code == 409:
            print("Leader has dirty working tree:", file=sys.stderr)
            for f in data.get("dirty_files", []):
                print(f"  {f}", file=sys.stderr)
            return 1
        print(f"Update failed ({e.code}): {data.get('error', str(e))}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1

    status = data.get("status", "unknown")

    if status == "up_to_date":
        print("Already up to date.")
        return 0

    if status == "preview":
        commits = data.get("pending_commits", [])
        print(f"{len(commits)} pending commit(s):")
        for c in commits:
            print(f"  {c}")
        print()
        print("Run without --dry-run to apply:")
        print("  maxim peer update")
        return 0

    if status == "updated":
        commits = data.get("commits_applied", [])
        print(f"Updated! {len(commits)} commit(s) applied:")
        for c in commits:
            print(f"  {c}")
        print()
        print("Restart maxim on the leader to load new code.")
        return 0

    print(f"Unexpected status: {status}")
    print(json.dumps(data, indent=2))
    return 1


__all__ = ["run_peer_connect_subcommand"]
