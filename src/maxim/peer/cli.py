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


def _clear_probe_cache(url: str | None = None) -> None:
    """Drop the P6 remote-probe cache so the next startup re-probes.

    Called from peer commands that change the leader's reachability:
    ``connect`` (new URL), ``forget`` (URL gone), ``restart`` (the leader
    is transiently unavailable during the restart), ``update`` (the
    leader process is replaced), ``llm`` (the leader's model is being
    swapped, server is unreachable for 30-90s).

    Best-effort: silently swallows any error so a probe-cache failure
    can never block a peer command.
    """
    try:
        from maxim.runtime import probe_cache

        if url:
            probe_cache.clear_cache_for_url(url)
        else:
            probe_cache.clear_cache()
    except Exception:
        pass


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
    if action == "key":
        return _cmd_key(list(argv[1:]))
    if action == "forget":
        return _cmd_forget()
    if action == "update":
        return _cmd_update(list(argv[1:]))
    if action == "restart":
        return _cmd_restart(list(argv[1:]))
    if action == "llm":
        return _cmd_llm(list(argv[1:]))
    if action == "version":
        return _cmd_version(list(argv[1:]))
    if action == "logs":
        return _cmd_logs(list(argv[1:]))
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
    print("  key              Print the stored API key (for export/scripting)")
    print("  key set <key>    Update the stored API key without re-running connect")
    print("  forget           Remove stored peer config")
    print("  test <url>       Verify a leader URL is reachable + authenticated")
    print("  update [url]     Pull + install on leader (--dry-run, --force)")
    print("  restart [url]    Soft-restart maxim on leader (reloads code)")
    print("  llm <model>     Hot-swap the LLM model on the leader")
    print("  version [url]    Show maxim version on leader (and local)")
    print("  logs [url]       Tail live logs from leader (-f to follow)")


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

    # Peer connections are to your own infrastructure (leader behind a
    # tunnel). Default to is_cloud=False so the cloud lane gate doesn't
    # block inference. Use --cloud for actual cloud providers (Anthropic, etc).
    is_cloud = "--cloud" in argv

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
    # New URL: drop any cached probe outcomes so the next startup re-probes
    # the freshly-configured leader instead of trusting a stale entry from
    # whatever was previously connected.
    _clear_probe_cache()
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
        _clear_probe_cache()
        print(f"✓ Removed peer config: {path}")
        return 0
    print(f"No peer config to remove at {path}")
    return 1


# ─── key ─────────────────────────────────────────────────────────────────


def _cmd_key(argv: list[str]) -> int:
    """Print or update the stored API key.

    maxim peer key          — print the raw key (for piping/export)
    maxim peer key set KEY  — update the key in-place (validates, preserves 0600)
    """
    if argv and argv[0] == "set":
        return _cmd_key_set(argv[1:])

    # Default: print the raw key
    cfg = read_peer_config()
    if cfg is None:
        print("No peer config found.", file=sys.stderr)
        print("Run: maxim peer connect <url>", file=sys.stderr)
        return 1
    # Print raw key only — safe for `export KEY=$(maxim peer key)`
    print(cfg.api_key)
    return 0


def _cmd_key_set(argv: list[str]) -> int:
    """Update the API key in the existing peer config."""
    cfg = read_peer_config()
    if cfg is None:
        print("No peer config found — run `maxim peer connect <url>` first.", file=sys.stderr)
        return 1

    # Get key from argument or prompt
    if argv and argv[0] not in ("-h", "--help"):
        key = argv[0].strip()
    else:
        if argv and argv[0] in ("-h", "--help"):
            print("Usage: maxim peer key set <api-key>")
            print("       maxim peer key set          (prompts for key)")
            return 0
        try:
            key = getpass.getpass("Paste the new API key (hidden input): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1

    if not key:
        print("✗ API key required.", file=sys.stderr)
        return 1

    # Validate: Bearer tokens must be latin-1 encodable (HTTP header constraint)
    try:
        key.encode("latin-1")
    except UnicodeEncodeError:
        print(
            "✗ API key contains non-ASCII characters — likely pasted from\n"
            "  decorative output. Paste only the raw key string.",
            file=sys.stderr,
        )
        return 1

    # Update config, preserving url/model/is_cloud
    updated = PeerConfig(url=cfg.url, api_key=key, model=cfg.model, is_cloud=cfg.is_cloud)
    path = write_peer_config(updated)
    print(f"✓ API key updated in {path}")
    print(f"  key: {truncate_key(key)}")
    return 0


# ─── helpers ──────────────────────────────────────────────────────────────


def _request_with_retry(
    url: str,
    *,
    method: str = "GET",
    headers: dict | None = None,
    body: bytes | None = None,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 1.0,
) -> bytes | None:
    """Make an HTTP request with exponential backoff on transient failures.

    Returns response body on success, None on failure after all retries.
    """
    import time
    import urllib.error
    import urllib.request

    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, method=method, data=body)
        req.add_header("User-Agent", "maxim-peer/1.0")
        if headers:
            for k, v in headers.items():
                req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503):
                # Transient — retry with backoff
                if attempt < max_retries:
                    wait = base_backoff * (2**attempt)
                    time.sleep(wait)
                    continue
            return None  # Permanent HTTP error
        except (urllib.error.URLError, OSError):
            # Connection refused, timeout, etc.
            if attempt < max_retries:
                wait = base_backoff * (2**attempt)
                time.sleep(wait)
                continue
            return None
    return None


def _check_proxy_ping(base_url: str, key: str | None = None, *, verbose: bool = False) -> dict | None:
    """Probe /v1/debug/ping to verify the tunnel reaches LeaderProxy.

    Returns the ping response dict, or None if the probe fails.
    """
    import json
    import urllib.error
    import urllib.request

    endpoint = f"{base_url}/v1/debug/ping"
    req = urllib.request.Request(
        endpoint,
        method="GET",
        headers={"User-Agent": "maxim-peer/1.0"},
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if verbose:
            print(f"[{e.code}]", end="", flush=True)
        return None
    except urllib.error.URLError as e:
        if verbose:
            reason = str(getattr(e, "reason", e))[:30]
            print(f"[{reason}]", end="", flush=True)
        return None
    except Exception as e:
        if verbose:
            print(f"[{type(e).__name__}]", end="", flush=True)
        return None


def _print_404_diagnosis(base_url: str) -> None:
    """Print diagnostic hints when an admin/debug endpoint returns 404."""
    print(file=sys.stderr)
    print("  This usually means the tunnel is routing to llama-cpp-server", file=sys.stderr)
    print("  (port 8100) instead of LeaderProxy (port 8099).", file=sys.stderr)
    print(file=sys.stderr)
    # Try to confirm by pinging the proxy
    ping = _check_proxy_ping(base_url)
    if ping and ping.get("service") == "LeaderProxy":
        print("  However, /v1/debug/ping DID reach LeaderProxy.", file=sys.stderr)
        print("  The endpoint may not exist in this version. Try updating the leader.", file=sys.stderr)
    else:
        print("  Quick checks on the leader machine:", file=sys.stderr)
        print("    curl -s http://localhost:8099/v1/debug/ping", file=sys.stderr)
        print("    grep service ~/.cloudflared/config.yml", file=sys.stderr)
        print("      (should show: service: http://localhost:8099)", file=sys.stderr)
        print(file=sys.stderr)
        print("  See: docs/troubleshooting/leader_proxy_debug.md", file=sys.stderr)


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
        return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved)
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
            if not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved):
                return True
        except ValueError:
            return True
    return False


def _run_peer_test(url: str, key: str) -> int:
    """Delegate to the existing `maxim peer test` implementation."""
    import os

    os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] = key
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
    force = False

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--branch":
            i += 1
            branch = argv[i] if i < len(argv) else "main"
        elif a in ("--dry-run", "--preview"):
            dry_run = True
        elif a in ("--force", "-f"):
            force = True
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
    body = json.dumps({"branch": branch, "dry_run": dry_run, "force": force}).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            # Cloudflare Bot Fight Mode blocks Python's default User-Agent.
            # Use a neutral UA to avoid error 1010.
            "User-Agent": "maxim-peer/1.0",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    # An update replaces the leader process. Drop the cached probe entry
    # so the next startup re-probes the new binary instead of trusting
    # the pre-update "ok".
    if not dry_run:
        _clear_probe_cache(url)

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
        if e.code == 404:
            print("Admin endpoint returned 404.", file=sys.stderr)
            _print_404_diagnosis(base)
            return 1
        if e.code == 409:
            print("Leader has dirty working tree:", file=sys.stderr)
            for f in data.get("dirty_files", []):
                print(f"  {f}", file=sys.stderr)
            hint = data.get("hint")
            if hint:
                print(f"\n  {hint}", file=sys.stderr)
            return 1
        if e.code == 401:
            print("Authentication failed.", file=sys.stderr)
            print("  Check API key matches leader. Run: maxim tunnel key show (on leader)", file=sys.stderr)
            return 1
        print(f"Update failed ({e.code}): {data.get('error', str(e))}", file=sys.stderr)
        if data.get("stderr"):
            print(f"  stderr: {data['stderr']}", file=sys.stderr)
        if data.get("stdout"):
            print(f"  stdout: {data['stdout']}", file=sys.stderr)
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
        print("Restart maxim on the leader to load new code:")
        print("  maxim peer restart")
        return 0

    print(f"Unexpected status: {status}")
    print(json.dumps(data, indent=2))
    return 1


def _cmd_restart(argv: list[str]) -> int:
    """Soft-restart maxim on the leader via /v1/admin/restart."""
    import json
    import urllib.error
    import urllib.request

    url: str | None = None
    key: str | None = None

    for a in argv:
        if a.startswith("http"):
            url = a

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL: maxim peer restart <url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/admin/restart"

    body = json.dumps({"delay_s": 1.5}).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "maxim-peer/1.0",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    # Restart drops the leader briefly. Clear the probe cache so the next
    # startup re-probes instead of believing a now-stale "ok" entry.
    _clear_probe_cache(url)

    print(f"Restarting leader ({base})...")
    print()

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            data = {"error": str(e)}
        if e.code == 403:
            print("Remote restart is disabled on the leader.", file=sys.stderr)
            print("  Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.", file=sys.stderr)
            return 1
        if e.code == 404:
            print("Admin endpoint returned 404.", file=sys.stderr)
            _print_404_diagnosis(base)
            return 1
        if e.code == 401:
            print("Authentication failed.", file=sys.stderr)
            print("  Check API key matches leader. Run: maxim tunnel key show (on leader)", file=sys.stderr)
            return 1
        print(f"Restart failed ({e.code}): {data.get('error', str(e))}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1

    status = data.get("status", "unknown")

    if status == "restarting":
        import time as _time

        uptime = data.get("uptime_s", "?")
        print(f"Leader is restarting (was up {uptime}s).")
        print("  Waiting for leader to come back...", end="", flush=True)

        # Poll until the leader responds again (exponential backoff, max 120s)
        # Two-phase: first wait for proxy to respond, then wait for LLM ready.
        proxy_up = False
        llm_ready = False
        _restart_start = _time.time()
        for attempt in range(15):  # 15 attempts, ~90s total
            wait = min(2.0 * (1.5**attempt), 10.0)  # cap at 10s between polls
            _time.sleep(wait)
            elapsed = int(_time.time() - _restart_start)
            print(f" {elapsed}s", end="", flush=True)
            ping = _check_proxy_ping(base, key, verbose=True)
            if ping is not None:
                proxy_up = True
                if ping.get("llm_ready", False):
                    llm_ready = True
                break  # Proxy responded — move to phase 2 (LLM polling)
        print()
        if llm_ready:
            print("  Leader is back online.")
        elif proxy_up:
            # Proxy responded but LLM not ready yet — keep polling for LLM.
            # Large models (14B+) can take 60-120s to load into VRAM.
            _llm_wait_start = _time.time()
            print("  Proxy is up, waiting for LLM model to load...", end="", flush=True)
            for attempt in range(30):  # 30 x 5s = 150s max — enough for large models
                _time.sleep(5.0)
                elapsed = int(_time.time() - _llm_wait_start)
                print(f" {elapsed}s", end="", flush=True)
                ping = _check_proxy_ping(base, key)
                if ping is not None and ping.get("llm_ready", False):
                    llm_ready = True
                    break
            print()
            if llm_ready:
                elapsed = int(_time.time() - _llm_wait_start)
                print(f"  Leader is back online (LLM ready, {elapsed}s).")
            else:
                print("  Leader proxy is up but LLM model is still loading.")
                print("  Inference requests will queue until the model is ready.")
                print("  Check with: maxim peer llm --status")
        else:
            print("  Leader did not respond within timeout.")
            print("  Check with: maxim peer version")
        return 0

    print(f"Unexpected status: {status}")
    print(json.dumps(data, indent=2))
    return 1


def _cmd_llm_status(argv: list[str]) -> int:
    """Show what LLM model is running on the leader."""
    import json
    import urllib.error
    import urllib.request

    url: str | None = None
    for a in argv:
        if a.startswith("http"):
            url = a

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL.", file=sys.stderr)
            return 1
        url = cfg.url

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    # Query debug/status (no auth required for debug endpoints)
    req = urllib.request.Request(
        f"{base}/v1/debug/status",
        headers={"User-Agent": "maxim-peer/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  Failed to query leader: {e}", file=sys.stderr)
        return 1

    model = data.get("llm_model") or data.get("active_model") or "unknown"
    maxim_uptime = data.get("maxim_uptime_s", data.get("uptime_s", 0))
    llm_uptime = data.get("llm_uptime_s")
    gpu = data.get("gpu") or {}
    gpu_util = gpu.get("utilization", "?")
    gpu_mem = gpu.get("memory_used", "?")
    gpu_total = gpu.get("memory_total", "?")
    lane = data.get("infer_lane") or {}

    print("  Leader status:")
    print(f"    Maxim uptime:  {int(maxim_uptime)}s")
    print(f"    LLM model:     {model}")
    if llm_uptime is not None:
        print(f"    LLM uptime:    {int(llm_uptime)}s")
    else:
        print("    LLM uptime:    n/a")
    if isinstance(gpu_util, (int, float)):
        print(f"    GPU:           {gpu_util}% util, {gpu_mem}/{gpu_total} MB VRAM")
    if lane:
        reqs = lane.get("total_requests", 0)
        in_flight = lane.get("in_flight", 0)
        latency = lane.get("avg_latency_ms")
        latency_str = f"{latency}ms" if latency is not None else "n/a"
        print(f"    Infer lane:    {reqs} requests, {in_flight} in-flight, avg {latency_str}")
    return 0


def _cmd_llm(argv: list[str]) -> int:
    """Hot-swap the LLM model on the leader via /v1/admin/llm-swap."""
    import json
    import time
    import urllib.error
    import urllib.request

    if not argv or argv[0] in ("-h", "--help"):
        print("Usage: maxim peer llm <model> [url]")
        print("       maxim peer llm --status [url]")
        print()
        print("Hot-swap the LLM running on the leader's llama-cpp-server.")
        print("Stops the current model, loads the new one, and health-checks.")
        print()
        print("Options:")
        print("  --status         Show what model is currently running on the leader")
        print()
        print("Examples:")
        print("  maxim peer llm qwen2.5-14b")
        print("  maxim peer llm mistral-7b")
        print("  maxim peer llm --status")
        return 0

    # Check for --status flag
    if "--status" in argv:
        return _cmd_llm_status([a for a in argv if a != "--status"])

    model = argv[0]
    url: str | None = None
    key: str | None = None

    for a in argv[1:]:
        if a.startswith("http"):
            url = a

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL: maxim peer llm <model> <url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/admin/llm-swap"

    body = json.dumps({"model": model}).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "maxim-peer/1.0",
        },
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    # The leader is unreachable for 30-90 s during a swap; nuke the cache
    # entry so the next probe sees the new model state, not the old "ok".
    _clear_probe_cache(url)

    print(f"  Swapping leader to {model}...")
    t0 = time.time()

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read())
        except Exception:
            data = {"error": str(e)}
        error_msg = data.get("error", str(e))
        hint = data.get("hint", "")
        print(f"  Error ({e.code}): {error_msg}", file=sys.stderr)
        if hint:
            print(f"  Hint: {hint}", file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"  Connection failed: {e.reason}", file=sys.stderr)
        return 1

    elapsed = round(time.time() - t0, 1)
    status = data.get("status", "")

    if status == "already_running":
        print(f"  Already running {data.get('model', model)} — no swap needed.")
        return 0

    if status == "swapped":
        prev = data.get("previous_model", "?")
        new_model = data.get("model", model)
        startup = data.get("startup_s", elapsed)
        print(f"  Done: {prev} → {new_model} ({startup}s startup, {elapsed}s total)")
        return 0

    print(f"  Unexpected response: {json.dumps(data, indent=2)}")
    return 1


def _cmd_version(argv: list[str]) -> int:
    """Show local and leader maxim version."""
    import json
    import urllib.error
    import urllib.request

    from maxim import get_version_info

    local = get_version_info()
    print(f"Local:  v{local['version']}", end="")
    if local.get("git_hash"):
        print(f" ({local['git_hash']})", end="")
    print()
    if local.get("git_message"):
        print(f"        {local['git_message']}")

    # Query leader
    url: str | None = None
    key: str | None = None

    for a in argv:
        if a.startswith("http"):
            url = a

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            return 0  # No leader configured, just show local
        url = cfg.url
        key = cfg.api_key

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/debug/version"

    req = urllib.request.Request(
        endpoint,
        method="GET",
        headers={"User-Agent": "maxim-peer/1.0"},
    )
    if key:
        req.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
    except Exception as e:
        print(f"Leader: (unreachable: {e})")
        return 1

    print(f"Leader: v{data.get('version', '?')}", end="")
    if data.get("git_hash"):
        print(f" ({data['git_hash']})", end="")
    print()
    if data.get("git_message"):
        print(f"        {data['git_message']}")

    # Highlight version mismatch
    if local.get("git_hash") and data.get("git_hash"):
        if local["git_hash"] != data["git_hash"]:
            print()
            print("  Version mismatch! Run: maxim peer update && maxim peer restart")

    return 0


def _cmd_logs(argv: list[str]) -> int:
    """Tail logs from the leader via /v1/debug/logs (polling)."""
    import json
    import time
    import urllib.error
    import urllib.request
    from datetime import datetime

    url: str | None = None
    key: str | None = None
    follow = False
    poll_interval = 2.0
    limit = 50

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--follow"):
            follow = True
        elif a == "--interval" and i + 1 < len(argv):
            i += 1
            poll_interval = max(0.5, float(argv[i]))
        elif a == "--limit" and i + 1 < len(argv):
            i += 1
            limit = int(argv[i])
        elif a.startswith("http"):
            url = a
        i += 1

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL: maxim peer logs <url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    since_seq = -1

    def _fetch(seq: int, n: int) -> dict | None:
        endpoint = f"{base}/v1/debug/logs?since_seq={seq}&limit={n}"
        req = urllib.request.Request(
            endpoint,
            method="GET",
            headers={"User-Agent": "maxim-peer/1.0"},
        )
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            print(f"  Error: HTTP {e.code}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"  Connection error: {e}", file=sys.stderr)
            return None

    def _print_entry(entry: dict) -> None:
        ts = datetime.fromtimestamp(entry["ts"]).strftime("%H:%M:%S.%f")[:-3]
        level = entry.get("level", "?")
        logger_name = entry.get("logger", "")
        # Shorten logger name: maxim.runtime.foo → runtime.foo
        if logger_name.startswith("maxim."):
            logger_name = logger_name[6:]
        msg = entry.get("message", "")

        # Color by level
        level_colors = {"ERROR": "\033[31m", "WARNING": "\033[33m", "INFO": "\033[36m", "DEBUG": "\033[90m"}
        color = level_colors.get(level, "")
        reset = "\033[0m" if color else ""

        print(f"{color}{ts} [{level:<7}] [{logger_name}] {msg}{reset}")

    # Initial fetch
    data = _fetch(since_seq, limit)
    if data is None:
        return 1

    for entry in data.get("entries", []):
        _print_entry(entry)
    since_seq = data.get("latest_seq", since_seq)

    if not follow:
        return 0

    # Follow mode — poll until Ctrl+C (with backoff on failure)
    print(f"\n--- Following (poll every {poll_interval}s, Ctrl+C to stop) ---\n")
    consecutive_failures = 0
    max_backoff = 30.0
    try:
        while True:
            if consecutive_failures > 0:
                backoff = min(poll_interval * (2**consecutive_failures), max_backoff)
                time.sleep(backoff)
            else:
                time.sleep(poll_interval)
            data = _fetch(since_seq, 200)
            if data is None:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print("  (leader unreachable — backing off)")
                continue
            if consecutive_failures > 0:
                print("  (reconnected)")
            consecutive_failures = 0
            for entry in data.get("entries", []):
                _print_entry(entry)
            since_seq = data.get("latest_seq", since_seq)
    except KeyboardInterrupt:
        print("\n  Stopped.")
        return 0


__all__ = ["run_peer_connect_subcommand"]
