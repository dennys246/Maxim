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
    """Dispatch peer config subcommands: connect / show / forget.

    **Plan 4 C2:** Role detection runs at the top of this function
    via :func:`maxim.runtime.role.detect_and_apply_role` so that drain
    / resume verbs (and any other verb that resolves a role-scoped
    ``~/.maxim/util/*.{role}.txt`` path) see the correct
    ``MAXIM_ROLE`` env var.

    **J1 guard (C2 pre-merge review):** ``cli.py::main`` already calls
    ``detect_and_apply_role(raw_argv)`` once before dispatching to the
    peer subcommand. Calling it a second time here would:

    - Re-emit ``role_detected`` with ``role_source=env_var`` instead
      of the original ``default`` / ``peer_yml`` / ``mesh_yml`` —
      confusing log noise for operators.
    - Re-fire ``role_divergence`` WARNING if leader_mode disagrees.
    - Re-run ``migrate_persisted_model_file`` (idempotent but wasted).

    So we guard on ``MAXIM_ROLE`` already being set. If it's present,
    we know the full detection pipeline ran upstream in ``main`` and
    skip the duplicate call. If it's absent (shouldn't happen post-C2
    on the normal entry path, but this is defensive — e.g. a direct
    import of ``run_peer_connect_subcommand`` from a test or script),
    we run the full detection so drain state paths resolve correctly.
    """
    import os as _os

    if not _os.environ.get("MAXIM_ROLE"):
        try:
            from maxim.runtime.role import detect_and_apply_role

            detect_and_apply_role(list(argv))
        except Exception as e:  # pragma: no cover - defensive
            import logging as _logging

            _logging.getLogger(__name__).debug("role detection skipped: %s", e)

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
    if action == "install":
        return _cmd_install(list(argv[1:]))
    if action == "deps":
        return _cmd_deps(list(argv[1:]))
    if action == "list-nodes":
        from maxim.peer.mesh_cli import run_list_nodes

        return run_list_nodes(list(argv[1:]))
    if action == "list-drained":
        from maxim.peer.mesh_cli import run_list_drained

        return run_list_drained(list(argv[1:]))
    if action == "init-mesh":
        from maxim.peer.mesh_setup import run_init_mesh

        return run_init_mesh(list(argv[1:]))
    if action == "add-node":
        from maxim.peer.mesh_setup import run_add_node

        return run_add_node(list(argv[1:]))
    if action == "remove-node":
        from maxim.peer.mesh_setup import run_remove_node

        return run_remove_node(list(argv[1:]))
    if action == "--node":
        from maxim.peer.mesh_cli import run_node_subcommand

        return run_node_subcommand(list(argv))
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
    print("  install <extras> Install optional extras on leader (e.g., semantic,llm-torch)")
    print("  deps [url]       Show installed packages on leader")
    print("  list-nodes       List mesh nodes + live status (Plan 4 C1)")
    print("  list-drained     List currently drained mesh nodes (Plan 4 C2)")
    print("  init-mesh        Synthesize mesh.yml from peer.yml (Plan 4 C3.1)")
    print("  add-node <name>  Add a node to mesh.yml (Plan 4 C3.2)")
    print("                   Required: --url <url>; optional: --role peer|leader, --force")
    print("  remove-node <n>  Remove a node from mesh.yml (Plan 4 C3.2)")
    print("                   Auto-clears drain state for the removed node")
    print("  --node <n> <v>   Per-node verbs: status|health|drain|resume|install|")
    print("                   update|restart|llm")


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

    from maxim.utils import http as _http

    for attempt in range(max_retries + 1):
        try:
            resp = _http.fetch_url(
                url,
                method=method,
                headers=headers or {},
                content=body,
                timeout=_http.TimeoutPolicy(
                    connect_s=min(timeout, 3.0),
                    read_s=timeout,
                    total_s=timeout + 1.0,
                ),
            )
            return resp.content
        except _http.HTTPRateLimited:
            if attempt < max_retries:
                time.sleep(base_backoff * (2**attempt))
                continue
            return None
        except _http.HTTPServerError as e:
            if e.status in (502, 503) and attempt < max_retries:
                time.sleep(base_backoff * (2**attempt))
                continue
            return None
        except (_http.HTTPError, OSError):
            if attempt < max_retries:
                time.sleep(base_backoff * (2**attempt))
                continue
            return None
    return None


def _check_proxy_ping(base_url: str, key: str | None = None, *, verbose: bool = False) -> dict | None:
    """Probe /v1/debug/ping to verify the tunnel reaches LeaderProxy.

    Returns the ping response dict, or None if the probe fails.
    """
    from maxim.utils import http as _http

    endpoint = f"{base_url}/v1/debug/ping"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        resp = _http.fetch_url(
            endpoint,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=5.0, total_s=6.0),
        )
        return resp.json()
    except _http.HTTPError as e:
        if verbose:
            tag = str(e.status) if e.status else type(e).__name__
            print(f"[{tag}]", end="", flush=True)
        return None
    except Exception as e:
        if verbose:
            print(f"[{type(e).__name__}]", end="", flush=True)
        return None


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
    """Update the leader via pip (default) or git (--dev).

    Plan 4 C3.5 + peer_update_pip_mode plan: HTTP wire-level logic lives
    in :func:`maxim.peer.admin_core.update_on_target`. This function
    handles arg parsing and URL/key resolution only.

    Usage::

        maxim peer update                     # auto-detect (pip or git)
        maxim peer update --dry-run           # preview only
        maxim peer update --version 0.3.1     # pin PyPI version
        maxim peer update --dev               # force git mode (origin/main)
        maxim peer update --dev feat/foo      # force git mode (origin/feat/foo)
        maxim peer update --dev --force       # stash dirty tree first
    """
    from maxim.peer.admin_core import update_on_target

    url: str | None = None
    key: str | None = None
    branch = "main"
    dry_run = False
    force = False
    mode = "auto"
    version: str | None = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--branch":
            i += 1
            branch = argv[i] if i < len(argv) else "main"
        elif a == "--dev":
            mode = "dev"
            # Next non-flag arg is the branch
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 1
                branch = argv[i]
        elif a == "--version":
            i += 1
            version = argv[i] if i < len(argv) else None
        elif a in ("--dry-run", "--preview"):
            dry_run = True
        elif a in ("--force", "-f"):
            force = True
        elif a.startswith("http"):
            url = a
        i += 1

    # ── Client-side validation ───────────────────────────────────────
    if version and mode == "dev":
        print("--version and --dev are mutually exclusive.", file=sys.stderr)
        print("  Use --version for pip updates, --dev <branch> for git.", file=sys.stderr)
        return 1

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Provide a URL: maxim peer update <url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    return update_on_target(
        url,
        key,
        branch=branch,
        dry_run=dry_run,
        force=force,
        mode=mode,
        version=version,
    )


def _cmd_restart(argv: list[str]) -> int:
    """Soft-restart maxim on the leader.

    Plan 4 C3.5: HTTP wire-level logic moved to
    :func:`maxim.peer.admin_core.restart_on_target`. This function
    handles arg parsing and URL/key resolution only.
    """
    from maxim.peer.admin_core import restart_on_target

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

    return restart_on_target(url, key)


def _cmd_llm_status(argv: list[str]) -> int:
    """Show what LLM model is running on the leader."""
    from maxim.utils import http as _http

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
    try:
        resp = _http.fetch_url(
            f"{base}/v1/debug/status",
            method="GET",
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        data = resp.json()
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
    """Hot-swap the LLM model on the leader.

    Plan 4 C3.6: HTTP wire-level logic moved to
    :func:`maxim.peer.admin_core.llm_swap_on_target`. This function
    handles arg parsing, --status subcommand, and URL/key resolution.
    """
    from maxim.peer.admin_core import llm_swap_on_target

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

    return llm_swap_on_target(url, key, model)


def _cmd_version(argv: list[str]) -> int:
    """Show local and leader maxim version."""
    from maxim import get_version_info
    from maxim.utils import http as _http

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

    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = _http.fetch_url(
            endpoint,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
        )
        data = resp.json()
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
    import time
    from datetime import datetime

    from maxim.utils import http as _http

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
        headers: dict[str, str] = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = _http.fetch_url(
                endpoint,
                method="GET",
                headers=headers,
                timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=10.0, total_s=12.0),
            )
            return resp.json()
        except _http.HTTPError as e:
            print(f"  Error: HTTP {e.status}", file=sys.stderr)
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


# ─── install ─────────────────────────────────────────────────────────────

# Plan 4 C3.3 fold: the shared install core (KNOWN_EXTRAS,
# classify_install_tokens, install_on_target) lives in
# :mod:`maxim.peer.install_core` so both this verb and the mesh-aware
# ``maxim peer --node <name> install`` verb can import from the same
# leaf module without inverting the cli → mesh_cli coupling direction.
# A CI grep in ``.github/workflows/test.yml`` locks the admin-install
# endpoint string to the core module — see install_core's module
# docstring for the full allow-list rationale.
from maxim.peer.install_core import (
    KNOWN_EXTRAS,
    _looks_like_url,
    classify_install_tokens,
    install_on_target,
)


def _cmd_install(argv: list[str]) -> int:
    """``maxim peer install <extras_or_packages> [url]`` — install on
    the connected leader (or the URL passed positionally).

    Usage:
        maxim peer install semantic
        maxim peer install semantic,llm-torch
        maxim peer install sentence-transformers   # raw pip package
        maxim peer install semantic https://other-leader.example.com/v1

    Plan 4 C3.3: this verb is the no-mesh.yml fallback for the
    positional-URL path. The mesh-aware form is
    ``maxim peer --node <name> install <extras>``, which composes
    drain → install → resume around the same
    :func:`maxim.peer.install_core.install_on_target` core.
    """
    url: str | None = None
    raw_tokens: list[str] = []

    for arg in argv:
        if _looks_like_url(arg):
            # Fold I3: was ``arg.startswith("http")`` which
            # false-positives on httpx / http-client / httpie.
            # ``_looks_like_url`` requires ``"://"``.
            url = arg
        elif arg.startswith("--"):
            pass  # future flags
        else:
            # Could be comma-separated extras or raw package names
            raw_tokens.extend(arg.split(","))

    if not raw_tokens:
        print("Usage: maxim peer install <extras_or_packages>", file=sys.stderr)
        print("  extras: " + ", ".join(sorted(KNOWN_EXTRAS)), file=sys.stderr)
        print("  Example: maxim peer install semantic", file=sys.stderr)
        print("  Example: maxim peer install sentence-transformers", file=sys.stderr)
        return 2

    key: str | None = None
    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Run: maxim peer connect <leader-url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    extras, packages = classify_install_tokens(raw_tokens)
    return install_on_target(url, key, extras, packages)


# ─── deps ────────────────────────────────────────────────────────────────


def _cmd_deps(argv: list[str]) -> int:
    """Show installed packages on leader via /v1/debug/deps."""
    from maxim.utils import http as _http

    url: str | None = None
    key: str | None = None

    for arg in argv:
        if arg.startswith("http"):
            url = arg

    if url is None:
        cfg = read_peer_config()
        if cfg is None:
            print("No peer config. Run: maxim peer connect <leader-url>", file=sys.stderr)
            return 1
        url = cfg.url
        key = cfg.api_key

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    endpoint = f"{base}/v1/debug/deps"
    headers: dict[str, str] = {}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        resp = _http.fetch_url(
            endpoint,
            method="GET",
            headers=headers,
            timeout=_http.TimeoutPolicy(connect_s=3.0, read_s=30.0, total_s=33.0),
        )
        data = resp.json()
    except _http.HTTPError as e:
        if e.status == 404:
            print(
                "Leader does not support deps endpoint (update leader first).",
                file=sys.stderr,
            )
        elif e.status == 401:
            print("Authentication failed.", file=sys.stderr)
        else:
            print(f"Failed (HTTP {e.status})", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed: {e}", file=sys.stderr)
        return 1

    packages = data.get("packages", {})
    extras = data.get("extras", {})

    if extras:
        print("Installed extras:")
        for name, installed in sorted(extras.items()):
            status = "installed" if installed else "not installed"
            print(f"  [{name}] {status}")
        print()

    if packages:
        print("Key packages:")
        for name, version in sorted(packages.items()):
            print(f"  {name}=={version}")

    return 0


__all__ = ["run_peer_connect_subcommand"]
