"""Shared HTTP cores for ``maxim peer`` admin verbs.

Plan 4 C3.5 + C3.6. Mirrors the :mod:`maxim.peer.install_core` pattern:
each function is the **single source of truth** for one admin endpoint's
wire-level shape (endpoint URL, headers, JSON body, timeout policy,
response classification, exit code contract).

Two callers per function:

- :mod:`maxim.peer.cli` — resolves URL from a positional arg or
  ``peer.yml``, then calls the core.
- :mod:`maxim.peer.mesh_cli` — resolves URL + key from
  ``mesh.yml::nodes`` by name, drains the node, calls the core,
  then resumes.

CI greps in ``.github/workflows/test.yml`` enforce that:

- ``/v1/admin/update`` appears only in this file + its test file +
  ``leader_proxy.py`` (server handler)
- ``/v1/admin/restart`` — same allow-list
- ``/v1/admin/llm-swap`` — same allow-list

Do NOT inline copies of the endpoint strings elsewhere.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def update_on_target(
    url: str,
    key: str | None,
    *,
    branch: str = "main",
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """POST an update request to a target URL and report results.

    Plan 4 C3.5: extracted from :func:`maxim.peer.cli._cmd_update`.
    Single source of truth for the ``/v1/admin/update`` wire shape.

    ``url`` may include a trailing ``/v1`` — it gets stripped for the
    endpoint composition.

    Exit code contract:

    - ``0`` — update succeeded (or already up-to-date, or dry-run preview)
    - ``1`` — update failed (HTTP error, dirty tree, auth, etc.)
    """
    import json

    from maxim.utils import http as _http

    # Lazy import to avoid circular dependency at module-load time.
    from maxim.peer.cli import _clear_probe_cache

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    endpoint = f"{base}/v1/admin/update"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # An update replaces the leader process. Drop the cached probe entry
    # so the next startup re-probes the new binary instead of trusting
    # the pre-update "ok".
    if not dry_run:
        _clear_probe_cache(url)

    print(f"{'Previewing' if dry_run else 'Updating'} leader ({base})...")
    print(f"  branch: {branch}")
    print()

    try:
        resp = _http.fetch_url(
            endpoint,
            method="POST",
            headers=headers,
            json={"branch": branch, "dry_run": dry_run, "force": force},
            timeout=_http.TimeoutPolicy(connect_s=5.0, read_s=180.0, total_s=185.0),
        )
        data = resp.json()
    except _http.HTTPError as e:
        data = {}
        if e.response is not None:
            try:
                data = e.response.json()
            except Exception:
                data = {"error": str(e)}
        if e.status == 403:
            print("Remote update is disabled on the leader.", file=sys.stderr)
            print("  Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.", file=sys.stderr)
            return 1
        if e.status == 404:
            print("Admin endpoint returned 404.", file=sys.stderr)
            _print_404_diagnosis(base, key)
            return 1
        if e.status == 409:
            print("Leader has dirty working tree:", file=sys.stderr)
            for f in data.get("dirty_files", []):
                print(f"  {f}", file=sys.stderr)
            hint = data.get("hint")
            if hint:
                print(f"\n  {hint}", file=sys.stderr)
            return 1
        if e.status == 401:
            print("Authentication failed.", file=sys.stderr)
            print(
                "  Check API key matches leader. Run: maxim tunnel key show (on leader)",
                file=sys.stderr,
            )
            return 1
        print(f"Update failed ({e.status}): {data.get('error', e.fix_hint)}", file=sys.stderr)
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


def restart_on_target(url: str, key: str | None) -> int:
    """POST a restart request and poll for recovery.

    Plan 4 C3.5: extracted from :func:`maxim.peer.cli._cmd_restart`.
    Single source of truth for the ``/v1/admin/restart`` wire shape.

    Two-phase recovery poll:
    1. Wait for LeaderProxy to respond to /v1/debug/ping (~90s)
    2. Wait for LLM model to load (~150s for large models)

    Exit code contract:

    - ``0`` — restart initiated (leader responded or timed out waiting)
    - ``1`` — restart failed (HTTP error, auth, etc.)
    """
    import json
    import time

    from maxim.utils import http as _http

    from maxim.peer.cli import _check_proxy_ping, _clear_probe_cache

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/admin/restart"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # Restart drops the leader briefly. Clear the probe cache so the next
    # startup re-probes instead of believing a now-stale "ok" entry.
    _clear_probe_cache(url)

    print(f"Restarting leader ({base})...")
    print()

    try:
        resp = _http.fetch_url(
            endpoint,
            method="POST",
            headers=headers,
            json={"delay_s": 1.5},
            timeout=_http.TimeoutPolicy(connect_s=3.0, read_s=30.0, total_s=33.0),
        )
        data = resp.json()
    except _http.HTTPError as e:
        data = {}
        if e.response is not None:
            try:
                data = e.response.json()
            except Exception:
                data = {"error": str(e)}
        if e.status == 403:
            print("Remote restart is disabled on the leader.", file=sys.stderr)
            print("  Set MAXIM_ALLOW_REMOTE_UPDATE=1 on the leader process.", file=sys.stderr)
            return 1
        if e.status == 404:
            print("Admin endpoint returned 404.", file=sys.stderr)
            _print_404_diagnosis(base, key)
            return 1
        if e.status == 401:
            print("Authentication failed.", file=sys.stderr)
            print(
                "  Check API key matches leader. Run: maxim tunnel key show (on leader)",
                file=sys.stderr,
            )
            return 1
        print(f"Restart failed ({e.status}): {data.get('error', e.fix_hint)}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Connection failed: {e}", file=sys.stderr)
        return 1

    status = data.get("status", "unknown")

    if status == "restarting":
        uptime = data.get("uptime_s", "?")
        print(f"Leader is restarting (was up {uptime}s).")
        print("  Waiting for leader to come back...", end="", flush=True)

        # Poll until the leader responds again (exponential backoff, max 120s)
        # Two-phase: first wait for proxy to respond, then wait for LLM ready.
        proxy_up = False
        llm_ready = False
        _restart_start = time.time()
        for attempt in range(15):  # 15 attempts, ~90s total
            wait = min(2.0 * (1.5**attempt), 10.0)  # cap at 10s between polls
            time.sleep(wait)
            elapsed = int(time.time() - _restart_start)
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
            _llm_wait_start = time.time()
            print("  Proxy is up, waiting for LLM model to load...", end="", flush=True)
            for attempt in range(30):  # 30 x 5s = 150s max — enough for large models
                time.sleep(5.0)
                elapsed = int(time.time() - _llm_wait_start)
                print(f" {elapsed}s", end="", flush=True)
                ping = _check_proxy_ping(base, key)
                if ping is not None and ping.get("llm_ready", False):
                    llm_ready = True
                    break
            print()
            if llm_ready:
                elapsed = int(time.time() - _llm_wait_start)
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


def llm_swap_on_target(url: str, key: str | None, model: str) -> int:
    """POST an LLM swap request to a target URL and report results.

    Plan 4 C3.6: extracted from :func:`maxim.peer.cli._cmd_llm`.
    Single source of truth for the ``/v1/admin/llm-swap`` wire shape.

    Exit code contract:

    - ``0`` — swap succeeded (or already running the requested model)
    - ``1`` — swap failed (HTTP error, auth, etc.)
    """
    import json
    import time

    from maxim.utils import http as _http

    from maxim.peer.cli import _clear_probe_cache

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    endpoint = f"{base}/v1/admin/llm-swap"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # The leader is unreachable for 30-90 s during a swap; nuke the cache
    # entry so the next probe sees the new model state, not the old "ok".
    _clear_probe_cache(url)

    print(f"  Swapping leader to {model}...")
    t0 = time.time()

    try:
        resp = _http.fetch_url(
            endpoint,
            method="POST",
            headers=headers,
            json={"model": model},
            timeout=_http.TimeoutPolicy(connect_s=5.0, read_s=180.0, total_s=185.0),
        )
        data = resp.json()
    except _http.HTTPError as e:
        data = {}
        if e.response is not None:
            try:
                data = e.response.json()
            except Exception:
                data = {"error": str(e)}
        error_msg = data.get("error", e.fix_hint)
        hint = data.get("hint", "")
        print(f"  Error ({e.status}): {error_msg}", file=sys.stderr)
        if hint:
            print(f"  Hint: {hint}", file=sys.stderr)
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


def _print_404_diagnosis(base_url: str, key: str | None = None) -> None:
    """Print diagnostic hints when an admin/debug endpoint returns 404.

    Shared helper for update + restart 404 errors. Delegates to
    :func:`maxim.peer.cli._check_proxy_ping` for the proxy probe.
    """
    from maxim.peer.cli import _check_proxy_ping

    print(file=sys.stderr)
    print("  This usually means the tunnel is routing to llama-cpp-server", file=sys.stderr)
    print("  (port 8100) instead of LeaderProxy (port 8099).", file=sys.stderr)
    print(file=sys.stderr)
    ping = _check_proxy_ping(base_url, key)
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


__all__ = [
    "llm_swap_on_target",
    "restart_on_target",
    "update_on_target",
]
