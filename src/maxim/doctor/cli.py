"""`maxim doctor` + `maxim peer` subcommand handlers."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence

from maxim.doctor.checks import CheckResult, run_all_checks
from maxim.doctor.platform_detect import PlatformInfo, detect_platform


DOCTOR_USAGE = """\
Usage: maxim doctor [OPTIONS]

Diagnose your Maxim environment + print platform-specific fix suggestions.

Options:
  --retry                  After printing, offer to re-run failed checks one at a time
  --json                   Output results as machine-readable JSON
  --as peer <url>          Run peer-mode checks against a remote leader URL
  --as leader              Force leader-mode checks (skip peer auto-detection)
  --as solo                Force solo-mode checks (no peer/leader wiring)
  --embodiment <REF>       Validate that an SEM component ref resolves
                           (e.g. weapons/rusty_sword) — same shape as the
                           agent flag. Adds an Embodiment section to the
                           report; fails if the ref is unknown.
  --last-decision          Print the most recent lane-routing decision (P9) and exit
  -h, --help               Show this help message

Examples:
  maxim doctor                                   Check local environment
  maxim doctor --retry                           Interactive fix loop
  maxim doctor --json                            JSON output for CI/scripts
  maxim doctor --last-decision                   Why did the last sim pick its model?
  maxim doctor --as peer https://maxim.example.com/v1
                                                 Check peer connectivity
  maxim doctor --embodiment weapons/rusty_sword  Validate SEM ref before running
"""


def run_doctor_subcommand(argv: Sequence[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(DOCTOR_USAGE)
        return 0
    if "--last-decision" in argv:
        return _print_last_decision()
    retry = "--retry" in argv
    as_json = "--json" in argv

    # Parse --as <role> [url]
    role, peer_url = _parse_as_flag(list(argv))

    # Parse --embodiment <REF>
    embodiment_ref = _parse_embodiment_flag(list(argv))

    info = detect_platform()
    sections = run_all_checks(
        info,
        role=role,
        peer_url=peer_url,
        embodiment_ref=embodiment_ref,
    )

    if as_json:
        _json_report(sections, info)
    else:
        _print_report(sections)

    if retry and not as_json:
        return _retry_loop(sections, info, role=role, peer_url=peer_url)

    # Exit code reflects the worst status seen
    worst = _worst_status(sections)
    return {"ok": 0, "warn": 0, "fail": 1, "info": 0}[worst]


def _print_last_decision() -> int:
    """Render the most recent ``LaneDecisionRecord`` to stdout (P9).

    Returns 0 on success, 1 if no log exists yet (e.g. fresh install).
    """
    try:
        from maxim.runtime.decision_log import format_record_human, read_last_record
    except ImportError as e:
        print(f"decision_log unavailable: {e}", file=sys.stderr)
        return 1
    record = read_last_record()
    if record is None:
        print("No lane decisions logged yet.")
        print("Run `maxim` (or any command that builds the LLM router) at least once,")
        print("then re-run `maxim doctor --last-decision`.")
        return 1
    print(format_record_human(record))
    return 0


def _parse_embodiment_flag(argv: list[str]) -> str | None:
    """Extract ``--embodiment <REF>`` from argv. Returns None if absent.

    Mirrors the agent-side ``--embodiment`` flag in `cli_parser.py`.
    The doctor variant validates the ref against ComponentRegistry
    BEFORE the user spends time spinning up an agent.

    On parse failure (missing value, value starts with ``-``), this
    function exits with code 2 and a usage message — silent drop is
    explicitly forbidden here per ``CLAUDE.md::doctor::Don't silently
    drop failures``. A user who typed ``--embodiment`` and got back a
    clean doctor report would walk away thinking everything is fine
    when their request was actually swallowed (cross-confirmed
    pre-merge review finding C1/I2).
    """
    try:
        idx = argv.index("--embodiment")
    except ValueError:
        return None
    if idx + 1 >= len(argv):
        print(
            "error: --embodiment requires a component ref (e.g. weapons/rusty_sword)",
            file=sys.stderr,
        )
        sys.exit(2)
    ref = argv[idx + 1]
    if ref.startswith("-"):
        print(
            f"error: --embodiment expected a component ref, got flag {ref!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    return ref


def _parse_as_flag(argv: list[str]) -> tuple[str | None, str | None]:
    """Extract ``--as <role> [url]`` from argv."""
    try:
        idx = argv.index("--as")
    except ValueError:
        return None, None
    if idx + 1 >= len(argv):
        return None, None
    role_str = argv[idx + 1].lower()
    if role_str not in ("peer", "leader", "solo"):
        print(f"Unknown role: {role_str!r} (expected peer, leader, or solo)", file=sys.stderr)
        return None, None
    peer_url = None
    if role_str == "peer" and idx + 2 < len(argv) and not argv[idx + 2].startswith("-"):
        peer_url = argv[idx + 2]
    return role_str, peer_url


def _print_report(sections: list[tuple[str, list[CheckResult]]]) -> None:
    print()
    for name, results in sections:
        print(f"━━━ {name} ━━━")
        for r in results:
            print(f"  {r.symbol} {r.name}: {r.message}")
            if r.fix:
                for fix_line in r.fix.splitlines():
                    print(f"    → {fix_line}" if fix_line else "    →")
        print()


def _json_report(sections: list[tuple[str, list[CheckResult]]], info: PlatformInfo) -> None:
    """Print machine-readable JSON to stdout."""
    output = {
        "platform": {
            "os": info.os,
            "runtime": info.runtime,
            "distro": info.distro,
            "arch": info.arch,
            "display_name": info.display_name,
        },
        "sections": [
            {
                "name": name,
                "checks": [
                    {
                        "name": r.name,
                        "status": r.status,
                        "message": r.message,
                        "fix": r.fix,
                    }
                    for r in results
                ],
            }
            for name, results in sections
        ],
        "worst_status": _worst_status(sections),
    }
    print(json.dumps(output, indent=2))


def _worst_status(sections: list[tuple[str, list[CheckResult]]]) -> str:
    worst = "ok"
    for _, results in sections:
        for r in results:
            if r.status == "fail":
                return "fail"
            if r.status == "warn":
                worst = "warn"
    return worst


def _retry_loop(
    sections: list[tuple[str, list[CheckResult]]],
    info: PlatformInfo,
    *,
    role: str | None = None,
    peer_url: str | None = None,
) -> int:
    """Walk through failing checks that have retry_id, wait for user fix, re-run."""
    from maxim.doctor.checks import (
        check_cloudflared,
        check_key_age,
        check_key_auth_smoke,
        check_key_permissions,
        check_peer_auth,
        check_peer_key_set,
        check_peer_url_reachable,
        check_server_reachable,
        check_tier_detection,
        check_tunnel_config,
        check_tunnel_config_sync,
        check_vram_pressure,
    )

    # Map retry_id → callable that re-runs the check.
    # Peer key is resolved once for peer checks.
    import os

    peer_key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
    if not peer_key:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                peer_key = cfg.api_key
        except Exception:
            pass

    retryable_fns: dict[str, object] = {
        "server": check_server_reachable,
        "cloudflared": lambda: check_cloudflared(info),
        "tunnel-config": check_tunnel_config,
        "tunnel-config-sync": check_tunnel_config_sync,
        "tier_detection": check_tier_detection,
        "vram_pressure": check_vram_pressure,
        "key-age": check_key_age,
        "key-permissions": check_key_permissions,
        "key-auth": check_key_auth_smoke,
        # Peer checks
        "peer-url": lambda: check_peer_url_reachable(peer_url) if peer_url else None,
        "peer-key": check_peer_key_set,
        "peer-auth": lambda: check_peer_auth(peer_url, peer_key) if peer_url else None,
    }

    # Plan 4 Stage C1: mesh node retry IDs are dynamic (``mesh_node_<name>``).
    # Register one re-probe callable per node that re-reads mesh.yml at
    # retry time so manual config edits between retries pick up without
    # re-running `maxim doctor` from scratch.
    def _make_mesh_node_reprobe(node_name: str):
        def _reprobe():
            from maxim.doctor.checks import _probe_mesh_node_to_check
            from maxim.peer.mesh_config import read_or_synthesize_mesh_config

            try:
                mesh = read_or_synthesize_mesh_config()
            except Exception:
                return None
            if mesh is None:
                return None
            node = mesh.get_node(node_name)
            if node is None:
                return None
            return _probe_mesh_node_to_check(node, mesh.cluster_key)

        return _reprobe

    # Plan 4 Stage C2 (CCR1 — triple-confirmed review finding): orphan
    # drain entries get their own re-probe so operators can run
    # `maxim peer --node <name> resume` to clean up the drain state
    # file, then hit Enter in the retry loop to confirm the orphan is
    # gone. The reprobe re-reads drain state + mesh.yml each iteration
    # and constructs a fresh check result:
    #
    # - If the entry is still orphaned → return the same warn
    # - If the operator ran `resume` (entry removed from drain state) →
    #   return an info CheckResult "orphan cleared" (status ok from
    #   retry loop's perspective, exits the loop for this id)
    # - If the operator edited mesh.yml to re-add the node → the entry
    #   is no longer an orphan, so read_drained_nodes doesn't surface
    #   it; return info "orphan cleared"
    def _make_orphan_reprobe(orphan_name: str):
        def _reprobe():
            from maxim.peer.drain_state import read_drained_nodes
            from maxim.peer.mesh_config import read_or_synthesize_mesh_config

            try:
                mesh = read_or_synthesize_mesh_config()
            except Exception:
                return None
            if mesh is None:
                # Mesh config went away entirely — can't classify.
                return None
            known_names = {n.name for n in mesh.nodes}
            drain_result = read_drained_nodes(known_names)
            if orphan_name not in drain_result.orphans:
                # Orphan cleared: either operator ran `resume` and it's
                # gone from drain state, or they re-added the node to
                # mesh.yml so it's now valid. Either way, report ok so
                # the retry loop exits for this id.
                return CheckResult(
                    name=f"Drain orphan {orphan_name}",
                    status="ok",
                    message=f"orphan cleared for {orphan_name!r}",
                )
            # Still orphaned — return the same warn so the operator
            # knows their edit didn't land.
            return CheckResult(
                name=f"Drain orphan {orphan_name}",
                status="warn",
                message=(f"drain state entry {orphan_name!r} still has no matching node in mesh.yml"),
                fix=(
                    f"Run `maxim peer --node {orphan_name} resume` to clean up "
                    f"the drain state, or re-add {orphan_name!r} to mesh.yml::nodes."
                ),
                retry_id=f"mesh_drain_orphan_{orphan_name}",
            )

        return _reprobe

    for _, results in sections:
        for r in results:
            if r.retry_id and r.retry_id.startswith("mesh_node_"):
                node_name = r.retry_id[len("mesh_node_") :]
                retryable_fns[r.retry_id] = _make_mesh_node_reprobe(node_name)
            elif r.retry_id and r.retry_id.startswith("mesh_drain_orphan_"):
                orphan_name = r.retry_id[len("mesh_drain_orphan_") :]
                retryable_fns[r.retry_id] = _make_orphan_reprobe(orphan_name)

    # Collect failing checks that have retry_id, in section order
    retryable_results: list[tuple[str, CheckResult]] = []
    for _, results in sections:
        for r in results:
            if r.retry_id and r.status not in ("ok", "info") and r.retry_id in retryable_fns:
                retryable_results.append((r.retry_id, r))

    if not retryable_results:
        print("━━━ Retry loop ━━━")
        print("All retryable checks passed.")
        return 0

    print("━━━ Retry loop ━━━")
    print("Press Enter after each fix to re-test. Ctrl+C to exit.")
    print()
    for retry_id, initial_result in retryable_results:
        check_fn = retryable_fns[retry_id]
        result = initial_result
        while result.status not in ("ok", "info"):
            print(f"  {result.symbol} {result.name}: {result.message}")
            if result.fix:
                for line in result.fix.splitlines():
                    print(f"    → {line}" if line else "    →")
            try:
                input("  [press Enter to re-test, or Ctrl+C to exit] ")
            except KeyboardInterrupt:
                print("\n  Exiting retry loop.")
                return 1
            new_result = check_fn()
            if new_result is None:
                break
            result = new_result
            print()
        print(f"  ✓ {result.name}: {result.message}")
        print()
    print("All retryable checks passed.")
    return 0


# ─── maxim peer test <url> ────────────────────────────────────────────────

PEER_USAGE = """\
Usage: maxim peer test <url> [--key KEY] [--model MODEL]

Verify a peer/leader URL is reachable and authenticated. Runs:
  - DNS resolution
  - HTTP(S) handshake
  - GET /v1/models
  - Chat completion round-trip

Options:
  --key KEY     Bearer token (default: $MAXIM_LANE_LARGE_REMOTE_API_KEY)
  --model M     Model name to send (default: auto-detect from /v1/models)

Returns 0 on full success, 1 on any failure.

Tip: For comprehensive peer diagnostics with retry support, use:
  maxim doctor --as peer <url>
"""


def run_peer_subcommand(argv: Sequence[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(PEER_USAGE)
        return 0 if argv else 2
    action = argv[0]
    if action != "test":
        print(f"Unknown action: {action}\n\n{PEER_USAGE}", file=sys.stderr)
        return 2
    if len(argv) < 2:
        print("Missing <url>\n\n" + PEER_USAGE, file=sys.stderr)
        return 2
    url = argv[1]
    key, model = _parse_peer_opts(list(argv[2:]))
    return _peer_test(url, key=key, model=model)


def _parse_peer_opts(opts: list[str]) -> tuple[str | None, str | None]:
    import os

    key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
    # Fall back to peer config if no key in env or args
    if not key:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                key = cfg.api_key
        except Exception:
            pass
    model = None
    i = 0
    while i < len(opts):
        if opts[i] == "--key" and i + 1 < len(opts):
            key = opts[i + 1]
            i += 2
        elif opts[i] == "--model" and i + 1 < len(opts):
            model = opts[i + 1]
            i += 2
        else:
            i += 1
    return key, model


def _peer_test(base_url: str, *, key: str | None, model: str | None) -> int:
    import socket
    import urllib.parse

    from maxim.utils import http as _http

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    print()
    print(f"Testing peer connection to {base_url}")

    # 1. DNS — we still resolve directly for the human-readable error.
    try:
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname or ""
        if host:
            socket.gethostbyname(host)
            print("  ✓ DNS resolves")
        else:
            print("  ✗ URL has no host")
            return 1
    except socket.gaierror as e:
        print(f"  ✗ DNS failed: {e}")
        print(f"    → Check the hostname: {host}")
        return 1

    # 2. Proxy identity check — /v1/debug/ping (LeaderProxy-only endpoint)
    extra_headers: dict[str, str] = {}
    if key:
        extra_headers["Authorization"] = f"Bearer {key}"
    ping_url = f"{base_url}/debug/ping"
    try:
        ping_resp = _http.fetch_url(
            ping_url,
            method="GET",
            headers=extra_headers,
            timeout=_http.TimeoutPolicy(connect_s=2.0, read_s=5.0, total_s=6.0),
        )
        ping_data = ping_resp.json()
        if ping_data.get("service") == "LeaderProxy":
            print(
                f"  ✓ LeaderProxy confirmed (up {ping_data.get('uptime_s', '?')}s, "
                f"auth={'on' if ping_data.get('auth_enabled') else 'off'})"
            )
        else:
            print(f"  ? /v1/debug/ping responded but service={ping_data.get('service', '?')}")
    except _http.HTTPClientError as e:
        if e.status == 404:
            print("  ? /v1/debug/ping returned 404 — may be talking to llama-cpp-server directly")
            print("    → Check tunnel routes to port 8099 (LeaderProxy), not 8100")
        else:
            print(f"  ? /v1/debug/ping returned HTTP {e.status}")
    except _http.HTTPError as e:
        print(f"  ? /v1/debug/ping unreachable: {e}")
    except Exception as e:
        print(f"  ? /v1/debug/ping unreachable: {e}")

    # 3-4. HTTPS handshake + /v1/models
    # Cloudflare's default bot-protection WAF rules block the
    # `Python-urllib/*` User-Agent with a 403. The _external endpoint
    # registered by maxim.utils.http sets User-Agent=maxim-peer/1.0 at
    # registration time — structurally impossible to forget.
    models_url = f"{base_url}/models"
    try:
        t0 = time.time()
        resp = _http.fetch_url(
            models_url,
            method="GET",
            headers=extra_headers,
            timeout=_http.TimeoutPolicy(connect_s=3.0, read_s=10.0, total_s=12.0),
        )
        data = resp.json()
    except _http.HTTPAuthError as e:
        # Key-drift detection (post-config-unification UX fold,
        # 2026-06-03): the canonical 401 cause is "your stored key
        # doesn't match the leader's current key." Direct the
        # operator at rotate-and-re-paste rather than the generic
        # "check the key" message. Aligns with the canonical
        # classifier at peer/probe_classify.py.
        print(f"  ✗ HTTP {e.status}: auth rejected — your stored API key may be stale")
        print("    → On the leader, print the current key:")
        print("        maxim tunnel key show")
        print("    → Then re-pair this peer with the fresh key:")
        print(f"        maxim peer connect {base_url.rstrip('/v1').rstrip('/')}")
        print("        (paste the new key at the prompt)")
        return 1
    except _http.HTTPClientError as e:
        print(f"  ✗ HTTP {e.status}: {e.fix_hint}")
        if e.status == 404:
            print(f"    → /v1/models returned 404. Is {base_url} the right base URL?")
        return 1
    except (_http.HTTPTimeout, _http.HTTPConnectionError) as e:
        print(f"  ✗ Connection failed: {e.fix_hint}")
        print(f"    → Is the server reachable? Try: curl -v {base_url}/models")
        return 1
    except _http.HTTPError as e:
        print(f"  ✗ Probe failed: {e}")
        return 1
    except (OSError, json.JSONDecodeError) as e:
        print(f"  ✗ Response parse failed: {e}")
        return 1
    print(f"  ✓ /v1/models returned 200 ({(time.time() - t0) * 1000:.0f} ms)")

    # Detect model name if not provided
    if model is None:
        try:
            model = data["data"][0]["id"]
        except (KeyError, IndexError, TypeError):
            model = "m"
    print(f"    model: {model}")

    # 4. Chat completion round-trip
    completion_url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    try:
        t0 = time.time()
        chat_resp = _http.fetch_url(
            completion_url,
            method="POST",
            headers={**extra_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=_http.TimeoutPolicy(connect_s=3.0, read_s=60.0, total_s=65.0),
        )
        data = chat_resp.json()
        dt = (time.time() - t0) * 1000
        text = data["choices"][0]["message"]["content"]
        print(f"  ✓ Chat completion in {dt:.0f} ms: {text!r}")
    except Exception as e:
        print(f"  ✗ Chat completion failed: {e}")
        return 1

    print()
    print(f"Ready to run: MAXIM_LANE_LARGE_REMOTE_URL={base_url} maxim")
    return 0


__all__ = ["run_doctor_subcommand", "run_peer_subcommand"]
