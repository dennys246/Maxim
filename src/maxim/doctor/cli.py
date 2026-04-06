"""`maxim doctor` + `maxim peer` subcommand handlers."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence

from maxim.doctor.checks import CheckResult, run_all_checks
from maxim.doctor.platform_detect import detect_platform


DOCTOR_USAGE = """\
Usage: maxim doctor [--retry]

Diagnose your Maxim environment + print platform-specific fix suggestions.

Options:
  --retry   After printing, offer to re-run failed checks one at a time
            (press Enter after you've applied each fix).
"""


def run_doctor_subcommand(argv: Sequence[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(DOCTOR_USAGE)
        return 0
    retry = "--retry" in argv

    info = detect_platform()
    sections = run_all_checks(info)
    _print_report(sections)

    if retry:
        return _retry_loop(info)

    # Exit code reflects the worst status seen
    worst = _worst_status(sections)
    return {"ok": 0, "warn": 0, "fail": 1}[worst]


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


def _worst_status(sections: list[tuple[str, list[CheckResult]]]) -> str:
    worst = "ok"
    for _, results in sections:
        for r in results:
            if r.status == "fail":
                return "fail"
            if r.status == "warn":
                worst = "warn"
    return worst


def _retry_loop(info) -> int:
    """Walk through failing checks, wait for user Enter, re-run each."""
    from maxim.doctor.checks import (
        check_cloudflared,
        check_llama_cpp_server_installed,
        check_server_reachable,
        check_tunnel_config,
        check_tunnel_config_sync,
    )

    retryable = {
        "server": check_server_reachable,
        "cloudflared": lambda: check_cloudflared(info),
        "tunnel-config": check_tunnel_config,
        "tunnel-config-sync": check_tunnel_config_sync,
    }
    print("━━━ Retry loop ━━━")
    print("Press Enter after each fix to re-test. Ctrl+C to exit.")
    print()
    for retry_id, check_fn in retryable.items():
        result = check_fn()
        if result.status == "ok":
            continue
        while result.status != "ok":
            print(f"  {result.symbol} {result.name}: {result.message}")
            if result.fix:
                for line in result.fix.splitlines():
                    print(f"    → {line}" if line else "    →")
            try:
                input("  [press Enter to re-test, or Ctrl+C to exit] ")
            except KeyboardInterrupt:
                print("\n  Exiting retry loop.")
                return 1
            result = check_fn()
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
  --key KEY     Bearer token (default: $MAXIM_LANE_INFER_REMOTE_API_KEY)
  --model M     Model name to send (default: auto-detect from /v1/models)

Returns 0 on full success, 1 on any failure.
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

    key = os.environ.get("MAXIM_LANE_INFER_REMOTE_API_KEY")
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
    import urllib.error
    import urllib.parse
    import urllib.request
    import socket

    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    print()
    print(f"Testing peer connection to {base_url}")

    # 1. DNS
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
    headers = {"User-Agent": "maxim-peer-test/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    ping_url = f"{base_url}/debug/ping"
    try:
        ping_req = urllib.request.Request(ping_url, headers=headers)
        with urllib.request.urlopen(ping_req, timeout=5) as resp:  # noqa: S310
            ping_data = json.loads(resp.read())
        if ping_data.get("service") == "LeaderProxy":
            print(
                f"  ✓ LeaderProxy confirmed (up {ping_data.get('uptime_s', '?')}s, auth={'on' if ping_data.get('auth_enabled') else 'off'})"
            )
        else:
            print(f"  ? /v1/debug/ping responded but service={ping_data.get('service', '?')}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("  ? /v1/debug/ping returned 404 — may be talking to llama-cpp-server directly")
            print("    → Check tunnel routes to port 8099 (LeaderProxy), not 8100")
        else:
            print(f"  ? /v1/debug/ping returned HTTP {e.code}")
    except Exception as e:
        print(f"  ? /v1/debug/ping unreachable: {e}")

    # 3-4. HTTPS handshake + /v1/models
    # NOTE: Cloudflare's default bot-protection WAF rules block the
    # `Python-urllib/*` User-Agent with a 403. Send a neutral UA so the
    # request looks like any other HTTP client — matches curl behavior.
    models_url = f"{base_url}/models"
    try:
        req = urllib.request.Request(models_url, headers=headers)
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10.0) as resp:  # noqa: S310
            body = resp.read().decode()
            data = json.loads(body)
    except urllib.error.HTTPError as e:
        print(f"  ✗ HTTP {e.code}: {e.reason}")
        if e.code == 401:
            print("    → Bearer token rejected. Check the key matches the leader's.")
            print("    → On the leader, run: maxim tunnel key show")
        elif e.code == 404:
            print(f"    → /v1/models returned 404. Is {base_url} the right base URL?")
        return 1
    except (urllib.error.URLError, socket.timeout) as e:
        print(f"  ✗ Connection failed: {e}")
        print(f"    → Is the server reachable? Try: curl -v {base_url}/models")
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
        req = urllib.request.Request(
            completion_url,
            data=json.dumps(payload).encode(),
            headers={**headers, "Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=60.0) as resp:  # noqa: S310
            data = json.loads(resp.read())
        dt = (time.time() - t0) * 1000
        text = data["choices"][0]["message"]["content"]
        print(f"  ✓ Chat completion in {dt:.0f} ms: {text!r}")
    except Exception as e:
        print(f"  ✗ Chat completion failed: {e}")
        return 1

    print()
    print(f"Ready to run: MAXIM_LANE_INFER_REMOTE_URL={base_url} maxim")
    return 0


__all__ = ["run_doctor_subcommand", "run_peer_subcommand"]
