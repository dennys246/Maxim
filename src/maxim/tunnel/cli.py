"""`maxim tunnel` subcommand handler.

Three actions:
  setup   — guided interactive setup: login, create tunnel, DNS route, write config.yml
  status  — show what's currently configured
  start   — launch cloudflared tunnel run as a foreground process (optional)
"""
from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

from maxim.tunnel.cloudflared import (
    CloudflaredNotInstalled,
    cloudflared_version,
    find_cloudflared,
    find_tunnel_credentials,
    install_hint,
    require_cloudflared,
    run_tunnel_create,
    run_tunnel_login,
    run_tunnel_route_dns,
)
from maxim.tunnel.config import (
    CONFIG_PATH,
    TunnelConfig,
    config_exists,
    read_config_summary,
    write_config_yml,
)
from maxim.tunnel.keys import (
    ENV_VAR,
    ensure_key,
    format_all_snippets,
    key_exists,
    key_file_path,
    read_key,
    rotate_key,
    truncate_for_display,
)


USAGE = """\
Usage: maxim tunnel <action> [options]

Actions:
  setup        Guided interactive tunnel setup (login, create, DNS, config.yml)
  status       Show current tunnel configuration
  start        Run cloudflared tunnel daemon in the foreground (--force to
               bypass the duplicate-instance check)
  tail         Stream cloudflared + llama-cpp-server logs (debug)
  key show     Print the current Maxim API key
  key rotate   Generate a new API key (invalidates peers)
  key export   Print copy-paste shell snippets for peers to set the key

Once `setup` completes, subsequent `maxim` runs auto-detect ~/.cloudflared/config.yml
and promote this machine to leader mode. Peers can point at your tunnel URL using
the API key from `key export`.
"""


DEFAULT_TUNNEL_NAME = "maxim-llm"
DEFAULT_LOCAL_PORT = 8100


def run_tunnel_subcommand(argv: Sequence[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2
    action = argv[0]
    if action == "setup":
        return _cmd_setup()
    if action == "status":
        return _cmd_status()
    if action == "start":
        return _cmd_start(list(argv[1:]))
    if action == "key":
        return _cmd_key(list(argv[1:]))
    if action == "tail":
        return _cmd_tail(list(argv[1:]))
    print(f"Unknown action: {action}\n\n{USAGE}", file=sys.stderr)
    return 2


# ─── setup ────────────────────────────────────────────────────────────────

def _cmd_setup() -> int:
    print("─" * 62)
    print("  Maxim tunnel setup — guided cloudflared configuration")
    print("─" * 62)
    print()

    # 1. Binary check
    cloudflared = find_cloudflared()
    if cloudflared is None:
        print("✗ cloudflared binary not found on PATH.")
        print()
        print("Install it first:")
        print(install_hint())
        print()
        print("Then re-run `maxim tunnel setup`.")
        return 1
    version = cloudflared_version() or "(unknown version)"
    print(f"✓ cloudflared found: {cloudflared} — {version}")
    print()

    # 2. Existing config check
    if config_exists():
        existing = read_config_summary() or {}
        print(f"! {CONFIG_PATH} already exists:")
        for k, v in existing.items():
            print(f"    {k}: {v}")
        print()
        answer = _prompt("Overwrite existing config? [y/N]: ", default="n").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted. Nothing changed.")
            return 1
        print()

    # 3. Cloudflare auth
    print("Step 1/4 — Authenticate with Cloudflare (browser opens)")
    print("  cloudflared will print a URL; click 'Authorize' on the domain you want to use.")
    print()
    _prompt("Press Enter to continue (or Ctrl+C to abort)...", default="")
    print()
    exit_code = run_tunnel_login()
    if exit_code != 0:
        print(f"✗ cloudflared tunnel login failed (exit code {exit_code}).")
        return 1
    print("✓ Cloudflare authenticated.")
    print()

    # 4. Create tunnel
    tunnel_name = _prompt(
        f"Step 2/4 — Tunnel name [{DEFAULT_TUNNEL_NAME}]: ",
        default=DEFAULT_TUNNEL_NAME,
    ).strip() or DEFAULT_TUNNEL_NAME
    print()
    print(f"Creating tunnel '{tunnel_name}'...")
    code, output = run_tunnel_create(tunnel_name)
    if code != 0:
        # Common case: tunnel already exists. Tell user but continue.
        if "already exists" in output.lower() or "tunnel with name" in output.lower():
            print(f"! Tunnel '{tunnel_name}' already exists — reusing it.")
        else:
            print(f"✗ Failed to create tunnel:\n{output}")
            return 1
    else:
        print(f"✓ Tunnel '{tunnel_name}' created.")
    print()

    # Locate the credentials file
    cred_file = find_tunnel_credentials(tunnel_name)
    if cred_file is None:
        print(
            f"✗ Could not locate credentials file for '{tunnel_name}' in ~/.cloudflared/.\n"
            f"  Run `cloudflared tunnel list` to inspect, then re-run `maxim tunnel setup`."
        )
        return 1
    print(f"✓ Found credentials: {cred_file}")
    print()

    # Extract tunnel UUID from credentials filename
    tunnel_id = cred_file.stem

    # 5. DNS route
    hostname = _prompt(
        "Step 3/4 — Hostname (e.g. maxim.yourdomain.com): ",
        default="",
    ).strip()
    if not hostname:
        print("✗ Hostname required. Aborted.")
        return 1
    print()
    print(f"Routing DNS: {hostname} → tunnel '{tunnel_name}'...")
    code, output = run_tunnel_route_dns(tunnel_name, hostname)
    if code != 0:
        # Could be already-routed; surface the output but continue
        if "already exists" in output.lower() or "already resolves" in output.lower():
            print(f"! DNS route for {hostname} already exists — reusing it.")
        else:
            print(f"✗ DNS route failed:\n{output}")
            print("  (If you need to change an existing route, use the Cloudflare dashboard.)")
            return 1
    else:
        print(f"✓ DNS routed: {hostname}")
    print()

    # 6. Choose local port + write config
    port_raw = _prompt(
        f"Step 4/4 — Local server port [{DEFAULT_LOCAL_PORT}]: ",
        default=str(DEFAULT_LOCAL_PORT),
    ).strip() or str(DEFAULT_LOCAL_PORT)
    try:
        port = int(port_raw)
    except ValueError:
        port = DEFAULT_LOCAL_PORT
    print()

    cfg = TunnelConfig(
        tunnel_id=tunnel_id,
        credentials_file=str(cred_file),
        hostname=hostname,
        local_service=f"http://localhost:{port}",
    )
    try:
        path = write_config_yml(cfg, overwrite=True)
    except OSError as e:
        print(f"✗ Failed to write config: {e}")
        return 1
    print(f"✓ Wrote {path}")
    print()

    # 7. Generate API key if not present
    key = ensure_key()
    print(f"✓ API key ready at {key_file_path()}")
    print(f"  {truncate_for_display(key)}")
    print()

    # 8. Summary
    print("─" * 62)
    print("  Setup complete")
    print("─" * 62)
    print(f"  Tunnel:    {tunnel_name} ({tunnel_id})")
    print(f"  Hostname:  https://{hostname}")
    print(f"  Local:     http://localhost:{port}")
    print(f"  API key:   {truncate_for_display(key)}")
    print()
    print("Next steps:")
    print("  1. Test the tunnel in the foreground first:")
    print(f"       cloudflared tunnel run {tunnel_name}")
    print("     (leave it running; Ctrl+C to stop)")
    print()
    print("     Once that works, install as a system service (runs on boot):")
    print(f"       sudo cloudflared --config {CONFIG_PATH} service install")
    print("       # NOTE: --config is required because sudo changes HOME to /root,")
    print("       #       so cloudflared can't find the config at ~/.cloudflared/config.yml")
    print()
    print("  2. Start Maxim normally — leader mode auto-detects the config + API key:")
    print("       maxim")
    print()
    print("  3. Share the API key with peers via a secure channel, then:")
    print("       maxim tunnel key export     # on this machine — shows copy-paste snippets")
    print()
    print(f"  4. On each peer:")
    print(f"       export MAXIM_LANE_INFER_REMOTE_URL=https://{hostname}/v1")
    print(f'       export {ENV_VAR}="<key-shared-securely>"')
    print("       export MAXIM_MAX_CLOUD_LANES=1")
    print("       maxim")
    print("─" * 62)
    return 0


# ─── status ───────────────────────────────────────────────────────────────

def _cmd_status() -> int:
    print("─" * 62)
    print("  Maxim tunnel status")
    print("─" * 62)

    cloudflared = find_cloudflared()
    if cloudflared is None:
        print("  cloudflared: ✗ not installed")
        print()
        print("  Install:")
        print("  " + install_hint().replace("\n", "\n  "))
        return 1
    version = cloudflared_version() or "(unknown)"
    print(f"  cloudflared: ✓ {cloudflared} — {version}")

    summary = read_config_summary()
    if summary is None:
        print(f"  config.yml:  ✗ not found at {CONFIG_PATH}")
        print()
        print("  Run: maxim tunnel setup")
        return 1
    print(f"  config.yml:  ✓ {CONFIG_PATH}")
    for k, v in summary.items():
        print(f"    {k}: {v}")

    # Check if the tunnel daemon is running by looking for cloudflared processes.
    daemon_running = _is_daemon_running()
    print(f"  daemon:      {'✓ running' if daemon_running else '✗ not running'}")
    if not daemon_running:
        print()
        tunnel_id = summary.get("tunnel", "")
        hint_name = tunnel_id or "<tunnel-name>"
        print(f"  Start with:  cloudflared tunnel run {hint_name}")
        print(f"  Or install:  sudo cloudflared --config {CONFIG_PATH} service install")
    print("─" * 62)
    return 0


def _is_daemon_running() -> bool:
    """Best-effort check: look for a cloudflared process with 'tunnel run'."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "cloudflared.*tunnel.*run"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


# ─── start ────────────────────────────────────────────────────────────────

def _cmd_start(extra_args: list[str]) -> int:
    """Run `cloudflared tunnel run` in the foreground.

    Useful for testing before installing as a service. Inherits stdout/stderr
    so the user sees connection logs directly. Refuses to start if another
    cloudflared process is already running (systemd service OR another
    foreground invocation) — duplicate tunnel connectors waste Cloudflare
    edge slots and confuse log streams.

    Bypass with `--force` if you actually want duplicate connectors.
    """
    summary = read_config_summary()
    if summary is None:
        print("✗ No tunnel configured. Run `maxim tunnel setup` first.", file=sys.stderr)
        return 1
    try:
        cloudflared = require_cloudflared()
    except CloudflaredNotInstalled as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1
    # Use the tunnel UUID from config.yml as the identifier
    tunnel_id = summary.get("tunnel", "")
    if not tunnel_id:
        print("✗ config.yml missing 'tunnel:' field.", file=sys.stderr)
        return 1

    # Pre-flight: is something already tunnelling? If so, refuse with a
    # clear explanation unless --force was passed.
    force = "--force" in extra_args
    if force:
        extra_args = [a for a in extra_args if a != "--force"]
    elif _cloudflared_service_active():
        print(
            "✗ cloudflared systemd service is already active.\n\n"
            "  Starting a second cloudflared would create a duplicate tunnel\n"
            "  connector (Cloudflare tolerates this but it wastes edge slots\n"
            "  and splits your log streams across two processes).\n\n"
            "  What do you want to do?\n\n"
            "    Watch logs:       maxim tunnel tail\n"
            "    Stop the service: sudo systemctl stop cloudflared\n"
            "    Force foreground: maxim tunnel start --force\n",
            file=sys.stderr,
        )
        return 1
    elif _cloudflared_process_running():
        print(
            "✗ cloudflared is already running as a foreground process.\n\n"
            "  Starting another would create a duplicate connector.\n\n"
            "  What do you want to do?\n\n"
            "    Ctrl+C the other process first, then re-run this command.\n"
            "    Or bypass:  maxim tunnel start --force\n",
            file=sys.stderr,
        )
        return 1

    print(f"Starting cloudflared tunnel: {tunnel_id}")
    print("(Ctrl+C to stop)")
    # Surface the service-install hint unless a service is already running.
    _print_service_install_hint()
    print()
    cmd = [cloudflared, "tunnel", "run", tunnel_id] + extra_args
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def _cloudflared_process_running() -> bool:
    """Best-effort check for ANY cloudflared tunnel run process (not just systemd)."""
    try:
        result = subprocess.run(
            ["pgrep", "-af", "cloudflared.*tunnel.*run"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _print_service_install_hint() -> None:
    """Suggest installing cloudflared as a persistent service if one isn't running.

    Detects whether a cloudflared systemd unit is already active on Linux;
    otherwise prints an OS-appropriate `service install` command. Skips the
    hint entirely when a service is already managing the tunnel (the
    foreground start is then a deliberate debug run).
    """
    if _cloudflared_service_active():
        return  # user already has a persistent service; don't nag

    import platform
    system = platform.system().lower()
    print()
    print("  ──────────────────────────────────────────────────────────────")
    print("  Tip: running foreground. For a persistent tunnel that survives")
    print("  reboots + terminal close, install cloudflared as a service:")
    if system == "linux":
        print(f"    sudo cloudflared --config {CONFIG_PATH} service install")
        print("    sudo systemctl enable --now cloudflared")
    elif system == "darwin":
        print(f"    sudo cloudflared --config {CONFIG_PATH} service install")
        print("    sudo launchctl start com.cloudflare.cloudflared")
    elif system == "windows":
        print(f'    cloudflared --config "{CONFIG_PATH}" service install')
        print("    # Then use Services.msc or `sc start cloudflared`")
    else:
        print(f"    sudo cloudflared --config {CONFIG_PATH} service install")
    print("  (Run `maxim tunnel start` any time to test foreground.)")
    print("  ──────────────────────────────────────────────────────────────")


def _cloudflared_service_active() -> bool:
    """Return True iff systemd reports the cloudflared service as active.

    POSIX-only, best-effort. Returns False on Windows, macOS, or when
    systemctl isn't reachable — so the hint prints on those platforms
    (there's no way to know if launchd / Windows Services already has it).
    """
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "cloudflared"],
            capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "active"


# ─── tail ─────────────────────────────────────────────────────────────────

TAIL_USAGE = """\
Usage: maxim tunnel tail [--since DURATION] [--filter REGEX]

Streams cloudflared's systemd logs alongside llama-cpp-server output (if
visible) so you can watch peer requests traverse the tunnel → local server
in real time.

Options:
  --since DURATION   Show logs from the last N (e.g., 5m, 1h). Default: 2m
  --filter REGEX     Only show lines matching this regex

Tip: combine with MAXIM_LANE_TRACE=1 on the peer side and
MAXIM_TUNNEL_ECHO=1 on the leader to get full request-id correlation.
"""


def _cmd_tail(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(TAIL_USAGE)
        return 0
    since = "2m"
    filter_re: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--since" and i + 1 < len(argv):
            since = argv[i + 1]
            i += 2
        elif a == "--filter" and i + 1 < len(argv):
            filter_re = argv[i + 1]
            i += 2
        else:
            print(f"Unknown option: {a}\n\n{TAIL_USAGE}", file=sys.stderr)
            return 2

    # Convert the user-friendly "2m"/"5h" form to journalctl's accepted
    # "-2 minutes ago" / "-5 hours ago" format. journalctl rejects the
    # compact form with "Failed to parse timestamp" on most distros.
    since_arg = _to_journalctl_since(since)

    # Prefer journalctl (systemd service); fall back to looking for live
    # cloudflared stderr. Either way we stream until Ctrl+C.
    cmd = ["journalctl", "-u", "cloudflared", "--since", since_arg, "-f", "--no-pager"]
    if filter_re:
        cmd.extend(["-g", filter_re])

    print(f"Streaming cloudflared logs (systemd service)...")
    print(f"  cmd: {' '.join(cmd)}")
    print("  Ctrl+C to stop.")
    print()
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "✗ `journalctl` not found. On this system, cloudflared may be running\n"
            "  as a foreground process or the systemd logs are in a different location.\n"
            "  If running `maxim tunnel start` in another terminal, its stderr shows\n"
            "  the same information directly.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


def _to_journalctl_since(value: str) -> str:
    """Translate '2m' / '5h' / '1d' into journalctl's accepted relative form.

    systemd-journalctl --since accepts a few formats; the ones that actually
    work on a stock Ubuntu 24.04 build:
      - absolute timestamp: '2024-01-02 15:04:05'
      - "N UNIT ago" with no leading dash: '2 minutes ago', '5 hours ago'
      - compact "-Nunit" (no space): '-2min', '-5h', '-1day'

    What does NOT work (despite some docs suggesting it): '-2 minutes ago'
    with both the dash AND the spaces. We translate to the "N UNIT ago"
    form since it's the most readable in log output.
    """
    import re
    text = value.strip()
    if not text:
        return "2 minutes ago"
    # Already in a journalctl-accepted form — leave it alone
    if " ago" in text or re.match(r"^\d{4}-\d{2}-\d{2}", text) or re.match(r"^-\d+[a-z]+$", text):
        return text
    # Match compact forms: "2m", "5h", "1d", "30s"
    m = re.fullmatch(r"(\d+)\s*([smhd])", text)
    if not m:
        return text  # pass through; let journalctl complain if still invalid
    n, unit = int(m.group(1)), m.group(2)
    unit_name = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
    return f"{n} {unit_name} ago"


# ─── key ──────────────────────────────────────────────────────────────────

def _cmd_key(subargs: list[str]) -> int:
    if not subargs:
        print("Usage: maxim tunnel key <show|rotate|export>", file=sys.stderr)
        return 2
    action = subargs[0]
    if action == "show":
        return _cmd_key_show()
    if action == "rotate":
        return _cmd_key_rotate()
    if action == "export":
        return _cmd_key_export()
    print(f"Unknown key action: {action}", file=sys.stderr)
    return 2


def _cmd_key_show() -> int:
    key = read_key()
    if key is None:
        print(f"No API key set. File: {key_file_path()}")
        print("Run: maxim tunnel key rotate")
        return 1
    print(f"Key file:   {key_file_path()}")
    print(f"Key:        {key}")
    print(f"Truncated:  {truncate_for_display(key)}")
    print()
    print("Share with peers via a secure channel (Signal, encrypted email, etc.)")
    print("They'll set it with: maxim tunnel key export")
    return 0


def _cmd_key_rotate() -> int:
    had_key = key_exists()
    old_display = truncate_for_display(read_key() or "")
    key = rotate_key()
    print(f"✓ Generated new API key: {truncate_for_display(key)}")
    print(f"  Saved to: {key_file_path()}")
    if had_key:
        print()
        print(f"⚠ Previous key ({old_display}) is now invalid.")
        print("  Peers using the old key will get 401 errors until you share the new one.")
        print("  Restart `maxim` on the leader machine to pick up the new key.")
    print()
    print("Run `maxim tunnel key export` for peer setup snippets.")
    return 0


def _cmd_key_export() -> int:
    key = read_key()
    if key is None:
        print("No API key set. Run: maxim tunnel key rotate")
        return 1
    print("─" * 62)
    print("  Share the snippet below matching your shell / OS.")
    print("─" * 62)
    print()
    print(format_all_snippets(key))
    print()
    print("─" * 62)
    print("  After setting the env var, the peer can connect:")
    print("    maxim")
    print("─" * 62)
    return 0


# ─── helpers ──────────────────────────────────────────────────────────────

def _prompt(msg: str, *, default: str) -> str:
    try:
        answer = input(msg)
    except EOFError:
        return default
    return answer if answer else default


__all__ = ["run_tunnel_subcommand"]
