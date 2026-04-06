"""Individual diagnostic checks run by `maxim doctor`.

Each check is a pure function: takes `PlatformInfo`, returns a `CheckResult`.
Results include a status, message, and platform-specific fix hint. The fix
hint is rendered verbatim in the doctor output — it should be copy-pasteable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from maxim.doctor.platform_detect import PlatformInfo, detect_wsl_ip

Status = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str | None = None       # platform-specific fix hint
    retry_id: str | None = None  # only set when retry is meaningful

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗"}[self.status]


# ─── environment checks ────────────────────────────────────────────────────

def check_gpu() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="torch not installed — CPU-only mode",
            fix="Install GPU stack: pip install -e '.[llm-torch]'",
        )
    if not torch.cuda.is_available():
        import os
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_vis == "":
            return CheckResult(
                name="GPU / CUDA",
                status="warn",
                message="CUDA hidden (CUDA_VISIBLE_DEVICES=''). Likely the Blackwell workaround.",
                fix=(
                    "If you have a Blackwell GPU (RTX 50xx), current behavior keeps CUDA\n"
                    "  enabled by default. If it's hidden, set: unset MAXIM_BLACKWELL_HIDE_CUDA"
                ),
            )
        return CheckResult(
            name="GPU / CUDA",
            status="warn",
            message="No CUDA device available — inference will be CPU-only",
        )
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    return CheckResult(
        name="GPU / CUDA",
        status="ok",
        message=f"{props.name} ({vram_gb:.1f} GB VRAM)",
    )


def check_llama_cpp_server_installed() -> CheckResult:
    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        return CheckResult(
            name="llama-cpp-server installed",
            status="fail",
            message="llama_cpp.server not installed — auto-spawn disabled",
            fix="pip install -e '.[llm-server]'",
        )
    return CheckResult(
        name="llama-cpp-server installed",
        status="ok",
        message="ready for auto-spawn",
    )


def check_server_reachable(port: int = 8100) -> CheckResult:
    """Probe the local auto-spawn port."""
    from maxim.runtime.lane_backends import _llm_server_responding_at
    url = f"http://127.0.0.1:{port}/v1"
    if _llm_server_responding_at(url, timeout_s=2.0):
        return CheckResult(
            name="Auto-spawn server",
            status="ok",
            message=f"responding at {url}",
            retry_id="server",
        )
    return CheckResult(
        name="Auto-spawn server",
        status="warn",
        message=f"nothing responding at {url}",
        fix="Run `maxim` in another terminal — auto-spawn will launch one.",
        retry_id="server",
    )


# ─── LAN access ────────────────────────────────────────────────────────────

def check_lan_access(info: PlatformInfo, port: int = 8100) -> CheckResult:
    """Platform-specific LAN-access guidance for exposing the local server.

    Doesn't test reachability from an external host (can't, from here). Just
    prints what the user needs to do to make the server LAN-reachable, with
    their actual IPs filled in.
    """
    from maxim.runtime.leader_mode import detect_role
    # Use resolved role (respects both MAXIM_ROLE env var and implicit
    # cloudflared-config detection), not just the raw env var.
    leader = detect_role().is_leader

    if info.runtime == "wsl2":
        wsl_ip = detect_wsl_ip() or "<wsl-ip>"
        host_ip = info.windows_host_ip or "<windows-host-ip>"
        fix = (
            f"WSL2 peers can't reach http://127.0.0.1:{port} directly — the\n"
            f"  Windows host needs to forward the port.\n\n"
            f"  1. In PowerShell on Windows (as admin):\n"
            f"       netsh interface portproxy add v4tov4 listenport={port} listenaddress=0.0.0.0 connectport={port} connectaddress={wsl_ip}\n"
            f'       netsh advfirewall firewall add rule name="Maxim LLM" dir=in action=allow protocol=TCP localport={port}\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  3. Peers connect via your Windows LAN IP:\n"
            f"       http://{host_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (WSL2)",
            status="warn" if not leader else "ok",
            message=(
                "WSL2 port-forwarding required for LAN peers" if not leader
                else f"leader mode + port-forwarded at {host_ip}:{port}"
            ),
            fix=fix if not leader else None,
        )

    if info.runtime in ("wsl1", "docker"):
        return CheckResult(
            name=f"LAN access ({info.runtime})",
            status="warn",
            message=f"LAN access untested on {info.runtime}",
            fix=(
                f"{info.runtime} has unusual networking. Consider switching to WSL2\n"
                f"  (mirrored networking mode) or running Maxim in the native host."
            ),
        )

    if info.os == "linux":
        from maxim.doctor.platform_detect import detect_lan_ip
        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        firewall_hint = {
            "ubuntu": f"sudo ufw allow {port}/tcp",
            "debian": f"sudo ufw allow {port}/tcp",
            "fedora": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "rhel":   f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "arch":   f"sudo ufw allow {port}/tcp  # (or your preferred firewall)",
        }.get(info.distro, f"Open TCP port {port} in your firewall")
        fix = (
            f"Linux peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. Allow port {port} in your firewall (if active):\n"
            f"       {firewall_hint}\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (Linux)",
            status="warn" if not leader else "ok",
            message="native Linux — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "macos":
        from maxim.doctor.platform_detect import detect_lan_ip
        lan_ip = detect_lan_ip() or "<your-lan-ip>"
        fix = (
            f"macOS peers reach you directly via your LAN IP:\n"
            f"  1. Run Maxim as leader:\n"
            f"       MAXIM_ROLE=leader maxim\n\n"
            f"  2. If macOS firewall is enabled, allow Python in:\n"
            f"       System Settings → Network → Firewall → Options\n\n"
            f"  3. Peers connect:\n"
            f"       http://{lan_ip}:{port}/v1\n"
        )
        return CheckResult(
            name="LAN access (macOS)",
            status="warn" if not leader else "ok",
            message="native macOS — peers reach LAN IP directly",
            fix=fix if not leader else None,
        )

    if info.os == "windows":
        fix = (
            f"Windows native:\n"
            f"  1. In PowerShell as admin, allow inbound:\n"
            f'       New-NetFirewallRule -DisplayName "Maxim LLM" -Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow\n\n'
            f"  2. Run Maxim as leader:\n"
            f"       $env:MAXIM_ROLE='leader'\n"
            f"       maxim\n"
        )
        return CheckResult(
            name="LAN access (Windows)",
            status="warn" if not leader else "ok",
            message="native Windows",
            fix=fix if not leader else None,
        )

    return CheckResult(
        name="LAN access",
        status="warn",
        message="unknown platform",
    )


# ─── cloudflared / tunnel ──────────────────────────────────────────────────

def check_cloudflared(info: PlatformInfo) -> CheckResult:
    from maxim.tunnel.cloudflared import cloudflared_version, find_cloudflared
    path = find_cloudflared()
    if path is None:
        install = {
            ("linux", "ubuntu"):
                "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
                "  sudo dpkg -i cloudflared.deb",
            ("linux", "debian"):
                "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
                "  sudo dpkg -i cloudflared.deb",
            ("linux", "fedora"):
                "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
                "  sudo rpm -i cloudflared.rpm",
            ("linux", "rhel"):
                "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
                "  sudo rpm -i cloudflared.rpm",
            ("linux", "arch"):
                "yay -S cloudflared  # or your preferred AUR helper",
            ("macos", "unknown"):
                "brew install cloudflare/cloudflare/cloudflared",
            ("windows", "unknown"):
                "Download https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe\n"
                "  (add to PATH)",
        }
        key = (info.os, info.distro if info.os == "linux" else "unknown")
        install_cmd = install.get(
            key,
            "See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/",
        )
        return CheckResult(
            name="cloudflared",
            status="warn",
            message="not installed — required for external tunnel access",
            fix=f"Install:\n  {install_cmd}\n  Then run: maxim tunnel setup",
            retry_id="cloudflared",
        )
    version = cloudflared_version() or "(unknown version)"
    return CheckResult(
        name="cloudflared",
        status="ok",
        message=f"{path} — {version}",
    )


def check_tunnel_config() -> CheckResult:
    from maxim.tunnel.config import CONFIG_PATH, read_config_summary
    summary = read_config_summary()
    if summary is None:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"no config at {CONFIG_PATH}",
            fix="Run: maxim tunnel setup",
            retry_id="tunnel-config",
        )
    hostname = summary.get("hostname", "?")
    return CheckResult(
        name="Tunnel config",
        status="ok",
        message=f"hostname={hostname}",
    )


# ─── API key ───────────────────────────────────────────────────────────────

def check_api_key() -> CheckResult:
    from maxim.tunnel.keys import key_exists, key_file_path, truncate_for_display, read_key
    if not key_exists():
        return CheckResult(
            name="API key",
            status="warn",
            message="no key set — server accepts all requests (localhost is fine)",
            fix=(
                "Generate one before exposing to LAN/tunnel:\n"
                "  maxim tunnel key rotate\n"
                "  maxim tunnel key export   # shell snippets for peers"
            ),
        )
    key = read_key() or ""
    return CheckResult(
        name="API key",
        status="ok",
        message=f"{truncate_for_display(key)} at {key_file_path()}",
    )


# ─── leader role ───────────────────────────────────────────────────────────

def check_role() -> CheckResult:
    from maxim.runtime.leader_mode import detect_role
    decision = detect_role()
    if decision.is_leader:
        return CheckResult(
            name="Role",
            status="ok",
            message=f"leader (bind={decision.bind_host}) — {decision.reason}",
        )
    return CheckResult(
        name="Role",
        status="ok",
        message=f"{decision.role} (bind={decision.bind_host}) — {decision.reason}",
    )


def _check_lane_metrics() -> list["CheckResult"]:
    """Report per-lane performance metrics if available (Phase 8)."""
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry
    except Exception:
        return []
    registry = get_metrics_registry()
    results: list[CheckResult] = []
    for name, metrics in registry.all_metrics().items():
        snap = metrics.snapshot()
        total = snap["jobs_completed"] + snap["jobs_failed"]
        if total == 0:
            continue
        msg = metrics.format_compact()
        status: Status = "ok"
        if snap["failure_rate"] > 0.2:
            status = "warn"
        if snap["failure_rate"] > 0.5:
            status = "fail"
        results.append(CheckResult(name=f"Lane: {name}", status=status, message=msg))
    return results


def run_all_checks(info: PlatformInfo) -> list[tuple[str, list[CheckResult]]]:
    """Return ordered [(section_name, results)] for `maxim doctor`."""
    sections = [
        ("Environment", [
            CheckResult(name="Platform", status="ok", message=info.display_name),
            CheckResult(name="Architecture", status="ok", message=info.arch),
            check_gpu(),
        ]),
        ("Local LLM", [
            check_llama_cpp_server_installed(),
            check_server_reachable(),
        ]),
        ("Role & Access", [
            check_role(),
            check_lan_access(info),
        ]),
        ("Tunnel (Cloudflare)", [
            check_cloudflared(info),
            check_tunnel_config(),
        ]),
        ("API key", [
            check_api_key(),
        ]),
    ]
    # Phase 8: lane metrics (only show if any calls have been recorded)
    lane_results = _check_lane_metrics()
    if lane_results:
        sections.append(("Lane Metrics", lane_results))
    return sections


__all__ = [
    "CheckResult",
    "Status",
    "check_gpu",
    "check_llama_cpp_server_installed",
    "check_server_reachable",
    "check_lan_access",
    "check_cloudflared",
    "check_tunnel_config",
    "check_api_key",
    "check_role",
    "run_all_checks",
]
