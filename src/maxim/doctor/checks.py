"""Individual diagnostic checks run by `maxim doctor`.

Each check is a pure function: takes `PlatformInfo`, returns a `CheckResult`.
Results include a status, message, and platform-specific fix hint. The fix
hint is rendered verbatim in the doctor output — it should be copy-pasteable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from maxim.doctor.platform_detect import PlatformInfo, detect_wsl_ip

logger = logging.getLogger(__name__)

Status = Literal["ok", "warn", "fail", "info"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    fix: str | None = None  # platform-specific fix hint
    retry_id: str | None = None  # only set when retry is meaningful

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "⚠", "fail": "✗", "info": "ℹ"}[self.status]


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
    vram_gb = props.total_memory / (1024**3)
    return CheckResult(
        name="GPU / CUDA",
        status="ok",
        message=f"{props.name} ({vram_gb:.1f} GB VRAM)",
    )


def check_tier_detection(caps=None) -> CheckResult:
    """Report which capability tiers are available on this hardware."""
    try:
        from maxim.runtime.capabilities import RuntimeCapabilities, detect_compute_resources
        from maxim.runtime.lane_models import detect_tiers

        if caps is None:
            has_gpu, gpu_type, vram_gb, ram_gb = detect_compute_resources()
            caps = RuntimeCapabilities(
                has_gpu=has_gpu,
                gpu_type=gpu_type,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
            )
        tiers = detect_tiers(caps)
    except ImportError as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection unavailable: {e}",
            fix="pip install -e '.[llm-llama]'  # or install pymaxim[llm-llama]",
        )
    except Exception as e:
        return CheckResult(
            name="LLM Tiers",
            status="warn",
            message=f"Tier detection failed: {e}",
        )

    tier_names = sorted(tiers.keys())
    profiles = {name: cfg.model_profile for name, cfg in tiers.items()}

    if "large" in tiers or "medium" in tiers:
        return CheckResult(
            name="LLM Tiers",
            status="ok",
            message=f"Tiers: {', '.join(tier_names)}. Profiles: {profiles}",
        )
    return CheckResult(
        name="LLM Tiers",
        status="warn",
        message=f"Only 'small' tier detected ({caps.ram_gb:.0f}GB RAM, GPU: {caps.gpu_type or 'none'})",
        fix=(
            "Agent inference needs a large or medium tier. Options:\n"
            "  --language-model mistral-7b          # if you have 8+ GB RAM\n"
            "  --cloud-fallback claude-sonnet       # use cloud for inference\n"
            "  --tier-model large=<remote-url>      # point to a remote leader"
        ),
        retry_id="tier_detection",
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


def check_llm_model_active() -> CheckResult:
    """Check if an LLM model is loaded and report which one."""
    try:
        from maxim.runtime.lane_backends import _active_model, _read_persisted_model

        active = _active_model
        persisted = _read_persisted_model()

        if active:
            return CheckResult(
                name="LLM model",
                status="ok",
                message=f"active model: {active}",
            )
        if persisted:
            return CheckResult(
                name="LLM model",
                status="warn",
                message=f"persisted model: {persisted} (not yet loaded)",
                fix=f"maxim peer llm {persisted}",
                retry_id="llm_model",
            )
        return CheckResult(
            name="LLM model",
            status="warn",
            message="no model configured — use `maxim peer llm <model>` to set one",
            fix="maxim peer llm qwen2.5-14b",
            retry_id="llm_model",
        )
    except Exception:
        return CheckResult(
            name="LLM model",
            status="info",
            message="model tracking not available (solo mode)",
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
                "WSL2 port-forwarding required for LAN peers"
                if not leader
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
            "rhel": f"sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload",
            "arch": f"sudo ufw allow {port}/tcp  # (or your preferred firewall)",
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
            (
                "linux",
                "ubuntu",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "debian",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
            "  sudo dpkg -i cloudflared.deb",
            (
                "linux",
                "fedora",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            (
                "linux",
                "rhel",
            ): "curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
            "  sudo rpm -i cloudflared.rpm",
            ("linux", "arch"): "yay -S cloudflared  # or your preferred AUR helper",
            ("macos", "unknown"): "brew install cloudflare/cloudflare/cloudflared",
            (
                "windows",
                "unknown",
            ): "Download https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe\n"
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
    service = summary.get("service", "")

    # Warn if tunnel points directly at llama-cpp-server (8100) instead of
    # LeaderProxy (8099). Without the proxy, peers bypass auth enforcement,
    # logging, GPU metrics, admission control, and admin endpoints.
    if ":8100" in service:
        return CheckResult(
            name="Tunnel config",
            status="warn",
            message=f"hostname={hostname} — tunnel points at port 8100 (llama-cpp-server directly)",
            fix=(
                "Update your tunnel config to route through the LeaderProxy (port 8099):\n"
                f"  Edit {CONFIG_PATH}\n"
                "  Change: service: http://localhost:8100\n"
                "  To:     service: http://localhost:8099\n"
                "  Then restart cloudflared:\n"
                "    sudo systemctl restart cloudflared   # Linux\n"
                "    cloudflared --config ~/.cloudflared/config.yml tunnel run   # manual"
            ),
        )

    return CheckResult(
        name="Tunnel config",
        status="ok",
        message=f"hostname={hostname} service={service}",
    )


def check_tunnel_config_sync() -> CheckResult:
    """Warn if the systemd cloudflared config differs from the user config.

    When cloudflared is installed as a systemd service, it reads from
    /etc/cloudflared/config.yml — NOT ~/.cloudflared/config.yml. After
    editing or regenerating the user config, the systemd copy must be
    updated too, or the tunnel will use stale settings.
    """
    import platform

    if platform.system() != "Linux":
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="skipped (non-Linux)",
        )

    from pathlib import Path

    user_cfg = Path.home() / ".cloudflared" / "config.yml"
    system_cfg = Path("/etc/cloudflared/config.yml")

    if not user_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no user config — nothing to compare",
        )
    if not system_cfg.is_file():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="no systemd config — cloudflared may not be a service",
        )

    try:
        user_text = user_cfg.read_text()
        system_text = system_cfg.read_text()
    except PermissionError:
        return CheckResult(
            name="Tunnel config sync",
            status="warn",
            message="cannot read /etc/cloudflared/config.yml (permission denied)",
            fix="Run: sudo cat /etc/cloudflared/config.yml",
        )

    if user_text.strip() == system_text.strip():
        return CheckResult(
            name="Tunnel config sync",
            status="ok",
            message="~/.cloudflared/config.yml and /etc/cloudflared/config.yml are in sync",
        )

    return CheckResult(
        name="Tunnel config sync",
        status="warn",
        message=(
            "~/.cloudflared/config.yml and /etc/cloudflared/config.yml DIFFER — "
            "systemd service may be using stale settings"
        ),
        fix=(
            "The systemd cloudflared service reads /etc/cloudflared/config.yml,\n"
            "not ~/.cloudflared/config.yml. After editing the user config, sync it:\n"
            "  sudo cp ~/.cloudflared/config.yml /etc/cloudflared/config.yml\n"
            "  sudo systemctl restart cloudflared"
        ),
        retry_id="tunnel-config-sync",
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


def check_key_age() -> CheckResult:
    """Warn when the API key file is older than 90 days."""
    import time

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key age",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        age_days = (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"cannot stat {path}",
        )
    if age_days > 90:
        return CheckResult(
            name="Key age",
            status="warn",
            message=f"key is {age_days:.0f} days old — consider rotating",
            fix="maxim tunnel key rotate && maxim tunnel key export",
            retry_id="key-age",
        )
    return CheckResult(
        name="Key age",
        status="ok",
        message=f"key is {age_days:.0f} days old",
    )


def check_key_permissions() -> CheckResult:
    """Fail if the key file is world-readable (POSIX only)."""
    import platform
    import stat

    if platform.system() == "Windows":
        return CheckResult(
            name="Key permissions",
            status="ok",
            message="skipped (Windows uses ACLs)",
        )

    from maxim.tunnel.keys import key_exists, key_file_path

    if not key_exists():
        return CheckResult(
            name="Key permissions",
            status="info",
            message="no key file — skipped",
        )
    path = key_file_path()
    try:
        mode = path.stat().st_mode
    except OSError:
        return CheckResult(
            name="Key permissions",
            status="warn",
            message=f"cannot stat {path}",
        )
    # Check group/other read bits
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        return CheckResult(
            name="Key permissions",
            status="fail",
            message=f"key file is world-readable (mode {oct(mode & 0o777)})",
            fix=f"chmod 600 {path}",
            retry_id="key-permissions",
        )
    return CheckResult(
        name="Key permissions",
        status="ok",
        message=f"mode {oct(mode & 0o777)}",
    )


def check_key_auth_smoke(port: int = 8100) -> CheckResult:
    """Verify the local server enforces the configured API key."""
    import json
    import urllib.error
    import urllib.request

    from maxim.tunnel.keys import key_exists, read_key

    if not key_exists():
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="no key configured — skipped",
        )
    key = read_key() or ""
    url = f"http://127.0.0.1:{port}/v1/models"
    headers = {"User-Agent": "maxim-doctor/1.0", "Authorization": f"Bearer {key}"}
    # Test with correct key
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
            json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return CheckResult(
                name="Key auth smoke",
                status="fail",
                message="server rejected our key (401)",
                fix="Key mismatch — regenerate: maxim tunnel key rotate",
                retry_id="key-auth",
            )
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"server returned HTTP {e.code}",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="info",
            message="server not reachable — skipped",
        )
    # Test with bogus key — should get 401 if auth is enforced
    bogus_headers = {"User-Agent": "maxim-doctor/1.0", "Authorization": "Bearer BOGUS"}
    try:
        req = urllib.request.Request(url, headers=bogus_headers)
        with urllib.request.urlopen(req, timeout=3) as resp:  # noqa: S310
            resp.read()
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message="server accepts ANY key — auth not enforced",
            fix=(
                "The server accepts requests without verifying the API key.\n"
                "  Fine for localhost, but insecure if exposed via tunnel.\n"
                "  Route through LeaderProxy (port 8099) to enforce auth."
            ),
        )
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return CheckResult(
                name="Key auth smoke",
                status="ok",
                message="auth enforced — valid key accepted, bogus key rejected",
            )
        return CheckResult(
            name="Key auth smoke",
            status="warn",
            message=f"bogus-key probe returned HTTP {e.code} (expected 401)",
        )
    except Exception:
        return CheckResult(
            name="Key auth smoke",
            status="ok",
            message="valid key accepted (bogus-key probe inconclusive)",
        )


# ─── disk + memory ────────────────────────────────────────────────────────


def check_disk_space() -> CheckResult:
    """Warn when free disk space is low."""
    import shutil
    from pathlib import Path

    from maxim.utils.paths import data_home

    try:
        check_path = data_home()
    except Exception:
        check_path = Path.home()
    try:
        usage = shutil.disk_usage(check_path)
    except OSError as e:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"cannot check: {e}",
        )
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    if free_gb < 2:
        return CheckResult(
            name="Disk space",
            status="fail",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — critically low",
            fix="Free disk space. Sim reports, model files, and logs consume storage over time.",
        )
    if free_gb < 10:
        return CheckResult(
            name="Disk space",
            status="warn",
            message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB — model downloads may fail",
        )
    return CheckResult(
        name="Disk space",
        status="ok",
        message=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
    )


def check_ram_headroom() -> CheckResult:
    """Report available system RAM."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
    except ImportError:
        # Fallback: read /proc/meminfo on Linux, sysctl on macOS
        import platform

        if platform.system() == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                avail_gb = info.get("MemAvailable", 0) / (1024**2)
                total_gb = info.get("MemTotal", 0) / (1024**2)
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read /proc/meminfo")
        elif platform.system() == "Darwin":
            import subprocess

            try:
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=5, text=True)
                total_gb = int(out.strip()) / (1024**3)
                # macOS doesn't expose "available" easily without psutil
                avail_gb = -1
            except Exception:
                return CheckResult(name="RAM", status="info", message="cannot read sysctl")
        else:
            return CheckResult(
                name="RAM",
                status="info",
                message="install psutil for RAM check: pip install psutil",
            )

    if avail_gb < 0:
        return CheckResult(
            name="RAM",
            status="ok",
            message=f"{total_gb:.1f} GB total (install psutil for available RAM)",
        )
    if avail_gb < 2:
        return CheckResult(
            name="RAM",
            status="warn",
            message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB — tight for LLM inference",
            fix="Close other applications or choose a smaller model (e.g., smollm-1.7b)",
        )
    return CheckResult(
        name="RAM",
        status="ok",
        message=f"{avail_gb:.1f} GB available of {total_gb:.1f} GB",
    )


# ─── inference coherence ──────────────────────────────────────────────────


def check_inference_coherence(port: int = 8100) -> CheckResult:
    """Send a fixed prompt and verify the LLM returns a sensible answer."""
    import json
    import time
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 8,
        "temperature": 0.0,
    }
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "User-Agent": "maxim-doctor/1.0"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read())
        dt_ms = (time.time() - t0) * 1000
    except Exception:
        return CheckResult(
            name="Inference coherence",
            status="info",
            message="server not reachable — skipped",
        )

    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError):
        return CheckResult(
            name="Inference coherence",
            status="fail",
            message="response has unexpected structure",
        )

    # Check if response contains "4" somewhere
    if "4" in text:
        return CheckResult(
            name="Inference coherence",
            status="ok",
            message=f"correct ({dt_ms:.0f} ms): {text!r}",
        )
    return CheckResult(
        name="Inference coherence",
        status="warn",
        message=f"unexpected answer ({dt_ms:.0f} ms): {text!r} — model may be misconfigured",
    )


# ─── peer-mode checks ────────────────────────────────────────────────────


def check_peer_url_reachable(url: str) -> CheckResult:
    """Resolve DNS and probe /v1/models on a remote leader."""
    import socket
    import time
    import urllib.error
    import urllib.parse
    import urllib.request

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or ""
    if not host:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"no hostname in URL: {url}",
        )
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"DNS resolution failed for {host}",
            fix=f"Check the hostname. Is the leader URL correct?\n  URL: {url}",
            retry_id="peer-url",
        )
    # HTTP probe
    models_url = url.rstrip("/")
    if not models_url.endswith("/v1"):
        models_url += "/v1"
    models_url += "/models"
    try:
        req = urllib.request.Request(models_url, headers={"User-Agent": "maxim-doctor/1.0"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            resp.read()
        dt_ms = (time.time() - t0) * 1000
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return CheckResult(
                name="Remote URL",
                status="ok",
                message=f"{host} reachable (auth required — checked separately)",
            )
        return CheckResult(
            name="Remote URL",
            status="warn",
            message=f"{host} returned HTTP {e.code}",
            retry_id="peer-url",
        )
    except Exception as e:
        return CheckResult(
            name="Remote URL",
            status="fail",
            message=f"cannot reach {host}: {e}",
            fix=f"Is the leader running? Try: curl -v {models_url}",
            retry_id="peer-url",
        )
    return CheckResult(
        name="Remote URL",
        status="ok",
        message=f"{host} reachable ({dt_ms:.0f} ms)",
    )


def check_peer_key_set() -> CheckResult:
    """Check that the peer has a remote API key configured."""
    import os

    key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
    if not key:
        try:
            from maxim.peer.config import read_peer_config

            cfg = read_peer_config()
            if cfg is not None:
                key = cfg.api_key
        except Exception as e:
            logger.debug("Could not read peer config for API key check: %s", e)
    if not key:
        return CheckResult(
            name="Peer API key",
            status="warn",
            message="no API key configured for remote access",
            fix=(
                "Get the key from the leader:\n"
                "  On leader: maxim tunnel key export\n"
                "  Then paste the export snippet here."
            ),
            retry_id="peer-key",
        )
    from maxim.tunnel.keys import truncate_for_display

    return CheckResult(
        name="Peer API key",
        status="ok",
        message=f"key set: {truncate_for_display(key)}",
    )


def check_peer_auth(url: str, key: str | None) -> CheckResult:
    """Send an authenticated request to the leader and verify 200."""
    import json
    import urllib.error
    import urllib.request

    if not key:
        return CheckResult(
            name="Peer auth",
            status="info",
            message="no key — skipped",
        )
    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    headers = {
        "User-Agent": "maxim-doctor/1.0",
        "Authorization": f"Bearer {key}",
    }
    try:
        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return CheckResult(
                name="Peer auth",
                status="fail",
                message="key rejected by leader (401)",
                fix=(
                    "Key mismatch. Ask the leader to regenerate:\n"
                    "  On leader: maxim tunnel key rotate\n"
                    "  Then: maxim tunnel key export"
                ),
                retry_id="peer-auth",
            )
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"leader returned HTTP {e.code}",
        )
    except Exception as e:
        return CheckResult(
            name="Peer auth",
            status="warn",
            message=f"connection failed: {e}",
        )
    return CheckResult(
        name="Peer auth",
        status="ok",
        message="authenticated successfully",
    )


def check_peer_model(url: str, key: str | None, expected_model: str | None = None) -> CheckResult:
    """Check if the leader advertises the expected model."""
    import json
    import urllib.error
    import urllib.request

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    models_url = f"{base}/models"
    headers = {"User-Agent": "maxim-doctor/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            data = json.loads(resp.read())
        model_ids = [m.get("id", "") for m in data.get("data", [])]
    except Exception:
        return CheckResult(
            name="Remote model",
            status="info",
            message="cannot query /v1/models — skipped",
        )
    if not model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message="leader reports no models loaded",
            fix="On the leader, load a model: maxim peer llm mistral-7b",
        )
    if expected_model and expected_model not in model_ids:
        return CheckResult(
            name="Remote model",
            status="warn",
            message=f"expected {expected_model!r}, leader has: {', '.join(model_ids)}",
            fix=f"On leader: maxim peer llm {expected_model}",
        )
    return CheckResult(
        name="Remote model",
        status="ok",
        message=f"leader model: {', '.join(model_ids)}",
    )


def check_peer_latency(url: str, key: str | None) -> CheckResult:
    """Measure round-trip latency to the leader (5 pings, report p50)."""
    import time
    import urllib.error
    import urllib.request

    base = url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    ping_url = f"{base}/models"
    headers = {"User-Agent": "maxim-doctor/1.0"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    timings: list[float] = []
    for probe_idx in range(5):
        try:
            req = urllib.request.Request(ping_url, headers=headers)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
                resp.read()
            timings.append((time.time() - t0) * 1000)
        except Exception as e:
            logger.debug("Latency probe %d failed: %s", probe_idx, e)
    if not timings:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message="all latency probes failed",
        )
    timings.sort()
    p50 = timings[len(timings) // 2]
    p95 = timings[int(len(timings) * 0.95)] if len(timings) >= 2 else timings[-1]
    if p50 > 200:
        return CheckResult(
            name="Peer latency",
            status="warn",
            message=f"p50={p50:.0f} ms, p95={p95:.0f} ms — high latency may slow real-time inference",
        )
    return CheckResult(
        name="Peer latency",
        status="ok",
        message=f"p50={p50:.0f} ms, p95={p95:.0f} ms ({len(timings)}/5 probes)",
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


def _detect_doctor_role(explicit: str | None = None, peer_url: str | None = None) -> tuple[str, str | None]:
    """Detect whether this machine is peer/leader/solo for doctor purposes.

    Returns ``(role, peer_url)`` where role is ``"peer"``, ``"leader"``,
    ``"solo"``, or ``"auto"`` (fall through to existing behaviour).
    """
    import os
    from urllib.parse import urlparse

    if explicit == "peer":
        url = peer_url or os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
        return "peer", url
    if explicit in ("leader", "solo"):
        return explicit, None
    # Auto-detect from env
    url = os.environ.get("MAXIM_LANE_LARGE_REMOTE_URL")
    if url:
        host = urlparse(url).hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            return "peer", url
    return "auto", None


def run_all_checks(
    info: PlatformInfo,
    *,
    role: str | None = None,
    peer_url: str | None = None,
    peer_key: str | None = None,
) -> list[tuple[str, list[CheckResult]]]:
    """Return ordered ``[(section_name, results)]`` for ``maxim doctor``.

    Parameters
    ----------
    role
        Explicit role override: ``"peer"``, ``"leader"``, ``"solo"``.
        ``None`` triggers auto-detection.
    peer_url
        Remote leader URL (used when role is ``"peer"``).
    peer_key
        API key for the remote leader.
    """
    detected_role, detected_url = _detect_doctor_role(role, peer_url)
    peer_url = peer_url or detected_url

    # ── environment (always) ──────────────────────────────────────────────
    sections: list[tuple[str, list[CheckResult]]] = [
        (
            "Environment",
            [
                CheckResult(name="Platform", status="ok", message=info.display_name),
                CheckResult(name="Architecture", status="ok", message=info.arch),
                check_gpu(),
                check_tier_detection(),
                check_disk_space(),
                check_ram_headroom(),
            ],
        ),
    ]

    if detected_role == "peer" and peer_url:
        # ── peer-mode sections ────────────────────────────────────────────
        # Resolve key from arg → env → peer config
        if not peer_key:
            import os

            peer_key = os.environ.get("MAXIM_LANE_LARGE_REMOTE_API_KEY")
        if not peer_key:
            try:
                from maxim.peer.config import read_peer_config

                cfg = read_peer_config()
                if cfg is not None:
                    peer_key = cfg.api_key
            except Exception as e:
                logger.debug("Could not read peer config in run_all_checks: %s", e)
        sections.append(
            (
                "Peer Connectivity",
                [
                    check_peer_url_reachable(peer_url),
                    check_peer_key_set(),
                    check_peer_auth(peer_url, peer_key),
                    check_peer_model(peer_url, peer_key),
                    check_peer_latency(peer_url, peer_key),
                ],
            )
        )
    else:
        # ── leader / solo sections ────────────────────────────────────────
        sections += [
            (
                "Local LLM",
                [
                    check_llama_cpp_server_installed(),
                    check_server_reachable(),
                    check_llm_model_active(),
                    check_inference_coherence(),
                ],
            ),
            (
                "Role & Access",
                [
                    check_role(),
                    check_lan_access(info),
                ],
            ),
            (
                "Tunnel (Cloudflare)",
                [
                    check_cloudflared(info),
                    check_tunnel_config(),
                    check_tunnel_config_sync(),
                ],
            ),
            (
                "API key",
                [
                    check_api_key(),
                    check_key_age(),
                    check_key_permissions(),
                    check_key_auth_smoke(),
                ],
            ),
        ]

    # Lane metrics (only show if any calls have been recorded)
    lane_results = _check_lane_metrics()
    if lane_results:
        sections.append(("Lane Metrics", lane_results))
    return sections


__all__ = [
    "CheckResult",
    "Status",
    "check_gpu",
    "check_tier_detection",
    "check_llama_cpp_server_installed",
    "check_server_reachable",
    "check_llm_model_active",
    "check_lan_access",
    "check_cloudflared",
    "check_tunnel_config",
    "check_tunnel_config_sync",
    "check_api_key",
    "check_key_age",
    "check_key_permissions",
    "check_key_auth_smoke",
    "check_disk_space",
    "check_ram_headroom",
    "check_inference_coherence",
    "check_peer_url_reachable",
    "check_peer_key_set",
    "check_peer_auth",
    "check_peer_model",
    "check_peer_latency",
    "check_role",
    "run_all_checks",
]
