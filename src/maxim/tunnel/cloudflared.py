"""Subprocess wrappers around the `cloudflared` binary.

cloudflared is a Go binary installed via OS package manager — we can't vendor
it via pip. These helpers handle detection, install hints, and running the
tunnel-management commands on the user's behalf.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class CloudflaredNotInstalled(RuntimeError):
    """Raised when the cloudflared binary can't be located."""


@dataclass(frozen=True)
class CloudflaredInfo:
    binary_path: str
    version: str


def find_cloudflared() -> str | None:
    """Return the path to the cloudflared binary, or None if not installed."""
    return shutil.which("cloudflared")


def install_hint() -> str:
    """OS-specific install command for cloudflared."""
    system = platform.system().lower()
    if system == "linux":
        # Try to detect the distro for a better hint
        try:
            with open("/etc/os-release") as f:
                os_release = f.read().lower()
            if "ubuntu" in os_release or "debian" in os_release:
                return (
                    "Ubuntu/Debian:\n"
                    "  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb\n"
                    "  sudo dpkg -i cloudflared.deb"
                )
            if "fedora" in os_release or "rhel" in os_release or "centos" in os_release:
                return (
                    "Fedora/RHEL:\n"
                    "  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-x86_64.rpm -o cloudflared.rpm\n"
                    "  sudo rpm -i cloudflared.rpm"
                )
        except OSError:
            pass
        return (
            "Linux: download from https://github.com/cloudflare/cloudflared/releases/latest\n"
            "  (pick the package for your distribution)"
        )
    if system == "darwin":
        return "macOS:  brew install cloudflare/cloudflare/cloudflared"
    if system == "windows":
        return (
            "Windows: download from https://github.com/cloudflare/cloudflared/releases/latest\n"
            "  (pick cloudflared-windows-amd64.exe)"
        )
    return "See https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"


def require_cloudflared() -> str:
    """Return the binary path, or raise CloudflaredNotInstalled with a hint."""
    path = find_cloudflared()
    if path is None:
        raise CloudflaredNotInstalled(
            "cloudflared binary not found on PATH.\n\n"
            "Install with:\n" + install_hint() + "\n\n"
            "Then re-run `maxim tunnel setup`."
        )
    return path


def cloudflared_version() -> str | None:
    """Return cloudflared's version string, or None if it can't be probed."""
    path = find_cloudflared()
    if path is None:
        return None
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip().splitlines()[0] if result.stdout else None


def run_tunnel_login() -> int:
    """Run `cloudflared tunnel login` — opens a browser for OAuth."""
    path = require_cloudflared()
    # Inherit stdout/stderr so user sees the URL cloudflared prints.
    return subprocess.call([path, "tunnel", "login"])


def run_tunnel_create(tunnel_name: str) -> tuple[int, str]:
    """Run `cloudflared tunnel create <name>`. Returns (exit_code, output)."""
    path = require_cloudflared()
    result = subprocess.run(
        [path, "tunnel", "create", tunnel_name],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return result.returncode, (result.stdout + result.stderr)


def run_tunnel_route_dns(tunnel_name: str, hostname: str) -> tuple[int, str]:
    """Route a hostname to the named tunnel."""
    path = require_cloudflared()
    result = subprocess.run(
        [path, "tunnel", "route", "dns", tunnel_name, hostname],
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    return result.returncode, (result.stdout + result.stderr)


def list_tunnels() -> tuple[int, str]:
    """List the user's tunnels."""
    path = require_cloudflared()
    result = subprocess.run(
        [path, "tunnel", "list"],
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    return result.returncode, (result.stdout + result.stderr)


def find_tunnel_credentials(tunnel_name: str) -> Path | None:
    """Find the .json credentials file for a tunnel in ~/.cloudflared/.

    cloudflared names credentials files <tunnel-uuid>.json. We match by
    inspecting `cloudflared tunnel list` output.
    """
    cfg_dir = Path.home() / ".cloudflared"
    if not cfg_dir.is_dir():
        return None
    code, output = list_tunnels()
    if code != 0:
        return None
    # The list output lines look like: "UUID  NAME  CREATED  CONNECTIONS"
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        uuid_candidate, name_candidate = parts[0], parts[1]
        if name_candidate == tunnel_name:
            cred_file = cfg_dir / f"{uuid_candidate}.json"
            if cred_file.is_file():
                return cred_file
    return None


__all__ = [
    "CloudflaredNotInstalled",
    "CloudflaredInfo",
    "find_cloudflared",
    "install_hint",
    "require_cloudflared",
    "cloudflared_version",
    "run_tunnel_login",
    "run_tunnel_create",
    "run_tunnel_route_dns",
    "list_tunnels",
    "find_tunnel_credentials",
]
