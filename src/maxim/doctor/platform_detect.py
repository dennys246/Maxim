"""Platform detection — OS, runtime environment, Linux distro.

Captures enough detail to give platform-specific fix suggestions
(e.g., `apt install cloudflared` vs `brew install`, WSL2 netsh commands).
"""
from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

OSName = Literal["linux", "macos", "windows", "unknown"]
Runtime = Literal["native", "wsl1", "wsl2", "docker", "unknown"]
Distro = Literal["ubuntu", "debian", "fedora", "rhel", "arch", "alpine", "other", "unknown"]


@dataclass(frozen=True)
class PlatformInfo:
    os: OSName
    os_release: str           # e.g., "Ubuntu 24.04", "macOS 14.2", "Windows 11"
    runtime: Runtime
    distro: Distro            # Linux only; "unknown" on macOS/Windows
    arch: str                 # e.g., "x86_64", "aarch64", "arm64"
    kernel_version: str
    windows_host_ip: str | None  # only set when runtime=wsl1/wsl2

    @property
    def is_wsl(self) -> bool:
        return self.runtime in ("wsl1", "wsl2")

    @property
    def display_name(self) -> str:
        """Human-readable platform name for diagnostic output."""
        if self.runtime == "wsl2":
            return f"WSL2 ({self.os_release}) on Windows"
        if self.runtime == "wsl1":
            return f"WSL1 ({self.os_release}) on Windows"
        if self.runtime == "docker":
            return f"Docker ({self.os_release})"
        return self.os_release


def detect_platform() -> PlatformInfo:
    system = platform.system().lower()
    os_name: OSName
    if system == "linux":
        os_name = "linux"
    elif system == "darwin":
        os_name = "macos"
    elif system == "windows":
        os_name = "windows"
    else:
        os_name = "unknown"

    runtime = _detect_runtime(os_name)
    distro = _detect_distro() if os_name == "linux" else "unknown"
    os_release = _detect_os_release(os_name, distro)
    kernel = platform.release()
    arch = platform.machine()
    host_ip = _detect_windows_host_ip() if runtime in ("wsl1", "wsl2") else None

    return PlatformInfo(
        os=os_name,
        os_release=os_release,
        runtime=runtime,
        distro=distro,
        arch=arch,
        kernel_version=kernel,
        windows_host_ip=host_ip,
    )


def _detect_runtime(os_name: OSName) -> Runtime:
    if os_name != "linux":
        return "native"
    # WSL indicators
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSLENV"):
        # WSL2 sets WSL_DISTRO_NAME; differentiate from WSL1 via kernel
        kernel = platform.release().lower()
        if "microsoft-standard" in kernel or "wsl2" in kernel:
            return "wsl2"
        if "microsoft" in kernel:
            return "wsl1"
        return "wsl2"  # Default to WSL2 when env is set but kernel ambiguous
    try:
        with open("/proc/version") as f:
            content = f.read().lower()
        if "microsoft" in content or "wsl" in content:
            return "wsl2" if "wsl2" in content else "wsl1"
    except OSError:
        pass
    # Docker indicator
    if Path("/.dockerenv").exists():
        return "docker"
    return "native"


def _detect_distro() -> Distro:
    try:
        with open("/etc/os-release") as f:
            content = f.read().lower()
    except OSError:
        return "unknown"
    if "ubuntu" in content:
        return "ubuntu"
    if "debian" in content:
        return "debian"
    if "fedora" in content:
        return "fedora"
    if "rhel" in content or "red hat" in content or "centos" in content:
        return "rhel"
    if "arch" in content or "manjaro" in content:
        return "arch"
    if "alpine" in content:
        return "alpine"
    return "other"


def _detect_os_release(os_name: OSName, distro: Distro) -> str:
    if os_name == "linux":
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return f"Linux ({distro})"
    if os_name == "macos":
        mac_ver = platform.mac_ver()[0] or ""
        return f"macOS {mac_ver}" if mac_ver else "macOS"
    if os_name == "windows":
        ver = platform.version()
        return f"Windows ({ver})"
    return "Unknown"


def _detect_windows_host_ip() -> str | None:
    """When running in WSL, find the Windows host's LAN IP.

    Strategy: run `ipconfig.exe` from inside WSL; parse the first non-loopback
    IPv4 address from "Wireless LAN adapter" or "Ethernet adapter" sections.
    Returns None on any failure.
    """
    try:
        result = subprocess.run(
            ["ipconfig.exe"],
            capture_output=True, text=True, timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    # Parse lines like: "   IPv4 Address. . . . . . . . . . . : 192.168.1.47"
    # Skip 127.x, 169.254.x, and the WSL virtual switch (usually 172.x/16)
    import re
    lines = result.stdout.splitlines()
    current_adapter = ""
    for line in lines:
        stripped = line.rstrip()
        if stripped and not stripped.startswith(" "):
            current_adapter = stripped.lower()
            continue
        match = re.search(r"IPv4.*?:\s*(\d+\.\d+\.\d+\.\d+)", stripped)
        if not match:
            continue
        ip = match.group(1)
        # Filter: skip WSL virtual switch + APIPA + loopback
        if ip.startswith(("127.", "169.254.")) or "wsl" in current_adapter or "vethernet" in current_adapter:
            continue
        return ip
    return None


def detect_wsl_ip() -> str | None:
    """From inside WSL, return the WSL2 distro's own IP."""
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    parts = result.stdout.strip().split()
    return parts[0] if parts else None


def detect_lan_ip() -> str | None:
    """Best-effort LAN IP discovery for native Linux/macOS."""
    # On Linux: hostname -I gives first non-loopback
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=2.0,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Fallback: socket connect trick
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except OSError:
        return None


__all__ = [
    "PlatformInfo",
    "OSName",
    "Runtime",
    "Distro",
    "detect_platform",
    "detect_wsl_ip",
    "detect_lan_ip",
]
