"""System-wide metrics collector — CPU, RAM, disk, GPU, network, WiFi.

Stdlib-only (no psutil dependency). Uses platform-specific commands with
graceful fallbacks when a metric source is unavailable. Each collector
function returns a dict or None, never raises.

Used by HeartbeatMonitor for periodic sampling and by LeaderProxy for
the /v1/debug/heartbeat endpoint.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def collect_all() -> dict[str, Any]:
    """Collect all available system metrics. Never raises."""
    return {
        "timestamp": time.time(),
        "gpu": collect_gpu(),
        "cpu": collect_cpu(),
        "memory": collect_memory(),
        "disk": collect_disk(),
        "network": collect_network_interfaces(),
        "wifi": collect_wifi_signal(),
        "platform": collect_platform(),
    }


# ─── GPU (nvidia-smi) ────────────────────────────────────────────────────

def collect_gpu() -> dict[str, Any] | None:
    """Query nvidia-smi for GPU metrics."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw,clocks.current.sm,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            return None
        return {
            "utilization_pct": _float(parts[0]),
            "vram_used_gb": round(_float(parts[1]) / 1024, 2),
            "vram_total_gb": round(_float(parts[2]) / 1024, 2),
            "temperature_c": _float(parts[3]),
            "power_draw_w": _float(parts[4]),
            "clock_mhz": _float(parts[5]),
            "name": parts[6].strip(),
        }
    except Exception:
        return None


# ─── CPU (stdlib /proc/stat or sysctl) ───────────────────────────────────

def collect_cpu() -> dict[str, Any] | None:
    """CPU usage percentage + load average."""
    try:
        load_1, load_5, load_15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        result: dict[str, Any] = {
            "load_1m": round(load_1, 2),
            "load_5m": round(load_5, 2),
            "load_15m": round(load_15, 2),
            "cores": cpu_count,
            "usage_pct": round(load_1 / cpu_count * 100, 1),
        }
        # Try /proc/stat for more precise instantaneous usage (Linux)
        stat = _read_proc_stat()
        if stat is not None:
            result["usage_pct"] = stat
        return result
    except Exception:
        return None


def _read_proc_stat() -> float | None:
    """Read /proc/stat for instantaneous CPU usage (Linux only)."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if parts[0] != "cpu":
            return None
        vals = [int(x) for x in parts[1:8]]
        idle = vals[3]
        total = sum(vals)
        # Need two samples — use a cached previous reading
        prev = getattr(_read_proc_stat, "_prev", None)
        _read_proc_stat._prev = (idle, total)  # type: ignore[attr-defined]
        if prev is None:
            return None
        d_idle = idle - prev[0]
        d_total = total - prev[1]
        if d_total == 0:
            return 0.0
        return round((1.0 - d_idle / d_total) * 100, 1)
    except Exception:
        return None


# ─── Memory (stdlib) ─────────────────────────────────────────────────────

def collect_memory() -> dict[str, Any] | None:
    """RAM usage. Linux: /proc/meminfo. macOS: vm_stat."""
    try:
        if platform.system() == "Linux":
            return _memory_linux()
        if platform.system() == "Darwin":
            return _memory_macos()
        return None
    except Exception:
        return None


def _memory_linux() -> dict[str, Any] | None:
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # kB
        total_gb = round(info.get("MemTotal", 0) / 1048576, 2)
        available_gb = round(info.get("MemAvailable", 0) / 1048576, 2)
        used_gb = round(total_gb - available_gb, 2)
        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "available_gb": available_gb,
            "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
        }
    except Exception:
        return None


def _memory_macos() -> dict[str, Any] | None:
    try:
        import sysctl_fallback  # noqa: F401 — not a real import, use subprocess
    except ImportError:
        pass
    try:
        # Total memory via sysctl
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        total_bytes = int(result.stdout.strip())
        total_gb = round(total_bytes / (1024 ** 3), 2)

        # vm_stat for page-level usage
        result = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=2,
        )
        pages: dict[str, int] = {}
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip().rstrip(".")
                try:
                    pages[key] = int(val)
                except ValueError:
                    pass

        page_size = 16384  # Apple Silicon default
        # Try to get actual page size
        try:
            ps_result = subprocess.run(
                ["sysctl", "-n", "hw.pagesize"],
                capture_output=True, text=True, timeout=2,
            )
            page_size = int(ps_result.stdout.strip())
        except Exception:
            pass

        free_pages = pages.get("Pages free", 0)
        inactive_pages = pages.get("Pages inactive", 0)
        available_gb = round((free_pages + inactive_pages) * page_size / (1024 ** 3), 2)
        used_gb = round(total_gb - available_gb, 2)

        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "available_gb": available_gb,
            "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
        }
    except Exception:
        return None


# ─── Disk ─────────────────────────────────────────────────────────────────

def collect_disk(path: str | None = None) -> dict[str, Any] | None:
    """Disk usage for the Maxim data directory."""
    try:
        if path is None:
            # Default: check the project root / data directory
            path = str(Path(__file__).resolve().parents[3])
        usage = shutil.disk_usage(path)
        return {
            "path": path,
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
            "usage_pct": round(usage.used / usage.total * 100, 1),
        }
    except Exception:
        return None


# ─── Network interfaces ──────────────────────────────────────────────────

def collect_network_interfaces() -> dict[str, Any] | None:
    """Basic network interface info — active interfaces + IPs."""
    try:
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "unknown"
        return {
            "hostname": hostname,
            "local_ip": local_ip,
        }
    except Exception:
        return None


# ─── WiFi signal ──────────────────────────────────────────────────────────

def collect_wifi_signal() -> dict[str, Any] | None:
    """WiFi signal strength + SSID. macOS: airport. Linux: iwconfig."""
    try:
        if platform.system() == "Darwin":
            return _wifi_macos()
        if platform.system() == "Linux":
            return _wifi_linux()
        return None
    except Exception:
        return None


def _wifi_macos() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "/System/Library/PrivateFrameworks/Apple80211.framework"
                "/Versions/Current/Resources/airport",
                "-I",
            ],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        info: dict[str, str] = {}
        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                info[key.strip()] = val.strip()
        rssi = _float(info.get("agrCtlRSSI", ""))
        noise = _float(info.get("agrCtlNoise", ""))
        return {
            "ssid": info.get("SSID", "unknown"),
            "rssi_dbm": rssi if rssi != 0 else None,
            "noise_dbm": noise if noise != 0 else None,
            "snr_db": round(rssi - noise, 1) if rssi and noise else None,
            "channel": info.get("channel"),
            "tx_rate_mbps": info.get("lastTxRate"),
        }
    except Exception:
        return None


def _wifi_linux() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["iwconfig"], capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            return None
        output = result.stdout + result.stderr
        ssid = None
        signal = None
        for line in output.split("\n"):
            if "ESSID:" in line:
                ssid = line.split("ESSID:")[1].strip().strip('"')
            if "Signal level=" in line:
                sig_part = line.split("Signal level=")[1].split()[0]
                signal = _float(sig_part)
        if ssid is None:
            return None
        return {
            "ssid": ssid,
            "rssi_dbm": signal,
        }
    except Exception:
        return None


# ─── Platform ─────────────────────────────────────────────────────────────

def collect_platform() -> dict[str, Any]:
    """Static platform info."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "pid": os.getpid(),
    }


# ─── helpers ──────────────────────────────────────────────────────────────

def _float(val: Any) -> float:
    """Best-effort float conversion, returns 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


__all__ = [
    "collect_all",
    "collect_gpu",
    "collect_cpu",
    "collect_memory",
    "collect_disk",
    "collect_network_interfaces",
    "collect_wifi_signal",
    "collect_platform",
]
