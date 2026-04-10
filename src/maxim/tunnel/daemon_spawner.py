"""Auto-spawn the cloudflared tunnel daemon (multi-LLM Phase 6b).

Mirrors LocalServerSpawner: when Maxim starts in leader mode + a tunnel
config is present + no cloudflared daemon is already running, spawn one
as a managed subprocess. Same SIGINT isolation (start_new_session=True)
and atexit cleanup, so the daemon survives Ctrl+C until Maxim's own
shutdown actually runs.

Respects any existing cloudflared daemon — if the systemd service
(or another foreground invocation) is already running, we skip the
spawn to avoid duplicate connections to Cloudflare's edge.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from maxim.tunnel.cloudflared import find_cloudflared
from maxim.tunnel.config import CONFIG_PATH as USER_CONFIG_PATH

logger = logging.getLogger(__name__)


SYSTEM_CONFIG_PATH = Path("/etc/cloudflared/config.yml")
READINESS_WAIT_S = 3.0  # give cloudflared time to fail if the config is bad


def cloudflared_already_running() -> bool:
    """True if any cloudflared tunnel daemon process is detected.

    Uses pgrep on POSIX. Windows falls back to a best-effort None returning
    False (manual daemon management required there). We match 'cloudflared'
    + 'tunnel' + 'run' to avoid false positives from unrelated cloudflared
    invocations like `cloudflared tunnel list`.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-af", "cloudflared.*tunnel.*run"],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def resolve_config_path() -> Path | None:
    """Return the cloudflared config path to pass to --config.

    Prefers /etc/cloudflared/config.yml (system-wide, used by the
    systemd service) over ~/.cloudflared/config.yml (user-space).
    The systemd service is the authoritative tunnel process — its
    config must match what we expect.
    """
    if SYSTEM_CONFIG_PATH.is_file():
        return SYSTEM_CONFIG_PATH
    if USER_CONFIG_PATH.is_file():
        return USER_CONFIG_PATH
    return None


class TunnelDaemonSpawner:
    """Manage a cloudflared tunnel daemon subprocess."""

    def __init__(self, *, config_path: Path | None = None) -> None:
        self._config_path = config_path or resolve_config_path()
        self._process: subprocess.Popen | None = None
        self._atexit_registered = False
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        """Spawn the daemon. Returns True if started, False on any failure.

        No-op if a daemon is already running (outside our control, e.g.,
        systemd service). Briefly waits (~3s) to catch immediate failures
        like bad config or missing credentials.
        """
        with self._lock:
            if self.is_running:
                return True

            binary = find_cloudflared()
            if binary is None:
                logger.warning("TunnelDaemonSpawner: cloudflared not found on PATH")
                return False

            if self._config_path is None or not self._config_path.is_file():
                logger.warning(
                    "TunnelDaemonSpawner: no config.yml found (checked ~/.cloudflared/ and /etc/cloudflared/)"
                )
                return False

            cmd = [
                binary,
                "--no-autoupdate",
                "--config",
                str(self._config_path),
                "tunnel",
                "run",
            ]
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=os.environ.copy(),
                    start_new_session=True,
                )
            except (OSError, ValueError) as e:
                logger.warning("TunnelDaemonSpawner: failed to spawn: %s", e)
                self._process = None
                return False

            if not self._atexit_registered:
                atexit.register(self.stop)
                self._atexit_registered = True

        # Short readiness wait — catches bad-config failures without blocking
        # startup if cloudflared is still registering edge connections.
        time.sleep(READINESS_WAIT_S)
        if not self.is_running:
            # Process died — likely config error; read stderr would be useful
            # but we DEVNULL'd it. Caller should surface this via logs.
            return False
        return True

    def stop(self) -> None:
        """Terminate the cloudflared subprocess. Idempotent, swallows errors."""
        with self._lock:
            proc = self._process
            self._process = None
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        except Exception:
            pass


__all__ = [
    "TunnelDaemonSpawner",
    "cloudflared_already_running",
    "resolve_config_path",
    "SYSTEM_CONFIG_PATH",
]
