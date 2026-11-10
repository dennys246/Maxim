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
from typing import Any

from maxim.tunnel.cloudflared import find_cloudflared
from maxim.utils.cancellation import spawn_registered, stop_child_group, unregister_child
from maxim.tunnel.config import CONFIG_PATH as USER_CONFIG_PATH
from maxim.utils.paths import cloudflared_system_dir

logger = logging.getLogger(__name__)


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
    config must match what we expect. The system path comes from
    ``utils/paths.py::cloudflared_system_dir`` on every call (role detection
    and the doctor read the same one).
    """
    system_config = cloudflared_system_dir() / "config.yml"
    if system_config.is_file():
        return system_config
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
                    "TunnelDaemonSpawner: no config.yml found (checked %s and %s)",
                    USER_CONFIG_PATH,
                    cloudflared_system_dir() / "config.yml",
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
                # Registered with the launch (#1344 PR 3a): a forced stop of maxim kills the daemon it started.
                self._process = spawn_registered(
                    lambda: subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=os.environ.copy(),
                        start_new_session=True,
                    ),
                    kind="tunnel",
                    stop=self.stop,
                )
            except (OSError, ValueError) as e:
                logger.warning("TunnelDaemonSpawner: failed to spawn: %s", e)
                self._process = None
                return False
            if self._process is None:
                logger.warning("TunnelDaemonSpawner: registry closed: a restart is in progress, no daemon started")
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
            stop_child_group(proc)  # its whole group, as a forced stop's kill (#1344 PR 3a)
        except Exception:
            pass
        finally:
            unregister_child(proc)


def auto_spawn_tunnel_daemon(logger: Any | None) -> None:
    """Spawn the cloudflared tunnel daemon if a tunnel config exists and no daemon is running. Called only by
    ``runtime/leader_mode.py::start_claimed_leader_services``, for the process that owns a leader's services (#1610;
    moved here from ``lane_backends``).

    Opt out with MAXIM_AUTO_SPAWN_TUNNEL=0.
    """
    if os.environ.get("MAXIM_AUTO_SPAWN_TUNNEL", "").strip().lower() in (
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
    ):
        return
    if cloudflared_already_running():
        if logger is not None:
            logger.info(
                "Cloudflared daemon already running — skipping auto-spawn (managed elsewhere, e.g. systemd service)"
            )
        return
    config_path = resolve_config_path()
    if config_path is None:
        if logger is not None:
            system_config = cloudflared_system_dir() / "config.yml"
            logger.debug(f"Tunnel auto-spawn skipped: no ~/.cloudflared/config.yml or {system_config} found.")
        return
    spawner = TunnelDaemonSpawner(config_path=config_path)
    if logger is not None:
        logger.info("Auto-spawning cloudflared tunnel daemon (config=%s)", config_path)
    if not spawner.start():
        if logger is not None:
            logger.warning(
                "Cloudflared daemon auto-spawn failed — tunnel will not be active "
                "until you start it manually: cloudflared --config %s tunnel run",
                config_path,
            )


__all__ = [
    "auto_spawn_tunnel_daemon",
    "TunnelDaemonSpawner",
    "cloudflared_already_running",
    "resolve_config_path",
]
