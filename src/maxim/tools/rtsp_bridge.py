"""RTSP bridge — exposes Reachy Mini camera as an RTSP stream.

Fully opt-in: this module is never eagerly imported. It is loaded lazily
by RTSPStreamingSkill.activate() only when needed, and wrapped in
try/except so a missing ffmpeg or MediaMTX never breaks the rest of Maxim.
"""

from __future__ import annotations

import logging
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

__all__ = ["RTSPBridge", "RTSPBridgeConfig"]


@dataclass(frozen=True)
class RTSPBridgeConfig:
    """Configuration for the RTSP bridge."""
    rtsp_url: str = "rtsp://localhost:8554/reachy"
    fps: int = 20
    # ffmpeg encoding settings
    preset: str = "ultrafast"       # CPU-friendly, low latency
    tune: str = "zerolatency"       # No B-frames, minimal buffering
    gop_size: int = 30              # Keyframe every ~1.5s at 20fps
    bitrate: str = "2M"             # Good quality for 720p
    # MediaMTX auto-management
    auto_start_mediamtx: bool = True   # Start MediaMTX if not already running
    mediamtx_path: str | None = None   # Path to binary (auto-detected if None)


class RTSPBridge:
    """Pipes Reachy Mini camera frames into MediaMTX via ffmpeg.

    Frame acquisition modes:
        - Standalone (no live loop): calls mini.media.get_frame() directly.
        - Coexist (live loop running): reads maxim._last_frame to avoid
          contending with frame_capture_worker for _media_lock. Deduplicates
          frames via _last_frame_ts to avoid encoding the same frame twice.

    Usage (standalone):
        bridge = RTSPBridge(maxim_instance)
        bridge.start()   # blocks until stop() called from another thread

    Usage (as skill):
        Wrapped by RTSPStreamingSkill — the skill handles lifecycle.
    """

    def __init__(self, maxim: Any, config: RTSPBridgeConfig | None = None):
        self.maxim = maxim
        self.config = config or RTSPBridgeConfig()
        self._proc: subprocess.Popen | None = None
        self._mediamtx_proc: subprocess.Popen | None = None  # managed MediaMTX
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = threading.Event()
        self._running = False  # True while _run() is actively looping
        self._expected_shape: tuple[int, int] | None = None  # (h, w)
        self._last_sent_ts: float | None = None  # dedup in coexist mode

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_event.is_set()

    def _get_rtsp_port(self) -> int:
        """Extract the RTSP port from the configured URL."""
        parsed = urlparse(self.config.rtsp_url)
        return parsed.port or 8554

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a TCP port is already listening (MediaMTX already running)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(("localhost", port)) == 0

    def _find_mediamtx(self) -> str | None:
        """Locate the mediamtx binary."""
        if self.config.mediamtx_path:
            return self.config.mediamtx_path
        return shutil.which("mediamtx")

    def _ensure_mediamtx(self) -> bool:
        """Start MediaMTX if not already running. Returns True if RTSP port is ready."""
        log = getattr(self.maxim, "log", logger)
        port = self._get_rtsp_port()

        if self._is_port_in_use(port):
            log.info("MediaMTX already running on port %d", port)
            return True

        if not self.config.auto_start_mediamtx:
            log.warning(
                "MediaMTX not running on port %d and auto_start_mediamtx=False",
                port,
            )
            return False

        binary = self._find_mediamtx()
        if binary is None:
            log.warning(
                "MediaMTX not running and binary not found in PATH. "
                "Install MediaMTX or set mediamtx_path in config. "
                "See docs/mediaMTX.md for setup instructions."
            )
            return False

        log.info("Starting MediaMTX from %s", binary)
        try:
            self._mediamtx_proc = subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            log.warning("Failed to start MediaMTX: %s", e)
            return False

        # Wait for MediaMTX to start accepting connections
        for _ in range(20):  # up to 2s
            if self._mediamtx_proc.poll() is not None:
                log.warning(
                    "MediaMTX exited immediately (code %d)",
                    self._mediamtx_proc.returncode,
                )
                self._mediamtx_proc = None
                return False
            if self._is_port_in_use(port):
                log.info("MediaMTX started on port %d (pid %d)", port, self._mediamtx_proc.pid)
                return True
            time.sleep(0.1)

        log.warning("MediaMTX started but port %d not ready after 2s", port)
        return True  # Optimistic — ffmpeg will fail-fast if it can't connect

    def _stop_mediamtx(self) -> None:
        """Stop MediaMTX if we started it."""
        proc = self._mediamtx_proc
        if proc is None:
            return
        log = getattr(self.maxim, "log", logger)
        try:
            proc.terminate()
            proc.wait(timeout=5.0)
            log.info("MediaMTX stopped (pid %d)", proc.pid)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.warning("MediaMTX force-killed (pid %d)", proc.pid)
        except Exception as e:
            log.warning("Error stopping MediaMTX: %s", e)
        self._mediamtx_proc = None

    def start(self, blocking: bool = True) -> None:
        """Start the RTSP bridge.

        Automatically starts MediaMTX if not already running and
        auto_start_mediamtx is enabled (default).

        Args:
            blocking: If True, blocks until stop() is called.
                      If False, runs in a background daemon thread.
        """
        if self.is_running:
            return

        self._stop_event.clear()
        self._started.clear()

        # Ensure MediaMTX is available before starting ffmpeg
        self._ensure_mediamtx()

        if blocking:
            self._run()
        else:
            self._thread = threading.Thread(
                target=self._run,
                name="maxim.rtsp_bridge",
                daemon=True,
            )
            self._thread.start()
            self._started.wait(timeout=5.0)

    def stop(self) -> None:
        """Stop the RTSP bridge gracefully. Also stops MediaMTX if we started it."""
        self._stop_event.set()
        self._kill_ffmpeg()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        self._stop_mediamtx()

    def _kill_ffmpeg(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception as e:
            log = getattr(self.maxim, "log", logger)
            log.debug("Error closing ffmpeg stdin: %s", e)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception as e:
            log = getattr(self.maxim, "log", logger)
            log.warning("Error waiting for ffmpeg to exit: %s", e)
            try:
                proc.kill()
            except Exception:
                pass  # Best effort — process may already be gone
        self._proc = None

    def _start_ffmpeg(self, w: int, h: int) -> subprocess.Popen | None:
        cfg = self.config
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(cfg.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", cfg.preset,
            "-tune", cfg.tune,
            "-g", str(cfg.gop_size),
            "-b:v", cfg.bitrate,
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            cfg.rtsp_url,
        ]
        try:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return None

    def _grab_frame(self) -> Any:
        """Acquire a frame using the appropriate mode.

        Coexist mode: if a live loop is running, read from _last_frame
        to avoid lock contention. Dedup via _last_frame_ts.
        Standalone mode: call mini.media.get_frame() directly.
        """
        live_event = getattr(self.maxim, "_live_stop_event", None)
        live_loop_active = live_event is not None and not live_event.is_set()

        if live_loop_active:
            ts = getattr(self.maxim, "_last_frame_ts", None)
            if ts is not None and ts == self._last_sent_ts:
                return None  # same frame, skip
            self._last_sent_ts = ts
            return getattr(self.maxim, "_last_frame", None)
        else:
            return self.maxim.mini.media.get_frame()

    def _run(self) -> None:
        """Main loop: grab frames -> pipe to ffmpeg -> RTSP."""
        log = getattr(self.maxim, "log", logger)
        cfg = self.config

        # Grab initial frame to detect resolution
        frame = None
        last_error = None
        for attempt in range(50):  # retry for up to ~5s
            if self._stop_event.is_set():
                return
            try:
                frame = self._grab_frame()
                last_error = None
            except Exception as e:
                last_error = e
                log.debug("Frame grab attempt %d failed: %s", attempt + 1, e)
            if frame is not None and hasattr(frame, "shape") and frame.size > 0:
                break
            time.sleep(0.1)

        if frame is None:
            if last_error:
                log.error(
                    "RTSP bridge: could not get initial frame after 50 attempts. "
                    "Last error: %s", last_error,
                )
            else:
                log.error("RTSP bridge: no frames available from camera")
            self._started.set()
            return

        h, w = frame.shape[:2]
        self._expected_shape = (h, w)

        log.info(
            "RTSP bridge starting: %dx%d @ %d fps -> %s",
            w, h, cfg.fps, cfg.rtsp_url,
        )

        self._proc = self._start_ffmpeg(w, h)
        if self._proc is None:
            log.warning(
                "RTSP bridge: ffmpeg not found. Install with: apt install ffmpeg"
            )
            self._started.set()
            return

        self._started.set()
        self._running = True
        frame_interval = 1.0 / cfg.fps

        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()

                if self._proc.poll() is not None:
                    log.error(
                        "RTSP bridge: ffmpeg exited (code %d). Is MediaMTX running?",
                        self._proc.returncode,
                    )
                    break

                frame = None
                try:
                    frame = self._grab_frame()
                except Exception as e:
                    log.warning("RTSP bridge frame error: %s", e)
                    time.sleep(0.01)
                    continue

                if frame is None or frame.size == 0:
                    time.sleep(0.005)
                    continue

                # Resolution change guard
                cur_h, cur_w = frame.shape[:2]
                if (cur_h, cur_w) != self._expected_shape:
                    log.warning(
                        "RTSP bridge: resolution changed %dx%d -> %dx%d, restarting ffmpeg",
                        self._expected_shape[1], self._expected_shape[0],
                        cur_w, cur_h,
                    )
                    self._kill_ffmpeg()
                    self._expected_shape = (cur_h, cur_w)
                    self._proc = self._start_ffmpeg(cur_w, cur_h)
                    if self._proc is None:
                        log.error("RTSP bridge: failed to restart ffmpeg")
                        break
                    continue

                try:
                    self._proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    log.error("RTSP bridge: ffmpeg pipe broken. Is MediaMTX running?")
                    break
                except OSError as e:
                    log.error("RTSP bridge: OS error writing to ffmpeg: %s", e)
                    break

                elapsed = time.monotonic() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self._running = False
            self._kill_ffmpeg()
            log.info("RTSP bridge stopped.")
