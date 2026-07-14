#!/usr/bin/env python3
"""Standalone Reachy Mini -> RTSP streamer.

Connects to a Reachy Mini via the reachy_mini SDK, reads camera frames, and
pipes them through ffmpeg to an RTSP endpoint (MediaMTX), which it can
auto-start. Imports only ``reachy_mini`` + the standard library — no
``maxim.*`` — so it runs as a pure "looks and streams" producer, fully
decoupled from the agentic movement/cognition stack.

Extracted from Maxim's ``RTSPBridge`` (standalone frame path only; coexist-mode
and every ``maxim`` coupling removed). Same proven ffmpeg pipeline
(rawvideo/bgr24 -> libx264 ultrafast/zerolatency -> rtsp/tcp).

This is the producer side of the Cardinal single-camera demo: ShredderSegmenter's
MaximCameraEngine consumes the RTSP feed this publishes.

Requires:
  - reachy_mini SDK installed (and a reachable robot daemon at runtime)
  - ffmpeg on PATH
  - a mediamtx binary on PATH (for --auto-start; on by default)

Usage:
  python scripts/reachy_streamer.py
  python scripts/reachy_streamer.py --url rtsp://localhost:8554/reachy --fps 20
  python scripts/reachy_streamer.py --connection-mode network   # robot on the LAN
  python scripts/reachy_streamer.py --sim                       # simulated robot, no hardware
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

log = logging.getLogger("reachy_streamer")

DEFAULT_URL = "rtsp://localhost:8554/reachy"


# ---------------------------------------------------------------------------
# MediaMTX
# ---------------------------------------------------------------------------


def _rtsp_port(url: str) -> int:
    """Best-effort parse of the port from an rtsp://host:port/path URL."""
    try:
        hostport = url.split("://", 1)[1].split("/", 1)[0]
        if ":" in hostport:
            return int(hostport.rsplit(":", 1)[1])
    except (IndexError, ValueError):
        pass
    return 8554


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _write_permissive_config() -> str:
    """Write a minimal MediaMTX config that allows publishing to any path.

    MediaMTX >=1.19 rejects publishing (RTSP 400 Bad Request) when started with
    an empty configuration — it needs at least ``paths: all_others:``. This is a
    dev/standalone convenience for --auto-start; for production run a managed
    MediaMTX with your own config and pass --mediamtx-config (or point --url at
    an already-running server).
    """
    fd, path = tempfile.mkstemp(prefix="reachy_mediamtx_", suffix=".yml")
    with os.fdopen(fd, "w") as f:
        f.write("paths:\n  all_others:\n")
    return path


def ensure_mediamtx(url: str, mediamtx_path: str | None, mediamtx_config: str | None, auto_start: bool):
    """Ensure something is serving RTSP on the target port.

    Returns the managed subprocess if we started one (so the caller can stop
    it), else None. Never raises — ffmpeg fails fast if the endpoint is down.
    """
    port = _rtsp_port(url)
    if _port_in_use("127.0.0.1", port):
        log.info("MediaMTX already reachable on port %d", port)
        return None
    if not auto_start:
        log.warning(
            "Nothing listening on RTSP port %d and --no-mediamtx set; "
            "start MediaMTX manually or ffmpeg will fail to connect.",
            port,
        )
        return None
    binary = mediamtx_path or shutil.which("mediamtx")
    if not binary:
        log.warning(
            "mediamtx not found on PATH; start it manually or pass "
            "--mediamtx-path. Continuing (ffmpeg will fail-fast if it can't connect)."
        )
        return None
    config = mediamtx_config or _write_permissive_config()
    if not mediamtx_config:
        log.info(
            "Auto-starting MediaMTX with a permissive dev config "
            "(paths: all_others). For production, run a managed MediaMTX and "
            "pass --mediamtx-config."
        )
    log.info("Starting MediaMTX (%s %s)...", binary, config)
    proc = subprocess.Popen([binary, config], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for the RTSP port to accept connections (up to ~3s) so ffmpeg doesn't
    # race MediaMTX's startup.
    for _ in range(30):
        if proc.poll() is not None:
            log.error("MediaMTX exited immediately (code %s)", proc.returncode)
            return None
        if _port_in_use("127.0.0.1", port):
            log.info("MediaMTX started (pid %d) on port %d", proc.pid, port)
            return proc
        time.sleep(0.1)
    log.warning(
        "MediaMTX started (pid %d) but port %d not accepting yet; continuing.",
        proc.pid,
        port,
    )
    return proc


# ---------------------------------------------------------------------------
# ffmpeg
# ---------------------------------------------------------------------------


def start_ffmpeg(width: int, height: int, args: argparse.Namespace):
    """Spawn ffmpeg reading raw BGR frames on stdin and publishing RTSP."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(args.fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        # Force 4:2:0: libx264 infers 4:4:4 High profile from bgr24 input, which
        # MediaMTX (and most H.264 consumers) reject with "400 Bad Request".
        "-pix_fmt",
        "yuv420p",
        "-preset",
        args.preset,
        "-tune",
        args.tune,
        "-g",
        str(args.gop_size),
        "-b:v",
        args.bitrate,
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        args.url,
    ]
    log.info("Starting ffmpeg: %dx%d @ %d fps -> %s", width, height, args.fps, args.url)
    try:
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        log.error("ffmpeg not found on PATH. Install ffmpeg.")
        return None


# ---------------------------------------------------------------------------
# Reachy Mini
# ---------------------------------------------------------------------------


def connect_robot(args: argparse.Namespace):
    """Connect to the robot via the reachy_mini SDK (late import).

    The camera is initialized during construction (media_backend='default'),
    so no separate 'start' call is needed — frames come from
    ``mini.media.get_frame()``.
    """
    from reachy_mini import ReachyMini  # late import so --help works without the SDK

    if args.sim:
        log.info("Connecting to a SIMULATED Reachy Mini (no hardware)...")
        return ReachyMini(
            spawn_daemon=True,
            use_sim=True,
            media_backend=args.media_backend,
            log_level=args.log_level,
        )
    log.info(
        "Connecting to Reachy Mini '%s' (connection_mode=%s)...",
        args.robot_name,
        args.connection_mode,
    )
    return ReachyMini(
        robot_name=args.robot_name,
        connection_mode=args.connection_mode,
        media_backend=args.media_backend,
        log_level=args.log_level,
    )


def run(args: argparse.Namespace) -> int:
    stop = {"flag": False}

    def _shutdown(signum, _frame):
        log.info("Signal %s received, shutting down...", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    mediamtx = ensure_mediamtx(args.url, args.mediamtx_path, args.mediamtx_config, not args.no_mediamtx)
    mini = None
    ffmpeg = None
    try:
        mini = connect_robot(args)

        # Grab an initial frame to detect resolution (retry ~5s: camera warm-up).
        frame = None
        for _ in range(50):
            if stop["flag"]:
                return 0
            frame = mini.media.get_frame()
            if frame is not None:
                break
            time.sleep(0.1)
        if frame is None:
            log.error("No camera frame after ~5s (camera unavailable?). Aborting.")
            return 1
        height, width = frame.shape[:2]

        ffmpeg = start_ffmpeg(width, height, args)
        if ffmpeg is None:
            return 1

        interval = 1.0 / max(1, args.fps)
        next_t = time.monotonic()
        frames = 0
        while not stop["flag"]:
            if ffmpeg.poll() is not None:
                log.error("ffmpeg exited (code %s). Stopping.", ffmpeg.returncode)
                return 1
            frame = mini.media.get_frame()
            if frame is not None:
                try:
                    ffmpeg.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    log.error("ffmpeg pipe closed. Stopping.")
                    return 1
                frames += 1
                if frames % (args.fps * 10) == 0:
                    log.info("Streaming... %d frames sent", frames)
            # Pace to target fps.
            next_t += interval
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()  # fell behind; resync rather than spiral
        log.info("Stopped after %d frames.", frames)
        return 0
    except Exception as e:  # noqa: BLE001 - top-level guard for a CLI entrypoint
        log.error("%s: %s", type(e).__name__, e)
        return 1
    finally:
        _cleanup(ffmpeg, mini, mediamtx)


def _cleanup(ffmpeg, mini, mediamtx) -> None:
    if ffmpeg is not None:
        try:
            if ffmpeg.stdin:
                ffmpeg.stdin.close()
            ffmpeg.terminate()
            ffmpeg.wait(timeout=5)
        except Exception:  # noqa: BLE001
            ffmpeg.kill()
    if mini is not None:
        try:
            mini.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
    if mediamtx is not None:
        try:
            mediamtx.terminate()
            mediamtx.wait(timeout=5)
        except Exception:  # noqa: BLE001
            mediamtx.kill()
    log.info("Cleanup complete.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone Reachy Mini -> RTSP streamer (reachy_mini + stdlib only).",
    )
    p.add_argument("--url", default=DEFAULT_URL, help=f"RTSP publish URL (default: {DEFAULT_URL})")
    p.add_argument("--fps", type=int, default=20, help="Target frame rate (default: 20)")
    p.add_argument("--robot-name", default="reachy_mini", help="Reachy robot name (default: reachy_mini)")
    p.add_argument(
        "--connection-mode",
        default="auto",
        choices=["auto", "localhost_only", "network"],
        help="SDK daemon discovery: auto (localhost then LAN), localhost_only, or network (default: auto)",
    )
    p.add_argument("--media-backend", default="default", help="reachy_mini media backend (default: 'default')")
    p.add_argument("--sim", action="store_true", help="Spawn a simulated robot (no hardware)")
    p.add_argument("--bitrate", default="2M", help="ffmpeg target bitrate (default: 2M)")
    p.add_argument("--gop-size", type=int, default=30, help="Keyframe interval in frames (default: 30)")
    p.add_argument("--preset", default="ultrafast", help="x264 preset (default: ultrafast)")
    p.add_argument("--tune", default="zerolatency", help="x264 tune (default: zerolatency)")
    p.add_argument("--no-mediamtx", action="store_true", help="Do not auto-start MediaMTX")
    p.add_argument("--mediamtx-path", default=None, help="Path to mediamtx binary (default: auto-detect on PATH)")
    p.add_argument(
        "--mediamtx-config",
        default=None,
        help="MediaMTX config for auto-start (default: a generated permissive dev config)",
    )
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    sys.exit(run(args))


if __name__ == "__main__":
    main()
