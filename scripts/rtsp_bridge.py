#!/usr/bin/env python3
"""Quick-start: expose Reachy Mini camera as RTSP without the full agentic loop.

Requires:
  - reachy_mini SDK installed
  - ffmpeg installed
  - MediaMTX running (./mediamtx)

Usage:
  python scripts/rtsp_bridge.py
  python scripts/rtsp_bridge.py --url rtsp://localhost:8554/reachy --fps 20
  python scripts/rtsp_bridge.py --sim  # simulation mode (no real robot)
"""

import argparse
import logging
import signal
import sys


def main():
    parser = argparse.ArgumentParser(description="Reachy Mini -> RTSP bridge")
    parser.add_argument(
        "--url", default="rtsp://localhost:8554/reachy", help="RTSP publish URL (default: rtsp://localhost:8554/reachy)"
    )
    parser.add_argument("--fps", type=int, default=20, help="Target frame rate (default: 20)")
    parser.add_argument("--robot-name", default="reachy_mini", help="Reachy robot name (default: reachy_mini)")
    parser.add_argument("--sim", action="store_true", help="Use simulation mode (no real robot)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("rtsp_bridge")

    # Late imports so --help works without reachy_mini installed
    from maxim.hardware import RobotRegistry
    from maxim.hardware.reachy import ReachyMiniController
    from maxim.hardware.simulation import SimulatedController
    from maxim.tools.rtsp_bridge import RTSPBridge, RTSPBridgeConfig

    # Connect via hardware abstraction layer
    registry = RobotRegistry()
    registry.register_controller_type("reachy_mini", ReachyMiniController)
    registry.register_controller_type("simulated", SimulatedController)

    robot_type = "simulated" if args.sim else "reachy_mini"
    config = (
        {"video_resolution": (640, 480), "simulate_delays": False}
        if args.sim
        else {"robot_name": args.robot_name, "media_backend": "default"}
    )

    log.info("Connecting to robot '%s' (type=%s)...", args.robot_name, robot_type)
    try:
        robot = registry.connect_robot(
            robot_id=args.robot_name,
            robot_type=robot_type,
            config=config,
            timeout=30.0,
            set_primary=True,
        )
        if robot is None:
            log.error("Failed to connect to robot")
            sys.exit(1)
        robot.start_recording()
    except Exception as e:
        log.error("Failed to connect: %s", e)
        sys.exit(1)

    # Build a minimal shim that satisfies RTSPBridge's interface
    class _Shim:
        def __init__(self, mini):
            self.mini = mini
            self.log = log
            self._live_stop_event = None  # Signals standalone mode
            self._last_frame = None
            self._last_frame_ts = None

    mini = getattr(robot, "mini", robot)
    bridge = RTSPBridge(
        _Shim(mini),
        RTSPBridgeConfig(rtsp_url=args.url, fps=args.fps),
    )

    def _shutdown(sig, frame):
        log.info("Shutting down...")
        bridge._stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("Streaming to %s (Ctrl+C to stop)", args.url)
    try:
        bridge.start(blocking=True)
    finally:
        log.info("Disconnected.")


if __name__ == "__main__":
    main()
