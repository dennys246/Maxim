import argparse
import logging
import os
import time

from src.conscience.selfy import Maxim
from src.utils.data_management import build_home
from src.utils.logging import configure_logging, log_exception


def life():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-name",
        default=os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini"),
        help="Reachy Mini daemon robot_name / zenoh namespace (default: $MAXIM_ROBOT_NAME or 'reachy_mini').",
    )
    parser.add_argument(
        "--home-dir",
        default=os.path.join("experiments", "maxim"),
        help="Reachy Mini home directory to save audio/pictures, models, and derivatives (default: 'experiments/maxim').",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Zenoh connection (default: 30).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Epochs to run Maxim for.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=warnings/errors, 1=info, 2=debug.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="passive-interaction",
        choices=["live", "train", "passive-interaction", "sleep"],
        help="Run mode: passive-interaction (track targets without ML), live (no training), train (update MotorCortex), sleep (audio-only; no wake_up).",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="true",
        help="Record + transcribe audio (True/False).",
    )
    parser.add_argument(
        "--audio_len",
        type=float,
        default=5.0,
        help="Seconds per transcription chunk (default: 5.0).",
    )

    args = parser.parse_args()

    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    if str(getattr(args, "mode", "passive-interaction")).strip().lower() == "sleep":
        args.audio = True

    run_id = time.strftime("%Y-%m-%d_%H%M%S")
    build_home(args.home_dir)
    log_path = os.path.join(args.home_dir, "logs", f"reachy_log_{run_id}.log")

    configure_logging(args.verbosity, log_file=log_path)
    logger = logging.getLogger("maxim")

    try:
        logger.info(
            "Starting Maxim (robot_name=%s, home_dir=%s, timeout=%.1fs, epochs=%d, mode=%s, log=%s)",
            args.robot_name,
            args.home_dir,
            float(args.timeout),
            int(args.epochs),
            args.mode,
            log_path,
        )

        maxim = Maxim(
            robot_name=args.robot_name,
            home_dir=args.home_dir,
            timeout=args.timeout,
            epochs=args.epochs,
            verbosity=args.verbosity,
            mode=args.mode,
            audio=bool(getattr(args, "audio", True)),
            audio_len=float(getattr(args, "audio_len", 5.0) or 5.0),
        )

        if str(getattr(args, "mode", "passive-interaction")).strip().lower() == "sleep":
            logger.info("Maxim sleeping (audio-only).")
            maxim.sleep(home_dir=args.home_dir, run_id=run_id)
        else:
            logger.info("✅ Maxim lives!")
            maxim.live(home_dir=args.home_dir, run_id=run_id)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
    except Exception as e:
        log_exception(logger, e, verbosity=getattr(args, "verbosity", 0), message="❌ Maxim stopped")
    
if __name__ == "__main__":

    life()
    
