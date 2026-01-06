from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Sequence

from maxim.utils.data_management import build_home
from maxim.utils.logging import configure_logging, log_exception


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim")
    parser.add_argument(
        "--robot-name",
        default=os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini"),
        help="Reachy Mini daemon robot_name / zenoh namespace (default: $MAXIM_ROBOT_NAME or 'reachy_mini').",
    )
    parser.add_argument(
        "--home-dir",
        default="data",
        help="Reachy Mini home directory to save run artifacts (audio/videos/images/transcript/logs) (default: 'data').",
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
        choices=["live", "train", "passive-interaction", "sleep", "agentic"],
        help="Run mode: passive-interaction (track targets without ML), live (no training), train (update MotorCortex), sleep (audio-only; no wake_up), agentic (run the agentic runtime loop).",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="reachy_mini",
        help="Agent name for --mode agentic (default: 'reachy_mini').",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="read_readme",
        help="Agentic goal for --mode agentic (string like 'read_readme', or JSON like '{\"tool_name\":\"read_file\",\"params\":{\"path\":\"README.md\"}}').",
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
    parser.add_argument(
        "--language-model",
        type=str,
        default=None,
        help="LLM profile name (overrides data/util/llm.json and $MAXIM_LLM_PROFILE).",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default=None,
        help="Vision segmentation model (default: YOLO8).",
    )
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    audio_raw = str(getattr(args, "audio", "true")).strip().lower()
    if audio_raw in ("1", "true", "t", "yes", "y", "on"):
        args.audio = True
    elif audio_raw in ("0", "false", "f", "no", "n", "off"):
        args.audio = False
    else:
        raise SystemExit(f"Invalid --audio value: {args.audio!r} (expected True/False)")

    if str(getattr(args, "mode", "passive-interaction")).strip().lower() == "sleep":
        args.audio = True

    language_model = getattr(args, "language_model", None)
    if language_model is not None:
        from maxim.models.language.router import list_llm_profiles, normalize_llm_profile

        selected = normalize_llm_profile(language_model)
        if selected:
            available = list_llm_profiles()
            if available and selected not in available:
                opts = ", ".join(available)
                raise SystemExit(f"Unknown --language-model {language_model!r}. Available: {opts}")
            os.environ["MAXIM_LLM_PROFILE"] = selected
        args.language_model = selected

    segmentation_model = getattr(args, "segmentation_model", None)
    if segmentation_model is not None:
        from maxim.models.vision.registry import list_segmentation_models, normalize_segmentation_model

        selected = normalize_segmentation_model(segmentation_model) or "YOLO8"
        available = list_segmentation_models()
        if available and selected not in available:
            opts = ", ".join(available)
            raise SystemExit(f"Unknown --segmentation-model {segmentation_model!r}. Available: {opts}")
        os.environ["MAXIM_SEGMENTATION_MODEL"] = selected
        args.segmentation_model = selected


def _reexec_with_mode(args: argparse.Namespace, *, mode: str) -> None:
    mode = str(mode or "").strip().lower()
    if not mode:
        return

    audio_flag = bool(getattr(args, "audio", True))
    if mode == "sleep":
        audio_flag = True

    argv = [
        sys.executable,
        "-m",
        "maxim.cli",
        "--robot-name",
        str(getattr(args, "robot_name", "reachy_mini")),
        "--home-dir",
        str(getattr(args, "home_dir", "data")),
        "--timeout",
        str(float(getattr(args, "timeout", 30.0) or 30.0)),
        "--epochs",
        str(int(getattr(args, "epochs", 1000) or 1000)),
        "--verbosity",
        str(int(getattr(args, "verbosity", 1) or 1)),
        "--mode",
        mode,
        "--agent",
        str(getattr(args, "agent", "reachy_mini")),
        "--goal",
        str(getattr(args, "goal", "read_readme")),
        "--audio",
        "true" if audio_flag else "false",
        "--audio_len",
        str(float(getattr(args, "audio_len", 5.0) or 5.0)),
    ]
    language_model = str(getattr(args, "language_model", "") or "").strip()
    if language_model:
        argv.extend(["--language-model", language_model])
    segmentation_model = str(getattr(args, "segmentation_model", "") or "").strip()
    if segmentation_model:
        argv.extend(["--segmentation-model", segmentation_model])
    os.execv(sys.executable, argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _normalize_args(args)

    build_home(args.home_dir)
    mode = str(getattr(args, "mode", "passive-interaction")).strip().lower()
    while True:
        run_id = time.strftime("%Y-%m-%d_%H%M%S")
        log_path = os.path.join(args.home_dir, "logs", f"reachy_log_{run_id}.log")

        configure_logging(args.verbosity, log_file=log_path, force=True)
        logger = logging.getLogger("maxim")

        maxim = None
        try:
            logger.info(
                "Starting Maxim (robot_name=%s, home_dir=%s, timeout=%.1fs, epochs=%d, mode=%s, log=%s)",
                args.robot_name,
                args.home_dir,
                float(args.timeout),
                int(args.epochs),
                mode,
                log_path,
            )

            if mode == "agentic":
                from maxim.agents import AgentList, GoalAgent, ReachyAgent
                from maxim.environment import ReachyEnv
                from maxim.runtime import (
                    build_decision_engine,
                    build_environment,
                    build_evaluators,
                    build_executor,
                    build_memory,
                    build_state,
                    build_tool_registry,
                    run_agent_loop,
                )

                goal_raw = str(getattr(args, "goal", "read_readme"))
                goal = goal_raw
                if goal_raw.strip().startswith("{"):
                    try:
                        goal = json.loads(goal_raw)
                    except Exception:
                        goal = goal_raw

                agents = AgentList(
                    [
                        ReachyAgent(),
                        GoalAgent(goal, name="GoalAgent"),
                    ]
                )
                agent_name = str(getattr(args, "agent", "") or "").strip()
                agent = agents.get_by_agent_name(agent_name)
                if agent is None:
                    available = ", ".join(agents.list_agent_names())
                    raise SystemExit(f"Unknown --agent {agent_name!r}. Available: {available}")

                registry = build_tool_registry()
                executor = build_executor(registry)
                decision_engine = build_decision_engine()
                if str(getattr(agent, "agent_name", "")).strip().lower() == "reachy_mini":
                    env = ReachyEnv(data_dir=args.home_dir)
                else:
                    env = build_environment()
                state = build_state(max_steps=int(args.epochs))
                memory = build_memory()
                evaluators = build_evaluators()

                run_agent_loop(
                    agent,
                    env,
                    state,
                    memory,
                    decision_engine,
                    executor,
                    evaluators=evaluators,
                    max_steps=int(args.epochs),
                    run_id=run_id,
                )
                return 0

            from maxim.conscience.selfy import Maxim

            audio_enabled = bool(getattr(args, "audio", True))
            if mode == "sleep":
                audio_enabled = True

            maxim = Maxim(
                robot_name=args.robot_name,
                home_dir=args.home_dir,
                timeout=args.timeout,
                epochs=args.epochs,
                verbosity=args.verbosity,
                mode=mode,
                audio=audio_enabled,
                audio_len=float(getattr(args, "audio_len", 5.0) or 5.0),
            )

            if mode == "sleep":
                logger.info("Maxim sleeping (audio-only).")
                maxim.sleep(home_dir=args.home_dir, run_id=run_id)
            else:
                logger.info("✅ Maxim lives!")
                maxim.live(home_dir=args.home_dir, run_id=run_id)

        except KeyboardInterrupt:
            logger.warning("Interrupted by user (Ctrl+C).")
            break
        except Exception as e:
            log_exception(
                logger,
                e,
                verbosity=getattr(args, "verbosity", 0),
                message="❌ Maxim stopped",
            )
            break
        finally:
            if maxim is not None:
                try:
                    maxim.shutdown()
                except Exception:
                    pass

        requested = getattr(maxim, "requested_mode", None) if maxim is not None else None
        if not requested:
            break
        requested = str(requested).strip().lower()
        if requested == "shutdown":
            logger.info("Shutdown requested.")
            break
        if requested in ("sleep", "passive-interaction", "train", "live", "agentic"):
            logger.info("Switching mode: %s -> %s", mode, requested)
            delay_s = 0.0
            try:
                delay_s = float(os.getenv("MAXIM_MODE_SWITCH_DELAY_S", "1.5") or 0.0)
            except Exception:
                delay_s = 1.5
            if delay_s > 0:
                logger.info("Waiting %.1fs before reconnect...", delay_s)
                time.sleep(delay_s)
            try:
                _reexec_with_mode(args, mode=requested)
            except Exception as e:
                logger.warning("Failed to restart Maxim for mode switch (%s); continuing in-process.", e)
                mode = requested
                continue
        logger.warning("Ignoring unknown requested_mode=%r", requested)
        break

    return 0


life = main


if __name__ == "__main__":
    raise SystemExit(main())
