"""CLI argument parser for Maxim.

Extracted from cli.py for modularity.
"""

from __future__ import annotations

import argparse
import os


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
        default=0,
        help="Epochs to run Maxim for (0 = unlimited).",
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
        default="exploration",
        choices=["live", "train", "reflection", "sleep", "agentic", "exploration"],
        help="Run mode: exploration (novelty-driven active discovery; DEFAULT), sleep (audio-only, no movement), live (no training), train (update MotorCortex), agentic (full perception-memory-goal architecture), reflection (introspection and memory consolidation).",
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
        help="Vision engine (default: rtm). Options: rtm, yolo (requires [yolo] extra).",
    )
    parser.add_argument(
        "--interactive",
        type=str,
        default="true",
        help="Enable interactive terminal input for keyword actions (True/False).",
    )
    parser.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Path for memory persistence (agentic mode). Default: {home_dir}/memory/memories.json",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset memory on startup (agentic mode).",
    )
    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Enable embedding-based memory similarity (requires sentence-transformers).",
    )
    parser.add_argument(
        "--autonomy",
        type=str,
        default="planning",
        choices=["planning", "supervised", "autonomous"],
        help="Initial autonomy level: planning (propose only), supervised (act within bounds), autonomous (full agency).",
    )
    parser.add_argument(
        "--autonomy-duration",
        type=float,
        default=None,
        help="Duration in seconds for timed autonomy (only applies to autonomous level).",
    )
    parser.add_argument(
        "--internet-access",
        action="store_true",
        default=True,
        help="Enable internet access for search and fetch tools (enabled by default).",
    )
    parser.add_argument(
        "--no-internet",
        action="store_true",
        help="Disable internet access.",
    )
    parser.add_argument(
        "--agentic-verbosity",
        type=int,
        default=None,  # Default to match --verbosity
        choices=[0, 1, 2, 3],
        help="Agentic logging verbosity: 0=quiet, 1=normal (goals/tools), 2=verbose (+perception/memory), 3=debug (+loop). Defaults to --verbosity level.",
    )
    parser.add_argument(
        "--no-agentic-console",
        action="store_true",
        dest="no_agentic_console",
        help="Disable agentic event output to console (enabled by default).",
    )
    # ─────────────────────────────────────────────────────────────────────────
    # TTS (Text-to-Speech) arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech for spoken responses (requires piper-tts).",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default="en_US-lessac-medium",
        help="TTS voice model name (default: en_US-lessac-medium).",
    )
    # ─────────────────────────────────────────────────────────────────────────
    # Communication (Twilio SMS/Voice)
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--comms",
        action="store_true",
        help="Enable Twilio SMS/Voice communication (requires TWILIO_* env vars).",
    )
    # ─────────────────────────────────────────────────────────────────────────
    # Exploration mode arguments
    # ─────────────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--explore",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Start in exploration mode with optional focus (e.g., --explore 'kitchen objects').",
    )
    parser.add_argument(
        "--exploration-duration",
        type=float,
        default=None,
        help="Duration in seconds for exploration session (default: unlimited).",
    )
    parser.add_argument(
        "--exploration-autonomy",
        type=str,
        default="supervised",
        choices=["supervised", "autonomous"],
        help="Autonomy level for exploration: supervised (default) or autonomous.",
    )
    parser.add_argument(
        "--exploration-allow-scripts",
        action="store_true",
        help="Allow writing and executing Python analysis scripts during exploration.",
    )
    parser.add_argument(
        "--exploration-allow-training",
        action="store_true",
        help="Allow model training during exploration (requires GPU).",
    )
    parser.add_argument(
        "--resume-session",
        type=str,
        default=None,
        help="Resume a previous exploration session by ID.",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List available exploration sessions and exit.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Python bytecode cache (__pycache__) before running.",
    )
    parser.add_argument(
        "--clear-memory",
        type=str,
        nargs="?",
        const="all",
        default=None,
        metavar="TYPE",
        help=(
            "Clear persistent memory and exit. "
            "Types: all (default), focus, bounds, escalation, fear, threshold, "
            "nac, scn, hippo, pain. Can specify multiple comma-separated types."
        ),
    )
    parser.add_argument(
        "--audit-architecture",
        action="store_true",
        help="Audit codebase for architecture rule violations and exit.",
    )
    return parser
