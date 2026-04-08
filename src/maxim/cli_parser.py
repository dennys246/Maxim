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
        "--llm",
        type=str,
        default=None,
        help="LLM profile name (overrides ~/.maxim/config/llm.json and $MAXIM_LLM_PROFILE).",
    )
    parser.add_argument(
        "--cloud-fallback",
        type=str,
        default=None,
        metavar="MODEL",
        help="Add a cloud model as fallback on the infer lane (e.g., claude-sonnet, gpt-4o-mini). "
        "Used when the primary (local/self-hosted) provider fails or is rate-limited.",
    )
    parser.add_argument(
        "--cloud-lane",
        type=str,
        nargs=2,
        default=None,
        metavar=("LANE", "MODEL"),
        help="Assign a cloud model to a specific lane (e.g., --cloud-lane review claude-haiku).",
    )
    parser.add_argument(
        "--cloud-budget",
        type=float,
        default=None,
        metavar="DOLLARS",
        help="Max session cost in USD for cloud providers (default: $5.00).",
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
        "--list-models",
        action="store_true",
        help="List available LLM model profiles and exit.",
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
            "nac, scn, hippo, pain, semantic. Can specify multiple comma-separated types."
        ),
    )
    parser.add_argument(
        "--audit-architecture",
        action="store_true",
        help="Audit codebase for architecture rule violations and exit.",
    )

    # ── Simulation ──────────────────────────────────────────────────────
    parser.add_argument(
        "--sim",
        type=str,
        nargs="?",
        const="interactive",
        default=None,
        metavar="GOAL_OR_PATH",
        help="Run simulation. String: goal for generative campaign. "
        "YAML path: run scenario/campaign directly. "
        "'agent'/'research'/'benchmark': legacy mode aliases. "
        "No argument: interactive REPL.",
    )
    parser.add_argument(
        "--sim-goal",
        "--goal",
        type=str,
        default=None,
        dest="sim_goal",
        metavar="GOAL",
        help="Simulation goal (alternative to passing goal as --sim value). Alias: --goal",
    )
    parser.add_argument(
        "--dm",
        action="store_true",
        default=False,
        help="Dungeon Master mode. With --sim <goal>: generate a campaign from the goal (future). "
        "With --sim <path.yaml>: run as DM campaign (auto-detected from YAML structure).",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Generate research report (Writer + Reviewer agents) after simulation completes.",
    )
    parser.add_argument(
        "--arc",
        type=str,
        default=None,
        dest="arc",
        metavar="PATH",
        help="Seed arc template YAML for generative mode. "
        "On small models, followed literally. On larger models, used as a starting point.",
    )
    parser.add_argument(
        "--aut-name",
        type=str,
        default="AUT",
        dest="aut_name",
        metavar="NAME",
        help="Display name for the agent-under-test in simulation logs. Default: AUT",
    )
    parser.add_argument(
        "--sim-interactive",
        action="store_true",
        dest="sim_interactive",
        help="Enable human-in-the-loop interaction during simulation. "
        "The narrator can pause for player choices, dice rolls, etc.",
    )
    parser.add_argument(
        "--replay-from",
        type=str,
        default=None,
        dest="replay_from",
        metavar="SESSION_ID",
        help="Replay recorded user interactions from a previous session. "
        "Used with --sim-interactive for deterministic re-runs.",
    )
    parser.add_argument(
        "--sim-persona",
        "--persona",
        type=str,
        default="adversarial",
        dest="sim_persona",
        metavar="PERSONA",
        help="Orchestrator persona for --sim agent (adversarial, cooperative, confused, "
        "escalating, campaign, refinement). Alias: --persona",
    )
    parser.add_argument(
        "--aut-model",
        type=str,
        default=None,
        dest="aut_model",
        metavar="MODEL",
        help="Separate LLM for the agent-under-test (e.g., mistral-7b). "
        "Orchestrator/research agents use --language-model. "
        "Enables dual-LLM mode for isolating memory vs context recall.",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        default=None,
        dest="campaign",
        metavar="PATH",
        help="Campaign YAML(s) for --sim research mode. Glob patterns accepted "
        "(e.g., scenarios/experiments/hippocampal_recall_*.yaml).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous simulation mode: never auto-complete, keep testing until /cancel. "
        "Best with --persona infinite.",
    )
    parser.add_argument(
        "--no-sim-env",
        action="store_true",
        help="Skip simulated filesystem with pain-triggering files (empty sandbox).",
    )
    parser.add_argument(
        "--sandbox",
        dest="sandbox_backend",
        type=str,
        default="auto",
        choices=["auto", "docker", "tmpdir"],
        help="Sandbox backend for simulation tool execution. "
        "'auto' (default) uses Docker if available, else tmpdir with a "
        "warning. 'docker' requires Docker. 'tmpdir' forces host-based "
        "tmpdir isolation.",
    )
    parser.add_argument(
        "--sandbox-image",
        type=str,
        default="python:3.12-slim",
        metavar="IMAGE",
        help="Docker image for the sandbox container. Defaults to "
        "python:3.12-slim. Catalog includes ubuntu:22.04, ubuntu:24.04, "
        "debian:12-slim, rockylinux:9, almalinux:9, alpine:3.19, "
        "and the RHEL UBI9 minimal image.",
    )
    parser.add_argument(
        "--sandbox-network",
        type=str,
        default="none",
        choices=["none", "bridge", "host"],
        help="Container network mode. 'none' (default) isolates the "
        "container from the network; 'bridge' enables outbound "
        "traffic; 'host' shares the host network stack.",
    )
    parser.add_argument(
        "--resume-sim",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous simulation session by ID (from data/sim_reports/). "
        "Restores AUT memory state and provides previous findings as context.",
    )
    parser.add_argument(
        "--sim-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Write scenario results to a JSON file (requires --sim).",
    )
    parser.add_argument(
        "--record-percepts",
        action="store_true",
        help="Record all percepts during a live session to ~/.maxim/sessions/ for replay.",
    )
    parser.add_argument(
        "--debug",
        "--sim-debug",
        dest="debug",
        nargs="?",
        const="all",
        default=None,
        metavar="SUBSYSTEMS",
        help="Enable debug tracing. Without args: all subsystems. With args: "
        "comma-separated subsystem names (e.g., --debug memory, --debug memory,causal). "
        "Subsystems: hippo/memory (capture/recall), nac/causal/reward (causal learning), "
        "atl/semantic/concepts (concept memory), scn/temporal/clock (time rhythms), "
        "all (everything). Also settable via MAXIM_HIPPO_TRACE=1, MAXIM_LANE_TRACE=1.",
    )
    parser.add_argument(
        "--show",
        dest="show_channels",
        default=None,
        metavar="CHANNELS",
        help="Filter simulation output by channel. Comma-separated: "
        "bio (hippocampus/NAc/SCN/ATL/pain/fear), exec (tool execution/LLM), "
        "sim (percepts/scenes/NPC/choices), memory (hippo/NAc/SCN/ATL), "
        "safety (fear/pain), all (everything, default). "
        "Example: --show bio,exec",
    )
    # ── Benchmark ───────────────────────────────────────────────────────
    parser.add_argument(
        "--benchmark",
        type=str,
        nargs="?",
        const="all",
        default=None,
        metavar="TIERS",
        help="Run tiered benchmarks. Values: tier1, tier2, tier3, all, "
        "or comma-separated (tier1,tier2). Requires --models. "
        "Promoted from --sim benchmark (legacy alias still works).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="MODEL1,MODEL2,...",
        help="Comma-separated model profiles for --sim benchmark. "
        "Each model is tested against all scenarios in the suite.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of runs per model in --sim benchmark (default: 1). Multiple runs enable variance measurement.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory for benchmark reports (default: ~/.maxim/benchmarks).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a previous benchmark_report.json for comparison.",
    )
    parser.add_argument(
        "--write-paper",
        action="store_true",
        default=False,
        dest="write_paper",
        help="Generate a comparative research paper from benchmark results. "
        "Uses the Writer agent from the research protocol.",
    )
    parser.add_argument(
        "--generate-simulation",
        type=str,
        default=None,
        metavar="DESCRIPTION",
        help="Generate a simulation scenario YAML from a natural language description. "
        'Example: --generate-simulation "user asks robot to pick up a cup"',
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for --generate-scenario (default: stdout).",
    )
    # ── Last run ────────────────────────────────────────────────────────
    parser.add_argument(
        "--last",
        nargs="?",
        const=1,
        type=int,
        default=None,
        metavar="N",
        help="Re-run a recent invocation. --last (most recent), --last 2 (second most recent), etc. Up to 5 saved.",
    )
    parser.add_argument(
        "--show-last",
        action="store_true",
        help="Show the last saved invocation and exit.",
    )
    parser.add_argument(
        "--clear-last",
        action="store_true",
        help="Clear the saved last invocation and exit.",
    )
    return parser
