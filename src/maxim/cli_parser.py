"""CLI argument parser for Maxim.

Extracted from cli.py for modularity.
"""

from __future__ import annotations

import argparse
import os


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="maxim")

    # ── Core ───────────────────────────────────────────────────────────
    core = parser.add_argument_group("core", "Runtime and model configuration")
    core.add_argument(
        "--display",
        type=str,
        default="bio",
        choices=["clean", "bio", "debug"],
        help="Output detail: bio (+ memory/learning annotations, DEFAULT), "
        "clean (narrative only), debug (+ full system traces).",
    )
    core.add_argument(
        "--interactive",
        nargs="?",
        const="true",
        default=None,
        help="Enable interactive terminal input. Use bare --interactive "
        "to enable, or --interactive false to disable. When unset, "
        "auto-enables for DM campaigns and disables for generative sims.",
    )
    core.add_argument(
        "--log-level",
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        dest="verbosity",
        help="Logging level: 0=warnings/errors, 1=info, 2=debug. "
        "Alias: --verbosity (deprecated, will be removed before 1.0).",
    )
    core.add_argument(
        "--mode",
        type=str,
        default="exploration",
        choices=["live", "train", "reflection", "sleep", "agentic", "exploration"],
        help="Run mode: exploration (DEFAULT), agentic (full agent loop), "
        "sleep (audio-only), live (no training), train (update MotorCortex), "
        "reflection (memory consolidation).",
    )
    core.add_argument(
        "--language-model",
        "--llm",
        type=str,
        default=None,
        help="LLM profile name (overrides ~/.maxim/config/llm.json and $MAXIM_LLM_PROFILE).",
    )
    core.add_argument(
        "--auto-download",
        action="store_true",
        default=False,
        dest="auto_download",
        help="Skip the interactive 'download model? [y/N]' prompt and proceed "
        "with any GGUF download required for the active LLM profile. "
        "Equivalent to MAXIM_AUTO_DOWNLOAD_MODELS=1. Use in scripts and "
        "headless deployments.",
    )
    core.add_argument(
        "--llm-n-ctx",
        type=int,
        default=None,
        dest="llm_n_ctx",
        metavar="N",
        help="Override auto-computed llama.cpp n_ctx (context window). "
        "Use for tuning against specific VRAM budgets. Warning: a value "
        "above the formula estimate may OOM the GPU at load time.",
    )
    core.add_argument(
        "--embodiment",
        type=str,
        default=None,
        metavar="REF",
        help="SEM component reference for the agent's body (e.g. "
        "'weapons/rusty_sword'). Loads the component, auto-generates "
        "affordance tools, and wires the pain cascade through NAc. "
        "Works with --sim across all modes (since 0.6).",
    )
    core.add_argument(
        "--deep-embodiment",
        action="store_true",
        default=False,
        help="Enable level-3 deep embodiment: per-sub-sensor damage routing "
        "via damage_affinities, individual sub-sensor exposure to the agent. "
        "Requires a capable LLM — level 2 (default) collapses sub-sensors "
        "to a single integrity value per body part. "
        "Env var: MAXIM_DEEP_EMBODIMENT=1.",
    )
    core.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Epochs to run Maxim for (0 = unlimited).",
    )
    core.add_argument(
        "--home-dir",
        default="data",
        help="Data directory for run artifacts (default: 'data').",
    )
    core.add_argument(
        "--list-models",
        action="store_true",
        help="List available LLM model profiles and exit.",
    )
    core.add_argument(
        "--delete-model",
        type=str,
        default=None,
        metavar="MODEL",
        help="Delete a downloaded local LLM model to free disk space.",
    )

    # ── Cloud LLM ──────────────────────────────────────────────────────
    cloud = parser.add_argument_group("cloud", "Cloud LLM provider options")
    cloud.add_argument(
        "--cloud-fallback",
        type=str,
        default=None,
        metavar="MODEL",
        help="Add a cloud model as fallback on the infer lane (e.g., claude-sonnet, gpt-4o-mini). "
        "Used when the primary (local/self-hosted) provider fails or is rate-limited.",
    )
    cloud.add_argument(
        "--cloud-lane",
        type=str,
        nargs=2,
        default=None,
        metavar=("LANE", "MODEL"),
        help="Assign a cloud model to a specific lane (e.g., --cloud-lane review claude-haiku).",
    )
    cloud.add_argument(
        "--cloud-budget",
        type=float,
        default=None,
        metavar="DOLLARS",
        help="Max session cost in USD for cloud providers (default: $5.00).",
    )

    # ── Autonomy ───────────────────────────────────────────────────────
    autonomy = parser.add_argument_group("autonomy", "Agent autonomy and permissions")
    autonomy.add_argument(
        "--autonomy",
        type=str,
        default="planning",
        choices=["planning", "supervised", "autonomous"],
        help="Initial autonomy level: planning (propose only), supervised (act within bounds), "
        "autonomous (full agency).",
    )
    autonomy.add_argument(
        "--autonomy-duration",
        type=float,
        default=None,
        help="Limit autonomous mode to N seconds, then revert to supervised. "
        "Only applies when --autonomy autonomous is set.",
    )
    inet = autonomy.add_mutually_exclusive_group()
    inet.add_argument(
        "--internet-access",
        action="store_true",
        default=True,
        help="Enable internet access for search and fetch tools (default).",
    )
    inet.add_argument(
        "--no-internet",
        action="store_true",
        help="Disable internet access for all tools.",
    )

    # ── Memory ─────────────────────────────────────────────────────────
    memory = parser.add_argument_group("memory", "Persistent memory management")
    memory.add_argument(
        "--memory-path",
        type=str,
        default=None,
        help="Path for memory persistence (agentic mode). Default: {home_dir}/memory/memories.json",
    )
    memory.add_argument(
        "--reset",
        action="store_true",
        help="Reset memory on startup (agentic mode).",
    )
    memory.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="Enable embedding-based memory similarity (requires sentence-transformers).",
    )
    memory.add_argument(
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

    # ── Robot/Hardware ─────────────────────────────────────────────────
    hw = parser.add_argument_group("hardware", "Robot and hardware options")
    hw.add_argument(
        "--robot-name",
        default=os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini"),
        help="Reachy Mini daemon robot_name / zenoh namespace (default: $MAXIM_ROBOT_NAME or 'reachy_mini').",
    )
    hw.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the Zenoh connection (default: 30).",
    )
    hw.add_argument(
        "--segmentation-model",
        type=str,
        default=None,
        help="Vision engine (default: rtm). Options: rtm, yolo (requires [yolo] extra).",
    )
    hw.add_argument(
        "--audio",
        type=str,
        default="true",
        help="Record + transcribe audio (True/False).",
    )
    hw.add_argument(
        "--audio_len",
        type=float,
        default=5.0,
        help="Seconds per transcription chunk (default: 5.0).",
    )
    hw.add_argument(
        "--tts",
        action="store_true",
        help="Enable text-to-speech for spoken responses (requires piper-tts).",
    )
    hw.add_argument(
        "--tts-model",
        type=str,
        default="en_US-lessac-medium",
        help="TTS voice model name (default: en_US-lessac-medium).",
    )
    hw.add_argument(
        "--comms",
        action="store_true",
        help="Enable Twilio SMS/Voice communication (requires TWILIO_* env vars).",
    )
    # ── Agentic ────────────────────────────────────────────────────────
    agentic = parser.add_argument_group("agentic", "Agent loop display and logging")
    agentic.add_argument(
        "--no-agentic-console",
        action="store_true",
        dest="no_agentic_console",
        help="Disable agentic event output to console.",
    )

    # ── Simulation ─────────────────────────────────────────────────────
    sim = parser.add_argument_group("simulation", "Simulation mode (--sim)")
    sim.add_argument(
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
    sim.add_argument(
        "--sim-goal",
        "--goal",
        type=str,
        default=None,
        dest="sim_goal",
        metavar="GOAL",
        help="Simulation goal (alternative to passing goal as --sim value). Alias: --goal",
    )
    sim.add_argument(
        "--dm",
        action="store_true",
        default=False,
        help="Dungeon Master mode. With --sim <goal>: generate a campaign. "
        "With --sim <path.yaml>: run as DM campaign (auto-detected from YAML).",
    )
    sim.add_argument(
        "--reap-orphans",
        action="store_true",
        default=False,
        help="Kill stale maxim sim processes detected at startup (safety net "
        "for runaway runs that may still be hitting the LLM backend). "
        "Equivalent to MAXIM_REAP_ORPHANS=1.",
    )
    sim.add_argument(
        "--sim-max-turns",
        type=int,
        default=50,
        dest="sim_max_turns",
        metavar="N",
        help="Maximum simulation turns before forced termination. "
        "When reached, the stall detector injects a max_turns termination "
        "into the bridge so the report shows finish_reason='max_turns'. "
        "Default: 50.",
    )
    sim.add_argument(
        "--research",
        action="store_true",
        help="Generate research report (Writer + Reviewer agents) after simulation completes.",
    )
    sim.add_argument(
        "--sim-persona",
        "--persona",
        type=str,
        default="adversarial",
        dest="sim_persona",
        metavar="PERSONA",
        help="Orchestrator persona for simulation (adversarial, cooperative, confused, "
        "escalating, campaign, refinement). Alias: --persona",
    )
    sim.add_argument(
        "--aut-model",
        type=str,
        default=None,
        dest="aut_model",
        metavar="MODEL",
        help="Separate LLM for the agent-under-test (e.g., mistral-7b). "
        "Orchestrator/research agents use --language-model.",
    )
    sim.add_argument(
        "--campaign",
        type=str,
        default=None,
        dest="campaign",
        metavar="PATH",
        help="Campaign YAML(s) for --sim research mode. Glob patterns accepted.",
    )
    sim.add_argument(
        "--continuous",
        action="store_true",
        help="Continuous simulation: never auto-complete, keep testing until /cancel.",
    )
    sim.add_argument(
        "--no-sim-env",
        action="store_true",
        help="Skip simulated filesystem with pain-triggering files (empty sandbox).",
    )
    sim.add_argument(
        "--sandbox",
        dest="sandbox_backend",
        type=str,
        default="auto",
        choices=["auto", "docker", "tmpdir"],
        help="Sandbox backend: 'auto' (Docker if available, else tmpdir), "
        "'docker' (requires Docker), 'tmpdir' (host-based isolation).",
    )
    sim.add_argument(
        "--sandbox-image",
        type=str,
        default="python:3.12-slim",
        metavar="IMAGE",
        help="Docker image for the sandbox container (default: python:3.12-slim).",
    )
    sim.add_argument(
        "--sandbox-network",
        type=str,
        default="none",
        choices=["none", "bridge", "host"],
        help="Container network mode: 'none' (isolated, default), 'bridge' (outbound traffic), 'host' (shared).",
    )
    sim.add_argument(
        "--resume-sim",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous simulation session by ID. "
        "Restores AUT memory state and provides previous findings as context.",
    )
    sim.add_argument(
        "--sim-report",
        type=str,
        default=None,
        metavar="PATH",
        help="Write scenario results to a JSON file (requires --sim).",
    )
    sim.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Global deterministic seed. Sets PYTHONHASHSEED, random, numpy, "
        "and torch seeds for reproducible fixture runs. Byte-identical "
        "determinism requires fixture-driven mode (no live LLM).",
    )

    # ── Asset Foundry ─────────────────────────────────────────────────
    foundry = parser.add_argument_group("foundry", "Asset Foundry — LLM-driven SEM component generation")
    foundry.add_argument(
        "--foundry",
        type=str,
        default=None,
        metavar="THEME",
        help="Run the Asset Foundry pipeline. Generates SEM components "
        "from a theme prompt, validates, tests, and scores them. "
        "Example: maxim --foundry 'cyberpunk weapons' --count 10",
    )
    foundry.add_argument(
        "--foundry-count",
        "--count",
        type=int,
        default=10,
        metavar="N",
        help="Number of components to generate (default: 10).",
    )
    foundry.add_argument(
        "--foundry-genre",
        type=str,
        default="fantasy",
        metavar="GENRE",
        help="Genre tag for generated components (default: fantasy).",
    )
    foundry.add_argument(
        "--foundry-category",
        type=str,
        default=None,
        metavar="CATEGORY",
        help="Specific category (weapons, creatures, npcs, items, etc.). If omitted, distributes across categories.",
    )
    foundry.add_argument(
        "--foundry-dry-run",
        action="store_true",
        default=False,
        help="Generate + validate only, skip gauntlet testing.",
    )

    # ── Auto-Curation ─────────────────────────────────────────────────
    curation = parser.add_argument_group("curation", "Pre-sim auto-curation of SEM components")
    curation.add_argument(
        "--auto-curate",
        action="store_true",
        default=False,
        help="Before sim, check genre/category coverage and run foundry "
        "to fill gaps. Promoted components go to ~/.maxim/components/.",
    )
    curation.add_argument(
        "--curate-threshold",
        type=int,
        default=5,
        metavar="N",
        help="Min components per genre/category before auto-curate triggers (default: 5).",
    )
    curation.add_argument(
        "--no-curate",
        action="store_true",
        default=False,
        help="Explicit opt-out of auto-curation (overrides config defaults).",
    )

    # ── Debug/Trace ────────────────────────────────────────────────────
    dbg = parser.add_argument_group("debug", "Debug tracing and output filtering")
    dbg.add_argument(
        "--trace",
        "--debug",
        "--sim-debug",
        dest="debug",
        nargs="?",
        const="all",
        default=None,
        metavar="SUBSYSTEMS",
        help="Enable detailed bio-system traces to stderr. Without args: all. "
        "With args: comma-separated (e.g., --trace hippo,nac). "
        "Subsystems: hippo/memory, nac/causal/reward, atl/semantic, "
        "scn/temporal, all. --debug and --sim-debug are synonyms.",
    )
    dbg.add_argument(
        "--show",
        dest="show_channels",
        default=None,
        metavar="CHANNELS",
        help="Filter simulation output by channel. Comma-separated: "
        "bio, bio-only (learning mechanisms only), exec, sim, memory, safety, all (default). "
        "Example: --show bio-only",
    )

    # ── Benchmark ──────────────────────────────────────────────────────
    bench = parser.add_argument_group("benchmark", "Multi-model benchmarking (--benchmark)")
    bench.add_argument(
        "--benchmark",
        type=str,
        nargs="?",
        const="all",
        default=None,
        metavar="TIERS",
        help="Run tiered benchmarks. Values: tier1, tier2, tier3, all, or comma-separated. Requires --models.",
    )
    bench.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="MODEL1,MODEL2,...",
        help="Comma-separated model profiles for benchmarking.",
    )
    bench.add_argument(
        "--runs",
        type=int,
        default=1,
        metavar="N",
        help="Runs per model in benchmark (default: 1). Multiple runs enable variance measurement.",
    )
    bench.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory for benchmark reports (default: ~/.maxim/benchmarks).",
    )
    bench.add_argument(
        "--baseline",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a previous benchmark_report.json for comparison.",
    )
    bench.add_argument(
        "--write-paper",
        action="store_true",
        default=False,
        dest="write_paper",
        help="Generate a comparative research paper from benchmark results.",
    )

    # ── Utilities ──────────────────────────────────────────────────────
    util = parser.add_argument_group("utilities", "Maintenance and generation tools")
    util.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear Python bytecode cache (__pycache__) before running.",
    )
    util.add_argument(
        "--audit-architecture",
        action="store_true",
        help="Audit codebase for architecture rule violations and exit.",
    )
    util.add_argument(
        "--generate-simulation",
        type=str,
        default=None,
        metavar="DESCRIPTION",
        help="Generate a simulation scenario YAML from a natural language description.",
    )
    util.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Output path for --generate-simulation (default: stdout).",
    )
    util.add_argument(
        "--last",
        nargs="?",
        const=1,
        type=int,
        default=None,
        metavar="N",
        help="Re-run a recent invocation. --last (most recent), --last 2 (second most recent).",
    )
    util.add_argument(
        "--show-last",
        action="store_true",
        help="Show the last saved invocation and exit.",
    )
    util.add_argument(
        "--clear-last",
        action="store_true",
        help="Clear the saved last invocation and exit.",
    )
    return parser
