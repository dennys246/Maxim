"""Simulation types and helpers — SimulationResult, resume context.

Extracted from orchestrator.py for single-responsibility decomposition.
Pure data structures and stateless helpers with no lifecycle dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Exit 4 is already the simulation hard-abort contract used by the D12 stall
# detector.  A clean unwind must report the same process outcome or campaign
# subprocesses can mistake an infrastructure abort for usable evidence.
SIMULATION_ABORT_EXIT_CODE = 4
SIMULATION_ABORT_FINISH_REASONS = frozenset(
    {
        "aborted",
        "aut_died",
        "cancel",
        "llm_wedged",
        "planning_failed",
        "stuck",
        "worker_unavailable",
    }
)
SIMULATION_FAILURE_FINISH_REASONS = SIMULATION_ABORT_FINISH_REASONS | {"error"}


def _normalize_finish_reason(finish_reason: str) -> str:
    """Normalize a persisted/runtime finish reason for policy checks."""
    return str(finish_reason or "").strip().lower()


def is_simulation_run_failure(finish_reason: str) -> bool:
    """Return whether a result is unusable because the run itself failed.

    This intentionally differs from the simulation's semantic verdict.  An
    orchestrator-confirmed ``failed``/``blocked``/``inconclusive`` outcome is
    valid experiment data; runtime aborts, cancellation, and ``error`` are not.
    """
    return _normalize_finish_reason(finish_reason) in SIMULATION_FAILURE_FINISH_REASONS


def simulation_exit_code(finish_reason: str) -> int:
    """Map a structured simulation finish reason to a process exit code.

    Library callers keep receiving structured results.  Process-level callers
    use this helper so clean and forced aborts share exit code 4, while a generic
    orchestrator exception retains the conventional exit code 1.
    """
    normalized = _normalize_finish_reason(finish_reason)
    if normalized in SIMULATION_ABORT_FINISH_REASONS:
        return SIMULATION_ABORT_EXIT_CODE
    if normalized == "error":
        return 1
    return 0


@dataclass
class SimulationResult:
    """Result from a completed simulation session.

    Carries all data needed for benchmarks, experiment analysis, and
    programmatic inspection.  Previously, detailed data was only
    persisted to session files; now it's available in-memory.
    """

    goal: str
    # Orchestrator flow-shape ("generative", "dm", "research", "benchmark",
    # or a caller-supplied label). Renamed from `persona` in 1.1 when the
    # persona system was hard-removed (persona_cleanup_and_mode_transition.md
    # Stages 3-5) — persisted pre-1.1 reports carry the old "persona" key,
    # which readers accept as a legacy alias.
    mode: str
    turns: int
    total_actions: int
    blocked_actions: int
    duration_s: float
    finish_reason: str = "unknown"
    summary: str = ""
    # Session identity (set after report is built)
    session_id: str = ""
    session_dir: str = ""
    campaign_analysis: dict[str, Any] = field(default_factory=dict)
    introspector: Any = None
    # Tool usage stats (from Executor.tool_usage_stats())
    tool_stats: dict[str, Any] = field(default_factory=dict)
    # Serialized action history (ActionRecord dicts)
    actions: list[dict[str, Any]] = field(default_factory=list)
    # Subsystem snapshot (from AUTIntrospector.benchmark_snapshot())
    subsystem_snapshot: dict[str, Any] = field(default_factory=dict)
    # JSON parse compliance (from json_parser counters)
    router_stats: dict[str, Any] = field(default_factory=dict)


def load_resume_context(session_id: str) -> dict[str, Any] | None:
    """Load a previous session's report and action log for resumption."""
    from maxim.utils.paths import sim_reports as _sim_reports_dir

    _reports_base = _sim_reports_dir()
    report_path = _reports_base / session_id / "report.json"
    if not report_path.exists():
        # Try fuzzy match — session_id might be a prefix
        reports_dir = _reports_base
        if reports_dir.exists():
            matches = sorted(
                [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith(session_id)],
                reverse=True,
            )
            if matches:
                report_path = matches[0] / "report.json"

    if not report_path.exists():
        logger.warning("Resume session not found: %s", session_id)
        return None

    try:
        with open(str(report_path), "r", encoding="utf-8") as f:
            report_data = json.load(f)
        from maxim.utils.format_version import check_format_version

        check_format_version(report_data, "session_report", log=logger)
        logger.info("Loaded previous session: %s", report_path.parent.name)
        return report_data
    except Exception as e:
        logger.warning("Failed to load resume session: %s", e)
        return None


def build_resume_prompt(report_data: dict[str, Any], goal: str, mode: str) -> str:
    """Build a context-rich prompt for resuming a previous simulation."""
    prev_goal = report_data.get("goal", "unknown")
    # Pre-1.1 reports persisted the mode under the legacy "persona" key.
    prev_mode = report_data.get("mode", report_data.get("persona", "unknown"))
    prev_turns = report_data.get("turns", 0)
    prev_actions = report_data.get("total_actions", 0)
    prev_blocked = report_data.get("blocked_actions", 0)
    prev_summary = report_data.get("llm_summary", "")
    prev_issues = report_data.get("llm_issues_found", [])
    prev_recommendations = report_data.get("llm_recommendations", [])
    prev_tool_usage = report_data.get("tool_usage", {})

    lines = [
        f"SIMULATION GOAL: {goal}",
        "",
        "You are RESUMING a previous simulation session.",
        f"You are the simulation orchestrator (mode: {mode}).",
        "",
        "## Previous Session Summary",
        f"Goal: {prev_goal}",
        f"Mode: {prev_mode}",
        f"Completed {prev_turns} turns, {prev_actions} actions ({prev_blocked} blocked)",
    ]

    if prev_summary:
        lines.append(f"Summary: {prev_summary}")

    if prev_issues:
        lines.append("Issues found:")
        for issue in prev_issues[:5]:
            lines.append(f"  - {issue}")

    if prev_recommendations:
        lines.append("Recommendations:")
        for rec in prev_recommendations[:5]:
            lines.append(f"  - {rec}")

    if prev_tool_usage:
        lines.append("Tool usage:")
        for tool, count in sorted(prev_tool_usage.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {tool}: {count}")

    lines.append("")
    lines.append(
        "Continue the simulation from where it left off. "
        "Build on the previous findings — don't repeat probes that already worked. "
        "Focus on areas the previous session identified as needing more testing. "
        "Use send_message to continue probing the agent."
    )

    return "\n".join(lines)


def build_basic_analysis(introspector: Any) -> dict[str, Any]:
    """Build a basic analysis dict for non-campaign runs (D-0b fix).

    Ensures research protocol always has analysis data to work with,
    even without a --campaign YAML.
    """
    if introspector is None:
        return {}
    try:
        return introspector.full_analysis(seed_keywords=[])
    except Exception as e:
        logger.debug("Basic analysis failed: %s", e)
        return {}
