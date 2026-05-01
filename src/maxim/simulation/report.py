"""Post-simulation report: persist all data and optionally run LLM roundup.

After each simulation run, this module:
1. Saves all action records, AUT state, and sim metadata to a session directory
2. Optionally runs an LLM analysis to summarize outcomes and flag issues
3. Produces a human-readable report on stdout and a machine-readable JSON on disk
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SimulationReport:
    """Complete post-simulation report with all captured data."""

    # Session identity
    session_id: str = ""
    timestamp: str = ""
    goal: str = ""
    persona: str = ""
    language_model: str = ""
    aut_model: str = ""  # Separate AUT model when dual-LLM mode is active

    # Timing
    duration_s: float = 0.0
    turns: int = 0
    finish_reason: str = "unknown"

    # Action summary
    total_actions: int = 0
    blocked_actions: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)
    tool_success_rates: dict[str, float] = field(default_factory=dict)
    response_texts: list[str] = field(default_factory=list)

    # AUT cognitive state (from InspectAUTTool / direct access)
    aut_memories_formed: int = 0
    aut_causal_links: int = 0
    aut_nac_summary: dict[str, Any] = field(default_factory=dict)

    # Cost
    cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    # LLM roundup (filled by analyze_simulation)
    llm_summary: str = ""
    llm_issues_found: list[str] = field(default_factory=list)
    llm_recommendations: list[str] = field(default_factory=list)

    # LLM-initiated finish (set when orchestrator calls finish_simulation)
    llm_finish_status: str = ""
    llm_finish_reason: str = ""
    llm_finish_summary: str = ""

    # Substrate metrics (populated by FixtureDrivenOrchestrator, S1)
    substrate_metrics: dict[str, Any] = field(default_factory=dict)

    # Bio-system telemetry (Track 3: rich event-level data for research)
    bio_telemetry_path: str = ""  # Path to bio_telemetry.jsonl if saved


def _count_tokens(text: str, llm_router: Any | None) -> int:
    """Best-effort token count for a static template.

    Prefers the live ``llm_router.get_token_counter()`` so the count
    matches what the budgeter would charge. Falls back to a 4-char/token
    heuristic when no router is available — the heuristic is good enough
    for the V1 phase delta (we only need order-of-magnitude separation
    between "0 tokens" Phase A and "~1k tokens" Phase G).

    Pre-merge review fold: the inner ``except`` is narrowed to
    ``AttributeError`` (router missing the method) plus ``TypeError``
    (counter signature drift) so a real counter bug propagates loudly
    instead of silently returning the heuristic.
    """
    if not text:
        return 0
    if llm_router is not None:
        try:
            counter = llm_router.get_token_counter()
        except AttributeError:
            counter = None
        if counter is not None:
            try:
                return int(counter.count_tokens(text))
            except (AttributeError, TypeError):
                pass
    return max(1, len(text) // 4)


def build_report(
    *,
    goal: str,
    persona: str,
    bridge: Any,
    duration_s: float,
    finish_reason: str,
    aut_hippocampus: Any | None = None,
    aut_nac: Any | None = None,
    aut_memory_hub: Any | None = None,
    llm_router: Any | None = None,
    language_model: str = "",
    llm_finish_context: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> SimulationReport:
    """Build a SimulationReport from all available data sinks.

    Plan 4 follow-up (2026-04-14): ``session_id`` is now optionally
    supplied by the caller. The simulation orchestrator pre-generates
    the timestamp at sim entry (so it can thread it into every
    LLMWorker's ``request_context``) and forwards the same value here
    so the report directory name matches the session_id in the JSONL
    log trace. Legacy callers that don't supply it fall back to the
    old behavior (self-generated timestamp at report-build time).
    """
    if session_id is None:
        session_id = time.strftime("%Y%m%d_%H%M%S")

    all_actions = bridge.get_all_actions()
    blocked = [a for a in all_actions if a.blocked]

    # Tool usage breakdown
    tool_counts: dict[str, int] = {}
    tool_successes: dict[str, int] = {}
    response_texts: list[str] = []

    for a in all_actions:
        tool_counts[a.tool_name] = tool_counts.get(a.tool_name, 0) + 1
        if a.result_success:
            tool_successes[a.tool_name] = tool_successes.get(a.tool_name, 0) + 1
        if a.tool_name in ("respond", "speak") and a.result_output:
            response_texts.append(str(a.result_output)[:500])

    tool_rates = {}
    for tool, count in tool_counts.items():
        tool_rates[tool] = round(tool_successes.get(tool, 0) / count, 3) if count else 0.0

    # AUT cognitive state
    aut_memories = 0
    aut_links = 0
    nac_summary: dict[str, Any] = {}

    if aut_hippocampus is not None:
        try:
            aut_memories = len(aut_hippocampus)
        except Exception:
            pass

    if aut_nac is not None:
        try:
            aut_links = sum(len(v) for v in aut_nac._links.values())
            # Grab top confident links
            top_links = []
            for sig_links in aut_nac._links.values():
                for link in sig_links:
                    top_links.append(
                        {
                            "event": getattr(link, "event_signature", ""),
                            "outcome": getattr(link, "outcome_signature", ""),
                            "confidence": round(getattr(link, "confidence", 0), 3),
                            "observations": getattr(link, "observation_count", 0),
                        }
                    )
            top_links.sort(key=lambda x: x["confidence"], reverse=True)
            nac_summary = {
                "total_links": aut_links,
                "top_links": top_links[:5],
            }
        except Exception:
            pass

    # Cost data — session_cost is exact USD accumulated by the router during
    # THIS sim's lifetime, incremented at router.py:786 on every LLM request.
    # We deliberately do NOT fall back to the CostTracker hourly rolling
    # window: that window is persisted across process invocations and counts
    # cost events from prior unrelated sims (including past cloud runs), so
    # a local mistral run with zero actual cost would otherwise display the
    # previous Claude run's cost as its own. session_cost == 0 means this
    # session spent nothing, full stop.
    cost_usd = 0.0
    input_tokens = 0
    output_tokens = 0
    if llm_router is not None:
        try:
            cost_usd = float(getattr(llm_router, "session_cost", 0.0) or 0.0)
            tracker = getattr(llm_router, "_cost_tracker", None)
            if tracker and hasattr(tracker, "get_session_tokens"):
                tokens = tracker.get_session_tokens()
                input_tokens = int(tokens.get("input_tokens", 0))
                output_tokens = int(tokens.get("output_tokens", 0))
        except Exception as e:
            logger.debug("cost/token lookup failed: %s", e)

    ctx = llm_finish_context or {}
    return SimulationReport(
        session_id=session_id,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        goal=goal,
        persona=persona,
        language_model=language_model,
        duration_s=round(duration_s, 1),
        turns=bridge.turn_count,
        finish_reason=finish_reason,
        total_actions=len(all_actions),
        blocked_actions=len(blocked),
        tool_usage=tool_counts,
        tool_success_rates=tool_rates,
        response_texts=response_texts[:10],  # Cap at 10
        aut_memories_formed=aut_memories,
        aut_causal_links=aut_links,
        aut_nac_summary=nac_summary,
        cost_usd=round(cost_usd, 4),
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        llm_finish_status=str(ctx.get("status", "")),
        llm_finish_reason=str(ctx.get("reason", "")),
        llm_finish_summary=str(ctx.get("summary", "")),
    )


def save_report(report: SimulationReport, base_dir: str | None = None) -> Path:
    """Persist the full report as JSON to a session directory."""
    if base_dir is None:
        from maxim.utils.paths import sim_reports

        base_dir = str(sim_reports())
    session_dir = Path(base_dir) / report.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save bio telemetry (Track 3: rich event-level data for research)
    try:
        from maxim.simulation.bio_telemetry import BioTelemetryCollector

        telemetry_path = BioTelemetryCollector().save(session_dir)
        if telemetry_path is not None:
            report.bio_telemetry_path = str(telemetry_path)
    except Exception as e:
        logger.debug("Bio telemetry save failed: %s", e)

    report_path = session_dir / "report.json"
    from maxim.utils.atomic_io import atomic_write_json
    from maxim.utils.format_version import with_format_version

    atomic_write_json(str(report_path), with_format_version(asdict(report)))

    logger.info("Simulation report saved: %s", report_path)
    return report_path


def save_action_log(bridge: Any, base_dir: str, session_id: str) -> Path | None:
    """Save all action records as JSONL for post-hoc analysis."""
    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    log_path = session_dir / "actions.jsonl"
    try:
        with open(str(log_path), "w", encoding="utf-8") as f:
            for a in bridge.get_all_actions():
                entry = {
                    "timestamp": a.timestamp,
                    "tool": a.tool_name,
                    "params": a.tool_args,
                    "success": a.result_success,
                    "output": str(a.result_output)[:1000] if a.result_output else None,
                    "error": a.result_error,
                    "blocked": a.blocked,
                    "block_reason": a.block_reason,
                }
                f.write(json.dumps(entry, default=str) + "\n")
        logger.info("Action log saved: %s (%d records)", log_path, len(bridge.get_all_actions()))
        return log_path
    except Exception as e:
        logger.warning("Failed to save action log: %s", e)
        return None


def save_aut_state(
    *,
    hippocampus: Any | None,
    nac: Any | None,
    base_dir: str,
    session_id: str,
) -> None:
    """Persist AUT hippocampus and NAc state for post-hoc analysis."""
    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    if hippocampus is not None:
        try:
            hippo_path = session_dir / "aut_hippocampus.json"
            hippocampus.save(str(hippo_path))
            logger.info("AUT hippocampus saved: %s (%d memories)", hippo_path, len(hippocampus))
        except Exception as e:
            logger.debug("Failed to save AUT hippocampus: %s", e)

    if nac is not None:
        try:
            nac_path = session_dir / "aut_nac.json"
            nac.save(str(nac_path))
            logger.info("AUT NAc saved: %s", nac_path)
        except Exception as e:
            logger.debug("Failed to save AUT NAc: %s", e)


def analyze_simulation(
    report: SimulationReport,
    llm_router: Any | None = None,
) -> SimulationReport:
    """Run LLM-powered post-simulation roundup.

    Summarizes outcomes, identifies bugs/issues, and provides recommendations.
    Modifies report in-place with llm_summary, llm_issues_found, llm_recommendations.
    """
    if llm_router is None:
        logger.info("No LLM router available, skipping simulation roundup")
        return report

    # Build analysis prompt from report data
    prompt = _build_roundup_prompt(report)

    try:
        response = llm_router.generate_json(
            prompt,
            system_override=(
                "You are a QA analyst reviewing a simulation test run of a robot assistant. "
                "Analyze the results and provide: 1) A concise summary, 2) Any issues or bugs found, "
                "3) Recommendations for improvement. Be specific and actionable."
            ),
            temperature=0.2,
            max_tokens=2048,
            request_context={"agent": "sim_roundup", "lane": "large"},
        )

        if isinstance(response, dict):
            report.llm_summary = str(response.get("summary", ""))
            issues = response.get("issues", response.get("issues_found", []))
            if isinstance(issues, list):
                report.llm_issues_found = [str(i) for i in issues]
            recs = response.get("recommendations", [])
            if isinstance(recs, list):
                report.llm_recommendations = [str(r) for r in recs]
        elif isinstance(response, str):
            report.llm_summary = response

        logger.info(
            "Simulation roundup complete: %d issues, %d recommendations",
            len(report.llm_issues_found),
            len(report.llm_recommendations),
        )

    except Exception as e:
        logger.warning("LLM roundup failed: %s", e)
        report.llm_summary = f"Roundup failed: {e}"

    return report


def _build_roundup_prompt(report: SimulationReport) -> str:
    """Build the analysis prompt from report data."""
    lines = [
        f"# Simulation Report — {report.goal}",
        f"Persona: {report.persona}",
        f"Model: {report.language_model}",
        f"Duration: {report.duration_s}s, Turns: {report.turns}",
        f"Finish reason: {report.finish_reason}",
        "",
        f"## Actions: {report.total_actions} total, {report.blocked_actions} blocked",
        "",
        "Tool usage:",
    ]
    for tool, count in sorted(report.tool_usage.items(), key=lambda x: -x[1]):
        rate = report.tool_success_rates.get(tool, 0)
        lines.append(f"  {tool}: {count} calls, {rate:.0%} success")

    if report.response_texts:
        lines.append("")
        lines.append("## AUT Responses (first 5):")
        for i, text in enumerate(report.response_texts[:5], 1):
            lines.append(f"  {i}. {text[:200]}")

    lines.append("")
    lines.append("## AUT Cognitive State:")
    lines.append(f"  Episodic memories formed: {report.aut_memories_formed}")
    lines.append(f"  Causal links learned: {report.aut_causal_links}")
    if report.aut_nac_summary.get("top_links"):
        lines.append("  Top causal links:")
        for link in report.aut_nac_summary["top_links"][:3]:
            lines.append(
                f"    {link['event']} → {link['outcome']} (conf={link['confidence']}, obs={link['observations']})"
            )

    if report.cost_usd > 0:
        lines.append("")
        lines.append(
            f"## Cost: ${report.cost_usd:.4f} ({report.total_input_tokens} in, {report.total_output_tokens} out)"
        )

    lines.append("")
    lines.append('Respond with JSON: {"summary": "...", "issues": ["..."], "recommendations": ["..."]}')

    return "\n".join(lines)


def print_report(report: SimulationReport) -> None:
    """Print a human-readable report to stdout via display_summary()."""
    from maxim.simulation.sim_logger import display_summary

    lines = [
        f"SIMULATION REPORT — {report.session_id}",
        f"  Goal: {report.goal}",
        f"  Persona: {report.persona}",
        f"  Model: {report.language_model}",
        f"  Duration: {report.duration_s}s | Turns: {report.turns}",
        f"  Finish: {report.finish_reason}",
    ]
    if report.llm_finish_status:
        lines.append(f"  Orchestrator status: {report.llm_finish_status}")
        if report.llm_finish_reason:
            lines.append(f"    Reason: {report.llm_finish_reason}")
        if report.llm_finish_summary:
            indented = "\n      ".join(report.llm_finish_summary.splitlines())
            lines.append(f"    Summary:\n      {indented}")
    lines.append(f"  Actions: {report.total_actions} ({report.blocked_actions} blocked)")
    lines.append(f"  AUT Memories: {report.aut_memories_formed} | Causal Links: {report.aut_causal_links}")
    if report.cost_usd > 0:
        lines.append(
            f"  Cost: ${report.cost_usd:.4f} ({report.total_input_tokens + report.total_output_tokens} tokens)"
        )

    if report.tool_usage:
        lines.append("")
        lines.append("  Tool Usage:")
        for tool, count in sorted(report.tool_usage.items(), key=lambda x: -x[1]):
            rate = report.tool_success_rates.get(tool, 0)
            lines.append(f"    {tool}: {count} calls ({rate:.0%} success)")

    if report.llm_summary:
        lines.append("")
        lines.append("  LLM Analysis:")
        lines.append(f"    {report.llm_summary}")

    if report.llm_issues_found:
        lines.append("")
        lines.append(f"  Issues Found ({len(report.llm_issues_found)}):")
        for issue in report.llm_issues_found:
            lines.append(f"    - {issue}")

    if report.llm_recommendations:
        lines.append("")
        lines.append(f"  Recommendations ({len(report.llm_recommendations)}):")
        for rec in report.llm_recommendations:
            lines.append(f"    - {rec}")

    display_summary(lines)
