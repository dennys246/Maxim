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

    # Lane / latency breakdown (from MetricsRegistry — populated when at least
    # one LLM call landed). Each lane name maps to its snapshot dict
    # (jobs_completed, p50_latency_ms, p99_latency_ms, etc).  Aggregate
    # latency fields are computed across lanes weighted by call count.
    lane_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Bio-system event counts (counted from sim_logger records — single
    # source of truth, no parallel counters to maintain).
    pain_events_count: int = 0
    fear_blocks_count: int = 0
    learn_events_count: int = 0

    # Provider failure attribution — empty dict if no provider had a
    # consecutive_errors > 0 or last_error set during the session.  Each
    # entry: provider_key → {consecutive_errors, last_error, backoff_s}.
    provider_failures: dict[str, dict[str, Any]] = field(default_factory=dict)

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

    # Lane / latency breakdown from MetricsRegistry.  Aggregates p50/p99
    # weighted by jobs_completed across lanes; the per-lane snapshots stay
    # in lane_breakdown for forensic inspection.
    lane_breakdown: dict[str, dict[str, Any]] = {}
    latency_p50_ms = 0.0
    latency_p99_ms = 0.0
    try:
        from maxim.models.language.lane_metrics import get_metrics_registry

        lane_breakdown = get_metrics_registry().snapshot()
        if lane_breakdown:
            total_calls = sum(int(snap.get("jobs_completed", 0) or 0) for snap in lane_breakdown.values())
            if total_calls > 0:
                p50_sum = 0.0
                p99_sum = 0.0
                for snap in lane_breakdown.values():
                    n = int(snap.get("jobs_completed", 0) or 0)
                    if n > 0:
                        p50_sum += float(snap.get("p50_latency_ms", 0.0) or 0.0) * n
                        p99_sum += float(snap.get("p99_latency_ms", 0.0) or 0.0) * n
                latency_p50_ms = round(p50_sum / total_calls, 1)
                latency_p99_ms = round(p99_sum / total_calls, 1)
    except Exception as e:
        logger.debug("lane metrics snapshot failed: %s", e)

    # Bio-system event counts from the sim_logger record buffer.  Counts
    # subsystems exactly once per emission, so a Reaction that fans out
    # to PainBus + ReactionBus shows as one PAIN event (the publisher) +
    # one REACTION event (the typed forward) — both meaningful.
    pain_events_count = 0
    fear_blocks_count = 0
    learn_events_count = 0
    try:
        from maxim.simulation.sim_logger import get_sim_records

        for rec in get_sim_records():
            sub = rec.get("subsystem", "")
            if sub == "PAIN":
                pain_events_count += 1
            elif sub == "BLOCKED":
                fear_blocks_count += 1
            elif sub == "LEARN":
                learn_events_count += 1
    except Exception as e:
        logger.debug("sim_log event count failed: %s", e)

    # Provider failure attribution from router._provider_states.  Only
    # surface providers that actually saw failures during the session —
    # healthy providers are noise.
    provider_failures: dict[str, dict[str, Any]] = {}
    if llm_router is not None:
        try:
            states = getattr(llm_router, "_provider_states", {}) or {}
            now = time.time()
            for key, state in states.items():
                errs = int(getattr(state, "consecutive_errors", 0) or 0)
                last_err = str(getattr(state, "last_error", "") or "")
                backoff_until = float(getattr(state, "backoff_until", 0.0) or 0.0)
                if errs > 0 or last_err:
                    provider_failures[key] = {
                        "consecutive_errors": errs,
                        "last_error": last_err[:200],
                        "backoff_remaining_s": round(max(0.0, backoff_until - now), 1),
                    }
        except Exception as e:
            logger.debug("provider failure snapshot failed: %s", e)

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
        lane_breakdown=lane_breakdown,
        latency_p50_ms=latency_p50_ms,
        latency_p99_ms=latency_p99_ms,
        pain_events_count=pain_events_count,
        fear_blocks_count=fear_blocks_count,
        learn_events_count=learn_events_count,
        provider_failures=provider_failures,
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
    """Save all action records as JSONL for post-hoc analysis.

    The first line is a header record carrying ``_format_version`` per
    CLAUDE.md CC1 (Stage 0b, release_0_9_1.md). Per-action records
    follow, one per line. Each carries Stage 0b telemetry fields
    (``agent_id``, ``session_id``, ``entity_class``) populated by
    ``InstrumentedExecutor`` from the bound ``RequestContext`` —
    ``None`` when the context was unbound at execution time (e.g.,
    pre-0b sims, headless API runs).

    Format-version evolution rule: appending optional fields to the
    per-action record is back-compat (existing parsers ignore unknown
    keys); removing or renaming fields requires a major bump.
    """
    session_dir = Path(base_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    log_path = session_dir / "actions.jsonl"
    try:
        with open(str(log_path), "w", encoding="utf-8") as f:
            # Stage 0b: format-version header (one-line schema marker
            # at the top of the JSONL — existing per-record parsers
            # ignore unknown top-level keys, so this is back-compat).
            header = {
                "_format_version": "1.0",
                "_record_kind": "header",
                "session_id": session_id,
            }
            f.write(json.dumps(header) + "\n")
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
                    # Stage 0b telemetry — None when RequestContext was unbound
                    # (pre-0b sims, headless API) or entity_class couldn't be
                    # derived from tool_args.
                    "agent_id": a.agent_id,
                    "session_id": a.session_id,
                    "entity_class": a.entity_class,
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
    ec: Any | None = None,
    atl: Any | None = None,
) -> None:
    """Persist AUT hippocampus, NAc, EC, and ATL state for post-hoc analysis.

    ``ec`` and ``atl`` are keyword-only with a None default for back-compat
    with existing callers; pass the live instances (``memory_hub.ec`` /
    ``memory_hub.atl``) to write ``aut_ec.json`` / ``aut_atl.json``
    alongside the hippocampus/NAc dumps. Required for substrate-divergence
    analyzers (``analysis/substrate_diff.py::ec_diff`` and the Roy-5a
    cosine-localization analyzer) that key off persisted EC centroid
    state. Missing args silently skip the corresponding file — same shape
    as hippocampus/NAc handling above.
    """
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

    if ec is not None:
        try:
            ec_path = session_dir / "aut_ec.json"
            ec.save(str(ec_path))
            logger.info("AUT EC saved: %s", ec_path)
        except Exception as e:
            logger.debug("Failed to save AUT EC: %s", e)

    if atl is not None:
        try:
            atl_path = session_dir / "aut_atl.json"
            atl.save(str(atl_path))
            logger.info("AUT ATL saved: %s", atl_path)
        except Exception as e:
            logger.debug("Failed to save AUT ATL: %s", e)


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


def print_report(report: SimulationReport, *, session_dir: Path | str | None = None) -> None:
    """Print a human-readable report to stdout via display_summary().

    Args:
        report: The simulation report to render.
        session_dir: Optional path of the saved session directory.  When
            provided, surfaced as the final line so users can locate
            ``report.json`` / ``actions.jsonl`` / ``aut_nac.json`` without
            having to grep startup banners.  Always produced — replaces
            the previously-conditional ``Report saved: ...`` banner that
            only fired on certain code paths.
    """
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

    # Bio-system event summary — surfaces what previously only existed in
    # the JSONL trail.  ``learn_events_count`` is the headline count of
    # substrate-weight changes / tier promotions / consolidations that
    # fired during the session.
    bio_parts = []
    if report.learn_events_count > 0:
        bio_parts.append(f"▲ Learn: {report.learn_events_count}")
    if report.pain_events_count > 0:
        bio_parts.append(f"Pain: {report.pain_events_count}")
    if report.fear_blocks_count > 0:
        bio_parts.append(f"Blocked: {report.fear_blocks_count}")
    if bio_parts:
        lines.append("  Bio events: " + " | ".join(bio_parts))

    if report.cost_usd > 0:
        lines.append(
            f"  Cost: ${report.cost_usd:.4f} ({report.total_input_tokens + report.total_output_tokens} tokens)"
        )

    # Latency summary — only show when at least one LLM call completed.
    if report.latency_p50_ms > 0:
        lines.append(f"  Latency: p50={report.latency_p50_ms:.0f}ms, p99={report.latency_p99_ms:.0f}ms")
        if report.lane_breakdown:
            for lane, snap in sorted(report.lane_breakdown.items()):
                jobs = int(snap.get("jobs_completed", 0) or 0)
                if jobs == 0:
                    continue
                p50 = float(snap.get("p50_latency_ms", 0) or 0)
                p99 = float(snap.get("p99_latency_ms", 0) or 0)
                fail = int(snap.get("jobs_failed", 0) or 0)
                fail_str = f", {fail} failed" if fail > 0 else ""
                lines.append(f"    {lane}: {jobs} calls, p50={p50:.0f}ms, p99={p99:.0f}ms{fail_str}")

    # Top causal links — already computed for the LLM analysis prompt
    # but never surfaced to the user before.
    top_links = report.aut_nac_summary.get("top_links", []) if report.aut_nac_summary else []
    if top_links:
        lines.append("")
        lines.append("  Top Causal Links:")
        for link in top_links[:5]:
            lines.append(
                f"    {link.get('event', '?')} → {link.get('outcome', '?')} "
                f"(conf={link.get('confidence', 0):.2f}, obs={link.get('observations', 0)})"
            )

    # Provider failures — silenced/cooled-down providers that the user
    # might want to know about (auth rejected, model missing, overloaded).
    if report.provider_failures:
        lines.append("")
        lines.append("  Provider Failures:")
        for provider, info in sorted(report.provider_failures.items()):
            errs = info.get("consecutive_errors", 0)
            last = info.get("last_error", "")
            cooldown = info.get("backoff_remaining_s", 0)
            cooldown_str = f", cooldown={cooldown:.0f}s" if cooldown > 0 else ""
            lines.append(f"    {provider}: {errs} errors, last={last!r}{cooldown_str}")

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

    # Always surface the saved session location so users can find the
    # full report.json / actions.jsonl / aut_nac.json artifacts without
    # having to grep startup banners.
    if session_dir is not None:
        lines.append("")
        lines.append(f"  Session: {session_dir}")

    display_summary(lines)


def emit_report_json(report: SimulationReport, path: str = "-") -> None:
    """Emit the full report as JSON for CI / grading pipelines.

    Args:
        report: Report to serialize.
        path: Destination — ``"-"`` writes to stdout, otherwise treated as
            a file path that will be created with parents.

    File writes are atomic (write-temp + rename) per the project
    persistence convention and carry ``_format_version`` per the CC1
    invariant so downstream consumers can detect schema drift.
    """
    from maxim.utils.format_version import with_format_version

    payload_dict = with_format_version(asdict(report))
    payload = json.dumps(payload_dict, default=str, indent=2)
    if path in ("-", ""):
        print(payload)
        return
    out = Path(path)
    if out.is_dir():
        raise IsADirectoryError(f"emit_report_json: path is a directory: {out!s}. Pass a file path or '-' for stdout.")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write — write to a sibling temp file, then os.replace so a
    # mid-write crash leaves the previous file intact instead of a
    # truncated half-JSON the CI pipeline cannot parse.
    from maxim.utils.atomic_io import atomic_write_text

    atomic_write_text(str(out), payload)
