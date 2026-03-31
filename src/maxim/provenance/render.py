"""Trace rendering — four render functions for different contexts.

Same data, different formats:
- render_trace: single cycle trace as markdown
- render_activities: background activity log
- render_summary: multi-cycle summary table
- render_session_report: full session export
"""

from __future__ import annotations

import time
from datetime import datetime

from maxim.provenance.types import (
    ProvenanceEntry,
    ProvenanceTrace,
    ProvenanceVerbosity,
)


def render_trace(
    trace: ProvenanceTrace,
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.COMPACT,
) -> str:
    """Render a single cycle trace as markdown."""
    ts = datetime.fromtimestamp(trace.started_at).strftime("%Y-%m-%d_%H%M%S")
    lines: list[str] = []

    if verbosity >= ProvenanceVerbosity.VERBOSE:
        lines.append(f"## Decision Trace [{ts}] session:{trace.session_id[:12]}")
    else:
        lines.append(f"## Decision Trace [{ts}]")

    lines.append("")

    base_ts = trace.started_at
    for entry in trace.entries:
        stage = f"**{entry.stage.value}**"
        if verbosity >= ProvenanceVerbosity.VERBOSE:
            entry_ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
            delta_ms = (entry.timestamp - base_ts) * 1000
            stage = f"**{entry.stage.value}** [{entry_ts}, +{delta_ms:.0f}ms]"

        lines.append(f"{stage} → {entry.action}")

        for ref in entry.sources:
            if verbosity >= ProvenanceVerbosity.VERBOSE:
                lines.append(f"  - {ref}")
            else:
                lines.append(f"  - {ref.short()}")

        if verbosity >= ProvenanceVerbosity.VERBOSE and entry.metadata:
            if "reasoning" in entry.metadata and entry.metadata["reasoning"]:
                lines.append(f'  Reasoning: "{entry.metadata["reasoning"]}"')
            if "alternatives" in entry.metadata:
                alts = entry.metadata["alternatives"]
                if alts:
                    lines.append(f"  Alternatives: {len(alts)} considered")

    return "\n".join(lines)


def render_activities(
    activities: list[ProvenanceEntry],
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.COMPACT,
) -> str:
    """Render background activity log as markdown."""
    if not activities:
        return ""

    lines = ["", "### Background Activity", ""]
    for entry in activities:
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
        line = f"**{entry.stage.value}** [{ts}] {entry.component} → {entry.action}"
        lines.append(line)
        if verbosity >= ProvenanceVerbosity.VERBOSE:
            for ref in entry.sources:
                lines.append(f"  - {ref}")

    return "\n".join(lines)


def render_summary(
    traces: list[ProvenanceTrace],
    activities: list[ProvenanceEntry] | None = None,
) -> str:
    """Render multi-cycle summary table with optional activity digest."""
    if not traces:
        return "No provenance traces available."

    lines = ["## Provenance Summary", ""]
    lines.append(f"**Cycles:** {len(traces)} completed")

    success_count = sum(
        1 for t in traces
        for e in t.entries
        if e.stage.value == "outcome" and "Success=True" in e.action
    )
    total = len(traces)
    if total > 0:
        lines.append(f"**Success rate:** {success_count}/{total} ({100*success_count//total}%)")

    lines.append("")
    lines.append("| # | Goal | Outcome | Duration |")
    lines.append("|---|------|---------|----------|")

    for i, trace in enumerate(traces, 1):
        goal = "?"
        outcome = "?"
        for entry in trace.entries:
            if entry.stage.value == "decision":
                goal = entry.action.split("(")[0].strip()
            if entry.stage.value == "outcome":
                outcome = entry.action
        duration = ""
        if trace.entries:
            duration = f"{(trace.entries[-1].timestamp - trace.started_at):.1f}s"
        lines.append(f"| {i} | {goal} | {outcome} | {duration} |")

    if activities:
        lines.append("")
        # Activity digest by component
        component_counts: dict[str, int] = {}
        for a in activities:
            component_counts[a.component] = component_counts.get(a.component, 0) + 1
        lines.append("### Activity Digest")
        for comp, count in sorted(component_counts.items()):
            lines.append(f"- {comp}: {count} operations")

    return "\n".join(lines)


def render_session_report(
    traces: list[ProvenanceTrace],
    activities: list[ProvenanceEntry],
    session_id: str,
    verbosity: ProvenanceVerbosity = ProvenanceVerbosity.VERBOSE,
) -> str:
    """Full session report for export/sharing."""
    lines = [f"# Session Report: {session_id[:12]}", ""]

    if traces:
        first_ts = min(t.started_at for t in traces)
        last_ts = max(
            e.timestamp
            for t in traces
            for e in t.entries
        ) if any(t.entries for t in traces) else first_ts
        duration = last_ts - first_ts
        started = datetime.fromtimestamp(first_ts).strftime("%Y-%m-%d %H:%M:%S")
        mins = int(duration // 60)
        secs = int(duration % 60)
        lines.append(f"**Started:** {started} | **Duration:** {mins}m {secs}s")
    else:
        lines.append("**No traces recorded.**")

    lines.append("")
    lines.append(render_summary(traces, activities))
    lines.append("")

    lines.append("## Cycle Details")
    lines.append("")
    for trace in traces:
        lines.append(render_trace(trace, verbosity=verbosity))
        lines.append("")

    if activities:
        lines.append(render_activities(activities, verbosity=verbosity))

    return "\n".join(lines)
