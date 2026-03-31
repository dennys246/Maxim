"""ExplainTool — surface provenance traces for user inspection.

Supports: current cycle, recent cycle, session summary, concept history,
session export (markdown and JSON).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from maxim.provenance.types import ProvenanceVerbosity
from maxim.tools.base import Tool

if TYPE_CHECKING:
    from maxim.provenance.collector import ProvenanceCollector


class ExplainTool(Tool):
    """Surface provenance trace for the current or recent decision."""

    name = "explain"
    description = (
        "Show why Maxim made a decision — what memories, concepts, "
        "and predictions informed it. Supports: current cycle, recent "
        "cycles, session summary, concept history, and session export."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "'current' | 'recent' | 'summary' | 'export' | "
                    "'export_json' | 'history' | 'concept:<name>' | run_id"
                ),
                "default": "current",
            },
            "verbosity": {
                "type": "integer",
                "description": "Detail level: 1=compact, 2=verbose",
                "default": 1,
            },
        },
    }

    def __init__(self, collector: ProvenanceCollector) -> None:
        self._collector = collector

    def execute(self, query: str = "current", verbosity: int = 1, **kw: Any) -> Any:
        from maxim.provenance.render import (
            render_activities,
            render_session_report,
            render_summary,
            render_trace,
        )

        v = ProvenanceVerbosity(min(verbosity, 2))

        if query == "summary":
            traces = self._collector.recent_traces(limit=20)
            activities = self._collector.recent_activities(limit=50)
            return render_summary(traces, activities)

        if query == "export":
            traces = self._collector.recent_traces(limit=100)
            activities = self._collector.recent_activities(limit=200)
            return render_session_report(
                traces, activities, self._collector.session_id, v,
            )

        if query == "export_json":
            traces = self._collector.recent_traces(limit=100)
            activities = self._collector.recent_activities(limit=200)
            return json.dumps(
                {
                    "session_id": self._collector.session_id,
                    "traces": [t.to_dict() for t in traces],
                    "activities": [a.to_dict() for a in activities],
                },
                indent=2,
                default=str,
            )

        if query == "history":
            return self._query_session_history()

        if query.startswith("concept:"):
            concept_name = query[len("concept:"):]
            return self._query_concept_history(concept_name)

        if query == "current":
            # Try in-progress trace first, fall back to most recent completed
            with self._collector._lock:
                in_progress = [
                    t for t in self._collector._traces.values()
                    if not t.completed
                ]
            if in_progress:
                trace = max(in_progress, key=lambda t: t.started_at)
            else:
                traces = self._collector.recent_traces(limit=1)
                trace = traces[0] if traces else None
        elif query == "recent":
            traces = self._collector.recent_traces(limit=1)
            trace = traces[0] if traces else None
        else:
            trace = self._collector.get_trace(query)

        if trace is None:
            return "No provenance trace available."

        result = render_trace(trace, verbosity=v)

        if v >= ProvenanceVerbosity.VERBOSE:
            activities = self._collector.recent_activities(limit=10)
            if activities:
                result += "\n" + render_activities(activities, verbosity=v)

        return result

    def _query_concept_history(self, concept_name: str) -> str:
        """Query cross-run concept history via ProvenanceStore."""
        if not self._collector._store:
            return "Provenance persistence not enabled."
        results = self._collector._store.query_concept(concept_name)
        if not results:
            return f"No provenance records found for concept '{concept_name}'."

        lines = [f"## Concept History: {concept_name}\n"]
        for r in results:
            session = r.get("session_id", "?")[:12]
            action = r.get("action", "")
            component = r.get("component", "")
            lines.append(f"- **{component}** [session:{session}] {action}")
        return "\n".join(lines)

    def _query_session_history(self) -> str:
        """List recent sessions from manifest."""
        if not self._collector._store:
            return "Provenance persistence not enabled."
        sessions = self._collector._store.load_recent_sessions(limit=10)
        if not sessions:
            return "No session history available."
        manifest = self._collector._store._load_manifest()
        lines = ["## Session History\n"]
        for sid in sessions:
            info = manifest.get(sid, {})
            traces = info.get("completed_traces", "?")
            lines.append(f"- `session:{sid[:12]}` — {traces} cycles")
        return "\n".join(lines)
