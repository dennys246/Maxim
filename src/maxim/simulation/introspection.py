"""Observer — programmatic access to AUT cognitive state.

Provides clean Python methods for querying the agent-under-test's memory,
causal links, predictions, energy, and system stats without going through
the tool dispatch layer.  Used by:

- Post-campaign analysis in the orchestrator (replaces registry hack)
- InspectAUTTool (delegates here instead of duplicating logic)
- Future standalone experiment scripts
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class Observer:
    """Read-only introspection into an AUT's cognitive subsystems.

    All methods return plain dicts/lists — no ToolOutput wrapping.
    Thread-safe: all reads are snapshot-style (no mutations).
    """

    ALLOWED_QUERIES = frozenset(
        {
            "memory_recall",
            "causal_links",
            "predict_outcome",
            "pain_history",
            "energy_status",
            "system_stats",
            "concept_query",
            "temporal_patterns",
        }
    )

    def __init__(
        self,
        *,
        hippocampus: Any = None,
        nac: Any = None,
        memory_hub: Any = None,
        energy_registry: Any = None,
        pain_detector: Any = None,
    ) -> None:
        self._hippocampus = hippocampus
        self._nac = nac
        self._memory_hub = memory_hub
        self._energy_registry = energy_registry
        self._pain_detector = pain_detector

    # ── Individual query methods ─────────────────────────────────────

    def memory_recall(self, keyword: str = "", limit: int = 10, goal: str = "", tool: str = "") -> dict:
        """Search episodic memory by keyword/goal/tool."""
        if self._hippocampus is None:
            return {"available": False, "reason": "hippocampus not wired"}
        limit = min(limit, 20)
        # Use keyword-based search if keyword given, else standard recall
        if keyword:
            memories = self._hippocampus.search_by_content(keyword, limit=limit)
        else:
            memories = self._hippocampus.recall(limit=limit, goal=goal or None, tool=tool or None)
        return {
            "available": True,
            "count": len(memories),
            "total_stored": len(self._hippocampus),
            "memories": [
                {
                    "id": getattr(m, "id", "?"),
                    "goal": getattr(getattr(m, "context", None), "goal", ""),
                    "tool": getattr(getattr(m, "action", None), "tool_used", ""),
                    "success": getattr(getattr(m, "outcome", None), "success", None),
                    "timestamp": getattr(m, "timestamp", 0),
                }
                for m in memories
            ],
        }

    def causal_links(self, event_signature: str = "") -> dict:
        """Query NAc causal links, optionally filtered by event signature."""
        if self._nac is None:
            return {"available": False, "reason": "NAc not wired"}
        if event_signature:
            links = self._nac.get_links_for_event(event_signature)
        else:
            links = []
            for sig_links in self._nac._links.values():
                links.extend(sig_links)
        return {
            "available": True,
            "link_count": len(links),
            "links": [
                {
                    "event": getattr(link, "event_signature", ""),
                    "outcome": getattr(link, "outcome_signature", ""),
                    "valence": str(getattr(link, "valence", "")),
                    "confidence": round(getattr(link, "confidence", 0), 3),
                    "observations": getattr(link, "observation_count", 0),
                }
                for link in links[:10]
            ],
        }

    def predict_outcome(self, event_signature: str, event_type: str = "tool") -> dict:
        """Get NAc prediction for a specific event."""
        if self._nac is None:
            return {"available": False, "reason": "NAc not wired"}
        if not event_signature:
            return {"available": True, "error": "event_signature required"}
        prediction = self._nac.predict(event_type, event_signature)
        if prediction is None:
            return {"available": True, "prediction": None, "reason": "no data for this event"}
        return {
            "available": True,
            "prediction": {
                "predicted_value": round(getattr(prediction, "predicted_value", 0), 3),
                "confidence": round(getattr(prediction, "confidence", 0), 3),
                "valence": str(getattr(prediction, "valence", "")),
                "observation_count": getattr(prediction, "observation_count", 0),
            },
        }

    def pain_history(self, limit: int = 10) -> dict:
        """Search episodic memory for pain-related entries."""
        if self._hippocampus is None:
            return {"available": False, "reason": "hippocampus not wired"}
        pain_memories = self._hippocampus.search_by_content("pain", limit=min(limit, 20))
        return {
            "available": True,
            "pain_memory_count": len(pain_memories),
            "memories": [
                {
                    "id": getattr(m, "id", "?"),
                    "timestamp": getattr(m, "timestamp", 0),
                }
                for m in pain_memories
            ],
        }

    def energy_status(self) -> dict:
        """Get energy/cost tracking stats."""
        if self._energy_registry is None:
            return {"available": False, "reason": "energy registry not wired"}
        try:
            stats = self._energy_registry.get_stats()
            return {"available": True, **stats}
        except Exception:
            return {"available": True, "error": "stats unavailable"}

    def system_stats(self) -> dict:
        """Aggregate health summary across all subsystems."""
        stats: dict[str, Any] = {}
        if self._hippocampus is not None:
            stats["hippocampus_memories"] = len(self._hippocampus)
        if self._nac is not None:
            total_links = sum(len(v) for v in self._nac._links.values())
            stats["nac_causal_links"] = total_links
        if self._memory_hub is not None:
            if hasattr(self._memory_hub, "atl") and self._memory_hub.atl:
                stats["atl_concepts"] = len(self._memory_hub.atl)
            if hasattr(self._memory_hub, "ec") and self._memory_hub.ec:
                stats["ec_signatures"] = len(self._memory_hub.ec)
        stats["available"] = True
        return stats

    def concept_query(self, name: str = "", category: str = "", limit: int = 5) -> dict:
        """Query ATL semantic concepts."""
        hub = self._memory_hub
        if hub is None or not hasattr(hub, "atl") or hub.atl is None:
            return {"available": False, "reason": "ATL not wired"}
        limit = min(limit, 10)
        concepts = hub.atl.recall(limit=limit, name=name or None, category=category or None)
        return {
            "available": True,
            "count": len(concepts),
            "concepts": [
                {
                    "name": getattr(c, "name", ""),
                    "category": getattr(c, "category", ""),
                    "confidence": round(getattr(c, "confidence", 0), 3),
                }
                for c in concepts
            ],
        }

    def temporal_patterns(self) -> dict:
        """Query SCN temporal rhythm state."""
        hub = self._memory_hub
        if hub is None or not hasattr(hub, "scn") or hub.scn is None:
            return {"available": False, "reason": "SCN not wired"}
        try:
            current = hub.scn.current_phase()
            return {
                "available": True,
                "current_phase": current if isinstance(current, dict) else str(current),
            }
        except Exception:
            return {"available": True, "error": "SCN query failed"}

    def pain_stats(self) -> dict:
        """Get direct pain signal counters from PainDetector.

        Unlike pain_history() which searches hippocampal memories for
        the word "pain", this returns actual PainDetector counters:
        total signals fired, per-type breakdown, etc.
        """
        if self._pain_detector is None:
            return {"available": False, "reason": "PainDetector not wired"}
        try:
            stats = self._pain_detector.get_stats()
            return {"available": True, **stats}
        except Exception:
            return {"available": True, "error": "PainDetector stats unavailable"}

    # ── Benchmark data collection ────────────────────────────────────

    def benchmark_snapshot(self, seed_keywords: list[str] | None = None) -> dict:
        """Comprehensive snapshot for benchmark metric computation.

        Extends full_analysis() with raw subsystem stats (graph topology,
        NAc observation counts, PainDetector counters) that the aggregated
        system_stats() method doesn't surface.
        """
        snapshot = self.full_analysis(seed_keywords=seed_keywords)

        # Raw hippocampus stats (graph topology, compression counts)
        if self._hippocampus is not None:
            try:
                snapshot["hippocampus_stats"] = self._hippocampus.stats()
            except Exception:
                pass

        # Raw NAc stats (event signatures, observation counts, priors)
        if self._nac is not None:
            try:
                snapshot["nac_stats"] = self._nac.stats()
            except Exception:
                pass

        # Direct pain stats (not memory-search proxy)
        pain = self.pain_stats()
        if pain.get("available"):
            snapshot["pain_stats"] = pain

        return snapshot

    # ── Dispatch (for InspectAUTTool compatibility) ──────────────────

    def dispatch(self, query: str, params: dict | None = None) -> Any:
        """Dispatch a named query with params dict.

        This is the bridge between the tool interface (query + params dict)
        and the typed methods above.
        """
        params = params or {}
        if query == "memory_recall":
            return self.memory_recall(
                keyword=params.get("keyword", ""),
                limit=int(params.get("limit", 5)),
                goal=params.get("goal", ""),
                tool=params.get("tool", ""),
            )
        elif query == "causal_links":
            return self.causal_links(event_signature=params.get("event_signature", ""))
        elif query == "predict_outcome":
            return self.predict_outcome(
                event_signature=params.get("event_signature", ""),
                event_type=params.get("event_type", "tool"),
            )
        elif query == "pain_history":
            return self.pain_history(limit=int(params.get("limit", 10)))
        elif query == "energy_status":
            return self.energy_status()
        elif query == "system_stats":
            return self.system_stats()
        elif query == "concept_query":
            return self.concept_query(
                name=params.get("name", ""),
                category=params.get("category", ""),
                limit=int(params.get("limit", 5)),
            )
        elif query == "temporal_patterns":
            return self.temporal_patterns()
        return {"error": f"unknown query: {query}"}

    # ── Batch analysis ───────────────────────────────────────────────

    def full_analysis(self, seed_keywords: list[str] | None = None) -> dict:
        """Run a comprehensive analysis across all subsystems.

        Useful for post-campaign experiment recording. Returns a dict
        with results from each subsystem.
        """
        analysis: dict[str, Any] = {}

        # Memory recall per seed keyword
        if seed_keywords:
            recall_results = {}
            for kw in seed_keywords:
                recall_results[kw] = self.memory_recall(keyword=kw)
            analysis["memory_recall"] = recall_results

        analysis["system_stats"] = self.system_stats()
        analysis["causal_links"] = self.causal_links()

        # Include concepts if ATL is available
        concepts = self.concept_query(limit=10)
        if concepts.get("available"):
            analysis["concepts"] = concepts

        # Include temporal state if SCN is available
        temporal = self.temporal_patterns()
        if temporal.get("available"):
            analysis["temporal"] = temporal

        return analysis

    def summarize(self, analysis: dict) -> str:
        """Produce a one-paragraph text summary of an analysis dict."""
        parts: list[str] = []

        stats = analysis.get("system_stats", {})
        if stats.get("available"):
            mem = stats.get("hippocampus_memories", 0)
            links = stats.get("nac_causal_links", 0)
            parts.append(f"{mem} episodic memories, {links} causal links")

        recall = analysis.get("memory_recall", {})
        for kw, result in recall.items():
            count = result.get("count", 0)
            parts.append(f"recall '{kw}': {count} hit(s)")

        causal = analysis.get("causal_links", {})
        if causal.get("available"):
            parts.append(f"{causal.get('link_count', 0)} total causal links")

        return "; ".join(parts) if parts else "No subsystem data available."


AUTIntrospector = Observer  # Deprecated alias — remove in 0.2.0
