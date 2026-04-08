# tools/introspection.py
"""Read-only introspection tools exposing biological subsystems to the LLM.

Each tool wraps existing query methods on Maxim's memory, prediction,
pain, temporal, and energy subsystems. No tool modifies agent state.
All output is formatted for LLM consumption with bounded size.
"""

from __future__ import annotations

import time
from typing import Any

from maxim.tools.base import Tool, ToolResult


# ---------------------------------------------------------------------------
# Tool 1: memory_recall — Episodic Memory Search
# ---------------------------------------------------------------------------


class MemoryRecallTool(Tool):
    """Search episodic memories with optional spreading activation."""

    name = "memory_recall"
    description = (
        "Search your episodic memories. Filter by goal, tool, success/failure, "
        "detected objects, or time range. Use 'expand=true' to find associated "
        "memories via spreading activation through the associative graph."
    )
    input_schema = {
        "query": (str, None),
        "tool_name": (str, None),
        "success": (bool, None),
        "object": (str, None),
        "person": (str, None),
        "mode": (str, None),
        "expand": (bool, False),
        "limit": (int, 5),
    }
    MANUAL_KEYWORDS = {
        "remember",
        "recall",
        "memory",
        "past",
        "history",
        "previous",
        "before",
        "last",
        "when",
        "episode",
    }

    def __init__(self, hippocampus: Any = None) -> None:
        super().__init__()
        self._hippocampus = hippocampus

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._hippocampus is None:
            return ToolResult(success=True, output={"available": False, "reason": "Hippocampus not available"})

        limit = min(int(kwargs.get("limit", 5)), 20)
        filters: dict[str, Any] = {}
        query_text = kwargs.get("query")
        if query_text:
            # Pass as query for relevance ranking (not just goal filter)
            filters["query"] = query_text
        if kwargs.get("tool_name"):
            filters["tool"] = kwargs["tool_name"]
        if kwargs.get("success") is not None:
            filters["success"] = kwargs["success"]
        if kwargs.get("object"):
            filters["object_detected"] = kwargs["object"]
        if kwargs.get("person"):
            filters["person_detected"] = kwargs["person"]
        if kwargs.get("mode"):
            filters["mode"] = kwargs["mode"]

        try:
            memories = self._hippocampus.recall(limit=limit, **filters)
        except Exception as e:
            return ToolResult(success=False, error=f"Memory recall failed: {e}")

        # Optionally expand via spreading activation
        if kwargs.get("expand") and memories:
            try:
                seed_ids = [m.id for m in memories[:3]]
                associated = self._hippocampus.recall_associated(
                    seed_ids=seed_ids,
                    limit=limit,
                )
                # Merge, dedup
                seen = {m.id for m in memories}
                for mem, activation in associated:
                    if mem.id not in seen:
                        memories.append(mem)
                        seen.add(mem.id)
                        if len(memories) >= limit:
                            break
            except Exception:
                pass  # Expansion is best-effort

        results = [_format_episodic_memory(m) for m in memories[:limit]]
        return ToolResult(
            success=True,
            output={
                "count": len(results),
                "memories": results,
                "expanded": bool(kwargs.get("expand")),
            },
        )


# ---------------------------------------------------------------------------
# Tool 2: predict_outcome — NAc Causal Predictions
# ---------------------------------------------------------------------------


class PredictOutcomeTool(Tool):
    """Query causal predictions for a tool before executing it."""

    name = "predict_outcome"
    description = (
        "Query what the causal learning system (NAc) predicts will happen "
        "if you execute a tool. Returns predicted outcome, confidence, "
        "expected delay, and all possible outcomes with their valences."
    )
    input_schema = {
        "tool_name": str,
        "context": (dict, None),
        "include_all_outcomes": (bool, True),
    }
    MANUAL_KEYWORDS = {
        "predict",
        "outcome",
        "expect",
        "happen",
        "result",
        "likely",
        "probability",
        "confidence",
        "risk",
    }

    def __init__(self, nac: Any = None) -> None:
        super().__init__()
        self._nac = nac

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._nac is None:
            return ToolResult(success=True, output={"available": False, "reason": "NAc not available"})

        tool_name = kwargs["tool_name"]
        context = kwargs.get("context")
        event_sig = f"tool:{tool_name}"

        try:
            primary = self._nac.predict("tool", event_sig, context)
        except Exception as e:
            return ToolResult(success=False, error=f"Prediction failed: {e}")

        result: dict[str, Any] = {"tool_name": tool_name}

        if primary is None:
            result["prediction"] = None
            result["message"] = f"No learned predictions for '{tool_name}' yet."
        else:
            result["prediction"] = {
                "predicted_outcome": primary.predicted_outcome,
                "valence": primary.predicted_valence.value,
                "value": round(primary.predicted_value, 3),
                "confidence": round(primary.confidence, 3),
                "expected_delay_s": round(primary.predicted_delay, 2),
                "delay_range_s": [round(primary.delay_bounds[0], 2), round(primary.delay_bounds[1], 2)],
                "context_match": round(primary.context_match, 3),
            }

        if kwargs.get("include_all_outcomes", True):
            try:
                all_outcomes = self._nac.predict_all_outcomes("tool", event_sig, context)
                result["all_outcomes"] = [
                    {
                        "outcome": o.predicted_outcome,
                        "valence": o.predicted_valence.value,
                        "value": round(o.predicted_value, 3),
                        "confidence": round(o.confidence, 3),
                        "delay_s": round(o.predicted_delay, 2),
                    }
                    for o in all_outcomes[:10]
                ]
            except Exception:
                result["all_outcomes"] = []

        return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# Tool 3: causal_links — Inspect Cause-Effect Relationships
# ---------------------------------------------------------------------------


class CausalLinksTool(Tool):
    """Inspect learned cause-effect relationships in the NAc."""

    name = "causal_links"
    description = (
        "Inspect your learned cause-effect relationships. See what events "
        "lead to what outcomes, filter by positive/negative valence, and "
        "trace which episodic memories contributed to each causal link."
    )
    input_schema = {
        "event": (str, None),
        "outcome": (str, None),
        "memory_id": (str, None),
        "valence": (str, None),
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {"cause", "effect", "causal", "link", "learn", "outcome", "because", "why", "consequence", "lead"}

    def __init__(self, nac: Any = None) -> None:
        super().__init__()
        self._nac = nac

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._nac is None:
            return ToolResult(success=True, output={"available": False, "reason": "NAc not available"})

        limit = min(int(kwargs.get("limit", 10)), 20)
        links = []

        try:
            if kwargs.get("event"):
                links = self._nac.get_links_for_event(kwargs["event"])
            elif kwargs.get("outcome"):
                links = self._nac.get_links_for_outcome(kwargs["outcome"])
            elif kwargs.get("memory_id"):
                links = self._nac.get_links_for(kwargs["memory_id"])
            else:
                # Return promotion candidates (high-confidence links)
                candidates = self._nac.get_promotion_candidates(min_confidence=0.3, min_observations=2)
                return ToolResult(
                    success=True,
                    output={
                        "count": len(candidates[:limit]),
                        "links": [
                            {
                                "pattern": c.pattern_name,
                                "confidence": round(c.confidence, 3),
                                "source_memories": len(c.source_memory_ids),
                                "metadata": c.metadata,
                            }
                            for c in candidates[:limit]
                        ],
                    },
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Causal link query failed: {e}")

        # Filter by valence
        valence_filter = kwargs.get("valence")
        if valence_filter:
            links = [lnk for lnk in links if lnk.outcome_valence.value == valence_filter]

        results = [_format_causal_link(lnk) for lnk in links[:limit]]
        return ToolResult(success=True, output={"count": len(results), "links": results})


# ---------------------------------------------------------------------------
# Tool 4: pain_history — Pain and Fear Queries
# ---------------------------------------------------------------------------


class PainHistoryTool(Tool):
    """Query pain signals and fear gate status."""

    name = "pain_history"
    description = (
        "Check your pain and fear history. See recent pain signals from "
        "tool failures, movement errors, and timeouts. Query whether the "
        "fear system would block a specific action."
    )
    input_schema = {
        "check_action": (str, None),
        "action_params": (dict, None),
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {"pain", "hurt", "fear", "danger", "safe", "risk", "error", "failure", "avoid", "block", "afraid"}

    def __init__(self, pain_detector: Any = None, fear_agent: Any = None) -> None:
        super().__init__()
        self._pain = pain_detector
        self._fear = fear_agent

    def execute(self, **kwargs: Any) -> ToolResult:
        result: dict[str, Any] = {}

        # Pain stats
        if self._pain is not None:
            try:
                result["pain_stats"] = self._pain.get_stats()
            except Exception:
                result["pain_stats"] = {"available": False}
        else:
            result["pain_stats"] = {"available": False, "reason": "PainDetector not available"}

        # Fear gate check
        if kwargs.get("check_action") and self._fear is not None:
            try:
                review = self._fear.review_action(
                    action_type=kwargs["check_action"],
                    action_params=kwargs.get("action_params", {}),
                )
                result["fear_check"] = {
                    "action": kwargs["check_action"],
                    "allowed": review.allow,
                    "risk_level": review.risk,
                    "findings": [str(f) for f in review.findings[:5]],
                }
            except Exception as e:
                result["fear_check"] = {"error": str(e)}

        return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# Tool 5: temporal_patterns — SCN Time-Based Queries
# ---------------------------------------------------------------------------


class TemporalPatternsTool(Tool):
    """Discover time-based patterns in experience."""

    name = "temporal_patterns"
    description = (
        "Discover time-based patterns in your experience. Find memories "
        "from specific times of day, days of week, or discover rhythmic "
        "patterns (e.g., failures cluster on Monday mornings)."
    )
    input_schema = {
        "hour": (int, None),
        "day": (int, None),
        "discover_rhythms": (bool, False),
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {
        "time",
        "when",
        "morning",
        "evening",
        "monday",
        "pattern",
        "rhythm",
        "schedule",
        "circadian",
        "daily",
        "weekly",
    }

    def __init__(self, scn: Any = None) -> None:
        super().__init__()
        self._scn = scn

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._scn is None:
            return ToolResult(success=True, output={"available": False, "reason": "SCN not available"})

        result: dict[str, Any] = {}

        try:
            if kwargs.get("discover_rhythms"):
                patterns = self._scn.find_rhythmic_patterns(min_occurrences=3)
                result["rhythmic_patterns"] = {
                    rhythm_type: [(bin_id, count) for bin_id, count in bins] for rhythm_type, bins in patterns.items()
                }
            else:
                memory_ids = self._scn.query_intersection(
                    hour=kwargs.get("hour"),
                    day=kwargs.get("day"),
                )
                limit = min(int(kwargs.get("limit", 10)), 20)
                result["memory_ids"] = list(memory_ids)[:limit]
                result["total_matches"] = len(memory_ids)
        except Exception as e:
            return ToolResult(success=False, error=f"Temporal query failed: {e}")

        return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# Tool 6: energy_status — Resource Consumption
# ---------------------------------------------------------------------------


class EnergyStatusTool(Tool):
    """Check energy and resource consumption."""

    name = "energy_status"
    description = (
        "Check your energy and resource consumption. See recent token usage, "
        "inference costs, and overall energy budget status."
    )
    input_schema = {
        "window_seconds": (float, 300.0),
    }
    MANUAL_KEYWORDS = {
        "energy",
        "budget",
        "cost",
        "token",
        "resource",
        "usage",
        "consumption",
        "efficiency",
        "expensive",
    }

    def __init__(self, energy_tracker: Any = None) -> None:
        super().__init__()
        self._tracker = energy_tracker

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._tracker is None:
            return ToolResult(success=True, output={"available": False, "reason": "EnergyTracker not available"})

        window = float(kwargs.get("window_seconds", 300.0))
        try:
            result = {
                "recent": self._tracker.get_window_stats(window),
                "total": self._tracker.get_total_stats(),
            }
        except Exception as e:
            return ToolResult(success=False, error=f"Energy query failed: {e}")

        return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# Tool 7: concept_query — Semantic Knowledge Search
# ---------------------------------------------------------------------------


class ConceptQueryTool(Tool):
    """Search semantic knowledge base for concepts and relationships."""

    name = "concept_query"
    description = (
        "Search your semantic knowledge base for concepts, facts, and "
        "relationships. Query by name, category, or explore relationships "
        "between concepts (IS_A, PART_OF, CAUSES, EXECUTES_WITH, etc.)."
    )
    input_schema = {
        "name": (str, None),
        "category": (str, None),
        "concept_id": (str, None),
        "relationship_type": (str, None),
        "limit": (int, 10),
    }
    MANUAL_KEYWORDS = {
        "concept",
        "know",
        "knowledge",
        "what",
        "define",
        "relationship",
        "related",
        "category",
        "semantic",
        "meaning",
    }

    def __init__(self, atl: Any = None) -> None:
        super().__init__()
        self._atl = atl

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._atl is None:
            return ToolResult(success=True, output={"available": False, "reason": "ATL not available"})

        limit = min(int(kwargs.get("limit", 10)), 20)

        # Relationship query
        if kwargs.get("concept_id"):
            try:
                rels = self._atl.find_by_relationship(
                    concept_id=kwargs["concept_id"],
                    rel_type=kwargs.get("relationship_type"),
                    limit=limit,
                )
                return ToolResult(
                    success=True,
                    output={
                        "concept_id": kwargs["concept_id"],
                        "relationships": [
                            {
                                "target_id": tid,
                                "type": rel.relationship_type,
                                "weight": round(rel.weight, 3),
                                "confidence": round(rel.confidence, 3),
                            }
                            for tid, rel in rels
                        ],
                    },
                )
            except Exception as e:
                return ToolResult(success=False, error=f"Relationship query failed: {e}")

        # Concept search
        filters: dict[str, Any] = {}
        if kwargs.get("name"):
            filters["name"] = kwargs["name"]
        if kwargs.get("category"):
            filters["category"] = kwargs["category"]

        try:
            concepts = self._atl.recall(limit=limit, **filters)
            results = [_format_concept(c) for c in concepts]
        except Exception as e:
            return ToolResult(success=False, error=f"Concept query failed: {e}")

        return ToolResult(success=True, output={"count": len(results), "concepts": results})


# ---------------------------------------------------------------------------
# Tool 8: scene_summary — Visual Scene
# ---------------------------------------------------------------------------


class SceneSummaryTool(Tool):
    """Get a summary of the current visual scene."""

    name = "scene_summary"
    description = (
        "Get a summary of the current visual scene. Shows the most salient "
        "objects, where you're currently looking, and what's novel or familiar."
    )
    input_schema = {
        "top_n": (int, 5),
        "include_attention": (bool, True),
    }
    MANUAL_KEYWORDS = {
        "see",
        "look",
        "scene",
        "visible",
        "object",
        "detect",
        "salient",
        "novel",
        "attention",
        "focus",
        "gaze",
    }

    def __init__(self, salience_network: Any = None, attention_network: Any = None) -> None:
        super().__init__()
        self._salience = salience_network
        self._attention = attention_network

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._salience is None and self._attention is None:
            return ToolResult(
                success=True, output={"available": False, "reason": "No vision subsystems available (headless mode)"}
            )

        result: dict[str, Any] = {"timestamp": time.time()}
        top_n = min(int(kwargs.get("top_n", 5)), 15)

        if self._salience is not None:
            try:
                result["salient_objects"] = self._salience.get_top_salient(n=top_n)
            except Exception:
                result["salient_objects"] = []

        if kwargs.get("include_attention", True) and self._attention is not None:
            try:
                result["current_focus"] = self._attention.get_current_focus()
                result["dwell_time_s"] = self._attention.get_dwell_time()
                result["next_suggested_target"] = self._attention.get_next_target()
            except Exception:
                pass

        return ToolResult(success=True, output=result)


# ---------------------------------------------------------------------------
# Tool 9: similarity_search — EC Similarity Queries
# ---------------------------------------------------------------------------


class SimilaritySearchTool(Tool):
    """Find similar situations via the Entorhinal Cortex."""

    name = "similarity_search"
    description = (
        "Find situations similar to a description or to a past memory. "
        "Uses multi-modal similarity (structural, temporal, semantic) to "
        "find related experiences across your memory."
    )
    input_schema = {
        "tool_name": (str, None),
        "memory_id": (str, None),
        "context": (dict, None),
        "limit": (int, 5),
    }
    MANUAL_KEYWORDS = {"similar", "like", "related", "comparable", "same", "analogy", "pattern", "match", "familiar"}

    def __init__(self, ec: Any = None) -> None:
        super().__init__()
        self._ec = ec

    def execute(self, **kwargs: Any) -> ToolResult:
        if self._ec is None:
            return ToolResult(success=True, output={"available": False, "reason": "EC not available"})

        limit = min(int(kwargs.get("limit", 5)), 20)

        try:
            if kwargs.get("memory_id"):
                results = self._ec.find_similar_by_memory(
                    memory_id=kwargs["memory_id"],
                    k=limit,
                )
            elif kwargs.get("tool_name"):
                results = self._ec.find_similar_situations_for_action(
                    tool_name=kwargs["tool_name"],
                    context=kwargs.get("context"),
                    k=limit,
                )
                # Convert (memory_id, SituationSignature) to (memory_id, score)
                results = [(mid, 1.0) for mid, _sig in results]
            else:
                return ToolResult(
                    success=True,
                    output={
                        "message": "Provide either tool_name or memory_id to search.",
                    },
                )
        except Exception as e:
            return ToolResult(success=False, error=f"Similarity search failed: {e}")

        return ToolResult(
            success=True,
            output={
                "count": len(results),
                "similar": [{"memory_id": mid, "similarity": round(score, 3)} for mid, score in results],
            },
        )


# ---------------------------------------------------------------------------
# Tool 10: system_stats — Aggregate Health Summary
# ---------------------------------------------------------------------------


class SystemStatsTool(Tool):
    """Aggregate health summary of all biological subsystems."""

    name = "system_stats"
    description = (
        "Get a health summary of all your biological subsystems: memory counts, "
        "causal link stats, energy usage, pain history, and learning progress."
    )
    input_schema = {}
    MANUAL_KEYWORDS = {"status", "health", "stats", "system", "summary", "overview", "diagnostic", "how", "doing"}

    def __init__(
        self,
        hippocampus: Any = None,
        nac: Any = None,
        ec: Any = None,
        atl: Any = None,
        energy_tracker: Any = None,
        pain_detector: Any = None,
        significance_learner: Any = None,
    ) -> None:
        super().__init__()
        self._hippocampus = hippocampus
        self._nac = nac
        self._ec = ec
        self._atl = atl
        self._energy = energy_tracker
        self._pain = pain_detector
        self._significance = significance_learner

    def execute(self, **kwargs: Any) -> ToolResult:
        stats: dict[str, Any] = {}

        for name, subsystem, method in [
            ("hippocampus", self._hippocampus, "stats"),
            ("nac", self._nac, "stats"),
            ("ec", self._ec, "stats"),
            ("atl", self._atl, "stats"),
            ("energy", self._energy, "get_total_stats"),
            ("pain", self._pain, "get_stats"),
        ]:
            if subsystem is not None:
                try:
                    fn = getattr(subsystem, method, None)
                    if fn:
                        stats[name] = fn()
                    else:
                        stats[name] = {"available": True, "stats_method": "not found"}
                except Exception as e:
                    stats[name] = {"error": str(e)}
            else:
                stats[name] = {"available": False}

        # Significance learner: report current heuristic weights
        if self._significance is not None:
            try:
                weights = {}
                for h in self._significance._heuristics:
                    weights[h.name] = round(h.weight, 4)
                stats["significance_weights"] = weights
                stats["significance_stuck"] = self._significance.detect_stuck()
            except Exception:
                pass

        return ToolResult(success=True, output=stats)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _format_episodic_memory(m: Any) -> dict[str, Any]:
    """Format an EpisodicMemory for LLM consumption."""
    result: dict[str, Any] = {"id": m.id, "timestamp": m.timestamp}

    if hasattr(m, "context"):
        ctx = m.context
        result["goal"] = getattr(ctx, "active_goal", None)
        result["mode"] = getattr(ctx, "active_mode", None)

    if hasattr(m, "action"):
        act = m.action
        result["tool"] = getattr(act, "tool_name", None)

    if hasattr(m, "outcome"):
        out = m.outcome
        if isinstance(out, dict):
            result["success"] = out.get("success")
            result["error"] = out.get("error")
            if "reflection" in out:
                result["reflection"] = out["reflection"]
        else:
            result["success"] = getattr(out, "success", None)
            result["error"] = getattr(out, "error", None)
            r = getattr(out, "result", None)
            if isinstance(r, dict) and "reflection" in r:
                result["reflection"] = r["reflection"]

    if hasattr(m, "perception"):
        p = m.perception
        result["salience"] = round(getattr(p, "salience", 0.0), 3)
        result["novelty"] = round(getattr(p, "novelty", 0.0), 3)
        objs = getattr(p, "detected_objects", [])
        if objs:
            result["detected_objects"] = objs[:5]

    return result


def _format_causal_link(link: Any) -> dict[str, Any]:
    """Format a CausalLink for LLM consumption."""
    result: dict[str, Any] = {
        "id": link.id,
        "event": link.event_signature,
        "outcome": link.outcome_signature,
        "valence": link.outcome_valence.value,
        "predicted_value": round(link.predicted_value, 3),
        "confidence": round(link.confidence, 3),
        "observations": link.observation_count,
    }
    if link.temporal_delta and hasattr(link.temporal_delta, "mean"):
        result["mean_delay_s"] = round(link.temporal_delta.mean, 2)
    if link.last_rpe is not None:
        result["last_rpe"] = round(link.last_rpe, 3)
    if link.memory_ids:
        result["contributing_memories"] = len(link.memory_ids)
    return result


def _format_concept(c: Any) -> dict[str, Any]:
    """Format a SemanticMemory/Concept for LLM consumption."""
    result: dict[str, Any] = {
        "id": c.id,
        "name": getattr(c, "name", ""),
        "category": getattr(c, "category", ""),
        "confidence": round(getattr(c, "confidence", 0.0), 3),
    }
    if hasattr(c, "definition") and c.definition:
        result["definition"] = c.definition[:200]
    if hasattr(c, "access_count"):
        result["access_count"] = c.access_count
    return result


__all__ = [
    "CausalLinksTool",
    "ConceptQueryTool",
    "EnergyStatusTool",
    "MemoryRecallTool",
    "PainHistoryTool",
    "PredictOutcomeTool",
    "SceneSummaryTool",
    "SimilaritySearchTool",
    "SystemStatsTool",
    "TemporalPatternsTool",
]
