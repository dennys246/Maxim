"""Agent self-introspection tools — learning, memory, performance, pain.

Read-only tools that let the agent reason about its own internal state.
Each tool reads from existing bio-system data sources and formats the
output for LLM consumption with bounded size (~4KB max).

Registered in build_tool_registry when the relevant data source exists.
"""

from __future__ import annotations

import time
from typing import Any

from maxim.tools.base import Tool, ToolOutput


# ---------------------------------------------------------------------------
# nac_stats — causal learning state
# ---------------------------------------------------------------------------


class NacStatsTool(Tool):
    """Inspect your causal learning state — what you've learned works and what doesn't."""

    name = "nac_stats"
    description = (
        "Show causal learning statistics: total observations, number of "
        "causal links, top-rewarded tools (by mean reward prediction error), "
        "and current reward bias. Helps you reason about what actions have "
        "been effective and what to avoid."
    )
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, nac: Any = None) -> None:
        super().__init__()
        self._nac = nac

    def execute(self, **kwargs: Any) -> ToolOutput:
        if self._nac is None:
            return ToolOutput(success=False, error="NAc (causal learning) not available")

        nac = self._nac
        with nac._lock:
            total_obs = nac._total_observations
            total_links = sum(len(links) for links in nac._links.values())
            pending_count = len(nac._pending_events)

            # Top-rewarded tool signatures by mean confidence × value
            tool_scores: dict[str, list[float]] = {}
            for sig, links in nac._links.items():
                for link in links:
                    tool_scores.setdefault(sig, []).append(link.confidence * link.value)

            top_tools = []
            for sig, scores in sorted(
                tool_scores.items(),
                key=lambda kv: sum(kv[1]) / len(kv[1]) if kv[1] else 0,
                reverse=True,
            )[:10]:
                mean_score = sum(scores) / len(scores) if scores else 0
                top_tools.append(
                    {
                        "signature": sig,
                        "links": len(scores),
                        "mean_score": round(mean_score, 3),
                    }
                )

            # Reward bias summary
            bias_entries = []
            for (agent_id, node_id), bias in sorted(
                nac._reward_bias.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]:
                if bias > 0.001:
                    bias_entries.append(
                        {
                            "node": node_id,
                            "bias": round(bias, 4),
                        }
                    )

        result = {
            "total_observations": total_obs,
            "total_causal_links": total_links,
            "pending_events": pending_count,
            "top_rewarded": top_tools,
            "reward_bias": bias_entries,
        }
        return ToolOutput(success=True, output=result)


# ---------------------------------------------------------------------------
# memory_pressure — hippocampal memory health
# ---------------------------------------------------------------------------


class MemoryPressureTool(Tool):
    """Check your memory health — how many memories, what tiers, consolidation state."""

    name = "memory_pressure"
    description = (
        "Show memory health: total episodic memories, per-tier breakdown "
        "(full vs compressed vs long-term), graph node/edge counts, and "
        "consolidation candidates. Helps you understand your memory load."
    )
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, hippocampus: Any = None) -> None:
        super().__init__()
        self._hippocampus = hippocampus

    def execute(self, **kwargs: Any) -> ToolOutput:
        if self._hippocampus is None:
            return ToolOutput(success=False, error="Hippocampus (memory) not available")

        stats = self._hippocampus.stats()
        result = {
            "total_memories": stats.get("total_memories", 0),
            "full_memories": stats.get("full_memories", 0),
            "compressed_memories": stats.get("compressed_memories", 0),
            "long_term_memories": stats.get("long_term_memories", 0),
            "consolidation_candidates": stats.get("consolidation_candidates", 0),
            "graph_nodes": stats.get("graph_nodes", 0),
            "graph_edges": stats.get("graph_edges", 0),
            "index_keys": stats.get("index_keys", 0),
        }
        return ToolOutput(success=True, output=result)


# ---------------------------------------------------------------------------
# loop_stats — runtime performance
# ---------------------------------------------------------------------------


class LoopStatsTool(Tool):
    """Check your own runtime performance — cycle timing and step count."""

    name = "loop_stats"
    description = (
        "Show runtime loop statistics: total steps executed, uptime, "
        "and current run ID. Helps you understand how long you've been "
        "running and how much work you've done."
    )
    input_schema = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, loop_controller: Any = None) -> None:
        super().__init__()
        self._lc = loop_controller

    def execute(self, **kwargs: Any) -> ToolOutput:
        if self._lc is None:
            return ToolOutput(success=False, error="Loop controller not available")

        lc = self._lc
        result = {
            "max_steps": lc.max_steps,
            "run_id": lc.run_id,
            "target_hz": round(1.0 / lc.target_period, 1) if lc.target_period > 0 else 0,
            "agent_name": lc.agent_name,
        }
        return ToolOutput(success=True, output=result)


# ---------------------------------------------------------------------------
# pain_triggers_active — current pain state
# ---------------------------------------------------------------------------


class PainTriggersActiveTool(Tool):
    """Check your current pain state — what hurts and why."""

    name = "pain_triggers_active"
    description = (
        "Show currently active pain signals: source entity, failure mode, "
        "intensity, and when each signal fired. Helps you reason about "
        "what's causing discomfort and whether to change approach."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Max number of recent signals to return",
                "default": 10,
            },
        },
    }

    def __init__(self, pain_bus: Any = None) -> None:
        super().__init__()
        self._pain_bus = pain_bus

    def execute(self, limit: int = 10, **kwargs: Any) -> ToolOutput:
        if self._pain_bus is None:
            return ToolOutput(success=False, error="PainBus not available")

        # Get recent pain reactions from the underlying reaction bus history
        recent_pain = self._pain_bus.reaction_bus.history(kind="pain")
        now = time.time()

        signals = []
        for reaction in recent_pain[-limit:]:
            ctx = reaction.context
            age_s = now - reaction.timestamp if hasattr(reaction, "timestamp") else None
            entry: dict[str, Any] = {
                "source": ctx.source if ctx else "unknown",
                "intensity": reaction.intensity,
            }
            if ctx and ctx.bindings:
                for key, snap in ctx.bindings.items():
                    entry[key] = snap.value if hasattr(snap, "value") else str(snap)
            if age_s is not None:
                entry["age_seconds"] = round(age_s, 1)
            signals.append(entry)

        result = {
            "active_count": len(signals),
            "signals": signals,
        }
        return ToolOutput(success=True, output=result)
