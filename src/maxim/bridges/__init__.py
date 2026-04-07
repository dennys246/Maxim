"""Bridge modules for cross-system memory integration.

Bridges enable bidirectional learning between the Hippocampus memory system
and external perception/decision/action systems. Each bridge:
- Enriches external systems with historical context from memory
- Updates memory based on new observations and outcomes

Available Bridges:
- SpatialMemoryBridge: Hippocampus <-> SpatialMap <-> AttentionNetwork
- SalienceMemoryBridge: Hippocampus <-> EC <-> SalienceNetwork
- PlanHistoryBridge: Hippocampus <-> NAc (plan template retrieval)
- EscalationLearningBridge: Hippocampus <-> SCN/NAc (learned thresholds)
- FearCircuitBridge: Hippocampus <-> FearAgent <-> NAc (learned risk patterns)
"""

from __future__ import annotations

from maxim.bridges.spatial_bridge import SpatialMemoryBridge
from maxim.bridges.salience_bridge import SalienceMemoryBridge
from maxim.bridges.planning_bridge import PlanHistoryBridge, PlanTemplate
from maxim.bridges.escalation_bridge import EscalationLearningBridge
from maxim.bridges.fear_bridge import FearCircuitBridge
from maxim.bridges.tool_pain_bridge import ToolPainBridge

__all__ = [
    "ToolPainBridge",
    "SpatialMemoryBridge",
    "SalienceMemoryBridge",
    "PlanHistoryBridge",
    "PlanTemplate",
    "EscalationLearningBridge",
    "FearCircuitBridge",
]
