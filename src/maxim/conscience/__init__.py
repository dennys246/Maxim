"""Maxim's conscience - safety enforcement and self-awareness."""

from __future__ import annotations

# FearAgent moved to maxim.agents - re-export for backward compatibility
from maxim.agents.fear_agent import (
    DangerCategory,
    FearAgent,
    Finding,
    ReviewResult,
    RiskLevel,
)

__all__ = [
    "DangerCategory",
    "FearAgent",
    "Finding",
    "ReviewResult",
    "RiskLevel",
]
