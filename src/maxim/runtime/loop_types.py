"""Typed dataclasses for agentic loop state (Phase 1 of loop modularization).

Replaces stringly-typed ``state.data["pending_*"]`` dicts with typed structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PendingConfirmation:
    """A tool action awaiting user yes/no/modify response."""

    action: dict[str, Any]
    reasoning: str
    confidence: float
    tool_name: str


@dataclass
class PendingModification:
    """User requested changes to a proposed action."""

    original_action: dict[str, Any]
    original_reasoning: str
    original_tool_name: str
    user_modification: str
    timestamp: float


@dataclass
class TimeoutRetry:
    """LLM timed out; awaiting user decision on retry."""

    original_request: Any
    timeout_s: float


@dataclass
class PlanModificationContext:
    """Context for revising a rejected/modified plan."""

    original_plan: str | None
    original_action: dict[str, Any] | None
    user_modification: str | None


@dataclass
class ActionFollowup:
    """Pending follow-up after a tool with followup_type completes."""

    tool: str
    result: str | None
    original_query: str
    followup_type: str  # "process" | "respond" | "engage"
    mode: str
    timestamp: float
