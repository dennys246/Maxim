"""Typed dataclasses for agentic loop state (Phase 1 of loop modularization).

Replaces stringly-typed ``state.data["pending_*"]`` dicts with typed structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.agents.llm_types import LLMProposal


@dataclass(frozen=True)
class PendingConfirmation:
    """A tool action awaiting the user's yes/no/modify, held on ``LoopController.pending_confirmation``.

    Built ONLY by ``from_proposal`` (#1133): ``source`` is the ``LLMProposal`` the action came from, so the
    confirmed action is credited to the situation it was PROPOSED in (``source.clusters``, the #1083
    rule), and ``tool_name`` is normalized once (``"unknown"`` when the action names none). Until #1133
    this lived in ``state.data["pending_confirmation"]`` as a dict and the proposal was dropped when the
    confirmation was asked, so the situation was lost.

    Runtime-ephemeral: lives on the controller for the ticks between the question and the answer, never
    persisted (``state.data``, which is snapshotted to disk, no longer carries it) and never crossing a
    wire, so CC3 forward-compat is out of scope.
    """

    action: dict[str, Any]
    reasoning: str
    confidence: float
    tool_name: str
    source: LLMProposal

    @classmethod
    def from_proposal(cls, proposal: LLMProposal) -> PendingConfirmation:
        """The one producer: the proposal the autonomy check parked for confirmation."""
        action = proposal.action or {}
        return cls(
            action=action,
            reasoning=proposal.reasoning,
            confidence=proposal.confidence,
            tool_name=action.get("tool_name") or "unknown",
            source=proposal,
        )

    @property
    def params(self) -> dict[str, Any]:
        """The action's params; a non-dict value reads as ``{}`` (the executor's own guard)."""
        raw = self.action.get("params")
        return raw if isinstance(raw, dict) else {}

    def policy_view(self) -> dict[str, Any]:
        """The dict ``sim.resolve_confirmation`` (a ``ResponsePolicy``) has always read.

        A compatibility shape: ``ResponsePolicy.resolve_confirmation(confirmation: dict)`` is public and may be
        overridden by a user's subclass, so it keeps receiving the dict it always did, not this record."""
        return {
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "tool_name": self.tool_name,
        }


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
