"""Skill base class — an atomic, reusable capability for Maxim.

A Skill encapsulates a single well-defined capability (e.g., streaming
video, tracking an object, recording audio). Skills expose tools for
the agentic runtime and provide LLM context about their current state.

Skills are composed into Protocols for higher-level operational profiles.
They can also be activated independently via the agentic runtime.

Lifecycle:
    IDLE → ACTIVATING → ACTIVE → DEACTIVATING → IDLE
                      → FAILED (on error)

Inspired by ROS2 action lifecycle, BehaviorTree.CPP skill states,
and SayCan affordance checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.tools.base import Tool

__all__ = ["Skill", "SkillConfig", "SkillState", "SkillResult"]


class SkillState(Enum):
    """Explicit lifecycle state for a skill.

    Provides richer information than a boolean is_active — the LLM and
    monitoring systems can distinguish "never started" from "failed" from
    "shutting down".

    States:
        IDLE         — not running, ready to activate
        ACTIVATING   — activate() in progress (setup, connecting, etc.)
        ACTIVE       — running normally
        FAILED       — activate() raised, or runtime error detected
        DEACTIVATING — deactivate() in progress (cleanup, releasing resources)
    """
    IDLE = "idle"
    ACTIVATING = "activating"
    ACTIVE = "active"
    FAILED = "failed"
    DEACTIVATING = "deactivating"


@dataclass(frozen=True)
class SkillResult:
    """Structured feedback from skill lifecycle transitions.

    Returned by activate/deactivate to give the protocol (and LLM) richer
    information than just success/failure. Feeds into context_for_llm()
    and health().

    Example:
        SkillResult(
            state=SkillState.ACTIVE,
            message="Streaming to rtsp://localhost:8554/reachy at 720p 20fps",
        )
    """
    state: SkillState
    message: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.state in (SkillState.ACTIVE, SkillState.IDLE)


@dataclass
class SkillConfig:
    """Base config for skills. Subclass for skill-specific settings."""
    pass


class Skill(ABC):
    """Atomic, reusable capability that Maxim can activate."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier (e.g., 'rtsp_streaming')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for LLM context."""
        ...

    @abstractmethod
    def tools(self) -> list[Tool]:
        """Tools this skill provides when active.

        Called during protocol activation to register tools dynamically.
        Return an empty list if the skill has no tools (e.g., a passive
        constraint skill).
        """
        ...

    @abstractmethod
    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        """Start the skill. Called by Protocol.on_activate().

        Args:
            maxim: The live Maxim instance. Skills store this reference
                   for accessing camera frames, motor queues, etc.
            context: Optional shared protocol context dict (blackboard).
                     Skills can read values written by earlier skills and
                     write values for downstream skills to consume.

        Returns:
            SkillResult with the new state and a human-readable message.
            Implementations should set self._state to ACTIVATING before
            starting work and ACTIVE on success (or FAILED on error).
        """
        ...

    @abstractmethod
    def deactivate(self) -> SkillResult:
        """Stop the skill and release resources.

        Returns:
            SkillResult with the new state (IDLE on success).
        """
        ...

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        """Check if preconditions are met before activation.

        Called by Protocol.on_activate() before starting any skills.
        If any skill's preconditions fail, the protocol aborts activation
        before partially starting skills (fail-fast).

        Returns:
            (ok, reason) — True if the skill can activate, False with a
            human-readable explanation if not.

        Override to check for required binaries, connections, hardware
        state, etc. Default returns (True, "").
        """
        return True, ""

    def bio_dependencies(self) -> list[str]:
        """Declare which biological subsystems this skill needs.

        Returns a list of system names: "hippocampus", "atl", "ips",
        "angular_gyrus", "nac", "scn". Protocol activation resolves
        these to live system references and injects them into
        self._bio before calling activate().

        Skills that don't need bio systems return [] (default).
        Skills that do can access them via self._bio["hippocampus"], etc.

        The full bio-skill integration (markdown skill definitions,
        concept-driven sequences, skills-as-memories) is planned in
        ATL concept memory plan phase A7. This interface is the
        foundation that ships now.
        """
        return []

    @property
    def state(self) -> SkillState:
        """Current lifecycle state of the skill.

        Subclasses should maintain self._state. Default returns IDLE.
        """
        return getattr(self, "_state", SkillState.IDLE)

    @property
    def is_active(self) -> bool:
        """Whether the skill is currently running. Convenience for state == ACTIVE."""
        return self.state == SkillState.ACTIVE

    def context_for_llm(self) -> str:
        """Optional context string injected into the LLM prompt.

        Override to provide real-time status (e.g., "Streaming at 20fps
        to rtsp://host:8554/reachy"). Default returns state info.
        """
        if self.state == SkillState.FAILED:
            result = getattr(self, "_last_result", None)
            error = result.error if result else "unknown"
            return f"Skill '{self.name}' FAILED: {error}"
        return ""

    def health(self) -> dict[str, Any]:
        """Optional health/status dict for monitoring.

        Default returns state and last result message.
        """
        info: dict[str, Any] = {"name": self.name, "state": self.state.value}
        result = getattr(self, "_last_result", None)
        if result is not None:
            info["message"] = result.message
            if result.error:
                info["error"] = result.error
            if result.metadata:
                info["metadata"] = result.metadata
        return info
