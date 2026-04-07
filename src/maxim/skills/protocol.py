"""Protocol — a named composition of Skills with constraints.

A Protocol bundles one or more Skills with workspace bounds, voice/CLI
phrases, and LLM context into an operational profile that Maxim can
activate on command.

Protocols are activated via:
  - Voice: "Maxim run shredder segmenter protocol"
  - CLI: typing the same phrase
  - Tool: the LLM calls run_protocol(name="shredder_segmenter")
  - Code: protocol_registry.activate("shredder_segmenter")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.skills.base import Skill, SkillResult

__all__ = ["Protocol", "WorkspaceBounds"]

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceBounds:
    """Workspace constraint override for a protocol.

    Values are in the same units as MovementMixin._get_workspace_limits():
    translations in mm, rotations in degrees. All values are positive
    (symmetric about zero). Set to None to use the default limit for
    that axis.

    Example: constrain gaze to ±30° yaw, ±20° pitch, leave others default:
        WorkspaceBounds(yaw=30.0, pitch=20.0)
    """

    x: float | None = None  # mm, forward/backward
    y: float | None = None  # mm, left/right
    z: float | None = None  # mm, up/down
    roll: float | None = None  # degrees
    pitch: float | None = None  # degrees
    yaw: float | None = None  # degrees


class Protocol(ABC):
    """Named operational profile composing Skills + constraints."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier (e.g., 'shredder_segmenter')."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    @abstractmethod
    def skills(self) -> list[Skill]:
        """Skills this protocol composes. Instantiated fresh each call."""
        ...

    def workspace_bounds(self) -> WorkspaceBounds | None:
        """Optional tighter workspace bounds while protocol is active.

        Returns None to use default bounds. Override to constrain.
        """
        return None

    def phrases(self) -> list[str]:
        """Voice/CLI phrases that activate this protocol.

        The first phrase is canonical. Others are aliases. Matching is
        case-insensitive substring. The protocol system automatically
        registers "stop {name}" phrases for deactivation.

        Override to customize. Default returns common patterns.
        """
        return [
            f"run {self.name.replace('_', ' ')} protocol",
            f"start {self.name.replace('_', ' ')} protocol",
            f"run {self.name.replace('_', ' ')}",
            f"start {self.name.replace('_', ' ')}",
        ]

    def stop_phrases(self) -> list[str]:
        """Voice/CLI phrases that deactivate this protocol."""
        return [
            f"stop {self.name.replace('_', ' ')} protocol",
            f"stop {self.name.replace('_', ' ')}",
            f"end {self.name.replace('_', ' ')} protocol",
        ]

    def context_for_llm(self) -> str:
        """LLM context when this protocol is active.

        Default aggregates skill contexts. Override to add protocol-level
        context (e.g., workspace constraints, operational goals).
        """
        parts = [f"Active protocol: {self.name} — {self.description}"]
        for skill in self._active_skills:
            ctx = skill.context_for_llm()
            if ctx:
                parts.append(f"  Skill [{skill.name}]: {ctx}")
        bounds = self.workspace_bounds()
        if bounds:
            constraints = []
            for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                val = getattr(bounds, axis)
                if val is not None:
                    unit = "mm" if axis in ("x", "y", "z") else "°"
                    constraints.append(f"{axis}=±{val}{unit}")
            if constraints:
                parts.append(f"  Workspace constrained: {', '.join(constraints)}")
        return "\n".join(parts)

    # --- Lifecycle (called by ProtocolRegistry) ---

    def on_activate(self, maxim: Any) -> None:
        """Activate all skills and apply constraints.

        Called by ProtocolRegistry. Override to add protocol-level setup
        (call super() to preserve skill activation).

        Activation sequence:
          1. Resolve bio dependencies for all skills
          2. Check all skill preconditions (fail-fast before starting anything)
          3. Activate skills in order, passing shared context dict
          4. On skill failure, delegate to on_skill_failed() for recovery decision
        """
        self._maxim = maxim
        self._active_skills = []
        # Preserve any pre-set context (e.g., _protocol_name from registry)
        pre_ctx = getattr(self, "_context", {})
        self._context: dict[str, Any] = {k: v for k, v in pre_ctx.items() if k.startswith("_")}

        # Phase 1: Resolve bio system dependencies
        bio_registry = self._resolve_bio_systems(maxim)
        skills = self.skills()
        for skill in skills:
            deps = skill.bio_dependencies()
            if deps:
                skill._bio = {name: bio_registry[name] for name in deps if name in bio_registry}
                missing = [d for d in deps if d not in bio_registry]
                if missing:
                    logger.warning(
                        "Skill '%s' requested bio systems %s but they are not available",
                        skill.name,
                        missing,
                    )

        # Phase 2: Check all preconditions before starting anything
        for skill in skills:
            ok, reason = skill.can_activate(maxim)
            if not ok:
                raise RuntimeError(f"Skill '{skill.name}' precondition failed: {reason}")

        # Phase 3: Activate skills in order with shared context
        for skill in skills:
            result = skill.activate(maxim, context=self._context)
            if result.ok:
                self._active_skills.append(skill)
                self._context["_active_skill_name"] = skill.name
            else:
                action = self.on_skill_failed(skill, result)
                if action == "abort":
                    # Roll back already-activated skills
                    self._rollback_skills()
                    raise RuntimeError(
                        f"Protocol '{self.name}' aborted: skill '{skill.name}' "
                        f"failed — {result.error or result.message}"
                    )
                elif action == "continue":
                    logger.warning(
                        "Skill '%s' failed but protocol continuing: %s",
                        skill.name,
                        result.error or result.message,
                    )

    def on_deactivate(self) -> None:
        """Deactivate all skills and release constraints.

        Called by ProtocolRegistry. Override to add protocol-level teardown
        (call super() to preserve skill deactivation).
        """
        failures = []
        for skill in reversed(self._active_skills):
            try:
                skill.deactivate()
            except Exception as e:
                failures.append(skill.name)
                logger.warning(
                    "Skill '%s' deactivation failed: %s",
                    skill.name,
                    e,
                    exc_info=True,
                )
        if failures:
            logger.error(
                "Deactivation incomplete: %d of %d skills failed to clean up: %s",
                len(failures),
                len(self._active_skills),
                ", ".join(failures),
            )
        self._active_skills = []
        self._context = {}
        self._maxim = None

    def on_skill_failed(self, skill: Skill, result: SkillResult) -> str:
        """Called when a skill fails during activation. Return recovery action.

        Override to implement protocol-specific degradation logic.

        Args:
            skill: The skill that failed.
            result: The SkillResult with error details.

        Returns:
            "abort"    — roll back all activated skills, fail the protocol
            "continue" — skip this skill, activate remaining skills
        """
        return "abort"

    def _rollback_skills(self) -> None:
        """Deactivate already-activated skills during a failed activation."""
        failures = []
        for skill in reversed(self._active_skills):
            try:
                skill.deactivate()
            except Exception as e:
                failures.append(skill.name)
                logger.warning(
                    "Skill '%s' rollback failed: %s",
                    skill.name,
                    e,
                    exc_info=True,
                )
        if failures:
            logger.error(
                "Rollback incomplete! %d skills not cleaned up: %s",
                len(failures),
                ", ".join(failures),
            )
        self._active_skills = []

    def _resolve_bio_systems(self, maxim: Any) -> dict[str, Any]:
        """Discover available biological subsystems on the Maxim instance.

        Resolves via MemoryHub (stored as _memory_hub on Maxim by
        agentic_runtime.py) for hippocampus, atl, angular_gyrus, scn.
        NAc is stored directly on Maxim as _nac.
        IPS is stateless and instantiated on demand.

        Returns a dict mapping system name to live instance.
        """
        systems: dict[str, Any] = {}

        # Resolve via MemoryHub (created in agentic_runtime.py)
        hub = getattr(maxim, "_memory_hub", None)
        if hub is not None:
            # Wrap hub in BioContext facade for skill-friendly access
            from maxim.skills.bio_context import BioContext

            systems["memory_hub"] = BioContext(hub)
            # Individual systems from hub (public attributes)
            hub_attrs = {
                "hippocampus": "hippocampus",
                "atl": "atl",
                "angular_gyrus": "angular_gyrus",
                "scn": "scn",
            }
            for name, attr in hub_attrs.items():
                obj = getattr(hub, attr, None)
                if obj is not None:
                    systems[name] = obj

        # NAc stored directly on Maxim (agentic_runtime.py:130)
        nac = getattr(maxim, "_nac", None)
        if nac is not None:
            systems["nac"] = nac

        # IPS is stateless — instantiate on demand
        try:
            from maxim.math.ips import IPS

            systems["ips"] = IPS()
        except ImportError:
            pass

        return systems

    def __init__(self) -> None:
        """Initialize protocol instance state.

        Subclasses MUST call super().__init__() if they override __init__.
        """
        self._maxim: Any = None
        self._active_skills: list[Skill] = []
        self._context: dict[str, Any] = {}

    @property
    def is_active(self) -> bool:
        return bool(self._active_skills)
