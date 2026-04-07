"""ProtocolRegistry — manages protocol lifecycle and integration.

Responsibilities:
  1. Register protocols (typically at startup in bootstrap.py)
  2. Activate/deactivate protocols on command
  3. Manage tool registration/deregistration
  4. Apply/restore workspace bounds
  5. Register voice/CLI phrases (permanent, at startup)
  6. Provide aggregate LLM context for all active protocols
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.skills.protocol import Protocol, WorkspaceBounds
    from maxim.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

__all__ = ["ProtocolRegistry"]


class ProtocolRegistry:
    """Central registry for protocol lifecycle management."""

    def __init__(
        self,
        maxim: Any,
        tool_registry: ToolRegistry,
    ) -> None:
        self._maxim = maxim
        self._tool_registry = tool_registry
        self._protocols: dict[str, Protocol] = {}
        self._active: dict[str, Protocol] = {}
        # Track tools registered by each protocol for clean removal
        self._protocol_tools: dict[str, list[str]] = {}
        # Track workspace bounds per protocol for multi-protocol composition
        self._saved_limits: dict[str, dict[str, float]] = {}

    def register(self, protocol: Protocol) -> None:
        """Register a protocol (does not activate it)."""
        self._protocols[protocol.name] = protocol
        logger.info("Registered protocol: %s", protocol.name)

    def activate(self, name: str) -> str:
        """Activate a registered protocol.

        Returns:
            Status message (success or error).
        """
        if name in self._active:
            return f"Protocol '{name}' is already active."

        protocol = self._protocols.get(name)
        if protocol is None:
            available = ", ".join(self._protocols.keys()) or "(none)"
            return f"Unknown protocol '{name}'. Available: {available}"

        log = getattr(self._maxim, "log", logger)

        # 1. Activate the protocol (starts skills)
        # Inject protocol name into shared context so skills can discover it
        protocol._context["_protocol_name"] = name
        try:
            protocol.on_activate(self._maxim)
        except Exception as e:
            log.error("Protocol '%s' activation failed: %s", name, e, exc_info=True)
            return f"Protocol '{name}' failed to activate: {e}"

        # 2. Register protocol tools
        tool_names = []
        for skill in protocol._active_skills:
            for tool in skill.tools():
                self._tool_registry.register(tool)
                tool_names.append(tool.name)
        self._protocol_tools[name] = tool_names

        # 3. Apply workspace bounds
        bounds = protocol.workspace_bounds()
        if bounds is not None:
            self._apply_workspace_bounds(name, bounds)

        # NOTE: Phrases are registered once at startup (Section 10.2) and
        # left permanently. They are NOT re-registered here or removed on
        # deactivate — otherwise deactivation would remove the activation
        # phrases needed for voice re-activation.

        self._active[name] = protocol
        log.info("Protocol '%s' activated with skills: %s", name, [s.name for s in protocol._active_skills])
        return f"Protocol '{name}' activated."

    def deactivate(self, name: str) -> str:
        """Deactivate a running protocol.

        Returns:
            Status message.
        """
        protocol = self._active.get(name)
        if protocol is None:
            return f"Protocol '{name}' is not active."

        log = getattr(self._maxim, "log", logger)

        # 1. Deactivate protocol (stops skills)
        protocol.on_deactivate()

        # 2. Deregister protocol tools
        for tool_name in self._protocol_tools.pop(name, []):
            self._tool_registry.deregister(tool_name)

        # 3. Restore workspace bounds
        self._restore_workspace_bounds(name)

        del self._active[name]
        log.info("Protocol '%s' deactivated.", name)
        return f"Protocol '{name}' deactivated."

    def get_active(self) -> list[Protocol]:
        return list(self._active.values())

    def get_available(self) -> list[str]:
        return list(self._protocols.keys())

    def all_skills(self) -> list:
        """Return flat list of all skills across all registered protocols."""
        skills = []
        for protocol in self._protocols.values():
            try:
                skills.extend(protocol.skills())
            except Exception:
                pass
        return skills

    def get_context_for_llm(self) -> str:
        """Aggregate LLM context from all active protocols."""
        if not self._active:
            return ""
        parts = []
        for protocol in self._active.values():
            parts.append(protocol.context_for_llm())
        return "\n".join(parts)

    # --- Internal helpers ---

    def _apply_workspace_bounds(self, name: str, bounds: WorkspaceBounds) -> None:
        """Override workspace limits via _workspace_limit_override attribute.

        Composes with existing overrides from other active protocols by
        taking the tightest (min) constraint across all active protocols.
        """
        override = {}
        for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
            val = getattr(bounds, axis)
            if val is not None:
                override[axis] = val
        self._saved_limits[name] = override
        self._recompute_workspace_override()

    def _restore_workspace_bounds(self, name: str) -> None:
        """Remove this protocol's bounds and recompute the composite."""
        self._saved_limits.pop(name, None)
        self._recompute_workspace_override()

    def _recompute_workspace_override(self) -> None:
        """Recompute the composite workspace override from all active protocols.

        Takes the tightest (min) constraint per axis across all active
        protocol bounds. If no protocols have bounds, clears the override.
        """
        if not self._saved_limits:
            self._maxim._workspace_limit_override = None
            return
        composite: dict[str, float] = {}
        for bounds_dict in self._saved_limits.values():
            for axis, val in bounds_dict.items():
                if axis in composite:
                    composite[axis] = min(composite[axis], val)
                else:
                    composite[axis] = val
        self._maxim._workspace_limit_override = composite

    def _register_phrases(self, name: str, protocol: Protocol) -> None:
        """Add voice/CLI phrase responses (thread-safe via R5 helpers)."""
        if not hasattr(self._maxim, "_register_phrase_response"):
            return

        for phrase in protocol.phrases():
            self._maxim._register_phrase_response(
                phrase,
                {
                    "call": "_protocol_activate",
                    "args": [name],
                    "requires_agentic": False,
                    "cooldown_s": 2.0,
                },
            )

        for phrase in protocol.stop_phrases():
            self._maxim._register_phrase_response(
                phrase,
                {
                    "call": "_protocol_deactivate",
                    "args": [name],
                    "requires_agentic": False,
                    "cooldown_s": 2.0,
                },
            )
