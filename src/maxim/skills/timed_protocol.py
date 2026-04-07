"""TimedProtocolSkill — auto-deactivate a protocol after a duration.

A passive skill (no tools) that starts a background timer on activation.
When the timer expires, it calls protocol_registry.deactivate() to cleanly
shut down the enclosing protocol.

Useful for ski recording sessions, security patrols, or any protocol
that should run for a bounded duration without manual intervention.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from maxim.skills.base import Skill, SkillConfig, SkillResult, SkillState

if TYPE_CHECKING:
    from maxim.tools.base import Tool

__all__ = ["TimedProtocolSkill", "TimedProtocolConfig"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimedProtocolConfig(SkillConfig):
    """Configuration for the timed protocol skill."""

    duration_minutes: float = 60.0  # Default: 1 hour


class TimedProtocolSkill(Skill):
    """Auto-deactivate the enclosing protocol after a set duration."""

    def __init__(self, config: TimedProtocolConfig | None = None):
        self._config = config or TimedProtocolConfig()
        self._state = SkillState.IDLE
        self._last_result: SkillResult | None = None
        self._timer: threading.Timer | None = None
        self._maxim: Any = None
        self._protocol_name: str | None = None

    @property
    def name(self) -> str:
        return "timed_protocol"

    @property
    def description(self) -> str:
        return f"Auto-stop protocol after {self._config.duration_minutes} minutes"

    def tools(self) -> list[Tool]:
        return []  # Passive skill — no tools

    def can_activate(self, maxim: Any) -> tuple[bool, str]:
        if self._config.duration_minutes <= 0:
            return False, "duration_minutes must be positive"
        return True, ""

    def activate(self, maxim: Any, context: dict[str, Any] | None = None) -> SkillResult:
        self._state = SkillState.ACTIVATING
        self._maxim = maxim

        # Read protocol name from shared context (set by ProtocolRegistry)
        self._protocol_name = (context or {}).get("_protocol_name")

        duration_s = self._config.duration_minutes * 60
        self._timer = threading.Timer(duration_s, self._on_timer_expired)
        self._timer.daemon = True
        self._timer.start()

        self._state = SkillState.ACTIVE
        self._last_result = SkillResult(
            state=SkillState.ACTIVE,
            message=f"Will auto-stop in {self._config.duration_minutes} minutes",
            metadata={"duration_minutes": self._config.duration_minutes},
        )
        return self._last_result

    def deactivate(self) -> SkillResult:
        self._state = SkillState.DEACTIVATING
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._maxim = None
        self._protocol_name = None
        self._state = SkillState.IDLE
        self._last_result = SkillResult(
            state=SkillState.IDLE,
            message="Timer cancelled",
        )
        return self._last_result

    def _on_timer_expired(self) -> None:
        """Called by the background timer when duration elapses."""
        log = getattr(self._maxim, "log", logger) if self._maxim else logger
        protocol_name = self._protocol_name

        if protocol_name is None:
            log.warning("TimedProtocolSkill timer expired but no protocol name set")
            return

        registry = getattr(self._maxim, "_protocol_registry", None)
        if registry is None:
            log.warning("TimedProtocolSkill timer expired but no protocol registry")
            return

        log.info("TimedProtocolSkill: duration expired, deactivating '%s'", protocol_name)
        try:
            registry.deactivate(protocol_name)
        except Exception as e:
            log.warning(
                "TimedProtocolSkill: failed to deactivate '%s': %s",
                protocol_name,
                e,
                exc_info=True,
            )

    def context_for_llm(self) -> str:
        if self.state == SkillState.ACTIVE:
            return f"Protocol will auto-stop after {self._config.duration_minutes} minutes."
        return super().context_for_llm()

    def health(self) -> dict[str, Any]:
        base = super().health()
        if self._timer is not None:
            base["duration_minutes"] = self._config.duration_minutes
            base["timer_alive"] = self._timer.is_alive()
        return base
