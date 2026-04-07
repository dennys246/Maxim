"""State manager for Maxim's operational state.

Manages transitions between processing states and operational modes (autonomy levels).
Provides callbacks for notifying agents of state changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from maxim.modes.definitions import (
    MaximState,
    OperationalMode,
    ProcessingState,
)

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agent notification of state changes."""

    def set_processing_state(self, state: str) -> None:
        """Called when processing state changes."""
        ...

    def set_operational_mode(self, mode: str) -> None:
        """Called when operational mode changes."""
        ...


@dataclass
class StateManagerConfig:
    """Configuration for state manager."""

    initial_mode: str = "passive"
    initial_processing_state: str = "awake"
    auto_wake_on_mode_change: bool = True


StateChangeCallback = Callable[[str, str, str], None]  # (state_type, old, new)


class StateManager:
    """Manages Maxim's operational state with agent notification.

    Tracks two orthogonal state dimensions:
    - Operational mode: passive, active, singularity (autonomy level)
    - Processing state: awake, sleep (resource usage)

    Example:
        manager = StateManager()
        manager.set_agent(my_agent)
        manager.set_operational_mode("active")
        manager.set_processing_state("sleep")
    """

    def __init__(
        self,
        config: StateManagerConfig | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        """Initialize state manager.

        Args:
            config: State manager configuration.
            log: Logger instance.
        """
        self.config = config or StateManagerConfig()
        self._log = log or logger
        self._agent: AgentProtocol | None = None
        self._callbacks: list[StateChangeCallback] = []

        # Initialize state
        self._state = MaximState(
            mode=OperationalMode(self.config.initial_mode),
            processing_state=ProcessingState(self.config.initial_processing_state),
        )

        # Shutdown flag
        self._shutdown_requested = False

    @property
    def state(self) -> MaximState:
        """Get current state."""
        return self._state

    @property
    def operational_mode(self) -> str:
        """Current operational mode as string."""
        return self._state.mode.value

    @property
    def processing_state(self) -> str:
        """Current processing state as string."""
        return self._state.processing_state.value

    @property
    def is_sleeping(self) -> bool:
        """Check if in sleep processing state."""
        return self._state.is_sleeping()

    @property
    def is_active(self) -> bool:
        """Check if in active operational mode."""
        return self._state.is_active()

    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def set_agent(self, agent: AgentProtocol | None) -> None:
        """Set the agent to notify of state changes.

        Args:
            agent: Agent implementing state change methods, or None.
        """
        self._agent = agent

    def add_callback(self, callback: StateChangeCallback) -> None:
        """Add a callback for state changes.

        Callback receives (state_type, old_value, new_value).
        state_type is one of: "processing_state", "strategy", "operational_mode"

        Args:
            callback: Function to call on state changes.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: StateChangeCallback) -> None:
        """Remove a state change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def set_processing_state(self, state: str) -> bool:
        """Set processing state (awake/sleep).

        Args:
            state: New processing state.

        Returns:
            True if state changed.
        """
        old = self._state.processing_state.value
        if state == old:
            return False

        try:
            new_state = ProcessingState(state)
        except ValueError:
            self._log.warning("Invalid processing state: %s", state)
            return False

        self._log.info("Processing state: %s -> %s", old, state)
        self._state.processing_state = new_state

        # Notify agent
        if self._agent is not None and hasattr(self._agent, "set_processing_state"):
            try:
                self._agent.set_processing_state(state)
            except Exception as e:
                self._log.warning("Failed to notify agent of processing state: %s", e)

        # Notify callbacks
        self._notify_callbacks("processing_state", old, state)

        return True

    def set_operational_mode(self, mode: str) -> bool:
        """Set operational mode (passive/active/singularity).

        If auto_wake_on_mode_change is True and currently sleeping,
        automatically wakes up.

        Args:
            mode: New operational mode.

        Returns:
            True if mode changed.
        """
        old = self._state.mode.value
        if mode == old:
            return False

        try:
            new_mode = OperationalMode(mode)
        except ValueError:
            self._log.warning("Invalid operational mode: %s", mode)
            return False

        self._log.info("Operational mode: %s -> %s", old, mode)
        self._state.mode = new_mode

        # Auto-wake if sleeping
        if self.config.auto_wake_on_mode_change and self.is_sleeping:
            self.set_processing_state("awake")

        # Notify agent
        if self._agent is not None and hasattr(self._agent, "set_operational_mode"):
            try:
                self._agent.set_operational_mode(mode)
            except Exception as e:
                self._log.warning("Failed to notify agent of mode: %s", e)

        # Notify callbacks
        self._notify_callbacks("operational_mode", old, mode)

        return True

    def request_shutdown(self) -> None:
        """Request system shutdown."""
        self._log.info("Shutdown requested")
        self._shutdown_requested = True

    def request_sleep(self) -> None:
        """Request sleep processing state."""
        self.set_processing_state("sleep")

    def request_wake(self) -> None:
        """Request awake processing state."""
        self.set_processing_state("awake")

    def request_mode_passive(self) -> None:
        self.set_operational_mode("passive")

    def request_mode_active(self) -> None:
        self.set_operational_mode("active")

    def request_mode_singularity(self) -> None:
        self.set_operational_mode("singularity")

    def _notify_callbacks(self, state_type: str, old: str, new: str) -> None:
        """Notify all callbacks of a state change."""
        for callback in self._callbacks:
            try:
                callback(state_type, old, new)
            except Exception as e:
                self._log.warning("State callback failed: %s", e)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "operational_mode": self.operational_mode,
            "processing_state": self.processing_state,
            "shutdown_requested": self._shutdown_requested,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Load state from dictionary."""
        if "operational_mode" in data:
            try:
                self._state.mode = OperationalMode(data["operational_mode"])
            except ValueError:
                pass

        if "processing_state" in data:
            try:
                self._state.processing_state = ProcessingState(data["processing_state"])
            except ValueError:
                pass

        # Silently ignore "strategy" key from old persisted state

        if "shutdown_requested" in data:
            self._shutdown_requested = bool(data["shutdown_requested"])


__all__ = [
    "StateManager",
    "StateManagerConfig",
    "AgentProtocol",
    "StateChangeCallback",
]
