"""Custom exception hierarchy for Maxim.

Provides specific exception types for different error categories,
enabling more precise error handling and better debugging.

Usage:
    from maxim.exceptions import ConnectionError, ToolExecutionError

    try:
        robot.connect()
    except ConnectionError as e:
        logger.error("Robot connection failed: %s", e)
        # Handle connection-specific recovery
"""

from __future__ import annotations

from typing import Any


class MaximError(Exception):
    """Base exception for all Maxim-related errors.

    All custom exceptions in Maxim inherit from this base class,
    allowing catch-all handling when needed:

        try:
            ...
        except MaximError as e:
            logger.error("Maxim operation failed: %s", e)
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        """Initialize with message and optional context.

        Args:
            message: Human-readable error description.
            context: Additional context for debugging (logged, not displayed).
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        return self.message


# ─────────────────────────────────────────────────────────────────────────────
# Connection Errors
# ─────────────────────────────────────────────────────────────────────────────


class ConnectionError(MaximError):
    """Robot connection issues.

    Raised when:
    - Initial connection to Reachy Mini fails
    - Connection is lost during operation
    - Reconnection attempts fail
    """

    pass


class ConnectionTimeoutError(ConnectionError):
    """Connection attempt timed out."""

    pass


class ConnectionLostError(ConnectionError):
    """Connection was lost during operation."""

    pass


class ReconnectionFailedError(ConnectionError):
    """Automatic reconnection failed after multiple attempts."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution Errors
# ─────────────────────────────────────────────────────────────────────────────


class ToolExecutionError(MaximError):
    """Tool execution failed.

    Raised when a tool (look_around, move_head, speak, etc.) fails
    during execution.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        tool_params: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if tool_name:
            ctx["tool_name"] = tool_name
        if tool_params:
            ctx["tool_params"] = tool_params
        super().__init__(message, context=ctx)
        self.tool_name = tool_name
        self.tool_params = tool_params


class ToolNotFoundError(ToolExecutionError):
    """Requested tool does not exist."""

    pass


class ToolValidationError(ToolExecutionError):
    """Tool parameters failed validation."""

    pass


class ToolTimeoutError(ToolExecutionError):
    """Tool execution timed out."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Model Errors
# ─────────────────────────────────────────────────────────────────────────────


class ModelError(MaximError):
    """ML model-related errors.

    Raised when:
    - Model loading fails
    - Inference fails
    - Model returns invalid output
    """

    def __init__(
        self,
        message: str,
        *,
        model_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if model_name:
            ctx["model_name"] = model_name
        super().__init__(message, context=ctx)
        self.model_name = model_name


class ModelLoadError(ModelError):
    """Failed to load ML model."""

    pass


class ModelInferenceError(ModelError):
    """Model inference failed."""

    pass


class ModelConfigError(ModelError):
    """Model configuration is invalid."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Memory Errors
# ─────────────────────────────────────────────────────────────────────────────


class MemoryError(MaximError):
    """Memory system errors.

    Raised when:
    - Memory storage/retrieval fails
    - Memory corruption is detected
    - Memory capacity is exceeded
    """

    pass


class MemoryCorruptionError(MemoryError):
    """Memory data is corrupted or inconsistent."""

    pass


class MemoryCapacityError(MemoryError):
    """Memory capacity limit exceeded."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Planning Errors
# ─────────────────────────────────────────────────────────────────────────────


class PlanningError(MaximError):
    """Planning and decision-making errors."""

    pass


class ConstraintViolation(PlanningError):
    """Safety or policy constraint was violated.

    Raised when a proposed action violates configured constraints,
    such as movement limits or forbidden operations.
    """

    def __init__(
        self,
        message: str,
        *,
        constraint_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if constraint_name:
            ctx["constraint_name"] = constraint_name
        super().__init__(message, context=ctx)
        self.constraint_name = constraint_name


class NoValidPlanError(PlanningError):
    """No valid plan could be generated for the given goal."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Errors
# ─────────────────────────────────────────────────────────────────────────────


class ConfigurationError(MaximError):
    """Configuration-related errors.

    Raised when:
    - Required configuration is missing
    - Configuration values are invalid
    - Configuration file parsing fails
    """

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Hardware Errors
# ─────────────────────────────────────────────────────────────────────────────


class HardwareError(MaximError):
    """Hardware-related errors.

    Raised when:
    - Motor control fails
    - Sensor reading fails
    - Hardware is in an unexpected state
    """

    pass


class MotorError(HardwareError):
    """Motor control error."""

    pass


class SensorError(HardwareError):
    """Sensor reading error."""

    pass


class CameraError(HardwareError):
    """Camera/vision hardware error."""

    pass


class AudioError(HardwareError):
    """Audio hardware error."""

    pass


# ─────────────────────────────────────────────────────────────────────────────
# Runtime Errors
# ─────────────────────────────────────────────────────────────────────────────


class RuntimeError(MaximError):
    """Runtime system errors.

    Raised for errors in the agent loop, thread management, etc.
    """

    pass


class ShutdownRequestedError(RuntimeError):
    """Shutdown was requested during operation.

    This is not an error per se, but a signal to stop processing.
    """

    pass


class AgentLoopError(RuntimeError):
    """Error in the agent loop."""

    pass


__all__ = [
    # Base
    "MaximError",
    # Connection
    "ConnectionError",
    "ConnectionTimeoutError",
    "ConnectionLostError",
    "ReconnectionFailedError",
    # Tools
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolTimeoutError",
    # Models
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "ModelConfigError",
    # Memory
    "MemoryError",
    "MemoryCorruptionError",
    "MemoryCapacityError",
    # Planning
    "PlanningError",
    "ConstraintViolation",
    "NoValidPlanError",
    # Configuration
    "ConfigurationError",
    "MissingConfigError",
    "InvalidConfigError",
    # Hardware
    "HardwareError",
    "MotorError",
    "SensorError",
    "CameraError",
    "AudioError",
    # Runtime
    "RuntimeError",
    "ShutdownRequestedError",
    "AgentLoopError",
]
