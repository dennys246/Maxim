"""Custom exception hierarchy for Maxim.

Provides specific exception types for different error categories,
enabling more precise error handling and better debugging.

Usage::

    from maxim.exceptions import MaximConnectionError, ToolExecutionError

    try:
        robot.connect()
    except MaximConnectionError as e:
        logger.error("Robot connection failed: %s", e)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal


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


class KeyFileUnusable(MaximError, RuntimeError):
    """A key file exists but cannot be used, so it is refused rather than read as absent or overwritten.

    Key-neutral base of every key-file refusal (#1402): ``tunnel.keys.ensure_key`` (the mesh key) and
    ``ensure_console_token`` raise it plainly; ``tunnel.keys.LeaderKeyError`` (a leader's mesh key, also a
    :class:`LeaderRefusal`) and its subclasses build on it. ``path`` is the resolved file, ``reason`` names what is
    wrong with it and ``fix_hint`` says how to repair it. Generating a new key over the file would silently rotate it
    and 401 every peer or device holding the old one.
    """

    def __init__(self, *, path: Path, reason: str, fix_hint: str, message: str | None = None) -> None:
        self.path = path
        self.reason = reason
        self.fix_hint = fix_hint
        super().__init__(message or f"key file unusable ({reason}): {path}")


class StopStillPending(MaximError, RuntimeError):
    """A console refused to start while an earlier process stop request is still set (#1438 PR 1b).

    The request is cleared only by the owner that made it, after a stop that finished; a stop that overran its budget
    (a campaign or a session end still running) leaves it set. Starting anyway would build loops and LLM workers that
    are born stopped. ``reason`` is the pending request's reason.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            f"refusing to start: an earlier stop ({reason!r}) has not finished; restart the process to recover"
        )


class LeaderRefusal(MaximError, RuntimeError):
    """A process refused to serve because doing so would be unauthenticated or over-exposed.

    Base of ``tunnel.keys.LeaderKeyError`` (a leader with no usable key),
    ``runtime.leader_proxy.ProxyAuthError`` (a proxy port held by something
    that is not an authenticated LeaderProxy) and
    ``runtime.local_server_spawner.LLMServerExposureError`` (a llama-cpp-server
    answering without its key, or reachable on the LAN). A broad
    ``except Exception`` that would degrade to "run without the LLM" must
    re-raise this first: a security refusal is not an availability fallback.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Connection Errors
# ─────────────────────────────────────────────────────────────────────────────


class MaximConnectionError(MaximError):
    """Robot connection issues.

    Raised when:
    - Initial connection to Reachy Mini fails
    - Connection is lost during operation
    - Reconnection attempts fail
    """

    pass


class ConnectionTimeoutError(MaximConnectionError):
    """Connection attempt timed out."""

    pass


class ConnectionLostError(MaximConnectionError):
    """Connection was lost during operation."""

    pass


class ReconnectionFailedError(MaximConnectionError):
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


class ToolEffectRefused(MaximError):
    """A registry refused to admit a tool because of its declared effect (#1315 item 2): the tool's effect exceeds
    the registry's ceiling, its class's ``effect`` was reassigned after creation, or it would replace a same-name
    tool under a ceiling below ``ALL``. ``effect`` and ``ceiling`` are sorted flag-name lists (the serialised form).
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str,
        effect: list[str],
        ceiling: list[str],
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message, context={**(context or {}), "tool_name": tool_name, "effect": effect, "ceiling": ceiling}
        )
        self.tool_name = tool_name
        self.effect = effect
        self.ceiling = ceiling


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


class MaximMemoryError(MaximError):
    """Memory system errors.

    Raised when:
    - Memory storage/retrieval fails
    - Memory corruption is detected
    - Memory capacity is exceeded
    """

    pass


class MemoryCorruptionError(MaximMemoryError):
    """Memory data is corrupted or inconsistent."""

    pass


class StoreOverwriteRefused(MaximMemoryError, FileExistsError):
    """A memory store refused to save over an existing file it never read (#939), or ``create.hippocampus`` /
    ``create.atl`` refused to start a new store over an existing file (#1071).

    Saving would replace memories this instance never loaded. Open the file with ``maxim.load.*`` (or
    the store's ``load()``) to keep them, save elsewhere, or declare the overwrite with
    ``save(overwrite=True)`` / ``allow_overwrite()``. Also a ``FileExistsError``, so ordinary
    file-error handling still sees it.
    """

    def __init__(self, message: str = "", *, path: str = "", store: str = "") -> None:
        # Defaults keep it picklable/copyable, like the other keyword-only Maxim errors.
        super().__init__(message, context={"path": path, "store": store})
        self.path = path
        self.store = store
        self.filename = path  # what a plain FileExistsError handler reads


class RunDirNotFound(MaximMemoryError, ValueError):
    """``utils/paths.py::resolve_run_dir`` found no directory; the message names every place it searched.

    Also a ``ValueError``, which is what it was before it joined the hierarchy (1.3.1).
    """


class RunDirAmbiguous(MaximMemoryError, ValueError):
    """A run ID or ID prefix names more than one directory; the message lists them (#932).

    Raised by ``resolve_run_dir`` and ``maxim.load.session``: a guess could read, or write to, the wrong
    run. Pass a path, or the full ID. Also a ``ValueError``.
    """


class SessionRefusal(MaximMemoryError, RuntimeError):
    """A memory-session claim was refused (#1438): the one class a loop's caller catches (PR 1/2 map it to a 409 or
    exit 2). Its subclasses say why."""


class SessionHeld(SessionRefusal):
    """A memory session is held by a live loop, so another owner cannot claim it (#1438 PR 0).

    Raised by ``integration.session_ownership.SessionOwnership.claim`` (through ``MemoryHub.on_session_start``) when
    a running agent loop holds the hub's session. It propagates as a typed abort of the loop that tried to claim it.
    """


class SessionStillEnding(SessionRefusal):
    """A memory session is still ending and did not finish within the claimer's deadline (#1438 PR 0)."""


class MemoryCapacityError(MaximMemoryError):
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


class ConfigDirUnreadable(ConfigurationError):
    """The operator config directory (``~/.config/maxim``) cannot be read, so no command can know its config.

    Raised by :func:`maxim.utils.paths.config_file_present`; the message names the directory and the
    ``chmod 700`` fix. Distinct from a malformed file, which role detection warns about and falls past (#1403).
    """


class UserProfilesInvalid(ConfigurationError):
    """The user's ``profiles.yml`` cannot be loaded (malformed YAML, a schema error, an unreadable file), so no run
    can know which models it may use (#1315 step 0a).

    Raised by ``models/language/config.py::_ensure_user_profiles`` and refused up front by
    ``require_valid_user_profiles`` at every run entry point. The message names the file; ``fix_hint`` says how to
    recover. ``maxim doctor`` still runs and reports the specific error.
    """

    fix_hint = "Fix the file, or move it aside, then run `maxim doctor` to check it."


class ComponentNotFoundError(ConfigurationError):
    """Requested SEM component template does not exist.

    Raised by ``maxim.create.entity()`` when the template reference
    (e.g., ``"npcs/guard"``) is not found in any component directory.
    The error message includes the list of available components.
    """

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    pass


class InvalidAgentIdError(InvalidConfigError, ValueError):
    """An agent id that the memory keys cannot hold.

    ``reward_bias`` persists as ``"<agent_id>:<node_id>"`` and the composite keys join on ``\\x1f``; every
    reader splits on the first separator, so an id containing either (or an empty id) would be misread
    as another agent. NAc refuses a key separator on its credit writes, so the id is refused where an agent
    is named (``AgentConfig``, ``build_bio_stack``), before any home is built
    (``decisions/nac.py::require_agent_id``). A ``ValueError`` too, for callers that caught the plain one.
    """

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


MotionRefusalKind = Literal[
    "non_finite", "duration", "method", "range", "invalid", "rate", "reference", "path", "limit"
]


class MotionRefused(HardwareError, ValueError):
    """A motion command refused before it reached a motor (#1496).

    Raised for a non-finite value, a duration outside [0.1 s, 10 s], an
    interpolation method that overshoots, a config value past a hard cap, or
    structurally invalid input. Also a ``ValueError``, so a refusal reads as bad
    input; that makes every ``except ValueError`` that substitutes a default
    motion a hazard: validation never sits inside such a handler
    (docs/plans/motion_finite_1496.md section 3).

    Attributes:
        kind: ``"non_finite"`` (NaN or Inf), ``"duration"`` (outside the range),
            ``"method"`` (an unknown or overshooting interpolation), ``"range"``
            (a config value past a hard cap, e.g. an orient delta over 90 deg),
            ``"invalid"`` (not a number, the wrong shape, an unknown pose), or one
            of #1499's: ``"rate"`` (an explicit duration faster than the rate cap),
            ``"reference"`` (no fresh, trusted readback to move from), ``"path"``
            (the move would sweep the neck mid-path) and ``"limit"`` (a turn that
            cannot fit within the body limit; no producer yet, PR 2's turn_around).
            No kind feeds the gaze or workspace-bounds learners (#1499 O4); a tool
            that reports one still reaches the tool-outcome path like any failure.
        reason: Human-readable reason, returned to the model by the tools.
    """

    def __init__(self, kind: MotionRefusalKind = "invalid", reason: str = "") -> None:
        super().__init__(reason, context={"kind": kind})
        self.kind: MotionRefusalKind = kind
        self.reason = reason

    def __reduce__(self) -> tuple[Any, ...]:
        # Positional (kind, reason), so pickle and copy rebuild it faithfully.
        return (type(self), (self.kind, self.reason))


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


class MaximRuntimeError(MaximError):
    """Runtime system errors.

    Raised for errors in the agent loop, thread management, etc.
    """

    pass


class ShutdownRequestedError(MaximRuntimeError):
    """Shutdown was requested during operation.

    This is not an error per se, but a signal to stop processing.
    """

    pass


class AgentLoopError(MaximRuntimeError):
    """Error in the agent loop."""

    pass


__all__ = [
    # Base
    "MaximError",
    "KeyFileUnusable",
    "StopStillPending",
    "LeaderRefusal",
    # Connection
    "MaximConnectionError",
    "ConnectionTimeoutError",
    "ConnectionLostError",
    "ReconnectionFailedError",
    # Tools
    "ToolExecutionError",
    "ToolNotFoundError",
    "ToolValidationError",
    "ToolTimeoutError",
    "ToolEffectRefused",
    # Models
    "ModelError",
    "ModelLoadError",
    "ModelInferenceError",
    "ModelConfigError",
    # Memory
    "MaximMemoryError",
    "MemoryCorruptionError",
    "StoreOverwriteRefused",
    "RunDirNotFound",
    "RunDirAmbiguous",
    "MemoryCapacityError",
    "SessionRefusal",
    "SessionHeld",
    "SessionStillEnding",
    # Planning
    "PlanningError",
    "ConstraintViolation",
    "NoValidPlanError",
    # Configuration
    "ConfigurationError",
    "ConfigDirUnreadable",
    "ComponentNotFoundError",
    "MissingConfigError",
    "InvalidConfigError",
    "InvalidAgentIdError",
    # Hardware
    "HardwareError",
    "MotorError",
    "MotionRefused",
    "SensorError",
    "CameraError",
    "AudioError",
    # Runtime
    "MaximRuntimeError",
    "ShutdownRequestedError",
    "AgentLoopError",
]
