"""Output mixin for agents to use the centralized output system.

Provides convenience methods for agents to write data to:
- Instance data directory
- Sandbox
- Shared outputs (cross-instance)
- Abstraction buffer

Usage:
    class MyAgent(Agent, AgentOutputMixin):
        def __init__(self, ...):
            super().__init__(...)
            self._init_output()  # Call in __init__

        def some_method(self):
            self._write_memory([...])
            self._write_to_sandbox("analysis.json", {...})
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.utils.agent_output import AgentOutputManager


logger = logging.getLogger(__name__)


class AgentOutputMixin:
    """Mixin that provides output methods for agents.

    Agents using this mixin get:
    - _write_memory(): Persist memories
    - _write_context(): Persist context snapshots
    - _write_artifact(): Write artifacts to sandbox
    - _write_result(): Write results
    - _write_to_sandbox(): Write arbitrary data to sandbox
    - _write_to_shared(): Write to cross-instance shared outputs
    - _log_to_stream(): Continuous logging to stream file
    """

    # Set by _init_output()
    _output_manager: AgentOutputManager | None = None
    _output_agent_name: str = "unknown"

    def _init_output(
        self,
        agent_name: str | None = None,
        output_manager: AgentOutputManager | None = None,
    ) -> None:
        """Initialize output capabilities.

        Call this in the agent's __init__ method.

        Args:
            agent_name: Name to use for outputs (defaults to class attribute)
            output_manager: Output manager (uses global if None)
        """
        # Try to get agent name from various sources
        if agent_name:
            self._output_agent_name = agent_name
        elif hasattr(self, "agent_name"):
            self._output_agent_name = getattr(self, "agent_name")
        elif hasattr(self, "name") and getattr(self, "name"):
            self._output_agent_name = getattr(self, "name")
        else:
            self._output_agent_name = type(self).__name__.lower()

        # Get output manager
        if output_manager:
            self._output_manager = output_manager
        else:
            from maxim.utils.agent_output import get_output_manager
            self._output_manager = get_output_manager()

    def _write_memory(
        self,
        memories: list[dict[str, Any]],
        share: bool = False,
    ) -> str | None:
        """Write memories to storage.

        Args:
            memories: List of memory dicts to persist
            share: Whether to also share with other instances

        Returns:
            Path to written file, or None if no output manager
        """
        if not self._output_manager:
            return None

        return self._output_manager.write_memory(
            agent_name=self._output_agent_name,
            memories=memories,
            share=share,
        )

    def _write_context(
        self,
        context: dict[str, Any],
        share: bool = True,
    ) -> str | None:
        """Write context snapshot.

        Args:
            context: Context dict to persist
            share: Whether to share with other instances (default True)

        Returns:
            Path to written file
        """
        if not self._output_manager:
            return None

        return self._output_manager.write_context(
            agent_name=self._output_agent_name,
            context=context,
            share=share,
        )

    def _write_artifact(
        self,
        name: str,
        content: Any,
        to_sandbox: bool = True,
    ) -> str | None:
        """Write an artifact (analysis result, generated content, etc.).

        Args:
            name: Artifact name (used as filename)
            content: Artifact content
            to_sandbox: Write to sandbox (default) or instance data

        Returns:
            Path to written file
        """
        if not self._output_manager:
            return None

        return self._output_manager.write_artifact(
            agent_name=self._output_agent_name,
            artifact_name=name,
            content=content,
            to_sandbox=to_sandbox,
        )

    def _write_result(
        self,
        result: dict[str, Any],
        share: bool = False,
    ) -> str | None:
        """Write a result (tool execution, goal completion, etc.).

        Args:
            result: Result dict
            share: Whether to share with other instances

        Returns:
            Path to written file
        """
        if not self._output_manager:
            return None

        return self._output_manager.write_result(
            agent_name=self._output_agent_name,
            result=result,
            share=share,
        )

    def _write_to_sandbox(
        self,
        filename: str,
        content: Any,
        subdir: str | None = None,
    ) -> str | None:
        """Write arbitrary content to sandbox.

        Args:
            filename: Name of file to create
            content: Content to write (will be JSON serialized if not string)
            subdir: Optional subdirectory within sandbox/outputs/{agent}/

        Returns:
            Path to written file
        """
        if not self._output_manager:
            return None

        from maxim.utils.agent_output import AgentOutput, OutputDestination

        # Build path
        if subdir:
            full_filename = os.path.join(subdir, filename)
        else:
            full_filename = filename

        output = AgentOutput(
            agent_name=self._output_agent_name,
            output_type="artifact",
            content=content,
            destination=OutputDestination.SANDBOX,
            filename=full_filename,
        )
        return self._output_manager.write(output)

    def _write_to_shared(
        self,
        filename: str,
        content: Any,
    ) -> str | None:
        """Write content to shared outputs (visible to other instances).

        Args:
            filename: Name of file to create
            content: Content to write

        Returns:
            Path to written file
        """
        if not self._output_manager:
            return None

        from maxim.utils.agent_output import AgentOutput, OutputDestination

        output = AgentOutput(
            agent_name=self._output_agent_name,
            output_type="shared",
            content=content,
            destination=OutputDestination.SHARED_OUTPUT,
            filename=filename,
        )
        return self._output_manager.write(output)

    def _log_to_stream(
        self,
        stream_type: str,
        record: dict[str, Any],
    ) -> None:
        """Write a record to a continuous log stream.

        Useful for high-frequency logging like perception events.

        Args:
            stream_type: Type of stream (e.g., "percepts", "detections")
            record: Record to append (timestamp added automatically)
        """
        if not self._output_manager:
            return

        self._output_manager.write_to_stream(
            agent_name=self._output_agent_name,
            stream_type=stream_type,
            record=record,
        )

    def _list_sandbox_outputs(self) -> list[dict[str, Any]]:
        """List this agent's outputs in sandbox."""
        if not self._output_manager:
            return []

        return self._output_manager.list_outputs(
            agent_name=self._output_agent_name,
            from_sandbox=True,
        )

    def _read_sandbox_output(self, filename: str) -> dict[str, Any] | None:
        """Read one of this agent's sandbox outputs."""
        if not self._output_manager:
            return None

        # Find the file
        outputs = self._list_sandbox_outputs()
        for out in outputs:
            if out["filename"] == filename:
                return self._output_manager.read_output(out["path"])

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Decorator for automatic output setup
# ─────────────────────────────────────────────────────────────────────────────


def with_output(agent_name: str | None = None):
    """Class decorator to automatically initialize output mixin.

    Usage:
        @with_output("my_agent")
        class MyAgent(Agent, AgentOutputMixin):
            pass
    """
    def decorator(cls):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._init_output(agent_name=agent_name)

        cls.__init__ = new_init
        return cls

    return decorator
