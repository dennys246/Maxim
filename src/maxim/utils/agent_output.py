"""Agent output system with sandbox-aware persistence.

Provides a unified way for agents to output data that:
1. Flows to the abstraction buffer (for LLM context)
2. Persists to instance-specific data directories
3. Optionally shares to cross-instance outputs

This replaces scattered persistence logic with a centralized system.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from maxim.utils.structured_logging import (
    AbstractionBuffer,
    LogRecord,
    get_abstraction_buffer,
)

if TYPE_CHECKING:
    from maxim.utils.filesystem_policy import FilesystemPolicy

logger = logging.getLogger(__name__)


class OutputDestination(Enum):
    """Where agent output should be written."""

    ABSTRACTION_ONLY = "abstraction"  # Only to in-memory buffer
    INSTANCE_DATA = "instance"  # Instance's data directory
    SANDBOX = "sandbox"  # Sandbox directory
    SHARED_OUTPUT = "shared"  # Shared outputs (visible to other instances)
    ALL = "all"  # All destinations


@dataclass
class AgentOutput:
    """A piece of output from an agent."""

    agent_name: str
    output_type: str  # "memory", "log", "result", "artifact", etc.
    content: Any
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Routing
    destination: OutputDestination = OutputDestination.INSTANCE_DATA
    filename: str | None = None  # Optional specific filename

    def to_log_record(self) -> LogRecord:
        """Convert to abstraction stream LogRecord."""
        data = self.metadata.copy()
        if isinstance(self.content, dict):
            # Include summary fields
            for key in ("summary", "count", "status", "success"):
                if key in self.content:
                    data[key] = self.content[key]
        elif isinstance(self.content, str) and len(self.content) < 200:
            data["content"] = self.content

        return LogRecord(
            timestamp=self.timestamp,
            level="INFO",
            source=self.agent_name,
            event=self.output_type,
            data=data,
        )

    def to_json_serializable(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        content = self.content
        if not isinstance(content, (dict, list, str, int, float, bool, type(None))):
            content = str(content)

        return {
            "agent_name": self.agent_name,
            "output_type": self.output_type,
            "content": content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AgentOutputManager:
    """Manages agent output routing and persistence.

    Centralizes all agent output handling with:
    - Instance-aware data directories
    - Sandbox integration
    - Cross-instance sharing
    - Abstraction buffer integration
    """

    def __init__(
        self,
        instance_id: str,
        base_data_dir: str = "data",
        sandbox_dir: str = "sandbox",
        shared_outputs_dir: str | None = None,
        policy: FilesystemPolicy | None = None,
        abstraction_buffer: AbstractionBuffer | None = None,
    ) -> None:
        """Initialize output manager.

        Args:
            instance_id: This instance's ID
            base_data_dir: Base data directory
            sandbox_dir: Sandbox directory
            shared_outputs_dir: Shared outputs directory (default: data/shared/outputs)
            policy: Optional filesystem policy for permission checks
            abstraction_buffer: Optional abstraction buffer (uses global if None)
        """
        self.instance_id = instance_id
        self.base_data_dir = os.path.abspath(base_data_dir)
        self.sandbox_dir = os.path.abspath(sandbox_dir)
        self.shared_outputs_dir = os.path.abspath(
            shared_outputs_dir or os.path.join(base_data_dir, "shared", "outputs")
        )
        self._policy = policy
        self._abstraction = abstraction_buffer or get_abstraction_buffer()

        # Computed paths
        self.instance_data_dir = os.path.join(self.base_data_dir, instance_id)
        self.instance_shared_dir = os.path.join(self.shared_outputs_dir, instance_id)

        # Output streams per agent (for buffered writes)
        self._streams: dict[str, list[AgentOutput]] = {}
        self._stream_files: dict[str, str] = {}  # agent -> current file path
        self._lock = threading.Lock()

        # Ensure directories exist
        self._ensure_directories()

        # Callbacks for output events
        self._on_output_callbacks: list[Callable[[AgentOutput], None]] = []

    def _ensure_directories(self) -> None:
        """Create required directories."""
        directories = [
            self.instance_data_dir,
            os.path.join(self.instance_data_dir, "memory"),
            os.path.join(self.instance_data_dir, "logs"),
            os.path.join(self.instance_data_dir, "artifacts"),
            self.instance_shared_dir,
            os.path.join(self.sandbox_dir, "outputs"),
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)

    def write(self, output: AgentOutput) -> str | None:
        """Write an agent output to appropriate destination(s).

        Returns:
            Path to written file, or None if only to abstraction buffer
        """
        # Always write to abstraction buffer
        self._abstraction.append(output.to_log_record())

        # Notify callbacks
        for callback in self._on_output_callbacks:
            try:
                callback(output)
            except Exception as e:
                logger.error(f"Output callback error: {e}")

        # Route to destination(s)
        if output.destination == OutputDestination.ABSTRACTION_ONLY:
            return None

        path = None
        if output.destination in (OutputDestination.INSTANCE_DATA, OutputDestination.ALL):
            path = self._write_to_instance(output)

        if output.destination in (OutputDestination.SANDBOX, OutputDestination.ALL):
            path = self._write_to_sandbox(output)

        if output.destination in (OutputDestination.SHARED_OUTPUT, OutputDestination.ALL):
            path = self._write_to_shared(output)

        return path

    def _write_to_instance(self, output: AgentOutput) -> str:
        """Write output to instance data directory."""
        subdir = self._get_subdir_for_type(output.output_type)
        dir_path = os.path.join(self.instance_data_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)

        filename = output.filename or self._generate_filename(output)
        path = os.path.join(dir_path, filename)

        self._write_json(path, output)
        return path

    def _write_to_sandbox(self, output: AgentOutput) -> str:
        """Write output to sandbox outputs directory."""
        dir_path = os.path.join(self.sandbox_dir, "outputs", output.agent_name)
        os.makedirs(dir_path, exist_ok=True)

        filename = output.filename or self._generate_filename(output)
        path = os.path.join(dir_path, filename)

        self._write_json(path, output)
        return path

    def _write_to_shared(self, output: AgentOutput) -> str:
        """Write output to shared outputs directory."""
        dir_path = os.path.join(self.instance_shared_dir, output.agent_name)
        os.makedirs(dir_path, exist_ok=True)

        filename = output.filename or self._generate_filename(output)
        path = os.path.join(dir_path, filename)

        self._write_json(path, output)
        return path

    def _write_json(self, path: str, output: AgentOutput) -> None:
        """Write output as JSON."""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(output.to_json_serializable(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to write output to {path}: {e}")

    def _get_subdir_for_type(self, output_type: str) -> str:
        """Map output type to subdirectory."""
        type_dirs = {
            "memory": "memory",
            "log": "logs",
            "artifact": "artifacts",
            "result": "results",
            "context": "context",
            "goal": "goals",
            "percept": "percepts",
        }
        return type_dirs.get(output_type, "misc")

    def _generate_filename(self, output: AgentOutput) -> str:
        """Generate filename for output."""
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(output.timestamp))
        return f"{output.agent_name}_{output.output_type}_{timestamp}.json"

    # ─────────────────────────────────────────────────────────────────────────
    # Streaming output (for continuous data like logs)
    # ─────────────────────────────────────────────────────────────────────────

    def open_stream(
        self,
        agent_name: str,
        stream_type: str,
        destination: OutputDestination = OutputDestination.INSTANCE_DATA,
    ) -> str:
        """Open a streaming output file for an agent.

        Returns:
            Path to the stream file
        """
        with self._lock:
            stream_key = f"{agent_name}:{stream_type}"

            # Choose directory based on destination
            if destination == OutputDestination.SANDBOX:
                dir_path = os.path.join(self.sandbox_dir, "outputs", agent_name)
            elif destination == OutputDestination.SHARED_OUTPUT:
                dir_path = os.path.join(self.instance_shared_dir, agent_name)
            else:
                subdir = self._get_subdir_for_type(stream_type)
                dir_path = os.path.join(self.instance_data_dir, subdir)

            os.makedirs(dir_path, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_name}_{stream_type}_{timestamp}.jsonl"
            path = os.path.join(dir_path, filename)

            self._streams[stream_key] = []
            self._stream_files[stream_key] = path

            return path

    def write_to_stream(
        self,
        agent_name: str,
        stream_type: str,
        record: dict[str, Any],
    ) -> None:
        """Write a record to an open stream."""
        stream_key = f"{agent_name}:{stream_type}"

        with self._lock:
            if stream_key not in self._stream_files:
                # Auto-open stream
                self.open_stream(agent_name, stream_type)

            path = self._stream_files[stream_key]

        # Write record as JSONL
        try:
            record_with_ts = {"timestamp": time.time(), **record}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_with_ts, default=str) + "\n")

            # Also send to abstraction buffer
            log_record = LogRecord(
                timestamp=time.time(),
                level="INFO",
                source=agent_name,
                event=stream_type,
                data=record,
            )
            self._abstraction.append(log_record)

        except Exception as e:
            logger.error(f"Failed to write to stream {stream_key}: {e}")

    def close_stream(self, agent_name: str, stream_type: str) -> None:
        """Close a streaming output."""
        stream_key = f"{agent_name}:{stream_type}"
        with self._lock:
            self._streams.pop(stream_key, None)
            self._stream_files.pop(stream_key, None)

    # ─────────────────────────────────────────────────────────────────────────
    # Convenience methods for common output types
    # ─────────────────────────────────────────────────────────────────────────

    def write_memory(
        self,
        agent_name: str,
        memories: list[dict[str, Any]],
        share: bool = False,
    ) -> str:
        """Write memories to appropriate location.

        Args:
            agent_name: Name of the agent
            memories: List of memory dicts to persist
            share: Whether to also write to shared outputs

        Returns:
            Path to written file
        """
        output = AgentOutput(
            agent_name=agent_name,
            output_type="memory",
            content={"memories": memories, "count": len(memories)},
            destination=OutputDestination.ALL if share else OutputDestination.INSTANCE_DATA,
            metadata={"memory_count": len(memories)},
        )
        return self.write(output)

    def write_context(
        self,
        agent_name: str,
        context: dict[str, Any],
        share: bool = True,
    ) -> str:
        """Write structured context snapshot.

        Args:
            agent_name: Name of the agent
            context: Context dict to persist
            share: Whether to write to shared outputs (default True for context)

        Returns:
            Path to written file
        """
        output = AgentOutput(
            agent_name=agent_name,
            output_type="context",
            content=context,
            destination=OutputDestination.SHARED_OUTPUT if share else OutputDestination.INSTANCE_DATA,
            metadata={"has_goal": bool(context.get("active_goal"))},
        )
        return self.write(output)

    def write_artifact(
        self,
        agent_name: str,
        artifact_name: str,
        content: Any,
        to_sandbox: bool = True,
    ) -> str:
        """Write an artifact (generated file, analysis result, etc.).

        Args:
            agent_name: Name of the agent
            artifact_name: Name for the artifact
            content: Artifact content
            to_sandbox: Whether to write to sandbox (default True)

        Returns:
            Path to written file
        """
        output = AgentOutput(
            agent_name=agent_name,
            output_type="artifact",
            content=content,
            destination=OutputDestination.SANDBOX if to_sandbox else OutputDestination.INSTANCE_DATA,
            filename=f"{artifact_name}.json",
            metadata={"artifact_name": artifact_name},
        )
        return self.write(output)

    def write_result(
        self,
        agent_name: str,
        result: dict[str, Any],
        share: bool = False,
    ) -> str:
        """Write a result (tool execution, goal completion, etc.)."""
        output = AgentOutput(
            agent_name=agent_name,
            output_type="result",
            content=result,
            destination=OutputDestination.SHARED_OUTPUT if share else OutputDestination.INSTANCE_DATA,
            metadata={
                "success": result.get("success"),
                "result_type": result.get("type"),
            },
        )
        return self.write(output)

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_output(self, callback: Callable[[AgentOutput], None]) -> None:
        """Register callback for all outputs."""
        self._on_output_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Reading outputs
    # ─────────────────────────────────────────────────────────────────────────

    def list_outputs(
        self,
        agent_name: str | None = None,
        output_type: str | None = None,
        from_sandbox: bool = False,
    ) -> list[dict[str, Any]]:
        """List available outputs.

        Args:
            agent_name: Filter by agent name
            output_type: Filter by output type
            from_sandbox: Look in sandbox instead of instance data

        Returns:
            List of output metadata dicts
        """
        if from_sandbox:
            base_dir = os.path.join(self.sandbox_dir, "outputs")
        else:
            base_dir = self.instance_data_dir

        outputs = []

        for root, _, files in os.walk(base_dir):
            for f in files:
                if not f.endswith(".json") and not f.endswith(".jsonl"):
                    continue

                path = os.path.join(root, f)
                rel_path = os.path.relpath(path, base_dir)

                # Parse agent and type from path/filename
                parts = rel_path.split(os.sep)
                file_agent = parts[0] if len(parts) > 1 else "unknown"
                file_type = parts[1] if len(parts) > 2 else "misc"

                # Apply filters
                if agent_name and file_agent != agent_name:
                    continue
                if output_type and output_type not in f:
                    continue

                outputs.append({
                    "path": path,
                    "agent": file_agent,
                    "type": file_type,
                    "filename": f,
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path),
                })

        outputs.sort(key=lambda x: x["modified"], reverse=True)
        return outputs

    def read_output(self, path: str) -> dict[str, Any] | None:
        """Read an output file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read output {path}: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────


_global_output_manager: AgentOutputManager | None = None


def get_output_manager() -> AgentOutputManager | None:
    """Get the global output manager."""
    return _global_output_manager


def set_output_manager(manager: AgentOutputManager) -> None:
    """Set the global output manager."""
    global _global_output_manager
    _global_output_manager = manager


def init_output_manager(
    instance_id: str,
    base_data_dir: str = "data",
    sandbox_dir: str = "sandbox",
    policy: FilesystemPolicy | None = None,
) -> AgentOutputManager:
    """Initialize the global output manager."""
    manager = AgentOutputManager(
        instance_id=instance_id,
        base_data_dir=base_data_dir,
        sandbox_dir=sandbox_dir,
        policy=policy,
    )
    set_output_manager(manager)
    return manager
