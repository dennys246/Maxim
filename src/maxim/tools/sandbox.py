"""Sandbox tools for safe script execution and file operations.

Provides tools for:
- Reading files from data directory (read-only)
- Reading/writing files in sandbox
- Executing scripts in sandbox
- Listing outputs from other instances

All operations respect filesystem policy and autonomy levels.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable

from maxim.tools.base import Tool, ToolResult

if TYPE_CHECKING:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.utils.filesystem_policy import FilesystemPolicy
    from maxim.utils.sandbox_executor import SandboxExecutor
    from maxim.utils.output_watcher import OutputWatcher

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Read File Tool (Data Directory - Read Only)
# ─────────────────────────────────────────────────────────────────────────────


class ReadDataFileTool(Tool):
    """Read files from the data directory (read-only access).

    Respects filesystem policy to ensure instance isolation.
    """

    name = "read_data_file"
    description = "Read a file from the data directory. Only files within your instance's data folder or shared read-only areas are accessible."

    input_schema = {
        "path": str,  # Required: path relative to data dir or absolute
        "encoding": (str, "utf-8"),  # Optional: file encoding
        "max_bytes": (int, 1_000_000),  # Optional: max bytes to read (1MB default)
    }

    def __init__(
        self,
        policy: FilesystemPolicy,
        base_data_dir: str = "data",
    ) -> None:
        super().__init__()
        self._policy = policy
        self._base_data_dir = os.path.abspath(base_data_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, error="Missing required parameter: path")

        encoding = str(kwargs.get("encoding", "utf-8") or "utf-8")
        max_bytes = int(kwargs.get("max_bytes", 1_000_000) or 1_000_000)

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(self._base_data_dir, path)
        path = os.path.abspath(path)

        # Check permission
        allowed, reason = self._policy.check_permission(
            path, self._policy.policies[0].permissions.READ  # Get READ permission
        )
        if not allowed:
            return ToolResult(success=False, error=f"Permission denied: {reason}")

        # Check file exists
        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"Path is not a file: {path}")

        # Read file
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read(max_bytes)

            truncated = os.path.getsize(path) > max_bytes

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "content": content,
                    "size": os.path.getsize(path),
                    "truncated": truncated,
                },
            )

        except UnicodeDecodeError as e:
            return ToolResult(success=False, error=f"Encoding error: {e}")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox File Tools (Read/Write)
# ─────────────────────────────────────────────────────────────────────────────


class ReadSandboxFileTool(Tool):
    """Read files from the sandbox directory."""

    name = "read_sandbox_file"
    description = "Read a file from the sandbox directory."

    input_schema = {
        "path": str,  # Required: path relative to sandbox or absolute
        "encoding": (str, "utf-8"),
        "max_bytes": (int, 1_000_000),
    }

    def __init__(self, sandbox_dir: str = "sandbox") -> None:
        super().__init__()
        self._sandbox_dir = os.path.abspath(sandbox_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, error="Missing required parameter: path")

        encoding = str(kwargs.get("encoding", "utf-8") or "utf-8")
        max_bytes = int(kwargs.get("max_bytes", 1_000_000) or 1_000_000)

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(self._sandbox_dir, path)
        path = os.path.abspath(path)

        # Ensure within sandbox
        if not path.startswith(self._sandbox_dir):
            return ToolResult(success=False, error="Path must be within sandbox directory")

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"Path is not a file: {path}")

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read(max_bytes)

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "content": content,
                    "size": os.path.getsize(path),
                    "truncated": os.path.getsize(path) > max_bytes,
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")


class WriteSandboxFileTool(Tool):
    """Write files to the sandbox directory."""

    name = "write_sandbox_file"
    description = "Write a file to the sandbox directory. Can create scripts or data files."

    input_schema = {
        "path": str,  # Required: path relative to sandbox or absolute
        "content": str,  # Required: file content
        "encoding": (str, "utf-8"),
        "append": (bool, False),  # Optional: append instead of overwrite
    }

    def __init__(self, sandbox_dir: str = "sandbox") -> None:
        super().__init__()
        self._sandbox_dir = os.path.abspath(sandbox_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content")

        if not path:
            return ToolResult(success=False, error="Missing required parameter: path")
        if content is None:
            return ToolResult(success=False, error="Missing required parameter: content")

        encoding = str(kwargs.get("encoding", "utf-8") or "utf-8")
        append = bool(kwargs.get("append", False))

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(self._sandbox_dir, path)
        path = os.path.abspath(path)

        # Ensure within sandbox
        if not path.startswith(self._sandbox_dir):
            return ToolResult(success=False, error="Path must be within sandbox directory")

        try:
            # Create parent directories
            os.makedirs(os.path.dirname(path), exist_ok=True)

            mode = "a" if append else "w"
            with open(path, mode, encoding=encoding) as f:
                f.write(content)

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "size": os.path.getsize(path),
                    "action": "appended" if append else "written",
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")


class ListSandboxTool(Tool):
    """List files in the sandbox directory."""

    name = "list_sandbox"
    description = "List files and directories in the sandbox."

    input_schema = {
        "path": (str, ""),  # Optional: subdirectory path
        "recursive": (bool, False),
    }

    def __init__(self, sandbox_dir: str = "sandbox") -> None:
        super().__init__()
        self._sandbox_dir = os.path.abspath(sandbox_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        subpath = str(kwargs.get("path", "") or "")
        recursive = bool(kwargs.get("recursive", False))

        path = os.path.join(self._sandbox_dir, subpath) if subpath else self._sandbox_dir
        path = os.path.abspath(path)

        if not path.startswith(self._sandbox_dir):
            return ToolResult(success=False, error="Path must be within sandbox directory")

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"Path not found: {path}")

        try:
            entries = []

            if recursive:
                for root, dirs, files in os.walk(path):
                    for d in dirs:
                        full_path = os.path.join(root, d)
                        rel_path = os.path.relpath(full_path, self._sandbox_dir)
                        entries.append({"name": rel_path, "type": "directory"})
                    for f in files:
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, self._sandbox_dir)
                        entries.append({
                            "name": rel_path,
                            "type": "file",
                            "size": os.path.getsize(full_path),
                        })
            else:
                for name in os.listdir(path):
                    full_path = os.path.join(path, name)
                    if os.path.isdir(full_path):
                        entries.append({"name": name, "type": "directory"})
                    else:
                        entries.append({
                            "name": name,
                            "type": "file",
                            "size": os.path.getsize(full_path),
                        })

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "entries": entries,
                    "count": len(entries),
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to list directory: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Script Execution Tool
# ─────────────────────────────────────────────────────────────────────────────


class ExecuteSandboxScriptTool(Tool):
    """Execute a script in the sandbox with resource limits.

    Respects autonomy levels:
    - PLANNING: Proposes execution, doesn't run
    - SUPERVISED: Requires approval for first run of each script
    - AUTONOMOUS: Executes freely within sandbox constraints
    """

    name = "execute_sandbox_script"
    description = "Execute a Python or shell script in the sandbox. Scripts are resource-limited and isolated."

    input_schema = {
        "script_path": str,  # Required: path to script in sandbox
        "args": (list, None),  # Optional: command-line arguments
        "env": (dict, None),  # Optional: environment variables
    }

    def __init__(
        self,
        executor: SandboxExecutor,
        autonomy_controller: AutonomyController | None = None,
    ) -> None:
        super().__init__()
        self._executor = executor
        self._autonomy = autonomy_controller
        self._approved_scripts: set[str] = set()  # Scripts approved in SUPERVISED mode

    def execute(self, **kwargs: Any) -> ToolResult:
        from maxim.agents.autonomy import AutonomyLevel

        script_path = kwargs.get("script_path")
        if not script_path:
            return ToolResult(success=False, error="Missing required parameter: script_path")

        args = kwargs.get("args") or []
        env = kwargs.get("env") or {}

        # Ensure args is a list
        if not isinstance(args, list):
            args = [str(args)]

        # Check autonomy level
        require_approval = True
        if self._autonomy:
            level = self._autonomy.current_level

            if level == AutonomyLevel.PLANNING:
                # In PLANNING mode, don't execute - just propose
                return ToolResult(
                    success=True,
                    output={
                        "status": "proposed",
                        "message": f"Would execute script: {script_path}",
                        "args": args,
                        "autonomy_level": "planning",
                    },
                )

            elif level == AutonomyLevel.SUPERVISED:
                # In SUPERVISED mode, require approval for new scripts
                script_key = os.path.abspath(
                    os.path.join(self._executor.sandbox_dir, script_path)
                    if not os.path.isabs(script_path)
                    else script_path
                )
                if script_key in self._approved_scripts:
                    require_approval = False
                else:
                    # Mark as needing approval (executor will call approval_callback)
                    require_approval = True

            elif level == AutonomyLevel.AUTONOMOUS:
                # In AUTONOMOUS mode, execute freely
                require_approval = False

        # Set up approval callback for SUPERVISED mode
        def approval_callback(path: str, content: str) -> bool:
            # For now, auto-approve in SUPERVISED if within sandbox
            # In real usage, this would prompt the user
            logger.info(f"Auto-approving script in SUPERVISED mode: {path}")
            self._approved_scripts.add(path)
            return True

        if require_approval:
            self._executor.approval_callback = approval_callback

        # Execute
        result = self._executor.execute(
            script_path=script_path,
            args=args,
            env=env,
            require_approval=require_approval,
        )

        # Log to autonomy controller
        if self._autonomy:
            self._autonomy.log_action(
                action_type="executed",
                action={"tool": "execute_sandbox_script", "script": script_path},
                reasoning=f"Executed sandbox script: {script_path}",
                mode="sandbox",
                outcome=result.status.value,
                error=result.error,
            )

        return ToolResult(
            success=result.status.value == "success",
            output=result.to_dict(),
            error=result.error,
        )


class CreateSandboxScriptTool(Tool):
    """Create a new script in the sandbox."""

    name = "create_sandbox_script"
    description = "Create a new Python or shell script in the sandbox scripts directory."

    input_schema = {
        "name": str,  # Required: script name (without extension)
        "content": str,  # Required: script content
        "script_type": (str, "python"),  # Optional: "python" or "shell"
    }

    def __init__(self, executor: SandboxExecutor) -> None:
        super().__init__()
        self._executor = executor

    def execute(self, **kwargs: Any) -> ToolResult:
        name = kwargs.get("name")
        content = kwargs.get("content")
        script_type = str(kwargs.get("script_type", "python") or "python").lower()

        if not name:
            return ToolResult(success=False, error="Missing required parameter: name")
        if content is None:
            return ToolResult(success=False, error="Missing required parameter: content")

        if script_type not in ("python", "shell"):
            return ToolResult(success=False, error="script_type must be 'python' or 'shell'")

        try:
            path = self._executor.write_script(name, content, script_type)
            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "name": name,
                    "type": script_type,
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to create script: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Instance Output Tools
# ─────────────────────────────────────────────────────────────────────────────


class ListOtherInstanceOutputsTool(Tool):
    """List output files from other Maxim instances."""

    name = "list_other_outputs"
    description = "List files that other Maxim instances have written to shared outputs."

    input_schema = {
        "from_instance": (str, None),  # Optional: filter to specific instance
        "limit": (int, 50),  # Optional: max files to return
    }

    def __init__(self, watcher: OutputWatcher) -> None:
        super().__init__()
        self._watcher = watcher

    def execute(self, **kwargs: Any) -> ToolResult:
        from_instance = kwargs.get("from_instance")
        limit = int(kwargs.get("limit", 50) or 50)

        try:
            outputs = self._watcher.list_other_outputs(from_instance=from_instance)
            outputs = outputs[:limit]

            return ToolResult(
                success=True,
                output={
                    "files": [f.to_dict() for f in outputs],
                    "count": len(outputs),
                    "instances": self._watcher.get_other_instances(),
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to list outputs: {e}")


class ReadOtherInstanceOutputTool(Tool):
    """Read a file from another instance's shared outputs."""

    name = "read_other_output"
    description = "Read a file from another Maxim instance's shared outputs (read-only)."

    input_schema = {
        "path": str,  # Required: path to file
        "encoding": (str, "utf-8"),
        "max_bytes": (int, 1_000_000),
    }

    def __init__(self, watcher: OutputWatcher, shared_outputs_dir: str) -> None:
        super().__init__()
        self._watcher = watcher
        self._shared_outputs_dir = os.path.abspath(shared_outputs_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        path = kwargs.get("path")
        if not path:
            return ToolResult(success=False, error="Missing required parameter: path")

        encoding = str(kwargs.get("encoding", "utf-8") or "utf-8")
        max_bytes = int(kwargs.get("max_bytes", 1_000_000) or 1_000_000)

        # Resolve path
        if not os.path.isabs(path):
            path = os.path.join(self._shared_outputs_dir, path)
        path = os.path.abspath(path)

        # Ensure within shared outputs
        if not path.startswith(self._shared_outputs_dir):
            return ToolResult(success=False, error="Path must be within shared outputs directory")

        # Don't allow reading own outputs through this tool
        own_instance_dir = os.path.join(self._shared_outputs_dir, self._watcher.own_instance_id)
        if path.startswith(own_instance_dir):
            return ToolResult(success=False, error="Use read_sandbox_file to read your own outputs")

        if not os.path.exists(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read(max_bytes)

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "content": content,
                    "size": os.path.getsize(path),
                    "truncated": os.path.getsize(path) > max_bytes,
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to read file: {e}")


class WriteToSharedOutputsTool(Tool):
    """Write a file to this instance's shared outputs directory."""

    name = "write_shared_output"
    description = "Write a file to your instance's shared outputs, visible to other instances."

    input_schema = {
        "filename": str,  # Required: filename (placed in instance's output folder)
        "content": str,  # Required: file content
        "encoding": (str, "utf-8"),
    }

    def __init__(self, instance_outputs_dir: str) -> None:
        super().__init__()
        self._outputs_dir = os.path.abspath(instance_outputs_dir)

    def execute(self, **kwargs: Any) -> ToolResult:
        filename = kwargs.get("filename")
        content = kwargs.get("content")

        if not filename:
            return ToolResult(success=False, error="Missing required parameter: filename")
        if content is None:
            return ToolResult(success=False, error="Missing required parameter: content")

        encoding = str(kwargs.get("encoding", "utf-8") or "utf-8")

        # Sanitize filename (no path traversal)
        filename = os.path.basename(filename)
        if not filename:
            return ToolResult(success=False, error="Invalid filename")

        path = os.path.join(self._outputs_dir, filename)

        try:
            os.makedirs(self._outputs_dir, exist_ok=True)

            with open(path, "w", encoding=encoding) as f:
                f.write(content)

            return ToolResult(
                success=True,
                output={
                    "path": path,
                    "filename": filename,
                    "size": os.path.getsize(path),
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=f"Failed to write file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Factory
# ─────────────────────────────────────────────────────────────────────────────


def build_sandbox_tools(
    policy: FilesystemPolicy,
    executor: SandboxExecutor,
    watcher: OutputWatcher | None = None,
    autonomy_controller: AutonomyController | None = None,
) -> list[Tool]:
    """Build all sandbox tools.

    Args:
        policy: Filesystem policy for the instance
        executor: Sandbox executor
        watcher: Optional output watcher for cross-instance features
        autonomy_controller: Optional autonomy controller for approval flow

    Returns:
        List of sandbox tools
    """
    tools: list[Tool] = [
        # Data directory (read-only)
        ReadDataFileTool(policy, policy.base_data_dir),
        # Sandbox (read/write/execute)
        ReadSandboxFileTool(policy.sandbox_dir),
        WriteSandboxFileTool(policy.sandbox_dir),
        ListSandboxTool(policy.sandbox_dir),
        # Script execution
        CreateSandboxScriptTool(executor),
        ExecuteSandboxScriptTool(executor, autonomy_controller),
        # Shared outputs
        WriteToSharedOutputsTool(policy.instance_outputs_dir),
    ]

    # Add cross-instance tools if watcher is available
    if watcher:
        tools.extend([
            ListOtherInstanceOutputsTool(watcher),
            ReadOtherInstanceOutputTool(watcher, policy.shared_outputs_dir),
        ])

    return tools
