from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maxim.utils.filesystem_policy import ModeFilesystemConfig


# ─────────────────────────────────────────────────────────────────────────────
# Path Sanitization Utilities
# ─────────────────────────────────────────────────────────────────────────────


class PathValidationError(ValueError):
    """Raised when path validation fails."""

    pass


def sanitize_path(path: str, *, max_length: int = 4096) -> str:
    """Sanitize a file path to prevent injection attacks.

    Args:
        path: The path to sanitize.
        max_length: Maximum allowed path length.

    Returns:
        The sanitized path.

    Raises:
        PathValidationError: If the path is invalid or potentially malicious.
    """
    if not path:
        raise PathValidationError("Path cannot be empty")

    # Check for null bytes (can truncate paths in C-based systems)
    if "\x00" in path:
        raise PathValidationError("Path contains null bytes")

    # Check for excessive length
    if len(path) > max_length:
        raise PathValidationError(f"Path exceeds maximum length of {max_length}")

    # Check for non-printable characters (except common ones like space, tab)
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", path):
        raise PathValidationError("Path contains invalid control characters")

    # Normalize the path to resolve . and .. segments
    normalized = os.path.normpath(path)

    return normalized


def validate_path_traversal(
    path: str,
    base_dir: str | None = None,
    allow_symlinks: bool = False,
) -> str:
    """Validate a path against traversal attacks.

    Args:
        path: The path to validate.
        base_dir: If provided, ensure path is within this directory.
        allow_symlinks: If False, resolve symlinks and re-check.

    Returns:
        The validated (and possibly resolved) path.

    Raises:
        PathValidationError: If the path is outside allowed bounds.
    """
    # First sanitize the path
    path = sanitize_path(path)

    # Resolve to absolute path
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    # If checking against base directory
    if base_dir:
        base_dir = os.path.realpath(base_dir) if not allow_symlinks else os.path.abspath(base_dir)

        # Resolve symlinks if not allowed
        check_path = os.path.realpath(path) if not allow_symlinks else path

        # Verify containment
        if not (check_path.startswith(base_dir + os.sep) or check_path == base_dir):
            raise PathValidationError(f"Path escapes base directory: {base_dir}")

        return check_path

    return path


from maxim.utils.gpu_compat import env_flag as _env_flag

from .base import Tool, ToolResult
from maxim.agents.bus import ToolErrorKind


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file from disk"
    input_schema = {
        "path": str,
        "tail_lines": (int, None),  # optional
    }

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        # If allowed_dirs is set, only allow reading from those directories
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

    def execute(self, path: str, tail_lines: int | None = None) -> ToolResult:
        try:
            # Sanitize and validate the path
            path = sanitize_path(path)

            # Check against allowed directories if configured
            if self._allowed_dirs:
                real_path = os.path.realpath(path)
                in_allowed = False
                for allowed in self._allowed_dirs:
                    if real_path.startswith(allowed + os.sep) or real_path == allowed:
                        in_allowed = True
                        break
                if not in_allowed:
                    return ToolResult(
                        success=False,
                        error="Path must be within allowed directories",
                    )
        except PathValidationError as e:
            return ToolResult(success=False, error=str(e))

        if tail_lines is not None:
            n = int(tail_lines)
            if n <= 0:
                return ""
            return self._read_tail_lines(path, n)

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _read_tail_lines(path: str | os.PathLike[str], n: int) -> str:
        block_size = 4096
        data = b""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0 and data.count(b"\n") <= n:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
            lines = data.splitlines()[-n:]
            return b"\n".join(lines).decode("utf-8", errors="replace")


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file on disk safely"
    input_schema = {
        "path": str,
        "content": str,
        "overwrite": (bool, False),  # optional, default False
    }

    # Maximum content size to prevent memory exhaustion (10 MB)
    MAX_CONTENT_SIZE: int = 10 * 1024 * 1024

    # Directories that should never be written to
    FORBIDDEN_PATHS: tuple[str, ...] = (
        "/bin",
        "/sbin",
        "/usr/bin",
        "/usr/sbin",
        "/usr/local/bin",
        "/etc",
        "/var",
        "/tmp",
        "/dev",
        "/proc",
        "/sys",
        "/boot",
        "/lib",
        "/lib64",
        "/opt",
        "/root",
        "/home",  # Protect all home directories by default
        "/System",
        "/Library",
        "/Applications",  # macOS system paths
        "/private/etc",
        "/private/var",  # macOS private paths
    )

    # File extensions that should never be written
    FORBIDDEN_EXTENSIONS: tuple[str, ...] = (
        ".exe",
        ".dll",
        ".so",
        ".dylib",  # Executables/libraries
        ".pem",
        ".key",
        ".crt",
        ".p12",  # Certificates/keys
        ".env",
        ".credentials",
        ".secret",  # Secrets
    )

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        mode_config: "ModeFilesystemConfig | None" = None,
    ) -> None:
        super().__init__()
        # If allowed_dirs is set, only allow writing within those directories
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None
        self._mode_config = mode_config

    def _is_path_allowed(self, path: Path) -> tuple[bool, str]:
        """Check if path is safe to write to (basic safety checks)."""
        # Resolve to real path (handles symlinks)
        real_path = os.path.realpath(str(path))

        # Check against forbidden paths
        for forbidden in self.FORBIDDEN_PATHS:
            if real_path.startswith(forbidden + os.sep) or real_path == forbidden:
                return False, f"Writing to {forbidden} is not allowed"

        # Check against forbidden extensions
        suffix = path.suffix.lower()
        if suffix in self.FORBIDDEN_EXTENSIONS:
            return False, f"Writing files with extension {suffix} is not allowed"

        # If allowlist is set, verify path is within allowed directories
        if self._allowed_dirs:
            in_allowed = False
            for allowed in self._allowed_dirs:
                if real_path.startswith(allowed + os.sep) or real_path == allowed:
                    in_allowed = True
                    break
            if not in_allowed:
                return False, f"Path must be within allowed directories: {self._allowed_dirs}"

        return True, ""

    def set_mode_config(self, config: "ModeFilesystemConfig") -> None:
        """Update the mode filesystem config (for runtime mode changes)."""
        self._mode_config = config

    def execute(self, **kwargs) -> ToolResult:
        try:
            raw_path = kwargs["path"]
            content = kwargs["content"]
            overwrite = kwargs.get("overwrite", False)

            # Check content size to prevent memory exhaustion
            if len(content) > self.MAX_CONTENT_SIZE:
                return ToolResult(
                    success=False,
                    error=f"Content size ({len(content)} bytes) exceeds maximum allowed ({self.MAX_CONTENT_SIZE} bytes)",
                )

            # Sanitize the path first
            try:
                validated_path = sanitize_path(raw_path)
            except PathValidationError as e:
                return ToolResult(success=False, error=str(e))

            path = Path(validated_path)

            # Check basic path restrictions (forbidden paths, extensions)
            allowed, reason = self._is_path_allowed(path)
            if not allowed:
                return ToolResult(success=False, error=reason)

            # Check mode-based write permission if config is set
            if self._mode_config is not None:
                from maxim.utils.filesystem_policy import check_write_permission

                perm_result = check_write_permission(str(path), self._mode_config)

                if not perm_result.allowed:
                    if perm_result.requires_approval:
                        # Return a special result indicating approval is needed
                        return ToolResult(
                            success=False,
                            error=perm_result.reason,
                            output={
                                "requires_approval": True,
                                "path": str(path),
                                "is_proposal": self._mode_config.cwd_access == "propose",
                            },
                        )
                    return ToolResult(success=False, error=perm_result.reason)

                # If approval is required but allowed, mark it in the result
                if perm_result.requires_approval:
                    # For supervised mode, we allow the write but flag it
                    # The autonomy controller should intercept this
                    pass  # Continue with the write, approval handled at higher level

            # Safety check for overwrite
            if path.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    error="File already exists (use overwrite=True to replace)",
                )

            # Ensure parent directories exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            path.write_text(content, encoding="utf-8")

            return ToolResult(success=True, output={"path": str(path), "size": len(content)})

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ExecuteFileTool(Tool):
    name = "execute_file"
    description = "Execute a file using the appropriate interpreter based on file extension."
    timeout: float = 10.0  # scripts get a shorter default than general tools
    input_schema = {
        "path": str,
        "timeout": (float, 10.0),  # optional seconds
    }

    EXT_INTERPRETER = {
        ".py": "python3",
        ".sh": "bash",
    }

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        mode_config: "ModeFilesystemConfig | None" = None,
    ) -> None:
        super().__init__()
        # If allowed_dirs is set, only allow executing from those directories
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None
        self._mode_config = mode_config

    def set_mode_config(self, config: "ModeFilesystemConfig") -> None:
        """Update the mode filesystem config (for runtime mode changes)."""
        self._mode_config = config

    def execute(self, **kwargs) -> ToolResult:
        try:
            if not _env_flag("MAXIM_ALLOW_EXECUTE_FILE", False):
                return ToolResult(
                    success=False,
                    error="ExecuteFileTool disabled. Set MAXIM_ALLOW_EXECUTE_FILE=1 to enable.",
                )

            raw_path = kwargs["path"]
            timeout = kwargs.get("timeout", self.timeout)

            # Sanitize and validate the path
            try:
                validated_path = sanitize_path(raw_path)
            except PathValidationError as e:
                return ToolResult(success=False, error=str(e))

            path = Path(validated_path)

            # Check against allowed directories if configured
            if self._allowed_dirs:
                real_path = os.path.realpath(str(path))
                in_allowed = False
                for allowed in self._allowed_dirs:
                    if real_path.startswith(allowed + os.sep) or real_path == allowed:
                        in_allowed = True
                        break
                if not in_allowed:
                    return ToolResult(
                        success=False,
                        error="Path must be within allowed directories",
                    )

            # Check mode-based execute permission if config is set
            if self._mode_config is not None:
                from maxim.utils.filesystem_policy import check_execute_permission

                perm_result = check_execute_permission(str(path), self._mode_config)

                if not perm_result.allowed:
                    return ToolResult(success=False, error=perm_result.reason)

                if perm_result.requires_approval:
                    # Return a special result indicating approval is needed
                    return ToolResult(
                        success=False,
                        error=perm_result.reason,
                        output={
                            "requires_approval": True,
                            "path": str(path),
                            "action": "execute",
                        },
                    )

            if not path.exists():
                return ToolResult(success=False, error="File does not exist")

            ext = path.suffix
            if ext not in self.EXT_INTERPRETER:
                return ToolResult(success=False, error=f"Unsupported file type: {ext} (allowed: .py, .sh)")

            interpreter = self.EXT_INTERPRETER[ext]

            # Run in subprocess (uses list form to prevent shell injection)
            result = subprocess.run(
                [interpreter, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,  # Explicitly disable shell to prevent injection
            )

            return ToolResult(
                success=(result.returncode == 0),
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "path": str(path),
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Execution timed out")
        except Exception as e:
            return ToolResult(success=False, error=str(e))


class EditFileTool(Tool):
    """Replace specific text in a file.

    When old_text matches multiple locations, use context_before and/or
    context_after to disambiguate which occurrence to replace.
    """

    name = "edit_file"
    description = (
        "Replace specific text in a file. Read the file first to see current contents. "
        "If old_text matches multiple locations, use context_before/context_after to disambiguate."
    )
    input_schema = {
        "path": str,
        "old_text": str,
        "new_text": str,
        "expected_count": (int, 1),
        "context_before": (str, None),  # Text that must appear immediately before old_text
        "context_after": (str, None),  # Text that must appear immediately after old_text
    }

    # Reuse WriteFileTool's safety lists
    FORBIDDEN_PATHS = WriteFileTool.FORBIDDEN_PATHS
    FORBIDDEN_EXTENSIONS = WriteFileTool.FORBIDDEN_EXTENSIONS

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

    def _is_path_allowed(self, path: Path) -> tuple[bool, str]:
        """Check if path is safe to write to (mirrors WriteFileTool)."""
        real_path = os.path.realpath(str(path))

        for forbidden in self.FORBIDDEN_PATHS:
            if real_path.startswith(forbidden + os.sep) or real_path == forbidden:
                return False, f"Writing to {forbidden} is not allowed"

        suffix = path.suffix.lower()
        if suffix in self.FORBIDDEN_EXTENSIONS:
            return False, f"Writing files with extension {suffix} is not allowed"

        if self._allowed_dirs:
            in_allowed = False
            for allowed in self._allowed_dirs:
                if real_path.startswith(allowed + os.sep) or real_path == allowed:
                    in_allowed = True
                    break
            if not in_allowed:
                return False, f"Path must be within allowed directories: {self._allowed_dirs}"

        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        try:
            raw_path = kwargs["path"]
            old_text = kwargs["old_text"]
            new_text = kwargs["new_text"]
            expected = kwargs.get("expected_count", 1)
            context_before = kwargs.get("context_before")
            context_after = kwargs.get("context_after")

            try:
                validated_path = sanitize_path(raw_path)
            except PathValidationError as e:
                return ToolResult(success=False, error=str(e))

            path = Path(validated_path)

            allowed, reason = self._is_path_allowed(path)
            if not allowed:
                return ToolResult(success=False, error=reason)

            try:
                content = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return ToolResult(
                    success=False,
                    error=f"File not found: {path}",
                    error_kind=ToolErrorKind.FILE_NOT_FOUND,
                )
            except PermissionError:
                return ToolResult(
                    success=False,
                    error=f"Permission denied: {path}",
                    error_kind=ToolErrorKind.PERMISSION_DENIED,
                )

            # Context-aware disambiguation: when context_before or context_after
            # is provided, search for the full pattern and replace only old_text.
            if context_before is not None or context_after is not None:
                return self._replace_with_context(
                    content,
                    path,
                    old_text,
                    new_text,
                    context_before or "",
                    context_after or "",
                )

            # Standard path: exact match without context
            count = content.count(old_text)
            if count == 0:
                lines = content.splitlines()
                near_matches = [
                    f"  line {i + 1}: {line.strip()}"
                    for i, line in enumerate(lines)
                    if old_text[:20] in line or old_text[-20:] in line
                ]
                hint = "\nNear matches:\n" + "\n".join(near_matches[:5]) if near_matches else ""
                return ToolResult(
                    success=False,
                    error=f"old_text not found in file.{hint}",
                    error_kind=ToolErrorKind.INVALID_INPUT,
                )
            if count != expected:
                lines = content.splitlines()
                match_lines = [i + 1 for i, line in enumerate(lines) if old_text in line]
                # Auto-suggest context when multiple matches found
                suggestions = self._suggest_context(content, old_text, match_lines)
                suggestion_hint = ""
                if suggestions:
                    suggestion_hint = "\nUse context_before/context_after to disambiguate. Suggestions:"
                    for s in suggestions[:3]:
                        suggestion_hint += f'\n  line {s["line"]}: context_before="{s["before"]}"'
                return ToolResult(
                    success=False,
                    error=f"Expected {expected} occurrences, found {count} at lines {match_lines}.{suggestion_hint}",
                    error_kind=ToolErrorKind.VALIDATION,
                )

            new_content = content.replace(old_text, new_text, expected)
            path.write_text(new_content, encoding="utf-8")
            return ToolResult(success=True, output=f"Replaced {count} occurrence(s)")

        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _replace_with_context(
        self,
        content: str,
        path: Path,
        old_text: str,
        new_text: str,
        context_before: str,
        context_after: str,
    ) -> ToolResult:
        """Replace old_text at the location uniquely identified by surrounding context."""
        full_pattern = context_before + old_text + context_after
        count = content.count(full_pattern)

        if count == 0:
            # Try without context to give helpful error
            bare_count = content.count(old_text)
            if bare_count == 0:
                return ToolResult(
                    success=False,
                    error="old_text not found in file (with or without context).",
                    error_kind=ToolErrorKind.INVALID_INPUT,
                )
            return ToolResult(
                success=False,
                error=(
                    f"old_text found {bare_count} time(s) but not with the given context. "
                    f"Check that context_before/context_after match the exact surrounding text."
                ),
                error_kind=ToolErrorKind.INVALID_INPUT,
            )

        if count > 1:
            return ToolResult(
                success=False,
                error=f"Context pattern still matches {count} locations. Provide more specific context.",
                error_kind=ToolErrorKind.VALIDATION,
            )

        # Unique match — replace only the old_text within the full pattern
        replacement = context_before + new_text + context_after
        new_content = content.replace(full_pattern, replacement, 1)
        path.write_text(new_content, encoding="utf-8")
        return ToolResult(success=True, output="Replaced 1 occurrence (context-disambiguated)")

    @staticmethod
    def _suggest_context(content: str, old_text: str, match_lines: list[int]) -> list[dict]:
        """Generate context suggestions for ambiguous matches."""
        lines = content.splitlines()
        suggestions = []
        for line_num in match_lines[:5]:
            idx = line_num - 1
            # Use the preceding line as context_before suggestion
            if idx > 0:
                prev_line = lines[idx - 1].rstrip()
                # Truncate long lines
                if len(prev_line) > 60:
                    prev_line = prev_line[:60] + "..."
                suggestions.append(
                    {
                        "line": line_num,
                        "before": prev_line + "\n",
                    }
                )
        return suggestions


class GlobTool(Tool):
    """Search for files and directories using glob patterns.

    Supports patterns like:
    - *.py - all Python files in current directory
    - **/*.py - all Python files recursively
    - src/**/*.ts - TypeScript files under src/
    - data/*.{json,yaml} - JSON and YAML files in data/
    """

    name = "glob"
    description = "Search for files and directories using glob patterns"
    input_schema = {
        "pattern": str,  # Required: glob pattern (e.g., "**/*.py")
        "path": (str, "."),  # Optional: base path to search from
        "max_results": (int, 100),  # Optional: max results to return
        "include_hidden": (bool, False),  # Optional: include hidden files
    }

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

    def execute(self, **kwargs) -> ToolResult:
        try:
            pattern = kwargs.get("pattern")
            if not pattern:
                return ToolResult(success=False, error="Missing required parameter: pattern")

            base_path = kwargs.get("path", ".") or "."
            max_results = int(kwargs.get("max_results", 100) or 100)
            include_hidden = bool(kwargs.get("include_hidden", False))

            # Sanitize and validate base path
            try:
                validated_path = sanitize_path(base_path)
            except PathValidationError as e:
                return ToolResult(success=False, error=str(e))

            base = Path(validated_path).resolve()

            # Check against allowed directories if configured
            if self._allowed_dirs:
                in_allowed = False
                for allowed in self._allowed_dirs:
                    if str(base).startswith(allowed + os.sep) or str(base) == allowed:
                        in_allowed = True
                        break
                if not in_allowed:
                    return ToolResult(
                        success=False,
                        error="Path must be within allowed directories",
                    )

            if not base.exists():
                return ToolResult(success=False, error=f"Base path does not exist: {base}")

            # Perform glob search
            str(base / pattern)
            matches = []

            for match in Path(base).glob(pattern):
                # Skip hidden files if not requested
                if not include_hidden and any(part.startswith(".") for part in match.parts):
                    continue

                # Security: ensure match is within base directory
                try:
                    real_match = match.resolve()
                    if self._allowed_dirs:
                        in_allowed = False
                        for allowed in self._allowed_dirs:
                            if str(real_match).startswith(allowed + os.sep) or str(real_match) == allowed:
                                in_allowed = True
                                break
                        if not in_allowed:
                            continue
                except (OSError, ValueError):
                    continue

                matches.append(
                    {
                        "path": str(match),
                        "name": match.name,
                        "is_file": match.is_file(),
                        "is_dir": match.is_dir(),
                        "size": match.stat().st_size if match.is_file() else None,
                    }
                )

                if len(matches) >= max_results:
                    break

            return ToolResult(
                success=True,
                output={
                    "pattern": pattern,
                    "base_path": str(base),
                    "matches": matches,
                    "count": len(matches),
                    "truncated": len(matches) >= max_results,
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class RequestDirectoryChangeTool(Tool):
    """Request to change the effective working directory.

    This tool is available in active mode to request changing the working
    directory boundary. The change affects what files can be accessed by
    filesystem tools.

    Note: This does NOT call os.chdir(). It updates the containment boundary
    used by filesystem tools. Only available when mode allows directory changes.
    """

    name = "request_directory_change"
    description = "Request to change the effective working directory for file operations"
    input_schema = {
        "path": str,  # Required: new directory path
        "reason": str,  # Required: why the change is needed
    }

    def __init__(self, can_change: bool = False) -> None:
        """Initialize directory change request tool.

        Args:
            can_change: Whether this mode allows directory changes.
        """
        super().__init__()
        self._can_change = can_change

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = kwargs.get("path")
            reason = kwargs.get("reason", "")

            if not path:
                return ToolResult(success=False, error="Missing required parameter: path")

            if not self._can_change:
                return ToolResult(
                    success=False,
                    error="Directory change not allowed in current mode. Only active and singularity modes can request directory changes.",
                )

            # Validate the path
            try:
                validated_path = sanitize_path(path)
            except PathValidationError as e:
                return ToolResult(success=False, error=str(e))

            # Resolve to absolute path
            abs_path = os.path.abspath(validated_path)

            if not os.path.isdir(abs_path):
                return ToolResult(
                    success=False,
                    error=f"Directory does not exist: {abs_path}",
                )

            # Import here to avoid circular dependency
            from maxim.utils.filesystem_policy import set_effective_cwd, ensure_sandbox_exists

            # Update effective CWD
            if not set_effective_cwd(abs_path):
                return ToolResult(
                    success=False,
                    error=f"Failed to change to directory: {abs_path}",
                )

            # Ensure sandbox exists in new directory
            sandbox_path = ensure_sandbox_exists(abs_path)

            return ToolResult(
                success=True,
                output={
                    "new_cwd": abs_path,
                    "sandbox_path": sandbox_path,
                    "reason": reason,
                    "message": f"Working directory changed to: {abs_path}",
                },
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class BashTool(Tool):
    """Execute shell commands with safety constraints.

    Common commands:
    - ls: List directory contents
    - cd: Change directory (note: doesn't persist between calls)
    - python script.py: Run Python scripts
    - cat file: Display file contents
    - grep pattern file: Search for patterns
    - find . -name "*.py": Find files by pattern
    """

    name = "bash"
    description = "Execute shell commands"
    input_schema = {
        "command": str,  # Required: command to execute
        "timeout": (float, 30.0),  # Optional: timeout in seconds
        "cwd": (str, None),  # Optional: working directory
    }

    # Commands that are always blocked for safety
    BLOCKED_COMMANDS = {
        "rm -rf /",
        "rm -rf /*",
        "dd",
        "mkfs",
        "fdisk",
        ":(){:|:&};:",
        "chmod 777 /",  # Fork bombs, dangerous perms
    }

    # Patterns that indicate dangerous operations
    DANGEROUS_PATTERNS = [
        r"rm\s+-rf\s+/",  # rm -rf starting from root
        r">\s*/dev/sd",  # Writing directly to block devices
        r"mv\s+/\s+",  # Moving root
        r"chmod\s+-R\s+777\s+/",  # Recursive chmod 777 from root
    ]

    def __init__(
        self,
        allowed_dirs: list[str] | None = None,
        blocked_commands: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None
        self._extra_blocked = blocked_commands or set()

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to execute."""
        # Check explicitly blocked commands
        if command.strip() in self.BLOCKED_COMMANDS or command.strip() in self._extra_blocked:
            return False, "This command is explicitly blocked for safety"

        # Check dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False, "Command contains dangerous pattern"

        return True, ""

    def execute(self, **kwargs) -> ToolResult:
        try:
            if not _env_flag("MAXIM_ALLOW_BASH", False):
                return ToolResult(
                    success=False,
                    error="BashTool disabled. Set MAXIM_ALLOW_BASH=1 to enable.",
                )

            command = kwargs.get("command")
            if not command:
                return ToolResult(success=False, error="Missing required parameter: command")

            timeout = float(kwargs.get("timeout", self.timeout) or self.timeout)
            cwd = kwargs.get("cwd")

            # Validate working directory if specified
            if cwd:
                try:
                    cwd = sanitize_path(cwd)
                    cwd = os.path.realpath(cwd)
                except PathValidationError as e:
                    return ToolResult(success=False, error=f"Invalid working directory: {e}")

                if self._allowed_dirs:
                    in_allowed = False
                    for allowed in self._allowed_dirs:
                        if cwd.startswith(allowed + os.sep) or cwd == allowed:
                            in_allowed = True
                            break
                    if not in_allowed:
                        return ToolResult(
                            success=False,
                            error="Working directory must be within allowed directories",
                        )
            elif self._allowed_dirs:
                # No cwd specified — default to the first allowed_dir rather than
                # inheriting the Python process CWD (which may be the user's repo).
                # Closes the sandbox escape gap when sims run with allowed_dirs set
                # to the sandbox tmpdir. See docs/plans/docker_sandbox_plan.md.
                cwd = self._allowed_dirs[0]

            # Check command safety
            is_safe, reason = self._is_command_safe(command)
            if not is_safe:
                return ToolResult(success=False, error=reason)

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            return ToolResult(
                success=(result.returncode == 0),
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Command timed out after {timeout} seconds")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
