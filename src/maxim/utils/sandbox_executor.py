"""Sandbox executor for safe script execution with resource limits.

Supports Python scripts and shell scripts with:
- Resource limits (CPU time, memory, file size)
- Restricted imports for Python
- Isolated working directory
- Output capture and timeout handling
"""

from __future__ import annotations

import hashlib
import logging
import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for TOCTOU protection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of sandbox execution."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    PERMISSION_DENIED = "permission_denied"
    BLOCKED = "blocked"


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""

    # Time limits
    timeout_seconds: float = 30.0  # Wall-clock timeout
    cpu_time_seconds: int = 25  # CPU time limit

    # Memory limits
    memory_bytes: int = 256 * 1024 * 1024  # 256 MB
    stack_bytes: int = 8 * 1024 * 1024  # 8 MB stack

    # File limits
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    max_open_files: int = 64

    # Process limits
    max_processes: int = 4  # Prevent fork bombs

    def apply_to_subprocess(self) -> None:
        """Apply resource limits to current process (call in subprocess)."""
        try:
            # CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_time_seconds, self.cpu_time_seconds + 5))

            # Memory limit (address space)
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))

            # Stack size
            resource.setrlimit(resource.RLIMIT_STACK, (self.stack_bytes, self.stack_bytes))

            # File size limit
            resource.setrlimit(resource.RLIMIT_FSIZE, (self.max_file_size_bytes, self.max_file_size_bytes))

            # Open files limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.max_open_files, self.max_open_files))

            # Process limit (prevent fork bombs)
            resource.setrlimit(resource.RLIMIT_NPROC, (self.max_processes, self.max_processes))

        except (ValueError, resource.error) as e:
            # Some limits may not be available on all platforms
            logger.warning(f"Could not set some resource limits: {e}")


@dataclass
class ExecutionResult:
    """Result of sandbox execution."""

    status: ExecutionStatus
    stdout: str = ""
    stderr: str = ""
    return_code: int | None = None
    execution_time_seconds: float = 0.0
    error: str | None = None
    output_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "execution_time_seconds": self.execution_time_seconds,
            "error": self.error,
            "output_files": self.output_files,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Python Sandbox
# ─────────────────────────────────────────────────────────────────────────────


# Restricted imports for Python sandbox
ALLOWED_PYTHON_IMPORTS = {
    # Standard library - safe modules
    "abc",
    "array",
    "base64",
    "bisect",
    "calendar",
    "collections",
    "colorsys",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "decimal",
    "difflib",
    "enum",
    "fractions",
    "functools",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "io",
    "itertools",
    "json",
    "logging",
    "math",
    "numbers",
    "operator",
    "pathlib",
    "pprint",
    "random",
    "re",
    "statistics",
    "string",
    "struct",
    "textwrap",
    "time",
    "typing",
    "unicodedata",
    "unittest",
    "uuid",
    "xml",
    "zipfile",
    # Data science (if available)
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "PIL",
    "cv2",
    # Common utilities
    "yaml",
    "toml",
}

BLOCKED_PYTHON_IMPORTS = {
    # System/OS access
    "os",
    "sys",
    "subprocess",
    "shutil",
    "glob",
    "tempfile",
    "pty",
    "tty",
    "termios",
    "resource",
    "signal",
    "fcntl",
    "grp",
    "pwd",
    "sysconfig",
    "syslog",
    # Network access
    "socket",
    "ssl",
    "http",
    "urllib",
    "urllib2",
    "urllib3",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    # Code execution
    "importlib",
    "imp",
    "code",
    "codeop",
    "compile",
    "exec",
    "eval",
    "ast",
    "dis",
    "inspect",
    "types",
    "builtins",
    "__builtins__",
    # Multiprocessing
    "multiprocessing",
    "threading",
    "concurrent",
    "asyncio",
    # Pickle (code execution risk)
    "pickle",
    "cPickle",
    "shelve",
    "marshal",
    # ctypes (arbitrary memory access)
    "ctypes",
    "cffi",
    # Debugging
    "pdb",
    "trace",
    "traceback",
}


def _create_restricted_python_wrapper(script_path: str, sandbox_dir: str) -> str:
    """Create a wrapper script that restricts imports."""
    wrapper = f'''
import sys
import builtins

# Blocked imports
BLOCKED = {BLOCKED_PYTHON_IMPORTS!r}

_original_import = builtins.__import__

def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    base_name = name.split(".")[0]
    if base_name in BLOCKED:
        raise ImportError(f"Import of '{{name}}' is not allowed in sandbox")
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _restricted_import

# Change to sandbox directory
import os
os.chdir({sandbox_dir!r})

# Restrict file access to sandbox
_original_open = builtins.open

def _restricted_open(file, mode="r", *args, **kwargs):
    file_path = os.path.abspath(str(file))
    sandbox_abs = os.path.abspath({sandbox_dir!r})
    if not file_path.startswith(sandbox_abs):
        raise PermissionError(f"Cannot access files outside sandbox: {{file}}")
    return _original_open(file, mode, *args, **kwargs)

builtins.open = _restricted_open

# Now execute the actual script
exec(open({script_path!r}).read())
'''
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Shell Sandbox
# ─────────────────────────────────────────────────────────────────────────────


BLOCKED_SHELL_COMMANDS = {
    # System modification
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){ :|:& };:",  # Fork bomb
    # Network
    "curl",
    "wget",
    "nc",
    "netcat",
    "ssh",
    "scp",
    "rsync",
    "ftp",
    "telnet",
    # Privilege escalation
    "sudo",
    "su",
    "doas",
    "pkexec",
    # Sensitive data
    "/etc/passwd",
    "/etc/shadow",
    "~/.ssh",
    ".env",
    "credentials",
    "secret",
    "password",
    "api_key",
    "token",
}


def _check_shell_script_safety(script_content: str) -> tuple[bool, str | None]:
    """Check if shell script is safe to execute.

    Returns:
        Tuple of (is_safe, reason_if_blocked)
    """
    content_lower = script_content.lower()

    for blocked in BLOCKED_SHELL_COMMANDS:
        if blocked.lower() in content_lower:
            return False, f"Script contains blocked command/pattern: {blocked}"

    # Check for network commands
    if any(cmd in content_lower for cmd in ["curl", "wget", "nc ", "netcat"]):
        return False, "Network commands are not allowed in sandbox"

    # Check for privilege escalation
    if any(cmd in content_lower for cmd in ["sudo ", "su ", "doas "]):
        return False, "Privilege escalation is not allowed"

    # Check for directory traversal
    if "../" in script_content and "/etc" in content_lower:
        return False, "Directory traversal to system directories is not allowed"

    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# Sandbox Executor
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SandboxExecutor:
    """Executes scripts in a sandboxed environment with resource limits."""

    sandbox_dir: str
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    allowed_extensions: set[str] = field(default_factory=lambda: {".py", ".sh"})

    # Callbacks for approval flow
    # Signature: (script_path, content, content_hash) -> approved
    approval_callback: Callable[[str, str, str], bool] | None = None

    # Track approved scripts by path AND content hash to prevent TOCTOU attacks
    # Key: script_path, Value: content_hash
    _approved_hashes: dict[str, str] = field(default_factory=dict, repr=False)

    # Prefix for temporary wrapper scripts (for identification during cleanup)
    TEMP_WRAPPER_PREFIX: str = "_maxim_wrapper_"

    def __post_init__(self) -> None:
        """Ensure sandbox directory exists and clean up stale temp files."""
        os.makedirs(self.sandbox_dir, exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_dir, "scripts"), exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_dir, "workspace"), exist_ok=True)
        os.makedirs(os.path.join(self.sandbox_dir, "outputs"), exist_ok=True)

        # Clean up orphaned temp wrapper files from previous runs
        self._cleanup_stale_temp_files()

    def _cleanup_stale_temp_files(self) -> None:
        """Remove orphaned temporary wrapper files from previous runs.

        These files can accumulate if the process is killed before cleanup.
        Only removes files matching our wrapper prefix that are older than 1 hour.
        """
        try:
            now = time.time()
            max_age_seconds = 3600  # 1 hour

            for filename in os.listdir(self.sandbox_dir):
                if not filename.startswith(self.TEMP_WRAPPER_PREFIX):
                    continue
                if not filename.endswith(".py"):
                    continue

                filepath = os.path.join(self.sandbox_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                try:
                    file_age = now - os.path.getmtime(filepath)
                    if file_age > max_age_seconds:
                        os.unlink(filepath)
                        logger.debug(f"Cleaned up stale temp file: {filename}")
                except Exception as e:
                    logger.debug(f"Failed to clean up temp file {filename}: {e}")

        except Exception as e:
            logger.debug(f"Failed to clean up stale temp files: {e}")

    def execute(
        self,
        script_path: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        working_dir: str | None = None,
        require_approval: bool = True,
    ) -> ExecutionResult:
        """Execute a script in the sandbox.

        Args:
            script_path: Path to script (must be in sandbox)
            args: Command-line arguments
            env: Environment variables (merged with restricted set)
            working_dir: Working directory (default: sandbox/workspace)
            require_approval: Whether to require approval callback

        Returns:
            ExecutionResult with status, output, etc.
        """
        args = args or []
        working_dir = working_dir or os.path.join(self.sandbox_dir, "workspace")

        # Validate script path
        script_path = os.path.abspath(script_path)
        sandbox_abs = os.path.abspath(self.sandbox_dir)

        if not script_path.startswith(sandbox_abs):
            return ExecutionResult(
                status=ExecutionStatus.PERMISSION_DENIED,
                error=f"Script must be in sandbox directory: {self.sandbox_dir}",
            )

        if not os.path.exists(script_path):
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Script not found: {script_path}",
            )

        # Check extension
        ext = os.path.splitext(script_path)[1].lower()
        if ext not in self.allowed_extensions:
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                error=f"Script type not allowed: {ext}. Allowed: {self.allowed_extensions}",
            )

        # Read script content for approval
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Could not read script: {e}",
            )

        # Compute content hash for TOCTOU protection
        content_hash = compute_content_hash(script_content)

        # Check if this exact script version was previously approved
        if script_path in self._approved_hashes:
            if self._approved_hashes[script_path] == content_hash:
                # Same content as previously approved - skip re-approval
                require_approval = False
            else:
                # Content changed since last approval - require re-approval
                logger.warning(
                    f"Script content changed since last approval: {script_path} "
                    f"(old hash: {self._approved_hashes[script_path][:16]}..., "
                    f"new hash: {content_hash[:16]}...)"
                )
                require_approval = True

        # Request approval if callback provided
        if require_approval and self.approval_callback:
            try:
                approved = self.approval_callback(script_path, script_content, content_hash)
                if not approved:
                    return ExecutionResult(
                        status=ExecutionStatus.BLOCKED,
                        error="Script execution not approved",
                    )
                # Store the approved hash
                self._approved_hashes[script_path] = content_hash
            except Exception as e:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=f"Approval check failed: {e}",
                )

        # Re-verify content hash before execution (TOCTOU protection)
        # Re-read the file to catch any modifications between approval and execution
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                final_content = f.read()
            final_hash = compute_content_hash(final_content)
            if final_hash != content_hash:
                return ExecutionResult(
                    status=ExecutionStatus.BLOCKED,
                    error="Script was modified between approval and execution (TOCTOU attack detected)",
                )
            script_content = final_content
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Could not verify script content: {e}",
            )

        # Execute based on type
        if ext == ".py":
            return self._execute_python(script_path, script_content, args, env, working_dir)
        elif ext == ".sh":
            return self._execute_shell(script_path, script_content, args, env, working_dir)
        else:
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                error=f"Unknown script type: {ext}",
            )

    def _execute_python(
        self,
        script_path: str,
        script_content: str,
        args: list[str],
        env: dict[str, str] | None,
        working_dir: str,
    ) -> ExecutionResult:
        """Execute a Python script with restrictions."""
        start_time = time.time()

        # Create wrapper script
        wrapper_code = _create_restricted_python_wrapper(script_path, self.sandbox_dir)

        # Create temporary wrapper file with identifiable prefix for cleanup
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=self.TEMP_WRAPPER_PREFIX,
            suffix=".py",
            dir=self.sandbox_dir,
            delete=False,
        ) as wrapper_file:
            wrapper_file.write(wrapper_code)
            wrapper_path = wrapper_file.name

        try:
            # Build environment
            safe_env = self._build_safe_env(env)

            # Build command
            cmd = [sys.executable, wrapper_path] + args

            # Execute with resource limits
            result = self._run_subprocess(cmd, safe_env, working_dir)
            result.execution_time_seconds = time.time() - start_time

            # Collect output files
            result.output_files = self._collect_output_files()

            return result

        finally:
            # Clean up wrapper
            try:
                os.unlink(wrapper_path)
            except Exception:
                pass

    def _execute_shell(
        self,
        script_path: str,
        script_content: str,
        args: list[str],
        env: dict[str, str] | None,
        working_dir: str,
    ) -> ExecutionResult:
        """Execute a shell script with safety checks."""
        start_time = time.time()

        # Safety check
        is_safe, reason = _check_shell_script_safety(script_content)
        if not is_safe:
            return ExecutionResult(
                status=ExecutionStatus.BLOCKED,
                error=reason,
            )

        # Build environment
        safe_env = self._build_safe_env(env)

        # Restrict PATH to minimal set
        safe_env["PATH"] = "/usr/bin:/bin"

        # Build command
        cmd = ["/bin/bash", script_path] + args

        # Execute with resource limits
        result = self._run_subprocess(cmd, safe_env, working_dir)
        result.execution_time_seconds = time.time() - start_time

        # Collect output files
        result.output_files = self._collect_output_files()

        return result

    def _build_safe_env(self, extra_env: dict[str, str] | None) -> dict[str, str]:
        """Build a safe environment for subprocess."""
        safe_env = {
            "HOME": self.sandbox_dir,
            "TMPDIR": os.path.join(self.sandbox_dir, "workspace"),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            # Sandbox markers
            "MAXIM_SANDBOX": "1",
            "MAXIM_SANDBOX_DIR": self.sandbox_dir,
        }

        # Add allowed extra env vars (filter out dangerous ones)
        if extra_env:
            blocked_env = {"LD_PRELOAD", "LD_LIBRARY_PATH", "DYLD_", "PYTHONPATH"}
            for key, value in extra_env.items():
                if not any(key.startswith(b) for b in blocked_env):
                    safe_env[key] = value

        return safe_env

    def _run_subprocess(
        self,
        cmd: list[str],
        env: dict[str, str],
        working_dir: str,
    ) -> ExecutionResult:
        """Run subprocess with resource limits and timeout."""

        def set_limits() -> None:
            """Called in subprocess to set resource limits."""
            self.resource_limits.apply_to_subprocess()

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=working_dir,
                preexec_fn=set_limits,
            )

            try:
                stdout, stderr = process.communicate(timeout=self.resource_limits.timeout_seconds)
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILED,
                    stdout=stdout.decode("utf-8", errors="replace"),
                    stderr=stderr.decode("utf-8", errors="replace"),
                    return_code=process.returncode,
                )

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                    stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                    error=f"Execution timed out after {self.resource_limits.timeout_seconds}s",
                )

        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
            )

    def _collect_output_files(self) -> list[str]:
        """Collect list of files in sandbox outputs directory."""
        outputs_dir = os.path.join(self.sandbox_dir, "outputs")
        if not os.path.exists(outputs_dir):
            return []

        output_files = []
        for root, _, files in os.walk(outputs_dir):
            for f in files:
                output_files.append(os.path.join(root, f))
        return output_files

    def write_script(
        self,
        name: str,
        content: str,
        script_type: str = "python",
    ) -> str:
        """Write a script to the sandbox scripts directory.

        Args:
            name: Script name (without extension)
            content: Script content
            script_type: "python" or "shell"

        Returns:
            Path to created script
        """
        ext = ".py" if script_type == "python" else ".sh"
        script_path = os.path.join(self.sandbox_dir, "scripts", f"{name}{ext}")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Make shell scripts executable
        if script_type == "shell":
            os.chmod(script_path, 0o755)

        return script_path

    def list_scripts(self) -> list[dict[str, Any]]:
        """List scripts in sandbox."""
        scripts_dir = os.path.join(self.sandbox_dir, "scripts")
        if not os.path.exists(scripts_dir):
            return []

        scripts = []
        for f in os.listdir(scripts_dir):
            path = os.path.join(scripts_dir, f)
            if os.path.isfile(path):
                ext = os.path.splitext(f)[1]
                scripts.append({
                    "name": f,
                    "path": path,
                    "type": "python" if ext == ".py" else "shell" if ext == ".sh" else "unknown",
                    "size": os.path.getsize(path),
                    "modified": os.path.getmtime(path),
                })
        return scripts


# ─────────────────────────────────────────────────────────────────────────────
# Global Sandbox Instance
# ─────────────────────────────────────────────────────────────────────────────


def _get_executor_singleton():
    """Lazy import to avoid circular dependency."""
    from maxim.utils.singleton import Singleton
    return Singleton("sandbox_executor")


def get_sandbox_executor() -> SandboxExecutor | None:
    """Get the global sandbox executor."""
    return _get_executor_singleton().get()


def set_sandbox_executor(executor: SandboxExecutor) -> None:
    """Set the global sandbox executor."""
    _get_executor_singleton().set(executor)


def init_sandbox_executor(
    sandbox_dir: str = "sandbox",
    resource_limits: ResourceLimits | None = None,
) -> SandboxExecutor:
    """Initialize the global sandbox executor."""
    return _get_executor_singleton().init(
        SandboxExecutor,
        sandbox_dir=sandbox_dir,
        resource_limits=resource_limits or ResourceLimits(),
    )
