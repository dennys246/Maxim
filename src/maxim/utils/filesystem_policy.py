"""Filesystem policy system for controlled access to data and sandbox directories.

Defines read/write/execute permissions per path pattern with instance isolation
and shared sandbox support for multi-Maxim coordination.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.utils.singleton import Singleton


class Permission(Flag):
    """File operation permissions."""

    NONE = 0
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()
    CREATE = auto()  # Create new files/directories
    DELETE = auto()  # Delete files/directories

    # Common combinations
    READ_ONLY = READ
    READ_WRITE = READ | WRITE | CREATE
    READ_WRITE_DELETE = READ | WRITE | CREATE | DELETE
    FULL = READ | WRITE | EXECUTE | CREATE | DELETE


@dataclass
class PathPolicy:
    """Policy for a specific path pattern."""

    pattern: str  # Glob pattern or exact path
    permissions: Permission
    description: str = ""
    instance_scoped: bool = False  # If True, {instance} placeholder is required
    recursive: bool = True  # Apply to subdirectories

    def matches(self, path: str, instance_id: str | None = None) -> bool:
        """Check if path matches this policy pattern."""
        resolved_pattern = self.pattern
        if self.instance_scoped:
            if instance_id is None:
                return False
            resolved_pattern = resolved_pattern.replace("{instance}", instance_id)

        # Normalize paths
        path = os.path.normpath(path)
        resolved_pattern = os.path.normpath(resolved_pattern)

        if self.recursive:
            # Match pattern or any parent directory
            if fnmatch.fnmatch(path, resolved_pattern):
                return True
            if fnmatch.fnmatch(path, f"{resolved_pattern}/*"):
                return True
            if fnmatch.fnmatch(path, f"{resolved_pattern}/**/*"):
                return True
            # Check if path starts with pattern (for directory matching)
            pattern_base = resolved_pattern.rstrip("*").rstrip("/")
            if path.startswith(pattern_base):
                return True
        else:
            if fnmatch.fnmatch(path, resolved_pattern):
                return True

        return False


@dataclass
class FilesystemPolicy:
    """Complete filesystem access policy for a Maxim instance."""

    instance_id: str
    base_data_dir: str = "data"
    sandbox_dir: str = "sandbox"
    policies: list[PathPolicy] = field(default_factory=list)

    # Computed paths
    _instance_data_dir: str = field(init=False, default="")
    _shared_dir: str = field(init=False, default="")
    _shared_outputs_dir: str = field(init=False, default="")

    def __post_init__(self) -> None:
        """Initialize computed paths and default policies."""
        self._instance_data_dir = os.path.join(self.base_data_dir, self.instance_id)
        self._shared_dir = os.path.join(self.base_data_dir, "shared")
        self._shared_outputs_dir = os.path.join(self._shared_dir, "outputs")

        if not self.policies:
            self.policies = self._default_policies()

    def _default_policies(self) -> list[PathPolicy]:
        """Create default policy set."""
        return [
            # Instance-specific data: full access
            PathPolicy(
                pattern=os.path.join(self.base_data_dir, "{instance}", "**"),
                permissions=Permission.READ_WRITE_DELETE,
                description="Instance-specific data directory",
                instance_scoped=True,
            ),
            # Other instances' data: no access
            PathPolicy(
                pattern=os.path.join(self.base_data_dir, "*", "**"),
                permissions=Permission.NONE,
                description="Other instances' data (blocked)",
                instance_scoped=False,
            ),
            # Shared models/config: read-only
            PathPolicy(
                pattern=os.path.join(self._shared_dir, "models", "**"),
                permissions=Permission.READ_ONLY,
                description="Shared models (read-only)",
            ),
            PathPolicy(
                pattern=os.path.join(self._shared_dir, "config", "**"),
                permissions=Permission.READ_ONLY,
                description="Shared config (read-only)",
            ),
            # Shared outputs: read all, write to own subfolder
            PathPolicy(
                pattern=os.path.join(self._shared_outputs_dir, "{instance}", "**"),
                permissions=Permission.READ_WRITE,
                description="Instance output folder in shared space",
                instance_scoped=True,
            ),
            PathPolicy(
                pattern=os.path.join(self._shared_outputs_dir, "**"),
                permissions=Permission.READ_ONLY,
                description="All shared outputs (read-only)",
            ),
            # Sandbox: read/write/execute
            PathPolicy(
                pattern=os.path.join(self.sandbox_dir, "**"),
                permissions=Permission.FULL,
                description="Sandbox directory (full access)",
            ),
            # Base data dir metadata: read-only
            PathPolicy(
                pattern=os.path.join(self.base_data_dir, "*.json"),
                permissions=Permission.READ_ONLY,
                description="Data directory metadata",
                recursive=False,
            ),
        ]

    def check_permission(
        self,
        path: str,
        required: Permission,
    ) -> tuple[bool, str]:
        """Check if operation is allowed on path.

        Args:
            path: Path to check
            required: Required permission(s)

        Returns:
            Tuple of (allowed, reason)
        """
        path = os.path.normpath(os.path.abspath(path))

        # Find matching policies (most specific first)
        matching_policies: list[PathPolicy] = []
        for policy in self.policies:
            if policy.matches(path, self.instance_id):
                matching_policies.append(policy)

        if not matching_policies:
            return False, f"No policy matches path: {path}"

        # Use most specific (longest pattern) policy
        # Instance-scoped policies take precedence
        matching_policies.sort(
            key=lambda p: (p.instance_scoped, len(p.pattern)),
            reverse=True,
        )
        policy = matching_policies[0]

        if required in policy.permissions:
            return True, policy.description
        else:
            missing = required & ~policy.permissions
            return False, f"Missing permission {missing.name} for {path} ({policy.description})"

    def can_read(self, path: str) -> bool:
        """Check if path can be read."""
        allowed, _ = self.check_permission(path, Permission.READ)
        return allowed

    def can_write(self, path: str) -> bool:
        """Check if path can be written."""
        allowed, _ = self.check_permission(path, Permission.WRITE)
        return allowed

    def can_execute(self, path: str) -> bool:
        """Check if path can be executed."""
        allowed, _ = self.check_permission(path, Permission.EXECUTE)
        return allowed

    def can_create(self, path: str) -> bool:
        """Check if file/directory can be created at path."""
        allowed, _ = self.check_permission(path, Permission.CREATE)
        return allowed

    def can_delete(self, path: str) -> bool:
        """Check if path can be deleted."""
        allowed, _ = self.check_permission(path, Permission.DELETE)
        return allowed

    @property
    def instance_data_dir(self) -> str:
        """Get instance-specific data directory."""
        return self._instance_data_dir

    @property
    def shared_dir(self) -> str:
        """Get shared directory."""
        return self._shared_dir

    @property
    def shared_outputs_dir(self) -> str:
        """Get shared outputs directory."""
        return self._shared_outputs_dir

    @property
    def instance_outputs_dir(self) -> str:
        """Get instance's output folder in shared space."""
        return os.path.join(self._shared_outputs_dir, self.instance_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize policy to dictionary."""
        return {
            "instance_id": self.instance_id,
            "base_data_dir": self.base_data_dir,
            "sandbox_dir": self.sandbox_dir,
            "policies": [
                {
                    "pattern": p.pattern,
                    "permissions": p.permissions.value,
                    "description": p.description,
                    "instance_scoped": p.instance_scoped,
                    "recursive": p.recursive,
                }
                for p in self.policies
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilesystemPolicy:
        """Deserialize policy from dictionary."""
        policies = [
            PathPolicy(
                pattern=p["pattern"],
                permissions=Permission(p["permissions"]),
                description=p.get("description", ""),
                instance_scoped=p.get("instance_scoped", False),
                recursive=p.get("recursive", True),
            )
            for p in data.get("policies", [])
        ]
        return cls(
            instance_id=data["instance_id"],
            base_data_dir=data.get("base_data_dir", "data"),
            sandbox_dir=data.get("sandbox_dir", "sandbox"),
            policies=policies if policies else [],  # Empty list triggers defaults
        )


# ─────────────────────────────────────────────────────────────────────────────
# Instance Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InstanceConfig:
    """Configuration for a Maxim instance in multi-instance setup."""

    instance_id: str
    display_name: str = ""
    base_data_dir: str = "data"
    sandbox_dir: str = "sandbox"

    # Directories this instance owns
    data_dir: str = field(init=False)
    outputs_dir: str = field(init=False)

    # Filesystem policy
    policy: FilesystemPolicy = field(init=False)

    def __post_init__(self) -> None:
        """Initialize computed fields."""
        if not self.display_name:
            self.display_name = self.instance_id

        self.data_dir = os.path.join(self.base_data_dir, self.instance_id)
        self.outputs_dir = os.path.join(self.base_data_dir, "shared", "outputs", self.instance_id)
        self.policy = FilesystemPolicy(
            instance_id=self.instance_id,
            base_data_dir=self.base_data_dir,
            sandbox_dir=self.sandbox_dir,
        )

    def ensure_directories(self) -> None:
        """Create all required directories for this instance."""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "audio"),
            os.path.join(self.data_dir, "videos"),
            os.path.join(self.data_dir, "memory"),
            os.path.join(self.data_dir, "logs"),
            os.path.join(self.data_dir, "transcript"),
            self.outputs_dir,
            self.sandbox_dir,
            os.path.join(self.sandbox_dir, "scripts"),
            os.path.join(self.sandbox_dir, "workspace"),
            os.path.join(self.sandbox_dir, "outputs"),
            os.path.join(self.base_data_dir, "shared", "models"),
            os.path.join(self.base_data_dir, "shared", "config"),
        ]
        for d in directories:
            os.makedirs(d, exist_ok=True)


def generate_instance_id(prefix: str = "maxim") -> str:
    """Generate a unique instance ID."""
    import uuid

    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_uuid}"


def get_instance_id_from_env(default_prefix: str = "maxim") -> str:
    """Get instance ID from environment or generate one."""
    instance_id = os.environ.get("MAXIM_INSTANCE_ID")
    if instance_id:
        return instance_id

    # Generate and set for consistency within process
    instance_id = generate_instance_id(default_prefix)
    os.environ["MAXIM_INSTANCE_ID"] = instance_id
    return instance_id


# ─────────────────────────────────────────────────────────────────────────────
# Global Policy Instance
# ─────────────────────────────────────────────────────────────────────────────


def _get_policy_singleton() -> "Singleton[FilesystemPolicy]":
    """Lazy import to avoid circular dependency."""
    from maxim.utils.singleton import Singleton

    return Singleton("filesystem_policy")


def get_filesystem_policy() -> FilesystemPolicy | None:
    """Get the global filesystem policy."""
    return _get_policy_singleton().get()


def set_filesystem_policy(policy: FilesystemPolicy) -> None:
    """Set the global filesystem policy."""
    _get_policy_singleton().set(policy)


def init_filesystem_policy(
    instance_id: str | None = None,
    base_data_dir: str = "data",
    sandbox_dir: str = "sandbox",
) -> FilesystemPolicy:
    """Initialize the global filesystem policy.

    Args:
        instance_id: Instance ID (generated if not provided)
        base_data_dir: Base data directory
        sandbox_dir: Sandbox directory

    Returns:
        The initialized policy
    """
    if instance_id is None:
        instance_id = get_instance_id_from_env()

    return _get_policy_singleton().init(
        FilesystemPolicy,
        instance_id=instance_id,
        base_data_dir=base_data_dir,
        sandbox_dir=sandbox_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode-Based Filesystem Containment
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_FOLDER_NAME = ".maxim_workspace"


class CwdAccessLevel:
    """Access levels for the current working directory."""

    NONE = "none"  # Cannot access CWD at all
    PROPOSE = "propose"  # Can propose plans to edit CWD (planning mode)
    SUPERVISED = "supervised"  # Can suggest edits, requires approval
    FULL = "full"  # Can freely edit within CWD


class SandboxExecutePolicy:
    """Policies for executing code in the sandbox."""

    BLOCKED = "blocked"  # Cannot execute at all
    APPROVAL_REQUIRED = "approval_required"  # Needs human/autonomy approval
    ALLOWED = "allowed"  # Can execute freely


@dataclass
class ModeFilesystemConfig:
    """Filesystem access configuration per operational mode.

    Attributes:
        mode: Operational mode name (passive, active, singularity)
        allowed_dirs: List of directories tools can access
        can_request_directory_change: Whether mode can request CWD change
        description: Human-readable description of access level
        workspace_write_always: Whether workspace is always writable (for logging/journaling)
        cwd_access: Level of access to CWD and subdirectories
        sandbox_execute: Policy for executing files in the sandbox
        accessible_folders: Additional folders beyond workspace/CWD that can be accessed
    """

    mode: str
    allowed_dirs: list[str]
    can_request_directory_change: bool = False
    description: str = ""
    # New graduated permission fields
    workspace_write_always: bool = True  # All modes can write to workspace for journaling
    cwd_access: str = CwdAccessLevel.NONE  # Access level for CWD
    sandbox_execute: str = SandboxExecutePolicy.BLOCKED  # Sandbox execution policy
    accessible_folders: list[str] = field(default_factory=list)  # Additional allowed folders


def get_workspace_path(cwd: str | None = None) -> str:
    """Get the workspace folder path within the current working directory.

    Args:
        cwd: Current working directory. Uses os.getcwd() if not provided.

    Returns:
        Absolute path to the workspace folder.
    """
    if cwd is None:
        cwd = os.getcwd()
    return os.path.join(os.path.realpath(cwd), WORKSPACE_FOLDER_NAME)


def ensure_workspace_exists(cwd: str | None = None) -> str:
    """Create the workspace folder and subdirectories if they don't exist.

    Structure:
        .maxim_workspace/
        ├── drafts/    - Proposed file edits, code drafts, work-in-progress
        ├── notes/     - Journaling, thinking notes, observations
        ├── plans/     - Structured plans for CWD modifications
        └── scratch/   - Temporary working files, ephemeral data

    Args:
        cwd: Current working directory. Uses os.getcwd() if not provided.

    Returns:
        Absolute path to the created/existing workspace folder.
    """
    workspace_path = get_workspace_path(cwd)
    for subdir in ("", "drafts", "notes", "plans", "scratch"):
        os.makedirs(os.path.join(workspace_path, subdir), exist_ok=True)
    return workspace_path




def get_mode_filesystem_config(
    mode: str,
    cwd: str | None = None,
    accessible_folders: list[str] | None = None,
) -> ModeFilesystemConfig:
    """Get filesystem configuration for an operational mode.

    Mode-based containment with graduated permissions:

    PLANNING (passive):
    - Workspace: Always writable (for logging/journaling)
    - Sandbox execute: Requires approval
    - CWD: Can propose plans to edit (no direct writes)
    - Can read CWD for context

    SUPERVISED (active):
    - Workspace: Always writable
    - Sandbox execute: Requires approval
    - CWD: Can suggest direct edits (requires approval)
    - Can request directory change

    AUTONOMOUS (singularity):
    - Workspace: Always writable
    - Sandbox execute: Allowed freely
    - CWD: Full read/write access
    - Full directory change capability

    Args:
        mode: Operational mode name (passive, active, singularity)
        cwd: Current working directory. Uses os.getcwd() if not provided.
        accessible_folders: Additional folders to grant access to.

    Returns:
        ModeFilesystemConfig with appropriate access settings.
    """
    if cwd is None:
        cwd = os.getcwd()
    cwd = os.path.realpath(cwd)
    workspace_path = get_workspace_path(cwd)
    extra_folders = accessible_folders or []

    if mode == "passive" or mode == "planning":
        # Planning mode: workspace is writable, CWD is read-only (propose plans)
        return ModeFilesystemConfig(
            mode="passive",
            allowed_dirs=[workspace_path, cwd] + extra_folders,  # Can read CWD
            can_request_directory_change=False,
            description=f"Workspace writable, CWD read-only (propose edits): {cwd}",
            workspace_write_always=True,
            cwd_access=CwdAccessLevel.PROPOSE,
            sandbox_execute=SandboxExecutePolicy.APPROVAL_REQUIRED,
            accessible_folders=extra_folders,
        )
    elif mode == "active" or mode == "supervised":
        # Supervised mode: workspace writable, CWD edits need approval
        return ModeFilesystemConfig(
            mode="active",
            allowed_dirs=[workspace_path, cwd] + extra_folders,
            can_request_directory_change=True,
            description=f"Workspace writable, CWD edits need approval: {cwd}",
            workspace_write_always=True,
            cwd_access=CwdAccessLevel.SUPERVISED,
            sandbox_execute=SandboxExecutePolicy.APPROVAL_REQUIRED,
            accessible_folders=extra_folders,
        )
    elif mode == "singularity" or mode == "autonomous":
        # Autonomous mode: full access within CWD and workspace
        return ModeFilesystemConfig(
            mode="singularity",
            allowed_dirs=[workspace_path, cwd] + extra_folders,  # Still bounded to CWD
            can_request_directory_change=True,
            description=f"Full access within CWD: {cwd}",
            workspace_write_always=True,
            cwd_access=CwdAccessLevel.FULL,
            sandbox_execute=SandboxExecutePolicy.ALLOWED,
            accessible_folders=extra_folders,
        )
    else:
        # Default to passive-like restrictions for unknown modes
        return ModeFilesystemConfig(
            mode=mode,
            allowed_dirs=[workspace_path],
            can_request_directory_change=False,
            description=f"Unknown mode '{mode}' - defaulting to workspace: {workspace_path}",
            workspace_write_always=True,
            cwd_access=CwdAccessLevel.NONE,
            sandbox_execute=SandboxExecutePolicy.BLOCKED,
            accessible_folders=[],
        )


# Global tracking of current working directory (can be changed in active mode)
_current_working_directory: str | None = None
_cwd_lock = __import__("threading").Lock()


def get_effective_cwd() -> str:
    """Get the effective current working directory.

    This may differ from os.getcwd() if active mode has requested a change.

    Returns:
        The effective working directory path.
    """
    global _current_working_directory
    with _cwd_lock:
        if _current_working_directory is None:
            _current_working_directory = os.getcwd()
        return _current_working_directory


def set_effective_cwd(path: str) -> bool:
    """Set the effective current working directory.

    Note: This does NOT call os.chdir(). It only updates the tracked directory
    used for containment boundaries.

    Args:
        path: New working directory path.

    Returns:
        True if successful, False if path doesn't exist.
    """
    global _current_working_directory
    path = os.path.realpath(path)
    if not os.path.isdir(path):
        return False

    with _cwd_lock:
        _current_working_directory = path
    return True


def reset_effective_cwd() -> None:
    """Reset effective CWD to actual os.getcwd()."""
    global _current_working_directory
    with _cwd_lock:
        _current_working_directory = os.getcwd()


# ─────────────────────────────────────────────────────────────────────────────
# Accessible Folders Management
# ─────────────────────────────────────────────────────────────────────────────

_accessible_folders: list[str] = []
_accessible_folders_lock = __import__("threading").Lock()


def get_accessible_folders() -> list[str]:
    """Get the list of additional accessible folders."""
    with _accessible_folders_lock:
        return list(_accessible_folders)


def add_accessible_folder(folder: str) -> bool:
    """Add a folder to the accessible folders list.

    Args:
        folder: Path to the folder to add.

    Returns:
        True if added successfully, False if folder doesn't exist.
    """
    global _accessible_folders
    folder = os.path.realpath(folder)
    if not os.path.isdir(folder):
        return False

    with _accessible_folders_lock:
        if folder not in _accessible_folders:
            _accessible_folders.append(folder)
    return True


def remove_accessible_folder(folder: str) -> bool:
    """Remove a folder from the accessible folders list.

    Args:
        folder: Path to the folder to remove.

    Returns:
        True if removed, False if not found.
    """
    global _accessible_folders
    folder = os.path.realpath(folder)

    with _accessible_folders_lock:
        if folder in _accessible_folders:
            _accessible_folders.remove(folder)
            return True
    return False


def clear_accessible_folders() -> None:
    """Clear all additional accessible folders."""
    global _accessible_folders
    with _accessible_folders_lock:
        _accessible_folders = []


def set_accessible_folders(folders: list[str]) -> None:
    """Set the accessible folders list (replaces existing).

    Args:
        folders: List of folder paths.
    """
    global _accessible_folders
    valid_folders = []
    for folder in folders:
        real_folder = os.path.realpath(folder)
        if os.path.isdir(real_folder):
            valid_folders.append(real_folder)

    with _accessible_folders_lock:
        _accessible_folders = valid_folders


# ─────────────────────────────────────────────────────────────────────────────
# Permission Checking Helpers
# ─────────────────────────────────────────────────────────────────────────────


def is_path_in_workspace(path: str, cwd: str | None = None) -> bool:
    """Check if a path is within the workspace directory.

    Args:
        path: Path to check.
        cwd: Current working directory. Uses effective CWD if not provided.

    Returns:
        True if path is within workspace.
    """
    if cwd is None:
        cwd = get_effective_cwd()
    workspace = get_workspace_path(cwd)
    real_path = os.path.realpath(path)
    return real_path.startswith(workspace + os.sep) or real_path == workspace




def is_path_in_cwd(path: str, cwd: str | None = None) -> bool:
    """Check if a path is within the current working directory.

    Args:
        path: Path to check.
        cwd: Current working directory. Uses effective CWD if not provided.

    Returns:
        True if path is within CWD.
    """
    if cwd is None:
        cwd = get_effective_cwd()
    cwd = os.path.realpath(cwd)
    real_path = os.path.realpath(path)
    return real_path.startswith(cwd + os.sep) or real_path == cwd


def is_path_in_accessible_folders(path: str) -> bool:
    """Check if a path is within any of the accessible folders.

    Args:
        path: Path to check.

    Returns:
        True if path is within an accessible folder.
    """
    real_path = os.path.realpath(path)
    for folder in get_accessible_folders():
        if real_path.startswith(folder + os.sep) or real_path == folder:
            return True
    return False


@dataclass
class WritePermissionResult:
    """Result of a write permission check."""

    allowed: bool
    requires_approval: bool = False
    reason: str = ""
    is_sandbox: bool = False
    is_cwd: bool = False


def check_write_permission(
    path: str,
    config: ModeFilesystemConfig,
    cwd: str | None = None,
) -> WritePermissionResult:
    """Check write permission for a path based on mode config.

    Args:
        path: Path to check.
        config: Mode filesystem configuration.
        cwd: Current working directory. Uses effective CWD if not provided.

    Returns:
        WritePermissionResult with allowed status and approval requirements.
    """
    if cwd is None:
        cwd = get_effective_cwd()

    in_sandbox = is_path_in_workspace(path, cwd)
    in_cwd = is_path_in_cwd(path, cwd)
    in_accessible = is_path_in_accessible_folders(path)

    # Workspace is always writable if workspace_write_always is True
    if in_sandbox and config.workspace_write_always:
        return WritePermissionResult(
            allowed=True,
            requires_approval=False,
            reason="Workspace is always writable",
            is_sandbox=True,
        )

    # Check CWD access based on cwd_access level
    if in_cwd or in_accessible:
        if config.cwd_access == CwdAccessLevel.FULL:
            return WritePermissionResult(
                allowed=True,
                requires_approval=False,
                reason="Full CWD access in singularity mode",
                is_cwd=True,
            )
        elif config.cwd_access == CwdAccessLevel.SUPERVISED:
            return WritePermissionResult(
                allowed=True,
                requires_approval=True,
                reason="CWD edit requires approval in supervised mode",
                is_cwd=True,
            )
        elif config.cwd_access == CwdAccessLevel.PROPOSE:
            return WritePermissionResult(
                allowed=False,
                requires_approval=True,
                reason="Can only propose edits in planning mode (submit as proposal)",
                is_cwd=True,
            )
        else:  # NONE
            return WritePermissionResult(
                allowed=False,
                requires_approval=False,
                reason="No CWD access in current mode",
                is_cwd=True,
            )

    # Path is outside allowed areas
    return WritePermissionResult(
        allowed=False,
        requires_approval=False,
        reason="Path is outside sandbox, CWD, and accessible folders",
    )


@dataclass
class ExecutePermissionResult:
    """Result of an execute permission check."""

    allowed: bool
    requires_approval: bool = False
    reason: str = ""


def check_execute_permission(
    path: str,
    config: ModeFilesystemConfig,
    cwd: str | None = None,
) -> ExecutePermissionResult:
    """Check execute permission for a path based on mode config.

    Args:
        path: Path to check.
        config: Mode filesystem configuration.
        cwd: Current working directory. Uses effective CWD if not provided.

    Returns:
        ExecutePermissionResult with allowed status and approval requirements.
    """
    if cwd is None:
        cwd = get_effective_cwd()

    in_sandbox = is_path_in_workspace(path, cwd)

    # Sandbox execution policy
    if in_sandbox:
        if config.sandbox_execute == SandboxExecutePolicy.ALLOWED:
            return ExecutePermissionResult(
                allowed=True,
                requires_approval=False,
                reason="Sandbox execution allowed in singularity mode",
            )
        elif config.sandbox_execute == SandboxExecutePolicy.APPROVAL_REQUIRED:
            return ExecutePermissionResult(
                allowed=True,
                requires_approval=True,
                reason="Sandbox execution requires approval",
            )
        else:  # BLOCKED
            return ExecutePermissionResult(
                allowed=False,
                requires_approval=False,
                reason="Sandbox execution blocked in current mode",
            )

    # Execution outside sandbox
    return ExecutePermissionResult(
        allowed=False,
        requires_approval=False,
        reason="Execution only allowed within sandbox",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CWD Tree Scanner (for LLM prompt context)
# ─────────────────────────────────────────────────────────────────────────────

_SCAN_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".eggs",
        "dist",
        "build",
        ".maxim_workspace",
        ".idea",
        ".vscode",
        ".DS_Store",
        ".ipynb_checkpoints",
    }
)


def scan_cwd_tree(
    cwd: str | None = None,
    max_depth: int = 2,
    max_entries: int = 30,
) -> list[str]:
    """Scan CWD directory tree for LLM context.

    Returns a compact listing of files and directories up to *max_depth*,
    skipping noise directories (`.git`, `__pycache__`, etc.).

    Args:
        cwd: Directory to scan.  Uses effective CWD if ``None``.
        max_depth: Maximum depth to recurse (0 = top-level only).
        max_entries: Maximum number of entries to return.

    Returns:
        List of formatted entry strings, directories first, then files.
    """
    if cwd is None:
        cwd = get_effective_cwd()

    dir_entries: list[str] = []
    file_entries: list[str] = []

    try:
        for root, dirs, files in os.walk(cwd):
            # Depth check
            rel_root = os.path.relpath(root, cwd)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            if depth > max_depth:
                dirs.clear()
                continue

            # Prune noise directories (modifying dirs in-place skips them)
            dirs[:] = sorted(d for d in dirs if d not in _SCAN_SKIP_DIRS)

            # Record subdirectories
            for d in dirs:
                rel = os.path.relpath(os.path.join(root, d), cwd)
                dir_entries.append(f"  {rel}/")

            # Record files
            for f in sorted(files):
                if f.startswith("."):
                    continue  # skip hidden files
                fpath = os.path.join(root, f)
                rel = os.path.relpath(fpath, cwd)
                try:
                    size = os.path.getsize(fpath)
                    size_str = f"{size}B" if size < 1024 else f"{size // 1024}KB"
                    file_entries.append(f"  {rel} ({size_str})")
                except OSError:
                    file_entries.append(f"  {rel}")
    except OSError:
        return []

    # Directories first, then files — both sorted
    combined = sorted(dir_entries) + sorted(file_entries)
    return combined[:max_entries]
