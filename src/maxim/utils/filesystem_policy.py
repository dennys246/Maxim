"""Filesystem policy system for controlled access to data and sandbox directories.

Defines read/write/execute permissions per path pattern with instance isolation
and shared sandbox support for multi-Maxim coordination.
"""

from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass, field
from enum import Flag, auto
from pathlib import Path
from typing import Any


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


_global_policy: FilesystemPolicy | None = None


def get_filesystem_policy() -> FilesystemPolicy | None:
    """Get the global filesystem policy."""
    return _global_policy


def set_filesystem_policy(policy: FilesystemPolicy) -> None:
    """Set the global filesystem policy."""
    global _global_policy
    _global_policy = policy


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

    policy = FilesystemPolicy(
        instance_id=instance_id,
        base_data_dir=base_data_dir,
        sandbox_dir=sandbox_dir,
    )
    set_filesystem_policy(policy)
    return policy
