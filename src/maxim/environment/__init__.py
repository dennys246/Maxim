"""Environment interfaces and implementations.

Environments expose observations of the world/state but should not perform side effects.
"""

from __future__ import annotations

from maxim.environment.base import Environment
from maxim.environment.filesystem_env import FileSystemEnv
from maxim.environment.internet_env import InternetEnv
from maxim.environment.reachy_env import ReachyEnv

__all__ = ["Environment", "FileSystemEnv", "InternetEnv", "ReachyEnv"]
