"""Integration layer for cross-system coordination.

This module provides the MemoryHub, which serves as the central coordinator
for all memory-subsystem bridges.
"""

from __future__ import annotations

from maxim.integration.memory_hub import MemoryHub, build_memory_hub

__all__ = ["MemoryHub", "build_memory_hub"]
