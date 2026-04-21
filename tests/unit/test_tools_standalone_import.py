"""Canary test: maxim.tools.* must be importable without pulling the full agent stack.

The circular import chain tools → agents → default_network → runtime → tools
was broken by moving ToolErrorKind to tools/base.py.  This test catches
regressions by importing tool modules in a subprocess with a clean sys.modules.
"""

from __future__ import annotations

import subprocess
import sys


def test_tools_base_imports_standalone():
    """tools/base.py imports without triggering the agents → runtime chain."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from maxim.tools.base import Tool, ToolOutput, ToolErrorKind; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"


def test_tools_narrative_imports_standalone():
    """tools/narrative.py (ThinkTool) imports without circular error."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from maxim.tools.narrative import ThinkTool; print('OK')",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"


def test_tool_error_kind_identity_across_paths():
    """ToolErrorKind from tools.base and agents.bus must be the same object."""
    from maxim.agents.bus import ToolErrorKind as from_bus
    from maxim.tools.base import ToolErrorKind as from_tools

    assert from_tools is from_bus, "Re-export diverged — different class objects"
