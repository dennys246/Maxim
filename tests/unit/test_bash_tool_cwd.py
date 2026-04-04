"""Regression tests for BashTool CWD confinement.

When allowed_dirs is set but no cwd parameter is provided, BashTool
must default cwd to the first allowed_dir — NOT inherit the Maxim
process's CWD. This closes the sandbox escape gap where bash
commands in simulation would run in the user's repo directory.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from maxim.tools.filesystem import BashTool


@pytest.fixture
def allow_bash(monkeypatch):
    """Enable BashTool for tests."""
    monkeypatch.setenv("MAXIM_ALLOW_BASH", "1")


@pytest.fixture
def sandbox(tmp_path):
    """Create a sandbox-like allowed directory."""
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


def _result_output(result) -> dict:
    """Normalize result output (success returns dict, failure returns error str)."""
    assert result.success, f"Command failed: {result.error}"
    return result.output


class TestBashCwdDefaulting:
    """When allowed_dirs is set but cwd is unspecified, default to first allowed_dir."""

    def test_pwd_runs_in_sandbox_not_process_cwd(self, allow_bash, sandbox):
        tool = BashTool(allowed_dirs=[sandbox])
        result = tool.execute(command="pwd")
        out = _result_output(result)
        assert out["stdout"].strip() == os.path.realpath(sandbox)

    def test_pwd_differs_from_process_cwd(self, allow_bash, sandbox):
        """Sanity check: process CWD is not the sandbox."""
        assert os.path.realpath(os.getcwd()) != os.path.realpath(sandbox)
        tool = BashTool(allowed_dirs=[sandbox])
        result = tool.execute(command="pwd")
        out = _result_output(result)
        assert out["stdout"].strip() != os.path.realpath(os.getcwd())

    def test_file_creation_lands_in_sandbox(self, allow_bash, sandbox):
        """Creating a file without cwd lands in the sandbox."""
        tool = BashTool(allowed_dirs=[sandbox])
        result = tool.execute(command="touch created_here.txt")
        assert result.success
        assert os.path.isfile(os.path.join(sandbox, "created_here.txt"))

    def test_explicit_cwd_still_respected(self, allow_bash, sandbox, tmp_path):
        """If cwd is explicitly provided and allowed, use it."""
        sub = tmp_path / "sandbox" / "sub"
        sub.mkdir()
        tool = BashTool(allowed_dirs=[sandbox])
        result = tool.execute(command="pwd", cwd=str(sub))
        out = _result_output(result)
        assert out["stdout"].strip() == os.path.realpath(str(sub))

    def test_explicit_cwd_outside_allowed_rejected(self, allow_bash, sandbox, tmp_path):
        """Explicit cwd outside allowed_dirs is still rejected."""
        outside = tmp_path / "outside"
        outside.mkdir()
        tool = BashTool(allowed_dirs=[sandbox])
        result = tool.execute(command="pwd", cwd=str(outside))
        assert not result.success
        assert "allowed" in result.error.lower()

    def test_no_allowed_dirs_inherits_process_cwd(self, allow_bash):
        """Backward-compat: without allowed_dirs, behavior is unchanged."""
        tool = BashTool(allowed_dirs=None)
        result = tool.execute(command="pwd")
        out = _result_output(result)
        # Without allowed_dirs, cwd is None → subprocess uses Python's CWD
        assert out["stdout"].strip() == os.path.realpath(os.getcwd())
