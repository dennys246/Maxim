"""Git version control tools."""
from __future__ import annotations

import subprocess
from typing import Any

from maxim.tools.base import Tool, ToolOutput
from maxim.agents.bus import ToolErrorKind


class GitDiffTool(Tool):
    """Show git differences between commits or working tree."""

    name = "git_diff"
    description = "Show git differences between commits or working tree"
    input_schema = {
        "ref1": (str, "HEAD"),
        "ref2": (str, None),
        "path": (str, None),
    }

    def execute(self, **kwargs) -> ToolOutput:
        ref1 = kwargs.get("ref1", "HEAD")
        ref2 = kwargs.get("ref2")
        path = kwargs.get("path")

        cmd = ["git", "diff", ref1]
        if ref2:
            cmd.append(ref2)
        if path:
            cmd.extend(["--", path])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return ToolOutput(success=False, error="git diff timed out", error_kind=ToolErrorKind.TIMEOUT)
        except FileNotFoundError:
            return ToolOutput(success=False, error="git not found", error_kind=ToolErrorKind.FILE_NOT_FOUND)

        if result.returncode != 0:
            return ToolOutput(success=False, error=result.stderr.strip(), error_kind=ToolErrorKind.EXTERNAL_FAILURE)

        return ToolOutput(success=True, output=result.stdout,
                         metadata={"ref1": ref1, "ref2": ref2, "path": path})


class GitCommitTool(Tool):
    """Commit staged changes to git."""

    name = "git_commit"
    description = "Commit staged changes to git"
    input_schema = {
        "message": str,
        "files": (list, None),
        "dry_run": (bool, False),
    }

    def execute(self, **kwargs) -> ToolOutput:
        message = kwargs["message"]
        files = kwargs.get("files") or []
        dry_run = kwargs.get("dry_run", False)

        try:
            if files:
                for f in files:
                    result = subprocess.run(["git", "add", f], capture_output=True, text=True, timeout=10)
                    if result.returncode != 0:
                        return ToolOutput(success=False, error=f"git add failed for {f}: {result.stderr.strip()}",
                                        error_kind=ToolErrorKind.EXTERNAL_FAILURE)

            cmd = ["git", "commit", "-m", message]
            if dry_run:
                cmd.insert(2, "--dry-run")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return ToolOutput(success=False, error="git commit timed out", error_kind=ToolErrorKind.TIMEOUT)
        except FileNotFoundError:
            return ToolOutput(success=False, error="git not found", error_kind=ToolErrorKind.FILE_NOT_FOUND)

        if result.returncode != 0:
            return ToolOutput(success=False, error=result.stderr.strip(), error_kind=ToolErrorKind.EXTERNAL_FAILURE)

        return ToolOutput(success=True, output=result.stdout.strip(),
                         metadata={"message": message, "dry_run": dry_run})
