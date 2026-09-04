"""Git version control tools.

``git_diff`` is opt-in via ``MAXIM_ALLOW_GIT_DIFF=1`` — the same mechanism
as ``MAXIM_ALLOW_BASH`` / ``MAXIM_ALLOW_EXECUTE_FILE`` in
``tools/filesystem.py``. Its argv is built from model-supplied strings with
no containment (``allowed_dirs``), and git options are file writes in
disguise (``--output=/path``), so a "read-only" tool was a write primitive.
Tests that set the flag rely on the autouse scrub
``tests/conftest.py::_isolate_maxim_tool_gate_env``.
"""

from __future__ import annotations

import subprocess

from maxim.tools.base import Tool, ToolErrorKind, ToolOutput
from maxim.utils.gpu_compat import env_flag as _env_flag


def _first_option_shaped(*values: str | None) -> str | None:
    """Return the first model-supplied argv element that starts with ``-``.

    Refs and paths reach ``git`` verbatim. A value such as
    ``--output=/etc/cron.d/x`` turns ``git diff`` into a file write outside
    any containment. ``--end-of-options`` is passed as well, but this reject
    is the guard that does not depend on the installed git's version.
    """
    for value in values:
        if value and value.startswith("-"):
            return value
    return None


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
        if not _env_flag("MAXIM_ALLOW_GIT_DIFF", False):
            return ToolOutput(
                success=False,
                error="GitDiffTool disabled. Set MAXIM_ALLOW_GIT_DIFF=1 to enable.",
                error_kind=ToolErrorKind.PERMISSION_DENIED,
            )

        ref1 = kwargs.get("ref1", "HEAD")
        ref2 = kwargs.get("ref2")
        path = kwargs.get("path")

        injected = _first_option_shaped(ref1, ref2, path)
        if injected is not None:
            return ToolOutput(
                success=False,
                error=f"git_diff refuses option-shaped argument {injected!r}: refs and paths must not start with '-'",
                error_kind=ToolErrorKind.INVALID_INPUT,
            )

        # ``--end-of-options`` (git >= 2.24, 2019) tells git that everything
        # after it is a revision/path, never an option — belt to the reject
        # above. Older git fails loudly on the unknown option rather than
        # silently running unguarded.
        cmd = ["git", "diff", "--end-of-options", ref1]
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

        return ToolOutput(success=True, output=result.stdout, metadata={"ref1": ref1, "ref2": ref2, "path": path})


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
                        return ToolOutput(
                            success=False,
                            error=f"git add failed for {f}: {result.stderr.strip()}",
                            error_kind=ToolErrorKind.EXTERNAL_FAILURE,
                        )

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

        return ToolOutput(success=True, output=result.stdout.strip(), metadata={"message": message, "dry_run": dry_run})
