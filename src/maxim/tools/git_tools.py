"""Git version control tools.

``git_diff`` is opt-in via ``MAXIM_ALLOW_GIT_DIFF=1`` — the same mechanism
as ``MAXIM_ALLOW_BASH`` / ``MAXIM_ALLOW_EXECUTE_FILE`` in
``tools/filesystem.py``. Its argv is built from model-supplied strings with
no containment (``allowed_dirs``), and git options are file writes in
disguise (``--output=/path``), so a "read-only" tool was a write primitive.
Tests that set the flag rely on the autouse scrub
``tests/conftest.py::_isolate_maxim_tool_gate_env``.

``git_commit`` is opt-in too, via ``MAXIM_ALLOW_GIT_COMMIT=1`` (#949, owner decision 2026-09-28): it
was the one host coding tool with no gate.

Both tools are contained (#949): they run in ``tool_workdir(allowed_dirs)`` -- the project when it lies
inside the mode's ``allowed_dirs``, else ``allowed_dirs[0]`` -- with ``GIT_CEILING_DIRECTORIES`` so git
never searches above that root, model-supplied paths must resolve inside ``allowed_dirs``, the
environment is the host-tool allowlist, and every call first checks that git's resolved repository and
worktree are inside ``allowed_dirs`` (a model-written ``.git`` gitfile cannot redirect it). Hooks,
fsmonitor, external diff drivers, textconv and commit signing are switched off. NOT closed: a model that
can write the repository's ``.git/config`` can still make git run code through clean/smudge filters,
which no git switch disables -- that is why both tools are opt-in (#957 tracks refusing ``.git/`` writes).
"""

from __future__ import annotations

import os
import subprocess

from maxim.tools.base import (
    GIT_HARDENING_ARGS,
    Tool,
    ToolErrorKind,
    ToolOutput,
    contained_path,
    git_env,
    git_root_refusal,
    tool_workdir,
)
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

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

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
        workdir = tool_workdir(self._allowed_dirs)
        refusal = git_root_refusal(workdir, self._allowed_dirs)
        if refusal is not None:
            return ToolOutput(
                success=False, error=f"git_diff refused: {refusal}", error_kind=ToolErrorKind.PERMISSION_DENIED
            )
        cmd = ["git", *GIT_HARDENING_ARGS, "diff", "--no-ext-diff", "--no-textconv", "--end-of-options", ref1]
        if ref2:
            cmd.append(ref2)
        if path:
            contained = contained_path(path, self._allowed_dirs, base=workdir)
            if contained is None:
                return ToolOutput(
                    success=False,
                    error=f"git_diff path {path!r} is outside the allowed directories",
                    error_kind=ToolErrorKind.PERMISSION_DENIED,
                )
            cmd.extend(["--", contained])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=workdir,
                env=git_env(workdir),
            )
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

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

    def execute(self, **kwargs) -> ToolOutput:
        if not _env_flag("MAXIM_ALLOW_GIT_COMMIT", False):
            return ToolOutput(
                success=False,
                error="GitCommitTool disabled. Set MAXIM_ALLOW_GIT_COMMIT=1 to enable.",
                error_kind=ToolErrorKind.PERMISSION_DENIED,
            )

        message = kwargs["message"]
        files = kwargs.get("files") or []
        dry_run = kwargs.get("dry_run", False)
        workdir = tool_workdir(self._allowed_dirs)
        env = git_env(workdir)
        refusal = git_root_refusal(workdir, self._allowed_dirs)
        if refusal is not None:
            return ToolOutput(
                success=False, error=f"git_commit refused: {refusal}", error_kind=ToolErrorKind.PERMISSION_DENIED
            )

        injected = _first_option_shaped(*(str(f) for f in files))
        if injected is not None:
            return ToolOutput(
                success=False,
                error=f"git_commit refuses option-shaped file {injected!r}: paths must not start with '-'",
                error_kind=ToolErrorKind.INVALID_INPUT,
            )
        staged: list[str] = []
        for f in files:
            contained = contained_path(f, self._allowed_dirs, base=workdir)
            if contained is None:
                return ToolOutput(
                    success=False,
                    error=f"git_commit refuses file {f!r}: outside the allowed directories",
                    error_kind=ToolErrorKind.PERMISSION_DENIED,
                )
            staged.append(contained)

        try:
            for f in staged:
                result = subprocess.run(
                    ["git", *GIT_HARDENING_ARGS, "add", "--", f],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=workdir,
                    env=env,
                )
                if result.returncode != 0:
                    return ToolOutput(
                        success=False,
                        error=f"git add failed for {f}: {result.stderr.strip()}",
                        error_kind=ToolErrorKind.EXTERNAL_FAILURE,
                    )

            cmd = ["git", *GIT_HARDENING_ARGS, "commit", "--no-gpg-sign", "-m", message]
            if dry_run:
                cmd.append("--dry-run")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=workdir, env=env)
        except subprocess.TimeoutExpired:
            return ToolOutput(success=False, error="git commit timed out", error_kind=ToolErrorKind.TIMEOUT)
        except FileNotFoundError:
            return ToolOutput(success=False, error="git not found", error_kind=ToolErrorKind.FILE_NOT_FOUND)

        if result.returncode != 0:
            return ToolOutput(success=False, error=result.stderr.strip(), error_kind=ToolErrorKind.EXTERNAL_FAILURE)

        return ToolOutput(success=True, output=result.stdout.strip(), metadata={"message": message, "dry_run": dry_run})
