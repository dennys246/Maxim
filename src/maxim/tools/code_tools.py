"""Code-aware tools for searching and analyzing source code."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from maxim.tools.base import Tool, ToolErrorKind, ToolOutput
from maxim.utils.gpu_compat import env_flag as _env_flag


class CodeSearchTool(Tool):
    """Search file contents with regex support."""

    name = "search_code"
    description = "Search file contents with regex support"
    input_schema = {
        "pattern": str,
        "path": (str, "."),
        "file_pattern": (str, None),
        "max_results": (int, 20),
        "context_lines": (int, 3),
    }

    def __init__(self, allowed_dirs: list[str] | None = None) -> None:
        super().__init__()
        self._allowed_dirs = [os.path.realpath(d) for d in allowed_dirs] if allowed_dirs else None

    def execute(self, **kwargs) -> ToolOutput:
        pattern = kwargs["pattern"]
        search_path = kwargs.get("path", ".")
        file_pattern = kwargs.get("file_pattern")
        max_results = kwargs.get("max_results", 20)
        context_lines = kwargs.get("context_lines", 3)

        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolOutput(success=False, error=f"Invalid regex: {e}", error_kind=ToolErrorKind.INVALID_INPUT)

        search_root = Path(search_path).resolve()
        if not search_root.exists():
            return ToolOutput(
                success=False, error=f"Path not found: {search_path}", error_kind=ToolErrorKind.FILE_NOT_FOUND
            )

        # Validate against allowed_dirs
        if self._allowed_dirs:
            real_path = str(search_root)
            if not any(real_path.startswith(d + os.sep) or real_path == d for d in self._allowed_dirs):
                return ToolOutput(
                    success=False, error="Path outside allowed directories", error_kind=ToolErrorKind.PERMISSION_DENIED
                )

        results = []
        files_searched = 0

        # Walk directory tree
        if search_root.is_file():
            files_to_search = [search_root]
        else:
            glob_pattern = file_pattern or "*"
            files_to_search = sorted(search_root.rglob(glob_pattern))

        for file_path in files_to_search:
            if not file_path.is_file():
                continue
            # Skip binary files and hidden directories
            if any(part.startswith(".") for part in file_path.relative_to(search_root).parts):
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except (PermissionError, OSError):
                continue

            files_searched += 1
            lines = content.splitlines()

            for i, line in enumerate(lines):
                if regex.search(line):
                    # Extract context
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = []
                    for j in range(start, end):
                        prefix = ">>>" if j == i else "   "
                        context.append(f"{prefix} {j + 1}: {lines[j]}")

                    results.append(
                        {
                            "file": str(file_path.relative_to(search_root)),
                            "line": i + 1,
                            "match": line.strip(),
                            "context": "\n".join(context),
                        }
                    )

                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break

        return ToolOutput(
            success=True,
            output=f"Found {len(results)} matches in {files_searched} files",
            metadata={"results": results, "files_searched": files_searched},
        )


class RunTestsTool(Tool):
    """Run test suite and return structured results."""

    name = "run_tests"
    description = "Run test suite and return structured results"
    timeout = 300.0  # 5 minutes for test suites
    input_schema = {
        "command": (str, "python -m pytest"),
        "test_path": (str, None),
        "timeout": (int, 120),
    }

    def execute(self, **kwargs) -> ToolOutput:
        # Opt-in like BashTool: ``command`` is an arbitrary model-supplied
        # argv run with no allowed_dirs and no dangerous-pattern filter, so
        # "run the tests" was an ungated shell primitive.
        if not _env_flag("MAXIM_ALLOW_RUN_TESTS", False):
            return ToolOutput(
                success=False,
                error="RunTestsTool disabled. Set MAXIM_ALLOW_RUN_TESTS=1 to enable.",
                error_kind=ToolErrorKind.PERMISSION_DENIED,
            )

        command = kwargs.get("command", "python -m pytest")
        test_path = kwargs.get("test_path")
        timeout = kwargs.get("timeout", 120)

        cmd_parts = command.split()
        if test_path:
            cmd_parts.append(test_path)

        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolOutput(
                success=False, error=f"Tests timed out after {timeout}s", error_kind=ToolErrorKind.TIMEOUT
            )
        except FileNotFoundError:
            return ToolOutput(
                success=False, error=f"Command not found: {cmd_parts[0]}", error_kind=ToolErrorKind.FILE_NOT_FOUND
            )

        success = result.returncode == 0
        output_text = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        error_text = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr

        return ToolOutput(
            success=success,
            output=output_text,
            error=error_text if not success else None,
            error_kind=ToolErrorKind.EXTERNAL_FAILURE if not success else None,
            metadata={"returncode": result.returncode, "command": " ".join(cmd_parts)},
        )
