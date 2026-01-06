from __future__ import annotations

from pathlib import Path
import os
import subprocess

from .base import Tool, ToolResult

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file from disk"
    input_schema = {
        "path": str,
        "tail_lines": (int, None),  # optional
    }

    def execute(self, path: str, tail_lines: int | None = None):
        if tail_lines is not None:
            n = int(tail_lines)
            if n <= 0:
                return ""
            return self._read_tail_lines(path, n)

        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _read_tail_lines(path: str | os.PathLike[str], n: int) -> str:
        block_size = 4096
        data = b""
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0 and data.count(b"\n") <= n:
                read_size = block_size if pos >= block_size else pos
                pos -= read_size
                f.seek(pos)
                data = f.read(read_size) + data
            lines = data.splitlines()[-n:]
            return b"\n".join(lines).decode("utf-8", errors="replace")
        

class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file on disk safely"
    input_schema = {
        "path": str,
        "content": str,
        "overwrite": (bool, False),  # optional, default False
    }

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = Path(kwargs["path"])
            content = kwargs["content"]
            overwrite = kwargs.get("overwrite", False)

            # Safety check
            if path.exists() and not overwrite:
                return ToolResult(
                    success=False,
                    error=f"File already exists: {path}"
                )

            # Ensure parent directories exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            path.write_text(content, encoding="utf-8")

            return ToolResult(
                success=True,
                output={"path": str(path), "size": len(content)}
            )

        except Exception as e:
            return ToolResult(success=False, error=str(e))


class ExecuteFileTool(Tool):
    name = "execute_file"
    description = "Execute a file using the appropriate interpreter based on file extension."
    input_schema = {
        "path": str,
        "timeout": (float, 10.0),  # optional seconds
    }

    EXT_INTERPRETER = {
        ".py": "python3",
        ".sh": "bash",
    }

    def execute(self, **kwargs) -> ToolResult:
        try:
            path = Path(kwargs["path"])
            timeout = kwargs.get("timeout", 10)

            if not path.exists():
                return ToolResult(success=False, error=f"File does not exist: {path}")

            ext = path.suffix
            if ext not in self.EXT_INTERPRETER:
                return ToolResult(success=False, error=f"Unsupported file type: {ext}")

            interpreter = self.EXT_INTERPRETER[ext]

            # Run in subprocess
            result = subprocess.run(
                [interpreter, str(path)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return ToolResult(
                success=(result.returncode == 0),
                output={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "path": str(path),
                }
            )

        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Execution timed out")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
