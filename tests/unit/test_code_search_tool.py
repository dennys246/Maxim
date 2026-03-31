"""Unit tests for CodeSearchTool (Phase 2, #11b)."""

from __future__ import annotations

from pathlib import Path

from maxim.agents.bus import ToolErrorKind
from maxim.tools.code_tools import CodeSearchTool


def _make_files(tmp_path: Path) -> Path:
    """Create a small project tree with searchable content."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "import os\ndef main():\n    print('hello')\n    return 0\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "def helper():\n    return 42\n\ndef another():\n    return 99\n",
        encoding="utf-8",
    )
    (src / "notes.txt").write_text(
        "This is a note.\nreturn value docs\n",
        encoding="utf-8",
    )
    return src


class TestCodeSearchTool:
    """Tests for CodeSearchTool."""

    def test_basic_regex_search(self, tmp_path: Path) -> None:
        """Find matches with correct line numbers."""
        src = _make_files(tmp_path)
        tool = CodeSearchTool()

        result = tool.execute(pattern=r"def \w+", path=str(src))

        assert result.success is True
        matches = result.metadata["results"]
        assert len(matches) >= 3  # main, helper, another
        # Verify line numbers are ints > 0
        for m in matches:
            assert isinstance(m["line"], int)
            assert m["line"] > 0

    def test_context_lines(self, tmp_path: Path) -> None:
        """context_lines=2 returns surrounding lines with >>> marker."""
        src = _make_files(tmp_path)
        tool = CodeSearchTool()

        result = tool.execute(
            pattern="return 42", path=str(src), context_lines=2,
        )

        assert result.success is True
        matches = result.metadata["results"]
        assert len(matches) >= 1
        ctx = matches[0]["context"]
        assert ">>>" in ctx  # match line marker
        assert "return 42" in ctx

    def test_file_pattern_filter(self, tmp_path: Path) -> None:
        """file_pattern='*.py' excludes .txt files."""
        src = _make_files(tmp_path)
        tool = CodeSearchTool()

        result = tool.execute(
            pattern="return", path=str(src), file_pattern="*.py",
        )

        assert result.success is True
        files_matched = {m["file"] for m in result.metadata["results"]}
        for f in files_matched:
            assert f.endswith(".py"), f"Non-.py file in results: {f}"

    def test_invalid_regex(self, tmp_path: Path) -> None:
        """Bad regex returns INVALID_INPUT error."""
        src = _make_files(tmp_path)
        tool = CodeSearchTool()

        result = tool.execute(pattern="[unclosed", path=str(src))

        assert result.success is False
        assert result.error_kind is ToolErrorKind.INVALID_INPUT
        assert "regex" in result.error.lower() or "Invalid" in result.error

    def test_no_matches(self, tmp_path: Path) -> None:
        """Pattern with no matches returns success with 0 results."""
        src = _make_files(tmp_path)
        tool = CodeSearchTool()

        result = tool.execute(
            pattern="zzz_nonexistent_pattern_zzz", path=str(src),
        )

        assert result.success is True
        assert len(result.metadata["results"]) == 0

    def test_max_results_limit(self, tmp_path: Path) -> None:
        """max_results caps the number of returned matches."""
        src = tmp_path / "many"
        src.mkdir()
        # Create a file with many matching lines
        lines = [f"item_{i} = value" for i in range(50)]
        (src / "big.py").write_text("\n".join(lines), encoding="utf-8")

        tool = CodeSearchTool()
        result = tool.execute(
            pattern="item_", path=str(src), max_results=3,
        )

        assert result.success is True
        assert len(result.metadata["results"]) == 3

    def test_path_not_found(self, tmp_path: Path) -> None:
        """Searching a nonexistent directory returns FILE_NOT_FOUND."""
        tool = CodeSearchTool()
        result = tool.execute(
            pattern="anything",
            path=str(tmp_path / "no_such_dir"),
        )

        assert result.success is False
        assert result.error_kind is ToolErrorKind.FILE_NOT_FOUND
