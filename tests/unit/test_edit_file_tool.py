"""Unit tests for EditFileTool (Phase 2, #11a)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maxim.agents.bus import ToolErrorKind
from maxim.tools.filesystem import EditFileTool


@pytest.fixture(autouse=True)
def _no_forbidden_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable FORBIDDEN_PATHS so tmp_path (under /private/var on macOS) works."""
    monkeypatch.setattr(EditFileTool, "FORBIDDEN_PATHS", ())
    monkeypatch.setattr(EditFileTool, "FORBIDDEN_EXTENSIONS", ())


class TestEditFileTool:
    """Tests for EditFileTool."""

    def test_successful_edit(self, tmp_path: Path) -> None:
        """Write a file, edit it, verify the change."""
        target = tmp_path / "hello.py"
        target.write_text("print('hello')\n", encoding="utf-8")

        tool = EditFileTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(
            path=str(target),
            old_text="hello",
            new_text="world",
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "print('world')\n"

    def test_old_text_not_found(self, tmp_path: Path) -> None:
        """old_text missing returns INVALID_INPUT with near-match hints."""
        target = tmp_path / "code.py"
        target.write_text("x = 42\nreturn x\n", encoding="utf-8")

        tool = EditFileTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(
            path=str(target),
            old_text="return y",
            new_text="return z",
        )

        assert result.success is False
        assert result.error_kind is ToolErrorKind.INVALID_INPUT
        assert "not found" in result.error

    def test_multiple_occurrences_unexpected(self, tmp_path: Path) -> None:
        """3 occurrences when expected_count=1 returns VALIDATION with line numbers."""
        content = "return None\nx = 1\nreturn None\ny = 2\nreturn None\n"
        target = tmp_path / "multi.py"
        target.write_text(content, encoding="utf-8")

        tool = EditFileTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            expected_count=1,
        )

        assert result.success is False
        assert result.error_kind is ToolErrorKind.VALIDATION
        assert "3" in result.error  # found 3 occurrences
        # Line numbers should be mentioned
        assert "1" in result.error
        assert "5" in result.error

    def test_multiple_occurrences_correct_count(self, tmp_path: Path) -> None:
        """expected_count=3 matches actual count; all 3 replaced."""
        content = "return None\nx = 1\nreturn None\ny = 2\nreturn None\n"
        target = tmp_path / "multi.py"
        target.write_text(content, encoding="utf-8")

        tool = EditFileTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            expected_count=3,
        )

        assert result.success is True
        new_content = target.read_text(encoding="utf-8")
        assert new_content.count("return 0") == 3
        assert "return None" not in new_content

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Editing a nonexistent file returns FILE_NOT_FOUND."""
        tool = EditFileTool(allowed_dirs=[str(tmp_path)])
        result = tool.execute(
            path=str(tmp_path / "does_not_exist.py"),
            old_text="a",
            new_text="b",
        )

        assert result.success is False
        assert result.error_kind is ToolErrorKind.FILE_NOT_FOUND

    def test_allowed_dirs_enforcement(self, tmp_path: Path) -> None:
        """File outside allowed_dirs is blocked."""
        safe = tmp_path / "safe"
        safe.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "secret.txt"
        target.write_text("secret data\n", encoding="utf-8")

        tool = EditFileTool(allowed_dirs=[str(safe)])
        result = tool.execute(
            path=str(target),
            old_text="secret",
            new_text="public",
        )

        assert result.success is False
        assert "allowed directories" in result.error
        # File should be unchanged
        assert target.read_text(encoding="utf-8") == "secret data\n"
