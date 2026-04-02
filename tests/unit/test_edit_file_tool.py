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


class TestEditFileToolContextDisambiguation:
    """Tests for context_before / context_after disambiguation."""

    def _make_file(self, tmp_path: Path, content: str) -> tuple[EditFileTool, Path]:
        target = tmp_path / "ambiguous.py"
        target.write_text(content, encoding="utf-8")
        return EditFileTool(allowed_dirs=[str(tmp_path)]), target

    def test_context_before_disambiguates(self, tmp_path: Path) -> None:
        """context_before selects the correct occurrence."""
        content = "def foo():\n    return None\n\ndef bar():\n    return None\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 42",
            context_before="def bar():\n    ",
        )

        assert result.success is True
        new_content = target.read_text(encoding="utf-8")
        assert new_content.count("return 42") == 1
        assert "def bar():\n    return 42" in new_content
        # foo's return None should be unchanged
        assert "def foo():\n    return None" in new_content

    def test_context_after_disambiguates(self, tmp_path: Path) -> None:
        """context_after selects the correct occurrence."""
        content = "return None\nx = 1\nreturn None\ny = 2\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            context_after="\ny = 2",
        )

        assert result.success is True
        new_content = target.read_text(encoding="utf-8")
        assert new_content.count("return 0") == 1
        assert new_content.count("return None") == 1  # First one unchanged

    def test_both_context_before_and_after(self, tmp_path: Path) -> None:
        """Both context_before and context_after for precise targeting."""
        content = "A\nreturn None\nB\nC\nreturn None\nD\nE\nreturn None\nF\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 99",
            context_before="C\n",
            context_after="\nD",
        )

        assert result.success is True
        new_content = target.read_text(encoding="utf-8")
        assert new_content.count("return 99") == 1
        assert new_content.count("return None") == 2  # Other two unchanged

    def test_context_no_match(self, tmp_path: Path) -> None:
        """Context pattern doesn't match but old_text exists."""
        content = "def foo():\n    return None\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            context_before="def bar():\n    ",  # Wrong context
        )

        assert result.success is False
        assert "not with the given context" in result.error

    def test_context_old_text_not_found(self, tmp_path: Path) -> None:
        """old_text itself doesn't exist."""
        content = "x = 1\ny = 2\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            context_before="x = 1\n",
        )

        assert result.success is False
        assert "not found" in result.error

    def test_context_still_ambiguous(self, tmp_path: Path) -> None:
        """Context doesn't uniquely identify — still multiple matches."""
        content = "if True:\n    return None\nif True:\n    return None\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
            context_before="if True:\n    ",
        )

        assert result.success is False
        assert "still matches" in result.error

    def test_multiple_match_suggests_context(self, tmp_path: Path) -> None:
        """When multiple matches found, error suggests context_before."""
        content = "def foo():\n    return None\ndef bar():\n    return None\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 0",
        )

        assert result.success is False
        assert "context_before" in result.error

    def test_context_preserves_surrounding_text(self, tmp_path: Path) -> None:
        """Only old_text is replaced; context_before/after are preserved."""
        content = "BEFORE_return None_AFTER\n"
        tool, target = self._make_file(tmp_path, content)

        result = tool.execute(
            path=str(target),
            old_text="return None",
            new_text="return 42",
            context_before="BEFORE_",
            context_after="_AFTER",
        )

        assert result.success is True
        assert target.read_text(encoding="utf-8") == "BEFORE_return 42_AFTER\n"
