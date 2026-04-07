"""Tests for GitDiffTool and GitCommitTool (Phase 3 #11d)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from maxim.agents.bus import ToolErrorKind
from maxim.tools.git_tools import GitCommitTool, GitDiffTool


# ---------------------------------------------------------------------------
# GitDiffTool
# ---------------------------------------------------------------------------


class TestGitDiffToolSuccess:
    """Successful git diff returns output with diff content."""

    def test_success(self) -> None:
        tool = GitDiffTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "diff --git a/foo.py b/foo.py\n+new line"
        mock_result.stderr = ""

        with patch("maxim.tools.git_tools.subprocess.run", return_value=mock_result):
            output = tool.run(ref1="HEAD")

        assert output.success is True
        assert "diff" in output.output


class TestGitDiffToolFailure:
    """Non-zero exit returns error with EXTERNAL_FAILURE."""

    def test_failure(self) -> None:
        tool = GitDiffTool()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "fatal: bad revision"

        with patch("maxim.tools.git_tools.subprocess.run", return_value=mock_result):
            output = tool.run(ref1="nonexistent")

        assert output.success is False
        assert output.error_kind == ToolErrorKind.EXTERNAL_FAILURE


# ---------------------------------------------------------------------------
# GitCommitTool
# ---------------------------------------------------------------------------


class TestGitCommitToolSuccess:
    """Successful commit returns success=True."""

    def test_success(self) -> None:
        tool = GitCommitTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "[main abc1234] fix: typo"

        with patch("maxim.tools.git_tools.subprocess.run", return_value=mock_result):
            output = tool.run(message="fix: typo")

        assert output.success is True


class TestGitCommitToolDryRun:
    """dry_run=True should add --dry-run flag to command."""

    def test_dry_run_flag(self) -> None:
        tool = GitCommitTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "dry run"

        with patch("maxim.tools.git_tools.subprocess.run", return_value=mock_result) as mock_run:
            tool.run(message="test commit", dry_run=True)

        # Find the commit call (not a git add call)
        commit_call = None
        for c in mock_run.call_args_list:
            cmd = c[0][0]
            if "commit" in cmd:
                commit_call = cmd
                break

        assert commit_call is not None
        assert "--dry-run" in commit_call


class TestGitCommitToolWithFiles:
    """Providing files should trigger git add for each file."""

    def test_git_add_called_for_each_file(self) -> None:
        tool = GitCommitTool()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "committed"

        with patch("maxim.tools.git_tools.subprocess.run", return_value=mock_result) as mock_run:
            tool.run(message="add files", files=["a.py", "b.py"])

        # Check that git add was called for each file
        add_calls = [c for c in mock_run.call_args_list if c[0][0][:2] == ["git", "add"]]
        assert len(add_calls) == 2
        assert add_calls[0][0][0][2] == "a.py"
        assert add_calls[1][0][0][2] == "b.py"


class TestGitDiffInAlwaysAllowed:
    """git_diff should be in AutonomyController.ALWAYS_ALLOWED_TOOLS."""

    def test_git_diff_always_allowed(self) -> None:
        from maxim.agents.autonomy import AutonomyController

        assert "git_diff" in AutonomyController.ALWAYS_ALLOWED_TOOLS
