"""P4d: ``git_diff`` / ``run_tests`` are opt-in shell primitives.

``GitDiffTool`` built ``git`` argv from model-supplied refs with no
containment — ``--output=/path`` is a file write — and ``RunTestsTool``
ran ``command.split()`` with no gate at all. Both now follow the
``MAXIM_ALLOW_BASH`` mechanism from ``tools/filesystem.py``; ``git_diff``
additionally refuses option-shaped arguments and passes
``--end-of-options``; the fear gate classifies all three git/test tools as
shell execution. The env gates are scrubbed by
``tests/conftest.py::_isolate_maxim_tool_gate_env``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.runtime.fear_gate import _classify_action
from maxim.tools.base import ToolErrorKind
from maxim.tools.code_tools import RunTestsTool
from maxim.tools.git_tools import GitDiffTool


def _ok_result(stdout: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = 0
    result.stdout = stdout
    result.stderr = ""
    return result


class TestGitDiffGate:
    def test_disabled_by_default_and_never_spawns(self):
        with patch("maxim.tools.git_tools.subprocess.run") as mock_run:
            output = GitDiffTool().run(ref1="HEAD")
        assert output.success is False
        assert output.error == "GitDiffTool disabled. Set MAXIM_ALLOW_GIT_DIFF=1 to enable."
        assert output.error_kind == ToolErrorKind.PERMISSION_DENIED
        mock_run.assert_not_called()

    @pytest.mark.parametrize("raw", ["0", "false", "off", ""])
    def test_falsey_values_keep_it_disabled(self, monkeypatch, raw):
        monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", raw)
        with patch("maxim.tools.git_tools.subprocess.run") as mock_run:
            assert GitDiffTool().run(ref1="HEAD").success is False
        mock_run.assert_not_called()

    def test_enabled_passes_end_of_options_before_refs(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
        with patch("maxim.tools.git_tools.subprocess.run", return_value=_ok_result("diff")) as mock_run:
            output = GitDiffTool().run(ref1="main", ref2="feature", path="src/x.py")
        assert output.success is True
        assert mock_run.call_args[0][0] == ["git", "diff", "--end-of-options", "main", "feature", "--", "src/x.py"]


class TestGitDiffArgumentInjection:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ref1": "--output=/tmp/pwned"},
            {"ref1": "HEAD", "ref2": "--output=/tmp/pwned"},
            {"ref1": "HEAD", "path": "--output=/tmp/pwned"},
            {"ref1": "-p"},
        ],
    )
    def test_option_shaped_arguments_are_refused_before_spawn(self, monkeypatch, kwargs):
        monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
        with patch("maxim.tools.git_tools.subprocess.run") as mock_run:
            output = GitDiffTool().run(**kwargs)
        assert output.success is False
        assert output.error_kind == ToolErrorKind.INVALID_INPUT
        assert "must not start with '-'" in (output.error or "")
        mock_run.assert_not_called()

    def test_ordinary_refs_are_not_mistaken_for_options(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
        with patch("maxim.tools.git_tools.subprocess.run", return_value=_ok_result()):
            assert GitDiffTool().run(ref1="HEAD~1", ref2="feature-branch", path="a-b.py").success is True


class TestRunTestsGate:
    def test_disabled_by_default_and_never_spawns(self):
        with patch("maxim.tools.code_tools.subprocess.run") as mock_run:
            output = RunTestsTool().run(command="python -m pytest")
        assert output.success is False
        assert output.error == "RunTestsTool disabled. Set MAXIM_ALLOW_RUN_TESTS=1 to enable."
        assert output.error_kind == ToolErrorKind.PERMISSION_DENIED
        mock_run.assert_not_called()

    def test_enabled_runs_the_command(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ALLOW_RUN_TESTS", "1")
        with patch("maxim.tools.code_tools.subprocess.run", return_value=_ok_result("3 passed")) as mock_run:
            output = RunTestsTool().run(command="python -m pytest", test_path="tests/")
        assert output.success is True
        assert mock_run.call_args[0][0] == ["python", "-m", "pytest", "tests/"]


class TestFearGateClassifiesSubprocessTools:
    @pytest.mark.parametrize("tool", ["run_tests", "git_diff", "git_commit"])
    def test_subprocess_tools_are_shell_exec(self, tool):
        assert _classify_action(tool) == "shell_exec"

    def test_existing_classes_unchanged(self):
        assert _classify_action("bash") == "shell_exec"
        assert _classify_action("write_file") == "file_write"
        assert _classify_action("search_code") == "tool_call"
