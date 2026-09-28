"""#949 -- the host coding tools run in a contained working root with an allowlisted environment.

`RunTestsTool` and the git tools ran in the PROCESS working directory, and every host coding tool
(`bash`, `execute_file`, `run_tests`, `git_diff`, `git_commit`) handed the full parent environment --
API keys included -- to a model-driven command. Owner decisions 2026-09-28: the working root is the
project when it lies inside the mode's `allowed_dirs`, else `allowed_dirs[0]`, with git capped at that
root; `git_commit` is opt-in like its siblings. These tests drive the real tools.
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

SECRET = "sk-test-949-do-not-leak"


@pytest.fixture
def secret_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("MAXIM_PEER_KEY", SECRET)
    monkeypatch.setenv("LC_ALL", "C")


def _git(path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True, text=True)


def _git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.invalid")
    _git(path, "config", "user.name", "t")
    _git(path, "config", "commit.gpgsign", "false")
    return path


def _project(tmp_path: Path) -> tuple[Path, list[str]]:
    """A project repo with a committed file, and the mode-style allowed_dirs [workspace, project]."""
    project = _git_repo(tmp_path / "project")
    (project / "a.txt").write_text("one\n")
    _git(project, "add", "a.txt")
    _git(project, "commit", "-qm", "init")
    workspace = project / ".maxim_workspace"
    workspace.mkdir()
    return project, [str(workspace), str(project)]


# ── the helpers ──────────────────────────────────────────────────────────────────────────────────


def test_host_tool_env_is_an_allowlist(secret_env, monkeypatch) -> None:
    from maxim.tools.base import host_tool_env

    monkeypatch.setenv("GIT_DIR", "/elsewhere/.git")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    env = host_tool_env()
    for dropped in ("ANTHROPIC_API_KEY", "MAXIM_PEER_KEY", "GIT_DIR", "SSH_AUTH_SOCK"):
        assert dropped not in env, dropped
    assert SECRET not in env.values()
    assert env.get("LC_ALL") == "C" and "PATH" in env


def test_the_working_root_is_the_project_when_it_is_allowed(tmp_path, monkeypatch) -> None:
    from maxim.tools.base import tool_workdir

    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    assert tool_workdir(allowed) == os.path.realpath(project)  # not the .maxim_workspace in allowed[0]
    monkeypatch.chdir(tmp_path)  # the process CWD is outside every allowed dir (a sim / console root)
    assert tool_workdir(allowed) == os.path.realpath(allowed[0])
    assert tool_workdir(None) is None


def test_contained_path_refuses_escapes(tmp_path) -> None:
    from maxim.tools.base import contained_path

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "root2").mkdir()
    (root / "link").symlink_to(tmp_path)  # a symlink that leaves the root
    allowed = [str(root)]
    assert contained_path("x.txt", allowed, base=str(root)) == os.path.realpath(root / "x.txt")
    for escape in ("../root2/x", str(tmp_path / "root2" / "x"), "link/elsewhere", "~/x", "", "   ", None, 7):
        assert contained_path(escape, allowed, base=str(root)) is None, escape


# ── the tools ────────────────────────────────────────────────────────────────────────────────────


def test_bash_runs_in_its_root_without_the_parent_secrets(tmp_path, monkeypatch, secret_env) -> None:
    from maxim.tools.filesystem import BashTool

    monkeypatch.setenv("MAXIM_ALLOW_BASH", "1")
    out = BashTool(allowed_dirs=[str(tmp_path)]).execute(command="pwd; env")
    assert out.success, out.error
    stdout = out.output["stdout"]
    assert stdout.splitlines()[0] == os.path.realpath(tmp_path)
    assert SECRET not in stdout and "PATH=" in stdout


def test_execute_file_runs_without_the_parent_secrets(tmp_path, monkeypatch, secret_env) -> None:
    from maxim.tools.filesystem import ExecuteFileTool

    monkeypatch.setenv("MAXIM_ALLOW_EXECUTE_FILE", "1")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "s.sh").write_text("pwd\nenv\n")
    out = ExecuteFileTool(allowed_dirs=[str(tmp_path)]).execute(path="s.sh")
    assert out.success, out.error
    stdout = out.output["stdout"]
    assert stdout.splitlines()[0] == os.path.realpath(tmp_path)
    assert SECRET not in stdout


def test_run_tests_runs_in_the_project_without_the_parent_secrets(tmp_path, monkeypatch, secret_env) -> None:
    from maxim.tools.code_tools import RunTestsTool

    monkeypatch.setenv("MAXIM_ALLOW_RUN_TESTS", "1")
    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    where = RunTestsTool(allowed_dirs=allowed).execute(command="pwd")
    # the PROJECT, not the .maxim_workspace in allowed[0] (where pytest would collect nothing)
    assert os.path.realpath(project) in str(where.output) + str(where.metadata)
    assert ".maxim_workspace" not in str(where.output) + str(where.metadata)
    env = RunTestsTool(allowed_dirs=allowed).execute(command="env")
    assert SECRET not in str(env.output) + str(env.metadata)


def test_run_tests_refuses_a_test_path_outside_its_root(tmp_path, monkeypatch) -> None:
    from maxim.tools.base import ToolErrorKind
    from maxim.tools.code_tools import RunTestsTool

    monkeypatch.setenv("MAXIM_ALLOW_RUN_TESTS", "1")
    (tmp_path / "root").mkdir()
    out = RunTestsTool(allowed_dirs=[str(tmp_path / "root")]).execute(command="true", test_path="../elsewhere")
    assert out.success is False and out.error_kind == ToolErrorKind.PERMISSION_DENIED


def test_git_diff_reads_the_project_by_repo_relative_path(tmp_path, monkeypatch, secret_env) -> None:
    """The active-mode regression both reviewers found: with allowed_dirs[0] as the root, a
    repo-relative path rebased into .maxim_workspace and returned success with an EMPTY diff."""
    from maxim.tools.git_tools import GitDiffTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    (project / "a.txt").write_text("two\n")
    out = GitDiffTool(allowed_dirs=allowed).execute(ref1="HEAD", path="a.txt")
    assert out.success, out.error
    assert "+two" in out.output


def test_git_diff_ignores_a_parent_git_dir(tmp_path, monkeypatch) -> None:
    """GIT_DIR in the parent environment would point git at ANOTHER repo; the scrub drops it."""
    from maxim.tools.git_tools import GitDiffTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
    project, allowed = _project(tmp_path)
    other = _git_repo(tmp_path / "other")
    monkeypatch.chdir(project)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))  # an empty repo: HEAD does not resolve there
    (project / "a.txt").write_text("two\n")
    out = GitDiffTool(allowed_dirs=allowed).execute(ref1="HEAD", path="a.txt")
    assert out.success, out.error
    assert "+two" in out.output


def test_git_never_searches_above_its_root(tmp_path, monkeypatch) -> None:
    """A root that is not itself a repo, nested inside one (a workspace inside the host repo, a sim
    root under an unrelated checkout), fails loudly instead of reaching the enclosing repo."""
    from maxim.tools.git_tools import GitCommitTool, GitDiffTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
    monkeypatch.setenv("MAXIM_ALLOW_GIT_COMMIT", "1")
    host = _git_repo(tmp_path / "host")
    (host / "base.txt").write_text("committed\n")
    _git(host, "add", "base.txt")
    _git(host, "commit", "-qm", "host base")  # HEAD exists, so a diff that reached the host would succeed
    (host / "staged.txt").write_text("the user's work\n")
    _git(host, "add", "staged.txt")  # something staged in the HOST repo
    nested = host / "nested_root"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)  # outside the allowed dir, so the root is allowed[0] = nested
    diff = GitDiffTool(allowed_dirs=[str(nested)]).execute(ref1="HEAD")
    assert diff.success is False
    commit = GitCommitTool(allowed_dirs=[str(nested)]).execute(message="should not happen")
    assert commit.success is False
    log = subprocess.run(["git", "-C", str(host), "log", "--oneline"], capture_output=True, text=True).stdout
    assert "should not happen" not in log


def test_git_commit_is_opt_in_contained_and_refuses_option_shaped_files(tmp_path, monkeypatch) -> None:
    from maxim.tools.base import ToolErrorKind
    from maxim.tools.git_tools import GitCommitTool

    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    (project / "b.txt").write_text("b\n")
    tool = GitCommitTool(allowed_dirs=allowed)
    disabled = tool.execute(message="m", files=["b.txt"])
    assert disabled.success is False and disabled.error_kind == ToolErrorKind.PERMISSION_DENIED

    monkeypatch.setenv("MAXIM_ALLOW_GIT_COMMIT", "1")
    option = tool.execute(message="m", files=["--exec=touch /tmp/x"])
    assert option.success is False and option.error_kind == ToolErrorKind.INVALID_INPUT
    outside = tool.execute(message="m", files=["../outside.txt"])
    assert outside.success is False and outside.error_kind == ToolErrorKind.PERMISSION_DENIED
    ok = tool.execute(message="add b", files=["b.txt"])
    assert ok.success, ok.error
    assert "add b" in _git(project, "log", "--oneline").stdout


def test_git_commit_never_runs_repository_hooks(tmp_path, monkeypatch) -> None:
    """A model that can write .git/hooks/* (active mode may write the project) must not turn
    git_commit into code execution."""
    from maxim.tools.git_tools import GitCommitTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_COMMIT", "1")
    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    marker = tmp_path / "hook_ran"
    hook = project / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    (project / "c.txt").write_text("c\n")
    out = GitCommitTool(allowed_dirs=allowed).execute(message="with hook", files=["c.txt"])
    assert out.success, out.error
    assert not marker.exists()


def test_the_registry_hands_the_containment_root_to_every_host_coding_tool(tmp_path) -> None:
    from maxim.runtime.bootstrap import build_tool_registry

    registry = build_tool_registry(operational_mode="active", allowed_dirs_override=[str(tmp_path)])
    root = os.path.realpath(tmp_path)
    for name in ("bash", "execute_file", "run_tests", "git_diff", "git_commit"):
        assert registry.get(name)._allowed_dirs == [root], name


def test_every_subprocess_in_tools_passes_an_explicit_env_and_cwd() -> None:
    """Structural guard (#949): a host tool that spawns a process without `env=` inherits every secret
    in the parent environment, and without `cwd=` runs wherever the process happens to be."""
    import maxim.tools

    tools_dir = Path(maxim.tools.__file__).parent
    offenders = []
    for path in sorted(tools_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "subprocess"
                and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}
            ):
                keywords = {kw.arg for kw in node.keywords}
                if not {"env", "cwd"} <= keywords:
                    offenders.append(f"{path.name}:{node.lineno} missing {sorted({'env', 'cwd'} - keywords)}")
    assert offenders == [], offenders


def test_a_gitfile_cannot_redirect_git_out_of_its_root(tmp_path, monkeypatch) -> None:
    """The ceiling limits the UPWARD search; a model-written `.git` FILE (`gitdir: ...`) points git
    anywhere. The resolved repository must be inside the allowed dirs, or git never runs."""
    from maxim.tools.base import ToolErrorKind
    from maxim.tools.git_tools import GitCommitTool, GitDiffTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
    monkeypatch.setenv("MAXIM_ALLOW_GIT_COMMIT", "1")
    host = _git_repo(tmp_path / "host")
    (host / "base.txt").write_text("committed\n")
    _git(host, "add", "base.txt")
    _git(host, "commit", "-qm", "host base")
    (host / "staged.txt").write_text("the user's work\n")
    _git(host, "add", "staged.txt")
    root = tmp_path / "sim_root"
    root.mkdir()
    (root / ".git").write_text(f"gitdir: {host / '.git'}\n")  # the redirect a model could write
    (root / "evil.txt").write_text("x\n")
    monkeypatch.chdir(tmp_path)
    diff = GitDiffTool(allowed_dirs=[str(root)]).execute(ref1="HEAD")
    assert diff.success is False and diff.error_kind == ToolErrorKind.PERMISSION_DENIED
    commit = GitCommitTool(allowed_dirs=[str(root)]).execute(message="escaped", files=["evil.txt"])
    assert commit.success is False and commit.error_kind == ToolErrorKind.PERMISSION_DENIED
    assert "escaped" not in _git(host, "log", "--oneline").stdout


def test_git_commit_never_runs_a_repo_local_gpg_program(tmp_path, monkeypatch) -> None:
    from maxim.tools.git_tools import GitCommitTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_COMMIT", "1")
    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    marker = tmp_path / "gpg_ran"
    script = tmp_path / "fake_gpg.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    script.chmod(0o755)
    _git(project, "config", "commit.gpgsign", "true")
    _git(project, "config", "gpg.program", str(script))
    (project / "d.txt").write_text("d\n")
    out = GitCommitTool(allowed_dirs=allowed).execute(message="unsigned", files=["d.txt"])
    assert out.success, out.error
    assert not marker.exists()


def test_git_diff_never_runs_an_external_diff_driver(tmp_path, monkeypatch) -> None:
    from maxim.tools.git_tools import GitDiffTool

    monkeypatch.setenv("MAXIM_ALLOW_GIT_DIFF", "1")
    project, allowed = _project(tmp_path)
    monkeypatch.chdir(project)
    marker = tmp_path / "driver_ran"
    script = tmp_path / "ext_diff.sh"
    script.write_text(f"#!/bin/sh\ntouch {marker}\n")
    script.chmod(0o755)
    _git(project, "config", "diff.external", str(script))
    (project / "a.txt").write_text("two\n")
    out = GitDiffTool(allowed_dirs=allowed).execute(ref1="HEAD", path="a.txt")
    assert out.success, out.error
    assert "+two" in out.output and not marker.exists()
