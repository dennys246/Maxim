"""``scripts/lint_child_env.py``: which spawns count as built on ``child_env`` (#1593)."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from scripts import lint_child_env as L

REPO = Path(__file__).resolve().parents[2]


def _hits(code: str) -> list[str]:
    return [spelling for _, spelling in L.unclean_spawns(textwrap.dedent(code))]


def test_an_inheriting_spawn_counts():
    assert _hits("import subprocess\nsubprocess.run(['python', '-V'])\n") == ["subprocess.run"]


def test_a_hand_copied_environment_counts():
    code = """
    import os, subprocess
    def f():
        subprocess.Popen(['x'], env={**os.environ, 'A': '1'})
        env = os.environ.copy()
        subprocess.run(['y'], env=env)
    """
    assert _hits(code) == ["subprocess.Popen", "subprocess.run"]


def test_child_env_direct_assigned_and_through_a_local_builder_are_clean():
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def _sub_env(extra):
        env = child_env()
        env.update(extra)
        return env
    def _sub_env_twice():
        return _sub_env({})
    def f():
        subprocess.run(['a'], env=child_env())
        env = child_env(A='1')
        subprocess.run(['b'], env=env)
        subprocess.run(['c'], env=_sub_env_twice())
        subprocess.run(['d'], env={**child_env(), 'B': '2'})
    """
    assert _hits(code) == []


def test_a_name_assigned_once_from_child_env_and_once_not_counts():
    code = """
    import os, subprocess
    from maxim.utils.child_env import child_env
    def f(flag):
        env = child_env()
        if flag:
            env = dict(os.environ)
        subprocess.run(['a'], env=env)
    """
    assert _hits(code) == ["subprocess.run"]


def test_reexec_without_an_environment_always_counts_and_execve_with_child_env_is_clean():
    code = """
    import os, sys
    from maxim.utils.child_env import child_env
    def restart():
        os.execv(sys.executable, sys.argv)
    def restart_clean():
        os.execve(sys.executable, sys.argv, child_env())
    def restart_dirty():
        os.execve(sys.executable, sys.argv, os.environ)
    """
    assert _hits(code) == ["os.execv", "os.execve"]


def test_asyncio_spawns_count():
    code = """
    import asyncio
    async def f():
        await asyncio.create_subprocess_exec('x')
    """
    assert _hits(code) == ["asyncio.create_subprocess_exec"]


def _calls(path: str, func: str) -> set[str]:
    tree = ast.parse((REPO / path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            return {
                n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            }
    raise AssertionError(f"{path}::{func} not found")


def test_every_sanctioned_builder_is_built_on_child_env():
    """SANCTIONED names cross-module builders the lint trusts; each must really reach child_env."""
    assert "child_env" in _calls("src/maxim/tools/base.py", "host_tool_env")
    assert "host_tool_env" in _calls("src/maxim/tools/base.py", "git_env")
    assert L.SANCTIONED == {"child_env", "host_tool_env", "git_env"}


def test_the_repo_scan_runs_and_skips_the_canonical_files():
    found = L.counts(REPO)
    assert found, "the scan found no spawns at all: the matcher is broken"
    assert not set(found) & L.CANONICAL
