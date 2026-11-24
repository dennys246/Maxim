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


def test_a_self_rebuilding_name_does_not_recurse_and_is_decided_by_its_other_assignments():
    clean = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f(x):
        env = child_env()
        env = {**env, "A": "1"}
        env = env if x else child_env()
        for k in ("B",):
            env = {**env, k: "1"}
        subprocess.run(["a"], env=env)
    """
    assert _hits(clean) == []
    dirty = """
    import os, subprocess
    def f():
        env = dict(os.environ)
        env = {**env, "A": "1"}
        subprocess.run(["a"], env=env)
    """
    assert _hits(dirty) == ["subprocess.run"]


def test_update_with_os_environ_undoes_child_env():
    code = """
    import os, subprocess
    from maxim.utils.child_env import child_env
    def f():
        env = child_env()
        env.update(os.environ)
        subprocess.run(["a"], env=env)
    """
    assert _hits(code) == ["subprocess.run"]


def test_a_local_method_named_like_a_mapping_method_does_not_clean_os_environ():
    code = """
    import os, subprocess
    from maxim.utils.child_env import child_env
    class C:
        def copy(self):
            return child_env()
        def run(self):
            subprocess.run(["a"], env=os.environ.copy())
            subprocess.run(["b"], env=self.copy())
    """
    assert _hits(code) == ["subprocess.run"]  # only the os.environ.copy() one


def test_a_builder_must_return_a_clean_environment_on_every_path():
    code = """
    import os, subprocess
    from maxim.utils.child_env import child_env
    def half(x):
        if x:
            return child_env()
        return dict(os.environ)
    def f():
        subprocess.run(["a"], env=half(True))
    """
    assert _hits(code) == ["subprocess.run"]


def test_conditional_and_filtering_comprehension_rules():
    code = """
    import os, subprocess
    from maxim.utils.child_env import child_env
    def f(x):
        subprocess.run(["a"], env=child_env() if x else child_env(A="1"))
        subprocess.run(["b"], env=child_env() if x else os.environ)
        subprocess.run(["c"], env={k: v for k, v in child_env().items() if k != "X"})
        subprocess.run(["d"], env={k: v for k, v in os.environ.items() if k != "X"})
    """
    assert [ln for ln, _ in L.unclean_spawns(textwrap.dedent(code))] == [6, 8]  # the "b" and "d" calls


def test_getoutput_takes_no_environment_and_always_counts():
    code = """
    import subprocess
    subprocess.getoutput("ls")
    subprocess.getstatusoutput("ls")
    """
    assert _hits(code) == ["subprocess.getoutput", "subprocess.getstatusoutput"]


def test_posix_spawn_reads_its_environment_from_the_third_argument():
    code = """
    import os
    from maxim.utils.child_env import child_env
    os.posix_spawn("/bin/true", ["true"], child_env())
    os.posix_spawn("/bin/true", ["true"], os.environ)
    """
    assert _hits(code) == ["os.posix_spawn"]


def test_a_name_with_no_grounded_assignment_is_unclean():
    """A self-reference is clean only next to an assignment that is clean on its own (Executor delta 2)."""
    cases = {
        "param_rebuilt": "def f(env):\n    env = {**env, 'A': '1'}\n    subprocess.run(['a'], env=env)\n",
        "param_ifexp": "def f(env, x):\n    env = env if x else child_env()\n    subprocess.run(['a'], env=env)\n",
        "self_only": "def f():\n    env = {**env}\n    subprocess.run(['a'], env=env)\n",
        "mutual": "def f():\n    a = b\n    b = a\n    subprocess.run(['a'], env=a)\n",
        "global": "env = dict(os.environ)\ndef f():\n    global env\n    env = {**env}\n    subprocess.run(['a'], env=env)\n",
    }
    header = "import os, subprocess\nfrom maxim.utils.child_env import child_env\n"
    for name, body in cases.items():
        assert _hits(header + body) == ["subprocess.run"], name


def test_an_alias_back_to_a_grounded_name_is_clean():
    """Delta round 3: `base = env` while `env` is being resolved must not make `env` dirty."""
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f():
        env = child_env()
        base = env
        env = {**base, "A": "1"}
        subprocess.run(["a"], env=env)
    def g():
        a = child_env()
        b = a
        a = b
        subprocess.run(["b"], env=a)
    """
    assert _hits(code) == []


def test_a_lambda_parameter_is_its_own_scope():
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f():
        env = child_env()
        g = lambda env: subprocess.run(["a"], env=env)
        return g
    """
    assert _hits(code) == ["subprocess.run"]


def test_the_scan_gives_the_same_answer_for_identical_files_parsed_in_turn():
    """The per-scope cache is keyed by id(): a stale entry from an earlier file must never leak into a later one."""
    clean = "import subprocess\nfrom maxim.utils.child_env import child_env\ndef f():\n    env = child_env()\n    subprocess.run(['a'], env=env)\n"
    dirty = "import os, subprocess\ndef f():\n    env = dict(os.environ)\n    subprocess.run(['a'], env=env)\n"
    for _ in range(20):
        assert L.unclean_spawns(clean) == []
        assert len(L.unclean_spawns(dirty)) == 1


def test_a_lambda_closing_over_an_outer_child_env_is_clean():
    """local_server_spawner launches through `lambda: subprocess.Popen(..., env=env)` over an outer child_env."""
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f(spawn):
        env = child_env()
        return spawn(lambda: subprocess.Popen(["a"], env=env))
    """
    assert _hits(code) == []


def test_an_ungrounded_cycle_cannot_hold_up_a_dirty_assignment():
    """Delta round 4: a grounded name that may also take a value from an ungrounded cycle is unclean."""
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f(x):
        a = {**b}
        b = {**a}
        env = child_env()
        if x:
            env = a
        subprocess.run(["a"], env=env)
    """
    assert _hits(code) == ["subprocess.run"]


def test_a_parameter_reassigned_on_one_path_is_unclean():
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f(env, x):
        if x:
            env = child_env()
        subprocess.run(["a"], env=env)
    """
    assert _hits(code) == ["subprocess.run"]


def test_a_lambda_parameter_inside_a_larger_expression_is_checked_in_the_lambda():
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f():
        env = child_env()
        return lambda env: subprocess.run(["a"], env={**env, "A": "1"})
    """
    assert _hits(code) == ["subprocess.run"]


def test_names_bound_in_ways_the_lint_cannot_evaluate_are_unclean():
    """Delta round 5's pre-existing blind spots: each binding form fails closed."""
    header = "import os, subprocess\nfrom maxim.utils.child_env import child_env\n"
    cases = {
        "walrus": "def f():\n    env = child_env()\n    if (env := os.environ.copy()):\n        pass\n    subprocess.run(['a'], env=env)\n",
        "for": "def f(envs):\n    env = child_env()\n    for env in envs:\n        pass\n    subprocess.run(['a'], env=env)\n",
        "with": "def f(cm):\n    env = child_env()\n    with cm as env:\n        pass\n    subprocess.run(['a'], env=env)\n",
        "tuple": "def f(g):\n    env = child_env()\n    env, x = g()\n    subprocess.run(['a'], env=env)\n",
        "augassign": "def f():\n    env = child_env()\n    env |= os.environ\n    subprocess.run(['a'], env=env)\n",
    }
    for name, body in cases.items():
        assert _hits(header + body) == ["subprocess.run"], name


def test_an_item_assignment_does_not_rebind_the_name():
    code = """
    import subprocess
    from maxim.utils.child_env import child_env
    def f(d):
        env = child_env()
        env["MPLBACKEND"] = "Agg"
        for k in ("A",):
            env[k] = d
        subprocess.run(["a"], env=env)
    """
    assert _hits(code) == []
