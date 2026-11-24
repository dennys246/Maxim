"""``maxim.utils.child_env``: a ``.pth`` hook's site-packages ``PYTHONPATH`` never reaches a child (#1593)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

from maxim.doctor import checks
from maxim.doctor.checks import check_maxim_install
from maxim.utils.child_env import child_env, own_site_dirs, strip_own_site_dirs

ROOTS = ("/venv/lib/python3.12/site-packages",)


def test_strip_drops_entries_inside_the_roots_and_keeps_the_rest_in_order():
    pp = os.pathsep.join(
        [
            "/venv/lib/python3.12/site-packages",
            "/venv/lib/python3.12/site-packages/gstreamer_python/lib/python3.12/site-packages",
            "/repo/src",
            "relative/src",
            "/other/venv/lib/python3.12/site-packages",
        ]
    )
    assert strip_own_site_dirs(pp, ROOTS) == os.pathsep.join(
        ["/repo/src", "relative/src", "/other/venv/lib/python3.12/site-packages"]
    )


def test_strip_keeps_empty_entries_which_python_reads_as_the_working_directory():
    assert strip_own_site_dirs(os.pathsep.join(["", "/venv/lib/python3.12/site-packages", "/x"]), ROOTS) == (
        os.pathsep.join(["", "/x"])
    )


def test_strip_does_not_match_a_sibling_that_only_shares_a_prefix():
    assert strip_own_site_dirs("/venv/lib/python3.12/site-packages-extra", ROOTS) == (
        "/venv/lib/python3.12/site-packages-extra"
    )


def test_child_env_removes_pythonpath_when_nothing_is_left():
    site_dir = own_site_dirs()[0]
    env = child_env({"PYTHONPATH": site_dir, "HOME": "/h"})
    assert "PYTHONPATH" not in env  # nothing left: removed, not set to ""
    assert env["HOME"] == "/h"


def test_child_env_keeps_other_variables_and_applies_overrides_last():
    site_dir = own_site_dirs()[0]
    env = child_env({"PYTHONPATH": os.pathsep.join([site_dir, "/repo/src"]), "GST_X": "1"}, MAXIM_ROLE="solo")
    assert env == {"PYTHONPATH": "/repo/src", "GST_X": "1", "MAXIM_ROLE": "solo"}
    assert child_env({"PYTHONPATH": "/a"}, PYTHONPATH=site_dir)["PYTHONPATH"] == site_dir  # an override is as given


def test_child_env_copies_instead_of_aliasing():
    source = {"A": "1"}
    env = child_env(source)
    env["A"] = "2"
    assert source == {"A": "1"}


_PARENT = textwrap.dedent(
    """
    import json, os, site, subprocess, sys
    site.addsitedir(sys.argv[1])  # processes the planted .pth, as interpreter startup would
    sys.path.insert(0, sys.argv[2])
    from maxim.utils.child_env import child_env
    probe = "import json, os, sys; print(json.dumps(os.environ.get('PYTHONPATH', '')))"
    # -S: the probe child runs no .pth hooks of its own, so it reports the environment it was GIVEN.
    inherited = subprocess.run([sys.executable, "-S", "-c", probe], capture_output=True, text=True, check=True)
    cleaned = subprocess.run(
        [sys.executable, "-S", "-c", probe], env=child_env(), capture_output=True, text=True, check=True
    )
    print(json.dumps([json.loads(inherited.stdout), json.loads(cleaned.stdout)]))
    """
)


def test_a_planted_pth_rewrite_reaches_an_inheriting_child_but_not_a_child_env_one(tmp_path: Path):
    """Two processes, as the bug needs: the hook rewrites the PARENT's os.environ, and only a CHILD sees it."""
    site_dir = own_site_dirs()[0]
    hook_dir = tmp_path / "hook"
    hook_dir.mkdir()
    (hook_dir / "zz_rewrite.pth").write_text(
        "import os; os.environ['PYTHONPATH'] = os.pathsep.join("
        f"[{site_dir!r}] + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p])\n"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    caller = str(tmp_path / "caller_src")
    out = subprocess.run(
        [sys.executable, "-c", _PARENT, str(hook_dir), src],
        env={**os.environ, "PYTHONPATH": caller},
        capture_output=True,
        text=True,
        check=True,
    )
    inherited, cleaned = json.loads(out.stdout.strip().splitlines()[-1])
    # The mechanism is real: without child_env the child gets site-packages FIRST (the deletion probe).
    assert inherited.split(os.pathsep)[0] == site_dir
    # With child_env the hook's entry is gone and the caller's own entry survives.
    assert site_dir not in cleaned.split(os.pathsep)
    assert caller in cleaned.split(os.pathsep)


def _site(tmp_path: Path, *, stray: bool) -> str:
    root = tmp_path / "site-packages"
    (root / "maxim").mkdir(parents=True)
    if stray:
        (root / "maxim" / "__init__.py").write_text("")
    return str(root)


def test_doctor_fails_on_a_stray_site_packages_copy_beside_a_checkout(tmp_path: Path):
    result = check_maxim_install((_site(tmp_path, stray=True),))
    assert result.status == "fail"
    assert "site-packages/maxim" in result.message
    assert result.fix and "pip uninstall pymaxim" in result.fix


def test_doctor_asks_a_child_and_is_ok_when_it_imports_the_same_tree(tmp_path: Path, monkeypatch):
    import maxim

    monkeypatch.setattr(checks, "_child_maxim_dir", lambda: Path(maxim.__file__).resolve().parent)
    assert check_maxim_install((_site(tmp_path, stray=False),)).status == "ok"


def test_doctor_fails_when_a_child_imports_another_tree(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checks, "_child_maxim_dir", lambda: tmp_path / "elsewhere" / "maxim")
    result = check_maxim_install((_site(tmp_path, stray=False),))
    assert result.status == "fail" and "a child process imports" in result.message


def test_doctor_warns_when_a_child_cannot_import_maxim(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(checks, "_child_maxim_dir", lambda: "ModuleNotFoundError: No module named 'maxim'")
    assert check_maxim_install((_site(tmp_path, stray=False),)).status == "warn"


def test_doctor_runs_the_real_child_probe():
    """No monkeypatch: the probe really spawns a child under child_env and reads back a directory."""
    child = checks._child_maxim_dir()
    assert isinstance(child, Path) and (child / "__init__.py").is_file()


def _imported_from(monkeypatch, root: Path) -> None:
    import maxim

    pkg = root / "maxim"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    monkeypatch.setattr(maxim, "__file__", str(pkg / "__init__.py"))


def test_doctor_fails_on_an_unclaimed_site_packages_copy_that_is_the_imported_one(tmp_path: Path, monkeypatch):
    """A path-style editable install appends src AFTER site-packages, so a stray copy wins in the process itself."""
    root = tmp_path / "site-packages"
    _imported_from(monkeypatch, root)
    monkeypatch.setattr(checks, "_claimed_by_a_distribution", lambda pkg_dir: False)
    result = check_maxim_install((str(root),))
    assert result.status == "fail" and "no installed package claims" in result.message


def test_doctor_is_ok_for_an_installed_package_a_distribution_claims(tmp_path: Path, monkeypatch):
    root = tmp_path / "site-packages"
    _imported_from(monkeypatch, root)
    monkeypatch.setattr(checks, "_claimed_by_a_distribution", lambda pkg_dir: True)
    assert check_maxim_install((str(root),)).status == "ok"


def test_claimed_by_a_distribution_reads_record_files(tmp_path: Path):
    """The real RECORD reader: a directory no distribution lists is unclaimed."""
    assert checks._claimed_by_a_distribution(tmp_path / "maxim") is False
    import pytest as _pytest

    pytest_dir = Path(_pytest.__file__).resolve().parent
    assert checks._claimed_by_a_distribution(pytest_dir) is True


def test_child_env_module_imports_only_the_standard_library():
    """scripts/_child_env.py loads child_env.py by path so harnesses never import the maxim package. A maxim (or any
    non-stdlib) import added to it would quietly import whichever maxim sys.path holds, the very gap it closes."""
    import ast

    source = Path(__file__).resolve().parents[2] / "src" / "maxim" / "utils" / "child_env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}, sorted(modules - set(sys.stdlib_module_names))


# ── the migrated spawn sites hand their child child_env, not the hook-rewritten env ─────────────────────────────────


def _polluted(monkeypatch) -> str:
    """Put this interpreter's site-packages FIRST on os.environ's PYTHONPATH, as the GStreamer .pth hook does."""
    site_dir = own_site_dirs()[0]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([site_dir, "/repo/src"]))
    return site_dir


def _capture_run(monkeypatch, target: str) -> list[dict]:
    envs: list[dict] = []

    def fake_run(cmd, **kwargs):
        envs.append(kwargs.get("env"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(f"{target}.subprocess.run", fake_run)
    return envs


def test_the_leader_restart_reexecs_with_child_env(monkeypatch):
    import threading

    from maxim.runtime import leader_proxy

    site_dir = _polluted(monkeypatch)
    execs: list[tuple] = []
    monkeypatch.setattr(leader_proxy.os, "execve", lambda exe, argv, env: execs.append((exe, argv, env)))
    monkeypatch.setattr(leader_proxy, "_stop_own_servers_for_restart", lambda: None)
    monkeypatch.setattr(leader_proxy.time, "sleep", lambda s: None)

    class _Inline:  # run the restart thread's target synchronously
        def __init__(self, target, **_kw):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(threading, "Thread", _Inline)
    h = object.__new__(leader_proxy._ProxyHandler)
    h.start_time = 0.0
    h.client_address = ("127.0.0.1", 1)
    h._admin_refused = lambda name: False
    h._read_body = lambda max_size: b""
    h._send_json = lambda code, body: None
    h._handle_admin_restart()
    assert len(execs) == 1
    env = execs[0][2]
    assert site_dir not in env.get("PYTHONPATH", "").split(os.pathsep)
    assert "/repo/src" in env["PYTHONPATH"].split(os.pathsep)


def test_the_leader_pip_calls_get_child_env(monkeypatch, tmp_path: Path):
    from maxim.runtime import leader_proxy

    site_dir = _polluted(monkeypatch)
    envs = _capture_run(monkeypatch, "maxim.runtime.leader_proxy")
    leader_proxy._get_latest_pypi_version()
    h = object.__new__(leader_proxy._ProxyHandler)
    h._send_json = lambda code, body: None
    h._run_pip_install(str(tmp_path))
    assert len(envs) == 2
    for env in envs:
        assert env is not None and site_dir not in env.get("PYTHONPATH", "").split(os.pathsep)


def test_the_matplotlib_preflight_gets_child_env(monkeypatch, tmp_path: Path):
    from maxim.utils import plotting

    site_dir = _polluted(monkeypatch)
    envs = _capture_run(monkeypatch, "maxim.utils.plotting")
    plotting.preflight_matplotlib_fonts(cache_dir=tmp_path / "mpl")
    assert envs and envs[0] is not None
    assert site_dir not in envs[0].get("PYTHONPATH", "").split(os.pathsep)
    assert envs[0]["MPLCONFIGDIR"] == str(tmp_path / "mpl")


def _fake_child(monkeypatch, *, returncode=1, stdout="", stderr="", raises=None):
    def fake_run(cmd, **kwargs):
        if raises is not None:
            raise raises
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_child_probe_reports_whitespace_only_stderr_instead_of_crashing(monkeypatch):
    _fake_child(monkeypatch, returncode=1, stderr="\n")
    assert checks._child_maxim_dir() == "exit 1, no output"


def test_child_probe_reports_the_last_stderr_line(monkeypatch):
    _fake_child(monkeypatch, returncode=1, stderr="Traceback\nModuleNotFoundError: No module named 'maxim'\n")
    assert checks._child_maxim_dir() == "ModuleNotFoundError: No module named 'maxim'"


def test_child_probe_reports_a_refused_or_failed_spawn(monkeypatch):
    _fake_child(monkeypatch, raises=RuntimeError("offline: spawn refused"))
    assert checks._child_maxim_dir() == "RuntimeError: offline: spawn refused"
    _fake_child(monkeypatch, raises=subprocess.TimeoutExpired(["x"], 60))
    assert checks._child_maxim_dir().startswith("TimeoutExpired")


def test_own_site_dirs_adds_the_user_site_when_enabled(monkeypatch, tmp_path: Path):
    import site

    monkeypatch.setattr(site, "ENABLE_USER_SITE", True)
    monkeypatch.setattr(site, "getusersitepackages", lambda: str(tmp_path / "user-site"))
    assert os.path.realpath(tmp_path / "user-site") in own_site_dirs()
    monkeypatch.setattr(site, "ENABLE_USER_SITE", False)
    assert os.path.realpath(tmp_path / "user-site") not in own_site_dirs()


def test_child_probe_downgrades_the_offline_latch_refusal_to_a_warning(tmp_path: Path, monkeypatch):
    """Under the offline latch the spawn is refused with OfflineRefusal (a RuntimeError); doctor must not crash."""
    from maxim._offline import OfflineRefusal

    _fake_child(monkeypatch, raises=OfflineRefusal("subprocess.Popen refused: offline"))
    result = check_maxim_install((_site(tmp_path, stray=False),))
    assert result.status == "warn" and "OfflineRefusal" in result.message
