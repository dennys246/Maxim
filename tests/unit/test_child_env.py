"""``maxim.utils.child_env``: a ``.pth`` hook's site-packages ``PYTHONPATH`` never reaches a child (#1593)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

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


def test_strip_does_not_match_a_sibling_that_only_shares_a_prefix():
    assert strip_own_site_dirs("/venv/lib/python3.12/site-packages-extra", ROOTS) == (
        "/venv/lib/python3.12/site-packages-extra"
    )


def test_child_env_removes_pythonpath_when_nothing_is_left():
    site_dir = own_site_dirs()[0]
    env = child_env({"PYTHONPATH": site_dir, "HOME": "/h"})
    assert "PYTHONPATH" not in env  # an empty value would put the child's cwd on sys.path
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


def test_doctor_is_ok_without_a_stray_copy(tmp_path: Path):
    assert check_maxim_install((_site(tmp_path, stray=False),)).status == "ok"


def test_doctor_is_ok_for_a_normal_pip_install():
    import maxim

    installed_root = str(Path(maxim.__file__).resolve().parent.parent)
    assert check_maxim_install((installed_root,)).status == "ok"
