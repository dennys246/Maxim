"""The LLM spawn-record directory is box-wide and never the developer's real one in a test (#1344 PR 3b, gate 9).

A TCP port is box-wide, so the record of who spawned the server on it cannot follow ``HOME`` or ``MAXIM_DATA_HOME``
(design round 5 D1): ``utils/paths.py::llm_spawn_record_dir`` reads the passwd home (``pwd.getpwuid``), override
``MAXIM_LLM_SPAWN_RECORD_DIR``. The suite isolates HOME, but the passwd home is the developer's real one, so conftest
exports the override into its isolation root at import and rebinds the reader's default, the way it does for
``MAXIM_CLOUDFLARED_SYSTEM_DIR`` (``test_cloudflared_system_dir_isolation.py``). These guards check the mechanism.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ENV = "MAXIM_LLM_SPAWN_RECORD_DIR"
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
POSIX = pytest.mark.skipif(sys.platform == "win32", reason="spawn records are POSIX only")


def red(why: str):
    return pytest.mark.xfail(strict=True, reason=f"PR 3b red gate 9: {why}")


@red("conftest exports the override into the isolation root")
def test_the_suite_points_the_record_dir_into_its_isolation_root():
    from maxim.utils.paths import llm_spawn_record_dir

    raw = os.environ.get(_ENV, "")
    assert raw, f"conftest must export {_ENV} beside HOME"
    assert Path(raw).parent == Path(os.environ["HOME"]).parent
    assert llm_spawn_record_dir() == Path(raw)


@POSIX
@red("the dir ignores HOME and MAXIM_DATA_HOME and honours the override")
def test_the_record_dir_ignores_home_and_data_home_and_honours_the_override(tmp_path, monkeypatch):
    from maxim.utils.paths import llm_spawn_record_dir

    before = llm_spawn_record_dir()
    monkeypatch.setenv("HOME", str(tmp_path / "other-home"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "other-data"))
    assert llm_spawn_record_dir() == before
    monkeypatch.setenv(_ENV, str(tmp_path / "records"))
    assert llm_spawn_record_dir() == tmp_path / "records"


@red("an in-process strip falls back to the isolation dir, never the passwd home")
def test_an_in_process_strip_still_never_reads_the_real_home(monkeypatch):
    from maxim.utils.paths import llm_spawn_record_dir

    isolated = Path(os.environ[_ENV])
    monkeypatch.delenv(_ENV)
    assert llm_spawn_record_dir() == isolated


@red("the child-env helpers keep the variable")
def test_the_strip_helpers_keep_the_record_dir(monkeypatch):
    from tests._child_env import ISOLATION_VARS, drop_maxim_overrides, without_maxim_overrides

    assert _ENV in ISOLATION_VARS
    environ = {"MAXIM_LLM_PROFILE": "x", _ENV: "/iso/records"}
    drop_maxim_overrides(environ)
    assert environ == {_ENV: "/iso/records"}
    assert without_maxim_overrides()[_ENV] == os.environ[_ENV]


@POSIX
@red("a fresh process without the override reads <passwd home>/.maxim/run/llm_server, whatever HOME says")
def test_a_child_without_the_override_reads_the_passwd_home(tmp_path):
    import pwd

    from tests._child_env import without_maxim_overrides

    env = {k: v for k, v in without_maxim_overrides().items() if k != _ENV}  # WITHOUT the isolation, on purpose
    env["HOME"] = str(tmp_path / "not-the-passwd-home")
    code = "from maxim.utils.paths import llm_spawn_record_dir; print(llm_spawn_record_dir())"
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**env, "PYTHONPATH": str(_REPO_SRC)},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    expected = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".maxim" / "run" / "llm_server"
    assert out.stdout.strip().splitlines()[-1] == str(expected)


@POSIX
@red("an unusable passwd home writes no record and never falls back to HOME")
@pytest.mark.parametrize("case", ["keyerror", "root", "missing"])
def test_an_unusable_passwd_home_yields_none(monkeypatch, tmp_path, case):
    import pwd

    import maxim.utils.paths as paths

    monkeypatch.delenv(_ENV)
    monkeypatch.setattr(paths, "_DEFAULT_LLM_SPAWN_RECORD_DIR", None)  # resolve from passwd, as outside the suite

    real = pwd.getpwuid(os.getuid())

    def getpwuid(uid):
        if case == "keyerror":
            raise KeyError(uid)
        pw_dir = "/" if case == "root" else str(tmp_path / "no-such-home")
        return pwd.struct_passwd((real.pw_name, real.pw_passwd, real.pw_uid, real.pw_gid, real.pw_gecos, pw_dir, "/bin/sh"))

    monkeypatch.setattr(pwd, "getpwuid", getpwuid)
    assert paths.llm_spawn_record_dir() is None
