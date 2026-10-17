"""tests/script_module_guard.py's positive control (#1339).

Each case runs a generated test file in a child pytest with ``tests/conftest.py`` as a plugin (the
``test_require_extras_lane.py`` pattern), so the guard's real hooks judge it: a leak errors the test (at teardown), its module
or its collection; a restore, ``monkeypatch.setitem`` and a first import pass; after a reported leak the original
entry is back. Only ``scripts/_provenance.py`` (stdlib only, no side effects at import) is really imported; the
``sys.path``-subdirectory case uses a stand-in module whose ``__file__`` names a script there.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_HEADER = f"""
import importlib.util, sys, types
from pathlib import Path
import pytest
SCRIPTS = Path({str(REPO / "scripts")!r})
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import _provenance as P

def fresh():
    spec = importlib.util.spec_from_file_location("_provenance", SCRIPTS / "_provenance.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
"""


def _outcomes(tmp_path: Path, files: dict[str, str], header: str = _HEADER) -> tuple[dict[str, str], str]:
    for name, body in files.items():
        (tmp_path / name).write_text(header + textwrap.dedent(body))
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "tests.conftest", "-p", "no:cacheprovider"]
        + [
            "-p",
            "no:randomly",
            "--no-cov",
            "--continue-on-collection-errors",
            *sorted(str(tmp_path / n) for n in files),
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    out = run.stdout + run.stderr
    found: dict[str, str] = {}
    for line in out.splitlines():
        m = re.match(r"(PASSED|FAILED|ERROR) \S*?([\w.]+\.py(?:::\w+)?)\b", line)
        if m:
            key = m.group(2)
            # a test that passed its call and errored at teardown is an ERROR for this purpose
            if found.get(key) != "ERROR":
                found[key] = m.group(1)
    return found, out


def test_a_test_that_leaves_a_script_module_replaced_or_removed_fails_and_innocent_ones_pass(tmp_path: Path) -> None:
    found, out = _outcomes(
        tmp_path,
        {
            "test_fn.py": """
            sys.path.insert(0, str(SCRIPTS / "orient_backbone"))
            stand_in = types.ModuleType("live_common")
            stand_in.__file__ = str(SCRIPTS / "orient_backbone" / "live_common.py")
            sys.modules["live_common"] = stand_in  # first registration, at collection

            def test_a_replace_leaks():
                sys.modules["_provenance"] = fresh()

            def test_b_the_original_is_back_after_a_reported_leak():
                assert sys.modules["_provenance"] is P

            def test_c_a_removal_leaks():
                del sys.modules["_provenance"]

            def test_d_replace_then_restore_passes():
                sys.modules["_provenance"] = fresh()
                sys.modules["_provenance"] = P

            def test_e_setitem_passes(monkeypatch):
                monkeypatch.setitem(sys.modules, "_provenance", fresh())

            def test_f_a_first_import_passes():
                assert "scripts._provenance" not in sys.modules
                import scripts._provenance  # noqa: F401

            def test_g_a_name_from_a_scripts_subdirectory_on_sys_path_is_guarded():
                other = types.ModuleType("live_common")
                other.__file__ = stand_in.__file__
                sys.modules["live_common"] = other
            """,
        },
    )
    expected = {
        "test_fn.py::test_a_replace_leaks": "ERROR",
        "test_fn.py::test_b_the_original_is_back_after_a_reported_leak": "PASSED",
        "test_fn.py::test_c_a_removal_leaks": "ERROR",
        "test_fn.py::test_d_replace_then_restore_passes": "PASSED",
        "test_fn.py::test_e_setitem_passes": "PASSED",
        "test_fn.py::test_f_a_first_import_passes": "PASSED",
        "test_fn.py::test_g_a_name_from_a_scripts_subdirectory_on_sys_path_is_guarded": "ERROR",
    }
    assert {k: found.get(k) for k in expected} == expected, out
    assert "sys.modules['_provenance'] was replaced" in out and "sys.modules['_provenance'] was removed" in out, out
    assert "sys.modules['live_common'] was replaced" in out, out


def test_a_collection_time_leak_is_a_collection_error_and_keeps_an_import_error(tmp_path: Path) -> None:
    found, out = _outcomes(
        tmp_path,
        {
            "test_a_first.py": "def test_ok():\n    pass\n",
            "test_b_leaks_at_import.py": 'sys.modules["_provenance"] = fresh()\n\ndef test_never():\n    pass\n',
            "test_c_leaks_then_raises.py": 'sys.modules["_provenance"] = fresh()\nraise RuntimeError("boom-1339")\n',
        },
    )
    assert found.get("test_a_first.py::test_ok") == "PASSED", out
    assert found.get("test_b_leaks_at_import.py") == "ERROR", out
    assert found.get("test_c_leaks_then_raises.py") == "ERROR", out
    assert "boom-1339" in out and out.count("sys.modules['_provenance'] was replaced") >= 2, out


def test_a_module_scoped_fixture_that_leaks_errors_its_module(tmp_path: Path) -> None:
    found, out = _outcomes(
        tmp_path,
        {
            "test_mod.py": """
            @pytest.fixture(scope="module")
            def leaky():
                sys.modules["_provenance"] = fresh()
                yield

            def test_uses_it(leaky):
                pass

            def test_last():
                pass
            """,
        },
    )
    assert found.get("test_mod.py::test_uses_it") == "PASSED", out
    assert found.get("test_mod.py::test_last") == "ERROR", out
    assert "in this module's fixtures" in out, out


def test_a_scripts_module_registered_under_its_own_name_is_guarded_without_scripts_on_sys_path(tmp_path: Path) -> None:
    """``import`` returns ``sys.modules[name]`` without consulting ``sys.path``, so the guard must not need it."""
    found, out = _outcomes(
        tmp_path,
        {
            "test_nopath.py": """
            def fresh(name="_provenance"):
                spec = importlib.util.spec_from_file_location(name, SCRIPTS / "_provenance.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

            P = fresh()
            sys.modules["_provenance"] = P  # first registration, at collection, scripts/ not on sys.path
            sys.modules["m1b5a_private"] = fresh("m1b5a_private")

            def test_a_replace_leaks():
                assert str(SCRIPTS) not in sys.path
                sys.modules["_provenance"] = fresh()

            def test_b_the_victim_sees_the_original():
                import _provenance

                assert _provenance is P

            def test_c_a_private_name_is_not_guarded():
                sys.modules["m1b5a_private"] = fresh("m1b5a_private")
            """,
        },
        header=(f"import importlib.util, sys\nfrom pathlib import Path\nSCRIPTS = Path({str(REPO / 'scripts')!r})\n"),
    )
    expected = {
        "test_nopath.py::test_a_replace_leaks": "ERROR",
        "test_nopath.py::test_b_the_victim_sees_the_original": "PASSED",
        "test_nopath.py::test_c_a_private_name_is_not_guarded": "PASSED",
    }
    assert {k: found.get(k) for k in expected} == expected, out
