"""tests/package_attribute_guard.py, the #1755 tripwire.

The in-process cases drive the guard's functions over a synthetic package tree (``_t1755pkg``), never the real
``maxim``. The child-pytest cases run generated tests with ``tests/conftest.py`` as a plugin (the
``test_script_module_guard.py`` pattern), so the real fixtures judge them on the real package. They reproduce both
#1755 shapes and check that the culprit errors, a dotted ``maxim.*`` patch in the NEXT test still resolves (the
state was repaired), and undone changes pass.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from tests import package_attribute_guard as guard

REPO = Path(__file__).resolve().parents[2]
ROOT = "_t1755pkg"


@pytest.fixture
def tree(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """``_t1755pkg`` with ``.sub`` (a package) and ``.sub.leaf``, each attached to its parent. Every name the cases
    register is removed afterwards (``monkeypatch.setitem`` restores a name it added by deleting it)."""
    root, sub, leaf = (types.ModuleType(n) for n in (ROOT, f"{ROOT}.sub", f"{ROOT}.sub.leaf"))
    root.sub = sub  # type: ignore[attr-defined]
    sub.leaf = leaf  # type: ignore[attr-defined]
    for mod in (root, sub, leaf):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setitem(sys.modules, f"{ROOT}.new", None)  # reserved for the cases that register it
    return types.SimpleNamespace(root=root, sub=sub, leaf=leaf)


def test_an_untouched_tree_has_no_leak(tree: types.SimpleNamespace) -> None:
    before = guard.snapshot(ROOT)
    assert before.root is tree.root and before.detached == frozenset()
    assert guard.check_and_restore(before) is None


def test_a_lost_submodule_attribute_is_reported_and_put_back(tree: types.SimpleNamespace) -> None:
    before = guard.snapshot(ROOT)
    del tree.sub.leaf
    message = guard.check_and_restore(before)
    assert message is not None and f"{ROOT}.sub.leaf is registered in sys.modules but detached" in message
    assert tree.sub.leaf is tree.leaf  # repaired, so later tests resolve the dotted path again


def test_an_attribute_gained_in_the_window_and_then_deleted_is_reported(tree: types.SimpleNamespace) -> None:
    """The review's shape: ``setattr(pkg, name, x, raising=False)`` before the first import, the import attaches the
    real module, the undo deletes it. Nothing about ``new`` existed when the window opened."""
    before = guard.snapshot(ROOT)
    new = types.ModuleType(f"{ROOT}.new")
    sys.modules[new.__name__] = new  # the import registers it ...
    tree.root.new = new  # type: ignore[attr-defined]  # ... and attaches it
    del tree.root.new  # type: ignore[attr-defined]  # the undo, with no old value, deletes the attribute
    message = guard.check_and_restore(before)
    assert message is not None and f"{ROOT}.new is registered in sys.modules but detached" in message
    assert tree.root.new is new


def test_a_repair_attaches_the_module_registered_now_not_a_stale_one(tree: types.SimpleNamespace) -> None:
    """Re-attaching the object seen at the start would split ``import`` (the new one) from attribute access."""
    before = guard.snapshot(ROOT)
    fresh = types.ModuleType(f"{ROOT}.sub.leaf")
    sys.modules[fresh.__name__] = fresh  # a re-import registered a fresh object ...
    del tree.sub.leaf  # ... and the package lost its attribute
    assert guard.check_and_restore(before) is not None
    assert tree.sub.leaf is fresh is sys.modules[f"{ROOT}.sub.leaf"]


def test_damage_inherited_from_before_the_window_is_not_blamed(tree: types.SimpleNamespace) -> None:
    del tree.sub.leaf
    before = guard.snapshot(ROOT)
    assert before.detached == frozenset({f"{ROOT}.sub.leaf"})
    assert guard.check_and_restore(before) is None


def test_a_replaced_root_package_is_reported_and_every_submodule_reattached(tree: types.SimpleNamespace) -> None:
    """The other #1755 shape. A submodule first imported in the window attached only to the fresh package object;
    after the original root goes back it is detached, so the repair attaches it too."""
    before = guard.snapshot(ROOT)
    fresh_root = types.ModuleType(ROOT)
    sys.modules[ROOT] = fresh_root
    new = types.ModuleType(f"{ROOT}.new")
    sys.modules[new.__name__] = new
    fresh_root.new = new  # type: ignore[attr-defined]
    message = guard.check_and_restore(before)
    assert message is not None and f"sys.modules[{ROOT!r}] was replaced by another module object" in message
    assert sys.modules[ROOT] is tree.root and tree.root.new is new


def test_a_removed_root_package_is_reported(tree: types.SimpleNamespace) -> None:
    before = guard.snapshot(ROOT)
    del sys.modules[ROOT]
    message = guard.check_and_restore(before)
    assert message is not None and f"sys.modules[{ROOT!r}] was removed" in message
    assert sys.modules[ROOT] is tree.root


def test_a_first_import_and_a_replaced_but_attached_submodule_pass(tree: types.SimpleNamespace) -> None:
    """Adding an attached submodule is an ordinary import; a re-imported submodule whose attribute points at some
    module is a stated blind spot (dotted patch paths still resolve), not a leak."""
    before = guard.snapshot(ROOT)
    new = types.ModuleType(f"{ROOT}.new")
    sys.modules[new.__name__] = new
    tree.root.new = new  # type: ignore[attr-defined]
    tree.sub.leaf = types.ModuleType(f"{ROOT}.sub.leaf")  # type: ignore[attr-defined]
    assert guard.check_and_restore(before) is None


def test_the_real_maxim_package_is_intact_here() -> None:
    import maxim.utils  # noqa: F401

    before = guard.snapshot()
    assert before.root is sys.modules["maxim"]
    assert before.detached == frozenset(), "a maxim submodule is already detached in this process"


_HEADER = '''
import importlib, pkgutil, sys
import pytest
import maxim
import maxim.utils
import maxim.utils.paths


def gain_then_undo_a_utils_submodule():
    """The #1755 undo shape on a ``maxim.utils`` submodule this process has not imported yet (sorted, first that
    loads): patch the name with no old value, import the real module (which attaches it), undo (which deletes it)."""
    for info in sorted(pkgutil.iter_modules(maxim.utils.__path__), key=lambda m: m.name):
        name = f"maxim.utils.{info.name}"
        if name in sys.modules or info.ispkg or info.name in vars(maxim.utils):
            continue
        mp = pytest.MonkeyPatch()
        mp.setattr(maxim.utils, info.name, object(), raising=False)  # notset before: the undo will delete it
        try:
            importlib.import_module(name)
        except ImportError:
            mp.undo()
            continue
        assert vars(maxim.utils)[info.name] is sys.modules[name]  # the import attached the real module
        mp.undo()
        return info.name
    raise AssertionError("no unloaded maxim.utils module to import")
'''


def _outcomes(tmp_path: Path, body: str) -> tuple[dict[str, str], str]:
    (tmp_path / "test_tripwire.py").write_text(_HEADER + textwrap.dedent(body))
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rA", "-p", "tests.conftest", "-p", "no:cacheprovider"]
        + ["-p", "no:randomly", "--no-cov", str(tmp_path / "test_tripwire.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    out = run.stdout + run.stderr
    found: dict[str, str] = {}
    for line in out.splitlines():
        m = re.match(r"(PASSED|FAILED|ERROR) \S*?::(\w+)\b", line)
        if m and found.get(m.group(2)) != "ERROR":  # passed its call, errored at teardown: an ERROR here
            found[m.group(2)] = m.group(1)
    return found, out


def test_the_conftest_fixtures_fail_the_culprit_and_spare_the_next_test(tmp_path: Path) -> None:
    found, out = _outcomes(
        tmp_path,
        """
        def test_a_lost_attribute_leaks():
            del maxim.utils  # what #1755's victims saw: maxim.utils registered, not an attribute

        def test_b_a_dotted_patch_resolves_after_the_reported_leak(monkeypatch):
            monkeypatch.setattr("maxim.utils.paths.Path", object)

        def test_c_a_replaced_package_leaks():
            sys.modules["maxim"] = type(maxim)("maxim")

        def test_d_the_package_is_back():
            assert sys.modules["maxim"] is maxim and "utils" in vars(maxim)

        def test_e_a_patch_before_the_first_import_undone_after_it_leaks():
            gain_then_undo_a_utils_submodule()

        def test_f_an_undone_delattr_passes(monkeypatch):
            monkeypatch.delattr(maxim, "utils")

        def test_g_an_undone_package_swap_passes(monkeypatch):
            monkeypatch.setitem(sys.modules, "maxim", type(maxim)("maxim"))
        """,
    )
    expected = {
        "test_a_lost_attribute_leaks": "ERROR",
        "test_b_a_dotted_patch_resolves_after_the_reported_leak": "PASSED",
        "test_c_a_replaced_package_leaks": "ERROR",
        "test_d_the_package_is_back": "PASSED",
        "test_e_a_patch_before_the_first_import_undone_after_it_leaks": "ERROR",
        "test_f_an_undone_delattr_passes": "PASSED",
        "test_g_an_undone_package_swap_passes": "PASSED",
    }
    assert found == expected, out
    assert "maxim.utils is registered in sys.modules but detached" in out, out
    assert "sys.modules['maxim'] was replaced by another module object" in out, out


def test_a_module_scoped_fixture_that_detaches_a_submodule_errors_its_module(tmp_path: Path) -> None:
    found, out = _outcomes(
        tmp_path,
        """
        @pytest.fixture(scope="module")
        def leaky():
            yield
            del maxim.utils  # after the per-test window closed; only the module window sees it

        def test_uses_it(leaky):
            pass

        def test_last():
            pass
        """,
    )
    assert found.get("test_uses_it") == "PASSED", out
    assert found.get("test_last") == "ERROR", out
    assert "in this module's fixtures" in out and "maxim.utils is registered in sys.modules but detached" in out, out
