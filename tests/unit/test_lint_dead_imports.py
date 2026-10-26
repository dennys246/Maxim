"""Tests for the dead-import lint (scripts/lint_dead_imports.py, #1436).

The guard's value is that it FIRES on an import of a module or name that no longer exists, at any
depth, and does NOT false-flag a live one. The fixtures build a tiny `src/maxim` + `scripts/` tree.
The repo-clean gate is strict-red until #1436's fix removes the ten dead imports it found.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location("lint_dead_imports", _REPO / "scripts" / "lint_dead_imports.py")
assert _SPEC and _SPEC.loader
_LINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LINT)


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, body in {"src/maxim/__init__.py": "", "scripts/.keep": "", **files}.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")
    return tmp_path


@pytest.mark.xfail(strict=True, reason="#1436: ten dead maxim imports in src/ and scripts/ (voice input among them)")
def test_the_repo_has_no_dead_maxim_imports():
    assert _LINT.check() == []


def test_a_lazy_import_of_a_deleted_module_fails(tmp_path):
    repo = _tree(tmp_path, {"src/maxim/a.py": "def f():\n    from maxim.gone import thing\n"})
    assert _LINT.check(repo) == ["src/maxim/a.py:2: module maxim.gone does not exist"]


def test_a_deleted_name_in_a_live_module_fails(tmp_path):
    repo = _tree(
        tmp_path,
        {"src/maxim/b.py": "def kept():\n    pass\n", "src/maxim/a.py": "def f():\n    from maxim.b import dropped\n"},
    )
    assert _LINT.check(repo) == ["src/maxim/a.py:2: maxim.b has no dropped"]


def test_live_targets_pass_submodules_reexports_try_blocks_and_relative_imports(tmp_path):
    repo = _tree(
        tmp_path,
        {
            "src/maxim/pkg/__init__.py": "from maxim.pkg.impl import Thing\ntry:\n    import json as fast\nexcept ImportError:\n    fast = None\n",
            "src/maxim/pkg/impl.py": "class Thing:\n    pass\nif True:\n    LATE = 1\n",
            "src/maxim/pkg/sub.py": "",
            "src/maxim/user.py": textwrap.dedent(
                """\
                import maxim.pkg.sub
                from maxim.pkg import Thing, fast, sub
                from maxim.pkg.impl import LATE
                class C:
                    from maxim.pkg import Thing as T
                def f():
                    from .pkg.impl import Thing
                """
            ),
        },
    )
    assert _LINT.check(repo) == []


def test_type_checking_imports_must_resolve_and_bind_nothing_at_runtime(tmp_path):
    repo = _tree(
        tmp_path,
        {
            "src/maxim/t.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from maxim.u import Only\n",
            "src/maxim/u.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    Only = int\n",
            "src/maxim/v.py": "def f():\n    from maxim.u import Only\n",
            "src/maxim/w.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from maxim.gone import X\n",
        },
    )
    assert _LINT.check(repo) == [
        "src/maxim/t.py:3: maxim.u has no Only",
        "src/maxim/v.py:2: maxim.u has no Only",
        "src/maxim/w.py:3: module maxim.gone does not exist",
    ]


def test_a_lazy_getattr_module_accepts_only_its_closed_literal_set(tmp_path):
    repo = _tree(
        tmp_path,
        {
            "src/maxim/lazy.py": "_NAMES = {'Known': 'x'}\ndef __getattr__(name):\n    return _NAMES[name]\n",
            "src/maxim/a.py": "def f():\n    from maxim.lazy import Known\n    from maxim.lazy import Unknown\n",
        },
    )
    assert _LINT.check(repo) == ["src/maxim/a.py:3: maxim.lazy has no Unknown"]


def test_scripts_are_checked_at_top_level_too(tmp_path):
    repo = _tree(tmp_path, {"scripts/run.py": "from maxim.gone import x\n"})
    assert _LINT.check(repo) == ["scripts/run.py:1: module maxim.gone does not exist"]


def test_non_maxim_imports_are_ignored(tmp_path):
    repo = _tree(tmp_path, {"src/maxim/a.py": "def f():\n    import numpy\n    from os import path\n"})
    assert _LINT.check(repo) == []


def test_main_reports_and_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(_LINT, "check", lambda: ["src/maxim/a.py:2: module maxim.gone does not exist"])
    assert _LINT.main() == 1
    assert "dead-import lint: FAIL (1)" in capsys.readouterr().out
    monkeypatch.setattr(_LINT, "check", lambda: [])
    assert _LINT.main() == 0
