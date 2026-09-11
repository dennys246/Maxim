"""Tests for the orphan-module ratchet (scripts/lint_orphan_modules.py).

The guard's whole value is that it FIRES on a new unreferenced module and does NOT
false-flag a live one. These pin: clean-on-main (zero orphans, empty grandfather),
`__main__` exempt, and a synthetic never-referenced module failing.

Note: the synthetic module's name is BUILT at runtime (never a single string literal in
this file) — otherwise this test file, which the lint scans, would itself "reference" the
name and the orphan would look live.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location("lint_orphan_modules", _REPO / "scripts" / "lint_orphan_modules.py")
assert _SPEC and _SPEC.loader
_LINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LINT)

# Built by concatenation so the full token never appears as a literal anywhere.
_FAKE = "zzq" + "orphan" + "probe" + "xyz"


def test_clean_on_main_zero_orphans():
    assert _LINT._orphans() == []
    assert _LINT.check() == 0


def test_grandfather_set_is_empty():
    # The honest baseline: every module is referenced somewhere in src/tests/scripts.
    assert _LINT._GRANDFATHERED == set()


def test_main_entrypoint_is_exempt():
    assert "__main__" in _LINT._DYNAMIC_ALLOW
    assert "__main__" not in _LINT._orphans()


def test_new_unreferenced_module_fails(monkeypatch, tmp_path):
    mod = tmp_path / f"{_FAKE}.py"
    mod.write_text("value = 1\n")  # content free of the module name
    real = _LINT._candidate_modules()
    real[_FAKE] = mod
    monkeypatch.setattr(_LINT, "_candidate_modules", lambda: real)
    assert _FAKE in _LINT._orphans()
    assert _LINT.check() == 1


def test_dynamic_allow_suppresses_a_would_be_orphan(monkeypatch, tmp_path):
    mod = tmp_path / f"{_FAKE}.py"
    mod.write_text("value = 1\n")
    real = _LINT._candidate_modules()
    real[_FAKE] = mod
    monkeypatch.setattr(_LINT, "_candidate_modules", lambda: real)
    monkeypatch.setattr(_LINT, "_DYNAMIC_ALLOW", _LINT._DYNAMIC_ALLOW | {_FAKE})
    assert _FAKE not in _LINT._orphans()
    assert _LINT.check() == 0
