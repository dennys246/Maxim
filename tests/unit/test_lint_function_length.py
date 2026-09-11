"""Tests for the god-function length ratchet (scripts/lint_function_length.py).

The ratchet's value is that it FIRES on growth and FAILS LOUD on a rename — a guard that
silently passes when the function moves would be worthless. These pin both behaviours plus
the clean-on-current-main invariant.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location("lint_function_length", _REPO / "scripts" / "lint_function_length.py")
assert _SPEC and _SPEC.loader
_LINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LINT)


def test_ratchet_is_clean_on_current_main():
    """The pinned baselines match (or exceed) the current spans."""
    assert _LINT.check() == 0


def test_every_pinned_function_actually_exists():
    """No baseline entry silently no-ops because the function moved."""
    for (rel, name), _pinned in _LINT._BASELINES.items():
        path = _REPO / rel
        assert path.exists(), f"pinned file missing: {rel}"
        assert _LINT._span_of(path, name) is not None, f"pinned function not found: {rel}::{name}"


def test_growth_fails(monkeypatch):
    """Tightening a pin below the real span (i.e. simulating growth past the pin) fails."""
    tightened = {k: max(1, v - 50) for k, v in _LINT._BASELINES.items()}
    monkeypatch.setattr(_LINT, "_BASELINES", tightened)
    assert _LINT.check() == 1


def test_rename_fails_loud(monkeypatch):
    """A pinned name that no longer exists must FAIL, not pass silently."""
    bogus = {("src/maxim/cli.py", "a_function_that_does_not_exist"): 100}
    monkeypatch.setattr(_LINT, "_BASELINES", bogus)
    assert _LINT.check() == 1


def test_span_of_measures_inclusive_line_count(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("def foo():\n    x = 1\n    return x\n")  # 3 lines, def on line 1
    assert _LINT._span_of(f, "foo") == 3
    assert _LINT._span_of(f, "missing") is None
