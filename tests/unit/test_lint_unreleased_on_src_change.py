"""Tests for the [Unreleased]-on-src-change lint (roadmap 16.10 enforcement).

Exercises the pure `verdict()` (git-free) across the four cases + the section parser.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "lint_unreleased_on_src_change", _REPO / "scripts" / "lint_unreleased_on_src_change.py"
)
assert _SPEC and _SPEC.loader
_LINT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LINT)

_BASE = """# Changelog

## [Unreleased]

## [1.2.1] - 2026-09-10 — "x"
- old thing
"""

_GREW = """# Changelog

## [Unreleased]

### Added
- a new entry

## [1.2.1] - 2026-09-10 — "x"
- old thing
"""

_RELEASED = """# Changelog

## [Unreleased]

## [1.3.0] - 2026-10-01 — "y"
- a new entry

## [1.2.1] - 2026-09-10 — "x"
- old thing
"""


def test_no_src_change_is_clean():
    ok, _ = _LINT.verdict(False, _BASE, _BASE)
    assert ok


def test_src_change_without_unreleased_growth_fails():
    ok, msg = _LINT.verdict(True, _BASE, _BASE)
    assert not ok
    assert "Unreleased" in msg


def test_src_change_with_unreleased_growth_passes():
    ok, _ = _LINT.verdict(True, _BASE, _GREW)
    assert ok


def test_release_transaction_is_exempt():
    # [Unreleased] emptied but a new version header appeared → exempt.
    ok, msg = _LINT.verdict(True, _BASE, _RELEASED)
    assert ok
    assert "release" in msg.lower()


def test_unreleased_lines_extraction():
    assert _LINT._unreleased_lines(_GREW) == {"- a new entry"}
    assert _LINT._unreleased_lines(_BASE) == set()
