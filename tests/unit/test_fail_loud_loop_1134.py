"""#1134 (fail-loud Stage 3): the loop controller and loop state report every swallow and join the measurement path.

``runtime/loop_controller.py`` and ``runtime/loop_state.py`` received the agent loop's D13 handlers and wake
predicates in the 1.3.2 decomposition (slice 2) but could not join ``scripts/lint_no_silent_swallows.py``'s
``MEASUREMENT_PATH`` at zero: each carried a bare ``except Exception: pass`` and broad handlers that logged without
the Stage-1 report (``log_swallowed_exception``), so a failure there (the shape that hid #1133) left no event.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = ("src/maxim/runtime/loop_controller.py", "src/maxim/runtime/loop_state.py")


def _lint():
    spec = importlib.util.spec_from_file_location(
        "lint_no_silent_swallows", ROOT / "scripts" / "lint_no_silent_swallows.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rel", FILES)
def test_every_broad_handler_reports(rel: str) -> None:
    lint = _lint()
    text = (ROOT / rel).read_text(encoding="utf-8")
    assert (lint.swallow_hits(text), lint.unreported_swallow_hits(text)) == ([], [])


def test_both_files_are_on_the_measurement_path() -> None:
    assert set(FILES) <= set(_lint().MEASUREMENT_PATH)
