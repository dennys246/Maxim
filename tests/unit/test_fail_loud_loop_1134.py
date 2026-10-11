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


# ── each loop_controller handler, made to fire: it reports and keeps its fallback ─────────────────────


def _controller():
    from unittest.mock import MagicMock

    from maxim.runtime.executor import Executor
    from maxim.runtime.loop_controller import LoopController

    state = MagicMock()
    state.data = {}
    executor = MagicMock(spec=Executor)
    executor.effective_operational_mode.return_value = None
    return LoopController(
        agent=MagicMock(),
        environment=MagicMock(),
        state=state,
        memory=MagicMock(),
        decision_engine=MagicMock(),
        executor=executor,
        autonomy_controller=MagicMock(),
    )


@pytest.fixture
def reports(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Every ``log_swallowed_exception`` call the controller makes, passed through to the real reporter."""
    import maxim.runtime.loop_controller as lc

    calls: list[dict] = []
    real = lc.log_swallowed_exception

    def _spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(lc, "log_swallowed_exception", _spy)
    return calls


def test_a_failing_prompt_reset_is_reported_and_never_raises(monkeypatch: pytest.MonkeyPatch, reports) -> None:
    from unittest.mock import MagicMock

    import maxim.simulation.sim_logger as sim_logger
    from maxim.runtime.loop_controller import LoopController

    display = MagicMock()
    display.set_prompt.side_effect = RuntimeError("display gone")
    monkeypatch.setattr(sim_logger, "get_active_display", lambda: display)
    LoopController._clear_confirmation_prompt()
    assert len(reports) == 1


def test_a_failing_memory_record_is_reported_and_the_input_still_lands(reports) -> None:
    ctrl = _controller()
    ctrl.memory.record_command.side_effect = RuntimeError("memory down")
    ctrl.store_user_input("look around", "cli")
    assert ctrl.state.data["pending_user_input"] == "look around"
    assert reports == [{"context": {"source_type": "cli"}}]


def test_a_failing_prefetch_is_reported_and_clears_the_prefetch(reports) -> None:
    from unittest.mock import MagicMock

    ctrl = _controller()
    ctrl.prefetcher = MagicMock()
    ctrl.prefetcher.prefetch_for_input.side_effect = RuntimeError("fs gone")
    ctrl.pending_prefetch = object()
    ctrl.run_prefetch("open notes.txt")
    assert ctrl.pending_prefetch is None
    assert len(reports) == 1
