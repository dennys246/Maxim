"""#1468: a robot session that stops on an exception must exit non-zero.

``cli.py::_main_impl``'s run loop catches any exception from the robot runtime, logs ``❌ Maxim stopped`` and
breaks out of the loop. It used to ``return 0`` after it: loud for a human at the terminal, invisible to a supervisor
or script (systemd, maxim-pulse's app, CI), which read the exit code. A missing robot module fails at import since
#1436, and this path turned that into exit 0 as well.

These drive the REAL ``cli.main`` down the robot path, with ``embodied_runtime.selfy.Maxim`` replaced by a class
that raises on construction (the runtime resolves it by module attribute at call time).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _run_robot_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, raised: BaseException) -> int:
    from maxim import cli
    from maxim.embodied_runtime import selfy

    class _Raises:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise raised

    monkeypatch.setattr(selfy, "Maxim", _Raises)
    return cli.main(["--mode", "live", "--home-dir", str(tmp_path / "robot"), "--interactive", "false"])


@pytest.mark.timeout(120)
def test_a_robot_session_that_stops_on_an_exception_exits_nonzero_1468(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert _run_robot_session(monkeypatch, tmp_path, RuntimeError("reachy unreachable")) == 1


@pytest.mark.timeout(120)
def test_an_operator_interrupt_still_exits_zero_1468(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ctrl+C is the operator ending the session, not a failure: it keeps exiting 0."""
    assert _run_robot_session(monkeypatch, tmp_path, KeyboardInterrupt()) == 0
