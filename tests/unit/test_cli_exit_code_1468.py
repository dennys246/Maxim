"""#1468: a robot session that stops on an exception must exit non-zero.

``cli.py::_main_impl``'s run loop catches any exception from the robot runtime, logs ``❌ Maxim stopped`` and
breaks out of the loop. It used to ``return 0`` after it: loud for a human at the terminal, invisible to a supervisor
or script (systemd, maxim-pulse's app, CI), which read the exit code. A missing robot module fails at import since
#1436, and this path turned that into exit 0 as well. The review round (owner decisions 2026-10-26) settled the
table: a crash exits 1; a typed backend error or leader refusal reaches ``cli.main`` and exits 2 with its fix or
reason; an operator's Ctrl+C exits 130 wherever it lands.

These drive the REAL ``cli.main`` down the robot path, with ``embodied_runtime.selfy.Maxim`` replaced by a class
that raises on construction (the runtime resolves it by module attribute at call time).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_cli_process_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """``cli.main`` writes process state a later test reads: ``last_runs.json`` under the data home (measured
    2026-10-26: ``test_last_run.py::TestDataHome`` failed after these ran in the full suite) and the env vars it
    exports at startup (``MAXIM_ROLE``). Each test gets its own HOME and data home, and ``os.environ`` is put back
    exactly as it was, whatever the run wrote."""
    from maxim.utils.paths import _reset_caches

    saved_env = dict(os.environ)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    _reset_caches()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_env)
        _reset_caches()


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
def test_an_operator_interrupt_exits_130_1468(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One code for one operator action (owner decision 2026-10-26): a Ctrl+C inside a run exits 130, as one that
    lands outside it does (``cli.main``'s handler), the Unix convention for SIGINT."""
    assert _run_robot_session(monkeypatch, tmp_path, KeyboardInterrupt()) == 130


@pytest.mark.timeout(120)
def test_a_backend_error_inside_a_run_exits_2_with_its_fix_hint_1468(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A typed backend error (here: auth) reaches ``cli.main``, which renders its ``fix_hint`` and exits 2 (owner
    decision 2026-10-26): a supervisor can tell "the operator must act" (2) from "crashed" (1)."""
    from maxim.models.language.types import BackendAuthFailed
    from maxim.utils import logging as maxim_logging

    warned: list[dict] = []
    monkeypatch.setattr(maxim_logging, "user_warn", lambda message, **kw: warned.append({"message": message, **kw}))
    assert _run_robot_session(monkeypatch, tmp_path, BackendAuthFailed("the provider refused the key")) == 2
    assert [w["fix"] for w in warned] == [BackendAuthFailed.fix_hint]


@pytest.mark.timeout(120)
def test_a_leader_refusal_inside_a_run_exits_2_and_says_why_1468(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``LeaderRefusal`` is a security refusal, not an availability fallback: a broad ``except Exception`` must not
    absorb it (its own docstring). ``cli.main`` prints it and exits 2, as server mode and the early leader boot do."""
    from maxim.exceptions import LeaderRefusal

    assert _run_robot_session(monkeypatch, tmp_path, LeaderRefusal("no usable leader key")) == 2
    assert "no usable leader key" in capsys.readouterr().err


@pytest.mark.timeout(120)
def test_a_run_that_fails_in_live_exits_nonzero_and_still_shuts_down_1468(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The commoner real failure: the runtime builds, then ``live()`` raises. The session exits 1, and the loop's
    teardown (``cli._end_robot_run``) still stops the runtime it built."""
    from maxim import cli
    from maxim.embodied_runtime import selfy

    shutdowns: list[str] = []

    class _FailsLive:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def live(self, **kwargs: object) -> None:
            raise RuntimeError("the robot's media loop died")

        def shutdown(self) -> None:
            shutdowns.append("shutdown")

    monkeypatch.setattr(selfy, "Maxim", _FailsLive)
    rc = cli.main(["--mode", "live", "--home-dir", str(tmp_path / "robot"), "--interactive", "false"])
    assert rc == 1
    assert shutdowns == ["shutdown"]
