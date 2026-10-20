"""#1337: ``maxim --clear-memory`` clears the data home's memory, all of it, and asks first.

Before: the CLI passed ``--home-dir`` (default ``"data"``, relative to the working directory) as the base,
so a plain ``maxim --clear-memory`` never touched ``~/.maxim`` or ``MAXIM_DATA_HOME``. And ``MEMORY_PATHS``
was a third hand-kept list of an agent's files, missing ``cerebellum.json``, every agent home outside
``agents/`` (``memory/``, written by the Reachy runtime and ``hive merge``; ``orchestrator/``; the
pre-#932 ``sessions/``) and several learners.

Owner decisions 2026-10-19: confirm y/N at a terminal, refuse without ``--yes`` off one; always the data
home; ``--home-dir``'s stores only when it is passed explicitly, with a warning naming ``./data`` memory
otherwise; ``sim_reports/`` (run archives) is not memory.

The red gate runs the shipped CLI in a subprocess (a real non-TTY stdin) over stores written by the real
writers, so a store a writer adds later is caught by the writer, not by a copy of the list.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[2] / "src"


def _populate(data_home: Path, monkeypatch) -> None:
    """Persisted memory as the real writers lay it out, in every agent home kind."""
    import maxim
    from maxim.memory.encoding import EncodingSignals
    from maxim.memory.types import Perception
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.utils import paths

    monkeypatch.setenv("MAXIM_DATA_HOME", str(data_home))
    paths._reset_caches()
    for name, home in (
        ("scout", None),  # agents/scout/, maxim.create.agent's home
        ("sim_orchestrator", data_home / "orchestrator"),  # the orchestrator's flat home
        ("api_agent", data_home / "sessions"),  # the pre-#932 maxim.run() agent
    ):
        agent = maxim.create.agent(name, persistence_dir=str(home) if home else None)
        agent.hippocampus.capture(
            perception=Perception(observations={"text": "the vault code is seven-three-nine"}),
            encoding=EncodingSignals.unmeasured("api"),
        )
        agent.shutdown()
    # memory/: build_bio_stack(persistence_dir=user_memory()), the Reachy runtime and `hive merge`
    stack = build_bio_stack(persistence_dir=str(data_home / "memory"), agent_id="reachy")
    stack.on_session_end()  # writes memory/cerebellum.json
    assert (data_home / "memory" / "cerebellum.json").is_file()


def _clear(data_home: Path, cwd: Path, *argv: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MAXIM_")}
    env.update(MAXIM_DATA_HOME=str(data_home), MAXIM_ROLE="solo", PYTHONPATH=str(REPO_SRC), HOME=str(cwd))
    return subprocess.run(
        [sys.executable, "-m", "maxim", "--clear-memory", *argv],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _memory_left(data_home: Path) -> list[str]:
    from maxim.runtime.agent_factory import persisted_agent_state

    left: list[Path] = []
    for home in [
        *sorted((data_home / "agents").glob("*")),
        *(data_home / d for d in ("memory", "orchestrator", "sessions")),
    ]:
        left += persisted_agent_state(home)
    return sorted(str(p.relative_to(data_home)) for p in left)


@pytest.mark.xfail(strict=True, reason="#1337: MEMORY_PATHS misses cerebellum and every agent home outside agents/")
def test_clear_memory_all_leaves_no_agent_state_in_any_agent_home(tmp_path: Path, monkeypatch) -> None:
    from maxim.cli_utils import clear_memory

    data_home = tmp_path / "home"
    _populate(data_home, monkeypatch)

    clear_memory("all")

    assert _memory_left(data_home) == []


@pytest.mark.xfail(strict=True, reason="#1337: --clear-memory resolves ./data and misses agent homes")
def test_clear_memory_yes_clears_every_agent_home_in_the_data_home(tmp_path: Path, monkeypatch) -> None:
    data_home, cwd = tmp_path / "home", tmp_path / "work"
    cwd.mkdir()
    _populate(data_home, monkeypatch)
    keep = data_home / "util" / "internet_access.json"  # operator state, never memory
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text(json.dumps({"enabled": False}))
    assert _memory_left(data_home), "the fixture wrote no memory"

    proc = _clear(data_home, cwd, "--yes")

    assert proc.returncode == 0, proc.stderr
    assert _memory_left(data_home) == []
    assert keep.exists()


@pytest.mark.xfail(strict=True, reason="#1337: --clear-memory deletes without asking")
def test_clear_memory_without_yes_off_a_terminal_refuses_and_deletes_nothing(tmp_path: Path, monkeypatch) -> None:
    data_home, cwd = tmp_path / "home", tmp_path / "work"
    cwd.mkdir()
    _populate(data_home, monkeypatch)
    before = _memory_left(data_home)

    proc = _clear(data_home, cwd)

    assert proc.returncode != 0
    assert "--yes" in proc.stderr
    assert _memory_left(data_home) == before
