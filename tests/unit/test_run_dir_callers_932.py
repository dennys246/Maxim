"""The run-directory callers that still answered "which directory is run X" on their own (#932).

``resolve_run_dir`` (1.3.1) is the one answer; ``Session.from_disk`` returned the NEWEST prefix match
(``load.session("2026")`` silently picked one of many runs), ``scripts/check_oscillator_coldstart.py``
hard-coded ``~/.maxim`` (ignoring ``MAXIM_DATA_HOME``), and the research / campaign reports were
written to a working-directory-relative ``./data/sim_reports`` that nothing reads.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)  # the working directory is a searched place: keep it empty
    _reset_caches()
    yield tmp_path / "home"
    _reset_caches()


def _sim(root: Path, sid: str, goal: str = "") -> Path:
    d = root / "sim_reports" / sid
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"goal": goal}))
    return d


# ── load.session / Session.from_disk ────────────────────────────────────


def test_session_exact_id_resolves_even_when_it_prefixes_newer_runs(data_home) -> None:
    """The newest-prefix-match rule loaded ``..._b`` for ``load.session("20260927_120000")``."""
    import maxim

    _sim(data_home, "20260927_120000", goal="the one asked for")
    _sim(data_home, "20260927_120000_b", goal="a newer run it prefixes")
    s = maxim.load.session("20260927_120000")
    assert s.id == "20260927_120000"
    assert s.goal == "the one asked for"


def test_session_path_resolves(data_home, tmp_path) -> None:
    import maxim

    elsewhere = tmp_path / "archive" / "20260101_000000"
    elsewhere.mkdir(parents=True)
    (elsewhere / "report.json").write_text(json.dumps({"goal": "archived"}))
    s = maxim.load.session(str(elsewhere))
    assert s.id == "20260101_000000"
    assert Path(s.dir) == elsewhere.resolve()
    assert s.goal == "archived"


def test_session_unique_prefix_still_resolves(data_home) -> None:
    import maxim

    _sim(data_home, "20260408_143022", goal="only match")
    _sim(data_home, "20260901_000000")
    assert maxim.load.session("20260408").id == "20260408_143022"


def test_session_ambiguous_prefix_refuses_and_lists_the_matches(data_home) -> None:
    import maxim
    from maxim.utils.paths import RunDirAmbiguous

    _sim(data_home, "20260408_143022")
    _sim(data_home, "20260408_180000")
    with pytest.raises(RunDirAmbiguous) as exc:
        maxim.load.session("20260408")
    assert "20260408_143022" in str(exc.value)
    assert "20260408_180000" in str(exc.value)


def test_session_missing_raises_file_not_found(data_home, tmp_path) -> None:
    import maxim

    _sim(data_home, "20260408_143022")
    with pytest.raises(FileNotFoundError):
        maxim.load.session("20990101")
    with pytest.raises(FileNotFoundError):
        maxim.load.session(str(tmp_path / "no" / "such" / "run"))


def test_session_id_found_in_two_places_is_refused(data_home) -> None:
    """The exact-ID half goes through resolve_run_dir, including its ambiguity refusal."""
    import maxim
    from maxim.utils.paths import RunDirAmbiguous

    _sim(data_home, "20260927_120000")
    (Path.cwd() / "20260927_120000").mkdir()
    with pytest.raises(RunDirAmbiguous):
        maxim.load.session("20260927_120000")


# ── report writers follow the data home ─────────────────────────────────


def test_campaign_analysis_is_written_under_the_data_home(data_home, monkeypatch) -> None:
    from maxim.simulation import campaign_runner

    monkeypatch.setattr(campaign_runner.time, "sleep", lambda s: None)
    campaign_runner.run_precampaign_turns(turns=[], bridge=None, introspector=None)
    written = list((data_home / "sim_reports").glob("campaign_analysis_*.json"))
    assert len(written) == 1
    assert not (Path.cwd() / "data").exists()


def test_research_session_dir_is_under_the_data_home(data_home, monkeypatch) -> None:
    from maxim.simulation import research_orchestrator, research_tools

    seen: list[Path] = []

    class _Stop(Exception):
        pass

    class _Log:
        def __init__(self, *, session_dir, agent_nickname):
            seen.append(Path(session_dir))
            raise _Stop

    monkeypatch.setattr(research_tools, "ExperimentLog", _Log)
    with pytest.raises(_Stop):
        research_orchestrator.start_research_mode(goal="g")
    assert len(seen) == 1
    assert seen[0].parent == data_home / "sim_reports"
    assert seen[0].name.startswith("research_")
    assert seen[0].is_dir()
    assert not (Path.cwd() / "data").exists()


# ── scripts/check_oscillator_coldstart.py ───────────────────────────────


def _coldstart():
    spec = importlib.util.spec_from_file_location(
        "check_oscillator_coldstart_932", REPO / "scripts" / "check_oscillator_coldstart.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_coldstart_resolves_a_bare_id_under_the_data_home(data_home) -> None:
    d = _sim(data_home, "20260927_120000")
    (d / "aut_scn.json").write_text("{}")
    assert _coldstart()._resolve_scn_path("20260927_120000", None) == (d / "aut_scn.json").resolve()


def test_coldstart_refuses_an_ambiguous_id(data_home) -> None:
    from maxim.utils.paths import RunDirAmbiguous

    _sim(data_home, "20260927_120000")
    (Path.cwd() / "20260927_120000").mkdir()
    with pytest.raises(RunDirAmbiguous):
        _coldstart()._resolve_scn_path("20260927_120000", None)


# ── runtime/agent_factory.py::migrate_agent_state + atomic_io.atomic_install_dir ──


def _legacy(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "hippocampus.json").write_text('{"legacy": 1}')
    (root / "nac.json").write_text('{"legacy": 2}')
    return root


def test_migration_yields_to_an_agent_another_process_installed_first(tmp_path, monkeypatch) -> None:
    """The rename never lands inside a populated home (shutil.move nested it there, then the originals went)."""
    from maxim.runtime.agent_factory import migrate_agent_state
    from maxim.utils import atomic_io

    legacy = _legacy(tmp_path / "sessions")
    agent_dir = tmp_path / "agents" / "api_agent"
    real = atomic_io.atomic_install_dir

    def other_process_wins(staged, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "hippocampus.json").write_text('{"other": true}')
        real(staged, dest)

    monkeypatch.setattr(atomic_io, "atomic_install_dir", other_process_wins)
    assert migrate_agent_state(legacy, agent_dir) is False
    assert json.loads((agent_dir / "hippocampus.json").read_text()) == {"other": True}
    assert sorted(p.name for p in agent_dir.iterdir()) == ["hippocampus.json"]  # nothing nested inside
    assert (legacy / "hippocampus.json").exists() and (legacy / "nac.json").exists()
    assert [p.name for p in (tmp_path / "agents").iterdir()] == ["api_agent"]  # stage discarded


def test_install_dir_refuses_a_populated_destination(tmp_path) -> None:
    from maxim.utils.atomic_io import atomic_install_dir

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "a.json").write_text("{}")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "b.json").write_text("{}")
    with pytest.raises(OSError):
        atomic_install_dir(str(staged), str(dest))
    assert sorted(p.name for p in dest.iterdir()) == ["b.json"]


def test_install_dir_fsyncs_the_data_before_the_rename(tmp_path, monkeypatch) -> None:
    """The rename must not reach disk ahead of the copies: every staged file is fsynced first."""
    from maxim.utils import atomic_io

    staged = tmp_path / "staged"
    (staged / "hub").mkdir(parents=True)
    (staged / "a.json").write_text("{}")
    (staged / "hub" / "b.json").write_text("{}")
    events: list[str] = []
    real_fsync_path, real_rename = atomic_io._fsync_path, atomic_io.os.rename

    def fsync_path(path):
        events.append(f"fsync {path}")
        real_fsync_path(path)

    def rename(src, dst):
        events.append("rename")
        real_rename(src, dst)

    monkeypatch.setattr(atomic_io, "_fsync_path", fsync_path)
    monkeypatch.setattr(atomic_io.os, "rename", rename)
    atomic_io.atomic_install_dir(str(staged), str(tmp_path / "dest"))
    before = events[: events.index("rename")]
    for f in ("a.json", "hub/b.json"):
        assert f"fsync {staged / f}" in before
    assert f"fsync {staged}" in before
    assert events[-1] == f"fsync {tmp_path}"  # the parent, after
    assert (tmp_path / "dest" / "hub" / "b.json").exists()


def test_an_interrupt_after_the_rename_propagates_and_keeps_the_installed_agent(tmp_path, monkeypatch, caplog) -> None:
    """Ctrl-C after the rename is neither swallowed nor blamed on another process (round-2 review)."""
    from maxim.runtime.agent_factory import migrate_agent_state
    from maxim.utils import atomic_io

    legacy = _legacy(tmp_path / "sessions")
    agent_dir = tmp_path / "agents" / "api_agent"
    real = atomic_io._fsync_path

    def interrupt_on_parent(path):
        if path == str(agent_dir.parent):  # fsynced only after the rename
            raise KeyboardInterrupt
        real(path)

    monkeypatch.setattr(atomic_io, "_fsync_path", interrupt_on_parent)
    with caplog.at_level("INFO", logger="maxim.runtime.agent_factory"), pytest.raises(KeyboardInterrupt):
        migrate_agent_state(legacy, agent_dir)
    assert json.loads((agent_dir / "hippocampus.json").read_text()) == {"legacy": 1}
    messages = [r.getMessage() for r in caplog.records]
    assert any("interrupted before removing the originals" in m for m in messages)
    assert not any("another process" in m or "could not be put back" in m for m in messages)


def test_an_interrupt_is_never_swallowed_even_when_a_rival_installed(tmp_path, monkeypatch) -> None:
    from maxim.runtime.agent_factory import migrate_agent_state
    from maxim.utils import atomic_io

    legacy = _legacy(tmp_path / "sessions")
    agent_dir = tmp_path / "agents" / "api_agent"

    def rival_then_interrupt(staged, dest):
        Path(dest).mkdir(parents=True, exist_ok=True)
        (Path(dest) / "nac.json").write_text("{}")
        raise KeyboardInterrupt

    monkeypatch.setattr(atomic_io, "atomic_install_dir", rival_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        migrate_agent_state(legacy, agent_dir)
    assert (legacy / "hippocampus.json").exists()


def test_a_partly_failed_removal_loses_no_carried_file(tmp_path, monkeypatch) -> None:
    """The stage holds the only complete copy once removal starts: it is merged back, never discarded."""
    import shutil

    from maxim.runtime.agent_factory import migrate_agent_state

    legacy = _legacy(tmp_path / "sessions")
    agent_dir = tmp_path / "agents" / "api_agent"
    (agent_dir / "notes").mkdir(parents=True)
    (agent_dir / "notes" / "a.txt").write_text("a")
    (agent_dir / "notes" / "b.txt").write_text("b")
    real_rmtree = shutil.rmtree

    def half_rmtree(path, *a, **k):
        if Path(path) == agent_dir / "notes":
            (agent_dir / "notes" / "a.txt").unlink()
            raise PermissionError("half way")
        return real_rmtree(path, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", half_rmtree)
    with pytest.raises(PermissionError):
        migrate_agent_state(legacy, agent_dir)
    assert (agent_dir / "notes" / "a.txt").read_text() == "a"
    assert (agent_dir / "notes" / "b.txt").read_text() == "b"
    assert (legacy / "hippocampus.json").exists()
    assert not (agent_dir / "hippocampus.json").exists()


def test_a_rival_installing_mid_migration_keeps_every_file_it_wrote(tmp_path, monkeypatch) -> None:
    """The collision cleanup removes empty directories only: a rival's hub state is never deleted."""
    import shutil

    from maxim.runtime.agent_factory import migrate_agent_state

    legacy = _legacy(tmp_path / "sessions")
    (legacy / "memory_hub").mkdir()
    (legacy / "memory_hub" / "hub.json").write_text("{}")
    agent_dir = tmp_path / "agents" / "api_agent"
    (agent_dir / "memory_hub").mkdir(parents=True)  # empty: no state, collides with the staged hub
    real_copy2 = shutil.copy2

    def rival_during_copy(src, dst, *a, **k):
        if not (agent_dir / "nac.json").exists():
            (agent_dir / "memory_hub" / "rival.json").write_text('{"rival": true}')
            (agent_dir / "nac.json").write_text('{"rival": true}')
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copy2", rival_during_copy)
    assert migrate_agent_state(legacy, agent_dir) is False
    assert json.loads((agent_dir / "memory_hub" / "rival.json").read_text()) == {"rival": True}
    assert json.loads((agent_dir / "nac.json").read_text()) == {"rival": True}
    assert (legacy / "memory_hub" / "hub.json").exists()


def test_the_stage_is_kept_when_a_carried_file_cannot_be_put_back(tmp_path, monkeypatch) -> None:
    import shutil

    from maxim.runtime.agent_factory import migrate_agent_state

    legacy = _legacy(tmp_path / "sessions")
    agent_dir = tmp_path / "agents" / "api_agent"
    (agent_dir / "notes").mkdir(parents=True)
    (agent_dir / "notes" / "a.txt").write_text("a")
    real_rmtree, real_copytree = shutil.rmtree, shutil.copytree

    def half_rmtree(path, *a, **k):
        if Path(path) == agent_dir / "notes":
            (agent_dir / "notes" / "a.txt").unlink()
            raise PermissionError("half way")
        return real_rmtree(path, *a, **k)

    def no_restore(src, dst, *a, **k):
        if Path(dst) == agent_dir / "notes":
            raise PermissionError("read-only")
        return real_copytree(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "rmtree", half_rmtree)
    monkeypatch.setattr(shutil, "copytree", no_restore)
    with pytest.raises(PermissionError):
        migrate_agent_state(legacy, agent_dir)
    kept = [p for p in (tmp_path / "agents").iterdir() if p.name.startswith(".api_agent.migrating-")]
    assert len(kept) == 1 and (kept[0] / "notes" / "a.txt").read_text() == "a"


def test_run_dir_errors_are_maxim_errors_and_value_errors(data_home) -> None:
    """Exported from ``maxim``, so inside the stable hierarchy (round-2 review), and still ValueErrors for
    the CLI handlers that catch them as such."""
    import maxim
    from maxim.utils import paths

    assert paths.RunDirAmbiguous is maxim.RunDirAmbiguous
    for cls in (paths.RunDirAmbiguous, paths.RunDirNotFound):
        assert issubclass(cls, maxim.MaximMemoryError) and issubclass(cls, ValueError)
    _sim(data_home, "20260408_143022")
    _sim(data_home, "20260408_180000")
    with pytest.raises(maxim.MaximError):
        maxim.load.session("20260408")


def test_the_downgrade_backup_advice_names_every_run_directory() -> None:
    """``docs/user/upgrading.md`` told users to back up ``~/.maxim/sessions/``, which no simulation writes
    (#940 item 3). Known answer: the backup sentence names each directory ``resolve_run_dir`` searches,
    and the sentence before its correction note does not name ``sessions``."""
    from maxim.utils.paths import RUN_DIR_KINDS

    text = (REPO / "docs" / "user" / "upgrading.md").read_text()
    sentence = next(line for line in text.splitlines() if line.startswith("If you anticipate a possible downgrade"))
    advice = sentence.split("*(Corrected", 1)[0]
    for directory in RUN_DIR_KINDS.values():
        assert f"`~/.maxim/{directory}/`" in advice, directory
    assert "sessions" not in advice
