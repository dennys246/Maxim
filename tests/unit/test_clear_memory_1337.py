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


def test_clear_memory_all_leaves_no_agent_state_in_any_agent_home(tmp_path: Path, monkeypatch) -> None:
    from maxim.cli_utils import clear_memory

    data_home = tmp_path / "home"
    _populate(data_home, monkeypatch)

    clear_memory("all")

    assert _memory_left(data_home) == []


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


def test_clear_memory_without_yes_off_a_terminal_refuses_and_deletes_nothing(tmp_path: Path, monkeypatch) -> None:
    data_home, cwd = tmp_path / "home", tmp_path / "work"
    cwd.mkdir()
    _populate(data_home, monkeypatch)
    before = _memory_left(data_home)

    proc = _clear(data_home, cwd)

    assert proc.returncode != 0
    assert "--yes" in proc.stderr
    assert _memory_left(data_home) == before


# ── The one definition of an agent's state ─────────────────────────────────────────────────────


def test_the_clear_types_cover_exactly_what_an_agent_home_persists() -> None:
    """``_AGENT_FILES`` splits ``AGENT_STATE_FILES`` (+ ``memory_hub/``) by type: a store file added there without a
    clear type would survive ``--clear-memory all`` silently."""
    from maxim.cli_utils import _AGENT_FILES
    from maxim.runtime.agent_factory import AGENT_STATE_FILES

    by_type = {name for names in _AGENT_FILES.values() for name in names}
    assert set(AGENT_STATE_FILES) | {"memory_hub"} <= by_type
    assert by_type - set(AGENT_STATE_FILES) - {"memory_hub"} == {"nac.meta.json"}  # hive merge's NAc record


def test_every_agent_type_is_cleared_in_every_agent_home() -> None:
    from maxim.cli_utils import _AGENT_FILES, AGENT_HOMES, MEMORY_PATHS

    for mem_type, names in _AGENT_FILES.items():
        for home in AGENT_HOMES:
            for name in names:
                assert f"{home}/{name}" in MEMORY_PATHS[mem_type]


@pytest.mark.parametrize(
    ("mem_type", "rel"),
    [
        ("bounds", "util/learned_bounds.json"),  # spatial/bounds_learner.py (the old key named an unwritten file)
        ("significance", "util/significance_weights.json"),
        ("short_term", "short_term_memory/stm_1.json"),
        ("tool_index", "memory/tool_index.json"),
        ("nac", "memory/nac.meta.json"),
        ("memory_hub", "agents/scout/memory_hub/hub.json"),
        ("hippo", "sessions/hippocampus.json.backup"),
    ],
)
def test_each_learner_the_inventory_found_is_cleared(tmp_path: Path, mem_type: str, rel: str) -> None:
    from maxim.cli_utils import clear_memory

    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")

    assert clear_memory(mem_type, home_dir=str(tmp_path))[mem_type] is True
    assert not target.exists()


# ── run_clear_memory: the confirmation and the bases ───────────────────────────────────────────


@pytest.fixture
def homes(tmp_path: Path, monkeypatch):
    """A data home and a working directory, each holding one NAc/EC pair; ``./data`` is the robot path's default."""
    from maxim.utils import paths

    data_home, cwd = tmp_path / "home", tmp_path / "work"
    for d in (data_home / "agents" / "scout", cwd / "data" / "memory"):
        d.mkdir(parents=True)
        for name in ("nac.json", "ec.json"):
            (d / name).write_text("{}")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(data_home))
    monkeypatch.chdir(cwd)
    paths._reset_caches()
    yield data_home, cwd
    paths._reset_caches()


def _run(scope: str = "all", *, explicit: bool = False, home_dir: str = "data", **kw) -> int:
    from maxim.cli_utils import run_clear_memory

    kw.setdefault("assume_yes", False)
    return run_clear_memory(scope, home_dir=home_dir, home_dir_explicit=explicit, **kw)


def test_a_terminal_yes_deletes_and_a_no_deletes_nothing(homes) -> None:
    data_home, _ = homes
    nac = data_home / "agents" / "scout" / "nac.json"
    assert _run(stdin_is_tty=True, input_fn=lambda _p: "n") == 1
    assert nac.exists()
    assert _run(stdin_is_tty=True, input_fn=lambda _p: "y") == 0
    assert not nac.exists()


def test_end_of_input_at_the_prompt_is_a_no(homes) -> None:
    def _eof(_prompt: str) -> str:
        raise EOFError

    data_home, _ = homes
    assert _run(stdin_is_tty=True, input_fn=_eof) == 1
    assert (data_home / "agents" / "scout" / "nac.json").exists()


def test_off_a_terminal_it_refuses_without_yes_and_never_prompts(homes, capsys) -> None:
    def _no_prompt(_prompt: str) -> str:
        raise AssertionError("prompted off a terminal")

    data_home, _ = homes
    assert _run(stdin_is_tty=False, input_fn=_no_prompt) == 2
    assert "--yes" in capsys.readouterr().err
    assert (data_home / "agents" / "scout" / "nac.json").exists()


def test_the_listing_names_every_path_before_asking(homes, capsys) -> None:
    data_home, _ = homes
    _run(stdin_is_tty=True, input_fn=lambda _p: "n")
    out = capsys.readouterr().err  # stderr: a redirected stdout must not hide the question (review fold)
    assert (
        str(data_home / "agents" / "scout" / "nac.json") in out
        and str(data_home / "agents" / "scout" / "ec.json") in out
    )


def test_an_unnamed_dot_data_is_warned_about_and_left(homes, capsys) -> None:
    _, cwd = homes
    assert _run(assume_yes=True) == 0
    err = capsys.readouterr().err
    assert "--home-dir data" in err and str(cwd / "data" / "memory" / "nac.json") in err
    assert (cwd / "data" / "memory" / "nac.json").exists()


def test_a_named_home_dir_is_cleared_with_the_data_home(homes) -> None:
    data_home, cwd = homes
    assert _run(assume_yes=True, explicit=True) == 0
    assert not (cwd / "data" / "memory" / "nac.json").exists()
    assert not (data_home / "agents" / "scout" / "nac.json").exists()


def test_an_unknown_type_deletes_nothing(homes, capsys) -> None:
    data_home, _ = homes
    assert _run("nac,bogus", assume_yes=True) == 2
    assert "bogus" in capsys.readouterr().err
    assert (data_home / "agents" / "scout" / "nac.json").exists()


def test_nothing_to_clear_exits_zero(homes) -> None:
    assert _run("planning", assume_yes=True) == 0


def test_a_failed_deletion_exits_nonzero(homes, monkeypatch, capsys) -> None:
    def _denied(self, *a, **kw):
        raise PermissionError("read-only")

    monkeypatch.setattr(Path, "unlink", _denied)
    assert _run(assume_yes=True) == 1
    assert "read-only" in capsys.readouterr().err


def test_sim_reports_are_never_memory(homes) -> None:
    data_home, _ = homes
    report = data_home / "sim_reports" / "s1" / "aut_hippocampus.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    assert _run(assume_yes=True) == 0
    assert report.exists()


def test_explicit_flag_reads_both_spellings_and_not_an_abbreviation() -> None:
    from maxim.cli_utils import explicit_flag

    assert explicit_flag(["--home-dir", "x"], "--home-dir") and explicit_flag(["--home-dir=x"], "--home-dir")
    assert not explicit_flag(["--home", "x"], "--home-dir") and not explicit_flag(None, "--home-dir")


def test_the_question_goes_to_stderr_not_stdout(homes, capsys) -> None:
    _run(stdin_is_tty=True, input_fn=lambda _p: "n")
    captured = capsys.readouterr()
    assert "[y/N]" in captured.err and "[y/N]" not in captured.out


@pytest.mark.parametrize("scope", ["ALL", " all "])
def test_all_is_case_and_space_insensitive(homes, scope) -> None:
    data_home, _ = homes
    assert _run(scope, assume_yes=True) == 0
    assert not (data_home / "agents" / "scout" / "nac.json").exists()


def test_an_empty_name_in_a_list_is_ignored(homes) -> None:
    data_home, _ = homes
    assert _run("nac,", assume_yes=True) == 0
    assert not (data_home / "agents" / "scout" / "ec.json").exists()  # its pair


def test_a_symlinked_store_is_unlinked_not_followed(homes, tmp_path) -> None:
    data_home, _ = homes
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.json").write_text("{}")
    (data_home / "agents" / "scout" / "memory_hub").symlink_to(outside, target_is_directory=True)

    assert _run(assume_yes=True) == 0
    assert not os.path.lexists(data_home / "agents" / "scout" / "memory_hub")
    assert (outside / "keep.json").exists()


def test_a_named_home_dir_inside_the_data_home_lists_each_path_once(homes, capsys) -> None:
    data_home, _ = homes
    _run(assume_yes=True, explicit=True, home_dir=str(data_home / "agents" / "scout" / ".." / ".."))
    err = capsys.readouterr().err
    assert err.count(str(data_home / "agents" / "scout" / "nac.json")) == 1


def test_a_clear_memory_run_is_never_saved_for_replay() -> None:
    from maxim.utils.last_run import should_save

    for argv in (["--mode", "live", "--clear-memory", "--yes"], ["--mode", "live", "--clear-memory=all", "--yes"]):
        assert should_save(argv) is False
    assert should_save(["--mode=live"]) is True


# ── The shipped CLI's --home-dir wiring (Executor fold: a unit call never reaches cli.py) ─────


def _dot_data_memory(cwd: Path) -> Path:
    nac = cwd / "data" / "memory" / "nac.json"
    nac.parent.mkdir(parents=True)
    nac.write_text("{}")
    (nac.parent / "ec.json").write_text("{}")
    return nac


def test_the_cli_leaves_an_unnamed_dot_data_and_says_so(tmp_path: Path) -> None:
    data_home, cwd = tmp_path / "home", tmp_path / "work"
    data_home.mkdir()
    cwd.mkdir()
    nac = _dot_data_memory(cwd)

    proc = _clear(data_home, cwd, "--yes")

    assert proc.returncode == 0, proc.stderr
    assert nac.exists()
    assert "--home-dir data" in proc.stderr and str(nac.resolve()) in proc.stderr


def test_the_cli_clears_a_named_home_dir(tmp_path: Path) -> None:
    data_home, cwd = tmp_path / "home", tmp_path / "work"
    data_home.mkdir()
    cwd.mkdir()
    nac = _dot_data_memory(cwd)

    proc = _clear(data_home, cwd, "--home-dir", "data", "--yes")

    assert proc.returncode == 0, proc.stderr
    assert not nac.exists()


# ── No writer builds an agent home outside AGENT_HOME_PATTERNS (Architecture fold) ─────────────


def _literal_agent_homes() -> dict[str, set[str]]:
    """Every agent home a writer in src/ names statically: a ``persistence_dir=`` / ``base_data_dir=`` argument
    built from ``user_memory()``, ``agent_data(...)`` or ``data_home() / "<literal>"``."""
    import ast

    def home_of(node: ast.AST) -> str | None:
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                if name == "user_memory":
                    return "memory"
                if name == "agent_data":
                    return "agents/*"
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) and isinstance(n.left, ast.Call):
                name = n.left.func.id if isinstance(n.left.func, ast.Name) else getattr(n.left.func, "attr", "")
                if name in ("data_home", "_data_home") and isinstance(n.right, ast.Constant):
                    return str(n.right.value)
        return None

    found: dict[str, set[str]] = {}
    for path in sorted((REPO_SRC / "maxim").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in ("persistence_dir", "base_data_dir") and (home := home_of(kw.value)):
                        found.setdefault(home, set()).add(str(path.relative_to(REPO_SRC)))
    return found


def test_every_agent_home_a_writer_names_is_one_clear_memory_clears() -> None:
    from maxim.utils.paths import AGENT_HOME_PATTERNS

    found = _literal_agent_homes()
    assert {"memory", "orchestrator"} <= set(found), found  # the scan still sees the known writers
    assert set(found) <= set(AGENT_HOME_PATTERNS), {h: f for h, f in found.items() if h not in AGENT_HOME_PATTERNS}


def test_the_legacy_api_agent_dir_is_an_agent_home() -> None:
    from maxim import api
    from maxim.utils.paths import AGENT_HOME_PATTERNS, RUN_DIR_KINDS

    assert api._LEGACY_API_AGENT_DIR in AGENT_HOME_PATTERNS
    assert f"{RUN_DIR_KINDS['agent']}/*" in AGENT_HOME_PATTERNS


@pytest.mark.parametrize("scope", ["", ",", " "])
def test_an_empty_type_list_refuses(homes, scope, capsys) -> None:
    data_home, _ = homes
    assert _run(scope, assume_yes=True) == 2
    assert (data_home / "agents" / "scout" / "nac.json").exists()
