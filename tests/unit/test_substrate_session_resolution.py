"""One answer to "which directory is run X" (roadmap 1.3.1): ``utils/paths.py::resolve_run_dir``.

``maxim substrate``/``hive pull`` looked a bare ``--session`` ID up in ``<data home>/sessions/``, which
nothing writes by ID -- a simulation's substrate lives in ``sim_reports/<id>/`` and a
``maxim.create.agent()`` home in ``agents/<name>/`` -- so neither ever resolved; ``maxim roy diff`` had its
own copy that swallowed every lookup error. Driven through the real CLI entry points.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    from maxim.utils.paths import _reset_caches

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)  # the working directory is a searched place: keep it empty
    _reset_caches()
    yield tmp_path / "home"
    _reset_caches()


def _sim(root, sid: str):
    d = root / "sim_reports" / sid
    d.mkdir(parents=True)
    (d / "aut_nac.json").write_text(json.dumps({"links": {}}))
    (d / "aut_ec.json").write_text(json.dumps({"substrate_nodes": {}}))
    return d


def _census(arg: str, capsys) -> tuple[int, str]:
    from maxim.hivemind.cli import run_substrate_subcommand

    capsys.readouterr()
    code = run_substrate_subcommand(["invalidate", "--session", arg])
    out, err = capsys.readouterr()
    return code, out + err


def test_a_simulations_id_resolves(data_home, capsys) -> None:
    d = _sim(data_home, "20260927_120000")
    code, text = _census("20260927_120000", capsys)
    assert code == 0, text
    assert str(d.resolve()) in text


def test_an_agents_name_resolves_as_a_receiver(data_home) -> None:
    """`hive pull --session <name>` promised a maxim.create.agent() home; the name never resolved."""
    import maxim
    from maxim.hivemind.cli import _expand_session_dir, _resolve_receiver_pair

    agent = maxim.create.agent(name="scout")
    agent.shutdown()
    home = _expand_session_dir("scout", agents=True)  # ingest's lookup (hive pull goes through it)
    assert home == (data_home / "agents" / "scout").resolve()
    assert _resolve_receiver_pair(home) is not None  # the receiver layout ingest reads


def test_an_id_in_two_places_is_refused_not_guessed(data_home, tmp_path, capsys) -> None:
    _sim(data_home, "dup")
    (tmp_path / "dup").mkdir()  # the working directory holds a different run of the same name
    code, text = _census("dup", capsys)
    assert code == 2 and "more than one directory" in text


def test_an_unknown_id_names_every_place_searched(data_home, tmp_path) -> None:
    from maxim.utils.paths import RunDirNotFound, resolve_run_dir

    with pytest.raises(RunDirNotFound) as exc:
        resolve_run_dir("nope", kinds=("sim", "agent"))
    for place in (tmp_path / "nope", data_home / "sim_reports" / "nope", data_home / "agents" / "nope"):
        assert str(place) in str(exc.value)


def test_a_path_is_only_a_path(data_home, tmp_path, capsys) -> None:
    d = _sim(tmp_path / "elsewhere", "x")
    assert _census(str(d), capsys)[0] == 0
    code, text = _census(str(tmp_path / "missing"), capsys)
    assert code == 2 and text.strip().endswith(f"directory not found: {tmp_path / 'missing'}")


def test_roy_diff_uses_the_shared_lookup_and_says_why_it_failed(data_home, capsys) -> None:
    from maxim.roy.cli import _run_diff

    assert _run_diff(["nope_a", "nope_b"]) == 2
    assert "looked in" in capsys.readouterr().err  # its own copy swallowed the lookup and printed a bare path


def test_the_lookup_creates_nothing(data_home) -> None:
    from maxim.utils.paths import RunDirNotFound, resolve_run_dir

    with pytest.raises(RunDirNotFound):
        resolve_run_dir("nope")
    assert not (data_home / "sim_reports").exists() and not (data_home / "agents").exists()


def test_export_takes_no_agent_name(data_home, capsys) -> None:
    """export reads only a simulation's aut_*.json: an agent name must not resolve to a home it can't read."""
    from maxim.hivemind.cli import run_substrate_subcommand

    (data_home / "agents" / "scout").mkdir(parents=True)
    capsys.readouterr()
    code = run_substrate_subcommand(["export", "out.zip", "--session", "scout", "--contributor-id", "c"])
    err = capsys.readouterr().err
    assert code == 2 and "looked in" in err and str(data_home / "agents") not in err


def test_an_id_found_only_in_the_working_directory_resolves(data_home, tmp_path) -> None:
    from maxim.utils.paths import resolve_run_dir

    (tmp_path / "local_run").mkdir()
    assert resolve_run_dir("local_run") == (tmp_path / "local_run").resolve()


def test_running_from_inside_sim_reports_is_not_ambiguous(data_home, monkeypatch) -> None:
    from maxim.utils.paths import resolve_run_dir

    d = _sim(data_home, "20260927_130000")
    monkeypatch.chdir(data_home / "sim_reports")  # the working directory's hit IS the data-home hit
    assert resolve_run_dir("20260927_130000") == d.resolve()


@pytest.mark.parametrize("arg", ["", "   "])
def test_an_empty_argument_is_refused_not_the_working_directory(data_home, arg) -> None:
    from maxim.utils.paths import RunDirNotFound, resolve_run_dir

    with pytest.raises(RunDirNotFound, match="empty"):
        resolve_run_dir(arg)


def test_ingest_resolves_an_agents_name_through_the_cli(data_home, tmp_path, capsys) -> None:
    """The wiring, not the helper: `substrate ingest --session <agent name>` gets past the lookup to the
    bundle (hive pull builds exactly this argv). Before, it stopped at "directory not found"."""
    import maxim
    from maxim.hivemind.cli import run_substrate_subcommand

    maxim.create.agent(name="scout").shutdown()
    bundle = tmp_path / "not_a_bundle.zip"
    bundle.write_bytes(b"not a zip")
    capsys.readouterr()
    code = run_substrate_subcommand(
        ["ingest", str(bundle), "--session", "scout", "--trust", "c", "--receiver-body", "minecraft_player"]
    )
    err = capsys.readouterr().err
    assert code != 0  # the junk bundle is refused...
    assert "looked in" not in err and "directory not found" not in err, err  # ...but the receiver resolved


@pytest.mark.parametrize("arg", [".scratch", "~/definitely_not_a_run_dir"])
def test_a_dot_or_tilde_argument_is_a_path_not_an_id(data_home, arg) -> None:
    """Path-shaped means path only: `.scratch` is never searched as an ID, even where one exists."""
    from maxim.utils.paths import RunDirNotFound, resolve_run_dir

    _sim(data_home, arg.lstrip("~/"))
    with pytest.raises(RunDirNotFound, match="directory not found"):
        resolve_run_dir(arg)
