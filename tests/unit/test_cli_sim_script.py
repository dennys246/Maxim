"""``--sim-script <file>``: the scripted narrator (Exp 63c's engine change; prereg §Apparatus).

The script's lines reach ``start_simulation_mode(pre_campaign_turns=...)`` (the shipped path: ``run_precampaign_turns``
-> ``bridge.send_and_wait``), loaded by ``experiment.load_campaign_turns(strict=True)``. It is accepted only on the
plain goal-string orchestrator call (a whitelist) and refused, before anything starts, everywhere else.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from maxim import cli
from maxim.cli_parser import _build_parser
from maxim.cli_utils import normalize_args
from maxim.simulation.experiment import load_campaign_turns

GOAL = "escape a dungeon with a sleeping guard"  # Exp 63c's goal: selects no arc
LINES = ["The guard is asleep in the dungeon.", "What comes to mind?", "Sum up your intention in a sentence."]


def _script(tmp_path: Path, lines=LINES, name: str = "s.yaml") -> Path:
    body = "percepts:\n" + "".join(f"  - cli_input: {line!r}\n" for line in lines)
    f = tmp_path / name
    f.write_text(body)
    return f


def _args(*argv: str):
    args = _build_parser().parse_args(list(argv))
    normalize_args(args)
    return args


def _ok_argv(script: Path, *extra: str, turns: int = len(LINES)) -> list[str]:
    return ["--sim", GOAL, "--sim-script", str(script), "--interactive", "false", "--sim-max-turns", str(turns), *extra]


def test_without_the_flag_nothing_is_loaded() -> None:
    assert cli._sim_script_turns(_args("--sim", GOAL)) is None


def test_the_plain_goal_call_loads_the_scripts_turns(tmp_path) -> None:
    f = _script(tmp_path)
    turns = cli._sim_script_turns(_args(*_ok_argv(f)))
    assert turns == load_campaign_turns(str(f))  # the existing loader's dicts, unchanged (the 63c content pin)
    assert [t["text"] for t in turns] == LINES
    assert {(t["salience"], t["novelty"]) for t in turns} == {(0.8, 0.7)}  # send_and_wait's own defaults


def test_it_combines_with_resume_sim(tmp_path) -> None:
    assert cli._sim_script_turns(_args(*_ok_argv(_script(tmp_path), "--resume-sim", "S1"))) is not None


def _refused(capsys, args) -> str:
    with pytest.raises(SystemExit) as exc:
        cli._sim_script_turns(args)
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: --sim-script refused: "), err
    return err


@pytest.mark.parametrize(
    ("argv", "reason"),
    [
        (["--sim-script", "S"], "goal string"),  # no --sim at all
        (["--sim", "scenarios/x.yaml", "--sim-script", "S"], "goal string"),  # a YAML --sim (ScenarioSource)
        (["--sim", "agent", "--sim-script", "S"], "goal string"),
        (["--sim", "research", "--sim-script", "S"], "goal string"),
        (["--sim", "benchmark", "--sim-script", "S"], "goal string"),
        (["--sim", "--sim-script", "S"], "goal string"),  # the REPL
        ([*_ok_argv(Path("S")), "--dm"], "--dm"),
        ([*_ok_argv(Path("S")), "--research"], "--research"),
        ([*_ok_argv(Path("S")), "--continuous"], "--continuous"),
        ([*_ok_argv(Path("S")), "--campaign", "c.yaml"], "--campaign"),
        (["--sim", GOAL, "--sim-script", "S", "--interactive", "true", "--sim-max-turns", "3"], "interactive"),
        (["--sim", GOAL, "--sim-script", "S", "--interactive", "false"], "--sim-max-turns is required"),
        (["--sim", GOAL, "--sim-script", "S", "--interactive", "false", "--sim-max-turns", "0"], "at least 1"),
        (["--sim", "  ", "--sim-script", "S", "--interactive", "false", "--sim-max-turns", "3"], "goal string"),
        # A WHITELIST: a dispatch that runs before the goal-string branch must not drop the script silently.
        ([*_ok_argv(Path("S")), "--foundry", "fantasy"], "--foundry"),
        ([*_ok_argv(Path("S")), "--list-models"], "--list-models"),
        ([*_ok_argv(Path("S")), "--embodiment", "bodies/base_humanoid"], "--embodiment"),
        ([*_ok_argv(Path("S")), "--benchmark", "tier1"], "--benchmark"),
        ([*_ok_argv(Path("S")), "--goal", "other"], "--sim-goal"),
        ([*_ok_argv(Path("S")), "--sim-mode", "adversarial"], "--sim-mode"),
    ],
)
def test_every_other_path_is_refused(capsys, argv, reason) -> None:
    assert reason in _refused(capsys, _args(*argv))


def test_interactive_auto_on_a_tty_is_refused(capsys, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    args = _args("--sim", GOAL, "--sim-script", str(_script(tmp_path)), "--sim-max-turns", "3")
    assert "interactive" in _refused(capsys, args)


def test_a_goal_that_selects_an_arc_is_refused(capsys, tmp_path) -> None:
    from maxim.simulation.arcs import BUILTIN_ARCS

    arc_goal = sorted(BUILTIN_ARCS)[0]  # an exact arc name always selects its arc
    assert "narrative arc" in _refused(capsys, _args("--sim", arc_goal, *_ok_argv(_script(tmp_path))[2:]))


def test_an_arc_probe_that_raises_refuses_never_reads_as_no_arc(capsys, monkeypatch, tmp_path) -> None:
    import maxim.simulation.arcs as arcs

    def boom(goal):
        raise RuntimeError("arc table unreadable")

    monkeypatch.setattr(arcs, "select_arc_for_goal", boom)
    assert "arc table unreadable" in _refused(capsys, _args(*_ok_argv(_script(tmp_path))))


@pytest.mark.parametrize(
    ("lines", "turns", "reason"),
    [
        (LINES, 4, "has 3 lines but --sim-max-turns is 4"),  # the runner ignores the cap
        (LINES, 2, "has 3 lines but --sim-max-turns is 2"),
        (["one", "", "three"], 3, "not a non-blank line"),  # a blank line: the default loader would drop it
        (["one", "   ", "three"], 3, "not a non-blank line"),  # whitespace-only: unaddressed, publishes nothing
    ],
)
def test_the_script_itself_is_checked(capsys, tmp_path, lines, turns, reason) -> None:
    args = _args(*_ok_argv(_script(tmp_path, lines), turns=turns))
    assert reason in _refused(capsys, args)


@pytest.mark.parametrize(
    "body",
    [
        "percepts: {}\n",
        "[1, 2]\n",
        "percepts:\n  - just a string\n",
        "percepts:\n  - cli_input: 7\n",
        "percepts:\n  - at: 0\n",
        "percepts:\n  - cli_input: hi\n    metadata: null\n",
        "percepts: [unclosed\n",  # not YAML: a refusal, never a traceback
    ],
)
def test_a_malformed_script_is_refused(capsys, tmp_path, body) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text(body)
    assert "could not be loaded" in _refused(capsys, _args(*_ok_argv(f, turns=1)))


def test_a_missing_script_is_refused(capsys, tmp_path) -> None:
    assert "could not be loaded" in _refused(capsys, _args(*_ok_argv(tmp_path / "nope.yaml")))


def test_the_default_loader_still_skips_a_blank_line(tmp_path) -> None:
    """``strict`` is opt-in: ``run_campaign``'s loader behaviour is unchanged (wiring S5)."""
    assert [t["text"] for t in load_campaign_turns(str(_script(tmp_path, ["one", "", "three"])))] == ["one", "three"]


def test_the_turns_reach_start_simulation_mode_through_the_cli(monkeypatch, tmp_path) -> None:
    """The production entry point: ``cli.main`` hands the script's turns to the orchestrator as ``pre_campaign_turns``
    with the resume id and the turn cap, and leaves the finish reason to the runner (no relabel)."""
    import maxim.simulation.orchestrator as orch

    seen: dict = {}

    class _Started(Exception):
        pass

    def fake_start(**kwargs):
        seen.update(kwargs)
        raise _Started

    monkeypatch.setattr(orch, "start_simulation_mode", fake_start)
    f = _script(tmp_path)
    with pytest.raises(_Started):
        cli.main([*_ok_argv(f, "--resume-sim", "S1"), "--sandbox", "tmpdir"])
    assert seen["pre_campaign_turns"] == load_campaign_turns(str(f))
    assert (seen["goal"], seen["resume_session"], seen["max_turns"]) == (GOAL, "S1", len(LINES))
    assert seen["generative"] is False and seen["continuous"] is False


def test_a_refusal_through_the_real_entry_point_starts_nothing(tmp_path) -> None:
    """``python -m maxim`` refuses with exit 1 and starts no sim (no report directory is written)."""
    from tests._child_env import isolation_env

    home = tmp_path / "home"
    env = {
        **isolation_env(),
        "MAXIM_DATA_HOME": str(home),
        "MAXIM_OFFLINE": "1",
        "MAXIM_ROLE": "solo",  # the box's own role never decides a test (a leader boot refuses without its key)
        "HOME": str(tmp_path),
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "maxim", *_ok_argv(_script(tmp_path), turns=9)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**env, "PYTHONPATH": str(Path(cli.__file__).resolve().parents[1])},
    )
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert "--sim-script refused: the script has 3 lines but --sim-max-turns is 9" in proc.stderr
    assert not (home / "sim_reports").exists()


def test_an_empty_script_never_becomes_a_narrator_run(capsys, tmp_path) -> None:
    """``[]`` is falsy at ``start_simulation_mode``'s ``if pre_campaign_turns:``: an empty script with a cap of 0 would
    run the narrator for 50 turns. Strict loading refuses it, and so does the cap."""
    f = tmp_path / "empty.yaml"
    f.write_text("percepts: []\n")
    assert "at least 1" in _refused(capsys, _args(*_ok_argv(f, turns=0)))
    with pytest.raises(ValueError, match="non-empty"):
        load_campaign_turns(str(f), strict=True)


def test_the_plain_63c_argv_passes_the_whitelist(tmp_path) -> None:
    """Exp 63c's phase argv (o19_verdict.phase_argv plus --sim-script) carries no option outside the whitelist."""
    args = _args(*_ok_argv(_script(tmp_path), "--resume-sim", "S1"))
    assert cli._sim_script_unexpected_options(args) == []


def test_the_whitelist_names_real_options() -> None:
    """Every whitelisted name is a parser option (a typo would whitelist nothing), and none is also exempted as
    normalized; any option NOT listed is refused with --sim-script (fail closed), including one added later."""
    from maxim.cli_parser import _build_parser

    dests = {a.dest for a in _build_parser()._actions if a.option_strings and a.dest != "help"}
    assert cli._SIM_SCRIPT_ALLOWED <= dests, cli._SIM_SCRIPT_ALLOWED - dests
    assert not (cli._SIM_SCRIPT_ALLOWED & cli._SIM_SCRIPT_NORMALIZED)


def test_resolve_sim_script_has_no_side_effects(capsys, tmp_path) -> None:
    """The harness-callable form: a reason, never an exit or output (Exp 63c's preflight runs it before its marker)."""
    turns, reason = cli.resolve_sim_script(_args(*_ok_argv(_script(tmp_path), turns=9)))
    assert turns is None and "has 3 lines but --sim-max-turns is 9" in reason
    turns, reason = cli.resolve_sim_script(_args(*_ok_argv(_script(tmp_path))))
    assert reason is None and [t["text"] for t in turns] == LINES
    assert cli.resolve_sim_script(_args("--sim", GOAL)) == (None, None)
    assert capsys.readouterr() == ("", "")


def test_resolve_sim_script_refuses_raw_args(tmp_path) -> None:
    """Un-normalized args would read --interactive false as the truthy string "false": a TypeError, never a guess."""
    raw = _build_parser().parse_args(_ok_argv(_script(tmp_path)))
    with pytest.raises(TypeError, match="normalize_args"):
        cli.resolve_sim_script(raw)


def test_normalizing_the_63c_argv_writes_nothing(monkeypatch, tmp_path) -> None:
    """A harness preflight normalizes the 63c argv before its marker: that must persist nothing and export no env
    (``normalize_args`` writes only behind --llm, --llm-n-ctx, --auto-download, the cloud and segmentation flags)."""
    import os

    import maxim.runtime.lane_backends as lane_backends

    writes: list = []
    monkeypatch.setattr(lane_backends, "_write_persisted_model", lambda *a, **k: writes.append(a))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    script = _script(tmp_path)
    before_env = dict(os.environ)
    before_files = sorted(p for p in tmp_path.rglob("*"))
    turns, reason = cli.resolve_sim_script(_args(*_ok_argv(script, "--resume-sim", "S1")))
    assert reason is None and turns
    assert dict(os.environ) == before_env and writes == []
    assert sorted(p for p in tmp_path.rglob("*")) == before_files
