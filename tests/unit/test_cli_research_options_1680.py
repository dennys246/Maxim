"""``--research`` under llm-primary (and ``--sim research``) runs the research protocol, whose own sim call takes
none of the goal-string sim's embodiment, sim-mode, full-turn, sandbox or resume options. Each one passed explicitly
is refused (exit 2, ``CliOptionsRefused``) instead of being dropped; ``--sim-max-turns`` is forwarded (#1680)."""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from maxim import cli
from maxim.cli_parser import _build_parser
from maxim.exceptions import CliOptionsRefused

DROPPED = [
    (["--embodiment", "bodies/infant_humanoid_chilled"], "--embodiment"),
    (["--no-embodiment"], "--no-embodiment"),
    (["--deep-embodiment"], "--deep-embodiment"),
    (["--sim-run-full-turns"], "--sim-run-full-turns"),
    (["--continuous"], "--continuous"),
    (["--no-sim-env"], "--no-sim-env"),
    (["--sandbox-image", "python:3.11-slim"], "--sandbox-image"),
    (["--sandbox-network", "bridge"], "--sandbox-network"),
    (["--resume-sim", "S1"], "--resume-sim"),
    (["--sim-mode", "dm"], "--sim-mode"),
]


def _args(*argv: str):
    return _build_parser().parse_args(["--sim", "warm the cradle", "--research", *argv])


@pytest.mark.parametrize(("argv", "flag"), DROPPED, ids=[f for _, f in DROPPED])
def test_each_option_the_research_protocol_would_drop_is_refused(argv, flag) -> None:
    refusal = cli._research_refusal(_args(*argv))
    assert isinstance(refusal, CliOptionsRefused)
    assert flag in str(refusal) and flag in refusal.fix_hint


def test_the_options_it_takes_are_not_refused() -> None:
    args = _args("--sim-max-turns", "7", "--campaign", "c.yaml", "--aut-model", "m", "--seed", "3")
    assert cli._research_refusal(args) is None


@pytest.mark.parametrize("sim", [["--sim", "warm the cradle", "--research"], ["--sim", "research"]])
def test_the_cli_refuses_with_exit_2_before_the_protocol_runs(monkeypatch, tmp_path, sim) -> None:
    # cli.main saves a --sim invocation under the data home: keep it in tmp, never the suite's shared HOME.
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        "maxim.simulation.research_orchestrator.start_research_mode",
        lambda **kw: pytest.fail("the research protocol ran with options it drops"),
    )
    warned = []
    monkeypatch.setattr("maxim.utils.logging.user_warn", lambda message, **kw: warned.append(message))
    code = cli.main([*sim, "--embodiment", "bodies/x", "--interactive", "false"])
    assert code == 2
    assert warned and "CliOptionsRefused" in warned[0] and "--embodiment" in warned[0]
    assert not (Path.home() / ".maxim" / "last_runs.json").exists()  # nothing leaked into the suite's HOME


@pytest.mark.parametrize("sim", [["--sim", "warm the cradle", "--research"], ["--sim", "research"]])
def test_sim_max_turns_is_forwarded(monkeypatch, tmp_path, sim) -> None:
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    seen = {}

    def research(**kw):
        seen.update(kw)
        return SimpleNamespace(finish_reason="completed", review_verdict="accept")

    monkeypatch.setattr("maxim.simulation.research_orchestrator.start_research_mode", research)
    with pytest.raises(SystemExit) as exc:
        cli.main([*sim, "--sim-max-turns", "7", "--interactive", "false"])
    assert exc.value.code == 0 and seen["max_turns"] == 7


def _args_reads(func) -> set[str]:
    """Every ``args.<name>`` / ``getattr(args, "<name>", ...)`` a function reads."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
            names.add(node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == "args"
            and isinstance(node.args[1], ast.Constant)
        ):
            names.add(node.args[1].value)
    return names


def test_every_goal_sim_option_is_forwarded_or_refused() -> None:
    """A new option ``_run_goal_sim`` honours must be forwarded to the research protocol or refused by it, never
    dropped: the drift that let --embodiment and --sim-max-turns vanish under --research (#1680)."""
    read = (
        _args_reads(cli._run_goal_sim)
        | _args_reads(cli._sim_turn_caps)
        | _args_reads(cli._resolve_mode)
        | _args_reads(cli._cli_llm_profile)
    )
    forwarded = {"sim_max_turns", "language_model", "aut_model", "sandbox_backend"} | {"campaign", "sim_goal"}
    checked_on_their_own = {"sim_mode"}
    missing = read - forwarded - checked_on_their_own - cli._RESEARCH_DROPPED_OPTIONS
    assert not missing, f"_run_goal_sim reads {sorted(missing)}: forward them to start_research_mode or refuse them"
