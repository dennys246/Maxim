"""``--research`` under llm-primary (and ``--sim research``) runs the research protocol, whose own sim call takes
none of the goal-string sim's embodiment, curation, sim-mode, full-turn, sandbox or resume options. Each one passed
explicitly is refused before anything runs (exit 2, ``CliOptionsRefused``) instead of being dropped;
``--sim-max-turns`` is forwarded, except with ``--campaign``, which delivers every line whatever the cap (#1680)."""

from __future__ import annotations

import ast
import inspect
import pickle
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
    (["--auto-curate"], "--auto-curate"),
    (["--sim-run-full-turns"], "--sim-run-full-turns"),
    (["--continuous"], "--continuous"),
    (["--sandbox-image", "python:3.11-slim"], "--sandbox-image"),
    (["--sandbox-network", "bridge"], "--sandbox-network"),
    (["--resume-sim", "S1"], "--resume-sim"),
    (["--sim-mode", "dm"], "--sim-mode"),
    (["--sim-mode", "generative"], "--sim-mode"),
    (["--campaign", "c.yaml", "--sim-max-turns", "3"], "--sim-max-turns"),
]
SIMS = [["--sim", "warm the cradle", "--research"], ["--sim", "research"]]


def _args(*argv: str):
    return _build_parser().parse_args(["--sim", "warm the cradle", "--research", *argv])


@pytest.mark.parametrize(("argv", "flag"), DROPPED, ids=[" ".join(a) for a, _ in DROPPED])
def test_each_option_the_research_protocol_would_drop_is_refused(argv, flag) -> None:
    refusal = cli._research_refusal(_args(*argv))
    assert isinstance(refusal, CliOptionsRefused)
    assert flag in str(refusal) and flag in refusal.fix_hint


@pytest.mark.parametrize(
    "argv",
    [
        ["--sim-max-turns", "7", "--aut-model", "m", "--seed", "3"],
        ["--campaign", "c.yaml"],
        ["--sim-mode", "research"],  # what the protocol runs
        ["--no-sim-env"],  # what the protocol always does
    ],
    ids=["forwarded", "campaign", "sim-mode-research", "no-sim-env"],
)
def test_the_options_it_takes_are_not_refused(argv) -> None:
    assert cli._research_refusal(_args(*argv)) is None


def _main(monkeypatch, tmp_path, argv, research):
    # cli.main saves a --sim invocation under the data home: keep it in tmp, never the suite's shared HOME.
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr("maxim.simulation.research_orchestrator.start_research_mode", research)
    warned = []
    monkeypatch.setattr("maxim.utils.logging.user_warn", lambda message, **kw: warned.append(message))
    try:
        code = cli.main([*argv, "--interactive", "false"])
    except SystemExit as exc:
        code = exc.code
    assert not (Path.home() / ".maxim" / "last_runs.json").exists()  # nothing leaked into the suite's HOME
    return code, warned


def _never(**kw):
    pytest.fail("the research protocol ran with options it drops")


@pytest.mark.parametrize("sim", SIMS, ids=["goal-research", "sim-research"])
@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["--embodiment", "bodies/x"], "--embodiment"),
        (["--sim-mode", "dm"], "--sim-mode"),
        (["--sim-mode", "generative"], "--sim-mode"),
        (["--campaign", "c.yaml", "--sim-max-turns", "3"], "--sim-max-turns"),
    ],
    ids=["embodiment", "sim-mode-dm", "sim-mode-generative", "campaign-cap"],
)
def test_the_cli_refuses_with_exit_2_before_the_protocol_runs(monkeypatch, tmp_path, sim, extra, flag) -> None:
    code, warned = _main(monkeypatch, tmp_path, [*sim, *extra], _never)
    assert code == 2
    assert warned and "CliOptionsRefused" in warned[0] and flag in warned[0]


def test_the_refusal_comes_before_curation_writes_anything(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "maxim.simulation.foundry.auto_curate", lambda **kw: pytest.fail("curation ran before the refusal")
    )
    code, warned = _main(monkeypatch, tmp_path, [*SIMS[0], "--auto-curate", "--embodiment", "bodies/x"], _never)
    assert code == 2 and "--auto-curate" in warned[0] and "--embodiment" in warned[0]


def _accepting(seen):
    def research(**kw):
        seen.update(kw)
        return SimpleNamespace(finish_reason="completed", review_verdict="accept")

    return research


@pytest.mark.parametrize("sim", SIMS, ids=["goal-research", "sim-research"])
@pytest.mark.parametrize(
    ("extra", "expected_turns"),
    [(["--sim-max-turns", "7"], 7), (["--sim-mode", "research"], 50), (["--no-sim-env"], 50)],
    ids=["max-turns-forwarded", "sim-mode-research", "no-sim-env"],
)
def test_what_it_takes_runs(monkeypatch, tmp_path, sim, extra, expected_turns) -> None:
    seen: dict = {}
    code, warned = _main(monkeypatch, tmp_path, [*sim, *extra], _accepting(seen))
    assert code == 0 and not warned and seen["max_turns"] == expected_turns


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


#: Read by ``_main_impl`` for a goal-string sim before it dispatches (the embodiment ``_run_goal_sim`` is handed, and
#: the curation run before it), so not visible in ``_run_goal_sim``'s own reads.
_MAIN_IMPL_GOAL_SIM_READS = {"embodiment", "no_embodiment", "deep_embodiment", "auto_curate"}
#: What the research protocol does on its own, whatever is passed: ``mode="research"`` (``--sim-mode`` other than
#: ``research`` is refused separately) and ``no_sim_env=True``.
_HONOURED_BY_DESIGN = {"sim_mode", "no_sim_env"}


def test_every_goal_sim_option_is_forwarded_or_refused() -> None:
    """A goal-sim option must be forwarded to the research protocol (read by ``_run_research_from_cli``) or refused
    by it, never dropped: the drift that let --embodiment and --sim-max-turns vanish under --research (#1680)."""
    read = (
        _args_reads(cli._run_goal_sim)
        | _args_reads(cli._sim_turn_caps)
        | _args_reads(cli._resolve_mode)
        | _args_reads(cli._cli_llm_profile)
        | _MAIN_IMPL_GOAL_SIM_READS
    )
    forwarded = _args_reads(cli._run_research_from_cli)
    missing = read - forwarded - _HONOURED_BY_DESIGN - cli._RESEARCH_DROPPED_OPTIONS
    assert not missing, f"the goal sim reads {sorted(missing)}: forward them to start_research_mode or refuse them"


@pytest.mark.xfail(strict=True, reason="#1680: CliOptionsRefused adopts _rebuild_with_state after #1621 merges")
def test_the_refusal_round_trips_through_pickle() -> None:
    refusal = CliOptionsRefused("--embodiment cannot run", fix_hint="Drop --embodiment.")
    back = pickle.loads(pickle.dumps(refusal))
    assert str(back) == str(refusal) and back.fix_hint == refusal.fix_hint
