"""#1806 PR 0 (CLI part): the legacy robot flags, the shared run-mode constant, the startup log's robot name and
the exp49 harness env.

- ``--robot`` is an explicit alias of ``--robot-name`` (same dest), so it no longer works only as an argparse
  prefix abbreviation, and ``--rob``/``--robo`` become ambiguous (wire N7).
- ``--robot-name`` defaults to ``None``: ``MAXIM_ROBOT_NAME`` is read where the name is resolved (wire F4).
- ``--mode``'s choices come from ``maxim.modes.definitions.RUN_MODES``; ``EMBODIED_RUN_MODES`` is the body subset.
- The startup log names the resolved robot name, never ``None``.
- ``scripts/exp49/run_trials.py::_trial_env`` drops ``MAXIM_ROBOT_NAME`` so a trial child is never a named robot run.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
from pathlib import Path

import pytest

from maxim import cli
from maxim.cli import _robot_name_for_log
from maxim.cli_parser import _build_parser
from maxim.modes.definitions import EMBODIED_RUN_MODES, RUN_MODES

REPO = Path(__file__).resolve().parents[2]


# ── --robot / --robot-name ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv, expected",
    [
        (["--robot", "alice"], "alice"),
        (["--robot=alice"], "alice"),
        (["--robot-name", "alice"], "alice"),
        (["--robot-name=alice"], "alice"),
    ],
)
def test_robot_is_an_alias_of_robot_name(argv, expected) -> None:
    assert _build_parser().parse_args(argv).robot_name == expected


@pytest.mark.parametrize("prefix", ["--rob", "--robo"])
def test_robot_prefixes_are_ambiguous(prefix, capsys) -> None:
    """Defining ``--robot`` explicitly makes the shorter prefixes match two options, which argparse refuses."""
    with pytest.raises(SystemExit) as exc:
        _build_parser().parse_args([prefix, "alice"])
    assert exc.value.code == 2
    assert "ambiguous" in capsys.readouterr().err


def test_robot_name_default_is_none_whatever_the_environment(monkeypatch) -> None:
    """The pre-fix default was ``os.environ.get("MAXIM_ROBOT_NAME", "reachy_mini")``, read at parser build time."""
    monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
    assert _build_parser().parse_args([]).robot_name is None
    monkeypatch.delenv("MAXIM_ROBOT_NAME")
    assert _build_parser().parse_args([]).robot_name is None


# ── the shared run-mode constant ────────────────────────────────────────────


def _mode_action():
    parser = _build_parser()
    return next(a for a in parser._actions if "--mode" in a.option_strings)


def test_mode_choices_are_the_shared_run_modes() -> None:
    assert tuple(_mode_action().choices) == RUN_MODES
    assert _mode_action().default in RUN_MODES


def test_embodied_run_modes_are_the_body_subset() -> None:
    assert EMBODIED_RUN_MODES == {"exploration", "live", "sleep", "train", "reflection"}
    assert EMBODIED_RUN_MODES < set(RUN_MODES)
    assert "agentic" in RUN_MODES and "agentic" not in EMBODIED_RUN_MODES


def test_the_run_mode_module_imports_no_cli_or_runtime_layer() -> None:
    """A future ``maxim.devices`` package reads these constants and must not reach cli/runtime/tools/agents."""
    src = (REPO / "src" / "maxim" / "modes" / "definitions.py").read_text(encoding="utf-8")
    forbidden = ("cli", "cli_parser", "cli_utils", "runtime", "tools", "agents", "planning", "console")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("maxim."):
            assert node.module.split(".")[1] not in forbidden, node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("maxim."):
                    assert alias.name.split(".")[1] not in forbidden, alias.name


# ── the startup log's robot name ────────────────────────────────────────────


def test_the_log_name_is_the_typed_name(monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
    assert _robot_name_for_log(_build_parser().parse_args(["--robot", "bob"])) == "bob (--robot-name)"


def test_the_log_name_falls_back_to_the_environment_then_the_default(monkeypatch) -> None:
    args = _build_parser().parse_args([])
    monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
    assert _robot_name_for_log(args) == "alice ($MAXIM_ROBOT_NAME)"
    monkeypatch.delenv("MAXIM_ROBOT_NAME")
    assert _robot_name_for_log(args) == "reachy_mini (default)"
    assert "None" not in _robot_name_for_log(args)


def test_the_startup_log_prints_the_resolved_name_not_the_parsed_one() -> None:
    """``args.robot_name`` is None for an untyped run, so the "Starting Maxim" line must not print it raw."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(cli._main_impl)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and str(node.args[0].value).startswith("Starting Maxim (robot_name=")
    ]
    assert len(calls) == 1
    name_arg = calls[0].args[1]
    assert isinstance(name_arg, ast.Call) and getattr(name_arg.func, "id", None) == "_robot_name_for_log"


# ── exp49 harness env ───────────────────────────────────────────────────────


def _load_run_trials():
    spec = importlib.util.spec_from_file_location(
        "exp49_run_trials_1806_pr0", REPO / "scripts" / "exp49" / "run_trials.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("arm", ["A", "C"])
def test_an_exp49_trial_child_is_never_a_named_robot_run(monkeypatch, tmp_path, arm) -> None:
    monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
    env = _load_run_trials()._trial_env(arm, tmp_path, tmp_path / "t.jsonl", min_confidence=None)
    assert "MAXIM_ROBOT_NAME" not in env


def test_the_daemon_name_chain_is_one_function_shared_by_selfy_and_the_log(monkeypatch) -> None:
    """#1806 PR 0: selfy dials and the CLI logs the SAME chain; an empty value counts as unset."""
    from maxim.hardware.config import resolve_daemon_name

    monkeypatch.setenv("MAXIM_ROBOT_NAME", "")
    assert resolve_daemon_name(None) == ("reachy_mini", "default")
    assert resolve_daemon_name("") == ("reachy_mini", "default")
    monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
    assert resolve_daemon_name(None) == ("alice", "$MAXIM_ROBOT_NAME")
    assert resolve_daemon_name("bob") == ("bob", "--robot-name")

    import inspect

    import maxim.embodied_runtime.selfy as selfy

    assert "resolve_daemon_name(robot_name)" in inspect.getsource(selfy.Maxim.__init__)
    assert 'os.getenv("MAXIM_ROBOT_NAME"' not in inspect.getsource(selfy.Maxim.__init__)
