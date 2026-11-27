"""Regression tests for the CLI quick-start gate: menu, server mode, or a run.

The gate decides whether ``maxim`` enters a run or shows the quick-start menu. ``maxim --llm qwen2.5-14b``
used to exit silently because ``--llm`` set ``args.language_model`` but the gate only checked
``MAXIM_LLM_ENABLED``. #1806 PR 0 made it read PARSED, EXPLICIT flags only: before, it compared the parsed
``--robot-name`` (whose argparse default was ``$MAXIM_ROBOT_NAME``) against ``reachy_mini`` and grepped the raw
argv for ``--robot``, so ``maxim --robot-name reachy_mini`` and ``maxim --robot=reachy_mini`` showed the menu
while a bare ``maxim`` with ``MAXIM_ROBOT_NAME`` set entered the robot runtime.

These tests call the real gate (``cli._quick_start_route`` / ``cli._requested_run_action``), not a copy of it.
"""

from __future__ import annotations

import pytest

from maxim.cli import _quick_start_route, _requested_run_action
from maxim.cli_parser import _build_parser


@pytest.fixture(autouse=True)
def _no_route_env(monkeypatch):
    for name in ("MAXIM_ROLE", "MAXIM_LLM_ENABLED", "MAXIM_ROBOT_NAME"):
        monkeypatch.delenv(name, raising=False)


def _route(argv: list[str]) -> str:
    args = _build_parser().parse_args(argv)
    return _quick_start_route(args, argv, sim_path=getattr(args, "sim", None))


class TestActionGate:
    def test_llm_flag_short_form_is_action(self):
        """`maxim --llm qwen2.5-14b` must count as a meaningful action (it used to print the menu and exit)."""
        assert _route(["--llm", "qwen2.5-14b"]) == "run"

    def test_language_model_long_form_is_action(self):
        assert _route(["--language-model", "mistral-7b"]) == "run"

    def test_bare_maxim_is_not_action(self):
        """Negative control: a bare `maxim` shows the menu."""
        assert _route([]) == "menu"

    def test_sim_flag_is_action(self):
        assert _route(["--sim", "test"]) == "run"

    @pytest.mark.parametrize("argv", [["--mode", "live"], ["--mode=exploration"]])
    def test_explicit_mode_is_action(self, argv):
        """Even `--mode exploration`, the parser's default value, counts once typed."""
        assert _route(argv) == "run"

    def test_leader_env_is_server_mode(self, monkeypatch):
        """A leader without an explicit action serves instead of entering a run."""
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        assert _route([]) == "server"

    def test_llm_enabled_env_is_server_mode(self, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_ENABLED", "1")
        assert _route([]) == "server"

    def test_an_action_beats_the_leader_server(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        assert _route(["--sim", "test"]) == "run"


class TestRobotFlagsGate:
    """#1806 PR 0 (wire F4): a typed robot flag is an action whatever its value; the environment never is."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["--robot-name", "reachy_mini"],
            ["--robot-name=reachy_mini"],
            ["--robot", "reachy_mini"],
            ["--robot=reachy_mini"],
            ["--robot", "alice"],
        ],
    )
    def test_a_typed_robot_name_enters_the_run(self, argv):
        """The pre-fix gate sent the default name, typed, to the menu (`!= "reachy_mini"` and an exact `--robot`
        token), so `--robot-name reachy_mini` and `--robot=reachy_mini` never reached the robot runtime."""
        assert _route(argv) == "run"

    def test_maxim_robot_name_alone_shows_the_menu(self, monkeypatch):
        """The pre-fix parser defaulted `--robot-name` to `$MAXIM_ROBOT_NAME`, so a bare `maxim` with it set (to
        anything but `reachy_mini`) entered the robot runtime."""
        monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
        assert _route([]) == "menu"
        args = _build_parser().parse_args([])
        assert args.robot_name is None
        assert _requested_run_action(args, [], sim_path=None) is False

    def test_maxim_robot_name_does_not_turn_a_leader_into_a_robot_run(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ROBOT_NAME", "alice")
        monkeypatch.setenv("MAXIM_ROLE", "leader")
        assert _route([]) == "server"
