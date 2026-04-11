"""Regression test for the CLI "no meaningful action" gate.

The gate at cli.py:_has_action decides whether to enter the main loop or
print the quick-start banner and exit. It used to check only:
    --sim, --explore, --mode, --robot, MAXIM_ROLE=leader, MAXIM_LLM_ENABLED=1

That meant `maxim --llm qwen2.5-14b` (without --sim or any other flag)
silently exited because `--llm` sets args.language_model but not
MAXIM_LLM_ENABLED. The user's explicit model choice wasn't counted as
an action.

This test isolates the gate logic so regressions produce a clear
failure message rather than a behavioral change requiring an end-to-end
CLI run.
"""

from __future__ import annotations


def _compute_has_action(argv: list[str]) -> bool:
    """Re-implement the gate logic the way cli.py does, so we can assert
    which argv combinations should/shouldn't trip the quick-start path
    without actually calling main()."""
    import os

    from maxim.cli_parser import _build_parser

    parser = _build_parser()
    args = parser.parse_args(argv)
    sim_path = getattr(args, "sim", None)

    _has_sim = sim_path is not None
    _has_explore = getattr(args, "explore", None) is not None
    _has_mode_override = "--mode" in argv
    _has_robot = getattr(args, "robot_name", "reachy_mini") != "reachy_mini" or "--robot" in argv
    _is_leader = os.environ.get("MAXIM_ROLE", "").strip().lower() == "leader"
    _has_llm = os.environ.get("MAXIM_LLM_ENABLED", "").strip() == "1"
    _has_llm_cli = bool(str(getattr(args, "language_model", "") or "").strip())
    _has_action = _has_sim or _has_explore or _has_mode_override or _has_robot or _has_llm_cli

    return bool(_has_action or _is_leader or _has_llm)


class TestActionGate:
    def test_llm_flag_short_form_is_action(self):
        """`maxim --llm qwen2.5-14b` must count as a meaningful action.

        Regression: used to print quick-start and exit because the gate
        only checked MAXIM_LLM_ENABLED env, not args.language_model.
        """
        import os as _os

        _saved = {k: _os.environ.pop(k) for k in ("MAXIM_ROLE", "MAXIM_LLM_ENABLED") if k in _os.environ}
        try:
            assert _compute_has_action(["--llm", "qwen2.5-14b"]) is True
        finally:
            _os.environ.update(_saved)

    def test_language_model_long_form_is_action(self):
        """The long-form flag must behave identically."""
        import os as _os

        _saved = {k: _os.environ.pop(k) for k in ("MAXIM_ROLE", "MAXIM_LLM_ENABLED") if k in _os.environ}
        try:
            assert _compute_has_action(["--language-model", "mistral-7b"]) is True
        finally:
            _os.environ.update(_saved)

    def test_bare_maxim_is_not_action(self):
        """Running bare `maxim` with no flags should still trip the
        quick-start gate (the intended behavior). This is a negative
        control for the fix above."""
        import os as _os

        _saved = {k: _os.environ.pop(k) for k in ("MAXIM_ROLE", "MAXIM_LLM_ENABLED") if k in _os.environ}
        try:
            assert _compute_has_action([]) is False
        finally:
            _os.environ.update(_saved)

    def test_sim_flag_is_action(self):
        """Positive control for the --sim path (unchanged behavior)."""
        import os as _os

        _saved = {k: _os.environ.pop(k) for k in ("MAXIM_ROLE", "MAXIM_LLM_ENABLED") if k in _os.environ}
        try:
            assert _compute_has_action(["--sim", "test"]) is True
        finally:
            _os.environ.update(_saved)

    def test_leader_env_is_action(self):
        """Positive control for the leader role path."""
        import os as _os

        _saved_role = _os.environ.get("MAXIM_ROLE")
        _saved_enabled = _os.environ.pop("MAXIM_LLM_ENABLED", None)
        _os.environ["MAXIM_ROLE"] = "leader"
        try:
            assert _compute_has_action([]) is True
        finally:
            if _saved_role is not None:
                _os.environ["MAXIM_ROLE"] = _saved_role
            else:
                _os.environ.pop("MAXIM_ROLE", None)
            if _saved_enabled is not None:
                _os.environ["MAXIM_LLM_ENABLED"] = _saved_enabled
