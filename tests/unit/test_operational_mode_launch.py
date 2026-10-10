"""#829 -- one vocabulary per axis, and an operator launch grant for the operational mode.

The CLI has two axes: the RUN mode (`--mode`: exploration, agentic, sleep, live, train, reflection) and
the OPERATIONAL mode (passive / active / singularity: capability). A runtime request for an operational
name re-exec'd with `--mode <name>`, which argparse rejects (exit 2), and there was no way to launch
with a chosen capability. Owner decisions 2026-09-28: a separate `--operational-mode` launch flag (the
human grant #924's record promises; singularity reachable only this way, loudly); runtime operational
requests only lower; `--mode agentic` gets a definition and an unknown mode fails closed.
"""

from __future__ import annotations

import argparse
import logging
from unittest.mock import MagicMock

import pytest

from maxim.cli import (
    _RUNTIME_SWITCHABLE_MODES,
    _after_run_mode_request,
    _apply_operational_grant,
    _current_operational_mode,
    _explicit_mode_flag,
    _runtime_mode_switch_allowed,
    _validate_operational_grant,
)
from maxim.cli_parser import _build_parser
from maxim.cli_utils import reexec_with_mode


def _args(**overrides) -> argparse.Namespace:
    ns = _build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


def _reexec_argv(monkeypatch, args, mode: str) -> list[str]:
    captured: list[list[str]] = []
    monkeypatch.setattr("maxim.cli_utils.os.execv", lambda exe, argv: captured.append(list(argv)))
    reexec_with_mode(args, mode=mode)
    assert captured, "reexec_with_mode did not exec"
    argv = captured[0]
    assert argv[1:3] == ["-m", "maxim.cli"]
    return argv[3:]


@pytest.mark.parametrize("requested", _RUNTIME_SWITCHABLE_MODES)
def test_every_runtime_switchable_name_survives_argparse(monkeypatch, requested) -> None:
    """The issue's own acceptance test: the re-exec a runtime request produces parses."""
    argv = _reexec_argv(monkeypatch, _args(mode="exploration"), requested)
    parsed = _build_parser().parse_args(argv)  # SystemExit (exit 2) on the pre-fix code
    if requested in ("passive", "active"):
        assert parsed.operational_mode == requested and parsed.mode == "exploration"
    else:
        assert parsed.mode == requested


def test_a_run_mode_switch_keeps_the_operators_grant(monkeypatch) -> None:
    argv = _reexec_argv(monkeypatch, _args(mode="exploration", operational_mode="active"), "sleep")
    parsed = _build_parser().parse_args(argv)
    assert parsed.mode == "sleep" and parsed.operational_mode == "active"


def test_the_launch_flag_accepts_the_three_operational_modes_only() -> None:
    for name in ("passive", "active", "singularity"):
        assert _build_parser().parse_args(["--operational-mode", name]).operational_mode == name
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--operational-mode", "live"])
    assert _build_parser().parse_args([]).operational_mode is None  # default: derived, as before


def test_the_current_operational_mode_is_the_grant_else_the_run_modes() -> None:
    assert _current_operational_mode(_args(mode="sleep", operational_mode="active")) == "active"
    assert _current_operational_mode(_args(mode="exploration")) == "active"
    assert _current_operational_mode(_args(mode="sleep")) == "passive"
    assert _current_operational_mode(_args(mode="agentic")) == "active"


def test_every_runtime_request_may_only_lower() -> None:
    allowed = _runtime_mode_switch_allowed
    assert allowed("active", current_operational="passive", granted=None) is False  # a raise
    assert allowed("passive", current_operational="active", granted=None) is True  # lowering
    assert allowed("singularity", current_operational="singularity", granted=None) is False  # #821
    assert allowed("sleep", current_operational="passive", granted=None) is True  # passive run mode
    # a RUN-mode request that would raise (sleep, passive -> live, active) is refused too
    assert allowed("live", current_operational="passive", granted=None) is False
    # ...but a run-mode switch under an operator grant keeps the grant, so it raises nothing
    assert allowed("live", current_operational="passive", granted="passive") is True


class _Maxim:
    def __init__(self, requested):
        self.requested_mode = requested


def test_a_request_for_the_current_state_does_not_restart_the_run(monkeypatch) -> None:
    """A model that kept asking for the mode it is already in would otherwise restart the run forever."""
    execs: list[str] = []
    monkeypatch.setattr("maxim.cli._reexec_with_mode", lambda args, mode: execs.append(mode))
    monkeypatch.setenv("MAXIM_MODE_SWITCH_DELAY_S", "0")
    args = _args(mode="exploration", operational_mode="passive")
    assert _after_run_mode_request(args, _Maxim("passive"), "exploration") is None
    assert _after_run_mode_request(args, _Maxim("exploration"), "exploration") is None
    assert execs == []
    lowering = _args(mode="exploration", operational_mode="active")
    _after_run_mode_request(lowering, _Maxim("passive"), "exploration")
    assert execs == ["passive"]


def test_a_failed_reexec_keeps_the_vocabulary_split(monkeypatch) -> None:
    def fail(args, mode):
        raise OSError("execv failed")

    monkeypatch.setattr("maxim.cli._reexec_with_mode", fail)
    monkeypatch.setenv("MAXIM_MODE_SWITCH_DELAY_S", "0")
    args = _args(mode="exploration", operational_mode="active")
    # an operational request keeps the RUN mode and changes the grant, in-process
    assert _after_run_mode_request(args, _Maxim("passive"), "exploration") == "exploration"
    assert args.operational_mode == "passive"


def test_the_flag_is_refused_where_it_would_be_ignored(capsys) -> None:
    assert _validate_operational_grant(_args(operational_mode="active", sim="x"), ["--mode", "agentic"]) == 2
    assert "not honoured with --sim" in capsys.readouterr().err
    assert _validate_operational_grant(_args(operational_mode="active"), ["--operational-mode", "active"]) == 2
    assert "needs an explicit --mode" in capsys.readouterr().err
    assert _validate_operational_grant(_args(operational_mode="active"), ["--mode", "agentic"]) is None
    assert _validate_operational_grant(_args(), []) is None  # no flag: nothing to check


def test_agentic_is_an_active_run_mode() -> None:
    from maxim.modes.definitions import get_mode

    assert get_mode("agentic") is not None and get_mode("agentic").name == "active"


# ── the dispatch gate ────────────────────────────────────────────────────────────────────────────


def _executor(mode_source=None):
    from maxim.runtime.executor import Executor
    from maxim.tools.registry import ToolRegistry

    executor = Executor(ToolRegistry())
    if mode_source is not None:
        executor.set_mode_source(mode_source)
    return executor


def test_the_launch_grant_is_the_mode_dispatch_enforces() -> None:
    executor = _executor(lambda: "exploration")  # the run mode implies active
    assert executor._mode_denial("bash") is None
    executor.set_operational_override("passive")
    assert executor._mode_denial("bash") is not None  # the operator's grant wins
    executor.set_operational_override("active")
    assert executor._mode_denial("bash") is None


def test_an_unknown_operational_grant_is_refused_at_launch() -> None:
    with pytest.raises(ValueError, match="unknown operational mode"):
        _executor().set_operational_override("godmode")


def test_an_unknown_run_mode_fails_closed() -> None:
    """Before #829 an unresolvable mode name restricted nothing (the `agentic` hole)."""
    denial = _executor(lambda: "no-such-mode")._mode_denial("bash")
    assert denial is not None and "enforced as passive" in denial
    # #963 (Q4) changed this pin: a mode source that names no mode restricted nothing; it is now the one default,
    # observe (passive). Only an executor with no mode source and no grant is unrestricted.
    assert _executor(lambda: None)._mode_denial("bash") is not None
    assert _executor()._mode_denial("bash") is None


def test_the_one_operational_mode_accessor() -> None:
    """``Executor.effective_operational_mode`` (#963, owner decisions Q1/Q4): the grant, else the mode source's name,
    ``observe`` for an empty or missing one, passive (fail closed) for a non-string; None only with neither a source
    nor a grant. The grant is stored raw (N2)."""
    assert _executor().effective_operational_mode() is None
    assert _executor(lambda: "live").effective_operational_mode() == "live"
    assert _executor(lambda: "").effective_operational_mode() == "observe"
    assert _executor(lambda: None).effective_operational_mode() == "observe"
    assert _executor(lambda: 7).effective_operational_mode() == "passive"
    granted = _executor(lambda: "live")
    granted.set_operational_override("PASSIVE")
    assert granted.effective_operational_mode() == "PASSIVE"  # raw, as the operator gave it
    assert granted._mode_denial("bash") is not None
    granted.set_operational_override(None)
    assert granted.effective_operational_mode() == "live"
    no_source = _executor()
    no_source.set_operational_override("active")
    assert no_source.effective_operational_mode() == "active"


def test_the_loop_reads_the_executors_mode_and_its_own_only_without_one() -> None:
    """``loop_state.operational_mode`` (#963): the executor's answer; the run mode (else ``observe``) only when there
    is no executor or the executor has neither a grant nor a source; a non-string answer is a ``TypeError``."""
    from maxim.runtime.executor import Executor
    from maxim.runtime.loop_state import operational_mode, run_mode, shutdown_requested

    class _State:
        def __init__(self, data: dict) -> None:
            self.data = data

    assert operational_mode(None, _State({"mode": "live"})) == "live"
    assert operational_mode(None, _State({})) == "observe"
    assert operational_mode(None, _State({"mode": ""})) == "observe"
    assert operational_mode(_executor(), _State({"mode": "exploration"})) == "exploration"
    assert operational_mode(_executor(lambda: "live"), _State({"mode": "observe"})) == "live"  # the executor's own
    wrapped = MagicMock(spec=Executor)
    wrapped.effective_operational_mode.return_value = MagicMock()
    with pytest.raises(TypeError, match="not str or None"):
        operational_mode(wrapped, _State({"mode": "live"}))
    assert run_mode(_State({"mode": 3})) is None and run_mode(_State({"mode": "sleep"})) == "sleep"
    assert run_mode(_State({"mode": ""})) is None and run_mode(_State({})) is None
    assert shutdown_requested(_State({"mode": "shutdown"})) and not shutdown_requested(_State({}))


def test_every_executor_wrapper_forwards_the_accessor() -> None:
    """The wrappers forward ``effective_operational_mode`` (``__getattr__``); none defines its own (#963)."""
    from maxim.runtime.fear_gate import FearGatedExecutor
    from maxim.runtime.pain_interceptor import AnticipatoryPainExecutor, PainInterceptorExecutor
    from maxim.simulation.instrumented_executor import InstrumentedExecutor

    inner = _executor(lambda: "live")
    inner.set_operational_override("passive")
    for wrapper_type in (FearGatedExecutor, PainInterceptorExecutor, AnticipatoryPainExecutor, InstrumentedExecutor):
        assert "effective_operational_mode" not in vars(wrapper_type), wrapper_type
    wrapped = [
        InstrumentedExecutor(inner, MagicMock()),
        AnticipatoryPainExecutor(inner, None),
        PainInterceptorExecutor.__new__(PainInterceptorExecutor),
        FearGatedExecutor.__new__(FearGatedExecutor),
    ]
    wrapped[2]._inner = inner
    wrapped[3]._executor = inner
    for w in wrapped:
        assert w.effective_operational_mode() == "passive", type(w)


def test_singularity_is_granted_loudly_on_every_honoured_path(caplog, capsys) -> None:
    """Announced at validation, which runs for BOTH runtimes (the CLI loop and the robot runtime)."""
    with caplog.at_level(logging.WARNING, logger="maxim.cli"):
        assert _validate_operational_grant(_args(operational_mode="singularity"), ["--mode", "live"]) is None
    assert any("singularity" in r.getMessage() and "launch" in r.getMessage() for r in caplog.records)
    assert "singularity" in capsys.readouterr().err
    executor = _executor(lambda: "exploration")
    _apply_operational_grant(executor, "singularity")
    assert executor.effective_operational_mode() == "singularity"
    _apply_operational_grant(executor, None)  # no flag: nothing changes
    assert executor.effective_operational_mode() == "singularity"


def test_what_the_model_is_shown_follows_the_grant() -> None:
    """The prompt roster and context prompt read ``loop_state.operational_mode`` -- the executor's own
    precedence, the one dispatch applies (#963) -- so a raising grant is not a silent no-op at the prompt.

    The roster's half is a SOURCE pin until the slice that extracts §6's prompt assembly (owner decision Q6,
    #963); the Default Network and the follow-up are pinned BEHAVIOURALLY, flipping the grant between two ticks
    (``test_loop_gates_characterization.py::test_the_default_network_follows_a_grant_set_between_ticks``,
    ``test_one_mode_accessor_963.py``), and ``scripts/lint_loop_mode_reads.py`` holds every other read."""

    from maxim.runtime import loop_state  # operational_mode's home (#963)

    class _State:
        data = {"mode": "observe"}

    executor = _executor(lambda: "observe")
    assert loop_state.operational_mode(executor, _State()) == "observe"
    executor.set_operational_override("active")  # the grant is a floor as well as a ceiling (#922)
    assert loop_state.operational_mode(executor, _State()) == "active"
    from tests.unit._loop_source import loop_source

    loop = loop_source()  # the loop's modules, wherever the block lives (1.3.2 decomposition)
    assert "mode_name = operational_mode(executor, state)" in loop  # the roster + context prompt


def test_the_mode_switch_tool_sees_the_grant() -> None:
    """Launched --mode exploration --operational-mode passive: the tool's current mode is the grant,
    so a request for active is refused as a raise (it used to read the run mode, 'active')."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.bootstrap import build_tool_registry

    class _Robot:
        mode = "exploration"
        launch_operational_mode = "passive"
        requested_mode = None

    robot = _Robot()
    registry = build_tool_registry(
        internet_launch_enabled=False,
        maxim=robot,
        autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
    )
    result = registry.get("mode_switch").execute(mode="active")
    assert result.success is False and robot.requested_mode is None


def test_the_robot_runtime_carries_the_grant() -> None:
    """Both runtimes: the CLI agent loop AND the robot runtime (Selfy) honour the flag. SOURCE pins until the
    slices that extract ``cli._main_impl`` and ``_start_agentic_runtime`` (owner decision Q6, #963)."""
    import inspect

    from maxim import cli
    from maxim.embodied_runtime import agentic_runtime
    from maxim.embodied_runtime.selfy import Maxim

    assert "operational_mode" in inspect.signature(Maxim.__init__).parameters
    cli_source = inspect.getsource(cli)
    assert 'operational_mode=getattr(args, "operational_mode", None)' in cli_source  # Maxim(...)
    assert '_apply_operational_grant(executor, getattr(args, "operational_mode", None))' in cli_source
    assert "_operational_mode = _registry_operational_mode(args, _is_sim_mode)" in cli_source
    assert "_validate_operational_grant(args, raw_argv)" in cli_source
    source = inspect.getsource(agentic_runtime)
    assert "executor.set_operational_override(_granted_mode)" in source
    assert '**({"operational_mode": _granted_mode} if _granted_mode else {})' in source
    from maxim.embodied_runtime import selfy

    assert '"passive", "active")' in inspect.getsource(selfy)  # a capability request never parks the head


def test_the_capability_predicate_and_dispatch_agree_on_an_unknown_mode() -> None:
    """Dispatch enforces an unknown mode as passive, so the predicate judges it as passive too: a switch
    out of it to active is a raise, not a free 'lowering'."""
    from maxim.modes.definitions import raises_capability

    assert raises_capability("no-such-mode", "active") is True
    assert raises_capability("no-such-mode", "passive") is False


def test_a_failed_reexec_into_a_run_mode_records_it(monkeypatch) -> None:
    """Otherwise the next request is judged against the OLD run mode: live -> sleep in-process, then a
    later `live` request would pass as active -> active, a raise without a grant."""
    monkeypatch.setattr("maxim.cli._reexec_with_mode", lambda args, mode: (_ for _ in ()).throw(OSError("x")))
    monkeypatch.setenv("MAXIM_MODE_SWITCH_DELAY_S", "0")
    args = _args(mode="live")
    assert _after_run_mode_request(args, _Maxim("sleep"), "live") == "sleep"
    assert args.mode == "sleep"
    assert _after_run_mode_request(args, _Maxim("live"), "sleep") is None  # passive -> active: refused


def test_both_forms_of_the_mode_flag_count_as_explicit(capsys) -> None:
    assert _validate_operational_grant(_args(operational_mode="active"), ["--mode=agentic"]) is None
    assert _explicit_mode_flag(["--mode=live"]) and _explicit_mode_flag(["--mode", "live"])
    assert not _explicit_mode_flag(["--model", "x"]) and not _explicit_mode_flag(None)


@pytest.mark.parametrize("flag", ["list_models", "audit_architecture", "clear_memory"])
def test_the_flag_is_refused_beside_an_action_that_runs_no_agent(capsys, flag) -> None:
    value = True if flag in ("list_models", "audit_architecture") else "all"
    args = _args(operational_mode="singularity", **{flag: value})
    assert _validate_operational_grant(args, ["--mode", "live"]) == 2
    err = capsys.readouterr().err
    assert "runs no agent" in err and "granted at launch" not in err  # no false singularity banner
