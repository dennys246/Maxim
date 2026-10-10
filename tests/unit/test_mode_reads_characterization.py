"""#963 characterization: what each mode reader in the agent loop sees, by run mode and launch grant.

Written BEFORE #963 routes every read through one accessor, and green on ``main``: it pinned that day's values,
defects included, so the fix's change is visible per test (the slice 3/4 protocol). The cells #963 changed say
so, with the value they had. The readers:

- the follow-up a tool queues (its ``followup_type`` and ``mode``), on all three dispatch paths that share
  ``tool_dispatch.execute_and_learn`` (autonomous, the policy-confirmed SUPERVISED path, the PLANNING approved
  path), observed through the REAL loop (``tests/unit/_execute_learn_driver.py``) with the probe registered as
  ``internet_search``, the one kind of tool whose follow-up type depends on the mode (``engage``);
- the autonomy audit's ``mode`` on every entry the run logs;
- the dispatch gate, composed as the loop composes it (``loop_setup._prepare_executor`` hands the executor its
  mode source; the grant is ``Executor.set_operational_override``).

The Default Network's mode is pinned beside the other pre-tick gate pins, in
``tests/unit/test_loop_gates_characterization.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit._execute_learn_driver import run_once

pytestmark = pytest.mark.timeout(60)

ENGAGE_TOOL = "internet_search"
LEVELS = ("autonomous", "supervised", "planning")

# (state mode, grant) -> (follow-up type, follow-up mode, audit mode).
_MATRIX: dict[str, tuple[str | None, str | None, tuple[str, str, str]]] = {
    "live": ("live", None, ("engage", "live", "live")),
    "observe": ("observe", None, ("engage", "observe", "observe")),
    "active": ("active", None, ("engage", "active", "active")),
    # #963 changed these: the grant decided dispatch, but the follow-up and the audit read the run mode
    # (("engage", "live", "live") and ("engage", "", "")); every reader now reads the operational mode.
    "passive_grant_over_live": ("live", "passive", ("respond", "passive", "passive")),
    "passive_grant_over_empty": ("", "passive", ("respond", "passive", "passive")),
    # No mode in the state (the CLI loop, ``maxim.run()``). #963 changed this: each reader picked its own default
    # (("engage", "live", "unknown")); the one default is ``observe``.
    "no_mode": (None, None, ("engage", "observe", "observe")),
}


def _run(monkeypatch: Any, tmp_path: Any, case: str, level: str) -> Any:
    state_mode, grant, _ = _MATRIX[case]
    return run_once(monkeypatch, tmp_path, tool=ENGAGE_TOOL, level=level, state_mode=state_mode, grant=grant)


@pytest.mark.parametrize("level", LEVELS)
@pytest.mark.parametrize("case", list(_MATRIX))
def test_the_follow_up_and_the_audit_by_run_mode_and_grant(monkeypatch, tmp_path, case, level):
    obs = _run(monkeypatch, tmp_path, case, level)
    assert [e["_tool"] for e in obs.executed] == [ENGAGE_TOOL]
    followup_type, followup_mode, audit_mode = _MATRIX[case][2]
    [fu] = obs.followups
    assert (fu.tool, fu.followup_type, fu.mode) == (ENGAGE_TOOL, followup_type, followup_mode)
    entries = obs.autonomy.get_audit_log()
    assert entries, "the run logged no audit entry"
    assert {e.mode for e in entries} == {audit_mode}


# ── the dispatch gate, composed as the loop composes it ──────────────────────────────────────────────

# (state mode, grant) -> does a host-acting tool (bash) pass the mode gate.
_DISPATCH: dict[str, tuple[str | None, str | None, bool]] = {
    "live": ("live", None, True),
    "observe": ("observe", None, False),
    "active": ("active", None, True),
    "passive_grant_over_live": ("live", "passive", False),
    "passive_grant_over_empty": ("", "passive", False),
    # An explicit empty mode. #963 (Q4) changed this: it restricted nothing (True) while the prompt roster showed
    # passive; the one default, observe, is passive.
    "empty": ("", None, False),
    "no_mode": (None, None, False),  # the one default, observe (the mode source's own default before #963)
}


@pytest.mark.parametrize("case", list(_DISPATCH))
def test_the_dispatch_gate_by_run_mode_and_grant(case):
    from maxim.runtime.bootstrap import build_executor
    from maxim.runtime.loop_setup import _prepare_executor
    from maxim.runtime.state import RuntimeState
    from maxim.tools.registry import ToolRegistry

    state_mode, grant, bash_runs = _DISPATCH[case]
    state = RuntimeState()
    if state_mode is not None:
        state.data["mode"] = state_mode
    executor = _prepare_executor(build_executor(ToolRegistry(), pain_bus=None, permissions=None), None, state)
    if grant is not None:
        executor.set_operational_override(grant)
    assert (executor._mode_denial("bash") is None) is bash_runs
    assert (executor._mode_denial("respond") is None) is True  # passive still runs the non-acting tools


def test_the_maxim_runtime_copy_is_the_run_mode_not_the_grant(monkeypatch, tmp_path):
    """The PerceptionAgent -> MemoryAgent -> ExecAgent axis (``maxim_runtime["mode"]``: sleep, reflection) is the
    RUN mode, and stays so (#963 owner decision Q5): a grant does not reach it, and no run mode seeds nothing."""
    obs = run_once(monkeypatch, tmp_path, tool=ENGAGE_TOOL, state_mode="live", grant="passive")
    assert obs.ctrl.state.data["maxim_runtime"]["mode"] == "live"
    monkeypatch.undo()
    obs = run_once(monkeypatch, tmp_path / "none", tool=ENGAGE_TOOL, state_mode=None)
    assert "mode" not in obs.ctrl.state.data["maxim_runtime"]
