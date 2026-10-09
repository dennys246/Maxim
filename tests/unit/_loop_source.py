"""The agent loop's source text, wherever in the loop's modules it lives (1.3.2 decomposition, slice 0).

Source pins that assert "the loop's code contains X" read the loop through these helpers rather than
``inspect.getsource(agent_loop.run_agentic_loop)``, so a slice that moves a block from
``runtime/agent_loop.py`` into a ``runtime/loop_<concern>.py`` module (the decomposition's flat layout,
owner decision 2026-10-05) keeps the pin green without editing it -- while deleting the code still
fails it. A pin about ONE function's body (an ordering inside it, an AST of that function) does NOT
use these: a slice that moves that code must update the pin consciously.

Resolved from THIS file's path, never ``maxim.__file__``: an installed editable package can shadow a
worktree, and the text that must be pinned is this checkout's.

The glob takes in EVERY ``runtime/loop_*.py``, which already includes three modules that predate the
decomposition: ``loop_controller.py``, ``loop_state.py`` and ``loop_types.py``. A "contains X" pin
therefore also matches X there, and a text pin can also hit X's own ``def`` line or a helper: prefer
a call-site form of the string, or ``loop_call_count`` (an AST count of CALLS) -- the slice-0
deletion probe found five pins that had gone vacuous that way.

Function-specific pins (update consciously, by owning slice). These read one function's body (an
ordering, a region split, an AST of that function) and are NOT routed through this helper:

  - ``test_substrate_action_budget.py::TestWiringPins::test_substrate_branch_consults_gate_before_proposing``
    (gate before propose inside §6b): reads ``loop_substrate.substrate_tick`` since slice 3, and pins one
    ``substrate_tick(`` call inside ``run_agentic_loop``'s ``_substrate_tick_due`` branch.
  - ``test_planning_liveness.py::test_idle_gate_uses_exact_worker_state`` and
    ``::test_completed_state_is_active_until_proposal_poll`` (§0.6 idle gate): read
    ``loop_gates.pre_tick_gate`` since slice 2 (the first also pins one ``pre_tick_gate(`` call in the loop).
  - ``test_experience_clock.py::_loop_calls`` / ``_setup_calls`` / ``_gate_calls`` (ASTs of
    ``run_agentic_loop``, ``loop_setup.build_loop_run`` and ``loop_gates.pre_tick_gate``: one
    ``_loop_bio_handles``, in the setup since slice 1; one ``_loop_live_tick`` carrying the setup's driver,
    in the pre-tick gate since slice 2, which the loop calls with ``experience_driver=_loop_xclock``).
  - ``test_planning_liveness.py::TestLoopWiringPins`` ``loop_src`` pins: ``test_single_gate_covers_every_failure_site``
    (the gate definition, read from ``loop_setup._planning_liveness_gate`` since slice 1, the loop's
    use of ``run.planning_liveness_on``); ``test_exhaustion_raises_after_teardown`` (raise after
    ``_end_bio_session``; the teardown and the raise stayed in the loop in slices 1 and 2 -- the gate
    returns ``GateOutcome.EXHAUSTED`` and the loop sets its flag -- so the pin is unchanged);
    ``test_proposal_time_stamped_on_any_proposal`` -- the slice that moves §2 (the proposal poll stamps it);
    NOT slice 4, which moved §5 (#1085) and left this pin untouched.
  - ``test_planning_liveness.py::test_bad_tool_name_is_recorded_for_correction`` (§2 region) -- §2 is
    LLM-primary, outside phase 1.
  - ``test_console_tool_allowlist.py::TestRosterAdvertisesOnlyPermittedTools::test_agent_loop_filters_the_advertised_roster_through_permits``,
    its ordering half (permits filter before ``last_surfaced_tools``) -- §2, outside phase 1.
  - ``test_tool_output_framing.py`` and ``test_planning_liveness.py``: ``inspect.getsource`` of the
    module-level helpers ``_followup_synthetic_input``, ``_drop_stale_proposal`` and
    ``_proposal_without_action_reason`` -- whichever slice moves the helper (it is an attribute
    lookup, so a re-export keeps it green).
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[2] / "src" / "maxim" / "runtime"


def loop_source_paths() -> list[Path]:
    """``runtime/agent_loop.py`` first, then every ``runtime/loop_*.py``, sorted by name."""
    paths = [RUNTIME / "agent_loop.py", *sorted(RUNTIME.glob("loop_*.py"))]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"loop source missing: {missing}")
    return paths


def loop_source_relpaths() -> set[str]:
    """The loop's modules as ``runtime/<name>.py`` (relative to the ``maxim`` package)."""
    return {f"runtime/{p.name}" for p in loop_source_paths()}


def loop_source() -> str:
    """The concatenated text of every loop module (a file separator between them)."""
    return "\n".join(f"# ---- {p.name} ----\n{p.read_text()}" for p in loop_source_paths())


def loop_source_trees() -> list[ast.Module]:
    """One parsed module per loop file (the concatenation is not one valid module: each file may
    carry its own ``from __future__`` import)."""
    return [ast.parse(p.read_text(), filename=str(p)) for p in loop_source_paths()]


def loop_call_count(name: str) -> int:
    """How many CALLS to ``name`` (``name(...)`` or ``x.name(...)``) the loop's modules make -- for a
    "the loop calls X" pin, where a text match would also hit X's own ``def`` line."""
    return sum(
        1
        for tree in loop_source_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == name
    )
