"""``start_simulation_mode``'s source, wherever its code lives (1.3.2 orchestrator decomposition, slice 0).

The orchestrator's counterpart of ``tests/unit/_loop_source.py``. Source and AST pins on
``simulation/orchestrator.py::start_simulation_mode`` read the function through these helpers, never
``inspect.getsource(orchestrator.start_simulation_mode)`` or a hand-rolled ``ast.parse``. An extraction
slice moves a block of the function into its own function, possibly in a new module under
``simulation/``. The slice that does so appends that function to ``PARTS`` (in the same commit as the
move), and every "the function contains X" pin follows the code there UNCHANGED. Deleting the code
still fails the pin.

Resolved from THIS file's path, never ``maxim.__file__``: an installed editable package can shadow a
worktree, and the text that must be pinned is this checkout's. (``test_finish_reason_1002.py`` used to
read ``maxim.__file__``.)

``PARTS`` lists FUNCTIONS, not whole modules, unlike ``loop_source()``. The orchestrator module also
holds helpers (``_resolve_finish_reason``, ``_finish_runner``, ``_restore_aut_from_session``...) whose
bodies repeat strings the pins look for at the CALL sites. A whole-module text would make those pins
vacuous: the slice-0 loop lesson, where five pins went vacuous that way.

Three views:

  - ``orchestrator_source()``: the concatenated text of every part, for "contains X" pins.
  - ``orchestrator_tree()``: an ``ast.Module`` holding every part's ``FunctionDef``, for ``ast.walk``
    pins (call counts, call arguments).
  - ``start_simulation_mode_def()``: the ONE function's ``FunctionDef``, for function-specific pins.

FUNCTION-SPECIFIC PINS. These read an ordering inside the one function (``lineno`` comparisons, an
``index()`` ordering, a ``split()`` region). They read ``start_simulation_mode_def()``, or an ordering of
``orchestrator_source()`` that only holds while both ends stay in one part. The slice that moves either
end updates them consciously:

  - ``test_report_provenance_m1.py``, ``_start_simulation_mode`` (the start stamp's ``lineno`` before
    every ``build_primary_router`` call; §0-§1).
  - ``test_finish_reason_1002.py``, ``_start_simulation_mode``. The operator-stop pin separates the
    run's own top-level ``KeyboardInterrupt`` handler (§12) from the nested readers' (§10) with
    ``n is not fn``. Its other pins (the ``_resolve_finish_reason`` arguments in §14, each runner's
    ``_finish_runner`` in §9) are walk-only, and move to ``orchestrator_tree()`` when a slice splits
    the function.
  - ``test_narrator_reliability.py::test_only_the_run_full_turns_flag_sets_the_finish_floor_and_an_observer_is_exempt``
    (``_is_observe_only = `` before ``FinishSimulationTool(``; §5).
  - ``test_planning_liveness.py::TestOrchestratorWiringPins::test_519_abort_clock_is_nudge_proof`` and
    ``TestDeadAutIsSurfaced::test_aut_worker_terminates_the_sim_on_loop_failure`` (``split()`` regions of
    the stall detector §11 and the ``sim.aut`` worker §8). They hold as long as each region moves whole.
  - ``test_planning_liveness.py::...::test_orchestrator_opts_in_and_aut_does_not`` (the region after the
    narrator loop call's ``percept_source=orchestrator_source``; §12).
  - ``tests/integration/test_persistent_agent_campaign.py::TestNonePathUnchanged::test_injection_branch_gates_are_wired``
    (two multi-line pins that carry the function body's indentation; §2-§3).

NOT ROUTED HERE (whole-MODULE readers, ``inspect.getsource(orchestrator)``). These follow the module, not
the function. A slice that moves the code they pin into a new module updates them:
``test_console_tool_allowlist.py``, ``test_config_sparse_1_2.py``, ``test_prompt_names_own_tools.py``
and ``test_stall_detector_with_registry.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SIMULATION = Path(__file__).resolve().parents[2] / "src" / "maxim" / "simulation"

# (module file under src/maxim/simulation/, function name). The first entry is the function itself;
# each extraction slice appends the functions it moved out of it.
PARTS: tuple[tuple[str, str], ...] = (
    ("orchestrator.py", "start_simulation_mode"),
    # slice 0 (owner decision SF-4, 2026-10-13): the TTY read loops, moved verbatim to the coverage-excluded
    # tty_stdin.py (the nested _stdin_reader_raw and the post-sim raw prompt loop)
    ("tty_stdin.py", "stdin_reader_raw"),
    ("tty_stdin.py", "read_post_sim_goal_raw"),
)

# Every module a slice moved ``start_simulation_mode`` code into, as a dotted name. The slice that creates one
# appends it here in the same commit as the move, and appends its functions to ``PARTS``.
# ``test_orchestrator_source.py`` holds the registry complete (its docstring lists the checks and their limits).
EXTRACTION_MODULES: tuple[str, ...] = ("maxim.simulation.tty_stdin",)

# Top-level functions of an extraction module that are NOT the run's code, with the reason: module-level
# helpers of ``orchestrator.py`` that moved with the code they serve. They are not in ``PARTS``, so a pin on
# the run cannot be satisfied by their text.
MOVED_MODULE_HELPERS: dict[tuple[str, str], str] = {
    ("tty_stdin.py", "_restore_terminal"): "module level in orchestrator.py at 17710609 (the atexit/SIGTERM "
    "terminal restore), moved with the raw reader that arms it",
    ("tty_stdin.py", "_sigterm_handler"): "module level in orchestrator.py at 17710609 (the SIGTERM hook the raw "
    "reader installs), moved with it",
}

# What ``orchestrator.py`` looked like at main 17710609, before any extraction slice, frozen: every ``maxim``
# module it imported (``maxim_imports``, resolved as ``maxim_imports()`` resolves them) and the top-level
# function names of each ``maxim.simulation`` module among them. Generated once from ``git show 17710609:...``;
# never regenerated.
PRE_SLICE_SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "orchestrator_pre_slice_snapshot.json"

# A ``maxim`` module ``orchestrator.py`` imports that it did not at 17710609, and that is NOT an extraction
# module (a fix needing a new dependency), with the reason. Empty at slice 0.
NEW_ORCHESTRATOR_IMPORTS: dict[str, str] = {}

# A new top-level function in a pre-slice ``maxim.simulation`` module that is NOT code moved out of the run
# (ordinary new code in, e.g., ``report.py``), as (file, name) with the reason. Empty at slice 0; the #1261 fix,
# which is not a slice, added the first entries.
NEW_PRE_SLICE_MODULE_FUNCTIONS: dict[tuple[str, str], str] = {
    ("tools.py", "_action_summary"): (
        "#1261: the narrator-facing per-action summary, factored out of SendMessageTool, ExtendSimulationTool and "
        "SpawnSubSimulationTool (three identical copies) to add the utterance; not orchestrator run code"
    ),
    ("report.py", "_provider_failures"): (
        "#997: build_report's provider-failure attribution block moved out verbatim (rule T, after the URL scrub "
        "touched it); report assembly, not orchestrator run code"
    ),
    ("report.py", "_record_dict"): (
        "#997: the one serialization point both report writers share, reducing endpoint URLs; not orchestrator run code"
    ),
}


def maxim_imports(text: str, package: str = "maxim.simulation") -> set[str]:
    """Every ``maxim`` module a source text imports, at any depth of the file, in every static form:
    ``import maxim.x``, ``from maxim.x import y``, ``from maxim.pkg import mod`` (``maxim.pkg.mod`` when
    ``maxim/pkg`` is a package) and relative imports, resolved against ``package`` (the file's own)."""
    src = SIMULATION.parents[1]
    out: set[str] = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names if a.name == "maxim" or a.name.startswith("maxim.")}
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                base = package.split(".")
                base = base[: len(base) - (n.level - 1)]
                module = ".".join(base + ([n.module] if n.module else []))
            else:
                module = n.module or ""
            if module != "maxim" and not module.startswith("maxim."):
                continue
            if (src / module.replace(".", "/") / "__init__.py").is_file():  # a package: its names may be modules
                out |= {f"{module}.{a.name}" for a in n.names}
            else:
                out.add(module)
    return out


def _function(path: Path, name: str) -> tuple[ast.FunctionDef, str]:
    if not path.is_file():
        raise FileNotFoundError(f"orchestrator source missing: {path}")
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    found = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise LookupError(f"expected one def {name!r} in {path.name}, found {len(found)}")
    return found[0], text


def _part_source(node: ast.FunctionDef, text: str) -> str:
    """The def's full lines, as ``inspect.getsource`` returns them."""
    lines = text.splitlines(keepends=True)
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return "".join(lines[first - 1 : node.end_lineno])


def start_simulation_mode_def() -> ast.FunctionDef:
    """THE function's AST: for pins about an ordering inside it (module docstring)."""
    file, name = PARTS[0]
    return _function(SIMULATION / file, name)[0]


def orchestrator_source() -> str:
    """The concatenated text of every part (a separator line names each part)."""
    chunks = []
    for file, name in PARTS:
        node, text = _function(SIMULATION / file, name)
        chunks.append(f"# ---- {file}::{name} ----\n{_part_source(node, text)}")
    return "\n".join(chunks)


def orchestrator_tree() -> ast.Module:
    """One module node whose body is every part's ``FunctionDef`` (``ast.walk`` covers all of them)."""
    return ast.Module(body=[_function(SIMULATION / f, n)[0] for f, n in PARTS], type_ignores=[])
