"""``tests/unit/_orchestrator_source.py``'s registry stays complete as the orchestrator is decomposed.

The source pins on ``start_simulation_mode`` follow its code through ``PARTS`` (that module's docstring). A
slice that moves code out and forgets to register it would leave every pin on the moved text reading a
function that no longer holds it, which fails loudly, or, worse, a pin that only counts occurrences going
quietly short. These tests make the registration itself mechanical, against the frozen pre-slice snapshot
(``PRE_SLICE_SNAPSHOT``, main 17710609):

  - every top-level function of an ``EXTRACTION_MODULES`` module is a ``PARTS`` entry, or a named
    ``MOVED_MODULE_HELPERS`` exception;
  - every ``maxim.simulation`` module ``orchestrator.py`` imports (relative imports resolved) beyond the
    snapshot is an ``EXTRACTION_MODULES`` entry, so a new extraction module cannot be left off the list;
  - every OTHER ``maxim`` module it imports beyond the snapshot is a named ``NEW_ORCHESTRATOR_IMPORTS``
    entry, so code moved to a module outside ``maxim.simulation`` cannot slip past the first check;
  - every top-level function of a PRE-SLICE ``maxim.simulation`` module the orchestrator imports that the
    snapshot does not hold is a ``PARTS`` entry or a named ``NEW_PRE_SLICE_MODULE_FUNCTIONS`` exception, so
    run code moved into an EXISTING module (``report.py``, ``campaign_runner.py``) shows. This freezes all
    15 such modules, not only the plausible targets: measured over the 30 days before 2026-10-13, 13
    commits changed a top-level ``def`` line in them (``report``, ``tools``, ``sim_types``, ``sim_logger``), a
    one-line exception each, which is the deliberate cost.

STATED LIMITS (what these do not see):

  - a module the orchestrator reaches only TRANSITIVELY (it imports ``a``, run code moved to ``b``, which
    ``a`` imports);
  - an extraction module outside ``maxim.simulation`` is caught only through the orchestrator's own new
    import (the third check), not by the function checks, which read ``maxim.simulation`` files;
  - ``importlib``/``__import__`` and any other dynamic import;
  - run code moved into a CLASS (a method), or into the body of a function that already existed in a
    pre-slice module: only new top-level function names are compared.
"""

from __future__ import annotations

import ast
import json

from tests.unit._orchestrator_source import (
    EXTRACTION_MODULES,
    MOVED_MODULE_HELPERS,
    NEW_ORCHESTRATOR_IMPORTS,
    NEW_PRE_SLICE_MODULE_FUNCTIONS,
    PARTS,
    PRE_SLICE_SNAPSHOT,
    SIMULATION,
    maxim_imports,
)

_PREFIX = "maxim.simulation."


def _file(module: str) -> str:
    """``maxim.simulation.tty_stdin`` -> ``tty_stdin.py`` (relative to ``SIMULATION``, as ``PARTS`` names it)."""
    assert module.startswith(_PREFIX), f"{module}: extraction modules live under maxim.simulation"
    return module[len(_PREFIX) :].replace(".", "/") + ".py"


def _top_level_functions(file: str) -> set[str]:
    tree = ast.parse((SIMULATION / file).read_text())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _snapshot() -> dict:
    data = json.loads(PRE_SLICE_SNAPSHOT.read_text())
    assert data["generated_at_commit"] == "17710609"
    return data


def _orchestrator_imports() -> set[str]:
    return maxim_imports((SIMULATION / "orchestrator.py").read_text())


def test_every_function_of_an_extraction_module_is_a_part() -> None:
    parts = set(PARTS)
    for module in EXTRACTION_MODULES:
        file = _file(module)
        missing = sorted(
            f for f in _top_level_functions(file) if (file, f) not in parts and (file, f) not in MOVED_MODULE_HELPERS
        )
        assert not missing, f"{file}: append {missing} to _orchestrator_source.PARTS (or name it a moved helper)"
    files = {_file(m) for m in EXTRACTION_MODULES}
    # Every named helper lives in a registered module and still exists there.
    assert {f for f, _ in MOVED_MODULE_HELPERS} <= files
    for file, name in MOVED_MODULE_HELPERS:
        assert name in _top_level_functions(file), (file, name)
    assert not set(PARTS) & set(MOVED_MODULE_HELPERS)


def test_a_new_simulation_import_of_the_orchestrator_is_a_registered_extraction_module() -> None:
    pre = {m for m in _snapshot()["maxim_imports"] if m.startswith(_PREFIX)}
    imported = {m for m in _orchestrator_imports() if m.startswith(_PREFIX)}
    unregistered = sorted(imported - pre - set(EXTRACTION_MODULES))
    assert not unregistered, f"orchestrator.py imports {unregistered}: add it to EXTRACTION_MODULES"
    # Each registered module is really imported by the orchestrator (the list does not go stale).
    assert set(EXTRACTION_MODULES) <= imported


def test_a_new_non_simulation_import_of_the_orchestrator_is_named() -> None:
    pre = set(_snapshot()["maxim_imports"])
    new = sorted(m for m in _orchestrator_imports() - pre if not m.startswith(_PREFIX))
    unnamed = [m for m in new if m not in NEW_ORCHESTRATOR_IMPORTS]
    assert not unnamed, (
        f"orchestrator.py newly imports {unnamed}: run code moved outside maxim.simulation must be registered "
        "(and is a limit of the function checks), and a new dependency is named in NEW_ORCHESTRATOR_IMPORTS"
    )
    assert set(NEW_ORCHESTRATOR_IMPORTS) <= set(new), (
        "a NEW_ORCHESTRATOR_IMPORTS entry the orchestrator no longer needs"
    )


def test_no_run_code_hides_in_a_pre_slice_simulation_module() -> None:
    parts = set(PARTS)
    frozen = _snapshot()["simulation_module_functions"]
    assert frozen and sum(map(len, frozen.values())) > 100  # the snapshot is the real one, not emptied
    seen: set[tuple[str, str]] = set()
    for module, before in frozen.items():
        file = _file(module)
        for name in sorted(_top_level_functions(file) - set(before)):
            seen.add((file, name))
            assert (file, name) in parts or (file, name) in NEW_PRE_SLICE_MODULE_FUNCTIONS, (
                f"{file}::{name} is new since 17710609: run code moved there goes in PARTS; other new code is "
                "named in NEW_PRE_SLICE_MODULE_FUNCTIONS with its reason"
            )
    assert set(NEW_PRE_SLICE_MODULE_FUNCTIONS) <= seen, "a stale NEW_PRE_SLICE_MODULE_FUNCTIONS entry"


def test_the_import_resolver_sees_every_static_form() -> None:
    """A negative control per form: each would otherwise be a way around the checks above."""
    text = (
        "import maxim.runtime.a\n"
        "from maxim.runtime.b import x\n"
        "from maxim.simulation import tools\n"
        "from .rel_mod import y\n"
        "from . import rel_pkg_mod\n"
        "from ..runtime.c import z\n"
        "def f():\n    from maxim.simulation.nested import w\n"
    )
    got = maxim_imports(text)
    assert {
        "maxim.runtime.a",
        "maxim.runtime.b",
        "maxim.simulation.tools",
        "maxim.simulation.rel_mod",
        "maxim.simulation.rel_pkg_mod",
        "maxim.runtime.c",
        "maxim.simulation.nested",
    } <= got
