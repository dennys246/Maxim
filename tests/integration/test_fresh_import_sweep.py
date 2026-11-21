"""Every ``maxim`` module imports on its own, in a fresh interpreter (#1315 step 0a, V9; atonia notes §1A.1).

The package inits are lazy (PEP 562), so importing one module no longer imports its whole package first. A cycle
that the old eager inits hid surfaces only when a module is the FIRST thing a process imports (step 0a found one:
``agents/base.py``'s bottom-of-module ``GoalAgent`` re-export). One child per module, so import order in a shared
process cannot mask a cycle. ``slow``: one interpreter per module.

A module that IS an optional extra's own (its classes subclass the extra's types, or it is the extra's server) is
declared in :data:`EXTRA_OWNED` (#1647; owner ruling 2026-11-21: such a module need not import without its extra).
The declaration is checked, not a skip: a listed module must import when its extra is installed, and when the extra
is absent its failure must be a refusal attributed to one of that extra's own import names. Every unlisted module
must import in a bare interpreter. Until #1647 the sweep exempted nothing and counted every exception, so the slow
lane (which installs ``console`` and ``sign`` but not ``training``) was red from 2026-11-01 on ``motor_cortex``,
while the console modules passed only because that lane happens to install their extra.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

from tests._child_env import isolation_env
from tests.conftest import _optional_dependencies, core_import_names, extra_import_names

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

# ``maxim.__main__`` runs the CLI on import: it is the entry point, not a library module.
SKIP = frozenset({"maxim.__main__"})

# module -> the pyproject extra it belongs to. Add a module here only when it cannot exist without the extra (its
# module-level code is the extra's API); a module that merely USES an extra imports it inside the function that needs
# it, through utils/optional_deps.py, and stays unlisted.
EXTRA_OWNED: dict[str, str] = {
    "maxim.console.schemas": "console",
    "maxim.console.server": "console",
    "maxim.models.movement.motor_cortex": "training",
}

# The child reports each failure as data, so the verdict below can attribute it. ``missing`` is the one dependency
# the failure is attributed to, or ``None`` when it attributes to nothing (and so can never be an extra's refusal).
# The rule, applied to the chain of the top error and its ``__cause__`` (else its unsuppressed ``__context__``),
# walked once per link (a seen-set: ``e.__cause__ = e`` must not hang the child):
#   1. Any link that is not an ImportError -> ``None``. The import failed on something else (a typed refusal
#      wrapping an AttributeError, or a defect masked in ``__context__`` by a fallback import of an absent name).
#   2. The top error must be the canonical refusal, ``OptionalDependencyError`` (by type name and module: the child
#      must not import ``maxim.utils.optional_deps`` itself, or every module after it is no longer imported first),
#      or a stdlib ImportError carrying its own ``.name``. A hand-written ``raise ImportError("...")`` names nothing
#      and is refused even when chained to a real missing module: CLAUDE.md's optional-deps rule bans that shape in
#      src/maxim, and ``require_optional_dependency`` is the refusal to use.
#   3. Then the DEEPEST ``.name`` in the chain wins, else the typed refusal's ``import_name``. Decision: the deepest
#      link is what was actually not found. ``require_optional_dependency("keras")`` with keras installed and
#      tensorflow absent raises ``OptionalDependencyError(keras)`` from ``ModuleNotFoundError(tensorflow)``: the
#      extra is partly absent, and it is tensorflow that is missing. The same rule attributes an installed name raised
#      from a missing own name (contrived) to the missing one; that is pinned by a test, not an accident.
_CHILD = (
    "import importlib, json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "def attribute(exc):\n"
    "    chain, seen, link = [], set(), exc\n"
    "    while link is not None and id(link) not in seen:\n"
    "        seen.add(id(link))\n"
    "        chain.append(link)\n"
    "        link = link.__cause__ if link.__cause__ is not None else (\n"
    "            None if link.__suppress_context__ else link.__context__)\n"
    "    if not all(isinstance(e, ImportError) for e in chain):\n"
    "        return None\n"
    "    typed = type(exc).__name__ == 'OptionalDependencyError' and (\n"
    "        type(exc).__module__ == 'maxim.utils.optional_deps')\n"
    "    if not (typed or exc.name):\n"
    "        return None\n"
    "    named = [e.name for e in chain if e.name]\n"
    "    return named[-1] if named else getattr(exc, 'import_name', None)\n"
    "fails = {}\n"
    "for name in json.loads(sys.argv[2]):\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception as exc:\n"
    "        fails[name] = {\n"
    "            'text': f'{type(exc).__name__}: {exc}'[:300],\n"
    "            'is_import_error': isinstance(exc, ImportError),\n"
    "            'type_module': type(exc).__module__,\n"
    "            'missing': attribute(exc),\n"
    "        }\n"
    "print(json.dumps(fails))\n"
)


def _installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def _top(name: str | None) -> str:
    return (name or "").split(".", 1)[0]


def sweep_verdict(
    module: str,
    failure: dict[str, Any] | None,
    *,
    installed: Callable[[str], bool] = _installed,
    owned: dict[str, str] = EXTRA_OWNED,
    own_names_of: Callable[[str], tuple[str, ...]] = extra_import_names,
) -> str | None:
    """Why ``module``'s fresh import is a defect, or ``None`` when it is acceptable. Pure, so the rule is testable.

    An unlisted module must import. A listed module must import when every import name of its extra is installed.
    When the extra is absent (even partly), the one acceptable failure is an ImportError attributed (``_CHILD``'s
    rule: no non-ImportError in the chain, a typed or named top error, the deepest name wins) to one of the extra's
    import names that is NOT installed here. Anything else is a defect like any other: an error raised by an
    installed package of the extra (``from keras.layers import Layr``, a keras ``ValueError`` at class definition)
    proves that package is present, so it cannot be the extra's absence. ``utils/optional_deps.py::import_refused``
    is not used: its blanket ImportError/OSError branch cannot say WHICH dependency refused."""
    if failure is None:
        return None
    extra = owned.get(module)
    if extra is None:
        return f"{module}: {failure['text']}"
    own_names = own_names_of(extra)
    if all(installed(n) for n in own_names):
        return f"{module}: fails with its extra [{extra}] installed: {failure['text']}"
    missing = _top(failure["missing"])
    if failure["is_import_error"] and missing in {_top(n) for n in own_names} and not installed(missing):
        return None
    return f"{module}: [{extra}] is absent, but this is not its refusal: {failure['text']}"


def _modules() -> tuple[list[str], list[str]]:
    """Every module name, plus the packages ``walk_packages`` could not descend into (an ``__init__`` that raises
    would otherwise drop its whole subtree from the sweep without a word)."""
    unwalkable: list[str] = []
    sys.path.insert(0, str(SRC))
    try:
        import maxim

        names = sorted(
            m.name
            for m in pkgutil.walk_packages(maxim.__path__, "maxim.", onerror=unwalkable.append)
            if m.name not in SKIP
        )
        return names, unwalkable
    finally:
        sys.path.remove(str(SRC))


def _import_failures(src: Path, names: list[str], env: dict[str, str]) -> dict[str, dict[str, Any]]:
    fails: dict[str, dict[str, Any]] = {}
    for name in names:
        proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
            [sys.executable, "-I", "-c", _CHILD, str(src), json.dumps([name])],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        lines = proc.stdout.strip().splitlines()
        if proc.returncode != 0 or not lines:
            text = f"child exited {proc.returncode}: {proc.stderr[-300:]}"
            fails[name] = {"text": text, "is_import_error": False, "type_module": "", "missing": None}
            continue
        fails.update(json.loads(lines[-1]))
    return fails


def _child_env(tmp_path: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path),
        **isolation_env(),  # beside HOME: the host's /etc/cloudflared never reaches the child
    }


@pytest.mark.slow
def test_every_module_imports_first_in_a_fresh_process(tmp_path: Path) -> None:
    names, unwalkable = _modules()
    assert not unwalkable, f"packages whose __init__ fails to import, so their subtrees were not swept: {unwalkable}"
    assert set(EXTRA_OWNED) <= set(names), (
        f"EXTRA_OWNED names modules that do not exist: {set(EXTRA_OWNED) - set(names)}"
    )
    fails = _import_failures(SRC, names, _child_env(tmp_path))
    defects = [v for v in (sweep_verdict(name, fails.get(name)) for name in names) if v is not None]
    assert not defects, "modules that fail when imported first:\n  " + "\n  ".join(defects)


# ── the verdict's rule (fast: no sweep) ───────────────────────────────────────────────────────────────────────────


def _failure(text: str, *, missing: str | None, is_import_error: bool = True, type_module: str = "builtins") -> dict:
    return {"text": text, "is_import_error": is_import_error, "type_module": type_module, "missing": missing}


_NOTHING_INSTALLED: Callable[[str], bool] = lambda _name: False  # noqa: E731
_EVERYTHING_INSTALLED: Callable[[str], bool] = lambda _name: True  # noqa: E731
_MOTOR = "maxim.models.movement.motor_cortex"


def test_an_unlisted_module_must_import_bare() -> None:
    failure = _failure("ModuleNotFoundError: No module named 'keras'", missing="keras")
    assert sweep_verdict("maxim.some.module", failure, installed=_NOTHING_INSTALLED) is not None
    assert sweep_verdict("maxim.some.module", None, installed=_NOTHING_INSTALLED) is None


def test_a_listed_module_passes_on_its_extras_typed_refusal() -> None:
    refusal = _failure("OptionalDependencyError: motor_cortex: required dependency 'keras'", missing="keras")
    assert sweep_verdict(_MOTOR, refusal, installed=_NOTHING_INSTALLED) is None
    declared = _failure("ModuleNotFoundError: No module named 'pydantic'", missing="pydantic")
    assert sweep_verdict("maxim.console.schemas", declared, installed=_NOTHING_INSTALLED) is None
    # Partly installed: keras present, tensorflow absent. The typed refusal's cause names tensorflow, which is not.
    keras_only = _failure("OptionalDependencyError: ... 'keras'", missing="tensorflow")
    assert sweep_verdict(_MOTOR, keras_only, installed=_KERAS_ONLY) is None


_KERAS_ONLY: Callable[[str], bool] = lambda name: name == "keras"  # noqa: E731
_TF_ONLY: Callable[[str], bool] = lambda name: name == "tensorflow"  # noqa: E731


@pytest.mark.parametrize(
    "failure",
    [
        # the pre-#1647 shape: a hand-written ImportError names no dependency
        _failure("ImportError: motor_cortex requires keras. Install with: ...", missing=None),
        _failure("ModuleNotFoundError: No module named 'cv2'", missing="cv2"),  # another extra's dependency
        _failure("ImportError: cannot import name 'x' from 'maxim.utils.config'", missing="maxim.utils.config"),
        _failure("NameError: name 'x' is not defined", missing=None, is_import_error=False),
        _failure("OSError: [Errno 2]", missing=None, is_import_error=False),  # no package to attribute it to
    ],
)
def test_a_listed_module_failing_for_another_reason_fails(failure: dict) -> None:
    assert sweep_verdict(_MOTOR, failure, installed=_NOTHING_INSTALLED) is not None


@pytest.mark.parametrize(
    ("failure", "installed"),
    [
        # (a) keras installed, tensorflow absent: a typo against the installed keras is a bug, not the extra's absence
        (_failure("ImportError: cannot import name 'Layr' from 'keras.layers'", missing="keras.layers"), _KERAS_ONLY),
        # (b) same: keras itself raised, so keras is present
        (_failure("ValueError: ...", missing=None, is_import_error=False, type_module="keras.src.layers"), _KERAS_ONLY),
        # (c) tensorflow installed, keras absent: tensorflow raised, so it is present and broken
        (_failure("AttributeError: ...", missing=None, is_import_error=False, type_module="tensorflow"), _TF_ONLY),
        # an ImportError naming an own package that IS installed
        (_failure("ModuleNotFoundError: No module named 'keras'", missing="keras"), _KERAS_ONLY),
    ],
)
def test_a_partly_installed_extra_accepts_only_a_missing_own_name(failure: dict, installed) -> None:
    assert sweep_verdict(_MOTOR, failure, installed=installed) is not None


def test_a_listed_module_must_import_when_its_extra_is_installed() -> None:
    refusal = _failure("OptionalDependencyError: motor_cortex: required dependency 'keras'", missing="keras")
    assert sweep_verdict(_MOTOR, refusal, installed=_EVERYTHING_INSTALLED) is not None


def test_extra_owned_names_real_extras() -> None:
    extras = _optional_dependencies()
    unknown = {e for e in EXTRA_OWNED.values() if e not in extras}
    assert not unknown, f"EXTRA_OWNED names extras pyproject does not declare: {unknown}"


def _module_level_dependencies(path: Path) -> set[str]:
    """The top-level names ``path`` needs at import, from its module-level statements (inside a module-level
    ``try``/``if``/``with`` too, never inside a function or class): ``import`` / ``from`` (absolute), and the
    constant first argument of a ``require_optional_dependency(...)`` or ``importlib.import_module(...)`` call,
    by name or attribute (``optional_deps.require_optional_dependency``), as a statement, an assignment, an
    annotated assignment or a ``with`` item."""
    import ast

    found: set[str] = set()
    loaders = {"require_optional_dependency", "import_module"}

    def note_call(expr: ast.expr | None) -> None:
        if not isinstance(expr, ast.Call) or not expr.args or not isinstance(expr.args[0], ast.Constant):
            return
        func = expr.func
        called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if called in loaders and isinstance(expr.args[0].value, str):
            found.add(_top(expr.args[0].value))

    def visit(nodes: list[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.Import):
                found.update(_top(alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(_top(node.module))
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.Expr)):
                note_call(node.value)
            elif isinstance(node, ast.With):
                for item in node.items:
                    note_call(item.context_expr)
                visit(node.body)
            elif isinstance(node, (ast.Try, ast.If)):
                visit(node.body)
                visit(node.orelse)
                for handler in getattr(node, "handlers", ()):
                    visit(handler.body)
                visit(getattr(node, "finalbody", []))

    visit(ast.parse(path.read_text()).body)
    return found


def test_each_extra_owned_module_imports_only_what_its_extra_declares() -> None:
    """The declaration the verdict relies on: every third-party name an extra-owned module needs at import is a
    DIRECT requirement of its extra (or of core). A name that arrives only transitively (pydantic through fastapi,
    until #1647 declared it) vanishes when the upstream drops it, and the sweep could not attribute its refusal."""
    core = set(core_import_names())
    undeclared = {}
    for module, extra in EXTRA_OWNED.items():
        path = SRC / (module.replace(".", "/") + ".py")
        needed = _module_level_dependencies(path) - set(sys.stdlib_module_names) - {"maxim", "__future__"}
        missing = needed - set(extra_import_names(extra)) - core
        if missing:
            undeclared[module] = sorted(missing)
    assert not undeclared, f"extra-owned modules import names their extra does not declare: {undeclared}"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os\nimport uvicorn.main\n", {"os", "uvicorn"}),
        ("from pydantic import BaseModel\nfrom . import sibling\n", {"pydantic"}),
        ("try:\n    import uvicorn\nexcept ImportError:\n    import cv2\n", {"uvicorn", "cv2"}),
        ("if True:\n    import fastapi\n", {"fastapi"}),
        ("keras = require_optional_dependency('keras', feature='x')\n", {"keras"}),
        ("require_optional_dependency('keras')\n", {"keras"}),
        ("keras: object = require_optional_dependency('keras')\n", {"keras"}),  # AnnAssign
        ("keras = optional_deps.require_optional_dependency('keras')\n", {"keras"}),  # attribute call
        ("yaml = importlib.import_module('yaml')\n", {"yaml"}),
        ("with contextlib.suppress(Exception):\n    import rich\n", {"rich"}),  # with body
        ("with importlib.import_module('cv2'):\n    pass\n", {"cv2"}),  # with item
        ("def f():\n    import cv2\nclass C:\n    import torch\n", set()),  # function and class bodies: not import
    ],
)
def test_the_declaration_check_sees_each_module_level_form(tmp_path: Path, source: str, expected: set[str]) -> None:
    probe = tmp_path / "probe.py"
    probe.write_text(source)
    assert _module_level_dependencies(probe) == expected


_ODE = "from maxim.utils.optional_deps import OptionalDependencyError\n"
# module -> (source, the child's attribution, accepted as the stand-in extra's refusal). The stand-in extra's import
# names are the zz_* names below (none installed) plus ``json`` (installed), so the verdict runs on real reports.
_CHILD_CASES: dict[str, tuple[str, str | None, bool]] = {
    "missing": ("import zz_absent_1647\n", "zz_absent_1647", True),
    "typed": (
        "from maxim.utils.optional_deps import require_optional_dependency\n"
        "require_optional_dependency('zz_absent_1647', extra='training')\n",
        "zz_absent_1647",
        True,
    ),
    # keras installed, tensorflow absent: the deepest name in the chain is the one not found
    "chained": (
        _ODE + "try:\n    import zz_inner_1647\n"
        "except ImportError as e:\n    raise OptionalDependencyError('zz_outer_1647', cause=e) from e\n",
        "zz_inner_1647",
        True,
    ),
    # contrived: an INSTALLED own name raised from a missing own one; the deepest (missing) name wins, by decision
    "installed_from_missing": (
        "try:\n    import zz_inner_1647\n"
        "except ImportError as e:\n    raise ModuleNotFoundError('json?', name='json') from e\n",
        "zz_inner_1647",
        True,
    ),
    "plain": ("raise ImportError('needs keras')\n", None, False),
    # S1: a hand-written ImportError chained to a real missing own name is still not the canonical refusal
    "plain_chained": (
        "try:\n    import zz_absent_1647\nexcept ImportError as e:\n    raise ImportError('needs zz') from e\n",
        None,
        False,
    ),
    # N2: a typed refusal wrapping a non-ImportError
    "typed_wraps_defect": (
        _ODE + "try:\n    {}.missing_attr\n"
        "except AttributeError as e:\n    raise OptionalDependencyError('zz_absent_1647', cause=e) from e\n",
        None,
        False,
    ),
    # N2: a defect masked in __context__ by a fallback import of an absent own name
    "masked_by_fallback": (
        "try:\n    {}.missing_attr\nexcept AttributeError:\n    import zz_absent_1647\n",
        None,
        False,
    ),
    # N1: a cause cycle must not hang the child
    "cause_cycle": (
        "e = ModuleNotFoundError('x', name='zz_absent_1647')\ne.__cause__ = e\nraise e\n",
        "zz_absent_1647",
        True,
    ),
}


def test_the_child_attributes_each_shape_and_the_verdict_follows(tmp_path: Path) -> None:
    """The child's report on REAL failures, then the verdict on that report, for a stand-in listed module of a
    stand-in extra: what is accepted is exactly a canonical or named refusal of an absent own name."""
    pkg = tmp_path / "zz1647"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name, (source, _, _) in _CHILD_CASES.items():
        (pkg / f"{name}.py").write_text(source)
    modules = [f"zz1647.{name}" for name in _CHILD_CASES]
    child = _CHILD.replace("sys.path.insert(0, sys.argv[1])\n", f"sys.path[:0] = [sys.argv[1], {str(SRC)!r}]\n")
    proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        [sys.executable, "-I", "-c", child, str(tmp_path), json.dumps(modules)],
        capture_output=True,
        text=True,
        env=_child_env(tmp_path),
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    fails = json.loads(proc.stdout.strip().splitlines()[-1])
    assert set(fails) == set(modules), "every case must fail to import"
    own = ("zz_absent_1647", "zz_inner_1647", "zz_outer_1647", "json")
    for name, (_, attributed, accepted) in _CHILD_CASES.items():
        report = fails[f"zz1647.{name}"]
        assert report["missing"] == attributed, (name, report)
        verdict = sweep_verdict(
            f"zz1647.{name}", report, owned={f"zz1647.{name}": "standin"}, own_names_of=lambda _e: own
        )
        assert (verdict is None) is accepted, (name, verdict)
