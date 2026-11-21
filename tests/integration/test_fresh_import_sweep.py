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
from tests.conftest import _optional_dependencies, extra_import_names

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

# Import names an extra brings in through its own requirements and that its modules import directly. ``pydantic``
# arrives with ``fastapi``; console/schemas.py imports it at module level.
IMPORTED_THROUGH: dict[str, tuple[str, ...]] = {"console": ("pydantic",)}

# The child reports each failure as data, so the verdict below can attribute it: ``missing`` is the dependency an
# ImportError names (``OptionalDependencyError.import_name``, else ``ImportError.name``; a hand-written
# ``ImportError("...")`` names none, so it attributes to nothing).
_CHILD = (
    "import importlib, json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "fails = {}\n"
    "for name in json.loads(sys.argv[2]):\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception as exc:\n"
    "        missing = getattr(exc, 'import_name', None) or (exc.name if isinstance(exc, ImportError) else None)\n"
    "        fails[name] = {\n"
    "            'text': f'{type(exc).__name__}: {exc}'[:300],\n"
    "            'is_import_error': isinstance(exc, ImportError),\n"
    "            'type_module': type(exc).__module__,\n"
    "            'missing': missing,\n"
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
) -> str | None:
    """Why ``module``'s fresh import is a defect, or ``None`` when it is acceptable. Pure, so the rule is testable.

    An unlisted module must import. A listed module must import when every import name of its extra is installed;
    when the extra is absent, its failure must be attributed to one of the extra's import names: an ImportError
    naming it, or an exception the package itself defines (``utils/optional_deps.py::import_refused``'s package
    rule). ``import_refused``'s blanket ImportError/OSError branch is not used here: it cannot say WHICH dependency
    refused, and a listed module failing on anything else is a defect like any other."""
    if failure is None:
        return None
    extra = owned.get(module)
    if extra is None:
        return f"{module}: {failure['text']}"
    own_names = extra_import_names(extra)
    if all(installed(n) for n in own_names):
        return f"{module}: fails with its extra [{extra}] installed: {failure['text']}"
    names = {_top(n) for n in own_names + IMPORTED_THROUGH.get(extra, ())}
    if failure["is_import_error"]:
        if _top(failure["missing"]) in names:
            return None
    elif _top(failure["type_module"]) in names:
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
    transitive = _failure("ModuleNotFoundError: No module named 'pydantic'", missing="pydantic")
    assert sweep_verdict("maxim.console.schemas", transitive, installed=_NOTHING_INSTALLED) is None
    own_error = _failure("RuntimeError: ...", missing=None, is_import_error=False, type_module="tensorflow.errors")
    assert sweep_verdict(_MOTOR, own_error, installed=_NOTHING_INSTALLED) is None


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


def test_a_listed_module_must_import_when_its_extra_is_installed() -> None:
    refusal = _failure("OptionalDependencyError: motor_cortex: required dependency 'keras'", missing="keras")
    assert sweep_verdict(_MOTOR, refusal, installed=_EVERYTHING_INSTALLED) is not None


def test_extra_owned_names_real_extras() -> None:
    extras = _optional_dependencies()
    unknown = {e for e in [*EXTRA_OWNED.values(), *IMPORTED_THROUGH] if e not in extras}
    assert not unknown, f"EXTRA_OWNED / IMPORTED_THROUGH name extras pyproject does not declare: {unknown}"


def test_the_child_reports_what_the_verdict_attributes(tmp_path: Path) -> None:
    """The child's report, on real failures: a hand-written ImportError names nothing, a missing module names
    itself, and the typed refusal names its import name."""
    pkg = tmp_path / "zz1647"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "plain.py").write_text("raise ImportError('needs keras')\n")
    (pkg / "missing.py").write_text("import zz_not_installed_1647\n")
    (pkg / "typed.py").write_text(
        "from maxim.utils.optional_deps import require_optional_dependency\n"
        "require_optional_dependency('zz_not_installed_1647', extra='training')\n"
    )
    env = _child_env(tmp_path)
    child = _CHILD.replace("sys.path.insert(0, sys.argv[1])\n", f"sys.path[:0] = [sys.argv[1], {str(SRC)!r}]\n")
    proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        [
            sys.executable,
            "-I",
            "-c",
            child,
            str(tmp_path),
            json.dumps(["zz1647.plain", "zz1647.missing", "zz1647.typed"]),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    fails = json.loads(proc.stdout.strip().splitlines()[-1])
    assert fails["zz1647.plain"]["missing"] is None
    assert fails["zz1647.missing"]["missing"] == "zz_not_installed_1647"
    assert fails["zz1647.typed"]["missing"] == "zz_not_installed_1647"
    assert all(f["is_import_error"] for f in fails.values())
