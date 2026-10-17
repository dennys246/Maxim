"""No test may leave a ``scripts/`` module replaced or removed under the name ``import`` resolves it to (#1339).

A script imports its helpers by name, often lazily (``analyze_exp56.py::main`` runs ``from _provenance import
...`` at CALL time), so it gets whatever ``sys.modules["_provenance"]`` holds then. A test that patches a helper
holds the object it imported at collection (``import _provenance as P``). A test that re-executed the script with
``spec_from_file_location("_provenance", ...)`` and ``sys.modules["_provenance"] = module``, and never put the old
one back, made those two different objects for every later test in the process: the patch landed on the stale one
and the script ran the real code. Serially the leaking file sorted after the patching one, so nothing failed; under
``--dist loadfile`` a worker ran them the other way round (CI run 39050775378: ``test_an_analyzer_refuses_with_3...``
returned 4, the real provenance check passing on a clean checkout). History:
docs/lessons/script-module-identity-leak.md.

``tests/conftest.py`` checks around every test (``_scripts_modules_not_replaced``), every test module
(``_scripts_modules_not_replaced_by_module``, which sees module- and class-scoped fixtures) and every collected
module (the ``pytest_make_collect_report`` hookwrapper). A leak fails the test, the module's last teardown or the
collection, and the original entries are then put back so later tests on that worker do not fail for the wrong
reason (the failure is already reported).

A name is guarded when its entry is a module loaded from this repo's ``scripts/`` and either (a) the name is that
file's OWN import name with ``scripts/`` or any directory below it as the anchor (``scripts/a/b.py`` registered as
``a.b`` or ``b``), whatever ``sys.path`` says, since ``import`` returns ``sys.modules[name]`` without consulting the
path; or (b) ``import`` would resolve the name to a ``scripts/`` file under the live ``sys.path``: ``scripts.a.b``
always, and every module name relative to a ``sys.path`` entry inside ``scripts/`` (such as
``scripts/orient_backbone``). Adding an entry that was absent is an ordinary first import and stays allowed;
replacing or removing one that was present is the leak. A test that needs a private copy loads it under a private
name (``m1b5a_...``) or registers it with ``monkeypatch.setitem(sys.modules, ...)``, which restores it.

Blind spots (stated, not covered):

- ``importlib.reload`` keeps the same object and replaces its contents; identity is unchanged, so it is not seen.
- A first import and a replacement inside one checked window (one test, one module, one collection) look like a
  first import: the snapshot never saw the original.
- Session- and package-scoped fixtures: their setup runs before the first module snapshot, their teardown after the
  last check.
- Scripts loaded from ANOTHER tree (e.g. a path built from ``maxim.__file__`` when an editable install points at
  a different checkout) are outside this ``scripts/`` and unguarded.
- A namespace package directory (``__file__`` is None) is unguarded; its submodules, which have files, are guarded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = (Path(__file__).resolve().parents[1] / "scripts").resolve()

_file_parts: dict[tuple[str, str], tuple[str, ...] | None] = {}
_entry_names: dict[tuple[str, str], frozenset[str]] = {}
_names_for_path: dict[tuple[tuple[str, ...], str], frozenset[str]] = {}


def _module_names(root: Path, prefix: str = "") -> frozenset[str]:
    names: set[str] = set()
    for path in root.rglob("*.py"):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            names.add(prefix + ".".join(parts))
    return frozenset(names)


_SCRIPTS_PACKAGE_NAMES = _module_names(SCRIPTS, "scripts.") | {"scripts"}


def _own_names(root: Path) -> frozenset[str]:
    """Every name a ``scripts/`` file is its own import name under, with ``scripts/`` or any directory below it as the
    anchor: ``scripts/a/b.py`` is ``a.b`` and ``b``. Independent of ``sys.path``."""
    names: set[str] = set()
    for dotted in _module_names(root):
        parts = dotted.split(".")
        names.update(".".join(parts[i:]) for i in range(len(parts)))
    return frozenset(names)


_OWN_NAMES = _own_names(SCRIPTS)


def _names_from_entry(entry: str, cwd: str) -> frozenset[str]:
    key = (entry, cwd if not os.path.isabs(entry) else "")
    names = _entry_names.get(key)
    if names is None:
        names = frozenset()
        try:
            root = Path(cwd, entry).resolve()
            if root.is_relative_to(SCRIPTS) and root.is_dir():
                names = _module_names(root)
        except (OSError, ValueError):
            pass  # an unresolvable path entry contributes no names; nothing can import from it either
        _entry_names[key] = names
    return names


def guarded_names() -> frozenset[str]:
    """Every name ``import`` could resolve to a ``scripts/`` file under the current ``sys.path``."""
    cwd = os.getcwd()
    key = (tuple(p for p in sys.path if isinstance(p, str)), cwd)
    names = _names_for_path.get(key)
    if names is None:
        names = _SCRIPTS_PACKAGE_NAMES.union(*(_names_from_entry(entry, cwd) for entry in key[0]))
        _names_for_path[key] = names
    return names


def _script_parts(module: object) -> tuple[str, ...] | None:
    """The module's file as dotted parts relative to ``scripts/`` (``a/b.py`` -> ``("a", "b")``), or None when it is
    not a ``scripts/`` file. Cached per ``__file__`` (and per cwd when that is relative)."""
    file = getattr(module, "__file__", None) if isinstance(module, ModuleType) else None
    if not isinstance(file, str):
        return None
    key = (file, "" if os.path.isabs(file) else os.getcwd())
    if key in _file_parts:
        return _file_parts[key]
    parts: tuple[str, ...] | None = None
    try:
        path = Path(key[1], file).resolve()
        if path.is_relative_to(SCRIPTS):
            rel = list(path.relative_to(SCRIPTS).with_suffix("").parts)
            parts = tuple(rel[:-1] if rel and rel[-1] == "__init__" else rel)
    except (OSError, ValueError):
        parts = None  # an unresolvable __file__ is not a scripts/ file
    _file_parts[key] = parts
    return parts


def _is_guarded(name: str, module: object, path_names: frozenset[str]) -> bool:
    """``name`` is guarded for this module when it is a ``scripts/`` module and either ``import`` resolves ``name`` to
    a ``scripts/`` file under the current ``sys.path``, or ``name`` is the module's OWN import name with some directory
    of ``scripts/`` as the anchor (a suffix of its relative path), whatever ``sys.path`` says: ``import`` returns
    ``sys.modules[name]`` without consulting the path. Private names (``m1b5a_...``) match neither."""
    parts = _script_parts(module)
    if parts is None:
        return False
    if name in path_names:
        return True
    split = tuple(name.split("."))
    return 0 < len(split) <= len(parts) and parts[-len(split) :] == split


def snapshot() -> dict[str, object]:
    """The ``scripts/`` modules currently registered under a guarded name."""
    modules = sys.modules
    path_names = guarded_names()
    return {
        name: mod
        for name in path_names | _OWN_NAMES
        if (mod := modules.get(name)) is not None and _is_guarded(name, mod, path_names)
    }


def leaks(before: dict[str, object]) -> list[str]:
    """Every entry in ``before`` that is now a different object, or gone."""
    out = []
    for name, mod in sorted(before.items()):
        now = sys.modules.get(name)
        if now is None:
            out.append(f"sys.modules[{name!r}] was removed")
        elif now is not mod:
            out.append(f"sys.modules[{name!r}] was replaced by another module object")
    return out


def restore(before: dict[str, object]) -> None:
    """Put back every entry in ``before`` (called only after a leak has been reported)."""
    for name, mod in before.items():
        sys.modules[name] = mod  # type: ignore[assignment]


def check_and_restore(before: dict[str, object]) -> str | None:
    """The failure message for any leak since ``before`` (entries restored), or None."""
    found = leaks(before)
    if not found:
        return None
    restore(before)
    return (
        "a scripts/ module was replaced or removed under the name import resolves it to, and not restored "
        "(tests/script_module_guard.py): " + "; ".join(found) + ". Load a private copy under a private name, "
        "or register it with monkeypatch.setitem(sys.modules, ...). The original entries have been put back."
    )
