"""No test may leave a registered ``maxim`` submodule detached from its package, or replace the package (#1755).

Two full local suites (Session V's 10a at f826e33c, the #1519 merge at 33084f04) failed the same way, from
``tests/unit/test_reachy_speaker_sink.py`` onward: ``module 'maxim' has no attribute 'utils'`` (and ``runtime``,
``tunnel``, ``embodied_runtime``, ``models``) for every test that monkeypatches a dotted ``maxim.*`` path. The
submodules were still in ``sys.modules`` but no longer attributes of their package, so ``maxim.__getattr__`` raised.
Neither run named the test that did it: every later test failed instead. Two shapes produce that state:

- a fresh ``maxim`` package object while its submodules stay registered (the package removed or replaced in
  ``sys.modules`` and re-imported);
- an undo deleting an attribute the package GAINED in the window, e.g. ``monkeypatch.setattr(maxim, "tunnel", x,
  raising=False)`` before ``maxim.tunnel`` is first imported: the import sets the real module as the attribute, and
  the undo, having no old value, deletes it. ``patch.dict(vars(pkg))`` and ``monkeypatch.delattr`` of a name imported
  in the window do the same.

So the check is an INVARIANT, not a diff against the attributes seen at the start: when the window closes, every
module registered as ``maxim.a.b`` whose registered package ``maxim.a`` is a module must hold it as attribute ``b``
(read from ``vars()``, so the lazy ``__getattr__`` is not run), and ``sys.modules["maxim"]`` must be the object it
was. Names already detached when the window opened are excluded, so a test is never blamed for damage it inherited.

``tests/conftest.py`` checks around every test (``_maxim_package_attributes_intact``), every test module
(``..._by_module``, which sees module- and class-scoped fixtures) and every collected module (the
``pytest_make_collect_report`` hookwrapper), the same three windows as ``tests/script_module_guard.py``. A leak fails
the test, the module's last teardown or the collection, naming what was detached, and the state is then repaired so
later tests do not fail for the wrong reason (the failure is already reported): the package object goes back, then
every detached submodule is re-attached as the object ``sys.modules`` NOW registers, so ``import`` and attribute
access agree (re-attaching a stale snapshot object would split them, the #1339 identity class).

Blind spots (stated, not covered):

- A submodule replaced by a fresh re-import whose package attribute still points at SOME module: attribute access
  and dotted patch paths still resolve, which is all #1755 breaks.
- A name detached before the window opened and still detached at its close: excluded (inherited), so the window that
  detached it is the one that reports it. Damage present before the first window (at conftest import) is never
  reported.
- Session- and package-scoped fixtures: their setup runs before the first module snapshot, their teardown after the
  last check.
- Damage done by a background thread is reported by whichever window is open when it lands, which may not be the
  test that started the thread.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

ROOT = "maxim"


@dataclass(frozen=True)
class Snapshot:
    """The root package object and the names already detached when the window opened.

    Test-only and in-process: never persisted and never crosses a wire, so CC3 does not apply."""

    root_name: str
    root: object
    detached: frozenset[str]


def _detached(root: str) -> list[str]:
    """Every registered ``root.a.b`` module whose registered package ``root.a`` is a module lacking attribute ``b``."""
    modules = sys.modules
    prefix = root + "."
    out = []
    for name, mod in list(modules.items()):
        if not name.startswith(prefix) or not isinstance(mod, ModuleType):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = modules.get(parent_name)
        if isinstance(parent, ModuleType) and attr not in vars(parent):
            out.append(name)
    return sorted(out)


def snapshot(root: str = ROOT) -> Snapshot:
    """The ``root`` package as registered now, and the names already detached (inherited, never blamed)."""
    return Snapshot(root, sys.modules.get(root), frozenset(_detached(root)))


def leaks(before: Snapshot) -> list[str]:
    """What broke in the window: the root package replaced or removed, or a submodule newly detached."""
    out = []
    now_root = sys.modules.get(before.root_name)
    if before.root is not None and now_root is not before.root:
        how = "removed" if now_root is None else "replaced by another module object"
        out.append(f"sys.modules[{before.root_name!r}] was {how}")
    out.extend(
        f"{name} is registered in sys.modules but detached from its package (no attribute {name.rpartition('.')[2]!r})"
        for name in _detached(before.root_name)
        if name not in before.detached
    )
    return out


def restore(before: Snapshot) -> None:
    """Put the root package back, then re-attach every detached submodule as the object ``sys.modules`` registers now
    (called only after a leak has been reported). Re-attaching after the root is back also covers submodules first
    imported in the window, which were attached only to the discarded package object."""
    if before.root is not None:
        sys.modules[before.root_name] = before.root  # type: ignore[assignment]
    for name in _detached(before.root_name):
        parent_name, _, attr = name.rpartition(".")
        setattr(sys.modules[parent_name], attr, sys.modules[name])


def check_and_restore(before: Snapshot) -> str | None:
    """The failure message for any leak since ``before`` (state repaired), or None."""
    found = leaks(before)
    if not found:
        return None
    restore(before)
    shown = found if len(found) <= 8 else [*found[:8], f"... and {len(found) - 8} more"]
    return (
        "a maxim submodule was left detached from its package, or the package was replaced in sys.modules "
        "(tests/package_attribute_guard.py, #1755): " + "; ".join(shown) + ". Every later test that patches a "
        "dotted maxim.* path would fail with \"module 'maxim' has no attribute ...\". Common causes: a "
        "monkeypatch.setattr(pkg, name, ..., raising=False) undone after the real submodule was imported, "
        "patch.dict(vars(pkg)), or removing the package from sys.modules. If this test only re-imported a package "
        "after an earlier failed import, the damage came from that import. The state has been repaired."
    )
