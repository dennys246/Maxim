"""The environment a child process inherits (#1593).

Some wheels rewrite ``os.environ`` at interpreter startup. On macOS, the GStreamer wheels that ``reachy_mini``
pulls in ship ``gstreamer_bundle.pth`` and ``gstreamer_ext_runtime.pth``. Both call
``gstreamer_libs.setup_python_environment()``, which does ``os.environ.update(...)`` with a ``PYTHONPATH`` that
starts with this venv's ``site-packages``. The process that ran the hook is unaffected, because its ``sys.path``
was computed first. Every child it spawns inherits the variable, though, and so:

- a child ``maxim`` imports the copy in ``site-packages``, not the checkout on the parent's ``PYTHONPATH``. A
  stray copy there shadowed the editable install on the rig from 2026-10-16 to 2026-11-03;
- a child that runs another interpreter, such as another repo's venv, sees this venv's ~190 packages as its own.
  Maxim Overdrive's lock check failed this way on 2026-11-23.

:func:`child_env` is the one place a spawn site gets its environment. It returns a copy without the
``PYTHONPATH`` entries that point into the RUNNING interpreter's own site-packages directories. A Python child
from the same venv loses nothing: its own ``.pth`` hooks run again and put back what they need. A child of
another interpreter would only be misled by them. Every other variable passes through unchanged, including the
GStreamer and GI paths the hook set: a GStreamer child, such as the plugin scanner, is spawned by libgstreamer
from ``os.environ``, never through here, so it keeps them. ``VIRTUAL_ENV`` is not touched, because it does not
change any interpreter's ``sys.path``.

Stdlib only, and no ``maxim`` imports. Scripts load this file by path (``scripts/_child_env.py``), so a harness
never has to import the ``maxim`` package it is checking. Regression guard:
``tests/unit/test_child_env.py`` (a two-process test with a planted ``.pth``) and ``scripts/lint_child_env.py``
(the spawn-site ratchet).
"""

from __future__ import annotations

import os
import site
import sys
from collections.abc import Mapping

__all__ = ["child_env", "own_site_dirs", "strip_own_site_dirs"]


def own_site_dirs() -> tuple[str, ...]:
    """The running interpreter's own site-packages directories, resolved: ``site.getsitepackages()``, plus the
    user site when it is enabled (a ``pip install --user`` puts a hook's entries there)."""
    dirs = [os.path.realpath(d) for d in site.getsitepackages()]
    if site.ENABLE_USER_SITE:
        dirs.append(os.path.realpath(site.getusersitepackages()))
    # A venv built without site.getsitepackages() support still has purelib under its prefix.
    purelib = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}")
    dirs.append(os.path.realpath(os.path.join(purelib, "site-packages")))
    return tuple(dict.fromkeys(dirs))


def _inside(path: str, roots: tuple[str, ...]) -> bool:
    if not os.path.isabs(path):
        return False  # a relative entry (``src``) is the caller's choice, never a hook's
    real = os.path.realpath(path)
    return any(real == root or real.startswith(root + os.sep) for root in roots)


def strip_own_site_dirs(pythonpath: str, roots: tuple[str, ...] | None = None) -> str:
    """``pythonpath`` without its entries inside ``roots`` (default :func:`own_site_dirs`), order kept.

    An empty entry (``:/x``, ``/x:``) is kept: Python reads it as the working directory, so an operator's
    ``export PYTHONPATH=$PYTHONPATH:/x`` keeps meaning what it meant."""
    roots = own_site_dirs() if roots is None else roots
    kept = [entry for entry in pythonpath.split(os.pathsep) if not _inside(entry, roots)]
    return os.pathsep.join(kept)


def child_env(environ: Mapping[str, str] | None = None, **overrides: str) -> dict[str, str]:
    """A copy of ``environ`` (default ``os.environ``) fit for a child process, with ``overrides`` applied last.

    ``PYTHONPATH`` keeps every entry outside this interpreter's site-packages, in order, empty entries included.
    When nothing is left the variable is removed (an empty ``PYTHONPATH`` and none mean the same to Python). An
    override is applied as given: a caller that sets ``PYTHONPATH`` gets exactly that value.

    Only THIS interpreter's site dirs are stripped. A ``PYTHONPATH`` that another venv's hook already rewrote,
    before this process started, passes through unchanged.
    """
    env = dict(os.environ if environ is None else environ)
    if "PYTHONPATH" in env:
        stripped = strip_own_site_dirs(env["PYTHONPATH"])
        if stripped:
            env["PYTHONPATH"] = stripped
        else:
            del env["PYTHONPATH"]
    env.update(overrides)
    return env
