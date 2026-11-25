"""``child_env`` for scripts, loaded from THIS repo's ``src/maxim/utils/child_env.py`` by path (#1593).

A harness must not import the ``maxim`` package to build its children's environment, because which ``maxim`` it
would get is the very question ``_provenance.assert_repo_interpreter`` answers. So this loads the one source file
by its path, never through ``sys.path``, and re-exports its functions. That module is stdlib only. One
definition serves ``src/`` and ``scripts/``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SOURCE = Path(__file__).resolve().parent.parent / "src" / "maxim" / "utils" / "child_env.py"
_spec = importlib.util.spec_from_file_location("_maxim_child_env_by_path", _SOURCE)
if _spec is None or _spec.loader is None:  # pragma: no cover - the file ships with the repo
    raise ImportError(f"cannot load {_SOURCE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

child_env = _module.child_env
own_site_dirs = _module.own_site_dirs
strip_own_site_dirs = _module.strip_own_site_dirs

__all__ = ["child_env", "own_site_dirs", "strip_own_site_dirs"]
