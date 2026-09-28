"""Centralized path resolution for Maxim data files.

Two categories of data:

1. **Bundled defaults** — shipped inside the wheel at ``src/maxim/_data/``.
   Read-only templates, seed components, seed encounters.
   Accessed via :func:`bundled_data`.

2. **User-generated data** — written at runtime to ``~/.maxim/`` (or
   ``$MAXIM_DATA_HOME``).  Memories, sim reports, learned state, plans,
   downloaded models.  Accessed via :func:`data_home`.

Every module that previously did ``Path("data/util/foo.json")`` should
use one of these helpers instead so that ``pip install pymaxim`` works
outside a repo checkout.

Environment variables:
    MAXIM_DATA_HOME  Override user data directory (default ``~/.maxim``)
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Bundled data (read-only, shipped in wheel)
# ---------------------------------------------------------------------------

_bundled_data_cache: Path | None = None


def bundled_data() -> Path:
    """Return path to package-bundled data directory (read-only defaults).

    This resolves to ``src/maxim/_data/`` during development and to the
    installed package's ``_data/`` directory after ``pip install``.
    Result is cached after first call.
    """
    global _bundled_data_cache
    if _bundled_data_cache is None:
        import importlib.resources

        _bundled_data_cache = Path(str(importlib.resources.files("maxim") / "_data"))
    return _bundled_data_cache


def bundled_templates() -> Path:
    """Shortcut for ``bundled_data() / "templates"``."""
    return bundled_data() / "templates"


# ---------------------------------------------------------------------------
# User data (read-write, created on first use)
# ---------------------------------------------------------------------------

_data_home_cache: Path | None = None


def data_home() -> Path:
    """Return user data directory, creating it on first access.

    Default: ``~/.maxim``.  Override with ``$MAXIM_DATA_HOME``.
    Result is cached after first call.  Call :func:`_reset_caches`
    to clear (used by tests that mock ``MAXIM_DATA_HOME``).
    """
    global _data_home_cache
    if _data_home_cache is not None:
        return _data_home_cache
    raw = os.environ.get("MAXIM_DATA_HOME", "")
    base = Path(raw) if raw else Path.home() / ".maxim"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        import sys

        sys.exit(
            f"Error: Cannot create data directory: {base}\n"
            f"Fix: Check permissions, or set MAXIM_DATA_HOME to a writable path."
        )
    _data_home_cache = base
    return base


def _reset_caches() -> None:
    """Clear path caches.  Used by tests that mock ``MAXIM_DATA_HOME``."""
    global _bundled_data_cache, _data_home_cache
    _bundled_data_cache = None
    _data_home_cache = None


def user_config() -> Path:
    """User config overrides — ``~/.maxim/config/``."""
    p = data_home() / "config"
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_memory() -> Path:
    """Memory persistence — ``~/.maxim/memory/``."""
    p = data_home() / "memory"
    p.mkdir(parents=True, exist_ok=True)
    return p


# The run directories a user names by ID. ``resolve_run_dir`` searches them and owns these names; the
# writers (sim_reports(), agent_data(), AgentFactory, load.agent, api.py's agent home) use the same
# constants, so a rename cannot leave the lookup behind.
_SIM_REPORTS = "sim_reports"
_AGENTS = "agents"
RUN_DIR_KINDS: dict[str, str] = {"sim": _SIM_REPORTS, "agent": _AGENTS}


def sim_reports() -> Path:
    """Simulation reports — ``~/.maxim/sim_reports/``."""
    p = data_home() / _SIM_REPORTS
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir() -> Path:
    """Downloaded models — ``~/.maxim/models/``."""
    p = data_home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def agent_data(agent_id: str) -> Path:
    """Per-agent data directory — ``~/.maxim/agents/{agent_id}/``."""
    p = data_home() / _AGENTS / agent_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def benchmarks_dir() -> Path:
    """Benchmark output — ``~/.maxim/benchmarks/``."""
    p = data_home() / "benchmarks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def provenance_dir() -> Path:
    """Provenance traces — ``~/.maxim/provenance/``."""
    p = data_home() / "provenance"
    p.mkdir(parents=True, exist_ok=True)
    return p


def sessions_dir() -> Path:
    """Live session recordings — ``~/.maxim/sessions/``."""
    p = data_home() / "sessions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def planning_dir() -> Path:
    """Planning state — ``~/.maxim/planning/``."""
    p = data_home() / "planning"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Convenience: resolve a config file with bundled fallback
# ---------------------------------------------------------------------------


def resolve_config(filename: str) -> Path:
    """Find a config file: check user config first, fall back to bundled.

    Args:
        filename: Config file name (e.g. ``"llm.json"``).

    Returns:
        Path to the file — user override if it exists, else bundled default.

    Raises:
        FileNotFoundError: If neither user nor bundled version exists.
    """
    user_path = user_config() / filename
    if user_path.is_file():
        return user_path

    bundled_path = bundled_templates() / filename
    if bundled_path.is_file():
        return bundled_path

    raise FileNotFoundError(
        f"Config file '{filename}' not found in {user_path} or {bundled_path}. "
        f"Copy a template from {bundled_templates()} to {user_config()} to get started."
    )


def store_file_path(path: str | os.PathLike[str]) -> str:
    """A persisted store's file path as the filesystem should see it: ``~`` expanded (#950).

    Every save/load of a memory store resolves its path through this, so ``~/...`` means the home
    directory wherever it is given -- a ``persistence_path``, a ``load()`` argument or a ``maxim.load``
    call -- and never a literal ``./~`` directory under the working directory.
    """
    return os.path.expanduser(os.fspath(path))


def resolve_user_state(relative_path: str) -> Path:
    """Resolve a user-state file path under ``~/.maxim/``.

    Creates parent directories as needed.  Use this for any file that was
    previously at ``data/util/foo.json`` — it becomes
    ``~/.maxim/foo.json``.

    Args:
        relative_path: Path relative to data_home, e.g. ``"util/nac_state.json"``
                        or ``"memory/hippocampus.json"``.
    """
    p = data_home() / relative_path
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class RunDirNotFound(ValueError):
    """``resolve_run_dir`` found no directory; the message names every place it searched."""


class RunDirAmbiguous(ValueError):
    """``resolve_run_dir`` found the ID in more than one place; the message names them."""


def resolve_run_dir(id_or_path: str, *, kinds: tuple[str, ...] = ("sim", "agent")) -> Path:
    """The directory a user means by ``id_or_path`` -- a simulation's or an agent's (1.3.1).

    The ONE answer to "which directory is run X" for the CLI (several places used to answer it on their
    own, and the substrate CLI looked in a directory nothing writes). Creates no run directory.

    - Anything that looks like a path (a separator, absolute, ``~`` or ``.``-prefixed) is a path, only --
      so an ID that starts with a dot is never searched by ID; pass its path.
    - A bare ID is looked up in the working directory and, per ``kinds``, in ``<data home>/sim_reports/``
      (``"sim"``: a simulation's reports and AUT substrate) and ``<data home>/agents/`` (``"agent"``: a
      ``maxim.create.agent()`` home). Found in exactly one place, that is the answer.

    Raises ``RunDirAmbiguous`` when a bare ID is found in more than one place (a guess could write to the
    wrong run) and ``RunDirNotFound`` naming every place searched when it is found in none.
    """
    unknown = [k for k in kinds if k not in RUN_DIR_KINDS]
    if unknown or not kinds:
        raise ValueError(f"resolve_run_dir: kinds must be drawn from {sorted(RUN_DIR_KINDS)}, got {kinds!r}")
    raw = str(id_or_path).strip()
    if not raw:
        # Path("") is ".": an empty argument would otherwise name the working directory.
        raise RunDirNotFound("empty run id or path")
    as_path = Path(raw).expanduser()
    if Path(raw).name != raw or raw.startswith(("~", ".")) or as_path.is_absolute():
        if as_path.is_dir():
            return as_path.resolve()
        raise RunDirNotFound(f"directory not found: {as_path}")
    home = data_home()
    places = [Path.cwd() / raw] + [home / RUN_DIR_KINDS[k] / raw for k in kinds]
    # De-duplicated: run from inside sim_reports/ (or agents/), the working directory IS the data-home hit.
    found = list(dict.fromkeys(place.resolve() for place in places if place.is_dir()))
    if len(found) == 1:
        return found[0]
    if found:
        listed = ", ".join(str(place) for place in found)
        raise RunDirAmbiguous(f"{raw!r} names more than one directory ({listed}); pass the path instead")
    listed = ", ".join(str(place) for place in places)
    raise RunDirNotFound(f"no directory named {raw!r}: looked in {listed}")
