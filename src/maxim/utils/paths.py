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


def sim_reports() -> Path:
    """Simulation reports — ``~/.maxim/sim_reports/``."""
    p = data_home() / "sim_reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir() -> Path:
    """Downloaded models — ``~/.maxim/models/``."""
    p = data_home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def agent_data(agent_id: str) -> Path:
    """Per-agent data directory — ``~/.maxim/agents/{agent_id}/``."""
    p = data_home() / "agents" / agent_id
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
