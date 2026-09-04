"""Save and restore recent CLI invocations for `maxim --last`.

Persists the last N unique invocations (simulation agent, agentic mode)
to ``<data_home>/last_runs.json`` — ``~/.maxim`` by default, or
``$MAXIM_DATA_HOME`` when set (resolved lazily through
``maxim.utils.paths`` so a read-only-root sandbox with only the data home
writable still works). Trivial commands (--help, --show-last, etc.) are
not saved.

Usage:
    maxim --last          # Re-run most recent
    maxim --last 3        # Re-run 3rd most recent
    maxim --show-last     # Show all saved runs
    maxim --clear-last    # Clear saved runs
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from maxim.utils.paths import resolve_user_state

# Args that indicate a "real" run worth saving
_SAVEABLE_INDICATORS = {"--sim", "--mode", "--generate-simulation"}

# Args that should NOT trigger a save (meta/utility commands)
_SKIP_INDICATORS = {
    "--last",
    "--show-last",
    "--clear-last",
    "--help",
    "-h",
    "--clear-cache",
    "--clear-memory",
    "--audit-architecture",
}

_MAX_SAVED_RUNS = 5


def _last_runs_path() -> Path:
    """Resolve ``<data_home>/last_runs.json`` at call time, never at import.

    A module-level ``Path.home() / ".maxim"`` constant bypassed
    ``MAXIM_DATA_HOME`` (and every test override of it); routing through
    :func:`maxim.utils.paths.resolve_user_state` keeps one source of truth.
    """
    return resolve_user_state("last_runs.json")


def should_save(argv: list[str]) -> bool:
    """Check if this invocation is worth saving."""
    if any(arg in argv for arg in _SKIP_INDICATORS):
        return False
    return any(arg in argv for arg in _SAVEABLE_INDICATORS)


def save_last_run(argv: list[str]) -> None:
    """Save the current invocation. Deduplicates and keeps last N unique."""
    try:
        runs = _load_all()

        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "args": argv,
            "command": "maxim " + " ".join(_quote_args(argv)),
        }

        # Remove duplicate if same args already saved
        cmd_key = " ".join(sorted(argv))
        runs = [r for r in runs if " ".join(sorted(r.get("args", []))) != cmd_key]

        # Prepend new entry (most recent first)
        runs.insert(0, entry)

        # Keep only last N
        runs = runs[:_MAX_SAVED_RUNS]

        from maxim.utils.atomic_io import atomic_write_json
        from maxim.utils.format_version import with_format_version

        atomic_write_json(str(_last_runs_path()), with_format_version({"runs": runs}))
    except Exception:
        pass  # Best-effort


def load_last_run(index: int = 1) -> dict[str, Any] | None:
    """Load a saved invocation by index (1 = most recent, 2 = second, etc.)."""
    runs = _load_all()
    idx = max(1, index) - 1  # Convert 1-based to 0-based
    if idx < len(runs):
        return runs[idx]
    return None


def load_all_runs() -> list[dict[str, Any]]:
    """Load all saved invocations."""
    return _load_all()


def clear_last_run() -> bool:
    """Delete all saved runs. Returns True if deleted."""
    try:
        path = _last_runs_path()
        if path.is_file():
            path.unlink()
            return True
    except Exception:
        pass
    return False


def format_last_run(data: dict[str, Any], index: int = 0) -> str:
    """Format a single run entry for display."""
    cmd = data.get("command", "maxim " + " ".join(data.get("args", [])))
    ts = data.get("timestamp", "unknown")
    prefix = f"  [{index}]" if index > 0 else "  "
    return f"{prefix} ({ts})  $ {cmd}"


def format_all_runs() -> str:
    """Format all saved runs for display."""
    runs = _load_all()
    if not runs:
        return "  No saved runs."
    lines = []
    for i, run in enumerate(runs, 1):
        lines.append(format_last_run(run, index=i))
    return "\n".join(lines)


def _load_all() -> list[dict[str, Any]]:
    """Load the runs file."""
    try:
        path = _last_runs_path()
        if path.is_file():
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
            from maxim.utils.format_version import check_format_version

            if isinstance(data, dict):
                # Detect v1.0-shape BEFORE check_format_version so a
                # corrupted v1.0 file (lost the "runs" key but kept the
                # _format_version marker) returns [] cleanly instead of
                # treating the version envelope as a single legacy run
                # (CC1 review fold, executor #4).
                has_v1_marker = "_format_version" in data
                check_format_version(data, "last_runs", log=logging.getLogger(__name__))
                # v1.0 wraps the list under "runs".
                runs = data.get("runs")
                if isinstance(runs, list):
                    return runs
                if has_v1_marker:
                    return []
                # Pre-1.0 stored a single run as a dict at root.
                return [data]
    except Exception:
        pass
    return []


def _quote_args(argv: list[str]) -> list[str]:
    """Quote args that contain spaces for display."""
    quoted = []
    for arg in argv:
        if " " in arg:
            quoted.append(f'"{arg}"')
        else:
            quoted.append(arg)
    return quoted
