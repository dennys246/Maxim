"""The scenario files a run actually read (baseline floors B0).

The loaders a sim's world comes from (an embodiment spec, a component resolved by ref, a reflex file, an arc,
a DM campaign, a scenario fixture) read their YAML through :func:`read_recorded` (and parse the bytes it returns), which
records ``{path, sha256}``. A sim takes a :func:`mark` at its start and reports :func:`since` that
mark. A file loaded BEFORE the mark (a DM campaign the CLI loads before starting the sim, a component spec
served from a registry cache) is carried on the object that holds it (``source_inputs``) and re-recorded with
:func:`renote` after the mark.

The list shows what is covered; a loader that does not use ``read_recorded`` is not in it, and neither is a world
the narrator generates at run time or ``simulation_generator.py``'s output. Scenarios that callers load before
the sim and hand over as plain turn dicts (``pre_campaign_turns``: ``api.simulate(scenario=...)``, the
``--research`` path, ``experiment.load_campaign_turns``) are not listed; their CONTENT enters the sim's
``scenario_sha256`` instead (``simulation/serving_provenance.py``). The record grows for the process's lifetime
(one small entry per load or cache hit); a sim reads only its own window of it.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable
from pathlib import Path


_LOCK = threading.Lock()
_READ: list[dict[str, str]] = []


def read_recorded(path: str | os.PathLike[str]) -> bytes:
    """``path``'s bytes, recording the path and the SHA-256 of exactly those bytes; the loader parses what this
    returns, so the hash covers what was parsed. Raises what reading raises (the loader's own error surface)."""
    resolved = os.path.realpath(path)
    data = Path(resolved).read_bytes()
    with _LOCK:
        _READ.append({"path": resolved, "sha256": hashlib.sha256(data).hexdigest()})
    return data


def renote(records: Iterable[dict[str, str]]) -> None:
    """Record again files read earlier (from an object's ``source_inputs`` or a cache)."""
    with _LOCK:
        _READ.extend({"path": r["path"], "sha256": r["sha256"]} for r in records)


def mark() -> int:
    with _LOCK:
        return len(_READ)


def since(start: int) -> list[dict[str, str]]:
    """The distinct files recorded after ``start`` (from :func:`mark`), in first-seen order: a file read or
    re-recorded many times (a cached component resolved every turn) appears once, so the window describes
    WHAT was read, not how often."""
    with _LOCK:
        seen: set[tuple[str, str]] = set()
        out = []
        for r in _READ[start:]:
            key = (r["path"], r["sha256"])
            if key not in seen:
                seen.add(key)
                out.append(dict(r))
        return out
