"""Persistence file-format version helper (v1.0 freeze-critical contract).

Every persisted JSON file in Maxim that represents long-lived bio-system,
runtime, or session state carries a ``_format_version`` field at its root.
Newly-written files default to :data:`FORMAT_VERSION` (``"1.0"``). Loaders
default to ``"0.x"`` when the field is absent and emit a one-time warning
per file type so an operator restoring a 0.8 session into a 1.0+ build
sees the version drift surfaced (instead of a silent partial load).

Why a separate field from ``snapshot.py``'s ``schema_version``
=============================================================

``maxim.memory.snapshot`` already defines an envelope-layer
``schema_version`` (int) for the six envelope-conformant bio-systems
(ATL, Hippocampus, NAc, SCN, PerceptTraceBuffer, CrossLayerGraph).
That field is **load-bearing** for the P3.5 Stage 2 forward-migration
registry — bumping it triggers a registered migration function. It is
intentionally only present on files written through the envelope
(``SessionSnapshot.write`` or ``wrap_envelope``).

``_format_version`` is the **broader** contract: every JSON file Maxim
writes carries it, including individual ``bio_system.save(path)`` files
that bypass the envelope, sim reports, plan documents, foundry batch
metadata, learned tool indices, web caches, and so on. The two
conventions coexist:

- ``schema_version`` (int) → envelope migration trigger (snapshot.py)
- ``_format_version`` (string) → file-format contract for everything else

A file written by ``atl.save(path)`` has BOTH at root: ``_format_version``
(this module) lives at the JSON root alongside the payload keys; the
existing ``"version": "1.0"`` payload-string is tombstoned per the
snapshot.py docstring. Re-bumping the tombstoned payload version is
forbidden — bump ``_format_version`` here when the file shape itself
changes, or bump the envelope ``schema_version`` and register a
migration when the inner shape needs structural transformation.

Loader contract
===============

Every loader that reads a JSON file written by Maxim should call
:func:`check_format_version` on the parsed dict before unpacking it.
The helper:

1. Returns the version string (``"1.0"`` for new files, ``"0.x"`` for
   pre-1.0 files missing the field).
2. Warns ONCE per ``file_type`` when the field is missing, so a sweep
   over many files (e.g., loading 12 NAc bias snapshots) doesn't spam
   the log.
3. Future-proofs a 1.1 schema bump: a loader can compare the returned
   version against an expected set and raise / down-convert as needed.

Loaders MUST NOT raise on missing ``_format_version``. The default
``"0.x"`` allows old files to continue loading; structural rejection
lives in the existing payload validation logic, not here.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Bump in lockstep with file-shape changes that older Maxim builds cannot
# parse cleanly. v1.0 is the inaugural value; pre-1.0 files have no field
# at all and are detected as ``"0.x"``.
FORMAT_VERSION: str = "1.0"

# Special sentinel returned when the field is absent. Keep this as a
# constant so callers can compare without re-typing the string.
LEGACY_VERSION: str = "0.x"

# The root field name. Underscore-prefixed so it sorts before payload
# keys in ``json.dumps(..., sort_keys=True)`` outputs and so callers
# scanning for "obvious metadata" can spot it.
_FIELD: str = "_format_version"

# Set of ``file_type`` strings we have already warned about for missing
# version. Process-wide, intentional — the warning is "you are loading
# a pre-1.0 file"; once the operator has seen it for a given file type
# we do not need to repeat it for every bias snapshot or session
# directory in a sweep. Tests that assert warning emission must clear
# this set up-front (use :func:`reset_warned_types` from a test fixture).
_warned_types: set[str] = set()


def with_format_version(payload: dict[str, Any], version: str = FORMAT_VERSION) -> dict[str, Any]:
    """Return ``payload`` annotated with ``_format_version`` at root.

    Mutates the input dict in place AND returns it (callers may chain).

    **Fail-loud on stale conflict** (CC1 review fold, arch B1 + executor #6).
    The pre-fold version used ``setdefault`` so an existing
    ``_format_version`` key was silently preserved. That hides exactly
    the failure mode this contract is meant to surface: a stale
    ``"0.x"`` (or otherwise wrong) literal anywhere in the payload
    survives into the persisted file. Per CLAUDE.md's silent-no-op
    invariant lesson, freeze-critical contracts must fail loudly when
    the assumption breaks. So:

    - If ``_format_version`` is absent, set it to ``version``.
    - If ``_format_version`` is already equal to ``version``, no-op.
    - If ``_format_version`` is present but different (a stale value
      survived a load-modify-save round-trip), raise ``ValueError`` so
      the caller sees the bug.

    Raises ``TypeError`` if ``payload`` is not a dict — the format
    contract is JSON-object-rooted, never list-rooted. Loaders relying
    on ``check_format_version`` would have nowhere to read the field
    from a list-rooted document.
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"with_format_version: payload must be dict, got {type(payload).__name__}; "
            "the _format_version contract requires a JSON object at root"
        )
    existing = payload.get(_FIELD)
    if existing is None:
        payload[_FIELD] = version
    elif existing != version:
        raise ValueError(
            f"with_format_version: payload already carries {_FIELD}={existing!r} "
            f"but writer requested {version!r}; refusing to silently re-stamp. "
            "If you are re-saving a payload that came from a versioned file, "
            "delete the field first or pass a matching version."
        )
    return payload


def check_format_version(
    data: Any,
    file_type: str,
    *,
    log: logging.Logger | None = None,
) -> str:
    """Inspect a parsed JSON document and return its ``_format_version``.

    Parameters
    ----------
    data
        The parsed JSON value. Typically a dict; if not, the function
        returns :data:`LEGACY_VERSION` without warning (the loader's
        own validation will surface the type error).
    file_type
        A short stable label identifying the kind of file (``"atl"``,
        ``"nac"``, ``"session_report"``, ...). Used to dedupe warnings:
        only one warning per file_type per process. Pick something
        stable across versions — file paths churn, file types do not.
    log
        Optional caller-supplied logger. Defaults to this module's
        logger so warnings carry the ``maxim.utils.format_version``
        path; callers that want their own module name in the log
        should pass ``log=<their logger>``.

    Returns
    -------
    str
        The version string. ``FORMAT_VERSION`` (``"1.0"``) for files
        written by 1.0+ builds, :data:`LEGACY_VERSION` (``"0.x"``)
        when the field is absent.
    """
    if not isinstance(data, dict):
        return LEGACY_VERSION

    raw = data.get(_FIELD)
    if isinstance(raw, str) and raw:
        return raw

    if file_type not in _warned_types:
        _warned_types.add(file_type)
        (log or logger).warning(
            "Loading pre-1.0 %s file (no _format_version at root); assuming 0.x. "
            "Re-save with this build to upgrade the file format. "
            "Future warnings for %s suppressed.",
            file_type,
            file_type,
        )
    return LEGACY_VERSION


def reset_warned_types() -> None:
    """Clear the per-file-type warning dedupe set.

    Test-only helper. Production code should not need to reset the
    dedupe state — the warning is intentionally one-shot per process.
    Tests that assert ``logger.warning`` emission must call this in a
    fixture so each test starts with a clean dedupe set; otherwise the
    second test that loads a pre-1.0 ATL file sees no warning and
    fails despite working code.
    """
    _warned_types.clear()


__all__ = [
    "FORMAT_VERSION",
    "LEGACY_VERSION",
    "check_format_version",
    "reset_warned_types",
    "with_format_version",
]
