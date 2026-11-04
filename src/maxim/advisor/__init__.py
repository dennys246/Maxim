"""``maxim.advisor``: the atonia composition, an observer that proposes and never acts (#1315; Experimental).

Design: ``docs/plans/atonia_gate_approach_notes.md`` §4 (item 1) and §5 (item 5), with §1A winning over both. User
guide: ``docs/user/advisor.md``.

- :func:`open` (``maxim.advisor._session.open_session``) builds an :class:`AdvisorSession`;
  ``session.ingest(events)`` returns one report; ``session.close()`` writes the log's close record.
- :mod:`maxim.advisor.log` is item 5's proposal and refusal log, the observer's one sanctioned write besides its
  reports.

Nothing is imported at ``import maxim.advisor``: each name loads on first use, so the package's import closure is
what the observer itself needs (§4.2).
"""

from __future__ import annotations

from typing import Any

_FROM_SESSION = {"open": "open_session", "AdvisorSession": "AdvisorSession", "AdvisorLimits": "AdvisorLimits"}
_FROM_EXCEPTIONS = (
    "AdvisorError",
    "AdvisorCompositionError",
    "AdvisorInputError",
    "AdvisorStopped",
    "AdvisorReportWriteFailed",
    "SealedRegistryError",
    "AdvisorLogBroken",
    "AdvisorLogMalformed",
)

_SUBMODULES = ("log",)

__all__ = sorted([*_FROM_SESSION, *_FROM_EXCEPTIONS, *_SUBMODULES])


def __getattr__(name: str) -> Any:
    if name in _FROM_SESSION:
        from maxim.advisor import _session

        return getattr(_session, _FROM_SESSION[name])
    if name in _SUBMODULES:  # ``maxim.advisor.log.read_log`` in a reader process that never opened a session
        import importlib

        return importlib.import_module(f"maxim.advisor.{name}")
    if name in _FROM_EXCEPTIONS:
        from maxim import exceptions

        return getattr(exceptions, name)
    raise AttributeError(f"module 'maxim.advisor' has no attribute {name!r}")
