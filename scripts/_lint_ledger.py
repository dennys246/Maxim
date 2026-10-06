"""Shared ledger rules for the ratchet lints that carry an owner-reviewed allowance list (#1091).

``lint_function_length.py`` (exceptions) and ``lint_coverage.py`` (the exclusion ledger) had copied both rules
verbatim. Like ``_lint_git.py``, this is a stdlib-only helper the lints import by path.

Not shared, on purpose: the TYPE_CHECKING detectors. ``lint_mypy_ratchet.py`` matches ``TYPE_CHECKING`` and ``MYPY``
because mypy treats both as true; ``lint_coverage.py`` matches ``TYPE_CHECKING`` only, mirroring coverage.py's
``exclude_lines`` in ``pyproject.toml``. One shared detector would widen the coverage gate's exemption to
``if MYPY:`` blocks, which coverage.py still measures.
"""

from __future__ import annotations

import re

REF_RE = re.compile(r"#\d+|https://github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+")


def ref_ok(ref: object) -> bool:
    """A ledger ref: an issue/PR number (``#123``) or a github.com PR/issue URL, nothing else."""
    return isinstance(ref, str) and bool(REF_RE.fullmatch(ref))


def append_only_problem(base: list, head: list, what: str) -> str | None:
    """The merge-base list, as parsed records, must be an exact prefix of HEAD's. ``what`` is the subject with its
    verb ("ledger is", "exceptions are"). None when it holds."""
    if head[: len(base)] == base:
        return None
    return f"{what} append-only: the merge-base list is not an exact prefix of this one (edited, reordered or removed)"
