"""Shared ledger rules for the ratchet lints that carry an owner-reviewed allowance list (#1091).

``lint_function_length.py`` (exceptions) and ``lint_coverage.py`` (the exclusion ledger) had copied both rules
verbatim. Like ``_lint_git.py``, this is a stdlib-only helper the lints import by path.

Not shared, on purpose: the TYPE_CHECKING detectors answer different questions. ``lint_mypy_ratchet.py`` asks what
mypy treats as always true (``TYPE_CHECKING`` and ``MYPY``). ``lint_coverage.py`` asks only whether a block that
coverage.py's ``if TYPE_CHECKING:`` regex ALREADY excluded is imports-only: both its callers (``exclusion_matches``
and the diff rule's ``f.excluded`` branch) reach the detector only on a line that regex matched, so its name set is
gated by the regex and never decides what is excluded. Sharing would couple a gate to the other lint's semantics
for no behaviour change.
"""

from __future__ import annotations

import re

REF_RE = re.compile(r"#\d+|https://github\.com/[\w.-]+/[\w.-]+/(?:pull|issues)/\d+")


def ref_ok(ref: object) -> bool:
    """A ledger ref: an issue/PR number (``#123``) or a github.com PR/issue URL, nothing else."""
    return isinstance(ref, str) and bool(REF_RE.fullmatch(ref))


def append_only_problem(*, base: list, head: list, what: str) -> str | None:
    """The merge-base list, as parsed records, must be an exact prefix of HEAD's. ``what`` is the subject with its
    verb ("ledger is", "exceptions are"). None when it holds."""
    if head[: len(base)] == base:
        return None
    return f"{what} append-only: the merge-base list is not an exact prefix of this one (edited, reordered or removed)"
