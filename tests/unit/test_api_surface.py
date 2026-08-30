"""The advertised public API surface must match the real one.

Score card 2026-08-27, Documentation-honesty "Upgrade to B−" condition (2):
``tests/unit/test_api_surface.py`` asserting ``len(_API_VERBS)`` equals the
number in README.md.

WHAT THIS CAUGHT ON ITS FIRST RUN (2026-08-30)
----------------------------------------------
README advertised **17** verb-based functions. ``_API_VERBS`` held **21**. The
README number had been wrong through at least two releases, and nothing could
notice, because the only place the two claims met was a human reading both.
That is the whole failure mode this file exists for: a prose count is a claim,
and an unchecked claim rots.

The same run found ``"list_registered_tools"`` listed twice in the
``_API_VERBS`` literal — invisible at runtime because it is a ``frozenset``,
which is precisely why it survived.

WHY THE README NUMBER IS THE ANCHOR, NOT A CONSTANT HERE
--------------------------------------------------------
Hard-coding the count in this file would move the rot rather than stop it: the
test would agree with itself while README drifted. So the count is PARSED out
of README.md at run time, and the assertion is that the two independent
statements agree. Adding a verb therefore fails this test until README is
updated in the same commit — which is the intended cost.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import maxim

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# The sentence under test, e.g. "- **Use the Python API** — 21 verb-based
# functions for programmatic access". Anchored on "verb-based functions" so a
# stray number elsewhere in README cannot satisfy it.
_README_VERB_COUNT = re.compile(r"(\d+)\s+verb-based functions")


def _readme_verb_count() -> int:
    text = README.read_text(encoding="utf-8")
    matches = _README_VERB_COUNT.findall(text)
    assert matches, (
        "README.md no longer states an 'N verb-based functions' count — either restore it or retire this guard deliberately"
    )
    assert len(set(matches)) == 1, f"README.md states conflicting verb counts: {sorted(set(matches))}"
    return int(matches[0])


def test_readme_verb_count_matches_api_verbs():
    """The advertised number and the real surface are one claim, checked."""
    advertised = _readme_verb_count()
    actual = len(maxim._API_VERBS)
    assert actual == advertised, (
        f"README.md advertises {advertised} verb-based functions but "
        f"maxim._API_VERBS has {actual}. Update BOTH in the same commit — "
        f"verbs: {sorted(maxim._API_VERBS)}"
    )


@pytest.mark.parametrize("verb", sorted(maxim._API_VERBS))
def test_every_advertised_verb_resolves_and_is_callable(verb):
    """A verb in the set but missing from api.py is an AttributeError at the
    user's first call. The lazy ``__getattr__`` means nothing else catches it."""
    resolved = getattr(maxim, verb)
    assert callable(resolved), f"maxim.{verb} resolved to a non-callable: {type(resolved)!r}"


def test_api_verbs_literal_has_no_duplicates():
    """``_API_VERBS`` is a frozenset, so a duplicated entry in the source
    literal is invisible at runtime — which is how one survived until
    2026-08-30. Count the string literals, not the set."""
    source = (REPO_ROOT / "src" / "maxim" / "__init__.py").read_text(encoding="utf-8")
    start = source.index("_API_VERBS = frozenset(")
    end = source.index(")", source.index("}", start))
    literals = re.findall(r'"([a-z_]+)"', source[start:end])
    duplicates = {name for name in literals if literals.count(name) > 1}
    assert not duplicates, f"duplicate entries in the _API_VERBS literal: {sorted(duplicates)}"
    assert set(literals) == set(maxim._API_VERBS), "the _API_VERBS literal and the imported frozenset disagree"
