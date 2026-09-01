"""Characterization tests pinning bugs ledger D51 — the LSH index is degenerate.

**These tests assert BROKEN behaviour on purpose.** They exist so the defect is
executable rather than only described, and so that whoever fixes D51 gets a loud,
specific failure telling them to update this file — instead of the fix landing
with nothing to notice it.

D51, measured 2026-08-30 and surfaced by the pymaxim.bio audit:

1. ``LSHIndex.add``/``query`` compute the bucket from ``semantic_hash`` alone and
   IDENTICALLY for every table — the loop variable ``i`` is bound and never used.
2. ``_hashers`` is built with distinct per-table seeds and never read, so the
   four tables are byte-identical and the multi-table probability amplification
   that is the point of multi-table LSH does not exist.
3. ``semantic_hash`` stays ``(0,)*8`` in production, so every item lands in ONE
   bucket and the query degrades to a linear scan.

Consequence: 0.3 / 3.0 / 12.6 ms at N = 100 / 1,000 / 4,000, one bucket holding
100% at every size. Called live from ``nac.py::distribute_reward``.

D51 is filed and deliberately NOT fixed — the repair needs a design decision
(what should each table hash? should this index use the structural/temporal/
context dimensions ``SituationSignature``'s docstring already promises? is the
surface load-bearing at all, given the substrate results rest on the exact scan
in ``pattern_complete_or_separate``?), not a patch.

WHEN YOU FIX D51: these tests SHOULD fail. Rewrite them to assert the new,
correct behaviour and delete the ledger row per the bugs-README expiry rule.
"""

from __future__ import annotations

from maxim.similarity.lsh import LSHIndex
from maxim.similarity.signature import SituationSignature

FIX_HINT = "If you fixed D51, this failure is expected — update this file and retire the ledger row."


def _sig(n: int) -> SituationSignature:
    """A signature whose NON-semantic dimensions all vary."""
    return SituationSignature(
        semantic_hash=(0,) * 8,  # what production actually produces
        structural_hash=n * 7919,
        temporal_hash=(n % 24, n % 7, n % 4, n % 12),
        context_hash=n * 104729,
        tool_name=f"tool_{n}",
        outcome_type="success" if n % 2 else "failure",
        mode="explore",
        goal_keywords=(f"kw{n}",),
    )


def _populated(count: int = 200) -> LSHIndex:
    index = LSHIndex()
    for n in range(count):
        index.add(f"m{n}", _sig(n))
    return index


def test_all_hash_tables_are_byte_identical() -> None:
    """D51(1)+(2): every table gets the same bucket key, so the 4x structure is
    four copies of one table — 4x memory and 4x probe cost for no amplification."""
    tables = [dict(t) for t in _populated()._tables]
    assert len(tables) == 4
    assert all(t == tables[0] for t in tables), FIX_HINT


def test_everything_lands_in_one_bucket() -> None:
    """D51(3): with `semantic_hash` degenerate, the index is a linear scan."""
    table = dict(_populated(count=200)._tables[0])
    assert len(table) == 1, FIX_HINT
    assert len(next(iter(table.values()))) == 200, FIX_HINT


def test_non_semantic_dimensions_do_not_affect_bucketing() -> None:
    """`SituationSignature`'s docstring says that when semantic hashes collide,
    "similarity queries rely only on structural, temporal, and context".
    `LSHIndex` never consults those — this pins that the docstring is wrong."""
    index = LSHIndex()
    index.add("a", _sig(1))
    index.add("b", _sig(999999))  # every non-semantic field differs
    table = dict(index._tables[0])
    assert len(table) == 1, FIX_HINT
    assert sorted(next(iter(table.values()))) == ["a", "b"], FIX_HINT


def test_per_table_hashers_are_built_and_never_used() -> None:
    """D51(2): distinct seeds are constructed and discarded. If a fix starts
    consuming them, the tables above stop being identical and this file fails."""
    index = LSHIndex()
    assert len(index._hashers) == index.num_tables
    assert len({h.seed for h in index._hashers}) == index.num_tables, "seeds are distinct..."
    src = __import__("inspect").getsource(LSHIndex)
    reads = [ln for ln in src.splitlines() if "_hashers" in ln and "self._hashers =" not in ln]
    # only the `if not self._hashers:` guard and the field declaration remain
    assert all("if not" in ln or ": list[SemanticLSH]" in ln for ln in reads), (
        f"_hashers is now READ somewhere — {FIX_HINT}"
    )
