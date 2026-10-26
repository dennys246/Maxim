"""#1153: ``AngularGyrus.recall`` built the index key from what it was handed, so ``category=MathCategory.PATTERN``
looked up ``category:MathCategory.PATTERN`` while the store indexes ``category:PATTERN``
(``_index_record`` writes ``record.category.name``). Every enum-keyed recall came back empty: MemoryAgent's
knowledge fallback has always returned nothing (a Dormant path, #845, so no live effect). The #845/#995 class: a
reader using a key the store doesn't have, silently empty. Fixed at the store, not the caller. Landed as a strict
xfail (``182d6f50``), flipped in place.
"""

from __future__ import annotations

import time


def _gyrus():
    from maxim.math.angular_gyrus import AngularGyrus
    from maxim.math.math_types import MathMemory
    from maxim.math.types import MathCategory

    ag = AngularGyrus()
    ag.store(MathMemory(id="p1", timestamp=time.time(), name="trend", category=MathCategory.PATTERN, confidence=0.9))
    ag.store(MathMemory(id="m1", timestamp=time.time(), name="mean", category=MathCategory.METHOD, confidence=0.9))
    return ag


def test_an_enum_category_finds_what_its_name_finds():
    from maxim.math.types import MathCategory

    ag = _gyrus()
    assert [r.id for r in ag.recall(category="PATTERN")] == ["p1"]
    assert [r.id for r in ag.recall(category=MathCategory.PATTERN)] == ["p1"]


def test_the_dormant_knowledge_fallback_now_reads_patterns():
    """The one enum caller, ``MemoryAgent``'s AG knowledge fallback (Dormant, #845): its call shape now finds the
    pattern. Through the store's own ``recall``, not a copy of the call."""
    from maxim.math.types import MathCategory

    ag = _gyrus()
    assert [r.id for r in ag.recall(limit=3, category=MathCategory.PATTERN, min_confidence=0.5, touch=False)] == ["p1"]
