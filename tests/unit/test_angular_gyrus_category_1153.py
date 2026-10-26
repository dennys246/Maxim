"""#1153: ``AngularGyrus.recall`` built the index key from what it was handed, so ``category=MathCategory.PATTERN``
looked up ``category:MathCategory.PATTERN`` while the store indexes ``category:PATTERN``
(``_index_record`` writes ``record.category.name``). Every enum-keyed recall came back empty: MemoryAgent's
knowledge fallback has always returned nothing from the AG (a Dormant path, #845, so no live effect). The #845/#995 class: a
reader using a key the store doesn't have, silently empty. Fixed at the store, not the caller. Landed as a strict
xfail (``182d6f50``), flipped in place.
"""

from __future__ import annotations

import time

import pytest


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


def test_a_category_name_in_any_case_finds_its_records_and_an_unknown_one_fails_loud():
    """Review fold (both lenses): the live math tool hands the model's string straight in, and its own store path
    upper-cases it, so "pattern" stored but never recalled. The store keys every form by the name; a name that is no
    category raises (owner decision 2026-10-25), and the tool returns that to the model as a tool error."""
    from maxim.math.angular_gyrus import UnknownMathCategory

    ag = _gyrus()
    for spelled in ("pattern", "Pattern", " PATTERN "):
        assert [r.id for r in ag.recall(category=spelled)] == ["p1"], spelled
    with pytest.raises(UnknownMathCategory, match="'patterns'.*PATTERN"):
        ag.recall(category="patterns")


def test_a_blank_category_is_no_filter():
    """Owner decision 2026-10-25: a model filling an optional parameter with "" means no filter."""
    ag = _gyrus()
    everything = sorted(r.id for r in ag.recall(limit=1000))
    assert {"m1", "p1"} <= set(everything)
    for blank in ("", "  "):
        assert sorted(r.id for r in ag.recall(limit=1000, category=blank)) == everything


def test_storing_and_recalling_follow_one_category_contract():
    """Owner decision 2026-10-25: store and recall are one contract. The tool used to file an unknown category under
    PATTERN (so "store trend" succeeded while "recall trend" raised); now both raise, both take any case, and an
    omitted or blank category stores under the PATTERN default."""
    from maxim.math.types import MathCategory
    from maxim.tools.math_tool import MathTool

    tool = MathTool.__new__(MathTool)
    tool._ag = _gyrus()
    methods_before = len(tool._ag.recall(limit=1000, category=MathCategory.METHOD))
    for spelled in ("method", "Method", "METHOD"):
        stored = tool.run(operation="store_memory", name=f"median_{spelled}", category=spelled)
        assert stored.success and stored.output["category"] == "METHOD", spelled
    for blank in (None, ""):
        stored = tool.run(operation="store_memory", name=f"guess_{blank!r}", category=blank)
        assert stored.success and stored.output["category"] == MathCategory.PATTERN.name
    refused = tool.run(operation="store_memory", name="x", category="trend")
    assert refused.success is False and "unknown math category 'trend'" in (refused.error or "")
    recalled = tool.run(operation="recall_memory", category="trend")
    assert recalled.success is False and "unknown math category 'trend'" in (recalled.error or "")
    stored_methods = tool.run(operation="recall_memory", category="method", limit=1000).output["count"]
    assert stored_methods == methods_before + 3


def test_the_math_tool_reports_an_unknown_category_to_the_model():
    from maxim.tools.math_tool import MathTool

    tool = MathTool.__new__(MathTool)
    tool._ag = _gyrus()
    out = tool.run(operation="recall_memory", category="patterns")
    assert out.success is False and "unknown math category" in (out.error or "")
    assert tool.run(operation="recall_memory", category="pattern").output["count"] == 1


def test_the_dormant_knowledge_fallback_reads_patterns_past_a_compressed_one():
    """The one enum caller, through the REAL consumer: ``MemoryAgent._build_knowledge_context`` with no episodic seeds
    falls back to the AG query. A compressed PATTERN record (no ``verbal``) ranked first used to raise inside the
    grandfathered ``except`` and drop the whole AG block, the full record with it (review fold, wire lens)."""
    from types import SimpleNamespace

    from maxim.agents.memory_agent import MemoryAgent
    from maxim.math.math_types import CompressedMathMemory, MathMemory
    from maxim.math.types import MathCategory

    ag = _gyrus()
    old = MathMemory(id="p0", timestamp=time.time(), name="old_trend", category=MathCategory.PATTERN, confidence=0.95)
    ag.store(old)
    ag._records["p0"] = CompressedMathMemory.from_math_record(old)

    hub = SimpleNamespace(atl=None, angular_gyrus=ag)
    agent = SimpleNamespace(
        _memory_hub=hub, _forming_pool={}, _get_concept_relationships=MemoryAgent._get_concept_relationships
    )
    entries = MemoryAgent._build_knowledge_context(agent)
    assert [e["concept_name"] for e in entries] == ["trend"]
