"""#991 -- a memory's success is read the same way for every record kind.

An ``EpisodicMemory`` keeps its success on ``.outcome``; a ``CompressedMemory`` carries it directly. The
enrichment pipeline probed ``getattr(mem, "success", False)``, which an episode never has, so every
episode surfaced to the LLM was tagged as a failure (``[-]``, valence -0.3), whatever its outcome. The
four bridges hand-rolled the same probe correctly; all five now read ``memory.types.record_success``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from maxim.memory.encoding import EncodingSignals
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.types import CompressedMemory, EpisodicMemory, Outcome, Perception, record_success

ENC = EncodingSignals.unmeasured("api")


@pytest.mark.parametrize("success", [True, False, None])
def test_record_success_reads_both_record_kinds(success) -> None:
    assert record_success(EpisodicMemory(id="e", timestamp=1.0, outcome=Outcome(success=success))) is success
    assert record_success(CompressedMemory(id="c", timestamp=1.0, success=success)) is success


def test_a_record_with_no_outcome_is_unknown() -> None:
    assert record_success(SimpleNamespace(id="x")) is None


def _pipeline_over(outcomes: dict[str, bool | None]):
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    ids = {}
    for word, success in outcomes.items():
        ids[word] = h.capture(
            perception=Perception(salience=0.5, observations={"text": f"the kettle {word}"}),
            outcome=Outcome(success=success),
            encoding=ENC,
        )
    return BioEnrichmentPipeline(hippocampus=h), ids


def test_an_enrichment_memory_takes_its_valence_from_its_own_outcome() -> None:
    """Every episode used to read -0.3. Success leans approach, failure avoid, unknown neither."""
    pipeline, ids = _pipeline_over({"whistled": True, "burned": False, "sat": None})
    summaries = {s.memory_id: s.valence for s in pipeline._query_hippocampus("kettle", ["kettle"])}
    assert summaries == {ids["whistled"]: 0.3, ids["burned"]: -0.3, ids["sat"]: 0.0}


def test_the_llm_sees_each_memory_marked_by_its_own_outcome() -> None:
    from maxim.integration.bio_enrichment import EnrichmentResult

    pipeline, _ = _pipeline_over({"whistled": True, "burned": False, "sat": None})
    memories = tuple(pipeline._query_hippocampus("kettle", ["kettle"]))
    text = pipeline.format_thought_response(EnrichmentResult(memories=memories))
    assert sorted(line.strip()[:3] for line in text.splitlines()[1:]) == ["[+]", "[-]", "[~]"]


@pytest.mark.parametrize("success", [True, False, None])
def test_an_episode_answers_success_itself(success) -> None:
    """The invariant on the type: a stray ``getattr(mem, "success", False)`` is now right for both kinds."""
    ep = EpisodicMemory(id="e", timestamp=1.0, outcome=Outcome(success=success))
    assert ep.success is success
    assert getattr(ep, "success", False) is success


def test_the_planner_shows_a_compressed_success_and_only_strategies_it_can_show() -> None:
    """It skipped any record without an ``outcome``, so a compressed success never counted. And every
    admitted strategy is credited as used, so each must render: one with no tool to show is not admitted."""
    from maxim.memory.types import Action
    from maxim.planning.adaptive_planner import AdaptivePlanner

    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    won = CompressedMemory(id="c-won", timestamp=1.0, success=True, tool_name="pour")
    toolless = CompressedMemory(id="c-toolless", timestamp=1.0, success=True)
    lost = CompressedMemory(id="c-lost", timestamp=1.0, success=False, tool_name="spill")
    ep = EpisodicMemory(id="e-won", timestamp=1.0, action=Action(tool_name="grasp"), outcome=Outcome(success=True))
    seen = EpisodicMemory(id="e-seen", timestamp=1.0, outcome=Outcome(success=None))
    h.recall = lambda **kwargs: [seen]  # a seed, so the planner reaches the associated recall
    h.recall_associated = lambda seed_ids, limit: [(m, 1.0) for m in (toolless, won, lost, ep, seen)]
    pctx = AdaptivePlanner(hippocampus=h)._gather_context({"description": "get water"}, "grasp", {}, None)
    assert [m.id for m in pctx.successful_strategies] == ["c-won", "e-won"]
    rendered = [line.strip() for line in pctx.to_llm_section().splitlines() if line.strip().startswith("- Used")]
    assert rendered == ["- Used pour → success", "- Used grasp → success"]


@pytest.mark.parametrize("success, label", [(True, "success"), (False, "failure"), (None, "unknown")])
def test_a_compressed_records_outcome_reaches_its_readers(success, label) -> None:
    from maxim.agents.memory_agent import MemoryAgent
    from maxim.similarity.signature import SituationSignature
    from maxim.tools.introspection import _format_episodic_memory

    compressed = CompressedMemory(id="c", timestamp=1.0, success=success)
    assert SituationSignature.from_memory(compressed).outcome_type == label  # was ""
    assert MemoryAgent._memory_to_context_item(compressed, 0.5)["content"]["success"] is success
    assert _format_episodic_memory(compressed).get("success") is success


def test_no_reader_probes_a_memory_records_success_by_attribute() -> None:
    """The hand-rolled probes are gone; a new one must use ``record_success`` (or the record's own
    ``.success``). Scoped to the names a memory record or its outcome goes by in ``src/``: a tool RESULT's
    ``.success`` is a different, always-bool field, and a single-line regex sees only these shapes."""
    import re
    from pathlib import Path

    import maxim

    probe = re.compile(
        r"""(getattr|hasattr)\((mem|memory|record|episode|ep|m|outcome|out), ["']success["']"""
        r"""|getattr\(getattr\(\w+, ["']outcome["'][^)]*\), ["']success["']"""
    )
    root = Path(maxim.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{n}"
        for path in root.rglob("*.py")
        if path.name != "types.py" or path.parent.name != "memory"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if probe.search(line)
    ]
    assert offenders == []
