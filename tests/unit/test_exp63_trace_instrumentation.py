"""Exp 63's enrichment-trace instrumentation (docs/experiments/exp63_carried_recall_prereg.md).

The trace names whose recall it was (`agent_id`, logging only), which memories surfaced (`memory_ids`), by which
path (`memory_paths`), and what the store held at the query's entry (`goal_path_horizon`, `goal_path_holes`). The
verdict recomputes recall from these, so each is pinned to what the pipeline actually did, on every path.
"""

from __future__ import annotations

import ast
import inspect
import logging
from types import SimpleNamespace

from maxim.integration.bio_enrichment import BioEnrichmentPipeline, EnrichmentContext
from maxim.memory.encoding import EncodingSignals
from maxim.memory.hippocampus import Hippocampus, HippocampusConfig
from maxim.memory.types import EpisodicMemory, Outcome, Perception

ENC = EncodingSignals.unmeasured("api")
GOAL = "escape a dungeon with a sleeping guard"


def _store(n: int) -> tuple[Hippocampus, list[str]]:
    h = Hippocampus(HippocampusConfig(auto_save_after_sleep=False))
    ids = [
        h.capture(
            perception=Perception(salience=0.5, observations={"text": f"memory {i}"}),
            outcome=Outcome(success=True),
            encoding=ENC,
        )
        for i in range(n)
    ]
    return h, ids


def _trace(caplog, pipeline: BioEnrichmentPipeline, goal: str | None = GOAL) -> dict:
    with caplog.at_level(logging.INFO, logger="maxim.integration.bio_enrichment"):
        ctx = EnrichmentContext(active_goal=goal) if goal else None
        pipeline.enrich("the guard stirs", context=ctx, bypass_gate=True)
    records = [r for r in caplog.records if getattr(r, "event", None) == "enrichment_trace"]
    assert len(records) == 1
    return records[0].data


def test_the_goal_path_traces_its_ids_in_order_and_the_store_at_entry(caplog) -> None:
    h, ids = _store(5)
    data = _trace(caplog, BioEnrichmentPipeline(hippocampus=h, trace_agent_id="aut"))
    expected = [m.id for m in h.recall(query=GOAL, limit=5)][:3]  # the goal path's own call: a known answer
    assert data["memory_ids"] == expected and data["memories"] == 3 and set(expected) <= set(ids)
    assert data["memory_paths"] == ["goal", "goal", "goal"]  # no encoder/EC: the graph path cannot run
    assert (data["goal_path_horizon"], data["goal_path_holes"]) == (4, [])
    assert data["agent_id"] == "aut"


def _mem(i: str) -> EpisodicMemory:
    return EpisodicMemory(id=i, timestamp=1.0, outcome=Outcome(success=True))


class _StubStore:
    """Each path returns fixed memories, so the labels, the de-duplication and the cap are known answers."""

    _memories: dict = {}

    def __init__(self, graph=(), goal=(), substring=(), view=(7, [])):
        self._graph, self._goal, self._substring, self._view = list(graph), list(goal), list(substring), view

    def stored_capture_seqs(self):
        return self._view

    def retrieve_on_cue(self, node_id, limit, multi_hop):
        return [("n1", 0.9)] if self._graph else []

    def recall(self, *, query=None, object_detected=None, limit=5):
        return self._graph if object_detected else self._goal

    def search_by_content(self, text, limit=5):
        return self._substring


def _graph_pipeline(store: _StubStore) -> BioEnrichmentPipeline:
    encoder = SimpleNamespace(embed=lambda text: [1.0], geometry_for=lambda emb, modality: None)
    ec = SimpleNamespace(
        pattern_complete_readonly=lambda emb, modality, geometry: SimpleNamespace(
            is_new=False, similarity=0.9, node_id="n1"
        )
    )
    atl = SimpleNamespace(get=lambda node_id: SimpleNamespace(name="guard"))
    return BioEnrichmentPipeline(hippocampus=store, encoder=encoder, ec=ec, atl=atl)


def test_every_path_is_labelled_deduplicated_and_capped(caplog) -> None:
    g, d, s1, s2 = _mem("g"), _mem("d"), _mem("s1"), _mem("s2")
    store = _StubStore(graph=[g], goal=[d, g], substring=[s1, d, s2], view=(7, [[5, 5]]))
    data = _trace(caplog, _graph_pipeline(store))
    assert data["memory_ids"] == ["g", "d", "s1"]
    assert data["memory_paths"] == ["graph", "goal", "substring"]
    assert (data["goal_path_horizon"], data["goal_path_holes"]) == (7, [[5, 5]])


def test_without_a_goal_the_substring_path_is_labelled(caplog) -> None:
    store = _StubStore(substring=[_mem("a"), _mem("b")], view=(1, []))
    data = _trace(caplog, BioEnrichmentPipeline(hippocampus=store), goal=None)
    assert data["memory_ids"] == ["a", "b"] and data["memory_paths"] == ["substring", "substring"]


def test_a_memory_that_fails_to_summarize_leaves_no_orphan_label(caplog) -> None:
    """A path is recorded with its summary, so the two lists stay parallel even when the graph path's own try
    swallows a failed summary."""
    bad = SimpleNamespace(id="bad", valence="not a number")  # float(valence) raises inside _add_memory
    store = _StubStore(graph=[bad], goal=[_mem("x"), _mem("y"), _mem("z")])
    data = _trace(caplog, _graph_pipeline(store))
    assert data["memory_ids"] == ["x", "y", "z"] and data["memory_paths"] == ["goal", "goal", "goal"]


def test_the_trace_label_changes_no_behaviour() -> None:
    """``agent_id`` switches on per-agent reads (aversions, reward bias); the trace label must not."""
    pipeline = BioEnrichmentPipeline(trace_agent_id="aut")
    assert pipeline._agent_id == "" and pipeline._trace_agent_id == "aut"


def test_the_horizon_is_what_is_stored_and_the_holes_what_is_not() -> None:
    h, _ = _store(3)
    assert h.stored_capture_seqs() == (2, [])
    h.next_capture_seq()  # reserved, not stored: recall cannot see it, and it is above the horizon
    assert h.stored_capture_seqs() == (2, [])
    late = h.next_capture_seq()  # 4: reserved; 3 stays in flight
    h.capture(
        perception=Perception(salience=0.5, observations={"text": "late"}),
        outcome=Outcome(success=True),
        encoding=ENC,
        capture_seq=late,
    )
    assert h.stored_capture_seqs() == (4, [[3, 3]])  # 3 is at or below the horizon but not stored: a hole
    assert Hippocampus(HippocampusConfig(auto_save_after_sleep=False)).stored_capture_seqs() == (-1, [])


def test_holes_start_at_the_load_watermark_and_come_as_ranges() -> None:
    """A long-lived store's evicted and dropped numbers never land again: only this process's own numbers can be in
    flight, so holes start at the load watermark, and runs of them are one range each."""
    h, _ = _store(3)
    for _ in range(4):
        h.next_capture_seq()  # 3..6 reserved, none stored
    h.capture(
        perception=Perception(salience=0.5, observations={"text": "late"}),
        outcome=Outcome(success=True),
        encoding=ENC,
        capture_seq=h.next_capture_seq(),  # 7
    )
    assert h.stored_capture_seqs() == (7, [[3, 6]])
    h._resume_capture_seq()  # as after a load: the watermark moves past every saved trace
    assert h.stored_capture_seqs() == (7, [])


def test_an_empty_store_traces_no_ids(caplog) -> None:
    data = _trace(
        caplog, BioEnrichmentPipeline(hippocampus=Hippocampus(HippocampusConfig(auto_save_after_sleep=False)))
    )
    assert data["memory_ids"] == [] and data["memory_paths"] == []
    assert (data["goal_path_horizon"], data["goal_path_holes"]) == (-1, [])


def test_the_bio_stack_labels_its_pipeline_with_its_agent() -> None:
    from maxim.runtime import bio_stack

    tree = ast.parse(inspect.getsource(bio_stack.build_bio_stack).lstrip())
    calls = [
        n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "BioEnrichmentPipeline"
    ]
    assert len(calls) == 1
    kw = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert kw.get("trace_agent_id") == "agent_id" and "agent_id" not in kw


def test_a_failed_horizon_read_loses_only_the_horizon_never_recall(caplog) -> None:
    class _NoView(_StubStore):
        def stored_capture_seqs(self):
            raise RuntimeError("no view")

    store = _NoView(goal=[_mem("x"), _mem("y")])
    data = _trace(caplog, BioEnrichmentPipeline(hippocampus=store))
    assert data["memory_ids"] == ["x", "y"] and data["memory_paths"] == ["goal", "goal"]
    assert data["goal_path_horizon"] is None and data["goal_path_holes"] == []
