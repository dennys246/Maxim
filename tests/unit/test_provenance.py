"""Unit tests for the Provenance system (P1-P8).

Tests cover:
- P1: Core types (PipelineStage, ProvenanceVerbosity, Ref, Entry, Trace)
- P2: ProvenanceCollector (two-tier, thread-safe, session-aware)
- P4: Trace rendering (compact/verbose/summary/session report)
- P5: ExplainTool (current/recent/summary/export/concept queries)
- P6: ProvenanceStore (session-aware, crash-safe persistence)
- P7: StructuredContext.provenance_context field
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any
from unittest.mock import Mock

import pytest

from maxim.provenance.types import (
    PipelineStage,
    ProvenanceEntry,
    ProvenanceRef,
    ProvenanceTrace,
    ProvenanceVerbosity,
)
from maxim.provenance.collector import ProvenanceCollector
from maxim.provenance.store import ProvenanceStore
from maxim.provenance.render import (
    render_activities,
    render_session_report,
    render_summary,
    render_trace,
)


# ─────────────────────────────────────────────────────────────────────────────
# P1: Core Types
# ─────────────────────────────────────────────────────────────────────────────


class TestPipelineStage:
    def test_all_nine_stages(self):
        stages = list(PipelineStage)
        assert len(stages) == 9
        names = {s.value for s in stages}
        assert "perception" in names
        assert "consolidation" in names

    def test_verbosity_levels(self):
        assert ProvenanceVerbosity.OFF == 0
        assert ProvenanceVerbosity.COMPACT == 1
        assert ProvenanceVerbosity.VERBOSE == 2
        assert ProvenanceVerbosity.COMPACT > ProvenanceVerbosity.OFF


class TestProvenanceRef:
    def test_str_format(self):
        ref = ProvenanceRef("atl", "abc12345-full-uuid", "kitchen (location)", 0.82)
        s = str(ref)
        assert "`atl:abc12345`" in s
        assert "kitchen (location)" in s
        assert "0.82" in s

    def test_short_format(self):
        ref = ProvenanceRef("hippocampus", "def67890", "episode")
        assert ref.short() == "`hippocampus:def67890` episode"

    def test_to_dict_roundtrip(self):
        ref = ProvenanceRef("atl", "id123", "label", 0.9)
        d = ref.to_dict()
        restored = ProvenanceRef.from_dict(d)
        assert restored == ref

    def test_from_dict_default_confidence(self):
        ref = ProvenanceRef.from_dict({"layer": "atl", "id": "x", "label": "y"})
        assert ref.confidence == 1.0


class TestProvenanceEntry:
    def test_to_dict_roundtrip(self):
        entry = ProvenanceEntry(
            timestamp=1000.0,
            stage=PipelineStage.DECISION,
            component="memory_agent",
            action="Action: navigate (confidence: 0.9)",
            sources=[ProvenanceRef("atl", "c1", "kitchen")],
            confidence=0.9,
            metadata={"reasoning": "test"},
        )
        d = entry.to_dict()
        restored = ProvenanceEntry.from_dict(d)
        assert restored.stage == PipelineStage.DECISION
        assert restored.component == "memory_agent"
        assert len(restored.sources) == 1
        assert restored.metadata == {"reasoning": "test"}


class TestProvenanceTrace:
    def test_add_and_complete(self):
        trace = ProvenanceTrace(
            trace_id="t1", run_id="r1", session_id="s1", started_at=time.time()
        )
        trace.add(PipelineStage.PERCEPTION, "perception_agent", "3 objects")
        trace.add(PipelineStage.DECISION, "memory_agent", "navigate")
        trace.complete()
        assert len(trace.entries) == 2
        assert trace.completed

    def test_to_dict_roundtrip(self):
        trace = ProvenanceTrace(
            trace_id="t1", run_id="r1", session_id="s1", started_at=1000.0
        )
        trace.add(PipelineStage.RECALL, "pattern_completer", "2 predictions")
        trace.complete()
        d = trace.to_dict()
        assert d["type"] == "trace"
        assert d["session_id"] == "s1"
        restored = ProvenanceTrace.from_dict(d)
        assert restored.trace_id == "t1"
        assert len(restored.entries) == 1
        assert restored.completed

    def test_persisted_and_lock_excluded_from_eq(self):
        t1 = ProvenanceTrace(trace_id="t", run_id="r", session_id="s", started_at=1.0)
        t2 = ProvenanceTrace(trace_id="t", run_id="r", session_id="s", started_at=1.0)
        t1._persisted = True
        assert t1 == t2  # _persisted has compare=False

    def test_thread_safe_add(self):
        trace = ProvenanceTrace(
            trace_id="t", run_id="r", session_id="s", started_at=time.time()
        )
        errors: list[Exception] = []

        def add_entries():
            try:
                for i in range(50):
                    trace.add(PipelineStage.RECALL, "test", f"entry-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_entries) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(trace.entries) == 200  # 4 threads x 50


# ─────────────────────────────────────────────────────────────────────────────
# P2: ProvenanceCollector
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectorTier1:
    def test_begin_trace_returns_none_when_off(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.OFF)
        assert c.begin_trace("run-1") is None

    def test_begin_trace_returns_trace_when_compact(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        trace = c.begin_trace("run-1")
        assert trace is not None
        assert trace.run_id == "run-1"
        assert trace.session_id == c.session_id

    def test_record_noop_when_off(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.OFF)
        c.record("run-1", PipelineStage.DECISION, "test", "action")
        # No error, no trace

    def test_record_adds_to_trace(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("run-1")
        c.record("run-1", PipelineStage.DECISION, "test", "navigate")
        trace = c.get_trace("run-1")
        assert len(trace.entries) == 1
        assert trace.entries[0].action == "navigate"

    def test_record_missing_trace_logs_debug(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        # Should not raise, just debug log
        c.record("nonexistent", PipelineStage.DECISION, "test", "action")

    def test_complete_trace(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("run-1")
        c.record("run-1", PipelineStage.OUTCOME, "test", "success")
        trace = c.complete_trace("run-1")
        assert trace.completed

    def test_recent_traces_sorted(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("old")
        c.complete_trace("old")
        time.sleep(0.01)
        c.begin_trace("new")
        c.complete_trace("new")

        recent = c.recent_traces(limit=2)
        assert len(recent) == 2
        assert recent[0].run_id == "new"

    def test_evict_old_traces(self):
        c = ProvenanceCollector(
            verbosity=ProvenanceVerbosity.COMPACT, max_traces=3
        )
        for i in range(5):
            c.begin_trace(f"run-{i}")
        assert len(c._traces) == 3


class TestCollectorTier2:
    def test_log_activity_stores_entries(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.log_activity(
            PipelineStage.FORMATION, "concept_extractor", "Extracted 3 concepts"
        )
        activities = c.recent_activities()
        assert len(activities) == 1
        assert activities[0].component == "concept_extractor"

    def test_log_activity_noop_when_off(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.OFF)
        c.log_activity(PipelineStage.FORMATION, "test", "action")
        assert c.recent_activities() == []

    def test_log_activity_persists_immediately(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        mock_store = Mock()
        c._store = mock_store

        c.log_activity(PipelineStage.LEARNING, "nac", "causal link")

        mock_store.write_activity.assert_called_once()

    def test_activity_eviction(self):
        c = ProvenanceCollector(
            verbosity=ProvenanceVerbosity.COMPACT, max_activities=5
        )
        for i in range(10):
            c.log_activity(PipelineStage.FORMATION, "test", f"action-{i}")
        assert len(c._activities) == 5

    def test_recent_activities_filtered_by_stage(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.log_activity(PipelineStage.FORMATION, "ext", "extracted")
        c.log_activity(PipelineStage.LEARNING, "nac", "learned")
        c.log_activity(PipelineStage.FORMATION, "ext", "extracted again")

        formation = c.recent_activities(stage=PipelineStage.FORMATION)
        assert len(formation) == 2
        learning = c.recent_activities(stage=PipelineStage.LEARNING)
        assert len(learning) == 1


class TestCollectorSessionLifecycle:
    def test_on_session_end_writes_summary(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        mock_store = Mock()
        c._store = mock_store

        c.begin_trace("r1")
        c.complete_trace("r1")
        c.log_activity(PipelineStage.FORMATION, "test", "action")

        stats = c.on_session_end()
        assert stats["total_traces"] == 1
        assert stats["completed_traces"] == 1
        assert stats["total_activities"] == 1
        mock_store.write_session_summary.assert_called_once()
        mock_store.close.assert_called_once()

    def test_on_session_end_skips_persisted_traces(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        mock_store = Mock()
        c._store = mock_store

        c.begin_trace("r1")
        c.complete_trace("r1")  # This persists and sets _persisted=True

        # Reset mock to only count on_session_end writes
        mock_store.write_trace.reset_mock()
        c.on_session_end()

        # Should NOT re-write already persisted trace
        mock_store.write_trace.assert_not_called()

    def test_on_session_end_empty_when_off(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.OFF)
        assert c.on_session_end() == {}

    def test_thread_safe_concurrent_access(self):
        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        errors: list[Exception] = []

        def tier1():
            try:
                for i in range(20):
                    rid = f"t1-{threading.current_thread().name}-{i}"
                    c.begin_trace(rid)
                    c.record(rid, PipelineStage.DECISION, "test", f"action-{i}")
                    c.complete_trace(rid)
            except Exception as e:
                errors.append(e)

        def tier2():
            try:
                for i in range(20):
                    c.log_activity(PipelineStage.FORMATION, "test", f"bg-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=tier1, name=f"tier1-{j}") for j in range(3)
        ] + [
            threading.Thread(target=tier2, name=f"tier2-{j}") for j in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ─────────────────────────────────────────────────────────────────────────────
# P4: Rendering
# ─────────────────────────────────────────────────────────────────────────────


def _make_trace(run_id: str = "r1", completed: bool = True) -> ProvenanceTrace:
    trace = ProvenanceTrace(
        trace_id="t1", run_id=run_id, session_id="sess123", started_at=time.time()
    )
    trace.add(
        PipelineStage.RECALL, "pattern_completer", "2 predictions",
        sources=[ProvenanceRef("hippocampus", "ep1", "observe (success=True)", 0.85)],
    )
    trace.add(
        PipelineStage.DECISION, "memory_agent",
        "Action: navigate (confidence: 0.90)",
        confidence=0.9,
        reasoning="Kitchen had target",
    )
    trace.add(PipelineStage.OUTCOME, "memory_agent", "Success=True, duration=1200ms")
    if completed:
        trace.complete()
    return trace


class TestRenderTrace:
    def test_compact_format(self):
        trace = _make_trace()
        result = render_trace(trace, ProvenanceVerbosity.COMPACT)
        assert "Decision Trace" in result
        assert "**recall**" in result
        assert "**decision**" in result
        assert "**outcome**" in result
        assert "session:" not in result  # No session in compact

    def test_verbose_format_includes_session_and_timestamps(self):
        trace = _make_trace()
        result = render_trace(trace, ProvenanceVerbosity.VERBOSE)
        assert "session:sess123" in result
        assert "ms]" in result  # Timestamp delta
        assert "confidence:" in result  # Full ref format

    def test_verbose_includes_reasoning(self):
        trace = _make_trace()
        result = render_trace(trace, ProvenanceVerbosity.VERBOSE)
        assert "Kitchen had target" in result


class TestRenderActivities:
    def test_empty_returns_empty(self):
        assert render_activities([]) == ""

    def test_renders_entries(self):
        entry = ProvenanceEntry(
            timestamp=time.time(),
            stage=PipelineStage.FORMATION,
            component="concept_extractor",
            action="Extracted 3 concepts",
        )
        result = render_activities([entry])
        assert "concept_extractor" in result
        assert "Extracted 3 concepts" in result


class TestRenderSummary:
    def test_no_traces(self):
        result = render_summary([])
        assert "No provenance traces" in result

    def test_with_traces(self):
        traces = [_make_trace("r1"), _make_trace("r2")]
        result = render_summary(traces)
        assert "2 completed" in result
        assert "navigate" in result

    def test_with_activity_digest(self):
        traces = [_make_trace()]
        activities = [
            ProvenanceEntry(
                timestamp=time.time(), stage=PipelineStage.FORMATION,
                component="concept_extractor", action="extracted",
            ),
            ProvenanceEntry(
                timestamp=time.time(), stage=PipelineStage.LEARNING,
                component="nac", action="learned",
            ),
        ]
        result = render_summary(traces, activities)
        assert "concept_extractor" in result
        assert "nac" in result


class TestRenderSessionReport:
    def test_includes_all_sections(self):
        traces = [_make_trace()]
        activities = [
            ProvenanceEntry(
                timestamp=time.time(), stage=PipelineStage.FORMATION,
                component="extractor", action="extracted",
            ),
        ]
        result = render_session_report(traces, activities, "sess123")
        assert "Session Report: sess123" in result
        assert "Cycle Details" in result
        assert "Background Activity" in result


# ─────────────────────────────────────────────────────────────────────────────
# P5: ExplainTool
# ─────────────────────────────────────────────────────────────────────────────


class TestExplainTool:
    def test_current_returns_in_progress_trace(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("active-run")
        c.record("active-run", PipelineStage.DECISION, "test", "navigating")

        tool = ExplainTool(c)
        result = tool.execute(query="current")
        assert "navigating" in result

    def test_recent_returns_completed_only(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("completed")
        c.record("completed", PipelineStage.OUTCOME, "test", "done")
        c.complete_trace("completed")

        c.begin_trace("in-progress")
        c.record("in-progress", PipelineStage.DECISION, "test", "ongoing")

        tool = ExplainTool(c)
        result = tool.execute(query="recent")
        assert "done" in result
        assert "ongoing" not in result

    def test_current_falls_back_to_recent(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("r1")
        c.record("r1", PipelineStage.OUTCOME, "test", "completed-action")
        c.complete_trace("r1")

        tool = ExplainTool(c)
        result = tool.execute(query="current")
        assert "completed-action" in result

    def test_no_trace_available(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        tool = ExplainTool(c)
        result = tool.execute(query="current")
        assert "No provenance trace" in result

    def test_summary(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("r1")
        c.complete_trace("r1")

        tool = ExplainTool(c)
        result = tool.execute(query="summary")
        assert "Summary" in result

    def test_export_json(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        c.begin_trace("r1")
        c.record("r1", PipelineStage.DECISION, "test", "nav")
        c.complete_trace("r1")

        tool = ExplainTool(c)
        result = tool.execute(query="export_json")
        data = json.loads(result)
        assert "session_id" in data
        assert len(data["traces"]) == 1

    def test_concept_query_no_store(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        tool = ExplainTool(c)
        result = tool.execute(query="concept:kitchen")
        assert "not enabled" in result

    def test_history_no_store(self):
        from maxim.tools.explain import ExplainTool

        c = ProvenanceCollector(verbosity=ProvenanceVerbosity.COMPACT)
        tool = ExplainTool(c)
        result = tool.execute(query="history")
        assert "not enabled" in result


# ─────────────────────────────────────────────────────────────────────────────
# P6: ProvenanceStore
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def store_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


class TestProvenanceStore:
    def test_write_trace_creates_session_file(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        trace = _make_trace()
        store.write_trace(trace)
        store.close()

        path = os.path.join(store_dir, f"{trace.session_id}.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.loads(f.readline())
        assert data["type"] == "trace"
        assert data["run_id"] == "r1"

    def test_write_activity(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        entry = ProvenanceEntry(
            timestamp=time.time(),
            stage=PipelineStage.FORMATION,
            component="extractor",
            action="extracted 3",
        )
        store.write_activity(entry, "sess1")
        store.close()

        records = store.load_session("sess1")
        assert len(records) == 1
        assert records[0]["type"] == "activity"

    def test_write_session_summary_updates_manifest(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        store.write_session_summary("sess1", {"completed_traces": 5})
        store.close()

        manifest = store._load_manifest()
        assert "sess1" in manifest
        assert manifest["sess1"]["completed_traces"] == 5

    def test_close_idempotent(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        store.close()
        store.close()  # Should not raise

    def test_load_session(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        trace = _make_trace()
        store.write_trace(trace)
        store.close()

        records = store.load_session(trace.session_id)
        assert len(records) == 1

    def test_load_recent_sessions(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        store.write_session_summary("old", {"completed_traces": 1})
        time.sleep(0.01)
        store.write_session_summary("new", {"completed_traces": 2})
        store.close()

        sessions = store.load_recent_sessions(limit=2)
        assert sessions[0] == "new"

    def test_query_concept(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        trace = _make_trace()
        store.write_trace(trace)
        store.write_session_summary(trace.session_id, {"completed_traces": 1})
        store.close()

        results = store.query_concept("observe")
        assert len(results) >= 1

    def test_load_nonexistent_session(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        assert store.load_session("nonexistent") == []

    def test_manifest_atomic_write(self, store_dir):
        store = ProvenanceStore(base_dir=store_dir)
        store.write_session_summary("s1", {"completed_traces": 1})
        store.write_session_summary("s2", {"completed_traces": 2})
        store.close()

        manifest = store._load_manifest()
        assert len(manifest) == 2
        # No .tmp file left behind
        assert not os.path.exists(os.path.join(store_dir, "sessions.tmp"))


# ─────────────────────────────────────────────────────────────────────────────
# P7: StructuredContext integration
# ─────────────────────────────────────────────────────────────────────────────


class TestStructuredContextProvenance:
    def test_provenance_context_field_exists(self):
        from maxim.agents.bus import StructuredContext
        ctx = StructuredContext(timestamp=time.time())
        assert hasattr(ctx, "provenance_context")
        assert ctx.provenance_context == ""
