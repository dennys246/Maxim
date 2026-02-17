"""Unit tests for Hippocampus - Associative memory graph.

Tests the core functionality of memory capture, indexing, recall,
and sleep consolidation.
"""

from __future__ import annotations

import time

import pytest


class TestHippocampusCapture:
    """Test memory capture."""

    def test_capture_assigns_unique_id(self, hippocampus, complete_memory_args):
        """Each capture gets a unique ID."""
        id1 = hippocampus.capture(**complete_memory_args)
        id2 = hippocampus.capture(**complete_memory_args)

        assert id1 != id2

    def test_capture_increments_count(self, hippocampus, complete_memory_args):
        """Capture increases memory count."""
        assert len(hippocampus) == 0

        hippocampus.capture(**complete_memory_args)
        assert len(hippocampus) == 1

        hippocampus.capture(**complete_memory_args)
        assert len(hippocampus) == 2

    def test_capture_builds_goal_index(self, hippocampus, complete_memory_args):
        """Capture indexes by goal."""
        hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(goal="find mug")
        assert len(memories) == 1

    def test_capture_builds_tool_index(self, hippocampus, complete_memory_args):
        """Capture indexes by tool name."""
        hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(tool="look_around")
        assert len(memories) == 1

    def test_capture_builds_success_index(self, hippocampus, complete_memory_args):
        """Capture indexes by success status."""
        hippocampus.capture(**complete_memory_args)

        successful = hippocampus.recall(success=True)
        failed = hippocampus.recall(success=False)

        assert len(successful) == 1
        assert len(failed) == 0

    def test_capture_builds_object_index(self, hippocampus, complete_memory_args):
        """Capture indexes by detected objects."""
        hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(object_detected="mug")
        assert len(memories) == 1

    def test_capture_builds_person_index(self, hippocampus, complete_memory_args):
        """Capture indexes by detected people."""
        hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(person_detected="person_1")
        assert len(memories) == 1


class TestHippocampusImmediatePromotion:
    """Test immediate promotion to long-term memory."""

    def test_high_salience_promotes_immediately(self, hippocampus, high_salience_perception):
        """High-salience memories are immediately promoted to long-term."""
        from maxim.memory.types import Action, Context, Decision, Outcome

        memory_id = hippocampus.capture(
            perception=high_salience_perception,
            context=Context(active_goal="emergency", active_mode="active"),
            decision=Decision(intent={"goal": "respond"}, reasoning="urgent", confidence=1.0),
            action=Action(tool_name="alert", tool_params={}),
            outcome=Outcome(success=True, result={}),
        )

        memory = hippocampus.get(memory_id)
        assert memory.long_term is True

    def test_normal_salience_not_promoted_immediately(self, hippocampus, complete_memory_args):
        """Normal salience memories are not immediately promoted."""
        memory_id = hippocampus.capture(**complete_memory_args)

        memory = hippocampus.get(memory_id)
        assert memory.long_term is False


class TestHippocampusRecall:
    """Test memory recall and queries."""

    def test_recall_with_no_filters_returns_all(self, hippocampus, complete_memory_args):
        """Recall with no filters returns all memories."""
        hippocampus.capture(**complete_memory_args)
        hippocampus.capture(**complete_memory_args)
        hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(limit=100)
        assert len(memories) == 3

    def test_recall_respects_limit(self, hippocampus, complete_memory_args):
        """Recall respects the limit parameter."""
        for _ in range(10):
            hippocampus.capture(**complete_memory_args)

        memories = hippocampus.recall(limit=5)
        assert len(memories) == 5

    def test_recall_filters_combine_with_and(self, hippocampus):
        """Multiple filters are AND-ed together."""
        from maxim.memory.types import Action, Context, Decision, Outcome, Perception

        # Memory A: goal=find, success=True
        hippocampus.capture(
            perception=Perception(observations={}, detected_objects=[], detected_people=[], salience=0.5, novelty=0.5),
            context=Context(active_goal="find", active_mode="explore"),
            decision=Decision(intent={}, reasoning="", confidence=0.5),
            action=Action(tool_name="look", tool_params={}),
            outcome=Outcome(success=True, result={}),
        )

        # Memory B: goal=find, success=False
        hippocampus.capture(
            perception=Perception(observations={}, detected_objects=[], detected_people=[], salience=0.5, novelty=0.5),
            context=Context(active_goal="find", active_mode="explore"),
            decision=Decision(intent={}, reasoning="", confidence=0.5),
            action=Action(tool_name="look", tool_params={}),
            outcome=Outcome(success=False, result={}),
        )

        # Memory C: goal=grasp, success=True
        hippocampus.capture(
            perception=Perception(observations={}, detected_objects=[], detected_people=[], salience=0.5, novelty=0.5),
            context=Context(active_goal="grasp", active_mode="explore"),
            decision=Decision(intent={}, reasoning="", confidence=0.5),
            action=Action(tool_name="grasp", tool_params={}),
            outcome=Outcome(success=True, result={}),
        )

        # Query: goal=find AND success=True -> only Memory A
        memories = hippocampus.recall(goal="find", success=True)
        assert len(memories) == 1

    def test_recall_returns_most_recent_first(self, hippocampus):
        """Recall returns memories in reverse chronological order."""
        from maxim.memory.types import Action, Context, Decision, Outcome, Perception

        ids = []
        for i in range(3):
            memory_id = hippocampus.capture(
                perception=Perception(observations={"i": i}, detected_objects=[], detected_people=[], salience=0.5, novelty=0.5),
                context=Context(active_goal=f"goal_{i}", active_mode="explore"),
                decision=Decision(intent={}, reasoning="", confidence=0.5),
                action=Action(tool_name="test", tool_params={}),
                outcome=Outcome(success=True, result={}),
            )
            ids.append(memory_id)
            time.sleep(0.01)  # Ensure distinct timestamps

        memories = hippocampus.recall()

        # Most recent should be first
        assert memories[0].id == ids[-1]

    def test_recall_time_filters(self, hippocampus, complete_memory_args):
        """Recall respects time range filters."""
        before = time.time()
        time.sleep(0.01)

        hippocampus.capture(**complete_memory_args)

        time.sleep(0.01)
        after = time.time()

        # Memory is within range
        memories = hippocampus.recall(time_after=before, time_before=after)
        assert len(memories) == 1

        # Memory is before range
        memories = hippocampus.recall(time_after=after + 1)
        assert len(memories) == 0


class TestHippocampusGet:
    """Test individual memory retrieval."""

    def test_get_returns_memory(self, hippocampus, complete_memory_args):
        """get() returns the memory by ID."""
        memory_id = hippocampus.capture(**complete_memory_args)

        memory = hippocampus.get(memory_id)

        assert memory is not None
        assert memory.id == memory_id

    def test_get_updates_access_tracking(self, hippocampus, complete_memory_args):
        """get() updates access timestamp and count."""
        memory_id = hippocampus.capture(**complete_memory_args)

        memory = hippocampus.get(memory_id)
        first_access_count = memory.access_count

        time.sleep(0.01)
        memory = hippocampus.get(memory_id)

        assert memory.access_count > first_access_count

    def test_get_returns_none_for_nonexistent(self, hippocampus):
        """get() returns None for non-existent ID."""
        memory = hippocampus.get("nonexistent_id")
        assert memory is None


class TestHippocampusSimilarRecall:
    """Test similarity-based recall."""

    def test_recall_similar_finds_matching_objects(self, hippocampus):
        """recall_similar finds memories with matching objects."""
        from maxim.memory.types import Action, Context, Decision, Outcome, Perception

        # Memory with mug
        hippocampus.capture(
            perception=Perception(observations={}, detected_objects=["mug"], detected_people=[], salience=0.5, novelty=0.5),
            context=Context(active_goal="find", active_mode="explore"),
            decision=Decision(intent={}, reasoning="", confidence=0.5),
            action=Action(tool_name="look", tool_params={}),
            outcome=Outcome(success=True, result={}),
        )

        # Memory with plate
        hippocampus.capture(
            perception=Perception(observations={}, detected_objects=["plate"], detected_people=[], salience=0.5, novelty=0.5),
            context=Context(active_goal="find", active_mode="explore"),
            decision=Decision(intent={}, reasoning="", confidence=0.5),
            action=Action(tool_name="look", tool_params={}),
            outcome=Outcome(success=True, result={}),
        )

        # Query with mug perception
        query_perception = Perception(
            observations={}, detected_objects=["mug"], detected_people=[], salience=0.5, novelty=0.5
        )

        similar = hippocampus.recall_similar(query_perception)

        # Should find the mug memory
        assert len(similar) >= 1
        assert any("mug" in m.perception.detected_objects for m in similar)


class TestHippocampusConsistency:
    """Test index consistency."""

    def test_check_consistency_on_clean_state(self, hippocampus, complete_memory_args):
        """Clean state has no consistency issues."""
        hippocampus.capture(**complete_memory_args)

        issues = hippocampus.check_consistency()
        assert issues == []

    def test_repair_consistency_fixes_issues(self, hippocampus, complete_memory_args):
        """Repairs detected consistency issues."""
        memory_id = hippocampus.capture(**complete_memory_args)

        # Manually corrupt: delete memory but leave index
        del hippocampus._memories[memory_id]

        issues = hippocampus.check_consistency()
        assert len(issues) > 0

        repaired = hippocampus.repair_consistency()
        assert repaired > 0

        issues_after = hippocampus.check_consistency()
        assert issues_after == []


class TestHippocampusPersistence:
    """Test save/load roundtrip."""

    def test_save_load_preserves_memories(self, hippocampus, complete_memory_args, tmp_path):
        """Memories survive save/load cycle."""
        memory_id = hippocampus.capture(**complete_memory_args)

        path = tmp_path / "hippocampus.json"
        hippocampus.save(str(path))

        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippocampus2 = Hippocampus(HippocampusConfig(persistence_path=None))
        hippocampus2.load(str(path))

        assert len(hippocampus2) == 1
        memory = hippocampus2.get(memory_id)
        assert memory is not None

    def test_save_load_preserves_indices(self, hippocampus, complete_memory_args, tmp_path):
        """Indices survive save/load cycle."""
        hippocampus.capture(**complete_memory_args)

        path = tmp_path / "hippocampus.json"
        hippocampus.save(str(path))

        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hippocampus2 = Hippocampus(HippocampusConfig(persistence_path=None))
        hippocampus2.load(str(path))

        memories = hippocampus2.recall(goal="find mug")
        assert len(memories) == 1


class TestHippocampusSleep:
    """Test sleep consolidation."""

    def test_sleep_returns_stats(self, hippocampus, complete_memory_args):
        """Sleep returns consolidation statistics."""
        hippocampus.capture(**complete_memory_args)

        results = hippocampus.sleep()

        assert "compressed" in results
        assert "removed" in results
        assert "preserved" in results

    def test_sleep_preserves_recent_memories(self, hippocampus, complete_memory_args):
        """Recently captured memories are preserved."""
        memory_id = hippocampus.capture(**complete_memory_args)

        results = hippocampus.sleep()

        # Recent memory should be preserved
        assert memory_id in hippocampus._memories
        assert results["preserved"] >= 1

    def test_sleep_with_disabled_consolidation(self, complete_memory_args):
        """Sleep does nothing when consolidation is disabled."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        config = HippocampusConfig(
            enable_sleep_consolidation=False,
            persistence_path=None,
        )
        hippocampus = Hippocampus(config)
        hippocampus.capture(**complete_memory_args)

        results = hippocampus.sleep()

        assert results["compressed"] == 0
        assert results["removed"] == 0


class TestHippocampusStats:
    """Test statistics collection."""

    def test_stats_returns_expected_keys(self, hippocampus, complete_memory_args):
        """Stats returns all expected keys."""
        hippocampus.capture(**complete_memory_args)

        stats = hippocampus.stats()

        assert "total_memories" in stats
        assert "compressed_memories" in stats
        assert "full_memories" in stats
        assert "index_keys" in stats

    def test_stats_tracks_captures(self, hippocampus, complete_memory_args):
        """Stats tracks memory capture count."""
        assert hippocampus.stats()["memories_captured"] == 0

        hippocampus.capture(**complete_memory_args)

        assert hippocampus.stats()["memories_captured"] == 1


class TestAsyncCapture:
    """Test passive hippocampus — async capture via background worker."""

    def _make_hippocampus(self):
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        return Hippocampus(HippocampusConfig(
            persistence_path=None,
            enable_associative_graph=False,
        ))

    def test_async_capture_processes_in_background(self):
        """capture_from_loop_async queues and processes in background."""
        hipp = self._make_hippocampus()
        hipp.start_capture_worker()
        try:
            hipp.capture_from_loop_async(
                observation={"detected_objects": ["cup"]},
                state=None,
                intent={"goal": "find cup"},
                decision={"action": "look"},
                action={"tool": "focus_interests", "params": {}},
                result=None,
                run_id="test-1",
            )
            flushed = hipp.flush(timeout=5.0)
            assert flushed is True
            assert len(hipp) == 1
        finally:
            hipp.stop_capture_worker()

    def test_async_capture_multiple(self):
        """Multiple async captures all get processed."""
        hipp = self._make_hippocampus()
        hipp.start_capture_worker()
        try:
            for i in range(10):
                hipp.capture_from_loop_async(
                    observation={"detected_objects": [f"obj_{i}"]},
                    state=None,
                    intent={"goal": f"goal_{i}"},
                    decision={"action": "look"},
                    action={"tool": "focus_interests", "params": {}},
                    result=None,
                    run_id=f"run-{i}",
                )
            flushed = hipp.flush(timeout=5.0)
            assert flushed is True
            assert len(hipp) == 10
        finally:
            hipp.stop_capture_worker()

    def test_async_capture_snapshots_state(self):
        """State is snapshotted at queue time, not at processing time."""
        from unittest.mock import Mock

        hipp = self._make_hippocampus()
        hipp.start_capture_worker()
        try:
            state = Mock()
            state.snapshot.return_value = {"mode": "active", "val": 1}
            state.data = {"mode": "active", "val": 1}

            hipp.capture_from_loop_async(
                observation={},
                state=state,
                intent={},
                decision={},
                action={"tool": "test"},
                result=None,
                run_id="snap-test",
            )

            # Mutate state after queueing
            state.snapshot.return_value = {"mode": "changed", "val": 99}

            flushed = hipp.flush(timeout=5.0)
            assert flushed is True
            assert len(hipp) == 1

            # Verify snapshot() was called during queueing
            state.snapshot.assert_called()
        finally:
            hipp.stop_capture_worker()

    def test_queue_overflow_drops_oldest(self):
        """When queue is full, oldest capture is dropped."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hipp = Hippocampus(HippocampusConfig(
            persistence_path=None,
            enable_associative_graph=False,
            capture_queue_size=3,  # tiny queue
        ))
        # Don't start worker — items stay queued

        for i in range(5):
            hipp.capture_from_loop_async(
                observation={"i": i},
                state=None,
                intent={},
                decision={},
                action={"tool": "test"},
                result=None,
                run_id=f"run-{i}",
            )

        # Queue should have 3 items (2 dropped)
        assert hipp._capture_queue.qsize() == 3

    def test_flush_returns_true_when_empty(self):
        """flush on empty queue returns True immediately."""
        hipp = self._make_hippocampus()
        assert hipp.flush(timeout=0.1) is True

    def test_flush_timeout_returns_false(self):
        """flush returns False if queue doesn't drain in time."""
        from maxim.memory.hippocampus import Hippocampus, HippocampusConfig

        hipp = Hippocampus(HippocampusConfig(
            persistence_path=None,
            enable_associative_graph=False,
            capture_queue_size=10,
        ))
        # Queue items but don't start worker
        for i in range(5):
            hipp.capture_from_loop_async(
                observation={},
                state=None,
                intent={},
                decision={},
                action={"tool": "test"},
                result=None,
                run_id=f"run-{i}",
            )
        # flush should timeout since no worker is draining
        result = hipp.flush(timeout=0.1)
        assert result is False

    def test_stop_drains_queue(self):
        """stop_capture_worker drains remaining items before stopping."""
        hipp = self._make_hippocampus()
        hipp.start_capture_worker()
        try:
            for i in range(5):
                hipp.capture_from_loop_async(
                    observation={"i": i},
                    state=None,
                    intent={},
                    decision={},
                    action={"tool": "test"},
                    result=None,
                    run_id=f"run-{i}",
                )
            hipp.stop_capture_worker()
            assert len(hipp) == 5
        except Exception:
            hipp.stop_capture_worker()
            raise

    def test_start_stop_lifecycle(self):
        """Worker can be started and stopped cleanly."""
        hipp = self._make_hippocampus()
        hipp.start_capture_worker()
        hipp.stop_capture_worker()
        # Can restart
        hipp.start_capture_worker()
        hipp.stop_capture_worker()