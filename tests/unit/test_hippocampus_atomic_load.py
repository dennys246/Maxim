"""Unit tests for Hippocampus atomic load (BUG-5).

Verifies that load() parses into temp structures before swapping,
so a corrupted payload does not destroy existing in-memory data.
"""

from __future__ import annotations

import json
import time

import pytest

from maxim.memory.hippocampus import Hippocampus, HippocampusConfig


def _make_hippocampus() -> Hippocampus:
    return Hippocampus(HippocampusConfig(
        persistence_path=None,
        enable_associative_graph=False,
    ))


def _valid_memory_dict(memory_id: str = "mem-1") -> dict:
    """Return a minimal valid serialized EpisodicMemory dict."""
    ts = time.time()
    return {
        "id": memory_id,
        "timestamp": ts,
        "created_at": ts,
        "accessed_at": ts,
        "access_count": 1,
        "long_term": False,
        "perception": {
            "observations": {"test": True},
            "detected_objects": ["cup"],
            "detected_people": [],
            "salience": 0.5,
            "novelty": 0.5,
        },
        "context": {"active_goal": "test", "active_mode": "explore"},
        "decision": {"intent": {}, "reasoning": "", "confidence": 0.5},
        "action": {"tool_name": "look", "tool_params": {}},
        "outcome": {"success": True, "result": {}},
    }


class TestAtomicLoad:
    """Test that load() is atomic — corrupted files don't destroy state."""

    def test_corrupted_payload_preserves_existing_data(
        self, hippocampus, complete_memory_args, tmp_path
    ):
        """If a payload contains a malformed entry, load() raises and
        existing in-memory data is not cleared."""
        # Populate with a real memory
        original_id = hippocampus.capture(**complete_memory_args)
        assert len(hippocampus) == 1

        # Write a file with one valid and one malformed memory
        bad_payload = {
            "version": "3.0",
            "memories": [
                _valid_memory_dict("good-1"),
                {"_missing_id_and_timestamp": True},  # will raise KeyError
            ],
            "context_index": {},
            "stats": {},
        }
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps(bad_payload))

        with pytest.raises((KeyError, TypeError, ValueError)):
            hippocampus.load(str(bad_file))

        # Existing data must still be intact
        assert len(hippocampus) == 1
        assert hippocampus.get(original_id) is not None

    def test_valid_save_load_roundtrip(self, hippocampus, complete_memory_args, tmp_path):
        """Normal save/load roundtrip produces the same memories."""
        id1 = hippocampus.capture(**complete_memory_args)
        id2 = hippocampus.capture(**complete_memory_args)

        path = tmp_path / "roundtrip.json"
        hippocampus.save(str(path))

        hipp2 = _make_hippocampus()
        hipp2.load(str(path))

        assert len(hipp2) == 2
        assert hipp2.get(id1) is not None
        assert hipp2.get(id2) is not None

    def test_empty_memories_list_loads_cleanly(self, tmp_path):
        """A payload with an empty memories list loads without error."""
        payload = {
            "version": "3.0",
            "memories": [],
            "context_index": {},
            "stats": {},
        }
        path = tmp_path / "empty.json"
        path.write_text(json.dumps(payload))

        hipp = _make_hippocampus()
        hipp.load(str(path))

        assert len(hipp) == 0
