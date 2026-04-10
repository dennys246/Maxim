"""Tests for LearnedToolIndex — keyword-weighted hashtable for tool relevance."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


from maxim.tools.learned_index import LearnedToolIndex, _tokenize


# ─────────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FakeTool:
    name: str = "grab"
    description: str = "Pick up and hold an object"
    input_schema: dict = None
    MANUAL_KEYWORDS: set = None

    def __post_init__(self):
        if self.input_schema is None:
            self.input_schema = {"object_id": str, "force": float}
        if self.MANUAL_KEYWORDS is None:
            self.MANUAL_KEYWORDS = set()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────


class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("pick up the red cup")
        assert "pick" in tokens
        assert "red" in tokens
        assert "cup" in tokens
        assert "the" not in tokens  # stopword
        assert "up" not in tokens  # too short

    def test_case_insensitive(self):
        assert _tokenize("GRAB") == _tokenize("grab")

    def test_splits_on_punctuation(self):
        tokens = _tokenize("read_file: hello-world")
        assert "read" in tokens
        assert "file" in tokens
        assert "hello" in tokens
        assert "world" in tokens


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistration:
    def test_auto_extracts_keywords(self):
        index = LearnedToolIndex()
        tool = FakeTool(name="edit_file", description="Replace specific text in a file")
        index.register_tool(tool)
        scores = index.score_tools("edit the file")
        assert "edit_file" in scores
        assert scores["edit_file"] > 0

    def test_compound_name_split(self):
        index = LearnedToolIndex()
        tool = FakeTool(name="read_file", description="Read file contents")
        index.register_tool(tool)
        # Both "read" and "file" should match independently
        assert index.score_tools("read something")["read_file"] > 0
        assert index.score_tools("a file")["read_file"] > 0

    def test_manual_keywords(self):
        index = LearnedToolIndex()
        tool = FakeTool(name="grab")
        index.register_tool(tool)
        index.register_manual_keywords("grab", {"seize", "clutch", "grasp"})
        scores = index.score_tools("grasp the handle")
        assert "grab" in scores

    def test_manual_keywords_from_tool_class(self):
        index = LearnedToolIndex()
        tool = FakeTool(name="grab", MANUAL_KEYWORDS={"seize", "clutch"})
        index.register_tool(tool)
        scores = index.score_tools("seize it")
        assert "grab" in scores

    def test_duplicate_registration_skipped(self):
        index = LearnedToolIndex()
        tool = FakeTool(name="grab")
        index.register_tool(tool)
        # Modify weight
        index.record_outcome("pick up cup", "grab", success=True)
        old_stats = index.stats()
        # Re-register — should skip, not clobber
        index.register_tool(tool)
        assert index.stats() == old_stats


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


class TestScoring:
    def _make_index(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab", description="Pick up and hold an object"))
        index.register_tool(FakeTool(name="scan", description="Search the scene for objects"))
        index.register_tool(FakeTool(name="read_file", description="Read file contents from disk"))
        return index

    def test_matching_tool_scores_higher(self):
        index = self._make_index()
        scores = index.score_tools("pick up the object")
        assert scores.get("grab", 0) > scores.get("read_file", 0)

    def test_no_match_returns_empty(self):
        index = self._make_index()
        scores = index.score_tools("xyzzy")
        assert len(scores) == 0

    def test_normalization_prevents_verbose_dominance(self):
        """A tool with 15 keywords shouldn't dominate one with 4."""
        index = LearnedToolIndex()
        verbose = FakeTool(
            name="verbose_tool",
            description="This tool does many things including reading writing editing searching filtering sorting grouping mapping reducing transforming",
        )
        concise = FakeTool(name="concise_tool", description="Read files")
        index.register_tool(verbose)
        index.register_tool(concise)
        # Both match "read" — concise shouldn't be buried
        scores = index.score_tools("read the data")
        # Both should score, verbose shouldn't dominate by 10x
        if "verbose_tool" in scores and "concise_tool" in scores:
            ratio = scores["verbose_tool"] / scores["concise_tool"]
            assert ratio < 3.0  # Reasonable bound


# ─────────────────────────────────────────────────────────────────────────────
# get_relevant_tools
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRelevantTools:
    def test_small_registry_returns_all_tools_as_relevant(self):
        """Below SMALL_REGISTRY_THRESHOLD the filter bypasses entirely.

        This is a deliberate design choice: at small registry sizes (<=60
        tools) the filter's token savings are less than the prompt overhead
        of splitting the tools section, and a cold-signal query is
        statistically likely to land on a random subset that hides the
        right tool. Returning the full registry as 'relevant' produces a
        single unified manifest with zero hallucination risk.
        """
        index = LearnedToolIndex()
        for i in range(10):
            index.register_tool(FakeTool(name=f"tool_{i}", description=f"Tool number {i}"))
        relevant, background = index.get_relevant_tools("xyzzy no match")
        assert len(relevant) == 10
        assert background == []

    def test_large_registry_cold_start_meets_min_tools_floor(self):
        """Above SMALL_REGISTRY_THRESHOLD, even a no-match query must return
        at least MIN_TOOLS relevant entries so the model has enough surface
        area to find a match. Previously returned only MIN_TOOLS=3 which
        caused systematic tool hallucination.
        """
        index = LearnedToolIndex()
        count = index.SMALL_REGISTRY_THRESHOLD + 10
        for i in range(count):
            index.register_tool(FakeTool(name=f"tool_{i}", description=f"Tool number {i}"))
        relevant, background = index.get_relevant_tools("xyzzy no match")
        assert len(relevant) >= index.MIN_TOOLS
        assert len(relevant) + len(background) == count

    def test_high_scoring_tools_are_relevant(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab", description="Pick up objects"))
        index.register_tool(FakeTool(name="scan", description="Search scene"))
        relevant, background = index.get_relevant_tools("pick up the object")
        assert "grab" in relevant

    def test_partitions_are_disjoint(self):
        index = LearnedToolIndex()
        for i in range(5):
            index.register_tool(FakeTool(name=f"tool_{i}", description=f"Tool {i}"))
        relevant, background = index.get_relevant_tools("tool 0")
        assert set(relevant) & set(background) == set()
        assert len(relevant) + len(background) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Learning
# ─────────────────────────────────────────────────────────────────────────────


class TestLearning:
    def test_success_strengthens_keywords(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        before = index.score_tools("pick object")
        for _ in range(10):
            index.record_outcome("pick up the object", "grab", success=True)
        after = index.score_tools("pick object")
        assert after.get("grab", 0) > before.get("grab", 0)

    def test_failure_does_not_weaken(self):
        """Failure counts observation but doesn't change weight."""
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        before = index.score_tools("pick object")
        for _ in range(10):
            index.record_outcome("pick up the object", "grab", success=False)
        after = index.score_tools("pick object")
        # Weights should be unchanged (failure = no-op on weight)
        assert abs(after.get("grab", 0) - before.get("grab", 0)) < 0.01

    def test_new_keyword_discovery(self):
        """Successful execution creates learned keywords from unmatched goal tokens."""
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab", description="Pick up objects"))
        # "mug" is not in grab's auto-extracted keywords
        assert index.score_tools("mug").get("grab", 0) == 0
        # After successful use with "mug" in goal:
        index.record_outcome("grab the mug from the table", "grab", success=True)
        # "mug" should now be a learned keyword
        assert index.score_tools("mug").get("grab", 0) > 0

    def test_new_keyword_capped(self):
        """At most NEW_KEYWORD_MAX_PER_OUTCOME new keywords per outcome."""
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab", description="Pick up objects"))
        original_count = index.stats()["total_keywords"]
        # Goal with many novel tokens
        index.record_outcome("alpha beta gamma delta epsilon zeta", "grab", success=True)
        new_count = index.stats()["total_keywords"]
        assert new_count - original_count <= index.NEW_KEYWORD_MAX_PER_OUTCOME

    def test_surfaced_unused_decays(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab", description="Pick up objects"))
        index.register_tool(FakeTool(name="scan", description="Search scene objects"))
        # Strengthen grab first
        for _ in range(5):
            index.record_outcome("pick up object", "grab", success=True)
        before = index.score_tools("object")
        # scan was surfaced but grab was used — scan's "object" keyword should decay
        for _ in range(10):
            index.record_surfaced_but_unused("find the object", ["grab", "scan"], "grab")
        after = index.score_tools("object")
        # scan's score for "object" should have decreased
        assert after.get("scan", 0) < before.get("scan", 0.5)

    def test_unknown_tool_outcome_ignored(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        # Recording outcome for unregistered tool should not raise
        index.record_outcome("do something", "nonexistent_tool", success=True)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, tmp_path: Path):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        # Train it
        for _ in range(5):
            index.record_outcome("pick up cup", "grab", success=True)

        path = str(tmp_path / "tool_index.json")
        index.save(path)

        # Load into fresh index with same tool registered
        index2 = LearnedToolIndex()
        index2.register_tool(FakeTool(name="grab"))
        index2.load(path)

        # Weights should be restored
        s1 = index.score_tools("pick up cup")
        s2 = index2.score_tools("pick up cup")
        assert abs(s1.get("grab", 0) - s2.get("grab", 0)) < 0.01

    def test_load_restores_learned_keywords(self, tmp_path: Path):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        index.record_outcome("grab the mug", "grab", success=True)
        assert index.score_tools("mug").get("grab", 0) > 0

        path = str(tmp_path / "tool_index.json")
        index.save(path)

        index2 = LearnedToolIndex()
        index2.register_tool(FakeTool(name="grab"))
        index2.load(path)
        # "mug" was learned — should be restored
        assert index2.score_tools("mug").get("grab", 0) > 0

    def test_load_ignores_unknown_tools(self, tmp_path: Path):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        path = str(tmp_path / "tool_index.json")
        index.save(path)

        index2 = LearnedToolIndex()
        # Don't register "grab" — it should be ignored on load
        index2.load(path)
        assert index2.stats()["tools_registered"] == 0

    def test_load_missing_file(self, tmp_path: Path):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        # Should not raise
        index.load(str(tmp_path / "nonexistent.json"))

    def test_atomic_write(self, tmp_path: Path):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        path = str(tmp_path / "tool_index.json")
        index.save(path)
        assert os.path.exists(path)
        assert not os.path.exists(f"{path}.tmp")
        # Verify valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "grab" in data


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_reflect_state(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        index.register_tool(FakeTool(name="scan"))
        stats = index.stats()
        assert stats["tools_registered"] == 2
        assert stats["total_keywords"] > 0
        assert stats["learned_keywords"] == 0

    def test_learned_keyword_counted(self):
        index = LearnedToolIndex()
        index.register_tool(FakeTool(name="grab"))
        index.record_outcome("grab the mug", "grab", success=True)
        stats = index.stats()
        assert stats["learned_keywords"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Thread safety
# ─────────────────────────────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_score_and_update(self):
        """Concurrent reads and writes should not raise."""
        import threading

        index = LearnedToolIndex()
        for i in range(10):
            index.register_tool(FakeTool(name=f"tool_{i}", description=f"Tool {i} does things"))

        errors = []

        def reader():
            for _ in range(100):
                try:
                    index.score_tools("tool things stuff")
                    index.get_relevant_tools("tool things stuff")
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(100):
                try:
                    index.record_outcome(f"do tool things {i}", f"tool_{i % 10}", success=True)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
