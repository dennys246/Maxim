"""Tests for the standalone experiment runner."""

from __future__ import annotations

import tempfile

import pytest

from maxim.simulation.experiment import ExperimentResult, load_campaign_turns


# ── ExperimentResult tests ───────────────────────────────────────────


class TestExperimentResult:
    def _make_result(self, **overrides):
        defaults = {
            "campaign_path": "test.yaml",
            "aut_model": "mistral-7b",
            "sim_turns": 7,
            "sim_duration_s": 45.0,
            "finish_reason": "complete",
            "analysis": {
                "memory_recall": {
                    "Verath": {"available": True, "count": 2, "total_stored": 14},
                    "nonexistent": {"available": True, "count": 0, "total_stored": 14},
                },
                "system_stats": {
                    "available": True,
                    "hippocampus_memories": 14,
                    "nac_causal_links": 5,
                },
            },
            "summary": "14 episodic memories, 5 causal links",
        }
        defaults.update(overrides)
        return ExperimentResult(**defaults)

    def test_memory_survived_true(self):
        result = self._make_result()
        assert result.memory_survived("Verath") is True

    def test_memory_survived_false(self):
        result = self._make_result()
        assert result.memory_survived("nonexistent") is False

    def test_memory_survived_unknown_keyword(self):
        result = self._make_result()
        assert result.memory_survived("unknown") is False

    def test_recall_count(self):
        result = self._make_result()
        assert result.recall_count("Verath") == 2
        assert result.recall_count("nonexistent") == 0
        assert result.recall_count("unknown") == 0

    def test_hippocampus_memories(self):
        result = self._make_result()
        assert result.hippocampus_memories == 14

    def test_causal_links(self):
        result = self._make_result()
        assert result.causal_links == 5

    def test_empty_analysis(self):
        result = self._make_result(analysis={})
        assert result.memory_survived("Verath") is False
        assert result.hippocampus_memories == 0
        assert result.causal_links == 0


# ── Campaign loading tests ───────────────────────────────────────────


class TestLoadCampaignTurns:
    def test_loads_valid_campaign(self):
        yaml_content = """
goal: test recall
percepts:
  - at: 0
    cli_input: "You meet Elara in a forest."
    metadata:
      experiment_role: seed
  - at: 2
    cli_input: "A bridge crosses a river."
    metadata:
      experiment_role: interference
  - at: 4
    cli_input: "A massive silver elm with a stone door."
    metadata:
      experiment_role: recall
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            turns = load_campaign_turns(f.name)

        assert len(turns) == 3
        assert turns[0]["cli_input"] == "You meet Elara in a forest."
        assert turns[0]["metadata"]["experiment_role"] == "seed"
        assert turns[1]["at"] == 2
        assert turns[2]["cli_input"] == "A massive silver elm with a stone door."

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_campaign_turns("/nonexistent/path.yaml")

    def test_empty_campaign(self):
        yaml_content = "goal: empty\npercepts: []\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            turns = load_campaign_turns(f.name)

        assert turns == []

    def test_skips_turns_without_cli_input(self):
        yaml_content = """
percepts:
  - at: 0
    cli_input: "Hello"
  - at: 1
    content: "pain_signal"
  - at: 2
    cli_input: "World"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            turns = load_campaign_turns(f.name)

        assert len(turns) == 2
        assert turns[0]["cli_input"] == "Hello"
        assert turns[1]["cli_input"] == "World"
