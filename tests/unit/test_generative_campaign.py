"""Tests for generative campaign system.

Covers:
- NarrativeArc / NarrativePhase construction and serialization
- Builtin arc templates
- Arc selection from goal keywords
- YAML arc loading
- Narrator two-call approach (decision + generation)
- Narrator single-call approach
- Phase advancement and completion detection
- Generative runner turn loop (with mock LLM + bridge)
- YAML export of generated campaigns
- Entity naming (AgentProfile extension)
"""

from __future__ import annotations

from unittest.mock import MagicMock

from maxim.mesh.identity import AgentProfile
from maxim.simulation.arcs import (
    BUILTIN_ARCS,
    NarrativeArc,
    NarrativePhase,
    load_arc_yaml,
    select_arc_for_goal,
)
from maxim.simulation.generative_runner import (
    GenerativeCampaignResult,
    export_campaign_yaml,
    run_generative_campaign,
)
from maxim.simulation.narrator import Narrator


# ---------------------------------------------------------------------------
# NarrativeArc
# ---------------------------------------------------------------------------


class TestNarrativeArc:
    def test_basic_construction(self):
        arc = NarrativeArc(
            name="test",
            description="A test arc",
            phases=[
                NarrativePhase("seed", "Plant a detail", 1, 2),
                NarrativePhase("recall", "Test recall", 1, 1),
            ],
        )
        assert arc.name == "test"
        assert len(arc.phases) == 2
        assert arc.total_turns_min == 2
        assert arc.total_turns_max == 3

    def test_phase_names(self):
        arc = NarrativeArc(
            name="test",
            description="",
            phases=[
                NarrativePhase("a", "...", 1, 1),
                NarrativePhase("b", "...", 1, 1),
            ],
        )
        assert arc.phase_names == ["a", "b"]

    def test_to_narrator_instructions(self):
        arc = NarrativeArc(
            name="test",
            description="A test",
            phases=[NarrativePhase("seed", "Plant a detail", 1, 2)],
        )
        instructions = arc.to_narrator_instructions()
        assert "NARRATIVE ARC: test" in instructions
        assert "SEED" in instructions
        assert "Plant a detail" in instructions

    def test_to_dict(self):
        arc = NarrativeArc(
            name="test",
            description="A test",
            phases=[NarrativePhase("seed", "Plant", 1, 2)],
        )
        d = arc.to_dict()
        assert d["name"] == "test"
        assert len(d["phases"]) == 1
        assert d["phases"][0]["turns"] == [1, 2]


# ---------------------------------------------------------------------------
# Builtin arcs
# ---------------------------------------------------------------------------


class TestBuiltinArcs:
    def test_memory_recall_exists(self):
        arc = BUILTIN_ARCS["memory_recall"]
        assert arc.name == "memory_recall"
        assert len(arc.phases) == 5
        assert arc.phases[0].name == "seed"

    def test_causal_learning_exists(self):
        arc = BUILTIN_ARCS["causal_learning"]
        assert len(arc.phases) == 3

    def test_safety_boundary_exists(self):
        arc = BUILTIN_ARCS["safety_boundary"]
        assert len(arc.phases) == 3

    def test_skill_learning_exists(self):
        arc = BUILTIN_ARCS["skill_learning"]
        assert len(arc.phases) == 7
        assert arc.phases[0].name == "introduction"
        assert arc.phases[-1].name == "reflection"

    def test_all_arcs_have_phases(self):
        for name, arc in BUILTIN_ARCS.items():
            assert len(arc.phases) > 0, f"Arc {name} has no phases"
            assert arc.total_turns_min > 0, f"Arc {name} has 0 min turns"


# ---------------------------------------------------------------------------
# Arc selection
# ---------------------------------------------------------------------------


class TestArcSelection:
    def test_memory_keywords(self):
        arc = select_arc_for_goal("test memory recall under interference")
        assert arc is not None
        assert arc.name == "memory_recall"

    def test_causal_keywords(self):
        arc = select_arc_for_goal("test causal learning and prediction")
        assert arc is not None
        assert arc.name == "causal_learning"

    def test_safety_keywords(self):
        arc = select_arc_for_goal("test safety boundaries and ethics")
        assert arc is not None
        assert arc.name == "safety_boundary"

    def test_skill_keywords(self):
        arc = select_arc_for_goal("test skill learning and practice")
        assert arc is not None
        assert arc.name == "skill_learning"

    def test_no_match(self):
        arc = select_arc_for_goal("evaluate quantum entanglement theories")
        assert arc is None

    def test_best_match_wins(self):
        # "memory recall" has 2 keywords matching, should win over partial matches
        arc = select_arc_for_goal("memory recall")
        assert arc is not None
        assert arc.name == "memory_recall"


# ---------------------------------------------------------------------------
# YAML arc loading
# ---------------------------------------------------------------------------


class TestArcYamlLoading:
    def test_load_arc(self, tmp_path):
        yaml_content = """
name: test_arc
description: A test arc
phases:
  - name: opening
    turns: [1, 2]
    instruction: "Open the story"
  - name: climax
    turns: [2, 3]
    instruction: "Build to climax"
  - name: resolution
    turns: [1, 1]
    instruction: "Resolve the story"
"""
        arc_path = tmp_path / "test_arc.yaml"
        arc_path.write_text(yaml_content)

        arc = load_arc_yaml(arc_path)
        assert arc.name == "test_arc"
        assert len(arc.phases) == 3
        assert arc.phases[0].turns_min == 1
        assert arc.phases[0].turns_max == 2
        assert arc.source == "yaml"

    def test_load_nonexistent(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_arc_yaml(tmp_path / "nonexistent.yaml")

    def test_single_turn_value(self, tmp_path):
        yaml_content = """
name: simple
description: Simple arc
phases:
  - name: only_phase
    turns: 5
    instruction: "Do something 5 times"
"""
        arc_path = tmp_path / "simple.yaml"
        arc_path.write_text(yaml_content)

        arc = load_arc_yaml(arc_path)
        assert arc.phases[0].turns_min == 5
        assert arc.phases[0].turns_max == 5


# ---------------------------------------------------------------------------
# Narrator
# ---------------------------------------------------------------------------


class TestNarrator:
    def _make_mock_llm(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "seed",
                "scene_type": "introduction",
                "notes": "A mysterious stranger",
                "done": False,
            }
        )
        llm.generate_text = MagicMock(return_value="The sun sets over the dusty road.")
        return llm

    def test_decide(self):
        llm = self._make_mock_llm()
        arc = BUILTIN_ARCS["memory_recall"]
        narrator = Narrator(llm, arc, goal="test memory")

        decision = narrator.decide()
        assert "phase" in decision
        assert decision["done"] is False

    def test_generate(self):
        llm = self._make_mock_llm()
        arc = BUILTIN_ARCS["memory_recall"]
        narrator = Narrator(llm, arc, goal="test memory")

        decision = {"phase": "seed", "scene_type": "intro", "notes": "start"}
        text = narrator.generate(decision)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_single_call(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "seed",
                "narrative": "A dark forest stretches before you.",
                "done": False,
            }
        )
        arc = BUILTIN_ARCS["memory_recall"]
        narrator = Narrator(llm, arc, goal="test memory")

        text, decision = narrator.generate_single()
        assert "forest" in text
        assert decision["done"] is False

    def test_phase_advancement(self):
        llm = self._make_mock_llm()
        # Create a short arc with 1-turn phases
        arc = NarrativeArc(
            name="short",
            description="Short test",
            phases=[
                NarrativePhase("a", "Phase A", 1, 1),
                NarrativePhase("b", "Phase B", 1, 1),
            ],
        )
        narrator = Narrator(llm, arc, goal="test")

        # First turn in phase A
        assert narrator.current_phase == "a"
        narrator.generate({"phase": "a", "scene_type": "x"})

        # Should auto-advance to B after 1 turn
        narrator.decide()
        assert narrator.current_phase == "b"

    def test_completion(self):
        llm = self._make_mock_llm()
        arc = NarrativeArc(
            name="tiny",
            description="Tiny",
            phases=[NarrativePhase("only", "Only phase", 1, 1)],
        )
        narrator = Narrator(llm, arc, goal="test")
        narrator.generate({"phase": "only"})
        narrator.decide()  # Should trigger done
        assert narrator.is_done

    def test_done_from_llm(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "seed",
                "scene_type": "final",
                "notes": "wrap up",
                "done": True,
            }
        )
        arc = BUILTIN_ARCS["memory_recall"]
        narrator = Narrator(llm, arc, goal="test")
        decision = narrator.decide()
        assert decision["done"] is True
        assert narrator.is_done


# ---------------------------------------------------------------------------
# Generative runner
# ---------------------------------------------------------------------------


class TestGenerativeRunner:
    def test_run_basic(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "seed",
                "scene_type": "intro",
                "notes": "",
                "done": False,
            }
        )
        llm.generate_text = MagicMock(return_value="The story begins.")

        bridge = MagicMock()
        bridge.send_and_wait = MagicMock(
            return_value={
                "response": "I look around.",
                "turn": 1,
                "timed_out": False,
            }
        )

        result = run_generative_campaign(
            goal="test memory recall",
            bridge=bridge,
            llm=llm,
            max_turns=3,
        )

        assert isinstance(result, GenerativeCampaignResult)
        assert result.total_turns > 0
        assert result.arc_name == "memory_recall"
        assert bridge.send_and_wait.called

    def test_max_turns_cap(self):
        llm = MagicMock()
        # Never set done=True → should hit max_turns
        llm.generate_json = MagicMock(
            return_value={
                "phase": "seed",
                "scene_type": "x",
                "notes": "",
                "done": False,
            }
        )
        llm.generate_text = MagicMock(return_value="Scene.")

        bridge = MagicMock()
        bridge.send_and_wait = MagicMock(return_value={"response": "ok"})

        result = run_generative_campaign(
            goal="test something",
            bridge=bridge,
            llm=llm,
            max_turns=5,
        )

        assert result.total_turns <= 5

    def test_custom_arc(self):
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "x",
                "scene_type": "y",
                "done": False,
            }
        )
        llm.generate_text = MagicMock(return_value="Custom scene.")

        bridge = MagicMock()
        bridge.send_and_wait = MagicMock(return_value={"response": "ok"})

        custom_arc = NarrativeArc(
            name="custom",
            description="Custom test",
            phases=[NarrativePhase("only", "Do it", 1, 1)],
        )

        result = run_generative_campaign(
            goal="test custom",
            bridge=bridge,
            llm=llm,
            arc=custom_arc,
            max_turns=5,
        )

        assert result.arc_name == "custom"

    def test_embodied_arc_does_not_deregister_conversational_tools(self):
        """Regression guard for PR D — the deregister block at
        generative_runner.py:365-372 was deleted because it forced
        a silent ``Tool not registered: 'respond'`` loop whenever the
        prompt_builder (which we did not consult before deregistering)
        still instructed the LLM to call ``respond``. The fix routes the
        intent through ``LLMRequest.is_embodied`` and gates prompt content
        instead. This test pins the deletion: an arc whose phases declare
        ``world_entities`` must NOT cause tool_registry.deregister to be
        invoked for ``respond`` or ``say``. See
        docs/plans/cradle_activation_fixes.md (Finding B).
        """
        llm = MagicMock()
        llm.generate_json = MagicMock(
            return_value={
                "phase": "intro",
                "scene_type": "embodied",
                "done": True,  # End immediately
            }
        )
        llm.generate_text = MagicMock(return_value="Scene.")

        bridge = MagicMock()
        bridge.send_and_wait = MagicMock(return_value={"response": "ok"})

        tool_registry = MagicMock()
        # Without the fix, the deregister loop fires on (respond, say)
        # and we'd see ≥1 call to tool_registry.deregister. The fix
        # deletes that loop entirely.
        embodied_arc = NarrativeArc(
            name="embodied_test",
            description="Arc with world_entities to trigger the embodied branch",
            phases=[
                NarrativePhase(
                    "intro",
                    "Look around",
                    1,
                    1,
                    world_entities=("items/cradle_food",),
                ),
            ],
        )

        run_generative_campaign(
            goal="test embodied",
            bridge=bridge,
            llm=llm,
            arc=embodied_arc,
            max_turns=2,
            tool_registry=tool_registry,
        )

        deregistered = [call.args[0] for call in tool_registry.deregister.call_args_list if call.args]
        assert "respond" not in deregistered, (
            f"respond should NOT be deregistered post-PR D; deregistered={deregistered}"
        )
        assert "say" not in deregistered, f"say should NOT be deregistered post-PR D; deregistered={deregistered}"


# ---------------------------------------------------------------------------
# YAML export
# ---------------------------------------------------------------------------


class TestYamlExport:
    def test_export(self, tmp_path):
        result = GenerativeCampaignResult(
            goal="test memory",
            arc_name="memory_recall",
            total_turns=3,
            turns=[
                {"turn": 1, "narrative": "The road...", "phase": "seed", "aut_response": "I walk."},
                {"turn": 2, "narrative": "A stranger...", "phase": "interference", "aut_response": "Hello."},
                {"turn": 3, "narrative": "Remember?", "phase": "recall", "aut_response": "Yes!"},
            ],
            decisions=[{"phase": "seed", "done": False}],
        )

        path = export_campaign_yaml(tmp_path, result)
        assert path.exists()
        assert path.name == "generated_campaign.yaml"

        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "generated_memory_recall"
        assert data["generated"] is True
        assert len(data["percepts"]) == 3
        assert data["percepts"][0]["text"] == "The road..."


# ---------------------------------------------------------------------------
# Entity naming
# ---------------------------------------------------------------------------


class TestEntityNaming:
    def test_entity_type_field(self):
        profile = AgentProfile(nickname="Verath", role="aut", entity_type="aut")
        assert profile.entity_type == "aut"

    def test_default_entity_type(self):
        profile = AgentProfile(nickname="worker", role="agent")
        assert profile.entity_type == "agent"

    def test_log_prefix(self):
        profile = AgentProfile(nickname="Verath", role="aut")
        assert profile.log_prefix == "[Verath]"

    def test_long_name_truncated_in_prefix(self):
        profile = AgentProfile(nickname="A" * 25, role="aut")
        assert len(profile.log_prefix) <= 23  # [17+...] = 22

    def test_to_dict_includes_entity_type(self):
        profile = AgentProfile(nickname="narrator", role="orchestrator", entity_type="npc")
        d = profile.to_dict()
        assert d["entity_type"] == "npc"

    def test_from_dict_entity_type(self):
        d = {"nickname": "narrator", "role": "orchestrator", "entity_type": "npc"}
        profile = AgentProfile.from_dict(d)
        assert profile.entity_type == "npc"

    def test_from_dict_default_entity_type(self):
        d = {"nickname": "old_agent", "role": "worker"}
        profile = AgentProfile.from_dict(d)
        assert profile.entity_type == "agent"


class TestAutTurnTimeout:
    """``generative_runner._aut_turn_timeout_s`` — env-var-configurable AUT
    per-turn response timeout. Default 30s for fast models; widen via
    MAXIM_SIM_AUT_TURN_TIMEOUT_S for reasoning models whose <think> chains
    exceed 30s/action. Regression for the 2026-06-11 DeepSeek-R1 fire where
    every turn timed out at 30s while R1 was still reasoning (~150s/action).
    """

    def _fn(self):
        from maxim.simulation.generative_runner import _aut_turn_timeout_s

        return _aut_turn_timeout_s

    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", raising=False)
        assert self._fn()() == 30.0

    def test_override(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", "300")
        assert self._fn()() == 300.0

    def test_clamp_floor(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", "1")
        assert self._fn()() == 5.0

    def test_clamp_ceiling(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", "99999")
        assert self._fn()() == 1800.0

    def test_malformed_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", "not-a-number")
        assert self._fn()() == 30.0

    def test_empty_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("MAXIM_SIM_AUT_TURN_TIMEOUT_S", "  ")
        assert self._fn()() == 30.0
