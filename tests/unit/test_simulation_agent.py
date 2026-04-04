"""Tests for the simulation agent: bridge, tools, and personas."""

from __future__ import annotations

import threading
import time

import pytest

from maxim.simulation.bridge import SimulationBridge
from maxim.simulation.sinks import ActionRecord, RecordingSink
from maxim.simulation.conversational_source import ConversationalSource
from maxim.simulation.personas import get_persona, list_personas, SIMULATION_PERSONAS


# ── SimulationBridge tests ──────────────────────────────────────────────────


class TestSimulationBridge:
    def test_creation(self):
        bridge = SimulationBridge()
        assert bridge.turn_count == 0
        assert isinstance(bridge.percept_source, ConversationalSource)
        assert isinstance(bridge.action_sink, RecordingSink)

    def test_send_and_wait_timeout(self):
        """send_and_wait returns timed_out=True when no actions appear."""
        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        result = bridge.send_and_wait("hello", timeout=0.5)
        assert result["turn"] == 1
        assert result["timed_out"] is True
        assert result["response"] is None
        assert result["actions"] == []

    def test_send_and_wait_detects_response(self):
        """send_and_wait returns when actions appear in the sink."""
        bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)

        # Simulate AUT responding after a short delay
        def fake_aut():
            time.sleep(0.2)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="I can help with that",
            ))

        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()

        result = bridge.send_and_wait("Do something")
        t.join(timeout=2)

        assert result["turn"] == 1
        assert result["timed_out"] is False
        assert result["response"] == "I can help with that"
        assert len(result["actions"]) == 1

    def test_send_and_wait_settle_captures_multi_action(self):
        """settle detection waits for all actions in a multi-action response."""
        bridge = SimulationBridge(response_timeout=5.0, settle_s=0.5)

        def fake_aut():
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="read_file",
                result_success=True,
                result_output="file contents",
            ))
            time.sleep(0.2)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="Here's what I found",
            ))

        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()

        result = bridge.send_and_wait("Read my file")
        t.join(timeout=2)

        assert len(result["actions"]) == 2
        assert result["response"] == "Here's what I found"

    def test_send_and_wait_collects_multiple_responses(self):
        """Response text joins all respond/speak outputs."""
        bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)

        def fake_aut():
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="First part",
            ))
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="speak",
                result_success=True,
                result_output="Second part",
            ))

        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()

        result = bridge.send_and_wait("Tell me something")
        t.join(timeout=2)

        assert result["response"] == "First part\nSecond part"

    def test_send_and_wait_stop_event(self):
        """send_and_wait returns early when stop_event is set."""
        stop = threading.Event()
        bridge = SimulationBridge(response_timeout=10.0, stop_event=stop)

        def cancel_soon():
            time.sleep(0.3)
            stop.set()

        t = threading.Thread(target=cancel_soon, daemon=True)
        t.start()

        start = time.time()
        result = bridge.send_and_wait("hello")
        elapsed = time.time() - start
        t.join(timeout=2)

        assert elapsed < 2.0  # Should return well before 10s timeout
        assert result["timed_out"] is True

    def test_blocked_actions_tracked(self):
        """Blocked actions are returned in the blocked list."""
        bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)

        def fake_aut():
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="bash",
                blocked=True,
                block_reason="FearAgent: dangerous command",
            ))
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="I can't do that",
            ))

        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()

        result = bridge.send_and_wait("Run rm -rf /")
        t.join(timeout=2)

        assert len(result["blocked"]) == 1
        assert result["blocked"][0].tool_name == "bash"
        assert "dangerous" in result["blocked"][0].block_reason

    def test_turn_count_increments(self):
        bridge = SimulationBridge(response_timeout=0.3, settle_s=0.1)
        bridge.send_and_wait("turn 1")
        bridge.send_and_wait("turn 2")
        bridge.send_and_wait("turn 3")
        assert bridge.turn_count == 3

    def test_inject_pain(self):
        bridge = SimulationBridge()
        bridge.inject_pain(pain_type="collision", intensity=0.8)
        assert bridge.turn_count == 1
        percept = bridge.percept_source.next_percept()
        assert percept is not None
        assert percept.source == "proprioception"
        assert percept.content == "pain_signal"

    def test_finish_signals_exhaustion(self):
        bridge = SimulationBridge()
        assert not bridge.percept_source.is_exhausted()
        bridge.finish()
        assert bridge.percept_source.is_exhausted()

    def test_get_all_actions(self):
        bridge = SimulationBridge()
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="respond",
            result_success=True, result_output="hi",
        ))
        assert len(bridge.get_all_actions()) == 1

    def test_get_actions_since(self):
        bridge = SimulationBridge()
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="a", result_success=True))
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="b", result_success=True))
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="c", result_success=True))
        since_1 = bridge.get_actions_since(1)
        assert len(since_1) == 2
        assert since_1[0].tool_name == "b"


# ── Tool tests ──────────────────────────────────────────────────────────────


class TestSimulationTools:
    def _make_bridge_with_actions(self) -> SimulationBridge:
        bridge = SimulationBridge(response_timeout=0.3, settle_s=0.1)
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="respond",
            result_success=True, result_output="Hello",
        ))
        bridge.action_sink.record(ActionRecord(
            timestamp=time.time(), tool_name="bash",
            blocked=True, block_reason="dangerous",
        ))
        bridge._turn_count = 2
        return bridge

    def test_observe_actions_full_history(self):
        from maxim.simulation.tools import ObserveActionsTool
        bridge = self._make_bridge_with_actions()
        tool = ObserveActionsTool(bridge=bridge)
        result = tool.run()
        assert result.success
        assert result.output["total_actions"] == 2
        assert result.output["returned"] == 2

    def test_observe_actions_since_index(self):
        from maxim.simulation.tools import ObserveActionsTool
        bridge = self._make_bridge_with_actions()
        tool = ObserveActionsTool(bridge=bridge)
        result = tool.run(since_index=1)
        assert result.success
        assert result.output["returned"] == 1

    def test_check_completion_in_progress(self):
        from maxim.simulation.tools import CheckCompletionTool
        bridge = self._make_bridge_with_actions()
        tool = CheckCompletionTool(bridge=bridge, goal="test safety")
        result = tool.run()
        assert result.success
        assert result.output["goal"] == "test safety"
        assert result.output["turns_completed"] == 2

    def test_analyze_results(self):
        from maxim.simulation.tools import AnalyzeResultsTool
        bridge = self._make_bridge_with_actions()
        tool = AnalyzeResultsTool(bridge=bridge)
        result = tool.run(focus="safety")
        assert result.success
        assert result.output["blocked_count"] == 1
        assert "safety_summary" in result.output

    def test_inject_pain_tool(self):
        from maxim.simulation.tools import InjectPainTool
        bridge = SimulationBridge()
        tool = InjectPainTool(bridge=bridge)
        result = tool.run(pain_type="collision", intensity=0.9)
        assert result.success
        assert result.output["pain_type"] == "collision"
        assert bridge.turn_count == 1

    def test_finish_simulation_tool(self):
        from maxim.simulation.tools import FinishSimulationTool
        bridge = SimulationBridge()
        orch_source = ConversationalSource()
        tool = FinishSimulationTool(bridge=bridge, orchestrator_source=orch_source)
        result = tool.run(reason="goal achieved", summary="All tests passed")
        assert result.success
        assert result.output["finished"] is True
        assert bridge.percept_source.is_exhausted()
        assert orch_source.is_exhausted()

    def test_send_message_requires_text(self):
        from maxim.simulation.tools import SendMessageTool
        bridge = SimulationBridge(response_timeout=0.3)
        tool = SendMessageTool(bridge=bridge)
        result = tool.run(text="")
        assert not result.success
        assert "required" in result.error


# ── Persona tests ───────────────────────────────────────────────────────────


class TestPersonas:
    def test_all_personas_defined(self):
        assert len(SIMULATION_PERSONAS) == 8
        for name in ("adversarial", "cooperative", "confused", "escalating", "campaign", "refinement", "researcher", "sweep"):
            assert name in SIMULATION_PERSONAS

    def test_get_persona(self):
        p = get_persona("adversarial")
        assert p is not None
        assert p.name == "adversarial"
        assert p.context_prompt  # Not empty

    def test_get_persona_case_insensitive(self):
        assert get_persona("ADVERSARIAL") is not None
        assert get_persona("Campaign") is not None

    def test_get_unknown_persona(self):
        assert get_persona("nonexistent") is None

    def test_list_personas(self):
        names = list_personas()
        assert "adversarial" in names
        assert "campaign" in names
        assert len(names) == 8

    def test_all_personas_have_context_prompt(self):
        for name, persona in SIMULATION_PERSONAS.items():
            assert persona.context_prompt, f"Persona '{name}' has empty context_prompt"
            assert persona.focus, f"Persona '{name}' has empty focus"

    def test_continuous_suffix_appended(self):
        """--continuous appends NEVER STOP instructions to any persona."""
        p = get_persona("adversarial", continuous=True)
        assert p is not None
        assert "NEVER" in p.context_prompt
        assert "finish_simulation" in p.context_prompt

    def test_continuous_does_not_mutate_original(self):
        """Continuous mode returns a copy, doesn't modify the original."""
        original = get_persona("adversarial")
        continuous = get_persona("adversarial", continuous=True)
        assert "NEVER" not in original.context_prompt
        assert "NEVER" in continuous.context_prompt


# ── Decomposition tool tests ────────────────────────────────────────────────


class TestExtendSimulationTool:
    def test_extend_uses_main_bridge_when_no_sub(self):
        """Without a sub-AUT, extend uses the main bridge."""
        from maxim.simulation.tools import ExtendSimulationTool

        bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        # Fake a response on the main bridge
        def fake_aut():
            time.sleep(0.1)
            bridge.action_sink.record(ActionRecord(
                timestamp=time.time(), tool_name="respond",
                result_success=True, result_output="Extended response",
            ))
        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()

        tool = ExtendSimulationTool(main_bridge=bridge, spawn_tool=None)
        result = tool.run(goal="go deeper")
        t.join(timeout=2)

        assert result.success
        assert result.output["extended_sub_simulation"] is False
        assert result.output["response"] == "Extended response"

    def test_extend_uses_sub_bridge_when_available(self):
        """With an active sub-AUT, extend uses the sub-bridge."""
        from maxim.simulation.tools import ExtendSimulationTool

        main_bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)
        sub_bridge = SimulationBridge(response_timeout=0.5, settle_s=0.2)

        # Fake response on sub-bridge
        def fake_sub_aut():
            time.sleep(0.1)
            sub_bridge.action_sink.record(ActionRecord(
                timestamp=time.time(), tool_name="respond",
                result_success=True, result_output="Sub-AUT extended",
            ))
        t = threading.Thread(target=fake_sub_aut, daemon=True)
        t.start()

        # Mock spawn_tool with active sub-bridge
        class MockSpawn:
            active_sub_bridge = sub_bridge

        tool = ExtendSimulationTool(main_bridge=main_bridge, spawn_tool=MockSpawn())
        result = tool.run(goal="go deeper on sub")
        t.join(timeout=2)

        assert result.success
        assert result.output["extended_sub_simulation"] is True
        assert result.output["response"] == "Sub-AUT extended"

    def test_extend_requires_goal(self):
        from maxim.simulation.tools import ExtendSimulationTool
        bridge = SimulationBridge(response_timeout=0.3)
        tool = ExtendSimulationTool(main_bridge=bridge)
        result = tool.run(goal="")
        assert not result.success
        assert "required" in result.error


class TestSpawnSubSimulationTool:
    def test_spawn_requires_goal(self):
        from maxim.simulation.tools import SpawnSubSimulationTool
        tool = SpawnSubSimulationTool(llm_router=None, stop_event=threading.Event())
        result = tool.run(goal="")
        assert not result.success

    def test_teardown_idempotent(self):
        """Teardown when nothing is active doesn't crash."""
        from maxim.simulation.tools import SpawnSubSimulationTool
        tool = SpawnSubSimulationTool(llm_router=None, stop_event=threading.Event())
        assert tool.active_sub_bridge is None
        tool._teardown_sub()  # Should not raise
        assert tool.active_sub_bridge is None

    @pytest.mark.slow
    def test_spawn_creates_sub_bridge(self):
        """Spawn creates a sub-bridge (slow: bootstraps AUT)."""
        from maxim.simulation.tools import SpawnSubSimulationTool

        stop = threading.Event()
        tool = SpawnSubSimulationTool(
            llm_router=None,
            stop_event=stop,
            parent_bridge=None,
            sim_tmpdir=".",
        )

        result = tool.run(goal="test something")
        assert result.success
        assert result.output["goal"] == "test something"
        assert tool.active_sub_bridge is not None

        # Cleanup
        stop.set()
        tool._teardown_sub()


class TestCheckCompletionContinuous:
    def test_continuous_mode_never_completes(self):
        from maxim.simulation.tools import CheckCompletionTool
        bridge = SimulationBridge(response_timeout=0.3)
        bridge._turn_count = 100  # Even at 100 turns
        tool = CheckCompletionTool(bridge=bridge, goal="test", continuous=True)
        result = tool.run()
        assert result.success
        assert result.output["complete"] is False
        assert "Continuous" in result.output["reason"]

    def test_non_continuous_completes_at_50(self):
        from maxim.simulation.tools import CheckCompletionTool
        bridge = SimulationBridge(response_timeout=0.3)
        bridge._turn_count = 50
        tool = CheckCompletionTool(bridge=bridge, goal="test", continuous=False)
        result = tool.run()
        assert result.success
        assert result.output["complete"] is True


class TestSpinnerPrefix:
    def test_spinner_with_prefix(self):
        from maxim.simulation.spinner import Spinner
        s = Spinner(prefix="│  ")
        assert s._prefix == "│  "

    def test_spinner_default_no_prefix(self):
        from maxim.simulation.spinner import Spinner
        s = Spinner()
        assert s._prefix == ""
