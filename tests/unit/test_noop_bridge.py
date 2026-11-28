"""Tests for NoOp deliberation marker and bridge F2 fix (Stage 4).

Covers:
- RecordingSink.record_noop() creates correct ActionRecord
- Bridge doesn't count noop markers as real actions (timed_out logic)
- Noop markers break the bridge timeout loop (action count grows)
- Bridge response text excludes noop markers
"""

from __future__ import annotations

import time

from maxim.simulation.sinks import ActionRecord, RecordingSink


class TestRecordingSinkNoop:
    def test_record_noop_creates_marker(self):
        sink = RecordingSink()
        sink.record_noop(reason="thinking")
        assert len(sink.actions) == 1
        action = sink.actions[0]
        assert action.tool_name == "_deliberation_noop"
        assert action.result_success is True
        assert "thinking" in str(action.result_output)

    def test_record_noop_default_reason(self):
        sink = RecordingSink()
        sink.record_noop()
        action = sink.actions[0]
        assert "deliberation tick" in str(action.result_output)

    def test_noop_increments_action_count(self):
        """Noop markers grow len(actions) — this is the F2 fix."""
        sink = RecordingSink()
        count_before = len(sink.actions)
        sink.record_noop()
        assert len(sink.actions) == count_before + 1

    def test_noop_not_found_by_find_actions(self):
        """find_actions skips blocked actions; noop is not blocked."""
        sink = RecordingSink()
        sink.record_noop()
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello",
            )
        )
        # find_actions without tool filter returns all non-blocked
        all_actions = sink.find_actions()
        assert len(all_actions) == 2  # both are non-blocked

        # find_actions with tool filter returns only matching
        responds = sink.find_actions(tool="respond")
        assert len(responds) == 1
        assert responds[0].tool_name == "respond"

    def test_noop_mixed_with_real_actions(self):
        """Mix of noops and real actions — both recorded."""
        sink = RecordingSink()
        sink.record_noop(reason="thinking 1")
        sink.record(
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello",
            )
        )
        sink.record_noop(reason="thinking 2")
        assert len(sink.actions) == 3


class TestBridgeNoopHandling:
    """Test that the bridge's timed_out logic excludes noop markers."""

    def test_noop_only_is_timed_out(self):
        """If only noop markers exist, bridge reports timed_out=True."""
        # Simulate what the bridge does with new_actions
        new_actions = [
            ActionRecord(
                timestamp=time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output="thinking",
            ),
        ]
        real_actions = [a for a in new_actions if a.tool_name != "_deliberation_noop"]
        assert len(real_actions) == 0
        # Bridge would set timed_out = len(real_actions) == 0
        timed_out = len(real_actions) == 0
        assert timed_out is True

    def test_real_action_is_not_timed_out(self):
        """If real actions exist alongside noops, not timed out."""
        new_actions = [
            ActionRecord(
                timestamp=time.time(),
                tool_name="_deliberation_noop",
                result_success=True,
                result_output="thinking",
            ),
            ActionRecord(
                timestamp=time.time(),
                tool_name="respond",
                result_success=True,
                result_output="hello world",
            ),
        ]
        real_actions = [a for a in new_actions if a.tool_name != "_deliberation_noop"]
        assert len(real_actions) == 1
        timed_out = len(real_actions) == 0
        assert timed_out is False

    def test_response_text_excludes_noop(self):
        """The real bridge's response text ignores noop markers: a marker carries no utterance (#1261)."""
        import threading

        from maxim.simulation.bridge import SimulationBridge

        bridge = SimulationBridge(response_timeout=5.0, settle_s=0.3)

        def fake_aut():
            time.sleep(0.1)
            bridge.action_sink.record_noop(reason="just thinking")
            bridge.action_sink.record(
                ActionRecord(
                    timestamp=time.time(),
                    tool_name="respond",
                    tool_args={"message": "actual response"},
                    result_success=True,
                    result_output={"delivered": True, "mode": "cli"},
                    utterance="actual response",
                )
            )

        t = threading.Thread(target=fake_aut, daemon=True)
        t.start()
        result = bridge.send_and_wait("hello")
        t.join(timeout=2)
        assert result["noop_count"] == 1
        assert result["response"] == "actual response"


def test_the_sub_simulation_report_counts_no_marker_as_an_action(monkeypatch, tmp_path):
    """#1743: a sub-sim AUT's declined turn records ``_deliberation_noop``; the report the narrator reads counts the
    agent's actions only (``ActionRecord.is_synthetic``), as every other counter does."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import maxim.runtime.agent_loop as agent_loop
    import maxim.simulation.bridge as bridge_mod
    from maxim.simulation.sinks import DELIBERATION_NOOP
    from maxim.simulation.tools import SpawnSubSimulationTool

    monkeypatch.chdir(tmp_path)
    records = [
        ActionRecord(timestamp=time.time(), tool_name="respond", result_success=True),
        ActionRecord(timestamp=time.time(), tool_name=DELIBERATION_NOOP, result_success=True),
    ]
    monkeypatch.setattr(agent_loop, "run_agentic_loop", lambda *a, **kw: None)
    monkeypatch.setattr("maxim.agents.maxim_agent.MaximAgent", lambda *a, **kw: MagicMock(name="sub_agent"))
    monkeypatch.setattr(
        bridge_mod.SimulationBridge,
        "send_and_wait",
        lambda self, goal: {"turn": 0, "response": None, "actions": records, "blocked": [], "timed_out": False},
    )
    from maxim.runtime.bootstrap import build_executor
    from maxim.tools.registry import ToolRegistry
    from tests.policy_helpers import TEST_POLICY

    tool = SpawnSubSimulationTool(llm_router=None, sim_tmpdir=str(tmp_path), sandbox_dirs=[str(tmp_path)])
    executor = build_executor(ToolRegistry([tool]), pain_bus=None, policy=TEST_POLICY)
    out = executor.execute({"tool_name": "spawn_sub_simulation", "params": {"goal": "hello"}})
    if tool._sub_thread is not None:
        tool._sub_thread.join(timeout=10)
    report = out.output if isinstance(out.output, dict) else SimpleNamespace(get=lambda k: None)
    assert out.success, out
    assert report.get("total_actions") == 1 and len(report.get("actions")) == 2


def test_the_last_marker_is_the_turns_final_word():
    """``nothing_done_kind`` reads the LAST no-op marker of a turn's records (#1743)."""
    from maxim.simulation.sinks import DELIBERATION_NOOP, NOOP_STALE, noop_error_reason, nothing_done_kind

    early = ActionRecord(timestamp=1.0, tool_name=DELIBERATION_NOOP, result_output=NOOP_STALE)
    late = ActionRecord(timestamp=2.0, tool_name=DELIBERATION_NOOP, result_output=noop_error_reason("boom"))
    assert nothing_done_kind([early, late]) == ("error", "boom")
    assert nothing_done_kind([late, early]) == ("stale", "")
    assert nothing_done_kind([ActionRecord(timestamp=0.0, tool_name="respond")]) is None


def test_an_error_reason_reduces_urls_before_the_cut():
    """A cut first would keep ``user:pw@`` (the URL scrub then cannot match a truncated URL), #997."""
    from maxim.simulation.sinks import noop_error_reason

    reason = noop_error_reason("x" * 30 + " for url 'http://user:pw@10.0.0.2:8100/a/very/long/path?token=abc'")
    assert reason.startswith("proposal_error:") and len(reason) <= len("proposal_error:") + 60
    assert "user" not in reason and "pw" not in reason and "token" not in reason
