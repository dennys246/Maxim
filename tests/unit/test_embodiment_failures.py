"""Tests for composed failures + persistence (Phase 2).

Covers:
- Compound failure triggers (all mode)
- Composed failure modes (composes field)
- Persistent failures with recovery
- Failure state export/import
- Save/load to JSON file
- ToolPainBridge embodiment pain handling
- Active failure tracking in stats
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from maxim.embodiment.body import Embodiment
from maxim.embodiment.sem import Entity, FailureMode, FailureTrigger

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios" / "embodiment"


# ---------------------------------------------------------------------------
# Composed failures
# ---------------------------------------------------------------------------


class TestComposedFailures:
    def test_compound_trigger_all(self):
        """Tennis elbow: both strain AND fatigue must exceed thresholds."""
        fm = FailureMode(
            name="tennis_elbow",
            composes=["strain", "fatigue"],
            triggers=[
                FailureTrigger("strain", ">", 0.6, 0.5),
                FailureTrigger("fatigue", ">", 0.5, 0.5),
            ],
            trigger_mode="all",
            pain_intensity=0.5,
        )
        # Only strain high → no fire
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.3}) is False
        # Only fatigue high → no fire
        assert fm.evaluate({"strain": 0.4, "fatigue": 0.6}) is False
        # Both high → fire
        assert fm.evaluate({"strain": 0.7, "fatigue": 0.6}) is True

    def test_composed_labels(self):
        fm = FailureMode(
            name="tennis_elbow",
            composes=["strain", "fatigue"],
            triggers=[FailureTrigger("strain", ">", 0.6)],
        )
        assert fm.composes == ["strain", "fatigue"]

    def test_persistent_with_recovery(self):
        fm = FailureMode(
            name="overheating",
            triggers=[FailureTrigger("temperature", ">", 70, 0.6)],
            persistent=True,
            recovery_condition=FailureTrigger("temperature", "<", 40),
            pain_intensity=0.6,
        )
        # Fire
        assert fm.evaluate({"temperature": 75}) is True
        assert fm.active is True

        # Still active — temperature dropped but not below recovery threshold
        assert fm.evaluate({"temperature": 55}) is True
        assert fm.active is True

        # Recovered
        assert fm.evaluate({"temperature": 35}) is False
        assert fm.active is False

    def test_embodiment_evaluates_composed(self):
        """Composed failure fires through the Embodiment runtime."""
        ent = Entity("elbow", "joint")
        ent.sensors["strain"] = _stub_sensor("strain", 0.7)
        ent.sensors["fatigue"] = _stub_sensor("fatigue", 0.6)
        ent.vital_metrics["strain"] = 0.7
        ent.vital_metrics["fatigue"] = 0.6
        ent.failure_modes.append(
            FailureMode(
                name="tennis_elbow",
                composes=["strain", "fatigue"],
                triggers=[
                    FailureTrigger("strain", ">", 0.6, 0.5),
                    FailureTrigger("fatigue", ">", 0.5, 0.5),
                ],
                trigger_mode="all",
                pain_intensity=0.5,
            )
        )

        pain_bus = MagicMock()
        emb = Embodiment(ent, pain_bus=pain_bus)
        events = emb.evaluate_failures()
        assert any(e.failure_name == "tennis_elbow" for e in events)

        # Stage 2: Embodiment publishes PainSignal via PainBus.publish with
        # the rich cause-description context dict, not a thin Reaction on
        # reaction_bus. The PainBus dual-dispatch path internally converts
        # to Reaction + fires both direct PainSignal subscribers and
        # reaction_bus subscribers.
        pain_bus.publish.assert_called()
        signal = pain_bus.publish.call_args[0][0]
        from maxim.proprioception.pain import PainSignal, PainType

        assert isinstance(signal, PainSignal)
        assert signal.intensity == 0.5
        assert signal.pain_type == PainType.EXTERNAL_SIGNAL
        assert signal.context["source"] == "embodiment"
        assert signal.context["entity"] == ent.full_path
        assert signal.context["failure_mode"] == "tennis_elbow"
        assert signal.context["composes"] == ["strain", "fatigue"]
        assert signal.context["entity_path"] == ent.full_path  # legacy key preserved

    def test_embodiment_includes_agent_id_in_pain_context(self):
        """``Embodiment(agent_id=X)`` propagates X into every PainSignal.context.

        Load-bearing for the reaction_bus subscriber in
        ``runtime/bio_stack.py``: without agent_id in the signal context,
        ``pain_signal_to_reaction`` produces a ReactionContext with
        ``agent_id=None``, the subscriber early-returns, and
        ``distribute()`` is never called for any pain reaction.
        """
        ent = Entity("arm", "limb")
        ent.sensors["strain"] = _stub_sensor("strain", 0.9)
        ent.vital_metrics["strain"] = 0.9
        ent.failure_modes.append(
            FailureMode(
                name="overstrain",
                triggers=[FailureTrigger("strain", ">", 0.5, 0.5)],
                pain_intensity=0.6,
            )
        )

        pain_bus = MagicMock()
        emb = Embodiment(ent, pain_bus=pain_bus, agent_id="agent_42")
        emb.evaluate_failures()

        pain_bus.publish.assert_called()
        signal = pain_bus.publish.call_args[0][0]
        assert signal.context["agent_id"] == "agent_42"


# ---------------------------------------------------------------------------
# Failure persistence
# ---------------------------------------------------------------------------


class TestFailurePersistence:
    def test_export_active_failures(self):
        ent = Entity("elbow", "joint")
        ent.vital_metrics["strain"] = 0.7
        fm = FailureMode(
            name="strain_overload",
            triggers=[FailureTrigger("strain", ">", 0.6, 0.6)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()  # Fire the failure

        state = emb.export_failure_state()
        assert len(state["active_failures"]) == 1
        assert state["active_failures"][0]["failure_name"] == "strain_overload"
        assert state["active_failures"][0]["persistent"] is True

    def test_export_no_active(self):
        ent = Entity("elbow", "joint")
        ent.vital_metrics["strain"] = 0.3
        fm = FailureMode(
            name="strain_overload",
            triggers=[FailureTrigger("strain", ">", 0.6, 0.6)],
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        state = emb.export_failure_state()
        assert len(state["active_failures"]) == 0

    def test_import_restores_active(self):
        ent = Entity("elbow", "joint")
        fm = FailureMode(
            name="overheating",
            triggers=[FailureTrigger("temperature", ">", 70, 0.6)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.import_failure_state(
            {
                "active_failures": [
                    {
                        "entity_path": "elbow",
                        "failure_name": "overheating",
                        "persistent": True,
                        "last_fired": 1000.0,
                    },
                ],
            }
        )

        assert fm.active is True
        assert fm.last_fired == 1000.0

    def test_save_load_roundtrip(self, tmp_path):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        path = str(tmp_path / "failures.json")
        emb.save_failures(path)

        # New embodiment, load state
        ent2 = Entity("shoulder", "joint")
        fm2 = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent2.failure_modes.append(fm2)
        emb2 = Embodiment(ent2)
        loaded = emb2.load_failures(path)
        assert loaded is True
        assert fm2.active is True

    def test_failure_history_in_export(self):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()

        state = emb.export_failure_state()
        assert len(state["failure_history"]) == 1
        assert state["failure_history"][0]["failure_name"] == "overextension"

    def test_active_failures_in_stats(self):
        ent = Entity("shoulder", "joint")
        ent.vital_metrics["angle"] = 180
        fm = FailureMode(
            name="overextension",
            triggers=[FailureTrigger("angle", ">", 175, 0.8)],
            persistent=True,
        )
        ent.failure_modes.append(fm)

        emb = Embodiment(ent)
        emb.evaluate_failures()
        stats = emb.stats()
        assert stats["active_failures"] == 1


# ---------------------------------------------------------------------------
# ToolPainBridge embodiment integration
# ---------------------------------------------------------------------------


class TestToolPainBridgeEmbodiment:
    def test_embodiment_pain_routed(self):
        """Embodiment pain signals are routed to _on_embodiment_pain."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.8,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "arm.shoulder",
                "entity_type": "joint",
                "failure_mode": "overextension",
                "composes": [],
                "sensor_readings": {"angle": 176},
            },
        )
        bridge._on_pain(signal)

        # Should have called record_outcome_full on NAc
        nac.record_outcome_full.assert_called_once()
        call_kwargs = nac.record_outcome_full.call_args
        assert call_kwargs[1]["outcome_type"] == "embodiment_failure"
        assert "arm.shoulder" in call_kwargs[1]["outcome_signature"]
        assert call_kwargs[1]["context"]["failure_mode"] == "overextension"

    def test_composed_metadata_passed(self):
        """Composition metadata flows through to NAc."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.5,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "arm.elbow",
                "entity_type": "joint",
                "failure_mode": "tennis_elbow",
                "composes": ["strain", "fatigue"],
                "sensor_readings": {"strain": 0.7, "fatigue": 0.6},
            },
        )
        bridge._on_pain(signal)

        call_kwargs = nac.record_outcome_full.call_args
        assert call_kwargs[1]["context"]["composes"] == ["strain", "fatigue"]

    def test_tool_pain_still_works(self):
        """Normal tool pain signals still route correctly."""
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])

        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        bridge = ToolPainBridge(nac=nac)

        # Non-embodiment signal should NOT call record_outcome_full
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=1000.0,
            context={"tool_name": "read_file", "invocation_id": "123"},
        )
        bridge._on_pain(signal)
        nac.record_outcome_full.assert_not_called()


# ---------------------------------------------------------------------------
# Stage 1 (sem_execution_hook) — tool-invoked embodiment failure attribution
# ---------------------------------------------------------------------------
#
# Pre-fix bug: when a tool-invoked SEM affordance produced embodiment
# failures, `_on_embodiment_pain` tried to attribute via NAc
# context-similarity. The pending tool event's context (`{"params": ...}`)
# shared zero keys with the outcome context (`{source, entity, failure_mode,
# ...}`), so the similarity score was 0/1 = 0.0 and the link never formed.
# NAc could not learn that the tool was negative.
#
# Fix: direct attribution via `record_tool_embodiment_failure` called by
# the executor, bypassing context similarity. `_on_embodiment_pain` is
# guarded to skip NAc attribution while a tool is in flight.


class TestToolEmbodimentAttribution:
    """Stage 1 regression guards for the tool→embodiment-pain attribution fix."""

    def test_record_tool_embodiment_failure_pops_pending_and_records_negative(self):
        """Direct-attribution API: pops pending event, records NEGATIVE
        via NAc.record_outcome (not record_outcome_full), and returns the RPE."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        mock_link = MagicMock()
        mock_link.last_rpe = 0.42
        mock_link.predicted_value = 0.5  # Real float — reflection path formats it.
        nac.record_outcome = MagicMock(return_value=[mock_link])

        bridge = ToolPainBridge(nac=nac)
        bridge.record_tool_start("rusty_sword_slash", "inv-1", context={"params": {"force": 0.9}})

        failures = [
            {"name": "shatter", "entity": "rusty_sword", "pain": 0.9},
        ]
        rpe = bridge.record_tool_embodiment_failure("rusty_sword_slash", "inv-1", failures)

        # Direct attribution — NOT context similarity.
        nac.record_outcome.assert_called_once()
        call_kwargs = nac.record_outcome.call_args.kwargs
        assert call_kwargs["event_type"] == "tool"
        assert call_kwargs["event_id"] == "tool:rusty_sword_slash"
        assert call_kwargs["outcome_valence"] == Valence.NEGATIVE
        # The outcome context carries the failure metadata for diagnostics.
        ctx = call_kwargs["context"]
        assert ctx["source"] == "embodiment"
        assert ctx["failure_mode"] == "shatter"
        assert ctx["entity"] == "rusty_sword"
        assert ctx["intensity"] == 0.9

        # Pending entry is popped (no leak).
        assert ("rusty_sword_slash", "inv-1") not in bridge._pending_tools
        assert rpe == 0.42

    def test_record_tool_embodiment_failure_no_pending_returns_zero(self):
        """Calling without a pending tool is a no-op — no NAc call, no crash."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        nac.record_outcome = MagicMock()

        bridge = ToolPainBridge(nac=nac)
        rpe = bridge.record_tool_embodiment_failure("ghost_tool", "inv-missing", [{"name": "x"}])

        nac.record_outcome.assert_not_called()
        assert rpe == 0.0

    def test_record_tool_embodiment_failure_empty_list_raises_value_error(self):
        """Empty failures list is a programming error — raise, don't silently
        succeed. The executor's guard is the load-bearing check; if the
        bridge is called directly with [], that's a contract violation."""
        import pytest
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        bridge = ToolPainBridge(nac=nac)
        bridge.record_tool_start("t", "inv", context={"params": {}})

        with pytest.raises(ValueError, match="non-empty"):
            bridge.record_tool_embodiment_failure("t", "inv", [])

    def test_tool_index_updated_negative_on_embodiment_failure(self):
        """Parity with record_tool_complete: learned tool index gets a
        negative outcome update. Pre-fix this was asymmetric — only
        positive outcomes updated the index, silently biasing the
        goal→tool selection toward whatever succeeded first."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])
        tool_index = MagicMock()

        bridge = ToolPainBridge(nac=nac, tool_index=tool_index)
        bridge.record_tool_start("sword_slash", "inv-ti", context={"goal": "defeat dummy"})

        bridge.record_tool_embodiment_failure(
            "sword_slash",
            "inv-ti",
            [{"name": "shatter", "entity": "sword", "pain": 0.9}],
        )

        tool_index.record_outcome.assert_called_once_with("defeat dummy", "sword_slash", success=False)

    def test_reflection_fires_on_surprising_embodiment_failure(self):
        """Parity with _on_pain: rpe > 0.3 triggers reflexion memory
        generation. Dropping this would silently regress the signal the
        hippocampus reflection path is specifically designed to capture."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        surprising_link = MagicMock()
        surprising_link.last_rpe = 0.55  # Above the 0.3 threshold.
        surprising_link.predicted_value = 0.5
        nac.record_outcome = MagicMock(return_value=[surprising_link])

        bridge = ToolPainBridge(nac=nac)
        bridge.record_tool_start("sword_slash", "inv-r", context={"params": {}})
        # Stub reflection path so we can assert without an LLM.
        bridge._generate_reflection = MagicMock(return_value="noted: sword fragile")
        bridge._store_reflection = MagicMock()

        bridge.record_tool_embodiment_failure(
            "sword_slash",
            "inv-r",
            [{"name": "shatter", "entity": "sword", "pain": 0.9}],
        )

        bridge._generate_reflection.assert_called_once()
        bridge._store_reflection.assert_called_once()

    def test_exactly_once_attribution_across_bridge_and_executor(self):
        """INVARIANT: when a tool in flight triggers synchronous embodiment
        pain during tool.run, the full flow (PainBus._on_pain callback →
        guard skip, executor → record_tool_embodiment_failure) records
        exactly ONE NAc outcome. Not zero (the pre-fix bug), not two
        (double-recording regression if the guard is ever removed)."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])
        nac.record_outcome_full = MagicMock(return_value=[])

        bridge = ToolPainBridge(nac=nac)
        bridge.record_tool_start("sword_slash", "inv-1x", context={"params": {}})
        # Clear the record_event() call count from record_tool_start.
        nac.reset_mock()

        # Simulate: during tool.run, embodiment fires pain → PainBus calls
        # _on_pain → _on_embodiment_pain guards because tool is pending.
        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.9,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "sword",
                "entity_type": "weapon",
                "failure_mode": "shatter",
                "composes": [],
                "sensor_readings": {},
            },
        )
        bridge._on_pain(signal)
        assert nac.record_outcome.call_count == 0, "guard should suppress"
        assert nac.record_outcome_full.call_count == 0, "guard should suppress full too"

        # Then: tool.run returns with side_effects; executor calls the
        # direct-attribution API.
        bridge.record_tool_embodiment_failure(
            "sword_slash",
            "inv-1x",
            [{"name": "shatter", "entity": "sword", "pain": 0.9}],
        )

        # EXACTLY ONCE across the combined flow.
        assert nac.record_outcome.call_count == 1
        assert nac.record_outcome_full.call_count == 0

    def test_on_embodiment_pain_guards_when_tool_pending(self):
        """With a pending tool, `_on_embodiment_pain` MUST skip NAc
        attribution — the executor will drive direct attribution after
        `tool.run` returns. This prevents double-recording AND prevents
        the broken context-similarity path from firing."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock()

        bridge = ToolPainBridge(nac=nac)
        # A tool is in flight.
        bridge.record_tool_start("rusty_sword_slash", "inv-2", context={"params": {"force": 0.8}})
        # Clear the mock history from record_tool_start's record_event call.
        nac.reset_mock()

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.8,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "rusty_sword",
                "entity_type": "weapon",
                "failure_mode": "shatter",
                "composes": [],
                "sensor_readings": {"durability": 0.05},
            },
        )
        bridge._on_pain(signal)

        # GUARD: with a pending tool, record_outcome_full must NOT fire.
        nac.record_outcome_full.assert_not_called()

    def test_on_embodiment_pain_falls_through_when_no_tool_pending(self):
        """Out-of-band embodiment pain (no tool in flight) still takes the
        context-similarity path via `record_outcome_full`. Preserves the
        autonomous-SEM-tick use case."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.proprioception.pain import PainSignal, PainType

        nac = MagicMock()
        nac.record_outcome_full = MagicMock(return_value=[])

        bridge = ToolPainBridge(nac=nac)
        # No pending tool — pure out-of-band case.

        signal = PainSignal(
            pain_type=PainType.EXTERNAL_SIGNAL,
            intensity=0.8,
            timestamp=1000.0,
            context={
                "source": "embodiment",
                "entity": "arm.shoulder",
                "entity_type": "joint",
                "failure_mode": "overextension",
                "composes": [],
                "sensor_readings": {"angle": 176},
            },
        )
        bridge._on_pain(signal)

        # Fall-through path — context-similarity attribution still runs.
        nac.record_outcome_full.assert_called_once()


class TestExecutorEmbodimentAttributionWiring:
    """End-to-end: Executor inspects ToolOutput.side_effects and routes
    embodiment failures through the bridge's direct-attribution API."""

    def _make_fake_tool_with_failures(self, failures):
        """A minimal Tool that returns a ToolOutput with side_effects populated."""
        from maxim.tools.base import Tool, ToolOutput

        class FakeAffordanceTool(Tool):
            name = "rusty_sword_slash"
            description = "fake sword slash"
            input_schema = {}

            def execute(self, **kwargs):
                return ToolOutput(
                    success=True,
                    output={"entity": "rusty_sword", "affordance": "slash"},
                    side_effects={"embodiment_failures": failures},
                )

        return FakeAffordanceTool()

    def _make_fake_tool_plain_success(self):
        from maxim.tools.base import Tool, ToolOutput

        class PlainTool(Tool):
            name = "plain_search"
            description = "plain tool"
            input_schema = {}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output={"hits": 3})

        return PlainTool()

    def test_executor_routes_embodiment_failures_to_bridge(self):
        """When a tool's ToolOutput carries side_effects['embodiment_failures'],
        the executor calls record_tool_embodiment_failure — NOT record_tool_complete."""
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        failures = [{"name": "shatter", "entity": "rusty_sword", "pain": 0.9}]
        tool = self._make_fake_tool_with_failures(failures)
        registry = ToolRegistry()
        registry.register(tool)

        bridge = MagicMock()
        bridge.record_tool_embodiment_failure = MagicMock(return_value=0.5)
        bridge.record_tool_complete = MagicMock()

        exe = Executor(tool_registry=registry, tool_pain_bridge=bridge)
        result = exe.execute({"tool_name": "rusty_sword_slash", "params": {}})

        assert result.success is True  # The TOOL succeeded; the entity failed.
        bridge.record_tool_start.assert_called_once()
        bridge.record_tool_embodiment_failure.assert_called_once()
        bridge.record_tool_complete.assert_not_called()

        # Assert the failures list was forwarded intact.
        call = bridge.record_tool_embodiment_failure.call_args
        assert call.args[0] == "rusty_sword_slash"
        # invocation_id is uuid, just check it's the same one used in record_tool_start
        assert call.args[2] == failures

    def test_executor_routes_plain_success_to_complete(self):
        """Regression guard: tools without side_effects still take the
        record_tool_complete success path."""
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        tool = self._make_fake_tool_plain_success()
        registry = ToolRegistry()
        registry.register(tool)

        bridge = MagicMock()
        bridge.record_tool_complete = MagicMock(return_value=0.0)
        bridge.record_tool_embodiment_failure = MagicMock()

        exe = Executor(tool_registry=registry, tool_pain_bridge=bridge)
        result = exe.execute({"tool_name": "plain_search", "params": {}})

        assert result.success is True
        bridge.record_tool_complete.assert_called_once()
        bridge.record_tool_embodiment_failure.assert_not_called()

    def test_executor_empty_side_effects_takes_complete_path(self):
        """Empty embodiment_failures list is indistinguishable from success."""
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        tool = self._make_fake_tool_with_failures([])  # Empty list.
        registry = ToolRegistry()
        registry.register(tool)

        bridge = MagicMock()
        bridge.record_tool_complete = MagicMock(return_value=0.0)
        bridge.record_tool_embodiment_failure = MagicMock()

        exe = Executor(tool_registry=registry, tool_pain_bridge=bridge)
        exe.execute({"tool_name": "rusty_sword_slash", "params": {}})

        bridge.record_tool_complete.assert_called_once()
        bridge.record_tool_embodiment_failure.assert_not_called()


class TestToolOutputSideEffects:
    """ToolOutput gains a typed side_effects field (backward-compatible default)."""

    def test_side_effects_defaults_to_none(self):
        from maxim.tools.base import ToolOutput

        out = ToolOutput(success=True, output="hello")
        assert out.side_effects is None

    def test_side_effects_carries_embodiment_failures(self):
        from maxim.tools.base import ToolOutput

        failures = [{"name": "shatter", "entity": "rusty_sword", "pain": 0.9}]
        out = ToolOutput(
            success=True,
            output={"entity": "rusty_sword"},
            side_effects={"embodiment_failures": failures},
        )
        assert out.side_effects == {"embodiment_failures": failures}

    def test_side_effects_survives_tool_run_wrapping(self):
        """Tool.run() preserves side_effects when execute() returns a ToolOutput."""
        from maxim.tools.base import Tool, ToolOutput

        class T(Tool):
            name = "t"
            description = "t"
            input_schema = {}

            def execute(self, **kwargs):
                return ToolOutput(
                    success=True,
                    output="x",
                    side_effects={"embodiment_failures": [{"name": "y"}]},
                )

        result = T().run()
        assert result.side_effects == {"embodiment_failures": [{"name": "y"}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_sensor(name: str, value: float):
    """Create a minimal SpecSensor for testing."""
    from maxim.embodiment.spec import SpecSensor

    return SpecSensor(
        _name=name,
        _entity_name="test",
        _unit="ratio",
        _schema={"type": "float", "range": [0, 1]},
        _initial=value,
    )


class TestNeutralToolCompletion:
    """D53: a tool that RAN but accomplished nothing books NEUTRAL.

    ``Executor`` hardcoded ``record_tool_complete(success=True)``, and the
    bridge hardcoded ``Valence.POSITIVE`` — a second door into the same
    causal bucket as ``tool_dispatch.record_outcome``'s collapse, so
    fixing only one leaves the flat positive intact.

    A clamped or unverifiable motion is a REFUSAL, not harm: it must not
    route through ``embodiment_failures`` (which asserts harm and would
    invert the bug), and it must not book POSITIVE (which asserts an
    achievement the tool itself reports did not happen).
    """

    def _tool(self, side_effects):
        from maxim.tools.base import Tool, ToolOutput

        class MotionTool(Tool):
            name = "reachy_turn_left"
            description = "fake motion"
            input_schema = {}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output={"ok": True}, side_effects=side_effects)

        return MotionTool()

    def _run(self, side_effects):
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(self._tool(side_effects))
        bridge = MagicMock()
        bridge.record_tool_complete = MagicMock(return_value=0.0)
        exe = Executor(tool_registry=registry, tool_pain_bridge=bridge)
        exe.execute({"tool_name": "reachy_turn_left", "params": {}})
        return bridge

    def test_executor_passes_neutral_for_ineffective_outcome(self):
        from maxim.decisions.causal_link import Valence

        bridge = self._run({"outcome_ineffective": True})
        bridge.record_tool_complete.assert_called_once()
        assert bridge.record_tool_complete.call_args.kwargs["outcome_valence"] is Valence.NEUTRAL

    def test_executor_passes_positive_for_effective_outcome(self):
        from maxim.decisions.causal_link import Valence

        for side in (None, {"outcome_ineffective": False}):
            bridge = self._run(side)
            assert bridge.record_tool_complete.call_args.kwargs["outcome_valence"] is Valence.POSITIVE

    def test_bridge_books_neutral_link_not_positive(self):
        """Against a real NAc: the neutral completion lands in NEITHER
        outcome bucket, so it contributes nothing to recommend_action."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.causal_link import Valence
        from maxim.decisions.nac import NAc, NACConfig

        nac = NAc(NACConfig())
        bridge = ToolPainBridge(nac=nac)

        bridge.record_tool_start("clamped_turn", "inv-1", {})
        bridge.record_tool_complete("clamped_turn", "inv-1", success=True, outcome_valence=Valence.NEUTRAL)
        bridge.record_tool_start("real_turn", "inv-2", {})
        bridge.record_tool_complete("real_turn", "inv-2", success=True, outcome_valence=Valence.POSITIVE)

        clamped = [lk for links in nac._links.values() for lk in links if "clamped_turn" in lk.event_signature]
        real = [lk for links in nac._links.values() for lk in links if "real_turn" in lk.event_signature]
        assert clamped and all(lk.outcome_valence is Valence.NEUTRAL for lk in clamped)
        assert real and all(lk.outcome_valence is Valence.POSITIVE for lk in real)

    def test_bridge_defaults_to_positive_for_existing_callers(self):
        """The historical contract is preserved when no tier is passed."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])
        bridge = ToolPainBridge(nac=nac)
        bridge.record_tool_start("t", "inv", {})
        bridge.record_tool_complete("t", "inv", success=True)
        assert nac.record_outcome.call_args.kwargs["outcome_valence"] is Valence.POSITIVE

    def test_neutral_does_not_strengthen_learned_tool_index(self):
        """An ineffective run is no evidence this tool serves this goal."""
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        nac.record_outcome = MagicMock(return_value=[])
        index = MagicMock()
        bridge = ToolPainBridge(nac=nac, tool_index=index)
        bridge.record_tool_start("t", "inv", {"goal": "find the sound"})
        bridge.record_tool_complete("t", "inv", success=True, outcome_valence=Valence.NEUTRAL)
        assert index.record_outcome.call_args.kwargs["success"] is False
