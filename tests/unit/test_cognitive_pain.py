"""Tests for Phase 3 cognitive pain subsystem.

Covers PainDetector tool-error methods, ToolPainBridge NAc integration,
ToolHarmPredictor, MonitorRegistry lifecycle, and Executor running-tool tracking.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.agents.bus import ToolErrorKind
from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence
from maxim.proprioception.pain import PainConfig, PainDetector, PainSignal, PainType


# ---------------------------------------------------------------------------
# 1. PainType has tool categories
# ---------------------------------------------------------------------------


class TestPainTypeToolCategories:
    """Verify PainType enum includes all tool-related members."""

    def test_tool_failure_exists(self) -> None:
        assert PainType.TOOL_FAILURE.value == "tool_failure"

    def test_tool_timeout_exists(self) -> None:
        assert PainType.TOOL_TIMEOUT.value == "tool_timeout"

    def test_tool_invalid_input_exists(self) -> None:
        assert PainType.TOOL_INVALID_INPUT.value == "tool_invalid_input"

    def test_tool_sustained_exists(self) -> None:
        assert PainType.TOOL_SUSTAINED.value == "tool_sustained"


# ---------------------------------------------------------------------------
# 2. PainSignal with defaults
# ---------------------------------------------------------------------------


class TestPainSignalDefaults:
    """PainSignal should have sensible defaults for movement fields."""

    def test_angular_velocity_default(self) -> None:
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=time.time(),
        )
        assert signal.angular_velocity == 0.0

    def test_translation_velocity_default(self) -> None:
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=time.time(),
        )
        assert signal.translation_velocity == 0.0

    def test_direction_reversals_default(self) -> None:
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=time.time(),
        )
        assert signal.direction_reversals == 0

    def test_context_default_is_empty_dict(self) -> None:
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.5,
            timestamp=time.time(),
        )
        assert signal.context == {}


# ---------------------------------------------------------------------------
# 3. record_tool_error emits signal
# ---------------------------------------------------------------------------


class TestRecordToolError:
    """record_tool_error should emit a PainSignal to callbacks."""

    def _make_detector(self) -> PainDetector:
        config = PainConfig(pain_cooldown_seconds=0.0)
        return PainDetector(config=config)

    def test_callback_receives_signal(self) -> None:
        detector = self._make_detector()
        received: list[PainSignal] = []
        detector.add_pain_callback(received.append)

        detector.record_tool_error(
            tool_name="web_search",
            error="connection refused",
            error_kind=ToolErrorKind.EXTERNAL_FAILURE,
        )

        assert len(received) == 1
        assert received[0].pain_type == PainType.TOOL_FAILURE
        assert received[0].context["tool_name"] == "web_search"

    def test_timeout_error_maps_to_tool_timeout(self) -> None:
        detector = self._make_detector()
        signal = detector.record_tool_error(
            tool_name="slow_tool",
            error="timed out",
            error_kind=ToolErrorKind.TIMEOUT,
        )
        assert signal is not None
        assert signal.pain_type == PainType.TOOL_TIMEOUT

    def test_invalid_input_maps_correctly(self) -> None:
        detector = self._make_detector()
        signal = detector.record_tool_error(
            tool_name="parser",
            error="bad json",
            error_kind=ToolErrorKind.INVALID_INPUT,
        )
        assert signal is not None
        assert signal.pain_type == PainType.TOOL_INVALID_INPUT


# ---------------------------------------------------------------------------
# 4. record_tool_error escalates intensity
# ---------------------------------------------------------------------------


class TestToolErrorEscalation:
    """Repeated errors for the same tool should increase intensity."""

    def test_intensity_increases_over_five_calls(self) -> None:
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)

        intensities: list[float] = []
        for _ in range(5):
            signal = detector.record_tool_error(
                tool_name="flaky_tool",
                error="fail",
                error_kind=ToolErrorKind.EXTERNAL_FAILURE,
            )
            assert signal is not None
            intensities.append(signal.intensity)

        # Last intensity should exceed first
        assert intensities[-1] > intensities[0]


# ---------------------------------------------------------------------------
# 5. record_tool_running below threshold
# ---------------------------------------------------------------------------


class TestRecordToolRunningBelowThreshold:
    """Tool running under expected time should not emit a signal."""

    def test_returns_none_when_ratio_under_one(self) -> None:
        detector = PainDetector()
        result = detector.record_tool_running(
            tool_name="fast_tool",
            elapsed=5.0,
            expected=10.0,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 6. record_tool_running above threshold
# ---------------------------------------------------------------------------


class TestRecordToolRunningAboveThreshold:
    """Tool running 2x expected should emit TOOL_SUSTAINED with intensity ~0.6."""

    def test_emits_sustained_signal(self) -> None:
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)
        signal = detector.record_tool_running(
            tool_name="slow_tool",
            elapsed=20.0,
            expected=10.0,
        )
        assert signal is not None
        assert signal.pain_type == PainType.TOOL_SUSTAINED
        # intensity = min(1.0, 0.3 * ratio) = min(1.0, 0.3 * 2.0) = 0.6
        assert abs(signal.intensity - 0.6) < 0.01


# ---------------------------------------------------------------------------
# 7. ToolPainBridge records start/complete
# ---------------------------------------------------------------------------


class TestToolPainBridgeStartComplete:
    """Bridge should forward tool lifecycle to NAc."""

    def test_start_records_event_and_complete_records_positive(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        detector = PainDetector()

        bridge = ToolPainBridge(nac=nac, pain_detector=detector)
        bridge.record_tool_start("web_search", "inv-1")
        nac.record_event.assert_called_once()
        call_kwargs = nac.record_event.call_args
        assert call_kwargs[1]["event_signature"] == "tool:web_search"

        bridge.record_tool_complete("web_search", "inv-1", success=True)
        nac.record_outcome.assert_called_once()
        call_kwargs = nac.record_outcome.call_args
        assert call_kwargs[1]["outcome_valence"] == Valence.POSITIVE


# ---------------------------------------------------------------------------
# 8. ToolPainBridge pain callback
# ---------------------------------------------------------------------------


class TestToolPainBridgePainCallback:
    """TOOL_FAILURE pain should record NEGATIVE outcome in NAc."""

    def test_on_pain_records_negative(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)

        bridge = ToolPainBridge(nac=nac, pain_detector=detector)
        # Start the tool so bridge tracks it
        bridge.record_tool_start("bad_tool", "inv-2")

        # Emit a pain signal with matching tool_name and invocation_id
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.8,
            timestamp=time.time(),
            context={"tool_name": "bad_tool", "invocation_id": "inv-2"},
        )
        bridge._on_pain(signal)

        # record_outcome should have been called with NEGATIVE
        outcome_calls = [
            c for c in nac.record_outcome.call_args_list if c[1].get("outcome_valence") == Valence.NEGATIVE
        ]
        assert len(outcome_calls) == 1


# ---------------------------------------------------------------------------
# 9. ToolPainBridge should_gate_tool
# ---------------------------------------------------------------------------


class TestToolPainBridgeShouldGate:
    """should_gate_tool should return True when NAc predicts negative."""

    def test_gates_when_nac_predicts_negative_high_confidence(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.causal_link import OutcomePrediction

        nac = MagicMock()
        prediction = OutcomePrediction(
            event_signature="tool:risky_tool",
            predicted_outcome="failure",
            predicted_valence=Valence.NEGATIVE,
            predicted_value=0.2,
            predicted_delay=1.0,
            delay_bounds=(0.5, 2.0),
            confidence=0.8,
            contributing_links=[],
            context_match=1.0,
        )
        nac.predict.return_value = prediction

        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        should_gate, reason = bridge.should_gate_tool("risky_tool")
        assert should_gate is True
        assert "risky_tool" in reason

    def test_does_not_gate_when_no_prediction(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        nac.predict.return_value = None

        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        should_gate, reason = bridge.should_gate_tool("safe_tool")
        assert should_gate is False
        assert reason == ""


# ---------------------------------------------------------------------------
# 10. ToolHarmPredictor with no history
# ---------------------------------------------------------------------------


class TestToolHarmPredictorNoHistory:
    """ToolHarmPredictor should return None when NAc has no links."""

    def test_returns_none_for_unknown_tool(self) -> None:
        from maxim.harm.tool_predictor import ToolHarmPredictor

        nac = MagicMock()
        nac.get_negative_outcomes.return_value = []

        predictor = ToolHarmPredictor(nac=nac)
        result = predictor.predict(
            action_type="tool_call",
            action_params={"tool_name": "new_tool"},
        )
        assert result is None

    def test_can_predict_only_tool_call(self) -> None:
        from maxim.harm.tool_predictor import ToolHarmPredictor

        nac = MagicMock()
        predictor = ToolHarmPredictor(nac=nac)
        assert predictor.can_predict("tool_call") is True
        assert predictor.can_predict("movement") is False


# ---------------------------------------------------------------------------
# 11. MonitorRegistry starts and stops
# ---------------------------------------------------------------------------


class TestMonitorRegistryLifecycle:
    """MonitorRegistry should start/stop without errors."""

    def test_start_stop_no_crash(self) -> None:
        from maxim.runtime.monitor_registry import MonitorRegistry, SignalMonitor

        class DummyMonitor(SignalMonitor):
            name = "dummy"

            def check(self) -> PainSignal | None:
                return None

        registry = MonitorRegistry(poll_interval=0.05)
        registry.register(DummyMonitor())
        registry.start()
        time.sleep(0.15)
        registry.stop()
        # No exception means success


# ---------------------------------------------------------------------------
# 12. MonitorRegistry fires callbacks
# ---------------------------------------------------------------------------


class TestMonitorRegistryCallbacks:
    """Monitor that returns a PainSignal should trigger callbacks."""

    def test_callback_receives_signal(self) -> None:
        from maxim.runtime.monitor_registry import MonitorRegistry, SignalMonitor

        emitted_signal = PainSignal(
            pain_type=PainType.TOOL_SUSTAINED,
            intensity=0.5,
            timestamp=time.time(),
        )

        class FiringMonitor(SignalMonitor):
            name = "firing"

            def __init__(self) -> None:
                self._fired = False

            def check(self) -> PainSignal | None:
                if not self._fired:
                    self._fired = True
                    return emitted_signal
                return None

        received: list[PainSignal] = []
        registry = MonitorRegistry(poll_interval=0.05)
        registry.register(FiringMonitor())
        registry.add_signal_callback(received.append)
        registry.start()
        time.sleep(0.2)
        registry.stop()

        assert len(received) >= 1
        assert received[0].pain_type == PainType.TOOL_SUSTAINED


# ---------------------------------------------------------------------------
# 13. Executor tracks running tool
# ---------------------------------------------------------------------------


class TestExecutorRunningTool:
    """Executor should track currently running tool via get_running_tool."""

    def test_running_tool_set_during_execution(self) -> None:
        from maxim.runtime.executor import Executor
        from maxim.tools.base import Tool, ToolOutput
        from maxim.tools.registry import ToolRegistry

        captured: list[Any] = []

        class SlowTool(Tool):
            name = "slow"
            description = "A tool that captures running state"
            input_schema: dict[str, Any] = {}

            def __init__(self, executor_ref: list[Executor]) -> None:
                super().__init__()
                self._executor_ref = executor_ref

            def execute(self, **kwargs: Any) -> ToolOutput:
                # During execution, get_running_tool should return info
                captured.append(self._executor_ref[0].get_running_tool())
                return ToolOutput(success=True, output="done")

        registry = ToolRegistry()
        executor_ref: list[Executor] = []
        tool = SlowTool(executor_ref)
        registry.register(tool)
        executor = Executor(tool_registry=registry)
        executor_ref.append(executor)

        executor.execute({"tool_name": "slow", "params": {}})

        # During execution, running tool should have been set
        assert len(captured) == 1
        assert captured[0] is not None
        name, start_time, inv_id = captured[0]
        assert name == "slow"
        assert isinstance(start_time, float)

        # After execution, should be None
        assert executor.get_running_tool() is None


# ---------------------------------------------------------------------------
# 14. ToolPainBridge SCN registration on tool pain
# ---------------------------------------------------------------------------


class TestToolPainBridgeSCNOnPain:
    """SCN should be notified when a tool pain signal is received."""

    def test_scn_register_called_on_tool_pain(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        scn = MagicMock()
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)

        bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)
        bridge.record_tool_start("flaky_api", "inv-10")

        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.7,
            timestamp=time.time(),
            context={"tool_name": "flaky_api", "invocation_id": "inv-10"},
        )
        bridge._on_pain(signal)

        scn.register.assert_called_once()
        call_args = scn.register.call_args
        assert call_args[0][0] == "tool:flaky_api"  # event_signature
        assert call_args[1]["significance"] == 0.7  # matches signal intensity

    def test_scn_not_called_when_none(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)

        bridge = ToolPainBridge(nac=nac, pain_detector=detector)  # no scn
        bridge.record_tool_start("flaky_api", "inv-11")

        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.7,
            timestamp=time.time(),
            context={"tool_name": "flaky_api", "invocation_id": "inv-11"},
        )
        bridge._on_pain(signal)
        # No error raised — SCN path is skipped gracefully


# ---------------------------------------------------------------------------
# 15. ToolPainBridge SCN registration on tool success
# ---------------------------------------------------------------------------


class TestToolPainBridgeSCNOnSuccess:
    """SCN should be notified with mild significance on tool success."""

    def test_scn_register_called_on_success(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        scn = MagicMock()
        detector = PainDetector()

        bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)
        bridge.record_tool_start("web_search", "inv-20")
        bridge.record_tool_complete("web_search", "inv-20", success=True)

        scn.register.assert_called_once()
        call_args = scn.register.call_args
        assert call_args[0][0] == "tool:web_search"
        assert call_args[1]["significance"] == 0.3  # mild positive

    def test_scn_not_called_on_failure_complete(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        scn = MagicMock()
        detector = PainDetector()

        bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)
        bridge.record_tool_start("bad_tool", "inv-21")
        bridge.record_tool_complete("bad_tool", "inv-21", success=False)

        scn.register.assert_not_called()


# ---------------------------------------------------------------------------
# 16. CausalLink stores last_rpe after update_prediction_rw
# ---------------------------------------------------------------------------


class TestCausalLinkLastRPE:
    """update_prediction_rw should populate last_rpe with abs(error)."""

    def _make_link(self, predicted_value: float = 0.5) -> CausalLink:
        return CausalLink(
            id="test-link",
            event_type="tool",
            event_signature="tool:test",
            event_context={},
            outcome_type="result",
            outcome_signature="result:test",
            outcome_valence=Valence.NEUTRAL,
            temporal_delta=TemporalDelta(observed_deltas=(1.0,)),
            predicted_value=predicted_value,
        )

    def test_last_rpe_none_initially(self) -> None:
        link = self._make_link()
        assert link.last_rpe is None

    def test_last_rpe_set_on_surprising_positive(self) -> None:
        link = self._make_link(predicted_value=0.2)
        error = link.update_prediction_rw(Valence.POSITIVE)
        # actual_reward=1.0, predicted=0.2 => error=0.8
        assert link.last_rpe == pytest.approx(abs(error))
        assert link.last_rpe == pytest.approx(0.8)

    def test_last_rpe_set_on_surprising_negative(self) -> None:
        link = self._make_link(predicted_value=0.9)
        error = link.update_prediction_rw(Valence.NEGATIVE)
        # actual_reward=0.0, predicted=0.9 => error=-0.9
        assert link.last_rpe == pytest.approx(0.9)
        assert error == pytest.approx(-0.9)

    def test_last_rpe_low_on_expected_outcome(self) -> None:
        link = self._make_link(predicted_value=0.95)
        link.update_prediction_rw(Valence.POSITIVE)
        # actual_reward=1.0, predicted=0.95 => |error|=0.05
        assert link.last_rpe == pytest.approx(0.05)

    def test_last_rpe_persists_in_serialization(self) -> None:
        link = self._make_link(predicted_value=0.3)
        link.update_prediction_rw(Valence.POSITIVE)
        data = link.to_dict()
        restored = CausalLink.from_dict(data)
        assert restored.last_rpe == pytest.approx(link.last_rpe)


# ---------------------------------------------------------------------------
# 17. ToolPainBridge exposes RPE on success
# ---------------------------------------------------------------------------


class TestToolPainBridgeRPEOnSuccess:
    """record_tool_complete should return RPE magnitude from NAc links."""

    def test_returns_rpe_from_links(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.nac import NAc

        nac = NAc()
        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        # First observation establishes a link
        bridge.record_tool_start("search", "inv-a")
        bridge.record_tool_complete("search", "inv-a", success=True)

        # Second observation — the link now has a predicted_value near 0.5-1.0,
        # so a POSITIVE outcome should produce a measurable RPE
        bridge.record_tool_start("search", "inv-b")
        rpe = bridge.record_tool_complete("search", "inv-b", success=True)
        assert isinstance(rpe, float)
        # RPE should also be stored on the bridge
        assert bridge._last_rpe == rpe

    def test_returns_zero_when_no_pending(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        nac = MagicMock()
        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        rpe = bridge.record_tool_complete("unknown", "inv-x", success=True)
        assert rpe == 0.0


# ---------------------------------------------------------------------------
# 18. ToolPainBridge stores RPE on pain
# ---------------------------------------------------------------------------


class TestToolPainBridgeRPEOnPain:
    """_on_pain should update _last_rpe from NAc links."""

    def test_last_rpe_updated_on_pain(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge

        # Use a real NAc so RPE is actually computed
        from maxim.decisions.nac import NAc

        nac = NAc()
        config = PainConfig(pain_cooldown_seconds=0.0)
        detector = PainDetector(config=config)
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        # Establish a link with a POSITIVE observation first
        bridge.record_tool_start("flaky", "inv-1")
        bridge.record_tool_complete("flaky", "inv-1", success=True)

        # Now trigger pain (NEGATIVE outcome) — should produce high RPE
        bridge.record_tool_start("flaky", "inv-2")
        signal = PainSignal(
            pain_type=PainType.TOOL_FAILURE,
            intensity=0.8,
            timestamp=time.time(),
            context={"tool_name": "flaky", "invocation_id": "inv-2"},
        )
        bridge._on_pain(signal)

        # RPE should be non-zero (surprising negative after positive history)
        assert bridge._last_rpe > 0.0


# ---------------------------------------------------------------------------
# 19. Executor get_last_rpe
# ---------------------------------------------------------------------------


class TestExecutorGetLastRPE:
    """Executor.get_last_rpe should reflect the bridge's most recent RPE."""

    def test_returns_zero_without_bridge(self) -> None:
        from maxim.runtime.executor import Executor
        from maxim.tools.registry import ToolRegistry

        executor = Executor(tool_registry=ToolRegistry())
        assert executor.get_last_rpe() == 0.0

    def test_returns_rpe_after_tool_execution(self) -> None:
        from maxim.bridges.tool_pain_bridge import ToolPainBridge
        from maxim.decisions.nac import NAc
        from maxim.runtime.executor import Executor
        from maxim.tools.base import Tool, ToolOutput
        from maxim.tools.registry import ToolRegistry

        nac = NAc()
        detector = PainDetector()
        bridge = ToolPainBridge(nac=nac, pain_detector=detector)

        class OkTool(Tool):
            name = "ok"
            description = "Always succeeds"
            input_schema: dict[str, Any] = {}

            def execute(self, **kwargs: Any) -> ToolOutput:
                return ToolOutput(success=True, output="ok")

        registry = ToolRegistry()
        registry.register(OkTool())
        executor = Executor(
            tool_registry=registry,
            pain_detector=detector,
            tool_pain_bridge=bridge,
        )

        # First call establishes the link
        executor.execute({"tool_name": "ok", "params": {}})
        # Second call updates RPE
        executor.execute({"tool_name": "ok", "params": {}})
        rpe = executor.get_last_rpe()
        assert isinstance(rpe, float)
        assert rpe >= 0.0
