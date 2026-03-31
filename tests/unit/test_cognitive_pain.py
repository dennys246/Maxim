"""Tests for Phase 3 cognitive pain subsystem.

Covers PainDetector tool-error methods, ToolPainBridge NAc integration,
ToolHarmPredictor, MonitorRegistry lifecycle, and Executor running-tool tracking.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

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
            c for c in nac.record_outcome.call_args_list
            if c[1].get("outcome_valence") == Valence.NEGATIVE
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
