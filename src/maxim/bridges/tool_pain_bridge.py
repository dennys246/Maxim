"""Bridges tool errors to NAc for causal learning.

Optionally integrates with SCN for temporal indexing of tool events,
enabling discovery of time-correlated failure patterns (e.g. API rate
limits at peak hours, resource exhaustion during batch windows).
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from maxim.decisions.causal_link import Valence
from maxim.decisions.nac import NAc
from maxim.proprioception.pain import PainDetector, PainSignal, PainType

if TYPE_CHECKING:
    from maxim.time.scn import SCN

logger = logging.getLogger(__name__)


class ToolPainBridge:
    """Connects tool execution outcomes to NAc causal learning.

    Learning flow:
    1. Tool is about to execute -> record_tool_start(tool_name, invocation_id)
    2. Tool completes successfully -> record_tool_complete(..., success=True)
    3. Tool fails -> PainDetector emits pain -> _on_pain() records NEGATIVE outcome
    4. Future calls: should_gate_tool() queries NAc before execution

    When an SCN instance is provided, tool events are also registered as
    temporal signatures so the system can learn time-of-day / day-of-week
    patterns in tool reliability.

    Example:
        bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)

        sig = bridge.record_tool_start("web_search", "inv-1")
        # ... tool executes ...
        bridge.record_tool_complete("web_search", "inv-1", success=True)

        # Before next call
        should_gate, reason = bridge.should_gate_tool("web_search")
    """

    def __init__(
        self,
        nac: NAc,
        pain_detector: PainDetector,
        scn: SCN | None = None,
    ) -> None:
        self._nac = nac
        self._scn = scn
        self._lock = threading.Lock()
        self._pending_tools: dict[tuple[str, str], str] = {}
        self._last_rpe: float = 0.0  # Most recent RPE magnitude (for salience)
        pain_detector.add_pain_callback(self._on_pain)

    def record_tool_start(
        self,
        tool_name: str,
        invocation_id: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Record that a tool invocation is starting.

        Args:
            tool_name: Name of the tool being executed.
            invocation_id: Unique ID for this invocation.
            context: Optional context about the invocation.

        Returns:
            Event signature for tracking.
        """
        event_signature = f"tool:{tool_name}"
        self._nac.record_event(
            event_type="tool",
            event_signature=event_signature,
            context=context,
        )
        with self._lock:
            self._pending_tools[(tool_name, invocation_id)] = event_signature
        return event_signature

    def record_tool_complete(
        self,
        tool_name: str,
        invocation_id: str,
        success: bool,
    ) -> float:
        """Record that a tool invocation completed.

        Args:
            tool_name: Name of the tool.
            invocation_id: Unique ID for this invocation.
            success: Whether the tool succeeded.

        Returns:
            RPE magnitude from NAc Rescorla-Wagner update (0.0 if no links).
            High values indicate surprising outcomes useful for memory salience.
        """
        with self._lock:
            event_signature = self._pending_tools.pop(
                (tool_name, invocation_id), None
            )
        if event_signature and success:
            links = self._nac.record_outcome(
                event_type="tool",
                event_id=event_signature,
                outcome_valence=Valence.POSITIVE,
            )
            rpe = max((l.last_rpe or 0.0 for l in links), default=0.0) if links else 0.0
            self._last_rpe = rpe

            # Register positive temporal signal with SCN
            if self._scn is not None:
                try:
                    from maxim.time.temporal_signature import TemporalSignature

                    self._scn.register(
                        event_signature,
                        TemporalSignature.now(),
                        significance=0.3,  # Mild positive signal
                    )
                except Exception:
                    pass  # SCN registration is best-effort

            return rpe
        return 0.0

    def _on_pain(self, signal: PainSignal) -> None:
        """Handle pain signals from tool failures."""
        if signal.pain_type not in (
            PainType.TOOL_FAILURE,
            PainType.TOOL_TIMEOUT,
            PainType.TOOL_INVALID_INPUT,
        ):
            return
        tool_name = signal.context.get("tool_name", "")
        invocation_id = signal.context.get("invocation_id", "")
        with self._lock:
            event_signature = self._pending_tools.pop(
                (tool_name, invocation_id), None
            )
        if event_signature:
            links = self._nac.record_outcome(
                event_type="tool",
                event_id=event_signature,
                outcome_valence=Valence.NEGATIVE,
                context=signal.context,
            )
            rpe = max((l.last_rpe or 0.0 for l in links), default=0.0) if links else 0.0
            self._last_rpe = rpe

            # Register temporal context with SCN
            if self._scn is not None:
                try:
                    from maxim.time.temporal_signature import TemporalSignature

                    self._scn.register(
                        event_signature,
                        TemporalSignature.now(),
                        significance=signal.intensity,
                    )
                except Exception:
                    pass  # SCN registration is best-effort

    def should_gate_tool(
        self,
        tool_name: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if a tool should be gated due to predicted failure.

        Args:
            tool_name: Name of the tool to check.
            context: Optional context for the prediction.

        Returns:
            Tuple of (should_gate, reason).
        """
        event_signature = f"tool:{tool_name}"
        prediction = self._nac.predict(
            event_type="tool",
            event_signature=event_signature,
            context=context,
        )
        if prediction is None:
            return False, ""
        if (
            prediction.predicted_valence == Valence.NEGATIVE
            and prediction.confidence >= 0.4
        ):
            return True, (
                f"NAc predicts '{tool_name}' will fail "
                f"(value={prediction.predicted_value:.2f}, "
                f"confidence={prediction.confidence:.2f})"
            )
        return False, ""


__all__ = ["ToolPainBridge"]
