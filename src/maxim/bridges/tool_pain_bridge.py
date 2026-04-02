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
    from maxim.memory.hippocampus import Hippocampus
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

    Reflexion integration:
    When a surprising failure occurs (RPE > 0.3), a verbal self-critique is
    generated and stored as an episodic memory in the hippocampus.  The
    AdaptivePlanner recalls these reflections via spreading activation to
    inform future planning decisions.

    Example:
        bridge = ToolPainBridge(nac=nac, pain_detector=detector, scn=scn)

        sig = bridge.record_tool_start("web_search", "inv-1")
        # ... tool executes ...
        bridge.record_tool_complete("web_search", "inv-1", success=True)

        # Before next call
        should_gate, reason = bridge.should_gate_tool("web_search")
    """

    # Rate-limit reflections: max 1 per tool_name per window
    _REFLECTION_COOLDOWN_S = 300.0  # 5 minutes

    def __init__(
        self,
        nac: NAc,
        pain_detector: PainDetector,
        scn: SCN | None = None,
        hippocampus: Hippocampus | None = None,
        llm: Any = None,
    ) -> None:
        self._nac = nac
        self._scn = scn
        self._hippocampus = hippocampus
        self._llm = llm
        self._lock = threading.Lock()
        self._pending_tools: dict[tuple[str, str], str] = {}
        self._last_rpe: float = 0.0  # Most recent RPE magnitude (for salience)
        self._last_reflection_time: dict[str, float] = {}  # tool_name → timestamp
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
            self._create_causal_edges(links)

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
            self._create_causal_edges(links)

            # Reflexion: generate and store verbal self-critique for surprising failures
            if rpe > 0.3:
                action_dict = {"tool_name": tool_name, "params": signal.context}
                error_str = signal.context.get("error", str(signal.pain_type.value))
                reflection = self._generate_reflection(links, action_dict, error_str, signal.context)
                if reflection:
                    self._store_reflection(reflection, action_dict)

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

    def _create_causal_edges(self, links: list) -> None:
        """Create CAUSES edges for surprising outcomes (RPE > 0.3).

        Connects older episodes in each link's memory_ids to the newest
        episode, threading a causal chain through the hippocampus graph.
        """
        if self._hippocampus is None or not links:
            return
        from maxim.agents.bus import EdgeType

        for link in links:
            if (link.last_rpe or 0.0) < 0.3:
                continue
            mem_ids = link.memory_ids
            if len(mem_ids) < 2:
                continue
            # Connect older episodes to the newest via CAUSES
            newest = mem_ids[-1]
            for older_id in mem_ids[-5:-1]:
                try:
                    self._hippocampus.graph.add_edge(
                        source=older_id,
                        target=newest,
                        edge_type=EdgeType.CAUSES,
                        weight=link.confidence,
                    )
                except Exception:
                    pass

    # ── Reflexion: verbal self-critique on surprising failures ──────────

    def _generate_reflection(
        self,
        links: list,
        action: dict[str, Any],
        error: str,
        context: dict[str, Any],
    ) -> str | None:
        """Generate a verbal reflection on surprising failure.

        Only called when RPE > 0.3 (surprising).  Rate-limited to one
        reflection per tool_name per ``_REFLECTION_COOLDOWN_S`` seconds.
        """
        tool_name = action.get("tool_name", "")

        # Rate-limit: skip if we recently reflected on this tool
        import time as _time

        now = _time.time()
        last = self._last_reflection_time.get(tool_name, 0.0)
        if now - last < self._REFLECTION_COOLDOWN_S:
            return None

        surprising_links = [l for l in links if (l.last_rpe or 0.0) > 0.3]
        if not surprising_links:
            return None

        self._last_reflection_time[tool_name] = now

        if self._llm is None:
            # Template-based fallback (no LLM required)
            link = surprising_links[0]
            ctx_summary = ", ".join(
                f"{k}={v}" for k, v in list(context.items())[:3]
            )
            return (
                f"Unexpected failure: {tool_name} failed with '{error}'. "
                f"Predicted success with value {link.predicted_value:.2f} "
                f"but actual was negative. "
                f"RPE={link.last_rpe:.2f}. Context: {ctx_summary}. "
                f"Consider: different parameters or alternative tool next time."
            )

        # LLM-generated reflection
        prompt = (
            f"A tool execution failed unexpectedly. Briefly explain what "
            f"went wrong and what to try differently next time.\n"
            f"Tool: {tool_name}\n"
            f"Error: {error}\n"
            f"Expected outcome value: {surprising_links[0].predicted_value:.2f}\n"
            f"Actual: failure (RPE={surprising_links[0].last_rpe:.2f})\n"
            f"Context: {context}\n"
            f"Reply in 1-2 sentences."
        )
        try:
            return self._llm.generate(prompt, max_tokens=100)
        except Exception as e:
            logger.debug("LLM reflection generation failed: %s", e)
            return None

    def _store_reflection(self, reflection: str, action: dict[str, Any]) -> None:
        """Store reflection as high-salience episodic memory in hippocampus."""
        if self._hippocampus is None or not reflection:
            return
        try:
            from maxim.memory.types import (
                Action as MemAction,
                Context as MemContext,
                Decision as MemDecision,
                Outcome as MemOutcome,
                Perception as MemPerception,
            )

            self._hippocampus.capture(
                perception=MemPerception(
                    salience=0.9,
                    novelty=0.8,
                    observations={"type": "reflection", "source": "reflexion"},
                ),
                context=MemContext(
                    active_goal=action.get("tool_name", ""),
                ),
                decision=MemDecision(
                    intent={"goal": "learn_from_failure"},
                ),
                action=MemAction(
                    tool_name=action.get("tool_name", ""),
                    tool_params=action.get("params", {}),
                ),
                outcome=MemOutcome(
                    success=False,
                    result={"reflection": reflection},
                ),
            )
        except Exception as e:
            logger.debug("Failed to store reflection: %s", e)

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
