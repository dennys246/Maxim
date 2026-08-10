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
from maxim.utils.logging import log_swallowed_exception

if TYPE_CHECKING:
    from maxim.decisions.temporal_credit import TemporalCreditDistributor
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
        pain_detector: PainDetector | None = None,
        scn: SCN | None = None,
        hippocampus: Hippocampus | None = None,
        llm: Any = None,
        pain_bus: Any | None = None,
        tool_index: Any = None,
        distributor: TemporalCreditDistributor | None = None,
        agent_id: str = "",
    ) -> None:
        self._nac = nac
        self._scn = scn
        self._hippocampus = hippocampus
        self._llm = llm
        self._tool_index = tool_index  # LearnedToolIndex for keyword weight updates
        self._distributor = distributor
        self._agent_id = agent_id
        self._lock = threading.Lock()
        self._pending_tools: dict[tuple[str, str], str] = {}
        self._pending_contexts: dict[tuple[str, str], dict[str, Any]] = {}  # (tool, inv_id) → context
        self._last_rpe: float = 0.0
        self._last_reflection_time: dict[str, float] = {}
        # Subscribe to pain signals via bus (preferred) or detector (legacy)
        if pain_bus is not None:
            pain_bus.subscribe(self._on_pain)
        elif pain_detector is not None:
            pain_detector.add_pain_callback(self._on_pain)

    def _emit_temporal_event(
        self,
        event_type: str,
        event_signature: str,
        activation: float,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Emit a TemporalEvent through the distributor (best-effort).

        Additive to the existing SCN registration — the distributor
        routes events through NAc eligibility traces and SCN phase
        anchoring for temporal credit attribution.
        """
        if self._distributor is None:
            return
        try:
            from uuid import uuid4

            from maxim.time.temporal_event import TemporalEvent
            from maxim.time.temporal_signature import TemporalSignature

            event = TemporalEvent(
                event_id=uuid4().hex,
                event_type=event_type,
                event_signature=event_signature,
                agent_id=self._agent_id,
                temporal_sig=TemporalSignature.now(),
                activation=activation,
                context=context or {},
            )
            self._distributor.record_event(event)
        except Exception:
            log_swallowed_exception()  # Temporal event emission is best-effort

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
            if context:
                self._pending_contexts[(tool_name, invocation_id)] = context
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
            event_signature = self._pending_tools.pop((tool_name, invocation_id), None)
            tool_context = self._pending_contexts.pop((tool_name, invocation_id), None)
        if event_signature and success:
            links = self._nac.record_outcome(
                event_type="tool",
                event_id=event_signature,
                outcome_valence=Valence.POSITIVE,
            )
            rpe = max((lnk.last_rpe or 0.0 for lnk in links), default=0.0) if links else 0.0
            self._last_rpe = rpe
            self._create_causal_edges(links)

            # Update learned tool index keyword weights
            if self._tool_index is not None and tool_context:
                goal_text = tool_context.get("goal", "")
                if goal_text:
                    self._tool_index.record_outcome(goal_text, tool_name, success=True)

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
                    log_swallowed_exception()  # SCN registration is best-effort

            # Emit temporal event for credit attribution
            self._emit_temporal_event(
                "tool",
                event_signature,
                activation=0.3,
                context={"outcome": "success", "tool_name": tool_name},
            )

            return rpe
        return 0.0

    def record_tool_embodiment_failure(
        self,
        tool_name: str,
        invocation_id: str,
        failures: list[dict[str, Any]],
    ) -> float:
        """Attribute embodiment failures to a pending tool event — directly.

        Called by the executor after ``tool.run()`` returns with
        ``ToolOutput.side_effects["embodiment_failures"]`` populated
        (the tool ran, but the body produced SEM failures). This is the
        production replacement for the pre-fix attribution path that
        relied on ``_on_embodiment_pain``'s context-similarity match —
        which silently failed because the pending tool event's context
        (``{"params": ...}``) shared zero keys with the rich outcome
        context emitted by ``body.py::_publish_pain``. See
        ``docs/plans/archive/sem_execution_hook.md`` Stage 1 for the full
        root-cause writeup.

        This method pops the pending tool event by ``(tool_name,
        invocation_id)`` and calls ``nac.record_outcome`` (not
        ``record_outcome_full``) with a direct event_id — NO context
        similarity, NO attribution ambiguity. Mirrors the shape of
        :meth:`record_tool_complete` on the failure side. Specifically,
        this method also:

        - updates ``_tool_index`` with a negative outcome (the learned
          tool index was previously asymmetric — only positive outcomes
          updated the keyword weights, so the index silently biased
          toward tools that happened to succeed first)
        - generates + stores a reflexion memory when ``rpe > 0.3``
          (surprising failures are exactly the signal reflexion is
          designed to learn from — the pre-fix ``_on_pain`` embodiment
          branch already did this, so dropping it would be a
          behavioral regression)

        Args:
            tool_name: Name of the tool whose execution produced the
                embodiment failures.
            invocation_id: Unique invocation ID from the executor.
            failures: Non-empty list of failure event dicts. Shape per
                entry: ``{"name": failure_mode, "entity": entity_path,
                "pain": intensity}``. The first entry's metadata
                populates the outcome context. Callers MUST ensure the
                list is non-empty — the executor's branch
                (``runtime/executor.py``) already guards on this, and
                an empty list here indicates a programming error.

        Returns:
            RPE magnitude from NAc's Rescorla-Wagner update. Zero if no
            pending event matched or no links formed.

        Raises:
            ValueError: If ``failures`` is empty. This is a precondition
                violation; the caller's guard is the load-bearing check.
        """
        if not failures:
            raise ValueError(
                "record_tool_embodiment_failure requires a non-empty failures list; "
                "the caller (runtime/executor.py) is responsible for gating on this."
            )

        with self._lock:
            event_signature = self._pending_tools.pop((tool_name, invocation_id), None)
            tool_context = self._pending_contexts.pop((tool_name, invocation_id), None)
        if not event_signature:
            return 0.0

        primary = failures[0]
        outcome_context: dict[str, Any] = {
            "source": "embodiment",
            "failure_mode": primary.get("name", ""),
            "entity": primary.get("entity", ""),
            "intensity": float(primary.get("pain", 0.0)),
            "failures": failures,
        }

        links = self._nac.record_outcome(
            event_type="tool",
            event_id=event_signature,
            outcome_valence=Valence.NEGATIVE,
            context=outcome_context,
        )
        rpe = max((lnk.last_rpe or 0.0 for lnk in links), default=0.0) if links else 0.0
        self._last_rpe = rpe
        self._create_causal_edges(links)

        # Parity with record_tool_complete's tool-index update, on the
        # negative side. Without this, the learned tool index silently
        # biases toward tools that happened to succeed first — an
        # asymmetric-bias bug the pre-merge review caught.
        if self._tool_index is not None and tool_context:
            goal_text = tool_context.get("goal", "")
            if goal_text:
                self._tool_index.record_outcome(goal_text, tool_name, success=False)

        # Reflexion: generate + store verbal self-critique for surprising
        # failures. Mirrors the `_on_pain` embodiment branch the pre-fix
        # path used — dropping it would be a silent behavioral regression
        # on exactly the signal reflexion exists to capture.
        if rpe > 0.3 and links:
            action_dict = {"tool_name": tool_name, "params": outcome_context}
            error_str = f"embodiment:{outcome_context['failure_mode']}"
            reflection = self._generate_reflection(links, action_dict, error_str, outcome_context)
            if reflection:
                self._store_reflection(reflection, action_dict)

        # Register temporal context with SCN (mirror record_tool_complete path).
        if self._scn is not None:
            try:
                from maxim.time.temporal_signature import TemporalSignature

                self._scn.register(
                    event_signature,
                    TemporalSignature.now(),
                    significance=outcome_context["intensity"] or 0.5,
                )
            except Exception:
                log_swallowed_exception()  # SCN registration is best-effort.

        # Emit temporal event for credit attribution
        self._emit_temporal_event(
            "pain",
            event_signature,
            activation=outcome_context.get("intensity") or 0.5,
            context={
                "outcome": "embodiment_failure",
                "tool_name": tool_name,
                "failure_mode": outcome_context.get("failure_mode", ""),
            },
        )

        return rpe

    def _on_pain(self, signal: PainSignal) -> None:
        """Handle pain signals from tool failures and embodiment failures."""
        # Embodiment-sourced failures (SEM entities)
        if signal.context.get("source") == "embodiment":
            self._on_embodiment_pain(signal)
            return

        if signal.pain_type not in (
            PainType.TOOL_FAILURE,
            PainType.TOOL_TIMEOUT,
            PainType.TOOL_INVALID_INPUT,
        ):
            return
        tool_name = signal.context.get("tool_name", "")
        invocation_id = signal.context.get("invocation_id", "")
        with self._lock:
            event_signature = self._pending_tools.pop((tool_name, invocation_id), None)
            self._pending_contexts.pop((tool_name, invocation_id), None)
        if event_signature:
            links = self._nac.record_outcome(
                event_type="tool",
                event_id=event_signature,
                outcome_valence=Valence.NEGATIVE,
                context=signal.context,
            )
            rpe = max((lnk.last_rpe or 0.0 for lnk in links), default=0.0) if links else 0.0
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
                    log_swallowed_exception()  # SCN registration is best-effort

            # Emit temporal event for credit attribution
            self._emit_temporal_event(
                "pain",
                event_signature,
                activation=signal.intensity,
                context={"outcome": "tool_pain", "tool_name": tool_name, "pain_type": signal.pain_type.value},
            )

    def _on_embodiment_pain(self, signal: PainSignal) -> None:
        """Handle pain signals from embodiment failures (SEM entities).

        Two paths:

        1. **Tool-invoked** (ANY pending tool is in flight): SKIP NAc
           attribution here. The executor will call
           :meth:`record_tool_embodiment_failure` after ``tool.run()``
           returns, using direct ``(tool_name, invocation_id)`` lookup.
           This is the production path for tool-invoked SEM affordances.
           The pre-fix context-similarity attribution silently failed
           for this case (``{"params": ...}`` vs the rich outcome
           context shared zero keys).

        2. **Out-of-band** (no pending tool): fall through to
           ``record_outcome_full`` with the rich signal context, letting
           NAc's temporal-window + context-similarity path attribute the
           pain to any other pending events. This is the autonomous-SEM
           tick path — a joint limit trips while no tool is running.

        **Broad-guard semantics** (cross-confirmed in the pre-merge
        review): the guard checks ``bool(self._pending_tools)`` — ANY
        pending tool suppresses attribution here, not just the specific
        tool whose modulator caused this pain. This is safe under the
        current **serialized-executor** contract (one ``Executor`` +
        one ``ToolPainBridge`` per agent instance; executions are
        serialized through ``Executor._lock``; concurrent executions
        happen only across agents and each agent has its own bridge).
        If a future refactor introduces concurrent in-flight tool
        executions sharing a single bridge, this guard becomes
        **over-broad** and will silently drop out-of-band embodiment
        pain for the duration of any pending tool. Narrow the guard
        at that point by iterating ``_pending_contexts`` and matching
        ``signal.context.get("entity")`` against stored per-pending
        entity metadata — which requires the executor to record entity
        at ``record_tool_start`` time.

        **Out-of-band path weakness** (also cross-confirmed): the
        context-similarity fall-through (``record_outcome_full`` below)
        remains subject to the same ``{"params": ...}`` vs rich-context
        mismatch that motivated this fix — it only attributes when a
        pending NON-tool event happens to share keys with the
        embodiment outcome. This is acceptable as Stage 1 scope
        because out-of-band embodiment pain has no direct-lookup key
        available. If this silently-drops-attribution case becomes a
        problem in practice, the remedy is a future stage that enriches
        the pending-event context on the non-tool path, not another
        band-aid here.

        The guard reads ``self._pending_tools`` under the existing lock
        — no new threadlocal, no new ContextVar, no re-entrancy hazard
        (explicitly forbidden by ``proprioception/pain_bus.py`` docstring
        after the Substrate P2 Stage 2 incident).
        """
        with self._lock:
            has_pending_tool = bool(self._pending_tools)
        if has_pending_tool:
            # Executor will attribute via record_tool_embodiment_failure
            # after tool.run returns. Skip here to avoid both (a) the
            # broken context-similarity path and (b) double-recording.
            # See the broad-guard-semantics note in the docstring for
            # the serialized-executor assumption.
            return

        entity_path = signal.context.get("entity", "")
        failure_mode = signal.context.get("failure_mode", "")
        composes = signal.context.get("composes", [])
        entity_type = signal.context.get("entity_type", "")

        # Build event signature from entity + failure
        event_signature = f"embodiment:{entity_path}:{failure_mode}"

        # Record in NAc as a causal observation
        links = self._nac.record_outcome_full(
            outcome_type="embodiment_failure",
            outcome_signature=event_signature,
            outcome_valence=Valence.NEGATIVE,
            context={
                "source": "embodiment",
                "entity": entity_path,
                "entity_type": entity_type,
                "failure_mode": failure_mode,
                "composes": composes,
                "intensity": signal.intensity,
                "sensor_readings": signal.context.get("sensor_readings", {}),
            },
        )

        rpe = max((lnk.last_rpe or 0.0 for lnk in links), default=0.0) if links else 0.0
        self._last_rpe = rpe
        self._create_causal_edges(links)

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
                log_swallowed_exception()

        # Emit temporal event for credit attribution
        self._emit_temporal_event(
            "pain",
            event_signature,
            activation=signal.intensity,
            context={"outcome": "embodiment_pain", "entity": entity_path, "failure_mode": failure_mode},
        )

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
                    log_swallowed_exception()

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

        surprising_links = [lnk for lnk in links if (lnk.last_rpe or 0.0) > 0.3]
        if not surprising_links:
            return None

        self._last_reflection_time[tool_name] = now

        if self._llm is None:
            # Template-based fallback (no LLM required)
            link = surprising_links[0]
            ctx_summary = ", ".join(f"{k}={v}" for k, v in list(context.items())[:3])
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
        tool_params: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Check if a tool should be gated due to predicted failure.

        Args:
            tool_name: Name of the tool to check.
            context: Optional context for the prediction.
            tool_params: Tool call parameters (for compound signature).

        Returns:
            Tuple of (should_gate, reason).
        """
        from maxim.runtime.tool_dispatch import build_tool_signature

        event_signature = build_tool_signature(tool_name, tool_params)
        prediction = self._nac.predict(
            event_type="tool",
            event_signature=event_signature,
            context=context,
        )
        if prediction is None:
            return False, ""
        if prediction.predicted_valence == Valence.NEGATIVE and prediction.confidence >= 0.4:
            return True, (
                f"NAc predicts '{tool_name}' will fail "
                f"(value={prediction.predicted_value:.2f}, "
                f"confidence={prediction.confidence:.2f})"
            )
        return False, ""


__all__ = ["ToolPainBridge"]
